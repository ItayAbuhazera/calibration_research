# Data/pacs_domain_adaptation.py
"""
PACS dataset loader for domain adaptation research
Train on 3 domains, test on 1 domain
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import glob
from pathlib import Path
from Data.pacs import PACSDataset

def get_domain_adaptation_loaders(target_domain, batch_size=128, random_seed=42, valid_size=0.1,
                                  train_transform=None, val_test_transform=None):
    """
    Get data loaders for domain adaptation: train on 3 domains, test on 1
    
    Args:
        target_domain: The domain to use for testing ('photo', 'art_painting', 'cartoon', 'sketch')
        batch_size: Batch size for data loading
        random_seed: Random seed for reproducibility
        valid_size: Fraction of source domains to use for validation
        train_transform: Transform to apply to training data
        val_test_transform: Transform to apply to validation and test data
    """
    all_domains = ['photo', 'art_painting', 'cartoon', 'sketch']
    source_domains = [d for d in all_domains if d != target_domain]
    
    print(f"🔄 Domain Adaptation Setup:")
    print(f"   Source domains (train): {source_domains}")
    print(f"   Target domain (test): {target_domain}")
    
    # Use default transforms if none are provided
    if train_transform is None or val_test_transform is None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        train_transform = train_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        
        val_test_transform = val_test_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])
        print("   Using default PACS transforms.")
    else:
        print("   Using provided custom transforms.")

    # Create datasets for each source domain
    source_datasets = []
    for domain in source_domains:
        dataset = PACSDataset(
            root='./data',
            domain=domain,
            train=True,
            transform=train_transform,
            download=True
        )
        source_datasets.append(dataset)
        print(f"   {domain}: {len(dataset)} images")
    
    # Combine source domains
    combined_source = ConcatDataset(source_datasets)
    print(f"   Combined source: {len(combined_source)} images")
    
    # Create train/validation split from source domains
    num_source = len(combined_source)
    indices = list(range(num_source))
    split = int(np.floor(valid_size * num_source))
    
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    # Create train and validation loaders
    train_loader = DataLoader(
        combined_source,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        combined_source,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    # Create test dataset for target domain
    test_dataset = PACSDataset(
        root='./data',
        domain=target_domain,
        train=False,  # Use full domain for testing
        transform=val_test_transform,
        download=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"   Target test: {len(test_dataset)} images")
    print(f"✅ Domain adaptation loaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def get_all_domain_shifts():
    """
    Get all possible domain shift combinations for comprehensive evaluation
    
    Returns:
        List of (source_domains, target_domain) tuples
    """
    all_domains = ['photo', 'art_painting', 'cartoon', 'sketch']
    domain_shifts = []
    
    for target in all_domains:
        source_domains = [d for d in all_domains if d != target]
        domain_shifts.append((source_domains, target))
    
    return domain_shifts

def evaluate_all_domain_shifts(model, device, batch_size=128):
    """
    Evaluate model on all possible domain shifts
    
    Args:
        model: Trained model
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with results for each domain shift
    """
    results = {}
    domain_shifts = get_all_domain_shifts()
    
    model.eval()
    
    for source_domains, target_domain in domain_shifts:
        print(f"\n🔍 Evaluating domain shift: {source_domains} → {target_domain}")
        
        # Get test loader for target domain
        test_dataset = PACSDataset(
            root='./data',
            domain=target_domain,
            train=False,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            download=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Evaluate
        correct = 0
        total = 0
        all_confidences = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probs, 1)
                
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
                
                all_confidences.extend(confidences.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        accuracy = 100.0 * correct / total
        
        # Calculate ECE
        from Metrics.metrics import expected_calibration_error
        ece = expected_calibration_error(all_confidences, all_predictions, all_labels)
        
        results[f"{'-'.join(source_domains)}_to_{target_domain}"] = {
            'accuracy': accuracy,
            'ece': ece,
            'source_domains': source_domains,
            'target_domain': target_domain,
            'total_samples': total
        }
        
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   ECE: {ece:.4f}")
    
    return results

# Integration with existing AugMix support
class PACSAugMixDomainAdaptation(Dataset):
    """PACS dataset for domain adaptation with AugMix support"""
    
    def __init__(self, source_domains, augmix_transform=None):
        self.source_datasets = []
        
        for domain in source_domains:
            dataset = PACSDataset(
                root='./data',
                domain=domain,
                train=True,
                transform=None,  # Will apply transform in __getitem__
                download=True
            )
            self.source_datasets.append(dataset)
        
        # Combine all source datasets
        self.combined_data = []
        for dataset in self.source_datasets:
            self.combined_data.extend([(dataset.samples[i], dataset.labels[i]) for i in range(len(dataset))])
        
        self.augmix_transform = augmix_transform
    
    def __len__(self):
        return len(self.combined_data)
    
    def __getitem__(self, idx):
        img_path, label = self.combined_data[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.augmix_transform:
            # Returns (clean, aug1, aug2)
            clean, aug1, aug2 = self.augmix_transform(image)
            return clean, aug1, aug2, label
        else:
            # Standard transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            return transform(image), label

def get_domain_adaptation_augmix_loaders(target_domain, batch_size=128, random_seed=42, 
                                       valid_size=0.1, augmix_transform=None):
    """Get AugMix loaders for domain adaptation"""
    all_domains = ['photo', 'art_painting', 'cartoon', 'sketch']
    source_domains = [d for d in all_domains if d != target_domain]
    
    # Training dataset with AugMix
    train_dataset = PACSAugMixDomainAdaptation(
        source_domains=source_domains,
        augmix_transform=augmix_transform
    )
    
    # Validation dataset without AugMix
    val_dataset = PACSAugMixDomainAdaptation(
        source_domains=source_domains,
        augmix_transform=None
    )
    
    # Create train/val split
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader