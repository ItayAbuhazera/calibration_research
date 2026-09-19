# Data/pacs_augmix.py
"""
PACS data loader with AugMix support
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Data.pacs import PACSDataset
from torch.utils.data.sampler import SubsetRandomSampler

class PACSAugMix(Dataset):
    """PACS dataset that returns AugMix triplets"""
    
    def __init__(self, root, domain=None, train=True, augmix_transform=None):
        self.pacs = PACSDataset(root=root, domain=domain, train=train, download=True)
        self.augmix_transform = augmix_transform
        
    def __len__(self):
        return len(self.pacs)
    
    def __getitem__(self, idx):
        image, label = self.pacs[idx]
        
        if self.augmix_transform:
            # Returns (clean, aug1, aug2)
            clean, aug1, aug2 = self.augmix_transform(image)
            return clean, aug1, aug2, label
        else:
            # Fallback to standard transform (use ImageNet normalization for PACS)
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            return transform(image), label

def get_augmix_loaders(batch_size, random_seed, valid_size=0.1, augmix_transform=None, domain=None):
    """Get AugMix training and validation loaders for PACS"""
    
    # Create dataset
    train_dataset = PACSAugMix(
        root='./data', 
        domain=domain,
        train=True, 
        augmix_transform=augmix_transform
    )
    
    val_dataset = PACSAugMix(
        root='./data',
        domain=domain,
        train=True,
        augmix_transform=None  # No augmentation for validation
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
    
    # Create loaders
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