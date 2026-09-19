# Data/pacs.py
"""
PACS dataset loader for domain adaptation research
PACS: Photo, Art, Cartoon, Sketch - 4 domains with 7 classes
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import glob
from pathlib import Path

class PACSDataset(Dataset):
    """
    PACS dataset with 4 domains (Photo, Art, Cartoon, Sketch) and 7 classes
    Classes: dog, elephant, giraffe, guitar, horse, house, person
    """
    
    DOMAINS = ['photo', 'art_painting', 'cartoon', 'sketch']
    CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    
    def __init__(self, root, domain=None, train=True, transform=None, download=True):
        """
        Args:
            root: Root directory where PACS dataset is stored
            domain: Specific domain to load ('photo', 'art_painting', 'cartoon', 'sketch')
                   If None, loads all domains
            train: If True, loads train split, otherwise test split
            transform: Transform to apply to images
            download: If True, attempts to download dataset (placeholder)
        """
        self.root = Path(root)
        self.domain = domain
        self.train = train
        self.transform = transform
        
        if download:
            self._download()
        
        # Load data
        self.samples = []
        self.labels = []
        self._load_data()
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
    
    def _download(self):
        """
        Download PACS dataset if not exists
        Note: This is a placeholder - in practice, PACS needs to be downloaded manually
        """
        pacs_dir = self.root / 'pacs'
        if not pacs_dir.exists():
            print("📁 PACS dataset not found. Please download manually from:")
            print("   https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk")
            print(f"   Extract to: {pacs_dir}")
            
            # Create directory structure as placeholder
            pacs_dir.mkdir(parents=True, exist_ok=True)
            for domain in self.DOMAINS:
                domain_dir = pacs_dir / domain
                domain_dir.mkdir(exist_ok=True)
                for class_name in self.CLASSES:
                    class_dir = domain_dir / class_name
                    class_dir.mkdir(exist_ok=True)
    
    def _load_data(self):
        """Load image paths and labels"""
        pacs_dir = self.root / 'pacs'
        
        domains_to_load = [self.domain] if self.domain else self.DOMAINS
        
        for domain in domains_to_load:
            domain_dir = pacs_dir / domain
            if not domain_dir.exists():
                continue
                
            for class_idx, class_name in enumerate(self.CLASSES):
                class_dir = domain_dir / class_name
                if not class_dir.exists():
                    continue
                
                # Load all images from this class directory
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                image_paths = []
                for ext in image_extensions:
                    image_paths.extend(glob.glob(str(class_dir / ext)))
                
                for img_path in image_paths:
                    self.samples.append(img_path)
                    self.labels.append(class_idx)
        
        print(f"✅ Loaded {len(self.samples)} PACS images")
        if self.domain:
            print(f"   Domain: {self.domain}")
        else:
            print("   All domains included")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_train_valid_loader(batch_size,
                           augment=True,
                           random_seed=42,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=0,
                           pin_memory=False,
                           domain=None):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the PACS dataset.
    
    Args:
        batch_size: how many samples per batch to load
        augment: whether to apply data augmentation
        random_seed: fix seed for reproducibility
        valid_size: percentage split of the training set used for validation
        shuffle: whether to shuffle the train/validation indices
        num_workers: number of subprocesses to use when loading the dataset
        pin_memory: whether to copy tensors into CUDA pinned memory
        domain: specific domain to load ('photo', 'art_painting', 'cartoon', 'sketch')
    
    Returns:
        train_loader: training set iterator
        valid_loader: validation set iterator
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # PACS uses ImageNet normalization (224x224 images)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Define transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    # Load the dataset
    data_dir = './data'
    train_dataset = PACSDataset(
        root=data_dir, 
        domain=domain,
        train=True,
        transform=train_transform,
        download=True
    )

    valid_dataset = PACSDataset(
        root=data_dir,
        domain=domain, 
        train=True,
        transform=valid_transform,
        download=False
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=0, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=0, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def get_test_loader(batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    domain=None):
    """
    Utility function for loading and returning a multi-process
    test iterator over the PACS dataset.
    
    Args:
        batch_size: how many samples per batch to load
        shuffle: whether to shuffle the dataset
        num_workers: number of subprocesses to use when loading the dataset
        pin_memory: whether to copy tensors into CUDA pinned memory
        domain: specific domain to load ('photo', 'art_painting', 'cartoon', 'sketch')
    
    Returns:
        data_loader: test set iterator
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = './data'
    dataset = PACSDataset(
        root=data_dir,
        domain=domain,
        train=False,  # Test set
        transform=transform,
        download=True
    )

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=pin_memory,
    )

    return data_loader

# Convenience function to get all domain loaders
def get_all_domain_loaders(batch_size, random_seed=42, valid_size=0.1):
    """
    Get data loaders for all PACS domains
    
    Returns:
        Dictionary with domain names as keys and (train_loader, val_loader, test_loader) tuples as values
    """
    domain_loaders = {}
    
    for domain in PACSDataset.DOMAINS:
        train_loader, val_loader = get_train_valid_loader(
            batch_size=batch_size,
            random_seed=random_seed,
            valid_size=valid_size,
            domain=domain
        )
        test_loader = get_test_loader(
            batch_size=batch_size,
            domain=domain
        )
        
        domain_loaders[domain] = (train_loader, val_loader, test_loader)
    
    return domain_loaders