# Data/svhn_augmix.py
"""
SVHN data loader with AugMix support
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import numpy as np

class SVHNAugMix(Dataset):
    """SVHN dataset that returns AugMix triplets"""
    
    def __init__(self, root, split='train', augmix_transform=None):
        self.svhn = datasets.SVHN(root=root, split=split, download=True)
        self.augmix_transform = augmix_transform
        
    def __len__(self):
        return len(self.svhn)
    
    def __getitem__(self, idx):
        image, label = self.svhn[idx]
        
        if self.augmix_transform:
            # Returns (clean, aug1, aug2)
            clean, aug1, aug2 = self.augmix_transform(image)
            return clean, aug1, aug2, label
        else:
            # Fallback to standard transform (use SVHN normalization - same as CIFAR)
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
            return transform(image), label

def get_augmix_loaders(batch_size, random_seed, valid_size=0.1, augmix_transform=None):
    """Get AugMix training and validation loaders for SVHN"""
    from torch.utils.data.sampler import SubsetRandomSampler
    
    # Create dataset - SVHN uses 'train' split for training data
    train_dataset = SVHNAugMix(
        root='./data', 
        split='train', 
        augmix_transform=augmix_transform
    )
    
    val_dataset = SVHNAugMix(
        root='./data',
        split='train',
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