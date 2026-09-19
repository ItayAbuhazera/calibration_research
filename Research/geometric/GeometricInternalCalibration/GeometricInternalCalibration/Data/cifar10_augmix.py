# Data/cifar10_augmix.py
"""
CIFAR-10 data loader with AugMix support
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import numpy as np

class CIFAR10AugMix(Dataset):
    """CIFAR-10 dataset that returns AugMix triplets"""
    
    def __init__(self, root, train=True, augmix_transform=None):
        self.cifar10 = datasets.CIFAR10(root=root, train=train, download=True)
        self.augmix_transform = augmix_transform
        
    def __len__(self):
        return len(self.cifar10)
    
    def __getitem__(self, idx):
        image, label = self.cifar10[idx]
        
        if self.augmix_transform:
            # Returns (clean, aug1, aug2)
            clean, aug1, aug2 = self.augmix_transform(image)
            return clean, aug1, aug2, label
        else:
            # Fallback to standard transform
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
            return transform(image), label

def get_augmix_loaders(batch_size, random_seed, valid_size=0.1, augmix_transform=None):
    """Get AugMix training and validation loaders"""
    from torch.utils.data.sampler import SubsetRandomSampler
    
    # Create dataset
    train_dataset = CIFAR10AugMix(
        root='./data', 
        train=True, 
        augmix_transform=augmix_transform
    )
    
    val_dataset = CIFAR10AugMix(
        root='./data',
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