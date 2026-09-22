"""
DINOv2-specific CIFAR-10 data loading with optimized transforms and batch sizes.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import logging

from utils.logging_config import get_logger
logger = get_logger(__name__)

def get_dinov2_loaders(batch_size=128, random_seed=42, valid_size=0.1, 
                       train_transform=None, val_test_transform=None):
    """
    Get DINOv2-optimized CIFAR-10 data loaders.
    
    Args:
        batch_size: Batch size for training (default 128 for DINOv2)
        random_seed: Random seed for reproducibility
        valid_size: Fraction of training data to use for validation
        train_transform: Custom training transforms (if None, use default DINOv2 transforms)
        val_test_transform: Custom validation/test transforms (if None, use default DINOv2 transforms)
    
    Returns:
        train_loader, val_loader: Training and validation data loaders
    """
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Default DINOv2 transforms if not provided
    if train_transform is None:
        from torchvision import transforms
        dinov2_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            dinov2_normalize,
        ])
    
    if val_test_transform is None:
        from torchvision import transforms
        dinov2_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            dinov2_normalize,
        ])
    
    # Load CIFAR-10 training data
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Split training data into train and validation
    train_size = int((1 - valid_size) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create data loaders with optimized settings for DINOv2
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Optimized for DINOv2
        pin_memory=True,
        drop_last=True  # Important for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f" DINOv2 CIFAR-10 Data Loaders Created:")
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Train samples: {len(train_dataset)}")
    logger.info(f"   Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader

def cifar10_test(batch_size=128, shuffle=False, transform=None):
    """
    Get CIFAR-10 test data loader with DINOv2-optimized settings.
    
    Args:
        batch_size: Batch size for testing
        shuffle: Whether to shuffle the test data
        transform: Custom transform (if None, use default DINOv2 test transform)
    
    Returns:
        test_loader: Test data loader
    """
    
    # Default DINOv2 test transform if not provided
    if transform is None:
        from torchvision import transforms
        dinov2_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            dinov2_normalize,
        ])
    
    # Load CIFAR-10 test data
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f" DINOv2 CIFAR-10 Test Loader Created:")
    logger.info(f"   Test batches: {len(test_loader)}")
    logger.info(f"   Test samples: {len(test_dataset)}")
    
    return test_loader 