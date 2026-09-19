# Data/cifar10_c.py
"""
CIFAR-10-C evaluation protocol following Hendrycks & Dietterich (2019)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from typing import List, Tuple, Optional

class CIFAR10C(Dataset):
    """
    CIFAR-10-C dataset with 15 corruption types at 5 severity levels
    
    Corruption types: gaussian_noise, shot_noise, impulse_noise, defocus_blur,
    glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness,
    contrast, elastic_transform, pixelate, jpeg_compression
    """
    
    # Define all 15 corruption types as in the original paper
    CORRUPTION_TYPES = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    def __init__(self, 
                 root: str, 
                 corruption_type: str,
                 severity: int = 5,  # 1-5, where 5 is most severe
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            root: Path to CIFAR-10-C dataset directory
            corruption_type: One of the 15 corruption types
            severity: Corruption severity level (1-5)
            transform: Optional transforms (usually just normalization for eval)
        """
        super().__init__()
        
        if corruption_type not in self.CORRUPTION_TYPES:
            raise ValueError(f"Unknown corruption: {corruption_type}")
        
        if severity not in range(1, 6):
            raise ValueError(f"Severity must be 1-5, got {severity}")
        
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform
        
        # Load corruption data (following CIFAR-10-C format)
        data_path = os.path.join(root, f"{corruption_type}.npy")
        labels_path = os.path.join(root, "labels.npy")
        
        # Each corruption file contains all 5 severity levels
        # Shape: [50000, 32, 32, 3] for each severity level
        all_data = np.load(data_path)
        all_labels = np.load(labels_path)
        
        # Extract data for specific severity level
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        
        self.data = all_data[start_idx:end_idx]
        self.targets = all_labels[start_idx:end_idx]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            # Convert numpy array to PIL Image for transforms
            from PIL import Image
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            # Convert to tensor
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img, target

def get_cifar10c_loader(root: str,
                       corruption_type: str,
                       severity: int = 5,
                       batch_size: int = 128,
                       num_workers: int = 4) -> DataLoader:
    """
    Get CIFAR-10-C data loader for specific corruption and severity
    
    Returns:
        DataLoader for evaluation (no data augmentation)
    """
    # Use same normalization as CIFAR-10 training
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    dataset = CIFAR10C(
        root=root,
        corruption_type=corruption_type,
        severity=severity,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for evaluation
        num_workers=0,
        pin_memory=True
    )
    
    return loader

def evaluate_all_corruptions(model, 
                           cifar10c_root: str,
                           device: torch.device,
                           severity: int = 5) -> dict:
    """
    Evaluate model on all 15 CIFAR-10-C corruption types
    
    Returns:
        Dictionary with results for each corruption type
    """
    from Metrics.metrics import test_classification_net, expected_calibration_error
    
    results = {}
    
    for corruption in CIFAR10C.CORRUPTION_TYPES:
        print(f"📊 Evaluating {corruption} at severity {severity}...")
        
        loader = get_cifar10c_loader(
            root=cifar10c_root,
            corruption_type=corruption,
            severity=severity
        )
        
        # Test model on this corruption
        _, accuracy, labels, predictions, confidences = test_classification_net(
            model, loader, device
        )
        
        # Calculate ECE
        ece = expected_calibration_error(confidences, predictions, labels)
        
        results[corruption] = {
            'accuracy': accuracy,
            'ece': ece,
            'num_samples': len(labels)
        }
        
        print(f"   Accuracy: {accuracy:.2f}%, ECE: {ece:.4f}")
    
    # Calculate mean performance across all corruptions
    mean_accuracy = np.mean([r['accuracy'] for r in results.values()])
    mean_ece = np.mean([r['ece'] for r in results.values()])
    
    results['mean'] = {
        'accuracy': mean_accuracy,
        'ece': mean_ece
    }
    
    print(f"\n📈 Mean Corruption Performance:")
    print(f"   Accuracy: {mean_accuracy:.2f}%, ECE: {mean_ece:.4f}")
    
    return results