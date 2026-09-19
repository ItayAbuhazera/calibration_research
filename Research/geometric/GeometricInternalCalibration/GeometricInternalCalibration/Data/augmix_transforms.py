# Data/augmix_transforms.py
"""
AugMix implementation following Hendrycks et al. (2020) - Generic version for multiple datasets
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import random

# Dataset normalization constants
DATASET_NORMALIZATION = {
    'cifar10': {
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010]
    },
    'cifar100': {
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761]
    },
    'svhn': {
        'mean': [0.4377, 0.4438, 0.4728],
        'std': [0.1980, 0.2010, 0.1970]
    },
    'tinyimagenet': {
        'mean': [0.4802, 0.4481, 0.3975],
        'std': [0.2770, 0.2691, 0.2821]
    },
    'pacs': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}

def get_normalization_params(dataset_name):
    """
    Get normalization parameters for a given dataset.
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'cifar100', 'svhn')
    
    Returns:
        tuple: (mean, std) for normalization
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_NORMALIZATION:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_NORMALIZATION.keys())}")
    
    norm_params = DATASET_NORMALIZATION[dataset_name]
    return norm_params['mean'], norm_params['std']

class AugMixTransforms:
    """
    AugMix augmentation strategy with mixing chains - Generic version for multiple datasets
    """
    
    def __init__(self, severity=3, width=3, depth=-1, alpha=1.0, dataset_name=None, 
                 mean=None, std=None):
        """
        Args:
            severity: Severity of augmentations (1-10)
            width: Number of augmentation chains to mix
            depth: Depth of augmentation chains (-1 for random 1-3)
            alpha: Parameter for Beta distribution mixing
            dataset_name: Name of dataset ('cifar10', 'cifar100', 'svhn') - used to auto-detect normalization
            mean: Custom normalization mean (overrides dataset_name)
            std: Custom normalization std (overrides dataset_name)
        """
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        
        # Determine normalization parameters
        if mean is not None and std is not None:
            # Use custom normalization values
            self.mean = mean
            self.std = std
            print(f"🔧 Using custom normalization: mean={mean}, std={std}")
        elif dataset_name is not None:
            # Auto-detect from dataset name
            self.mean, self.std = get_normalization_params(dataset_name)
            print(f"🔧 Using {dataset_name.upper()} normalization: mean={self.mean}, std={self.std}")
        else:
            # Fallback to CIFAR-10 (backward compatibility)
            self.mean, self.std = get_normalization_params('cifar10')
            print("⚠️ No dataset specified, falling back to CIFAR-10 normalization")
        
        # Define augmentation operations (following AugMix paper)
        self.augment_ops = [
            self.autocontrast, self.equalize, self.posterize, self.rotate,
            self.solarize, self.shear_x, self.shear_y, self.translate_x,
            self.translate_y, self.brightness, self.color, self.contrast, 
            self.sharpness, self.identity, self.invert
        ]
    
    def __call__(self, image):
        """Apply AugMix efficiently with minimal conversions"""
        # Convert to tensor once
        if not isinstance(image, torch.Tensor):
            image_tensor = transforms.ToTensor()(image)
        else:
            image_tensor = image
            
        # Create augmented versions with normalization inside mixing
        aug1_tensor = self._apply_augmix_tensor(image_tensor)
        aug2_tensor = self._apply_augmix_tensor(image_tensor)
        
        # Apply normalization to clean image using dataset-specific values
        normalize = transforms.Normalize(self.mean, self.std)
        
        return (
            normalize(image_tensor),
            aug1_tensor,  # Already normalized inside _apply_augmix_tensor
            aug2_tensor   # Already normalized inside _apply_augmix_tensor
        )
    
    def _apply_augmix_tensor(self, image_tensor):
        """Apply AugMix operations with normalization AFTER mixing (paper-faithful)"""
        # Convert to PIL once and cache it
        image_pil = transforms.ToPILImage()(image_tensor)
        
        # Sample mixing weights
        ws = np.random.dirichlet([self.alpha] * self.width)
        m = np.random.beta(self.alpha, self.alpha)
        
        # Create augmentation chains WITHOUT normalization
        mixed_tensor = torch.zeros_like(image_tensor)
        
        for i in range(self.width):
            aug_pil = image_pil.copy()
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            
            for _ in range(depth):
                op = np.random.choice(self.augment_ops)
                aug_pil = op(aug_pil)
            
            aug_tensor = transforms.ToTensor()(aug_pil)
            # Mix WITHOUT normalization (paper-faithful)
            mixed_tensor += ws[i] * aug_tensor
        
        # Final mixing with original (still no normalization)
        result = (1 - m) * image_tensor + m * mixed_tensor
        
        # Apply normalization AFTER all mixing is complete (paper-faithful)
        normalize = transforms.Normalize(self.mean, self.std)
        return normalize(torch.clamp(result, 0, 1))
    
    # Augmentation operations (updated implementations)
    def autocontrast(self, img):
        return ImageOps.autocontrast(img)
    
    def equalize(self, img):
        return ImageOps.equalize(img)
    
    def posterize(self, img):
        """Paper-faithful posterize: maps severity 1-10 to bits 8-4 exactly"""
        # AugMix paper uses AutoAugment ranges: level 0-9 maps to bits 8-4
        # Our severity 1-10 should map the same way
        bits = max(1, min(8, 8 - int((self.severity - 1) * 4 / 9)))
        return ImageOps.posterize(img, bits)
    
    def rotate(self, img):
        """Enhanced rotation with exact paper ranges"""
        # AugMix paper: rotation up to 30 degrees
        degrees = (self.severity / 10.0) * 30.0
        angle = np.random.uniform(-degrees, degrees)
        # Use white fill instead of gray for consistency with paper
        return img.rotate(angle, fillcolor=(255, 255, 255))
    
    def solarize(self, img):
        # Original maps severity to threshold 256->0
        threshold = int(256 - (self.severity / 10.0) * 256)
        threshold = max(0, min(255, threshold))
        return ImageOps.solarize(img, threshold)
    
    def invert(self, img):
        """Invert operation - inverts pixel values"""
        return ImageOps.invert(img)
    
    def brightness(self, img):
        """Enhanced brightness with exact AutoAugment ranges"""
        # AutoAugment brightness: factor range [0.1, 1.9]
        factor = 0.1 + (self.severity / 10.0) * 1.8
        factor = max(0.1, min(1.9, factor))  # Ensure bounds
        return ImageEnhance.Brightness(img).enhance(factor)
    
    def color(self, img):
        factor = 0.1 + (self.severity / 10.0) * 1.8
        factor = max(0.1, min(1.9, factor))  # Ensure bounds
        return ImageEnhance.Color(img).enhance(factor)
    
    def contrast(self, img):
        factor = 0.1 + (self.severity / 10.0) * 1.8
        factor = max(0.1, min(1.9, factor))  # Ensure bounds
        return ImageEnhance.Contrast(img).enhance(factor)
    
    def sharpness(self, img):
        factor = 0.1 + (self.severity / 10.0) * 1.8
        factor = max(0.1, min(1.9, factor))  # Ensure bounds
        return ImageEnhance.Sharpness(img).enhance(factor)
    
    def shear_x(self, img):
        shear = np.random.uniform(-0.3, 0.3) * self.severity / 10
        return img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0), 
                            fillcolor=(128, 128, 128))
    
    def shear_y(self, img):
        shear = np.random.uniform(-0.3, 0.3) * self.severity / 10
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))
    
    def translate_x(self, img):
        shift = np.random.uniform(-0.45, 0.45) * self.severity / 10 * img.size[0]
        return img.transform(img.size, Image.AFFINE, (1, 0, shift, 0, 1, 0))
    
    def translate_y(self, img):
        shift = np.random.uniform(-0.45, 0.45) * self.severity / 10 * img.size[1]
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, shift))

    def identity(self, img):
        """Identity operation - returns image unchanged"""
        return img

def jensen_shannon_loss(logits_clean, logits_aug1, logits_aug2, temperature=1.0):
    """
    Paper-faithful Jensen-Shannon divergence with enhanced numerical stability
    """
    # Apply temperature scaling
    logits_clean = logits_clean / temperature
    logits_aug1 = logits_aug1 / temperature  
    logits_aug2 = logits_aug2 / temperature
    
    # Convert to probabilities with log-sum-exp stability
    def stable_softmax(logits):
        logits_max = torch.max(logits, dim=1, keepdim=True)[0]
        logits_stable = logits - logits_max
        return F.softmax(logits_stable, dim=1)
    
    p_clean = stable_softmax(logits_clean)
    p_aug1 = stable_softmax(logits_aug1)
    p_aug2 = stable_softmax(logits_aug2)
    
    # Mixture distribution M = (P1 + P2 + P3) / 3
    p_mixture = (p_clean + p_aug1 + p_aug2) / 3.0
    
    # Clamp for numerical stability
    eps = 1e-8
    p_mixture = torch.clamp(p_mixture, min=eps, max=1.0)
    p_clean = torch.clamp(p_clean, min=eps, max=1.0)
    p_aug1 = torch.clamp(p_aug1, min=eps, max=1.0)
    p_aug2 = torch.clamp(p_aug2, min=eps, max=1.0)
    
    # Jensen-Shannon divergence: (KL(P1||M) + KL(P2||M) + KL(P3||M)) / 3
    def kl_div_safe(p, q):
        """Safe KL divergence computation"""
        return torch.sum(p * torch.log(p / q), dim=1).mean()
    
    js_loss = (kl_div_safe(p_clean, p_mixture) + 
               kl_div_safe(p_aug1, p_mixture) + 
               kl_div_safe(p_aug2, p_mixture)) / 3.0
    
    return js_loss