"""
Extended loss functions including novel geometric losses
Builds on existing loss.py structure
"""

import torch
from torch.nn import functional as F
from Losses.focal_loss import FocalLoss
from Losses.focal_loss_adaptive_gamma import FocalLossAdaptive
from Losses.mmce import MMCE, MMCE_weighted
from Losses.brier_score import BrierScore

# Import new geometric losses
from Losses.geometric_losses import (
    constellation_loss_wrapper, 
    geometric_focal_wrapper,
    GEOMETRIC_LOSS_FUNCTIONS,
    fast_separation_loss
)

# Keep all existing loss functions
def cross_entropy(logits, targets, **kwargs):
    return F.cross_entropy(logits, targets, reduction='sum')

def focal_loss(logits, targets, **kwargs):
    return FocalLoss(gamma=kwargs['gamma'])(logits, targets)

def focal_loss_adaptive(logits, targets, **kwargs):
    return FocalLossAdaptive(gamma=kwargs['gamma'],
                             device=kwargs['device'])(logits, targets)

def mmce(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE(kwargs['device'])(logits, targets)
    return ce + (kwargs['lamda'] * mmce)

def mmce_weighted(logits, targets, **kwargs):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE_weighted(kwargs['device'])(logits, targets)
    return ce + (kwargs['lamda'] * mmce)

def brier_score(logits, targets, **kwargs):
    return BrierScore()(logits, targets)

# NEW: Geometric loss functions
# def constellation_loss(logits, targets, **kwargs):
#     """Constellation loss with multi-scale geometric optimization"""
#     # Extract features if available, otherwise use logits
#     features = kwargs.get('features', logits)
#     return constellation_loss_wrapper(logits, targets, features=features, **kwargs)

def geometric_focal_loss(logits, targets, **kwargs):
    """Geometric focal calibration loss"""
    features = kwargs.get('features', logits)
    return geometric_focal_wrapper(logits, targets, features=features, **kwargs)

def constellation_separation_loss(logits, targets, **kwargs):
    """Constellation loss with emphasis on separation"""
    features = kwargs.get('features', logits)
    return constellation_loss_wrapper(
        logits, targets, features=features, 
        alpha=1.0, beta=0.5, gamma=1.5, **kwargs
    )

def combined_geometric_loss(logits, targets, **kwargs):
    """Combined loss: cross-entropy + geometric calibration"""
    ce_loss = cross_entropy(logits, targets, **kwargs)
    geo_loss = constellation_loss(logits, targets, **kwargs)
    lambda_geo = kwargs.get('lambda_geo', 0.5)
    return ce_loss + lambda_geo * geo_loss

def dual_focal_loss(logits, targets, **kwargs):
    """Dual focal loss for calibration"""
    from Losses.dual_focal_loss import DualFocalLoss
    gamma = kwargs.get('gamma', 0)
    return DualFocalLoss(gamma=gamma)(logits, targets)
"""
Dataset	     Model	    γ
cifar100	resnet50	5
cifar100	resnet110	6.1
cifar100	wide_resnet	3.9
cifar100	densenet121	3.4
-------------	--------------	-------
cifar10	resnet50	5
cifar10	resnet110	4.5
cifar10	wide_resnet	2.6
cifar10	densenet121	5
-------------	--------------	-------
tiny_imagenet	resnet50	2.3
"""


# Convenience function to get loss function
def get_loss_function(loss_name, **kwargs):
    """
    Get loss function by name
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional parameters for the loss function
    
    Returns:
        Loss function
    """
    if loss_name not in LOSS_FUNCTIONS:
        available = list(LOSS_FUNCTIONS.keys())
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {available}")
    
    loss_fn = LOSS_FUNCTIONS[loss_name]
    
    # Return wrapper that includes kwargs
    def loss_wrapper(logits, targets, **additional_kwargs):
        combined_kwargs = {**kwargs, **additional_kwargs}
        return loss_fn(logits, targets, **combined_kwargs)
    
    return loss_wrapper
def constellation_loss_original(logits, targets, **kwargs):
    """Original constellation loss (gamma=0) - baseline for comparison"""
    features = kwargs.get('features', logits)  # Use logits as features if not provided
    return _constellation_loss_impl(features, targets, 
                                  alpha=kwargs.get('const_alpha', 1.0),
                                  beta=kwargs.get('const_beta', 1.0), 
                                  gamma=0.0)  # KEY: gamma=0 for original

def constellation_loss(logits, targets, **kwargs):
    """Enhanced constellation loss with fast separation (gamma>0)"""
    features = kwargs.get('features', logits)
    return _constellation_loss_impl(features, targets,
                                  alpha=kwargs.get('const_alpha', 1.0),
                                  beta=kwargs.get('const_beta', 1.0),
                                  gamma=kwargs.get('const_gamma', 1.0))

def geometric_focal_calibration_loss(logits, targets, **kwargs):
    """Geometric focal calibration loss"""
    features = kwargs.get('features', logits)
    device = features.device
    
    # 1. Focal classification loss
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    gamma = kwargs.get('gamma', 3.0)
    focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
    
    # 2. Geometric calibration component
    lambda_geo = kwargs.get('lambda_geo', 1.0)
    probabilities = F.softmax(logits, dim=1)
    max_probs, predictions = torch.max(probabilities, dim=1)
    
    # Simple geometric calibration using feature distances
    calibration_loss = torch.tensor(0.0, device=device)
    batch_size = features.size(0)
    
    if batch_size > 1:
        dist_matrix = torch.cdist(features, features, p=2)
        
        for i in range(batch_size):
            pred_class = predictions[i]
            confidence = max_probs[i]
            
            # Find nearest neighbors of same and different predicted classes
            same_class_mask = (predictions == pred_class) & (torch.arange(batch_size, device=device) != i)
            diff_class_mask = (predictions != pred_class)
            
            if same_class_mask.sum() > 0 and diff_class_mask.sum() > 0:
                min_same_dist = torch.min(dist_matrix[i, same_class_mask])
                min_diff_dist = torch.min(dist_matrix[i, diff_class_mask])
                
                if min_diff_dist > min_same_dist:
                    geometric_confidence = torch.sigmoid((min_diff_dist - min_same_dist))
                    calibration_error = torch.abs(confidence - geometric_confidence)
                    calibration_loss += calibration_error
        
        calibration_loss = calibration_loss / batch_size
    
    return focal_loss + lambda_geo * calibration_loss

def _constellation_loss_impl(features, labels, alpha=1.0, beta=1.0, gamma=1.0):
    """Fully GPU-optimized constellation loss with vectorized operations"""
    device = features.device
    features = F.normalize(features, dim=1)
    batch_size = features.size(0)
    
    if labels.device != device:
        labels = labels.to(device)
    
    unique_labels, inverse_indices = torch.unique(labels, return_inverse=True)
    num_classes = len(unique_labels)
    
    if num_classes <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 1. 🚀 VECTORIZED: Compute class centroids
    # Create one-hot encoding for class memberships
    class_mask = labels.unsqueeze(1) == unique_labels.unsqueeze(0)  # [batch_size, num_classes]
    class_counts = class_mask.sum(dim=0).float()  # [num_classes]
    
    # Filter out empty classes
    valid_classes_mask = class_counts > 0
    if valid_classes_mask.sum() <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    valid_class_mask = class_mask[:, valid_classes_mask].float()  # ✅ Convert to float!
    valid_class_counts = class_counts[valid_classes_mask]  # [valid_classes]
    
    # Compute centroids: sum of features for each class / class count
    centroids = torch.mm(features.t(), valid_class_mask) / valid_class_counts.unsqueeze(0)  # [feature_dim, valid_classes]
    centroids = F.normalize(centroids.t(), dim=1)  # [valid_classes, feature_dim]
    
    num_valid_classes = centroids.size(0)
    if num_valid_classes <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 2. Centroid separation loss (maximize minimum inter-class distance)
    centroid_dists = torch.cdist(centroids, centroids, p=2)
    eye_mask = torch.eye(num_valid_classes, device=device) * 1e9
    masked_dists = centroid_dists + eye_mask
    min_centroid_dist = torch.min(masked_dists)
    centroid_separation_loss = -min_centroid_dist
    
    # 3. 🚀 VECTORIZED: Intra-class compactness loss
    compactness_loss = torch.tensor(0.0, device=device)
    
    # For each valid class, compute distances to centroid
    for i in range(num_valid_classes):
        class_i_mask = valid_class_mask[:, i] > 0.5  # ✅ Convert back to boolean
        if class_i_mask.sum() > 1:  # Only if class has more than 1 sample
            class_features = features[class_i_mask]  # [class_size, feature_dim]
            centroid_i = centroids[i]  # [feature_dim]
            distances = torch.norm(class_features - centroid_i, dim=1, p=2)  # [class_size]
            compactness_loss += distances.sum()
    
    # Normalize by total number of samples
    compactness_loss = compactness_loss / batch_size
    
    # 4. 🚀 FULLY VECTORIZED: Individual separation loss
    separation_loss = torch.tensor(0.0, device=device)
    if gamma > 0:
        # Compute all pairwise distances at once
        dist_matrix = torch.cdist(features, features, p=2)  # [batch_size, batch_size]
        
        # Create vectorized masks for same/different classes
        labels_expanded = labels.unsqueeze(1)  # [batch_size, 1]
        labels_tiled = labels.unsqueeze(0)     # [1, batch_size]
        
        same_class_mask = (labels_expanded == labels_tiled)  # [batch_size, batch_size]
        diff_class_mask = ~same_class_mask                   # [batch_size, batch_size]
        
        # Exclude self-comparisons (diagonal)
        eye_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        same_class_mask = same_class_mask & ~eye_mask
        
        # 🚀 Vectorized minimum distance computation
        # For same-class: mask invalid entries with infinity
        same_dist_masked = dist_matrix.masked_fill(~same_class_mask, float('inf'))
        min_same_dists, _ = torch.min(same_dist_masked, dim=1)  # [batch_size]
        
        # For different-class: mask invalid entries with infinity
        diff_dist_masked = dist_matrix.masked_fill(~diff_class_mask, float('inf'))
        min_diff_dists, _ = torch.min(diff_dist_masked, dim=1)  # [batch_size]
        
        # Check which samples have valid neighbors
        has_same_neighbors = same_class_mask.sum(dim=1) > 0   # [batch_size]
        has_diff_neighbors = diff_class_mask.sum(dim=1) > 0   # [batch_size]
        valid_samples = has_same_neighbors & has_diff_neighbors  # [batch_size]
        
        if valid_samples.sum() > 0:
            # Get valid distances
            valid_same_dists = min_same_dists[valid_samples]  # [valid_count]
            valid_diff_dists = min_diff_dists[valid_samples]  # [valid_count]
            
            # Only consider samples with positive separation (diff > same)
            positive_separation = valid_diff_dists > valid_same_dists  # [valid_count]
            
            if positive_separation.sum() > 0:
                # Compute fast separation scores
                pos_same_dists = valid_same_dists[positive_separation]
                pos_diff_dists = valid_diff_dists[positive_separation]
                fast_separations = (pos_diff_dists - pos_same_dists) / 2.0
                
                # Average separation loss (maximize separation = minimize negative)
                separation_loss = -fast_separations.mean()
    
    # Combine all loss components
    total_loss = (alpha * centroid_separation_loss + 
                  beta * compactness_loss + 
                  gamma * separation_loss)
    
    return total_loss


# ✅ FIXED: Correct function name without asterisks
def constellation_loss_impl(features, labels, alpha=1.0, beta=1.0, gamma=1.0):
    """Public wrapper for constellation loss implementation."""
    return _constellation_loss_impl(features, labels, alpha, beta, gamma)

# Loss function configurations for experiments
LOSS_CONFIGS = {
    'cross_entropy': {},
    'focal_loss': {'gamma': 2.0},
    'constellation': {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0},
    'constellation_loss_original': {'const_alpha': 1.0, 'const_beta': 1.0, 'const_gamma': 0.0},
    'geometric_focal': {'gamma': 3.0, 'lambda_geo': 1.0},
    'constellation_separation': {'alpha': 1.0, 'beta': 0.5, 'gamma': 1.5},
    'combined_geometric': {'lambda_geo': 0.5},
}

# Extended loss function registry
LOSS_FUNCTIONS = {
    # Existing losses
    'cross_entropy': cross_entropy,
    'focal_loss': focal_loss,
    'focal_loss_adaptive': focal_loss_adaptive,
    'mmce': mmce,
    'mmce_weighted': mmce_weighted,
    'brier_score': brier_score,
    'dual_focal_loss': dual_focal_loss,
    
    # NEW: Geometric losses
    'constellation': constellation_loss,
    'geometric_focal': geometric_focal_loss,
    'constellation_separation': constellation_separation_loss,
    'combined_geometric': combined_geometric_loss,

}

__all__ = [
    'cross_entropy', 'focal_loss', 'focal_loss_adaptive',
    'mmce', 'mmce_weighted', 'brier_score',
    'dual_focal_loss', 'constellation_loss',
    'constellation_loss_original', 'geometric_focal_calibration_loss',
    'fast_separation_loss', 'constellation_loss_impl'
]
