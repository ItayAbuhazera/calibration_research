import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Optional, Tuple

from utils.logging_config import get_logger
logger = get_logger(__name__)

def constellation_loss(features, labels, alpha=1.0, beta=1.0, gamma=1.0, **kwargs):
    """
    Constellation Loss: Multi-scale geometric optimization combining:
    - Centroid separation (maximize inter-class distances)  
    - Intra-class compactness (minimize intra-class distances)
    - Individual separation (fast separation approximation)
    
    Args:
        features: Feature tensor [B, D]
        labels: Target labels [B]
        alpha: Weight for centroid separation
        beta: Weight for intra-class compactness  
        gamma: Weight for individual separation
    """
    device = features.device
    features = F.normalize(features, dim=1)  # Note: Normalization implies cosine similarity
    batch_size = features.size(0)
    
    if labels.device != device:
        labels = labels.to(device)
    
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    
    if num_classes <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 1. Compute class centroids
    centroids = []
    valid_classes = []
    for label_val in unique_labels:
        mask = (labels == label_val)
        if mask.sum() > 0:
            centroid = F.normalize(torch.mean(features[mask], dim=0), dim=0)
            centroids.append(centroid)
            valid_classes.append(label_val)
    
    if len(centroids) <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    centroids = torch.stack(centroids)
    
    # 2. Centroid separation loss (maximize minimum inter-class distance)
    centroid_dists = torch.cdist(centroids, centroids, p=2)
    eye_mask = torch.eye(len(centroids), device=device).bool()
    masked_dists = centroid_dists.masked_fill(eye_mask, float('inf'))
    min_centroid_dist = torch.min(masked_dists)
    centroid_separation_loss = -min_centroid_dist  # Negative to maximize distance
    
    # 3. Intra-class compactness loss
    compactness_loss = torch.tensor(0.0, device=device)
    total_samples = 0
    
    for i, label_val in enumerate(valid_classes):
        mask = (labels == label_val)
        if mask.sum() > 1:
            class_features = features[mask]
            centroid = centroids[i]
            distances = torch.norm(class_features - centroid, dim=1, p=2)
            compactness_loss += distances.sum()
            total_samples += mask.sum()
    
    if total_samples > 0:
        compactness_loss = compactness_loss / total_samples
    
    # 4. Individual separation loss (vectorized)
    dist_matrix = torch.cdist(features, features, p=2)
    same_class_mask = (labels[:, None] == labels[None, :]) & (~torch.eye(batch_size, device=device).bool())
    diff_class_mask = (labels[:, None] != labels[None, :])
    
    min_same_dist = torch.min(dist_matrix.masked_fill(~same_class_mask, float('inf')), dim=1)[0]
    min_diff_dist = torch.min(dist_matrix.masked_fill(~diff_class_mask, float('inf')), dim=1)[0]
    
    delta_i = 0.5 * (min_diff_dist - min_same_dist)
    valid_mask = (same_class_mask.sum(dim=1) > 0) & (diff_class_mask.sum(dim=1) > 0)
    
    if valid_mask.sum() > 0:
        separation_loss = -delta_i[valid_mask].mean()  # Negative to maximize separation
    else:
        separation_loss = torch.tensor(0.0, device=device)
    
    # Combine losses
    total_loss = (alpha * centroid_separation_loss + 
                  beta * compactness_loss + 
                  gamma * separation_loss)
    
    return total_loss

def geometric_focal_calibration_loss(features, logits, targets, gamma=3.0, lambda_geo=1.0, **kwargs):
    """
    Geometric Focal Calibration Loss combining:
    - Focal loss for classification
    - Geometric separation for calibration
    
    Args:
        features: Feature tensor [B, D]
        logits: Model logits [B, C]
        targets: Target labels [B]
        gamma: Focal loss gamma parameter
        lambda_geo: Weight for geometric calibration loss
    """
    device = features.device
    
    # 1. Focal classification loss
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
    
    # 2. Geometric calibration component
    probabilities = F.softmax(logits, dim=1)
    max_probs, predictions = torch.max(probabilities, dim=1)
    
    calibration_loss = torch.tensor(0.0, device=device)
    batch_size = features.size(0)
    
    if batch_size > 1:
        dist_matrix = torch.cdist(features, features, p=2)
        
        for i in range(batch_size):
            pred_class = predictions[i]
            confidence = max_probs[i]
            
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

def fast_separation_loss(features, labels, **kwargs):
    """
    Fast-Separation Loss: Encourages better separation of features in the latent space.
    
    Args:
        features: Feature tensor [B, D]
        labels: Target labels [B]
    """
    device = features.device
    features = F.normalize(features, dim=1)  # Note: Normalization implies cosine similarity
    batch_size = features.size(0)
    
    if labels.device != device:
        labels = labels.to(device)
    
    if len(torch.unique(labels)) <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    dist_matrix = torch.cdist(features, features, p=2)
    
    same_class_mask = (labels[:, None] == labels[None, :]) & (~torch.eye(batch_size, device=device).bool())
    diff_class_mask = (labels[:, None] != labels[None, :])
    
    min_same_dist = torch.min(dist_matrix.masked_fill(~same_class_mask, float('inf')), dim=1)[0]
    min_diff_dist = torch.min(dist_matrix.masked_fill(~diff_class_mask, float('inf')), dim=1)[0]
    
    delta_i = 0.5 * (min_diff_dist - min_same_dist)
    valid_mask = (same_class_mask.sum(dim=1) > 0) & (diff_class_mask.sum(dim=1) > 0)
    
    if valid_mask.sum() > 0:
        loss = -delta_i[valid_mask].mean()
    else:
        loss = torch.tensor(0.0, device=device)
    
    return loss

def fast_separation_loss_wrapper(logits, targets, features=None, **kwargs):
    """Wrapper to integrate fast-separation loss with existing loss interface"""
    if features is None:
        features = logits
    return fast_separation_loss(features, targets, **kwargs)

def constellation_loss_wrapper(logits, targets, features=None, **kwargs):
    """Wrapper to integrate constellation loss with existing loss interface"""
    if features is None:
        features = logits
    return constellation_loss(features, targets, **kwargs)

def geometric_focal_wrapper(logits, targets, features=None, **kwargs):
    """Wrapper to integrate geometric focal loss with existing loss interface"""
    if features is None:
        features = logits
    return geometric_focal_calibration_loss(features, logits, targets, **kwargs)

# Add to existing loss function registry
GEOMETRIC_LOSS_FUNCTIONS = {
    'constellation': constellation_loss_wrapper,
    'geometric_focal': geometric_focal_wrapper,
    'constellation_separation': lambda logits, targets, **kwargs: constellation_loss_wrapper(
        logits, targets, alpha=1.0, beta=0.5, gamma=1.5, **kwargs
    ),
    'fast_separation': fast_separation_loss_wrapper,
}