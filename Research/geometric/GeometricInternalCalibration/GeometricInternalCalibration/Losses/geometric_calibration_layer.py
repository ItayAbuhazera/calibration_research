"""
Geometric Calibration Layer - integrates with existing model architectures
Works with existing training loops and data loaders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Dict, Any

from utils.logging_config import get_logger
logger = get_logger(__name__)


class SimpleFeatureExtractor:
    """Extract features from common architectures"""
    
    def __init__(self, model, architecture='resnet'):
        self.model = model
        self.architecture = architecture.lower()
        self.features = None
        self._hook_registered = False
    
    def register_hook(self):
        """Register hook to extract features before final layer"""
        if self._hook_registered:
            return
            
        if 'resnet' in self.architecture:
            if hasattr(self.model, 'avgpool'):
                self.model.avgpool.register_forward_hook(self._feature_hook)
            elif hasattr(self.model, 'fc'):
                # Find the layer before FC
                for name, module in self.model.named_modules():
                    if 'avgpool' in name or ('layer4' in name and len(list(module.children())) == 0):
                        module.register_forward_hook(self._feature_hook)
                        break
        elif 'densenet' in self.architecture:
            if hasattr(self.model, 'features'):
                self.model.features.register_forward_hook(self._feature_hook)
        
        self._hook_registered = True
    
    def _feature_hook(self, module, input, output):
        """Hook to capture features"""
        if isinstance(output, torch.Tensor):
            if len(output.shape) > 2:
                self.features = torch.flatten(output, 1)
            else:
                self.features = output
    
    def extract_features(self, x):
        """Extract features by running forward pass"""
        if not self._hook_registered:
            self.register_hook()
        
        self.features = None
        logits = self.model(x)
        return logits, self.features


class GeometricCalibrationLayer(nn.Module):
    """Geometric calibration layer using stability scores"""
    
    def __init__(self, num_classes, warmup_batches=50):
        super().__init__()
        self.num_classes = num_classes
        self.warmup_batches = warmup_batches
        self.batch_count = 0
        self.calibrated = False
        
        # Simple linear mapping from stability to confidence
        self.stability_to_confidence = nn.Linear(1, 1)
        nn.init.normal_(self.stability_to_confidence.weight, mean=1.0, std=0.1)
        nn.init.constant_(self.stability_to_confidence.bias, 0.0)
        
        # Feature bank for stability computation
        self.register_buffer('feature_bank', torch.empty(0, 0))
        self.register_buffer('label_bank', torch.empty(0, dtype=torch.long))
        self.max_bank_size = 2000
        
        # Calibration tracking
        self.register_buffer('calibration_errors', torch.empty(0))
        
    def update_feature_bank(self, features, labels):
        """Update feature bank for stability computation"""
        if not self.training:
            return
            
        features = features.detach()
        labels = labels.detach()
        device = features.device
        
        if self.feature_bank.numel() == 0:
            self.feature_bank = features
            self.label_bank = labels
        else:
            self.feature_bank = torch.cat([self.feature_bank.to(device), features], dim=0)
            self.label_bank = torch.cat([self.label_bank.to(device), labels], dim=0)
            
            if len(self.feature_bank) > self.max_bank_size:
                self.feature_bank = self.feature_bank[-self.max_bank_size:]
                self.label_bank = self.label_bank[-self.max_bank_size:]
    
    def compute_stability_scores(self, features, predictions):
        """Compute simplified stability scores"""
        batch_size = features.size(0)
        device = features.device
        
        if self.feature_bank.numel() == 0 or len(self.feature_bank) < 10:
            return torch.zeros(batch_size, device=device)
        
        stability_scores = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            query_feature = features[i]
            predicted_class = predictions[i]
            
            # Find distances to same and different class samples in bank
            same_class_mask = (self.label_bank == predicted_class)
            diff_class_mask = (self.label_bank != predicted_class)
            
            if same_class_mask.sum() > 0 and diff_class_mask.sum() > 0:
                same_features = self.feature_bank[same_class_mask]
                diff_features = self.feature_bank[diff_class_mask]
                
                same_dists = torch.norm(query_feature.unsqueeze(0) - same_features, dim=1, p=2)
                diff_dists = torch.norm(query_feature.unsqueeze(0) - diff_features, dim=1, p=2)
                
                min_same = torch.min(same_dists)
                min_diff = torch.min(diff_dists)
                
                if min_diff > min_same:
                    stability_scores[i] = (min_diff - min_same) / 2.0
        
        return stability_scores
    
    def forward(self, logits, features=None, targets=None):
        """Apply geometric calibration"""
        device = logits.device
        batch_size = logits.size(0)
        
        # Get base predictions
        base_probs = F.softmax(logits, dim=1)
        max_probs, predictions = torch.max(base_probs, dim=1)
        
        # Update training state
        if self.training:
            self.batch_count += 1
            if self.batch_count >= self.warmup_batches:
                self.calibrated = True
        
        # If no features provided or not calibrated yet, return base probabilities
        if features is None or not self.calibrated:
            return base_probs
        
        # Update feature bank during training
        if self.training and targets is not None:
            self.update_feature_bank(features, targets)
        
        # Compute stability scores
        stability_scores = self.compute_stability_scores(features, predictions)
        
        # Apply calibration mapping
        raw_confidence = self.stability_to_confidence(stability_scores.unsqueeze(1)).squeeze(1)
        calibrated_confidence = torch.sigmoid(raw_confidence)
        calibrated_confidence = torch.clamp(calibrated_confidence, 0.1, 0.95)
        
        # Create calibrated probability distribution
        calibrated_probs = base_probs.clone()
        
        for i in range(batch_size):
            pred_class = predictions[i]
            new_confidence = calibrated_confidence[i]
            
            # Set predicted class probability
            calibrated_probs[i, pred_class] = new_confidence
            
            # Redistribute remaining probability
            remaining_prob = 1.0 - new_confidence
            other_classes = torch.arange(self.num_classes, device=device) != pred_class
            
            if other_classes.sum() > 0:
                other_probs = base_probs[i, other_classes]
                other_sum = other_probs.sum()
                
                if other_sum > 1e-8:
                    calibrated_probs[i, other_classes] = other_probs * (remaining_prob / other_sum)
                else:
                    calibrated_probs[i, other_classes] = remaining_prob / other_classes.sum()
        
        # Track calibration quality
        if self.training and targets is not None:
            with torch.no_grad():
                actual_correct = (predictions == targets).float()
                cal_errors = torch.abs(calibrated_confidence - actual_correct)
                
                if len(self.calibration_errors) > 1000:
                    self.calibration_errors = self.calibration_errors[-500:]
                self.calibration_errors = torch.cat([self.calibration_errors.to(device), cal_errors])
        
        return calibrated_probs


class GeometricModelWrapper(nn.Module):
    """Wrapper to add geometric calibration to existing models"""
    
    def __init__(self, base_model, num_classes, architecture='resnet'):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.architecture = architecture
        
        # Feature extractor
        self.feature_extractor = SimpleFeatureExtractor(base_model, architecture)
        
        # Geometric calibration layer
        self.calibration_layer = GeometricCalibrationLayer(num_classes)
        
    def forward(self, x, targets=None, return_components=False):
        """Forward pass with geometric calibration"""
        # Get logits and features
        logits, features = self.feature_extractor.extract_features(x)
        
        # Apply geometric calibration
        if self.calibration_layer.calibrated and features is not None:
            calibrated_probs = self.calibration_layer(logits, features, targets)
            
            if return_components:
                return {
                    'logits': logits,
                    'features': features,
                    'calibrated_probabilities': calibrated_probs,
                    'calibration_stats': {
                        'calibrated': True,
                        'feature_bank_size': len(self.calibration_layer.feature_bank)
                    }
                }
            return calibrated_probs
        else:
            # During warmup, still update calibration layer but return logits
            if features is not None:
                _ = self.calibration_layer(logits, features, targets)
            
            if return_components:
                return {
                    'logits': logits,
                    'features': features,
                    'calibrated_probabilities': F.softmax(logits, dim=1),
                    'calibration_stats': {
                        'calibrated': False,
                        'warmup_batch': self.calibration_layer.batch_count
                    }
                }
            return logits


def create_geometric_model(base_model, num_classes, architecture='resnet'):
    """Factory function to create geometric calibration model"""
    return GeometricModelWrapper(base_model, num_classes, architecture)


def apply_geometric_calibration_to_existing_model(model, num_classes, architecture='resnet'):
    """Apply geometric calibration to an existing trained model"""
    # Create wrapper
    geo_model = GeometricModelWrapper(model, num_classes, architecture)
    
    # Copy weights
    if hasattr(model, 'state_dict'):
        geo_model.base_model.load_state_dict(model.state_dict())
    
    return geo_model