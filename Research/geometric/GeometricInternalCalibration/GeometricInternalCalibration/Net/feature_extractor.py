import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from utils.logging_config import get_logger
logger = get_logger(__name__)

class FeatureExtractorWrapper(nn.Module):
    """Wrapper to extract features from a model."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.features = None
        self._register_feature_hook()
    
    def _register_feature_hook(self):
        """Register hook to extract features from the right layer."""
        modules = list(self.base_model.named_modules())
        
        # For ResNet, find the avgpool or the layer before FC
        target_layer = None
        for name, module in reversed(modules):
            # Skip the final Linear/classifier layer
            if isinstance(module, nn.Linear):
                continue
            # Look for pooling layers (these have rich features)
            elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                target_layer = module
                logger.info(f"🎯 Found pooling layer: {name}")
                break
            # For ResNet, look for the last conv block
            elif 'layer4' in name and isinstance(module, nn.Sequential):
                target_layer = module
                logger.info(f"🎯 Found conv block: {name}")
                break
        
        if target_layer is not None:
            target_layer.register_forward_hook(self._feature_hook)
            logger.info(f"🔗 Registered feature hook on: {target_layer}")
        else:
            logger.error("❌ Could not find suitable layer for feature extraction")
    
    def _feature_hook(self, module, input, output):
        """Hook to capture features."""
        self.features = output
    
    def forward(self, x, return_features=False):
        """Forward pass with optional feature return."""
        self.features = None
        logits = self.base_model(x)
        
        if return_features:
            if self.features is None:
                logger.warning("⚠️ No features captured, falling back to logits")
                return logits, logits
            
            # Reshape features if needed (for pooling layers)
            if len(self.features.shape) == 4:  # [B, C, H, W]
                features = self.features.mean(dim=[2, 3])  # Global average pooling
            else:
                features = self.features
            
            # Normalize features
            features = F.normalize(features, dim=1)
            return logits, features
        
        return logits 