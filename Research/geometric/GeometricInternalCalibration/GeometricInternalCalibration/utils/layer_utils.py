"""
Shared utilities for layer discovery and filtering.

Provides consistent filtering of activation and non-feature-bearing layers
across all layer discovery mechanisms.
"""
import torch.nn as nn
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Patterns to exclude from feature layer selection
EXCLUDED_LAYER_PATTERNS = [
    'relu', 'gelu', 'silu', 'leakyrelu', 'sigmoid', 'tanh', 
    'dropout', 'identity', 'activation'
]

# Module types to exclude (for isinstance checks)
EXCLUDED_LAYER_TYPES = (
    nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh,  # Activations
    nn.Dropout, nn.Dropout2d, nn.Dropout3d,  # Dropout
    nn.Identity,  # Identity
)


def is_feature_layer(name: str, module: Optional[nn.Module] = None) -> bool:
    """
    Check if layer contains learned features (not activations/dropout).
    
    Args:
        name: Layer name (e.g., "features.0.relu" or "layer1.0.conv1")
        module: Optional PyTorch module instance for type checking
    
    Returns:
        True if layer should be included, False if it's an activation/dropout
    """
    name_lower = name.lower()
    
    # Check name patterns
    if any(pattern in name_lower for pattern in EXCLUDED_LAYER_PATTERNS):
        return False
    
    # Check module type if provided
    if module is not None:
        if isinstance(module, EXCLUDED_LAYER_TYPES):
            return False
        module_type = type(module).__name__.lower()
        if any(pattern in module_type for pattern in EXCLUDED_LAYER_PATTERNS):
            return False
    
    return True


def filter_feature_layers(
    layers: List[Dict[str, Any]], 
    model: Optional[nn.Module] = None
) -> List[Dict[str, Any]]:
    """
    Filter out activation and non-feature layers from discovered layers.
    
    Args:
        layers: List of layer dictionaries with at least a 'name' key
        model: Optional model to check module types (if layers have 'idx' key)
    
    Returns:
        Filtered list of feature-bearing layers
    """
    if not layers:
        return []
    
    # Build module index mapping if model is provided
    idx_to_module = {}
    if model is not None:
        modules = list(model.modules())
        idx_to_module = {i: m for i, m in enumerate(modules)}
    
    filtered = []
    for layer in layers:
        layer_name = layer.get('name', '')
        layer_idx = layer.get('idx')
        
        # Get module if available
        module = None
        if layer_idx is not None and layer_idx in idx_to_module:
            module = idx_to_module[layer_idx]
        
        # Check if this is a feature layer
        if is_feature_layer(layer_name, module):
            filtered.append(layer)
    
    excluded_count = len(layers) - len(filtered)
    if excluded_count > 0:
        logger.info(f"Filtered {excluded_count} non-feature layers (activations, dropout, etc.)")
        excluded_names = [l.get('name', 'unknown') for l in layers if l not in filtered]
        if len(excluded_names) <= 10:
            logger.debug(f"Excluded layers: {excluded_names}")
        else:
            logger.debug(f"Excluded layers: {excluded_names[:10]}... (and {len(excluded_names) - 10} more)")
    
    return filtered

