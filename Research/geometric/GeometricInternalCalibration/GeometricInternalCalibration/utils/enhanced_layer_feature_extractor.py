"""
Enhanced Layer Feature Extractor for Layer Selection Optimization Experiments
Supports forced layer selection and systematic evaluation across architecture layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import time
import os

# Import memory safety components
from utils.shm_retry_wrapper import (
    SharedMemoryMonitor, MemorySafeDataLoader, retry_on_memory_error,
    setup_memory_safe_pytorch, cleanup_all_torch_memory, get_adaptive_batch_size,
    enable_emergency_recovery_mode, create_memory_safe_hash
)

from utils.logging_config import get_logger
logger = get_logger(__name__)


def process_batch_with_retry(
    model: torch.nn.Module, 
    data: torch.Tensor, 
    device: torch.device,
    base_batch_size: int = 32,
    min_batch_size: int = 4
) -> torch.Tensor:
    """
    Process a batch with automatic retry and batch size reduction for CUDA OOM.
    
    This function handles CUDA out-of-memory errors separately from shared memory
    errors by reducing batch size and retrying.
    
    Args:
        model: PyTorch model to run
        data: Input data tensor [B, C, H, W]
        device: Device to run on
        base_batch_size: Initial batch size to try
        min_batch_size: Minimum batch size before giving up
        
    Returns:
        Model output tensor
    """
    monitor = SharedMemoryMonitor()
    current_batch_size = min(base_batch_size, data.shape[0])
    
    while current_batch_size >= min_batch_size:
        try:
            # Check memory before processing
            memory_usage = monitor.get_usage_percentage()
            if memory_usage > 80:
                logger.warning(f"High memory usage ({memory_usage:.1f}%) before batch processing")
                cleanup_all_torch_memory()
            
            # Process in chunks if batch is larger than current batch size
            if data.shape[0] <= current_batch_size:
                # Single batch processing
                logger.debug(f"Processing single batch of size {data.shape[0]} with limit {current_batch_size}")
                return model(data)
            else:
                # Multi-chunk processing
                logger.debug(f"Processing {data.shape[0]} samples in chunks of {current_batch_size}")
                outputs = []
                
                for i in range(0, data.shape[0], current_batch_size):
                    chunk = data[i:i + current_batch_size]
                    chunk_output = model(chunk)
                    outputs.append(chunk_output)
                    
                    # Clear cache between chunks
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                return torch.cat(outputs, dim=0)
                
        except RuntimeError as e:
            error_msg = str(e).lower()
            
            # Handle CUDA OOM separately
            if "out of memory" in error_msg or "cuda out of memory" in error_msg:
                logger.warning(f"CUDA OOM with batch size {current_batch_size}: {e}")
                
                # Reduce batch size
                current_batch_size = max(current_batch_size // 2, min_batch_size)
                logger.info(f"Reducing batch size to {current_batch_size}")
                
                # Clean up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cleanup_all_torch_memory()
                
                continue
            else:
                # Other runtime errors - re-raise
                logger.error(f"Non-OOM runtime error in batch processing: {e}")
                raise
                
        except Exception as e:
            # Non-runtime errors - re-raise
            logger.error(f"Non-runtime error in batch processing: {e}")
            raise
    
    # If we get here, all batch sizes failed
    logger.error(f"Batch processing failed with all sizes from {base_batch_size} to {min_batch_size}")
    raise RuntimeError(f"Failed to process batch with any size from {base_batch_size} to {min_batch_size}")


class EnhancedLayerFeatureExtractor:
    """
    Enhanced feature extractor with systematic layer selection capabilities.
    
    Supports:
    1. Forced layer selection for systematic experiments
    2. Multi-layer extraction for comparative analysis
    3. Architecture-specific layer candidates
    4. Performance metrics collection
    
    Compatible with original FeatureExtractor interface for plug-and-play usage.
    """
    
    def __init__(self, model, model_name: str, forced_layer: Optional[str] = None, use_compression: bool = True, enable_caching: bool = True, cache_dir: Optional[str] = None):
        """
        Initialize enhanced feature extractor.
        
        Args:
            model: Neural network model
            model_name: Name of the model architecture (e.g., 'resnet18', 'densenet121')
            forced_layer: Specific layer to extract from (for systematic experiments)
            use_compression: Whether to use spatial compression (True) or keep full dimensions (False)
            enable_caching: Whether to enable feature caching for expensive extractions
            cache_dir: Directory for feature cache (default: ./feature_cache)
        """
        self.model = model
        self.model_name = model_name.lower()
        self.forced_layer = forced_layer
        self.use_compression = use_compression  # 🔧 RENAMED: use_spp → use_compression
        self.enable_caching = enable_caching
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./feature_cache")
        
        # Create cache directory if caching is enabled
        if self.enable_caching:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"🗂️ Feature caching enabled: {self.cache_dir}")
        
        # Layer candidates per architecture
        self.layer_candidates = self._get_layer_candidates()
        
        # Current extraction state
        self.current_layer = forced_layer
        self.layer_features = {}
        self.hooks = []
        self.extraction_times = {}
        
        # 🔧 COMPATIBILITY: Add same interface as original FeatureExtractor
        self.features = None  # Current batch features (for compatibility)
        self.hook = None      # Single hook reference (for compatibility)
        
        # Performance tracking
        self.extraction_metrics = {
            'feature_dimensions': {},
            'extraction_times': {},
            'memory_usage': {},
            'layer_availability': {},
            'cache_hits': {},
            'cache_misses': {}
        }
        
        # 🔧 NEW: Initialize GPU memory management
        self._initialize_gpu_memory_management()
        
        # Initialize extraction based on forced layer
        if self.forced_layer:
            self._initialize_extraction()
        
        # logger.info(f"🔧 EnhancedLayerFeatureExtractor initialized:")
        # logger.info(f"   Model: {self.model_name}")
        # logger.info(f"   Forced layer: {self.forced_layer}")
        # logger.info(f"   Use compression: {self.use_compression}")
        # logger.info(f"   Enable caching: {self.enable_caching}")
        # logger.info(f"   Available layers: {list(self.layer_candidates.keys())}")
        # logger.info(f"   ✅ Compatible with original FeatureExtractor interface")
    
    def _initialize_gpu_memory_management(self):
        """
        🔧 NEW: Initialize GPU memory management to prevent CUDA errors.
        """
        if torch.cuda.is_available():
            # Clear any existing cache
            torch.cuda.empty_cache()
            
            # Set memory management options
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False  # Disable for consistent memory usage
            
            # Set conservative memory allocation
            try:
                # Disable expandable segments which cause the CUDA error
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:False'
                
                # Force memory pool initialization
                dummy = torch.zeros(1, device='cuda')
                del dummy
                torch.cuda.empty_cache()
                
                logger.debug("🔧 GPU memory management initialized")
                
            except Exception as e:
                logger.warning(f"🔧 GPU memory management setup failed: {e}")
    
    def _get_layer_candidates(self) -> Dict[str, Dict[str, Any]]:
        """
        🔧 UPDATED: Get layer candidates with improved DenseNet support.
        """
        candidates = {}
        
        if 'resnet' in self.model_name:
            # ResNet family - existing logic remains the same
            if 'resnet18' in self.model_name:
                candidates = {
                    'layer1': {
                        'expected_dim': 64,
                        'rationale': 'Early features, basic patterns',
                        'depth': 1
                    },
                    'layer2': {
                        'expected_dim': 128,
                        'rationale': 'Mid-level features, moderate complexity',
                        'depth': 2
                    },
                    'layer3': {
                        'expected_dim': 256,
                        'rationale': 'High-level features, good discrimination',
                        'depth': 3
                    },
                    'layer4': {
                        'expected_dim': 512,
                        'rationale': 'Final conv layer, maximum specialization',
                        'depth': 4
                    },
                    'layer4.1': {
                        'expected_dim': 512,
                        'rationale': 'Final residual block, highest semantic features',
                        'depth': 5
                    }
                }
            elif 'resnet50' in self.model_name:
                candidates = {
                    'layer1': {'expected_dim': 256, 'rationale': 'Early features', 'depth': 1},
                    'layer2': {'expected_dim': 512, 'rationale': 'Mid-level features', 'depth': 2},
                    'layer3': {'expected_dim': 1024, 'rationale': 'High-level features', 'depth': 3},
                    'layer4': {'expected_dim': 2048, 'rationale': 'Final conv layer', 'depth': 4},
                    'layer4.1': {'expected_dim': 2048, 'rationale': 'Final residual block', 'depth': 5}
                }
            else:
                # Generic ResNet
                candidates = {
                    'layer1': {'expected_dim': 'variable', 'rationale': 'Early features', 'depth': 1},
                    'layer2': {'expected_dim': 'variable', 'rationale': 'Mid-level features', 'depth': 2},
                    'layer3': {'expected_dim': 'variable', 'rationale': 'High-level features', 'depth': 3},
                    'layer4': {'expected_dim': 'variable', 'rationale': 'Final conv layer', 'depth': 4},
                    'layer4.1': {'expected_dim': 'variable', 'rationale': 'Final residual block', 'depth': 5}
                }
        
        elif 'densenet' in self.model_name:
            # 🔧 IMPROVED: DenseNet support for both standard and custom architectures
            
            # Check model structure to determine naming convention
            if hasattr(self.model, 'features'):
                # Standard torchvision DenseNet
                # logger.info("🔍 Detected standard torchvision DenseNet")
                candidates = {
                    'denseblock1': {
                        'expected_dim': 'variable',
                        'rationale': 'Early dense features, basic patterns',
                        'depth': 1
                    },
                    'transition1': {
                        'expected_dim': 'variable', 
                        'rationale': 'First transition, feature compression',
                        'depth': 2
                    },
                    'denseblock2': {
                        'expected_dim': 'variable',
                        'rationale': 'Mid-level dense features',
                        'depth': 3
                    },
                    'transition2': {
                        'expected_dim': 'variable',
                        'rationale': 'Second transition, further compression', 
                        'depth': 4
                    },
                    'denseblock3': {
                        'expected_dim': 'variable',
                        'rationale': 'High-level dense features, optimal balance',
                        'depth': 5
                    },
                    'transition3': {
                        'expected_dim': 'variable',
                        'rationale': 'Final transition, maximum compression',
                        'depth': 6
                    },
                    'denseblock4': {
                        'expected_dim': 'variable',
                        'rationale': 'Final dense features, maximum specialization',
                        'depth': 7
                    },
                    'final_norm': {
                        'expected_dim': 1024,
                        'rationale': 'Final normalization, ready for classification',
                        'depth': 8
                    }
                }
            else:
                # Custom DenseNet architecture (like yours)
                # logger.info("🔍 Detected custom DenseNet architecture")
                candidates = {
                    # Support both naming conventions
                    'dense1': {
                        'expected_dim': 'variable',
                        'rationale': 'First dense block, early features',
                        'depth': 1
                    },
                    'denseblock1': {  # Alias for dense1
                        'expected_dim': 'variable',
                        'rationale': 'First dense block, early features', 
                        'depth': 1
                    },
                    'trans1': {
                        'expected_dim': 'variable',
                        'rationale': 'First transition, feature compression',
                        'depth': 2
                    },
                    'transition1': {  # Alias for trans1
                        'expected_dim': 'variable',
                        'rationale': 'First transition, feature compression',
                        'depth': 2
                    },
                    'dense2': {
                        'expected_dim': 'variable',
                        'rationale': 'Second dense block, mid-level features',
                        'depth': 3
                    },
                    'denseblock2': {  # Alias for dense2
                        'expected_dim': 'variable',
                        'rationale': 'Second dense block, mid-level features',
                        'depth': 3
                    },
                    'trans2': {
                        'expected_dim': 'variable',
                        'rationale': 'Second transition, further compression',
                        'depth': 4
                    },
                    'transition2': {  # Alias for trans2
                        'expected_dim': 'variable',
                        'rationale': 'Second transition, further compression',
                        'depth': 4
                    },
                    'dense3': {
                        'expected_dim': 'variable',
                        'rationale': 'Third dense block, high-level features',
                        'depth': 5
                    },
                    'denseblock3': {  # Alias for dense3
                        'expected_dim': 'variable',
                        'rationale': 'Third dense block, high-level features',
                        'depth': 5
                    },
                    'trans3': {
                        'expected_dim': 'variable',
                        'rationale': 'Final transition, maximum compression',
                        'depth': 6
                    },
                    'transition3': {  # Alias for trans3
                        'expected_dim': 'variable',
                        'rationale': 'Final transition, maximum compression',
                        'depth': 6
                    },
                    'dense4': {
                        'expected_dim': 'variable',
                        'rationale': 'Final dense block, maximum specialization',
                        'depth': 7
                    },
                    'denseblock4': {  # Alias for dense4
                        'expected_dim': 'variable',
                        'rationale': 'Final dense block, maximum specialization',
                        'depth': 7
                    },
                    'bn': {
                        'expected_dim': 'variable',
                        'rationale': 'Final batch normalization',
                        'depth': 8
                    },
                    'final_norm': {  # Alias for bn
                        'expected_dim': 'variable',
                        'rationale': 'Final normalization, ready for classification',
                        'depth': 8
                    }
                }
        
        elif 'dinov2' in self.model_name:
            # 🔧 NEW: DINOv2 support for transformer-based models
            candidates = {
                'block_0': {
                    'expected_dim': 384,  # DINOv2 small
                    'rationale': 'Early transformer block, basic patterns',
                    'depth': 1
                },
                'block_2': {
                    'expected_dim': 384,
                    'rationale': 'Early-mid transformer block',
                    'depth': 2
                },
                'block_5': {
                    'expected_dim': 384,
                    'rationale': 'Mid-level transformer block, good balance',
                    'depth': 3
                },
                'block_8': {
                    'expected_dim': 384,
                    'rationale': 'Mid-late transformer block, high-level features',
                    'depth': 4
                },
                'block_11': {
                    'expected_dim': 384,
                    'rationale': 'Late transformer block, maximum specialization',
                    'depth': 5
                },
                'norm': {
                    'expected_dim': 384,
                    'rationale': 'Final normalization layer',
                    'depth': 6
                }
            }
        
        else:
            # Generic architecture
            candidates = {
                'features': {'expected_dim': 'variable', 'rationale': 'Generic features', 'depth': 1},
                'pooling': {'expected_dim': 'variable', 'rationale': 'Pooling layer', 'depth': 2}
            }
        
        return candidates
    
    def _initialize_extraction(self):
        """Initialize the feature extraction system."""
        if self.forced_layer:
            # Single layer extraction for systematic experiments
            self._register_single_layer_hook(self.forced_layer)
        else:
            # Multi-layer extraction for analysis
            self._register_multi_layer_hooks()
    
    def _register_single_layer_hook(self, layer_name: str):
        """Register hook for a single specific layer."""
        target_module = self._find_layer_module(layer_name)
        if target_module:
            hook = target_module.register_forward_hook(self._make_hook(layer_name))
            self.hooks.append(hook)
            
            # 🔧 COMPATIBILITY: Set single hook reference for original interface
            self.hook = hook
            
            # logger.info(f"🔗 Registered single layer hook: {layer_name}")
        else:
            logger.error(f"❌ Could not find layer: {layer_name}")
    
    def _register_multi_layer_hooks(self):
        """Register hooks for multiple layers."""
        for layer_name in self.layer_candidates.keys():
            target_module = self._find_layer_module(layer_name)
            if target_module:
                hook = target_module.register_forward_hook(self._make_hook(layer_name))
                self.hooks.append(hook)
                logger.info(f"🔗 Registered multi-layer hook: {layer_name}")
    
    def _find_layer_module(self, layer_name: str) -> Optional[nn.Module]:
        """
        Enhanced layer finding with architecture-specific logic.
        
        Args:
            layer_name: Name of the layer to find
            
        Returns:
            The module corresponding to the layer, or None if not found
        """
        # Unified logging for all architectures
        # logger.info(f"🔍 Looking for layer '{layer_name}' in model {self.model_name}")
        # logger.info(f"   Available attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}")
        
        if 'resnet' in self.model_name:
            return self._find_resnet_layer(layer_name)
        elif 'densenet' in self.model_name:
            return self._find_densenet_layer(layer_name)
        elif 'dinov2' in self.model_name:
            return self._find_dinov2_layer(layer_name)
        else:
            # Fallback for generic models
            logger.warning(f"⚠️ Unknown model architecture '{self.model_name}'. Using generic layer search.")
            return self._find_generic_layer(layer_name)
    
    def _find_resnet_layer(self, layer_name: str) -> Optional[nn.Module]:
        """Find ResNet layer with proper hierarchy navigation."""
        # Direct attribute access for ResNet layers (most reliable)
        if layer_name == 'layer1' and hasattr(self.model, 'layer1'):
            return self.model.layer1
        elif layer_name == 'layer2' and hasattr(self.model, 'layer2'):
            return self.model.layer2
        elif layer_name == 'layer3' and hasattr(self.model, 'layer3'):
            return self.model.layer3
        elif layer_name == 'layer4' and hasattr(self.model, 'layer4'):
            return self.model.layer4
        elif layer_name == 'avgpool' and hasattr(self.model, 'avgpool'):
            return self.model.avgpool
        
        # Handle sub-layer cases (e.g., layer4.1, layer3.1, etc.)
        if '.' in layer_name:
            main_layer, sub_layer = layer_name.split('.')
            if hasattr(self.model, main_layer):
                main_module = getattr(self.model, main_layer)
                if hasattr(main_module, sub_layer):
                    return getattr(main_module, sub_layer)
                # Try to access by index if it's a Sequential
                elif hasattr(main_module, '__getitem__'):
                    try:
                        sub_index = int(sub_layer)
                        return main_module[sub_index]
                    except (ValueError, IndexError):
                        pass
        
        # Fallback: search through named modules with exact matching
        for name, module in self.model.named_modules():
            if layer_name == name:
                return module
        
        # Last resort: fuzzy matching for edge cases
        for name, module in self.model.named_modules():
            if layer_name.lower() in name.lower() and any(
                pattern in name.lower() for pattern in ['layer', 'pool', 'conv']
            ):
                # logger.info(f"🔍 Found layer '{layer_name}' via fuzzy matching: {name}")
                return module
        
        logger.warning(f"⚠️ Could not find ResNet layer: {layer_name}")
        logger.warning(f"   Available layers: {[name for name, _ in self.model.named_modules() if 'layer' in name or 'pool' in name]}")
        return None
    
    def _find_densenet_layer(self, layer_name: str) -> Optional[nn.Module]:
        """
        🔧 FIXED: Find DenseNet layer with support for custom architecture.
        
        Supports both:
        1. Standard torchvision: features.denseblock1, features.transition1
        2. Custom architecture: dense1, dense2, trans1, trans2
        """
        # First, check if this is a standard torchvision DenseNet
        if hasattr(self.model, 'features'):
            # logger.info("🔍 Found standard torchvision DenseNet with 'features' module")
            return self._find_standard_densenet_layer(layer_name)
        
        # If no 'features' module, check for custom DenseNet architecture
        # logger.info("🔍 Detected custom DenseNet architecture without 'features' module")
        
        # Map layer names to custom DenseNet attributes
        custom_layer_mapping = {
            # Dense blocks
            'denseblock1': ['dense1'],
            'denseblock2': ['dense2'], 
            'denseblock3': ['dense3'],
            'denseblock4': ['dense4'],
            
            # Transitions
            'transition1': ['trans1'],
            'transition2': ['trans2'],
            'transition3': ['trans3'],
            
            # Alternative names (in case of different naming)
            'dense1': ['dense1'],
            'dense2': ['dense2'],
            'dense3': ['dense3'], 
            'dense4': ['dense4'],
            'trans1': ['trans1'],
            'trans2': ['trans2'],
            'trans3': ['trans3'],
            
            # Final normalization
            'final_norm': ['bn', 'norm', 'final_bn', 'final_norm'],
            'norm': ['bn', 'norm'],
            'bn': ['bn']
        }
        
        # logger.info(f"🔍 Looking for layer '{layer_name}' in custom DenseNet")
        # logger.info(f"   Available attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}")
        
        if layer_name in custom_layer_mapping:
            candidate_names = custom_layer_mapping[layer_name]
            
            for candidate_name in candidate_names:
                if hasattr(self.model, candidate_name):
                    module = getattr(self.model, candidate_name)
                    # logger.info(f"✅ Found custom DenseNet layer '{layer_name}' as attribute '{candidate_name}'")
                    # logger.info(f"   Module type: {type(module).__name__}")
                    return module
                else:
                    logger.debug(f"   ❌ Candidate '{candidate_name}' not found")
        
        # If direct mapping fails, try fuzzy matching
        # logger.info(f"🔍 Trying fuzzy matching for '{layer_name}'")
        for attr_name in dir(self.model):
            if attr_name.startswith('_'):
                continue
                
            # Check if the attribute name contains our target layer name
            if layer_name.lower() in attr_name.lower():
                module = getattr(self.model, attr_name)
                if isinstance(module, nn.Module):
                    # logger.info(f"✅ Found layer '{layer_name}' via fuzzy matching: '{attr_name}'")
                    # logger.info(f"   Module type: {type(module).__name__}")
                    return module
        
        # Last resort: search through all modules
        # logger.info(f"🔍 Searching through all named modules for '{layer_name}'")
        for name, module in self.model.named_modules():
            if layer_name in name and len(list(module.children())) > 0:
                # logger.info(f"✅ Found layer '{layer_name}' in named modules: '{name}'")
                # logger.info(f"   Module type: {type(module).__name__}")
                return module
        
        # If we still can't find it, provide detailed debugging info
        logger.error(f"❌ Could not find DenseNet layer: {layer_name}")
        logger.error(f"   Available model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_') and hasattr(self.model, attr)]}")
        logger.error(f"   Available named modules (first 10):")
        for i, (name, module) in enumerate(self.model.named_modules()):
            if i < 10:
                logger.error(f"      {name}: {type(module).__name__}")
            else:
                logger.error(f"      ... (and {len(list(self.model.named_modules())) - 10} more)")
                break
        
        return None

    def _find_standard_densenet_layer(self, layer_name: str) -> Optional[nn.Module]:
        """
        Find layer in standard torchvision DenseNet (with features module).
        """
        if not hasattr(self.model, 'features'):
            return None
        
        features = self.model.features
        
        # Map layer names to actual module paths with multiple candidates
        layer_mapping = {
            'denseblock1': ['denseblock1'],
            'transition1': ['transition1'],
            'denseblock2': ['denseblock2'], 
            'transition2': ['transition2'],
            'denseblock3': ['denseblock3'],
            'transition3': ['transition3'],
            'denseblock4': ['denseblock4'],
            'final_norm': ['norm5', 'norm', 'final_norm']
        }
        
        if layer_name in layer_mapping:
            for candidate_name in layer_mapping[layer_name]:
                # Try direct attribute access first
                if hasattr(features, candidate_name):
                    # logger.info(f"✅ Found standard DenseNet layer '{layer_name}' as features.{candidate_name}")
                    return getattr(features, candidate_name)
        
        # Fallback: search through all features submodules
        for name, module in features.named_children():
            if layer_name in name:
                # logger.info(f"✅ Found DenseNet layer '{layer_name}' via submodule search: features.{name}")
                return module
        
        # Last resort: search through all model modules
        for name, module in self.model.named_modules():
            if layer_name in name and any(
                pattern in name.lower() for pattern in ['denseblock', 'transition', 'norm']
            ):
                # logger.info(f"✅ Found DenseNet layer '{layer_name}' via full search: {name}")
                return module
        
        logger.warning(f"⚠️ Could not find standard DenseNet layer: {layer_name}")
        logger.warning(f"   Available features submodules: {[name for name, _ in features.named_children()]}")
        return None
    
    def _find_generic_layer(self, layer_name: str) -> Optional[nn.Module]:
        """Find layer in generic architecture with intelligent search."""
        # First try exact name matching
        for name, module in self.model.named_modules():
            if layer_name == name:
                return module
        
        # Then try attribute access
        if hasattr(self.model, layer_name):
            return getattr(self.model, layer_name)
        
        # Finally try fuzzy matching
        for name, module in self.model.named_modules():
            if layer_name.lower() in name.lower():
                # logger.info(f"🔍 Found generic layer '{layer_name}' via fuzzy matching: {name}")
                return module
        
        logger.warning(f"⚠️ Could not find generic layer: {layer_name}")
        logger.warning(f"   Available modules: {[name for name, _ in list(self.model.named_modules())[:10]]}...")
        return None
    
    def _make_hook(self, layer_name: str):
        """Create a forward hook for feature extraction with smart spatial handling."""
        def hook_fn(module, input, output):
            start_time = time.time()
            
            # Handle tuple outputs from transformer blocks (like DINOv2)
            features_tensor = None
            if isinstance(output, tuple):
                if len(output) > 0 and isinstance(output[0], torch.Tensor):
                    features_tensor = output[0]
            elif isinstance(output, torch.Tensor):
                features_tensor = output

            if features_tensor is None:
                logger.warning(f"⚠️ Hook for layer '{layer_name}' received an unexpected output type: {type(output)}")
                return

            # 🔧 DEBUG: Add tensor dimension debugging for DenseNet layers
            if 'dense' in layer_name.lower() and len(features_tensor.shape) == 4:
                self.debug_tensor_dimensions(features_tensor, layer_name)
            
            # Apply smart reduction which handles 4D, 3D, and 2D tensors
            features = self._smart_spatial_reduction(features_tensor, layer_name)
            
            # Store features in layer-specific dictionary
            self.layer_features[layer_name] = features.detach()
            
            # 🔧 COMPATIBILITY: Set features attribute for original interface
            # Only set if this is the forced layer or if no forced layer is specified
            if self.forced_layer is None or layer_name == self.forced_layer:
                self.features = features.detach()
            
            # Record extraction time
            extraction_time = (time.time() - start_time) * 1000  # Convert to ms
            self.extraction_times[layer_name] = extraction_time
            
            # Record feature dimensions
            if features is not None and features.shape is not None and len(features.shape) > 1:
                self.extraction_metrics['feature_dimensions'][layer_name] = features.shape[1]
        
        return hook_fn
    
    def _smart_spatial_reduction(self, features: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        🔧 FIXED: Smart spatial reduction with enhanced memory management and consistent dimensions.
        
        Args:
            features: [B, C, H, W] tensor or [B, hidden_dim] for transformers
            layer_name: Name of the layer for strategy selection
            
        Returns:
            Reduced features (compressed) or flattened features (no compression)
        """
        batch_size = features.shape[0]
        
        # Clear GPU cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Handle 3D transformer features [B, seq_len, hidden_dim]
        if len(features.shape) == 3:
            original_dims = features.shape[1] * features.shape[2]
            # logger.info(f"🔧 {layer_name}: 3D transformer features detected {features.shape} ({original_dims:,} dims)")
            # Select the CLS token, which is the first token in the sequence
            features = features[:, 0, :] # Shape: [B, hidden_dim]
            # logger.info(f"   Selected CLS token. New shape: {features.shape}")
        
        # Handle 1D transformer features (like DINOv2's CLS token)
        if len(features.shape) == 2:  # [B, hidden_dim]
            original_dims = features.shape[1]
            # logger.info(f"🔧 {layer_name}: 1D transformer features detected ({original_dims:,} dims)")
            
            # Strategy selection for 1D features
            if not self.use_compression:
                # NO COMPRESSION: Keep all dimensions
                # logger.info(f"🔧 {layer_name}: No compression - keeping all {original_dims:,} dims")
                return features
            else:
                # COMPRESSION: Use PCA-like reduction for 1D features
                target_dims = self._get_1d_compression_target(layer_name, original_dims)
                
                if original_dims <= target_dims:
                    # logger.info(f"🔧 {layer_name}: No compression needed ({original_dims:,} ≤ {target_dims:,})")
                    return features
                else:
                    # Apply compression to 1D features
                    compressed_features = self._compress_1d_features(features, target_dims)
                    reduction_factor = original_dims / target_dims
                    # logger.info(f"🔧 {layer_name}: 1D compression {original_dims:,} → {target_dims:,} dims (reduction: {reduction_factor:.1f}x)")
                    return compressed_features
        
        # Handle 4D spatial features (original logic)
        if len(features.shape) != 4:
            # logger.warning(f"⚠️ {layer_name}: Expected 4D tensor but got {len(features.shape)}D. Flattening.")
            return torch.flatten(features, 1)

        batch_size, channels, height, width = features.shape
        original_dims = channels * height * width
        
        # Strategy selection based on layer and compression setting
        strategy = self._get_spatial_strategy(layer_name, original_dims)
        
        try:
            if strategy['method'] == 'flatten':
                # NO COMPRESSION: Keep all spatial dimensions
                result = torch.flatten(features, 1)
                logger.info(f"🔧 {layer_name}: No compression - flattened to {result.shape[1]:,} dims")
                
            elif strategy['method'] == 'spatial_pyramid_pooling':
                result = self._spatial_pyramid_pooling(features, strategy['levels'])
                reduction_factor = original_dims / result.shape[1] if result.shape[1] > 0 else float('inf')
                logger.info(f"🔧 {layer_name}: SPP compression {original_dims:,} → {result.shape[1]:,} dims (reduction: {reduction_factor:.1f}x)")
                
            elif strategy['method'] == 'adaptive_pooling':
                result = self._adaptive_spatial_pooling(features, strategy['target_size'])
                reduction_factor = original_dims / result.shape[1] if result.shape[1] > 0 else float('inf')
                logger.info(f"🔧 {layer_name}: Adaptive compression {original_dims:,} → {result.shape[1]:,} dims (reduction: {reduction_factor:.1f}x)")
                
            elif strategy['method'] == 'pca_spatial_preserving':
                result = self._pca_spatial_preserving(features, strategy['target_dims'])
                reduction_factor = original_dims / result.shape[1] if result.shape[1] > 0 else float('inf')
                logger.info(f"🔧 {layer_name}: PCA spatial compression {original_dims:,} → {result.shape[1]:,} dims (reduction: {reduction_factor:.1f}x)")
                
            elif strategy['method'] == 'global_avg_pooling':
                result = features.mean(dim=[2, 3])
                reduction_factor = original_dims / result.shape[1] if result.shape[1] > 0 else float('inf')
                logger.info(f"🔧 {layer_name}: Global avg compression {original_dims:,} → {result.shape[1]:,} dims (reduction: {reduction_factor:.1f}x)")
                
            else:
                # Fallback: flatten with warning
                logger.warning(f"⚠️ Unknown strategy '{strategy['method']}' for {layer_name} - using flattening")
                result = torch.flatten(features, 1)
                logger.info(f"🔧 {layer_name}: Fallback flatten to {result.shape[1]:,} dims")
            
            # Clear GPU cache after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return result
            
        except Exception as e:
            logger.error(f"🔧 Spatial reduction failed for {layer_name}: {e}")
            # Emergency fallback: simple flattening
            result = torch.flatten(features, 1)
            logger.warning(f"🔧 {layer_name}: Emergency fallback flatten to {result.shape[1]:,} dims")
            return result
    
    def _get_1d_compression_target(self, layer_name: str, original_dims: int) -> int:
        """
        Get target dimensions for 1D feature compression.
        
        Args:
            layer_name: Name of the layer
            original_dims: Original number of dimensions
            
        Returns:
            Target number of dimensions after compression
        """
        # Conservative compression targets for DINOv2 layers
        if 'block_0' in layer_name or 'block_2' in layer_name:
            return min(512, original_dims)  # Early blocks: moderate compression
        elif 'block_5' in layer_name or 'block_8' in layer_name:
            return min(1024, original_dims)  # Middle blocks: light compression
        elif 'block_11' in layer_name or 'norm' in layer_name:
            return min(2048, original_dims)  # Late blocks: minimal compression
        else:
            return min(1024, original_dims)  # Default: moderate compression

    def _compress_1d_features(self, features: torch.Tensor, target_dims: int) -> torch.Tensor:
        """
        Compress 1D features using random projection.
        
        Args:
            features: [B, original_dims] tensor
            target_dims: Target number of dimensions
            
        Returns:
            Compressed features [B, target_dims]
        """
        batch_size, original_dims = features.shape
        
        if original_dims <= target_dims:
            return features
        
        # Use random projection for compression
        try:
            # Create random projection matrix
            projection = torch.randn(original_dims, target_dims, 
                                   device=features.device, dtype=features.dtype)
            
            # Normalize projection matrix
            projection = projection / torch.norm(projection, dim=0, keepdim=True)
            
            # Apply projection
            compressed = features @ projection
            
            return compressed
            
        except Exception as e:
            logger.warning(f"🔧 Random projection failed: {e}, using simple truncation")
            # Fallback: simple truncation
            return features[:, :target_dims]
    
    def _calculate_compressed_dimensions(self, features_shape: tuple, strategy: dict) -> int:
        """
        Calculate the exact output dimensions for a given compression strategy.
        
        Args:
            features_shape: (B, C, H, W) shape of input features
            strategy: Compression strategy dictionary
            
        Returns:
            Number of output dimensions after compression
        """
        if len(features_shape) != 4:
            return features_shape[1] if len(features_shape) > 1 else 1
            
        batch_size, channels, height, width = features_shape
        
        if strategy['method'] == 'flatten':
            return channels * height * width
        
        elif strategy['method'] == 'spatial_pyramid_pooling':
            levels = strategy.get('levels', [1, 2, 4])
            total_pooled_elements = sum(level * level for level in levels)
            return channels * total_pooled_elements
        
        elif strategy['method'] == 'adaptive_pooling':
            target_size = strategy.get('target_size', 4)
            return channels * target_size * target_size
        
        elif strategy['method'] == 'pca_spatial_preserving':
            return strategy.get('target_dims', 1024)
        
        elif strategy['method'] == 'global_avg_pooling':
            return channels
        
        else:
            # Fallback: assume same as flatten
            return channels * height * width
    
    def _estimate_channels_from_layer(self, layer_name: str) -> int:
        """
        Estimate the number of channels for a given layer based on architecture.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Estimated number of channels
        """
        if 'resnet18' in self.model_name:
            channel_map = {
                'layer1': 64,
                'layer2': 128, 
                'layer3': 256,
                'layer4': 512,
                'layer4.1': 512
            }
        elif 'resnet50' in self.model_name:
            channel_map = {
                'layer1': 256,
                'layer2': 512,
                'layer3': 1024, 
                'layer4': 2048,
                'layer4.1': 2048
            }
        elif 'densenet' in self.model_name:
            # DenseNet channels vary, use conservative estimates
            channel_map = {
                'dense1': 256, 'denseblock1': 256,
                'trans1': 128, 'transition1': 128,
                'dense2': 512, 'denseblock2': 512,
                'trans2': 256, 'transition2': 256,
                'dense3': 1024, 'denseblock3': 1024,
                'trans3': 512, 'transition3': 512,
                'dense4': 1024, 'denseblock4': 1024,
                'bn': 1024, 'final_norm': 1024
            }
        else:
            # Generic fallback
            channel_map = {}
        
        return channel_map.get(layer_name, 512)  # Conservative default
    
    def _get_spatial_strategy(self, layer_name: str, original_dims: int) -> dict:
        """
        🔧 FIXED: Better strategy selection that prevents SPP dimension increases
        """
        
        # 🔧 FIX: For DenseNet layers, respect the `use_compression` flag.
        # When `use_compression` is False, we should always flatten to get raw features.
        if 'densenet' in self.model_name and not self.use_compression:
            logger.info(f"🔧 DenseNet layer '{layer_name}' without compression: flattened to {original_dims:,} dims")
            return {
                'method': 'flatten',
                'reason': f"No compression for DenseNet layer {layer_name}"
            }

        # 🔧 DENSENET LAYERS: Use more conservative compression
        if layer_name in ['dense1', 'denseblock1']:
            logger.info(f"🔧 DenseNet dense1 forced compression: {original_dims:,} dims")
            return {
                'method': 'adaptive_pooling',
                'target_size': 4,  # Fixed 4x4 = 16 spatial positions
                'reason': 'DenseNet dense1 - adaptive pooling (SPP can increase dims)'
            }
        
        elif layer_name in ['dense2', 'denseblock2']:
            logger.info(f"🔧 DenseNet dense2 forced compression: {original_dims:,} dims")
            return {
                'method': 'adaptive_pooling', 
                'target_size': 3,  # Fixed 3x3 = 9 spatial positions
                'reason': 'DenseNet dense2 - adaptive pooling'
            }
        
        elif layer_name in ['dense3', 'denseblock3']:
            logger.info(f"🔧 DenseNet dense3 forced compression: {original_dims:,} dims")
            # This is the problematic layer - use very aggressive compression
            return {
                'method': 'adaptive_pooling',
                'target_size': 2,  # Fixed 2x2 = 4 spatial positions (very aggressive)
                'reason': 'DenseNet dense3 - very aggressive adaptive pooling'
            }
        
        elif layer_name in ['dense4', 'denseblock4']:
            logger.info(f"🔧 DenseNet dense4 forced compression: {original_dims:,} dims")
            return {
                'method': 'adaptive_pooling',
                'target_size': 2,  # Fixed 2x2 = 4 spatial positions
                'reason': 'DenseNet dense4 - aggressive adaptive pooling'
            }
        
        elif layer_name in ['trans1', 'transition1', 'trans2', 'transition2', 'trans3', 'transition3']:
            logger.info(f"🔧 DenseNet transition forced compression: {original_dims:,} dims")
            return {
                'method': 'adaptive_pooling',
                'target_size': 3,  # Fixed 3x3 = 9 spatial positions
                'reason': 'DenseNet transition - adaptive pooling'
            }
        
        elif layer_name in ['bn', 'final_norm']:
            logger.info(f"🔧 DenseNet bn forced compression: {original_dims:,} dims")
            return {
                'method': 'adaptive_pooling',
                'target_size': 1,  # Global average pooling (1x1)
                'reason': 'DenseNet bn - global average pooling'
            }
        
        # 🔧 RESNET LAYERS: Original logic
        # LAYER1 EXCEPTION: Always use compression (too many dimensions otherwise)
        if layer_name in ['layer1']:
            logger.info(f"🔧 Layer1 forced compression: {original_dims:,} dims too large for no compression")
            if original_dims > 50000:  # Very high dimensions
                return {
                    'method': 'spatial_pyramid_pooling',
                    'levels': [1, 2, 4],
                    'reason': 'Layer1 - forced compression with SPP for very high dims'
                }
            elif original_dims > 20000:  # High dimensions
                return {
                    'method': 'spatial_pyramid_pooling',
                    'levels': [1, 2],
                    'reason': 'Layer1 - forced compression with SPP'
                }
            else:
                return {
                    'method': 'adaptive_pooling',
                    'target_size': 8,
                    'reason': 'Layer1 - forced compression with adaptive pooling'
                }
        
        # OTHER LAYERS: Respect compression setting
        if not self.use_compression:
            # NO COMPRESSION: Keep full spatial dimensions (flatten)
            return {
                'method': 'flatten',
                'reason': f'No compression - preserving all {original_dims:,} spatial dimensions'
            }
        
        # COMPRESSION ENABLED: Use smart spatial reduction
        
        # Early layers (layer2): preserve spatial info with moderate compression
        if layer_name in ['layer2']:
            # Estimate spatial dimensions for compression benefit check
            channels = self._estimate_channels_from_layer(layer_name)
            spatial_pixels = original_dims // channels if channels > 0 else 256
            height = width = int(spatial_pixels ** 0.5)
            
            # Test different compression strategies
            if original_dims > 30000:
                candidate_strategy = {
                    'method': 'spatial_pyramid_pooling',
                    'levels': [1, 2, 4],
                    'reason': 'Early layer - multi-scale SPP compression'
                }
            elif original_dims > 15000:
                candidate_strategy = {
                    'method': 'spatial_pyramid_pooling',
                    'levels': [1, 2],
                    'reason': 'Early layer - SPP compression'
                }
            else:
                candidate_strategy = {
                    'method': 'adaptive_pooling',
                    'target_size': 8,
                    'reason': 'Early layer - adaptive pooling compression'
                }
            
            # Check if compression is beneficial
            estimated_shape = (1, channels, height, width)
            compressed_dims = self._calculate_compressed_dimensions(estimated_shape, candidate_strategy)
            
            if compressed_dims < original_dims:
                logger.info(f"🔧 {layer_name}: Compression beneficial - {original_dims:,} → {compressed_dims:,} dims (reduction: {original_dims/compressed_dims:.1f}x)")
                return candidate_strategy
            else:
                logger.info(f"🔧 {layer_name}: Compression would increase dims ({original_dims:,} → {compressed_dims:,}), using flatten")
                return {
                    'method': 'flatten',
                    'reason': f'Early layer - compression would increase dims ({original_dims:,} → {compressed_dims:,}), using flatten'
                }
        
        # Middle layers (layer3): moderate compression
        elif layer_name in ['layer3']:
            # Estimate spatial dimensions for compression benefit check
            channels = self._estimate_channels_from_layer(layer_name)
            spatial_pixels = original_dims // channels if channels > 0 else 64
            height = width = int(spatial_pixels ** 0.5)
            
            # Test compression strategy
            if original_dims > 20000:
                candidate_strategy = {
                    'method': 'spatial_pyramid_pooling',
                    'levels': [1, 2],
                    'reason': 'Mid-layer - SPP compression'
                }
            else:
                candidate_strategy = {
                    'method': 'adaptive_pooling',
                    'target_size': 4,
                    'reason': 'Mid-layer - adaptive pooling compression'
                }
            
            # Check if compression is beneficial
            estimated_shape = (1, channels, height, width)
            compressed_dims = self._calculate_compressed_dimensions(estimated_shape, candidate_strategy)
            
            if compressed_dims < original_dims:
                logger.info(f"🔧 {layer_name}: Compression beneficial - {original_dims:,} → {compressed_dims:,} dims (reduction: {original_dims/compressed_dims:.1f}x)")
                return candidate_strategy
            else:
                logger.info(f"🔧 {layer_name}: Compression would increase dims ({original_dims:,} → {compressed_dims:,}), using flatten")
                return {
                    'method': 'flatten',
                    'reason': f'Mid-layer - compression would increase dims ({original_dims:,} → {compressed_dims:,}), using flatten'
                }
        
        # 🔧 FIXED: Late layers (layer4, layer4.1) - Use SPP for spatial consistency
        elif layer_name in ['layer4', 'layer4.1']:
            # Estimate spatial dimensions for compression benefit check
            channels = self._estimate_channels_from_layer(layer_name)
            spatial_pixels = original_dims // channels if channels > 0 else 16
            height = width = int(spatial_pixels ** 0.5)
            
            # Test SPP compression with conservative levels
            candidate_strategy = {
                'method': 'spatial_pyramid_pooling',
                'levels': [1, 2],  # Conservative: 1x1 + 2x2 = 5 pooled regions
                'reason': 'Late layer - SPP compression maintaining spatial structure'
            }
            
            # Calculate compressed dimensions 
            estimated_shape = (1, channels, height, width)
            compressed_dims = self._calculate_compressed_dimensions(estimated_shape, candidate_strategy)
            
            # Check if compression is beneficial
            if compressed_dims < original_dims:
                logger.info(f"🔧 {layer_name}: SPP beneficial - {original_dims:,} → {compressed_dims:,} dims (reduction: {original_dims/compressed_dims:.1f}x)")
                return candidate_strategy
            else:
                # SPP would increase dimensions, use flatten instead
                logger.info(f"🔧 {layer_name}: SPP would increase dims ({original_dims:,} → {compressed_dims:,}), using flatten")
                return {
                    'method': 'flatten',
                    'reason': f'Late layer - SPP would increase dims ({original_dims:,} → {compressed_dims:,}), using flatten'
                }
        
        # Default: moderate compression
        else:
            # Estimate spatial dimensions for compression benefit check
            channels = self._estimate_channels_from_layer(layer_name)
            spatial_pixels = original_dims // channels if channels > 0 else 16
            height = width = int(spatial_pixels ** 0.5)
            
            # Test compression strategy
            if original_dims > 15000:
                candidate_strategy = {
                    'method': 'adaptive_pooling',
                    'target_size': 4,
                    'reason': 'Default - adaptive pooling compression'
                }
            else:
                candidate_strategy = {
                    'method': 'spatial_pyramid_pooling',
                    'levels': [1, 2],
                    'reason': 'Default - SPP compression preserving spatial structure'
                }
            
            # Check if compression is beneficial
            estimated_shape = (1, channels, height, width)
            compressed_dims = self._calculate_compressed_dimensions(estimated_shape, candidate_strategy)
            
            if compressed_dims < original_dims:
                logger.info(f"🔧 {layer_name}: Default compression beneficial - {original_dims:,} → {compressed_dims:,} dims (reduction: {original_dims/compressed_dims:.1f}x)")
                return candidate_strategy
            else:
                logger.info(f"🔧 {layer_name}: Default compression would increase dims ({original_dims:,} → {compressed_dims:,}), using flatten")
                return {
                    'method': 'flatten',
                    'reason': f'Default - compression would increase dims ({original_dims:,} → {compressed_dims:,}), using flatten'
                }
    
    def _spatial_pyramid_pooling(self, features: torch.Tensor, levels: list) -> torch.Tensor:
        """
        Multi-scale spatial pooling that preserves spatial structure.
        
        Args:
            features: [B, C, H, W] tensor
            levels: List of pooling grid sizes
            
        Returns:
            Flattened features preserving multi-scale spatial info
        """
        batch_size, channels, height, width = features.shape
        pooled_features = []
        
        for level in levels:
            # Adaptive pooling to create level x level grid
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (level, level))
            # Flatten: [B, C, level, level] -> [B, C*level*level]
            pooled = pooled.view(batch_size, -1)
            pooled_features.append(pooled)
        
        # Concatenate all levels: [B, C*(1+4+16)] for levels=[1,2,4]
        result = torch.cat(pooled_features, dim=1)
        logger.debug(f"🔧 SPP applied with levels {levels}: {features.shape} → {result.shape}")
        return result
    
    def _random_spatial_sampling(self, features: torch.Tensor, max_samples: int) -> torch.Tensor:
        """
        Randomly sample spatial locations to reduce dimensionality
        while maintaining spatial diversity.
        
        Args:
            features: [B, C, H, W] tensor
            max_samples: Maximum number of spatial samples
            
        Returns:
            Sampled features
        """
        batch_size, channels, height, width = features.shape
        
        if height * width <= max_samples:
            # If already small enough, just flatten
            return features.view(batch_size, channels * height * width)
        
        # Randomly sample spatial locations
        n_samples = min(max_samples, height * width)
        spatial_indices = torch.randperm(height * width, device=features.device)[:n_samples]
        
        # Reshape and sample
        features_flat = features.view(batch_size, channels, height * width)
        sampled_features = features_flat[:, :, spatial_indices]
        
        result = sampled_features.view(batch_size, channels * n_samples)
        logger.debug(f"🔧 Random sampling applied: {features.shape} → {result.shape} ({n_samples} samples)")
        return result
    
    def _adaptive_spatial_pooling(self, features: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Adaptive spatial pooling to a target size.
        
        Args:
            features: [B, C, H, W] tensor
            target_size: Target spatial size
            
        Returns:
            Pooled features
        """
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (target_size, target_size))
        result = pooled.view(features.shape[0], -1)
        logger.debug(f"🔧 Adaptive pooling applied: {features.shape} → {result.shape}")
        return result
    
    def _pca_spatial_preserving(self, features: torch.Tensor, target_dims: int) -> torch.Tensor:
        """
        🔧 FIXED: Enhanced PCA with guaranteed consistent output dimensions.
        
        Args:
            features: [B, C, H, W] tensor
            target_dims: Target number of dimensions after compression
            
        Returns:
            Compressed features as tensor [B, target_dims] (guaranteed consistent size)
        """
        batch_size, channels, height, width = features.shape
        original_dims = channels * height * width
        
        logger.debug(f"🔧 PCA spatial preserving: input shape {features.shape}, target dims {target_dims}")
        
        # If already small enough, pad or truncate to exact target dimensions
        if original_dims <= target_dims:
            result = features.view(batch_size, -1)
            if result.shape[1] < target_dims:
                # Pad with zeros to reach target dimensions
                padding = torch.zeros(batch_size, target_dims - result.shape[1], 
                                    device=features.device, dtype=features.dtype)
                result = torch.cat([result, padding], dim=1)
            elif result.shape[1] > target_dims:
                # Truncate to target dimensions
                result = result[:, :target_dims]
            logger.debug(f"🔧 No PCA needed: {features.shape} → {result.shape}")
            return result
        
        # Reshape to [B, C*H*W] for PCA
        features_flat = features.view(batch_size, -1)  # [B, C*H*W]
        
        # 🔧 CRITICAL FIX: Always ensure exact target dimensions
        try:
            # Method 1: Simplified spatial-aware compression with guaranteed dimensions
            if height * width > 1:
                # Calculate optimal spatial reduction to get close to target
                spatial_ratio = max(1, (original_dims // target_dims) ** 0.5)
                target_h = max(1, int(height / spatial_ratio))
                target_w = max(1, int(width / spatial_ratio))
                
                # Use adaptive pooling to reduce spatial dimensions
                pooled = torch.nn.functional.adaptive_avg_pool2d(features, (target_h, target_w))
                result = pooled.view(batch_size, -1)  # [B, C*target_h*target_w]
                
                # Ensure exact target dimensions
                if result.shape[1] < target_dims:
                    # Pad with zeros
                    padding = torch.zeros(batch_size, target_dims - result.shape[1], 
                                        device=features.device, dtype=features.dtype)
                    result = torch.cat([result, padding], dim=1)
                elif result.shape[1] > target_dims:
                    # Use random projection to reduce to exact target dimensions
                    projection = torch.randn(result.shape[1], target_dims, 
                                           device=result.device, dtype=result.dtype)
                    projection = projection / torch.norm(projection, dim=0, keepdim=True)
                    result = result @ projection
                
            else:
                # Already 1D spatial, use random projection
                if features_flat.shape[1] > target_dims:
                    projection = torch.randn(features_flat.shape[1], target_dims, 
                                           device=features.device, dtype=features.dtype)
                    projection = projection / torch.norm(projection, dim=0, keepdim=True)
                    result = features_flat @ projection
                else:
                    # Pad to target dimensions
                    padding = torch.zeros(batch_size, target_dims - features_flat.shape[1], 
                                        device=features.device, dtype=features.dtype)
                    result = torch.cat([features_flat, padding], dim=1)
            
        except Exception as e:
            logger.warning(f"🔧 PCA failed ({e}), using simple truncation/padding")
            # Fallback: simple truncation or padding
            if features_flat.shape[1] > target_dims:
                result = features_flat[:, :target_dims]
            else:
                padding = torch.zeros(batch_size, target_dims - features_flat.shape[1], 
                                    device=features.device, dtype=features.dtype)
                result = torch.cat([features_flat, padding], dim=1)
        
        # 🔧 VERIFICATION: Ensure exact dimensions
        assert result.shape[1] == target_dims, f"PCA output dimension mismatch: {result.shape[1]} != {target_dims}"
        
        logger.debug(f"🔧 PCA applied: {features.shape} → {result.shape} (exactly {target_dims} dims)")
        return result
    
    def debug_tensor_dimensions(self, features: torch.Tensor, layer_name: str):
        """Debug method to understand tensor structure"""
        if len(features.shape) == 4:
            batch, channels, height, width = features.shape
            total_dims = channels * height * width
            # logger.info(f"🔍 DEBUG {layer_name}: Shape=[{batch}, {channels}, {height}, {width}], Total={total_dims:,}")
            # logger.info(f"🔍 DEBUG {layer_name}: Spatial={height}x{width}={height*width} pixels, Channels={channels}")
            
            # Calculate what different compression methods would yield
            spp_dims = channels * (1 + 4 + 16)  # [1,2,4] levels
            adaptive_4x4 = channels * 16
            adaptive_2x2 = channels * 4
            global_avg = channels
            
            # logger.info(f"🔍 DEBUG {layer_name}: SPP[1,2,4]→{spp_dims:,}, Adaptive4x4→{adaptive_4x4:,}, Adaptive2x2→{adaptive_2x2:,}, GlobalAvg→{global_avg:,}")
        else:
            logger.info(f"🔍 DEBUG {layer_name}: Non-4D tensor shape: {features.shape}")
    
    @retry_on_memory_error(max_retries=3, initial_delay=15.0)
    def extract_features(self, data_loader, device, target_layer: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features from the specified layer with memory safety and caching.
        
        Args:
            data_loader: Data loader for the dataset
            device: Device to run extraction on
            target_layer: Specific layer to extract from (overrides forced_layer if provided)
            
        Returns:
            Tuple of (features, logits, labels)
        """
        layer_to_use = target_layer or self.forced_layer
        if not layer_to_use:
            raise ValueError("No target layer specified for feature extraction")
        
        # Initialize memory monitoring
        monitor = SharedMemoryMonitor()
        monitor.log_memory_statistics(f"before {layer_to_use} extraction")
        
        # Check cache first if enabled
        if self.enable_caching:
            cache_key = self._create_cache_key(data_loader, layer_to_use, self.use_compression)
            cached_result = self._load_features_from_cache(cache_key)
            
            if cached_result is not None:
                self.extraction_metrics['cache_hits'][layer_to_use] = self.extraction_metrics['cache_hits'].get(layer_to_use, 0) + 1
                logger.info(f"🎯 Cache hit for {layer_to_use} - skipping expensive extraction")
                return cached_result
            else:
                self.extraction_metrics['cache_misses'][layer_to_use] = self.extraction_metrics['cache_misses'].get(layer_to_use, 0) + 1
                logger.info(f"📊 Cache miss for {layer_to_use} - proceeding with extraction")
        
        logger.info(f"🔍 Extracting features from layer: {layer_to_use}")
        
        self.model.eval()
        all_features = []
        all_logits = []
        all_labels = []
        
        start_time = time.time()
        base_batch_size = getattr(data_loader, 'batch_size', 32)
        current_batch_size = base_batch_size
        extraction_attempt = 0
        max_extraction_attempts = 3
        
        while extraction_attempt < max_extraction_attempts:
            try:
                # Reset for fresh attempt
                all_features.clear()
                all_logits.clear()
                all_labels.clear()
                
                with torch.no_grad():
                    for batch_idx, (data, targets) in enumerate(data_loader):
                        try:
                            # Memory checkpoint before batch
                            memory_usage = monitor.get_usage_percentage()
                            
                            # Dynamic batch size reduction if memory usage is high
                            if memory_usage > 70.0:
                                adaptive_size = get_adaptive_batch_size(current_batch_size, memory_usage)
                                if adaptive_size != current_batch_size:
                                    current_batch_size = adaptive_size
                                    logger.info(f"Reduced batch size to {current_batch_size} (memory: {memory_usage:.1f}%)")
                            
                            data, targets = data.to(device), targets.to(device)
                            
                            # Process batch with retry logic for CUDA OOM
                            try:
                                # Forward pass (triggers hooks)
                                logits = process_batch_with_retry(self.model, data, device, current_batch_size)
                                
                                # Get features from target layer
                                if layer_to_use in self.layer_features:
                                    features = self.layer_features[layer_to_use]
                                    all_features.append(features.cpu().numpy())
                                    all_logits.append(logits.cpu().numpy())
                                    all_labels.append(targets.cpu().numpy())
                                    
                                    # Clear layer features to free memory
                                    del self.layer_features[layer_to_use]
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                
                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    logger.warning(f"CUDA OOM in batch {batch_idx}, cleaning memory and retrying")
                                    cleanup_all_torch_memory()
                                    
                                    # Retry with smaller batch size
                                    current_batch_size = max(current_batch_size // 2, 4)
                                    logits = process_batch_with_retry(self.model, data, device, current_batch_size)
                                    
                                    if layer_to_use in self.layer_features:
                                        features = self.layer_features[layer_to_use]
                                        all_features.append(features.cpu().numpy())
                                        all_logits.append(logits.cpu().numpy())
                                        all_labels.append(targets.cpu().numpy())
                                        
                                        del self.layer_features[layer_to_use]
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()
                                else:
                                    raise
                            
                            # Progress logging with memory stats
                            if batch_idx % 50 == 0:
                                memory_usage_after = monitor.get_usage_percentage()
                                logger.info(f"   Processed batch {batch_idx}/{len(data_loader)} "
                                           f"(memory: {memory_usage_after:.1f}%, batch_size: {current_batch_size})")
                                
                                # Cleanup every 100 batches
                                if batch_idx % 100 == 0 and batch_idx > 0:
                                    cleanup_all_torch_memory()
                                    
                        except Exception as e:
                            logger.error(f"Error processing batch {batch_idx}: {e}")
                            # Clean up memory on error
                            cleanup_all_torch_memory()
                            raise
                
                # If we get here, extraction was successful
                break
                
            except Exception as e:
                extraction_attempt += 1
                logger.warning(f"Extraction attempt {extraction_attempt}/{max_extraction_attempts} failed: {e}")
                
                if extraction_attempt >= max_extraction_attempts:
                    logger.error(f"All {max_extraction_attempts} extraction attempts failed")
                    raise
                
                # Progressive recovery strategy
                if extraction_attempt == 2:  # Second failure - enable emergency mode
                    logger.warning("🚨 Enabling emergency recovery mode for feature extraction")
                    enable_emergency_recovery_mode()
                    current_batch_size = max(current_batch_size // 4, 1)
                
                # Cleanup and wait before retry
                cleanup_all_torch_memory()
                monitor.wait_for_memory(threshold=60.0, timeout=120.0)
                time.sleep(10.0 * extraction_attempt)
        
        total_time = time.time() - start_time
        
        if not all_features:
            raise ValueError(f"No features extracted from layer: {layer_to_use}")
        
        # Memory checkpoint before concatenation
        monitor.log_memory_statistics(f"before {layer_to_use} concatenation")
        
        try:
            features = np.concatenate(all_features, axis=0)
            logits = np.concatenate(all_logits, axis=0)
            labels = np.concatenate(all_labels, axis=0)
        except Exception as e:
            logger.error(f"Error concatenating arrays: {e}")
            cleanup_all_torch_memory()
            raise
        
        # Save to cache if enabled
        if self.enable_caching:
            try:
                self._save_features_to_cache(features, logits, labels, cache_key)
            except Exception as e:
                logger.warning(f"Failed to save features to cache: {e}")
        
        # Record performance metrics
        self.extraction_metrics['extraction_times'][layer_to_use] = total_time
        self.extraction_metrics['layer_availability'][layer_to_use] = True
        
        # Final memory checkpoint
        monitor.log_memory_statistics(f"after {layer_to_use} extraction")
        
        logger.info(f"✅ Extracted features: {features.shape}, logits: {logits.shape}, labels: {labels.shape}")
        logger.info(f"   Extraction time: {total_time:.2f}s")
        
        return features, logits, labels
    
    # 🔧 COMPATIBILITY: Add original FeatureExtractor interface method
    @retry_on_memory_error(max_retries=3, initial_delay=15.0)
    def extract_features_compatible(self, data_loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features using the original FeatureExtractor interface with memory safety.
        
        This method matches the original FeatureExtractor.extract_features signature
        and behavior for plug-and-play compatibility.
        
        Args:
            data_loader: Data loader for the dataset
            device: Device to run extraction on
            
        Returns:
            Tuple of (features, logits, labels)
        """
        # Use forced layer if specified, otherwise use default layer
        layer_to_use = self.forced_layer
        if not layer_to_use:
            # Use default layer based on architecture (like original FeatureExtractor)
            if 'resnet18' in self.model_name:
                layer_to_use = 'layer4'
            elif 'resnet50' in self.model_name:
                layer_to_use = 'layer3'
            elif 'densenet121' in self.model_name:
                layer_to_use = 'denseblock3'
            else:
                # Fallback to first available layer
                layer_to_use = list(self.layer_candidates.keys())[0]
        
        # Initialize memory monitoring
        monitor = SharedMemoryMonitor()
        monitor.log_memory_statistics(f"before compatible {layer_to_use} extraction")
        
        logger.info(f"🔍 Extracting features (compatible mode) from layer: {layer_to_use}")
        
        self.model.eval()
        all_features = []
        all_logits = []
        all_labels = []
        
        base_batch_size = getattr(data_loader, 'batch_size', 32)
        current_batch_size = base_batch_size
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                try:
                    # Memory checkpoint before batch
                    memory_usage = monitor.get_usage_percentage()
                    
                    # Dynamic batch size reduction if memory usage is high
                    if memory_usage > 70.0:
                        adaptive_size = get_adaptive_batch_size(current_batch_size, memory_usage)
                        if adaptive_size != current_batch_size:
                            current_batch_size = adaptive_size
                            logger.info(f"Reduced compatible batch size to {current_batch_size} (memory: {memory_usage:.1f}%)")
                    
                    data, targets = data.to(device), targets.to(device)
                    
                    # Forward pass with retry logic
                    logits = process_batch_with_retry(self.model, data, device, current_batch_size)
                    
                    # Get features (captured by hook) - use compatibility features attribute
                    if self.features is not None:
                        # Flatten spatial dimensions but keep feature dimension (like original)
                        if len(self.features.shape) == 4:  # [B, C, H, W]
                            features = self.features.mean(dim=[2, 3])  # Global average pooling
                        else:
                            features = self.features
                        
                        all_features.append(features.cpu().numpy())
                        all_logits.append(logits.cpu().numpy())
                        all_labels.append(targets.cpu().numpy())
                        
                        # Clear features to free memory
                        self.features = None
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Progress logging with memory stats
                    if batch_idx % 50 == 0:
                        memory_usage_after = monitor.get_usage_percentage()
                        logger.info(f"   Processed batch {batch_idx}/{len(data_loader)} "
                                   f"(memory: {memory_usage_after:.1f}%, batch_size: {current_batch_size})")
                        
                        # Cleanup every 100 batches
                        if batch_idx % 100 == 0 and batch_idx > 0:
                            cleanup_all_torch_memory()
                            
                except Exception as e:
                    logger.error(f"Error processing compatible batch {batch_idx}: {e}")
                    cleanup_all_torch_memory()
                    raise
        
        # Memory checkpoint before concatenation
        monitor.log_memory_statistics(f"before compatible {layer_to_use} concatenation")
        
        try:
            features = np.concatenate(all_features, axis=0)
            logits = np.concatenate(all_logits, axis=0)
            labels = np.concatenate(all_labels, axis=0)
        except Exception as e:
            logger.error(f"Error concatenating compatible arrays: {e}")
            cleanup_all_torch_memory()
            raise
        
        # Final memory checkpoint
        monitor.log_memory_statistics(f"after compatible {layer_to_use} extraction")
        
        logger.info(f"✅ Extracted features (compatible): {features.shape}, logits: {logits.shape}, labels: {labels.shape}")
        return features, logits, labels
    
    def extract_multi_layer_features(self, data_loader, device) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Extract features from all available layers for comparative analysis.
        
        Args:
            data_loader: Data loader for the dataset
            device: Device to run extraction on
            
        Returns:
            Dictionary mapping layer names to (features, logits, labels) tuples
        """
        # logger.info(f"🔍 Extracting features from all available layers")
        
        self.model.eval()
        layer_data = defaultdict(lambda: {'features': [], 'logits': [], 'labels': []})
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                data, targets = data.to(device), targets.to(device)
                
                # Forward pass (triggers all hooks)
                logits = self.model(data)
                
                # Collect features from all layers
                for layer_name, features in self.layer_features.items():
                    layer_data[layer_name]['features'].append(features.cpu().numpy())
                    layer_data[layer_name]['logits'].append(logits.cpu().numpy())
                    layer_data[layer_name]['labels'].append(targets.cpu().numpy())
                
                if batch_idx % 50 == 0:
                    logger.info(f"   Processed batch {batch_idx}/{len(data_loader)}")
        
        # Concatenate all batches for each layer
        results = {}
        for layer_name, data_dict in layer_data.items():
            if data_dict['features']:
                features = np.concatenate(data_dict['features'], axis=0)
                logits = np.concatenate(data_dict['logits'], axis=0)
                labels = np.concatenate(data_dict['labels'], axis=0)
                results[layer_name] = (features, logits, labels)
                
                logger.info(f"   ✅ {layer_name}: {features.shape}")
        
        return results
    
    def get_layer_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for all layers."""
        return {
            'model_name': self.model_name,
            'forced_layer': self.forced_layer,
            'use_compression': self.use_compression,
            'available_layers': list(self.layer_candidates.keys()),
            'extraction_metrics': self.extraction_metrics.copy(),
            'layer_candidates': self.layer_candidates.copy()
        }
    
    def save_layer_analysis(self, output_path: str):
        """Save layer analysis results to file."""
        analysis_data = {
            'model_name': self.model_name,
            'forced_layer': self.forced_layer,
            'use_compression': self.use_compression,
            'layer_candidates': self.layer_candidates,
            'extraction_metrics': self.extraction_metrics,
            'timestamp': time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"💾 Layer analysis saved to: {output_path}")
    
    def cleanup(self):
        """
        🔧 FIXED: Enhanced cleanup with proper hook removal and GPU memory management.
        """
        # logger.info(f"🧹 Cleaning up EnhancedLayerFeatureExtractor")
        
        # Remove all hooks
        hooks_removed = 0
        for hook in self.hooks:
            try:
                hook.remove()
                hooks_removed += 1
            except Exception as e:
                logger.warning(f"🔧 Failed to remove hook: {e}")
        
        self.hooks.clear()
        self.layer_features.clear()
        
        # 🔧 COMPATIBILITY: Clear compatibility attributes
        self.features = None
        self.hook = None
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"   Removed {hooks_removed} hooks successfully")
    
    def __del__(self):
        """
        🔧 NEW: Destructor to ensure cleanup on object deletion.
        """
        try:
            if hasattr(self, 'hooks') and self.hooks:
                self.cleanup()
        except Exception:
            pass  # Silent cleanup in destructor
    
    def get_smart_compression_config(self, cal_method: str) -> dict:
        """
        Get smart compression configuration for StabilitySpace integration.
        
        This method provides compression settings that work with the existing
        StabilitySpace compression infrastructure while preserving spatial information.
        
        Args:
            cal_method: Calibration method name
            
        Returns:
            Compression configuration dictionary
        """
        if not self.forced_layer:
            return {'use_compression': False}
        
        # Get expected dimensions for this layer
        layer_info = self.layer_candidates.get(self.forced_layer, {})
        expected_dim = layer_info.get('expected_dim', 'unknown')
        
        # Estimate actual dimensions based on layer
        estimated_dims = self._estimate_layer_dimensions(self.forced_layer)
        
        # Determine if compression is needed
        compression_needed = self._should_use_compression(cal_method, self.forced_layer, estimated_dims)
        
        if not compression_needed:
            return {'use_compression': False}
        
        # Get compression strategy
        strategy = self._get_compression_strategy(cal_method, self.forced_layer, estimated_dims)
        
        logger.info(f"🔧 Smart compression for {self.forced_layer} + {cal_method}:")
        logger.info(f"   Estimated dimensions: {estimated_dims:,}")
        logger.info(f"   Strategy: {strategy['method']}")
        logger.info(f"   Reason: {strategy['reason']}")
        
        return {
            'use_compression': True,
            'compression_mode': strategy['method'],
            'compression_param': strategy['target_dims'],
            'strategy': strategy,
            'estimated_dims': estimated_dims
        }
    
    def _estimate_layer_dimensions(self, layer_name: str) -> int:
        """Estimate the actual dimensions for a layer after intelligent spatial reduction."""
        if 'resnet18' in self.model_name:
            if layer_name == 'layer1':
                # 64 channels × 32×32 spatial = 65,536 original
                if self.use_compression:
                    # SPP with levels [1,2,4] is beneficial: 64 × (1+4+16) = 1,344
                    return 64 * (1 + 4 + 16)
                else:
                    return 64 * 32 * 32  # Flattened
            elif layer_name == 'layer2':
                # 128 channels × 16×16 spatial = 32,768 original
                if self.use_compression:
                    # SPP with levels [1,2] is beneficial: 128 × (1+4) = 640
                    return 128 * (1 + 4)
                else:
                    return 128 * 16 * 16  # Flattened
            elif layer_name == 'layer3':
                # 256 channels × 8×8 spatial = 16,384 original
                if self.use_compression:
                    # SPP with levels [1,2] is beneficial: 256 × (1+4) = 1,280
                    return 256 * (1 + 4)
                else:
                    return 256 * 8 * 8  # Flattened
            elif layer_name == 'layer4':
                # 512 channels × 4×4 spatial = 8,192 original
                if self.use_compression:
                    # 🔧 FIXED: SPP with levels [1,2] is beneficial: 512 × (1+4) = 2,560
                    return 512 * (1 + 4)
                else:
                    return 512 * 4 * 4  # Flattened
            elif layer_name == 'layer4.1':
                # 512 channels × 2×2 spatial = 2,048 original
                if self.use_compression:
                    # 🔧 FIXED: SPP would increase dims (512×5=2,560 > 2,048), so use flatten
                    return 512 * 2 * 2  # No compression, flatten only
                else:
                    return 512 * 2 * 2  # Flattened
        elif 'resnet50' in self.model_name:
            if layer_name == 'layer1':
                # 256 channels × 32×32 spatial = 262,144 original
                if self.use_compression:
                    # SPP with levels [1,2,4] is beneficial: 256 × (1+4+16) = 5,376
                    return 256 * (1 + 4 + 16)
                else:
                    return 256 * 32 * 32  # Flattened
            elif layer_name == 'layer2':
                # 512 channels × 16×16 spatial = 131,072 original
                if self.use_compression:
                    # SPP with levels [1,2] is beneficial: 512 × (1+4) = 2,560
                    return 512 * (1 + 4)
                else:
                    return 512 * 16 * 16  # Flattened
            elif layer_name == 'layer3':
                # 1024 channels × 8×8 spatial = 65,536 original
                if self.use_compression:
                    # SPP with levels [1,2] is beneficial: 1024 × (1+4) = 5,120
                    return 1024 * (1 + 4)
                else:
                    return 1024 * 8 * 8  # Flattened
            elif layer_name == 'layer4':
                # 2048 channels × 4×4 spatial = 32,768 original
                if self.use_compression:
                    # SPP with levels [1,2] is beneficial: 2048 × (1+4) = 10,240
                    return 2048 * (1 + 4)
                else:
                    return 2048 * 4 * 4  # Flattened
            elif layer_name == 'layer4.1':
                # 2048 channels × 2×2 spatial = 8,192 original
                if self.use_compression:
                    # SPP would increase dims (2048×5=10,240 > 8,192), so use flatten
                    return 2048 * 2 * 2  # No compression, flatten only
                else:
                    return 2048 * 2 * 2  # Flattened
        
        # Default estimation (conservative)
        return 5000  # Conservative default
    
    def _should_use_compression(self, cal_method: str, layer_name: str, estimated_dims: int) -> bool:
        """Determine if compression should be used based on method and dimensions."""
        
        # Since SPP is already reducing dimensions significantly, be more conservative
        # Only use additional compression for very high dimensions after SPP
        
        # Always use compression for very high-dimensional semantic methods (after SPP)
        if ('semantic' in cal_method and 
            layer_name in ['layer1', 'layer2'] and 
            estimated_dims > 8000):  # Reduced threshold since SPP already helps
            return True
        
        # Use compression for FAISS methods with high dimensions (after SPP)
        if ('faiss' in cal_method and estimated_dims > 8000):  # Reduced threshold
            return True
        
        # Use compression for any method with very high dimensions (after SPP)
        if estimated_dims > 15000:  # Reduced threshold
            return True
        
        return False
    
    def _get_compression_strategy(self, cal_method: str, layer_name: str, estimated_dims: int) -> dict:
        """Get optimal compression strategy for the given configuration."""
        
        # Early layers with high dimensions: aggressive compression
        if layer_name in ['layer1', 'layer2'] and estimated_dims > 50000:
            return {
                'method': 'pca',
                'target_dims': 512,
                'reason': f'High-dimensional early layer ({estimated_dims:,} dims) - aggressive PCA'
            }
        
        # Early layers with moderate dimensions: moderate compression
        elif layer_name in ['layer1', 'layer2'] and estimated_dims > 20000:
            return {
                'method': 'pca',
                'target_dims': 1024,
                'reason': f'Moderate-dimensional early layer ({estimated_dims:,} dims) - moderate PCA'
            }
        
        # FAISS methods with high dimensions: conservative compression
        elif 'faiss' in cal_method and estimated_dims > 15000:
            return {
                'method': 'pca',
                'target_dims': 2048,
                'reason': f'FAISS method with high dimensions ({estimated_dims:,} dims) - conservative PCA'
            }
        
        # Middle layers: light compression
        elif layer_name in ['layer3'] and estimated_dims > 10000:
            return {
                'method': 'pca',
                'target_dims': 2048,
                'reason': f'Mid-layer with moderate dimensions ({estimated_dims:,} dims) - light PCA'
            }
        
        # Late layers: minimal compression
        elif layer_name in ['layer4', 'layer4.1'] and estimated_dims > 8000:
            return {
                'method': 'pca',
                'target_dims': 4096,
                'reason': f'Late layer with moderate dimensions ({estimated_dims:,} dims) - minimal PCA'
            }
        
        # Default: no compression
        else:
            return {
                'method': 'none',
                'target_dims': estimated_dims,
                'reason': f'Manageable dimensions ({estimated_dims:,} dims) - no compression needed'
            }
    
    def debug_model_layers(self):
        """
        🔧 NEW: Debug function to discover available layers in any model.
        """
        logger.info(f"🔬 DEBUGGING MODEL LAYERS for {self.model_name}")
        logger.info("=" * 50)
        
        # Print model attributes
        logger.info("📋 Model attributes (not starting with _):")
        model_attrs = [attr for attr in dir(self.model) if not attr.startswith('_') and hasattr(self.model, attr)]
        for attr in model_attrs:
            attr_obj = getattr(self.model, attr)
            if isinstance(attr_obj, nn.Module):
                logger.info(f"   ✅ {attr}: {type(attr_obj).__name__}")
            else:
                logger.info(f"   📄 {attr}: {type(attr_obj).__name__} (not a module)")
        
        # Print named modules (first level)
        logger.info("")
        logger.info("📋 Named modules (first level):")
        for name, module in self.model.named_children():
            logger.info(f"   ✅ {name}: {type(module).__name__}")
            
            # For each child, show its children too
            child_modules = list(module.named_children())
            if child_modules:
                logger.info(f"      └─ Children: {[child_name for child_name, _ in child_modules[:5]]}")
                if len(child_modules) > 5:
                    logger.info(f"         (and {len(child_modules) - 5} more...)")
        
        # Try to identify the architecture type
        logger.info("")
        # logger.info("🔍 Architecture analysis:")
        # if hasattr(self.model, 'features'):
            # logger.info("   📁 Has 'features' module → Likely torchvision model")
        # if hasattr(self.model, 'dense1'):
            # logger.info("   🟦 Has 'dense1' → Likely custom DenseNet")
        # if hasattr(self.model, 'layer1'):
            # logger.info("   🟧 Has 'layer1' → Likely ResNet")
        
        logger.info("=" * 50)
    
    def _create_cache_key(self, data_loader, layer_name: str, compression: bool) -> str:
        """
        Create a unique cache key for feature extraction.
        
        Args:
            data_loader: DataLoader used for extraction
            layer_name: Name of the layer
            compression: Whether compression is used
            
        Returns:
            Unique cache key string
        """
        try:
            # Create a hash based on dataset, model, layer, and compression settings
            dataset_info = {
                'dataset_size': len(data_loader.dataset) if hasattr(data_loader.dataset, '__len__') else 'unknown',
                'batch_size': getattr(data_loader, 'batch_size', 'unknown'),
                'dataset_type': type(data_loader.dataset).__name__
            }
            
            cache_components = {
                'model_name': self.model_name,
                'layer_name': layer_name,
                'compression': compression,
                'dataset_info': dataset_info
            }
            
            cache_key = create_memory_safe_hash(cache_components)
            return f"{self.model_name}_{layer_name}_{cache_key}_{'compressed' if compression else 'uncompressed'}"
            
        except Exception as e:
            logger.warning(f"Failed to create cache key: {e}")
            return f"fallback_{self.model_name}_{layer_name}_{int(time.time())}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the full path for a cache file."""
        return self.cache_dir / f"{cache_key}.pt"
    
    def _save_features_to_cache(self, features: np.ndarray, logits: np.ndarray, labels: np.ndarray, cache_key: str) -> bool:
        """
        Save extracted features to cache.
        
        Args:
            features: Extracted features
            logits: Model logits
            labels: Ground truth labels
            cache_key: Cache key for the file
            
        Returns:
            True if successfully saved
        """
        if not self.enable_caching:
            return False
        
        try:
            cache_path = self._get_cache_path(cache_key)
            
            cache_data = {
                'features': features,
                'logits': logits,
                'labels': labels,
                'timestamp': time.time(),
                'model_name': self.model_name,
                'layer_name': self.forced_layer,
                'compression': self.use_compression,
                'feature_shape': features.shape,
                'logits_shape': logits.shape,
                'labels_shape': labels.shape
            }
            
            torch.save(cache_data, cache_path)
            logger.info(f"💾 Features cached: {cache_path}")
            logger.info(f"   Shapes: features {features.shape}, logits {logits.shape}, labels {labels.shape}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to save features to cache: {e}")
            return False
    
    def _load_features_from_cache(self, cache_key: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load features from cache if available.
        
        Args:
            cache_key: Cache key for the file
            
        Returns:
            Tuple of (features, logits, labels) if found, None otherwise
        """
        if not self.enable_caching:
            return None
        
        try:
            cache_path = self._get_cache_path(cache_key)
            
            if not cache_path.exists():
                return None
            
            # Check file age (optional: could implement cache expiration)
            file_age = time.time() - cache_path.stat().st_mtime
            max_age = 7 * 24 * 3600  # 7 days
            
            if file_age > max_age:
                logger.info(f"🗑️ Cache expired ({file_age/3600:.1f}h old): {cache_path}")
                cache_path.unlink()
                return None
            
            cache_data = torch.load(cache_path, map_location='cpu')
            
            # Validate cache data
            required_keys = ['features', 'logits', 'labels']
            if not all(key in cache_data for key in required_keys):
                logger.warning(f"Invalid cache data: {cache_path}")
                return None
            
            features = cache_data['features']
            logits = cache_data['logits']
            labels = cache_data['labels']
            
            logger.info(f"📂 Features loaded from cache: {cache_path}")
            logger.info(f"   Shapes: features {features.shape}, logits {logits.shape}, labels {labels.shape}")
            logger.info(f"   Cache age: {file_age/3600:.1f}h")
            
            return features, logits, labels
            
        except Exception as e:
            logger.warning(f"Failed to load features from cache: {e}")
            return None
    
    def clear_cache(self, max_age_hours: Optional[float] = None) -> Dict[str, int]:
        """
        Clear feature cache, optionally filtering by age.
        
        Args:
            max_age_hours: Maximum age in hours (None = clear all)
            
        Returns:
            Dictionary with cleanup statistics
        """
        if not self.enable_caching or not self.cache_dir.exists():
            return {'files_removed': 0, 'space_freed_mb': 0}
        
        removed_count = 0
        space_freed = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.pt"):
            try:
                should_remove = False
                
                if max_age_hours is None:
                    should_remove = True
                else:
                    file_age = current_time - cache_file.stat().st_mtime
                    if file_age > (max_age_hours * 3600):
                        should_remove = True
                
                if should_remove:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    removed_count += 1
                    space_freed += file_size
                    
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        space_freed_mb = space_freed / (1024 * 1024)
        
        logger.info(f"🧹 Cache cleanup: {removed_count} files removed, {space_freed_mb:.1f}MB freed")
        
        return {
            'files_removed': removed_count,
            'space_freed_mb': space_freed_mb
        }

    def _find_dinov2_layer(self, layer_name: str) -> Optional[nn.Module]:
        """Enhanced DINOv2 layer finder that handles both custom and Hugging Face models."""
        logger.info(f"🔍 DEBUG: Starting DINOv2 layer search for '{layer_name}'")

        # Determine the base model to search within
        base_model = self.model
        model_type = type(self.model).__name__
        logger.info(f"   📋 DEBUG: Model type: {model_type}")
        
        # Handle custom DINOv2Classifier wrapper
        if hasattr(self.model, 'backbone'):
            base_model = self.model.backbone
            logger.info("   ✅ DEBUG: Found 'backbone' attribute, searching within it.")
        else:
            logger.info("   ⚠️ DEBUG: No 'backbone' attribute found on the main model.")

        # Handle Hugging Face Dinov2ForImageClassification
        if hasattr(base_model, 'dinov2'):
            base_model = base_model.dinov2
            logger.info("   ✅ DEBUG: Found 'dinov2' attribute (Hugging Face model)")

        logger.info(f"   📋 DEBUG: Base model type: {type(base_model).__name__}")
        logger.info(f"   📋 DEBUG: Available attributes: {[attr for attr in dir(base_model) if not attr.startswith('_')][:10]}...")

        # Try different layer access patterns for DINOv2
        layer_access_patterns = [
            # Pattern 1: Standard transformer encoder structure
            lambda m, idx: m.encoder.layer[idx] if hasattr(m, 'encoder') and hasattr(m.encoder, 'layer') else None,
            
            # Pattern 2: Direct blocks access (timm-style)
            lambda m, idx: m.blocks[idx] if hasattr(m, 'blocks') and isinstance(m.blocks, (nn.ModuleList, nn.Sequential)) else None,
            
            # Pattern 3: Hugging Face DINOv2 encoder layers
            lambda m, idx: m.encoder.layers[idx] if hasattr(m, 'encoder') and hasattr(m.encoder, 'layers') else None,
            
            # Pattern 4: Alternative encoder structure
            lambda m, idx: getattr(m, f'layer_{idx}') if hasattr(m, f'layer_{idx}') else None,
        ]

        # Handle transformer blocks
        if layer_name.startswith('block_'):
            try:
                block_index = int(layer_name.split('_')[1])
                logger.info(f"   🔢 DEBUG: Extracted block index: {block_index}")
                
                # Try each access pattern
                for i, pattern in enumerate(layer_access_patterns):
                    try:
                        layer = pattern(base_model, block_index)
                        if layer is not None:
                            logger.info(f"   ✅ DEBUG: Found layer using pattern {i+1}: {type(layer).__name__}")
                            return layer
                    except (AttributeError, IndexError, TypeError) as e:
                        logger.debug(f"   ⚠️ DEBUG: Pattern {i+1} failed: {e}")
                        
            except (ValueError, IndexError) as e:
                logger.warning(f"   ⚠️ DEBUG: Invalid DINOv2 block name: {layer_name} - {e}")

        # Handle the final normalization layer
        if layer_name == 'norm':
            norm_candidates = ['layernorm', 'norm', 'ln_f', 'final_norm']
            for norm_name in norm_candidates:
                if hasattr(base_model, norm_name):
                    norm_layer = getattr(base_model, norm_name)
                    logger.info(f"   ✅ DEBUG: Found normalization layer '{norm_name}': {type(norm_layer).__name__}")
                    return norm_layer
            logger.warning("   ⚠️ DEBUG: No normalization layer found")

        # Enhanced debugging: show actual model structure
        logger.info("   🔍 DEBUG: Exploring model structure...")
        try:
            if hasattr(base_model, 'encoder'):
                encoder = base_model.encoder
                logger.info(f"      📋 Encoder type: {type(encoder).__name__}")
                logger.info(f"      📋 Encoder attributes: {[attr for attr in dir(encoder) if not attr.startswith('_')][:10]}...")
                
                # Check for layers/layer attribute
                if hasattr(encoder, 'layer'):
                    layers = encoder.layer
                    logger.info(f"      📋 Found encoder.layer with {len(layers)} layers")
                elif hasattr(encoder, 'layers'):
                    layers = encoder.layers
                    logger.info(f"      📋 Found encoder.layers with {len(layers)} layers")
        except Exception as e:
            logger.warning(f"   ⚠️ DEBUG: Error exploring structure: {e}")

        logger.error(f"❌ DEBUG: DINOv2 layer '{layer_name}' not found after all checks. Returning None.")
        return None

    def _extract_features_with_hook(self, data_loader, layer_name: str):
        """Enhanced feature extraction with proper DINOv2 output handling."""
        logger.info(f"🔬 Extracting features from layer: {layer_name}")
        
        # Find and register hook
        target_layer = self._find_dinov2_layer(layer_name) if 'dinov2' in self.model_name else self._find_layer_module(layer_name)
        
        if target_layer is None:
            logger.error(f"❌ Layer '{layer_name}' not found!")
            return None
        
        # Storage for features
        all_features = []
        all_labels = []
        
        # Hook function
        def feature_hook(module, input, output):
            # Handle different output types
            if hasattr(output, 'last_hidden_state'):
                # Hugging Face transformer output
                features = output.last_hidden_state
            elif isinstance(output, torch.Tensor):
                # Regular tensor output
                features = output
            else:
                # Try to extract tensor from output
                features = output[0] if isinstance(output, (tuple, list)) else output
            
            # Store features
            if isinstance(features, torch.Tensor):
                all_features.append(features.detach().cpu())
            else:
                logger.warning(f"⚠️ Unexpected feature type: {type(features)}")
        
        # Register hook
        hook = target_layer.register_forward_hook(feature_hook)
        
        try:
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(data_loader):
                    if batch_idx >= 5:  # Limit for optimization
                        break
                    
                    data = data.to(next(self.model.parameters()).device)
                    targets = targets.to(next(self.model.parameters()).device)
                    
                    # Clear previous features
                    all_features.clear()
                    
                    try:
                        # Forward pass to trigger hook
                        _ = self.model(data)
                        
                        if all_features:
                            features = all_features[0]
                            if self.use_compression:
                                features = self._smart_spatial_reduction(features, layer_name)
                            all_labels.extend(targets.cpu().numpy())
                            
                            if batch_idx == 0:
                                logger.info(f"   ✅ Feature shape: {features.shape}")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Batch {batch_idx} failed with error: {e}")
                        continue
        
        finally:
            hook.remove()
        
        if not all_features:
            logger.error(f"❌ No features extracted from {layer_name}")
            return None
        
        logger.info(f"✅ Extracted features from {len(all_features)} batches")
        return torch.cat(all_features, dim=0) if all_features else None