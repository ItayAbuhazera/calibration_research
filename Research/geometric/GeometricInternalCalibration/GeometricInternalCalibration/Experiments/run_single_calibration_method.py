# Experiments/run_single_calibration_method.py
"""
Run a single post-hoc calibration method for memory efficiency
"""

import argparse
import sys
import os
import json
import time
import logging
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Removed shared memory locking mechanism

# Import the existing calibration infrastructure
from Experiments.run_post_hoc_calibration import (
    load_trained_model, get_data_loaders, FeatureExtractor,
    PyTorchModelAdapter, extract_raw_data_and_features
)

# Import calibrators
from Calibrators.temperature_scaling import TemperatureScaling
from Calibrators.isotonic_regression import IsotonicRegressionCalibrator
from Calibrators.geometric_calibrator import GeometricCalibrator
from Calibrators.platt_scaling import PlattScaling
from Calibrators.beta_calibration import BetaCalibration
from Calibrators.dirichlet_calibration import DirichletCalibration

# Import metrics
from Metrics.metrics import expected_calibration_error

# Import corruption evaluation utilities
from Data.cifar10_c import get_cifar10c_loader, CIFAR10C
from Data.cifar100_c import get_cifar100c_loader, CIFAR100C

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from utils.logging_config import get_logger
logger = get_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """
    🔧 FIXED: Custom JSON encoder that handles NumPy types for results serialization.
    
    Converts NumPy numeric types to native Python types that can be JSON serialized:
    - np.integer → int
    - np.floating → float  
    - np.ndarray → list
    - np.bool_ → bool
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):  # NumPy scalars
            return obj.item()
        return super().default(obj)

def setup_cuda_memory_management():
    """
    🔧 FIXED: Setup CUDA memory management to prevent expandable segments error.
    """
    import os
    
    # Critical fix for CUDA expandable segments error
    # Use consistent settings across all components
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:False'
    
    if torch.cuda.is_available():
        # Additional CUDA settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False  # Consistent memory usage
        torch.backends.cudnn.deterministic = True  # Reproducible results
        
        # Clear any existing cache
        torch.cuda.empty_cache()
        
        # Force memory pool initialization to prevent allocation issues
        try:
            # Allocate and immediately free a small tensor to initialize the memory pool
            dummy = torch.zeros(1, device='cuda')
            del dummy
            torch.cuda.empty_cache()
            logger.debug("🔧 CUDA memory pool initialized")
        except Exception as e:
            logger.warning(f"🔧 CUDA memory pool initialization failed: {e}")
        
        logger.info("🔧 CUDA memory management configured")
        logger.info(f"   GPU: {torch.cuda.get_device_name()}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        logger.info(f"   Allocation config: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")

# Removed memory estimation function for shared memory locks

# Removed OOM error checking function

# Removed OOM error handling function

def extract_logits(model, data_loader, device):
    """
    🔧 FIXED: Lightweight function to extract only logits and labels with proper memory management.
    """
    # Setup memory management
    setup_cuda_memory_management()
    
    model.eval()
    all_logits = []
    all_labels = []
    
    logger.info(f"🧠 Extracting logits with lightweight function...")
    
    with torch.no_grad():
        for i, (data, targets) in enumerate(data_loader):
            try:
                data, targets = data.to(device), targets.to(device)
                
                # Clear GPU cache before forward pass (more frequent for early batches)
                if torch.cuda.is_available():
                    if i < 5:  # Clear more frequently for first few batches
                        torch.cuda.empty_cache()
                    elif i % 10 == 0:  # Then every 10 batches
                        torch.cuda.empty_cache()
                
                # Forward pass with error handling
                try:
                    output = model(data)
                except RuntimeError as e:
                    if "expandable_segment" in str(e) or "CUDA" in str(e):
                        logger.error(f"❌ CUDA memory error in batch {i}: {e}")
                        logger.info("🔧 Attempting to recover by clearing cache and reducing batch...")
                        torch.cuda.empty_cache()
                        
                        # Try with smaller batch if possible
                        if data.shape[0] > 1:
                            half_batch = data.shape[0] // 2
                            try:
                                output = model(data[:half_batch])
                                logger.info(f"🔧 Successfully processed half batch ({half_batch} samples)")
                            except Exception as e2:
                                logger.error(f"❌ Even half batch failed: {e2}")
                                continue
                        else:
                            logger.error("❌ Cannot reduce batch size further, skipping batch")
                            continue
                    else:
                        raise e
                
                # Handle different output types (tensor vs ImageClassifierOutput)
                if hasattr(output, 'logits'):
                    # HuggingFace ImageClassifierOutput
                    logits = output.logits
                else:
                    # Direct tensor output
                    logits = output
                
                all_logits.append(logits.cpu())
                all_labels.append(targets.cpu())
                
                if i % 100 == 0:
                    logger.info(f"   Processed batch {i}/{len(data_loader)}")
                    
            except Exception as e:
                logger.error(f"❌ Error in logit extraction batch {i}: {e}")
                # Skip this batch
                continue
    
    if not all_logits:
        raise RuntimeError("No logits were extracted successfully")
    
    final_logits = torch.cat(all_logits).numpy()
    final_labels = torch.cat(all_labels).numpy()
    
    logger.info(f"✅ Extracted logits: {final_logits.shape}, labels: {final_labels.shape}")
    return final_logits, final_labels

def get_calibration_method_config(cal_method: str, use_compression: bool = True) -> dict:
    """
    Get calibration method configuration with smart compression support.
    
    Args:
        cal_method: Calibration method name
        use_compression: Whether to use spatial compression for feature extraction
        
    Returns:
        Configuration dictionary
    """
    configs = {
        # === SIMPLE METHODS ===
        'uncalibrated': {
            'name': 'Uncalibrated',
            'description': 'No calibration baseline',
            'memory_efficient': True,
            'requires_features': False,
            'batch_size_factor': 1.0,
            'use_semantic': False,
            'use_faiss': False,
            'use_augmix': False,
            'smart_compression': False,
            'use_compression': use_compression  # 🔧 RENAMED: use_spp → use_compression
        },
        'temperature': {
            'name': 'Temperature_Scaling',
            'description': 'Temperature scaling (Platt scaling)',
            'memory_efficient': True,
            'requires_features': False,
            'batch_size_factor': 1.0,
            'use_semantic': False,
            'use_faiss': False,
            'use_augmix': False,
            'smart_compression': False,
            'use_compression': use_compression  # 🔧 RENAMED: use_spp → use_compression
        },
        'platt': {
            'name': 'Platt_Scaling',
            'description': 'Platt scaling for binary and multiclass calibration',
            'memory_efficient': True,
            'requires_features': False,
            'batch_size_factor': 1.0,
            'use_semantic': False,
            'use_faiss': False,
            'use_augmix': False,
            'smart_compression': False,
            'use_compression': use_compression
        },
        'beta': {
            'name': 'Beta_Calibration',
            'description': 'Beta calibration for multiclass problems',
            'memory_efficient': True,
            'requires_features': False,
            'batch_size_factor': 1.0,
            'use_semantic': False,
            'use_faiss': False,
            'use_augmix': False,
            'smart_compression': False,
            'use_compression': use_compression
        },
        'dirichlet': {
            'name': 'Dirichlet_Calibration',
            'description': 'Dirichlet calibration for multiclass problems',
            'memory_efficient': True,
            'requires_features': False,
            'batch_size_factor': 1.0,
            'use_semantic': False,
            'use_faiss': False,
            'use_augmix': False,
            'smart_compression': False,
            'use_compression': use_compression
        },
        'isotonic': {
            'name': 'Isotonic_Regression',
            'description': 'Isotonic regression calibration',
            'memory_efficient': True,
            'requires_features': False,
            'batch_size_factor': 1.0,
            'use_semantic': False,
            'use_faiss': False,
            'use_augmix': False,
            'smart_compression': False,
            'use_compression': use_compression  # 🔧 RENAMED: use_spp → use_compression
        },
        
        # === GEOMETRIC FAST SEPARATION ===
        'geometric_semantic_fast': {
            'name': 'Geometric_Semantic_Fast_Separation',
            'description': 'Uses extracted FEATURES for geometric calculations',
            'memory_efficient': False,
            'requires_features': True,
            'batch_size_factor': 0.8,  # Can be more aggressive now with compression
            'use_semantic': True,   # 🧠 Key: Uses FEATURES for geometric calculations
            'use_faiss': False,
            'use_augmix': False,
            'smart_compression': True,  # 🔧 Enable smart compression
            'compression_threshold': 8192,  # Only compress if > 8192 dims (after compression)
            'use_compression': use_compression  # 🔧 RENAMED: use_spp → use_compression
        },
        'geometric_physical_fast': {
            'name': 'Geometric_Physical_Fast_Separation',
            'description': 'Uses raw IMAGES for geometric calculations',
            'memory_efficient': False,
            'requires_features': True,
            'batch_size_factor': 0.8,
            'use_semantic': False,  # 🖼️ Key: Uses RAW IMAGES for geometric calculations
            'use_faiss': False,
            'use_augmix': False,
            'smart_compression': False,  # Physical methods don't need compression
            'use_compression': use_compression  # 🔧 RENAMED: use_spp → use_compression (for feature extraction)
        },
        
        # === GEOMETRIC FAISS ===
        'geometric_semantic_faiss': {
            'name': 'Geometric_Semantic_FAISS',
            'description': 'Uses extracted FEATURES + FAISS for fast similarity search',
            'memory_efficient': False,
            'requires_features': True,
            'batch_size_factor': 0.6,  # FAISS still needs more memory
            'use_semantic': True,   # 🧠 Key: Uses FEATURES for geometric calculations
            'use_faiss': True,
            'use_augmix': False,
            'smart_compression': True,  # 🔧 Enable smart compression for FAISS
            'compression_threshold': 8192,  # Only compress if > 8192 dims (after compression)
            'use_compression': use_compression  # 🔧 RENAMED: use_spp → use_compression
        },
        'geometric_physical_faiss': {
            'name': 'Geometric_Physical_FAISS',
            'description': 'Uses raw IMAGES + FAISS for fast similarity search',
            'memory_efficient': False,
            'requires_features': True,
            'batch_size_factor': 0.6,
            'use_semantic': False,  # 🖼️ Key: Uses RAW IMAGES for geometric calculations
            'use_faiss': True,
            'use_augmix': False,
            'smart_compression': False,  # Physical methods don't need compression
            'use_compression': use_compression  # 🔧 RENAMED: use_spp → use_compression (for feature extraction)
        },
        
        # === GEOMETRIC AUGMIX ===
        'geometric_semantic_fast_augmix': {
            'name': 'Geometric_Semantic_Fast_Separation_AugMix',
            'description': 'Uses extracted FEATURES + AugMix data augmentation',
            'memory_efficient': False,
            'requires_features': True,
            'batch_size_factor': 0.4,  # AugMix needs more memory
            'use_semantic': True,   # �� Key: Uses FEATURES for geometric calculations
            'use_faiss': False,
            'use_augmix': True,
            'smart_compression': True,  # 🔧 Enable smart compression
            'compression_threshold': 8192,  # Only compress if > 8192 dims (after compression)
            'use_compression': use_compression  # 🔧 RENAMED: use_spp → use_compression
        },
        'geometric_physical_fast_augmix': {
            'name': 'Geometric_Physical_Fast_Separation_AugMix',
            'description': 'Uses raw IMAGES + AugMix data augmentation',
            'memory_efficient': False,
            'requires_features': True,
            'batch_size_factor': 0.4,
            'use_semantic': False,  # 🖼️ Key: Uses RAW IMAGES for geometric calculations
            'use_faiss': False,
            'use_augmix': True,
            'smart_compression': False,  # Physical methods don't need compression
            'use_compression': use_compression  # 🔧 RENAMED: use_spp → use_compression (for feature extraction)
        },
        'geometric_semantic_faiss_augmix': {
            'name': 'Geometric_Semantic_FAISS_AugMix',
            'description': 'Uses extracted FEATURES + FAISS + AugMix (highest memory)',
            'memory_efficient': False,
            'requires_features': True,
            'batch_size_factor': 0.2,  # FAISS+AugMix needs most memory
            'use_semantic': True,   # 🧠 Key: Uses FEATURES for geometric calculations
            'use_faiss': True,
            'use_augmix': True,
            'smart_compression': True,  # 🔧 Enable smart compression for FAISS+AugMix
            'compression_threshold': 8192,  # Only compress if > 8192 dims (after compression)
            'use_compression': use_compression  # 🔧 RENAMED: use_spp → use_compression
        },
        'geometric_physical_faiss_augmix': {
            'name': 'Geometric_Physical_FAISS_AugMix',
            'description': 'Uses RAW IMAGES + FAISS + AugMix (highest memory)',
            'memory_efficient': False,
            'requires_features': True,
            'batch_size_factor': 0.2,
            'use_semantic': False,  # 🖼️ Key: Uses RAW IMAGES for geometric calculations
            'use_faiss': True,
            'use_augmix': True,
            'smart_compression': False,  # Physical methods don't need compression
            'use_compression': use_compression  # 🔧 RENAMED: use_spp → use_compression (for feature extraction)
        }
    }
    
    if cal_method not in configs:
        raise ValueError(f"Unknown calibration method: {cal_method}")
    
    return configs[cal_method]

def get_enhanced_layer_calibration_config(cal_method: str) -> dict:
    """
    Generate layer-specific calibration method configurations for systematic experiments.
    
    Args:
        cal_method: Method name with layer specification (e.g., 'geometric_semantic_fast_layer3')
        
    Returns:
        Enhanced configuration with layer-specific settings
    """
    # Parse method and layer from the calibration method name
    method_parts = cal_method.split('_')
    
    # Find the layer part
    layer_name = None
    base_method = None
    
    # Check for known layer patterns
    layer_patterns = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 
                     'denseblock1', 'denseblock2', 'denseblock3', 'denseblock4',
                     'transition1', 'transition2', 'transition3', 'final_norm']
    
    for i, part in enumerate(method_parts):
        if part in layer_patterns:
            layer_name = part
            base_method = '_'.join(method_parts[:i])
            break
    
    if not layer_name or not base_method:
        logger.warning(f"⚠️ Could not parse layer from method name: {cal_method}")
        return {}
    
    # Get base configuration
    base_config = get_calibration_method_config(base_method)
    if not base_config:
        logger.warning(f"⚠️ Unknown base method: {base_method}")
        return {}
    
    # Create enhanced configuration
    enhanced_config = {
        **base_config,
        'forced_layer': layer_name,
        'layer_optimization': True,
        'description': f"{base_config.get('description', base_config.get('name', base_method))} with forced {layer_name} extraction",
        'name': f"{base_config.get('name', base_method)}_{layer_name}"
    }
    
    # Adjust batch size for layer-specific experiments (may need more memory)
    if enhanced_config.get('batch_size_factor', 1.0) > 0.1:
        enhanced_config['batch_size_factor'] = max(0.1, enhanced_config['batch_size_factor'] * 0.8)
    
    logger.info(f"🔧 Generated layer-specific config: {cal_method}")
    logger.info(f"   Base method: {base_method}")
    logger.info(f"   Forced layer: {layer_name}")
    logger.info(f"   Batch size factor: {enhanced_config['batch_size_factor']}")
    
    return enhanced_config

def evaluate_corruption_robustness(calibrator, method_name, cal_method_config, 
                                   model_adapter, feature_extractor, dataset_name, 
                                   device, batch_size, cal_method, cifar10c_dir=None, cifar100c_dir=None):
    """
    🔧 FIXED: Corruption evaluation with proper data flow for geometric physical calibration
    """
    logger.info(f"🔬 Evaluating corruption robustness on {dataset_name.upper()}-C...")

    if dataset_name == 'cifar10':
        if not cifar10c_dir or not os.path.exists(cifar10c_dir):
            logger.warning("CIFAR-10-C directory not provided or not found. Skipping corruption evaluation.")
            return None
        corruption_data_dir = cifar10c_dir
        loader_fn = get_cifar10c_loader
        corruption_types = CIFAR10C.CORRUPTION_TYPES
    elif dataset_name == 'cifar100':
        if not cifar100c_dir or not os.path.exists(cifar100c_dir):
            logger.warning("CIFAR-100-C directory not provided or not found. Skipping corruption evaluation.")
            return None
        corruption_data_dir = cifar100c_dir
        loader_fn = get_cifar100c_loader
        corruption_types = CIFAR100C.CORRUPTION_TYPES
    else:
        logger.warning(f"Corruption evaluation not supported for {dataset_name}.")
        return None

    # Detect method type
    is_simple_method = not cal_method_config.get('requires_features', False)
    use_semantic = cal_method_config.get('use_semantic', True)

    logger.info(f"🔍 Corruption evaluation - Method type: {'Simple' if is_simple_method else 'Geometric'}")
    logger.info(f"🔍 Geometric mode: {'Semantic' if use_semantic else 'Physical'}")

    # Initialize detailed results storage
    detailed_results = {
        'corruption_details': {},
        'severity_summaries': {},
        'corruption_summaries': {},
        'overall_summary': {}
    }
    
    corruption_accuracies = []
    corruption_eces = []
    
    # Track results by severity and corruption type
    severity_results = {f'severity_{i}': {'accuracies': [], 'eces': []} for i in range(1, 6)}
    corruption_type_results = {corruption: {'accuracies': [], 'eces': []} for corruption in corruption_types}

    logger.info(f"   Testing {len(corruption_types)} corruption types × 5 severities = {len(corruption_types) * 5} combinations")

    for corruption in corruption_types:
        logger.info(f"   🔍 Testing corruption: {corruption}")
        detailed_results['corruption_details'][corruption] = {}
        
        for severity in range(1, 6):
            try:
                # Load corrupted test data
                c_loader = loader_fn(
                    root=corruption_data_dir,
                    corruption_type=corruption,
                    severity=severity,
                    batch_size=batch_size,
                    num_workers=0
                )

                # 🔧 Extract data based on calibration method requirements (without locking)
                if is_simple_method:
                    # Simple methods: only need logits
                    c_logits, c_labels = extract_logits(model_adapter.model, c_loader, device)
                    c_raw, c_features = None, None
                else:
                    # Geometric methods: need both raw data and features
                    c_raw, c_features, c_logits, c_labels = extract_raw_data_and_features(
                        feature_extractor, c_loader, device
                    )
                    # Validate extraction for geometric methods
                    if c_raw is None or c_features is None:
                        raise RuntimeError("Failed to extract raw data or features for geometric method")

                # 🔧 CRITICAL FIX: Apply calibration with correct data types
                if is_simple_method:
                    # Simple methods: use logits only
                    cal_probs = calibrator.calibrate(c_logits)
                else:
                    # Geometric methods: use correct data based on semantic/physical mode
                    if use_semantic:
                        # Semantic geometric: use features for geometric calc, raw for model predictions
                        if cal_method_config.get('use_augmix', False):
                            cal_probs = calibrator.calibrate_batched(
                                c_features,           # ✅ Features for geometric calculations
                                X_test_original=c_raw, # ✅ Raw images for model predictions
                                batch_size=batch_size
                            )
                        else:
                            cal_probs = calibrator.calibrate(
                                c_features,           # ✅ Features for geometric calculations
                                X_test_original=c_raw  # ✅ Raw images for model predictions
                            )
                    else:
                        # Physical geometric: use raw for both geometric calc AND model predictions
                        if cal_method_config.get('use_augmix', False):
                            cal_probs = calibrator.calibrate_batched(
                                c_raw,                # ✅ Raw images for geometric calculations
                                X_test_original=c_raw, # ✅ Same raw images for model predictions
                                batch_size=batch_size
                            )
                        else:
                            cal_probs = calibrator.calibrate(
                                c_raw,                # ✅ Raw images for geometric calculations 
                                X_test_original=c_raw  # ✅ Same raw images for model predictions
                            )

                # Validate calibration output
                if cal_probs is None or cal_probs.shape[0] == 0:
                    raise RuntimeError("Empty calibration output")
                
                # Calculate metrics
                cal_preds = np.argmax(cal_probs, axis=1)
                cal_confs = np.max(cal_probs, axis=1)
                accuracy = 100.0 * np.mean(cal_preds == c_labels)
                ece = expected_calibration_error(cal_confs, cal_preds, c_labels)

                # Store detailed results
                detailed_results['corruption_details'][corruption][f'severity_{severity}'] = {
                    'accuracy': float(accuracy),
                    'ece': float(ece),
                    'num_samples': len(c_labels),
                    'corruption_type': corruption,
                    'severity_level': severity
                }

                # Add to tracking lists
                corruption_accuracies.append(accuracy)
                corruption_eces.append(ece)
                
                # Track by severity and corruption type
                severity_results[f'severity_{severity}']['accuracies'].append(accuracy)
                severity_results[f'severity_{severity}']['eces'].append(ece)
                corruption_type_results[corruption]['accuracies'].append(accuracy)
                corruption_type_results[corruption]['eces'].append(ece)

                # Print/log summary for this corruption and severity
                logger.info(f"      📊 {corruption} (severity {severity}): Accuracy={accuracy:.2f}%, ECE={ece:.4f}")
                
            except Exception as e:
                logger.error(f"      ❌ Failed {corruption} severity {severity}: {e}")
                # Store error information and continue
                detailed_results['corruption_details'][corruption][f'severity_{severity}'] = {
                    'accuracy': 0.0,
                    'ece': 1.0,
                    'num_samples': 0,
                    'corruption_type': corruption,
                    'severity_level': severity,
                    'error': str(e)
                }
                
                corruption_accuracies.append(0.0)
                corruption_eces.append(1.0)
                severity_results[f'severity_{severity}']['accuracies'].append(0.0)
                severity_results[f'severity_{severity}']['eces'].append(1.0)
                corruption_type_results[corruption]['accuracies'].append(0.0)
                corruption_type_results[corruption]['eces'].append(1.0)

    # Calculate severity-wise summaries
    for severity_key, severity_data in severity_results.items():
        if severity_data['accuracies']:
            detailed_results['severity_summaries'][severity_key] = {
                'mean_accuracy': float(np.mean(severity_data['accuracies'])),
                'std_accuracy': float(np.std(severity_data['accuracies'])),
                'mean_ece': float(np.mean(severity_data['eces'])),
                'std_ece': float(np.std(severity_data['eces'])),
                'num_corruptions': len(severity_data['accuracies']),
                'severity_level': int(severity_key.split('_')[1])
            }

    # Calculate corruption-type summaries
    for corruption_type, corruption_data in corruption_type_results.items():
        if corruption_data['accuracies']:
            detailed_results['corruption_summaries'][corruption_type] = {
                'mean_accuracy': float(np.mean(corruption_data['accuracies'])),
                'std_accuracy': float(np.std(corruption_data['accuracies'])),
                'mean_ece': float(np.mean(corruption_data['eces'])),
                'std_ece': float(np.std(corruption_data['eces'])),
                'num_severities': len(corruption_data['accuracies']),
                'corruption_type': corruption_type
            }

    # Calculate overall statistics
    mean_corruption_acc = np.mean(corruption_accuracies) if corruption_accuracies else 0.0
    mean_corruption_ece = np.mean(corruption_eces) if corruption_eces else 1.0
    
    detailed_results['overall_summary'] = {
        'mean_corruption_accuracy': float(mean_corruption_acc),
        'mean_corruption_ece': float(mean_corruption_ece),
        'std_corruption_accuracy': float(np.std(corruption_accuracies)) if corruption_accuracies else 0.0,
        'std_corruption_ece': float(np.std(corruption_eces)) if corruption_eces else 0.0,
        'total_corruption_combinations': len(corruption_accuracies),
        'successful_evaluations': len([acc for acc in corruption_accuracies if acc > 0]),
        'failed_evaluations': len([acc for acc in corruption_accuracies if acc == 0])
    }

    logger.info(f"   ✅ Mean Corruption Accuracy: {mean_corruption_acc:.2f}%")
    logger.info(f"   ✅ Mean Corruption ECE: {mean_corruption_ece:.4f}")
    logger.info(f"   📊 Total evaluations: {len(corruption_accuracies)} ({detailed_results['overall_summary']['successful_evaluations']} successful)")

    return {
        'mean_corruption_accuracy': float(mean_corruption_acc),
        'mean_corruption_ece': float(mean_corruption_ece),
        'corruption_detailed_results': detailed_results
    }

def run_single_calibration_method(model, feature_extractor, train_loader, val_loader, test_loader, 
                                 device, num_classes, method_name, dataset_name, cal_method, 
                                 batch_size=128, enable_corruption_eval=False, cifar10c_dir=None, 
                                 cifar100c_dir=None, use_compression=True, forced_layer=None):
    """
    🔧 FIXED: Enhanced calibration method with proper memory management and error handling.
    """
    # Setup CUDA memory management first
    setup_cuda_memory_management()
    
    logger.info(f"🚀 Running calibration method: {cal_method}")
    logger.info(f"   Training method: {method_name}")
    logger.info(f"   Dataset: {dataset_name}")
    logger.info(f"   Use compression: {use_compression}")
    
    # Get calibration method configuration
    config = get_calibration_method_config(cal_method, use_compression=use_compression)
    
    logger.info(f"🔍 Running {config['name']} calibration...")
    if 'description' in config:
        logger.info(f"   📋 Method: {config['description']}")
    
    # Layer optimization detection
    is_layer_optimization = (forced_layer is not None) or config.get('layer_optimization', False)
    forced_layer = forced_layer or config.get('forced_layer', None)
    
    # Enhanced feature extractor setup for geometric methods
    if is_layer_optimization and forced_layer and config['requires_features']:
        logger.info(f"🎯 Layer Optimization Mode: Using forced layer '{forced_layer}'")
        
        if hasattr(feature_extractor, 'forced_layer') and feature_extractor.forced_layer:
            logger.info(f"🔧 Layer optimization: Using existing forced layer {feature_extractor.forced_layer}")
            if hasattr(feature_extractor, 'use_compression'):
                logger.info(f"🔧 Existing feature extractor compression setting: {feature_extractor.use_compression}")
        else:
            from utils.enhanced_layer_feature_extractor import EnhancedLayerFeatureExtractor
            
            # Clean up existing feature extractor first
            if hasattr(feature_extractor, 'cleanup'):
                feature_extractor.cleanup()
            
            feature_extractor = EnhancedLayerFeatureExtractor(
                model=model,
                model_name=method_name,
                forced_layer=forced_layer,
                use_compression=use_compression
            )
            logger.info(f"✅ Feature extractor integrated for {method_name} with compression={use_compression}")
    
    # Memory-aware batch size adjustment
    effective_batch_size = max(1, int(batch_size * config['batch_size_factor']))
    logger.info(f"🔧 Using batch size: {effective_batch_size} (factor: {config['batch_size_factor']})")
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 Cleared GPU cache")
    
    try:
        # Extract data based on method requirements
        if config['requires_features']:
            logger.info("🔍 Extracting both raw data and features for geometric calibration...")
            
            # Extract data directly without locking
            try:
                # Extract features 
                train_raw, train_features, train_logits, train_labels = extract_raw_data_and_features(
                    feature_extractor, train_loader, device)
                val_raw, val_features, val_logits, val_labels = extract_raw_data_and_features(
                    feature_extractor, val_loader, device)
                test_raw, test_features, test_logits, test_labels = extract_raw_data_and_features(
                    feature_extractor, test_loader, device)
                
                logger.info("✅ Feature extraction completed successfully!")
                logger.info(f"📊 Extracted data shapes:")
                logger.info(f"   Raw images: {val_raw.shape} (for physical geometric & model predictions)")
                logger.info(f"   Features: {val_features.shape} (for semantic geometric calculations)")
                logger.info(f"   Logits: {val_logits.shape} (for simple calibration methods)")
                
            except Exception as e:
                logger.error(f"❌ Data extraction failed: {e}")
                
                return {
                    'accuracy': 0.0,
                    'ece': 1.0,
                    'num_samples': 0,
                    'error': f'Data extraction failed: {str(e)}',
                    'calibration_time': 0.0,
                    'effective_batch_size': effective_batch_size
                }
                
        else:
            logger.info("🔍 Extracting logits only for simple calibration methods (memory efficient)...")
            
            # Clean up any hooks to avoid interference
            if hasattr(feature_extractor, 'cleanup'):
                feature_extractor.cleanup()
            
            train_raw, train_features, train_logits, train_labels = None, None, None, None
            val_raw, val_features, test_raw, test_features = None, None, None, None
            
            try:
                # Extract logits directly without locking
                val_logits, val_labels = extract_logits(model, val_loader, device)
                test_logits, test_labels = extract_logits(model, test_loader, device)
                
                logger.info("✅ Logit extraction completed successfully!")
                logger.info(f"📊 Extracted val_logits shape: {val_logits.shape}")
                
            except Exception as e:
                logger.error(f"❌ Logit extraction failed: {e}")
                return {
                    'accuracy': 0.0,
                    'ece': 1.0,
                    'num_samples': 0,
                    'error': f'Logit extraction failed: {str(e)}',
                    'calibration_time': 0.0,
                    'effective_batch_size': effective_batch_size
                }
        
        # Check uncalibrated accuracy baseline
        uncal_probs = F.softmax(torch.tensor(test_logits), dim=1).numpy()
        uncal_preds = np.argmax(uncal_probs, axis=1)
        uncal_accuracy = 100.0 * np.mean(uncal_preds == test_labels)
        
        logger.info(f"📊 Initial uncalibrated accuracy on clean data: {uncal_accuracy:.2f}%")
        
        # Check accuracy threshold
        if uncal_accuracy < 50.0:
            logger.warning(f"📉 Initial accuracy ({uncal_accuracy:.2f}%) is below 50% threshold. Aborting calibration.")
            return {
                'accuracy': uncal_accuracy,
                'ece': 1.0,
                'num_samples': len(test_labels),
                'error': f'Initial accuracy {uncal_accuracy:.2f}% is below 50% threshold.',
                'calibration_time': 0.0,
                'effective_batch_size': effective_batch_size
            }
        
        # Create model adapter
        model_adapter = PyTorchModelAdapter(model, device, dataset_name, feature_extractor)
        
        # Initialize and run calibrator
        start_time = time.time()
        calibrator = None
        
        if cal_method == 'uncalibrated':
            test_probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
            calibration_time = time.time() - start_time
            
        elif cal_method == 'temperature':
            calibrator = TemperatureScaling()
            calibrator.fit(val_logits, val_labels)
            start_time = time.time()
            test_probs = calibrator.calibrate(test_logits)
            calibration_time = time.time() - start_time
            
        elif cal_method == 'isotonic':
            calibrator = IsotonicRegressionCalibrator()
            calibrator.fit(val_logits, val_labels)
            start_time = time.time()
            test_probs = calibrator.calibrate(test_logits)
            calibration_time = time.time() - start_time
            
        elif cal_method == 'platt':
            calibrator = PlattScaling()
            calibrator.fit(val_logits, val_labels)
            start_time = time.time()
            test_probs = calibrator.calibrate(test_logits)
            calibration_time = time.time() - start_time
            
        elif cal_method == 'beta':
            calibrator = BetaCalibration()
            calibrator.fit(val_logits, val_labels)
            start_time = time.time()
            test_probs = calibrator.calibrate(test_logits)
            calibration_time = time.time() - start_time
            
        elif cal_method == 'dirichlet':
            calibrator = DirichletCalibration()
            calibrator.fit(val_logits, val_labels)
            start_time = time.time()
            test_probs = calibrator.calibrate(test_logits)
            calibration_time = time.time() - start_time
            
        elif cal_method.startswith('geometric'):
            # Configure geometric calibrator
            use_semantic = config.get('use_semantic', True)
            use_faiss = config.get('use_faiss', True) 
            use_augmix = config.get('use_augmix', False)
            smart_compression = config.get('smart_compression', False)
            
            # Force fast_separation for CIFAR-100 to avoid memory issues
            if dataset_name == 'cifar100' and use_faiss:
                logger.warning("🔄 Forcing fast_separation for CIFAR-100 to avoid memory issues")
                use_faiss = False
            
            # Semantic vs Physical distinction
            if use_semantic:
                train_embed_data = train_features
                val_embed_data = val_features  
                test_embed_data = test_features
                logger.info("🧠 Semantic Geometric Calibration: Using extracted features")
            else:
                train_embed_data = train_raw
                val_embed_data = val_raw
                test_embed_data = test_raw
                logger.info("🖼️ Physical Geometric Calibration: Using raw images")
            
            # Initialize geometric calibrator with memory management
            try:
                calibrator = GeometricCalibrator(
                    model=model_adapter,
                    X_train_embed=train_embed_data,
                    y_train=train_labels,
                    X_train_original=train_raw,
                    compression_mode=None,  # Compression handled by feature extractor
                    compression_param=None,
                    metric='l2',
                    library='faiss' if use_faiss else 'fast_separation',
                    use_binning=True,
                    n_bins=200,
                    use_augmix=use_augmix,
                    augmix_versions=3 if use_augmix else 0,
                    augmix_severity=3 if use_augmix else 0,
                    augmix_width=3 if use_augmix else 0
                )
                
                logger.info(f"🔧 Fitting geometric calibrator with batched processing...")
                if val_embed_data is not None and val_raw is not None:
                    logger.info(f"   📊 Embedding data shape: {val_embed_data.shape}")
                    logger.info(f"   📊 Original data shape: {val_raw.shape}")
                
                # Fit with memory-efficient batching
                calibrator.fit(val_embed_data, val_labels, X_val_original=val_raw, fit_batch_size=effective_batch_size)
                
                # Calibrate test data
                start_time = time.time()
                test_probs = calibrator.calibrate_batched(test_embed_data, X_test_original=test_raw, batch_size=effective_batch_size)
                calibration_time = time.time() - start_time
                
            except Exception as e:
                logger.error(f"❌ Geometric calibration failed: {e}")
                
                return {
                    'accuracy': 0.0,
                    'ece': 1.0,
                    'num_samples': len(test_labels),
                    'error': f'Geometric calibration failed: {str(e)}',
                    'calibration_time': 0.0,
                    'effective_batch_size': effective_batch_size
                }
        
        else:
            raise ValueError(f"Unknown calibration method: {cal_method}")
        
        # Evaluate performance
        test_preds = np.argmax(test_probs, axis=1)
        test_confs = np.max(test_probs, axis=1)
        
        accuracy = 100.0 * np.mean(test_preds == test_labels)
        ece = expected_calibration_error(test_confs, test_preds, test_labels)
        
        result = {
            'accuracy': accuracy,
            'ece': ece,
            'num_samples': len(test_labels),
            'calibration_time': calibration_time,
            'effective_batch_size': effective_batch_size
        }
        
        # Add layer optimization information
        if is_layer_optimization and forced_layer:
            result['layer_optimization'] = {
                'forced_layer': forced_layer,
                'layer_optimization_enabled': True
            }
            
            if hasattr(feature_extractor, 'get_layer_performance_metrics'):
                result['layer_metrics'] = feature_extractor.get_layer_performance_metrics()
        
        logger.info(f"   ✅ {config['name']}: Accuracy={accuracy:.2f}%, ECE={ece:.4f}, Time={calibration_time:.2f}s")
        
        # Corruption evaluation if enabled
        if enable_corruption_eval:
            try:
                # 🔧 FIXED: Handle corruption evaluation for ALL calibration methods
                if cal_method in ['uncalibrated', 'temperature', 'isotonic', 'platt', 'beta', 'dirichlet']:
                    # For simple methods, create a dummy calibrator and config for compatibility
                    class SimpleCalibrator:
                        def __init__(self, method_name, test_logits):
                            self.method_name = method_name
                            self.test_logits = test_logits
                            self.is_fitted = True
                        
                        def calibrate(self, logits):
                            if self.method_name == 'uncalibrated':
                                return torch.softmax(torch.tensor(logits), dim=1).numpy()
                            elif self.method_name == 'temperature':
                                # Use temperature scaling
                                temp_calibrator = TemperatureScaling()
                                temp_calibrator.fit(val_logits, val_labels)
                                return temp_calibrator.calibrate(logits)
                            elif self.method_name == 'isotonic':
                                # Use isotonic regression
                                iso_calibrator = IsotonicRegressionCalibrator()
                                iso_calibrator.fit(val_logits, val_labels)
                                return iso_calibrator.calibrate(logits)
                            elif self.method_name == 'platt':
                                platt_calibrator = PlattScaling()
                                platt_calibrator.fit(val_logits, val_labels)
                                return platt_calibrator.calibrate(logits)
                            elif self.method_name == 'beta':
                                beta_calibrator = BetaCalibration()
                                beta_calibrator.fit(val_logits, val_labels)
                                return beta_calibrator.calibrate(logits)
                            elif self.method_name == 'dirichlet':
                                dirichlet_calibrator = DirichletCalibration()
                                dirichlet_calibrator.fit(val_logits, val_labels)
                                return dirichlet_calibrator.calibrate(logits)
                            else:
                                return torch.softmax(torch.tensor(logits), dim=1).numpy()
                    
                    # Create dummy calibrator and config for simple methods
                    dummy_calibrator = SimpleCalibrator(cal_method, test_logits)
                    simple_config = {
                        'use_semantic': False,
                        'use_faiss': False,
                        'use_augmix': False
                    }
                    
                    corruption_results = evaluate_corruption_robustness(
                        dummy_calibrator, method_name, simple_config, model_adapter, feature_extractor,
                        dataset_name, device, effective_batch_size,
                        cal_method, cifar10c_dir=cifar10c_dir, cifar100c_dir=cifar100c_dir
                    )
                else:
                    # For geometric methods, use the existing logic
                    corruption_results = evaluate_corruption_robustness(
                        calibrator, method_name, config, model_adapter, feature_extractor,
                        dataset_name, device, effective_batch_size,
                        cal_method, cifar10c_dir=cifar10c_dir, cifar100c_dir=cifar100c_dir
                    )
                
                if corruption_results:
                    result.update(corruption_results)
                    mca = result.get('mean_corruption_accuracy', -1)
                    mce = result.get('mean_corruption_ece', -1)
                    logger.info(f"   🌪️ {config['name']} Corruption: mAcc={mca:.2f}%, mECE={mce:.4f}")
            except Exception as e:
                logger.warning(f"⚠️ Corruption evaluation failed: {e}")
                import traceback
                logger.warning(f"📋 Corruption evaluation traceback:\n{traceback.format_exc()}")
        
        return result
        
    except Exception as e:
        logger.error(f"   ❌ {config['name']} failed: {e}")
        import traceback
        logger.error(f"   📋 Full traceback:\n{traceback.format_exc()}")
        
        return {
            'accuracy': 0.0,
            'ece': 1.0,
            'num_samples': len(test_labels) if 'test_labels' in locals() else 0,
            'error': str(e),
            'calibration_time': 0.0,
            'effective_batch_size': effective_batch_size
        }
    
    finally:
        # Always clear GPU cache and cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if hasattr(feature_extractor, 'cleanup'):
            try:
                feature_extractor.cleanup()
            except Exception as e:
                logger.warning(f"⚠️ Final cleanup failed: {e}")

def construct_model_path(base_dir: str, method: str, dataset: str, model: str, seed: int, target_domain: str = None) -> str:
    """Construct the path to a trained model, with special handling for PACS domain adaptation and DINOv2 models."""
    if method.startswith('baseline_'):
        exp_type = 'baseline'
    elif method in ['augmix', 'augmix_constellation']:
        exp_type = 'baseline'
    else:
        exp_type = 'geometric'
    
    if dataset == 'pacs':
        if not target_domain:
            # Default to 'sketch' if not provided, or handle as an error
            # For this case, let's align with the calling logic which might depend on this path.
            # A more robust solution might be to raise an error if it's strictly required.
            target_domain = 'sketch' # Or raise ValueError("target_domain is required for PACS dataset")
        
        all_domains = ["photo", "art_painting", "cartoon", "sketch"]
        source_domains = [d for d in all_domains if d != target_domain]
        source_domains_str = "-".join(source_domains)
        
        model_name = f"{method}_pacs_DA_{source_domains_str}_to_{target_domain}_{model}_seed{seed}"
        model_path = os.path.join(
            base_dir, exp_type, method, 'pacs', f"DA_to_{target_domain}", model, f"seed{seed}", model_name, "best_model.pth"
        )
    else:
        # Special handling for DINOv2 models
        is_baseline_method = method.startswith('baseline_') or method in ['augmix', 'augmix_constellation']
        
        # Calibration comparison path is only for specific, non-baseline DINOv2 experiments
        if model in ['dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant'] and not is_baseline_method:
            # Use calibration comparison path structure
            model_name = f"{method}_{dataset}_{model}_seed{seed}"
            # Look for the model in calibration_comparison directory with job ID suffix
            import glob
            calibration_pattern = os.path.join(base_dir, "calibration_comparison", f"{model_name}*", "best_model.pth")
            logger.info(f"🔍 Looking for DINOv2 model with pattern: {calibration_pattern}")
            matching_files = glob.glob(calibration_pattern)
            logger.info(f"🔍 Found {len(matching_files)} matching files: {matching_files}")
            
            if matching_files:
                # Use the first matching file (should be the only one for a given seed)
                model_path = matching_files[0]
                logger.info(f"🔍 Found DINOv2 model in calibration comparison: {model_path}")
            else:
                # Fallback to standard path structure for non-baseline DINOv2 if not found in calibration_comparison
                model_path = os.path.join(
                    base_dir, exp_type, method, dataset, model, f"seed{seed}", model_name, "best_model.pth"
                )
                logger.warning(f"⚠️ DINOv2 model not found in calibration comparison, trying standard path: {model_path}")
        else:
            # Standard path structure for all other models, including from-scratch and baseline DINOv2 models
            model_name = f"{method}_{dataset}_{model}_seed{seed}"
            model_path = os.path.join(
                base_dir, exp_type, method, dataset, model, f"seed{seed}", model_name, "best_model.pth"
            )
        
    return model_path

def load_trained_model(model_path: str, model_name: str, num_classes: int, device, dataset: str = 'cifar10'):
    """Load a trained model from checkpoint, handling model wrappers and architecture mismatches."""
    
    # 🔧 FIXED: Use the correct model architecture based on the dataset
    if dataset == 'pacs':
        # For PACS, we need ImageNet-style models
        import torchvision.models as models
        if model_name == 'resnet18':
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'resnet50':
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'densenet121':
            model = models.densenet121(weights=None)
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        else:
            raise ValueError(f"Unknown model for PACS: {model_name}")
    else:
        # Original logic for CIFAR/SVHN
        if model_name == 'resnet18':
            from Net.resnet import resnet18
            model = resnet18(num_classes=num_classes)
        elif model_name == 'resnet50':
            from Net.resnet import resnet50
            model = resnet50(num_classes=num_classes)
        elif model_name == 'densenet121':
            from Net.densenet import densenet121
            model = densenet121(num_classes=num_classes)
        elif model_name in ['dinov2_small', 'dinov2_small_scratch']:
            # DINOv2 small model
            is_baseline_model = '/baseline/' in model_path
            if 'scratch' in model_name or is_baseline_model:
                # Load from-scratch DINOv2 model (your custom trained model)
                from Net.dinov2 import DINOv2Classifier
                model = DINOv2Classifier(
                    num_classes=num_classes,
                    model_size='small',
                    freeze_backbone=False,  # Train from scratch
                    use_pretrained=False   # From scratch
                )
                logger.info("🔧 Loading from-scratch DINOv2 small model (or baseline model)")
            else:
                # Load pre-trained HuggingFace DINOv2 model
                from transformers import AutoModelForImageClassification
                model = AutoModelForImageClassification.from_pretrained(
                    "facebook/dinov2-small",
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
                logger.info("🔧 Loading pre-trained HuggingFace DINOv2 small model")
        elif model_name in ['dinov2_base', 'dinov2_base_scratch']:
            # DINOv2 base model
            is_baseline_model = '/baseline/' in model_path
            if 'scratch' in model_name or is_baseline_model:
                # Load from-scratch DINOv2 model (your custom trained model)
                from Net.dinov2 import DINOv2Classifier
                model = DINOv2Classifier(
                    num_classes=num_classes,
                    model_size='base',
                    freeze_backbone=False,  # Train from scratch
                    use_pretrained=False   # From scratch
                )
                logger.info("🔧 Loading from-scratch DINOv2 base model (or baseline model)")
            else:
                # Load pre-trained HuggingFace DINOv2 model
                from transformers import AutoModelForImageClassification
                model = AutoModelForImageClassification.from_pretrained(
                    "facebook/dinov2-base",
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
                logger.info("🔧 Loading pre-trained HuggingFace DINOv2 base model")
        elif model_name in ['dinov2_large', 'dinov2_large_scratch']:
            # DINOv2 large model
            is_baseline_model = '/baseline/' in model_path
            if 'scratch' in model_name or is_baseline_model:
                from Net.dinov2 import DINOv2Classifier
                model = DINOv2Classifier(
                    num_classes=num_classes,
                    model_size='large',
                    freeze_backbone=False,
                    use_pretrained=False
                )
                logger.info("🔧 Loading from-scratch DINOv2 large model (or baseline model)")
            else:
                from transformers import AutoModelForImageClassification
                model = AutoModelForImageClassification.from_pretrained(
                    "facebook/dinov2-large",
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
                logger.info("🔧 Loading pre-trained HuggingFace DINOv2 large model")
        elif model_name in ['dinov2_giant', 'dinov2_giant_scratch']:
            # DINOv2 giant model
            is_baseline_model = '/baseline/' in model_path
            if 'scratch' in model_name or is_baseline_model:
                from Net.dinov2 import DINOv2Classifier
                model = DINOv2Classifier(
                    num_classes=num_classes,
                    model_size='giant',
                    freeze_backbone=False,
                    use_pretrained=False
                )
                logger.info("🔧 Loading from-scratch DINOv2 giant model (or baseline model)")
            else:
                from transformers import AutoModelForImageClassification
                model = AutoModelForImageClassification.from_pretrained(
                    "facebook/dinov2-giant",
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
                logger.info("🔧 Loading pre-trained HuggingFace DINOv2 giant model")
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # 🔧 NEW: Check for empty/corrupt file before loading
    if os.path.getsize(model_path) == 0:
        raise IOError(f"Model file is empty or corrupt: {model_path}. The training process may have failed.")
    
    state_dict = torch.load(model_path, map_location=device)
    
    # 🔧 FIXED: Remap state_dict keys to handle 'downsample' vs 'shortcut' mismatch
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('downsample', 'shortcut')
        if new_key.startswith('module.'):
            new_key = new_key[len('module.'):]
        new_state_dict[new_key] = value
    
    # Check for wrapped model prefix
    sample_key = next(iter(new_state_dict.keys()))
    
    if sample_key.startswith('base_model.'):
        logger.info("🔄 Detected wrapped model - extracting base_model weights")
        base_model_state_dict = {}
        for key, value in new_state_dict.items():
            if key.startswith('base_model.'):
                new_key = key[len('base_model.'):]
                base_model_state_dict[new_key] = value
        
        # Additional fix for shortcut vs downsample in wrapped models
        final_state_dict = {}
        for key, value in base_model_state_dict.items():
            new_key = key.replace('downsample', 'shortcut')
            final_state_dict[new_key] = value
            
        model.load_state_dict(final_state_dict, strict=False)
        logger.info("✅ Successfully loaded wrapped model weights with remapping")
        
    else:
        # Standard unwrapped model
        model.load_state_dict(new_state_dict, strict=False)
        logger.info("✅ Loaded standard model weights with remapping")
    
    model = model.to(device)
    logger.info(f"✅ Model loaded and moved to {device}")
    return model

def main():
    """Main function with enhanced CUDA memory management."""
    
    parser = argparse.ArgumentParser(description="Run single post-hoc calibration method")
    
    # Model specification
    parser.add_argument('--method', type=str, required=True,
                       help='Training method (e.g., baseline_focal, augmix, etc.)')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['cifar10', 'cifar100', 'svhn', 'pacs'],
                       help='Dataset name')
    parser.add_argument('--model', type=str, required=True,
                       help='Model architecture')
    parser.add_argument('--seed', type=int, required=True,
                       help='Random seed used in training')
    parser.add_argument('--calibration_method', type=str, required=True,
                       help='Specific calibration method to run')
    
    # Paths
    parser.add_argument('--results_dir', type=str, 
                       default='aaai_full_experiments/results',
                       help='Base directory containing Phase 1 results')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output JSON file for results')
    
    # Parameters
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Base batch size (will be adjusted per method)')
    # PACS target domain
    parser.add_argument('--target_domain', type=str, default=None,
                       help='PACS target domain')
    parser.add_argument('--pacs_mode', type=str, default='domain_adaptation',
                       help='PACS mode')

    # Layer optimization (optional)
    parser.add_argument('--forced_layer', type=str, default=None,
                       help='Force specific layer for feature extraction (for layer optimization)')
    
    # 🔧 NEW: Compression Configuration
    parser.add_argument('--use_compression', type=str, default='true', choices=['true', 'false'],
                       help='Use spatial compression (true) or keep full dimensions (false)')
    
    parser.add_argument('--enable_corruption_eval', action='store_true',
                       help='Enable evaluation on corrupted datasets (e.g., CIFAR-10-C)')
    parser.add_argument('--cifar10c_dir', type=str, default=None,
                       help='Path to CIFAR-10-C dataset root directory')
    parser.add_argument('--cifar100c_dir', type=str, default=None,
                       help='Path to CIFAR-100-C dataset root directory')
    
    args = parser.parse_args()
    
    # 🔧 NEW: Check if results already exist
    if os.path.exists(args.output_file):
        logger.info(f"✅ Results already exist: {args.output_file}")
        logger.info(f"   Skipping computation to avoid duplicate work")
        logger.info(f"   Experiment config: {args.method} + {args.dataset} + {args.model} + {args.calibration_method} + seed{args.seed}")
        if args.forced_layer:
            logger.info(f"   Forced layer: {args.forced_layer}")
        logger.info(f"   Use compression: {args.use_compression}")
        logger.info(f"   Exiting peacefully...")
        sys.exit(0)
    
    # 🔧 CRITICAL: Setup CUDA memory management first thing
    setup_cuda_memory_management()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 Using device: {device}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load model
    model_path = construct_model_path(
        args.results_dir, args.method, args.dataset, args.model, args.seed, 
        target_domain=args.target_domain
    )
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    # Get data loaders
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset, args.batch_size, args.seed, args.target_domain
    )
    
    # Load model and setup feature extraction
    model = load_trained_model(model_path, args.model, num_classes, device, dataset=args.dataset)
    
    # Enhanced feature extractor setup
    if args.forced_layer is not None or 'dinov2' in args.model:
        # Use EnhancedLayerFeatureExtractor for layer optimization or DINOv2 models
        logger.info(f"🔬 Using EnhancedLayerFeatureExtractor")
        if args.forced_layer:
            logger.info(f"   Forced layer: {args.forced_layer}")
        if 'dinov2' in args.model:
            logger.info(f"   DINOv2 model detected: {args.model}")
        
        from utils.enhanced_layer_feature_extractor import EnhancedLayerFeatureExtractor
        feature_extractor = EnhancedLayerFeatureExtractor(
            model=model,
            model_name=args.model,
            forced_layer=args.forced_layer,
            use_compression=args.use_compression.lower() == 'true'
        )
    else:
        logger.info(f"🔗 Using standard FeatureExtractor")
        feature_extractor = FeatureExtractor(model, args.model)
    
    # Convert compression string to boolean
    use_compression = args.use_compression.lower() == 'true'
    
    try:
        # Run single calibration method
        result = run_single_calibration_method(
            model, feature_extractor, train_loader, val_loader, test_loader,
            device, num_classes, args.method, args.dataset, args.calibration_method, args.batch_size,
            enable_corruption_eval=args.enable_corruption_eval,
            cifar10c_dir=args.cifar10c_dir,
            cifar100c_dir=args.cifar100c_dir,
            use_compression=use_compression,
            forced_layer=args.forced_layer
        )
        
        # Save result
        output_data = {
            'experiment_config': vars(args),
            'forced_layer': args.forced_layer,
            'results': {
                get_calibration_method_config(args.calibration_method)['name']: result
            },
            'timestamp': time.time()
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"✅ Results saved to: {args.output_file}")
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        import traceback
        logger.error(f"📋 Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)
        
    finally:
        # Final cleanup
        if hasattr(feature_extractor, 'cleanup'):
            try:
                feature_extractor.cleanup()
            except Exception as e:
                logger.warning(f"⚠️ Final cleanup failed: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()