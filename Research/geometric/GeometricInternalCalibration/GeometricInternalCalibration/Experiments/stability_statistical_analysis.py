#!/usr/bin/env python3
"""
Stability Score Statistical Analysis: Semantic Feature Quality Assessment
Enhanced version with robust error handling and improved integration

This script provides the missing piece for Section 4.2:
1. Calculates stability scores for clean vs corrupted images
2. Compares stability score distributions across layers  
3. Shows why layer3/trans3 provide better calibration-relevant geometric properties
4. Explains the "feature collapse" phenomenon in deeper layers

ENHANCED FEATURES:
- StabilitySpace Integration: Uses the same calculation as geometric calibration for consistency
- Enhanced Correlation Analysis: Computes BOTH Spearman (monotonic) and Pearson (linear) correlations
- Comprehensive Result Validation: Ensures complete and valid results before skipping
- 5-Digit Precision: All numerical outputs with consistent precision
- Robust NaN Handling: Graceful handling of feature collapse scenarios
- Multi-dimensional Analysis: Class and severity-specific insights for comprehensive mode

STABILITY CALCULATION METHODS:
- Primary: StabilitySpace class (consistency with calibration)
- Fallback: Manual calculation (when StabilitySpace unavailable)
- Libraries: fast_separation (fastest), faiss (robust), kdtree, separation
- Metrics: l2 (default), l1, cosine, linf
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import sys
import os
from collections import defaultdict
import pandas as pd
from scipy import stats
import argparse
import warnings
import glob
warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'Experiments' else SCRIPT_DIR

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'Experiments'))
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))

# Try to import required modules with fallbacks
try:
    from enhanced_layer_feature_extractor import EnhancedLayerFeatureExtractor
except ImportError as e:
    print(f"⚠️ Warning: Could not import enhanced_layer_feature_extractor: {e}")
    print("   Falling back to basic feature extraction")
    EnhancedLayerFeatureExtractor = None

try:
    from run_post_hoc_calibration import load_trained_model
except ImportError as e:
    print(f"⚠️ Warning: Could not import load_trained_model: {e}")
    print("   Will use basic model loading")
    load_trained_model = None

# Try to import StabilitySpace with fallback
try:
    from utils.stability_space import StabilitySpace
    STABILITY_SPACE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import StabilitySpace: {e}")
    print("   Will use basic stability calculation")
    STABILITY_SPACE_AVAILABLE = False
    StabilitySpace = None

# CIFAR-10 classes
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# CIFAR-100 classes (fine labels)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

def get_class_names(dataset):
    """Get class names for the specified dataset."""
    if dataset == 'cifar10':
        return CIFAR10_CLASSES
    elif dataset == 'cifar100':
        return CIFAR100_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def get_num_classes(dataset):
    """Get number of classes for the specified dataset."""
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def safe_correlation(x, y, correlation_type='spearman'):
    """
    Safely calculate correlation with NaN handling.
    
    Args:
        x: First variable
        y: Second variable  
        correlation_type: 'spearman' or 'pearson'
    
    Returns:
        correlation, p_value (both 0.0 if calculation fails or results in NaN)
    """
    try:
        # Convert to numpy arrays and check for valid data
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Check for sufficient data and variance
        if len(x) < 3 or len(y) < 3:
            return 0.0, 1.0
            
        if len(set(x)) < 2 or len(set(y)) < 2:
            # No variance in data
            return 0.0, 1.0
            
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid_mask) < 3:
            return 0.0, 1.0
            
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        # Double-check variance after cleaning
        if len(set(x_clean)) < 2 or len(set(y_clean)) < 2:
            return 0.0, 1.0
        
        # Calculate correlation
        if correlation_type == 'spearman':
            corr, p_val = stats.spearmanr(x_clean, y_clean)
        elif correlation_type == 'pearson':
            corr, p_val = stats.pearsonr(x_clean, y_clean)
        else:
            raise ValueError(f"Unknown correlation type: {correlation_type}")
        
        # Check for NaN results
        if np.isnan(corr) or np.isnan(p_val):
            return 0.0, 1.0
            
        return float(corr), float(p_val)
        
    except (ValueError, TypeError, ZeroDivisionError, Exception) as e:
        # Any error - return safe defaults
        return 0.0, 1.0

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Stability Score Statistical Analysis for AAAI Paper Section 4.2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script complements the k-NN analysis by measuring actual geometric calibration quality.

IMPORTANT: StabilitySpace Integration
- Uses the same StabilitySpace class as geometric calibration for consistency
- GPU vs CPU may produce slightly different results due to precision (float32 vs float64)
- The algorithm is mathematically identical, differences are numerical only
- Use --force-manual-calculation for exact manual computation
- Use --stability-library and --stability-metric to match calibration settings

Examples:
  # Standard analysis with StabilitySpace (recommended)
  python stability_statistical_analysis.py --model resnet18 --layer layer3 --seed 12
  
  # Force manual calculation for exact reproducibility
  python stability_statistical_analysis.py --model resnet18 --layer layer3 --force-manual-calculation
  
  # Match specific calibration settings
  python stability_statistical_analysis.py --model resnet18 --layer layer3 --stability-library faiss --stability-metric cosine
  
  # Comprehensive multi-dimensional analysis across classes and severities
  python stability_statistical_analysis.py --model resnet18 --layer layer3 --comprehensive-analysis --samples-per-class-severity 20
  
  # CIFAR-100 analysis with comprehensive dimensional insights
  python stability_statistical_analysis.py --model densenet121 --layer trans3 --dataset cifar100 --comprehensive-analysis
  
  # Multiple corruptions with dimensional insights
  python stability_statistical_analysis.py --model resnet50 --layer layer4 --all-corruptions --comprehensive-analysis

Available models: resnet18, resnet50, densenet121
Available datasets: cifar10 (10 classes), cifar100 (100 classes)
Available layers:
  - ResNet: layer1, layer2, layer3, layer4, layer4.1
  - DenseNet: dense1, trans1, dense2, trans2, dense3, trans3, dense4, bn
  - Physical Space: pixel (for raw pixel analysis)
Available corruptions: defocus_blur, gaussian_noise, contrast, snow, etc.
        """
    )
    
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'densenet121'],
                       help='Model to analyze (default: resnet18)')
    
    parser.add_argument('--layer', type=str, default='layer3',
                       help='Layer to analyze (default: layer3)')
    
    parser.add_argument('--seed', type=int, default=12,
                       help='Random seed (default: 12)')
    
    parser.add_argument('--training-method', type=str, default='constellation',
                       choices=['constellation', 'contrastive', 'baseline_cross_entropy', 
                               'baseline_focal', 'baseline_focal_adaptive', 'baseline_brier', 
                               'baseline_mmce', 'baseline_mmce_weighted', 'augmix', 
                               'augmix_constellation', 'ce_fast_separation', 
                               'constellation_original', 'geometric_focal_calibration'],
                       help='Training method (default: constellation)')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset (default: cifar10)')
    
    parser.add_argument('--corruption', type=str, default='defocus_blur',
                       help='Single corruption type to analyze (default: defocus_blur)')
    
    parser.add_argument('--corruptions', type=str, nargs='+', 
                       default=['defocus_blur', 'gaussian_noise', 'contrast', 'snow', 'motion_blur'],
                       help='Multiple corruption types to analyze (default: defocus_blur gaussian_noise contrast snow motion_blur)')
    
    parser.add_argument('--all-corruptions', action='store_true',
                       help='Process all available corruption types in CIFAR-10-C')
    
    parser.add_argument('--num-images', type=int, default=500,
                       help='Number of images to analyze (default: 500)')
    
    parser.add_argument('--max-analyze', type=int, default=1000,
                       help='Maximum examples to analyze (default: 1000)')
    
    parser.add_argument('--output-dir', type=str, default='section4_stability_analysis',
                       help='Base output directory (default: section4_stability_analysis)')
    
    parser.add_argument('--results-dir', type=str, default='aaai_full_experiments/results',
                       help='Base results directory for finding models (default: aaai_full_experiments/results)')
    
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for feature extraction (default: 100)')
    
    parser.add_argument('--cifar10c-dir', type=str, default='data/cifar10-c',
                       help='CIFAR-10-C data directory (default: data/cifar10-c)')
    
    parser.add_argument('--cifar100c-dir', type=str, default='data/cifar100-c',
                       help='CIFAR-100-C data directory (default: data/cifar100-c)')
    
    parser.add_argument('--comprehensive-analysis', action='store_true',
                       help='Enable comprehensive multi-dimensional analysis across classes and severities')
    
    parser.add_argument('--samples-per-class-severity', type=int, default=30,
                       help='Samples per class-severity combination for comprehensive analysis (default: 30)')
    
    parser.add_argument('--use-stability-space', action='store_true', default=True,
                       help='Use StabilitySpace class for consistency with calibration (default: True)')
    
    parser.add_argument('--stability-library', type=str, default='fast_separation',
                       choices=['fast_separation', 'faiss', 'kdtree', 'separation'],
                       help='StabilitySpace library to use (default: fast_separation)')
    
    parser.add_argument('--stability-metric', type=str, default='l2',
                       choices=['l1', 'l2', 'linf', 'cosine'],
                       help='Distance metric for stability calculation (default: l2)')
    
    parser.add_argument('--force-manual-calculation', action='store_true',
                       help='Force manual stability calculation instead of StabilitySpace')
    
    return parser.parse_args()

def construct_output_dir(args, corruption_type=None):
    """Construct configuration-specific output directory."""
    if corruption_type:
        output_dir = Path(args.output_dir) / "results" / args.training_method / args.dataset / args.model / f"{args.layer}_stability" / f"seed{args.seed}" / corruption_type
    else:
        # Base directory for multi-corruption runs
        output_dir = Path(args.output_dir) / "results" / args.training_method / args.dataset / args.model / f"{args.layer}_stability" / f"seed{args.seed}"
    return output_dir

def get_training_method_category(training_method):
    """Determine the category (baseline/geometric) for a training method."""
    baseline_methods = [
        'augmix', 'augmix_constellation', 'baseline_brier', 'baseline_cross_entropy', 
        'baseline_focal', 'baseline_focal_adaptive', 'baseline_mmce', 'baseline_mmce_weighted'
    ]
    
    geometric_methods = [
        'ce_fast_separation', 'constellation', 'constellation_original', 'geometric_focal_calibration'
    ]
    
    if training_method in baseline_methods:
        return 'baseline'
    elif training_method in geometric_methods:
        return 'geometric'
    else:
        print(f"⚠️ Unknown training method category for '{training_method}', defaulting to 'geometric'")
        return 'geometric'

def check_model_exists(training_method, dataset, model, seed, results_dir):
    """Check if trained model exists using glob pattern matching."""
    # Determine the correct category and build the full path
    category = get_training_method_category(training_method)
    full_results_dir = Path(results_dir) / category
    
    print(f"🔍 Looking for {training_method} in category: {category}")
    print(f"   Base directory: {full_results_dir}")
    print(f"   Dataset: {dataset}")
    
    # Try multiple possible patterns - from most specific to most general
    patterns = [
        # Original expected structure
        str(full_results_dir / training_method / dataset / model / f"seed{seed}" / f"{training_method}_{dataset}_{model}_seed{seed}" / "best_model.pth"),
        str(full_results_dir / training_method / dataset / model / f"seed{seed}" / "best_model.pth"),
        str(full_results_dir / training_method / dataset / model / f"seed{seed}" / "model.pth"),
        
        # Simpler structure: category/method/model_seed/
        str(full_results_dir / training_method / f"{model}_seed{seed}" / "best_model.pth"),
        str(full_results_dir / training_method / f"{model}_seed{seed}" / "model.pth"),
        str(full_results_dir / training_method / f"{model}_seed{seed}" / f"{training_method}_{dataset}_{model}_seed{seed}" / "best_model.pth"),
        
        # Even simpler: category/method/
        str(full_results_dir / training_method / "best_model.pth"),
        str(full_results_dir / training_method / "model.pth"),
        str(full_results_dir / training_method / f"{training_method}_{dataset}_{model}_seed{seed}.pth"),
        str(full_results_dir / training_method / f"{model}_seed{seed}.pth"),
        
        # Alternative naming patterns
        str(full_results_dir / training_method / f"{training_method}_{model}_seed{seed}" / "best_model.pth"),
        str(full_results_dir / training_method / f"{dataset}_{model}_seed{seed}" / "best_model.pth"),
        
        # Use glob patterns for flexible matching
        str(full_results_dir / training_method / "*" / f"*seed{seed}*" / "best_model.pth"),
        str(full_results_dir / training_method / "*" / f"*seed{seed}*" / "model.pth"),
        str(full_results_dir / training_method / f"*seed{seed}*" / "best_model.pth"),
        str(full_results_dir / training_method / f"*seed{seed}*" / "model.pth"),
    ]
    
    for pattern in patterns:
        matching_files = glob.glob(pattern)
        if matching_files:
            model_path = Path(matching_files[0])
            print(f"✅ Found model: {model_path}")
            return model_path
    
    print(f"❌ Model not found for: {training_method} | {dataset} | {model} | seed{seed}")
    print(f"   Searched patterns:")
    for pattern in patterns:
        print(f"     {pattern}")
    return None

def get_available_corruptions(dataset='cifar10', corruption_dir=None):
    """Get all available corruption types from CIFAR-10-C or CIFAR-100-C."""
    
    if corruption_dir:
        corruption_paths = [Path(corruption_dir)]
    else:
        corruption_paths = [
            PROJECT_ROOT / 'data' / f'{dataset}-c',
            PROJECT_ROOT / 'Data' / f'{dataset}-c',
            Path(f'./data/{dataset}-c'),
            Path(f'/datasets/{dataset}-c')
        ]
    
    for corruption_path in corruption_paths:
        if corruption_path.exists():
            corruption_files = list(corruption_path.glob('*.npy'))
            if corruption_files:
                corruptions = [f.stem for f in corruption_files]
                # Remove labels.npy if present
                if 'labels' in corruptions:
                    corruptions.remove('labels')
                return sorted(corruptions)
    
    # Fallback list if data directory not found
    return [
        'defocus_blur', 'gaussian_noise', 'contrast', 'snow', 'motion_blur',
        'frost', 'brightness', 'fog', 'zoom_blur', 'elastic_transform',
        'jpeg_compression', 'pixelate', 'gaussian_blur', 'glass_blur', 'impulse_noise',
        'shot_noise', 'speckle_noise', 'spatter', 'saturate'
    ]

def load_corruption_data(corruption_type, dataset='cifar10', corruption_dir=None, severity=5, return_all_severities=False):
    """Load specific corruption data from CIFAR-10-C or CIFAR-100-C."""
    
    # Determine corruption directory
    if corruption_dir:
        base_corruption_dir = Path(corruption_dir)
    else:
        if dataset == 'cifar10':
            base_corruption_dir = PROJECT_ROOT / 'data' / 'cifar10-c'
        elif dataset == 'cifar100':
            base_corruption_dir = PROJECT_ROOT / 'data' / 'cifar100-c'
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    corruption_path = base_corruption_dir / f'{corruption_type}.npy'
    
    if not corruption_path.exists():
        # Try alternative locations
        alt_paths = [
            PROJECT_ROOT / 'Data' / f'{dataset}-c' / f'{corruption_type}.npy',
            Path(f'/datasets/{dataset}-c') / f'{corruption_type}.npy',
            Path(f'./data/{dataset}-c') / f'{corruption_type}.npy'
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                corruption_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Corruption file not found: {corruption_type}.npy for {dataset} in any of {[corruption_path] + alt_paths}")
    
    corruption_data = np.load(corruption_path)
    
    if return_all_severities:
        # Return all severities as a dictionary
        all_severities = {}
        for sev in range(1, 6):
            start_idx = (sev - 1) * 10000
            end_idx = sev * 10000
            all_severities[sev] = corruption_data[start_idx:end_idx]
        
        print(f"✅ Loaded all severities for {corruption_type} on {dataset.upper()}")
        return all_severities
    else:
        # Select specific severity level (1-5, where 5 is most severe)
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        corrupted_images = corruption_data[start_idx:end_idx]
        
        print(f"✅ Loaded {len(corrupted_images)} images for {corruption_type} on {dataset.upper()} (severity {severity})")
        return corrupted_images

def load_basic_model(model_path, model_name, num_classes=10):
    """Basic model loading fallback when load_trained_model is not available."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🔧 Loading {model_name} with {num_classes} classes")
    
    if model_name == 'resnet18':
        from Net.resnet import resnet18
        model = resnet18(num_classes=num_classes)
    elif model_name == 'resnet50':
        from Net.resnet import resnet50
        model = resnet50(num_classes=num_classes)
    elif model_name == 'densenet121':
        from Net.densenet import densenet121
        model = densenet121(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def load_single_model_data(model_name, layer_name, training_method='constellation', seed=12, results_dir='aaai_full_experiments/results', corruption_type='defocus_blur', dataset='cifar10', corruption_dir=None):
    """Load CIFAR-10/CIFAR-100 data and a single specified model."""
    
    # Determine number of classes
    num_classes = 10 if dataset == 'cifar10' else 100
    
    print(f"📁 Loading {dataset.upper()} test data...")
    
    # Load clean test data based on dataset
    if dataset == 'cifar10':
        # Load CIFAR-10 test data
        test_batch_path = PROJECT_ROOT / 'data' / 'cifar-10-batches-py' / 'test_batch'
        
        # Try alternative locations if not found
        if not test_batch_path.exists():
            alt_paths = [
                PROJECT_ROOT / 'Data' / 'cifar-10-batches-py' / 'test_batch',
                Path('./data/cifar-10-batches-py/test_batch'),
                Path('/datasets/cifar-10-batches-py/test_batch')
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    test_batch_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"CIFAR-10 test data not found in any of {[test_batch_path] + alt_paths}")
        
        with open(test_batch_path, 'rb') as fo:
            test_batch = pickle.load(fo, encoding='bytes')
        
        images = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(test_batch[b'labels'])
        
    elif dataset == 'cifar100':
        # Load CIFAR-100 test data
        test_batch_path = PROJECT_ROOT / 'data' / 'cifar-100-python' / 'test'
        
        # Try alternative locations if not found
        if not test_batch_path.exists():
            alt_paths = [
                PROJECT_ROOT / 'Data' / 'cifar-100-python' / 'test',
                Path('./data/cifar-100-python/test'),
                Path('/datasets/cifar-100-python/test')
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    test_batch_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"CIFAR-100 test data not found in any of {[test_batch_path] + alt_paths}")
        
        with open(test_batch_path, 'rb') as fo:
            test_batch = pickle.load(fo, encoding='bytes')
        
        images = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(test_batch[b'fine_labels'])  # Use fine labels for CIFAR-100
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f"✅ Loaded {len(images)} clean {dataset.upper()} test images")
    
    # Load corruption data
    print(f"📁 Loading {corruption_type} corruption data for {dataset.upper()}...")
    corrupted_images = load_corruption_data(corruption_type, dataset=dataset, corruption_dir=corruption_dir, severity=5)
    
    # Special case: pixel space analysis (no model needed)
    if layer_name.lower() in ['pixel', 'pixels', 'raw', 'physical']:
        print(f"🖼️ Pixel space analysis - no model loading required")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return images, labels, corrupted_images, None, device
    
    # Load specified model for semantic analysis
    print(f"🤖 Loading {model_name} model for {dataset.upper()}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = check_model_exists(training_method, dataset, model_name, seed, results_dir)
    if model_path is None:
        return None, None, None, None, None
    
    try:
        if load_trained_model:
            model = load_trained_model(str(model_path), model_name, num_classes, device)
        else:
            model = load_basic_model(model_path, model_name, num_classes)
        
        model.eval()
        print(f"✅ Loaded {model_name} for {dataset.upper()}")
        print(f"   Model path: {model_path}")
        print(f"   Number of classes: {num_classes}")
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None, None, None, None, None
    
    return images, labels, corrupted_images, model, device

class BasicFeatureExtractor:
    """Basic feature extractor fallback when EnhancedLayerFeatureExtractor is not available."""
    
    def __init__(self, model, model_name, forced_layer):
        self.model = model
        self.model_name = model_name
        self.forced_layer = forced_layer
        self.layer_features = {}
        self.hooks = []
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Setup hooks to capture intermediate layer outputs."""
        def make_hook(name):
            def hook(module, input, output):
                # Store the output, handling different shapes
                if isinstance(output, torch.Tensor):
                    self.layer_features[name] = output.detach()
                elif isinstance(output, (list, tuple)):
                    self.layer_features[name] = output[0].detach()
            return hook
        
        # Register hooks for the requested layer
        for name, module in self.model.named_modules():
            if name == self.forced_layer or name.endswith(self.forced_layer):
                handle = module.register_forward_hook(make_hook(self.forced_layer))
                self.hooks.append(handle)
                print(f"✅ Registered hook for layer: {name}")
                break
        
        if not self.hooks:
            print(f"⚠️ Warning: Could not find layer {self.forced_layer}")
            # Register hook for the last conv layer as fallback
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    last_conv_name = name
            
            if 'last_conv_name' in locals():
                handle = module.register_forward_hook(make_hook(self.forced_layer))
                self.hooks.append(handle)
                print(f"✅ Fallback: Using layer {last_conv_name}")
    
    def cleanup(self):
        """Remove hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

def extract_features_batch_robust(images, model, model_name, layer_name, device, batch_size=100):
    """Extract features in batches with robust tensor handling."""
    
    # Handle special case for pixel space analysis
    if layer_name.lower() in ['pixel', 'pixels', 'raw']:
        print(f"🖼️ Using raw pixel features (flattened) for physical space analysis...")
        # Normalize images and flatten
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        
        normalized_images = (images.astype(np.float32) / 255.0 - mean) / std
        flattened_features = normalized_images.reshape(len(images), -1)
        
        print(f"✅ Pixel features: {flattened_features.shape}")
        return flattened_features
    
    print(f"🧠 Extracting features from {model_name}-{layer_name} (stability analysis)...")
    
    # Create extractor with fallback
    if EnhancedLayerFeatureExtractor:
        try:
            extractor = EnhancedLayerFeatureExtractor(
                model=model,
                model_name=model_name,
                forced_layer=layer_name,
                use_compression=True
            )
        except Exception as e:
            print(f"⚠️ EnhancedLayerFeatureExtractor failed: {e}")
            print("   Falling back to BasicFeatureExtractor")
            extractor = BasicFeatureExtractor(model, model_name, layer_name)
    else:
        extractor = BasicFeatureExtractor(model, model_name, layer_name)
    
    features = []
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc=f"Extracting {model_name}-{layer_name}"):
            batch = images[i:i+batch_size]
            
            # Normalize batch
            batch_norm = (batch.astype(np.float32) / 255.0 - mean) / std
            batch_tensor = torch.from_numpy(batch_norm).permute(0, 3, 1, 2).float().to(device)
            batch_tensor = batch_tensor.contiguous()
            
            # Extract features
            _ = model(batch_tensor)
            batch_features = []
            
            for j in range(len(batch_tensor)):
                if layer_name in extractor.layer_features and extractor.layer_features[layer_name] is not None:
                    feat = extractor.layer_features[layer_name][j].cpu().numpy()
                    # Flatten features for distance calculation
                    feat = feat.flatten()
                    batch_features.append(feat)
                else:
                    available_layers = list(extractor.layer_features.keys())
                    print(f"⚠️  No features found for {layer_name}. Available: {available_layers}")
                    # Use a default size that should work for most layers
                    batch_features.append(np.zeros(4096))
            
            features.extend(batch_features)
            extractor.layer_features.clear()
    
    extractor.cleanup()
    return np.array(features)

def comprehensive_stability_sampling(corrupted_images_by_severity, labels, samples_per_class_severity=100, dataset='cifar10'):
    """
    Create comprehensive sampling plan across all classes and severities.
    Returns selected samples with metadata for dimensional analysis.
    """
    num_classes = get_num_classes(dataset)
    class_names = get_class_names(dataset)
    
    print(f"🎯 Creating comprehensive sampling across classes and severities for {dataset.upper()}...")
    print(f"   Dataset: {dataset.upper()} ({num_classes} classes)")
    print(f"   Samples per class-severity combination: {samples_per_class_severity}")
    
    selected_samples = []
    sampling_metadata = []
    
    # Sample across all classes and severities
    for class_id in range(num_classes):
        class_mask = (labels == class_id)
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) == 0:
            continue
            
        for severity in range(1, 6):  # Severities 1-5
            if severity not in corrupted_images_by_severity:
                continue
                
            # Sample images from this class at this severity
            if len(class_indices) >= samples_per_class_severity:
                selected_indices = np.random.choice(class_indices, samples_per_class_severity, replace=False)
            else:
                selected_indices = class_indices
            
            for idx in selected_indices:
                selected_samples.append({
                    'global_index': int(idx),
                    'corrupted_image': corrupted_images_by_severity[severity][idx],
                    'clean_label': int(labels[idx])
                })
                
                sampling_metadata.append({
                    'global_index': int(idx),
                    'class_id': int(class_id),
                    'class_name': class_names[class_id],
                    'severity_level': int(severity),
                    'combo_id': f'class{class_id}_sev{severity}'
                })
    
    print(f"✅ Selected {len(selected_samples)} samples across {len(set(m['combo_id'] for m in sampling_metadata))} class-severity combinations")
    return selected_samples, sampling_metadata

def calculate_stability_scores_with_stability_space(clean_features, corrupted_features, labels, predictions, max_analyze=1000, metadata=None, metric='l2', library='fast_separation'):
    """
    Calculate stability scores using the actual StabilitySpace class for consistency with calibration.
    
    Args:
        clean_features: Clean training features for stability space
        corrupted_features: Corrupted features to analyze
        labels: True labels for clean features
        predictions: Predicted labels for corrupted features
        max_analyze: Maximum number of samples to analyze
        metadata: Optional metadata for comprehensive analysis
        metric: Distance metric ('l2', 'l1', 'cosine')
        library: Library to use ('fast_separation', 'faiss', 'kdtree')
    
    Returns:
        List of stability results with same format as manual calculation
    """
    print(f"🚀 Calculating stability scores using StabilitySpace (library: {library}, metric: {metric})...")
    
    # If comprehensive metadata is provided, use all samples (already pre-selected)
    if metadata is not None and len(metadata) == len(corrupted_features):
        indices_to_analyze = list(range(len(corrupted_features)))
        print(f"📊 Using comprehensive sampling with metadata: {len(indices_to_analyze)} samples")
    else:
        # Limit analysis for efficiency (legacy behavior)
        num_analyze = min(len(corrupted_features), max_analyze)
        indices_to_analyze = np.random.choice(len(corrupted_features), num_analyze, replace=False)
        print(f"📊 Using random sampling: {len(indices_to_analyze)} samples")
    
    # Extract samples to analyze
    selected_corrupted_features = corrupted_features[indices_to_analyze]
    selected_predictions = predictions[indices_to_analyze]
    
    # Initialize StabilitySpace with clean training data
    print(f"🔧 Initializing StabilitySpace with {len(clean_features)} training samples...")
    try:
        stability_space = StabilitySpace(
            X_train=clean_features,
            y_train=labels,
            compression=None,  # No compression for analysis
            library=library,
            metric=metric,
            num_labels=len(np.unique(labels)),
            use_cuda=torch.cuda.is_available()
        )
        print(f"✅ StabilitySpace initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize StabilitySpace: {e}")
        print(f"   Falling back to manual calculation...")
        # Fallback to enhanced method
        if metadata is not None:
            return calculate_stability_scores_enhanced(clean_features, corrupted_features, labels, predictions, max_analyze, metadata)
        else:
            return calculate_stability_scores_basic(clean_features, corrupted_features, labels, predictions, max_analyze, metadata)
    
    # Calculate stability scores using StabilitySpace
    print(f"🔬 Computing stability scores for {len(selected_corrupted_features)} samples...")
    try:
        # StabilitySpace expects predictions as class indices (not one-hot)
        if len(selected_predictions.shape) > 1 and selected_predictions.shape[1] > 1:
            selected_predictions = np.argmax(selected_predictions, axis=1)
        
        # Call the actual StabilitySpace method
        stability_scores = stability_space.calc_stab(
            X_val=selected_corrupted_features,
            y_val_pred=selected_predictions,
            timeout=None  # No timeout for analysis
        )
        
        print(f"✅ StabilitySpace computation complete")
        
    except Exception as e:
        print(f"❌ StabilitySpace computation failed: {e}")
        print(f"   Falling back to manual calculation...")
        # Fallback to enhanced method
        if metadata is not None:
            return calculate_stability_scores_enhanced(clean_features, corrupted_features, labels, predictions, max_analyze, metadata)
        else:
            return calculate_stability_scores_basic(clean_features, corrupted_features, labels, predictions, max_analyze, metadata)
    
    # Format results to match the expected structure
    stability_results = []
    
    for idx, i in enumerate(indices_to_analyze):
        predicted_class = int(selected_predictions[idx])
        true_class = int(labels[i]) if i < len(labels) else int(predictions[i])  # Handle index bounds
        stability_score = float(stability_scores[idx])
        
        # Calculate additional metrics that the original code provided
        # Note: StabilitySpace doesn't directly provide min_friend_dist and min_unfriend_dist
        # We can estimate them from the stability score, but they won't be exact
        # For consistency with existing analysis, we'll compute basic metrics
        
        is_correct_prediction = (predicted_class == true_class)
        
        result = {
            'index': int(i),
            'stability_score': stability_score,
            'min_friend_dist': 0.0,  # StabilitySpace doesn't expose these directly
            'min_unfriend_dist': max(0.0, stability_score * 2.0),  # Rough estimation
            'predicted_class': predicted_class,
            'true_class': true_class,
            'is_correct': is_correct_prediction,
            'separation_ratio': 1.0 if stability_score > 0 else 0.0  # Simplified ratio
        }
        
        # Add metadata if available
        if metadata is not None and idx < len(metadata):
            result.update({
                'class_id': metadata[idx]['class_id'],
                'class_name': metadata[idx]['class_name'],
                'severity_level': metadata[idx]['severity_level'],
                'combo_id': metadata[idx]['combo_id']
            })
        
        stability_results.append(result)
    
    # Log comparison stats with manual calculation
    manual_scores = []
    try:
        # Quick sample check with manual calculation (first 10 samples)
        sample_size = min(10, len(selected_corrupted_features))
        for i in range(sample_size):
            query_feature = selected_corrupted_features[i]
            predicted_class = selected_predictions[i]
            
            # Manual calculation for comparison
            same_class_mask = (labels == predicted_class)
            same_class_features = clean_features[same_class_mask]
            
            if len(same_class_features) > 0:
                friend_distances = np.linalg.norm(same_class_features - query_feature, axis=1)
                min_friend_dist = np.min(friend_distances)
            else:
                min_friend_dist = float('inf')
            
            diff_class_mask = (labels != predicted_class)
            diff_class_features = clean_features[diff_class_mask]
            
            if len(diff_class_features) > 0:
                unfriend_distances = np.linalg.norm(diff_class_features - query_feature, axis=1)
                min_unfriend_dist = np.min(unfriend_distances)
            else:
                min_unfriend_dist = 0
            
            if min_friend_dist != float('inf'):
                manual_score = (min_unfriend_dist - min_friend_dist) / 2.0
            else:
                manual_score = 0
            
            manual_scores.append(manual_score)
        
        # Compare StabilitySpace vs Manual calculation
        if manual_scores:
            stability_space_sample = [stability_scores[i] for i in range(len(manual_scores))]
            
            print(f"📊 Calculation comparison (first {len(manual_scores)} samples):")
            print(f"   StabilitySpace mean: {np.mean(stability_space_sample):.6f}")
            print(f"   Manual mean: {np.mean(manual_scores):.6f}")
            print(f"   Difference: {abs(np.mean(stability_space_sample) - np.mean(manual_scores)):.6f}")
            print(f"   Max difference: {max(abs(a - b) for a, b in zip(stability_space_sample, manual_scores)):.6f}")
            
            # Check for significant discrepancies
            max_diff = max(abs(a - b) for a, b in zip(stability_space_sample, manual_scores))
            if max_diff > 1e-3:
                print(f"⚠️ Warning: Significant difference detected between StabilitySpace and manual calculation!")
                print(f"   This might be due to GPU/CPU precision differences or different algorithms")
                print(f"   StabilitySpace library: {library}, metric: {metric}")
        
    except Exception as e:
        print(f"⚠️ Could not perform comparison calculation: {e}")
    
    print(f"✅ Formatted {len(stability_results)} stability results")
    return stability_results

def calculate_stability_scores_enhanced(clean_features, corrupted_features, labels, predictions, max_analyze=1000, metadata=None):
    """
    Enhanced stability score calculation with metadata support for dimensional analysis.
    """
    print(f"🔬 Calculating stability scores for {len(corrupted_features)} samples (enhanced method)...")
    
    # If comprehensive metadata is provided, use all samples (already pre-selected)
    if metadata is not None and len(metadata) == len(corrupted_features):
        indices_to_analyze = list(range(len(corrupted_features)))
        print(f"📊 Using comprehensive sampling with metadata: {len(indices_to_analyze)} samples")
    else:
        # Limit analysis for efficiency (legacy behavior)
        num_analyze = min(len(corrupted_features), max_analyze)
        indices_to_analyze = np.random.choice(len(corrupted_features), num_analyze, replace=False)
        print(f"📊 Using random sampling: {len(indices_to_analyze)} samples")
    
    stability_results = []
    
    for i in tqdm(indices_to_analyze, desc="Computing stability scores"):
        query_feature = corrupted_features[i]
        predicted_class = predictions[i]
        true_class = labels[i]
        
        try:
            # Find same class examples (friends)
            same_class_mask = (labels == predicted_class)
            same_class_features = clean_features[same_class_mask]
            
            if len(same_class_features) > 0:
                friend_distances = np.linalg.norm(same_class_features - query_feature, axis=1)
                min_friend_dist = np.min(friend_distances)
            else:
                min_friend_dist = float('inf')
            
            # Find different class examples (unfriends)
            diff_class_mask = (labels != predicted_class)
            diff_class_features = clean_features[diff_class_mask]
            
            if len(diff_class_features) > 0:
                unfriend_distances = np.linalg.norm(diff_class_features - query_feature, axis=1)
                min_unfriend_dist = np.min(unfriend_distances)
            else:
                min_unfriend_dist = 0
            
            # Calculate stability score
            if min_friend_dist != float('inf'):
                stability_score = (min_unfriend_dist - min_friend_dist) / 2.0
            else:
                stability_score = 0  # No same-class examples found
            
            # Calculate additional metrics
            is_correct_prediction = (predicted_class == true_class)
            
            result = {
                'index': int(i),
                'stability_score': float(stability_score),
                'min_friend_dist': float(min_friend_dist),
                'min_unfriend_dist': float(min_unfriend_dist),
                'predicted_class': int(predicted_class),
                'true_class': int(true_class),
                'is_correct': bool(is_correct_prediction),
                'separation_ratio': float(min_unfriend_dist / (min_friend_dist + 1e-8))
            }
            
            # Add metadata if available
            if metadata is not None and i < len(metadata):
                result.update({
                    'class_id': metadata[i]['class_id'],
                    'class_name': metadata[i]['class_name'],
                    'severity_level': metadata[i]['severity_level'],
                    'combo_id': metadata[i]['combo_id']
                })
            
            stability_results.append(result)
            
        except Exception as e:
            print(f"⚠️  Error calculating stability for sample {i}: {e}")
            continue
    
    return stability_results

def calculate_stability_scores_basic(clean_features, corrupted_features, labels, predictions, max_analyze=1000, metadata=None):
    """
    Basic stability score calculation when StabilitySpace is not available.
    Enhanced with metadata support for dimensional analysis.
    
    Stability Score = (distance_to_nearest_unfriend - distance_to_nearest_friend) / 2
    """
    print(f"🔬 Calculating stability scores for {len(corrupted_features)} samples (basic method)...")
    
    # If comprehensive metadata is provided, use all samples (already pre-selected)
    if metadata is not None and len(metadata) == len(corrupted_features):
        indices_to_analyze = list(range(len(corrupted_features)))
        print(f"📊 Using comprehensive sampling with metadata: {len(indices_to_analyze)} samples")
    else:
        # Limit analysis for efficiency (legacy behavior)
        num_analyze = min(len(corrupted_features), max_analyze)
        indices_to_analyze = np.random.choice(len(corrupted_features), num_analyze, replace=False)
        print(f"📊 Using random sampling: {len(indices_to_analyze)} samples")
    
    stability_results = []
    
    for i in tqdm(indices_to_analyze, desc="Computing stability scores"):
        query_feature = corrupted_features[i]
        predicted_class = predictions[i]
        true_class = labels[i]
        
        try:
            # Find same class examples (friends)
            same_class_mask = (labels == predicted_class)
            same_class_features = clean_features[same_class_mask]
            
            if len(same_class_features) > 0:
                friend_distances = np.linalg.norm(same_class_features - query_feature, axis=1)
                min_friend_dist = np.min(friend_distances)
            else:
                min_friend_dist = float('inf')
            
            # Find different class examples (unfriends)
            diff_class_mask = (labels != predicted_class)
            diff_class_features = clean_features[diff_class_mask]
            
            if len(diff_class_features) > 0:
                unfriend_distances = np.linalg.norm(diff_class_features - query_feature, axis=1)
                min_unfriend_dist = np.min(unfriend_distances)
            else:
                min_unfriend_dist = 0
            
            # Calculate stability score
            if min_friend_dist != float('inf'):
                stability_score = (min_unfriend_dist - min_friend_dist) / 2.0
            else:
                stability_score = 0  # No same-class examples found
            
            # Calculate additional metrics
            is_correct_prediction = (predicted_class == true_class)
            
            result = {
                'index': int(i),
                'stability_score': float(stability_score),
                'min_friend_dist': float(min_friend_dist),
                'min_unfriend_dist': float(min_unfriend_dist),
                'predicted_class': int(predicted_class),
                'true_class': int(true_class),
                'is_correct': bool(is_correct_prediction),
                'separation_ratio': float(min_unfriend_dist / (min_friend_dist + 1e-8))
            }
            
            # Add metadata if available
            if metadata is not None and i < len(metadata):
                result.update({
                    'class_id': metadata[i]['class_id'],
                    'class_name': metadata[i]['class_name'],
                    'severity_level': metadata[i]['severity_level'],
                    'combo_id': metadata[i]['combo_id']
                })
            
            stability_results.append(result)
            
        except Exception as e:
            print(f"⚠️  Error calculating stability for sample {i}: {e}")
            continue
    
    return stability_results

def calculate_stability_scores(clean_features, corrupted_features, labels, predictions, max_analyze=1000, metadata=None, use_stability_space=True, stability_library='fast_separation', stability_metric='l2'):
    """
    Calculate stability scores using the optimal available method.
    Priority: StabilitySpace (for consistency) > Enhanced > Basic
    """
    if STABILITY_SPACE_AVAILABLE and use_stability_space:
        print(f"🚀 Using StabilitySpace for exact consistency with calibration (library: {stability_library}, metric: {stability_metric})")
        try:
            # Try the specified library first, then fallback options
            libraries_to_try = [stability_library]
            if stability_library != 'fast_separation':
                libraries_to_try.append('fast_separation')
            if stability_library != 'faiss':
                libraries_to_try.append('faiss')
            
            for library in libraries_to_try:
                try:
                    return calculate_stability_scores_with_stability_space(
                        clean_features, corrupted_features, labels, predictions, 
                        max_analyze, metadata, metric=stability_metric, library=library
                    )
                except Exception as e:
                    print(f"⚠️ {library} failed: {e}")
                    continue
            
            # If all StabilitySpace libraries fail, fall back to manual
            print("⚠️ All StabilitySpace libraries failed, falling back to manual calculation")
            
        except Exception as e:
            print(f"⚠️ StabilitySpace initialization failed: {e}")
    elif not STABILITY_SPACE_AVAILABLE:
        print("📊 StabilitySpace not available, using manual calculation")
    else:
        print("📊 Manual calculation forced by user")
    
    # Fallback to manual calculation
    print("📊 Using manual stability calculation")
    if metadata is not None:
        return calculate_stability_scores_enhanced(clean_features, corrupted_features, labels, predictions, max_analyze, metadata)
    else:
        return calculate_stability_scores_basic(clean_features, corrupted_features, labels, predictions, max_analyze, metadata)

def analyze_stability_distribution(stability_results):
    """Analyze the distribution of stability scores."""
    print("📊 Analyzing stability score distribution...")
    
    scores = [r['stability_score'] for r in stability_results]
    correct_predictions = [r for r in stability_results if r['is_correct']]
    incorrect_predictions = [r for r in stability_results if not r['is_correct']]
    
    correct_scores = [r['stability_score'] for r in correct_predictions]
    incorrect_scores = [r['stability_score'] for r in incorrect_predictions]
    
    # Calculate percentile normalization (like in geometric calibrator)
    scores_array = np.array(scores)
    percentile_scores = []
    
    for score in scores:
        percentile = (np.sum(scores_array <= score) / len(scores_array)) * 100
        percentile_scores.append(percentile)
    
    # Calculate percentile-based statistics
    percentile_range = np.max(percentile_scores) - np.min(percentile_scores)
    percentile_iqr = np.percentile(percentile_scores, 75) - np.percentile(percentile_scores, 25)
    
    analysis = {
        'overall_stats': {
            'mean_stability': np.mean(scores),
            'std_stability': np.std(scores),
            'median_stability': np.median(scores),
            'min_stability': np.min(scores),
            'max_stability': np.max(scores),
            'stability_range': np.max(scores) - np.min(scores),
            'positive_stability_pct': np.sum(np.array(scores) > 0) / len(scores) * 100,
            'total_samples': len(scores),
            # NEW: Percentile-based metrics
            'percentile_range': float(percentile_range),
            'percentile_iqr': float(percentile_iqr),
            'variance_to_mean_ratio': np.std(scores) / (abs(np.mean(scores)) + 1e-8),
            'calibration_informativeness': float(percentile_iqr / 50.0)  # Normalized to 0-2 scale
        },
        'correct_prediction_stats': {
            'mean_stability': np.mean(correct_scores) if correct_scores else 0,
            'std_stability': np.std(correct_scores) if correct_scores else 0,
            'mean_percentile': np.mean([percentile_scores[i] for i, r in enumerate(stability_results) if r['is_correct']]) if correct_scores else 0,
            'count': len(correct_scores)
        },
        'incorrect_prediction_stats': {
            'mean_stability': np.mean(incorrect_scores) if incorrect_scores else 0,
            'std_stability': np.std(incorrect_scores) if incorrect_scores else 0,
            'mean_percentile': np.mean([percentile_scores[i] for i, r in enumerate(stability_results) if not r['is_correct']]) if incorrect_scores else 0,
            'count': len(incorrect_scores)
        },
        'percentile_scores': percentile_scores  # Store for visualization
    }
    
    # Calculate BOTH Spearman and Pearson correlations for comprehensive analysis
    if len(scores) > 1:
        correct_binary = [1 if r['is_correct'] else 0 for r in stability_results]
        
        # Spearman correlation (PRIMARY - rank-based, monotonic, ideal for isotonic regression)
        spearman_corr, spearman_p = safe_correlation(scores, correct_binary, 'spearman')
        analysis['stability_correctness_spearman'] = spearman_corr
        analysis['stability_correctness_spearman_p'] = spearman_p
        
        # Pearson correlation (SECONDARY - linear relationship assessment)
        pearson_corr, pearson_p = safe_correlation(scores, correct_binary, 'pearson')
        analysis['stability_correctness_pearson'] = pearson_corr
        analysis['stability_correctness_pearson_p'] = pearson_p
        
        # Backward compatibility: use Spearman as primary correlation
        analysis['stability_correctness_correlation'] = analysis['stability_correctness_spearman']
        analysis['correlation_p_value'] = analysis['stability_correctness_spearman_p']
        
        # Percentile correlations - both Spearman and Pearson
        percentile_spearman_corr, percentile_spearman_p = safe_correlation(percentile_scores, correct_binary, 'spearman')
        analysis['percentile_correctness_spearman'] = percentile_spearman_corr
        analysis['percentile_correctness_spearman_p'] = percentile_spearman_p
        
        percentile_pearson_corr, percentile_pearson_p = safe_correlation(percentile_scores, correct_binary, 'pearson')
        analysis['percentile_correctness_pearson'] = percentile_pearson_corr
        analysis['percentile_correctness_pearson_p'] = percentile_pearson_p
        
        # Backward compatibility: use Spearman as primary percentile correlation
        analysis['percentile_correctness_correlation'] = analysis['percentile_correctness_spearman']
        analysis['percentile_correlation_p_value'] = analysis['percentile_correctness_spearman_p']
    
    return analysis

def calculate_dimensional_analysis(stability_results, dataset='cifar10'):
    """
    Calculate multi-dimensional analysis across classes, severities, and combinations.
    This provides granular insights for identifying optimal layers per dimension.
    """
    print(f"🔍 Performing multi-dimensional stability analysis for {dataset.upper()}...")
    
    # Check if we have metadata for dimensional analysis
    has_metadata = len(stability_results) > 0 and 'class_id' in stability_results[0]
    
    if not has_metadata:
        print("⚠️ No metadata available for dimensional analysis. Returning overall analysis only.")
        return {
            'overall_analysis': analyze_stability_distribution(stability_results),
            'has_dimensional_data': False
        }
    
    # Separate analysis by different dimensions
    dimensional_analysis = {
        'overall_analysis': analyze_stability_distribution(stability_results),
        'by_class_analysis': calculate_class_wise_analysis(stability_results, dataset),
        'by_severity_analysis': calculate_severity_wise_analysis(stability_results),
        'by_combo_analysis': calculate_combo_wise_analysis(stability_results),
        'insights': generate_dimensional_insights(stability_results, dataset),
        'has_dimensional_data': True
    }
    
    return dimensional_analysis

def calculate_class_wise_analysis(stability_results, dataset='cifar10'):
    """Calculate stability analysis separately for each class in the dataset."""
    num_classes = get_num_classes(dataset)
    class_names = get_class_names(dataset)
    class_analysis = {}
    
    for class_id in range(num_classes):
        class_name = class_names[class_id]
        class_data = [r for r in stability_results if r.get('class_id') == class_id]
        
        if len(class_data) >= 5:  # Minimum samples for analysis
            # Calculate class-specific correlations
            scores = [r['stability_score'] for r in class_data]
            correct_binary = [1 if r['is_correct'] else 0 for r in class_data]
            
            # Spearman and Pearson correlations with NaN handling
            spearman_corr, spearman_p = safe_correlation(scores, correct_binary, 'spearman')
            pearson_corr, pearson_p = safe_correlation(scores, correct_binary, 'pearson')
            
            class_analysis[f'class_{class_id}'] = {
                'class_id': class_id,
                'class_name': class_name,
                'sample_count': len(class_data),
                'mean_stability': np.mean(scores),
                'std_stability': np.std(scores),
                'positive_stability_pct': np.sum(np.array(scores) > 0) / len(scores) * 100,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'accuracy': np.mean(correct_binary) * 100,
                'variance_to_mean_ratio': np.std(scores) / (abs(np.mean(scores)) + 1e-8)
            }
    
    return class_analysis

def calculate_severity_wise_analysis(stability_results):
    """Calculate stability analysis separately for each severity level."""
    severity_analysis = {}
    
    for severity in range(1, 6):
        severity_data = [r for r in stability_results if r.get('severity_level') == severity]
        
        if len(severity_data) >= 5:  # Minimum samples for analysis
            # Calculate severity-specific correlations
            scores = [r['stability_score'] for r in severity_data]
            correct_binary = [1 if r['is_correct'] else 0 for r in severity_data]
            
            # Spearman and Pearson correlations with NaN handling
            spearman_corr, spearman_p = safe_correlation(scores, correct_binary, 'spearman')
            pearson_corr, pearson_p = safe_correlation(scores, correct_binary, 'pearson')
            
            severity_analysis[f'severity_{severity}'] = {
                'severity_level': severity,
                'sample_count': len(severity_data),
                'mean_stability': np.mean(scores),
                'std_stability': np.std(scores),
                'positive_stability_pct': np.sum(np.array(scores) > 0) / len(scores) * 100,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'accuracy': np.mean(correct_binary) * 100,
                'variance_to_mean_ratio': np.std(scores) / (abs(np.mean(scores)) + 1e-8)
            }
    
    return severity_analysis

def calculate_combo_wise_analysis(stability_results):
    """Calculate stability analysis for each class-severity combination."""
    combo_analysis = {}
    
    # Group by combination ID
    combo_groups = defaultdict(list)
    for result in stability_results:
        combo_id = result.get('combo_id')
        if combo_id:
            combo_groups[combo_id].append(result)
    
    for combo_id, combo_data in combo_groups.items():
        if len(combo_data) >= 3:  # Minimum samples for analysis
            scores = [r['stability_score'] for r in combo_data]
            correct_binary = [1 if r['is_correct'] else 0 for r in combo_data]
            
            # Get class and severity from first sample
            class_id = combo_data[0]['class_id']
            class_name = combo_data[0]['class_name']
            severity_level = combo_data[0]['severity_level']
            
            # Spearman and Pearson correlations with NaN handling
            spearman_corr, spearman_p = safe_correlation(scores, correct_binary, 'spearman')
            pearson_corr, pearson_p = safe_correlation(scores, correct_binary, 'pearson')
            
            combo_analysis[combo_id] = {
                'combo_id': combo_id,
                'class_id': class_id,
                'class_name': class_name,
                'severity_level': severity_level,
                'sample_count': len(combo_data),
                'mean_stability': np.mean(scores),
                'std_stability': np.std(scores),
                'positive_stability_pct': np.sum(np.array(scores) > 0) / len(scores) * 100,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'accuracy': np.mean(correct_binary) * 100,
                'variance_to_mean_ratio': np.std(scores) / (abs(np.mean(scores)) + 1e-8)
            }
    
    return combo_analysis

def generate_dimensional_insights(stability_results, dataset='cifar10'):
    """Generate specific insights for classes, severities, and optimal layer identification."""
    
    num_classes = get_num_classes(dataset)
    
    # Get dimensional analyses
    class_analysis = calculate_class_wise_analysis(stability_results, dataset)
    severity_analysis = calculate_severity_wise_analysis(stability_results)
    
    insights = {
        'class_insights': {},
        'severity_insights': {},
        'recommendations': {},
        'summary_statistics': {}
    }
    
    # Class-specific insights
    for class_key, class_data in class_analysis.items():
        class_name = class_data['class_name']
        spearman_corr = class_data['spearman_correlation']
        mean_stability = class_data['mean_stability']
        variance_ratio = class_data['variance_to_mean_ratio']
        
        quality_score = (abs(spearman_corr) * 0.4 + 
                        variance_ratio * 0.3 + 
                        class_data['positive_stability_pct'] / 100 * 0.3)
        
        insights['class_insights'][class_name] = {
            'spearman_correlation': spearman_corr,
            'mean_stability': mean_stability,
            'variance_ratio': variance_ratio,
            'quality_score': quality_score,
            'interpretation': f"For {class_name} class: Spearman={spearman_corr:.3f}, Stability={mean_stability:.3f}, Quality={quality_score:.3f}"
        }
    
    # Severity-specific insights
    for severity_key, severity_data in severity_analysis.items():
        severity_level = severity_data['severity_level']
        spearman_corr = severity_data['spearman_correlation']
        mean_stability = severity_data['mean_stability']
        variance_ratio = severity_data['variance_to_mean_ratio']
        
        quality_score = (abs(spearman_corr) * 0.4 + 
                        variance_ratio * 0.3 + 
                        severity_data['positive_stability_pct'] / 100 * 0.3)
        
        insights['severity_insights'][f'severity_{severity_level}'] = {
            'severity_level': severity_level,
            'spearman_correlation': spearman_corr,
            'mean_stability': mean_stability,
            'variance_ratio': variance_ratio,
            'quality_score': quality_score,
            'interpretation': f"For severity {severity_level}: Spearman={spearman_corr:.3f}, Stability={mean_stability:.3f}, Quality={quality_score:.3f}"
        }
    
    # Overall recommendations
    if class_analysis:
        best_class = max(class_analysis.values(), key=lambda x: abs(x['spearman_correlation']))
        insights['recommendations']['best_class_for_layer'] = {
            'class_name': best_class['class_name'],
            'spearman_correlation': best_class['spearman_correlation'],
            'reason': f"Highest absolute Spearman correlation ({best_class['spearman_correlation']:.3f})"
        }
    
    if severity_analysis:
        best_severity = max(severity_analysis.values(), key=lambda x: abs(x['spearman_correlation']))
        insights['recommendations']['best_severity_for_layer'] = {
            'severity_level': best_severity['severity_level'],
            'spearman_correlation': best_severity['spearman_correlation'],
            'reason': f"Highest absolute Spearman correlation ({best_severity['spearman_correlation']:.3f})"
        }
    
    # Summary statistics (adapted for variable number of classes)
    all_spearman = [data['spearman_correlation'] for data in class_analysis.values()]
    all_variance_ratios = [data['variance_to_mean_ratio'] for data in class_analysis.values()]
    
    # Filter out NaN values for statistics
    valid_spearman = [x for x in all_spearman if not np.isnan(x)]
    valid_variance_ratios = [x for x in all_variance_ratios if not np.isnan(x)]
    
    insights['summary_statistics'] = {
        'dataset': dataset,
        'num_classes': num_classes,
        'mean_spearman_across_classes': np.mean(valid_spearman) if valid_spearman else 0,
        'std_spearman_across_classes': np.std(valid_spearman) if valid_spearman else 0,
        'mean_variance_ratio_across_classes': np.mean(valid_variance_ratios) if valid_variance_ratios else 0,
        'classes_with_positive_correlation': sum(1 for corr in valid_spearman if corr > 0),
        'classes_with_strong_correlation': sum(1 for corr in valid_spearman if abs(corr) > 0.3),
        'percentage_classes_positive_correlation': (sum(1 for corr in valid_spearman if corr > 0) / num_classes * 100) if num_classes > 0 else 0,
        'percentage_classes_strong_correlation': (sum(1 for corr in valid_spearman if abs(corr) > 0.3) / num_classes * 100) if num_classes > 0 else 0
    }
    
    return insights

def create_stability_visualization(stability_results, analysis, output_dir, model_name, layer_name, corruption_type):
    """Create comprehensive stability analysis visualization."""
    print("📈 Creating stability analysis visualization...")
    
    # Handle both analysis structures (direct and comprehensive)
    if 'overall_analysis' in analysis:
        # Comprehensive analysis structure
        core_analysis = analysis['overall_analysis']
        is_comprehensive = True
    else:
        # Direct analysis structure
        core_analysis = analysis
        is_comprehensive = False
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data
    scores = [r['stability_score'] for r in stability_results]
    correct_predictions = [r for r in stability_results if r['is_correct']]
    incorrect_predictions = [r for r in stability_results if not r['is_correct']]
    
    correct_scores = [r['stability_score'] for r in correct_predictions]
    incorrect_scores = [r['stability_score'] for r in incorrect_predictions]
    
    # 1. Stability Score Distribution
    ax = axes[0, 0]
    ax.hist(scores, bins=50, alpha=0.7, color='skyblue', density=True)
    ax.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
    ax.axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero Stability')
    ax.set_xlabel('Stability Score')
    ax.set_ylabel('Density')
    ax.set_title(f'Stability Score Distribution\n{model_name}-{layer_name} on {corruption_type}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Correct vs Incorrect Predictions
    ax = axes[0, 1]
    if correct_scores and incorrect_scores:
        ax.hist([correct_scores, incorrect_scores], bins=30, alpha=0.7, 
                label=['Correct Predictions', 'Incorrect Predictions'],
                color=['green', 'red'], density=True)
        ax.set_xlabel('Stability Score')
        ax.set_ylabel('Density')
        ax.set_title('Stability by Prediction Correctness', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Stability vs Distance Ratio
    ax = axes[0, 2]
    separation_ratios = [r['separation_ratio'] for r in stability_results]
    scatter = ax.scatter(scores, separation_ratios, c=[r['is_correct'] for r in stability_results], 
                        cmap='RdYlGn', alpha=0.6)
    ax.set_xlabel('Stability Score')
    ax.set_ylabel('Unfriend/Friend Distance Ratio')
    ax.set_title('Stability vs Separation Ratio', fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Correct Prediction')
    ax.grid(True, alpha=0.3)
    
    # 4. Statistics Summary
    ax = axes[1, 0]
    ax.axis('off')
    
    # Get percentile scores for additional info
    percentile_scores = core_analysis.get('percentile_scores', [])
    
    # Build stats text with conditional comprehensive analysis info
    comprehensive_note = "\n    🔍 COMPREHENSIVE ANALYSIS MODE" if is_comprehensive else ""
    
    stats_text = f"""
    STABILITY STATISTICS:{comprehensive_note}
    
    Overall:
    Mean: {core_analysis['overall_stats']['mean_stability']:.5f} ± {core_analysis['overall_stats']['std_stability']:.5f}
    Range: [{core_analysis['overall_stats']['min_stability']:.5f}, {core_analysis['overall_stats']['max_stability']:.5f}]
    Stability Range: {core_analysis['overall_stats'].get('stability_range', 0):.5f}
    Positive: {core_analysis['overall_stats']['positive_stability_pct']:.1f}%
    
    Calibration Quality:
    Variance/Mean Ratio: {core_analysis['overall_stats'].get('variance_to_mean_ratio', 0):.2f}
    Percentile Range: {core_analysis['overall_stats'].get('percentile_range', 0):.1f}%
    Calibration Info: {core_analysis['overall_stats'].get('calibration_informativeness', 0):.2f}
    
    Correct vs Incorrect:
    Correct Mean %ile: {core_analysis['correct_prediction_stats'].get('mean_percentile', 0):.1f}
    Incorrect Mean %ile: {core_analysis['incorrect_prediction_stats'].get('mean_percentile', 0):.1f}
    
    Correlations (Isotonic Regression Analysis):
    Spearman (Monotonic): {core_analysis.get('stability_correctness_spearman', 0):.5f} (p={core_analysis.get('stability_correctness_spearman_p', 1):.5f})
    Pearson (Linear): {core_analysis.get('stability_correctness_pearson', 0):.5f} (p={core_analysis.get('stability_correctness_pearson_p', 1):.5f})
    
    Percentile Correlations:
    Spearman: {core_analysis.get('percentile_correctness_spearman', 0):.5f} (p={core_analysis.get('percentile_correctness_spearman_p', 1):.5f})
    Pearson: {core_analysis.get('percentile_correctness_pearson', 0):.5f} (p={core_analysis.get('percentile_correctness_pearson_p', 1):.5f})
    
    🎯 Spearman > Pearson = Non-linear monotonic (optimal for isotonic mapping)
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # 5. Calibration Quality Metrics
    ax = axes[1, 1]
    
    # Create stability bins and calculate accuracy per bin
    n_bins = 10
    if len(scores) > n_bins:
        score_bins = np.percentile(scores, np.linspace(0, 100, n_bins + 1))
        bin_centers = []
        bin_accuracies = []
        
        for i in range(len(score_bins) - 1):
            bin_mask = (np.array(scores) >= score_bins[i]) & (np.array(scores) < score_bins[i + 1])
            if np.sum(bin_mask) > 0:
                bin_center = (score_bins[i] + score_bins[i + 1]) / 2
                bin_accuracy = np.mean([stability_results[j]['is_correct'] for j in range(len(stability_results)) if bin_mask[j]])
                bin_centers.append(bin_center)
                bin_accuracies.append(bin_accuracy)
        
        if bin_centers:
            ax.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Stability Score (Binned)')
            ax.set_ylabel('Accuracy')
            ax.set_title('Stability vs Accuracy\n(Calibration Quality)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Calculate correlation for calibration quality
            if len(bin_centers) > 2:
                cal_corr, _ = safe_correlation(bin_centers, bin_accuracies, 'pearson')
                ax.text(0.05, 0.95, f'Stability-Accuracy\nCorrelation: {cal_corr:.5f}', 
                       transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 6. Feature Quality Assessment
    ax = axes[1, 2]
    
    # Calculate feature quality metrics
    friend_dists = [r['min_friend_dist'] for r in stability_results if r['min_friend_dist'] != float('inf')]
    unfriend_dists = [r['min_unfriend_dist'] for r in stability_results]
    
    ax.hist([friend_dists, unfriend_dists], bins=30, alpha=0.7,
            label=['Friend Distances', 'Unfriend Distances'],
            color=['blue', 'orange'], density=True)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Density')
    ax.set_title('Friend vs Unfriend Distance Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Stability Analysis: {model_name}-{layer_name} on {corruption_type}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Stability analysis plot saved to {output_dir}/stability_analysis.png")

def get_pixel_space_predictions(corrupted_images, clean_images, labels, max_samples=1000):
    """
    Get predictions for pixel space using nearest neighbor classification.
    Since we don't have a model for pixel space, we use k-NN classification.
    """
    print("🔮 Getting pixel space predictions using k-NN classification...")
    
    # Flatten and normalize images
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    
    clean_pixels_norm = (clean_images.astype(np.float32) / 255.0 - mean) / std
    clean_pixels_flat = clean_pixels_norm.reshape(len(clean_images), -1)
    
    corrupted_pixels_norm = (corrupted_images.astype(np.float32) / 255.0 - mean) / std
    corrupted_pixels_flat = corrupted_pixels_norm.reshape(len(corrupted_images), -1)
    
    # Limit samples for efficiency
    if len(corrupted_pixels_flat) > max_samples:
        indices = np.random.choice(len(corrupted_pixels_flat), max_samples, replace=False)
        corrupted_pixels_flat = corrupted_pixels_flat[indices]
    else:
        indices = np.arange(len(corrupted_pixels_flat))
    
    # Use k-NN classification with k=5
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(clean_pixels_flat, labels)
    
    predictions = knn.predict(corrupted_pixels_flat)
    
    print(f"✅ Got k-NN predictions for {len(predictions)} samples")
    
    # Return predictions with original indices
    full_predictions = np.zeros(len(corrupted_images), dtype=int)
    full_predictions[indices] = predictions
    
    return full_predictions

def validate_existing_results(corruption_output_dir, corruption_type, args):
    """
    Comprehensively validate existing results to determine if they're complete and usable.
    
    Args:
        corruption_output_dir: Path to corruption-specific output directory
        corruption_type: Name of corruption being processed
        args: Command line arguments
    
    Returns:
        bool: True if results are complete and should be skipped, False if need to reprocess
    """
    
    # Expected files for complete results
    expected_files = [
        corruption_output_dir / 'stability_analysis.png',
        corruption_output_dir / 'stability_detailed_results.json'
    ]
    
    # Add CSV file for comprehensive mode
    if args.comprehensive_analysis:
        expected_files.append(corruption_output_dir / 'stability_data_with_metadata.csv')
    
    # Check if all expected files exist
    if not all(f.exists() for f in expected_files):
        missing_files = [f.name for f in expected_files if not f.exists()]
        print(f"⚠️ {corruption_type}: Missing files {missing_files}, reprocessing...")
        return False
    
    try:
        # Load and validate JSON structure
        json_file = corruption_output_dir / 'stability_detailed_results.json'
        with open(json_file, 'r') as f:
            existing_results = json.load(f)
        
        # Validate basic structure
        required_top_level = ['experiment_config', 'detailed_results']
        for key in required_top_level:
            if key not in existing_results:
                print(f"⚠️ {corruption_type}: Missing top-level key '{key}', reprocessing...")
                return False
        
        # Validate experiment config matches current parameters
        config = existing_results.get('experiment_config', {})
        config_checks = [
            ('model', args.model),
            ('layer', args.layer),
            ('seed', args.seed),
            ('training_method', args.training_method),
            ('corruption', corruption_type),
            ('comprehensive_analysis', args.comprehensive_analysis)
        ]
        
        # Check stability space settings if they exist in config
        if 'use_stability_space' in config:
            use_stability_space = not args.force_manual_calculation and args.use_stability_space
            config_checks.extend([
                ('use_stability_space', use_stability_space),
                ('stability_library', args.stability_library),
                ('stability_metric', args.stability_metric)
            ])
        
        for key, expected_value in config_checks:
            if config.get(key) != expected_value:
                print(f"⚠️ {corruption_type}: Config mismatch for '{key}': expected {expected_value}, got {config.get(key)}, reprocessing...")
                return False
        
        # Mode-specific validation
        if args.comprehensive_analysis:
            # Comprehensive mode: check for dimensional_analysis
            if 'dimensional_analysis' not in existing_results:
                print(f"⚠️ {corruption_type}: Missing 'dimensional_analysis' for comprehensive mode, reprocessing...")
                return False
            
            dimensional_analysis = existing_results['dimensional_analysis']
            
            # Check dimensional analysis structure
            required_dimensional_keys = ['overall_analysis', 'by_class_analysis', 'by_severity_analysis', 'insights', 'has_dimensional_data']
            for key in required_dimensional_keys:
                if key not in dimensional_analysis:
                    print(f"⚠️ {corruption_type}: Missing dimensional analysis key '{key}', reprocessing...")
                    return False
            
            # Check that has_dimensional_data is True
            if not dimensional_analysis.get('has_dimensional_data', False):
                print(f"⚠️ {corruption_type}: Dimensional analysis flag is False, reprocessing...")
                return False
            
            # Get analysis for correlation validation from overall_analysis
            stability_analysis = dimensional_analysis.get('overall_analysis', {})
            
            # Validate samples_per_class_severity matches if provided
            if args.samples_per_class_severity != config.get('samples_per_class_severity'):
                print(f"⚠️ {corruption_type}: samples_per_class_severity mismatch, reprocessing...")
                return False
                
        else:
            # Standard mode: check for stability_analysis
            if 'stability_analysis' not in existing_results:
                print(f"⚠️ {corruption_type}: Missing 'stability_analysis' for standard mode, reprocessing...")
                return False
            
            stability_analysis = existing_results['stability_analysis']
        
        # Validate stability analysis structure (common for both modes)
        if not isinstance(stability_analysis, dict):
            print(f"⚠️ {corruption_type}: stability_analysis is not a dict, reprocessing...")
            return False
        
        # Check for overall_stats
        if 'overall_stats' not in stability_analysis:
            print(f"⚠️ {corruption_type}: Missing 'overall_stats' in stability_analysis, reprocessing...")
            return False
        
        overall_stats = stability_analysis['overall_stats']
        
        # Validate essential overall_stats fields
        required_stats = [
            'mean_stability', 'std_stability', 'min_stability', 'max_stability',
            'positive_stability_pct', 'total_samples', 'variance_to_mean_ratio',
            'calibration_informativeness'
        ]
        
        for stat in required_stats:
            if stat not in overall_stats:
                print(f"⚠️ {corruption_type}: Missing overall stat '{stat}', reprocessing...")
                return False
            
            # Check for NaN values in critical stats
            value = overall_stats[stat]
            if isinstance(value, (int, float)) and np.isnan(value):
                print(f"⚠️ {corruption_type}: NaN value in overall stat '{stat}', reprocessing...")
                return False
        
        # Validate enhanced correlation analysis
        required_correlations = [
            'stability_correctness_spearman',
            'stability_correctness_pearson', 
            'percentile_correctness_spearman',
            'percentile_correctness_pearson'
        ]
        
        missing_correlations = []
        for corr_key in required_correlations:
            if corr_key not in stability_analysis:
                missing_correlations.append(corr_key)
            else:
                # Check if correlation value is valid (not NaN, and reasonable range)
                corr_value = stability_analysis[corr_key]
                if not isinstance(corr_value, (int, float)) or np.isnan(corr_value):
                    missing_correlations.append(f"{corr_key} (invalid)")
        
        if missing_correlations:
            print(f"⚠️ {corruption_type}: Missing/invalid correlations {missing_correlations}, reprocessing...")
            return False
        
        # Validate detailed_results structure
        detailed_results = existing_results.get('detailed_results', [])
        if not isinstance(detailed_results, list) or len(detailed_results) == 0:
            print(f"⚠️ {corruption_type}: Invalid or empty detailed_results, reprocessing...")
            return False
        
        # Sample check first result
        if detailed_results:
            sample_result = detailed_results[0]
            required_result_keys = ['stability_score', 'is_correct', 'predicted_class', 'true_class']
            
            for key in required_result_keys:
                if key not in sample_result:
                    print(f"⚠️ {corruption_type}: Missing detailed result key '{key}', reprocessing...")
                    return False
            
            # For comprehensive mode, check for metadata
            if args.comprehensive_analysis:
                metadata_keys = ['class_id', 'class_name', 'severity_level', 'combo_id']
                for key in metadata_keys:
                    if key not in sample_result:
                        print(f"⚠️ {corruption_type}: Missing metadata key '{key}' in comprehensive mode, reprocessing...")
                        return False
        
        # Validate visualization file size (basic check)
        png_file = corruption_output_dir / 'stability_analysis.png'
        if png_file.stat().st_size < 10000:  # Less than 10KB suggests corrupted image
            print(f"⚠️ {corruption_type}: PNG file too small ({png_file.stat().st_size} bytes), reprocessing...")
            return False
        
        # If we made it here, everything looks good
        print(f"✅ {corruption_type}: Complete and valid results found, skipping...")
        return True
        
    except (json.JSONDecodeError, KeyError, FileNotFoundError, AttributeError, TypeError) as e:
        print(f"⚠️ {corruption_type}: Error validating results ({type(e).__name__}: {e}), reprocessing...")
        return False
    except Exception as e:
        print(f"⚠️ {corruption_type}: Unexpected error validating results ({e}), reprocessing...")
        return False

def convert_numpy_types_with_precision(obj):
    """Convert numpy types to Python types with 5 decimal precision for floats."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types_with_precision(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types_with_precision(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return round(float(obj), 5)  # 5 decimal places
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, float):
        return round(obj, 5)  # 5 decimal places for regular floats too
    else:
        return obj

def safe_get_core_analysis(analysis_result):
    """Safely extract core analysis from either direct or comprehensive structure with error handling."""
    try:
        if isinstance(analysis_result, dict):
            if 'overall_analysis' in analysis_result:
                # Comprehensive analysis structure
                core = analysis_result['overall_analysis']
                if isinstance(core, dict) and 'overall_stats' in core:
                    return core
                else:
                    print(f"⚠️ Invalid overall_analysis structure: {type(core)}")
                    return None
            elif 'overall_stats' in analysis_result:
                # Direct analysis structure
                return analysis_result
            else:
                print(f"⚠️ No recognized analysis structure found. Keys: {list(analysis_result.keys())}")
                return None
        else:
            print(f"⚠️ Expected dict, got {type(analysis_result)}")
            return None
    except Exception as e:
        print(f"⚠️ Error extracting core analysis: {e}")
        return None

def main():
    """Run stability score statistical analysis."""
    args = parse_args()
    
    # Determine corruptions to process
    # Check if --corruptions was explicitly provided by checking if it's in sys.argv
    corruptions_explicitly_provided = '--corruptions' in sys.argv
    
    # Check for common mistake: --corruption all (should be --all-corruptions)
    if args.corruption.lower() == 'all':
        print("⚠️ Warning: You used '--corruption all' but 'all' is not a valid corruption type.")
        print("   Did you mean '--all-corruptions'? Converting to all-corruptions mode...")
        args.all_corruptions = True
    
    if args.all_corruptions:
        corruption_dir = getattr(args, f'{args.dataset}c_dir', None)
        corruptions_to_process = get_available_corruptions(dataset=args.dataset, corruption_dir=corruption_dir)
        print(f"🌪️ Processing ALL available corruptions for {args.dataset.upper()}: {len(corruptions_to_process)} types")
    elif corruptions_explicitly_provided:
        # User explicitly provided --corruptions argument
        corruptions_to_process = args.corruptions
        print(f"🌪️ Processing multiple corruptions: {corruptions_to_process}")
    else:
        # Single corruption mode (either explicit --corruption or default behavior)
        corruptions_to_process = [args.corruption]
        print(f"🌪️ Processing single corruption: {args.corruption}")
    
    print("\n🔧 STABILITY CALCULATION METHOD:")
    if not args.force_manual_calculation and args.use_stability_space and STABILITY_SPACE_AVAILABLE:
        print(f"   ✅ Using StabilitySpace for consistency with calibration")
        print(f"   📚 Library: {args.stability_library}")
        print(f"   📏 Metric: {args.stability_metric}")
        print(f"   🖥️  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print(f"   💡 Note: Results should match geometric calibration exactly")
        
        if torch.cuda.is_available():
            print(f"   ⚠️  GPU calculations use float32 precision (vs CPU float64)")
            print(f"   ⚠️  Small numerical differences from manual calculation expected")
    elif args.force_manual_calculation:
        print(f"   🔧 Manual calculation forced by user")
        print(f"   📏 Using L2 distance with manual implementation")
        print(f"   💡 Note: Exact manual calculation, may differ from calibration")
    elif not STABILITY_SPACE_AVAILABLE:
        print(f"   ⚠️  StabilitySpace not available, using manual calculation")
        print(f"   💡 Note: Install missing dependencies for StabilitySpace")
    else:
        print(f"   📊 Using manual calculation (StabilitySpace disabled)")
    print("")
    print("STABILITY SCORE STATISTICAL ANALYSIS")
    print("Geometric Feature Quality Assessment for AAAI Paper Section 4.2")
    print("Enhanced with StabilitySpace Integration for Calibration Consistency")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Seed: {args.seed}")
    print(f"Training Method: {args.training_method}")
    print(f"Corruptions: {corruptions_to_process}")
    print(f"Max Analyze: {args.max_analyze}")
    print(f"Use StabilitySpace: {not args.force_manual_calculation and args.use_stability_space}")
    if not args.force_manual_calculation and args.use_stability_space:
        print(f"StabilitySpace Library: {args.stability_library}")
        print(f"StabilitySpace Metric: {args.stability_metric}")
    print("="*80)
    
    # Load data and model ONCE (shared across all corruptions)
    print(f"\nStep 1: Loading clean {args.dataset.upper()} data and model...")
    
    # Determine number of classes
    num_classes = 10 if args.dataset == 'cifar10' else 100
    
    # Load clean test data based on dataset
    if args.dataset == 'cifar10':
        # Load CIFAR-10 test data
        test_batch_path = PROJECT_ROOT / 'data' / 'cifar-10-batches-py' / 'test_batch'
        
        # Try alternative locations if not found
        if not test_batch_path.exists():
            alt_paths = [
                PROJECT_ROOT / 'Data' / 'cifar-10-batches-py' / 'test_batch',
                Path('./data/cifar-10-batches-py/test_batch'),
                Path('/datasets/cifar-10-batches-py/test_batch')
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    test_batch_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"CIFAR-10 test data not found in any of {[test_batch_path] + alt_paths}")
        
        with open(test_batch_path, 'rb') as fo:
            test_batch = pickle.load(fo, encoding='bytes')
        
        clean_images = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(test_batch[b'labels'])
        
    elif args.dataset == 'cifar100':
        # Load CIFAR-100 test data
        test_batch_path = PROJECT_ROOT / 'data' / 'cifar-100-python' / 'test'
        
        # Try alternative locations if not found
        if not test_batch_path.exists():
            alt_paths = [
                PROJECT_ROOT / 'Data' / 'cifar-100-python' / 'test',
                Path('./data/cifar-100-python/test'),
                Path('/datasets/cifar-100-python/test')
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    test_batch_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"CIFAR-100 test data not found in any of {[test_batch_path] + alt_paths}")
        
        with open(test_batch_path, 'rb') as fo:
            test_batch = pickle.load(fo, encoding='bytes')
        
        clean_images = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(test_batch[b'fine_labels'])  # Use fine labels for CIFAR-100
        
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"✅ Loaded {len(clean_images)} clean {args.dataset.upper()} test images")
    
    # Handle model loading based on layer type
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    
    if args.layer.lower() in ['pixel', 'pixels', 'raw', 'physical']:
        print(f"🖼️ Pixel space analysis - no model loading required")
    else:
        # Load trained model for semantic analysis
        print(f"🤖 Loading {args.model} model for {args.dataset.upper()}...")
        
        model_path = check_model_exists(args.training_method, args.dataset, args.model, args.seed, args.results_dir)
        if model_path is None:
            print("❌ No model found. Cannot proceed.")
            sys.exit(1)
        
        try:
            if load_trained_model:
                model = load_trained_model(str(model_path), args.model, num_classes, device)
            else:
                model = load_basic_model(model_path, args.model, num_classes)
            
            model.eval()
            print(f"✅ Loaded {args.model} for {args.dataset.upper()}")
            print(f"   Model path: {model_path}")
            print(f"   Number of classes: {num_classes}")
        except Exception as e:
            print(f"❌ Failed to load {args.model}: {e}")
            sys.exit(1)
    
    # Extract clean features ONCE (shared across all corruptions)
    print("\nStep 2: Extracting clean features...")
    clean_features = extract_features_batch_robust(
        clean_images, model, args.model, args.layer, device, args.batch_size
    )
    print(f"✅ Clean feature extraction complete: {clean_features.shape}")
    
    # Create base output directory
    base_output_dir = construct_output_dir(args)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each corruption sequentially
    all_corruption_results = {}
    successful_corruptions = []
    failed_corruptions = []
    skipped_corruptions = []
    
    for corruption_idx, corruption_type in enumerate(corruptions_to_process):
        print(f"\n{'='*80}")
        print(f"PROCESSING CORRUPTION {corruption_idx + 1}/{len(corruptions_to_process)}: {corruption_type.upper()}")
        print(f"{'='*80}")
        
        try:
            # Create corruption-specific output directory
            corruption_output_dir = construct_output_dir(args, corruption_type)
            corruption_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if complete and valid results already exist
            should_skip = validate_existing_results(corruption_output_dir, corruption_type, args)
            
            if should_skip:
                # Load existing results for summary
                try:
                    with open(corruption_output_dir / 'stability_detailed_results.json', 'r') as f:
                        existing_results = json.load(f)
                    
                    # Extract analysis for summary using safe method
                    if args.comprehensive_analysis:
                        analysis = existing_results.get('dimensional_analysis', {})
                    else:
                        analysis = existing_results.get('stability_analysis', {})
                    
                    if analysis:
                        all_corruption_results[corruption_type] = analysis
                        successful_corruptions.append(corruption_type)
                        skipped_corruptions.append(corruption_type)
                        print(f"📊 {corruption_type}: Loaded existing results for summary")
                    
                except Exception as e:
                    print(f"⚠️ {corruption_type}: Error loading existing results for summary: {e}")
                
                continue
            
            # Load corruption data - comprehensive mode loads all severities
            print(f"📁 Loading {corruption_type} corruption data for {args.dataset.upper()}...")
            try:
                corruption_dir = getattr(args, f'{args.dataset}c_dir', None)
                
                if args.comprehensive_analysis:
                    # Load all severities for comprehensive analysis
                    corrupted_images_by_severity = load_corruption_data(
                        corruption_type, dataset=args.dataset, corruption_dir=corruption_dir, 
                        return_all_severities=True
                    )
                    
                    # Create comprehensive sampling plan
                    selected_samples, sampling_metadata = comprehensive_stability_sampling(
                        corrupted_images_by_severity, labels, args.samples_per_class_severity, args.dataset
                    )
                    
                    # Extract images and create corresponding labels/predictions arrays
                    corrupted_images = np.array([sample['corrupted_image'] for sample in selected_samples])
                    sample_labels = np.array([sample['clean_label'] for sample in selected_samples])
                    
                    print(f"✅ Comprehensive sampling: {len(corrupted_images)} images across {len(set(m['combo_id'] for m in sampling_metadata))} class-severity combinations")
                    
                else:
                    # Standard mode: single severity
                    corrupted_images = load_corruption_data(
                        corruption_type, dataset=args.dataset, corruption_dir=corruption_dir, severity=5
                    )
                    sample_labels = labels
                    sampling_metadata = None
                    
            except FileNotFoundError as e:
                print(f"⚠️ Skipping {corruption_type}: {e}")
                failed_corruptions.append(corruption_type)
                continue
            
            # Extract corrupted features
            print(f"🧠 Extracting corrupted features for {corruption_type}...")
            corrupted_features = extract_features_batch_robust(
                corrupted_images, model, args.model, args.layer, device, args.batch_size
            )
            print(f"✅ Corrupted feature extraction complete: {corrupted_features.shape}")
            
            # Get predictions on corrupted images
            print(f"🔮 Getting predictions for {corruption_type}...")
            
            if args.layer.lower() in ['pixel', 'pixels', 'raw', 'physical']:
                # Use k-NN for pixel space predictions
                predictions = get_pixel_space_predictions(corrupted_images, clean_images, labels, len(corrupted_images))
                # Note: predictions are already aligned with corrupted_images, no indexing needed
            else:
                # Use model for semantic space predictions  
                model.eval()
                predictions = []
                
                mean = np.array([0.4914, 0.4822, 0.4465])
                std = np.array([0.2023, 0.1994, 0.2010])
                
                with torch.no_grad():
                    for i in tqdm(range(0, len(corrupted_images), args.batch_size), desc="Getting predictions"):
                        batch = corrupted_images[i:i+args.batch_size]
                        batch_norm = (batch.astype(np.float32) / 255.0 - mean) / std
                        batch_tensor = torch.from_numpy(batch_norm).permute(0, 3, 1, 2).float().to(device)
                        
                        outputs = model(batch_tensor)
                        batch_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                        predictions.extend(batch_predictions)
                
                predictions = np.array(predictions)
            
            print(f"✅ Got predictions for {len(predictions)} samples")
            
            # Calculate stability scores with metadata - use enhanced version for comprehensive mode
            print(f"🔬 Calculating stability scores for {corruption_type}...")
            use_stability_space = not args.force_manual_calculation and args.use_stability_space
            
            if args.comprehensive_analysis:
                # For comprehensive analysis, use full labels array to match clean_features dimensions
                # The selected samples' indices will be handled by the metadata
                print(f"🔍 Comprehensive mode: Using all {len(clean_features)} clean features for stability reference")
                stability_results = calculate_stability_scores(
                    clean_features, corrupted_features, labels, predictions, 
                    args.max_analyze, sampling_metadata, 
                    use_stability_space=use_stability_space,
                    stability_library=args.stability_library,
                    stability_metric=args.stability_metric
                )
            else:
                stability_results = calculate_stability_scores(
                    clean_features, corrupted_features, sample_labels, predictions, 
                    args.max_analyze, sampling_metadata,
                    use_stability_space=use_stability_space,
                    stability_library=args.stability_library,
                    stability_metric=args.stability_metric
                )
            print(f"✅ Calculated stability scores for {len(stability_results)} samples")
            
            # Analyze stability distribution - use dimensional analysis if comprehensive mode
            if args.comprehensive_analysis:
                print(f"🔍 Performing comprehensive dimensional analysis for {corruption_type}...")
                analysis = calculate_dimensional_analysis(stability_results, args.dataset)
            else:
                print(f"📊 Analyzing stability distribution for {corruption_type}...")
                analysis = analyze_stability_distribution(stability_results)
            
            # Create visualization
            print(f"📈 Creating visualizations for {corruption_type}...")
            create_stability_visualization(stability_results, analysis, corruption_output_dir, 
                                          args.model, args.layer, corruption_type)
            
            # Save detailed results with 5-digit precision
            print(f"💾 Saving results for {corruption_type}...")
            
            # Prepare final results structure based on analysis mode
            final_results = {
                'experiment_config': {
                    'model': args.model,
                    'layer': args.layer,
                    'seed': args.seed,
                    'training_method': args.training_method,
                    'corruption': corruption_type,
                    'max_analyze': args.max_analyze,
                    'comprehensive_analysis': args.comprehensive_analysis,
                    'samples_per_class_severity': args.samples_per_class_severity if args.comprehensive_analysis else None,
                    'use_stability_space': not args.force_manual_calculation and args.use_stability_space,
                    'stability_library': args.stability_library,
                    'stability_metric': args.stability_metric,
                    'force_manual_calculation': args.force_manual_calculation
                },
                'detailed_results': stability_results
            }
            
            # Add analysis results based on mode
            if args.comprehensive_analysis:
                final_results['dimensional_analysis'] = analysis
                # Keep backward compatibility
                final_results['stability_analysis'] = analysis.get('overall_analysis', {})
            else:
                final_results['stability_analysis'] = analysis
            
            with open(corruption_output_dir / 'stability_detailed_results.json', 'w') as f:
                json.dump(convert_numpy_types_with_precision(final_results), f, indent=2)
            
            # Save CSV data for easy analysis (especially useful for comprehensive mode)
            if args.comprehensive_analysis and stability_results:
                try:
                    # Apply 5-digit precision to stability_results before creating DataFrame
                    precision_results = convert_numpy_types_with_precision(stability_results)
                    df = pd.DataFrame(precision_results)
                    
                    # Ensure all numeric columns are rounded to 5 decimal places
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    df[numeric_columns] = df[numeric_columns].round(5)
                    
                    df.to_csv(corruption_output_dir / 'stability_data_with_metadata.csv', index=False)
                    print(f"✅ CSV data saved to {corruption_output_dir}/stability_data_with_metadata.csv")
                except Exception as e:
                    print(f"⚠️ Could not save CSV: {e}")
            
            # Store results for summary
            all_corruption_results[corruption_type] = analysis
            successful_corruptions.append(corruption_type)
            
            # Print corruption summary with enhanced interpretation
            if args.comprehensive_analysis:
                # Comprehensive analysis summary
                overall = analysis.get('overall_analysis', {}).get('overall_stats', {})
                dimensional_insights = analysis.get('insights', {})
                
                print(f"\n📊 {corruption_type.upper()} COMPREHENSIVE ANALYSIS SUMMARY:")
                print(f"   Overall Mean Stability: {overall.get('mean_stability', 0):.3f} ± {overall.get('std_stability', 0):.3f}")
                print(f"   Overall Spearman Correlation: {analysis.get('overall_analysis', {}).get('stability_correctness_spearman', 0):.3f}")
                
                # Class-specific insights
                if 'recommendations' in dimensional_insights:
                    best_class = dimensional_insights['recommendations'].get('best_class_for_layer', {})
                    best_severity = dimensional_insights['recommendations'].get('best_severity_for_layer', {})
                    
                    if best_class:
                        print(f"   🏆 Best Class for {args.layer}: {best_class['class_name']} (Spearman: {best_class['spearman_correlation']:.3f})")
                    if best_severity:
                        print(f"   🏆 Best Severity for {args.layer}: Level {best_severity['severity_level']} (Spearman: {best_severity['spearman_correlation']:.3f})")
                
                # Summary statistics
                summary_stats = dimensional_insights.get('summary_statistics', {})
                if summary_stats:
                    num_classes = summary_stats.get('num_classes', get_num_classes(args.dataset))
                    mean_spearman = summary_stats.get('mean_spearman_across_classes', 0)
                    if np.isnan(mean_spearman):
                        mean_spearman = 0.0
                    print(f"   📈 Mean Spearman Across Classes: {mean_spearman:.3f}")
                    print(f"   📈 Classes with Strong Correlation (>0.3): {summary_stats.get('classes_with_strong_correlation', 0)}/{num_classes}")
                    print(f"   📈 Classes with Positive Correlation: {summary_stats.get('classes_with_positive_correlation', 0)}/{num_classes}")
                    print(f"   📈 Strong Correlation Rate: {summary_stats.get('percentage_classes_strong_correlation', 0):.1f}%")
                    print(f"   📈 Positive Correlation Rate: {summary_stats.get('percentage_classes_positive_correlation', 0):.1f}%")
                
                # Top 3 classes by quality
                class_insights = dimensional_insights.get('class_insights', {})
                if class_insights:
                    # Filter out NaN correlations before sorting
                    valid_classes = [(name, data) for name, data in class_insights.items() 
                                   if not np.isnan(data.get('spearman_correlation', 0))]
                    
                    if valid_classes:
                        sorted_classes = sorted(valid_classes, key=lambda x: abs(x[1]['spearman_correlation']), reverse=True)[:3]
                        print(f"   🥇 Top 3 Classes for {args.layer}:")
                        for i, (class_name, data) in enumerate(sorted_classes, 1):
                            corr = data['spearman_correlation']
                            quality = data.get('quality_score', 0)
                            if np.isnan(quality):
                                quality = 0.0
                            print(f"      {i}. {class_name}: Spearman={corr:.3f}, Quality={quality:.3f}")
                    else:
                        print(f"   ⚠️ No valid class correlations found (all NaN)")
                
            else:
                # Standard analysis summary
                overall = analysis['overall_stats']
                print(f"\n📊 {corruption_type.upper()} SUMMARY:")
                print(f"   Mean Stability: {overall['mean_stability']:.3f} ± {overall['std_stability']:.3f}")
                print(f"   Stability Range: {overall.get('stability_range', 0):.3f} (wider = better for calibration)")
                print(f"   Variance/Mean Ratio: {overall.get('variance_to_mean_ratio', 0):.2f} (higher = richer info)")
                print(f"   Positive Stability: {overall['positive_stability_pct']:.1f}%")
                print(f"   Calibration Informativeness: {overall.get('calibration_informativeness', 0):.2f} (0-2 scale)")
                if 'stability_correctness_spearman' in analysis:
                    print(f"   Spearman ↔ Correctness: {analysis['stability_correctness_spearman']:.3f} (monotonic)")
                    print(f"   Pearson ↔ Correctness: {analysis.get('stability_correctness_pearson', 0):.3f} (linear)")
                    print(f"   Percentile Spearman: {analysis.get('percentile_correctness_spearman', 0):.3f}")
                    print(f"   Percentile Pearson: {analysis.get('percentile_correctness_pearson', 0):.3f}")
            
            # Interpretation guide (works for both modes)
            # Get overall stats from appropriate structure
            if args.comprehensive_analysis:
                overall_stats = analysis.get('overall_analysis', {}).get('overall_stats', {})
            else:
                overall_stats = analysis.get('overall_stats', {})
            
            variance_ratio = overall_stats.get('variance_to_mean_ratio', 0)
            if variance_ratio > 5:
                print(f"   🎯 EXCELLENT: High variance indicates rich calibration information!")
            elif variance_ratio > 2:
                print(f"   ✅ GOOD: Moderate variance provides useful calibration signal")
            else:
                print(f"   ⚠️ CONCERNING: Low variance suggests potential feature collapse")
            
            # Clean up corrupted features to save memory
            del corrupted_features, corrupted_images, predictions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error processing {corruption_type}: {e}")
            failed_corruptions.append(corruption_type)
            continue
    
    # Create summary analysis across all corruptions
    if len(successful_corruptions) > 1:
        print(f"\n{'='*80}")
        print("CREATING MULTI-CORRUPTION SUMMARY")
        print(f"{'='*80}")
        
        create_multi_corruption_summary(all_corruption_results, base_output_dir, args)
    
    # Final summary
    print(f"\n{'='*80}")
    print("STABILITY ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"🎯 Model: {args.model}-{args.layer}")
    
    # Count processed vs skipped
    total_requested = len(corruptions_to_process)
    processed_count = len(successful_corruptions) - len(skipped_corruptions)
    skipped_count = len(skipped_corruptions)
    failed_count = len(failed_corruptions)
    
    print(f"📊 Processing Summary:")
    print(f"   Total requested: {total_requested}")
    print(f"   ✅ Successful (total): {len(successful_corruptions)}")
    print(f"   🔄 Newly processed: {processed_count}")
    print(f"   ⏭️ Skipped (valid existing): {skipped_count}")
    print(f"   ❌ Failed: {failed_count}")
    
    if failed_corruptions:
        print(f"\n⚠️ Failed corruptions: {failed_corruptions}")
    
    if skipped_corruptions:
        print(f"\n⏭️ Skipped corruptions (valid results exist): {skipped_corruptions}")
    
    print(f"\n📁 Results saved to: {base_output_dir}")
    print("🎯 Generated files:")
    for corruption in successful_corruptions:
        status = "⏭️ (existing)" if corruption in skipped_corruptions else "🔄 (processed)"
        print(f"   📂 {corruption}/ {status}")
        print(f"      - stability_analysis.png")
        print(f"      - stability_detailed_results.json")
    
    if len(successful_corruptions) > 1:
        print(f"   📊 multi_corruption_summary.png")
        print(f"   📊 multi_corruption_summary.json")
    # Print interpretation for single corruption results
    if len(successful_corruptions) == 1:
        corruption = successful_corruptions[0] 
        analysis = all_corruption_results[corruption]
        
        # Handle both analysis structures (direct and comprehensive)
        if 'overall_analysis' in analysis:
            # Comprehensive analysis structure
            overall = analysis['overall_analysis']['overall_stats']
            analysis_core = analysis['overall_analysis']
        else:
            # Direct analysis structure
            overall = analysis['overall_stats']
            analysis_core = analysis
        
        print(f"\n🔍 INTERPRETING YOUR {corruption.upper()} RESULTS:")
        print("="*50)
        
        variance_ratio = overall.get('variance_to_mean_ratio', 0)
        if variance_ratio > 5:
            print("🎯 EXCELLENT: High variance/mean ratio indicates RICH calibration information!")
            print("   This means layer provides diverse stability scores perfect for confidence mapping.")
        elif variance_ratio > 2:
            print("✅ GOOD: Moderate variance provides useful calibration signal.")
        else:
            print("⚠️ CONCERNING: Low variance may indicate feature collapse.")
            
        pos_stability = overall['positive_stability_pct']
        if pos_stability > 55:
            print(f"✅ {pos_stability:.1f}% positive stability is good for semantic advantage.")
        elif pos_stability > 45:
            print(f"📊 {pos_stability:.1f}% positive stability shows moderate semantic advantage.")
        else:
            print(f"⚠️ {pos_stability:.1f}% positive stability suggests semantic struggles with this corruption.")
            
        spearman_corr = analysis_core.get('stability_correctness_spearman', 0)
        pearson_corr = analysis_core.get('stability_correctness_pearson', 0)
        
        print(f"🔗 Spearman (monotonic): {spearman_corr:.3f} | Pearson (linear): {pearson_corr:.3f}")
        
        if spearman_corr > 0.3:
            print(f"🎯 EXCELLENT: Strong Spearman correlation = reliable isotonic regression mapping!")
        elif spearman_corr > 0.15:
            print(f"📊 GOOD: Moderate Spearman correlation = useful monotonic relationship.")
        else:
            print(f"⚠️ CONCERNING: Weak Spearman correlation = poor monotonic reliability.")
        
        # Compare Spearman vs Pearson
        if spearman_corr > pearson_corr + 0.05:
            print(f"✅ Spearman > Pearson: Non-linear monotonic relationship (IDEAL for isotonic regression)")
        elif abs(spearman_corr - pearson_corr) < 0.05:
            print(f"📊 Spearman ≈ Pearson: Linear monotonic relationship (good for isotonic regression)")
        else:
            print(f"⚠️ Pearson > Spearman: May indicate non-monotonic or weak relationship")
    
    # Enhanced interpretation guide
    print("\n" + "="*60)
    print("ENHANCED INTERPRETATION GUIDE FOR YOUR PAPER")
    print("="*60)
    
    if args.comprehensive_analysis and successful_corruptions:
        print("🎯 COMPREHENSIVE DIMENSIONAL ANALYSIS INSIGHTS:")
        print("")
        
        # Generate comprehensive insights across all corruptions
        print("📊 CLASS-SPECIFIC INSIGHTS:")
        print("   Use these to state: 'For automobile class, layer X is optimal'")
        print("")
        
        print("📊 SEVERITY-SPECIFIC INSIGHTS:")
        print("   Use these to state: 'For severity 5, layer Y achieves best correlation'")
        print("")
        
        print("🔍 TO EXTRACT YOUR SPECIFIC INSIGHTS:")
        print(f"   1. Check dimensional_analysis.insights.class_insights in saved JSON files")
        print(f"   2. Look at by_class_analysis for per-class Spearman correlations")
        print(f"   3. Check by_severity_analysis for per-severity optimal layers")
        print(f"   4. Use recommendations.best_class_for_layer and best_severity_for_layer")
        print("")
        
        print("🎯 EXAMPLE INSIGHTS YOU CAN NOW MAKE:")
        print("   • 'For severity 5 corruptions, dense1 achieves optimal Spearman correlation (0.45)'")
        print("   • 'Automobile class shows strongest stability-correctness relationship with trans3 (0.52)' (CIFAR-10)")
        print("   • 'Apple class demonstrates superior geometric features in layer2' (CIFAR-100)")
        print("   • 'Across all severities, layer3 maintains consistent geometric quality'")
        print("   • 'Bird and airplane classes benefit most from deeper semantic features'")
        print("")
        
        print("📈 DATA EXPORT FOR FURTHER ANALYSIS:")
        print(f"   • stability_data_with_metadata.csv contains per-sample data with class/severity")
        print(f"   • Filter by class_name='automobile' to analyze automobile-specific patterns")
        print(f"   • Filter by severity_level=5 to focus on highest corruption")
        print(f"   • Group by combo_id for class-severity specific insights")
        print("")
        
        # Show example commands for data analysis
        print("🔧 EXAMPLE ANALYSIS COMMANDS:")
        print("   # Find best layer for each class:")
        print("   grep -A 5 'best_class_for_layer' */stability_detailed_results.json")
        print("")
        print("   # Find best layer for each severity:")
        print("   grep -A 5 'best_severity_for_layer' */stability_detailed_results.json")
        print("")
        print("   # Compare across multiple layers:")
        print("   python compare_layers_comprehensive.py --results-dir ./section4_stability_analysis/")
        
    else:
        print("🎯 HIGH VARIANCE = EXCELLENT for calibration!")
        print("   • Variance > Mean → Rich stability spectrum for confidence mapping")
        print("   • Wide percentile range → Diverse geometric properties")
        print("   • High informativeness score → Optimal calibration potential")
        print("")
        print("🔍 What to look for across layers:")
        print("   • Layer3/Trans3: High variance + strong Spearman correlation → OPTIMAL")
        print("   • Layer4/Dense4: Low variance despite good k-NN → Feature collapse")
        print("   • Pixel space: Poor correlation → Insufficient abstraction")
        print("")
        print("💡 FOR DIMENSIONAL INSIGHTS, RERUN WITH:")
        print(f"   python {sys.argv[0]} --comprehensive-analysis --samples-per-class-severity 20")
    
    print("")
    print("📊 Key metrics for your paper:")
    print("   • Variance/Mean Ratio: >5 = excellent, >2 = good, <1 = concerning")
    print("   • Calibration Informativeness: >1.5 = excellent, >1.0 = good")
    print("   • Percentile Range: >80% = excellent separation")
    print("")
    print("🔗 CORRELATION ANALYSIS for ISOTONIC REGRESSION:")
    print("   • Spearman (PRIMARY): Rank-based, captures monotonic relationships")
    print("     - >0.3 = excellent isotonic mapping potential")
    print("     - >0.15 = good monotonic relationship")
    print("     - <0.15 = poor isotonic regression reliability")
    print("   • Pearson (SECONDARY): Linear relationship assessment")
    print("     - Spearman > Pearson: Non-linear monotonic (IDEAL for isotonic)")
    print("     - Spearman ≈ Pearson: Linear monotonic (good for isotonic)")
    print("     - Pearson > Spearman: Non-monotonic or weak (concerning)")
    print("")
    print("🎯 Your narrative: 'High stability variance indicates rich geometric")
    print("   structure optimal for calibration. Strong Spearman correlations")
    print("   demonstrate monotonic stability-correctness relationships ideal for")
    print("   isotonic regression mapping, while feature collapse in deeper")
    print("   layers reduces calibration-relevant information despite maintaining")
    print("   class separability.'")
    
    # Exit with appropriate code
    sys.exit(0 if successful_corruptions else 1)

def create_multi_corruption_summary(all_corruption_results, output_dir, args):
    """Create summary analysis across multiple corruptions."""
    print("📊 Creating multi-corruption summary...")
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Mean Stability by Corruption
    ax = axes[0, 0]
    corruptions = list(all_corruption_results.keys())
    
    # Safely extract core analysis for each corruption with error handling
    core_analyses = []
    valid_corruptions = []
    
    for corruption_name in corruptions:
        core_analysis = safe_get_core_analysis(all_corruption_results[corruption_name])
        if core_analysis is not None:
            core_analyses.append(core_analysis)
            valid_corruptions.append(corruption_name)
        else:
            print(f"⚠️ Skipping {corruption_name} due to invalid analysis structure")
    
    if not core_analyses:
        print("❌ No valid corruption analyses found for summary!")
        return
    
    corruptions = valid_corruptions  # Use only valid corruptions
    
    mean_stabilities = [core_analysis['overall_stats']['mean_stability'] for core_analysis in core_analyses]
    std_stabilities = [core_analysis['overall_stats']['std_stability'] for core_analysis in core_analyses]
    
    bars = ax.bar(range(len(corruptions)), mean_stabilities, yerr=std_stabilities, 
                  capsize=5, alpha=0.7, color='skyblue')
    ax.set_xticks(range(len(corruptions)))
    ax.set_xticklabels(corruptions, rotation=45, ha='right')
    ax.set_ylabel('Mean Stability Score')
    ax.set_title(f'Mean Stability by Corruption\n{args.model}-{args.layer}', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean_val in zip(bars, mean_stabilities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Positive Stability Percentage
    ax = axes[0, 1]
    pos_percentages = [core_analysis['overall_stats']['positive_stability_pct'] for core_analysis in core_analyses]
    
    bars = ax.bar(range(len(corruptions)), pos_percentages, alpha=0.7, color='lightgreen')
    ax.set_xticks(range(len(corruptions)))
    ax.set_xticklabels(corruptions, rotation=45, ha='right')
    ax.set_ylabel('Positive Stability (%)')
    ax.set_title('Positive Stability Percentage by Corruption', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, pct in zip(bars, pos_percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. Stability-Correctness Correlation (Spearman - Primary for Isotonic Regression)
    ax = axes[1, 0]
    spearman_correlations = [core_analysis.get('stability_correctness_spearman', core_analysis.get('stability_correctness_correlation', 0)) for core_analysis in core_analyses]
    pearson_correlations = [core_analysis.get('stability_correctness_pearson', 0) for core_analysis in core_analyses]
    
    # Use Spearman as primary with Pearson as comparison
    bars = ax.bar(range(len(corruptions)), spearman_correlations, alpha=0.7, color='orange', label='Spearman (Monotonic)')
    
    # Add Pearson as lighter bars for comparison
    ax.bar(range(len(corruptions)), pearson_correlations, alpha=0.4, color='lightcoral', 
           label='Pearson (Linear)', width=0.6)
    
    ax.set_xticks(range(len(corruptions)))
    ax.set_xticklabels(corruptions, rotation=45, ha='right')
    ax.set_ylabel('Stability-Correctness Correlation')
    ax.set_title('Spearman vs Pearson Correlations by Corruption\n(Primary: Spearman for Isotonic Regression)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    
    for bar, corr in zip(bars, spearman_correlations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{corr:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Summary Statistics Table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate overall statistics using Spearman correlations as primary
    overall_mean_stability = np.mean(mean_stabilities)
    overall_mean_positive = np.mean(pos_percentages)
    
    # Filter out NaN values for correlations
    valid_spearman = [c for c in spearman_correlations if not np.isnan(c) and c != 0]
    valid_pearson = [c for c in pearson_correlations if not np.isnan(c) and c != 0]
    
    overall_mean_spearman = np.mean(valid_spearman) if valid_spearman else 0
    overall_mean_pearson = np.mean(valid_pearson) if valid_pearson else 0
    
    # Find best corruptions (handle NaN values)
    valid_spearman_with_names = [(corr, name) for corr, name in zip(spearman_correlations, corruptions) if not np.isnan(corr)]
    valid_pearson_with_names = [(corr, name) for corr, name in zip(pearson_correlations, corruptions) if not np.isnan(corr)]
    
    best_stability_corruption = corruptions[np.argmax(mean_stabilities)]
    best_spearman_corruption = max(valid_spearman_with_names, key=lambda x: abs(x[0]))[1] if valid_spearman_with_names else "None"
    best_pearson_corruption = max(valid_pearson_with_names, key=lambda x: abs(x[0]))[1] if valid_pearson_with_names else "None"
    most_consistent_corruption = corruptions[np.argmax(pos_percentages)]
    
    best_spearman_value = max(valid_spearman_with_names, key=lambda x: abs(x[0]))[0] if valid_spearman_with_names else 0
    best_pearson_value = max(valid_pearson_with_names, key=lambda x: abs(x[0]))[0] if valid_pearson_with_names else 0
    
    stats_text = f"""
    MULTI-CORRUPTION SUMMARY
    {args.model}-{args.layer}
    
    Corruptions Analyzed: {len(corruptions)}
    
    Overall Statistics:
    • Mean Stability: {overall_mean_stability:.5f}
    • Mean Positive %: {overall_mean_positive:.1f}%
    • Mean Spearman: {overall_mean_spearman:.5f}
    • Mean Pearson: {overall_mean_pearson:.5f}
    
    Best Corruption (Stability):
    {best_stability_corruption} ({max(mean_stabilities):.5f})
    
    Best Spearman Correlation:
    {best_spearman_corruption} ({best_spearman_value:.5f})
    
    Best Pearson Correlation:
    {best_pearson_corruption} ({best_pearson_value:.5f})
    
    Most Consistent (High Positive %):
    {most_consistent_corruption} ({max(pos_percentages):.1f}%)
    
    🎯 Spearman optimal for isotonic mapping
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    
    plt.suptitle(f'Multi-Corruption Stability Analysis: {args.model}-{args.layer}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_corruption_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary data with NaN handling
    summary_data = {
        'model': args.model,
        'layer': args.layer,
        'seed': args.seed,
        'training_method': args.training_method,
        'corruptions_analyzed': corruptions,
        'overall_statistics': {
            'mean_stability_across_corruptions': float(overall_mean_stability),
            'mean_positive_percentage': float(overall_mean_positive),
            'mean_spearman_correlation': float(overall_mean_spearman),
            'mean_pearson_correlation': float(overall_mean_pearson),
            # Backward compatibility
            'mean_correlation': float(overall_mean_spearman),
            'best_stability_corruption': best_stability_corruption,
            'best_spearman_correlation_corruption': best_spearman_corruption,
            'best_pearson_correlation_corruption': best_pearson_corruption,
            # Backward compatibility
            'best_correlation_corruption': best_spearman_corruption,
            'most_consistent_corruption': most_consistent_corruption
        },
        'per_corruption_results': all_corruption_results
    }
    
    with open(output_dir / 'multi_corruption_summary.json', 'w') as f:
        json.dump(convert_numpy_types_with_precision(summary_data), f, indent=2)
    
    print(f"✅ Multi-corruption summary saved to {output_dir}/multi_corruption_summary.png")
    print(f"✅ Multi-corruption data saved to {output_dir}/multi_corruption_summary.json")

if __name__ == "__main__":
    main()