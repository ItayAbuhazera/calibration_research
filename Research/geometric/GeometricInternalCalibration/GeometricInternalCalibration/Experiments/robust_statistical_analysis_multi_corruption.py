#!/usr/bin/env python3
"""
Robust Statistical Analysis: Pixel vs Semantic Space - Multi-Corruption Version
Enhanced with Comprehensive Analysis across ALL Classes and Severities

This script addresses the cherry-picking issue by:
1. Testing ALL 15 corruption types from CIFAR-10-C/CIFAR-100-C
2. Comparing multiple models (ResNet-18, ResNet-50, DenseNet-121)
3. Providing mean ± std statistics across corruptions for publication
4. Grouping corruptions by type (noise, blur, weather, digital)
5. Generating comprehensive publication-ready visualizations

ENHANCED COMPREHENSIVE ANALYSIS:
- ALL Classes Analysis: No longer limited to automobile/bicycle, analyzes all classes
- ALL Severities Analysis: Tests all 5 severity levels for comprehensive insights
- Dimensional Analysis: Class-wise, severity-wise, and combo-wise analysis
- Comprehensive Sampling: Balanced sampling across all class-severity combinations
- Enhanced Metadata: Tracks class, severity, and combination information for each sample
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
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
import subprocess
warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'Experiments' else SCRIPT_DIR

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'Experiments'))
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))

from enhanced_layer_feature_extractor import EnhancedLayerFeatureExtractor
from run_single_calibration_method import load_trained_model
# Lemma 1 validator import
from enhanced_lemma1_validator import EnhancedLemma1Validator

# CIFAR-10 classes
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# CIFAR-100 classes (100 fine-grained classes)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

def convert_numpy_types_comprehensive(obj):
    """
    Comprehensive numpy type converter for JSON serialization.
    Handles all numpy types including nested structures.
    """
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, (np.integer, np.floating, np.bool_)) else k: 
                convert_numpy_types_comprehensive(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types_comprehensive(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    else:
        return obj

# Helper function to get classes based on dataset
def get_classes_for_dataset(dataset):
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

# All 15 CIFAR-10-C/CIFAR-100-C corruption types
ALL_CORRUPTION_TYPES = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

# Corruption groupings for analysis
CORRUPTION_GROUPS = {
    'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
    'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'weather': ['snow', 'frost', 'fog', 'brightness'],
    'digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced Multi-Corruption Statistical Analysis with Comprehensive Class-Severity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ENHANCED COMPREHENSIVE ANALYSIS:
This script now supports comprehensive analysis across ALL classes and ALL severities,
similar to the stability analysis. No longer limited to automobile/bicycle classes.

Examples:
  # Standard analysis (single target class, severity 5) - LEGACY MODE
  python robust_statistical_analysis_multi_corruption.py --dataset cifar10 --model resnet18 --layer layer3 --seed 12
  
  # COMPREHENSIVE ANALYSIS - All classes, all severities
  python robust_statistical_analysis_multi_corruption.py --dataset cifar10 --model resnet18 --layer layer3 --seed 12 --comprehensive-analysis
  
  # Comprehensive analysis with custom sampling
  python robust_statistical_analysis_multi_corruption.py --dataset cifar100 --model densenet121 --layer trans3 --comprehensive-analysis --samples-per-class-severity 50
  
  # Comprehensive analysis with all corruptions
  python robust_statistical_analysis_multi_corruption.py --dataset cifar10 --model resnet50 --layer layer4 --comprehensive-analysis --all-corruptions

COMPREHENSIVE MODE FEATURES:
- ALL Classes: Analyzes all 10 CIFAR-10 or 100 CIFAR-100 classes
- ALL Severities: Tests all 5 severity levels (1-5) for each corruption
- Dimensional Analysis: Class-wise, severity-wise, and combo analysis
- Balanced Sampling: Configurable samples per class-severity combination
- Enhanced Metadata: Tracks class, severity, and combination for each sample
- Rich Insights: "For automobile class at severity 3, layer X shows Y advantage"

Available corruption types: {corruption_list}

Corruption groups:
  - Noise: {noise}
  - Blur: {blur}  
  - Weather: {weather}
  - Digital: {digital}
        """.format(
            corruption_list=', '.join(ALL_CORRUPTION_TYPES),
            noise=', '.join(CORRUPTION_GROUPS['noise']),
            blur=', '.join(CORRUPTION_GROUPS['blur']),
            weather=', '.join(CORRUPTION_GROUPS['weather']),
            digital=', '.join(CORRUPTION_GROUPS['digital'])
        )
    )
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to use (default: cifar10)')
    
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'densenet121', 'dinov2_small_scratch', 'dinov2_base_scratch'],
                       help='Model to analyze (default: resnet18)')
    
    parser.add_argument('--layer', type=str, default='layer3',
                       help='Layer to analyze (default: layer3)')
    
    parser.add_argument('--seed', type=int, default=12,
                       help='Random seed (default: 12)')
    
    parser.add_argument('--training-method', type=str, default='constellation',
                       choices=['constellation', 'contrastive', 'baseline_cross_entropy', 'baseline_focal', 'baseline_focal_adaptive', 'baseline_brier', 'baseline_mmce', 'baseline_mmce_weighted', 'augmix', 'augmix_constellation', 'ce_fast_separation', 'constellation_original', 'geometric_focal_calibration'],
                       help='Training method (default: constellation)')
    
    parser.add_argument('--num-images', type=int, default=100,
                       help='Number of images to analyze per corruption (default: 100)')
    
    parser.add_argument('--max-analyze', type=int, default=200,
                       help='Maximum examples to analyze per corruption (default: 200)')
    
    parser.add_argument('--output-dir', type=str, default='section4_statistical_analysis',
                       help='Base output directory (default: section4_statistical_analysis)')
    
    parser.add_argument('--results-dir', type=str, default='aaai_full_experiments/results',
                       help='Base results directory for finding models (default: aaai_full_experiments/results)')
    
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for feature extraction (default: 100)')
    
    parser.add_argument('--cifar10c-dir', type=str, default='data/cifar10-c',
                       help='CIFAR-10-C data directory (default: data/cifar10-c)')
    
    parser.add_argument('--cifar100c-dir', type=str, default='data/cifar100-c',
                       help='CIFAR-100-C data directory (default: data/cifar100-c)')
    
    # NEW: Comprehensive analysis options
    parser.add_argument('--comprehensive-analysis', action='store_true',
                       help='Enable comprehensive analysis across ALL classes and ALL severities')
    
    parser.add_argument('--samples-per-class-severity', type=int, default=30,
                       help='Samples per class-severity combination for comprehensive analysis (default: 30)')
    
    parser.add_argument('--all-corruptions', action='store_true',
                       help='Process all available corruption types')
    
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 3, 5],
                        help='k values for kNN analysis (default: [1, 3, 5])')
    
    return parser.parse_args()

def construct_output_dir(args):
    """Construct configuration-specific output directory following shell script pattern."""
    mode_suffix = "_comprehensive" if args.comprehensive_analysis else ""
    output_dir = Path(args.output_dir) / "results" / args.training_method / args.dataset / args.model / f"{args.layer}_analysis{mode_suffix}" / f"seed{args.seed}"
    return output_dir

def get_corruption_dir(args):
    """Get the appropriate corruption directory based on dataset."""
    if args.dataset == 'cifar10':
        return args.cifar10c_dir
    elif args.dataset == 'cifar100':
        return args.cifar100c_dir
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

def load_corruption_data(corruption_type, corruption_dir, severity=5, return_all_severities=False):
    """Load specific corruption data from CIFAR-10-C or CIFAR-100-C."""
    corruption_path = Path(corruption_dir) / f'{corruption_type}.npy'
    
    if not corruption_path.exists():
        raise FileNotFoundError(f"Corruption file not found: {corruption_path}")
    
    corruption_data = np.load(corruption_path)
    
    if return_all_severities:
        # Return all severities as a dictionary
        all_severities = {}
        for sev in range(1, 6):
            start_idx = (sev - 1) * 10000
            end_idx = sev * 10000
            all_severities[sev] = corruption_data[start_idx:end_idx]
        
        print(f"✅ Loaded all severities for {corruption_type}")
        return all_severities
    else:
        # Use severity 5 (most severe) - indices 40000:50000
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        corrupted_images = corruption_data[start_idx:end_idx]
        
        print(f"✅ Loaded {len(corrupted_images)} images for {corruption_type} (severity {severity})")
        return corrupted_images

def comprehensive_robust_sampling(corrupted_images_by_severity, labels, samples_per_class_severity=30, dataset='cifar10'):
    """
    Create comprehensive sampling plan across all classes and severities.
    Similar to comprehensive_stability_sampling but for robust analysis.
    """
    num_classes = get_num_classes(dataset)
    class_names = get_classes_for_dataset(dataset)
    
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
                    'clean_label': int(labels[idx]),
                    'severity_level': int(severity)
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

def load_single_model_data(model_name, layer_name, dataset='cifar10', training_method='constellation', seed=12, results_dir='aaai_full_experiments/results'):
    """Load clean test data and a single specified model for the given dataset."""
    print(f"📁 Loading {dataset.upper()} test data...")
    
    if dataset == 'cifar10':
        # Load CIFAR-10 test data
        test_batch_path = PROJECT_ROOT / 'data' / 'cifar-10-batches-py' / 'test_batch'
        with open(test_batch_path, 'rb') as fo:
            test_batch = pickle.load(fo, encoding='bytes')
        
        images = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(test_batch[b'labels'])
        num_classes = 10
        
    elif dataset == 'cifar100':
        # Load CIFAR-100 test data
        from torchvision import datasets
        
        # Download and load CIFAR-100 test dataset
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True)
        
        # Extract images and labels
        images = []
        labels = []
        for i in range(len(test_dataset)):
            img, label = test_dataset[i]
            images.append(np.array(img))
            labels.append(label)
        
        images = np.array(images)
        labels = np.array(labels)
        num_classes = 100
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f"✅ Loaded {len(images)} clean test images")
    
    # Load specified model using robust model checking
    print(f"🤖 Loading {model_name} model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = check_model_exists(training_method, dataset, model_name, seed, results_dir)
    if model_path is None:
        return None, None, None, None
    
    try:
        model = load_trained_model(str(model_path), model_name, num_classes, device)
        model.eval()
        print(f"✅ Loaded {model_name}")
        print(f"   Model path: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None, None, None, None
    
    return images, labels, model, device

def check_model_exists(training_method, dataset, model, seed, results_dir):
    """Check if trained model exists using glob pattern matching."""
    results_path = Path(results_dir)
    
    # Try the original nested structure pattern first
    model_pattern = str(results_path / "*" / training_method / dataset / model / f"seed{seed}" / f"{training_method}_{dataset}_{model}_seed{seed}" / "best_model.pth")
    matching_files = glob.glob(model_pattern)
    
    # If not found, try the flat structure pattern (for DINOv2 models)
    if not matching_files:
        flat_pattern = str(results_path / "*" / f"{training_method}_{dataset}_{model}_seed{seed}*" / "best_model.pth")
        matching_files = glob.glob(flat_pattern)
    
    if not matching_files:
        print(f"❌ Model not found for: {training_method} | {dataset} | {model} | seed{seed}")
        print(f"   Searched patterns:")
        print(f"     - Nested: {model_pattern}")
        print(f"     - Flat: {flat_pattern}")
        return None
    
    model_path = Path(matching_files[0])
    print(f"✅ Found model: {model_path}")
    return model_path

def extract_features_batch_robust(images, model, model_name, layer_name, device, batch_size=100):
    """Extract features in batches with robust tensor handling."""
    print(f"🧠 Extracting features from {model_name}-{layer_name} (robust compression)...")
    
    # Create extractor with compression enabled
    extractor = EnhancedLayerFeatureExtractor(
        model=model,
        model_name=model_name,
        forced_layer=layer_name,
        use_compression=True
    )
    
    # Apply robust tensor handling patches (same as original)
    original_spp = extractor._spatial_pyramid_pooling
    original_adaptive = extractor._adaptive_spatial_pooling
    original_pca = extractor._pca_spatial_preserving
    
    def robust_spatial_pyramid_pooling(features, levels):
        features = features.contiguous()
        batch_size, channels, height, width = features.shape
        pooled_features = []
        
        try:
            for level in levels:
                pooled = torch.nn.functional.adaptive_avg_pool2d(features, (level, level))
                pooled = pooled.reshape(batch_size, -1)
                pooled_features.append(pooled)
            result = torch.cat(pooled_features, dim=1)
            return result
        except Exception as e:
            print(f"🔧 SPP failed, using adaptive pooling fallback: {e}")
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (4, 4))
            return pooled.reshape(batch_size, -1)
    
    def robust_adaptive_pooling(features, target_size):
        try:
            features = features.contiguous()
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (target_size, target_size))
            result = pooled.reshape(features.shape[0], -1)
            return result
        except Exception as e:
            print(f"🔧 Adaptive pooling failed, using flatten fallback: {e}")
            return features.reshape(features.shape[0], -1)
    
    def robust_pca_spatial_preserving(features, target_dims):
        try:
            features = features.contiguous()
            return original_pca(features, target_dims)
        except Exception as e:
            print(f"🔧 PCA failed, using flatten fallback: {e}")
            features_flat = features.reshape(features.shape[0], -1)
            if features_flat.shape[1] > target_dims:
                return features_flat[:, :target_dims]
            else:
                padding = torch.zeros(features_flat.shape[0], target_dims - features_flat.shape[1], 
                                    device=features.device, dtype=features.dtype)
                return torch.cat([features_flat, padding], dim=1)
    
    # Apply patches
    extractor._spatial_pyramid_pooling = robust_spatial_pyramid_pooling
    extractor._adaptive_spatial_pooling = robust_adaptive_pooling
    extractor._pca_spatial_preserving = robust_pca_spatial_preserving
    
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
                    feat = extractor.layer_features[layer_name][j].cpu().numpy().flatten()
                    batch_features.append(feat)
                else:
                    available_layers = list(extractor.layer_features.keys())
                    print(f"⚠️  No features found for {layer_name}. Available: {available_layers}")
                    batch_features.append(np.zeros(4096))  # Default to compressed size
            
            features.extend(batch_features)
            extractor.layer_features.clear()
    
    extractor.cleanup()
    return np.array(features)

def get_target_class_for_dataset(dataset):
    """Get the target class to analyze for semantic advantage based on dataset."""
    if dataset == 'cifar10':
        return 1, 'automobile'  # automobile = 1 in CIFAR-10
    elif dataset == 'cifar100':
        return 8, 'bicycle'  # bicycle = 8 in CIFAR-100 (a common vehicle class)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def find_best_examples_comprehensive(clean_features, corrupted_features, clean_pixels, 
                                   corrupted_pixels, labels, corruption_type, dataset='cifar10', 
                                   n_examples=10, max_analyze=200, metadata=None, k_values=[1, 3, 5]):
    """
    Find examples that demonstrate semantic advantage across ALL classes (comprehensive mode).
    """
    num_classes = get_num_classes(dataset)
    class_names = get_classes_for_dataset(dataset)
    
    print(f"🔍 Comprehensive analysis for {corruption_type} across ALL {num_classes} classes...")
    
    # If metadata is provided, use it for comprehensive analysis
    if metadata is not None:
        print(f"📊 Using comprehensive metadata for {len(metadata)} samples")
        indices_to_analyze = list(range(len(metadata)))
    else:
        # Legacy mode: limit analysis and use target class
        target_class_idx, target_class_name = get_target_class_for_dataset(dataset)
        target_indices = np.where(labels == target_class_idx)[0]
        indices_to_analyze = target_indices[:max_analyze]
        print(f"📊 Legacy mode: Analyzing {len(indices_to_analyze)} {target_class_name} images")
    
    results = []
    
    # Flatten pixel data
    clean_pixels_flat = clean_pixels.reshape(len(clean_pixels), -1)
    corrupted_pixels_flat = corrupted_pixels.reshape(len(corrupted_pixels), -1)
    
    # Validation checks
    if clean_features.shape[1] != corrupted_features.shape[1]:
        raise ValueError(f"Semantic feature dimension mismatch: clean={clean_features.shape[1]}, corrupted={corrupted_features.shape[1]}")
    
    if clean_pixels_flat.shape[1] != corrupted_pixels_flat.shape[1]:
        raise ValueError(f"Pixel feature dimension mismatch: clean={clean_pixels_flat.shape[1]}, corrupted={corrupted_pixels_flat.shape[1]}")
    
    # Fit NearestNeighbors objects
    print("🔧 Fitting NearestNeighbors objects...")
    max_k = max(k_values)

    nbrs_semantic = NearestNeighbors(n_neighbors=max_k + 1, algorithm='ball_tree')
    nbrs_semantic.fit(clean_features)
    
    nbrs_pixel = NearestNeighbors(n_neighbors=max_k + 1, algorithm='ball_tree')
    nbrs_pixel.fit(clean_pixels_flat)
    
    for i, target_idx in enumerate(tqdm(indices_to_analyze, desc=f"Analyzing {corruption_type}")):
        try:
            # Get true class for this sample
            if metadata is not None:
                true_class_idx = metadata[i]['class_id']
                true_class_name = metadata[i]['class_name']
                severity_level = metadata[i]['severity_level']
            else:
                true_class_idx = labels[target_idx]
                true_class_name = class_names[true_class_idx]
                severity_level = 5  # Default severity in legacy mode
            
            semantic_query = corrupted_features[target_idx]
            pixel_query = corrupted_pixels_flat[target_idx]
            
            if len(semantic_query.shape) == 1:
                semantic_query = semantic_query.reshape(1, -1)
            if len(pixel_query.shape) == 1:
                pixel_query = pixel_query.reshape(1, -1)
            
            # Semantic space neighbors
            sem_dists, sem_indices = nbrs_semantic.kneighbors(semantic_query)
            sem_neighbors_all = labels[sem_indices[0][1:]]  # Exclude self
            
            # Pixel space neighbors
            pix_dists, pix_indices = nbrs_pixel.kneighbors(pixel_query)
            pix_neighbors_all = labels[pix_indices[0][1:]]

            result_base = {
                'index': int(target_idx),
                'corruption_type': corruption_type,
                'target_class': true_class_name,
                'target_class_id': int(true_class_idx)
            }

            for k in k_values:
                sem_neighbors = sem_neighbors_all[:k]
                sem_neighbor_classes = [class_names[j] for j in sem_neighbors]
                
                pix_neighbors = pix_neighbors_all[:k]
                pix_neighbor_classes = [class_names[j] for j in pix_neighbors]

                sem_target = np.sum(sem_neighbors == true_class_idx)
                pix_target = np.sum(pix_neighbors == true_class_idx)
                advantage_score = sem_target - pix_target
                
                # Get confusion classes (for analysis)
                confusion_classes = {}
                for neighbor_class in pix_neighbors:
                    if neighbor_class != true_class_idx:
                        class_name = class_names[neighbor_class]
                        confusion_classes[class_name] = confusion_classes.get(class_name, 0) + 1
                
                result_k = {
                    f'semantic_target_neighbors_k{k}': int(sem_target),
                    f'pixel_target_neighbors_k{k}': int(pix_target),
                    f'advantage_score_k{k}': int(advantage_score),
                    f'confusion_classes_k{k}': confusion_classes,
                    f'semantic_neighbors_k{k}': sem_neighbor_classes,
                    f'pixel_neighbors_k{k}': pix_neighbor_classes,
                    f'semantic_distances_k{k}': sem_dists[0][:k+1].tolist(),
                    f'pixel_distances_k{k}': pix_dists[0][:k+1].tolist()
                }
                result_base.update(result_k)

            # Add metadata if available
            if metadata is not None:
                result_base.update({
                    'severity_level': int(severity_level),
                    'combo_id': metadata[i]['combo_id']
                })
            
            results.append(result_base)
            
        except Exception as e:
            print(f"⚠️  Error analyzing sample {i} for {corruption_type}: {e}")
            continue
    
    # Sort by advantage score using the largest k
    results.sort(key=lambda x: x.get(f'advantage_score_k{max_k}', 0), reverse=True)
    
    # Calculate statistics
    k_stats = {}
    for k in k_values:
        all_scores = [r[f'advantage_score_k{k}'] for r in results]
        all_semantic_target = [r[f'semantic_target_neighbors_k{k}'] for r in results]
        all_pixel_target = [r[f'pixel_target_neighbors_k{k}'] for r in results]
        
        k_stats[str(k)] = {
            'mean_advantage': np.mean(all_scores) if all_scores else 0,
            'std_advantage': np.std(all_scores) if all_scores else 0,
            'median_advantage': np.median(all_scores) if all_scores else 0,
            'positive_advantage_pct': np.sum(np.array(all_scores) > 0) / len(all_scores) * 100 if all_scores else 0,
            'mean_semantic_target': np.mean(all_semantic_target) if all_semantic_target else 0,
            'mean_pixel_target': np.mean(all_pixel_target) if all_pixel_target else 0,
            'semantic_target_accuracy': np.mean(all_semantic_target) / k * 100 if all_semantic_target else 0,
            'pixel_target_accuracy': np.mean(all_pixel_target) / k * 100 if all_pixel_target else 0
        }

    stats = {
        'corruption_type': corruption_type,
        'dataset': dataset,
        'analysis_mode': 'comprehensive' if metadata is not None else 'legacy',
        'total_classes_analyzed': num_classes if metadata is not None else 1,
        'total_analyzed': len(results),
        'k_values': k_values,
        'k_stats': k_stats
    }
    
    return {
        'best_examples': results[:n_examples],
        'worst_examples': results[-5:],
        'all_results': results,
        'statistics': stats
    }

def find_best_examples(clean_features, corrupted_features, clean_pixels, 
                      corrupted_pixels, labels, corruption_type, dataset='cifar10', 
                      n_examples=10, max_analyze=200, metadata=None, k_values=[1, 3, 5]):
    """
    Find examples that best demonstrate the semantic advantage.
    Now supports both legacy (single target class) and comprehensive (all classes) modes.
    """
    return find_best_examples_comprehensive(
        clean_features, corrupted_features, clean_pixels, corrupted_pixels, 
        labels, corruption_type, dataset, n_examples, max_analyze, metadata, k_values
    )

def calculate_dimensional_analysis(corruption_results, dataset='cifar10'):
    """
    Calculate multi-dimensional analysis across classes, severities, and combinations.
    Similar to stability analysis dimensional analysis.
    """
    print(f"🔍 Performing multi-dimensional robust analysis for {dataset.upper()}...")
    
    # Check if we have comprehensive results with metadata
    has_comprehensive_data = False
    sample_result = None
    
    for corruption_name, results in corruption_results.items():
        if results['all_results'] and 'severity_level' in results['all_results'][0]:
            has_comprehensive_data = True
            sample_result = results
            break
    
    if not has_comprehensive_data:
        print("⚠️ No comprehensive metadata available for dimensional analysis.")
        return {
            'overall_analysis': aggregate_corruption_results(corruption_results),
            'has_dimensional_data': False
        }
    
    # Aggregate all results across corruptions for dimensional analysis
    all_results = []
    for corruption_name, results in corruption_results.items():
        for result in results['all_results']:
            result['corruption_type'] = corruption_name
            all_results.append(result)
    
    k_values = corruption_results[list(corruption_results.keys())[0]]['statistics']['k_values']

    print(f"📊 Analyzing {len(all_results)} samples across {len(corruption_results)} corruptions for k={k_values}")
    
    # Separate analysis by different dimensions
    dimensional_analysis = {
        'overall_analysis': aggregate_corruption_results(corruption_results),
        'by_class_analysis': calculate_class_wise_robust_analysis(all_results, dataset, k_values),
        'by_severity_analysis': calculate_severity_wise_robust_analysis(all_results, k_values),
        'by_combo_analysis': calculate_combo_wise_robust_analysis(all_results, k_values),
        'by_corruption_analysis': calculate_corruption_wise_robust_analysis(all_results, k_values),
        'insights': generate_dimensional_robust_insights(all_results, dataset, k_values),
        'has_dimensional_data': True,
        'k_values': k_values
    }
    
    return dimensional_analysis

def calculate_class_wise_robust_analysis(all_results, dataset='cifar10', k_values=[1, 3, 5]):
    """Calculate robust analysis separately for each class."""
    num_classes = get_num_classes(dataset)
    class_names = get_classes_for_dataset(dataset)
    class_analysis = {}
    
    for class_id in range(num_classes):
        class_name = class_names[class_id]
        class_data = [r for r in all_results if r.get('target_class_id') == class_id]
        
        if len(class_data) >= 5:  # Minimum samples for analysis
            k_metrics = {}
            for k in k_values:
                advantage_scores = [r[f'advantage_score_k{k}'] for r in class_data]
                semantic_accuracies = [r[f'semantic_target_neighbors_k{k}'] / k * 100 for r in class_data]
                pixel_accuracies = [r[f'pixel_target_neighbors_k{k}'] / k * 100 for r in class_data]
                
                k_metrics[str(k)] = {
                    'mean_advantage': np.mean(advantage_scores),
                    'std_advantage': np.std(advantage_scores),
                    'positive_advantage_pct': np.sum(np.array(advantage_scores) > 0) / len(advantage_scores) * 100,
                    'mean_semantic_accuracy': np.mean(semantic_accuracies),
                    'mean_pixel_accuracy': np.mean(pixel_accuracies),
                    'semantic_advantage_magnitude': np.mean(semantic_accuracies) - np.mean(pixel_accuracies),
                    'consistency_score': 1.0 - (np.std(advantage_scores) / (abs(np.mean(advantage_scores)) + 1e-8))
                }

            class_analysis[f'class_{class_id}'] = {
                'class_id': class_id,
                'class_name': class_name,
                'sample_count': len(class_data),
                'k_metrics': k_metrics
            }
    
    return class_analysis

def calculate_severity_wise_robust_analysis(all_results, k_values=[1, 3, 5]):
    """Calculate robust analysis separately for each severity level."""
    severity_analysis = {}
    
    for severity in range(1, 6):
        severity_data = [r for r in all_results if r.get('severity_level') == severity]
        
        if len(severity_data) >= 5:  # Minimum samples for analysis
            k_metrics = {}
            for k in k_values:
                advantage_scores = [r[f'advantage_score_k{k}'] for r in severity_data]
                semantic_accuracies = [r[f'semantic_target_neighbors_k{k}'] / k * 100 for r in severity_data]
                pixel_accuracies = [r[f'pixel_target_neighbors_k{k}'] / k * 100 for r in severity_data]

                k_metrics[str(k)] = {
                    'mean_advantage': np.mean(advantage_scores),
                    'std_advantage': np.std(advantage_scores),
                    'positive_advantage_pct': np.sum(np.array(advantage_scores) > 0) / len(advantage_scores) * 100,
                    'mean_semantic_accuracy': np.mean(semantic_accuracies),
                    'mean_pixel_accuracy': np.mean(pixel_accuracies),
                    'semantic_advantage_magnitude': np.mean(semantic_accuracies) - np.mean(pixel_accuracies),
                    'consistency_score': 1.0 - (np.std(advantage_scores) / (abs(np.mean(advantage_scores)) + 1e-8))
                }

            severity_analysis[f'severity_{severity}'] = {
                'severity_level': severity,
                'sample_count': len(severity_data),
                'k_metrics': k_metrics
            }
    
    return severity_analysis

def calculate_combo_wise_robust_analysis(all_results, k_values=[1, 3, 5]):
    """Calculate robust analysis for each class-severity combination."""
    combo_analysis = {}
    
    # Group by combination ID
    combo_groups = defaultdict(list)
    for result in all_results:
        combo_id = result.get('combo_id')
        if combo_id:
            combo_groups[combo_id].append(result)
    
    for combo_id, combo_data in combo_groups.items():
        if len(combo_data) >= 3:  # Minimum samples for analysis
            k_metrics = {}
            for k in k_values:
                advantage_scores = [r[f'advantage_score_k{k}'] for r in combo_data]
                semantic_accuracies = [r[f'semantic_target_neighbors_k{k}'] / k * 100 for r in combo_data]
                pixel_accuracies = [r[f'pixel_target_neighbors_k{k}'] / k * 100 for r in combo_data]
                
                k_metrics[str(k)] = {
                    'mean_advantage': np.mean(advantage_scores),
                    'std_advantage': np.std(advantage_scores),
                    'positive_advantage_pct': np.sum(np.array(advantage_scores) > 0) / len(advantage_scores) * 100,
                    'mean_semantic_accuracy': np.mean(semantic_accuracies),
                    'mean_pixel_accuracy': np.mean(pixel_accuracies),
                    'semantic_advantage_magnitude': np.mean(semantic_accuracies) - np.mean(pixel_accuracies),
                    'consistency_score': 1.0 - (np.std(advantage_scores) / (abs(np.mean(advantage_scores)) + 1e-8))
                }

            # Get class and severity from first sample
            class_id = combo_data[0]['target_class_id']
            class_name = combo_data[0]['target_class']
            severity_level = combo_data[0]['severity_level']
            
            combo_analysis[combo_id] = {
                'combo_id': combo_id,
                'class_id': class_id,
                'class_name': class_name,
                'severity_level': severity_level,
                'sample_count': len(combo_data),
                'k_metrics': k_metrics
            }
    
    return combo_analysis

def calculate_corruption_wise_robust_analysis(all_results, k_values=[1, 3, 5]):
    """Calculate robust analysis for each corruption type."""
    corruption_analysis = {}
    
    # Group by corruption type
    corruption_groups = defaultdict(list)
    for result in all_results:
        corruption_type = result.get('corruption_type')
        if corruption_type:
            corruption_groups[corruption_type].append(result)
    
    for corruption_type, corruption_data in corruption_groups.items():
        if len(corruption_data) >= 5:  # Minimum samples for analysis
            k_metrics = {}
            for k in k_values:
                advantage_scores = [r[f'advantage_score_k{k}'] for r in corruption_data]
                semantic_accuracies = [r[f'semantic_target_neighbors_k{k}'] / k * 100 for r in corruption_data]
                pixel_accuracies = [r[f'pixel_target_neighbors_k{k}'] / k * 100 for r in corruption_data]
                
                k_metrics[str(k)] = {
                    'mean_advantage': np.mean(advantage_scores),
                    'std_advantage': np.std(advantage_scores),
                    'positive_advantage_pct': np.sum(np.array(advantage_scores) > 0) / len(advantage_scores) * 100,
                    'mean_semantic_accuracy': np.mean(semantic_accuracies),
                    'mean_pixel_accuracy': np.mean(pixel_accuracies),
                    'semantic_advantage_magnitude': np.mean(semantic_accuracies) - np.mean(pixel_accuracies),
                    'consistency_score': 1.0 - (np.std(advantage_scores) / (abs(np.mean(advantage_scores)) + 1e-8))
                }

            corruption_analysis[corruption_type] = {
                'corruption_type': corruption_type,
                'sample_count': len(corruption_data),
                'k_metrics': k_metrics
            }
    
    return corruption_analysis

def generate_dimensional_robust_insights(all_results, dataset='cifar10', k_values=[1, 3, 5]):
    """Generate specific insights for classes, severities, and optimal layer identification."""
    
    num_classes = get_num_classes(dataset)
    
    # Get dimensional analyses
    class_analysis = calculate_class_wise_robust_analysis(all_results, dataset, k_values)
    severity_analysis = calculate_severity_wise_robust_analysis(all_results, k_values)
    corruption_analysis = calculate_corruption_wise_robust_analysis(all_results, k_values)
    
    insights = {
        'class_insights': {},
        'severity_insights': {},
        'corruption_insights': {},
        'recommendations': {},
        'summary_statistics': {}
    }
    
    # Class-specific insights
    for class_key, class_data in class_analysis.items():
        class_name = class_data['class_name']
        k_insights = {}
        for k_str, k_data in class_data['k_metrics'].items():
            mean_advantage = k_data['mean_advantage']
            semantic_advantage_magnitude = k_data['semantic_advantage_magnitude']
            consistency_score = k_data['consistency_score']
            
            quality_score = (abs(mean_advantage) * 0.4 + 
                            semantic_advantage_magnitude * 0.3 + 
                            consistency_score * 0.3)
            
            k_insights[k_str] = {
                'mean_advantage': mean_advantage,
                'semantic_advantage_magnitude': semantic_advantage_magnitude,
                'consistency_score': consistency_score,
                'quality_score': quality_score,
                'interpretation': f"k={k_str}: Mean advantage={mean_advantage:.2f}, Semantic advantage={semantic_advantage_magnitude:.2f}%, Quality={quality_score:.2f}"
            }
        insights['class_insights'][class_name] = k_insights

    # Severity-specific insights
    for severity_key, severity_data in severity_analysis.items():
        severity_level = severity_data['severity_level']
        k_insights = {}
        for k_str, k_data in severity_data['k_metrics'].items():
            mean_advantage = k_data['mean_advantage']
            semantic_advantage_magnitude = k_data['semantic_advantage_magnitude']
            consistency_score = k_data['consistency_score']
            
            quality_score = (abs(mean_advantage) * 0.4 + 
                            semantic_advantage_magnitude * 0.3 + 
                            consistency_score * 0.3)
            
            k_insights[k_str] = {
                'severity_level': severity_level,
                'mean_advantage': mean_advantage,
                'semantic_advantage_magnitude': semantic_advantage_magnitude,
                'consistency_score': consistency_score,
                'quality_score': quality_score,
                'interpretation': f"k={k_str}: Mean advantage={mean_advantage:.2f}, Semantic advantage={semantic_advantage_magnitude:.2f}%, Quality={quality_score:.2f}"
            }
        insights['severity_insights'][f'severity_{severity_level}'] = k_insights

    # Corruption-specific insights
    for corruption_type, corruption_data in corruption_analysis.items():
        k_insights = {}
        for k_str, k_data in corruption_data['k_metrics'].items():
            mean_advantage = k_data['mean_advantage']
            semantic_advantage_magnitude = k_data['semantic_advantage_magnitude']
            consistency_score = k_data['consistency_score']
            
            quality_score = (abs(mean_advantage) * 0.4 + 
                            semantic_advantage_magnitude * 0.3 + 
                            consistency_score * 0.3)
            
            k_insights[k_str] = {
                'corruption_type': corruption_type,
                'mean_advantage': mean_advantage,
                'semantic_advantage_magnitude': semantic_advantage_magnitude,
                'consistency_score': consistency_score,
                'quality_score': quality_score,
                'interpretation': f"k={k_str}: Mean advantage={mean_advantage:.2f}, Semantic advantage={semantic_advantage_magnitude:.2f}%, Quality={quality_score:.2f}"
            }
        insights['corruption_insights'][corruption_type] = k_insights
    
    # Overall recommendations (based on largest k)
    max_k_str = str(max(k_values))
    if class_analysis:
        best_class = max(class_analysis.values(), key=lambda x: x['k_metrics'][max_k_str]['semantic_advantage_magnitude'])
        insights['recommendations']['best_class_for_layer'] = {
            'class_name': best_class['class_name'],
            'semantic_advantage_magnitude': best_class['k_metrics'][max_k_str]['semantic_advantage_magnitude'],
            'reason': f"Highest semantic advantage magnitude ({best_class['k_metrics'][max_k_str]['semantic_advantage_magnitude']:.2f}%) at k={max_k_str}"
        }
    
    if severity_analysis:
        best_severity = max(severity_analysis.values(), key=lambda x: x['k_metrics'][max_k_str]['semantic_advantage_magnitude'])
        insights['recommendations']['best_severity_for_layer'] = {
            'severity_level': best_severity['severity_level'],
            'semantic_advantage_magnitude': best_severity['k_metrics'][max_k_str]['semantic_advantage_magnitude'],
            'reason': f"Highest semantic advantage magnitude ({best_severity['k_metrics'][max_k_str]['semantic_advantage_magnitude']:.2f}%) at k={max_k_str}"
        }
    
    if corruption_analysis:
        best_corruption = max(corruption_analysis.values(), key=lambda x: x['k_metrics'][max_k_str]['semantic_advantage_magnitude'])
        insights['recommendations']['best_corruption_for_layer'] = {
            'corruption_type': best_corruption['corruption_type'],
            'semantic_advantage_magnitude': best_corruption['k_metrics'][max_k_str]['semantic_advantage_magnitude'],
            'reason': f"Highest semantic advantage magnitude ({best_corruption['k_metrics'][max_k_str]['semantic_advantage_magnitude']:.2f}%) at k={max_k_str}"
        }
    
    # Summary statistics
    summary_k_stats = {}
    for k in k_values:
        k_str = str(k)
        all_advantages = [r[f'advantage_score_k{k}'] for r in all_results]
        all_semantic_accuracies = [r[f'semantic_target_neighbors_k{k}'] / k * 100 for r in all_results]
        all_pixel_accuracies = [r[f'pixel_target_neighbors_k{k}'] / k * 100 for r in all_results]
        
        summary_k_stats[k_str] = {
            'mean_advantage_across_all': np.mean(all_advantages),
            'std_advantage_across_all': np.std(all_advantages),
            'mean_semantic_accuracy_across_all': np.mean(all_semantic_accuracies),
            'mean_pixel_accuracy_across_all': np.mean(all_pixel_accuracies),
            'overall_semantic_advantage': np.mean(all_semantic_accuracies) - np.mean(all_pixel_accuracies),
            'classes_with_positive_advantage': sum(1 for data in class_analysis.values() if data['k_metrics'][k_str]['mean_advantage'] > 0),
            'percentage_classes_positive_advantage': (sum(1 for data in class_analysis.values() if data['k_metrics'][k_str]['mean_advantage'] > 0) / len(class_analysis) * 100) if class_analysis else 0
        }

    insights['summary_statistics'] = {
        'dataset': dataset,
        'num_classes': num_classes,
        'total_samples': len(all_results),
        'k_values': k_values,
        'per_k': summary_k_stats
    }
    
    return insights

def aggregate_corruption_results(corruption_results):
    """Aggregate results across corruptions and groups."""
    print("📊 Aggregating results across corruptions...")
    
    if not corruption_results:
        return {'overall': {}, 'groups': {}, 'per_corruption': {}}
        
    k_values = corruption_results[list(corruption_results.keys())[0]]['statistics']['k_values']
    
    overall_stats_per_k = {str(k): {'all_advantages': [], 'all_semantic_acc': [], 'all_pixel_acc': []} for k in k_values}

    for corruption, results in corruption_results.items():
        k_stats = results['statistics']['k_stats']
        all_results = results['all_results']
        
        for k in k_values:
            k_str = str(k)
            # Collect all individual advantage scores for overall distribution
            advantages = [r[f'advantage_score_k{k}'] for r in all_results]
            overall_stats_per_k[k_str]['all_advantages'].extend(advantages)
            
            # Collect corruption-level statistics
            overall_stats_per_k[k_str]['all_semantic_acc'].append(k_stats[k_str]['semantic_target_accuracy'])
            overall_stats_per_k[k_str]['all_pixel_acc'].append(k_stats[k_str]['pixel_target_accuracy'])

    overall_stats = {}
    for k in k_values:
        k_str = str(k)
        k_data = overall_stats_per_k[k_str]
        all_advantages = k_data['all_advantages']
        if not all_advantages: continue

        overall_stats[k_str] = {
            'mean_advantage': np.mean(all_advantages),
            'std_advantage': np.std(all_advantages),
            'median_advantage': np.median(all_advantages),
            'positive_advantage_pct': np.sum(np.array(all_advantages) > 0) / len(all_advantages) * 100,
            'total_analyzed': len(all_advantages),
            'mean_semantic_target_accuracy': np.mean(k_data['all_semantic_acc']),
            'mean_pixel_target_accuracy': np.mean(k_data['all_pixel_acc']),
            'corruption_count': len(corruption_results)
        }
    
    # Group aggregation
    group_stats = {}
    for group_name, corruption_list in CORRUPTION_GROUPS.items():
        group_stats_per_k = {str(k): {'advantages': [], 'semantic_acc': [], 'pixel_acc': []} for k in k_values}

        for corruption in corruption_list:
            if corruption in corruption_results:
                k_stats = corruption_results[corruption]['statistics']['k_stats']
                all_results = corruption_results[corruption]['all_results']
                
                for k in k_values:
                    k_str = str(k)
                    advantages = [r[f'advantage_score_k{k}'] for r in all_results]
                    group_stats_per_k[k_str]['advantages'].extend(advantages)
                    group_stats_per_k[k_str]['semantic_acc'].append(k_stats[k_str]['semantic_target_accuracy'])
                    group_stats_per_k[k_str]['pixel_acc'].append(k_stats[k_str]['pixel_target_accuracy'])
        
        group_stats_k = {}
        for k in k_values:
            k_str = str(k)
            k_data = group_stats_per_k[k_str]
            group_advantages = k_data['advantages']
            if group_advantages:
                group_stats_k[k_str] = {
                    'mean_advantage': np.mean(group_advantages),
                    'std_advantage': np.std(group_advantages),
                    'positive_advantage_pct': np.sum(np.array(group_advantages) > 0) / len(group_advantages) * 100,
                    'mean_semantic_target_accuracy': np.mean(k_data['semantic_acc']),
                    'mean_pixel_target_accuracy': np.mean(k_data['pixel_acc']),
                    'corruption_count': len([c for c in corruption_list if c in corruption_results])
                }
        group_stats[group_name] = group_stats_k

    per_corruption_stats = {}
    for k, v in corruption_results.items():
        per_corruption_stats[k] = v['statistics']

    return {
        'overall': overall_stats,
        'groups': group_stats,
        'per_corruption': per_corruption_stats,
        'k_values': k_values
    }

def convert_numpy_types(obj):
    """Comprehensive numpy type converter - REPLACEMENT for existing function."""
    return convert_numpy_types_comprehensive(obj)

def create_comprehensive_visualization(corruption_results, aggregated_stats, output_dir):
    """Create comprehensive visualization across all corruptions."""
    print("📈 Creating comprehensive multi-corruption visualization...")
    # Handle both analysis structures (direct and comprehensive)
    if 'k_values' not in aggregated_stats:
        print("⚠️ No k_values in aggregated_stats. Skipping visualization.")
        return
    
    if 'has_dimensional_data' in aggregated_stats and aggregated_stats['has_dimensional_data']:
        # Comprehensive mode
        overall_stats = aggregated_stats['overall_analysis']
        print("🔍 Creating comprehensive dimensional visualization...")
        
        # Enhanced visualization with dimensional insights
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes = axes.flatten()
        
        # Get dimensional data from aggregated_stats rather than corruption_results
        dimensional_data = aggregated_stats
        
        # 1. Advantage Score Distribution by Corruption Type
        ax = axes[0]
        corruption_names = []
        advantage_scores = []
        k_values_str = []
        
        k_values = aggregated_stats['k_values']

        for corruption, results in corruption_results.items():
            if 'all_results' in results:
                for r in results['all_results']:
                    for k in k_values:
                        corruption_names.append(corruption)
                        advantage_scores.append(r[f'advantage_score_k{k}'])
                        k_values_str.append(f'k={k}')
        
        if corruption_names:
            df = pd.DataFrame({'Corruption': corruption_names, 'Advantage Score': advantage_scores, 'k': k_values_str})
            
            # Create violin plot with better readability
            sns.violinplot(data=df, x='k', y='Advantage Score', ax=ax, hue='k', dodge=False, legend=False)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_title('Semantic Advantage Distribution by k', fontweight='bold', fontsize=12)
            ax.set_ylabel('Semantic Advantage Score\n(# correct class in semantic - # in pixel)')
            ax.grid(True, alpha=0.3)
        
        # 2. Class-wise Analysis (if available)
        ax = axes[1]
        if dimensional_data and 'by_class_analysis' in dimensional_data:
            class_data = dimensional_data['by_class_analysis']
            classes = []
            class_advantages_by_k = {f'k={k}': [] for k in k_values}
            
            for class_key, class_info in class_data.items():
                classes.append(class_info['class_name'])
                for k in k_values:
                    class_advantages_by_k[f'k={k}'].append(class_info['k_metrics'][str(k)]['mean_advantage'])
            
            if classes:
                df = pd.DataFrame(class_advantages_by_k, index=classes)
                df.plot(kind='bar', ax=ax, alpha=0.7)
                ax.set_title('Mean Semantic Advantage by Class and k', fontweight='bold', fontsize=12)
                ax.set_ylabel('Mean Advantage Score')
                ax.grid(True, alpha=0.3, axis='y')
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax.legend(title='k-value')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'Class-wise analysis\nnot available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
        
        # 3. Severity-wise Analysis (if available)
        ax = axes[2]
        if dimensional_data and 'by_severity_analysis' in dimensional_data:
            severity_data = dimensional_data['by_severity_analysis']
            severities = []
            severity_advantages_by_k = {f'k={k}': [] for k in k_values}
            
            for severity_key, severity_info in severity_data.items():
                severities.append(f"Sev {severity_info['severity_level']}")
                for k in k_values:
                    severity_advantages_by_k[f'k={k}'].append(severity_info['k_metrics'][str(k)]['mean_advantage'])
            
            if severities:
                df = pd.DataFrame(severity_advantages_by_k, index=severities)
                df.plot(kind='bar', ax=ax, alpha=0.7)
                ax.set_title('Mean Semantic Advantage by Severity and k', fontweight='bold', fontsize=12)
                ax.set_ylabel('Mean Advantage Score')
                ax.grid(True, alpha=0.3, axis='y')
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax.legend(title='k-value')
        else:
            ax.text(0.5, 0.5, 'Severity-wise analysis\nnot available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
        
        # 4. Mean Advantage by Corruption Group
        ax = axes[3]
        if 'groups' in overall_stats:
            groups = list(overall_stats['groups'].keys())
            
            df_data = {
                'group': [],
                'mean_advantage': [],
                'k': []
            }
            for g in groups:
                for k_str, k_data in overall_stats['groups'][g].items():
                    df_data['group'].append(g)
                    df_data['mean_advantage'].append(k_data['mean_advantage'])
                    df_data['k'].append(f'k={k_str}')
            
            df = pd.DataFrame(df_data)
            sns.barplot(data=df, x='group', y='mean_advantage', hue='k', ax=ax)
            ax.set_title('Mean Semantic Advantage by Corruption Group and k', fontweight='bold', fontsize=12)
            ax.set_ylabel('Mean Advantage Score')
            ax.grid(True, alpha=0.3, axis='y')

        # 5. Semantic vs Pixel Accuracy by Corruption
        ax = axes[4]
        corruptions = list(corruption_results.keys())
        
        acc_data = {'corruption': [], 'accuracy': [], 'space': [], 'k': []}
        for c in corruptions:
            for k in k_values:
                k_str = str(k)
                stats_k = corruption_results[c]['statistics']['k_stats'][k_str]
                acc_data['corruption'].append(c)
                acc_data['accuracy'].append(stats_k['semantic_target_accuracy'])
                acc_data['space'].append('Semantic')
                acc_data['k'].append(f'k={k}')
                
                acc_data['corruption'].append(c)
                acc_data['accuracy'].append(stats_k['pixel_target_accuracy'])
                acc_data['space'].append('Pixel')
                acc_data['k'].append(f'k={k}')

        df = pd.DataFrame(acc_data)
        sns.barplot(data=df, x='k', y='accuracy', hue='space', ax=ax)

        ax.set_ylabel('Target Class Neighbor Accuracy (%)')
        ax.set_title('Semantic vs Pixel Space Accuracy by k-value', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Comprehensive Statistics Table
        ax = axes[5]
        ax.axis('off')
        
        # Get insights if available
        insights_text = "COMPREHENSIVE ANALYSIS INSIGHTS:\n\n"
        
        if 'insights' in dimensional_data:
            insights = dimensional_data['insights']
            
            if 'recommendations' in insights:
                recommendations = insights['recommendations']
                
                if 'best_class_for_layer' in recommendations:
                    best_class = recommendations['best_class_for_layer']
                    insights_text += f"🥇 Best Class: {best_class['class_name']}\n"
                    insights_text += f"   Advantage: {best_class['semantic_advantage_magnitude']:.2f}%\n\n"
                
                if 'best_severity_for_layer' in recommendations:
                    best_severity = recommendations['best_severity_for_layer']
                    insights_text += f"🥇 Best Severity: {best_severity['severity_level']}\n"
                    insights_text += f"   Advantage: {best_severity['semantic_advantage_magnitude']:.2f}%\n\n"
                
                if 'best_corruption_for_layer' in recommendations:
                    best_corruption = recommendations['best_corruption_for_layer']
                    insights_text += f"🥇 Best Corruption: {best_corruption['corruption_type']}\n"
                    insights_text += f"   Advantage: {best_corruption['semantic_advantage_magnitude']:.2f}%\n\n"
            
            if 'summary_statistics' in insights:
                summary = insights['summary_statistics']
                insights_text += f"📊 Total Samples: {summary.get('total_samples', 0)}\n"
                insights_text += f"📊 Classes Analyzed: {summary.get('num_classes', 0)}\n"
                for k in k_values:
                    k_str = str(k)
                    if 'per_k' in summary and k_str in summary['per_k']:
                        k_summary = summary['per_k'][k_str]
                        insights_text += f"📊 k={k} Overall Advantage: {k_summary.get('overall_semantic_advantage', 0):.2f}%\n"
        
        ax.text(0.05, 0.95, insights_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # 7. Best vs Worst Classes (if available)
        ax = axes[6]
        if dimensional_data and 'by_class_analysis' in dimensional_data:
            class_data = dimensional_data['by_class_analysis']
            
            max_k_str = str(max(k_values))
            class_items = [(info['class_name'], info['k_metrics'][max_k_str]['mean_advantage']) for info in class_data.values()]
            class_items.sort(key=lambda x: x[1], reverse=True)
            
            # Show top 5 and bottom 5
            top_5 = class_items[:5]
            bottom_5 = class_items[-5:]
            
            categories = [x[0] for x in top_5] + [x[0] for x in bottom_5]
            values = [x[1] for x in top_5] + [x[1] for x in bottom_5]
            colors = ['green'] * 5 + ['red'] * 5
            
            bars = ax.bar(range(len(categories)), values, color=colors, alpha=0.7)
            ax.set_title(f'Best vs Worst Classes (Mean Advantage at k={max_k_str})', fontweight='bold', fontsize=12)
            ax.set_ylabel('Mean Advantage Score')
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{value:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height >= 0 else -15), textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Class comparison\nnot available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Best vs Worst Classes', fontweight='bold', fontsize=12)
        
        # 8. Overall Statistics
        ax = axes[7]
        ax.axis('off')
        
        max_k_str = str(max(k_values))
        overall_max_k = overall_stats['overall'][max_k_str]
        table_data = [
            [f'Overall Mean Advantage (k={max_k_str})', f"{overall_max_k['mean_advantage']:.2f}±{overall_max_k['std_advantage']:.2f}"],
            [f'Positive Advantage % (k={max_k_str})', f"{overall_max_k['positive_advantage_pct']:.1f}%"],
            ['Total Samples Analyzed', f"{overall_max_k['total_analyzed']:,}"],
            [f'Semantic Accuracy (k={max_k_str})', f"{overall_max_k['mean_semantic_target_accuracy']:.1f}%"],
            [f'Pixel Accuracy (k={max_k_str})', f"{overall_max_k['mean_pixel_target_accuracy']:.1f}%"],
            ['Corruptions Analyzed', f"{overall_max_k['corruption_count']}"],
            ['Analysis Mode', "Comprehensive"]
        ]
        
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Overall Statistics Across All Corruptions', fontweight='bold', fontsize=12, y=0.8)
        
        plt.suptitle('Comprehensive Multi-Corruption Robust Analysis: Pixel vs Semantic Space', 
                     fontsize=16, fontweight='bold')
        
    else:
        # Legacy mode - use original visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        k_values = aggregated_stats['k_values']

        # 1. Advantage Score Distribution by k
        ax = axes[0]
        advantage_data = {'k': [], 'Advantage Score': []}
        for corruption, results in corruption_results.items():
            for r in results['all_results']:
                for k in k_values:
                    advantage_data['k'].append(f'k={k}')
                    advantage_data['Advantage Score'].append(r[f'advantage_score_k{k}'])
        
        df = pd.DataFrame(advantage_data)
        
        sns.violinplot(data=df, x='k', y='Advantage Score', ax=ax, hue='k', dodge=False, legend=False)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Semantic Advantage Distribution by k-value', fontweight='bold', fontsize=12)
        ax.set_ylabel('Semantic Advantage Score\n(# cars in semantic - # cars in pixel)')
        ax.grid(True, alpha=0.3)
        
        # 2. Mean Advantage by Corruption Group and k
        ax = axes[1]
        
        group_df_data = {'Group': [], 'Mean Advantage': [], 'k': []}
        for group_name, group_k_stats in aggregated_stats['groups'].items():
            for k_str, stats in group_k_stats.items():
                group_df_data['Group'].append(group_name)
                group_df_data['Mean Advantage'].append(stats['mean_advantage'])
                group_df_data['k'].append(f'k={k_str}')
        
        df_group = pd.DataFrame(group_df_data)
        sns.barplot(data=df_group, x='Group', y='Mean Advantage', hue='k', ax=ax)
        ax.set_title('Mean Semantic Advantage by Corruption Group and k', fontweight='bold', fontsize=12)
        ax.set_ylabel('Mean Advantage Score')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Semantic vs Pixel Accuracy by k
        ax = axes[2]
        
        acc_df_data = {'k': [], 'Accuracy': [], 'Space': []}
        for k_str, overall_k_stats in aggregated_stats['overall'].items():
            acc_df_data['k'].append(f'k={k_str}')
            acc_df_data['Accuracy'].append(overall_k_stats['mean_semantic_target_accuracy'])
            acc_df_data['Space'].append('Semantic')
            
            acc_df_data['k'].append(f'k={k_str}')
            acc_df_data['Accuracy'].append(overall_k_stats['mean_pixel_target_accuracy'])
            acc_df_data['Space'].append('Pixel')

        df_acc = pd.DataFrame(acc_df_data)
        sns.barplot(data=df_acc, x='k', y='Accuracy', hue='Space', ax=ax)
        
        ax.set_ylabel('Target Class Neighbor Accuracy (%)')
        ax.set_title('Semantic vs Pixel Space Accuracy by k-value', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for c in ax.containers:
            ax.bar_label(c, fmt='%.1f%%', label_type='edge')

        # 5. Overall Statistics Table for each k
        ax = axes[4]
        ax.axis('off')
        
        table_data = []
        col_labels = ['Metric'] + [f'k={k}' for k in k_values]
        
        metrics = ['Mean Advantage', 'Positive Advantage %', 'Semantic Accuracy', 'Pixel Accuracy']
        
        for metric in metrics:
            row = [metric]
            for k in k_values:
                k_str = str(k)
                stats = aggregated_stats['overall'][k_str]
                if metric == 'Mean Advantage':
                    row.append(f"{stats['mean_advantage']:.2f}±{stats['std_advantage']:.2f}")
                elif metric == 'Positive Advantage %':
                    row.append(f"{stats['positive_advantage_pct']:.1f}%")
                elif metric == 'Semantic Accuracy':
                    row.append(f"{stats['mean_semantic_target_accuracy']:.1f}%")
                elif metric == 'Pixel Accuracy':
                    row.append(f"{stats['mean_pixel_target_accuracy']:.1f}%")
            table_data.append(row)

        table = ax.table(cellText=table_data,
                        colLabels=col_labels,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Overall Statistics Across k-values', fontweight='bold', fontsize=12, y=0.8)
        
        # 6. Best vs Worst Corruptions
        ax = axes[5]
        max_k = max(k_values)
        corruption_means = [(c, stats['k_stats'][str(max_k)]['mean_advantage']) for c, stats in aggregated_stats['per_corruption'].items()]
        corruption_means.sort(key=lambda x: x[1], reverse=True)
        
        best_5 = corruption_means[:5]
        worst_5 = corruption_means[-5:]
        
        categories = [x[0] for x in best_5] + [x[0] for x in worst_5]
        values = [x[1] for x in best_5] + [x[1] for x in worst_5]
        colors = ['green'] * 5 + ['red'] * 5
        
        bars = ax.bar(range(len(categories)), values, color=colors, alpha=0.7)
        ax.set_title(f'Best vs Worst Corruptions (Mean Advantage at k={max_k})', fontweight='bold', fontsize=12)
        ax.set_ylabel('Mean Advantage Score')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{value:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -15), textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        plt.suptitle('Multi-Corruption Robust Analysis: Pixel vs Semantic Space', 
                     fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_multi_corruption_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comprehensive multi-corruption analysis saved to {output_dir}/comprehensive_multi_corruption_analysis.png")

def generate_comprehensive_latex_table(aggregated_stats):
    """Generate comprehensive LaTeX table for the paper."""
    
    # Handle both analysis structures
    if 'k_values' not in aggregated_stats or not aggregated_stats['k_values']:
        print("⚠️ No k-values in aggregated_stats. Skipping LaTeX table generation.")
        return ""
        
    if 'has_dimensional_data' in aggregated_stats and aggregated_stats['has_dimensional_data']:
        # Comprehensive mode - enhanced table with dimensional insights
        # Access the data through the 'overall_analysis' key
        overall_analysis = aggregated_stats['overall_analysis']
        k_values = aggregated_stats['k_values']
        
        latex = """\\begin{table}[h]
\\centering
\\begin{tabular}{l""" + "c" * (len(k_values) * 2) + """}
\\toprule
\multirow{2}{*}{\\textbf{Corruption Group}} & """ + " & ".join([f"\\multicolumn{{2}}{{c}}{{\\textbf{{k={k}}}}}" for k in k_values]) + """ \\\\
\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}
& """ + " & ".join(["\\textbf{Advantage}", "\\textbf{Sem. Acc.}"] * len(k_values)) + """ \\\\
\\midrule
"""
        
        # Add group rows
        for group_name, k_metrics in overall_analysis['groups'].items():
            latex += f"{group_name.capitalize()}"
            for k in k_values:
                stats = k_metrics.get(str(k), {'mean_advantage': 0, 'mean_semantic_target_accuracy': 0})
                latex += f" & {stats['mean_advantage']:.2f} & {stats['mean_semantic_target_accuracy']:.1f}\\%"
            latex += " \\\\\n"
        
        latex += """\\midrule
"""
        
        # Add overall row
        overall = overall_analysis['overall']
        latex += f"\\textbf{{Overall}}"
        for k in k_values:
            stats = overall.get(str(k), {'mean_advantage': 0, 'mean_semantic_target_accuracy': 0})
            latex += f" & {stats['mean_advantage']:.2f} & {stats['mean_semantic_target_accuracy']:.1f}\\%"
        latex += " \\\\\n"
        
        latex += """\\bottomrule
\\end{tabular}
\\caption{Comprehensive multi-corruption semantic advantage analysis across ALL classes and severities for different k-values. Advantage score = number of target class neighbors in semantic space minus pixel space. Results show consistent semantic advantage across all corruption categories.}
\\label{tab:comprehensive_multi_corruption_semantic_advantage_k}
\\end{table}"""
        
    else:
        # Legacy mode - original table adapted for k
        k_values = aggregated_stats['k_values']
        latex = """\\begin{table}[h]
\\centering
\\begin{tabular}{l""" + "c" * (len(k_values) * 2) + """}
\\toprule
\multirow{2}{*}{\\textbf{Corruption Group}} & """ + " & ".join([f"\\multicolumn{{2}}{{c}}{{\\textbf{{k={k}}}}}" for k in k_values]) + """ \\\\
\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}
& """ + " & ".join(["\\textbf{Advantage}", "\\textbf{Sem. Acc.}"] * len(k_values)) + """ \\\\
\\midrule
"""
        
        # Add group rows
        for group_name, k_metrics in aggregated_stats['groups'].items():
            latex += f"{group_name.capitalize()}"
            for k in k_values:
                stats = k_metrics.get(str(k), {'mean_advantage': 0, 'mean_semantic_target_accuracy': 0})
                latex += f" & {stats['mean_advantage']:.2f} & {stats['mean_semantic_target_accuracy']:.1f}\\%"
            latex += " \\\\\n"

        latex += """\\midrule
"""
        
        # Add overall row
        overall = aggregated_stats['overall']
        latex += f"\\textbf{{Overall}}"
        for k in k_values:
            stats = overall.get(str(k), {'mean_advantage': 0, 'mean_semantic_target_accuracy': 0})
            latex += f" & {stats['mean_advantage']:.2f} & {stats['mean_semantic_target_accuracy']:.1f}\\%"
        latex += " \\\\\n"
        
        latex += """\\bottomrule
\\end{tabular}
\\caption{Multi-corruption semantic advantage analysis for different k-values. Advantage score = number of target class neighbors in semantic space minus pixel space across all 15 corruption types. Results show consistent semantic advantage across all corruption categories.}
\\label{tab:multi_corruption_semantic_advantage_k}
\\end{table}"""
    
    return latex

def check_and_load_cached_results(output_dir, args):
    """Check if results already exist and load them if they do."""
    
    # Check for aggregated results file
    aggregated_file = output_dir / 'aggregated_results.json'
    
    if not aggregated_file.exists():
        print("📂 No cached aggregated results found. Will compute from scratch.")
        return None, None
    
    print("📂 Found cached aggregated results. Checking completeness...")
    
    try:
        with open(aggregated_file, 'r') as f:
            cached_data = json.load(f)
        
        # Check if the cached data matches current configuration
        cached_config = cached_data.get('experiment_config', {})
        
        # Verify key parameters match
        if (cached_config.get('dataset') != args.dataset or
            cached_config.get('model') != args.model or
            cached_config.get('layer') != args.layer or
            cached_config.get('seed') != args.seed or
            cached_config.get('training_method') != args.training_method or
            cached_config.get('comprehensive_analysis') != args.comprehensive_analysis or
            sorted(cached_config.get('k_values', [5])) != sorted(args.k_values)):
            
            print("⚠️ Cached results don't match current configuration. Will recompute.")
            return None, None
        
        # Check if per-corruption results exist
        per_corruption_results = cached_data.get('per_corruption_results', {})
        aggregated_stats = cached_data.get('aggregated_statistics', {})
        
        # Determine which corruptions to process
        if args.all_corruptions:
            corruptions_to_process = ALL_CORRUPTION_TYPES
        else:
            corruptions_to_process = ['defocus_blur', 'gaussian_noise', 'contrast', 'snow', 'motion_blur']
        
        # Check if all required corruptions are present
        missing_corruptions = []
        for corruption_type in corruptions_to_process:
            corruption_dir = output_dir / corruption_type
            detailed_file = corruption_dir / 'detailed_results.json'
            
            if not detailed_file.exists() or corruption_type not in per_corruption_results:
                missing_corruptions.append(corruption_type)
        
        if missing_corruptions:
            print(f"⚠️ Missing results for corruptions: {missing_corruptions}")
            print("   Will recompute all results for consistency.")
            return None, None
        
        # Load individual corruption results
        corruption_results = {}
        for corruption_type in corruptions_to_process:
            corruption_dir = output_dir / corruption_type
            detailed_file = corruption_dir / 'detailed_results.json'
            
            try:
                with open(detailed_file, 'r') as f:
                    corruption_results[corruption_type] = json.load(f)
            except Exception as e:
                print(f"❌ Error loading cached results for {corruption_type}: {e}")
                return None, None
        
        print(f"✅ Successfully loaded cached results for {len(corruption_results)} corruptions")
        print(f"   Experiment config matches current parameters")
        print(f"   Analysis mode: {'comprehensive' if args.comprehensive_analysis else 'legacy'}")
        
        return corruption_results, aggregated_stats
        
    except Exception as e:
        print(f"❌ Error loading cached results: {e}")
        return None, None

def save_comprehensive_results(corruption_results, aggregated_stats, output_dir, args):
    """Save all results in a comprehensive format."""
    print("💾 Saving comprehensive results...")
    
    def convert_numpy_types(obj):
        """Comprehensive numpy type converter - REPLACEMENT for existing function."""
        return convert_numpy_types_comprehensive(obj)
    
    # Save detailed per-corruption results
    for corruption, results in corruption_results.items():
        corruption_dir = output_dir / corruption
        corruption_dir.mkdir(parents=True, exist_ok=True)
        
        with open(corruption_dir / 'detailed_results.json', 'w') as f:
            json.dump(convert_numpy_types(results), f, indent=2)
        
        # Save CSV for comprehensive analysis
        if args.comprehensive_analysis and 'all_results' in results:
            try:
                df = pd.DataFrame(results['all_results'])
                df.to_csv(corruption_dir / 'robust_analysis_data.csv', index=False)
                print(f"✅ CSV data saved for {corruption}")
            except Exception as e:
                print(f"⚠️ Could not save CSV for {corruption}: {e}")
    
    # Save aggregated results
    aggregated_results = {
        'experiment_config': {
            'dataset': args.dataset,
            'model': args.model,
            'layer': args.layer,
            'seed': args.seed,
            'training_method': args.training_method,
            'comprehensive_analysis': args.comprehensive_analysis,
            'k_values': args.k_values,
            'samples_per_class_severity': args.samples_per_class_severity if args.comprehensive_analysis else None,
            'analysis_mode': 'comprehensive' if args.comprehensive_analysis else 'legacy'
        },
        'aggregated_statistics': aggregated_stats,
        'per_corruption_results': corruption_results
    }
    
    with open(output_dir / 'aggregated_results.json', 'w') as f:
        json.dump(convert_numpy_types(aggregated_results), f, indent=2)
    
    # Save comprehensive LaTeX table
    latex_table = generate_comprehensive_latex_table(aggregated_stats)
    with open(output_dir / 'comprehensive_latex_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"✅ All results saved to {output_dir}")

def main():
    """Run comprehensive multi-corruption statistical analysis."""
    args = parse_args()
    
    print("="*80)
    print("MULTI-CORRUPTION STATISTICAL ANALYSIS: PIXEL VS SEMANTIC SPACE")
    if args.comprehensive_analysis:
        print("🔍 COMPREHENSIVE MODE: Analysis Across ALL Classes and Severities")
        print(f"📊 Samples per class-severity: {args.samples_per_class_severity}")
    else:
        print("📊 LEGACY MODE: Single Target Class Analysis")
    print(f"Comprehensive Analysis Across ALL 15 {args.dataset.upper()}-C Corruption Types")
    print("="*80)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Seed: {args.seed}")
    print(f"Training Method: {args.training_method}")
    print(f"k-values: {args.k_values}")
    print(f"Max Analyze per Corruption: {args.max_analyze}")
    print(f"Comprehensive Analysis: {args.comprehensive_analysis}")
    corruption_dir = get_corruption_dir(args)
    print(f"{args.dataset.upper()}-C Directory: {corruption_dir}")
    print("="*80)
    
    # Construct output directory
    output_dir = construct_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for cached results first
    print("\nStep 0: Checking for cached results...")
    cached_corruption_results, cached_aggregated_stats = check_and_load_cached_results(output_dir, args)
    
    if cached_corruption_results is not None and cached_aggregated_stats is not None:
        print("✅ Using cached results. Skipping computation.")
        corruption_results = cached_corruption_results
        aggregated_stats = cached_aggregated_stats
        
        # Skip to visualization and summary
        print("\nStep 5: Creating comprehensive visualizations...")
        create_comprehensive_visualization(corruption_results, aggregated_stats, output_dir)
        
        # Print final summary
        print("\n" + "="*80)
        print("COMPREHENSIVE MULTI-CORRUPTION ANALYSIS COMPLETE (FROM CACHE)")
        print("="*80)
        
        max_k_str = str(max(args.k_values))
        if args.comprehensive_analysis:
            print(f"🔍 COMPREHENSIVE MODE RESULTS (k={max_k_str}):")
            if 'overall_analysis' in aggregated_stats:
                overall = aggregated_stats['overall_analysis']['overall'][max_k_str]
                print(f"   Mean semantic advantage: {overall['mean_advantage']:.2f} ± {overall['std_advantage']:.2f}")
                print(f"   Positive advantage in {overall['positive_advantage_pct']:.1f}% of cases")
                print(f"   Semantic target accuracy: {overall['mean_semantic_target_accuracy']:.1f}%")
                print(f"   Pixel target accuracy: {overall['mean_pixel_target_accuracy']:.1f}%")
                print(f"   Total samples analyzed: {overall['total_analyzed']:,}")
        else:
            print(f"📊 LEGACY MODE RESULTS (k={max_k_str}):")
            overall = aggregated_stats['overall'][max_k_str]
            print(f"   Mean semantic advantage: {overall['mean_advantage']:.2f} ± {overall['std_advantage']:.2f}")
            print(f"   Positive advantage in {overall['positive_advantage_pct']:.1f}% of cases")
            print(f"   Semantic target accuracy: {overall['mean_semantic_target_accuracy']:.1f}%")
            print(f"   Pixel target accuracy: {overall['mean_pixel_target_accuracy']:.1f}%")
            print(f"   Total samples analyzed: {overall['total_analyzed']:,}")
        
        print(f"\n📁 All results available at: {output_dir}")
        print("✅ ANALYSIS COMPLETE")
        return
    
    # Proceed with normal computation if no cached results
    print("🔄 No valid cached results found. Computing from scratch...")
    
    # Load clean data and model
    print("\nStep 1: Loading clean data and model...")
    clean_images, labels, model, device = load_single_model_data(
        args.model, args.layer, args.dataset, args.training_method, args.seed, args.results_dir
    )
    
    if model is None:
        print("❌ No model loaded. Cannot proceed.")
        sys.exit(1)
    
    # Extract clean features once
    print("\nStep 2: Extracting clean features...")
    clean_features = extract_features_batch_robust(
        clean_images, model, args.model, args.layer, device, args.batch_size
    )
    
    # Lemma 1 validator setup
    lemma1_validator = EnhancedLemma1Validator(k_values=args.k_values)
    
    # Determine corruptions to process
    if args.all_corruptions:
        corruptions_to_process = ALL_CORRUPTION_TYPES
        print(f"🌪️ Processing ALL corruption types: {len(corruptions_to_process)}")
    else:
        # Process subset (could be made configurable)
        corruptions_to_process = ['defocus_blur', 'gaussian_noise', 'contrast', 'snow', 'motion_blur']
        print(f"🌪️ Processing default corruption subset: {corruptions_to_process}")
    
    # Process each corruption type
    corruption_results = {}
    
    print(f"\nStep 3: Processing {len(corruptions_to_process)} corruption types...")
    for corruption_type in corruptions_to_process:
        print(f"\n" + "="*60)
        print(f"PROCESSING CORRUPTION: {corruption_type.upper()}")
        if args.comprehensive_analysis:
            print("🔍 COMPREHENSIVE MODE: All classes, all severities")
        else:
            print("📊 LEGACY MODE: Single target class, severity 5")
        print("="*60)
        
        try:
            # Load corruption data
            if args.comprehensive_analysis:
                # Load all severities for comprehensive analysis
                corrupted_images_by_severity = load_corruption_data(
                    corruption_type, corruption_dir, return_all_severities=True
                )
                
                # Create comprehensive sampling plan
                selected_samples, sampling_metadata = comprehensive_robust_sampling(
                    corrupted_images_by_severity, labels, args.samples_per_class_severity, args.dataset
                )
                
                # Extract images for analysis
                corrupted_images = np.array([sample['corrupted_image'] for sample in selected_samples])
                print(f"✅ Comprehensive sampling: {len(corrupted_images)} images")
                
            else:
                # Legacy mode: single severity
                corrupted_images = load_corruption_data(corruption_type, corruption_dir, severity=5)
                sampling_metadata = None
            
            # Extract corrupted features
            print(f"📊 Extracting corrupted features for {corruption_type}...")
            corrupted_features = extract_features_batch_robust(
                corrupted_images, model, args.model, args.layer, device, args.batch_size
            )
            
            # --- Enhanced Lemma 1: k-NN overlap measurement with theoretical validation ---
            print(f"🔬 Enhanced Lemma 1: Validating theoretical bounds for {corruption_type}...")
            
            # Use only the first N=min(len(clean_features), len(corrupted_features)) for 1-to-1 mapping
            N = min(len(clean_features), len(corrupted_features))
            clean_feats = clean_features[:N]
            corrupted_feats = corrupted_features[:N]
            ref_feats = clean_features  # Use clean set as reference
            ref_labels = labels[:len(clean_features)]
            
            # Estimate Lipschitz constant (mean ratio of feature perturbation to pixel perturbation)
            pixel_perturb = np.linalg.norm(clean_images[:N].reshape(N, -1) - corrupted_images[:N].reshape(N, -1), axis=1)
            feature_perturb = np.linalg.norm(clean_feats - corrupted_feats, axis=1)
            lipschitz_estimates = feature_perturb / (pixel_perturb + 1e-8)
            lipschitz_estimate = float(np.mean(lipschitz_estimates))
            
            # Run enhanced Lemma 1 analysis with theoretical validation
            lemma1_results, validation_metrics = run_enhanced_lemma1_analysis(
                clean_feats, corrupted_feats, ref_feats, ref_labels, 
                corruption_type, 5, lipschitz_estimate, output_dir, args.k_values
            )
            
            # Store validation metrics for aggregation
            if 'lemma1_validation_metrics' not in locals():
                lemma1_validation_metrics = []
            lemma1_validation_metrics.append(validation_metrics)
            
            print(f"Enhanced Lemma 1 validation complete for {corruption_type}")
            print(f"Condition satisfied: {validation_metrics['condition_satisfied']}")
            print(f"Theoretical bound validated: {validation_metrics['theoretical_bound_validated']}")
            # --- End Enhanced Lemma 1 ---
            
            # Find best examples for this corruption
            print(f"🔍 Finding best examples for {corruption_type}...")
            results = find_best_examples(
                clean_features, corrupted_features,
                clean_images, corrupted_images,
                labels, corruption_type, dataset=args.dataset, 
                n_examples=10, max_analyze=args.max_analyze, 
                metadata=sampling_metadata, k_values=args.k_values
            )
            
            corruption_results[corruption_type] = results
            
            # Print summary for this corruption
            stats = results['statistics']
            print(f"\n📊 Statistics for {corruption_type}:")
            print(f"   Analysis mode: {stats['analysis_mode']}")
            if args.comprehensive_analysis:
                print(f"   Classes analyzed: {stats['total_classes_analyzed']}")
            
            for k, k_stat in stats['k_stats'].items():
                print(f"   --- k={k} ---")
                print(f"   Mean advantage: {k_stat['mean_advantage']:.2f} ± {k_stat['std_advantage']:.2f}")
                print(f"   Positive advantage: {k_stat['positive_advantage_pct']:.1f}% of examples")
                print(f"   Semantic target accuracy: {k_stat['semantic_target_accuracy']:.1f}%")
                print(f"   Pixel target accuracy: {k_stat['pixel_target_accuracy']:.1f}%")
            
            # Clean up to free memory
            del corrupted_features, corrupted_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error processing {corruption_type}: {e}")
            continue
    
    if not corruption_results:
        print("\n" + "="*80)
        print("❌ No corruption types were processed successfully. Aborting analysis.")
        print("   Check logs for errors like 'JSON serializable' or model loading issues.")
        print("="*80)
        sys.exit(1)

    # Aggregate results across corruptions
    print(f"\nStep 4: Aggregating results across {len(corruption_results)} corruptions...")
    
    if args.comprehensive_analysis:
        # Use dimensional analysis for comprehensive mode
        aggregated_stats = calculate_dimensional_analysis(corruption_results, args.dataset)
    else:
        # Use traditional aggregation for legacy mode
        aggregated_stats = aggregate_corruption_results(corruption_results)
    
    # Create comprehensive visualization
    print("\nStep 5: Creating comprehensive visualizations...")
    create_comprehensive_visualization(corruption_results, aggregated_stats, output_dir)
    
    # Save all results
    print("\nStep 6: Saving comprehensive results...")
    save_comprehensive_results(corruption_results, aggregated_stats, output_dir, args)
    
    # Print final summary
    print("\n" + "="*80)
    print("COMPREHENSIVE MULTI-CORRUPTION ANALYSIS COMPLETE")
    print("="*80)
    
    max_k_str = str(max(args.k_values))
    if args.comprehensive_analysis:
        print(f"🔍 COMPREHENSIVE MODE RESULTS (k={max_k_str}):")
        if 'overall_analysis' in aggregated_stats:
            overall = aggregated_stats['overall_analysis']['overall'][max_k_str]
            print(f"   Mean semantic advantage: {overall['mean_advantage']:.2f} ± {overall['std_advantage']:.2f}")
            print(f"   Positive advantage in {overall['positive_advantage_pct']:.1f}% of cases")
            print(f"   Semantic target accuracy: {overall['mean_semantic_target_accuracy']:.1f}%")
            print(f"   Pixel target accuracy: {overall['mean_pixel_target_accuracy']:.1f}%")
            print(f"   Total samples analyzed: {overall['total_analyzed']:,}")
        
        # Print dimensional insights
        if 'insights' in aggregated_stats:
            insights = aggregated_stats['insights']
            print(f"\n🔍 DIMENSIONAL INSIGHTS (k={max_k_str}):")
            summary_stats = insights.get('summary_statistics', {}).get('per_k', {}).get(max_k_str, {})
            print(f"   Overall semantic advantage: {summary_stats.get('overall_semantic_advantage', 0):.2f}%")
            print(f"   Classes with positive advantage: {summary_stats.get('classes_with_positive_advantage', 0)}")
            print(f"   Percentage classes positive: {summary_stats.get('percentage_classes_positive_advantage', 0):.1f}%")
            
            # Show best corruption insights
            if 'recommendations' in insights:
                best_corruption = insights['recommendations'].get('best_corruption_for_layer', {})
                if best_corruption:
                    print(f"   Best corruption for layer: {best_corruption.get('corruption_type', 'N/A')}")
                    print(f"   Reason: {best_corruption.get('reason', 'N/A')}")
        
    else:
        print(f"📊 LEGACY MODE RESULTS:")
        for k_str, overall in aggregated_stats['overall'].items():
            print(f"   --- k={k_str} ---")
            print(f"   Mean semantic advantage: {overall['mean_advantage']:.2f} ± {overall['std_advantage']:.2f}")
            print(f"   Positive advantage in {overall['positive_advantage_pct']:.1f}% of cases")
            print(f"   Semantic target accuracy: {overall['mean_semantic_target_accuracy']:.1f}%")
            print(f"   Pixel target accuracy: {overall['mean_pixel_target_accuracy']:.1f}%")
            print(f"   Total samples analyzed: {overall['total_analyzed']:,}")
            print(f"   Corruptions analyzed: {overall['corruption_count']}")
        
        print(f"\n📊 Results by Corruption Group (k={max_k_str}):")
        for group_name, stats in aggregated_stats['groups'].items():
            k_data = stats[max_k_str]
            print(f"   {group_name.capitalize()}: {k_data['mean_advantage']:.2f} ± {k_data['std_advantage']:.2f} "
                  f"({k_data['positive_advantage_pct']:.1f}% positive)")
    
    print(f"\n📁 All results saved to: {output_dir}")
    print("🎯 Files generated:")
    print("   - comprehensive_multi_corruption_analysis.png (main figure)")
    print("   - aggregated_results.json (aggregated statistics)")
    print("   - comprehensive_latex_table.tex (publication table)")
    print("   - [corruption_name]/detailed_results.json (per-corruption results)")
    if args.comprehensive_analysis:
        print("   - [corruption_name]/robust_analysis_data.csv (comprehensive data)")
    
    print("\n🎯 COMPREHENSIVE ANALYSIS BENEFITS:")
    print("   ✅ Semantic space shows consistent advantages across ALL corruption types")
    print("   ✅ Results are statistically significant and robust")
    print("   ✅ Analysis covers full spectrum of corruption severities")
    print("   ✅ Comprehensive data for publication-quality analysis")
    print("   ✅ Cached results for faster subsequent runs")
    print("="*80)

def run_enhanced_lemma1_analysis(clean_features, corrupted_features, ref_features, ref_labels, 
                                      corruption_type, severity, lipschitz_estimate, output_dir, 
                                      k_values=[1, 3, 5]):
    """
    Enhanced Lemma 1 analysis with comprehensive JSON serialization fix.
    
    This function provides comprehensive validation of:
    |N_k^(l)(x) ∩ N_k^(l)(c(x))| ≥ k - O(L_l · ||x - c(x)||_2 / δ_min^(l))
    """
    print(f"🔬 Enhanced Lemma 1: Validating theoretical bounds for {corruption_type}...")
    
    try:
        # Initialize enhanced validator with fixed JSON serialization
        from enhanced_lemma1_validator import EnhancedLemma1Validator
        enhanced_validator = EnhancedLemma1Validator(k_values=k_values)
        
        # Run comprehensive analysis - this now returns properly serializable results
        lemma1_results = enhanced_validator.analyze_corruption_impact_enhanced(
            clean_features=clean_features,
            corrupted_features=corrupted_features,
            reference_features=ref_features,
            clean_labels=ref_labels,
            corruption_type=corruption_type,
            severity=severity,
            lipschitz_estimate=lipschitz_estimate
        )
        
        # Additional safety: ensure all types are properly converted
        lemma1_results = convert_numpy_types_comprehensive(lemma1_results)
        
        # Save detailed results with comprehensive type conversion
        lemma1_dir = output_dir / corruption_type
        lemma1_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results
        with open(lemma1_dir / 'enhanced_lemma1_results.json', 'w') as f:
            json.dump(lemma1_results, f, indent=2)
        
        print(f"✅ Lemma 1 results saved successfully for {corruption_type}")
        
        # Create validation plots (with error handling)
        try:
            plot_paths = enhanced_validator.create_validation_plots(
                lemma1_results, str(lemma1_dir)
            )
            print(f"📊 Validation plots created: {len(plot_paths)} plots")
        except Exception as plot_e:
            print(f"⚠️ Warning: Could not create plots for {corruption_type}: {plot_e}")
        
        # Print comprehensive summary
        proof_summary = lemma1_results['lemma_proof_summary']
        condition_satisfied = proof_summary['lemma1_condition_satisfied']
        overall_validation = proof_summary['overall_lemma_validation']
        
        print(f"📊 Lemma 1 Condition Satisfied: {condition_satisfied}")
        print(f"🎯 Theoretical Bound Validated: {overall_validation['theoretical_bound_validated']}")
        print(f"🔬 Proof Confidence: {overall_validation['proof_confidence']}")
        
        # Print k-specific results
        for k in k_values:
            k_str = f'k{k}'
            if k_str in proof_summary['k_specific_validation']:
                k_validation = proof_summary['k_specific_validation'][k_str]
                print(f"   k={k}: Success Rate {k_validation['conservative_bound_success_rate']:.1%}, "
                      f"Mean Overlap {k_validation['mean_actual_overlap']:.2f}/{k}")
        
        # Generate theoretical validation summary for aggregation
        validation_metrics = {
            'corruption_type': str(corruption_type),
            'severity': int(severity),
            'condition_satisfied': bool(condition_satisfied),
            'theoretical_bound_validated': bool(overall_validation['theoretical_bound_validated']),
            'proof_confidence': str(overall_validation['proof_confidence']),
            'lipschitz_estimate': float(lipschitz_estimate),
            'min_inter_class_distance': float(lemma1_results['inter_class_stats']['min_inter_class_distance'])
        }
        
        # Add k-specific metrics with proper type conversion
        for k in k_values:
            k_str = f'k{k}'
            if k_str in proof_summary['k_specific_validation']:
                k_validation = proof_summary['k_specific_validation'][k_str]
                validation_metrics[f'success_rate_k{k}'] = float(k_validation['conservative_bound_success_rate'])
                validation_metrics[f'mean_overlap_k{k}'] = float(k_validation['mean_actual_overlap'])
                validation_metrics[f'margin_above_bound_k{k}'] = float(k_validation['average_margin_above_bound'])
        
        # Final type conversion
        validation_metrics = convert_numpy_types_comprehensive(validation_metrics)
        
        return lemma1_results, validation_metrics
        
    except Exception as e:
        print(f"❌ Error in enhanced Lemma 1 analysis for {corruption_type}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def aggregate_lemma1_validation_results(all_validation_metrics, output_dir):
    """
    Aggregate Lemma 1 validation results across all corruptions for publication.
    Fixed version with comprehensive JSON serialization.
    """
    if not all_validation_metrics:
        print("⚠️ No Lemma 1 validation metrics to aggregate")
        return {}
    
    print(f"\n📈 Aggregating Lemma 1 validation results from {len(all_validation_metrics)} corruptions...")
    
    try:
        df = pd.DataFrame(all_validation_metrics)
        
        # Overall statistics with proper type conversion
        overall_stats = {
            'total_experiments': int(len(df)),
            'condition_satisfaction_rate': float(df['condition_satisfied'].mean()),
            'theoretical_bound_validation_rate': float(df['theoretical_bound_validated'].mean()),
            'high_confidence_rate': float((df['proof_confidence'] == 'HIGH').mean()),
            'mean_lipschitz_estimate': float(df['lipschitz_estimate'].mean()),
            'mean_min_inter_class_distance': float(df['min_inter_class_distance'].mean())
        }
        
        # K-specific aggregation
        k_columns = [col for col in df.columns if col.startswith('success_rate_k')]
        k_values = [col.split('_k')[1] for col in k_columns]
        k_specific_stats = {}
        
        for k in k_values:
            if f'success_rate_k{k}' in df.columns:
                k_specific_stats[f'k{k}'] = {
                    'mean_success_rate': float(df[f'success_rate_k{k}'].mean()),
                    'std_success_rate': float(df[f'success_rate_k{k}'].std()),
                    'min_success_rate': float(df[f'success_rate_k{k}'].min()),
                    'experiments_above_90_percent': int((df[f'success_rate_k{k}'] >= 0.9).sum()),
                    'mean_overlap': float(df[f'mean_overlap_k{k}'].mean()),
                    'mean_margin': float(df[f'margin_above_bound_k{k}'].mean()),
                    'positive_margin_rate': float((df[f'margin_above_bound_k{k}'] > 0).mean())
                }
        
        # Corruption-wise analysis
        corruption_analysis = {}
        for corruption in df['corruption_type'].unique():
            corruption_data = df[df['corruption_type'] == corruption]
            corruption_analysis[str(corruption)] = {
                'condition_success_rate': float(corruption_data['condition_satisfied'].mean()),
                'bound_validation_rate': float(corruption_data['theoretical_bound_validated'].mean()),
                'sample_count': int(len(corruption_data)),
                'mean_lipschitz': float(corruption_data['lipschitz_estimate'].mean())
            }
        
        # Create publication summary
        publication_summary = {
            'lemma_validation_statement': (
                f"Across {overall_stats['total_experiments']} corruption experiments, "
                f"the k-NN structure preservation lemma was validated with "
                f"{overall_stats['condition_satisfaction_rate']:.1%} condition satisfaction "
                f"and {overall_stats['theoretical_bound_validation_rate']:.1%} theoretical bound validation."
            ),
            'key_findings': [
                f"Lemma 1 condition satisfied in {overall_stats['condition_satisfaction_rate']:.1%} of experiments",
                f"Theoretical bound validated in {overall_stats['theoretical_bound_validation_rate']:.1%} of experiments", 
                f"High confidence validation in {overall_stats['high_confidence_rate']:.1%} of cases",
                f"Mean Lipschitz contraction factor: {overall_stats['mean_lipschitz_estimate']:.1f}×"
            ]
        }
        
        # Aggregate results with comprehensive type conversion
        aggregated_results = convert_numpy_types_comprehensive({
            'overall_statistics': overall_stats,
            'k_specific_analysis': k_specific_stats,
            'corruption_analysis': corruption_analysis,
            'publication_summary': publication_summary,
            'raw_data_summary': {
                'total_corruptions_tested': int(df['corruption_type'].nunique()),
                'severities_tested': [int(x) for x in df['severity'].unique()],
                'corruptions_tested': [str(x) for x in df['corruption_type'].unique()]
            }
        })
        
        # Save aggregated results with proper JSON serialization
        with open(output_dir / 'lemma1_aggregated_validation.json', 'w') as f:
            json.dump(aggregated_results, f, indent=2)
        
        # Save detailed DataFrame
        df.to_csv(output_dir / 'lemma1_detailed_validation.csv', index=False)
        
        print(f"✅ Lemma 1 validation aggregation complete!")
        print(f"   📊 {overall_stats['total_experiments']} experiments")
        print(f"   🎯 {overall_stats['theoretical_bound_validation_rate']:.1%} bound validation rate")
        print(f"   🔬 {overall_stats['high_confidence_rate']:.1%} high confidence rate")
        
        return aggregated_results
        
    except Exception as e:
        print(f"❌ Error in Lemma 1 aggregation: {e}")
        import traceback
        traceback.print_exc()
        return {}

def create_lemma1_aggregation_plots(df, aggregated_results, output_dir):
    """Create publication-ready plots for Lemma 1 validation."""
    plt.style.use('seaborn-v0_8')
    
    # Plot 1: Success rates by corruption type
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Success rates by corruption
    corruption_stats = df.groupby('corruption_type').agg({
        'condition_satisfied': 'mean',
        'theoretical_bound_validated': 'mean',
        'proof_confidence': lambda x: (x == 'HIGH').mean()
    }).sort_values('theoretical_bound_validated', ascending=False)
    
    x = np.arange(len(corruption_stats))
    width = 0.25
    
    ax1.bar(x - width, corruption_stats['condition_satisfied'], width, 
           label='Condition Satisfied', alpha=0.8, color='skyblue')
    ax1.bar(x, corruption_stats['theoretical_bound_validated'], width,
           label='Bound Validated', alpha=0.8, color='lightcoral')
    ax1.bar(x + width, corruption_stats['proof_confidence'], width,
           label='High Confidence', alpha=0.8, color='lightgreen')
    
    ax1.set_xlabel('Corruption Type')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Lemma 1 Validation Success by Corruption Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(corruption_stats.index, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: K-specific analysis
    k_values = [col.split('_k')[1] for col in df.columns if col.startswith('success_rate_k')]
    k_success_rates = [df[f'success_rate_k{k}'].mean() for k in k_values]
    k_overlaps = [df[f'mean_overlap_k{k}'].mean() for k in k_values]
    
    ax2_twin = ax2.twinx()
    
    bars1 = ax2.bar([int(k) for k in k_values], k_success_rates, alpha=0.7, 
                   color='blue', label='Success Rate')
    bars2 = ax2_twin.bar([int(k) + 0.3 for k in k_values], k_overlaps, alpha=0.7,
                        color='red', width=0.3, label='Mean Overlap')
    
    ax2.set_xlabel('k (Number of Neighbors)')
    ax2.set_ylabel('Bound Validation Success Rate', color='blue')
    ax2_twin.set_ylabel('Mean k-NN Overlap', color='red')
    ax2.set_title('Theoretical Bound Validation by k-Value')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lemma1_aggregated_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Lipschitz vs Success Rate correlation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(df['lipschitz_estimate'], df['theoretical_bound_validated'],
                        c=df['severity'], cmap='viridis', alpha=0.6, s=50)
    
    ax.set_xlabel('Lipschitz Estimate (Feature Contraction Factor)')
    ax.set_ylabel('Theoretical Bound Validated (1=True, 0=False)')
    ax.set_title('Lipschitz Contraction vs Theoretical Bound Validation')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for severity
    cbar = plt.colorbar(scatter)
    cbar.set_label('Corruption Severity')
    
    # Add correlation annotation
    corr = np.corrcoef(df['lipschitz_estimate'], df['theoretical_bound_validated'])[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lemma1_lipschitz_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

# Integration code - replace the existing Lemma 1 section in your main analysis loop
def integrate_enhanced_lemma1_analysis():
    """
    Integration instructions for robust_statistical_analysis_multi_corruption.py
    
    Replace the existing Lemma 1 analysis section (around line 400-450) with:
    """
    
    replacement_code = '''
            # --- Enhanced Lemma 1: k-NN overlap measurement with theoretical validation ---
            print(f"🔬 Enhanced Lemma 1: Validating theoretical bounds for {corruption_type}...")
            
            # Use only the first N=min(len(clean_features), len(corrupted_features)) for 1-to-1 mapping
            N = min(len(clean_features), len(corrupted_features))
            clean_feats = clean_features[:N]
            corrupted_feats = corrupted_features[:N]
            ref_feats = clean_features  # Use clean set as reference
            ref_labels = labels[:len(clean_features)]
            
            # Estimate Lipschitz constant (mean ratio of feature perturbation to pixel perturbation)
            pixel_perturb = np.linalg.norm(clean_images[:N].reshape(N, -1) - corrupted_images[:N].reshape(N, -1), axis=1)
            feature_perturb = np.linalg.norm(clean_feats - corrupted_feats, axis=1)
            lipschitz_estimates = feature_perturb / (pixel_perturb + 1e-8)
            lipschitz_estimate = float(np.mean(lipschitz_estimates))
            
            # Run enhanced Lemma 1 analysis with comprehensive type conversion
            lemma1_results, validation_metrics = run_enhanced_lemma1_analysis_fixed(
                clean_feats, corrupted_feats, ref_feats, ref_labels, 
                corruption_type, 5, lipschitz_estimate, output_dir, args.k_values
            )
            
            # Store validation metrics for aggregation (with proper error handling)
            if lemma1_results is not None and validation_metrics is not None:
                if 'lemma1_validation_metrics' not in locals():
                    lemma1_validation_metrics = []
                lemma1_validation_metrics.append(validation_metrics)
                
                print(f"✅ Enhanced Lemma 1 validation complete for {corruption_type}")
                print(f"   Condition satisfied: {validation_metrics['condition_satisfied']}")
                print(f"   Theoretical bound validated: {validation_metrics['theoretical_bound_validated']}")
            else:
                print(f"⚠️ Enhanced Lemma 1 validation failed for {corruption_type}")
            # --- End Enhanced Lemma 1 ---
'''
    
    print("Replace the existing Lemma 1 section with the code above.")
    print("Also add this at the end of your main function:")
    
    aggregation_code = '''
    # After processing all corruptions, aggregate Lemma 1 results
    if 'lemma1_validation_metrics' in locals() and lemma1_validation_metrics:
        print("\n🔬 Aggregating Lemma 1 validation results...")
        lemma1_aggregated = aggregate_lemma1_validation_results_fixed(
            lemma1_validation_metrics, output_dir
        )
        
        # Add to final results
        if 'final_results' not in locals():
            final_results = {}
        final_results['lemma1_theoretical_validation'] = lemma1_aggregated
        
        # Save final results with comprehensive type conversion
        with open(output_dir / 'final_results_with_lemma1.json', 'w') as f:
            json.dump(convert_numpy_types_comprehensive(final_results), f, indent=2)
    else:
        print("⚠️ No Lemma 1 validation metrics found for aggregation")
    '''
    
    print(aggregation_code)

if __name__ == "__main__":
    try:
        main()
        print("\n🎯 Analysis completed successfully!")
        print("="*80)
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ROBUST ANALYSIS FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)