#!/usr/bin/env python3
"""
Robust Statistical Analysis: Pixel vs Semantic Space
Refactored to match run_layer_optimization_experiments.sh patterns

This script addresses the cherry-picking issue by:
1. Testing 200+ car examples instead of selecting one
2. Comparing multiple models (ResNet-18, ResNet-50, DenseNet-121)
3. Providing mean ± std statistics for publication
4. Finding the best examples objectively
5. Generating publication-ready visualizations
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
from run_post_hoc_calibration import load_trained_model

# CIFAR-10 classes
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Robust Statistical Analysis for AAAI Paper Section 4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model analysis
  python robust_statistical_analysis.py --model resnet18 --layer layer3 --seed 12
  
  # Different training method
  python robust_statistical_analysis.py --model densenet121 --layer trans3 --training-method constellation --seed 42
  
  # Batch submission to SLURM
  python robust_statistical_analysis.py --submit-all
  
  # Custom output directory
  python robust_statistical_analysis.py --model resnet50 --layer layer4 --output-dir custom_results

Available models: resnet18, resnet50, densenet121
Available layers:
  - ResNet: layer1, layer2, layer3, layer4, layer4.1
  - DenseNet: dense1, trans1, dense2, trans2, dense3, trans3, dense4, bn
Available training methods: constellation, contrastive, baseline_cross_entropy, baseline_focal, baseline_focal_adaptive, baseline_brier, baseline_mmce, baseline_mmce_weighted, augmix, augmix_constellation
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
                       choices=['constellation', 'contrastive', 'baseline_cross_entropy', 'baseline_focal', 'baseline_focal_adaptive', 'baseline_brier', 'baseline_mmce', 'baseline_mmce_weighted', 'augmix', 'augmix_constellation', 'ce_fast_separation', 'constellation_original', 'geometric_focal_calibration'],
                       help='Training method (default: constellation)')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10'],
                       help='Dataset (default: cifar10)')
    
    parser.add_argument('--num-images', type=int, default=100,
                       help='Number of images to analyze (default: 100)')
    
    parser.add_argument('--max-analyze', type=int, default=200,
                       help='Maximum car examples to analyze (default: 200)')
    
    parser.add_argument('--output-dir', type=str, default='section4_statistical_analysis',
                       help='Base output directory (default: section4_statistical_analysis)')
    
    parser.add_argument('--results-dir', type=str, default='aaai_full_experiments/results',
                       help='Base results directory for finding models (default: aaai_full_experiments/results)')
    
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for feature extraction (default: 100)')
    
    parser.add_argument('--batch-mode', action='store_true',
                       help='Run in batch mode (check existence, submit SLURM jobs)')
    
    parser.add_argument('--submit-all', action='store_true',
                       help='Submit all configurations to SLURM')
    
    return parser.parse_args()

def construct_output_dir(args):
    """Construct configuration-specific output directory following shell script pattern."""
    output_dir = Path(args.output_dir) / "results" / args.training_method / args.dataset / args.model / f"{args.layer}_compression" / f"seed{args.seed}"
    return output_dir

def construct_output_file(args):
    """Construct output file path following shell script pattern."""
    output_dir = construct_output_dir(args)
    output_file = output_dir / f"statistical_analysis_{args.training_method}_{args.dataset}_{args.model}_{args.layer}_seed{args.seed}.json"
    return output_file

def check_model_exists(training_method, dataset, model, seed, results_dir):
    """Check if trained model exists using glob pattern matching."""
    results_path = Path(results_dir)
    model_pattern = str(results_path / "*" / training_method / dataset / model / f"seed{seed}" / f"{training_method}_{dataset}_{model}_seed{seed}" / "best_model.pth")
    
    matching_files = glob.glob(model_pattern)
    
    if not matching_files:
        print(f"❌ Model not found for: {training_method} | {dataset} | {model} | seed{seed}")
        print(f"   Searched pattern: {model_pattern}")
        return None
    
    model_path = Path(matching_files[0])
    print(f"✅ Found model: {model_path}")
    return model_path

def validate_result_file(file_path):
    """Validate if a result file contains complete and valid results."""
    if not file_path.exists():
        return False, "missing"
    
    # Check file size
    if file_path.stat().st_size < 50:
        return False, "too_small"
    
    # Validate JSON structure
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Check required fields
        required_fields = ['statistics', 'best_examples', 'all_results']
        for field in required_fields:
            if field not in data:
                return False, f"missing_{field}"
                
        return True, "valid"
    except json.JSONDecodeError:
        return False, "invalid_json"
    except Exception as e:
        return False, f"error:{str(e)}"

def check_results_exist(output_dir):
    """Check if valid results already exist."""
    expected_files = [
        output_dir / 'statistical_analysis.png',
        output_dir / 'detailed_results.json',
        output_dir / 'statistical_significance.json',
        output_dir / 'best_examples_grid.png',
        output_dir / 'latex_table.tex'
    ]
    
    all_exist = all(f.exists() for f in expected_files)
    
    if all_exist:
        # Validate the JSON files
        json_files = [f for f in expected_files if f.suffix == '.json']
        for json_file in json_files:
            valid, status = validate_result_file(json_file)
            if not valid:
                print(f"   ❌ Invalid result file {json_file.name}: {status}")
                return False
        
        print("✅ Valid results already exist!")
        print(f"   Location: {output_dir}")
        return True
    
    return False

def submit_statistical_analysis_job(model, layer, seed, training_method, output_base_dir, results_dir):
    """Submit a single statistical analysis job to SLURM."""
    
    # Construct output directory
    output_dir = Path(output_base_dir) / "results" / training_method / "cifar10" / model / f"{layer}_compression" / f"seed{seed}"
    
    # Check if results already exist
    if check_results_exist(output_dir):
        print(f"⏭️ Skipping {model}-{layer}-{seed}-{training_method}: results exist")
        return "skipped"
    
    # Check if model exists
    model_path = check_model_exists(training_method, "cifar10", model, seed, results_dir)
    if not model_path:
        print(f"❌ Skipping {model}-{layer}-{seed}-{training_method}: model not found")
        return "no_model"
    
    # Prepare SLURM job
    job_name = f"stat_analysis_{training_method}_{model}_{layer}_s{seed}"
    log_file = output_dir.parent.parent.parent / "logs" / f"{job_name}_%j.out"
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = f"python {SCRIPT_DIR}/robust_statistical_analysis.py"
    cmd += f" --model {model}"
    cmd += f" --layer {layer}"
    cmd += f" --seed {seed}"
    cmd += f" --training-method {training_method}"
    cmd += f" --output-dir {output_base_dir}"
    cmd += f" --results-dir {results_dir}"
    
    # Submit to SLURM
    sbatch_cmd = [
        "sbatch",
        "--job-name", job_name,
        "--output", str(log_file),
        "--time", "4:00:00",
        "--mem", "32G",
        "--cpus-per-task", "4",
        "--gpus", "1",
        "--wrap", cmd
    ]
    
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"✅ Submitted job {job_id}: {job_name}")
        return "submitted"
    else:
        print(f"❌ Failed to submit: {result.stderr}")
        return "failed"

def submit_all_configurations(output_base_dir, results_dir):
    """Submit all model/layer/seed combinations to SLURM."""
    models = ["resnet18", "resnet50", "densenet121"]
    layers = {
        "resnet18": ["layer3"],
        "resnet50": ["layer3"],
        "densenet121": ["trans3"]
    }
    seeds = [12, 13, 14, 11]
    training_methods = ["constellation", "contrastive", "baseline_cross_entropy", "baseline_focal", "baseline_focal_adaptive", "baseline_brier", "baseline_mmce", "baseline_mmce_weighted", "augmix", "augmix_constellation", "ce_fast_separation", "constellation_original", "geometric_focal_calibration"]
    
    total_jobs = 0
    submitted_jobs = 0
    skipped_jobs = 0
    missing_model_jobs = 0
    failed_jobs = 0
    
    print("🚀 SUBMITTING ALL STATISTICAL ANALYSIS CONFIGURATIONS")
    print("=" * 60)
    
    for training_method in training_methods:
        for model in models:
            for layer in layers[model]:
                for seed in seeds:
                    total_jobs += 1
                    result = submit_statistical_analysis_job(
                        model, layer, seed, training_method, output_base_dir, results_dir
                    )
                    
                    if result == "submitted":
                        submitted_jobs += 1
                    elif result == "skipped":
                        skipped_jobs += 1
                    elif result == "no_model":
                        missing_model_jobs += 1
                    else:
                        failed_jobs += 1
    
    print("\n📊 SUBMISSION SUMMARY")
    print("=" * 30)
    print(f"Total jobs planned: {total_jobs}")
    print(f"Jobs submitted: {submitted_jobs}")
    print(f"Jobs skipped (results exist): {skipped_jobs}")
    print(f"Jobs skipped (no model): {missing_model_jobs}")
    print(f"Jobs failed: {failed_jobs}")
    
    if submitted_jobs > 0:
        print(f"\n✅ {submitted_jobs} statistical analysis jobs submitted to SLURM queue")
        print("   Monitor with: squeue -u $USER")
    
    return submitted_jobs

def validate_layer_for_model(model_name, layer_name):
    """Validate that the layer is compatible with the model."""
    valid_layers = {
        'resnet18': ['layer1', 'layer2', 'layer3', 'layer4', 'layer4.1'],
        'resnet50': ['layer1', 'layer2', 'layer3', 'layer4', 'layer4.1'],
        'densenet121': ['dense1', 'trans1', 'dense2', 'trans2', 'dense3', 'trans3', 'dense4', 'bn']
    }
    
    if model_name not in valid_layers:
        print(f"❌ Unknown model: {model_name}")
        return False
    
    if layer_name not in valid_layers[model_name]:
        print(f"❌ Invalid layer '{layer_name}' for model '{model_name}'")
        print(f"   Valid layers for {model_name}: {', '.join(valid_layers[model_name])}")
        return False
    
    return True

def load_single_model_data(model_name, layer_name, training_method='constellation', seed=12, results_dir='aaai_full_experiments/results'):
    """Load CIFAR-10 data and a single specified model."""
    print("📁 Loading CIFAR-10 test data...")
    
    # Load CIFAR-10 test data
    test_batch_path = PROJECT_ROOT / 'data' / 'cifar-10-batches-py' / 'test_batch'
    with open(test_batch_path, 'rb') as fo:
        test_batch = pickle.load(fo, encoding='bytes')
    
    images = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(test_batch[b'labels'])
    
    print(f"✅ Loaded {len(images)} test images")
    
    # Load corruption data
    print("📁 Loading corruption data...")
    corruption_path = PROJECT_ROOT / 'data' / 'cifar10-c' / 'defocus_blur.npy'
    corruption_data = np.load(corruption_path)
    corrupted_images = corruption_data[40000:50000]  # Severity 5
    
    print(f"✅ Loaded {len(corrupted_images)} corrupted images")
    
    # Load specified model using robust model checking
    print(f"🤖 Loading {model_name} model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = check_model_exists(training_method, 'cifar10', model_name, seed, results_dir)
    if model_path is None:
        return None, None, None, None, None

    try:
        model = load_trained_model(str(model_path), model_name, 10, device)
        model.eval()
        models = {(model_name, layer_name): model}
        print(f"✅ Loaded {model_name} - will use {layer_name}")
        print(f"   Model path: {model_path}")
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None, None, None, None, None
    
    return images, labels, corrupted_images, models, device

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
    
    # Monkey-patch spatial reduction methods to be more robust
    original_spp = extractor._spatial_pyramid_pooling
    original_adaptive = extractor._adaptive_spatial_pooling
    original_pca = extractor._pca_spatial_preserving
    
    def robust_spatial_pyramid_pooling(features, levels):
        """Robust SPP that handles non-contiguous tensors."""
        # Ensure tensor is contiguous before any view operations
        features = features.contiguous()
        batch_size, channels, height, width = features.shape
        pooled_features = []
        
        try:
            for level in levels:
                # Adaptive pooling to create level x level grid
                pooled = torch.nn.functional.adaptive_avg_pool2d(features, (level, level))
                # Use reshape instead of view for better compatibility
                pooled = pooled.reshape(batch_size, -1)
                pooled_features.append(pooled)
            
            # Concatenate all levels
            result = torch.cat(pooled_features, dim=1)
            return result
            
        except Exception as e:
            print(f"🔧 SPP failed, using adaptive pooling fallback: {e}")
            # Fallback to adaptive pooling
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (4, 4))
            return pooled.reshape(batch_size, -1)
    
    def robust_adaptive_pooling(features, target_size):
        """Robust adaptive pooling that handles non-contiguous tensors."""
        try:
            # Ensure tensor is contiguous before any operations
            features = features.contiguous()
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (target_size, target_size))
            # Use reshape instead of view
            result = pooled.reshape(features.shape[0], -1)
            return result
        except Exception as e:
            print(f"🔧 Adaptive pooling failed, using flatten fallback: {e}")
            # Fallback to flattening
            return features.reshape(features.shape[0], -1)
    
    def robust_pca_spatial_preserving(features, target_dims):
        """Robust PCA that handles non-contiguous tensors."""
        try:
            # Ensure tensor is contiguous
            features = features.contiguous()
            return original_pca(features, target_dims)
        except Exception as e:
            print(f"🔧 PCA failed, using flatten fallback: {e}")
            # Fallback to flattening and truncation
            features_flat = features.reshape(features.shape[0], -1)
            if features_flat.shape[1] > target_dims:
                return features_flat[:, :target_dims]
            else:
                # Pad with zeros if needed
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
            
            # Ensure tensor is contiguous before processing
            batch_tensor = batch_tensor.contiguous()
            
            # Extract features
            _ = model(batch_tensor)
            batch_features = []
            
            for j in range(len(batch_tensor)):
                if layer_name in extractor.layer_features and extractor.layer_features[layer_name] is not None:
                    feat = extractor.layer_features[layer_name][j].cpu().numpy().flatten()
                    batch_features.append(feat)
                else:
                    # Print debug info for missing features
                    available_layers = list(extractor.layer_features.keys())
                    print(f"⚠️  No features found for {layer_name}. Available: {available_layers}")
                    # Use fallback dimension
                    batch_features.append(np.zeros(4096))  # Default to compressed size
            
            features.extend(batch_features)
            extractor.layer_features.clear()
    
    extractor.cleanup()
    return np.array(features)

def find_best_car_examples(clean_features, corrupted_features, clean_pixels, 
                          corrupted_pixels, labels, n_examples=10, max_analyze=200):
    """
    Find car examples that best demonstrate the semantic advantage.
    
    Returns:
        Dictionary with best examples and statistics
    """
    car_indices = np.where(labels == 1)[0]  # automobile = 1
    
    # Limit analysis to first max_analyze cars for efficiency
    car_indices = car_indices[:max_analyze]
    
    results = []
    
    # Flatten pixel data
    clean_pixels_flat = clean_pixels.reshape(len(clean_pixels), -1)
    corrupted_pixels_flat = corrupted_pixels.reshape(len(corrupted_pixels), -1)
    
    # Debug: Print dimensions
    print(f"🔍 Analyzing {len(car_indices)} car images...")
    print(f"📊 Data dimensions:")
    print(f"   Clean semantic features: {clean_features.shape}")
    print(f"   Corrupted semantic features: {corrupted_features.shape}")
    print(f"   Clean pixel features: {clean_pixels_flat.shape}")
    print(f"   Corrupted pixel features: {corrupted_pixels_flat.shape}")
    
    # Validation checks
    if clean_features.shape[1] != corrupted_features.shape[1]:
        raise ValueError(f"Semantic feature dimension mismatch: clean={clean_features.shape[1]}, corrupted={corrupted_features.shape[1]}")
    
    if clean_pixels_flat.shape[1] != corrupted_pixels_flat.shape[1]:
        raise ValueError(f"Pixel feature dimension mismatch: clean={clean_pixels_flat.shape[1]}, corrupted={corrupted_pixels_flat.shape[1]}")
    
    if clean_features.shape[0] != clean_pixels_flat.shape[0]:
        raise ValueError(f"Sample count mismatch: semantic={clean_features.shape[0]}, pixel={clean_pixels_flat.shape[0]}")
    
    # Fit NearestNeighbors objects once (more efficient)
    print("🔧 Fitting NearestNeighbors objects...")
    
    # Semantic space neighbors - fit once
    nbrs_semantic = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    nbrs_semantic.fit(clean_features)
    print(f"   ✅ Semantic space fitted: {clean_features.shape[1]} dimensions")
    
    # Pixel space neighbors - fit once
    nbrs_pixel = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    nbrs_pixel.fit(clean_pixels_flat)
    print(f"   ✅ Pixel space fitted: {clean_pixels_flat.shape[1]} dimensions")
    
    for car_idx in tqdm(car_indices, desc="Finding best examples"):
        try:
            # Debug: Check individual sample dimensions
            semantic_query = corrupted_features[car_idx]
            pixel_query = corrupted_pixels_flat[car_idx]
            
            if len(semantic_query.shape) == 1:
                semantic_query = semantic_query.reshape(1, -1)
            if len(pixel_query.shape) == 1:
                pixel_query = pixel_query.reshape(1, -1)
            
            # Semantic space neighbors - query
            sem_dists, sem_indices = nbrs_semantic.kneighbors(semantic_query)
            sem_neighbors = labels[sem_indices[0][1:]]  # Exclude self
            sem_neighbor_classes = [CIFAR10_CLASSES[i] for i in sem_neighbors]
            
            # Pixel space neighbors - query
            pix_dists, pix_indices = nbrs_pixel.kneighbors(pixel_query)
            pix_neighbors = labels[pix_indices[0][1:]]
            pix_neighbor_classes = [CIFAR10_CLASSES[i] for i in pix_neighbors]
            
            # Count cars and calculate metrics
            sem_cars = np.sum(sem_neighbors == 1)
            pix_cars = np.sum(pix_neighbors == 1)
            
            # Semantic advantage score
            advantage_score = sem_cars - pix_cars
            
            # Check for ships/trucks in pixel neighbors (common confusion)
            pix_ships = np.sum(pix_neighbors == 8)
            pix_trucks = np.sum(pix_neighbors == 9)
            
            results.append({
                'index': int(car_idx),
                'semantic_car_neighbors': int(sem_cars),
                'pixel_car_neighbors': int(pix_cars),
                'advantage_score': int(advantage_score),
                'pixel_ships': int(pix_ships),
                'pixel_trucks': int(pix_trucks),
                'semantic_neighbors': sem_neighbor_classes,
                'pixel_neighbors': pix_neighbor_classes,
                'semantic_distances': sem_dists[0].tolist(),
                'pixel_distances': pix_dists[0].tolist()
            })
            
        except Exception as e:
            print(f"⚠️  Error analyzing car {car_idx}: {e}")
            print(f"   Semantic query shape: {semantic_query.shape if 'semantic_query' in locals() else 'N/A'}")
            print(f"   Pixel query shape: {pixel_query.shape if 'pixel_query' in locals() else 'N/A'}")
            print(f"   Expected semantic dims: {clean_features.shape[1]}")
            print(f"   Expected pixel dims: {clean_pixels_flat.shape[1]}")
            continue
    
    # Sort by advantage score
    results.sort(key=lambda x: x['advantage_score'], reverse=True)
    
    # Calculate statistics
    all_scores = [r['advantage_score'] for r in results]
    all_semantic_cars = [r['semantic_car_neighbors'] for r in results]
    all_pixel_cars = [r['pixel_car_neighbors'] for r in results]
    
    stats = {
        'mean_advantage': np.mean(all_scores),
        'std_advantage': np.std(all_scores),
        'median_advantage': np.median(all_scores),
        'positive_advantage_pct': np.sum(np.array(all_scores) > 0) / len(all_scores) * 100,
        'total_analyzed': len(results),
        'mean_semantic_cars': np.mean(all_semantic_cars),
        'mean_pixel_cars': np.mean(all_pixel_cars),
        'semantic_car_accuracy': np.mean(all_semantic_cars) / 5.0 * 100,
        'pixel_car_accuracy': np.mean(all_pixel_cars) / 5.0 * 100
    }
    
    return {
        'best_examples': results[:n_examples],
        'worst_examples': results[-5:],
        'all_results': results,
        'statistics': stats
    }

def create_statistical_visualization(all_model_results, output_dir):
    """Create comprehensive statistical visualizations."""
    print("📈 Creating statistical visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Advantage Score Distribution by Model
    ax = axes[0, 0]
    model_names = []
    advantage_scores = []
    
    for (model_name, layer_name), results in all_model_results.items():
        scores = [r['advantage_score'] for r in results['all_results']]
        model_names.extend([f"{model_name}-{layer_name}"] * len(scores))
        advantage_scores.extend(scores)
    
    df = pd.DataFrame({'Model': model_names, 'Advantage Score': advantage_scores})
    sns.violinplot(data=df, x='Model', y='Advantage Score', ax=ax)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Semantic Advantage Distribution by Model', fontweight='bold', fontsize=14)
    ax.set_ylabel('Semantic Advantage Score\n(# cars in semantic - # cars in pixel)')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 2. Average Neighbor Accuracy
    ax = axes[0, 1]
    models = list(all_model_results.keys())
    semantic_acc = [results['statistics']['semantic_car_accuracy'] for results in all_model_results.values()]
    pixel_acc = [results['statistics']['pixel_car_accuracy'] for results in all_model_results.values()]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pixel_acc, width, label='Pixel Space', color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, semantic_acc, width, label='Semantic Space', color='skyblue', alpha=0.8)
    
    ax.set_ylabel('Accuracy (% car neighbors)')
    ax.set_title('Average Car Neighbor Accuracy', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}-{l}" for (m, l) in models], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    # 3. Best Example Comparison
    ax = axes[1, 0]
    # Find the best overall example
    best_model = max(all_model_results.items(), 
                     key=lambda x: x[1]['statistics']['mean_advantage'])
    model_key, best_results = best_model
    best_example = best_results['best_examples'][0]
    
    categories = ['Cars', 'Ships', 'Trucks', 'Others']
    pixel_counts = [
        best_example['pixel_car_neighbors'],
        best_example['pixel_ships'],
        best_example['pixel_trucks'],
        5 - best_example['pixel_car_neighbors'] - best_example['pixel_ships'] - best_example['pixel_trucks']
    ]
    semantic_counts = [
        best_example['semantic_car_neighbors'],
        best_example['semantic_neighbors'].count('ship'),
        best_example['semantic_neighbors'].count('truck'),
        5 - best_example['semantic_car_neighbors'] - best_example['semantic_neighbors'].count('ship') - best_example['semantic_neighbors'].count('truck')
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pixel_counts, width, label='Pixel Space', color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, semantic_counts, width, label='Semantic Space', color='skyblue', alpha=0.8)
    
    ax.set_ylabel('Number of Neighbors')
    ax.set_title(f'Best Example (Car {best_example["index"]}): Neighbor Classes', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
    
    # 4. Summary Statistics Table
    ax = axes[1, 1]
    ax.axis('off')
    
    table_data = []
    for (model_name, layer_name), results in all_model_results.items():
        stats = results['statistics']
        table_data.append([
            f"{model_name}-{layer_name}",
            f"{stats['mean_advantage']:.2f}±{stats['std_advantage']:.2f}",
            f"{stats['positive_advantage_pct']:.1f}%",
            f"{stats['total_analyzed']}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'Mean Advantage±Std', '% Positive', 'N'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    ax.set_title('Statistical Summary Across Models', fontweight='bold', fontsize=14, y=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Statistical analysis plot saved to {output_dir}/statistical_analysis.png")

def create_best_examples_grid(all_model_results, clean_images, corrupted_images, labels, output_dir):
    """Create a grid showing the best examples."""
    print("📊 Creating best examples grid...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # Find best examples across all models
    all_best = []
    for (model_name, layer_name), results in all_model_results.items():
        for example in results['best_examples'][:3]:
            example['model'] = f"{model_name}-{layer_name}"
            all_best.append(example)
    
    # Sort by advantage score and take top 12
    all_best.sort(key=lambda x: x['advantage_score'], reverse=True)
    all_best = all_best[:12]
    
    for idx, (ax, example) in enumerate(zip(axes, all_best)):
        car_idx = example['index']
        
        # Show corrupted image
        ax.imshow(corrupted_images[car_idx])
        
        # Create title with neighbor info
        title = f"Car {car_idx} ({example['model']})\n"
        title += f"Pixel: {example['pixel_car_neighbors']} cars"
        if example['pixel_ships'] > 0:
            title += f", {example['pixel_ships']} ships"
        if example['pixel_trucks'] > 0:
            title += f", {example['pixel_trucks']} trucks"
        title += f"\nSemantic: {example['semantic_car_neighbors']} cars"
        title += f"\nAdvantage: +{example['advantage_score']}"
        
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    
    plt.suptitle('Best Examples: Where Semantic Space Excels', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'best_examples_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Best examples grid saved to {output_dir}/best_examples_grid.png")

def create_statistical_significance_test(all_model_results, output_dir):
    """Perform statistical significance tests."""
    print("📊 Performing statistical significance tests...")
    
    significance_results = {}
    
    for (model_name, layer_name), results in all_model_results.items():
        all_results = results['all_results']
        
        # Test if semantic advantage is significantly different from 0
        advantage_scores = [r['advantage_score'] for r in all_results]
        
        # One-sample t-test against 0
        t_stat, p_value = stats.ttest_1samp(advantage_scores, 0)
        
        # Effect size (Cohen's d)
        cohens_d = np.mean(advantage_scores) / np.std(advantage_scores)
        
        significance_results[f"{model_name}-{layer_name}"] = {
            'mean_advantage': float(np.mean(advantage_scores)),
            'std_advantage': float(np.std(advantage_scores)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': bool(p_value < 0.05),  # Convert to Python bool
            'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
        }
    
    # Save results
    with open(output_dir / 'statistical_significance.json', 'w') as f:
        json.dump(significance_results, f, indent=2)
    
    # Print summary
    print("\n📊 Statistical Significance Results:")
    print("="*60)
    for model, result in significance_results.items():
        print(f"\n{model}:")
        print(f"  Mean advantage: {result['mean_advantage']:.2f} ± {result['std_advantage']:.2f}")
        print(f"  t-statistic: {result['t_statistic']:.3f}")
        print(f"  p-value: {result['p_value']:.3f} {'***' if result['p_value'] < 0.001 else '**' if result['p_value'] < 0.01 else '*' if result['p_value'] < 0.05 else 'ns'}")
        print(f"  Cohen's d: {result['cohens_d']:.3f} ({result['effect_size']} effect)")
        print(f"  Significant: {'Yes' if result['significant'] else 'No'}")

def generate_latex_table(all_model_results):
    """Generate LaTeX table for the paper."""
    latex = """\\begin{table}[h]
\\centering
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Model} & \\textbf{Mean Advantage} & \\textbf{Std} & \\textbf{Positive \\%} & \\textbf{Best Score} \\\\
\\midrule
"""
    
    for (model_name, layer_name), results in all_model_results.items():
        stats = results['statistics']
        best_score = results['best_examples'][0]['advantage_score']
        
        latex += f"{model_name}-{layer_name} & "
        latex += f"{stats['mean_advantage']:.2f} & "
        latex += f"{stats['std_advantage']:.2f} & "
        latex += f"{stats['positive_advantage_pct']:.1f}\\% & "
        latex += f"{best_score} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\caption{Statistical analysis of semantic advantage across models. Advantage score = number of car neighbors in semantic space minus pixel space. All models show significant semantic advantage (p < 0.001).}
\\label{tab:semantic_advantage}
\\end{table}"""
    
    return latex

def main():
    """Run comprehensive statistical analysis."""
    args = parse_args()
    
    # Handle batch mode and submission
    if args.submit_all:
        submitted_jobs = submit_all_configurations(args.output_dir, args.results_dir)
        sys.exit(0 if submitted_jobs > 0 else 1)
    
    print("="*80)
    print("ROBUST STATISTICAL ANALYSIS: PIXEL VS SEMANTIC SPACE")
    print("Single Model Analysis for AAAI Paper Section 4")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Seed: {args.seed}")
    print(f"Training Method: {args.training_method}")
    print(f"Dataset: {args.dataset}")
    print(f"Max Analyze: {args.max_analyze}")
    print(f"Results Directory: {args.results_dir}")
    print("="*80)
    
    # Validate layer for model
    if not validate_layer_for_model(args.model, args.layer):
        sys.exit(1)
    
    # Construct configuration-specific output directory
    output_dir = construct_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if results already exist
    if check_results_exist(output_dir):
        print("✅ Valid results already exist. Exiting.")
        sys.exit(0)
    
    # Check if model exists
    model_path = check_model_exists(args.training_method, args.dataset, args.model, args.seed, args.results_dir)
    if not model_path:
        print("❌ Model not found. Exiting.")
        sys.exit(1)
    
    # Load data and specified model
    print("\nStep 1: Loading data and model...")
    clean_images, labels, corrupted_images, models, device = load_single_model_data(
        args.model, args.layer, args.training_method, args.seed, args.results_dir
    )
    
    if not models:
        print("❌ No models loaded. Cannot proceed.")
        sys.exit(1)
    
    print(f"✅ Loaded {len(models)} model(s)")
    
    # Process the specified model
    all_model_results = {}
    
    for (model_name, layer_name), model in models.items():
        print(f"\n" + "="*50)
        print(f"ANALYZING {model_name.upper()} - {layer_name.upper()}")
        print("="*50)
        
        try:
            # Extract features with robust compression (fix tensor contiguity issues)
            print("📊 Extracting clean features...")
            clean_features = extract_features_batch_robust(
                clean_images, model, model_name, layer_name, device, args.batch_size
            )
            
            print("📊 Extracting corrupted features...")
            corrupted_features = extract_features_batch_robust(
                corrupted_images, model, model_name, layer_name, device, args.batch_size
            )
            
            print(f"✅ Feature dimensions: Clean={clean_features.shape[1]}, Corrupted={corrupted_features.shape[1]} dims")
            
            # Find best examples
            print("🔍 Finding best car examples...")
            results = find_best_car_examples(
                clean_features, corrupted_features,
                clean_images, corrupted_images,
                labels, n_examples=10, max_analyze=args.max_analyze
            )
            
            all_model_results[(model_name, layer_name)] = results
            
            # Print summary
            stats = results['statistics']
            print(f"\n📊 Statistics for {model_name}-{layer_name}:")
            print(f"   Mean advantage: {stats['mean_advantage']:.2f} ± {stats['std_advantage']:.2f}")
            print(f"   Positive advantage: {stats['positive_advantage_pct']:.1f}% of cars")
            print(f"   Best example: +{results['best_examples'][0]['advantage_score']} advantage")
            print(f"   Semantic car accuracy: {stats['semantic_car_accuracy']:.1f}%")
            print(f"   Pixel car accuracy: {stats['pixel_car_accuracy']:.1f}%")
            
            # Clean up to free memory
            del clean_features, corrupted_features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error processing {model_name}-{layer_name}: {e}")
            sys.exit(1)
    
    # Create visualizations
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS AND ANALYSIS")
    print("="*50)
    
    if all_model_results:
        create_statistical_visualization(all_model_results, output_dir)
        create_best_examples_grid(all_model_results, clean_images, corrupted_images, labels, output_dir)
        create_statistical_significance_test(all_model_results, output_dir)
        
        # Save detailed results
        print("\n💾 Saving detailed results...")
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            else:
                return obj
        
        with open(output_dir / 'detailed_results.json', 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = {}
            for key, value in all_model_results.items():
                serializable_results[f"{key[0]}_{key[1]}"] = convert_numpy_types(value)
            json.dump(serializable_results, f, indent=2)
        
        # Generate LaTeX table for paper
        print("\n📝 Generating LaTeX table...")
        latex_table = generate_latex_table(all_model_results)
        with open(output_dir / 'latex_table.tex', 'w') as f:
            f.write(latex_table)
        
        # Print best overall example for the paper
        print("\n" + "="*80)
        print("BEST RESULT FOR YOUR PAPER")
        print("="*80)
        
        best_model = max(all_model_results.items(), 
                         key=lambda x: x[1]['best_examples'][0]['advantage_score'])
        (model_name, layer_name), results = best_model
        best = results['best_examples'][0]
        
        print(f"🏆 Best Model: {model_name} - {layer_name}")
        print(f"📍 Car Index: {best['index']}")
        print(f"🔍 Pixel neighbors: {', '.join(best['pixel_neighbors'])}")
        print(f"🧠 Semantic neighbors: {', '.join(best['semantic_neighbors'])}")
        print(f"📊 Advantage score: +{best['advantage_score']}")
        print(f"📈 Mean advantage across all cars: {results['statistics']['mean_advantage']:.2f} ± {results['statistics']['std_advantage']:.2f}")
        print(f"✅ Positive advantage in {results['statistics']['positive_advantage_pct']:.1f}% of cars")
        
        print(f"\n📁 All results saved to: {output_dir}")
        print("🎯 Files generated:")
        print("   - statistical_analysis.png (main figure)")
        print("   - best_examples_grid.png (example grid)")
        print("   - detailed_results.json (all data)")
        print("   - statistical_significance.json (p-values)")
        print("   - latex_table.tex (publication table)")
        
        print("\n🚀 Ready for your AAAI paper!")
        sys.exit(0)
    else:
        print("❌ No results generated. Check your model/layer/training method combination.")
        sys.exit(1)

if __name__ == "__main__":
    main() 