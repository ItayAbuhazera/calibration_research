"""
Test script to compare max pooling vs average pooling in SPP for geometric calibration.

This script validates the design choice of using average pooling in SPP layers
by comparing it against max pooling (as used in the original SPP paper).

Expected output: CSV file with comparison results and statistical analysis.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import logging
import argparse
import csv
import time
import glob
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple
from scipy import stats
from tqdm import tqdm

# Import utilities
from Experiments.run_post_hoc_calibration import (
    PyTorchModelAdapter,
    get_data_loaders,
    load_trained_model,
    construct_model_path
)
from Experiments.compare_dac_geometric import (
    extract_features_from_multiple_layers,
    combine_features,
    run_geometric_calibration,
    get_layer_names_for_model,
    find_layer_module,
    calculate_ece,
    calculate_accuracy
)
from Calibrators.geometric_calibrator_new import GeometricCalibrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from utils.logging_config import get_logger
logger = get_logger(__name__)


# Representative configs - now auto-discovered (kept for reference)
# REPRESENTATIVE_CONFIGS = [
#     {
#         'training_method': 'baseline',
#         'dataset': 'cifar10',
#         'model_name': 'resnet18',
#         'seed': 11
#     },
#     {
#         'training_method': 'baseline',
#         'dataset': 'cifar100',
#         'model_name': 'densenet121',
#         'seed': 11
#     },
#     {
#         'training_method': 'baseline',
#         'dataset': 'cifar10',
#         'model_name': 'resnet50',
#         'seed': 11
#     }
# ]


def get_fixed_5_layers(model_name: str, model: torch.nn.Module) -> List[str]:
    """Get fixed 5 layers for geo_comb experiment."""
    all_layers = get_layer_names_for_model(model_name, model)
    # Select 5 representative layers (typically main blocks)
    if len(all_layers) >= 5:
        # For most models, take layers 1, 2, 3, 4, 5 (0-indexed: 0, 1, 2, 3, 4)
        return all_layers[:5]
    else:
        return all_layers


def get_random_4_layers(model_name: str, model: torch.nn.Module, seed: int = 42) -> List[str]:
    """Get random 4 layers for random layer selection experiment."""
    import random
    all_layers = get_layer_names_for_model(model_name, model)
    if len(all_layers) < 4:
        return all_layers
    random.seed(seed)
    return random.sample(all_layers, 4)


def find_available_models(results_base_dir: str, min_accuracy: float = 50.0) -> List[Dict[str, Any]]:
    """
    Scan results directory and find available trained models with >50% accuracy.
    
    Expected structure:
    results_base_dir/
      training_method_category/
        training_method_name/
          dataset/
            model_name/
              seedXX/
                training_method_dataset_model_seedXX/
                  best_model.pth
                  test_results.json  # Contains accuracy
    """
    available = []
    
    # Pattern: results_base_dir/*/*/*/seed*/*/best_model.pth
    pattern = os.path.join(results_base_dir, '*', '*', '*', '*', 'seed*', '*', 'best_model.pth')
    model_paths = glob.glob(pattern)
    
    logger.info(f"Found {len(model_paths)} potential model paths")
    
    for model_path in model_paths:
        try:
            # Parse path components
            # Example: baseline/baseline_cross_entropy/cifar10/resnet18/seed18/baseline_cross_entropy_cifar10_resnet18_seed18/best_model.pth
            model_path_obj = Path(model_path)
            results_base_dir_obj = Path(results_base_dir)
            
            # Get relative path from results_base_dir
            try:
                rel_path = model_path_obj.relative_to(results_base_dir_obj)
                rel_parts = rel_path.parts
            except ValueError:
                # If not relative, try to find common path
                # This handles cases where paths don't share a common root
                parts = model_path_obj.parts
                base_parts = results_base_dir_obj.parts
                
                # Find where they diverge
                common_len = 0
                for i in range(min(len(parts), len(base_parts))):
                    if parts[i] == base_parts[i]:
                        common_len += 1
                    else:
                        break
                
                if common_len > 0:
                    rel_parts = parts[common_len:]
                else:
                    # Fallback: try to find 'results' in path
                    if 'results' in parts:
                        results_idx = parts.index('results')
                        rel_parts = parts[results_idx + 1:]
                    else:
                        logger.debug(f"Could not parse path: {model_path}")
                        continue
            
            if len(rel_parts) < 6:
                continue
            
            training_category = rel_parts[0]  # e.g., 'baseline'
            training_method = rel_parts[1]     # e.g., 'baseline_cross_entropy'
            dataset = rel_parts[2]             # e.g., 'cifar10'
            model_name = rel_parts[3]          # e.g., 'resnet18'
            seed_dir = rel_parts[4]            # e.g., 'seed18'
            
            # Extract seed number
            seed = int(seed_dir.replace('seed', ''))
            
            # Check for results.json to get accuracy
            results_dir = os.path.dirname(model_path)
            results_path = os.path.join(results_dir, 'results.json')
            
            accuracy = None
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    # Try multiple possible keys for accuracy
                    accuracy = results.get(
                        'test_accuracy',
                        results.get(
                            'accuracy',
                            results.get('final_test_accuracy')
                        )
                    )
            
            # Skip if accuracy too low
            if accuracy is not None and accuracy < min_accuracy:
                logger.debug(f"Skipping {model_path}: accuracy {accuracy:.2f}% < {min_accuracy}%")
                continue
            
            available.append({
                'training_method': training_method,
                'dataset': dataset,
                'model_name': model_name,
                'seed': seed,
                'model_path': model_path,
                'accuracy': accuracy
            })
            
        except Exception as e:
            logger.debug(f"Error parsing {model_path}: {e}")
            continue
    
    return available


def select_representative_configs(
    available_models: List[Dict[str, Any]], 
    n_configs: int = 6
) -> List[Dict[str, Any]]:
    """
    Select diverse representative configs for testing.
    
    Priority:
    1. Different architectures (ResNet-18, DenseNet-121, ResNet-50)
    2. Different datasets (CIFAR-10, CIFAR-100)
    3. Different training methods (cross_entropy, brier, augmix)
    4. Highest accuracy within each group
    """
    # Group by (model_name, dataset)
    groups = defaultdict(list)
    
    for model in available_models:
        key = (model['model_name'], model['dataset'])
        groups[key].append(model)
    
    # Sort each group by accuracy (highest first)
    for key in groups:
        groups[key].sort(key=lambda x: x['accuracy'] if x['accuracy'] else 0, reverse=True)
    
    # Priority order for selection (expanded)
    priorities = [
        ('resnet18', 'cifar10'),        # Standard baseline
        ('resnet18', 'cifar100'),       # Same model, harder dataset
        ('resnet50', 'cifar10'),        # Larger model
        ('resnet50', 'cifar100'),       # Large model + hard dataset
        ('densenet121', 'cifar10'),     # Different architecture (safe memory)
        ('wideresnet28x10', 'cifar10'), # Very wide model
    ]
    
    selected = []
    for key in priorities:
        if key in groups and len(selected) < n_configs:
            selected.append(groups[key][0])  # Take highest accuracy
    
    # Fill remaining with any available
    for key, models in groups.items():
        if len(selected) >= n_configs:
            break
        if models[0] not in selected:
            selected.append(models[0])
    
    selected = selected[:n_configs]
    
    # FILTER OUT MEMORY-INTENSIVE CONFIGS
    filtered = []
    for config in selected:
        model = config['model_name']
        dataset = config['dataset']
        
        # Memory rules based on empirical OOM testing:
        # - ResNet-50 + CIFAR-100: 20,496 dims → OOM
        # - DenseNet-121 + CIFAR-100: 15,120 dims → OOM
        # - WideResNet + CIFAR-100: likely OOM
        # - ResNet-18 + CIFAR-100: ~5k dims → OK
        
        skip = False
        skip_reason = None
        
        if model == 'densenet121' and dataset == 'cifar100':
            skip = True
            skip_reason = "DenseNet-121 + CIFAR-100 creates ~15k dims (OOM)"
        elif model == 'resnet50' and dataset == 'cifar100':
            skip = True
            skip_reason = "ResNet-50 + CIFAR-100 creates ~20k dims (OOM)"
        elif 'wideresnet' in model.lower() and dataset == 'cifar100':
            skip = True
            skip_reason = "WideResNet + CIFAR-100 likely causes OOM"
        
        if skip:
            logger.warning(f"⚠️  Skipping {model}_{dataset}: {skip_reason}")
            continue
        
        filtered.append(config)
    
    # If we filtered out configs, try to find memory-safe replacements
    if len(filtered) < n_configs:
        logger.info(f"Need {n_configs - len(filtered)} memory-safe replacement configs...")
        
        # Priority replacements (memory-friendly)
        safe_priorities = [
            ('resnet18', 'cifar10'),
            ('resnet18', 'cifar100'),   # ResNet-18 safe with CIFAR-100
            ('resnet50', 'cifar10'),
            ('densenet121', 'cifar10'), # DenseNet safe with CIFAR-10
            ('resnet18', 'svhn'),
            ('densenet121', 'svhn'),
            ('wideresnet28x10', 'cifar10'),  # WideResNet safe with CIFAR-10
        ]
        
        for key in safe_priorities:
            if len(filtered) >= n_configs:
                break
            if key in groups:
                candidate = groups[key][0]
                already_included = any(
                    c['model_name'] == candidate['model_name'] and
                    c['dataset'] == candidate['dataset'] and
                    c['seed'] == candidate['seed']
                    for c in filtered
                )
                if not already_included:
                    logger.info(f"  ✓ Adding safe replacement: {candidate['model_name']}_{candidate['dataset']}")
                    filtered.append(candidate)
    
    # Log final selection with memory estimates
    logger.info(f"\nFinal {len(filtered)} memory-safe configs:")
    for config in filtered:
        model = config['model_name']
        dataset = config['dataset']
        if model == 'resnet18':
            est_dims = "~5k" if dataset == 'cifar100' else "~4k"
        elif model == 'resnet50':
            est_dims = "~8k" if dataset == 'cifar10' else "~20k (AVOID)"
        elif model == 'densenet121':
            est_dims = "~8k" if dataset == 'cifar10' else "~15k (AVOID)"
        elif 'wideresnet' in model.lower():
            est_dims = "~6-8k" if dataset == 'cifar10' else "~20k (AVOID)"
        else:
            est_dims = "unknown"
        
        logger.info(f"  ✓ {model}_{dataset}_seed{config['seed']} (est. {est_dims} dims)")
    
    return filtered


def construct_model_path_fixed(
    results_base_dir: str,
    training_method: str,
    dataset: str,
    model_name: str,
    seed: int
) -> str:
    """
    Construct model path matching actual directory structure.
    
    Structure: results_base_dir/category/training_method/dataset/model_name/seedXX/checkpoint_name/best_model.pth
    
    The category is derived from training_method (e.g., 'baseline_cross_entropy' -> 'baseline')
    """
    # Infer category from training_method
    if 'baseline' in training_method.lower():
        category = 'baseline'
    elif 'augmix' in training_method.lower():
        category = 'augmix'
    else:
        category = training_method.split('_')[0]
    
    checkpoint_name = f"{training_method}_{dataset}_{model_name}_seed{seed}"
    
    model_path = os.path.join(
        results_base_dir,
        category,
        training_method,
        dataset,
        model_name,
        f'seed{seed}',
        checkpoint_name,
        'best_model.pth'
    )
    
    return model_path


def run_single_experiment(
    model: torch.nn.Module,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    layer_names: List[str],
    compression_ratio: float,
    pooling_mode: str,
    batch_size: int = 128
) -> Dict[str, Any]:
    """
    Run a single geometric calibration experiment.
    
    Returns:
        Dictionary with ECE, accuracy, and feature extraction time.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running experiment: {pooling_mode} pooling, CR={compression_ratio}x, {len(layer_names)} layers")
    logger.info(f"Layers: {layer_names}")
    logger.info(f"{'='*80}")
    
    # Create dataloaders
    from torch.utils.data import DataLoader, TensorDataset
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
        batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )
    
    # Extract features
    start_time = time.perf_counter()
    train_features_list = extract_features_from_multiple_layers(
        model, train_loader, layer_names, device,
        compression_ratio=compression_ratio,
        pooling_mode=pooling_mode
    )
    val_features_list = extract_features_from_multiple_layers(
        model, val_loader, layer_names, device,
        compression_ratio=compression_ratio,
        pooling_mode=pooling_mode
    )
    test_features_list = extract_features_from_multiple_layers(
        model, test_loader, layer_names, device,
        compression_ratio=compression_ratio,
        pooling_mode=pooling_mode
    )
    feature_extraction_time = time.perf_counter() - start_time
    
    # Combine features if multiple layers
    if len(layer_names) > 1:
        train_features = combine_features(train_features_list)
        val_features = combine_features(val_features_list)
        test_features = combine_features(test_features_list)
        
        # Apply post-concatenation compression
        from utils.compression_utils import SmartCompression
        post_compression_ratio = 8.0 if len(train_labels) > 10000 else 4.0  # CIFAR100 vs CIFAR10
        compressor = SmartCompression(
            method='random_projection',
            compression_ratio=post_compression_ratio,
            random_state=42
        )
        
        train_features_np = train_features.numpy() if torch.is_tensor(train_features) else train_features
        val_features_np = val_features.numpy() if torch.is_tensor(val_features) else val_features
        test_features_np = test_features.numpy() if torch.is_tensor(test_features) else test_features
        
        train_features = torch.from_numpy(compressor(train_features_np, train=True))
        val_features = torch.from_numpy(compressor(val_features_np, train=False))
        test_features = torch.from_numpy(compressor(test_features_np, train=False))
    else:
        train_features = train_features_list[0]
        val_features = val_features_list[0]
        test_features = test_features_list[0]
    
    # Create geometric calibrator
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features.numpy(),
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,
        device=str(device),
    )
    
    # FIT the calibrator on validation set
    fit_start = time.perf_counter()
    geo_cal.fit(
        X_val_embed=val_features.numpy(),
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size,
    )
    fit_time = time.perf_counter() - fit_start
    
    # CALIBRATE on test set
    calibrate_start = time.perf_counter()
    test_calibrated_probs = geo_cal.calibrate_batched(
        X_test_embed=test_features.numpy(),
        X_test_original=test_raw,
        batch_size=batch_size,
    )
    calibrate_time = time.perf_counter() - calibrate_start
    
    # Compute metrics
    ece = calculate_ece(test_calibrated_probs, test_labels)
    
    accuracy = calculate_accuracy(test_calibrated_probs, test_labels)
    
    logger.info(
        f"Results: ECE={ece*100:.3f}%, Accuracy={accuracy:.2f}%, "
        f"Feature time={feature_extraction_time:.2f}s, "
        f"Fit time={fit_time:.2f}s, Calibrate time={calibrate_time:.2f}s"
    )
    
    return {
        'ece': ece * 100,  # Convert to percentage
        'accuracy': accuracy,
        'feature_extraction_time_s': feature_extraction_time
    }


def run_all_experiments(
    config: Dict[str, str],
    results_base_dir: str,
    device: torch.device,
    batch_size: int = 128
) -> List[Dict[str, Any]]:
    """Run all experiments for a single config."""
    training_method = config['training_method']
    dataset = config['dataset']
    model_name = config['model_name']
    seed = config['seed']
    model_path = config.get('model_path')  # Use pre-discovered path
    
    logger.info(f"\n{'#'*80}")
    logger.info(f"Processing config: {training_method}_{dataset}_{model_name}_seed{seed}")
    logger.info(f"{'#'*80}")
    
    # Use provided model_path if available, otherwise construct
    if model_path is None:
        model_path = construct_model_path_fixed(
            results_base_dir,
            training_method,
            dataset,
            model_name,
            seed
        )
    
    if not os.path.exists(model_path):
        logger.warning(f"Model not found: {model_path}, skipping...")
        return []
    
    # Load data
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        dataset,
        batch_size,
        seed=seed
    )
    
    # Load model
    model = load_trained_model(
        model_path,
        model_name,
        num_classes,
        device,
        dataset=dataset
    )
    
    model_adapter = PyTorchModelAdapter(model, device, dataset)
    
    # Extract raw data
    train_raw = []
    train_labels = []
    for batch in train_loader:
        train_raw.append(batch[0].numpy())
        train_labels.append(batch[1].numpy())
    train_raw = np.concatenate(train_raw, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    val_raw = []
    val_labels = []
    for batch in val_loader:
        val_raw.append(batch[0].numpy())
        val_labels.append(batch[1].numpy())
    val_raw = np.concatenate(val_raw, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    
    test_raw = []
    test_labels = []
    for batch in test_loader:
        test_raw.append(batch[0].numpy())
        test_labels.append(batch[1].numpy())
    test_raw = np.concatenate(test_raw, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    # Get layer selections
    fixed_5_layers = get_fixed_5_layers(model_name, model)
    random_4_layers = get_random_4_layers(model_name, model, seed=seed)
    
    logger.info(f"Fixed 5 layers: {fixed_5_layers}")
    logger.info(f"Random 4 layers: {random_4_layers}")
    
    # Run all combinations
    results = []
    
    for pooling_mode in ['avg', 'max']:
        for layer_selection, layer_names in [('fixed_5', fixed_5_layers), ('random_4', random_4_layers)]:
            for compression_ratio in [4.0, 16.0]:
                try:
                    result = run_single_experiment(
                        model, model_adapter,
                        train_raw, train_labels,
                        val_raw, val_labels,
                        test_raw, test_labels,
                        device, layer_names,
                        compression_ratio, pooling_mode,
                        batch_size
                    )
                    
                    results.append({
                        'config': f"{training_method}_{dataset}_{model_name}_seed{seed}",
                        'layer_selection': layer_selection,
                        'compression_ratio': int(compression_ratio),
                        'pooling_mode': pooling_mode,
                        **result
                    })
                except Exception as e:
                    logger.error(f"Error in experiment {pooling_mode}/{layer_selection}/CR{compression_ratio}: {e}")
                    import traceback
                    traceback.print_exc()
    
    return results


def compute_statistical_analysis(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistical analysis comparing max vs avg pooling."""
    # Group by config, layer_selection, compression_ratio
    grouped = {}
    for result in all_results:
        key = (result['config'], result['layer_selection'], result['compression_ratio'])
        if key not in grouped:
            grouped[key] = {'avg': [], 'max': []}
        grouped[key][result['pooling_mode']].append(result['ece'])
    
    # Compute paired differences
    differences = []
    for key, values in grouped.items():
        if len(values['avg']) > 0 and len(values['max']) > 0:
            # Take mean if multiple runs
            avg_ece = np.mean(values['avg'])
            max_ece = np.mean(values['max'])
            differences.append(max_ece - avg_ece)  # max - avg (positive means max is better)
    
    if len(differences) == 0:
        return {}
    
    differences = np.array(differences)
    
    # Paired t-test
    # H0: mean difference = 0 (no difference)
    t_stat, p_value = stats.ttest_1samp(differences, 0.0)
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
    
    # TOST equivalence test with ±0.2% margin
    equivalence_margin = 0.2
    t1, p1 = stats.ttest_1samp(differences, -equivalence_margin)
    t2, p2 = stats.ttest_1samp(differences, equivalence_margin)
    tost_p_value = max(p1, p2)
    is_equivalent = tost_p_value < 0.05
    
    ci_low = float(mean_diff - 1.96 * std_diff / np.sqrt(len(differences)))
    ci_high = float(mean_diff + 1.96 * std_diff / np.sqrt(len(differences)))
    
    return {
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff),
        'n_pairs': len(differences),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'tost_p_value': float(tost_p_value),
        'tost_equivalence_margin': equivalence_margin,
        'tost_is_equivalent': is_equivalent,
        'confidence_interval_95': [ci_low, ci_high]
    }


def generate_verdict(statistics: Dict[str, Any]) -> str:
    """Generate verdict based on statistical analysis."""
    if not statistics:
        return "Insufficient data for analysis"
    
    mean_diff = statistics['mean_difference']
    p_value = statistics['p_value']
    is_equivalent = statistics.get('tost_is_equivalent', False)
    ci = statistics.get('confidence_interval_95', [0, 0])
    
    if is_equivalent:
        verdict = "Statistically equivalent - Either pooling works"
        interpretation = (
            f"TOST equivalence test confirms max and average pooling are equivalent "
            f"within ±{statistics.get('tost_equivalence_margin', 0.2):.1f}% margin "
            f"(p={statistics.get('tost_p_value', 1.0):.3f}). "
            f"Mean difference: {mean_diff:.3f}% (95% CI: [{ci[0]:.3f}%, {ci[1]:.3f}%]). "
            "Choose based on theoretical preference or literature alignment."
        )
    elif abs(mean_diff) < 0.2 and p_value > 0.05:
        verdict = "Likely equivalent - Need more data for confirmation"
        interpretation = (
            f"Mean difference is small ({mean_diff:.3f}%) but sample size insufficient "
            f"for statistical equivalence (TOST p={statistics.get('tost_p_value', 1.0):.3f}). "
            f"Recommend n≥24 pairs for definitive conclusion."
        )
    elif abs(mean_diff) < 0.5:
        verdict = "Practically equivalent but statistically different"
        interpretation = (
            f"Difference ({mean_diff:.3f}%) is practically negligible "
            f"but statistically detectable (p={p_value:.3f}). "
            f"Either choice is acceptable."
        )
    elif mean_diff > 0:
        verdict = "Max pooling is better"
        interpretation = (
            f"Max pooling provides {mean_diff:.3f}% improvement (p={p_value:.3f}). "
            f"Recommend switching to max pooling."
        )
    else:
        verdict = "Average pooling is better"
        interpretation = (
            f"Average pooling is {abs(mean_diff):.3f}% better (p={p_value:.3f}). "
            f"Keep average pooling."
        )
    
    return f"{verdict}\n{interpretation}"
    
    # Legacy logic removed in favor of TOST-driven decisions (handled above)


def main():
    parser = argparse.ArgumentParser(description='Test SPP pooling modes (max vs avg)')
    parser.add_argument('--results-base-dir', type=str,
                        default='aaai_full_experiments/results',
                        help='Base directory for experiment results')
    parser.add_argument('--output-dir', type=str,
                        default='spp_pooling_test_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for data loading')
    parser.add_argument('--min-accuracy', type=float, default=50.0,
                        help='Minimum model accuracy to include')
    parser.add_argument('--n-configs', type=int, default=6,
                        help='Number of representative configs to test')
    parser.add_argument('--skip-configs', type=str, nargs='*', default=[],
                        help='Config patterns to skip (e.g., "densenet121_cifar100")')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Auto-discover available models
    logger.info("Scanning for available models...")
    available_models = find_available_models(args.results_base_dir, args.min_accuracy)
    logger.info(f"Found {len(available_models)} models with accuracy >= {args.min_accuracy}%")
    
    if len(available_models) == 0:
        logger.error("No suitable models found! Check results_base_dir and min_accuracy threshold.")
        return
    
    # Select representative configs
    selected_configs = select_representative_configs(available_models, args.n_configs)
    logger.info(f"\nSelected {len(selected_configs)} representative configs:")
    for config in selected_configs:
        acc_str = f"{config['accuracy']:.2f}%" if config['accuracy'] else "N/A"
        logger.info(f"  - {config['training_method']}_{config['dataset']}_{config['model_name']}_seed{config['seed']}")
        logger.info(f"    Accuracy: {acc_str} | Path: {config['model_path']}")
    
    # Run all experiments with incremental saving
    all_results = []
    csv_path = os.path.join(args.output_dir, 'spp_pooling_comparison.csv')
    
    # Load existing results if file exists
    if os.path.exists(csv_path):
        logger.info(f"Loading existing results from {csv_path}")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            all_results = list(reader)
            # Convert string types back
            for r in all_results:
                r['compression_ratio'] = int(r['compression_ratio'])
                r['ece'] = float(r['ece'])
                r['accuracy'] = float(r['accuracy'])
                r['feature_extraction_time_s'] = float(r['feature_extraction_time_s'])
        logger.info(f"Loaded {len(all_results)} existing results")
        
        # Determine which configs are already done
        completed_configs = set(r['config'] for r in all_results)
        logger.info(f"Already completed: {completed_configs}")
    else:
        completed_configs = set()
    
    for config in tqdm(selected_configs, desc="Processing configs"):
        config_name = f"{config['training_method']}_{config['dataset']}_{config['model_name']}_seed{config['seed']}"
        
        # Check skip patterns
        skip = False
        for pattern in args.skip_configs:
            if pattern in config_name:
                logger.info(f"Skipping {config_name} - matches pattern '{pattern}'")
                skip = True
                break
        
        # Skip if already completed
        if skip or config_name in completed_configs:
            if config_name in completed_configs:
                logger.info(f"Skipping {config_name} - already completed")
            continue
        
        try:
            results = run_all_experiments(
                config,
                args.results_base_dir,
                device,
                args.batch_size
            )
            
            if results:
                all_results.extend(results)
                
                # SAVE IMMEDIATELY after each config
                logger.info(f"Saving results after {config_name}...")
                fieldnames = ['config', 'layer_selection', 'compression_ratio', 'pooling_mode',
                             'ece', 'accuracy', 'feature_extraction_time_s']
                
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_results)
                
                logger.info(f"✓ Saved {len(all_results)} total results to {csv_path}")
        
        except Exception as e:
            logger.error(f"FATAL ERROR processing {config_name}: {e}")
            logger.error("Saving partial results before continuing...")
            
            # Save what we have so far
            if all_results:
                fieldnames = ['config', 'layer_selection', 'compression_ratio', 'pooling_mode',
                             'ece', 'accuracy', 'feature_extraction_time_s']
                
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_results)
                
                logger.info(f"✓ Saved {len(all_results)} partial results")
            
            # Continue to next config instead of crashing
            continue
    
    logger.info(f"\nAll configs processed. Final results: {len(all_results)}")
    
    # Save results to CSV (final write to ensure consistency)
    logger.info(f"\nSaving results to: {csv_path}")
    
    if all_results:
        fieldnames = ['config', 'layer_selection', 'compression_ratio', 'pooling_mode',
                     'ece', 'accuracy', 'feature_extraction_time_s']
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        logger.info(f"Saved {len(all_results)} results to CSV")
        
        # Compute statistical analysis
        statistics = compute_statistical_analysis(all_results)
        
        if statistics:
            logger.info("\n" + "="*80)
            logger.info("STATISTICAL ANALYSIS")
            logger.info("="*80)
            logger.info(f"Mean difference (max - avg): {statistics['mean_difference']:.4f}%")
            logger.info(f"Std difference: {statistics['std_difference']:.4f}%")
            logger.info(f"Number of pairs: {statistics['n_pairs']}")
            logger.info(f"t-statistic: {statistics['t_statistic']:.4f}")
            logger.info(f"p-value: {statistics['p_value']:.4f}")
            logger.info(f"Cohen's d (effect size): {statistics['cohens_d']:.4f}")
            
            verdict = generate_verdict(statistics)
            logger.info("\n" + "="*80)
            logger.info("VERDICT")
            logger.info("="*80)
            logger.info(verdict)
            
            # Save statistics
            stats_path = os.path.join(args.output_dir, 'statistical_analysis.txt')
            with open(stats_path, 'w') as f:
                f.write("SPP Pooling Mode Comparison - Statistical Analysis\n")
                f.write("="*80 + "\n\n")
                f.write(f"Mean difference (max - avg): {statistics['mean_difference']:.4f}%\n")
                f.write(f"Std difference: {statistics['std_difference']:.4f}%\n")
                f.write(f"Number of pairs: {statistics['n_pairs']}\n")
                f.write(f"t-statistic: {statistics['t_statistic']:.4f}\n")
                f.write(f"p-value: {statistics['p_value']:.4f}\n")
                f.write(f"Cohen's d (effect size): {statistics['cohens_d']:.4f}\n\n")
                f.write("VERDICT:\n")
                f.write(verdict + "\n")
            
            logger.info(f"\nSaved statistical analysis to: {stats_path}")
    else:
        logger.warning("No results collected!")


if __name__ == "__main__":
    main()

