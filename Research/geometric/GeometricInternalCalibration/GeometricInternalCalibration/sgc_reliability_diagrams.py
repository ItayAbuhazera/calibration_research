"""
SGC Reliability Diagram Generation Script

This script generates publication-quality reliability diagrams for 
Randomized Geometric Calibration (SGC/RGCL) and its variants.

Convert to Jupyter notebook using: jupytext --to notebook sgc_reliability_diagrams.py
"""

# ============================================================================
# Cell 1: Imports and Configuration
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Project imports
import sys
sys.path.insert(0, '.')

from Experiments.run_post_hoc_calibration import (
    PyTorchModelAdapter,
    get_data_loaders,
    load_trained_model,
    construct_model_path
)
from Experiments.compare_dac_geometric import (
    extract_and_aggregate_sgc_features,
    normalize_discovered_layers,
    filter_non_feature_layers,
    calculate_ece,
    calculate_accuracy
)
from Experiments.multi_layer_ensemble import discover_model_layers
from Experiments.compare_dac_geometric import extract_features_from_multiple_layers
from Calibrators.geometric_calibrator_new import GeometricCalibrator
from Calibrators.temperature_scaling import TemperatureScaling
from utils.coordinate_extraction import (
    discover_coordinate_space,
    plan_coordinate_extraction,
    extract_coordinate_features
)
from utils.compression_utils import SmartCompression

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Configuration
CONFIG = {
    'models': ['resnet50', 'resnet101', 'densenet121'],
    'datasets': ['cifar10', 'cifar100', 'tiny_imagenet'],
    'seeds': [11, 12, 13, 14, 15, 16],
    'methods': ['uncalibrated', 'temperature_scaling', 'sgc', 'sgc_lite'],
    'sgc_params': {'L': 6, 'd': 256},
    'sgc_lite_params': {'K': 256},
    'output_dir': Path('reliability_data'),
    'figures_dir': Path('figures'),
    'batch_size': 128,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_bins': 10,  # For reliability diagrams
    'valid_size': 0.1,  # 10% validation split
}

# Create output directories
CONFIG['output_dir'].mkdir(exist_ok=True)
CONFIG['figures_dir'].mkdir(exist_ok=True)

print(f"Configuration loaded. Device: {CONFIG['device']}")
print(f"Output directory: {CONFIG['output_dir']}")
print(f"Figures directory: {CONFIG['figures_dir']}")


# ============================================================================
# Cell 2: Model and Data Loading Utilities
# ============================================================================

def get_num_classes(dataset_name: str) -> int:
    """Get number of classes for a dataset."""
    dataset_map = {
        'cifar10': 10,
        'cifar100': 100,
        'tiny_imagenet': 200,
    }
    return dataset_map.get(dataset_name.lower(), 10)


def load_model_and_data(
    model_name: str,
    dataset_name: str,
    seed: int,
    base_dir: str = 'checkpoints',
    training_method: str = 'cross_entropy'
) -> Tuple[torch.nn.Module, Dict[str, Any], PyTorchModelAdapter]:
    """
    Load pretrained model and data loaders.
    
    Returns:
        model: PyTorch model in eval mode
        data: Dict with 'train', 'val', 'test' loaders and raw arrays
        model_adapter: Adapter for model predictions
    """
    num_classes = get_num_classes(dataset_name)
    
    # Construct model path
    model_path = construct_model_path(
        base_dir=base_dir,
        method=training_method,
        dataset=dataset_name,
        model=model_name,
        seed=seed
    )
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load model
    device = torch.device(CONFIG['device'])
    model = load_trained_model(
        model_path=model_path,
        model_name=model_name,
        num_classes=num_classes,
        device=device,
        dataset=dataset_name
    )
    model.eval()
    
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset=dataset_name,
        batch_size=CONFIG['batch_size'],
        valid_size=CONFIG['valid_size'],
        random_seed=seed,
        augment=False  # No augmentation for calibration
    )
    
    # Extract raw arrays for compatibility
    def extract_raw(loader):
        images, labels = [], []
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                img, lbl = batch[0], batch[1]
            else:
                img, lbl = batch, None
            images.append(img.numpy() if isinstance(img, torch.Tensor) else img)
            if lbl is not None:
                labels.append(lbl.numpy() if isinstance(lbl, torch.Tensor) else lbl)
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0) if labels else None
        return images, labels
    
    train_raw, train_labels = extract_raw(train_loader)
    val_raw, val_labels = extract_raw(val_loader)
    test_raw, test_labels = extract_raw(test_loader)
    
    # Create model adapter
    model_adapter = PyTorchModelAdapter(model, device=device)
    
    data = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_raw': train_raw,
        'train_labels': train_labels,
        'val_raw': val_raw,
        'val_labels': val_labels,
        'test_raw': test_raw,
        'test_labels': test_labels,
    }
    
    return model, data, model_adapter


# ============================================================================
# Cell 3: Layer Extraction Utilities
# ============================================================================

def get_extractable_layers(
    model: torch.nn.Module,
    dataset_name: str,
    device: torch.device
) -> List[str]:
    """
    Discover all extractable intermediate representations.
    
    Returns:
        List of layer names that can be used for feature extraction
    """
    # Determine input shape
    if dataset_name.lower() in ['tiny_imagenet', 'tinyimagenet']:
        input_shape = (1, 3, 64, 64)
    else:
        input_shape = (1, 3, 32, 32)  # CIFAR
    
    # Discover layers
    discovered_layers = normalize_discovered_layers(
        model, discover_model_layers(model, device=device, input_shape=input_shape)
    )
    
    # Filter non-feature layers
    filtered_layers, _ = filter_non_feature_layers(discovered_layers, model)
    
    return [d['name'] for d in filtered_layers]


def sample_random_layers(
    all_layer_names: List[str],
    num_layers: int,
    seed: int
) -> List[str]:
    """Sample L layers uniformly at random."""
    np.random.seed(seed)
    if num_layers > len(all_layer_names):
        return all_layer_names
    return list(np.random.choice(all_layer_names, size=num_layers, replace=False))


# ============================================================================
# Cell 4: SGC Implementation
# ============================================================================

def run_sgc_calibration(
    model: torch.nn.Module,
    model_adapter: PyTorchModelAdapter,
    data: Dict[str, Any],
    device: torch.device,
    L: int = 6,
    d: int = 256,
    seed: int = 42,
    pooling_mode: str = 'max'
) -> Dict[str, np.ndarray]:
    """
    Run SGC (RGCL) calibration: Random layer sampling + SPP + projection + geometric scoring.
    
    Returns:
        Dict with 'max_probs', 'correct', 'predictions', 'labels'
    """
    # Get extractable layers
    all_layers = get_extractable_layers(model, 'cifar10', device)  # dataset_name not critical here
    
    # Sample random layers
    selected_layers = sample_random_layers(all_layers, L, seed)
    print(f"Selected {len(selected_layers)} layers: {selected_layers[:3]}...")
    
    # Extract and aggregate SGC features
    train_features, val_features, test_features, _ = extract_and_aggregate_sgc_features(
        model=model,
        layer_names=selected_layers,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        test_loader=data['test_loader'],
        device=device,
        target_dim=d,
        seed=seed,
        pooling_mode=pooling_mode
    )
    
    # Fit geometric calibrator
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=data['train_labels'],
        library='fast_separation',
        auto_select_layer=False,
        device=str(device),
        scoring_method='separation'
    )
    
    geo_cal.fit(
        X_val_embed=val_features,
        y_val=data['val_labels'],
        X_val_original=data['val_raw'],
        fit_batch_size=CONFIG['batch_size']
    )
    
    # Calibrate test set
    calibrated_probs = geo_cal.calibrate_batched(
        X_test_embed=test_features,
        X_test_original=data['test_raw'],
        batch_size=CONFIG['batch_size']
    )
    
    # Extract per-sample results
    max_probs = np.max(calibrated_probs, axis=1)
    predictions = np.argmax(calibrated_probs, axis=1)
    correct = (predictions == data['test_labels']).astype(float)
    
    return {
        'max_probs': max_probs,
        'correct': correct,
        'predictions': predictions,
        'labels': data['test_labels']
    }


def run_sgc_lite_calibration(
    model: torch.nn.Module,
    model_adapter: PyTorchModelAdapter,
    data: Dict[str, Any],
    device: torch.device,
    K: int = 256,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Run SGC-Lite (RGCC) calibration: Coordinate sampling + geometric scoring.
    
    Returns:
        Dict with 'max_probs', 'correct', 'predictions', 'labels'
    """
    # Determine input shape
    dataset_name = 'cifar10'  # Will be inferred from data shape
    if data['test_raw'].shape[-1] == 64:
        input_shape = (1, 3, 64, 64)
        dataset_name = 'tiny_imagenet'
    else:
        input_shape = (1, 3, 32, 32)
    
    # Discover coordinate space
    layer_map, total_size = discover_coordinate_space(
        model=model,
        input_shape=input_shape,
        device=str(device)
    )
    
    # Plan coordinate extraction
    sampling_plan, global_order = plan_coordinate_extraction(
        total_size=total_size,
        num_coordinates=K,
        layer_map=layer_map,
        seed=seed
    )
    
    # Extract coordinate features
    def extract_coords(loader):
        coords_list = []
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)
            coords = extract_coordinate_features(
                model=model,
                images=images,
                sampling_plan=sampling_plan,
                layer_map=layer_map,
                global_order=global_order,
                device=device
            )
            coords_list.append(coords.cpu().numpy())
        return np.concatenate(coords_list, axis=0)
    
    train_coords = extract_coords(data['train_loader'])
    val_coords = extract_coords(data['val_loader'])
    test_coords = extract_coords(data['test_loader'])
    
    # L2 normalize
    train_coords = normalize(train_coords, norm='l2', axis=1)
    val_coords = normalize(val_coords, norm='l2', axis=1)
    test_coords = normalize(test_coords, norm='l2', axis=1)
    
    # Fit geometric calibrator
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_coords,
        y_train=data['train_labels'],
        library='fast_separation',
        auto_select_layer=False,
        device=str(device),
        scoring_method='separation'
    )
    
    geo_cal.fit(
        X_val_embed=val_coords,
        y_val=data['val_labels'],
        X_val_original=data['val_raw'],
        fit_batch_size=CONFIG['batch_size']
    )
    
    # Calibrate test set
    calibrated_probs = geo_cal.calibrate_batched(
        X_test_embed=test_coords,
        X_test_original=data['test_raw'],
        batch_size=CONFIG['batch_size']
    )
    
    # Extract per-sample results
    max_probs = np.max(calibrated_probs, axis=1)
    predictions = np.argmax(calibrated_probs, axis=1)
    correct = (predictions == data['test_labels']).astype(float)
    
    return {
        'max_probs': max_probs,
        'correct': correct,
        'predictions': predictions,
        'labels': data['test_labels']
    }


# ============================================================================
# Cell 5: Baseline Methods
# ============================================================================

def get_uncalibrated_confidences(
    model_adapter: PyTorchModelAdapter,
    data: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Return max softmax probabilities (uncalibrated baseline)."""
    probs = model_adapter.predict_proba(data['test_raw'], batch_size=CONFIG['batch_size'])
    max_probs = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == data['test_labels']).astype(float)
    
    return {
        'max_probs': max_probs,
        'correct': correct,
        'predictions': predictions,
        'labels': data['test_labels']
    }


def run_temperature_scaling(
    model_adapter: PyTorchModelAdapter,
    data: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Fit temperature scaling on validation, apply to test."""
    # Get logits
    val_logits = model_adapter.predict_logits(data['val_raw'], batch_size=CONFIG['batch_size'])
    test_logits = model_adapter.predict_logits(data['test_raw'], batch_size=CONFIG['batch_size'])
    
    # Fit temperature scaling
    temp_scaler = TemperatureScaling(cross_validate='ece')
    temp_scaler.fit(logits=val_logits, labels=data['val_labels'])
    
    # Calibrate test set
    calibrated_probs = temp_scaler.calibrate(logits=test_logits)
    
    max_probs = np.max(calibrated_probs, axis=1)
    predictions = np.argmax(calibrated_probs, axis=1)
    correct = (predictions == data['test_labels']).astype(float)
    
    return {
        'max_probs': max_probs,
        'correct': correct,
        'predictions': predictions,
        'labels': data['test_labels']
    }


# ============================================================================
# Cell 6: Main Experiment Loop
# ============================================================================

def run_single_experiment(
    model_name: str,
    dataset_name: str,
    seed: int,
    method: str
) -> Optional[Dict[str, np.ndarray]]:
    """
    Run one experiment configuration.
    
    Returns:
        Dict with 'max_probs', 'correct', 'predictions', 'labels' or None if failed
    """
    try:
        print(f"\n{'='*60}")
        print(f"Running: {model_name} | {dataset_name} | seed={seed} | {method}")
        print(f"{'='*60}")
        
        # Load model and data
        model, data, model_adapter = load_model_and_data(
            model_name=model_name,
            dataset_name=dataset_name,
            seed=seed
        )
        
        device = torch.device(CONFIG['device'])
        
        # Run method
        if method == 'uncalibrated':
            results = get_uncalibrated_confidences(model_adapter, data)
        elif method == 'temperature_scaling':
            results = run_temperature_scaling(model_adapter, data)
        elif method == 'sgc':
            results = run_sgc_calibration(
                model=model,
                model_adapter=model_adapter,
                data=data,
                device=device,
                L=CONFIG['sgc_params']['L'],
                d=CONFIG['sgc_params']['d'],
                seed=seed
            )
        elif method == 'sgc_lite':
            results = run_sgc_lite_calibration(
                model=model,
                model_adapter=model_adapter,
                data=data,
                device=device,
                K=CONFIG['sgc_lite_params']['K'],
                seed=seed
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute ECE for logging
        ece = calculate_ece(
            np.eye(get_num_classes(dataset_name))[results['predictions']] * results['max_probs'][:, None],
            results['labels']
        )
        acc = calculate_accuracy(
            np.eye(get_num_classes(dataset_name))[results['predictions']] * results['max_probs'][:, None],
            results['labels']
        )
        print(f"ECE: {ece:.4f}, Accuracy: {acc:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_experiments():
    """
    Loop over all configurations, save results.
    Save format: {model}_{dataset}_seed{N}_{method}_per_sample.npz
    """
    for model_name in CONFIG['models']:
        for dataset_name in CONFIG['datasets']:
            for seed in CONFIG['seeds']:
                for method in CONFIG['methods']:
                    # Check if already exists
                    filename = CONFIG['output_dir'] / f"{model_name}_{dataset_name}_seed{seed}_{method}_per_sample.npz"
                    if filename.exists():
                        print(f"Skipping {filename.name} (already exists)")
                        continue
                    
                    # Run experiment
                    results = run_single_experiment(model_name, dataset_name, seed, method)
                    
                    if results is not None:
                        # Save results
                        np.savez(
                            filename,
                            max_probs=results['max_probs'],
                            correct=results['correct'],
                            predictions=results['predictions'],
                            labels=results['labels']
                        )
                        print(f"Saved: {filename}")
                    else:
                        print(f"Failed: {filename}")


# ============================================================================
# Cell 7: Data Aggregation for Plotting
# ============================================================================

def load_and_group_results(results_dir: Path) -> Dict[Tuple[str, str, str], List[Dict]]:
    """
    Group saved .npz files by (model, dataset, method).
    
    Returns:
        dict[(model, dataset, method)] -> list of per-seed data
    """
    grouped = defaultdict(list)
    
    for npz_file in results_dir.glob("*_per_sample.npz"):
        # Parse filename: {model}_{dataset}_seed{N}_{method}_per_sample.npz
        parts = npz_file.stem.replace('_per_sample', '').split('_')
        
        # Find seed index
        seed_idx = None
        for i, part in enumerate(parts):
            if part.startswith('seed'):
                seed_idx = i
                break
        
        if seed_idx is None:
            continue
        
        # Reconstruct model, dataset, method
        model_parts = parts[:seed_idx]
        seed_str = parts[seed_idx]
        method_parts = parts[seed_idx+1:]
        
        # Handle model names with underscores (e.g., resnet50, densenet121)
        if len(model_parts) == 1:
            model_name = model_parts[0]
            dataset_name = None
        elif len(model_parts) == 2:
            model_name = model_parts[0]
            dataset_name = model_parts[1]
        else:
            # More complex parsing
            model_name = model_parts[0]
            dataset_name = '_'.join(model_parts[1:])
        
        # Try to identify dataset from known list
        known_datasets = ['cifar10', 'cifar100', 'tiny_imagenet', 'tinyimagenet']
        for ds in known_datasets:
            if ds in '_'.join(model_parts):
                dataset_name = ds
                # Remove dataset from model_parts
                model_parts = [p for p in model_parts if p != ds.replace('_', '')]
                model_name = model_parts[0] if model_parts else 'unknown'
                break
        
        method_name = '_'.join(method_parts)
        
        # Load data
        data = np.load(npz_file)
        grouped[(model_name, dataset_name, method_name)].append({
            'seed': seed_str,
            'max_probs': data['max_probs'],
            'correct': data['correct'],
            'predictions': data['predictions'],
            'labels': data['labels']
        })
    
    return dict(grouped)


def compute_binned_statistics(
    grouped_data: List[Dict],
    n_bins: int = 10,
    bin_type: str = 'equal_width'
) -> Dict[str, np.ndarray]:
    """
    For each group:
    1. Compute bin accuracies per seed
    2. Aggregate: mean ± std across seeds
    
    Returns:
        dict with bin_centers, mean_acc, std_acc, mean_conf, sample_counts, ece_mean, ece_std
    """
    all_max_probs = [d['max_probs'] for d in grouped_data]
    all_correct = [d['correct'] for d in grouped_data]
    
    # Compute bins
    if bin_type == 'equal_width':
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else:  # equal_samples
        # Use percentiles across all seeds
        all_probs_flat = np.concatenate(all_max_probs)
        bin_edges = np.percentile(all_probs_flat, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute statistics per seed
    per_seed_stats = []
    per_seed_eces = []
    
    for max_probs, correct in zip(all_max_probs, all_correct):
        # Bin assignments
        bin_indices = np.digitize(max_probs, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute per-bin statistics
        bin_accs = []
        bin_confs = []
        bin_counts = []
        
        for b in range(n_bins):
            mask = bin_indices == b
            if np.any(mask):
                bin_accs.append(np.mean(correct[mask]))
                bin_confs.append(np.mean(max_probs[mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_accs.append(np.nan)
                bin_confs.append(np.nan)
                bin_counts.append(0)
        
        per_seed_stats.append({
            'acc': np.array(bin_accs),
            'conf': np.array(bin_confs),
            'count': np.array(bin_counts)
        })
        
        # Compute ECE for this seed
        ece = np.nansum(bin_counts / len(max_probs) * np.abs(np.array(bin_accs) - np.array(bin_confs)))
        per_seed_eces.append(ece)
    
    # Aggregate across seeds
    mean_acc = np.nanmean([s['acc'] for s in per_seed_stats], axis=0)
    std_acc = np.nanstd([s['acc'] for s in per_seed_stats], axis=0)
    mean_conf = np.nanmean([s['conf'] for s in per_seed_stats], axis=0)
    sample_counts = np.mean([s['count'] for s in per_seed_stats], axis=0)
    
    ece_mean = np.mean(per_seed_eces)
    ece_std = np.std(per_seed_eces)
    
    return {
        'bin_centers': bin_centers,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'mean_conf': mean_conf,
        'sample_counts': sample_counts,
        'ece_mean': ece_mean,
        'ece_std': ece_std,
        'n_seeds': len(grouped_data)
    }


# ============================================================================
# Cell 8: Plotting Functions
# ============================================================================

def plot_single_method_reliability(
    stats: Dict[str, np.ndarray],
    title: str,
    save_path: Path,
    figsize: Tuple[float, float] = (3.5, 3.5)
) -> None:
    """
    Publication-quality reliability diagram for one method.
    - Equal-width bins
    - Error bars (±1 std across seeds)
    - Histogram of sample counts at bottom
    - ECE ± std in title
    - 300 DPI, PDF output
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
    
    bin_centers = stats['bin_centers']
    mean_acc = stats['mean_acc']
    std_acc = stats['std_acc']
    mean_conf = stats['mean_conf']
    sample_counts = stats['sample_counts']
    ece_mean = stats['ece_mean']
    ece_std = stats['ece_std']
    
    # Top plot: Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration', alpha=0.7)
    ax1.errorbar(
        mean_conf, mean_acc,
        yerr=std_acc,
        fmt='o-',
        capsize=3,
        capthick=1.5,
        linewidth=1.5,
        markersize=5,
        label='Model'
    )
    ax1.set_ylabel('Empirical Accuracy', fontsize=11)
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Title with ECE
    title_with_ece = f"{title}\nECE = {ece_mean:.2%} ± {ece_std:.2%}"
    ax1.set_title(title_with_ece, fontsize=10)
    
    # Bottom plot: Histogram
    ax2.bar(bin_centers, sample_counts, width=0.08, alpha=0.6, color='gray')
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_xlabel('Confidence', fontsize=11)
    ax2.set_xlim([0, 1])
    
    plt.tight_layout()
    
    # Save as both PNG and PDF
    plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path.with_suffix('.png')} and {save_path.with_suffix('.pdf')}")


def plot_method_comparison(
    all_stats: Dict[str, Dict[str, np.ndarray]],
    model: str,
    dataset: str,
    save_path: Path,
    figsize: Tuple[float, float] = (14, 3.5)
) -> None:
    """
    Side-by-side comparison: Uncalibrated | Temp Scaling | SGC | SGC-Lite
    - 1x4 subplot grid
    - Shared y-axis scale
    - Method name + ECE in subplot titles
    """
    method_order = ['uncalibrated', 'temperature_scaling', 'sgc', 'sgc_lite']
    method_labels = {
        'uncalibrated': 'Uncalibrated',
        'temperature_scaling': 'Temperature Scaling',
        'sgc': 'SGC (RGCL)',
        'sgc_lite': 'SGC-Lite (RGCC)'
    }
    
    fig, axes = plt.subplots(1, len(method_order), figsize=figsize, sharey=True)
    
    for idx, method in enumerate(method_order):
        if method not in all_stats:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].set_title(method_labels.get(method, method), fontsize=10)
            continue
        
        stats = all_stats[method]
        ax = axes[idx]
        
        bin_centers = stats['bin_centers']
        mean_acc = stats['mean_acc']
        std_acc = stats['std_acc']
        mean_conf = stats['mean_conf']
        ece_mean = stats['ece_mean']
        ece_std = stats['ece_std']
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        # Reliability curve
        ax.errorbar(
            mean_conf, mean_acc,
            yerr=std_acc,
            fmt='o-',
            capsize=2,
            capthick=1,
            linewidth=1.2,
            markersize=4
        )
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.set_ylabel('Empirical Accuracy', fontsize=11)
        ax.set_xlabel('Confidence', fontsize=10)
        
        # Title with ECE
        title = f"{method_labels.get(method, method)}\nECE = {ece_mean:.2%} ± {ece_std:.2%}"
        ax.set_title(title, fontsize=10)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison: {save_path.with_suffix('.png')} and {save_path.with_suffix('.pdf')}")


# ============================================================================
# Cell 9: Generate All Figures
# ============================================================================

if __name__ == '__main__':
    # Load and group results
    grouped_results = load_and_group_results(CONFIG['output_dir'])
    
    print(f"Loaded {len(grouped_results)} experiment groups")
    for key, data_list in list(grouped_results.items())[:5]:
        print(f"  {key}: {len(data_list)} seeds")
    
    # Generate plots for each (model, dataset) combination
    for (model, dataset, method), data_list in grouped_results.items():
        if len(data_list) == 0:
            continue
        
        # Compute aggregated statistics
        stats = compute_binned_statistics(data_list, n_bins=CONFIG['n_bins'])
        
        # Plot single method
        title = f"{model.upper()} on {dataset.upper()}\n{method.upper()}"
        save_path = CONFIG['figures_dir'] / f"{model}_{dataset}_{method}_aggregated"
        plot_single_method_reliability(stats, title, save_path)
    
    # Generate comparison plots
    for model in CONFIG['models']:
        for dataset in CONFIG['datasets']:
            # Collect all methods for this (model, dataset)
            all_stats = {}
            for method in CONFIG['methods']:
                key = (model, dataset, method)
                if key in grouped_results and len(grouped_results[key]) > 0:
                    all_stats[method] = compute_binned_statistics(
                        grouped_results[key],
                        n_bins=CONFIG['n_bins']
                    )
            
            if len(all_stats) > 0:
                save_path = CONFIG['figures_dir'] / f"{model}_{dataset}_comparison"
                plot_method_comparison(all_stats, model, dataset, save_path)
    
    print(f"\nAll plots saved to: {CONFIG['figures_dir']}")

