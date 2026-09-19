#!/usr/bin/env python3
"""
Random Layer Ablation experiments using the established geometric calibration
pipeline with fixed target dimensions.

MODES
-----
1) LAYER MODE (default):
   - Extract features from random model layers.
   - SPP+JL compression (16x CIFAR10, 32x CIFAR100) -> concatenate -> project to target dim.

2) PIXEL MODE (--pixel-mode):
   - Use raw input pixels as features (e.g., 3072 dims for CIFAR).
   - Standard: optional Stage 1 compression to mimic SPP+JL (16x/32x) then project.
   - Raw: --no-first-stage-compression to skip Stage 1 (pixels -> project).
   - If target_dim equals the current dim (e.g., 3072), apply identity projection.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import gc
from typing import Any, Dict, List, Sequence
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import normalize
from Data.svhn import get_test_loader as svhn_test
from Data.tiny_imagenet import get_data_loader as tiny_imagenet_get_data_loader

# === Baseline calibration imports (added) ===
from Experiments.layer_selection import (
    run_standard_baselines,
    UncertaintyMetrics,
    NumpyEncoder,
)
from Calibrators.tulip import select_tulip_layers  # Add this import

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Experiments.compare_dac_geometric import (  # noqa: E402
    calculate_accuracy,
    calculate_ece,
    combine_features,
    extract_features_from_multiple_layers,
    get_layer_names_for_model,
)
from Experiments.multi_layer_ensemble import discover_model_layers  # noqa: E402
from Experiments.run_post_hoc_calibration import (  # noqa: E402
    PyTorchModelAdapter,
    construct_model_path,
    get_data_loaders,
    load_trained_model,
)
from Calibrators.geometric_calibrator_new import GeometricCalibrator  # noqa: E402
from utils.compression_utils import SmartCompression  # noqa: E402
from Calibrators.calibration_utils import (  # noqa: E402
    compute_ood_auroc,
    compute_fpr_at_tpr,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
from utils.logging_config import get_logger
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_model_hooks(model):
    """Remove all forward hooks from model and its modules."""
    model._forward_hooks.clear()
    for module in model.modules():
        module._forward_hooks.clear()


def make_json_serializable(obj: Any):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_incremental_results(output_path: str, results: Dict[str, Any]):
    """
    Save results incrementally to avoid data loss if job crashes.
    Uses atomic file replacement for safety.
    """
    temp_path = output_path + ".tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(make_json_serializable(results), f, indent=2)
        os.replace(temp_path, output_path)  # Atomic operation
        logger.debug(f"Saved incremental results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save incremental results: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise


def build_final_results(
    existing_results: Dict[str, Any],
    geo_best_results: List[Dict[str, Any]],
    best_per_target_dim: Dict[int, Dict[str, Any]],
    geo_comb_results: List[Dict[str, Any]],
    random_ablation_results: Dict[str, Any],
    args,
    baseline_results: Dict[str, Any] = None,
    tulip_best_results: List[Dict[str, Any]] = None,
    tulip_comb_results: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the final results dictionary from current state.
    Used for incremental saving.
    
    Note: geo_best_results and tulip_best_results may be empty lists
    if --skip-single-layers flag is used. This is intentional and valid.
    """
    final_results = {
        "experiment_info": {
            "dataset": args.dataset,
            "model_name": args.model_name,
            "training_method": args.training_method,
            "seed": args.seed,
            "mode": "layers",
            "no_first_stage_compression": False,
        },
        # === Baseline calibration results (added) ===
        "baselines": baseline_results if baseline_results else {},
        "geo_best": {
            "all_single_layers": geo_best_results,
            "best_per_target_dim": best_per_target_dim,
            "note": "Oracle: best single layer from DAC fixed 4 layers",
        },
        "geo_comb": {
            "results": geo_comb_results,
            "note": "Fixed 4 layers concatenated (DAC paper selection)",
        },
        "tulip_best": {
            "all_single_layers": tulip_best_results if tulip_best_results else [],
            "note": "Oracle: best single layer from TULIP selected layers",
        },
        "tulip_comb": {
            "results": tulip_comb_results if tulip_comb_results else [],
            "note": "All TULIP layers concatenated (TULIP strategic selection)",
        },
        "random_layer_ablation": {
            "experiment_config": random_ablation_results["experiment_config"],
            "results_by_target_dim": random_ablation_results["results_by_target_dim"],
        }
    }
    
    # Add comparison if computed
    if args.num_random_seeds > 1:
        if "comparison_to_fixed" in random_ablation_results:
            final_results["random_layer_ablation"]["comparison_to_fixed"] = random_ablation_results["comparison_to_fixed"]
    
    # Merge with existing results if any (like DAC baselines)
    if existing_results:
        existing_results.update(final_results)
        final_results = existing_results
    
    return final_results


def dataloader_from_numpy(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False, drop_last: bool = False, num_workers: int = 0
) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def extract_raw_from_loader(loader: DataLoader) -> (np.ndarray, np.ndarray):
    xs, ys = [], []
    for batch in loader:
        data, labels = batch[:2]
        xs.append(data.numpy())
        ys.append(labels.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def normalize_discovered_layers(model: torch.nn.Module, discovered: List[Dict[str, Any]]):
    """
    Ensure each discovered layer has a descriptive name derived from named_modules.
    """
    idx_to_name = {i: name for i, (name, _) in enumerate(model.named_modules())}
    normalized = []
    for entry in discovered:
        idx = entry.get("idx")
        entry = dict(entry)
        entry["name"] = idx_to_name.get(idx, entry.get("name", f"layer_{idx}"))
        normalized.append(entry)
    return normalized


def group_layers_by_block(discovered_layers: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Group layers by block using name patterns.
    Handles both ResNet (layer1, layer2, ...) and DenseNet (dense1, dense2, ...).
    """
    groups: Dict[str, List[int]] = {
        "block1": [], 
        "block2": [], 
        "block3": [], 
        "block4": []
    }
    
    for layer in discovered_layers:
        name = layer.get("name", "")
        idx = layer.get("idx")
        if idx is None:
            continue
        
        lowered = name.lower()
        
        # ResNet patterns: layer1, layer2, layer3, layer4
        # DenseNet patterns: dense1, dense2, dense3, dense4 or denseblock1, denseblock2, etc.
        if "layer1" in lowered or "dense1" in lowered or "denseblock1" in lowered:
            groups["block1"].append(idx)
        elif "layer2" in lowered or "dense2" in lowered or "denseblock2" in lowered:
            groups["block2"].append(idx)
        elif "layer3" in lowered or "dense3" in lowered or "denseblock3" in lowered:
            groups["block3"].append(idx)
        elif "layer4" in lowered or "dense4" in lowered or "denseblock4" in lowered:
            groups["block4"].append(idx)
    
    # Only return groups that have layers
    return {k: v for k, v in groups.items() if v}


def get_block_for_idx(idx: int, block_groups: Dict[str, List[int]]) -> str:
    """
    Get the block name for a given layer index.
    Returns the first block that contains this index, or None if not found.
    """
    for block_name, indices in block_groups.items():
        if idx in indices:
            return block_name
    return None


def probe_layer_dimensions(
    model: torch.nn.Module,
    discovered_layers: List[Dict[str, Any]],
    device: torch.device,
    dataset: str = "cifar10",
    model_name: str = None,
) -> Dict[int, int]:
    """
    Probe dimensions for each discovered layer using a single dummy batch.
    Returns dict mapping layer idx -> output dimension after SPP.
    """
    from Experiments.compare_dac_geometric import extract_features_from_multiple_layers
    
    # Determine input shape based on dataset and model
    is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
    if is_dinov2:
        sample_shape = (2, 3, 224, 224)  # DINOv2 requires 224x224
    elif dataset.lower() in ["cifar10", "cifar100"]:
        sample_shape = (2, 3, 32, 32)  # batch=2 to avoid batch norm issues
    elif dataset.lower() in ["tiny_imagenet", "tinyimagenet"]:
        sample_shape = (2, 3, 64, 64)
    else:
        sample_shape = (2, 3, 224, 224)
    
    dummy_input = torch.randn(sample_shape, device=device)
    dummy_loader = DataLoader(
        TensorDataset(dummy_input.cpu(), torch.zeros(sample_shape[0])),
        batch_size=sample_shape[0],
        drop_last=False,
    )
    
    layer_dims = {}
    for layer in discovered_layers:
        layer_name = layer["name"]
        layer_idx = layer["idx"]
        try:
            clear_model_hooks(model)
            features = extract_features_from_multiple_layers(
                model, dummy_loader, [layer_name], device, spp_only=True, pooling_mode="max"
            )
            if features and len(features) > 0:
                layer_dims[layer_idx] = features[0].shape[1]
        except Exception as e:
            logger.debug(f"Failed to probe layer {layer_name}: {e}")
            layer_dims[layer_idx] = -1  # Mark as unknown/failed
        finally:
            clear_model_hooks(model)
    
    return layer_dims


def select_random_layers(
    candidates: Sequence[int], num_layers: int, rng: random.Random
) -> List[int]:
    if len(candidates) < num_layers:
        raise ValueError(f"Requested {num_layers} layers but only {len(candidates)} available")
    return sorted(rng.sample(list(candidates), num_layers))


def compute_trial_statistics(trials: List[Dict[str, Any]], group_by: List[str]) -> Dict[str, Any]:
    """
    Compute statistics across random trials.
    
    Args:
        trials: List of trial results
        group_by: Keys to group by (e.g., ['num_layers'] for fixed target_dim)
    
    Returns:
        Dictionary with aggregated statistics
    """
    if not trials:
        return {}
    
    # Convert to DataFrame for easy grouping
    df = pd.DataFrame(trials)
    
    # Group by specified keys
    grouped = df.groupby(group_by)
    
    stats = {}
    for group_key, group_data in grouped:
        # Handle single vs multiple grouping keys
        if len(group_by) == 1:
            key_str = f"{group_by[0]}_{group_key}"
        else:
            key_str = "_".join([f"{k}_{v}" for k, v in zip(group_by, group_key)])
        
        n_trials = len(group_data)
        
        stats[key_str] = {
            'n_trials': n_trials,
            'ece': {
                'mean': float(group_data['ece'].mean()),
                'std': float(group_data['ece'].std()) if n_trials > 1 else 0.0,
                'min': float(group_data['ece'].min()),
                'max': float(group_data['ece'].max()),
                'median': float(group_data['ece'].median()),
            },
            'accuracy': {
                'mean': float(group_data['accuracy'].mean()),
                'std': float(group_data['accuracy'].std()) if n_trials > 1 else 0.0,
                'min': float(group_data['accuracy'].min()),
                'max': float(group_data['accuracy'].max()),
            },
            'samples_per_second': {
                'mean': float(group_data['samples_per_second'].mean()) if 'samples_per_second' in group_data.columns else 0.0,
                'std': float(group_data['samples_per_second'].std()) if n_trials > 1 and 'samples_per_second' in group_data.columns else 0.0,
                'min': float(group_data['samples_per_second'].min()) if 'samples_per_second' in group_data.columns else 0.0,
                'max': float(group_data['samples_per_second'].max()) if 'samples_per_second' in group_data.columns else 0.0,
                'median': float(group_data['samples_per_second'].median()) if 'samples_per_second' in group_data.columns else 0.0,
            },
            'config': dict(zip(group_by, group_key if isinstance(group_key, tuple) else [group_key])),
        }
        
        # Add best trial info
        best_trial_idx = group_data['ece'].idxmin()
        best_trial = group_data.loc[best_trial_idx]
        stats[key_str]['best_trial'] = {
            'trial_idx': int(best_trial.get('trial_idx', -1)),
            'trial_seed': int(best_trial.get('trial_seed', -1)),
            'ece': float(best_trial['ece']),
            'accuracy': float(best_trial['accuracy']),
            'samples_per_second': float(best_trial.get('samples_per_second', 0.0)),
        }
        
        # Add worst trial info
        worst_trial_idx = group_data['ece'].idxmax()
        worst_trial = group_data.loc[worst_trial_idx]
        stats[key_str]['worst_trial'] = {
            'trial_idx': int(worst_trial.get('trial_idx', -1)),
            'trial_seed': int(worst_trial.get('trial_seed', -1)),
            'ece': float(worst_trial['ece']),
            'accuracy': float(worst_trial['accuracy']),
            'samples_per_second': float(worst_trial.get('samples_per_second', 0.0)),
        }
        
        # Coefficient of variation (relative variability)
        cv = (stats[key_str]['ece']['std'] / stats[key_str]['ece']['mean'] * 100) if stats[key_str]['ece']['mean'] > 0 else 0.0
        stats[key_str]['ece']['coefficient_of_variation_pct'] = cv
    
    return stats


def print_trial_statistics(stats: Dict[str, Any], strategy_name: str):
    """Print trial statistics in a readable format."""
    if not stats:
        return
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TRIAL STATISTICS: {strategy_name.upper()}")
    logger.info(f"{'='*80}")
    logger.info(f"{'Config':<30} {'N':<4} {'Mean ECE':<12} {'Std':<10} {'Min':<10} {'Max':<10} {'CV%':<8}")
    logger.info("-"*80)
    
    for key, data in sorted(stats.items()):
        config_str = ", ".join([f"{k}={v}" for k, v in data['config'].items()])
        n = data['n_trials']
        mean_ece = data['ece']['mean'] * 100
        std_ece = data['ece']['std'] * 100
        min_ece = data['ece']['min'] * 100
        max_ece = data['ece']['max'] * 100
        cv = data['ece']['coefficient_of_variation_pct']
        
        logger.info(f"{config_str:<30} {n:<4} {mean_ece:>10.3f}% {std_ece:>8.3f}% {min_ece:>8.3f}% {max_ece:>8.3f}% {cv:>6.2f}%")
    
    # Overall insights
    all_cvs = [data['ece']['coefficient_of_variation_pct'] for data in stats.values()]
    avg_cv = np.mean(all_cvs)
    max_cv = np.max(all_cvs)
    
    logger.info(f"\nVariability Analysis:")
    logger.info(f"  Average CV: {avg_cv:.2f}%")
    logger.info(f"  Max CV: {max_cv:.2f}%")
    
    if avg_cv < 5:
        logger.info(f"  ✓ VERY STABLE: Random selection is highly consistent")
    elif avg_cv < 10:
        logger.info(f"  ✓ STABLE: Random selection shows acceptable variance")
    elif avg_cv < 20:
        logger.info(f"  ⚠ MODERATE VARIANCE: Consider more trials for reliability")
    else:
        logger.info(f"  ❌ HIGH VARIANCE: Random selection is unstable, need more trials or better strategy")


# ---------------------------------------------------------------------------
# Core trial logic
# ---------------------------------------------------------------------------
def run_random_sampling_trial(
    model: torch.nn.Module,
    model_adapter: PyTorchModelAdapter,
    device: torch.device,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    target_dim: int,
    selected_layer_indices: List[int] = None,
    selected_layer_names: List[str] = None,
    all_discovered_layers: List[Dict[str, Any]] = None,
    batch_size: int = 128,
    dataset: str = None,
    seed: int = 0,
    pixel_mode: bool = False,
    no_first_stage_compression: bool = False,
    ood_raw: np.ndarray = None,
) -> Dict[str, Any]:
    
    # [Layer Name Resolution Logic - Same as before]
    if pixel_mode:
        layer_names = ["pixels"]
        logger.info("PIXEL MODE: using raw image as features.")
    elif selected_layer_names is not None:
        layer_names = selected_layer_names
        logger.info(f"Selected layers (by name): {layer_names}")
    elif selected_layer_indices is not None and all_discovered_layers is not None:
        idx_to_position = {layer["idx"]: pos for pos, layer in enumerate(all_discovered_layers)}
        layer_positions = [idx_to_position[idx] for idx in selected_layer_indices]
        layer_names = [all_discovered_layers[pos]["name"] for pos in layer_positions]
        logger.info(f"Selected layer indices (model): {selected_layer_indices}")
        logger.info(f"Selected layers: {layer_names}")
    else:
        raise ValueError("Must provide either pixel_mode or layer selection.")

    # Dataloaders
    # Use drop_last=False for train/val - SPP pooling handles variable batch sizes
    train_loader = dataloader_from_numpy(train_raw, train_labels, batch_size, drop_last=False, num_workers=0)
    val_loader = dataloader_from_numpy(val_raw, val_labels, batch_size, drop_last=False, num_workers=0)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    ood_loader = None
    if ood_raw is not None:
        ood_loader = DataLoader(
            TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
            batch_size=batch_size,
            shuffle=False,
        )

    t_start = time.perf_counter()
    extraction_time = 0.0
    compression_time = 0.0
    spp_jl_ratio = 1.0
    ood_features_compressed = None

    if pixel_mode:
        # [PIXEL MODE CODE - UNCHANGED]
        logger.info(f"Using raw pixels. Input shape: {train_raw.shape}")
        t0 = time.perf_counter()
        train_features = torch.from_numpy(train_raw).float().view(len(train_raw), -1)
        val_features = torch.from_numpy(val_raw).float().view(len(val_raw), -1)
        test_features = torch.from_numpy(test_raw).float().view(len(test_raw), -1)
        if ood_raw is not None:
            ood_features = torch.from_numpy(ood_raw).float().view(len(ood_raw), -1)
        extraction_time = time.perf_counter() - t0
        concatenated_dim = train_features.shape[1]
        layer_contributions = {"pixels": {"dims": int(concatenated_dim), "percentage": 100.0}}

        if no_first_stage_compression:
            spp_jl_ratio = 1.0
        else:
            spp_jl_ratio = 32.0 if dataset == "cifar100" else 16.0
    
    else:
        # === NEW: Sequential Memory-Efficient Pipeline ===
        logger.info("Extracting features with SPP only (Sequential Mode)")
        
        # 1. Probing Step: Get dimensions without loading full dataset
        # We run just ONE batch to get feature shapes for logging percentages
        t_probe = time.perf_counter()
        dummy_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_raw[:batch_size]), torch.zeros(batch_size)), 
            batch_size=batch_size,
            drop_last=True  # Ensure consistent batch size to avoid dimension mismatches
        )
        # Extract each layer individually to avoid concatenation errors with different spatial dimensions
        layer_dims = []
        for lname in layer_names:
            clear_model_hooks(model)  # Clear hooks before extraction to prevent leakage
            dummy_feature_single = extract_features_from_multiple_layers(
                model, dummy_loader, [lname], device, spp_only=True, pooling_mode="max"
            )
            layer_dims.append(dummy_feature_single[0].shape[1])
            del dummy_feature_single
        total_dims = sum(layer_dims)
        del dummy_loader
        
        # Log contributions upfront
        layer_contributions = {
            layer_names[i]: {
                "dims": int(layer_dims[i]),
                "percentage": float(layer_dims[i] / total_dims * 100),
            } for i in range(len(layer_names))
        }
        logger.info("Layer contributions (calculated from probe):")
        for lname, contrib in layer_contributions.items():
            logger.info(f"  {lname}: {contrib['dims']} dims ({contrib['percentage']:.1f}%)")

        # 2. Initialize Accumulators
        train_features_sum = np.zeros((len(train_raw), target_dim), dtype=np.float32)
        val_features_sum = np.zeros((len(val_raw), target_dim), dtype=np.float32)
        test_features_sum = np.zeros((len(test_raw), target_dim), dtype=np.float32)

        t1 = time.perf_counter()

        # 3. Sequential Loop: Extract -> Project -> Sum -> Delete
        for i, lname in enumerate(layer_names):
            logger.info(f"Processing layer {lname} ({layer_dims[i]} dims)...")
            
            # A. Extract FULL dataset for THIS LAYER ONLY
            # This minimizes memory usage to Max(Layer_N) instead of Sum(All_Layers)
            clear_model_hooks(model)  # Clear hooks before extraction to prevent leakage
            t_ex = time.perf_counter()
            single_layer_train = extract_features_from_multiple_layers(
                model, train_loader, [lname], device, spp_only=True, pooling_mode="max"
            )[0]
            single_layer_val = extract_features_from_multiple_layers(
                model, val_loader, [lname], device, spp_only=True, pooling_mode="max"
            )[0]
            single_layer_test = extract_features_from_multiple_layers(
                model, test_loader, [lname], device, spp_only=True, pooling_mode="max"
            )[0]
            extraction_time += (time.perf_counter() - t_ex)

            # B. Project immediately
            compressor = SmartCompression(
                method="random_projection",
                target_dims=target_dim,
                random_state=seed + i, # Crucial: Independent seed per layer
            )
            
            # Note: numpy() conversion handles both Torch tensors and numpy arrays
            train_proj = compressor(single_layer_train.numpy() if hasattr(single_layer_train, 'numpy') else single_layer_train, train=True)
            val_proj = compressor(single_layer_val.numpy() if hasattr(single_layer_val, 'numpy') else single_layer_val, train=False)
            test_proj = compressor(single_layer_test.numpy() if hasattr(single_layer_test, 'numpy') else single_layer_test, train=False)

            # C. Add to sum
            train_features_sum += train_proj
            val_features_sum += val_proj
            test_features_sum += test_proj

            # D. AGGRESSIVE CLEANUP
            del single_layer_train, single_layer_val, single_layer_test
            del train_proj, val_proj, test_proj
            gc.collect()

        compression_time = time.perf_counter() - t1

        # 4. Final Normalization
        logger.info("Applying final L2 normalization...")
        train_features_compressed = normalize(train_features_sum, axis=1, norm="l2")
        val_features_compressed = normalize(val_features_sum, axis=1, norm="l2")
        test_features_compressed = normalize(test_features_sum, axis=1, norm="l2")

        del train_features_sum, val_features_sum, test_features_sum
        gc.collect()

        concatenated_dim = total_dims
        actual_output_dim = target_dim
        actual_compression_ratio = total_dims / target_dim

    # [PIXEL MODE POST-COMPRESSION LOGIC - Same as before]
    if pixel_mode:
        t1 = time.perf_counter()
        if target_dim != concatenated_dim:
            logger.info(f"Applying post-concatenation compression: {concatenated_dim} -> {target_dim}")
            compressor = SmartCompression(method="random_projection", target_dims=target_dim, random_state=seed)
            train_features_compressed = compressor(train_features.numpy(), train=True)
            val_features_compressed = compressor(val_features.numpy(), train=False)
            test_features_compressed = compressor(test_features.numpy(), train=False)
            if ood_raw is not None:
                ood_features_compressed = compressor(ood_features.numpy(), train=False)
        else:
             # Identity logic...
             train_features_compressed = train_features.numpy()
             val_features_compressed = val_features.numpy()
             test_features_compressed = test_features.numpy()
             if ood_raw is not None:
                 ood_features_compressed = ood_features.numpy()
        compression_time += time.perf_counter() - t1
        actual_output_dim = train_features_compressed.shape[1]

    # [GEOMETRIC CALIBRATION STEP - Same as before]
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features_compressed,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,
        device=str(device),
    )

    t2 = time.perf_counter()
    geo_cal.fit(
        X_val_embed=val_features_compressed,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size,
    )
    fit_time = time.perf_counter() - t2

    t3 = time.perf_counter()
    calibrated_probs = geo_cal.calibrate_batched(
        X_test_embed=test_features_compressed,
        X_test_original=test_raw,
        batch_size=batch_size,
    )
    calibrate_time = time.perf_counter() - t3

    ece = float(calculate_ece(calibrated_probs, test_labels))
    acc = float(calculate_accuracy(calibrated_probs, test_labels))

    # OOD evaluation (SVHN as OOD for CIFAR datasets)
    ood_auroc = None
    ood_fpr95 = None
    ood_diagnostics = {}  # Initialize to empty dict in case OOD evaluation is skipped
    if ood_raw is not None:
        # Compute OOD embeddings using the same memory-efficient pipeline as ID data
        if not pixel_mode and ood_loader is not None and ood_features_compressed is None:
            logger.info("Extracting features for OOD data (layer mode)...")
            
            # Initialize accumulator for OOD features
            ood_features_sum = np.zeros((len(ood_raw), target_dim), dtype=np.float32)
            
            # Sequential processing: extract -> project -> sum -> delete (same as ID data)
            for i, lname in enumerate(layer_names):
                logger.info(f"Processing OOD layer {lname}...")
                clear_model_hooks(model)
                
                single_layer_ood = extract_features_from_multiple_layers(
                    model, ood_loader, [lname], device, spp_only=True, pooling_mode="max"
                )[0]
                
                # Use same seed as ID data for this layer to ensure consistent projection
                compressor_ood = SmartCompression(
                    method="random_projection",
                    target_dims=target_dim,
                    random_state=seed + i,
                )
                
                ood_proj = compressor_ood(
                    single_layer_ood.numpy() if hasattr(single_layer_ood, "numpy") else single_layer_ood,
                    train=True,
                )
                
                ood_features_sum += ood_proj
                
                # Aggressive cleanup after each layer
                del single_layer_ood, ood_proj, compressor_ood
                gc.collect()
                
                # Optional: clear CUDA cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Final normalization
            ood_features_compressed = normalize(ood_features_sum, axis=1, norm="l2")
            del ood_features_sum
            gc.collect()

        if ood_features_compressed is not None:
            logger.info("Running geometric calibration on OOD data for OOD metrics...")
            ood_probs = geo_cal.calibrate_batched(
                X_test_embed=ood_features_compressed,
                X_test_original=ood_raw,
                batch_size=batch_size,
            )
            id_scores = 1.0 - np.max(calibrated_probs, axis=1)
            ood_scores = 1.0 - np.max(ood_probs, axis=1)
            ood_auroc = compute_ood_auroc(id_scores, ood_scores)
            ood_fpr95 = compute_fpr_at_tpr(id_scores, ood_scores)

            # Diagnostic: Analyze probability distribution patterns
            id_max_probs = np.max(calibrated_probs, axis=1)
            ood_max_probs = np.max(ood_probs, axis=1)

            # Entropy calculations (higher = more uniform/uncertain)
            id_entropies = -np.sum(calibrated_probs * np.log(calibrated_probs + 1e-10), axis=1)
            ood_entropies = -np.sum(ood_probs * np.log(ood_probs + 1e-10), axis=1)
            max_entropy = np.log(calibrated_probs.shape[1])  # log(num_classes)

            # Uncertainty scores (used for OOD detection)
            id_uncertainties = 1.0 - id_max_probs
            ood_uncertainties = 1.0 - ood_max_probs

            # Compute statistics
            ood_diagnostics = {
                # ID (in-distribution) statistics
                "id_max_prob_mean": float(id_max_probs.mean()),
                "id_max_prob_std": float(id_max_probs.std()),
                "id_max_prob_median": float(np.median(id_max_probs)),
                "id_entropy_mean": float(id_entropies.mean()),
                "id_entropy_std": float(id_entropies.std()),
                "id_entropy_normalized": float(id_entropies.mean() / max_entropy),  # 0=peaked, 1=uniform
                "id_uncertainty_mean": float(id_uncertainties.mean()),
                "id_uncertainty_std": float(id_uncertainties.std()),

                # OOD (out-of-distribution) statistics
                "ood_max_prob_mean": float(ood_max_probs.mean()),
                "ood_max_prob_std": float(ood_max_probs.std()),
                "ood_max_prob_median": float(np.median(ood_max_probs)),
                "ood_entropy_mean": float(ood_entropies.mean()),
                "ood_entropy_std": float(ood_entropies.std()),
                "ood_entropy_normalized": float(ood_entropies.mean() / max_entropy),
                "ood_uncertainty_mean": float(ood_uncertainties.mean()),
                "ood_uncertainty_std": float(ood_uncertainties.std()),

                # Separation metrics
                "max_prob_gap": float(id_max_probs.mean() - ood_max_probs.mean()),  # Positive = good separation
                "entropy_gap": float(ood_entropies.mean() - id_entropies.mean()),   # Positive = OOD more uncertain
                "uncertainty_gap": float(ood_uncertainties.mean() - id_uncertainties.mean()),

                # Distribution overlap
                "max_prob_overlap": float(
                    np.minimum(id_max_probs, ood_max_probs[: len(id_max_probs)]).mean()
                ) if len(ood_max_probs) >= len(id_max_probs) else float("nan"),
            }

            logger.info(
                f"  ID  max_prob: {ood_diagnostics['id_max_prob_mean']:.3f} ± "
                f"{ood_diagnostics['id_max_prob_std']:.3f}"
            )
            logger.info(
                f"  OOD max_prob: {ood_diagnostics['ood_max_prob_mean']:.3f} ± "
                f"{ood_diagnostics['ood_max_prob_std']:.3f}"
            )
            logger.info(f"  Separation gap: {ood_diagnostics['max_prob_gap']:.3f} (positive = good)")
            logger.info(
                f"  ID  entropy/max: {ood_diagnostics['id_entropy_normalized']:.3f} (1.0 = uniform)"
            )
            logger.info(
                f"  OOD entropy/max: {ood_diagnostics['ood_entropy_normalized']:.3f}"
            )
        else:
            ood_diagnostics = {}
    total_time = time.perf_counter() - t_start
    samples_per_second = len(test_raw) / total_time if total_time > 0 else 0.0

    result = {
        "layer_names": layer_names,
        "layer_contributions": layer_contributions,
        "spp_jl_compression_ratio": float(spp_jl_ratio),
        "concatenated_dim": int(concatenated_dim),
        "target_dim": int(target_dim),
        "actual_feature_dim": int(actual_output_dim),
        "post_concatenation_compression_ratio": float(actual_compression_ratio),
        "ece": ece,
        "accuracy": acc,
        "ood_auroc": ood_auroc,
        "ood_fpr95": ood_fpr95,
        "ood_diagnostics": ood_diagnostics,
        "extraction_time_s": extraction_time,
        "compression_time_s": compression_time,
        "fit_time_s": fit_time,
        "calibrate_time_s": calibrate_time,
        "total_time_s": total_time,
        "samples_per_second": samples_per_second,
    }
    
    if selected_layer_indices is not None:
        result["selected_layers"] = selected_layer_indices
    
    return result


# ---------------------------------------------------------------------------
# OOD-only evaluation driver
# ---------------------------------------------------------------------------
def run_ood_only_evaluation(args, output_path: str):
    """
    Run ONLY OOD detection evaluation on existing results without recomputing
    the random ablation search. Adds ood_auroc and ood_fpr95 fields in-place.
    """
    logger.info("\n" + "=" * 80)
    logger.info("=== OOD-ONLY MODE ENABLED ===")
    logger.info("=" * 80)
    logger.info(f"Loading existing results from: {output_path}")

    if not os.path.exists(output_path):
        logger.error(f"❌ Results file does not exist for OOD-only mode: {output_path}")
        return

    # Load existing results
    with open(output_path, "r") as f:
        existing_results: Dict[str, Any] = json.load(f)

    logger.info(f"Loaded {len(existing_results)} top-level sections")
    logger.info("Will add OOD metrics to existing experiments")

    # Skip unsupported datasets
    if args.dataset == "svhn":
        logger.info("⚠️ Dataset is SVHN; cannot use SVHN as both ID and OOD. Skipping OOD-only evaluation.")
        return
    if args.dataset not in ["cifar10", "cifar100"] and args.dataset.lower() not in ['tiny_imagenet', 'tinyimagenet']:
        logger.info(f"⚠️ OOD-only mode currently supports only CIFAR-10/100 and Tiny ImageNet. Dataset={args.dataset}. Skipping.")
        return

    # Skip DINOv2 (no fixed layers / inconsistent baselines)
    is_dinov2 = "dino" in args.model_name.lower()
    if is_dinov2:
        logger.info("⚠️ Detected DINOv2 model. Skipping OOD-only evaluation (no fixed layer baselines).")
        return

    logger.info(f"OOD dataset: SVHN")

    # Determine output path for updated results
    if getattr(args, "ood_output_dir", None):
        os.makedirs(args.ood_output_dir, exist_ok=True)
        ood_output_path = os.path.join(args.ood_output_dir, os.path.basename(output_path))
        logger.info(f"OOD results will be saved to: {ood_output_path}")
    else:
        ood_output_path = output_path
        logger.info("OOD results will overwrite the original results file.")

    # Load ID data and model
    device = torch.device(args.device)

    logger.info(f"Loading {args.dataset} train/val/test loaders...")
    # Determine required image size for DINOv2 models
    is_dinov2 = 'dinov2' in args.model_name.lower()
    image_size = 224 if is_dinov2 else None  # None means use default (32 for CIFAR)
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset, args.batch_size, seed=args.seed, image_size=image_size
    )
    train_raw, train_labels = extract_raw_from_loader(train_loader)
    val_raw, val_labels = extract_raw_from_loader(val_loader)
    test_raw, test_labels = extract_raw_from_loader(test_loader)

    logger.info("Loading SVHN test set as OOD dataset...")
    try:
        svhn_loader = svhn_test(batch_size=args.batch_size, shuffle=False)
        ood_raw, _ = extract_raw_from_loader(svhn_loader)
    except Exception as e:
        logger.error(f"❌ Failed to load SVHN for OOD evaluation: {e}")
        return

    # Load model and adapter
    model_path = construct_model_path(
        args.results_base_dir, args.training_method, args.dataset, args.model_name, args.seed
    )
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found at {model_path}")
        return

    model = load_trained_model(model_path, args.model_name, num_classes, device, dataset=args.dataset)
    model_adapter = PyTorchModelAdapter(model, device, args.dataset)

    def update_result_with_ood_metrics(result: Dict[str, Any], section_name: str) -> bool:
        """
        For a single result dict, compute and attach OOD metrics if needed.
        Returns True if metrics were newly computed, False otherwise.
        """
        # Skip only if all OOD outputs (metrics + diagnostics) already exist
        has_auroc = result.get("ood_auroc") is not None
        has_fpr95 = result.get("ood_fpr95") is not None
        has_diag = result.get("ood_diagnostics") is not None
        if has_auroc and has_fpr95 and has_diag:
            return False

        # Determine target_dim
        target_dim_raw = result.get("target_dim")
        if target_dim_raw is None:
            logger.warning(f"[{section_name}] Missing target_dim; skipping result.")
            return False
        try:
            target_dim = int(target_dim_raw)
        except (TypeError, ValueError):
            logger.warning(f"[{section_name}] Invalid target_dim={target_dim_raw}; skipping result.")
            return False

        layer_names = result.get("layer_names")

        # Handle pixel-mode experiments (layer_names == ['pixels'])
        pixel_mode = isinstance(layer_names, list) and len(layer_names) == 1 and layer_names[0] == "pixels"

        if not pixel_mode and not layer_names:
            logger.warning(f"[{section_name}] Missing layer_names; cannot reconstruct configuration. Skipping.")
            return False

        # Use trial-specific seed if available for reproducible compression
        trial_seed = result.get("trial_seed", args.seed)
        try:
            trial_seed = int(trial_seed)
        except (TypeError, ValueError):
            trial_seed = args.seed

        logger.info(f"[{section_name}] Computing OOD metrics for target_dim={target_dim}, layers={layer_names}")

        try:
            if pixel_mode:
                trial_res = run_random_sampling_trial(
                    model=model,
                    model_adapter=model_adapter,
                    device=device,
                    train_raw=train_raw,
                    train_labels=train_labels,
                    val_raw=val_raw,
                    val_labels=val_labels,
                    test_raw=test_raw,
                    test_labels=test_labels,
                    target_dim=target_dim,
                    batch_size=args.batch_size,
                    dataset=args.dataset,
                    seed=trial_seed,
                    pixel_mode=True,
                    no_first_stage_compression=args.no_first_stage_compression,
                    ood_raw=ood_raw,
                )
            else:
                trial_res = run_random_sampling_trial(
                    model=model,
                    model_adapter=model_adapter,
                    device=device,
                    train_raw=train_raw,
                    train_labels=train_labels,
                    val_raw=val_raw,
                    val_labels=val_labels,
                    test_raw=test_raw,
                    test_labels=test_labels,
                    target_dim=target_dim,
                    selected_layer_names=layer_names,
                    batch_size=args.batch_size,
                    dataset=args.dataset,
                    seed=trial_seed,
                    pixel_mode=False,
                    no_first_stage_compression=args.no_first_stage_compression,
                    ood_raw=ood_raw,
                )
        except Exception as e:
            logger.warning(f"[{section_name}] OOD evaluation failed for this result: {e}")
            return False

        result["ood_auroc"] = trial_res.get("ood_auroc")
        result["ood_fpr95"] = trial_res.get("ood_fpr95")
        if "ood_diagnostics" in trial_res:
            result["ood_diagnostics"] = trial_res["ood_diagnostics"]
        return True

    # Helper to log and process lists of results
    def process_results_list(section_name: str, results_list: List[Dict[str, Any]]):
        if not results_list:
            return
        n_total = len(results_list)
        n_existing = sum(
            1 for r in results_list
            if r.get("ood_auroc") is not None and r.get("ood_fpr95") is not None
        )
        n_to_compute = n_total - n_existing
        logger.info(f"Processing {section_name}: {n_total} results")
        logger.info(f"  - {n_existing} already have OOD metrics")
        logger.info(f"  - {n_to_compute} need OOD metrics")

        updated = 0
        for r in results_list:
            if update_result_with_ood_metrics(r, section_name):
                updated += 1
                # Incremental save after each successful update
                save_incremental_results(ood_output_path, existing_results)

        logger.info(f"Finished {section_name}: {updated} results updated with OOD metrics")

    # Process geo_best
    geo_best = existing_results.get("geo_best", {})
    geo_best_list = geo_best.get("all_single_layers", [])
    process_results_list("geo_best/all_single_layers", geo_best_list)

    # Process geo_comb
    geo_comb = existing_results.get("geo_comb", {})
    geo_comb_list = geo_comb.get("results", [])
    process_results_list("geo_comb/results", geo_comb_list)

    # Process tulip_best
    tulip_best = existing_results.get("tulip_best", {})
    tulip_best_list = tulip_best.get("all_single_layers", [])
    process_results_list("tulip_best/all_single_layers", tulip_best_list)

    # Process tulip_comb
    tulip_comb = existing_results.get("tulip_comb", {})
    tulip_comb_list = tulip_comb.get("results", [])
    process_results_list("tulip_comb/results", tulip_comb_list)

    # Process random_layer_ablation trials
    rla = existing_results.get("random_layer_ablation", {})
    results_by_target_dim = rla.get("results_by_target_dim", {})

    for td_key, td_data in results_by_target_dim.items():
        # Keys may be str or int; just use string in section name
        section_td = f"random_layer_ablation/td={td_key}"
        for strategy in ["global_random", "block_random", "first_k_layers", "last_k_layers"]:
            if strategy in td_data:
                process_results_list(f"{section_td}/{strategy}", td_data[strategy])

    # Process baselines for OOD metrics
    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING BASELINES FOR OOD METRICS")
    logger.info("=" * 80)

    try:
        num_baselines_updated = add_ood_to_baselines(
            existing_results=existing_results,
            model_adapter=model_adapter,
            train_raw=train_raw,
            train_labels=train_labels,
            val_raw=val_raw,
            val_labels=val_labels,
            test_raw=test_raw,
            test_labels=test_labels,
            ood_raw=ood_raw,
            batch_size=args.batch_size,
            device=device,
            dataset=args.dataset,
        )
        logger.info(f"Updated {num_baselines_updated} baseline methods with OOD metrics")

        # Save after baseline updates
        save_incremental_results(ood_output_path, existing_results)
    except Exception as e:
        logger.error(f"Failed to process baselines: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # Final save to ensure consistency
    save_incremental_results(ood_output_path, existing_results)
    logger.info("✅ OOD-only evaluation complete.")


def add_ood_to_baselines(
    existing_results: Dict[str, Any],
    model_adapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    ood_raw: np.ndarray,
    batch_size: int,
    device: torch.device,
    dataset: str,
) -> int:
    """
    Add OOD metrics to baseline calibration methods.

    Returns number of baselines updated.
    """
    from Calibrators.temperature_scaling import TemperatureScaling
    from Calibrators.isotonic_regression import TopLabelIsotonicCalibrator
    from Calibrators.platt_scaling import PlattScaling
    from Calibrators.beta_calibration import BetaCalibration
    from Calibrators.density_aware_calibration import (
        DensityAwareCalibrator,
        get_dac_target_layers,
        extract_dac_features,
        get_dac_k_value,
    )
    from Experiments.layer_selection import compress_with_spp_jl
    from Calibrators.geometric_calibrator_new import GeometricCalibrator

    baselines = existing_results.get("baselines", {})
    if not baselines:
        logger.warning("No baselines section found")
        return 0

    updated_count = 0

    # Helper: Get ID/OOD test predictions (uncalibrated)
    logger.info("Precomputing uncalibrated probabilities for ID and OOD...")
    test_probs_uncal = model_adapter.predict_proba(test_raw, batch_size=batch_size)
    ood_probs_uncal = model_adapter.predict_proba(ood_raw, batch_size=batch_size)

    # === 1. UNCALIBRATED ===
    if "uncalibrated" in baselines:
        baseline = baselines["uncalibrated"]
        if baseline.get("ood_auroc") is None and baseline.get("error") is None:
            logger.info("Computing OOD metrics for: uncalibrated")
            try:
                id_scores = 1.0 - np.max(test_probs_uncal, axis=1)
                ood_scores = 1.0 - np.max(ood_probs_uncal, axis=1)
                baseline["ood_auroc"] = compute_ood_auroc(id_scores, ood_scores)
                baseline["ood_fpr95"] = compute_fpr_at_tpr(id_scores, ood_scores)
                updated_count += 1
            except Exception as e:
                logger.warning(f"Failed to compute OOD for uncalibrated: {e}")

    # === 2. TEMPERATURE SCALING ===
    if "temperature_scaling" in baselines:
        baseline = baselines["temperature_scaling"]
        if baseline.get("ood_auroc") is None and baseline.get("error") is None:
            logger.info("Computing OOD metrics for: temperature_scaling")
            try:
                logits_val = model_adapter.predict_logits(val_raw, batch_size=batch_size)
                logits_test = model_adapter.predict_logits(test_raw, batch_size=batch_size)
                logits_ood = model_adapter.predict_logits(ood_raw, batch_size=batch_size)

                ts = TemperatureScaling()
                ts.fit(logits_val, val_labels)
                test_probs = ts.calibrate(logits_test)
                ood_probs = ts.calibrate(logits_ood)

                id_scores = 1.0 - np.max(test_probs, axis=1)
                ood_scores = 1.0 - np.max(ood_probs, axis=1)
                baseline["ood_auroc"] = compute_ood_auroc(id_scores, ood_scores)
                baseline["ood_fpr95"] = compute_fpr_at_tpr(id_scores, ood_scores)
                updated_count += 1
            except Exception as e:
                logger.warning(f"Failed to compute OOD for temperature_scaling: {e}")

    # === 3. ISOTONIC REGRESSION (Top-label) ===
    if "isotonic_toplabel" in baselines:
        baseline = baselines["isotonic_toplabel"]
        if baseline.get("ood_auroc") is None and baseline.get("error") is None:
            logger.info("Computing OOD metrics for: isotonic_toplabel")
            try:
                logits_val = model_adapter.predict_logits(val_raw, batch_size=batch_size)
                logits_test = model_adapter.predict_logits(test_raw, batch_size=batch_size)
                logits_ood = model_adapter.predict_logits(ood_raw, batch_size=batch_size)

                iso = TopLabelIsotonicCalibrator()
                iso.fit(logits_val, val_labels)
                test_probs = iso.calibrate(logits_test)
                ood_probs = iso.calibrate(logits_ood)

                id_scores = 1.0 - np.max(test_probs, axis=1)
                ood_scores = 1.0 - np.max(ood_probs, axis=1)
                baseline["ood_auroc"] = compute_ood_auroc(id_scores, ood_scores)
                baseline["ood_fpr95"] = compute_fpr_at_tpr(id_scores, ood_scores)
                updated_count += 1
            except Exception as e:
                logger.warning(f"Failed to compute OOD for isotonic_toplabel: {e}")

    # === 4. PLATT SCALING ===
    if "platt_scaling" in baselines:
        baseline = baselines["platt_scaling"]
        if baseline.get("ood_auroc") is None and baseline.get("error") is None:
            logger.info("Computing OOD metrics for: platt_scaling")
            try:
                logits_val = model_adapter.predict_logits(val_raw, batch_size=batch_size)
                logits_test = model_adapter.predict_logits(test_raw, batch_size=batch_size)
                logits_ood = model_adapter.predict_logits(ood_raw, batch_size=batch_size)

                platt = PlattScaling()
                platt.fit(logits_val, val_labels)
                test_probs = platt.calibrate(logits_test)
                ood_probs = platt.calibrate(logits_ood)

                id_scores = 1.0 - np.max(test_probs, axis=1)
                ood_scores = 1.0 - np.max(ood_probs, axis=1)
                baseline["ood_auroc"] = compute_ood_auroc(id_scores, ood_scores)
                baseline["ood_fpr95"] = compute_fpr_at_tpr(id_scores, ood_scores)
                updated_count += 1
            except Exception as e:
                logger.warning(f"Failed to compute OOD for platt_scaling: {e}")

    # === 5. BETA CALIBRATION ===
    if "beta_calibration" in baselines:
        baseline = baselines["beta_calibration"]
        if baseline.get("ood_auroc") is None and baseline.get("error") is None:
            logger.info("Computing OOD metrics for: beta_calibration")
            try:
                logits_val = model_adapter.predict_logits(val_raw, batch_size=batch_size)
                logits_test = model_adapter.predict_logits(test_raw, batch_size=batch_size)
                logits_ood = model_adapter.predict_logits(ood_raw, batch_size=batch_size)

                beta = BetaCalibration()
                beta.fit(logits_val, val_labels)
                test_probs = beta.calibrate(logits_test)
                ood_probs = beta.calibrate(logits_ood)

                id_scores = 1.0 - np.max(test_probs, axis=1)
                ood_scores = 1.0 - np.max(ood_probs, axis=1)
                baseline["ood_auroc"] = compute_ood_auroc(id_scores, ood_scores)
                baseline["ood_fpr95"] = compute_fpr_at_tpr(id_scores, ood_scores)
                updated_count += 1
            except Exception as e:
                logger.warning(f"Failed to compute OOD for beta_calibration: {e}")

    # === 6. DENSITY-AWARE CALIBRATION (DAC) ===
    if "density_aware_calibration" in baselines:
        baseline = baselines["density_aware_calibration"]
        if baseline.get("ood_auroc") is None and baseline.get("error") is None:
            logger.info("Computing OOD metrics for: density_aware_calibration")
            try:
                raw_model = model_adapter.model
                dev = torch.device(device)

                # Infer model name heuristically
                model_name = raw_model.__class__.__name__.lower()
                if "resnet" in model_name and hasattr(raw_model, "layer1"):
                    num_blocks = [
                        len(raw_model.layer1),
                        len(raw_model.layer2),
                        len(raw_model.layer3),
                        len(raw_model.layer4),
                    ]
                    if num_blocks == [2, 2, 2, 2]:
                        model_name = "resnet18"
                    elif num_blocks == [3, 4, 6, 3]:
                        model_name = "resnet50"
                    elif num_blocks == [3, 4, 23, 3]:
                        model_name = "resnet101"
                    elif num_blocks == [3, 8, 36, 3]:
                        model_name = "resnet152"
                elif "densenet" in model_name:
                    model_name = "densenet121"

                layer_names = get_dac_target_layers(model_name, raw_model)

                # Create loaders
                # Use drop_last=True for train/val to ensure consistent batch sizes and avoid dimension mismatches
                tr_loader = DataLoader(
                    TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True,
                )
                va_loader = DataLoader(
                    TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=True,
                )
                te_loader = DataLoader(
                    TensorDataset(torch.from_numpy(test_raw), torch.from_numpy(test_labels)),
                    batch_size=batch_size,
                    shuffle=False,
                )
                ood_loader = DataLoader(
                    TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
                    batch_size=batch_size,
                    shuffle=False,
                )

                # Extract features
                train_feats_list, _, train_labels_dac = extract_dac_features(raw_model, tr_loader, layer_names, dev)
                val_feats_list, logits_val_dac, val_labels_dac = extract_dac_features(
                    raw_model, va_loader, layer_names, dev
                )
                test_feats_list, logits_test_dac, _ = extract_dac_features(raw_model, te_loader, layer_names, dev)
                ood_feats_list, logits_ood_dac, _ = extract_dac_features(raw_model, ood_loader, layer_names, dev)

                # Fit DAC
                k_value = get_dac_k_value(dataset)
                dac = DensityAwareCalibrator(k=k_value, use_gpu=(str(device) == "cuda"))
                dac.fit(
                    train_features_list=[f.numpy() for f in train_feats_list],
                    val_features_list=[f.numpy() for f in val_feats_list],
                    val_logits=logits_val_dac.numpy(),
                    val_labels=val_labels_dac.numpy(),
                )

                # Calibrate
                test_probs = dac.calibrate([f.numpy() for f in test_feats_list], logits_test_dac.numpy())
                ood_probs = dac.calibrate([f.numpy() for f in ood_feats_list], logits_ood_dac.numpy())

                id_scores = 1.0 - np.max(test_probs, axis=1)
                ood_scores = 1.0 - np.max(ood_probs, axis=1)
                baseline["ood_auroc"] = compute_ood_auroc(id_scores, ood_scores)
                baseline["ood_fpr95"] = compute_fpr_at_tpr(id_scores, ood_scores)
                updated_count += 1
            except Exception as e:
                logger.warning(f"Failed to compute OOD for density_aware_calibration: {e}")

    # === 7. GEOMETRIC PHYSICAL SPACE (SPP+JL on pixels) ===
    if "geometric_physical_space" in baselines:
        baseline = baselines["geometric_physical_space"]
        if baseline.get("ood_auroc") is None and baseline.get("error") is None:
            logger.info("Computing OOD metrics for: geometric_physical_space")
            try:
                FINAL_DIM = 1024
                PYR = [4, 2, 1]

                Xtr_c = compress_with_spp_jl(
                    train_raw,
                    final_output_dim=FINAL_DIM,
                    pyramid_levels=PYR,
                    seed=42,
                    batch_size=batch_size,
                    device=str(device),
                )
                Xva_c = compress_with_spp_jl(
                    val_raw,
                    final_output_dim=FINAL_DIM,
                    pyramid_levels=PYR,
                    seed=42,
                    batch_size=batch_size,
                    device=str(device),
                )
                Xte_c = compress_with_spp_jl(
                    test_raw,
                    final_output_dim=FINAL_DIM,
                    pyramid_levels=PYR,
                    seed=42,
                    batch_size=batch_size,
                    device=str(device),
                )
                Xood_c = compress_with_spp_jl(
                    ood_raw,
                    final_output_dim=FINAL_DIM,
                    pyramid_levels=PYR,
                    seed=42,
                    batch_size=batch_size,
                    device=str(device),
                )

                geo = GeometricCalibrator(
                    model=model_adapter,
                    X_train_embed=Xtr_c,
                    y_train=train_labels,
                    auto_select_layer=False,
                    library="fast_separation",
                    device=str(device),
                )
                geo.fit(
                    X_val_embed=Xva_c,
                    X_val_original=val_raw,
                    y_val=val_labels,
                    fit_batch_size=batch_size,
                )

                test_probs = geo.calibrate_batched(
                    X_test_embed=Xte_c,
                    X_test_original=test_raw,
                    batch_size=batch_size,
                )
                ood_probs = geo.calibrate_batched(
                    X_test_embed=Xood_c,
                    X_test_original=ood_raw,
                    batch_size=batch_size,
                )

                id_scores = 1.0 - np.max(test_probs, axis=1)
                ood_scores = 1.0 - np.max(ood_probs, axis=1)
                baseline["ood_auroc"] = compute_ood_auroc(id_scores, ood_scores)
                baseline["ood_fpr95"] = compute_fpr_at_tpr(id_scores, ood_scores)
                updated_count += 1
            except Exception as e:
                logger.warning(f"Failed to compute OOD for geometric_physical_space: {e}")

    return updated_count


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------
def run_experiments(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Corruption benchmark validation
    # ------------------------------------------------------------------
    if args.corruption_type is not None and args.corruption_severity is None:
        raise ValueError(
            "If --corruption-type is set, you must also set "
            "--corruption-severity to an integer in [1, 2, 3, 4, 5]."
        )
    
    prefix = "pixel_ablation" if args.pixel_mode else "ablation"
    if args.pixel_mode and args.no_first_stage_compression:
        prefix += "_no_stage1_comp"
    # Base output filename
    if args.corruption_type:
        output_path = os.path.join(
            args.output_dir,
            f"{prefix}_{args.training_method}_{args.dataset}_{args.model_name}_seed{args.seed}_{args.corruption_type}_sev{args.corruption_severity}.json",
        )
    else:
        output_path = os.path.join(
            args.output_dir,
            f"{prefix}_{args.training_method}_{args.dataset}_{args.model_name}_seed{args.seed}.json",
        )

    # OOD-only mode: just augment existing results with OOD metrics and exit
    if getattr(args, "ood_only", False):
        run_ood_only_evaluation(args, output_path)
        return

    # ===== LOAD EXISTING RESULTS IF AVAILABLE =====
    existing_results: Dict[str, Any] = {}
    if os.path.exists(output_path):
        logger.info(f"Found existing results file: {output_path}")
        logger.info("Loading existing results from compare_dac_geometric.py...")
        try:
            with open(output_path, "r") as f:
                existing_results = json.load(f)
            logger.info(f"Loaded {len(existing_results)} existing sections")

            if "random_layer_ablation" in existing_results:
                rla_data = existing_results["random_layer_ablation"]
                
                # Check if first_k_layers and last_k_layers exist for all target dimensions
                missing_baselines = False
                if "results_by_target_dim" in rla_data:
                    results_by_td = rla_data["results_by_target_dim"]
                    for target_dim, target_data in results_by_td.items():
                        if "first_k_layers" not in target_data or "last_k_layers" not in target_data:
                            missing_baselines = True
                            logger.info(f"Missing first_k_layers or last_k_layers for target_dim={target_dim}, will add them")
                            break
                        elif not target_data.get("first_k_layers") or not target_data.get("last_k_layers"):
                            missing_baselines = True
                            logger.info(f"Empty first_k_layers or last_k_layers for target_dim={target_dim}, will add them")
                            break
                else:
                    missing_baselines = True
                    logger.info("No results_by_target_dim found, will add baselines")
                
                if not missing_baselines:
                    logger.warning("⚠️ Random ablation results already exist in this file!")
                    logger.warning("   Delete the 'random_layer_ablation' section if you want to recompute.")
                    logger.info("   Exiting without changes.")
                    return
                else:
                    logger.info("✅ Random ablation results exist but missing first_k_layers or last_k_layers")
                    logger.info("   Will add missing deterministic baselines incrementally")
        except Exception as e:
            logger.error(f"Failed to load existing results: {e}")
            logger.info("Starting fresh results file")
            existing_results = {}
    else:
        logger.info(f"No existing results found at {output_path}")
        logger.info("Creating new results file")

    # === Initialize baseline results (added) ===
    baseline_results = None

    device = torch.device(args.device)
    set_global_seed(args.seed)

    # Log experiment scope based on skip flag
    if args.skip_single_layers:
        logger.info("\n" + "⏭️  " * 20)
        logger.info("SINGLE-LAYER EXPERIMENTS DISABLED (--skip-single-layers)")
        logger.info("  Will skip: geo_best, tulip_best")
        logger.info("  Will run: geo_comb, tulip_comb, random ablation")
        logger.info("⏭️  " * 20 + "\n")

    # Load data
    # Determine required image size for DINOv2 models
    is_dinov2 = 'dinov2' in args.model_name.lower()
    image_size = 224 if is_dinov2 else None  # None means use default (32 for CIFAR)
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset,
        args.batch_size,
        seed=args.seed,
        corruption_type=args.corruption_type,
        corruption_severity=args.corruption_severity,
        image_size=image_size,
    )
    train_raw, train_labels = extract_raw_from_loader(train_loader)
    val_raw, val_labels = extract_raw_from_loader(val_loader)
    test_raw, test_labels = extract_raw_from_loader(test_loader)
    
    # OOD data: SVHN test set for CIFAR-10/100 and Tiny ImageNet
    ood_raw = None
    if args.corruption_type:
        # For corruption benchmarks, SVHN is not a meaningful OOD dataset.
        logger.info("Skipping OOD evaluation for corruption benchmark (corrupted CIFAR).")
    elif args.skip_ood:
        logger.info("Skipping OOD evaluation (--skip-ood flag set)")
    elif args.dataset in ["cifar10", "cifar100"] or args.dataset.lower() in ['tiny_imagenet', 'tinyimagenet']:
        logger.info("Loading SVHN test set as OOD dataset...")
        ood_loader = svhn_test(batch_size=args.batch_size, shuffle=False)
        ood_raw, _ = extract_raw_from_loader(ood_loader)

    # Load model
    model_path = construct_model_path(
        args.results_base_dir, args.training_method, args.dataset, args.model_name, args.seed
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = load_trained_model(model_path, args.model_name, num_classes, device, dataset=args.dataset)
    model_adapter = PyTorchModelAdapter(model, device, args.dataset)

    # ===== COMPUTE BASELINES (ONCE PER MODEL) =====  # (added)
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE CALIBRATION METHODS")
    logger.info("=" * 80)
    
    if args.corruption_type:
        baseline_results_path = os.path.join(
            args.output_dir,
            f"baseline_{args.training_method}_{args.dataset}_{args.model_name}_seed{args.seed}_{args.corruption_type}_sev{args.corruption_severity}.json"
        )
    else:
        baseline_results_path = os.path.join(
            args.output_dir,
            f"baseline_{args.training_method}_{args.dataset}_{args.model_name}_seed{args.seed}.json"
        )

    
    if os.path.exists(baseline_results_path):
        logger.info(f"✅ Baselines already computed, loading from: {baseline_results_path}")
        with open(baseline_results_path, 'r') as f:
            baseline_results = json.load(f)
    else:
        logger.info("Computing baseline calibration methods (this runs once per model)...")
        logger.info("  - Uncalibrated")
        logger.info("  - Temperature Scaling")
        logger.info("  - Isotonic Regression (Top-Label)")
        logger.info("  - Platt Scaling")
        logger.info("  - Beta Calibration")
        logger.info("  - DAC (Density-Aware Calibration)")
        logger.info("  - PSC (Probabilistic Skip Connections)")
        logger.info("  - TULIP (Transitional Uncertainty)")
        logger.info("  - Geometric Physical Space (SPP+JL on pixels)")
        
        # Note: candidate_indices will be computed in the next section
        # We'll compute baselines after layer discovery if needed
        baseline_results = None  # Placeholder, will compute after layer discovery

    # ===== PIXEL MODE SHORT-CIRCUIT =====
    if args.pixel_mode:
        logger.info("\n" + "=" * 80)
        logger.info(f"PIXEL MODE (no layers). Stage1 compression skipped: {args.no_first_stage_compression}")
        logger.info("=" * 80)

        pixel_results = existing_results.get("pixel_results_by_dim", {})
        for target_dim in args.target_dims:
            td_key = str(target_dim)
            if td_key in pixel_results:
                logger.info(f"Target dim {target_dim} already computed. Skipping.")
                continue

            logger.info(f"Running pixel experiment -> target_dim={target_dim}")
            res = run_random_sampling_trial(
                model=model,
                model_adapter=model_adapter,
                device=device,
                train_raw=train_raw,
                train_labels=train_labels,
                val_raw=val_raw,
                val_labels=val_labels,
                test_raw=test_raw,
                test_labels=test_labels,
                target_dim=target_dim,
                batch_size=args.batch_size,
                dataset=args.dataset,
                seed=args.seed,
                pixel_mode=True,
                no_first_stage_compression=args.no_first_stage_compression,
                ood_raw=ood_raw,
            )

            pixel_results[td_key] = make_json_serializable(res)
            existing_results["pixel_results_by_dim"] = pixel_results
            existing_results["experiment_info"] = {
                "dataset": args.dataset,
                "model_name": args.model_name,
                "training_method": args.training_method,
                "seed": args.seed,
                "mode": "pixels",
                "no_first_stage_compression": args.no_first_stage_compression,
            }
            save_incremental_results(output_path, existing_results)

        logger.info("Pixel experiments complete.")
        logger.info(f"Results saved to: {output_path}")
        return

    # ===== DISCOVER ALL LAYERS (not just main blocks) =====
    logger.info("\n" + "=" * 80)
    logger.info("DISCOVERING ALL MODEL LAYERS FOR RANDOM SAMPLING")
    logger.info("=" * 80)

    # Determine input shape for layer discovery (DINOv2 needs 224x224)
    is_dinov2 = 'dinov2' in args.model_name.lower()
    if is_dinov2:
        input_shape = (1, 3, 224, 224)
    elif args.dataset.lower() in ['tiny_imagenet', 'tinyimagenet']:
        input_shape = (1, 3, 64, 64)
    else:
        input_shape = (1, 3, 32, 32)  # Default for CIFAR/SVHN
    
    discovered_layers = normalize_discovered_layers(
        model, discover_model_layers(model, device=device, input_shape=input_shape)
    )
    candidate_indices = [d["idx"] for d in discovered_layers]

    logger.info(f"Discovered {len(candidate_indices)} total layers")
    logger.info("Sample of discovered layers:")
    for i in [0, 1, 2, -3, -2, -1]:
        if 0 <= i < len(discovered_layers) or -len(discovered_layers) <= i < 0:
            d = discovered_layers[i]
            logger.info(f"  idx={d['idx']:3d}, name={d['name']}")

    # Check if this is a DINOv2 model (skip fixed layer selection for DINOv2)
    is_dinov2 = "dino" in args.model_name.lower()
    
    # For DINOv2, filter out high-dimension layers to prevent OOM
    MAX_LAYER_DIM_DINOV2 = 50000  # Threshold: attention layers ~32k, MLP layers ~172k
    
    if is_dinov2:
        logger.info("\n" + "=" * 80)
        logger.info("DINOv2 DETECTED: PROBING LAYER DIMENSIONS FOR OOM PREVENTION")
        logger.info("=" * 80)
        
        layer_dims = probe_layer_dimensions(model, discovered_layers, device, args.dataset, args.model_name)
        
        # Log dimension distribution
        dim_values = [d for d in layer_dims.values() if d > 0]
        if dim_values:
            logger.info(f"Layer dimension stats: min={min(dim_values)}, max={max(dim_values)}, median={np.median(dim_values):.0f}")
        
        # Filter out layers exceeding threshold
        original_count = len(candidate_indices)
        filtered_indices = [
            idx for idx in candidate_indices 
            if layer_dims.get(idx, 0) > 0 and layer_dims.get(idx, float('inf')) <= MAX_LAYER_DIM_DINOV2
        ]
        
        # Log excluded layers
        excluded = [idx for idx in candidate_indices if idx not in filtered_indices]
        if excluded:
            logger.warning(f"Excluding {len(excluded)} high-dimension layers (>{MAX_LAYER_DIM_DINOV2} dims):")
            idx_to_layer = {l["idx"]: l for l in discovered_layers}
            for idx in excluded[:5]:  # Show first 5
                layer = idx_to_layer.get(idx)
                if layer:
                    dim = layer_dims.get(idx, "unknown")
                    logger.warning(f"  {layer['name']}: {dim} dims")
            if len(excluded) > 5:
                logger.warning(f"  ... and {len(excluded) - 5} more")
        
        candidate_indices = filtered_indices
        discovered_layers = [l for l in discovered_layers if l["idx"] in candidate_indices]
        logger.info(f"Filtered: {original_count} -> {len(candidate_indices)} layers")
        
        # Re-compute block groups after filtering
        block_groups = group_layers_by_block(discovered_layers)
        logger.info(f"Updated block groups: {list(block_groups.keys())}")
        for block_name, indices in block_groups.items():
            logger.info(f"  {block_name}: {len(indices)} layers")
    else:
        block_groups = group_layers_by_block(discovered_layers)
        logger.info(f"Block groups: {list(block_groups.keys())}")
        for block_name, indices in block_groups.items():
            logger.info(f"  {block_name}: {len(indices)} layers")

    # === Compute baselines after layer discovery (added) ===
    # Now compute baselines if not already done (needs candidate_indices for DAC)
    if baseline_results is None and not os.path.exists(baseline_results_path):
        try:
            # Create model-specific output directory for baselines
            if args.corruption_type:
                baseline_output_dir = os.path.join(
                    args.output_dir,
                    "baselines",
                    f"{args.training_method}_{args.dataset}_{args.model_name}_seed{args.seed}_{args.corruption_type}_sev{args.corruption_severity}"
                )
            else:
                baseline_output_dir = os.path.join(
                    args.output_dir,
                    "baselines",
                    f"{args.training_method}_{args.dataset}_{args.model_name}_seed{args.seed}"
                )
            os.makedirs(baseline_output_dir, exist_ok=True)
            logger.info(f"Created baseline output directory: {baseline_output_dir}")
            
            baseline_results = run_standard_baselines(
                model_adapter=model_adapter,
                train_raw=train_raw, 
                train_labels=train_labels,
                val_raw=val_raw, 
                val_labels=val_labels,
                test_raw=test_raw, 
                test_labels=test_labels,
                batch_size=args.batch_size,
                output_dir=baseline_output_dir,  # USE MODEL-SPECIFIC DIR
                candidate_layers_all=candidate_indices,
                seed=args.seed,
                device=args.device
            )
            
            # Save baselines separately for quick reference (redundant but for consistency)
            with open(baseline_results_path, 'w') as f:
                json.dump(baseline_results, f, indent=2, cls=NumpyEncoder)
            logger.info(f"💾 Baseline results also saved to: {baseline_results_path}")
            
        except Exception as e:
            logger.warning(f"Failed to compute baselines: {e}")
            logger.warning("Continuing with random ablation experiments only")
            baseline_results = {}
    elif baseline_results is None:
        # File exists, load it
        with open(baseline_results_path, 'r') as f:
            baseline_results = json.load(f)

    # ===== GET FIXED 4 LAYERS (DAC paper's choice) =====
    # Skip for DINOv2 models since DAC paper didn't specify fixed layers for DINOv2
    fixed_layer_names = []
    if args.skip_fixed_layers:
        logger.info("\n" + "=" * 80)
        logger.info("SKIPPING FIXED LAYER SELECTION (--skip-fixed-layers enabled)")
        logger.info("=" * 80)
    elif not is_dinov2:
        logger.info("\n" + "=" * 80)
        logger.info("GETTING FIXED 4 LAYERS (DAC PAPER SELECTION)")
        logger.info("=" * 80)
        fixed_layer_names = get_layer_names_for_model(args.model_name, model=model)
        logger.info(f"Fixed layers from DAC paper: {fixed_layer_names}")
    else:
        logger.info("\n" + "=" * 80)
        logger.info("SKIPPING FIXED LAYER SELECTION FOR DINOv2")
        logger.info("=" * 80)
        logger.info("DAC paper did not specify fixed layers for DINOv2 architecture.")
        logger.info("Proceeding with random layer ablation only.")

    # Get TULIP's strategically selected layers
    tulip_layer_names = None
    if not is_dinov2:
        try:
            logger.info("Getting TULIP's strategically selected layers...")
            tulip_layer_names = select_tulip_layers(model, args.model_name, num_fallback=7)
            logger.info(f"TULIP selected layers: {tulip_layer_names}")
            logger.info(f"  Number of layers: {len(tulip_layer_names)}")
        except Exception as e:
            logger.warning(f"Failed to get TULIP layers: {e}")
            tulip_layer_names = None

    # ===== SETUP RANDOM ABLATION EXPERIMENT =====
    logger.info("\n" + "=" * 80)
    logger.info("RANDOM LAYER ABLATION CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model training seed: {args.seed}")
    logger.info(f"Random trials per configuration: {args.num_random_seeds}")
    logger.info(f"Layer counts to test: {args.layer_counts}")
    logger.info(f"Target dimensions to test: {args.target_dims}")

    rng = random.Random(args.seed)
    target_dims = args.target_dims
    layer_counts = args.layer_counts

    # Check if we have existing random_layer_ablation results to merge
    existing_rla = existing_results.get("random_layer_ablation", {})
    existing_results_by_td = existing_rla.get("results_by_target_dim", {})
    
    random_ablation_results: Dict[str, Any] = {
        "experiment_config": {
            "dataset": args.dataset,
            "model_name": args.model_name,
            "training_method": args.training_method,
            "seed": args.seed,
            "num_random_trials": args.num_random_seeds,
            "target_dims": target_dims,
            "layer_counts": layer_counts,
            "total_available_layers": len(candidate_indices),
            "note": "Fixed target dimensions with varying layer counts",
            "skip_block_random": args.skip_block_random,
        },
        "results_by_target_dim": {},  # NEW structure
    }
    
    # Initialize results_by_target_dim with existing data if available
    for target_dim in target_dims:
        if str(target_dim) in existing_results_by_td:
            # Copy existing results for this target_dim
            random_ablation_results["results_by_target_dim"][target_dim] = existing_results_by_td[str(target_dim)].copy()
            # Ensure first_k_layers and last_k_layers keys exist (may be missing in old results)
            if "first_k_layers" not in random_ablation_results["results_by_target_dim"][target_dim]:
                random_ablation_results["results_by_target_dim"][target_dim]["first_k_layers"] = []
            if "last_k_layers" not in random_ablation_results["results_by_target_dim"][target_dim]:
                random_ablation_results["results_by_target_dim"][target_dim]["last_k_layers"] = []
            logger.info(f"Loaded existing results for target_dim={target_dim}")
        else:
            # Initialize empty structure
            random_ablation_results["results_by_target_dim"][target_dim] = {
                "global_random": [],
                "block_random": [],
                "first_k_layers": [],
                "last_k_layers": [],
            }
    
    # Initialize geo_best and geo_comb results (needed for incremental saving)
    # Load existing results if available
    existing_geo_best = existing_results.get("geo_best", {})
    existing_geo_comb = existing_results.get("geo_comb", {})
    
    geo_best_results = existing_geo_best.get("all_single_layers", [])
    # Convert best_per_target_dim keys to int if they're strings (JSON keys are strings)
    best_per_target_dim_raw = existing_geo_best.get("best_per_target_dim", {})
    best_per_target_dim = {}
    for key, value in best_per_target_dim_raw.items():
        try:
            int_key = int(key)
            best_per_target_dim[int_key] = value
        except (ValueError, TypeError):
            best_per_target_dim[key] = value
    geo_comb_results = existing_geo_comb.get("results", [])

    # Load TULIP results if they exist
    existing_tulip_best = existing_results.get("tulip_best", {})
    existing_tulip_comb = existing_results.get("tulip_comb", {})
    
    tulip_best_results = existing_tulip_best.get("all_single_layers", [])
    tulip_comb_results = existing_tulip_comb.get("results", [])
    
    # Check if TULIP experiments are complete
    tulip_best_complete = False
    tulip_comb_complete = False
    if args.skip_single_layers:
        tulip_best_complete = True  # Mark as complete if we're skipping it
        logger.info(f"⏭️  TULIP_BEST marked as complete (--skip-single-layers flag)")
    elif not is_dinov2 and tulip_layer_names:
        # Check tulip_best completeness
        expected_tulip_best_count = len(tulip_layer_names) * len(target_dims)
        if len(tulip_best_results) >= expected_tulip_best_count:
            existing_combinations = set()
            for result in tulip_best_results:
                layer_name = result.get("layer_names", [None])[0]
                target_dim = result.get("target_dim")
                if layer_name and target_dim is not None:
                    try:
                        target_dim = int(target_dim)
                    except (ValueError, TypeError):
                        pass
                    existing_combinations.add((layer_name, target_dim))
            
            expected_combinations = set()
            for layer_name in tulip_layer_names:
                for target_dim in target_dims:
                    expected_combinations.add((layer_name, target_dim))
            
            if existing_combinations == expected_combinations:
                tulip_best_complete = True
                logger.info(f"✅ TULIP_BEST results already exist ({len(tulip_best_results)} results)")
        
        # Check tulip_comb completeness
        if len(tulip_comb_results) >= len(target_dims):
            existing_tulip_comb_tds = set()
            for r in tulip_comb_results:
                td = r.get("target_dim")
                if td is not None:
                    try:
                        td = int(td)
                    except (ValueError, TypeError):
                        pass
                    existing_tulip_comb_tds.add(td)
            if existing_tulip_comb_tds == set(target_dims):
                tulip_comb_complete = True
                logger.info(f"✅ TULIP_COMB results already exist ({len(tulip_comb_results)} results)")
    
    # Check if geo_best is complete (has results for all layers and all target_dims)
    geo_best_complete = False
    if args.skip_single_layers:
        geo_best_complete = True  # Mark as complete if we're skipping it
        logger.info(f"⏭️  GEO_BEST marked as complete (--skip-single-layers flag)")
    elif not is_dinov2 and fixed_layer_names:
        expected_geo_best_count = len(fixed_layer_names) * len(target_dims)
        if len(geo_best_results) >= expected_geo_best_count:
            # Check if we have results for all combinations
            existing_combinations = set()
            for result in geo_best_results:
                layer_name = result.get("layer_names", [None])[0]
                target_dim = result.get("target_dim")
                if layer_name and target_dim is not None:
                    # Normalize target_dim to int for comparison
                    try:
                        target_dim = int(target_dim)
                    except (ValueError, TypeError):
                        pass
                    existing_combinations.add((layer_name, target_dim))
            
            expected_combinations = set()
            for layer_name in fixed_layer_names:
                for target_dim in target_dims:
                    expected_combinations.add((layer_name, target_dim))
            
            if existing_combinations == expected_combinations:
                geo_best_complete = True
                logger.info(f"✅ GEO_BEST results already exist ({len(geo_best_results)} results)")
                logger.info(f"   Skipping GEO_BEST experiments")
    
    # Check if geo_comb is complete (has results for all target_dims)
    geo_comb_complete = False
    if not is_dinov2 and fixed_layer_names:
        if len(geo_comb_results) >= len(target_dims):
            existing_geo_comb_tds = set()
            for r in geo_comb_results:
                td = r.get("target_dim")
                if td is not None:
                    # Normalize target_dim to int for comparison
                    try:
                        td = int(td)
                    except (ValueError, TypeError):
                        pass
                    existing_geo_comb_tds.add(td)
            if existing_geo_comb_tds == set(target_dims):
                geo_comb_complete = True
                logger.info(f"✅ GEO_COMB results already exist ({len(geo_comb_results)} results)")
                logger.info(f"   Skipping GEO_COMB experiments")

    # ===== GEO_BEST: Test each fixed layer individually (oracle) =====
    # Skip for DINOv2 models, if already complete, or if --skip-single-layers flag is set
    if not is_dinov2 and fixed_layer_names and not geo_best_complete and not args.skip_single_layers:
        logger.info("\n" + "=" * 80)
        logger.info("GEO_BEST: TESTING EACH FIXED LAYER INDIVIDUALLY")
        logger.info("=" * 80)

        logger.info(f"Testing target dimensions: {target_dims}")

        for layer_name in fixed_layer_names:
            logger.info(f"\nTesting single layer: {layer_name}")

            for target_dim in target_dims:
                # Check if this combination already exists
                existing_for_combo = False
                for r in geo_best_results:
                    r_layer = r.get("layer_names", [None])[0]
                    r_td = r.get("target_dim")
                    # Normalize target_dim for comparison
                    try:
                        r_td = int(r_td) if r_td is not None else None
                    except (ValueError, TypeError):
                        pass
                    if r_layer == layer_name and r_td == target_dim:
                        existing_for_combo = True
                        break
                
                if existing_for_combo:
                    logger.info(f"  ⏭️  Skipping {layer_name} for target_dim={target_dim}: already exists")
                    continue
                
                logger.info(f"  Target dimension: {target_dim}")
                trial_result = run_random_sampling_trial(
                    model=model,
                    model_adapter=model_adapter,
                    device=device,
                    train_raw=train_raw,
                    train_labels=train_labels,
                    val_raw=val_raw,
                    val_labels=val_labels,
                    test_raw=test_raw,
                    test_labels=test_labels,
                    target_dim=target_dim,  # Use target_dim instead of compression_ratio
                    selected_layer_names=[layer_name],  # Use names directly!
                    batch_size=args.batch_size,
                    dataset=args.dataset,
                    seed=args.seed,
                    ood_raw=ood_raw,
                )
                trial_result.update({"strategy": "geo_best_single_layer"})
                geo_best_results.append(make_json_serializable(trial_result))
                
                # Save incrementally after each geo_best trial
                final_results = build_final_results(
                    existing_results, geo_best_results, best_per_target_dim,
                    geo_comb_results, random_ablation_results, args,
                    baseline_results=baseline_results,
                )
                save_incremental_results(output_path, final_results)

        # Find best single layer (oracle) for each target dimension
        for target_dim in target_dims:
            results_for_td = [r for r in geo_best_results if r["target_dim"] == target_dim]
            if results_for_td:
                best = min(results_for_td, key=lambda x: x["ece"])
                best_per_target_dim[target_dim] = make_json_serializable(best)
                logger.info(f"Best layer for target_dim={target_dim}: {best['layer_names'][0]} (ECE={best['ece']:.6f})")
    elif args.skip_single_layers:
        logger.info("\n" + "=" * 80)
        logger.info("GEO_BEST: SKIPPED (--skip-single-layers flag)")
        logger.info("=" * 80)

    # ===== TULIP_BEST: Test each TULIP layer individually =====
    # Skip if already complete or if --skip-single-layers flag is set
    if not is_dinov2 and tulip_layer_names and not tulip_best_complete and not args.skip_single_layers:
        logger.info("\n" + "=" * 80)
        logger.info("TULIP_BEST: TESTING EACH TULIP LAYER INDIVIDUALLY")
        logger.info("=" * 80)

        logger.info(f"Testing target dimensions: {target_dims}")

        for layer_name in tulip_layer_names:
            logger.info(f"\nTesting single TULIP layer: {layer_name}")

            for target_dim in target_dims:
                # Check if this combination already exists
                existing_for_combo = False
                for r in tulip_best_results:
                    r_layer = r.get("layer_names", [None])[0]
                    r_td = r.get("target_dim")
                    try:
                        r_td = int(r_td) if r_td is not None else None
                    except (ValueError, TypeError):
                        pass
                    if r_layer == layer_name and r_td == target_dim:
                        existing_for_combo = True
                        break
                
                if existing_for_combo:
                    logger.info(f"  ⏭️  Skipping {layer_name} for target_dim={target_dim}: already exists")
                    continue
                
                logger.info(f"  Target dimension: {target_dim}")
                trial_result = run_random_sampling_trial(
                    model=model,
                    model_adapter=model_adapter,
                    device=device,
                    train_raw=train_raw,
                    train_labels=train_labels,
                    val_raw=val_raw,
                    val_labels=val_labels,
                    test_raw=test_raw,
                    test_labels=test_labels,
                    target_dim=target_dim,
                    selected_layer_names=[layer_name],
                    batch_size=args.batch_size,
                    dataset=args.dataset,
                    seed=args.seed,
                    ood_raw=ood_raw,
                )
                trial_result.update({"strategy": "tulip_best_single_layer"})
                tulip_best_results.append(make_json_serializable(trial_result))
                
                # Save incrementally
                final_results = build_final_results(
                    existing_results, geo_best_results, best_per_target_dim,
                    geo_comb_results, random_ablation_results, args,
                    baseline_results=baseline_results,
                    tulip_best_results=tulip_best_results,
                    tulip_comb_results=tulip_comb_results,
                )
                save_incremental_results(output_path, final_results)
    elif args.skip_single_layers:
        logger.info("\n" + "=" * 80)
        logger.info("TULIP_BEST: SKIPPED (--skip-single-layers flag)")
        logger.info("=" * 80)

    # ===== GEO_COMB: Concatenate all fixed layers =====
    # Skip for DINOv2 models or if already complete
    if not is_dinov2 and fixed_layer_names and not geo_comb_complete:
        logger.info("\n" + "=" * 80)
        logger.info("GEO_COMB: CONCATENATING ALL FIXED LAYERS")
        logger.info("=" * 80)
        for target_dim in target_dims:
            # Check if this target_dim already exists
            existing_for_td = False
            for r in geo_comb_results:
                r_td = r.get("target_dim")
                # Normalize target_dim for comparison
                try:
                    r_td = int(r_td) if r_td is not None else None
                except (ValueError, TypeError):
                    pass
                if r_td == target_dim:
                    existing_for_td = True
                    break
            
            if existing_for_td:
                logger.info(f"  ⏭️  Skipping target_dim={target_dim}: already exists")
                continue
            
            logger.info(f"  Target dimension: {target_dim}")
            trial_result = run_random_sampling_trial(
                model=model,
                model_adapter=model_adapter,
                device=device,
                train_raw=train_raw,
                train_labels=train_labels,
                val_raw=val_raw,
                val_labels=val_labels,
                test_raw=test_raw,
                test_labels=test_labels,
                target_dim=target_dim,  # Use target_dim instead of compression_ratio
                selected_layer_names=fixed_layer_names,  # Use names directly!
                batch_size=args.batch_size,
                dataset=args.dataset,
                seed=args.seed,
                ood_raw=ood_raw,
            )
            trial_result.update({
                "strategy": "geo_comb_fixed_layers",
                "num_layers": len(fixed_layer_names)
            })
            geo_comb_results.append(make_json_serializable(trial_result))
            logger.info(f"    ECE: {trial_result['ece']:.6f}")
            
            # Save incrementally after each geo_comb trial
            final_results = build_final_results(
                existing_results, geo_best_results, best_per_target_dim,
                geo_comb_results, random_ablation_results, args,
                baseline_results=baseline_results,
                tulip_best_results=tulip_best_results,
                tulip_comb_results=tulip_comb_results,
            )
            save_incremental_results(output_path, final_results)

    # ===== TULIP_COMB: Concatenate all TULIP layers =====
    if not is_dinov2 and tulip_layer_names and not tulip_comb_complete:
        logger.info("\n" + "=" * 80)
        logger.info("TULIP_COMB: CONCATENATING ALL TULIP LAYERS")
        logger.info("=" * 80)
        for target_dim in target_dims:
            # Check if this target_dim already exists
            existing_for_td = False
            for r in tulip_comb_results:
                r_td = r.get("target_dim")
                try:
                    r_td = int(r_td) if r_td is not None else None
                except (ValueError, TypeError):
                    pass
                if r_td == target_dim:
                    existing_for_td = True
                    break
            
            if existing_for_td:
                logger.info(f"  ⏭️  Skipping target_dim={target_dim}: already exists")
                continue
            
            logger.info(f"  Target dimension: {target_dim}")
            trial_result = run_random_sampling_trial(
                model=model,
                model_adapter=model_adapter,
                device=device,
                train_raw=train_raw,
                train_labels=train_labels,
                val_raw=val_raw,
                val_labels=val_labels,
                test_raw=test_raw,
                test_labels=test_labels,
                target_dim=target_dim,
                selected_layer_names=tulip_layer_names,
                batch_size=args.batch_size,
                dataset=args.dataset,
                seed=args.seed,
                 ood_raw=ood_raw,
            )
            trial_result.update({
                "strategy": "tulip_comb_fixed_layers",
                "num_layers": len(tulip_layer_names)
            })
            tulip_comb_results.append(make_json_serializable(trial_result))
            logger.info(f"    ECE: {trial_result['ece']:.6f}")
            
            # Save incrementally
            final_results = build_final_results(
                existing_results, geo_best_results, best_per_target_dim,
                geo_comb_results, random_ablation_results, args,
                baseline_results=baseline_results,
                tulip_best_results=tulip_best_results,
                tulip_comb_results=tulip_comb_results,
            )
            save_incremental_results(output_path, final_results)

    # ------------------------------------------------------------------
    # Global random strategy - iterate over target dimensions
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("GLOBAL RANDOM STRATEGY")
    logger.info("=" * 80)

    # For each target dimension, test different layer configurations
    for target_dim in target_dims:
        logger.info(f"\n{'='*80}")
        logger.info(f"TARGET DIMENSION: {target_dim}")
        logger.info(f"{'='*80}")

        # Initialize structure if not already loaded from existing results
        if target_dim not in random_ablation_results["results_by_target_dim"]:
            random_ablation_results["results_by_target_dim"][target_dim] = {
                "global_random": [],
                "block_random": [],
                "first_k_layers": [],
                "last_k_layers": [],
            }
        
        target_data = random_ablation_results["results_by_target_dim"][target_dim]
        
        for num_layers in layer_counts:
            # FIXED: Count existing trials for this specific (num_layers, target_dim) combination
            existing_trials = [
                r for r in target_data.get("global_random", [])
                if r.get("num_layers") == num_layers
            ]
            
            # Skip only if we have ALL required trials
            if len(existing_trials) >= args.num_random_seeds:
                logger.info(
                    f"⏭️  Skipping num_layers={num_layers} for target_dim={target_dim}: "
                    f"already have {len(existing_trials)}/{args.num_random_seeds} trials"
                )
                continue
            
            if len(candidate_indices) < num_layers:
                logger.warning(f"Skipping num_layers={num_layers}: only {len(candidate_indices)} candidates.")
                continue

            # Calculate how many additional trials we need
            trials_already_done = len(existing_trials)
            trials_needed = args.num_random_seeds - trials_already_done
            
            if trials_needed > 0:
                logger.info(
                    f"\nTesting {num_layers} layers -> {target_dim} dims "
                    f"({trials_already_done}/{args.num_random_seeds} trials exist, running {trials_needed} more)..."
                )
            
            # FIXED: Start from where we left off, not from 0
            for trial_idx in range(trials_already_done, args.num_random_seeds):
                trial_seed = args.seed + trial_idx
                rng.seed(trial_seed)
                selected = select_random_layers(candidate_indices, num_layers, rng)

                logger.info(
                    f"  [Trial {trial_idx+1}/{args.num_random_seeds}] "
                    f"layers={num_layers}, target_dim={target_dim}, seed={trial_seed}"
                )
                trial_result = run_random_sampling_trial(
                    model=model,
                    model_adapter=model_adapter,
                    device=device,
                    selected_layer_indices=selected,
                    all_discovered_layers=discovered_layers,
                    train_raw=train_raw,
                    train_labels=train_labels,
                    val_raw=val_raw,
                    val_labels=val_labels,
                    test_raw=test_raw,
                    test_labels=test_labels,
                    target_dim=target_dim,  # NEW: pass target_dim instead of compression_ratio
                    batch_size=args.batch_size,
                    dataset=args.dataset,
                    seed=trial_seed,
                    ood_raw=ood_raw,
                )
                trial_result.update(
                    {
                        "strategy": "global_random",
                        "num_layers": num_layers,
                        "trial_idx": trial_idx,
                        "trial_seed": trial_seed,
                    }
                )
                random_ablation_results["results_by_target_dim"][target_dim]["global_random"].append(
                    make_json_serializable(trial_result)
                )
                
                # SAVE IMMEDIATELY after each trial to avoid data loss
                final_results = build_final_results(
                    existing_results, geo_best_results, best_per_target_dim,
                    geo_comb_results, random_ablation_results, args,
                    baseline_results=baseline_results,
                    tulip_best_results=tulip_best_results,
                    tulip_comb_results=tulip_comb_results,
                )
                save_incremental_results(output_path, final_results)

    # ------------------------------------------------------------------
    # Block random strategy - iterate over target dimensions
    # ------------------------------------------------------------------
    if args.skip_block_random:
        logger.info("\n" + "=" * 80)
        logger.info("BLOCK RANDOM STRATEGY: SKIPPED (--skip-block-random flag set)")
        logger.info("=" * 80)
        logger.info("Note: Results showed block_random and global_random are identical.")
    elif block_groups:
        logger.info("\n" + "=" * 80)
        logger.info("BLOCK RANDOM STRATEGY")
        logger.info("=" * 80)
        block_keys = sorted(block_groups.keys())
        logger.info(f"Blocks available: {block_keys}")

        # Iterate over target dimensions (already initialized in global_random loop)
        for target_dim in target_dims:
            logger.info(f"\n{'='*80}")
            logger.info(f"TARGET DIMENSION: {target_dim} (Block Random)")
            logger.info(f"{'='*80}")

            target_data = random_ablation_results["results_by_target_dim"].get(target_dim, {})

            for num_layers in layer_counts:
                # FIXED: Count existing trials for this specific (num_layers, target_dim) combination
                existing_trials = [
                    r for r in target_data.get("block_random", [])
                    if r.get("num_layers") == num_layers
                ]
                
                # Skip only if we have ALL required trials
                if len(existing_trials) >= args.num_random_seeds:
                    logger.info(
                        f"⏭️  Skipping block_random num_layers={num_layers} for target_dim={target_dim}: "
                        f"already have {len(existing_trials)}/{args.num_random_seeds} trials"
                    )
                    continue
                
                # Check if we have enough layers across all blocks
                total_available = sum(len(block_groups[bk]) for bk in block_keys)
                if total_available < num_layers:
                    logger.warning(f"Skipping num_layers={num_layers}: only {total_available} layers available across blocks.")
                    continue

                # Calculate how many additional trials we need
                trials_already_done = len(existing_trials)
                trials_needed = args.num_random_seeds - trials_already_done
                
                if trials_needed > 0:
                    logger.info(
                        f"\nTesting {num_layers} random layers from blocks -> {target_dim} dims "
                        f"({trials_already_done}/{args.num_random_seeds} trials exist, running {trials_needed} more)..."
                    )
                
                # FIXED: Start from where we left off, not from 0
                for trial_idx in range(trials_already_done, args.num_random_seeds):
                        trial_seed = args.seed + trial_idx  # Use same seed base as global_random for fair comparison
                        rng.seed(trial_seed)

                        # Strategy: distribute layers across blocks as evenly as possible
                        selected = []

                        if num_layers <= len(block_keys):
                            # Pick from subset of blocks (e.g., 2 layers = 2 blocks)
                            chosen_blocks = rng.sample(block_keys, num_layers)
                            for bk in chosen_blocks:
                                if block_groups[bk]:
                                    selected.append(rng.choice(block_groups[bk]))
                        else:
                            # Pick multiple layers per block
                            layers_per_block = num_layers // len(block_keys)
                            remainder = num_layers % len(block_keys)

                            for i, bk in enumerate(block_keys):
                                if not block_groups[bk]:
                                    continue

                                # Some blocks get extra layer if there's remainder
                                n_from_block = layers_per_block + (1 if i < remainder else 0)
                                n_from_block = min(n_from_block, len(block_groups[bk]))

                                selected.extend(rng.sample(block_groups[bk], n_from_block))

                        selected = sorted(selected)

                        if not selected:
                            logger.warning("No block-based layers selected; skipping.")
                            continue

                        # Get unique blocks represented in selection (for logging)
                        selected_blocks = set()
                        for idx in selected:
                            block_name = get_block_for_idx(idx, block_groups)
                            if block_name:
                                selected_blocks.add(block_name)

                        logger.info(
                            f"  [Trial {trial_idx+1}/{args.num_random_seeds}] "
                            f"layers={num_layers}, blocks={len(selected_blocks)}, "
                            f"target_dim={target_dim}, seed={trial_seed}"
                        )
                        trial_result = run_random_sampling_trial(
                            model=model,
                            model_adapter=model_adapter,
                            device=device,
                            selected_layer_indices=selected,
                            all_discovered_layers=discovered_layers,
                            train_raw=train_raw,
                            train_labels=train_labels,
                            val_raw=val_raw,
                            val_labels=val_labels,
                            test_raw=test_raw,
                            test_labels=test_labels,
                            target_dim=target_dim,  # Use target_dim instead of compression_ratio
                            batch_size=args.batch_size,
                            dataset=args.dataset,
                            seed=trial_seed,
                            ood_raw=ood_raw,
                        )
                        trial_result.update(
                            {
                                "strategy": "block_random",
                                "num_layers": len(selected),  # Should equal num_layers
                                "trial_idx": trial_idx,
                                "trial_seed": trial_seed,
                            }
                        )
                        random_ablation_results["results_by_target_dim"][target_dim]["block_random"].append(
                            make_json_serializable(trial_result)
                        )
                        
                        # SAVE IMMEDIATELY after each trial to avoid data loss
                        final_results = build_final_results(
                            existing_results, geo_best_results, best_per_target_dim,
                            geo_comb_results, random_ablation_results, args,
                            baseline_results=baseline_results,
                            tulip_best_results=tulip_best_results,
                            tulip_comb_results=tulip_comb_results,
                        )
                        save_incremental_results(output_path, final_results)
    else:
        if not args.skip_block_random:
            logger.warning("Block grouping failed (no matching blocks); skipping block_random strategy.")

    # ===== DETERMINISTIC BASELINES: FIRST-K AND LAST-K =====
    # logger.info("\n" + "=" * 80)
    # logger.info("DETERMINISTIC BASELINES: FIRST-K AND LAST-K LAYERS")
    # logger.info("=" * 80)
    # logger.info("First-K: Shallow layers only (physical features)")
    # logger.info("Last-K: Deep layers only (semantic features)")

    # # Iterate over target dimensions and layer counts
    # for target_dim in target_dims:
    #     logger.info(f"\n{'='*80}")
    #     logger.info(f"TARGET DIMENSION: {target_dim} (Deterministic Baselines)")
    #     logger.info(f"{'='*80}")

    #     # Ensure target_dim structure exists with all required keys
    #     if target_dim not in random_ablation_results["results_by_target_dim"]:
    #         random_ablation_results["results_by_target_dim"][target_dim] = {
    #             "global_random": [],
    #             "block_random": [],
    #             "first_k_layers": [],
    #             "last_k_layers": [],
    #         }
    #     else:
    #         # Ensure first_k_layers and last_k_layers keys exist (may be missing in old results)
    #         if "first_k_layers" not in random_ablation_results["results_by_target_dim"][target_dim]:
    #             random_ablation_results["results_by_target_dim"][target_dim]["first_k_layers"] = []
    #         if "last_k_layers" not in random_ablation_results["results_by_target_dim"][target_dim]:
    #             random_ablation_results["results_by_target_dim"][target_dim]["last_k_layers"] = []
        
    #     target_data = random_ablation_results["results_by_target_dim"][target_dim]
        
    #     # Check which baselines already exist
    #     existing_first_k = {r.get("num_layers") for r in target_data.get("first_k_layers", [])}
    #     existing_last_k = {r.get("num_layers") for r in target_data.get("last_k_layers", [])}

    #     for num_layers in layer_counts:
    #         # Safety check: skip if k > len(candidate_indices)
    #         if num_layers > len(candidate_indices):
    #             logger.warning(f"Skipping num_layers={num_layers}: only {len(candidate_indices)} candidate layers available.")
    #             continue

    #         # ===== FIRST-K LAYERS (Shallow/Physical) =====
    #         # Skip if this configuration already exists
    #         if num_layers in existing_first_k:
    #             logger.info(f"⏭️  Skipping first_k_layers for k={num_layers}, target_dim={target_dim}: already exists")
    #         else:
    #             logger.info(f"\nTesting First-K layers: k={num_layers} -> {target_dim} dims...")
    #             first_k_indices = candidate_indices[:num_layers]
                
    #             # Get layer names for logging (candidate_indices contains model idx, need to find in discovered_layers)
    #             idx_to_layer = {layer["idx"]: layer for layer in discovered_layers}
    #             first_k_names = [idx_to_layer[idx]["name"] for idx in first_k_indices]
    #             logger.info(f"  Selected layers (first {num_layers}): {first_k_names}")

    #             trial_result = run_random_sampling_trial(
    #                 model=model,
    #                 model_adapter=model_adapter,
    #                 device=device,
    #                 selected_layer_indices=first_k_indices,
    #                 all_discovered_layers=discovered_layers,
    #                 train_raw=train_raw,
    #                 train_labels=train_labels,
    #                 val_raw=val_raw,
    #                 val_labels=val_labels,
    #                 test_raw=test_raw,
    #                 test_labels=test_labels,
    #                 target_dim=target_dim,
    #                 batch_size=args.batch_size,
    #                 dataset=args.dataset,
    #                 seed=args.seed,
    #             )
    #             trial_result.update({
    #                 "strategy": "first_k_layers",
    #                 "num_layers": num_layers,
    #             })
    #             random_ablation_results["results_by_target_dim"][target_dim]["first_k_layers"].append(
    #                 make_json_serializable(trial_result)
    #             )
    #             logger.info(f"  ECE: {trial_result['ece']:.6f}, Accuracy: {trial_result['accuracy']:.4f}")
                
    #             # Save incrementally after each baseline trial
    #             final_results = build_final_results(
    #                 existing_results, geo_best_results, best_per_target_dim,
    #                 geo_comb_results, random_ablation_results, args,
    #                 baseline_results=baseline_results,
    #                 tulip_best_results=tulip_best_results,
    #                 tulip_comb_results=tulip_comb_results,
    #             )
    #             save_incremental_results(output_path, final_results)

    #         # ===== LAST-K LAYERS (Deep/Semantic) =====
    #         # Skip if this configuration already exists
    #         if num_layers in existing_last_k:
    #             logger.info(f"⏭️  Skipping last_k_layers for k={num_layers}, target_dim={target_dim}: already exists")
    #         else:
    #             logger.info(f"\nTesting Last-K layers: k={num_layers} -> {target_dim} dims...")
    #             last_k_indices = candidate_indices[-num_layers:]
                
    #             # Get layer names for logging (candidate_indices contains model idx, need to find in discovered_layers)
    #             idx_to_layer = {layer["idx"]: layer for layer in discovered_layers}
    #             last_k_names = [idx_to_layer[idx]["name"] for idx in last_k_indices]
    #             logger.info(f"  Selected layers (last {num_layers}): {last_k_names}")

    #             trial_result = run_random_sampling_trial(
    #                 model=model,
    #                 model_adapter=model_adapter,
    #                 device=device,
    #                 selected_layer_indices=last_k_indices,
    #                 all_discovered_layers=discovered_layers,
    #                 train_raw=train_raw,
    #                 train_labels=train_labels,
    #                 val_raw=val_raw,
    #                 val_labels=val_labels,
    #                 test_raw=test_raw,
    #                 test_labels=test_labels,
    #                 target_dim=target_dim,
    #                 batch_size=args.batch_size,
    #                 dataset=args.dataset,
    #                 seed=args.seed,
    #             )
    #             trial_result.update({
    #                 "strategy": "last_k_layers",
    #                 "num_layers": num_layers,
    #             })
    #             random_ablation_results["results_by_target_dim"][target_dim]["last_k_layers"].append(
    #                 make_json_serializable(trial_result)
    #             )
    #             logger.info(f"  ECE: {trial_result['ece']:.6f}, Accuracy: {trial_result['accuracy']:.4f}")
                
    #             # Save incrementally after each baseline trial
    #             final_results = build_final_results(
    #                 existing_results, geo_best_results, best_per_target_dim,
    #                 geo_comb_results, random_ablation_results, args,
    #                 baseline_results=baseline_results,
    #                 tulip_best_results=tulip_best_results,
    #                 tulip_comb_results=tulip_comb_results,
    #             )
    #             save_incremental_results(output_path, final_results)

    # ===== VALIDATE TRIAL COUNTS =====
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATING TRIAL COUNTS")
    logger.info("=" * 80)
    
    for target_dim in target_dims:
        target_data = random_ablation_results["results_by_target_dim"].get(target_dim, {})
        for strategy in ["global_random", "block_random"]:
            if strategy not in target_data:
                continue
            trial_counts = {}
            for trial in target_data[strategy]:
                nl = trial.get("num_layers")
                if nl is not None:
                    trial_counts[nl] = trial_counts.get(nl, 0) + 1
            
            for nl, count in trial_counts.items():
                if count != args.num_random_seeds:
                    logger.warning(
                        f"⚠️  {strategy}: target_dim={target_dim}, num_layers={nl} has {count} trials, "
                        f"expected {args.num_random_seeds}"
                    )
                else:
                    logger.debug(
                        f"✓ {strategy}: target_dim={target_dim}, num_layers={nl} has correct count ({count} trials)"
                    )
    
    # ===== COMPUTE STATISTICS ACROSS RANDOM TRIALS =====
    if args.num_random_seeds > 1:
        logger.info("\n" + "=" * 80)
        logger.info("COMPUTING STATISTICS ACROSS RANDOM TRIALS")
        logger.info("=" * 80)
        
        # After all experiments, compute statistics per target dimension
        for target_dim in target_dims:
            logger.info(f"\n{'='*80}")
            logger.info(f"STATISTICS FOR TARGET_DIM={target_dim}")
            logger.info(f"{'='*80}")
            
            if target_dim not in random_ablation_results["results_by_target_dim"]:
                continue
            
            target_dim_results = random_ablation_results["results_by_target_dim"][target_dim]
            
            # Global random statistics
            if target_dim_results["global_random"]:
                global_stats = compute_trial_statistics(
                    target_dim_results["global_random"],
                    group_by=['num_layers']  # Only group by num_layers now
                )
                target_dim_results["global_random_statistics"] = global_stats
                print_trial_statistics(global_stats, f"Global Random (target_dim={target_dim})")
                
                # Find best configuration for this target_dim
                best_trial = min(target_dim_results["global_random"], key=lambda x: x['ece'])
                logger.info(f"\nBest configuration for target_dim={target_dim} (Global Random):")
                logger.info(f"  Layers: {best_trial['num_layers']}")
                logger.info(f"  ECE: {best_trial['ece']*100:.3f}%")
                logger.info(f"  Samples/sec: {best_trial['samples_per_second']:.1f}")
                target_dim_results["best_global_random"] = make_json_serializable(best_trial)
            
            # Block random statistics
            if target_dim_results["block_random"]:
                block_stats = compute_trial_statistics(
                    target_dim_results["block_random"],
                    group_by=['num_layers']  # Only group by num_layers now
                )
                target_dim_results["block_random_statistics"] = block_stats
                print_trial_statistics(block_stats, f"Block Random (target_dim={target_dim})")
                
                # Find best configuration for this target_dim
                best_trial = min(target_dim_results["block_random"], key=lambda x: x['ece'])
                logger.info(f"\nBest configuration for target_dim={target_dim} (Block Random):")
                logger.info(f"  Layers: {best_trial['num_layers']}")
                logger.info(f"  ECE: {best_trial['ece']*100:.3f}%")
                logger.info(f"  Samples/sec: {best_trial['samples_per_second']:.1f}")
                target_dim_results["best_block_random"] = make_json_serializable(best_trial)
        
        # Compare random strategies to fixed baseline (per target dimension)
        # Skip comparison for DINOv2 models since there's no fixed baseline
        if not is_dinov2:
            logger.info("\n" + "=" * 80)
            logger.info("COMPARING RANDOM TO FIXED BASELINE")
            logger.info("=" * 80)
        else:
            logger.info("\n" + "=" * 80)
            logger.info("SKIPPING FIXED BASELINE COMPARISON (DINOv2)")
            logger.info("=" * 80)
            logger.info("No fixed baseline available for DINOv2 models.")
        
        comparison_summary = {}  # Store comparison results
        
        # For each target dimension, compare random stats to geo_comb
        # Only do comparison if we have fixed layers (not DINOv2)
        for target_dim in target_dims:
            logger.info(f"\nTarget Dimension: {target_dim}")
            
            # Get fixed baseline (geo_comb) - only for non-DINOv2 models
            if not is_dinov2:
                geo_comb_for_td = [r for r in geo_comb_results if r['target_dim'] == target_dim]
                if geo_comb_for_td:
                    fixed_ece = geo_comb_for_td[0]['ece']
                    fixed_samples_per_second = geo_comb_for_td[0].get('samples_per_second', 0.0)
                    logger.info(f"  Fixed (geo_comb): ECE={fixed_ece*100:.3f}%, Throughput={fixed_samples_per_second:.1f} samples/sec")
                else:
                    fixed_ece = None
                    fixed_samples_per_second = None
                
                # Get TULIP fixed baseline
                tulip_comb_for_td = [r for r in tulip_comb_results if r['target_dim'] == target_dim]
                if tulip_comb_for_td:
                    tulip_ece = tulip_comb_for_td[0]['ece']
                    tulip_samples_per_second = tulip_comb_for_td[0].get('samples_per_second', 0.0)
                    logger.info(f"  TULIP (tulip_comb): ECE={tulip_ece*100:.3f}%, Throughput={tulip_samples_per_second:.1f} samples/sec")
                else:
                    tulip_ece = None
                    tulip_samples_per_second = None
            else:
                fixed_ece = None
                fixed_samples_per_second = None
                tulip_ece = None
                tulip_samples_per_second = None
            
            comparison_summary[f"target_dim_{target_dim}"] = {
                "target_dim": target_dim,
                "fixed_ece": float(fixed_ece) if fixed_ece is not None else None,
                "fixed_samples_per_second": float(fixed_samples_per_second) if fixed_samples_per_second is not None else None,
                "tulip_ece": float(tulip_ece) if tulip_ece is not None else None,
                "tulip_samples_per_second": float(tulip_samples_per_second) if tulip_samples_per_second is not None else None,
            }
            
            # Get statistics for this target dimension
            if target_dim in random_ablation_results["results_by_target_dim"]:
                target_dim_results = random_ablation_results["results_by_target_dim"][target_dim]
                
                # Global random stats (best num_layers configuration)
                if "best_global_random" in target_dim_results:
                    best_gr = target_dim_results["best_global_random"]
                    g_ece = best_gr['ece']
                    g_samples_per_second_best = best_gr.get('samples_per_second', 0.0)
                    
                    # Get mean/std from statistics if available
                    if "global_random_statistics" in target_dim_results:
                        # Find stats for the best num_layers
                        best_num_layers = best_gr['num_layers']
                        stats_key = f"num_layers_{best_num_layers}"
                        if stats_key in target_dim_results["global_random_statistics"]:
                            stats = target_dim_results["global_random_statistics"][stats_key]
                            g_mean = stats['ece']['mean']
                            g_std = stats['ece']['std']
                            g_cv = stats['ece']['coefficient_of_variation_pct']
                            g_samples_mean = stats.get('samples_per_second', {}).get('mean', g_samples_per_second_best)
                            g_samples_std = stats.get('samples_per_second', {}).get('std', 0.0)
                        else:
                            g_mean = g_ece
                            g_std = 0.0
                            g_cv = 0.0
                            g_samples_mean = g_samples_per_second_best
                            g_samples_std = 0.0
                    else:
                        g_mean = g_ece
                        g_std = 0.0
                        g_cv = 0.0
                        g_samples_mean = g_samples_per_second_best
                        g_samples_std = 0.0
                    
                    logger.info(f"  Global Random (best): ECE={g_mean*100:.3f}% ± {g_std*100:.3f}%, Throughput={g_samples_mean:.1f} ± {g_samples_std:.1f} samples/sec")
                    
                    # Check 120 FPS claim
                    if g_samples_mean >= 120:
                        logger.info(f"    ✓ Throughput >= 120 FPS")
                    else:
                        logger.info(f"    ⚠ Throughput below 120 FPS (mean={g_samples_mean:.1f})")
                    
                    comparison_summary[f"target_dim_{target_dim}"]["global_random"] = {
                        "mean_ece": float(g_mean),
                        "std_ece": float(g_std),
                        "cv_pct": float(g_cv),
                        "best_num_layers": int(best_gr['num_layers']),
                        "mean_samples_per_second": float(g_samples_mean),
                        "std_samples_per_second": float(g_samples_std),
                        "meets_120fps": bool(g_samples_mean >= 120),
                    }
                    
                    if fixed_ece is not None:
                        diff = (g_mean - fixed_ece) * 100
                        diff_pct = (diff / (fixed_ece * 100)) * 100 if fixed_ece > 0 else 0
                        logger.info(f"    vs Fixed ECE: {diff:+.3f}% ({diff_pct:+.1f}%)")
                        
                        comparison_summary[f"target_dim_{target_dim}"]["global_random"]["diff_from_fixed"] = float(diff)
                        comparison_summary[f"target_dim_{target_dim}"]["global_random"]["diff_from_fixed_pct"] = float(diff_pct)
                    
                    # Compare with TULIP
                    if tulip_ece is not None:
                        diff_t = (g_mean - tulip_ece) * 100
                        diff_t_pct = (diff_t / (tulip_ece * 100)) * 100 if tulip_ece > 0 else 0
                        logger.info(f"    vs TULIP ECE: {diff_t:+.3f}% ({diff_t_pct:+.1f}%)")
                        
                        comparison_summary[f"target_dim_{target_dim}"]["global_random"]["diff_from_tulip"] = float(diff_t)
                        comparison_summary[f"target_dim_{target_dim}"]["global_random"]["diff_from_tulip_pct"] = float(diff_t_pct)
                    
                    if fixed_samples_per_second is not None and fixed_samples_per_second > 0:
                        diff_samples = g_samples_mean - fixed_samples_per_second
                        diff_samples_pct = (diff_samples / fixed_samples_per_second) * 100
                        logger.info(f"    vs Fixed Throughput: {diff_samples:+.1f} samples/sec ({diff_samples_pct:+.1f}%)")
                        
                        comparison_summary[f"target_dim_{target_dim}"]["global_random"]["diff_samples_per_second"] = float(diff_samples)
                        comparison_summary[f"target_dim_{target_dim}"]["global_random"]["diff_samples_per_second_pct"] = float(diff_samples_pct)
                
                # Block random stats (best num_layers configuration)
                if "best_block_random" in target_dim_results:
                    best_br = target_dim_results["best_block_random"]
                    b_ece = best_br['ece']
                    b_samples_per_second_best = best_br.get('samples_per_second', 0.0)
                    
                    # Get mean/std from statistics if available
                    if "block_random_statistics" in target_dim_results:
                        # Find stats for the best num_layers
                        best_num_layers = best_br['num_layers']
                        stats_key = f"num_layers_{best_num_layers}"
                        if stats_key in target_dim_results["block_random_statistics"]:
                            stats = target_dim_results["block_random_statistics"][stats_key]
                            b_mean = stats['ece']['mean']
                            b_std = stats['ece']['std']
                            b_cv = stats['ece']['coefficient_of_variation_pct']
                            b_samples_mean = stats.get('samples_per_second', {}).get('mean', b_samples_per_second_best)
                            b_samples_std = stats.get('samples_per_second', {}).get('std', 0.0)
                        else:
                            b_mean = b_ece
                            b_std = 0.0
                            b_cv = 0.0
                            b_samples_mean = b_samples_per_second_best
                            b_samples_std = 0.0
                    else:
                        b_mean = b_ece
                        b_std = 0.0
                        b_cv = 0.0
                        b_samples_mean = b_samples_per_second_best
                        b_samples_std = 0.0
                    
                    logger.info(f"  Block Random (best): ECE={b_mean*100:.3f}% ± {b_std*100:.3f}%, Throughput={b_samples_mean:.1f} ± {b_samples_std:.1f} samples/sec")
                    
                    # Check 120 FPS claim
                    if b_samples_mean >= 120:
                        logger.info(f"    ✓ Throughput >= 120 FPS")
                    else:
                        logger.info(f"    ⚠ Throughput below 120 FPS (mean={b_samples_mean:.1f})")
                    
                    comparison_summary[f"target_dim_{target_dim}"]["block_random"] = {
                        "mean_ece": float(b_mean),
                        "std_ece": float(b_std),
                        "cv_pct": float(b_cv),
                        "best_num_layers": int(best_br['num_layers']),
                        "mean_samples_per_second": float(b_samples_mean),
                        "std_samples_per_second": float(b_samples_std),
                        "meets_120fps": bool(b_samples_mean >= 120),
                    }
                    
                    if fixed_ece is not None:
                        diff = (b_mean - fixed_ece) * 100
                        diff_pct = (diff / (fixed_ece * 100)) * 100 if fixed_ece > 0 else 0
                        logger.info(f"    vs Fixed ECE: {diff:+.3f}% ({diff_pct:+.1f}%)")
                        
                        comparison_summary[f"target_dim_{target_dim}"]["block_random"]["diff_from_fixed"] = float(diff)
                        comparison_summary[f"target_dim_{target_dim}"]["block_random"]["diff_from_fixed_pct"] = float(diff_pct)
                    
                    # Compare with TULIP
                    if tulip_ece is not None:
                        diff_t = (b_mean - tulip_ece) * 100
                        diff_t_pct = (diff_t / (tulip_ece * 100)) * 100 if tulip_ece > 0 else 0
                        logger.info(f"    vs TULIP ECE: {diff_t:+.3f}% ({diff_t_pct:+.1f}%)")
                        
                        comparison_summary[f"target_dim_{target_dim}"]["block_random"]["diff_from_tulip"] = float(diff_t)
                        comparison_summary[f"target_dim_{target_dim}"]["block_random"]["diff_from_tulip_pct"] = float(diff_t_pct)
                    
                    if fixed_samples_per_second is not None and fixed_samples_per_second > 0:
                        diff_samples = b_samples_mean - fixed_samples_per_second
                        diff_samples_pct = (diff_samples / fixed_samples_per_second) * 100
                        logger.info(f"    vs Fixed Throughput: {diff_samples:+.1f} samples/sec ({diff_samples_pct:+.1f}%)")
                        
                        comparison_summary[f"target_dim_{target_dim}"]["block_random"]["diff_samples_per_second"] = float(diff_samples)
                        comparison_summary[f"target_dim_{target_dim}"]["block_random"]["diff_samples_per_second_pct"] = float(diff_samples_pct)
        
        # Add comparison summary to results (only if we have comparisons)
        if comparison_summary:
            random_ablation_results["comparison_to_fixed"] = comparison_summary
    else:
        logger.info("\n⚠ Only 1 random seed used - skipping statistical analysis")
        logger.info("  Run with --num-random-seeds 3 or more for variance analysis")

    # ===== MERGE ALL RESULTS =====
    logger.info("\n" + "=" * 80)
    logger.info("MERGING ALL RESULTS")
    logger.info("=" * 80)

    # ===== EXPERIMENT SCOPE SUMMARY =====
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SCOPE SUMMARY")
    logger.info("=" * 80)
    
    if args.skip_single_layers:
        logger.info("✅ Multi-layer experiments completed:")
        logger.info("   - GEO_COMB (DAC fixed layers)")
        if tulip_layer_names:
            logger.info("   - TULIP_COMB (TULIP selected layers)")
        logger.info("   - Random ablation (global + block strategies)")
        logger.info("")
        logger.info("⏭️  Single-layer experiments skipped:")
        logger.info("   - GEO_BEST (individual DAC layers)")
        if tulip_layer_names:
            logger.info("   - TULIP_BEST (individual TULIP layers)")
    else:
        logger.info("✅ All experiments completed:")
        logger.info("   - GEO_BEST (individual DAC layers)")
        logger.info("   - GEO_COMB (DAC fixed layers)")
        if tulip_layer_names:
            logger.info("   - TULIP_BEST (individual TULIP layers)")
            logger.info("   - TULIP_COMB (TULIP selected layers)")
        logger.info("   - Random ablation (global + block strategies)")

    # ===== BASELINE COMPARISON SUMMARY =====  # (added)
    if baseline_results:
        logger.info("\n" + "=" * 80)
        logger.info("BASELINE CALIBRATION SUMMARY")
        logger.info("=" * 80)
        
        baseline_methods = [
            "uncalibrated", "temperature_scaling", "isotonic_toplabel",
            "platt_scaling", "beta_calibration", "density_aware_calibration",
            "probabilistic_skip_connections", "tulip", "geometric_physical_space"
        ]
        
        for method in baseline_methods:
            if method in baseline_results and baseline_results[method].get("ece") is not None:
                ece = baseline_results[method]["ece"]
                acc = baseline_results[method].get("acc", baseline_results[method].get("accuracy", 0))
                logger.info(f"  {method:35s}: ECE={ece:.6f}, Acc={acc:.4f}")
        
        # Compare best random ablation to best baseline
        if random_ablation_results and "results_by_target_dim" in random_ablation_results:
            best_random_ece = float('inf')
            for td_data in random_ablation_results["results_by_target_dim"].values():
                for trial in td_data.get("global_random", []):
                    best_random_ece = min(best_random_ece, trial.get("ece", float('inf')))
            
            baseline_eces = [
                baseline_results[m]["ece"]
                for m in baseline_methods
                if m in baseline_results and baseline_results[m].get("ece") is not None
            ]
            if baseline_eces:
                best_baseline_ece = min(baseline_eces)
                
                logger.info(f"\n📊 Best Random Ablation ECE: {best_random_ece:.6f}")
                logger.info(f"📊 Best Baseline ECE: {best_baseline_ece:.6f}")
                
                if best_random_ece < best_baseline_ece:
                    improvement = (best_baseline_ece - best_random_ece) / best_baseline_ece * 100
                    logger.info(f"✅ Random selection beats baselines by {improvement:.2f}%")
                else:
                    gap = (best_random_ece - best_baseline_ece) / best_baseline_ece * 100
                    logger.info(f"⚠️  Best baseline is {gap:.2f}% better than random selection")

    # Build final results (reuse the helper function)
    final_results = build_final_results(
        existing_results, geo_best_results, best_per_target_dim,
        geo_comb_results, random_ablation_results, args,
        baseline_results=baseline_results,
        tulip_best_results=tulip_best_results,
        tulip_comb_results=tulip_comb_results,
    )

    if existing_results:
        logger.info("Merged with any existing results")
    else:
        logger.info("No existing results to merge, creating new file")
    
    logger.info(f"Final file will have {len(final_results)} sections:")
    for key in final_results.keys():
        logger.info(f"  - {key}")

    # ===== SAVE FINAL RESULTS =====
    # Use the same incremental save function for consistency
    save_incremental_results(output_path, final_results)

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Total configurations tested:")
    logger.info(f"  GEO_BEST (single layers): {len(geo_best_results)}")
    logger.info(f"  GEO_COMB (fixed 4 layers): {len(geo_comb_results)}")
    
    total_global = sum(len(random_ablation_results["results_by_target_dim"][td]["global_random"]) 
                       for td in target_dims if td in random_ablation_results["results_by_target_dim"])
    total_block = sum(len(random_ablation_results["results_by_target_dim"][td]["block_random"]) 
                      for td in target_dims if td in random_ablation_results["results_by_target_dim"])
    logger.info(f"  Global random: {total_global}")
    logger.info(f"  Block random: {total_block}")


def parse_args():
    parser = argparse.ArgumentParser(description="Random Layer Ablation with geometric calibration")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--training-method", type=str, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--results-base-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-random-seeds", type=int, default=1)
    parser.add_argument(
        "--target-dims",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024, 2048, 3072, 4096],
        help="Target dimensions to test (fixed output sizes).",
    )
    parser.add_argument(
        "--layer-counts",
        nargs="+",
        type=int,
        default=[1, 2, 4, 6, 8, 12],
        help="Number of layers to sample for global strategy.",
    )
    parser.add_argument(
        "--skip-block-random",
        action="store_true",
        help="Skip block_random strategy (saves ~50%% computation time). Results showed it's identical to global_random.",
    )
    parser.add_argument(
        "--pixel-mode",
        action="store_true",
        help="Use raw pixels as features instead of model layers.",
    )
    parser.add_argument(
        "--no-first-stage-compression",
        action="store_true",
        help="Skip the initial SPP+JL-like compression. In pixel mode this means pixels -> target_dim directly.",
    )
    parser.add_argument(
        "--skip-fixed-layers",
        action="store_true",
        help="Skip fixed-layer (GEO_BEST / GEO_COMB) experiments and only run random-layer ablation.",
    )
    parser.add_argument(
        "--skip-single-layers",
        action="store_true",
        default=False,
        help=(
            "Skip single-layer oracle experiments (geo_best, tulip_best). "
            "Only run multi-layer experiments (geo_comb, tulip_comb, random ablation)."
        ),
    )
    parser.add_argument(
        "--ood-only",
        action="store_true",
        help=(
            "Run only OOD detection evaluation on existing results. "
            "Loads existing experiments and adds OOD metrics without recomputing calibration search."
        ),
    )
    parser.add_argument(
        "--ood-output-dir",
        type=str,
        default=None,
        help=(
            "Optional separate directory for OOD-augmented results. "
            "If not specified, updates the original results file in-place."
        ),
    )
    parser.add_argument(
        "--skip-ood",
        action="store_true",
        help="Skip OOD (SVHN) evaluation to save memory. Useful for large models or limited GPU memory.",
    )
    parser.add_argument(
        "--corruption-type",
        type=str,
        default=None,
        help="CIFAR-10-C / CIFAR-100-C corruption type (e.g., 'gaussian_noise', 'motion_blur'). "
             "If set, only the test loader will be corrupted; train/val remain clean.",
    )
    parser.add_argument(
        "--corruption-severity",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Severity level for CIFAR-C corruption (1-5). Must be set when --corruption-type is used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(args)