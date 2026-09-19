#!/usr/bin/env python3
"""
Global Coordinate Ablation Experiment

Tests Gil's "Global Coordinate" approach where we sample K random coordinates
from the entire model's activation space instead of selecting layers + SPP + JL.

Key difference from layer-based approach:
- Layer-based: Select L layers -> SPP pool each -> JL project each -> Concatenate -> Final projection
- Coordinate-based: Flatten ALL activations -> Sample K coordinates directly -> Done

The geometric calibration (StabilitySpace, GeometricCalibrator) is identical.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import normalize

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.coordinate_extraction import (
    discover_coordinate_space,
    plan_coordinate_extraction,
    plan_nested_coordinate_extraction,
    extract_coordinate_features,
    coerce_to_tensor,
    find_layer_module,
)
from Experiments.run_post_hoc_calibration import (
    PyTorchModelAdapter,
    construct_model_path,
    get_data_loaders,
    load_trained_model,
)
from Experiments.compare_dac_geometric import calculate_accuracy, calculate_ece

# Import helper functions from run_random_layer_ablation
from Experiments.run_random_layer_ablation import (
    set_global_seed,
    make_json_serializable,
    extract_raw_from_loader,
    save_incremental_results,
)

from Calibrators.geometric_calibrator_new import GeometricCalibrator
from Calibrators.density_aware_calibration import (
    DensityAwareCalibrator,
    get_dac_k_value,
    get_dac_target_layers,
    extract_dac_features,
)

from utils.logging_config import get_logger
logger = get_logger(__name__)


def validate_and_correct_coordinate_space(
    model: torch.nn.Module,
    layer_map: List[Dict],
    real_batch: torch.Tensor,
    device: torch.device,
) -> Tuple[List[Dict], int]:
    """
    Validate and correct the coordinate space by running a real batch using
    call-aware identifiers (base_name + call_count).
    """
    from collections import defaultdict

    model = model.to(device).eval()

    needed_base_names = set(layer["base_name"] for layer in layer_map)
    actual_sizes: Dict[str, int] = {}
    hooks = []

    def make_validation_hook(base_name: str):
        state = {"count": 0}

        def hook(module, _input, output):
            feat = coerce_to_tensor(output)
            if feat is None or not torch.is_tensor(feat):
                return
            unique_id = f"{base_name}#{state['count']}"
            state["count"] += 1
            actual_sizes[unique_id] = int(np.prod(feat.shape[1:]))

        return hook

    for name, module in model.named_modules():
        if name in needed_base_names:
            hooks.append(module.register_forward_hook(make_validation_hook(name)))

    try:
        with torch.no_grad():
            _ = model(real_batch)
    except Exception as e:
        logger.warning(f"Forward pass failed during validation: {e}")
    finally:
        for hook in hooks:
            hook.remove()

    corrections_made = 0
    corrected_layer_map = []
    current_idx = 0

    for layer in layer_map:
        unique_id = layer["name"]
        discovered_size = layer["size"]

        if unique_id in actual_sizes:
            actual_size = actual_sizes[unique_id]
            if actual_size != discovered_size:
                logger.warning(
                    f"Shape mismatch for layer '{unique_id}': "
                    f"discovered={discovered_size:,}, actual={actual_size:,}. Correcting."
                )
                corrections_made += 1
                size_to_use = actual_size
            else:
                size_to_use = discovered_size
        else:
            logger.warning(
                f"Layer call '{unique_id}' not found in actual forward pass. "
                f"Using discovered size {discovered_size:,}."
            )
            size_to_use = discovered_size

        corrected_layer_map.append(
            {
                **layer,
                "start_idx": current_idx,
                "end_idx": current_idx + size_to_use,
                "size": size_to_use,
            }
        )
        current_idx += size_to_use

    total_size = current_idx

    if corrections_made > 0:
        logger.info(
            f"✓ Corrected {corrections_made} layer size mismatch(es). "
            f"Total coordinate space: {total_size:,} (was {sum(l['size'] for l in layer_map):,})"
        )
    else:
        logger.info(f"✓ All layer sizes validated. Total coordinate space: {total_size:,}")

    return corrected_layer_map, total_size


def build_depth_grouped_dac_features(
    sampling_plan: Dict[str, np.ndarray],
    layer_map: List[Dict],
    features: np.ndarray,
    num_groups: int = 5,
) -> List[np.ndarray]:
    """
    Split extracted coordinate features into groups based on network depth.
    
    Groups coordinates by which network block they came from:
    - Group 0: pre-block layers (conv1, bn1, maxpool, pool0, norm0)
    - Group 1-4: block1 through block4 / dense1 through dense4
    
    Args:
        sampling_plan: Dict mapping layer_name -> local indices that were sampled
        layer_map: List of layer info dicts with 'name', 'start_idx', 'end_idx'
        features: Extracted features of shape (N_samples, K)
        num_groups: Number of depth groups (default 5 to match DAC structure)
    
    Returns:
        List of num_groups numpy arrays. Each has shape (N_samples, K_group).
        If a group is empty, returns array of zeros with shape (N_samples, 1).
    """
    def get_depth_group(layer_name: str) -> int:
        """Classify layer into depth group 0-4 based on name."""
        name_lower = layer_name.lower()
        
        # Pre-block (group 0): early layers before main blocks
        pre_block_patterns = ['conv1', 'bn1', 'maxpool', 'conv0', 'bn0', 'pool0', 
                              'norm0', 'features.conv0', 'features.norm0', 'features.pool0']
        if any(pattern in name_lower for pattern in pre_block_patterns):
            return 0
        
        # Block 1 (group 1)
        if any(x in name_lower for x in ['layer1', '.block1', 'dense1', 'denseblock1']):
            return 1
        # Include trans1 in block1 (transition after block1)
        if 'trans1' in name_lower or 'transition1' in name_lower:
            return 1
            
        # Block 2 (group 2)
        if any(x in name_lower for x in ['layer2', '.block2', 'dense2', 'denseblock2']):
            return 2
        if 'trans2' in name_lower or 'transition2' in name_lower:
            return 2
            
        # Block 3 (group 3)
        if any(x in name_lower for x in ['layer3', '.block3', 'dense3', 'denseblock3']):
            return 3
        if 'trans3' in name_lower or 'transition3' in name_lower:
            return 3
            
        # Block 4 (group 4)
        if any(x in name_lower for x in ['layer4', '.block4', 'dense4', 'denseblock4']):
            return 4
        # Final norm/bn before classifier
        if any(x in name_lower for x in ['norm5', 'bn2', 'features.norm5']):
            return 4
        
        # Default: assign to middle group
        return 2
    
    # Build mapping: feature column index -> depth group
    # We need to track which columns came from which layers
    # The sampling_plan tells us which layers have coordinates and how many
    
    column_to_group = []
    
    # Iterate through layer_map in order (features are concatenated in this order)
    for layer_info in layer_map:
        layer_name = layer_info['name']
        if layer_name not in sampling_plan:
            continue
        
        group = get_depth_group(layer_name)
        num_coords = len(sampling_plan[layer_name])
        column_to_group.extend([group] * num_coords)
    
    column_to_group = np.array(column_to_group)
    
    # Handle case where features might have different length due to reordering
    if len(column_to_group) != features.shape[1]:
        logger.warning(f"Column-to-group mapping length ({len(column_to_group)}) != "
                      f"features width ({features.shape[1]}). Using modulo assignment.")
        # Fallback: assign columns round-robin to groups
        column_to_group = np.arange(features.shape[1]) % num_groups
    
    # Split features by group
    groups = []
    group_sizes = []
    
    for g in range(num_groups):
        mask = (column_to_group == g)
        num_in_group = mask.sum()
        group_sizes.append(num_in_group)
        
        if num_in_group > 0:
            groups.append(features[:, mask])
        else:
            # Empty group: create single-column zeros (DAC needs at least 1 dim per layer)
            groups.append(np.zeros((features.shape[0], 1), dtype=features.dtype))
    
    logger.info(f"  Depth-based split: {group_sizes} coords per group (total {sum(group_sizes)})")
    
    return groups


def log_feature_diagnostics(features_list: List[torch.Tensor], mode_name: str, split: str = "train"):
    """Log feature statistics to compare pipelines."""
    logger.info(f"  Feature diagnostics for mode={mode_name}, split={split}:")
    for i, f in enumerate(features_list):
        f_np = f.numpy() if isinstance(f, torch.Tensor) else f
        l2_norms = np.linalg.norm(f_np, ord=2, axis=1)
        logger.info(f"    Layer {i}: shape={f_np.shape}, dtype={f_np.dtype}")
        logger.info(f"      Values: mean={f_np.mean():.4f}, std={f_np.std():.4f}, min={f_np.min():.4f}, max={f_np.max():.4f}")
        logger.info(f"      L2 norms: mean={l2_norms.mean():.4f}, std={l2_norms.std():.4f}, min={l2_norms.min():.4f}, max={l2_norms.max():.4f}")
        # Check if values are all zeros or very small
        if np.allclose(f_np, 0, atol=1e-6):
            logger.warning(f"      ⚠️  Layer {i} is all zeros!")
        elif np.abs(f_np.mean()) < 1e-6 and f_np.std() < 1e-6:
            logger.warning(f"      ⚠️  Layer {i} has very small values (mean≈0, std≈0)")


def run_dac_with_features(
    feature_layers_train: List[torch.Tensor],
    feature_layers_val: List[torch.Tensor],
    feature_layers_test: List[torch.Tensor],
    logits_train: torch.Tensor,
    logits_val: torch.Tensor,
    logits_test: torch.Tensor,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    dataset_name: str,
    device: torch.device,
    include_logits_as_layer: bool = True,
    mode_name: str = "unknown",
) -> Dict[str, Any]:
    """
    Run DAC calibration using the exact same pipeline as compare_dac_geometric.py.
    
    This function ensures consistency with the working implementation by:
    - Keeping features as torch tensors (not numpy arrays)
    - Passing logits as torch tensors
    - Letting DAC handle normalization internally
    
    Args:
        feature_layers_*: List of torch tensors, one per layer. 
                          These should NOT be pre-normalized - DAC handles normalization internally.
        logits_*: Torch tensors of logits
        include_logits_as_layer: If True, append logits as an additional density layer
    
    Returns:
        Dict with ece, accuracy, timing, diagnostics
    """
    k_value = get_dac_k_value(dataset_name)
    
    # Build feature lists - matching compare_dac_geometric.py exactly
    if include_logits_as_layer:
        train_features_with_logits = feature_layers_train + [logits_train]
        val_features_with_logits = feature_layers_val + [logits_val]
        test_features_with_logits = feature_layers_test + [logits_test]
    else:
        train_features_with_logits = feature_layers_train
        val_features_with_logits = feature_layers_val
        test_features_with_logits = feature_layers_test
    
    # Diagnostic logging to compare pipelines
    log_feature_diagnostics(train_features_with_logits, mode_name, "train")
    
    # Fit DAC - pass torch tensors directly (not numpy!)
    t_fit = time.perf_counter()
    dac = DensityAwareCalibrator(k=k_value, use_gpu=(device.type == 'cuda'))
    dac.fit(train_features_with_logits, val_features_with_logits, logits_val, val_labels)
    fit_time = time.perf_counter() - t_fit
    
    # Diagnostic: Check KNN distances and preprocessed features after fitting
    logger.info(f"  KNN Distance diagnostics for mode={mode_name}:")
    for i, scorer in enumerate(dac.layer_scorers):
        # Check preprocessed features (what actually goes into FAISS)
        val_feat_raw = val_features_with_logits[i]
        if isinstance(val_feat_raw, torch.Tensor):
            val_feat_raw = val_feat_raw.detach().cpu().numpy()
        val_feat_preprocessed = scorer._preprocess(val_features_with_logits[i])
        
        logger.info(f"    Layer {i} preprocessed features:")
        logger.info(f"      Shape: {val_feat_preprocessed.shape}")
        logger.info(f"      Values: mean={val_feat_preprocessed.mean():.6f}, std={val_feat_preprocessed.std():.6f}")
        logger.info(f"      L2 norms (should be ~1.0): mean={np.linalg.norm(val_feat_preprocessed, axis=1).mean():.6f}")
        
        # Get validation scores to check distribution
        val_scores = scorer.score(val_features_with_logits[i])
        logger.info(f"    Layer {i} KNN distances (validation): "
                   f"mean={np.mean(val_scores):.4f}, std={np.std(val_scores):.4f}, "
                   f"min={np.min(val_scores):.4f}, max={np.max(val_scores):.4f}")
        if np.mean(val_scores) > 100:
            logger.warning(f"      ⚠️  Layer {i} has very large KNN distances (mean > 100)!")
        if np.mean(val_scores) < 0.01:
            logger.warning(f"      ⚠️  Layer {i} has very small KNN distances (mean < 0.01)!")
    
    # Calibrate
    t_cal = time.perf_counter()
    calibrated_probs = dac.calibrate(test_features_with_logits, logits_test)
    calibrate_time = time.perf_counter() - t_cal
    
    # Metrics
    ece = float(calculate_ece(calibrated_probs, test_labels))
    acc = float(calculate_accuracy(calibrated_probs, test_labels))
    
    # Diagnostics
    num_classes = calibrated_probs.shape[1]
    max_confidences = np.max(calibrated_probs, axis=1)
    
    # Get temperature diagnostics if available
    temp_mean = temp_std = temp_min = temp_max = None
    if hasattr(dac, '_get_temperature'):
        test_knn_scores = []
        for i, scorer in enumerate(dac.layer_scorers):
            s_l = scorer.score(test_features_with_logits[i])
            test_knn_scores.append(s_l)
        knn_scores = np.column_stack(test_knn_scores)
        temperatures = dac._get_temperature(knn_scores)
        temp_mean = float(np.mean(temperatures))
        temp_std = float(np.std(temperatures))
        temp_min = float(np.min(temperatures))
        temp_max = float(np.max(temperatures))
    
    result = {
        'ece': ece,
        'accuracy': acc,
        'fit_time_s': fit_time,
        'calibrate_time_s': calibrate_time,
        'k_value': k_value,
        'num_layers': len(train_features_with_logits),
        'diagnostics': {
            'mean_confidence': float(np.mean(max_confidences)),
            'std_confidence': float(np.std(max_confidences)),
            'min_max_confidence': [float(np.min(max_confidences)), float(np.max(max_confidences))],
            'expected_uniform': float(1.0 / num_classes),
        }
    }
    
    if temp_mean is not None:
        result['diagnostics']['temperature'] = {
            'mean': temp_mean,
            'std': temp_std,
            'min': temp_min,
            'max': temp_max,
        }
    
    # Sanity check logging
    logger.info(f"  DAC Sanity Check:")
    logger.info(f"    ECE: {ece*100:.2f}% (expected: 2-4% for original mode)")
    logger.info(f"    Mean conf: {result['diagnostics']['mean_confidence']:.4f} (expected: 0.5-0.9)")
    if ece > 0.5:
        logger.error(f"    ⚠️  DAC ECE > 50% indicates pipeline mismatch with compare_dac_geometric.py!")
    if temp_mean is not None:
        logger.info(f"    Temperature: {temp_mean:.2f}±{temp_std:.2f} (expected: 1-5)")
    
    return result


def run_coordinate_trial(
    model: torch.nn.Module,
    model_adapter: PyTorchModelAdapter,
    device: torch.device,
    model_name: str,
    layer_map: List[Dict],
    total_coordinate_size: int,
    num_coordinates: int,  # K - the number of coordinates to sample
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    train_logits: torch.Tensor,
    val_logits: torch.Tensor,
    test_logits: torch.Tensor,
    dataset_name: str,
    batch_size: int,
    seed: int,
    dac_mode: str = "depth",
) -> Dict[str, Any]:
    """
    Run a single coordinate-based calibration trial.
    
    Args:
        num_coordinates: K - number of global coordinates to sample
        layer_map: From discover_coordinate_space (reusable across trials)
        total_coordinate_size: N - total activation space size
    
    Returns:
        Dict with ece, accuracy, timing, and metadata
    """
    t_start = time.perf_counter()
    
    # 1. Plan coordinate extraction (this is where randomness enters)
    t_plan = time.perf_counter()
    sampling_plan, global_order = plan_coordinate_extraction(
        total_size=total_coordinate_size,
        num_coordinates=num_coordinates,
        layer_map=layer_map,
        seed=seed,
    )
    plan_time = time.perf_counter() - t_plan
    
    # Log which layers got sampled
    layers_with_coords = list(sampling_plan.keys())
    coords_per_layer = {k: len(v) for k, v in sampling_plan.items()}
    logger.info(f"  Sampling plan: {len(layers_with_coords)} layers touched")
    logger.info(f"  Top 5 layers by coordinate count: {sorted(coords_per_layer.items(), key=lambda x: -x[1])[:5]}")
    
    # 2. Extract features using coordinate sampling
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    t_extract = time.perf_counter()
    
    train_features = extract_coordinate_features(
        model, train_loader, sampling_plan, layer_map, global_order, device=str(device)
    )
    val_features = extract_coordinate_features(
        model, val_loader, sampling_plan, layer_map, global_order, device=str(device)
    )
    test_features = extract_coordinate_features(
        model, test_loader, sampling_plan, layer_map, global_order, device=str(device)
    )
    
    extraction_time = time.perf_counter() - t_extract
    logger.info(f"  Extracted features: train={train_features.shape}, val={val_features.shape}, test={test_features.shape}")
    
    # 3. Prepare features for both calibrators
    # For Geometric: L2 normalize (same as layer-based approach)
    train_features_np = normalize(train_features.numpy(), norm='l2', axis=1)
    val_features_np = normalize(val_features.numpy(), norm='l2', axis=1)
    test_features_np = normalize(test_features.numpy(), norm='l2', axis=1)
    
    # For DAC: Keep as torch tensors (DAC handles normalization internally)
    # train_features, val_features, test_features are already torch tensors
    
    # 4. Geometric calibration (identical to layer-based)
    t_fit_geo = time.perf_counter()
    
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features_np,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,
        device=str(device),
    )
    
    geo_cal.fit(
        X_val_embed=val_features_np,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size,
    )
    fit_time_geo = time.perf_counter() - t_fit_geo
    
    t_calibrate_geo = time.perf_counter()
    calibrated_probs_geo = geo_cal.calibrate_batched(
        X_test_embed=test_features_np,
        X_test_original=test_raw,
        batch_size=batch_size,
    )
    calibrate_time_geo = time.perf_counter() - t_calibrate_geo
    
    # Compute Geometric metrics
    ece_geo = float(calculate_ece(calibrated_probs_geo, test_labels))
    acc_geo = float(calculate_accuracy(calibrated_probs_geo, test_labels))
    
    # 5. DAC calibration - try multiple modes to find what works
    dac_k = get_dac_k_value(dataset_name) if dac_mode != "skip" else None
    
    # Skip DAC if requested
    if dac_mode == "skip":
        logger.info(f"\n  {'='*60}")
        logger.info(f"  Skipping DAC calibration (--dac-mode skip)")
        logger.info(f"  {'='*60}")
        
        # Set DAC results to None/defaults
        dac_results = {}
        ece_dac = None
        acc_dac = None
        fit_time_dac = 0.0
        calibrate_time_dac = 0.0
        dac_num_pseudo_layers = None
        dac_total_coords = None
        dac_extract_time = 0.0
    else:
        # Determine which modes to run
        if dac_mode == "all":
            modes_to_run = ["single", "interleaved", "depth", "original"]
        else:
            modes_to_run = [dac_mode]
        
        # Store results for all modes
        dac_results = {}
        
        for current_mode in modes_to_run:
            logger.info(f"\n  {'='*60}")
            logger.info(f"  Running DAC mode: {current_mode}")
            logger.info(f"  {'='*60}")
            
            t_fit_dac = time.perf_counter()
            
            if current_mode == "single":
                # Mode 1: Single layer - use all K coordinates as one "layer" (no logits)
                # Keep as torch tensors
                train_features_dac = [train_features]
                val_features_dac = [val_features]
                test_features_dac = [test_features]
                
                dac_num_pseudo_layers = 1
                dac_total_coords = num_coordinates
                dac_extract_time = 0.0
                
                logger.info(f"  DAC mode=single: {num_coordinates} coords as 1 layer (no logits)")
                
            elif current_mode == "interleaved":
                # Mode 2: Sample 5*K, split interleaved into 5 pseudo-layers
                dac_num_pseudo_layers = 5
                dac_total_coords = dac_num_pseudo_layers * num_coordinates
                
                t_dac_extract = time.perf_counter()
                dac_seed = seed + 10000
                
                dac_sampling_plan, dac_global_order = plan_coordinate_extraction(
                    total_size=total_coordinate_size,
                    num_coordinates=dac_total_coords,
                    layer_map=layer_map,
                    seed=dac_seed,
                )
                
                train_dac_all = extract_coordinate_features(
                    model, train_loader, dac_sampling_plan, layer_map, dac_global_order, device=str(device)
                )
                val_dac_all = extract_coordinate_features(
                    model, val_loader, dac_sampling_plan, layer_map, dac_global_order, device=str(device)
                )
                test_dac_all = extract_coordinate_features(
                    model, test_loader, dac_sampling_plan, layer_map, dac_global_order, device=str(device)
                )
                
                dac_extract_time = time.perf_counter() - t_dac_extract
                
                # Interleaved split - keep as torch tensors
                train_dac_layers = []
                val_dac_layers = []
                test_dac_layers = []
                
                for layer_idx in range(dac_num_pseudo_layers):
                    indices = np.arange(layer_idx, dac_total_coords, dac_num_pseudo_layers)
                    
                    train_dac_layers.append(train_dac_all[:, indices])
                    val_dac_layers.append(val_dac_all[:, indices])
                    test_dac_layers.append(test_dac_all[:, indices])
                
                # Add logits as torch tensors (already on CPU from pre-computation)
                train_features_dac = train_dac_layers + [train_logits]
                val_features_dac = val_dac_layers + [val_logits]
                test_features_dac = test_dac_layers + [test_logits]
                
                logger.info(f"  DAC mode=interleaved: 5 x {num_coordinates} = {dac_total_coords} coords + logits = 6 layers")
                
            elif current_mode == "depth":
                # Mode 3: Use same K coordinates as geometric, group by network depth
                dac_num_pseudo_layers = 5
                dac_total_coords = num_coordinates  # Same as geometric!
                dac_extract_time = 0.0  # Reusing geometric extraction
                
                # Build depth-grouped features from the geometric features
                # build_depth_grouped_dac_features returns numpy arrays, convert back to torch
                train_depth_groups_np = build_depth_grouped_dac_features(
                    sampling_plan, layer_map, train_features.numpy(), num_groups=5
                )
                val_depth_groups_np = build_depth_grouped_dac_features(
                    sampling_plan, layer_map, val_features.numpy(), num_groups=5
                )
                test_depth_groups_np = build_depth_grouped_dac_features(
                    sampling_plan, layer_map, test_features.numpy(), num_groups=5
                )
                
                # Convert numpy arrays back to torch tensors
                train_dac_layers = [torch.from_numpy(g).float() for g in train_depth_groups_np]
                val_dac_layers = [torch.from_numpy(g).float() for g in val_depth_groups_np]
                test_dac_layers = [torch.from_numpy(g).float() for g in test_depth_groups_np]
                
                # Add logits as torch tensors
                train_features_dac = train_dac_layers + [train_logits]
                val_features_dac = val_dac_layers + [val_logits]
                test_features_dac = test_dac_layers + [test_logits]
                
                group_dims = [g.shape[1] for g in train_depth_groups_np]
                logger.info(f"  DAC mode=depth: {group_dims} coords by depth + logits = 6 layers")
            
            elif current_mode == "original":
                # Mode 4: Use DAC's original engineered layers as sanity check
                # This should produce ~1-4% ECE if implementation is correct (per DAC paper)
                model_name_for_dac = model_name.lower()
                dac_layer_names = get_dac_target_layers(model_name_for_dac, model)

                logger.info(f"  DAC mode=original: Using engineered layers {dac_layer_names}")

                # Extract features using DAC's original method - returns torch tensors AND logits
                t_dac_extract = time.perf_counter()
                train_feats_original, train_logits_original, _ = extract_dac_features(
                    model, train_loader, dac_layer_names, device
                )
                val_feats_original, val_logits_original, _ = extract_dac_features(
                    model, val_loader, dac_layer_names, device
                )
                test_feats_original, test_logits_original, _ = extract_dac_features(
                    model, test_loader, dac_layer_names, device
                )
                dac_extract_time = time.perf_counter() - t_dac_extract

                # Use logits from extract_dac_features (not pre-computed) - already torch tensors
                train_features_dac = train_feats_original + [train_logits_original]
                val_features_dac = val_feats_original + [val_logits_original]
                test_features_dac = test_feats_original + [test_logits_original]

                dac_num_pseudo_layers = len(train_features_dac)
                dac_total_coords = sum(f.shape[1] for f in train_features_dac)

                logger.info(f"  DAC original layer dims: {[f.shape[1] for f in train_features_dac]}")
            
            else:
                raise ValueError(f"Unknown dac_mode: {current_mode}")
            
            # Debug: log feature dimensions for each DAC layer
            logger.info(f"  DAC layer dims: {[f.shape[1] for f in train_features_dac]}")
            
            # Use the helper function for consistent DAC evaluation
            # Extract logits separately for the helper function
            if current_mode == "original":
                # Logits already extracted and included in feature lists
                logits_train_for_dac = train_logits_original
                logits_val_for_dac = val_logits_original
                logits_test_for_dac = test_logits_original
                feature_layers_train_for_dac = train_feats_original
                feature_layers_val_for_dac = val_feats_original
                feature_layers_test_for_dac = test_feats_original
            elif current_mode == "single":
                # Single mode: no logits as layer
                logits_train_for_dac = train_logits
                logits_val_for_dac = val_logits
                logits_test_for_dac = test_logits
                feature_layers_train_for_dac = train_features_dac
                feature_layers_val_for_dac = val_features_dac
                feature_layers_test_for_dac = test_features_dac
            else:
                # Other modes: logits are last element of feature lists
                logits_train_for_dac = train_features_dac[-1]
                logits_val_for_dac = val_features_dac[-1]
                logits_test_for_dac = test_features_dac[-1]
                feature_layers_train_for_dac = train_features_dac[:-1]
                feature_layers_val_for_dac = val_features_dac[:-1]
                feature_layers_test_for_dac = test_features_dac[:-1]
            
            dac_result = run_dac_with_features(
                feature_layers_train=feature_layers_train_for_dac,
                feature_layers_val=feature_layers_val_for_dac,
                feature_layers_test=feature_layers_test_for_dac,
                logits_train=logits_train_for_dac,
                logits_val=logits_val_for_dac,
                logits_test=logits_test_for_dac,
                val_labels=val_labels,
                test_labels=test_labels,
                dataset_name=dataset_name,
                device=device,
                include_logits_as_layer=(current_mode != "single"),
                mode_name=current_mode,
            )
            
            fit_time_dac = dac_result['fit_time_s']
            calibrate_time_dac = dac_result['calibrate_time_s']
            ece_dac = dac_result['ece']
            acc_dac = dac_result['accuracy']
            
            # Store results with mode suffix
            suffix = f"_{current_mode}" if dac_mode == "all" else ""
            dac_results[f"ece_dac{suffix}"] = ece_dac
            dac_results[f"accuracy_dac{suffix}"] = acc_dac
            dac_results[f"fit_time_dac_s{suffix}"] = fit_time_dac
            dac_results[f"calibrate_time_dac_s{suffix}"] = calibrate_time_dac
            dac_results[f"dac_num_pseudo_layers{suffix}"] = dac_num_pseudo_layers if current_mode != "single" else 1
            dac_results[f"dac_total_coordinates{suffix}"] = dac_total_coords
            dac_results[f"dac_extraction_time_s{suffix}"] = dac_extract_time
            
            logger.info(f"  ✓ {current_mode}: ECE={ece_dac*100:.2f}%, Acc={acc_dac:.4f}")
    
    # For backward compatibility, use the last mode's results as the main fields
    # If "all" mode, use "depth" as the default
    if dac_mode == "skip":
        # Already set above
        pass
    elif dac_mode == "all":
        ece_dac = dac_results["ece_dac_depth"]
        acc_dac = dac_results["accuracy_dac_depth"]
        fit_time_dac = dac_results["fit_time_dac_s_depth"]
        calibrate_time_dac = dac_results["calibrate_time_dac_s_depth"]
        dac_num_pseudo_layers = dac_results["dac_num_pseudo_layers_depth"]
        dac_total_coords = dac_results["dac_total_coordinates_depth"]
        dac_extract_time = dac_results["dac_extraction_time_s_depth"]
    else:
        ece_dac = dac_results["ece_dac"]
        acc_dac = dac_results["accuracy_dac"]
        fit_time_dac = dac_results["fit_time_dac_s"]
        calibrate_time_dac = dac_results["calibrate_time_dac_s"]
        dac_num_pseudo_layers = dac_results["dac_num_pseudo_layers"]
        dac_total_coords = dac_results["dac_total_coordinates"]
        dac_extract_time = dac_results["dac_extraction_time_s"]
    
    total_time = time.perf_counter() - t_start
    
    result_dict = {
        "num_coordinates": num_coordinates,
        "total_coordinate_space": total_coordinate_size,
        "num_layers_touched": len(layers_with_coords),
        "coords_per_layer": coords_per_layer,
        # Geometric results
        "ece_geometric": ece_geo,
        "accuracy_geometric": acc_geo,
        "fit_time_geometric_s": fit_time_geo,
        "calibrate_time_geometric_s": calibrate_time_geo,
        # DAC results (backward compatibility - uses last mode or depth if "all")
        "ece_dac": ece_dac,
        "accuracy_dac": acc_dac,
        "fit_time_dac_s": fit_time_dac,
        "calibrate_time_dac_s": calibrate_time_dac,
        # Backward compatibility aliases
        "ece": ece_geo,
        "accuracy": acc_geo,
        "fit_time_s": fit_time_geo,
        "calibrate_time_s": calibrate_time_geo,
        # Common metadata
        "plan_time_s": plan_time,
        "extraction_time_s": extraction_time,
        "total_time_s": total_time,
        "seed": seed,
        "dac_k": dac_k,
        "dac_mode": dac_mode,
        "dac_num_pseudo_layers": dac_num_pseudo_layers if dac_mode != "single" else 1,
        "dac_total_coordinates": dac_total_coords,
        "dac_extraction_time_s": dac_extract_time,
    }
    
    # Add all DAC mode results if "all" mode was used
    result_dict.update(dac_results)
    
    return result_dict


def run_nested_trial(
    model: torch.nn.Module,
    model_adapter: PyTorchModelAdapter,
    device: torch.device,
    model_name: str,
    layer_map: List[Dict],
    total_coordinate_size: int,
    coordinate_counts: List[int],
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    train_logits: torch.Tensor,
    val_logits: torch.Tensor,
    test_logits: torch.Tensor,
    dataset_name: str,
    batch_size: int,
    seed: int,
    dac_mode: str = "depth",
) -> Dict[int, Dict[str, Any]]:
    """
    Run nested sampling trial - extract once for max_k, then slice for each K.
    
    Returns:
        Dict mapping K -> result_dict
    """
    t_start = time.perf_counter()
    
    sorted_counts = sorted(coordinate_counts)
    max_k = max(sorted_counts)
    
    # Create all nested plans
    t_plan = time.perf_counter()
    nested_plans, master_indices = plan_nested_coordinate_extraction(
        total_size=total_coordinate_size,
        coordinate_counts=sorted_counts,
        layer_map=layer_map,
        seed=seed,
    )
    plan_time = time.perf_counter() - t_plan
    
    # Extract features ONCE for max_k
    max_sampling_plan, max_global_order = nested_plans[max_k]
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    t_extract = time.perf_counter()
    train_features_max = extract_coordinate_features(
        model, train_loader, max_sampling_plan, layer_map, max_global_order, device=str(device)
    )
    val_features_max = extract_coordinate_features(
        model, val_loader, max_sampling_plan, layer_map, max_global_order, device=str(device)
    )
    test_features_max = extract_coordinate_features(
        model, test_loader, max_sampling_plan, layer_map, max_global_order, device=str(device)
    )
    extraction_time = time.perf_counter() - t_extract
    
    logger.info(f"  Extracted max_k={max_k} features in {extraction_time:.2f}s")
    
    # Now evaluate each K by slicing
    results = {}
    
    for k in sorted_counts:
        logger.info(f"    Evaluating K={k}...")
        t_k = time.perf_counter()
        
        # Slice features to first K dimensions
        train_features = train_features_max[:, :k]
        val_features = val_features_max[:, :k]
        test_features = test_features_max[:, :k]
        
        # L2 normalize for Geometric
        train_features_np = normalize(train_features.numpy(), norm='l2', axis=1)
        val_features_np = normalize(val_features.numpy(), norm='l2', axis=1)
        test_features_np = normalize(test_features.numpy(), norm='l2', axis=1)
        
        # Geometric calibration
        t_fit_geo = time.perf_counter()
        geo_cal = GeometricCalibrator(
            model=model_adapter,
            X_train_embed=train_features_np,
            y_train=train_labels,
            library="fast_separation",
            auto_select_layer=False,
            device=str(device),
        )
        
        geo_cal.fit(
            X_val_embed=val_features_np,
            y_val=val_labels,
            X_val_original=val_raw,
            fit_batch_size=batch_size,
        )
        fit_time_geo = time.perf_counter() - t_fit_geo
        
        t_calibrate_geo = time.perf_counter()
        calibrated_probs_geo = geo_cal.calibrate_batched(
            X_test_embed=test_features_np,
            X_test_original=test_raw,
            batch_size=batch_size,
        )
        calibrate_time_geo = time.perf_counter() - t_calibrate_geo
        
        ece_geo = float(calculate_ece(calibrated_probs_geo, test_labels))
        acc_geo = float(calculate_accuracy(calibrated_probs_geo, test_labels))
        
        # DAC calibration - try multiple modes to find what works
        # Get sampling plan for this K (needed for depth mode)
        sampling_plan_k, _ = nested_plans[k]
        
        dac_k = get_dac_k_value(dataset_name) if dac_mode != "skip" else None
        
        # Skip DAC if requested
        if dac_mode == "skip":
            logger.info(f"      Skipping DAC calibration (--dac-mode skip)")
            
            # Set DAC results to None/defaults
            dac_results = {}
            ece_dac = None
            acc_dac = None
            fit_time_dac = 0.0
            calibrate_time_dac = 0.0
            dac_num_pseudo_layers = None
            dac_total_coords = None
            dac_extract_time = 0.0
        else:
            # Determine which modes to run
            if dac_mode == "all":
                modes_to_run = ["single", "interleaved", "depth", "original"]
            else:
                modes_to_run = [dac_mode]
            
            # Store results for all modes
            dac_results = {}
            
            # Cache for original DAC features (independent of K)
            original_train_features_dac = None
            original_val_features_dac = None
            original_test_features_dac = None
            original_train_logits_dac = None
            original_val_logits_dac = None
            original_test_logits_dac = None
            original_dac_num_pseudo_layers = None
            original_dac_total_coords = None
            original_dac_extract_time = None
            
            for current_mode in modes_to_run:
                logger.info(f"      {'-'*50}")
                logger.info(f"      Running DAC mode: {current_mode}")
                logger.info(f"      {'-'*50}")
                
                t_fit_dac = time.perf_counter()
                
                if current_mode == "single":
                    # Mode 1: Single layer - use all K coordinates as one "layer" (no logits)
                    # Keep as torch tensors
                    train_features_dac = [train_features]
                    val_features_dac = [val_features]
                    test_features_dac = [test_features]
                    
                    dac_num_pseudo_layers = 1
                    dac_total_coords = k
                    dac_extract_time = 0.0
                    
                    logger.info(f"      DAC mode=single: {k} coords as 1 layer (no logits)")
                    
                elif current_mode == "interleaved":
                    # Mode 2: Sample 5*K, split interleaved into 5 pseudo-layers
                    dac_num_pseudo_layers = 5
                    dac_total_coords = dac_num_pseudo_layers * k
                    
                    t_dac_extract = time.perf_counter()
                    dac_seed = seed + 10000
                    
                    dac_sampling_plan, dac_global_order = plan_coordinate_extraction(
                        total_size=total_coordinate_size,
                        num_coordinates=dac_total_coords,
                        layer_map=layer_map,
                        seed=dac_seed,
                    )
                    
                    train_dac_all = extract_coordinate_features(
                        model, train_loader, dac_sampling_plan, layer_map, dac_global_order, device=str(device)
                    )
                    val_dac_all = extract_coordinate_features(
                        model, val_loader, dac_sampling_plan, layer_map, dac_global_order, device=str(device)
                    )
                    test_dac_all = extract_coordinate_features(
                        model, test_loader, dac_sampling_plan, layer_map, dac_global_order, device=str(device)
                    )
                    
                    dac_extract_time = time.perf_counter() - t_dac_extract
                    
                    # Interleaved split - keep as torch tensors
                    train_dac_layers = []
                    val_dac_layers = []
                    test_dac_layers = []
                    
                    for layer_idx in range(dac_num_pseudo_layers):
                        indices = np.arange(layer_idx, dac_total_coords, dac_num_pseudo_layers)
                        
                        train_dac_layers.append(train_dac_all[:, indices])
                        val_dac_layers.append(val_dac_all[:, indices])
                        test_dac_layers.append(test_dac_all[:, indices])
                    
                    # Add logits as torch tensors
                    train_features_dac = train_dac_layers + [train_logits]
                    val_features_dac = val_dac_layers + [val_logits]
                    test_features_dac = test_dac_layers + [test_logits]
                    
                    logger.info(f"      DAC mode=interleaved: 5 x {k} = {dac_total_coords} coords + logits = 6 layers")
                    
                elif current_mode == "depth":
                    # Mode 3: Use same K coordinates as geometric, group by network depth
                    dac_num_pseudo_layers = 5
                    dac_total_coords = k  # Same as geometric!
                    dac_extract_time = 0.0  # Reusing geometric extraction
                    
                    # Build depth-grouped features from the geometric features
                    # build_depth_grouped_dac_features returns numpy arrays, convert back to torch
                    train_depth_groups_np = build_depth_grouped_dac_features(
                        sampling_plan_k, layer_map, train_features.numpy(), num_groups=5
                    )
                    val_depth_groups_np = build_depth_grouped_dac_features(
                        sampling_plan_k, layer_map, val_features.numpy(), num_groups=5
                    )
                    test_depth_groups_np = build_depth_grouped_dac_features(
                        sampling_plan_k, layer_map, test_features.numpy(), num_groups=5
                    )
                    
                    # Convert numpy arrays back to torch tensors
                    train_dac_layers = [torch.from_numpy(g).float() for g in train_depth_groups_np]
                    val_dac_layers = [torch.from_numpy(g).float() for g in val_depth_groups_np]
                    test_dac_layers = [torch.from_numpy(g).float() for g in test_depth_groups_np]
                    
                    # Add logits as torch tensors
                    train_features_dac = train_dac_layers + [train_logits]
                    val_features_dac = val_dac_layers + [val_logits]
                    test_features_dac = test_dac_layers + [test_logits]
                    
                    group_dims = [g.shape[1] for g in train_depth_groups_np]
                    logger.info(f"      DAC mode=depth: {group_dims} coords by depth + logits = 6 layers")
                
                elif current_mode == "original":
                    # Mode 4: Use DAC's original engineered layers as sanity check
                    # Features do not depend on K, so compute once and reuse
                    if original_train_features_dac is None:
                        model_name_for_dac = model_name.lower()
                        dac_layer_names = get_dac_target_layers(model_name_for_dac, model)

                        logger.info(f"      DAC mode=original: Using engineered layers {dac_layer_names}")

                        t_dac_extract = time.perf_counter()
                        train_feats_original, train_logits_original, _ = extract_dac_features(
                            model, train_loader, dac_layer_names, device
                        )
                        val_feats_original, val_logits_original, _ = extract_dac_features(
                            model, val_loader, dac_layer_names, device
                        )
                        test_feats_original, test_logits_original, _ = extract_dac_features(
                            model, test_loader, dac_layer_names, device
                        )
                        original_dac_extract_time = time.perf_counter() - t_dac_extract

                        # Use logits from extract_dac_features (not pre-computed) - already torch tensors
                        original_train_features_dac = train_feats_original + [train_logits_original]
                        original_val_features_dac = val_feats_original + [val_logits_original]
                        original_test_features_dac = test_feats_original + [test_logits_original]
                        
                        original_train_logits_dac = train_logits_original
                        original_val_logits_dac = val_logits_original
                        original_test_logits_dac = test_logits_original

                        original_dac_num_pseudo_layers = len(original_train_features_dac)
                        original_dac_total_coords = sum(f.shape[1] for f in original_train_features_dac)

                        logger.info(
                            f"      DAC original layer dims: "
                            f"{[f.shape[1] for f in original_train_features_dac]}"
                        )

                    train_features_dac = original_train_features_dac
                    val_features_dac = original_val_features_dac
                    test_features_dac = original_test_features_dac
                    dac_num_pseudo_layers = original_dac_num_pseudo_layers
                    dac_total_coords = original_dac_total_coords
                    dac_extract_time = original_dac_extract_time or 0.0
                
                else:
                    raise ValueError(f"Unknown dac_mode: {current_mode}")
                
                # Debug: log feature dimensions for each DAC layer
                logger.info(f"      DAC layer dims: {[f.shape[1] for f in train_features_dac]}")
                
                # Use the helper function for consistent DAC evaluation
                # Extract logits separately for the helper function
                if current_mode == "original":
                    # Logits already extracted and included in feature lists
                    logits_train_for_dac = original_train_logits_dac
                    logits_val_for_dac = original_val_logits_dac
                    logits_test_for_dac = original_test_logits_dac
                    feature_layers_train_for_dac = original_train_features_dac[:-1]
                    feature_layers_val_for_dac = original_val_features_dac[:-1]
                    feature_layers_test_for_dac = original_test_features_dac[:-1]
                elif current_mode == "single":
                    # Single mode: no logits as layer
                    logits_train_for_dac = train_logits
                    logits_val_for_dac = val_logits
                    logits_test_for_dac = test_logits
                    feature_layers_train_for_dac = train_features_dac
                    feature_layers_val_for_dac = val_features_dac
                    feature_layers_test_for_dac = test_features_dac
                else:
                    # Other modes: logits are last element of feature lists
                    logits_train_for_dac = train_features_dac[-1]
                    logits_val_for_dac = val_features_dac[-1]
                    logits_test_for_dac = test_features_dac[-1]
                    feature_layers_train_for_dac = train_features_dac[:-1]
                    feature_layers_val_for_dac = val_features_dac[:-1]
                    feature_layers_test_for_dac = test_features_dac[:-1]
                
                dac_result = run_dac_with_features(
                    feature_layers_train=feature_layers_train_for_dac,
                    feature_layers_val=feature_layers_val_for_dac,
                    feature_layers_test=feature_layers_test_for_dac,
                    logits_train=logits_train_for_dac,
                    logits_val=logits_val_for_dac,
                    logits_test=logits_test_for_dac,
                    val_labels=val_labels,
                    test_labels=test_labels,
                    dataset_name=dataset_name,
                    device=device,
                    include_logits_as_layer=(current_mode != "single"),
                    mode_name=current_mode,
                )
                
                fit_time_dac = dac_result['fit_time_s']
                calibrate_time_dac = dac_result['calibrate_time_s']
                ece_dac = dac_result['ece']
                acc_dac = dac_result['accuracy']
                
                # Store results with mode suffix
                suffix = f"_{current_mode}" if dac_mode == "all" else ""
                dac_results[f"ece_dac{suffix}"] = ece_dac
                dac_results[f"accuracy_dac{suffix}"] = acc_dac
                dac_results[f"fit_time_dac_s{suffix}"] = fit_time_dac
                dac_results[f"calibrate_time_dac_s{suffix}"] = calibrate_time_dac
                dac_results[f"dac_num_pseudo_layers{suffix}"] = dac_num_pseudo_layers if current_mode != "single" else 1
                dac_results[f"dac_total_coordinates{suffix}"] = dac_total_coords
                dac_results[f"dac_extraction_time_s{suffix}"] = dac_extract_time
                
                logger.info(f"      ✓ {current_mode}: ECE={ece_dac*100:.2f}%, Acc={acc_dac:.4f}")
        
        # For backward compatibility, use the last mode's results as the main fields
        # If "all" mode, use "depth" as the default
        if dac_mode == "skip":
            # Already set above
            pass
        elif dac_mode == "all":
            ece_dac = dac_results["ece_dac_depth"]
            acc_dac = dac_results["accuracy_dac_depth"]
            fit_time_dac = dac_results["fit_time_dac_s_depth"]
            calibrate_time_dac = dac_results["calibrate_time_dac_s_depth"]
            dac_num_pseudo_layers = dac_results["dac_num_pseudo_layers_depth"]
            dac_total_coords = dac_results["dac_total_coordinates_depth"]
            dac_extract_time = dac_results["dac_extraction_time_s_depth"]
        else:
            ece_dac = dac_results["ece_dac"]
            acc_dac = dac_results["accuracy_dac"]
            fit_time_dac = dac_results["fit_time_dac_s"]
            calibrate_time_dac = dac_results["calibrate_time_dac_s"]
            dac_num_pseudo_layers = dac_results["dac_num_pseudo_layers"]
            dac_total_coords = dac_results["dac_total_coordinates"]
            dac_extract_time = dac_results["dac_extraction_time_s"]
        
        # Get coords_per_layer for this K
        coords_per_layer = {name: len(indices) for name, indices in sampling_plan_k.items()}
        
        result_dict = {
            "num_coordinates": k,
            "total_coordinate_space": total_coordinate_size,
            "num_layers_touched": len(sampling_plan_k),
            "coords_per_layer": coords_per_layer,
            # Geometric results
            "ece_geometric": ece_geo,
            "accuracy_geometric": acc_geo,
            "fit_time_geometric_s": fit_time_geo,
            "calibrate_time_geometric_s": calibrate_time_geo,
            # DAC results (backward compatibility - uses last mode or depth if "all")
            "ece_dac": ece_dac,
            "accuracy_dac": acc_dac,
            "fit_time_dac_s": fit_time_dac,
            "calibrate_time_dac_s": calibrate_time_dac,
            # Backward compatibility aliases
            "ece": ece_geo,
            "accuracy": acc_geo,
            "fit_time_s": fit_time_geo,
            "calibrate_time_s": calibrate_time_geo,
            # Common metadata
            "plan_time_s": plan_time / len(sorted_counts),  # Amortized
            "extraction_time_s": extraction_time / len(sorted_counts),  # Amortized
            "total_time_s": time.perf_counter() - t_k,
            "seed": seed,
            "dac_k": dac_k,
            "dac_mode": dac_mode,
            "dac_num_pseudo_layers": dac_num_pseudo_layers if dac_mode != "single" else 1,
            "dac_total_coordinates": dac_total_coords,
            "dac_extraction_time_s": dac_extract_time,
            "sampling_strategy": "nested",
        }
        
        # Add all DAC mode results if "all" mode was used
        result_dict.update(dac_results)
        
        results[k] = result_dict
        
        if dac_mode == "skip":
            logger.info(f"      Geometric: ECE={ece_geo:.6f}, Acc={acc_geo:.4f} | DAC: skipped")
        else:
            logger.info(f"      Geometric: ECE={ece_geo:.6f}, Acc={acc_geo:.4f} | "
                       f"DAC: ECE={ece_dac:.6f}, Acc={acc_dac:.4f}")
    
    return results


def run_experiments(args):
    """Main experiment driver - supports both independent and nested sampling."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine output filename based on sampling strategy
    strategy_suffix = "_nested" if args.nested_sampling else "_independent"
    
    if args.corruption_type:
        output_path = os.path.join(
            args.output_dir,
            f"coordinate_ablation_{args.training_method}_{args.dataset}_{args.model_name}_seed{args.seed}{strategy_suffix}_{args.corruption_type}_sev{args.corruption_severity}.json"
        )
    else:
        output_path = os.path.join(
            args.output_dir,
            f"coordinate_ablation_{args.training_method}_{args.dataset}_{args.model_name}_seed{args.seed}{strategy_suffix}.json"
        )
    
    # Load existing results if any
    existing_results = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing_results = json.load(f)
        logger.info(f"Loaded existing results from {output_path}")
    
    # Setup
    set_global_seed(args.seed)
    device = torch.device(args.device)
    
    # Determine required image size for DINOv2 models
    is_dinov2 = 'dinov2' in args.model_name.lower()
    image_size = 224 if is_dinov2 else None
    
    # Load data
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset,
        args.batch_size,
        seed=args.seed,
        image_size=image_size,
        corruption_type=getattr(args, 'corruption_type', None),
        corruption_severity=getattr(args, 'corruption_severity', None),
    )
    
    # Extract raw numpy arrays
    train_raw, train_labels = extract_raw_from_loader(train_loader)
    val_raw, val_labels = extract_raw_from_loader(val_loader)
    test_raw, test_labels = extract_raw_from_loader(test_loader)
    
    logger.info(f"Data shapes: train={train_raw.shape}, val={val_raw.shape}, test={test_raw.shape}")
    
    # Load model
    model_path = construct_model_path(
        args.results_base_dir, args.training_method, args.dataset, args.model_name, args.seed
    )
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    
    model = load_trained_model(model_path, args.model_name, num_classes, device, dataset=args.dataset)
    model_adapter = PyTorchModelAdapter(model, device, args.dataset)
    
    # Extract logits once (used for DAC calibration)
    logger.info(f"\n{'='*80}")
    logger.info("EXTRACTING LOGITS")
    logger.info(f"{'='*80}")
    
    def get_logits_and_labels(loader):
        model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 2:
                    data, labels = batch
                else:
                    data = batch[0]
                    labels = torch.zeros(len(data), dtype=torch.long)
                
                data = data.to(device)
                logits = model(data)
                logits_list.append(logits.cpu())
                labels_list.append(labels)
        return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)
    
    train_logits, _ = get_logits_and_labels(train_loader)
    val_logits, _ = get_logits_and_labels(val_loader)
    test_logits, _ = get_logits_and_labels(test_loader)
    
    logger.info(f"Extracted logits: train={train_logits.shape}, val={val_logits.shape}, test={test_logits.shape}")
    
    # Discover coordinate space (do this ONCE, reuse for all trials)
    # Use actual input shape from data to ensure consistency
    input_shape = (1,) + train_raw.shape[1:]  # e.g., (1, 3, 32, 32)
    
    # Verify the shape matches what we'll actually use during extraction
    # Get a sample batch to double-check
    sample_batch = next(iter(train_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_data = sample_batch[0]
    else:
        sample_data = sample_batch
    actual_input_shape = (1,) + sample_data.shape[1:]
    
    if input_shape != actual_input_shape:
        logger.warning(f"Input shape mismatch: computed {input_shape}, actual batch shape {actual_input_shape}")
        logger.warning(f"Using actual shape: {actual_input_shape}")
        input_shape = actual_input_shape
    
    logger.info(f"\n{'='*80}")
    logger.info("DISCOVERING COORDINATE SPACE")
    logger.info(f"{'='*80}")
    logger.info(f"Using input shape: {input_shape}")
    
    layer_map, total_size = discover_coordinate_space(
        model, input_shape=input_shape, device=str(device)
    )
    
    logger.info(f"Initial discovery: {total_size:,} scalars across {len(layer_map)} layers")
    
    # Validate and correct coordinate space using a real batch
    logger.info(f"\n{'='*80}")
    logger.info("VALIDATING COORDINATE SPACE WITH REAL BATCH")
    logger.info(f"{'='*80}")
    
    # Get a fresh real batch from the data loader for validation
    # Ensure we use the same batch size that will be used during extraction
    validation_batch = next(iter(train_loader))
    if isinstance(validation_batch, (list, tuple)):
        real_batch = validation_batch[0]
    else:
        real_batch = validation_batch
    
    # Convert to tensor and move to device if needed
    if not isinstance(real_batch, torch.Tensor):
        real_batch = torch.from_numpy(real_batch) if isinstance(real_batch, np.ndarray) else real_batch
    real_batch = real_batch.to(device)
    
    # Validate and correct - this will catch most size mismatches
    layer_map, total_size = validate_and_correct_coordinate_space(
        model=model,
        layer_map=layer_map,
        real_batch=real_batch,
        device=device,
    )
    
    # Note: Some layers may still have dynamic shapes that vary between batches.
    # The extraction code handles this gracefully by filtering out-of-bounds indices.
    # If you see many warnings, it may indicate the model has conditional logic or
    # batch-dependent behavior that causes layer sizes to vary.
    
    logger.info(f"✓ Validated coordinate space: {total_size:,} scalars across {len(layer_map)} layers")
    
    # Initialize results structure
    results_by_num_coordinates = existing_results.get("results_by_num_coordinates", {})
    
    # Test different K values (similar to target_dims in layer-based)
    # These should roughly match the layer-based target_dims for fair comparison
    coordinate_counts = sorted(args.coordinate_counts)  # e.g., [64, 128, 256, 512, 1024, 2048, 4096]
    
    logger.info(f"\n{'='*80}")
    logger.info("RUNNING COORDINATE ABLATION EXPERIMENTS")
    logger.info(f"{'='*80}")
    logger.info(f"Testing K values: {coordinate_counts}")
    logger.info(f"Trials per K: {args.num_random_seeds}")
    logger.info(f"Sampling strategy: {'NESTED' if args.nested_sampling else 'INDEPENDENT'}")
    
    if args.nested_sampling:
        # NESTED SAMPLING: Run all K values together per trial
        for trial_idx in range(args.num_random_seeds):
            # Check if this trial already exists for all K values
            trial_exists = all(
                any(t.get("trial_idx") == trial_idx for t in results_by_num_coordinates.get(str(k), []))
                for k in coordinate_counts
            )
            
            if trial_exists:
                logger.info(f"\nTrial {trial_idx+1}: All K values already complete, skipping")
                continue
            
            trial_seed = args.seed + trial_idx
            logger.info(f"\n{'='*80}")
            logger.info(f"NESTED Trial {trial_idx+1}/{args.num_random_seeds} (seed={trial_seed})")
            logger.info(f"{'='*80}")
            
            try:
                trial_results = run_nested_trial(
                    model=model,
                    model_adapter=model_adapter,
                    device=device,
                    model_name=args.model_name,
                    layer_map=layer_map,
                    total_coordinate_size=total_size,
                    coordinate_counts=coordinate_counts,
                    train_raw=train_raw,
                    train_labels=train_labels,
                    val_raw=val_raw,
                    val_labels=val_labels,
                    test_raw=test_raw,
                    test_labels=test_labels,
                    train_logits=train_logits,
                    val_logits=val_logits,
                    test_logits=test_logits,
                    dataset_name=args.dataset,
                    batch_size=args.batch_size,
                    seed=trial_seed,
                    dac_mode=args.dac_mode,
                )
                
                # Store results for each K
                for k, result in trial_results.items():
                    k_key = str(k)
                    if k_key not in results_by_num_coordinates:
                        results_by_num_coordinates[k_key] = []
                    result["trial_idx"] = trial_idx
                    results_by_num_coordinates[k_key].append(result)
                
                # Save incrementally
                final_output = {
                    "experiment_config": {
                        "dataset": args.dataset,
                        "model_name": args.model_name,
                        "training_method": args.training_method,
                        "seed": args.seed,
                        "total_coordinate_space": total_size,
                        "num_layers_discovered": len(layer_map),
                        "method": "global_coordinate_sampling",
                        "sampling_strategy": "nested",
                        "coordinate_counts": coordinate_counts,
                        "num_random_trials": args.num_random_seeds,
                    },
                    "results_by_num_coordinates": results_by_num_coordinates,
                }
                save_incremental_results(output_path, final_output)
                
            except Exception as e:
                logger.error(f"Trial {trial_idx+1} failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
    
    else:
        # INDEPENDENT SAMPLING: Original behavior - sample independently for each K
        for num_coords in coordinate_counts:
            k_key = str(num_coords)
            
            if k_key not in results_by_num_coordinates:
                results_by_num_coordinates[k_key] = []
            
            # Check how many trials already exist
            existing_trials = len(results_by_num_coordinates[k_key])
            
            if existing_trials >= args.num_random_seeds:
                logger.info(f"\nK={num_coords}: All {args.num_random_seeds} trials already completed, skipping")
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"K={num_coords} ({existing_trials}/{args.num_random_seeds} trials done)")
            logger.info(f"{'='*80}")
            
            for trial_idx in range(existing_trials, args.num_random_seeds):
                trial_seed = args.seed + trial_idx
                
                logger.info(f"\n  Trial {trial_idx+1}/{args.num_random_seeds} (seed={trial_seed})")
                
                try:
                    trial_result = run_coordinate_trial(
                        model=model,
                        model_adapter=model_adapter,
                        device=device,
                        model_name=args.model_name,
                        layer_map=layer_map,
                        total_coordinate_size=total_size,
                        num_coordinates=num_coords,
                        train_raw=train_raw,
                        train_labels=train_labels,
                        val_raw=val_raw,
                        val_labels=val_labels,
                        test_raw=test_raw,
                        test_labels=test_labels,
                        train_logits=train_logits,
                        val_logits=val_logits,
                        test_logits=test_logits,
                        dataset_name=args.dataset,
                        batch_size=args.batch_size,
                        seed=trial_seed,
                        dac_mode=args.dac_mode,
                    )
                    
                    trial_result["trial_idx"] = trial_idx
                    trial_result["sampling_strategy"] = "independent"
                    results_by_num_coordinates[k_key].append(trial_result)
                    
                    # Log both results for comparison
                    logger.info(f"    ✓ Geometric: ECE={trial_result['ece_geometric']*100:.2f}%, "
                              f"Acc={trial_result['accuracy_geometric']:.4f} | "
                              f"DAC: ECE={trial_result['ece_dac']*100:.2f}%, "
                              f"Acc={trial_result['accuracy_dac']:.4f}")
                    logger.info(f"    ✓ Times: extract={trial_result['extraction_time_s']:.2f}s, "
                              f"geo_fit={trial_result['fit_time_geometric_s']:.2f}s, "
                              f"dac_fit={trial_result['fit_time_dac_s']:.2f}s, "
                              f"geo_cal={trial_result['calibrate_time_geometric_s']:.2f}s, "
                              f"dac_cal={trial_result['calibrate_time_dac_s']:.2f}s")
                    
                    # Save incrementally after each trial
                    final_output = {
                        "experiment_config": {
                            "dataset": args.dataset,
                            "model_name": args.model_name,
                            "training_method": args.training_method,
                            "seed": args.seed,
                            "total_coordinate_space": total_size,
                            "num_layers_discovered": len(layer_map),
                            "method": "global_coordinate_sampling",
                            "sampling_strategy": "independent",
                            "coordinate_counts": coordinate_counts,
                            "num_random_trials": args.num_random_seeds,
                        },
                        "results_by_num_coordinates": results_by_num_coordinates,
                    }
                    
                    save_incremental_results(output_path, final_output)
                    
                except Exception as e:
                    logger.error(f"    ✗ Trial {trial_idx+1} failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Continue with next trial
                    continue
    
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Sampling strategy: {'NESTED' if args.nested_sampling else 'INDEPENDENT'}")
    
    # Print summary
    logger.info(f"\nSummary by K:")
    for k_key in sorted(results_by_num_coordinates.keys(), key=int):
        trials = results_by_num_coordinates[k_key]
        if trials:
            # Geometric summary
            eces_geo = [t['ece_geometric'] for t in trials if 'ece_geometric' in t]
            if eces_geo:
                mean_ece_geo = np.mean(eces_geo) * 100
                std_ece_geo = np.std(eces_geo) * 100 if len(eces_geo) > 1 else 0
            
            # DAC summary - check for all modes
            dac_modes = []
            # Check if we have mode-specific results (from "all" mode)
            mode_keys = ['ece_dac_single', 'ece_dac_interleaved', 'ece_dac_depth', 'ece_dac_original']
            for mode_key in mode_keys:
                eces_mode = [t[mode_key] for t in trials if mode_key in t and t[mode_key] is not None]
                if eces_mode:
                    mean_ece = np.mean(eces_mode) * 100
                    std_ece = np.std(eces_mode) * 100 if len(eces_mode) > 1 else 0
                    mode_name = mode_key.replace('ece_dac_', '')
                    dac_modes.append((mode_name, mean_ece, std_ece))
            
            # If no mode-specific results, use backward compatibility field
            if not dac_modes:
                eces_dac = [t['ece_dac'] for t in trials if 'ece_dac' in t and t['ece_dac'] is not None]
                if eces_dac:
                    mean_ece_dac = np.mean(eces_dac) * 100
                    std_ece_dac = np.std(eces_dac) * 100 if len(eces_dac) > 1 else 0
                    # Try to get the actual mode name from the first trial
                    mode_name = trials[0].get('dac_mode', 'default') if trials else 'default'
                    dac_modes.append((mode_name, mean_ece_dac, std_ece_dac))
            
            if eces_geo:
                logger.info(f"  K={k_key}: {len(trials)} trials")
                logger.info(f"    Geometric: ECE={mean_ece_geo:.2f}±{std_ece_geo:.2f}%")
                if dac_modes:
                    for mode_name, mean_ece, std_ece in dac_modes:
                        logger.info(f"    DAC ({mode_name}): ECE={mean_ece:.2f}±{std_ece:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Global Coordinate Ablation Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name (e.g., cifar10, cifar100)")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name (e.g., resnet18, densenet121)")
    parser.add_argument("--training-method", type=str, required=True,
                       help="Training method (e.g., baseline, augmix)")
    parser.add_argument("--seed", type=int, default=11,
                       help="Base random seed")
    parser.add_argument("--results-base-dir", type=str, required=True,
                       help="Base directory containing trained models")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for data processing")
    parser.add_argument("--num-random-seeds", type=int, default=5,
                       help="Number of random trials per K value")
    parser.add_argument(
        "--coordinate-counts", nargs="+", type=int,
        default=[64, 128, 256, 512, 1024, 2048, 4096],
        help="Number of coordinates (K) to test"
    )
    parser.add_argument(
        "--corruption-type", type=str, default=None,
        help="Corruption type for CIFAR-C (e.g., gaussian_noise)"
    )
    parser.add_argument(
        "--corruption-severity", type=int, default=None,
        help="Corruption severity level (1-5)"
    )
    parser.add_argument(
        "--dac-mode", type=str, default="depth",
        choices=["single", "interleaved", "depth", "original", "all", "skip"],
        help=(
            "DAC feature structuring mode: "
            "single (1 layer, no logits), "
            "interleaved (5*K split), "
            "depth (semantic grouping), "
            "original (engineered DAC layers), "
            "all (run all modes), "
            "skip (skip DAC calibration entirely)"
        )
    )
    parser.add_argument(
        "--nested-sampling",
        action="store_true",
        help="Use nested sampling (K=8 subset of K=16 subset of K=32...). "
             "Default is independent sampling for each K."
    )
    return parser.parse_args()


if __name__ == "__main__":
    from utils.logging_config import setup_logging

    setup_logging()
    args = parse_args()
    run_experiments(args)

