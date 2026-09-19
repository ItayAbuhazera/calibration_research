"""
Alpha Ablation Study for MarginTailCVaRCalculator

This experiment tests different alpha (α) values for the MarginTailCVaRCalculator metric
to determine the optimal tail percentage for each dataset.

Hypothesis:
- CIFAR-100 (fine-grained, 100 classes): may prefer α = 0.10-0.15 (tighter tail)
- CIFAR-10 (coarse, 10 classes): may prefer α = 0.20-0.30 (broader tail)
- SVHN (digits, high accuracy): unclear, needs testing

The script tests alpha values: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Ensure project root on path for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Calibrators.geometric_calibrator_new import GeometricCalibrator, LayerMetrics
from Calibrators.metrics import MarginTailCVaRCalculator, AdHocLayerSelector, CrossModalLayerMapper
from Calibrators.nc_metrics import NC1CollapseMetricCalculator, NC4SeparabilityCalculator
from Experiments.run_post_hoc_calibration import PyTorchModelAdapter, get_data_loaders, load_trained_model
from Calibrators.calibration_utils import compute_ece

from utils.logging_config import get_logger
logger = get_logger(__name__)


# Alpha values to test
ALPHA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


@dataclass
class AlphaExperimentConfig:
    """Configuration for a single alpha experiment."""
    dataset: str
    model: str
    training_method: str
    seed: int
    alpha: float
    normalization_method: str = "tanh"
    normalization_scale: float = 2.0
    output_file: Path = None


def run_alpha_experiment(
    config: AlphaExperimentConfig,
    checkpoint_base_dir: Path,
    device: str = 'cuda',
    batch_size: int = 512,
    bins: int = 15,
    compression_ratio: Optional[float] = None,
    normalization_methods: Optional[List[str]] = None,
    normalization_scales: Optional[List[float]] = None,
) -> Dict:
    """
    Run a single alpha experiment: compute all geometric metrics once, then apply all selection strategies.
    
    Args:
        config: Experiment configuration (normalization_method/scale in config are defaults/fallbacks)
        normalization_methods: List of normalization methods to test (default: use config.normalization_method)
        normalization_scales: List of normalization scales to test (default: use config.normalization_scale)
    
    Returns:
        Dictionary with experiment results for ALL selection strategies:
        {
            'selected_layer_per_config': {
                # CVaR variants (existing)
                'margin_cvar_tanh2.0': {'layer': X, 'test_ece': Y, 'uncalibrated_ece': Z, ...},
                'margin_cvar_linear': {'layer': X, 'test_ece': Y, ...},
                # Single metrics (new)
                'calibration_decisiveness': {'layer': X, 'test_ece': Y, ...},
                'nc1_collapse': {'layer': X, 'test_ece': Y, ...},
                'nc4_separability': {'layer': X, 'test_ece': Y, ...},
                # Filtered/ensemble (new)
                'pareto_nc1_nc4': {'layer': X, 'test_ece': Y, ...},
                'ensemble_margin_nc4': {'layer': X, 'test_ece': Y, ...}
            },
            'metadata': {...},
            'raw_metrics': [...]  # All geometric metrics (CVaR, NC1, NC4, CalibrationDecisiveness) for all layers
        }
    """
    # Determine normalization variants to test
    if normalization_methods is None:
        normalization_methods = [config.normalization_method] if hasattr(config, 'normalization_method') else ['tanh']
    if normalization_scales is None:
        normalization_scales = [config.normalization_scale] if hasattr(config, 'normalization_scale') else [2.0]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Running Alpha Experiment (MULTI-METRIC MODE)")
    logger.info(f"{'='*80}")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Model: {config.model}")
    logger.info(f"Training: {config.training_method}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Alpha: {config.alpha}")
    logger.info(f"Normalization Methods: {normalization_methods}")
    logger.info(f"Normalization Scales: {normalization_scales}")
    logger.info(f"Metrics: CVaR, NC1, NC4, CalibrationDecisiveness")
    logger.info(f"Selection Strategies: CVaR variants, single metrics, pareto, ensemble")
    logger.info(f"Output: {config.output_file}")
    logger.info(f"{'='*80}\n")

    # Determine number of classes based on dataset
    dataset_num_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'svhn': 10,
    }
    num_classes = dataset_num_classes.get(config.dataset)
    if num_classes is None:
        logger.error(f"Unknown dataset: {config.dataset}. Expected one of {list(dataset_num_classes.keys())}")
        return None

    # Construct model path following the standard pattern:
    # checkpoint_base_dir / training_method / dataset / model / seed{N} / {training_method}_{dataset}_{model}_seed{N} / best_model.pth
    dynamic_folder_name = f"{config.training_method}_{config.dataset}_{config.model}_seed{config.seed}"
    model_path = (
        checkpoint_base_dir
        / config.training_method
        / config.dataset
        / config.model
        / f"seed{config.seed}"
        / dynamic_folder_name
        / "best_model.pth"
    )

    # Try multiple possible checkpoint paths in case structure varies
    possible_paths = [
        model_path,
        # Alternative: without dynamic folder name
        checkpoint_base_dir / config.training_method / config.dataset / config.model / f"seed{config.seed}" / "best_model.pth",
        # Alternative: old structure
        checkpoint_base_dir / config.dataset / config.model / config.training_method / f"seed{config.seed}" / "best_model.pth",
    ]

    found_path = None
    for path in possible_paths:
        if path.exists():
            found_path = path
            logger.info(f"Found model checkpoint at: {path}")
            break

    if found_path is None:
        logger.error(f"Failed to find model checkpoint. Tried the following paths:")
        for i, path in enumerate(possible_paths, 1):
            logger.error(f"  {i}. {path}")
        return None

    # Load model and data
    try:
        model = load_trained_model(
            model_path=str(found_path),
            model_name=config.model,
            num_classes=num_classes,
            device=device,
            dataset=config.dataset
        )
        logger.info(f"✓ Model loaded successfully from: {found_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {found_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    try:
        train_loader, val_loader, test_loader, num_classes = get_data_loaders(
            dataset=config.dataset,
            batch_size=batch_size,
        )
        logger.info(f"Loaded data loaders for {config.dataset}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None

    # Convert dataloaders to numpy arrays
    logger.info("Converting data to numpy arrays...")

    def loader_to_numpy(loader):
        X_list, y_list = [], []
        for X_batch, y_batch in loader:
            X_list.append(X_batch.numpy())
            y_list.append(y_batch.numpy())
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    X_train, y_train = loader_to_numpy(train_loader)
    X_val, y_val = loader_to_numpy(val_loader)
    X_test, y_test = loader_to_numpy(test_loader)

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Create model adapter
    model_adapter = PyTorchModelAdapter(model, device=device)

    # Pre-calculate validation probabilities for efficiency
    logger.info("Pre-calculating validation probabilities...")
    val_probs = model_adapter.predict_proba(X_val, batch_size=min(256, batch_size))

    # Determine compression ratio based on dataset
    if compression_ratio is None:
        compression_ratio = 32.0 if config.dataset == "cifar100" else 16.0
        logger.info(f"Auto-selected compression ratio: {compression_ratio}x")

    # =====================================================================
    # STEP 1: Compute all geometric metrics for all layers (ONCE)
    # =====================================================================
    logger.info("="*80)
    logger.info("STEP 1: Computing geometric metrics for all layers")
    logger.info("="*80)
    
    # Create calculators
    raw_cvar_calculator = MarginTailCVaRCalculator(
        alpha=config.alpha,
        normalization_method="tanh",  # Dummy, we'll use compute_raw_cvar
        normalization_scale=2.0
    )
    nc1_calculator = NC1CollapseMetricCalculator()
    nc4_calculator = NC4SeparabilityCalculator()
    
    # Create calibrator to get layer features
    logger.info("Extracting features for all candidate layers...")
    
    raw_pytorch_model = model_adapter.model
    candidate_layers = CrossModalLayerMapper.get_candidate_indices(raw_pytorch_model)
    device_obj = torch.device(device)
    
    # Split validation data (33-67 split for layer selection)
    from sklearn.model_selection import train_test_split
    total_samples = len(X_val)
    cal_train_indices, cal_eval_indices = train_test_split(
        np.arange(total_samples),
        test_size=0.67, train_size=0.33,
        stratify=y_val, random_state=42
    )
    X_cal_train, y_cal_train = X_val[cal_train_indices], y_val[cal_train_indices]
    X_cal_eval, y_cal_eval = X_val[cal_eval_indices], y_val[cal_eval_indices]
    
    # Get probabilities for cal_eval set (for CalibrationDecisiveness)
    cal_eval_probs = val_probs[cal_eval_indices]
    cal_eval_max_probs = np.max(cal_eval_probs, axis=1)
    calibration_decisiveness_global = float(np.mean(np.abs(cal_eval_max_probs - 0.5)))
    logger.info(f"  Global CalibrationDecisiveness (cal_eval set): {calibration_decisiveness_global:.4f}")
    
    # Create dataloaders
    cal_train_dataloader = DataLoader(
        TensorDataset(torch.from_numpy(X_cal_train.astype(np.float32)), torch.from_numpy(y_cal_train)),
        batch_size=batch_size
    )
    cal_eval_dataloader = DataLoader(
        TensorDataset(torch.from_numpy(X_cal_eval.astype(np.float32)), torch.from_numpy(y_cal_eval)),
        batch_size=batch_size
    )
    train_dataloader = DataLoader(
        TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train)),
        batch_size=batch_size
    )
    
    # Extract features and compute all metrics for each layer
    layer_selector = AdHocLayerSelector(
        metrics=[],  # Empty - we'll compute metrics manually
        score_weights={},
        cache_activations=False,
        compression_ratio=compression_ratio,
    )
    
    layer_geometric_scores = {}  # {layer_idx: {'raw_cvar': ..., 'raw_std': ..., 'nc1': ..., 'nc4': ..., 'calibration_decisiveness': ...}}
    layer_features = {}  # Store features for later use
    
    logger.info(f"Computing metrics for {len(candidate_layers)} layers...")
    for layer_idx in candidate_layers:
        logger.info(f"  Processing layer {layer_idx}...")
        
        # Extract features
        features_cal_eval, labels_cal_eval = layer_selector._extract_features(
            raw_pytorch_model, cal_eval_dataloader, layer_idx, device_obj
        )
        
        # Ensure features are torch.Tensor for NC metrics
        if isinstance(features_cal_eval, np.ndarray):
            features_cal_eval_tensor = torch.from_numpy(features_cal_eval).to(device_obj)
        else:
            features_cal_eval_tensor = features_cal_eval.to(device_obj)
        
        if isinstance(labels_cal_eval, np.ndarray):
            labels_cal_eval_tensor = torch.from_numpy(labels_cal_eval).to(device_obj)
        else:
            labels_cal_eval_tensor = labels_cal_eval.to(device_obj)
        
        # Compute raw CVaR
        raw_data = raw_cvar_calculator.compute_raw_cvar(
            features_cal_eval.numpy() if isinstance(features_cal_eval, torch.Tensor) else features_cal_eval,
            labels_cal_eval.numpy() if isinstance(labels_cal_eval, torch.Tensor) else labels_cal_eval
        )
        
        # Compute NC1 (within-class variability)
        nc1_score = nc1_calculator.compute(
            features_cal_eval_tensor,
            labels_cal_eval_tensor
        )
        
        # Compute NC4 (nearest centroid classifier accuracy)
        nc4_score = nc4_calculator.compute(
            features_cal_eval_tensor,
            labels_cal_eval_tensor
        )
        
        # CalibrationDecisiveness is computed from model probabilities (same for all layers)
        # We use the global value computed above
        
        layer_geometric_scores[layer_idx] = {
            'raw_cvar': raw_data['cvar'],
            'raw_std': raw_data['std'],
            'nc1_collapse': nc1_score,
            'nc4_separability': nc4_score,
            'calibration_decisiveness': calibration_decisiveness_global,  # Same for all layers (from model probs)
        }
        
        # Store features for this layer (we'll need them for calibration)
        layer_features[layer_idx] = {
            'cal_eval': (features_cal_eval, labels_cal_eval),
        }
    
    logger.info(f"✓ Computed all metrics for {len(candidate_layers)} layers")
    logger.info(f"  Metrics: CVaR, NC1, NC4, CalibrationDecisiveness")
    
    # =====================================================================
    # STEP 2: Apply all selection strategies and select best layer for each
    # =====================================================================
    logger.info("="*80)
    logger.info("STEP 2: Applying selection strategies and selecting layers")
    logger.info("="*80)
    
    selected_layer_per_config = {}
    # Cache calibration results by layer to avoid recalibrating the same layer
    layer_calibration_cache = {}  # {layer_idx: {'test_ece': float, 'uncalibrated_ece': float}}
    
    # Pre-compute uncalibrated ECE once (same for all variants)
    logger.info("Computing uncalibrated ECE (once for all variants)...")
    uncalibrated_probs = model_adapter.predict_proba(X_test, batch_size=batch_size)
    uncalibrated_ece = compute_ece(uncalibrated_probs, y_test, n_bins=bins)
    logger.info(f"  Uncalibrated ECE: {uncalibrated_ece:.4f}")
    
    # Helper function to calibrate a layer and cache results
    def calibrate_layer(layer_idx: int, strategy_name: str) -> float:
        """Calibrate a layer and return test ECE. Uses cache if available."""
        if layer_idx in layer_calibration_cache:
            logger.info(f"  ⚡ Reusing calibration results for layer {layer_idx} (already computed)")
            cached_result = layer_calibration_cache[layer_idx]
            return cached_result['test_ece']
        
        # Calibrate this layer for the first time
        logger.info(f"  Calibrating with layer {layer_idx} (first time for {strategy_name})...")
        
        # Get features for selected layer (full validation set for final calibration)
        val_dataloader_full = DataLoader(
            TensorDataset(torch.from_numpy(X_val.astype(np.float32)), torch.from_numpy(y_val)),
            batch_size=batch_size
        )
        val_features_final, _ = layer_selector._extract_features(
            raw_pytorch_model, val_dataloader_full, layer_idx, device_obj
        )
        train_features_final, _ = layer_selector._extract_features(
            raw_pytorch_model, train_dataloader, layer_idx, device_obj
        )
        
        X_train_embed = train_features_final.numpy() if isinstance(train_features_final, torch.Tensor) else train_features_final
        X_val_embed = val_features_final.numpy() if isinstance(val_features_final, torch.Tensor) else val_features_final
        
        # Create calibrator in feature mode (not auto_select_layer mode)
        calibrator = GeometricCalibrator(
            model=model_adapter,
            X_train_embed=X_train_embed,
            y_train=y_train,
            library="fast_separation",
            auto_select_layer=False,  # Feature mode - we provide embeddings directly
            device=device,
            compression_ratio=compression_ratio,
        )
        
        # Fit calibrator (need X_val_original for feature mode)
        calibrator.fit(X_val_embed=X_val_embed, X_val_original=X_val, y_val=y_val, fit_batch_size=batch_size, model_probs=val_probs)
        
        # Calibrate test set - extract features first, then calibrate
        test_dataloader = DataLoader(
            TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test)),
            batch_size=batch_size
        )
        test_features, _ = layer_selector._extract_features(
            raw_pytorch_model, test_dataloader, layer_idx, device_obj
        )
        X_test_embed = test_features.numpy() if isinstance(test_features, torch.Tensor) else test_features
        
        # Calibrate using feature mode (X_test_embed) but also need X_test_original for predict_proba
        calibrated_probs = calibrator.calibrate(X_test_embed=X_test_embed, X_test_original=X_test)
        
        # Calculate ECE
        test_ece = compute_ece(calibrated_probs, y_test, n_bins=bins)
        
        # Cache the result for this layer
        layer_calibration_cache[layer_idx] = {
            'test_ece': float(test_ece),
            'uncalibrated_ece': float(uncalibrated_ece),
        }
        
        return float(test_ece)
    
    # Strategy 1: CVaR normalization variants (existing)
    logger.info("")
    logger.info("Strategy: CVaR normalization variants")
    logger.info("-" * 80)
    for norm_method in normalization_methods:
        for norm_scale in normalization_scales:
            # Skip scale for linear method
            if norm_method == "linear":
                config_key = "margin_cvar_linear"
                logger.info(f"Testing: {config_key}")
            else:
                # Format: margin_cvar_tanh2.0 (not margin_cvar_tanh_scale2p0)
                config_key = f"margin_cvar_{norm_method}{norm_scale:.1f}"
                logger.info(f"Testing: {config_key}")
            
            # Apply normalization to raw CVaR values and find best layer
            normalized_scores = {}
            for layer_idx in candidate_layers:
                layer_data = layer_geometric_scores[layer_idx]
                normalized_score = raw_cvar_calculator.normalize_cvar(
                    layer_data['raw_cvar'],
                    layer_data['raw_std'],
                    normalization_method=norm_method,
                    normalization_scale=norm_scale
                )
                normalized_scores[layer_idx] = normalized_score
            
            # Select best layer (highest normalized score)
            best_layer = max(normalized_scores.items(), key=lambda x: x[1])[0]
            best_score = normalized_scores[best_layer]
            
            logger.info(f"  ✓ Best layer: {best_layer} (normalized CVaR score: {best_score:.4f})")
            
            # Calibrate and get ECE
            test_ece = calibrate_layer(best_layer, config_key)
            
            # Store results
            layer_data = layer_geometric_scores[best_layer]
            selected_layer_per_config[config_key] = {
                'layer': int(best_layer),
                'test_ece': float(test_ece),
                'uncalibrated_ece': float(uncalibrated_ece),
                'ece_improvement': float(uncalibrated_ece - test_ece),
                'normalized_cvar_score': float(best_score),
                'raw_cvar': float(layer_data['raw_cvar']),
                'raw_std': float(layer_data['raw_std']),
            }
            
            logger.info(f"  ✓ {config_key}: Layer {best_layer}, ECE {test_ece:.4f} (improvement: {uncalibrated_ece - test_ece:.4f})")
    
    # Strategy 2: CalibrationDecisiveness (select layer with highest decisiveness)
    logger.info("")
    logger.info("Strategy: CalibrationDecisiveness")
    logger.info("-" * 80)
    config_key = "calibration_decisiveness"
    logger.info(f"Testing: {config_key}")
    
    # CalibrationDecisiveness is the same for all layers (computed from model probs)
    # So we select a random layer or use the first one (doesn't matter)
    # Actually, since decisiveness is the same, we'll use the layer with best CVaR as tiebreaker
    decisiveness_scores = {idx: layer_geometric_scores[idx]['calibration_decisiveness'] for idx in candidate_layers}
    # All should be the same, but use max anyway
    best_layer = max(decisiveness_scores.items(), key=lambda x: x[1])[0]
    best_score = decisiveness_scores[best_layer]
    
    logger.info(f"  ✓ Selected layer: {best_layer} (decisiveness: {best_score:.4f}, same for all layers)")
    
    # Use layer with best normalized CVaR (tanh scale 2.0) as representative
    # Since decisiveness is the same, we pick the best CVaR layer
    normalized_scores = {}
    for layer_idx in candidate_layers:
        layer_data = layer_geometric_scores[layer_idx]
        normalized_score = raw_cvar_calculator.normalize_cvar(
            layer_data['raw_cvar'],
            layer_data['raw_std'],
            normalization_method="tanh",
            normalization_scale=2.0
        )
        normalized_scores[layer_idx] = normalized_score
    best_layer = max(normalized_scores.items(), key=lambda x: x[1])[0]
    
    test_ece = calibrate_layer(best_layer, config_key)
    
    layer_data = layer_geometric_scores[best_layer]
    selected_layer_per_config[config_key] = {
        'layer': int(best_layer),
        'test_ece': float(test_ece),
        'uncalibrated_ece': float(uncalibrated_ece),
        'ece_improvement': float(uncalibrated_ece - test_ece),
        'calibration_decisiveness': float(layer_data['calibration_decisiveness']),
    }
    
    logger.info(f"  ✓ {config_key}: Layer {best_layer}, ECE {test_ece:.4f} (improvement: {uncalibrated_ece - test_ece:.4f})")
    
    # Strategy 3: NC1 Collapse (select layer with highest NC1, filter NC1 > 0.2)
    logger.info("")
    logger.info("Strategy: NC1 Collapse")
    logger.info("-" * 80)
    config_key = "nc1_collapse"
    logger.info(f"Testing: {config_key}")
    
    nc1_scores = {idx: layer_geometric_scores[idx]['nc1_collapse'] for idx in candidate_layers}
    # Filter for NC1 > 0.2 threshold
    filtered_layers = {idx: score for idx, score in nc1_scores.items() if score > 0.2}
    
    if filtered_layers:
        best_layer = max(filtered_layers.items(), key=lambda x: x[1])[0]
        best_score = filtered_layers[best_layer]
        logger.info(f"  ✓ Best layer: {best_layer} (NC1: {best_score:.4f}, {len(filtered_layers)}/{len(candidate_layers)} layers pass NC1 > 0.2 threshold)")
    else:
        # If no layers pass threshold, use highest NC1 anyway
        best_layer = max(nc1_scores.items(), key=lambda x: x[1])[0]
        best_score = nc1_scores[best_layer]
        logger.info(f"  ⚠ No layers pass NC1 > 0.2 threshold, using best: {best_layer} (NC1: {best_score:.4f})")
    
    test_ece = calibrate_layer(best_layer, config_key)
    
    layer_data = layer_geometric_scores[best_layer]
    selected_layer_per_config[config_key] = {
        'layer': int(best_layer),
        'test_ece': float(test_ece),
        'uncalibrated_ece': float(uncalibrated_ece),
        'ece_improvement': float(uncalibrated_ece - test_ece),
        'nc1_collapse': float(layer_data['nc1_collapse']),
    }
    
    logger.info(f"  ✓ {config_key}: Layer {best_layer}, ECE {test_ece:.4f} (improvement: {uncalibrated_ece - test_ece:.4f})")
    
    # Strategy 4: NC4 Separability (select layer with highest NC4)
    logger.info("")
    logger.info("Strategy: NC4 Separability")
    logger.info("-" * 80)
    config_key = "nc4_separability"
    logger.info(f"Testing: {config_key}")
    
    nc4_scores = {idx: layer_geometric_scores[idx]['nc4_separability'] for idx in candidate_layers}
    best_layer = max(nc4_scores.items(), key=lambda x: x[1])[0]
    best_score = nc4_scores[best_layer]
    
    logger.info(f"  ✓ Best layer: {best_layer} (NC4: {best_score:.4f})")
    
    test_ece = calibrate_layer(best_layer, config_key)
    
    layer_data = layer_geometric_scores[best_layer]
    selected_layer_per_config[config_key] = {
        'layer': int(best_layer),
        'test_ece': float(test_ece),
        'uncalibrated_ece': float(uncalibrated_ece),
        'ece_improvement': float(uncalibrated_ece - test_ece),
        'nc4_separability': float(layer_data['nc4_separability']),
    }
    
    logger.info(f"  ✓ {config_key}: Layer {best_layer}, ECE {test_ece:.4f} (improvement: {uncalibrated_ece - test_ece:.4f})")
    
    # Strategy 5: Pareto NC1-NC4 (filter NC1 > 0.2, then max NC4)
    logger.info("")
    logger.info("Strategy: Pareto NC1-NC4 (filter NC1 > 0.2, then max NC4)")
    logger.info("-" * 80)
    config_key = "pareto_nc1_nc4"
    logger.info(f"Testing: {config_key}")
    
    # Filter for NC1 > 0.2
    filtered_layers = {idx: layer_geometric_scores[idx] for idx in candidate_layers 
                      if layer_geometric_scores[idx]['nc1_collapse'] > 0.2}
    
    if filtered_layers:
        # Among filtered layers, select highest NC4
        nc4_scores_filtered = {idx: layer_geometric_scores[idx]['nc4_separability'] for idx in filtered_layers.keys()}
        best_layer = max(nc4_scores_filtered.items(), key=lambda x: x[1])[0]
        best_nc1 = layer_geometric_scores[best_layer]['nc1_collapse']
        best_nc4 = layer_geometric_scores[best_layer]['nc4_separability']
        logger.info(f"  ✓ Best layer: {best_layer} (NC1: {best_nc1:.4f}, NC4: {best_nc4:.4f}, {len(filtered_layers)}/{len(candidate_layers)} layers pass NC1 > 0.2)")
    else:
        # If no layers pass threshold, use highest NC4 anyway
        nc4_scores = {idx: layer_geometric_scores[idx]['nc4_separability'] for idx in candidate_layers}
        best_layer = max(nc4_scores.items(), key=lambda x: x[1])[0]
        best_nc1 = layer_geometric_scores[best_layer]['nc1_collapse']
        best_nc4 = layer_geometric_scores[best_layer]['nc4_separability']
        logger.info(f"  ⚠ No layers pass NC1 > 0.2 threshold, using best NC4: {best_layer} (NC1: {best_nc1:.4f}, NC4: {best_nc4:.4f})")
    
    test_ece = calibrate_layer(best_layer, config_key)
    
    layer_data = layer_geometric_scores[best_layer]
    selected_layer_per_config[config_key] = {
        'layer': int(best_layer),
        'test_ece': float(test_ece),
        'uncalibrated_ece': float(uncalibrated_ece),
        'ece_improvement': float(uncalibrated_ece - test_ece),
        'nc1_collapse': float(layer_data['nc1_collapse']),
        'nc4_separability': float(layer_data['nc4_separability']),
    }
    
    logger.info(f"  ✓ {config_key}: Layer {best_layer}, ECE {test_ece:.4f} (improvement: {uncalibrated_ece - test_ece:.4f})")
    
    # Strategy 6: Ensemble Margin-NC4 (0.6 * normalized_margin_cvar + 0.4 * NC4)
    logger.info("")
    logger.info("Strategy: Ensemble Margin-NC4 (0.6*margin_cvar + 0.4*NC4)")
    logger.info("-" * 80)
    config_key = "ensemble_margin_nc4"
    logger.info(f"Testing: {config_key}")
    
    # Use tanh scale 2.0 for margin CVaR normalization
    ensemble_scores = {}
    for layer_idx in candidate_layers:
        layer_data = layer_geometric_scores[layer_idx]
        # Normalize CVaR
        normalized_cvar = raw_cvar_calculator.normalize_cvar(
            layer_data['raw_cvar'],
            layer_data['raw_std'],
            normalization_method="tanh",
            normalization_scale=2.0
        )
        # Get NC4
        nc4_score = layer_data['nc4_separability']
        # Ensemble: 0.6 * margin_cvar + 0.4 * NC4
        ensemble_score = 0.6 * normalized_cvar + 0.4 * nc4_score
        ensemble_scores[layer_idx] = ensemble_score
    
    best_layer = max(ensemble_scores.items(), key=lambda x: x[1])[0]
    best_score = ensemble_scores[best_layer]
    layer_data = layer_geometric_scores[best_layer]
    normalized_cvar_best = raw_cvar_calculator.normalize_cvar(
        layer_data['raw_cvar'],
        layer_data['raw_std'],
        normalization_method="tanh",
        normalization_scale=2.0
    )
    nc4_best = layer_data['nc4_separability']
    
    logger.info(f"  ✓ Best layer: {best_layer} (ensemble: {best_score:.4f} = 0.6*{normalized_cvar_best:.4f} + 0.4*{nc4_best:.4f})")
    
    test_ece = calibrate_layer(best_layer, config_key)
    
    selected_layer_per_config[config_key] = {
        'layer': int(best_layer),
        'test_ece': float(test_ece),
        'uncalibrated_ece': float(uncalibrated_ece),
        'ece_improvement': float(uncalibrated_ece - test_ece),
        'ensemble_score': float(best_score),
        'normalized_cvar_score': float(normalized_cvar_best),
        'nc4_separability': float(nc4_best),
    }
    
    logger.info(f"  ✓ {config_key}: Layer {best_layer}, ECE {test_ece:.4f} (improvement: {uncalibrated_ece - test_ece:.4f})")
    
    # Log optimization summary
    total_variants = len(selected_layer_per_config)
    unique_layers_calibrated = len(layer_calibration_cache)
    calibrations_saved = total_variants - unique_layers_calibrated
    logger.info("")
    logger.info("="*80)
    logger.info("Calibration Optimization Summary")
    logger.info("="*80)
    logger.info(f"  Total selection strategies tested: {total_variants}")
    logger.info(f"  Unique layers calibrated: {unique_layers_calibrated}")
    logger.info(f"  Calibrations saved (reused): {calibrations_saved}")
    if calibrations_saved > 0:
        logger.info(f"  ⚡ Efficiency gain: {calibrations_saved}/{total_variants} = {100*calibrations_saved/total_variants:.1f}% reduction")
    logger.info("="*80)
    
    # =====================================================================
    # STEP 3: Compile final results
    # =====================================================================
    logger.info("="*80)
    logger.info("STEP 3: Compiling results")
    logger.info("="*80)
    
    # Serialize raw metrics (all geometric metrics for all layers)
    raw_metrics = []
    for layer_idx in candidate_layers:
        layer_data = layer_geometric_scores[layer_idx]
        raw_metrics.append({
            'layer_idx': int(layer_idx),
            'raw_cvar': float(layer_data['raw_cvar']),
            'raw_std': float(layer_data['raw_std']),
            'nc1_collapse': float(layer_data['nc1_collapse']),
            'nc4_separability': float(layer_data['nc4_separability']),
            'calibration_decisiveness': float(layer_data['calibration_decisiveness']),
        })
    
    results = {
        "alpha": config.alpha,
        "selected_layer_per_config": selected_layer_per_config,
        "metadata": {
            "dataset": config.dataset,
            "model": config.model,
            "training_method": config.training_method,
            "seed": config.seed,
            "num_classes": int(num_classes),
            "compression_ratio": float(compression_ratio),
            "normalization_methods_tested": normalization_methods,
            "normalization_scales_tested": normalization_scales,
        },
        "raw_metrics": raw_metrics,
    }
    
    # Save results
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    with config.output_file.open('w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ Results saved to: {config.output_file}")
    logger.info(f"  Tested {len(selected_layer_per_config)} normalization variants")
    
    return results


def generate_experiment_configs(
    datasets: List[str],
    models: List[str],
    training_methods: List[str],
    seeds: List[int],
    alphas: List[float],
    output_base_dir: Path,
    normalization_methods: Optional[List[str]] = None,
    normalization_scales: Optional[List[float]] = None,
) -> List[AlphaExperimentConfig]:
    """
    Generate all experiment configurations.
    Each config represents one (dataset, model, training_method, seed, alpha) combo.
    All normalization variants will be tested within each experiment.
    """
    configs = []
    
    # Default normalization methods if not provided
    if normalization_methods is None:
        normalization_methods = ["tanh"]
    
    # Default normalization scales if not provided
    if normalization_scales is None:
        normalization_scales = [2.0]

    for dataset in datasets:
        for model in models:
            for training_method in training_methods:
                for seed in seeds:
                    for alpha in alphas:
                        # Format alpha for filename (e.g., 0.05 -> 005)
                        alpha_str = f"{int(alpha * 100):03d}"
                        
                        # Simple filename: seed{seed}_alpha{alpha_str}.json
                        # No method/scale in filename since all variants are in one file
                        filename = f"seed{seed}_alpha{alpha_str}.json"

                        output_file = (
                            output_base_dir
                            / dataset
                            / model
                            / training_method
                            / filename
                        )

                        # Store normalization methods/scales in config for reference
                        # (default values, actual lists passed to run_alpha_experiment)
                        config = AlphaExperimentConfig(
                            dataset=dataset,
                            model=model,
                            training_method=training_method,
                            seed=seed,
                            alpha=alpha,
                            normalization_method=normalization_methods[0],  # Default/fallback
                            normalization_scale=normalization_scales[0],  # Default/fallback
                            output_file=output_file,
                        )
                        # Store full lists as attributes (will be passed to run_alpha_experiment)
                        config.normalization_methods = normalization_methods
                        config.normalization_scales = normalization_scales
                        configs.append(config)

    return configs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Alpha ablation study for MarginTailCVaRCalculator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data and model configuration
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=["cifar10", "cifar100", "svhn"],
        help="Datasets to test"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=["resnet18", "resnet50", "densenet121"],
        help="Models to test"
    )
    parser.add_argument(
        "--training-methods",
        type=str,
        nargs='+',
        default=["augmix", "baseline_brier", "baseline_cross_entropy"],
        help="Training methods to test"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs='+',
        default=[1, 2, 3],
        help="Seeds to test"
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs='+',
        default=ALPHA_VALUES,
        help="Alpha values to test"
    )
    parser.add_argument(
        "--normalization-methods",
        type=str,
        nargs='+',
        default=None,
        choices=["linear", "tanh"],
        help="Normalization methods to test: 'linear' (old method) or 'tanh' (new method). Default: ['tanh']"
    )
    parser.add_argument(
        "--normalization-scales",
        type=float,
        nargs='+',
        default=None,
        help="Normalization scale factors for tanh normalization (default: [2.0]). Only used with 'tanh' method. Larger values = less sensitive to CVaR changes."
    )

    # Paths
    parser.add_argument(
        "--checkpoint-base-dir",
        type=Path,
        default=Path("aaai_full_experiments/results/baseline"),
        help="Base directory containing trained model checkpoints"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/alpha_ablation_margin_cvar"),
        help="Output directory for results"
    )

    # Compute configuration
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for data processing"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=15,
        help="Number of bins for ECE calculation"
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=None,
        help="Compression ratio (None=auto: 32x for CIFAR-100, 16x for others)"
    )

    # Experiment control
    parser.add_argument(
        "--skip-existing",
        action='store_true',
        help="Skip experiments with existing output files"
    )
    parser.add_argument(
        "--dry-run",
        action='store_true',
        help="Show what would be done without running"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    # Optional: single experiment mode
    parser.add_argument(
        "--single-experiment",
        action='store_true',
        help="Run a single experiment (first config only) for testing"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Generate all experiment configurations
    configs = generate_experiment_configs(
        datasets=args.datasets,
        models=args.models,
        training_methods=args.training_methods,
        seeds=args.seeds,
        alphas=args.alphas,
        output_base_dir=args.output_dir,
        normalization_methods=args.normalization_methods,
        normalization_scales=args.normalization_scales,
    )

    logger.info(f"Generated {len(configs)} experiment configurations")
    logger.info(f"  Datasets: {args.datasets}")
    logger.info(f"  Models: {args.models}")
    logger.info(f"  Training methods: {args.training_methods}")
    logger.info(f"  Seeds: {args.seeds}")
    logger.info(f"  Alphas: {args.alphas}")
    logger.info(f"  Normalization methods: {args.normalization_methods if args.normalization_methods else ['tanh']}")
    logger.info(f"  Normalization scales: {args.normalization_scales if args.normalization_scales else [2.0]}")

    # Filter out existing experiments if requested
    if args.skip_existing:
        original_count = len(configs)
        configs = [c for c in configs if not c.output_file.exists()]
        skipped = original_count - len(configs)
        logger.info(f"Skipping {skipped} existing experiments")

    if args.single_experiment and configs:
        logger.info("Single experiment mode: running first configuration only")
        configs = configs[:1]

    if not configs:
        logger.info("No experiments to run")
        return 0

    if args.dry_run:
        logger.info("DRY RUN - showing first 5 experiments:")
        for config in configs[:5]:
            method_info = f", method={config.normalization_method}" if config.normalization_method != "tanh" else ""
            scale_info = f", scale={config.normalization_scale}" if (config.normalization_method == "tanh" and config.normalization_scale != 2.0) else ""
            logger.info(f"  {config.dataset}/{config.model}/{config.training_method}/seed{config.seed}/alpha={config.alpha}{method_info}{scale_info}")
        logger.info(f"... and {len(configs) - 5} more" if len(configs) > 5 else "")
        return 0

    # Run experiments
    logger.info(f"\nStarting {len(configs)} experiments...")

    success_count = 0
    failure_count = 0
    failed_configs = []

    for i, config in enumerate(tqdm(configs, desc="Running experiments")):
        logger.info(f"\n{'='*80}")
        logger.info(f"Experiment {i+1}/{len(configs)}")
        logger.info(f"{'='*80}")

        try:
            result = run_alpha_experiment(
                config=config,
                checkpoint_base_dir=args.checkpoint_base_dir,
                device=args.device,
                batch_size=args.batch_size,
                bins=args.bins,
                compression_ratio=args.compression_ratio,
            )

            if result is not None:
                success_count += 1
                logger.info(f"✓ Experiment {i+1} completed successfully")
            else:
                failure_count += 1
                failed_configs.append(config)
                logger.error(f"✗ Experiment {i+1} failed")

        except Exception as e:
            failure_count += 1
            failed_configs.append(config)
            logger.error(f"✗ Experiment {i+1} failed with exception: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total experiments: {len(configs)}")
    logger.info(f"✓ Successful: {success_count}")
    logger.info(f"✗ Failed: {failure_count}")

    if failed_configs:
        logger.info("\nFailed configurations:")
        for config in failed_configs:
            method_info = f", method={config.normalization_method}" if config.normalization_method != "tanh" else ""
            scale_info = f", scale={config.normalization_scale}" if (config.normalization_method == "tanh" and config.normalization_scale != 2.0) else ""
            logger.info(f"  - {config.dataset}/{config.model}/{config.training_method}/seed{config.seed}/alpha={config.alpha}{method_info}{scale_info}")

    logger.info(f"\nResults saved to: {args.output_dir}")

    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
