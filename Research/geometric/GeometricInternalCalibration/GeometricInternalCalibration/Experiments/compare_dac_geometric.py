import torch
import numpy as np
import os
import logging
import argparse
import json
import time
from typing import Dict, Any, List, Union, Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Import DAC implementation
from Calibrators.density_aware_calibration import (
    DensityAwareCalibrator,
    extract_dac_features,
    extract_dac_features_memory_efficient,
    get_dac_target_layers,
    get_dac_k_value,
    find_layer_module
)
from Calibrators.density_aware_calibration_pytorch import create_dac_pytorch
# Import TULIP layer selection
from Calibrators.tulip import select_tulip_layers

# Import your existing utilities
from Experiments.run_post_hoc_calibration import (
    PyTorchModelAdapter,
    get_data_loaders,
    load_trained_model,
    construct_model_path  # This handles the path correctly
)
from Experiments.layer_selection import (
    run_standard_baselines,
    CIFAR_C_CORRUPTIONS,
    load_cifar_c_loader,
    compute_mce_for_method,
    get_all_data_as_numpy,
    UncertaintyMetrics
)
from Experiments.multi_layer_ensemble import discover_model_layers
from Calibrators.geometric_calibrator_new import GeometricCalibrator
from Metrics.metrics import expected_calibration_error
from Calibrators.metrics import (
    MarginTailCVaRCalculator,
    MultiScaleClassHomogeneityCalculator,
    LocalIntrinsicDimensionalityCalculator
)
from Calibrators.nc_metrics import (
    NC1CollapseMetricCalculator,
    NC4SeparabilityCalculator
)
from utils.stability_space import StabilitySpace
from utils.coordinate_extraction import (
    discover_coordinate_space,
    plan_coordinate_extraction,
    extract_coordinate_features,
    coerce_to_tensor,
)
from utils.layer_utils import filter_feature_layers, is_feature_layer
from sklearn.preprocessing import normalize

# ============================================================================
# DAC CALIBRATOR FACTORY
# ============================================================================

def make_dac_calibrator(k: int, device: torch.device, backend: str):
    """
    Factory function to create a DAC calibrator with the specified kNN backend.
    
    This allows fair timing comparisons between FAISS and PyTorch kNN implementations,
    matching the PyTorch-based kNN used in RGCL/RGCC methods.
    
    Args:
        k: Number of nearest neighbors
        device: PyTorch device (for determining GPU availability)
        backend: kNN backend to use ("faiss" or "torch")
        
    Returns:
        DAC calibrator instance (DensityAwareCalibrator or DensityAwareCalibratorPyTorch)
        
    Raises:
        ValueError: If backend is not "faiss" or "torch"
    """
    if backend == "faiss":
        return DensityAwareCalibrator(k=k, use_gpu=(device.type == "cuda"))
    elif backend == "torch":
        return create_dac_pytorch(k=k, use_gpu=(device.type == "cuda"))
    else:
        raise ValueError(f"Invalid DAC kNN backend: {backend}. Must be 'faiss' or 'torch'")


# ============================================================================
# OOD EVALUATION UTILITIES
# ============================================================================

# ============================================================================
# CIFAR-C CORRUPTION EVALUATION UTILITIES
# ============================================================================

def make_uncalibrated_predict_fn(model_adapter, batch_size):
    """Create predict_fn for uncalibrated baseline."""
    def predict_fn(Xc):
        return model_adapter.predict_proba(Xc, batch_size=batch_size)
    return predict_fn


def make_temperature_scaling_predict_fn(model_adapter, ts_calibrator, batch_size):
    """Create predict_fn for temperature scaling."""
    def predict_fn(Xc):
        logits = model_adapter.predict_logits(Xc, batch_size=batch_size)
        return ts_calibrator.calibrate(logits)
    return predict_fn


def make_geometric_predict_fn(
    model, model_adapter, geo_calibrator, layer_names, 
    batch_size, device, pooling_mode='max', target_dimension=None, seed=42
):
    """Create predict_fn for geometric calibration methods."""
    def predict_fn(Xc):
        # Extract features from corrupted images
        # Convert Xc to DataLoader
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xc).float()),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
        
        # Extract features
        features_list = extract_features_from_multiple_layers(
            model, test_loader, layer_names, device,
            compression_ratio=None,
            target_dimension=target_dimension,
            pooling_mode=pooling_mode,
            spp_only=False,
            normalize_spp_output=True
        )
        
        # Combine features if multiple layers
        if len(features_list) > 1:
            test_features = combine_features(features_list)
        else:
            test_features = features_list[0]
        
        # Convert to numpy
        test_features_np = test_features.cpu().numpy()
        
        # Get calibrated probabilities
        calibrated_probs = geo_calibrator.calibrate(
            X_test_embed=test_features_np,
            X_test_original=Xc
        )
        
        return calibrated_probs
    return predict_fn


def make_sgc_predict_fn(
    model, model_adapter, geo_calibrator, selected_layers,
    batch_size, device, pooling_mode='max', target_dim=256, seed=42
):
    """Create predict_fn for SGC (Semantic Geometric Calibration) methods."""
    def predict_fn(Xc):
        # Extract and aggregate SGC features
        # Convert Xc to DataLoader
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xc).float()),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
        
        # Create dummy train/val loaders (not used for feature extraction in this case)
        dummy_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xc[:1]).float()),
            batch_size=1,
            shuffle=False
        )
        
        # Extract SGC features
        _, _, test_features, _ = extract_and_aggregate_sgc_features(
            model, selected_layers,
            dummy_loader, dummy_loader, test_loader,
            device, target_dim=target_dim, seed=seed, pooling_mode=pooling_mode
        )
        
        # Get calibrated probabilities
        calibrated_probs = geo_calibrator.calibrate(
            X_test_embed=test_features,
            X_test_original=Xc
        )
        
        return calibrated_probs
    return predict_fn


def make_coordinate_predict_fn(
    model, model_adapter, geo_calibrator, layer_map, sampling_plan,
    batch_size, device, pooling_mode='max', seed=42
):
    """Create predict_fn for coordinate-based calibration."""
    def predict_fn(Xc):
        # Extract coordinate features
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xc).float()),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
        
        # Extract coordinate features
        test_coords = extract_coordinate_features(
            model, test_loader, layer_map, sampling_plan,
            device=device, pooling_mode=pooling_mode, seed=seed
        )
        
        # Get calibrated probabilities
        calibrated_probs = geo_calibrator.calibrate(
            X_test_embed=test_coords,
            X_test_original=Xc
        )
        
        return calibrated_probs
    return predict_fn


def evaluate_corruption_robustness(
    results: Dict[str, Any],
    model: torch.nn.Module,
    model_adapter: PyTorchModelAdapter,
    dataset_name: str,
    test_transform,
    batch_size: int,
    device: torch.device,
    train_raw: np.ndarray = None,
    train_labels: np.ndarray = None,
    val_raw: np.ndarray = None,
    val_labels: np.ndarray = None,
    cifar_c_dir: str = None,
    pooling_mode: str = 'max',
    target_dimension: int = 256,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluate all computed calibration methods on CIFAR-C corruptions.
    Uses FAISS backend for fast SGC evaluation.
    
    Returns:
        Dict with mCE scores for each method: {'method_name': {'mce': float, ...}}
    """
    logger.info("\n" + "="*80)
    logger.info("CIFAR-C CORRUPTION EVALUATION (mCE) - FAISS Backend")
    logger.info("="*80)
    
    corruption_results = {}
    
    # Helper to get labels
    def labels_getter():
        return np.zeros(10000, dtype=np.int64)
    
    # 1. Uncalibrated baseline
    if 'standard_baselines' in results and 'uncalibrated' in results['standard_baselines']:
        logger.info("\nEvaluating: Uncalibrated")
        try:
            predict_fn = make_uncalibrated_predict_fn(model_adapter, batch_size)
            mce = compute_mce_for_method(
                'uncalibrated', predict_fn, labels_getter,
                dataset_name, test_transform, batch_size, cifar_c_dir
            )
            corruption_results['uncalibrated'] = {'mce': float(mce)}
        except Exception as e:
            logger.error(f"Failed to compute mCE for uncalibrated: {e}")
            corruption_results['uncalibrated'] = {'mce': None, 'error': str(e)}
    
    # 2. Temperature Scaling
    if 'standard_baselines' in results and 'temperature_scaling' in results['standard_baselines']:
        logger.info("\nEvaluating: Temperature Scaling")
        try:
            ts_data = results['standard_baselines']['temperature_scaling']
            from Calibrators.temperature_scaling import TemperatureScaling
            ts_cal = TemperatureScaling()
            if 'temperature' in ts_data:
                ts_cal.temperature = float(ts_data['temperature'])
                predict_fn = make_temperature_scaling_predict_fn(model_adapter, ts_cal, batch_size)
                mce = compute_mce_for_method(
                    'temperature_scaling', predict_fn, labels_getter,
                    dataset_name, test_transform, batch_size, cifar_c_dir
                )
                corruption_results['temperature_scaling'] = {'mce': float(mce)}
            else:
                corruption_results['temperature_scaling'] = {'mce': None, 'error': 'temperature not found'}
        except Exception as e:
            logger.error(f"Failed to compute mCE for temperature_scaling: {e}")
            corruption_results['temperature_scaling'] = {'mce': None, 'error': str(e)}
    
    # 3. SGC methods using FAISS backend for speed
    # Helper function to evaluate SGC-style methods with FAISS
    def evaluate_sgc_method_faiss(method_key: str, method_data: dict, display_name: str = None):
        """Evaluate a single SGC-style method using FAISS backend."""
        if display_name is None:
            display_name = method_key
            
        if not isinstance(method_data, dict):
            return None
            
        if method_data.get('skipped') or method_data.get('error'):
            logger.info(f"Skipping {display_name}: marked as skipped or has error")
            return None
            
        if 'ece' not in method_data:
            logger.info(f"Skipping {display_name}: no ece in results")
            return None
            
        logger.info(f"\nEvaluating: {display_name}")
        
        try:
            if train_raw is None or train_labels is None or val_raw is None or val_labels is None:
                logger.warning(f"Cannot reconstruct {display_name} calibrator: missing train/val data")
                return {'mce': None, 'error': 'missing train/val data'}
            
            # Extract layer information
            selected_layers = method_data.get('selected_layers') or method_data.get('layer_names') or []
            if not selected_layers:
                logger.warning(f"No selected_layers/layer_names found for {display_name}")
                return {'mce': None, 'error': 'no selected_layers'}
            
            # Import FAISS calibrator (fallback to GeometricCalibrator if not available)
            try:
                from Experiments.compare_sgc_backends import SGCCalibratorFast
                use_faiss = True
            except ImportError:
                logger.warning(f"FAISS backend not available for {display_name}, falling back to GeometricCalibrator")
                use_faiss = False
            
            # Create dataloaders
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_raw).float()),
                batch_size=batch_size, shuffle=False
            )
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_raw).float()),
                batch_size=batch_size, shuffle=False
            )
            
            # Extract and aggregate features
            train_features, val_features, _, _ = extract_and_aggregate_sgc_features(
                model, selected_layers,
                train_loader, val_loader, val_loader,  # dummy test loader
                device, target_dim=target_dimension, seed=seed, pooling_mode=pooling_mode
            )
            
            if use_faiss:
                # Get validation predictions
                val_probs = model_adapter.predict_proba(val_raw, batch_size=batch_size)
                val_predictions = np.argmax(val_probs, axis=1)
                
                # Fit FAISS calibrator
                sgc_cal = SGCCalibratorFast(use_gpu=(device.type == 'cuda'))
                sgc_cal.fit(
                    X_train_features=train_features,
                    y_train=train_labels,
                    X_val_features=val_features,
                    y_val=val_labels,
                    val_predictions=val_predictions
                )
                
                # Create predict function for corrupted data
                def predict_fn(Xc):
                    # Extract features from corrupted images
                    test_loader_c = DataLoader(
                        TensorDataset(torch.from_numpy(Xc).float()),
                        batch_size=batch_size, shuffle=False
                    )
                    _, _, test_features, _ = extract_and_aggregate_sgc_features(
                        model, selected_layers,
                        train_loader, val_loader, test_loader_c,
                        device, target_dim=target_dimension, seed=seed, pooling_mode=pooling_mode
                    )
                    
                    # Get predictions
                    test_probs = model_adapter.predict_proba(Xc, batch_size=batch_size)
                    test_predictions = np.argmax(test_probs, axis=1)
                    
                    # Calibrate
                    calibrated_probs = sgc_cal.calibrate(
                        X_test_features=test_features,
                        predictions=test_predictions,
                        logits=test_probs
                    )
                    return calibrated_probs
            else:
                # Fallback to GeometricCalibrator
                geo_cal = GeometricCalibrator(
                    model=model_adapter,
                    X_train_embed=train_features,
                    y_train=train_labels,
                    library='fast_separation',
                    auto_select_layer=False,
                    device=str(device)
                )
                geo_cal.fit(
                    X_val_embed=val_features,
                    y_val=val_labels,
                    X_val_original=val_raw
                )
                
                # Create predict function using original method
                predict_fn = make_sgc_predict_fn(
                    model, model_adapter, geo_cal, selected_layers,
                    batch_size, device, pooling_mode, target_dimension, seed
                )
            
            mce = compute_mce_for_method(
                method_key, predict_fn, labels_getter,
                dataset_name, test_transform, batch_size, cifar_c_dir
            )
            return {'mce': float(mce)}
            
        except Exception as e:
            logger.error(f"Failed to compute mCE for {display_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'mce': None, 'error': str(e)}
    
    # Evaluate root-level SGC methods
    root_level_methods = [
        'global_random_separation',
        'global_random_trust_score',
        'global_random',  # Legacy name
        'sgc_random',  # Legacy name
        'sgc_faiss',
        'coordinate_sampling_separation',
        'coordinate_sampling_trust_score',
        'coordinate_sampling',  # Legacy name
        'geometric_concatenated',  # Legacy name
        'dac_with_coordinate_features',  # Legacy name
        'last_layer_only_baseline',
    ]
    for method_key in root_level_methods:
        if method_key in results:
            result = evaluate_sgc_method_faiss(method_key, results[method_key])
            if result is not None:
                corruption_results[method_key] = result
    
    # Evaluate ablation methods
    if 'ablation' in results and isinstance(results['ablation'], dict):
        ablation_methods = [
            'dac_with_sgc_aggregated_features',  # Legacy name
            'dac_with_random_layer_selection',
            'sgc_with_dac_preprocessing',  # Legacy name
            'sgc_with_dac_preprocessing_separation',
            'sgc_with_dac_preprocessing_trust_score',
            'sgc_with_dac_layers',  # Legacy name
        ]
        for method_key in ablation_methods:
            if method_key in results['ablation']:
                display_name = f"ablation.{method_key}"
                result = evaluate_sgc_method_faiss(method_key, results['ablation'][method_key], display_name)
                if result is not None:
                    corruption_results[display_name] = result
    
    logger.info("\n" + "="*80)
    logger.info("CORRUPTION EVALUATION COMPLETE")
    logger.info("="*80)
    for method, data in corruption_results.items():
        if 'mce' in data and data['mce'] is not None:
            logger.info(f"  {method}: mCE = {data['mce']:.6f}")
        else:
            logger.warning(f"  {method}: mCE computation failed")
    
    return corruption_results

def compute_ood_metrics(
    calibrated_probs_id: np.ndarray,
    calibrated_probs_ood: np.ndarray,
    id_labels: np.ndarray,
    ood_labels: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute OOD detection and calibration metrics.
    
    Args:
        calibrated_probs_id: Calibrated probabilities on in-distribution test set
        calibrated_probs_ood: Calibrated probabilities on OOD set (using same calibrator)
        id_labels: True labels for ID test set
        ood_labels: True labels for OOD set (optional, for ECE calculation)
    
    Returns:
        Dict with:
        - ood_auroc: AUROC for OOD detection (higher = better detection)
        - ood_fpr95: FPR at 95% TPR (lower = better)
        - ood_ece: ECE on OOD data (measures calibration transfer)
        - ood_accuracy: Accuracy on OOD (expect ~random for true OOD)
        - id_confidence_mean: Mean confidence on ID data
        - ood_confidence_mean: Mean confidence on OOD data
        - confidence_gap: ID - OOD confidence (positive = good separation)
    """
    from Calibrators.calibration_utils import compute_ood_auroc, compute_fpr_at_tpr
    
    # OOD detection scores (1 - max_prob, so higher = more uncertain)
    id_scores = 1.0 - np.max(calibrated_probs_id, axis=1)
    ood_scores = 1.0 - np.max(calibrated_probs_ood, axis=1)
    
    # Detection metrics
    ood_auroc = compute_ood_auroc(id_scores, ood_scores)
    ood_fpr95 = compute_fpr_at_tpr(id_scores, ood_scores)
    
    # Confidence statistics
    id_confidence_mean = float(np.max(calibrated_probs_id, axis=1).mean())
    ood_confidence_mean = float(np.max(calibrated_probs_ood, axis=1).mean())
    confidence_gap = id_confidence_mean - ood_confidence_mean
    
    result = {
        'ood_auroc': float(ood_auroc),
        'ood_fpr95': float(ood_fpr95),
        'id_confidence_mean': id_confidence_mean,
        'ood_confidence_mean': ood_confidence_mean,
        'confidence_gap': float(confidence_gap),
    }
    
    # OOD calibration quality (ECE on OOD using ID-fitted calibrator)
    # Note: For true OOD like SVHN on CIFAR, accuracy should be ~10% (random)
    # ECE measures if the calibrator's confidence matches this low accuracy
    if ood_labels is not None:
        ood_ece = calculate_ece(calibrated_probs_ood, ood_labels)
        ood_adaptive_ece = calculate_adaptive_ece(calibrated_probs_ood, ood_labels)
        ood_brier = calculate_brier_score(calibrated_probs_ood, ood_labels)
        ood_accuracy = calculate_accuracy(calibrated_probs_ood, ood_labels)
        result['ood_ece'] = float(ood_ece)
        result['ood_adaptive_ece'] = float(ood_adaptive_ece)
        result['ood_brier'] = float(ood_brier)
        result['ood_accuracy'] = float(ood_accuracy)
        
        # Log interpretation
        logger.info(f"  OOD ECE: {ood_ece:.4f} (lower = better calibration transfer)")
        logger.info(f"  OOD Adaptive ECE: {ood_adaptive_ece:.4f} (lower = better calibration transfer)")
        logger.info(f"  OOD Brier: {ood_brier:.4f} (lower = better calibration transfer)")
        logger.info(f"  OOD Accuracy: {ood_accuracy:.4f} (expected ~0.10 for random)")
        
        # Interpretation: is the model well-calibrated on OOD?
        if ood_confidence_mean < 0.2 and ood_ece < 0.1:
            logger.info(f"  -> Good calibration transfer: low confidence, low ECE")
        elif ood_confidence_mean > 0.5 and ood_ece > 0.3:
            logger.info(f"  -> Poor calibration transfer: overconfident on OOD")
    
    return result


def extract_raw_from_loader(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Extract raw data and labels from a DataLoader."""
    xs, ys = [], []
    for batch in loader:
        data, labels = batch[:2]
        xs.append(data.numpy())
        ys.append(labels.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def compute_baseline_ood_metrics(
    results: Dict[str, Any],
    model_adapter,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    ood_raw: np.ndarray,
    ood_labels: np.ndarray,
    batch_size: int,
    device: str,
    train_raw: np.ndarray = None,
    train_labels: np.ndarray = None,
    dac_knn_backend: str = 'faiss',
) -> Dict[str, Any]:
    """
    Add OOD metrics to standard baseline calibration methods.
    Only computes metrics for baselines that don't already have them.
    
    Modifies results dict in-place and returns it.
    """
    if ood_raw is None:
        logger.info("No OOD data provided, skipping baseline OOD metrics")
        return results
    
    baselines_section = results.get('standard_baselines')
    if baselines_section is None:
        logger.warning("standard_baselines missing in results; skipping baseline OOD metrics")
        results['standard_baselines'] = {}
        return results
    
    # Get logits for all sets
    logger.info("Extracting logits for OOD evaluation...")
    val_logits = model_adapter.predict_logits(val_raw, batch_size=batch_size)
    test_logits = model_adapter.predict_logits(test_raw, batch_size=batch_size)
    ood_logits = model_adapter.predict_logits(ood_raw, batch_size=batch_size)
    
    # Helper to compute softmax
    def softmax(x, axis=1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    # === UNCALIBRATED ===
    if 'uncalibrated' in baselines_section:
        if baselines_section['uncalibrated'].get('ood_auroc') is None:
            logger.info("Adding OOD metrics to: uncalibrated")
            try:
                test_probs = softmax(test_logits, axis=1)
                ood_probs = softmax(ood_logits, axis=1)
                ood_metrics = compute_ood_metrics(test_probs, ood_probs, test_labels, ood_labels=ood_labels)
                baselines_section['uncalibrated'].update(ood_metrics)
            except Exception as e:
                logger.error(f"Failed to compute OOD for uncalibrated: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # === TEMPERATURE SCALING ===
    if 'temperature_scaling' in baselines_section:
        if baselines_section['temperature_scaling'].get('ood_auroc') is None:
            logger.info("Adding OOD metrics to: temperature_scaling")
            try:
                from Calibrators.temperature_scaling import TemperatureScaling
                ts = TemperatureScaling()
                ts.fit(val_logits, val_labels)
                test_probs = ts.calibrate(test_logits)
                ood_probs = ts.calibrate(ood_logits)
                ood_metrics = compute_ood_metrics(test_probs, ood_probs, test_labels, ood_labels=ood_labels)
                baselines_section['temperature_scaling'].update(ood_metrics)
            except Exception as e:
                logger.error(f"Failed to compute OOD for temperature_scaling: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # === ISOTONIC TOPLABEL ===
    if 'isotonic_toplabel' in baselines_section:
        if baselines_section['isotonic_toplabel'].get('ood_auroc') is None:
            logger.info("Adding OOD metrics to: isotonic_toplabel")
            try:
                from Calibrators.isotonic_regression import TopLabelIsotonicCalibrator
                iso = TopLabelIsotonicCalibrator()
                iso.fit(val_logits, val_labels)
                test_probs = iso.calibrate(test_logits)
                ood_probs = iso.calibrate(ood_logits)
                ood_metrics = compute_ood_metrics(test_probs, ood_probs, test_labels, ood_labels=ood_labels)
                baselines_section['isotonic_toplabel'].update(ood_metrics)
            except Exception as e:
                logger.error(f"Failed to compute OOD for isotonic_toplabel: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # === PLATT SCALING ===
    if 'platt_scaling' in baselines_section:
        if baselines_section['platt_scaling'].get('ood_auroc') is None:
            logger.info("Adding OOD metrics to: platt_scaling")
            try:
                from Calibrators.platt_scaling import PlattScaling
                platt = PlattScaling()
                platt.fit(val_logits, val_labels)
                test_probs = platt.calibrate(test_logits)
                ood_probs = platt.calibrate(ood_logits)
                ood_metrics = compute_ood_metrics(test_probs, ood_probs, test_labels, ood_labels=ood_labels)
                baselines_section['platt_scaling'].update(ood_metrics)
            except Exception as e:
                logger.error(f"Failed to compute OOD for platt_scaling: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # === BETA CALIBRATION ===
    if 'beta_calibration' in baselines_section:
        if baselines_section['beta_calibration'].get('ood_auroc') is None:
            logger.info("Adding OOD metrics to: beta_calibration")
            try:
                from Calibrators.beta_calibration import BetaCalibration
                beta = BetaCalibration()
                beta.fit(val_logits, val_labels)
                test_probs = beta.calibrate(test_logits)
                ood_probs = beta.calibrate(ood_logits)
                ood_metrics = compute_ood_metrics(test_probs, ood_probs, test_labels, ood_labels=ood_labels)
                baselines_section['beta_calibration'].update(ood_metrics)
            except Exception as e:
                logger.error(f"Failed to compute OOD for beta_calibration: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # === DENSITY AWARE CALIBRATION (DAC) ===
    if 'density_aware_calibration' in baselines_section:
        dac_result = baselines_section['density_aware_calibration']
        if not dac_result.get('skipped', False) and dac_result.get('ood_auroc') is None:
            logger.info("Adding OOD metrics to: density_aware_calibration")
            try:
                if train_raw is None or train_labels is None:
                    logger.warning("train_raw and train_labels required for DAC OOD metrics, skipping")
                else:
                    from Calibrators.density_aware_calibration import (
                        DensityAwareCalibrator, get_dac_target_layers,
                        extract_dac_features, get_dac_k_value
                    )
                    
                    # Get the model from adapter
                    model = model_adapter.model
                    model_name = model.__class__.__name__.lower()
                    
                    # Infer model name for DAC layer selection
                    if 'resnet' in model_name and hasattr(model, 'layer1'):
                        num_blocks = [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)]
                        if num_blocks == [2, 2, 2, 2]:
                            model_name = "resnet18"
                        elif num_blocks == [3, 4, 6, 3]:
                            model_name = "resnet50"
                        elif num_blocks == [3, 4, 23, 3]:
                            model_name = "resnet101"
                        elif num_blocks == [3, 8, 36, 3]:
                            model_name = "resnet152"
                    elif 'densenet' in model_name:
                        model_name = "densenet121"
                    
                    layer_names = get_dac_target_layers(model_name, model)
                    dev = torch.device(device)
                    
                    # Create data loaders
                    train_loader = DataLoader(
                        TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                        batch_size=batch_size, shuffle=False
                    )
                    val_loader = DataLoader(
                        TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                        batch_size=batch_size, shuffle=False
                    )
                    test_loader = DataLoader(
                        TensorDataset(torch.from_numpy(test_raw), torch.from_numpy(test_labels)),
                        batch_size=batch_size, shuffle=False
                    )
                    ood_loader = DataLoader(
                        TensorDataset(torch.from_numpy(ood_raw), torch.from_numpy(ood_labels)),
                        batch_size=batch_size, shuffle=False
                    )
                    
                    # Extract features
                    train_feats, _, train_lbls = extract_dac_features(model, train_loader, layer_names, dev)
                    val_feats, val_logits_dac, val_lbls = extract_dac_features(model, val_loader, layer_names, dev)
                    test_feats, test_logits_dac, _ = extract_dac_features(model, test_loader, layer_names, dev)
                    ood_feats, ood_logits_dac, _ = extract_dac_features(model, ood_loader, layer_names, dev)
                    
                    # Fit DAC
                    dataset_name = model_adapter.dataset_name if hasattr(model_adapter, 'dataset_name') else "cifar10"
                    k_value = get_dac_k_value(dataset_name)
                    dev = torch.device(device)
                    
                    # Handle "both" mode: run both backends
                    backends_to_run = ['faiss', 'torch'] if dac_knn_backend == 'both' else [dac_knn_backend]
                    
                    for backend in backends_to_run:
                        dac = make_dac_calibrator(k=k_value, device=dev, backend=backend)
                        
                        # Synchronize GPU before timing
                        if dev.type == 'cuda':
                            torch.cuda.synchronize()
                        fit_start = time.perf_counter()
                        
                        dac.fit(
                            train_features_list=[f.numpy() for f in train_feats],
                            val_features_list=[f.numpy() for f in val_feats],
                            val_logits=val_logits_dac.numpy(),
                            val_labels=val_lbls.numpy(),
                        )
                        
                        if dev.type == 'cuda':
                            torch.cuda.synchronize()
                        fit_time = time.perf_counter() - fit_start
                        
                        # Calibrate
                        if dev.type == 'cuda':
                            torch.cuda.synchronize()
                        cal_start = time.perf_counter()
                        
                        test_probs = dac.calibrate([f.numpy() for f in test_feats], test_logits_dac.numpy())
                        ood_probs = dac.calibrate([f.numpy() for f in ood_feats], ood_logits_dac.numpy())
                        
                        if dev.type == 'cuda':
                            torch.cuda.synchronize()
                        cal_time = time.perf_counter() - cal_start
                        
                        ood_metrics = compute_ood_metrics(test_probs, ood_probs, test_labels, ood_labels=ood_labels)
                        ood_metrics['knn_backend'] = backend
                        ood_metrics['knn_backend_impl'] = dac.__class__.__module__ + '.' + dac.__class__.__name__
                        ood_metrics['fit_time'] = fit_time
                        ood_metrics['calibrate_time'] = cal_time
                        
                        # Store results under appropriate key
                        if backend == 'faiss' or dac_knn_backend == 'faiss':
                            baselines_section['density_aware_calibration'].update(ood_metrics)
                        else:
                            # Store torch results under separate key
                            key = 'density_aware_calibration_torch'
                            if key not in baselines_section:
                                baselines_section[key] = {}
                            baselines_section[key].update(ood_metrics)
            except Exception as e:
                logger.error(f"Failed to compute OOD for density_aware_calibration: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # === GEOMETRIC PHYSICAL SPACE ===
    if 'geometric_physical_space' in baselines_section:
        if baselines_section['geometric_physical_space'].get('ood_auroc') is None:
            logger.info("Adding OOD metrics to: geometric_physical_space")
            try:
                if train_raw is None or train_labels is None:
                    logger.warning("train_raw and train_labels required for geometric_physical_space OOD metrics, skipping")
                else:
                    from Experiments.layer_selection import compress_with_spp_jl
                    from Calibrators.geometric_calibrator_new import GeometricCalibrator
                    
                    FINAL_DIM = 1024
                    PYR = [4, 2, 1]
                    
                    # Compress all sets with same parameters
                    Xtr_c = compress_with_spp_jl(train_raw, final_output_dim=FINAL_DIM, pyramid_levels=PYR,
                                                 seed=42, batch_size=batch_size, device=device)
                    Xva_c = compress_with_spp_jl(val_raw, final_output_dim=FINAL_DIM, pyramid_levels=PYR,
                                                 seed=42, batch_size=batch_size, device=device)
                    Xte_c = compress_with_spp_jl(test_raw, final_output_dim=FINAL_DIM, pyramid_levels=PYR,
                                                 seed=42, batch_size=batch_size, device=device)
                    Xood_c = compress_with_spp_jl(ood_raw, final_output_dim=FINAL_DIM, pyramid_levels=PYR,
                                                  seed=42, batch_size=batch_size, device=device)
                    
                    # Fit geometric calibrator
                    geo = GeometricCalibrator(
                        model=model_adapter,
                        X_train_embed=Xtr_c,
                        y_train=train_labels,
                        auto_select_layer=False,
                        library="fast_separation",
                        device=device,
                    )
                    geo.fit(X_val_embed=Xva_c, X_val_original=val_raw, y_val=val_labels, fit_batch_size=batch_size)
                    
                    # Calibrate test and OOD
                    test_probs = geo.calibrate_batched(X_test_embed=Xte_c, X_test_original=test_raw, batch_size=batch_size)
                    ood_probs = geo.calibrate_batched(X_test_embed=Xood_c, X_test_original=ood_raw, batch_size=batch_size)
                    
                    ood_metrics = compute_ood_metrics(test_probs, ood_probs, test_labels, ood_labels=ood_labels)
                    baselines_section['geometric_physical_space'].update(ood_metrics)
            except Exception as e:
                logger.error(f"Failed to compute OOD for geometric_physical_space: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    results['standard_baselines'] = baselines_section
    return results


def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    """L2 normalize numpy array along axis=1 (per sample)."""
    return normalize(x, axis=1, norm='l2')


def make_json_serializable(obj):
    """Recursively convert numpy arrays and remove non-serializable objects."""
    import numpy as np

    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            # Skip non-serializable objects that were only needed during computation
            if key in ['calibrator', 'val_calibrated_probs', 'model_probs']:
                continue
            cleaned[key] = make_json_serializable(value)
        return cleaned
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    else:
        return obj


def save_per_sample_data(
    calibrated_probs: np.ndarray,
    test_labels: np.ndarray,
    method_name: str,
    output_dir: str,
    stability_scores: np.ndarray = None,
    model_name: str = None,
    dataset_name: str = None,
    training_method: str = None,
    seed: int = None
):
    """
    Save per-sample predictions for creating reliability diagrams and risk-coverage curves.
    
    Args:
        calibrated_probs: Calibrated probabilities, shape (n_test, n_classes)
        test_labels: True labels, shape (n_test,)
        method_name: Name of the calibration method (for filename)
        output_dir: Base output directory
        stability_scores: Optional stability scores (for geometric methods), shape (n_test,)
        model_name: Optional model name for path construction
        dataset_name: Optional dataset name for path construction
        training_method: Optional training method for path construction
        seed: Optional seed for path construction
    """
    # Construct output directory path
    if model_name and dataset_name and training_method and seed is not None:
        out_dir = os.path.join(output_dir, model_name, dataset_name, training_method, str(seed))
    else:
        # Fallback: use output_dir directly if metadata not provided
        out_dir = output_dir
    per_sample_dir = os.path.join(out_dir, 'per_sample_data')
    os.makedirs(per_sample_dir, exist_ok=True)
    
    per_sample_data = {
        'max_probs': np.max(calibrated_probs, axis=1),  # shape (n_test,)
        'correct': (np.argmax(calibrated_probs, axis=1) == test_labels).astype(np.int8),  # shape (n_test,)
        'pred_labels': np.argmax(calibrated_probs, axis=1),  # shape (n_test,)
    }
    
    # Add stability scores if available (for geometric methods)
    if stability_scores is not None:
        per_sample_data['stability_scores'] = stability_scores
    
    # Sanitize method name for filename
    safe_method_name = method_name.replace(' ', '_').replace('/', '_').lower()
    
    # Build filename with metadata if available
    if model_name and dataset_name and training_method and seed is not None:
        # Include all metadata in filename for easy identification
        safe_model_name = model_name.replace(' ', '_').replace('/', '_').lower()
        safe_dataset_name = dataset_name.replace(' ', '_').replace('/', '_').lower()
        safe_training_method = training_method.replace(' ', '_').replace('/', '_').lower()
        filename = f'{safe_model_name}_{safe_dataset_name}_{safe_training_method}_seed{seed}_{safe_method_name}_per_sample.npz'
    else:
        # Fallback: just use method name if metadata not available
        filename = f'{safe_method_name}_per_sample.npz'
    
    filepath = os.path.join(per_sample_dir, filename)
    
    np.savez_compressed(filepath, **per_sample_data)
    logger.info(f"Saved per-sample data to {filepath}")

def save_results_incrementally(results_dict: Dict[str, Any], output_file: str):
    """Save results to JSON file incrementally."""
    try:
        cleaned = make_json_serializable(results_dict)
        with open(output_file, 'w') as f:
            json.dump(cleaned, f, indent=2)
        logger.info(f"Results saved incrementally to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results incrementally: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from utils.logging_config import get_logger
logger = get_logger(__name__)


def load_existing_results(output_file: str) -> Union[Dict[str, Any], None]:
    """Load existing results if file exists, return None otherwise."""
    if not os.path.exists(output_file):
        return None
    try:
        with open(output_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load existing results from {output_file}: {e}")
        return None


def is_metrics_section_complete(results: Dict[str, Any]) -> bool:
    """Check if metric_guided_calibration is present and non-empty."""
    if 'metric_guided_calibration' not in results:
        return False
    metrics = results['metric_guided_calibration']
    return bool(metrics)


def is_section_complete(results: Dict[str, Any], section_name: str, required_fields: List[str] = None) -> bool:
    """
    Check if a specific section exists and has required fields.

    Args:
        results: Full results dictionary
        section_name: Name of section to check (supports nested like 'ablation.dac_with_sgc_aggregated_features')
        required_fields: List of required fields within that section (default: ['ece'])

    Returns:
        True if section exists and has all required fields
    """
    if required_fields is None:
        required_fields = ['ece']

    # Handle nested sections (e.g., 'ablation.dac_with_sgc_aggregated_features')
    parts = section_name.split('.')
    current: Any = results

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]

    # Empty-section check
    if current is None:
        return False
    if isinstance(current, (list, dict)) and not current:
        return False
    
    # If section is marked as skipped, consider it complete (no need to rerun)
    if isinstance(current, dict) and current.get('skipped', False):
        return True

    # Helper to check required fields in a single mapping
    def _mapping_has_fields(mapping: Dict[str, Any]) -> bool:
        for field in required_fields:
            if field not in mapping:
                return False
        return True

    # If this is a dict of dicts (e.g., geometric_original layer results), require every entry to have fields
    if isinstance(current, dict):
        # If it already has the required fields at top-level, just check that.
        if _mapping_has_fields(current):
            return True
        # Otherwise, if all values are dicts, ensure each has the required fields.
        if all(isinstance(v, dict) for v in current.values()):
            return all(_mapping_has_fields(v) for v in current.values())
        # Fallback: treat as a single mapping.
        return _mapping_has_fields(current)

    # If this is a list of dicts (e.g., metric_guided_calibration), ensure each item has required fields.
    if isinstance(current, list):
        if not current:
            return False
        for item in current:
            if not isinstance(item, dict):
                return False
            if not _mapping_has_fields(item):
                return False
        return True

    # For scalar/other types, we can't meaningfully check fields; require non-None / non-empty.
    return bool(current)


# ---- Resume / completeness helpers -----------------------------------------

GEO_REQUIRED_FIELDS = ["ece", "calibrator_params"]
DAC_REQUIRED_FIELDS = ["ece", "layer_weights"]

# Module-level flag for force rerun geometric (set in main())
_FORCE_RERUN_GEOMETRIC = False

# List of geometric calibration sections that should be force-rerun when --force-rerun-geometric is set
GEOMETRIC_SECTIONS_TO_RERUN = [
    'global_random',
    'global_random_separation',
    'global_random_trust_score',
    'sgc_with_dac_layers',
    'sgc_with_dac_layers_separation',
    'sgc_with_dac_layers_trust_score',
    'sgc_with_tulip_layers',
    'sgc_with_tulip_layers_separation',
    'sgc_with_tulip_layers_trust_score',
    'geometric_concatenated',
    'last_layer_only_baseline',
    'coordinate_sampling',
    'coordinate_sampling_separation',
    'coordinate_sampling_trust_score',
    'coordinate_spp',
    'coordinate_spp_separation',
    'coordinate_spp_trust_score',
    'coordinate_spp_dac',
    'pixel_geometric',
    'metric_guided_calibration',
    'geometric_original',  # Single-layer geometric
    'geometric_precomputed',  # Precomputed geometric
    'sgc_faiss',  # FAISS version of SGC
    'coordinate_sampling_faiss',  # FAISS version
    'coordinate_spp_faiss',  # FAISS version
    # Nested sections under ablation
    'ablation.sgc_with_dac_preprocessing',
    'ablation.sgc_with_dac_preprocessing_separation',
    'ablation.sgc_with_dac_preprocessing_trust_score',
    'ablation.random_single_layer_dac_preprocessing',
]

def is_result_complete(entry: Any, required_fields: List[str] = None) -> bool:
    """
    Returns True iff `entry` is a dict-like result that:
      - is a dict
      - does NOT represent an error
      - contains all required fields, and they are not None

    This is used to support partial re-runs when you add new result keys
    (e.g., 'calibrator_params') and older cached results predate the change.
    """
    if required_fields is None:
        required_fields = GEO_REQUIRED_FIELDS

    if not isinstance(entry, dict):
        return False

    # Treat any recorded failure as incomplete so we re-run it.
    if "error" in entry:
        return False

    for k in required_fields:
        if k not in entry or entry[k] is None:
            return False

    return True


def should_rerun_section(results: Dict[str, Any], section_name: str, required_fields: List[str] = None, require_ood: bool = False) -> bool:
    """
    Wrapper around is_section_complete: re-run the section if it is missing
    required fields (e.g., newly added keys).
    
    Args:
        results: Results dictionary
        section_name: Name of the section to check
        required_fields: List of required field names
        require_ood: If True, also check for OOD metrics (ood_auroc)
    """
    # Check if force rerun geometric is enabled and this is a geometric section
    if _FORCE_RERUN_GEOMETRIC:
        # Check if section_name matches any geometric section (exact match or starts with)
        for geo_section in GEOMETRIC_SECTIONS_TO_RERUN:
            if section_name == geo_section or section_name.startswith(geo_section + '_') or section_name.startswith(geo_section + '.'):
                logger.info(f"  {section_name}: force rerun (--force-rerun-geometric flag set)")
                return True
    
    if required_fields is None:
        required_fields = GEO_REQUIRED_FIELDS
    
    # Check if section is complete with required fields
    is_complete = is_section_complete(results, section_name, required_fields=required_fields)
    
    # If OOD is required but missing, need to rerun
    if require_ood and is_complete:
        section = results.get(section_name, {})
        if section.get('ood_auroc') is None:
            logger.info(f"  {section_name}: missing OOD metrics, will recompute")
            return True
    
    return not is_complete


def _convert_layer_metrics_nones_to_nan(layer_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Utility for patch-mode: convert None values (from JSON) back to np.nan so that
    selection utilities like select_best_layer_by_metrics can operate as before.
    """
    converted = {}
    for layer_name, metrics in layer_metrics.items():
        new_metrics = {}
        for k, v in metrics.items():
            if v is None:
                new_metrics[k] = np.nan
            else:
                new_metrics[k] = v
        converted[layer_name] = new_metrics
    return converted


def _recompute_layer_metrics_for_patch(
    args,
    device: torch.device
) -> (Dict[str, Dict[str, float]], Dict[str, str]):
    """
    Patch-mode helper: if layer_quality_metrics is missing from an existing
    results JSON, recompute the layer metrics and metric-based selections
    without touching other (already computed) sections.
    This will load the model, extract features and run the metric pipeline,
    but it will NOT recompute DAC / TS / ablations etc.
    """
    logger.info("Recomputing layer metrics for patch mode (this may be expensive)...")

    # Load data
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset,
        args.batch_size,
        seed=args.seed
    )

    # Load model
    model_path = construct_model_path(
        args.results_base_dir,
        args.training_method,
        args.dataset,
        args.model_name,
        args.seed
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    logger.info(f"[Patch mode] Loading model from: {model_path}")
    model = load_trained_model(
        model_path,
        args.model_name,
        num_classes,
        device,
        dataset=args.dataset
    )

    # Extract raw data
    logger.info("[Patch mode] Extracting raw data for metrics...")

    def extract_raw_data(loader):
        raw_data = []
        labels = []
        for data, target in loader:
            raw_data.append(data.numpy())
            labels.append(target.numpy())
        return np.concatenate(raw_data), np.concatenate(labels)

    train_raw, train_labels = extract_raw_data(train_loader)
    val_raw, val_labels = extract_raw_data(val_loader)
    test_raw, test_labels = extract_raw_data(test_loader)

    # Model adapter
    model_adapter = PyTorchModelAdapter(model, device, args.dataset)

    # Extract features once (all layers)
    precomputed_features = extract_all_features_once(
        model, args.model_name,
        train_raw, train_labels,
        val_raw, val_labels,
        test_raw, test_labels,
        device, args.batch_size,
        dataset=args.dataset,
        pooling_mode=args.spp_pooling_mode,
        target_dimension=getattr(args, 'target_dimension', None)
    )

    # Validation logits for metrics
    logger.info("[Patch mode] Computing validation logits for metrics...")
    val_logits = model_adapter.predict_proba(val_raw)

    # Compute layer metrics (no precomputed calibrations here, so this may fit
    # a geometric calibrator per layer internally where needed).
    layer_metrics = compute_all_layer_metrics_once(
        precomputed_features,
        train_labels,
        val_labels,
        val_logits,
        model_adapter=model_adapter,
        val_raw=val_raw,
        precomputed_calibrations=None
    )

    # Select best layers per metric
    metric_selections = select_best_layer_by_metrics(layer_metrics)

    return layer_metrics, metric_selections


def compute_fast_separation_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
    query_pred: np.ndarray,
    use_cuda: bool = True
) -> np.ndarray:
    """
    Compute fast separation scores using the same method as GeometricCalibrator.
    This wraps our existing StabilitySpace implementation for consistency.
    
    Args:
        train_features: Training feature vectors [n_train, d]
        train_labels: Training labels [n_train]
        query_features: Query feature vectors [n_query, d]
        query_pred: Predicted labels for query points [n_query] - determines which 
                    class to use for same/other distance computation
        use_cuda: Whether to use GPU acceleration
        
    Returns:
        Separation scores for each query point [n_query]
    """
    # Create StabilitySpace instance using the same settings as GeometricCalibrator
    stability_space = StabilitySpace(
        X_train=train_features,
        y_train=train_labels,
        compression=None,
        library='fast_separation',
        metric='l2',
        use_cuda=use_cuda
    )
    
    # Use the same calculation as GeometricCalibrator
    # This calls _stability_fast_separation_gpu internally
    separation_scores = stability_space.calc_stab(query_features, query_pred)
    
    return separation_scores


def compute_oracle_separation_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    use_cuda: bool = True
) -> np.ndarray:
    """
    Compute oracle separation scores (using TRUE labels).
    Use this for layer quality assessment - measures how well the layer
    naturally separates the true classes.
    
    Args:
        train_features: Training feature vectors [n_train, d]
        train_labels: Training labels [n_train]
        query_features: Query feature vectors [n_query, d]
        query_labels: TRUE labels [n_query]
        use_cuda: Whether to use GPU acceleration
        
    Returns:
        Oracle separation scores for each query point [n_query]
    """
    return compute_fast_separation_scores(
        train_features=train_features,
        train_labels=train_labels,
        query_features=query_features,
        query_pred=query_labels,  # Use true labels as "predictions"
        use_cuda=use_cuda
    )


def normalize_discovered_layers(model: torch.nn.Module, discovered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure each discovered layer has a descriptive name derived from named_modules.
    Copied from run_random_layer_ablation.py for consistency.
    """
    idx_to_name = {i: name for i, (name, _) in enumerate(model.named_modules())}
    normalized = []
    for entry in discovered:
        idx = entry.get("idx")
        entry = dict(entry)
        entry["name"] = idx_to_name.get(idx, entry.get("name", f"layer_{idx}"))
        normalized.append(entry)
    return normalized


def filter_non_feature_layers(discovered_layers: List[Dict[str, Any]], model: torch.nn.Module) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter out activation layers and other non-feature-bearing layers from discovered layers.
    
    These layers (ReLU, GELU, SiLU, Dropout, etc.) can have inconsistent output shapes
    and don't contain meaningful learned representations suitable for calibration.
    
    Args:
        discovered_layers: List of discovered layer dictionaries
        model: PyTorch model to check module types
    
    Returns:
        Tuple of (filtered_layers, excluded_layers)
    """
    filtered = filter_feature_layers(discovered_layers, model)
    excluded = [l for l in discovered_layers if l not in filtered]
    return filtered, excluded


def compute_calibration_separation_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
    query_raw: np.ndarray,
    model_adapter: 'PyTorchModelAdapter',
    use_cuda: bool = True
) -> np.ndarray:
    """
    Compute calibration separation scores (using MODEL PREDICTIONS).
    Use this for actual calibration - separation relative to what the model
    predicts, not the true label.
    
    Args:
        train_features: Training feature vectors [n_train, d]
        train_labels: Training labels [n_train]
        query_features: Query feature vectors [n_query, d]
        query_raw: Raw input data for query points (for model prediction)
        model_adapter: Model adapter to get predictions
        use_cuda: Whether to use GPU acceleration
        
    Returns:
        Calibration separation scores for each query point [n_query]
    """
    # Get model predictions
    logits = model_adapter.predict_proba(query_raw)
    query_pred = np.argmax(logits, axis=1)
    
    return compute_fast_separation_scores(
        train_features=train_features,
        train_labels=train_labels,
        query_features=query_features,
        query_pred=query_pred,  # Use model predictions
        use_cuda=use_cuda
    )


def calculate_accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Calculate classification accuracy."""
    predictions = np.argmax(probs, axis=1)
    return 100.0 * np.mean(predictions == labels)


def np_softmax(x):
    """Numerically stable softmax"""
    max_val = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_val)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def calculate_ece(probs: np.ndarray, labels: np.ndarray) -> float:
    """Calculate Expected Calibration Error."""
    predictions = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    return expected_calibration_error(confidences, predictions, labels)


def calculate_calibration_mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Calculate Maximum Calibration Error (worst non-empty confidence-bin gap)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    max_error = 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        else:
            in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])

        if np.any(in_bin):
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            max_error = max(max_error, abs(accuracy_in_bin - avg_confidence_in_bin))
    return float(max_error)


def calculate_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Calculate Brier Score."""
    num_classes = probs.shape[1]
    one_hot_labels = np.eye(num_classes)[labels]
    return np.mean(np.sum((probs - one_hot_labels)**2, axis=1))


def calculate_adaptive_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Calculate Adaptive Expected Calibration Error (ECE with equal-frequency binning)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    # Adaptive binning: equal number of samples per bin
    n_samples = len(confidences)
    sorted_indices = np.argsort(confidences)
    sorted_confidences = confidences[sorted_indices]
    
    # Create bin boundaries with equal number of samples per bin
    bin_boundaries = np.interp(
        np.linspace(0, n_samples, n_bins + 1),
        np.arange(n_samples),
        sorted_confidences
    )
    
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Include samples in this bin
        if i == n_bins - 1:
            # Last bin includes upper boundary
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


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
    sampling_plan: Dict[str, List[int]],
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


def compute_layer_quality_metrics(
    train_features: torch.Tensor,
    train_labels: np.ndarray,
    val_features: torch.Tensor,
    val_labels: np.ndarray,
    val_logits: np.ndarray,
    layer_name: str,
    model_adapter: 'PyTorchModelAdapter' = None,
    val_raw: np.ndarray = None,
    precomputed_val_probs: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute comprehensive quality metrics for a single layer.
    
    Args:
        precomputed_val_probs: Optional pre-computed calibrated probabilities on validation set.
            If provided, skips GeometricCalibrator fitting and uses these directly.
    
    Returns dict with all metric values for this layer.
    """
    # Convert to numpy/torch as needed
    train_np = train_features.numpy() if isinstance(train_features, torch.Tensor) else train_features
    val_np = val_features.numpy() if isinstance(val_features, torch.Tensor) else val_features
    
    # Convert to torch tensors for metric calculators
    train_torch = torch.from_numpy(train_np) if isinstance(train_np, np.ndarray) else train_features
    val_torch = torch.from_numpy(val_np) if isinstance(val_np, np.ndarray) else val_features
    
    if not isinstance(train_labels, torch.Tensor):
        train_labels_torch = torch.from_numpy(train_labels) if isinstance(train_labels, np.ndarray) else torch.tensor(train_labels)
    else:
        train_labels_torch = train_labels
    
    if not isinstance(val_labels, torch.Tensor):
        val_labels_torch = torch.from_numpy(val_labels) if isinstance(val_labels, np.ndarray) else torch.tensor(val_labels)
    else:
        val_labels_torch = val_labels
    
    metrics = {}
    
    # 1. Margin-based metrics (computed on validation set for layer selection)
    try:
        margin_calc = MarginTailCVaRCalculator(alpha=0.2)
        # Compute on validation set since we're selecting layers based on validation performance
        metrics['margin_tail_cvar'] = margin_calc.compute(val_torch, val_labels_torch)
    except Exception as e:
        logger.warning(f"Failed to compute margin_tail_cvar for {layer_name}: {e}")
        metrics['margin_tail_cvar'] = np.nan
    
    # 2. Calibration decisiveness (computed from ACTUAL calibrated probabilities)
    try:
        if precomputed_val_probs is not None:
            # Use precomputed calibrated probabilities (skip fitting)
            logger.info(f"  Using precomputed calibrated probabilities for calibration_decisiveness")
            calibrated_confidences = np.max(precomputed_val_probs, axis=1)
            metrics['calibration_decisiveness'] = float(np.mean(calibrated_confidences))
            logger.info(f"  Calibrated decisiveness: {metrics['calibration_decisiveness']:.6f}")
        elif model_adapter is not None and val_raw is not None:
            from Calibrators.geometric_calibrator_new import GeometricCalibrator
            
            # Convert to numpy if needed
            train_np = train_features.numpy() if torch.is_tensor(train_features) else train_features
            val_np = val_features.numpy() if torch.is_tensor(val_features) else val_features
            
            # Create geometric calibrator using THIS layer's features
            geo_cal = GeometricCalibrator(
                model=model_adapter,
                X_train_embed=train_np,
                y_train=train_labels,
                library="fast_separation",
                auto_select_layer=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Fit calibrator on validation set
            geo_cal.fit(
                X_val_embed=val_np,
                y_val=val_labels,
                X_val_original=val_raw,
                fit_batch_size=128
            )
            
            # Get calibrated probabilities on validation set
            calibrated_probs = geo_cal.calibrate_batched(
                X_test_embed=val_np,
                X_test_original=val_raw,
                batch_size=128
            )
            
            # Extract max confidences and compute mean
            calibrated_confidences = np.max(calibrated_probs, axis=1)
            metrics['calibration_decisiveness'] = float(np.mean(calibrated_confidences))
            
            logger.info(f"  Calibrated decisiveness: {metrics['calibration_decisiveness']:.6f}")
        else:
            logger.warning(f"model_adapter or val_raw not provided, skipping calibration_decisiveness")
            metrics['calibration_decisiveness'] = np.nan
            
    except Exception as e:
        logger.warning(f"Failed to compute calibration_decisiveness for {layer_name}: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        metrics['calibration_decisiveness'] = np.nan
    
    # 3. Multiscale separation
    try:
        multiscale_calc = MultiScaleClassHomogeneityCalculator(k_list=[5, 15, 30])
        metrics['multiscale_separation'] = multiscale_calc.compute(val_torch, val_labels_torch)
    except Exception as e:
        logger.warning(f"Failed to compute multiscale_separation for {layer_name}: {e}")
        metrics['multiscale_separation'] = np.nan
    
    # 4. Local Intrinsic Dimensionality
    try:
        lid_calc = LocalIntrinsicDimensionalityCalculator(k=20)
        metrics['lid'] = lid_calc.compute(val_torch, val_labels_torch)
    except Exception as e:
        logger.warning(f"Failed to compute LID for {layer_name}: {e}")
        metrics['lid'] = np.nan
    
    # 5. Neural Collapse metrics
    try:
        nc1_calc = NC1CollapseMetricCalculator()
        metrics['nc1'] = nc1_calc.compute(val_torch, val_labels_torch)
    except Exception as e:
        logger.warning(f"Failed to compute NC1 for {layer_name}: {e}")
        metrics['nc1'] = np.nan
    
    try:
        # NC4: Compute centroids from train, evaluate on val
        # This matches the user's spec: compute_nc4_score(train_np, train_labels, val_np)
        train_unique_labels = torch.unique(train_labels_torch)
        val_unique_labels = torch.unique(val_labels_torch)
        
        if len(train_unique_labels) < 2 or len(val_unique_labels) < 2:
            metrics['nc4'] = 0.0
        else:
            # Compute centroids from train features
            mu_c_list = []
            label_to_idx = {label.item(): idx for idx, label in enumerate(train_unique_labels)}
            
            for c in train_unique_labels:
                class_features = train_torch[train_labels_torch == c]
                if len(class_features) == 0:
                    mu_c_list.append(torch.zeros(train_torch.shape[1], device=train_torch.device))
                else:
                    mu_c_list.append(class_features.mean(dim=0))
            
            mus = torch.stack(mu_c_list)  # Shape [C, D_feat]
            
            # Evaluate on validation features
            dists = torch.cdist(val_torch, mus)  # Shape [N_val, C]
            preds_idx = torch.argmin(dists, dim=1)
            preds_class = train_unique_labels[preds_idx]
            
            # Calculate NCC accuracy on validation set
            acc = (preds_class == val_labels_torch).float().mean()
            metrics['nc4'] = float(acc.cpu().item())
    except Exception as e:
        logger.warning(f"Failed to compute NC4 for {layer_name}: {e}")
        metrics['nc4'] = np.nan
    
    return metrics


def compute_all_layer_metrics_once(
    precomputed_features: Dict[str, Any],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    val_logits: np.ndarray,
    model_adapter: 'PyTorchModelAdapter' = None,
    val_raw: np.ndarray = None,
    precomputed_calibrations: Dict[str, Dict[str, Any]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute all quality metrics for all layers.
    
    Args:
        precomputed_calibrations: Optional dict mapping layer_name to calibration results.
            If provided, uses precomputed_val_probs from results to skip redundant fitting.
            Expected format: {layer_name: {'val_calibrated_probs': np.ndarray, ...}}
    
    Returns dict: {layer_name: {metric_name: value}}
    """
    logger.info("\n" + "="*80)
    logger.info("COMPUTING LAYER QUALITY METRICS (ONE-TIME COMPUTATION)")
    logger.info("="*80)
    
    layer_names = precomputed_features['layer_names']
    train_features_list = precomputed_features['train_features_per_layer']
    val_features_list = precomputed_features['val_features_per_layer']
    
    all_metrics = {}
    
    start_time = time.perf_counter()
    
    for i, layer_name in enumerate(layer_names):
        logger.info(f"\nComputing metrics for layer {i+1}/{len(layer_names)}: {layer_name}")
        
        # Use precomputed validation probabilities if available
        val_probs = None
        if precomputed_calibrations is not None and layer_name in precomputed_calibrations:
            if 'val_calibrated_probs' in precomputed_calibrations[layer_name]:
                val_probs = precomputed_calibrations[layer_name]['val_calibrated_probs']
                logger.info(f"  Using precomputed calibrated probabilities (skipping redundant fitting)")
        
        metrics = compute_layer_quality_metrics(
            train_features_list[i],
            train_labels,
            val_features_list[i],
            val_labels,
            val_logits,
            layer_name,
            model_adapter=model_adapter,
            val_raw=val_raw,
            precomputed_val_probs=val_probs
        )
        
        all_metrics[layer_name] = metrics
        
        # Log computed metrics
        for metric_name, value in metrics.items():
            if not np.isnan(value):
                logger.info(f"  {metric_name:30s}: {value:.6f}")
    
    computation_time = time.perf_counter() - start_time
    logger.info(f"\nMetrics computation completed in {computation_time:.2f}s")
    logger.info("="*80)
    
    return all_metrics


def select_best_layer_by_metrics(
    layer_metrics: Dict[str, Dict[str, float]],
    metric_names: List[str] = None,
    include_median: bool = True  # New parameter
) -> Dict[str, str]:
    """
    Select Max, Min, and optionally Median layer for each metric.
    Returns a dict with keys like:
      - 'lid (Max)' -> 'layer1'
      - 'lid (Min)' -> 'layer4'
      - 'lid (Med)' -> 'layer2' (if include_median=True)
    """
    if metric_names is None:
        # Get all metrics present in the first layer
        first_layer = list(layer_metrics.values())[0]
        metric_names = list(first_layer.keys())
    
    selections = {}
    logger.info("\nExtremal layer selection by metric:")
    
    for metric_name in metric_names:
        layer_values = {}
        for layer_name, metrics in layer_metrics.items():
            # Collect valid values for this metric across all layers
            if metric_name in metrics and not np.isnan(metrics[metric_name]):
                layer_values[layer_name] = metrics[metric_name]
        
        if not layer_values:
            logger.warning(f"  No valid values for metric: {metric_name}")
            continue
        
        # 1. Select Layer that MAXIMIZES the metric
        max_layer = max(layer_values, key=layer_values.get)
        max_val = layer_values[max_layer]
        selections[f"{metric_name} (Max)"] = max_layer
        
        # 2. Select Layer that MINIMIZES the metric
        min_layer = min(layer_values, key=layer_values.get)
        min_val = layer_values[min_layer]
        selections[f"{metric_name} (Min)"] = min_layer
        
        # 3. MEDIAN (optional - finds layer closest to median value)
        if include_median:
            median_val = np.median(list(layer_values.values()))
            median_layer = min(layer_values, key=lambda k: abs(layer_values[k] - median_val))
            median_layer_val = layer_values[median_layer]
            selections[f"{metric_name} (Med)"] = median_layer
        
        logger.info(f"  {metric_name:25s} | Max: {max_layer:10s} ({max_val:.4f}) | Min: {min_layer:10s} ({min_val:.4f})" + 
                   (f" | Med: {median_layer:10s} ({median_layer_val:.4f})" if include_median else ""))
    
    # 4. PSC-STYLE SELECTION: NC1 filter + Max NC4
    # PSC paper approach: Select layers with high within-class variability (NC1 > 0.2)
    # then among those, pick the one with best class separability (Max NC4)
    logger.info("\nPSC-style selection (NC1 > 0.2, Max NC4):")
    
    try:
        nc1_threshold = 0.2
        
        # Get all layers with both NC1 and NC4 metrics
        layers_with_nc = {}
        for layer_name, metrics in layer_metrics.items():
            if 'nc1' in metrics and 'nc4' in metrics:
                if not np.isnan(metrics['nc1']) and not np.isnan(metrics['nc4']):
                    layers_with_nc[layer_name] = {
                        'nc1': metrics['nc1'],
                        'nc4': metrics['nc4']
                    }
        
        if not layers_with_nc:
            logger.warning("  No layers with valid NC1 and NC4 metrics")
        else:
            # Filter layers with NC1 > threshold
            filtered_layers = {
                name: vals for name, vals in layers_with_nc.items()
                if vals['nc1'] > nc1_threshold
            }
            
            if not filtered_layers:
                logger.warning(f"  No layers with NC1 > {nc1_threshold}")
                nc1_values_list = []
                for name, vals in layers_with_nc.items():
                    nc1_val = vals['nc1']
                    nc1_values_list.append(f"{name}: {nc1_val:.4f}")
                logger.info(f"  Available NC1 values: {nc1_values_list}")
            else:
                # Among filtered layers, select the one with maximum NC4
                psc_layer = max(filtered_layers, key=lambda k: filtered_layers[k]['nc4'])
                psc_nc1 = filtered_layers[psc_layer]['nc1']
                psc_nc4 = filtered_layers[psc_layer]['nc4']
                
                selections['PSC (NC1>0.2, Max NC4)'] = psc_layer
                
                logger.info(f"  Filtered {len(filtered_layers)}/{len(layers_with_nc)} layers with NC1 > {nc1_threshold}")
                logger.info(f"  Selected: {psc_layer} (NC1={psc_nc1:.4f}, NC4={psc_nc4:.4f})")
    
    except Exception as e:
        logger.warning(f"  Failed PSC selection: {e}")
        import traceback
        logger.debug(f"  Traceback: {traceback.format_exc()}")
    
    return selections


def extract_features_directly(model, dataloader, layer_name, device):
    """
    Extract features from a specific layer by name.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        layer_name: Layer name (str, supports dot notation)
        device: torch device
        
    Returns:
        features (torch.Tensor), labels (torch.Tensor)
    """
    model.eval()
    
    # Check if this is a DINO model that needs resizing
    needs_resize = hasattr(model, 'backbone') and model.backbone is not None
    
    # Get the target layer using the same helper as DAC
    target_layer = find_layer_module(model, layer_name)
    if target_layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    # Register hook
    features_list = []
    labels_list = []
    
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            feat = output.detach()
        elif isinstance(output, (tuple, list)):
            feat = output[0].detach() if len(output) > 0 else None
        else:
            return
        
        if feat is not None:
            # Apply global average pooling if 4D
            if feat.ndim == 4:
                feat = feat.mean(dim=[2, 3])
            features_list.append(feat.cpu())
    
    hook = target_layer.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    data, labels = batch
                else:
                    data, labels = batch[0], torch.zeros(len(batch[0]), dtype=torch.long)
                
                data = data.to(device)
                
                # Resize input for DINO models if needed (they expect 224x224)
                if needs_resize and data.size(-1) != 224:
                    data = torch.nn.functional.interpolate(
                        data, size=(224, 224), mode='bilinear', align_corners=False
                    )
                
                labels_list.append(labels)
                
                # Forward pass
                _ = model(data)
        
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
    finally:
        hook.remove()
    
    return features, labels


def extract_features_from_multiple_layers(
    model,
    dataloader,
    layer_names: List[str],
    device,
    compression_ratio: float = None,
    target_dimension: int = None,
    pooling_mode: str = 'max',
    spp_only: bool = False,  # When True, use SPP only (no JL); when False, use SPP+JL
    normalize_spp_output: bool = True,  # NEW PARAMETER
    return_on_device: bool = False,  # Phase 1: If True, keep features on device instead of moving to CPU
) -> List[torch.Tensor]:
    """
    Extract features from multiple layers using SPP-based pooling.

    There are two modes:
      - spp_only=True  → SPP only (no per-layer random projection). Output dim per layer
                         is C * sum(pyramid_levels^2).
      - spp_only=False → SPP + fixed JL projection (FixedSizeSPP_JL) with either
                         compression_ratio (legacy) or target_dimension (fixed d).
    
    Args:
        target_dimension: If provided, use fixed target dimension d instead of compression_ratio.
                         Takes precedence over compression_ratio.
        normalize_spp_output: If True, the FixedSizeSPP_JL module applies L2 normalization to
                         each layer's projected feature vector. If False, SPP/JL returns raw
                         (unnormalized) projected features, and the caller is responsible for
                         any subsequent normalization (e.g., in an ablation over normalization
                         strategies). Default True to preserve legacy behavior.
        return_on_device: Phase 1 optimization. If True, return features on the original device
                         (GPU) instead of moving to CPU. This eliminates unnecessary CPU↔GPU
                         transfers when features will be used immediately on GPU. Default False
                         for backward compatibility.
    """
    # Only require compression_ratio or target_dimension when we actually use JL projection
    if not spp_only and compression_ratio is None and target_dimension is None:
        raise ValueError("Either compression_ratio or target_dimension must be provided when spp_only=False")
    
    if pooling_mode not in ['avg', 'max']:
        raise ValueError(f"pooling_mode must be 'avg' or 'max', got '{pooling_mode}'")
    
    logger.info(f"🔧 Using SPP pooling mode: {pooling_mode}")
    
    model.eval()
    
    # Check if this is a DINO model that needs resizing
    needs_resize = hasattr(model, 'backbone') and model.backbone is not None
    
    # Storage for features from each layer
    layer_features = {name: [] for name in layer_names}
    
    # Track current batch's features (overwritten on each hook call, appended after forward pass)
    current_batch_features = {name: None for name in layer_names}
    
    # SPP+JL projectors for different channel sizes (shared across layers)
    spp_projectors = {}
    
    # Register hooks for all layers
    hooks = []
    
    def make_hook(layer_name):
        def hook_fn(module, input, output):
            from utils.tensor_utils import coerce_to_tensor
            
            raw = output
            feat = coerce_to_tensor(raw)
            
            if feat is None or not torch.is_tensor(feat):
                return
            
            # Apply SPP(+optional JL) compression (EXACT SAME as extract_features_directly)
            if feat.ndim == 3:
                import math
                B, T, C = feat.shape
                TT = T - 1
                has_square_grid = (TT > 0 and int(math.isqrt(TT)) ** 2 == TT)
                tokens = feat[:, 1:, :] if has_square_grid else feat
                Tuse = TT if has_square_grid else T
                s = int(math.isqrt(Tuse))
                if s * s == Tuse:
                    fmap = tokens.transpose(1, 2).reshape(B, C, s, s)
                    num_channels = fmap.size(1)

                    if num_channels not in spp_projectors:
                        if spp_only:
                            from utils.compression_utils import SPP_Only

                            spp_projectors[num_channels] = SPP_Only(
                                pyramid_levels=[4, 2, 1],
                                pooling_mode=pooling_mode,
                            ).to(device)
                        else:
                            from utils.compression_utils import FixedSizeSPP_JL

                            if target_dimension is not None:
                                spp_projectors[num_channels] = FixedSizeSPP_JL(
                                    in_channels=num_channels,
                                    final_output_dim=target_dimension,
                                    pooling_mode=pooling_mode,
                                    normalize_output=normalize_spp_output,
                                ).to(device)
                            else:
                                spp_projectors[num_channels] = FixedSizeSPP_JL(
                                    in_channels=num_channels,
                                    compression_ratio=compression_ratio,
                                    pooling_mode=pooling_mode,
                                    normalize_output=normalize_spp_output,
                                ).to(device)

                    projector = spp_projectors[num_channels]
                    feat = projector(fmap)
                else:
                    feat = tokens.mean(dim=1)
            elif feat.ndim > 3:
                num_channels = feat.size(1)

                if num_channels not in spp_projectors:
                    if spp_only:
                        from utils.compression_utils import SPP_Only

                        spp_projectors[num_channels] = SPP_Only(
                            pyramid_levels=[4, 2, 1],
                            pooling_mode=pooling_mode,
                        ).to(device)
                    else:
                        from utils.compression_utils import FixedSizeSPP_JL

                        if target_dimension is not None:
                            spp_projectors[num_channels] = FixedSizeSPP_JL(
                                in_channels=num_channels,
                                final_output_dim=target_dimension,
                                pooling_mode=pooling_mode,
                            ).to(device)
                        else:
                            spp_projectors[num_channels] = FixedSizeSPP_JL(
                                in_channels=num_channels,
                                compression_ratio=compression_ratio,
                                pooling_mode=pooling_mode,
                            ).to(device)

                projector = spp_projectors[num_channels]
                feat = projector(feat)
            elif feat.ndim == 2:
                pass
            else:
                return
            
            # OVERWRITE current batch's feature (ensures only LAST call per forward pass is kept)
            # Phase 1: Conditionally keep on device to avoid unnecessary CPU transfers
            if return_on_device:
                current_batch_features[layer_name] = feat.detach()
            else:
                current_batch_features[layer_name] = feat.detach().cpu()
        return hook_fn
    
    # Register hooks
    for layer_name in layer_names:
        target_layer = find_layer_module(model, layer_name)
        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")
        hook = target_layer.register_forward_hook(make_hook(layer_name))
        hooks.append(hook)
    
    try:
        with torch.no_grad():
            for batch in dataloader:
                data = batch[0] if len(batch) >= 1 else batch
                data = data.to(device)
                
                # Resize input for DINO models if needed (they expect 224x224)
                if needs_resize and data.size(-1) != 224:
                    data = torch.nn.functional.interpolate(
                        data, size=(224, 224), mode='bilinear', align_corners=False
                    )
                
                # Reset current batch features before forward pass
                for name in layer_names:
                    current_batch_features[name] = None
                
                _ = model(data)
                
                # After forward pass, append the captured features (only last output per layer)
                for name in layer_names:
                    if current_batch_features[name] is not None:
                        layer_features[name].append(current_batch_features[name])
        
        features_list = []
        for layer_name in layer_names:
            if not layer_features[layer_name]:
                raise ValueError(f"No features extracted for layer '{layer_name}'")
            features = torch.cat(layer_features[layer_name], dim=0)
            features_list.append(features)
    finally:
        for hook in hooks:
            hook.remove()
    
    return features_list


def combine_features(features_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Combine features from multiple layers by concatenation.
    
    Args:
        features_list: List of feature tensors, one per layer
        
    Returns:
        Combined feature tensor (concatenated along feature dimension)
    """
    # Concatenate along feature dimension (dim=1)
    combined = torch.cat(features_list, dim=1)
    return combined


def get_layer_names_for_model(model_name: str, model: torch.nn.Module = None) -> List[str]:
    """
    Get layer names for a given model architecture.
    Aligned with DAC paper (Table 6, page 14): PRE-BLOCK + BLOCK-1-4 (main block outputs only).
    
    The paper explicitly states they choose "the last layer of each block" - meaning just
    the main block outputs (layer1, layer2, layer3, layer4), not individual sub-blocks.
    
    Args:
        model_name: Model architecture name
        model: Optional model instance for dynamic layer detection
    
    Returns:
        List of layer names (strings for use with find_layer_module)
        For ResNet: [PRE-BLOCK, layer1, layer2, layer3, layer4] = 5 layers
        For DenseNet: [PRE-BLOCK, denseblock1, denseblock2, denseblock3, denseblock4] = 5 layers
    """
    model_name_lower = model_name.lower()
    
    if 'resnet' in model_name_lower:
        # DAC paper specification: PRE-BLOCK + BLOCK-1-4 (main block outputs only)
        # Total: 5 intermediate layers (logits added separately)
        layer_names = []
        
        # PRE-BLOCK: Try maxpool first (ImageNet), then conv1 (CIFAR)
        if model is not None:
            if find_layer_module(model, 'maxpool') is not None:
                layer_names.append('maxpool')
            elif find_layer_module(model, 'conv1') is not None:
                layer_names.append('conv1')
        else:
            # Static fallback: prefer maxpool (ImageNet standard), fallback to conv1
            layer_names.append('maxpool')  # Will be validated when used
        
        # BLOCK-1 to BLOCK-4: Main block outputs only (no sub-blocks)
        layer_names.extend(['layer1', 'layer2', 'layer3', 'layer4'])
        
        logger.info(f"ResNet layers (DAC paper spec): {layer_names}")
        logger.info(f"  Total: {len(layer_names)} intermediate layers (logits added separately)")
        
        return layer_names
    
    elif 'densenet' in model_name_lower:
        # DAC paper specification: PRE-BLOCK + BLOCK-1-4 (dense blocks only, no transitions)
        # Total: 5 intermediate layers (logits added separately)
        # Use the same logic as get_dac_target_layers to ensure consistency
        layer_names = get_dac_target_layers(model_name, model)
        
        logger.info(f"DenseNet layers (DAC paper spec): {layer_names}")
        logger.info(f"  Total: {len(layer_names)} intermediate layers (logits added separately)")
        
        return layer_names
    
    elif 'dinov2' in model_name_lower:
        # DINOv2 models: Use PRE-BLOCK equivalent + 4 evenly spaced transformer blocks
        # DINOv2 small/base have 12 blocks (0-11), large/giant have 24 blocks (0-23)
        # For consistency with DAC paper (5 layers), we select:
        # - block_0 (first block, equivalent to PRE-BLOCK)
        # - 4 evenly spaced blocks: block_2, block_5, block_8, block_11 (for 12-block models)
        #   or block_4, block_9, block_14, block_19 (for 24-block models)
        
        # Determine model size to select appropriate blocks
        if 'giant' in model_name_lower:
            # DINOv2 giant has 24 blocks
            layer_names = ['block_0', 'block_4', 'block_9', 'block_14', 'block_19']
        elif 'large' in model_name_lower:
            # DINOv2 large has 24 blocks
            layer_names = ['block_0', 'block_4', 'block_9', 'block_14', 'block_19']
        else:
            # DINOv2 small/base have 12 blocks (0-11)
            # Use block_0, block_2, block_5, block_8, block_11 (similar to multi_model_validation.py)
            layer_names = ['block_0', 'block_2', 'block_5', 'block_8', 'block_11']
        
        logger.info(f"DINOv2 layers (DAC paper spec, 5 layers): {layer_names}")
        logger.info(f"  Total: {len(layer_names)} intermediate layers (logits added separately)")
        
        return layer_names
    
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


def get_semantic_layer_name(layer_name: str, model_name: str) -> str:
    """
    Convert technical layer names to semantic names for better readability.
    
    Examples:
        ResNet: 'conv1' → 'Conv1 (Pre-block)', 'layer1' → 'Block1', 'fc' → 'Logits'
        DenseNet: 'dense1' → 'Dense1', 'trans1' → 'Trans1'
    """
    model_lower = model_name.lower()
    
    if 'resnet' in model_lower:
        # ResNet naming
        if layer_name == 'conv1':
            return 'Conv1 (Pre-block)'
        elif layer_name == 'maxpool':
            return 'MaxPool (Pre-block)'
        elif layer_name.startswith('layer1'):
            if layer_name == 'layer1':
                return 'Block1'
            else:
                sub = layer_name.split('.')[-1]
                return f'Block1.{sub}'
        elif layer_name.startswith('layer2'):
            if layer_name == 'layer2':
                return 'Block2'
            else:
                sub = layer_name.split('.')[-1]
                return f'Block2.{sub}'
        elif layer_name.startswith('layer3'):
            if layer_name == 'layer3':
                return 'Block3'
            else:
                sub = layer_name.split('.')[-1]
                return f'Block3.{sub}'
        elif layer_name.startswith('layer4'):
            if layer_name == 'layer4':
                return 'Block4'
            else:
                sub = layer_name.split('.')[-1]
                return f'Block4.{sub}'
        elif layer_name == 'fc' or layer_name == 'classifier':
            return 'Logits'
        else:
            return layer_name.capitalize()
    
    elif 'densenet' in model_lower:
        # DenseNet naming
        if 'dense' in layer_name.lower():
            # Extract block number: dense1, dense2, features.denseblock1, etc.
            if 'denseblock' in layer_name:
                num = layer_name.split('denseblock')[-1].split('.')[0]
                return f'DenseBlock{num}'
            elif layer_name.startswith('dense'):
                num = layer_name.replace('dense', '')
                return f'DenseBlock{num}'
        elif 'trans' in layer_name.lower():
            # Extract transition number
            if 'transition' in layer_name:
                num = layer_name.split('transition')[-1].split('.')[0]
                return f'Transition{num}'
            elif layer_name.startswith('trans'):
                num = layer_name.replace('trans', '')
                return f'Transition{num}'
        elif layer_name == 'bn' or 'norm' in layer_name.lower():
            return 'BatchNorm (Penultimate)'
        elif layer_name == 'classifier' or layer_name == 'fc':
            return 'Logits'
        else:
            return layer_name.capitalize()
    
    else:
        return layer_name.capitalize()


def get_penultimate_layer_name(model_name: str, all_layer_names: List[str]) -> str:
    """
    Get the penultimate (final feature) layer name for different architectures.
    This is the last layer before the classifier head.
    
    Args:
        model_name: Name of the model (e.g., 'resnet50', 'densenet121')
        all_layer_names: List of all available layer names from the model
        
    Returns:
        Name of the penultimate layer
    """
    model_lower = model_name.lower()
    
    if 'resnet' in model_lower:
        # ResNet: layer4 is the final feature block
        # Get the last layer4 sub-layer (e.g., layer4.2.conv2 or layer4.2)
        candidates = [n for n in all_layer_names if 'layer4' in n.lower()]
        if candidates:
            return candidates[-1]  # Last layer4 sub-layer
        # Fallback: second-to-last layer (before classifier)
        return all_layer_names[-2] if len(all_layer_names) >= 2 else all_layer_names[-1]
    
    elif 'densenet' in model_lower:
        # DenseNet: features.denseblock4 or norm5 is typically the final feature layer
        candidates = [n for n in all_layer_names if 'denseblock4' in n.lower() or 'norm5' in n.lower()]
        if candidates:
            return candidates[-1]
        # Fallback: second-to-last layer
        return all_layer_names[-2] if len(all_layer_names) >= 2 else all_layer_names[-1]
    
    elif 'dinov2' in model_lower or 'vit' in model_lower:
        # Vision Transformer: last transformer block output
        # Look for the last block or layer before classifier
        candidates = [n for n in all_layer_names if 'block' in n.lower() or 'layer' in n.lower()]
        if candidates:
            return candidates[-1]
        # Fallback: second-to-last layer
        return all_layer_names[-2] if len(all_layer_names) >= 2 else all_layer_names[-1]
    
    else:
        # Fallback: second-to-last layer (before classifier)
        return all_layer_names[-2] if len(all_layer_names) >= 2 else all_layer_names[-1]


def extract_all_features_once(
    model: torch.nn.Module,
    model_name: str,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    dataset: str = None,
    pooling_mode: str = 'max',
    target_dimension: int = None,
    normalize_spp_output: bool = True
) -> Dict[str, Any]:
    """
    Extract features from all layers ONCE and return them for reuse.
    
    Args:
        target_dimension: If provided, use fixed target dimension d instead of compression_ratio.
        normalize_spp_output: If True, the FixedSizeSPP_JL module applies L2 normalization to
                         each layer's projected feature vector. If False, SPP/JL returns raw
                         (unnormalized) projected features. Default True to preserve legacy behavior.
    
    Returns:
        Dictionary containing:
        - train_features_per_layer: List[torch.Tensor]
        - val_features_per_layer: List[torch.Tensor]
        - test_features_per_layer: List[torch.Tensor]
        - layer_names: List[str]
    """
    logger.info("\n" + "="*80)
    logger.info("EXTRACTING FEATURES FROM ALL LAYERS (ONE-TIME COMPUTATION)")
    logger.info("="*80)
    
    all_layer_names = get_layer_names_for_model(model_name, model)
    logger.info(f"Extracting from {len(all_layer_names)} layers: {all_layer_names}")
    
    # Use target_dimension if provided, otherwise fall back to compression_ratio
    if target_dimension is not None:
        logger.info(f"Using fixed target dimension: d={target_dimension}")
        compression_ratio = None
    else:
        # Determine compression ratio based on dataset
        if dataset is None:
            compression_ratio = 16.0  # Default for CIFAR10
            logger.warning(f"Dataset not provided, using default compression_ratio: {compression_ratio}x")
        else:
            dataset_lower = dataset.lower()
            if dataset_lower == "cifar100":
                compression_ratio = 32.0
            elif dataset_lower in ["tiny_imagenet", "tinyimagenet"]:
                compression_ratio = 32.0  # Similar to CIFAR-100 due to larger images
            else:
                compression_ratio = 16.0  # Default for CIFAR10
            logger.info(f"Using compression ratio: {compression_ratio}x for {dataset}")
    
    # Create dataloaders
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
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    train_features_list = extract_features_from_multiple_layers(
        model, train_loader, all_layer_names, device, 
        compression_ratio=compression_ratio,
        target_dimension=target_dimension,
        pooling_mode=pooling_mode,
        normalize_spp_output=normalize_spp_output
    )
    val_features_list = extract_features_from_multiple_layers(
        model, val_loader, all_layer_names, device, 
        compression_ratio=compression_ratio,
        target_dimension=target_dimension,
        pooling_mode=pooling_mode,
        normalize_spp_output=normalize_spp_output
    )
    test_features_list = extract_features_from_multiple_layers(
        model, test_loader, all_layer_names, device, 
        compression_ratio=compression_ratio,
        target_dimension=target_dimension,
        pooling_mode=pooling_mode,
        normalize_spp_output=normalize_spp_output
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    extraction_time = time.perf_counter() - start_time
    
    logger.info(f"\nFeature extraction completed in {extraction_time:.2f}s")
    for i, layer_name in enumerate(all_layer_names):
        logger.info(f"  Layer {i+1} ({layer_name}):")
        logger.info(f"    Train: {train_features_list[i].shape}")
        logger.info(f"    Val:   {val_features_list[i].shape}")
        logger.info(f"    Test:  {test_features_list[i].shape}")
    
    logger.info("="*80)
    
    return {
        'train_features_per_layer': train_features_list,
        'val_features_per_layer': val_features_list,
        'test_features_per_layer': test_features_list,
        'layer_names': all_layer_names,
        'extraction_time_s': extraction_time
    }


def compute_all_separation_scores_once(
    train_features_per_layer: List[torch.Tensor],
    train_labels: np.ndarray,
    val_features_per_layer: List[torch.Tensor],
    val_labels: np.ndarray,
    test_features_per_layer: List[torch.Tensor],
    test_labels: np.ndarray,
    layer_names: List[str],
    use_cuda: bool = True,
    model_adapter: 'PyTorchModelAdapter' = None,
    val_raw: np.ndarray = None,
    test_raw: np.ndarray = None
) -> Dict[str, Any]:
    """
    Compute geometric separation scores for all layers ONCE.
    Uses the same StabilitySpace implementation as GeometricCalibrator.
    
    Computes TWO types of separation scores:
    1. Oracle separation (using TRUE labels) - for layer quality assessment
    2. Calibration separation (using MODEL PREDICTIONS) - for actual calibration
    
    Args:
        train_features_per_layer: Features from each layer for training set
        train_labels: Training labels
        val_features_per_layer: Features from each layer for validation set
        val_labels: Validation TRUE labels
        test_features_per_layer: Features from each layer for test set
        test_labels: Test TRUE labels
        layer_names: Names of layers
        use_cuda: Whether to use GPU acceleration
        model_adapter: Model adapter for getting predictions (required for calibration separation)
        val_raw: Raw validation data (required for calibration separation)
        test_raw: Raw test data (required for calibration separation)
    
    Returns:
        Dictionary with:
        - val_oracle_separation_matrix: [n_val, n_layers] - using true labels
        - test_oracle_separation_matrix: [n_test, n_layers] - using true labels
        - val_calibration_separation_matrix: [n_val, n_layers] - using model predictions (if model_adapter provided)
        - test_calibration_separation_matrix: [n_test, n_layers] - using model predictions (if model_adapter provided)
        - computation_time_s: float
    """
    logger.info("\n" + "="*80)
    logger.info("COMPUTING SEPARATION SCORES (USING GEOMETRIC'S UTILS)")
    logger.info("="*80)
    
    n_layers = len(layer_names)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    # Get model predictions if model_adapter is provided
    compute_calibration_sep = model_adapter is not None and val_raw is not None and test_raw is not None
    if compute_calibration_sep:
        logger.info("Computing BOTH oracle (true labels) and calibration (model predictions) separations")
        val_pred = np.argmax(model_adapter.predict_proba(val_raw), axis=1)
        test_pred = np.argmax(model_adapter.predict_proba(test_raw), axis=1)
        logger.info(f"  Val accuracy: {100.0 * np.mean(val_pred == val_labels):.2f}%")
        logger.info(f"  Test accuracy: {100.0 * np.mean(test_pred == test_labels):.2f}%")
    else:
        logger.info("Computing ORACLE separation only (true labels)")
        val_pred = None
        test_pred = None
    
    # Storage for both types of separation
    val_oracle_sep_per_layer = []
    test_oracle_sep_per_layer = []
    val_calib_sep_per_layer = []
    test_calib_sep_per_layer = []
    
    for i, layer_name in enumerate(layer_names):
        logger.info(f"  Computing separation for layer {i+1}/{n_layers}: {layer_name}")
        
        # Convert to numpy
        train_np = train_features_per_layer[i].numpy() if isinstance(train_features_per_layer[i], torch.Tensor) else train_features_per_layer[i]
        val_np = val_features_per_layer[i].numpy() if isinstance(val_features_per_layer[i], torch.Tensor) else val_features_per_layer[i]
        test_np = test_features_per_layer[i].numpy() if isinstance(test_features_per_layer[i], torch.Tensor) else test_features_per_layer[i]
        
        # 1. Oracle separation (using TRUE labels) - for layer quality assessment
        val_oracle_sep = compute_fast_separation_scores(
            train_features=train_np,
            train_labels=train_labels,
            query_features=val_np,
            query_pred=val_labels,  # TRUE labels
            use_cuda=use_cuda
        )
        test_oracle_sep = compute_fast_separation_scores(
            train_features=train_np,
            train_labels=train_labels,
            query_features=test_np,
            query_pred=test_labels,  # TRUE labels
            use_cuda=use_cuda
        )
        
        val_oracle_sep_per_layer.append(val_oracle_sep)
        test_oracle_sep_per_layer.append(test_oracle_sep)
        
        logger.info(f"    Oracle Val range: [{val_oracle_sep.min():.4f}, {val_oracle_sep.max():.4f}]")
        
        # 2. Calibration separation (using MODEL PREDICTIONS) - for actual calibration
        if compute_calibration_sep:
            val_calib_sep = compute_fast_separation_scores(
                train_features=train_np,
                train_labels=train_labels,
                query_features=val_np,
                query_pred=val_pred,  # MODEL predictions
                use_cuda=use_cuda
            )
            test_calib_sep = compute_fast_separation_scores(
                train_features=train_np,
                train_labels=train_labels,
                query_features=test_np,
                query_pred=test_pred,  # MODEL predictions
                use_cuda=use_cuda
            )
            
            val_calib_sep_per_layer.append(val_calib_sep)
            test_calib_sep_per_layer.append(test_calib_sep)
            
            logger.info(f"    Calib Val range: [{val_calib_sep.min():.4f}, {val_calib_sep.max():.4f}]")
    
    # Stack into matrices
    val_oracle_matrix = np.column_stack(val_oracle_sep_per_layer)
    test_oracle_matrix = np.column_stack(test_oracle_sep_per_layer)
    
    result = {
        'val_oracle_separation_matrix': val_oracle_matrix,
        'test_oracle_separation_matrix': test_oracle_matrix,
        # Keep backward compatible keys (defaulting to oracle)
        'val_separation_matrix': val_oracle_matrix,
        'test_separation_matrix': test_oracle_matrix,
    }
    
    if compute_calibration_sep:
        val_calib_matrix = np.column_stack(val_calib_sep_per_layer)
        test_calib_matrix = np.column_stack(test_calib_sep_per_layer)
        result['val_calibration_separation_matrix'] = val_calib_matrix
        result['test_calibration_separation_matrix'] = test_calib_matrix
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    computation_time = time.perf_counter() - start_time
    result['computation_time_s'] = computation_time
    
    logger.info(f"\nSeparation computation completed in {computation_time:.2f}s")
    logger.info(f"  Val oracle matrix shape: {val_oracle_matrix.shape}")
    logger.info(f"  Test oracle matrix shape: {test_oracle_matrix.shape}")
    if compute_calibration_sep:
        logger.info(f"  Val calibration matrix shape: {val_calib_matrix.shape}")
        logger.info(f"  Test calibration matrix shape: {test_calib_matrix.shape}")
    logger.info("="*80)
    
    return result


def run_geometric_calibration(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    layer_name_or_names: Union[str, List[str]],
    method_name: str,
    batch_size: int = 128,
    precomputed_features: Dict[str, Any] = None,
    precomputed_probs: np.ndarray = None,
    return_model_probs: bool = False,
    return_val_probs: bool = False,
    seed: int = 42,
    pooling_mode: str = 'max',
    target_dimension: int = None,
    scoring_method: str = 'separation',  # Options: 'separation' or 'trust_score'
    ood_raw: np.ndarray = None,  # OOD data for evaluation
    ood_labels: np.ndarray = None,  # OOD labels for ECE calculation
    # OPTIMIZATION: Cached predictions and loaders
    precomputed_test_probs: np.ndarray = None,
    precomputed_ood_probs: np.ndarray = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
    ood_loader: DataLoader = None,
    ood_feature_cache: Dict[str, np.ndarray] = None,  # OPTIMIZATION: Cache for OOD features
    output_dir: str = None,  # Optional output directory for per-sample data saving
    training_method: str = None  # Training method for filename construction
) -> Dict[str, Any]:
    """
    Run geometric calibration on specified layer(s).
    
    Args:
        layer_name_or_names: Single layer name (str) or list of layer names for combination
        method_name: Name for the method (e.g., "Geometric (Layer1)", "Geometric (Combined)")
        precomputed_features: Optional dict with pre-extracted features to avoid recomputation
        precomputed_probs: Optional array of pre-computed logits/probs for apples-to-apples throughput
        return_model_probs: If True, returns computed model probabilities in the result dict
        return_val_probs: If True, returns calibrated probabilities on validation set and the calibrator object
    
    Returns:
        Dictionary with results including ECE, accuracy, timing.
        If return_val_probs=True, also includes 'val_calibrated_probs' and 'calibrator' keys.
    """
    logger.info("\n" + "="*80)
    score_label = "Trust Score" if scoring_method == 'trust_score' else "Separation"
    logger.info(f"RUNNING GEOMETRIC CALIBRATION: {method_name} ({score_label})")
    logger.info("="*80)
    logger.info("Feature preprocessing: SPP+JL compression with dataset-specific ratio (Geometric method)")
    
    # Determine if single layer or multiple layers
    if isinstance(layer_name_or_names, str):
        layer_names = [layer_name_or_names]
        is_combined = False
    else:
        layer_names = layer_name_or_names
        is_combined = True
    
    if not is_combined:
        semantic_name = get_semantic_layer_name(layer_names[0], model_name)
        logger.info(f"Using layer: {layer_names[0]} ({semantic_name})")
    else:
        logger.info(f"Using {len(layer_names)} layers (combined)")
    
    # Initialize compressor variable to None (will be set if needed for OOD)
    compressor = None
    
    # Check if we can use precomputed features
    if precomputed_features is not None:
        logger.info("Using precomputed features (skipping extraction)")
        
        all_layer_names = precomputed_features['layer_names']
        train_features_all = precomputed_features['train_features_per_layer']
        val_features_all = precomputed_features['val_features_per_layer']
        test_features_all = precomputed_features['test_features_per_layer']
        
        # Extract the specific layers we need
        if is_combined:
            # Extract ONLY the requested layers from precomputed features
            # layer_names contains the specific layers to combine (e.g., main_blocks)
            # NOT all_layer_names which contains everything
            # Filter layer_names to only include those that exist in all_layer_names
            available_layer_set = set(all_layer_names)
            filtered_layer_names = [name for name in layer_names if name in available_layer_set]
            if len(filtered_layer_names) != len(layer_names):
                missing = set(layer_names) - available_layer_set
                logger.warning(f"Some requested layers not found in precomputed features: {missing}")
                logger.warning(f"Using {len(filtered_layer_names)} available layers instead of {len(layer_names)} requested")
            if not filtered_layer_names:
                raise ValueError(f"None of the requested layers {layer_names} found in precomputed features. Available layers: {all_layer_names[:10]}...")
            layer_indices = [all_layer_names.index(name) for name in filtered_layer_names]
            layer_names = filtered_layer_names  # Update layer_names to filtered list
            train_features_list = [train_features_all[i] for i in layer_indices]
            val_features_list = [val_features_all[i] for i in layer_indices]
            test_features_list = [test_features_all[i] for i in layer_indices]
            
            logger.info(f"Combining {len(layer_indices)} requested layers from {len(all_layer_names)} available layers")
            for idx, name in zip(layer_indices, layer_names):
                logger.info(f"  Using layer {idx}: {name} (shape: {train_features_all[idx].shape})")
            
            train_features = combine_features(train_features_list)
            val_features = combine_features(val_features_list)
            test_features = combine_features(test_features_list)
            
            # Apply post-concatenation JL compression (consistent with SPP+JL pipeline)
            dataset_lower = dataset_name.lower()
            if dataset_lower == "cifar100":
                post_compression_ratio = 8.0
            elif dataset_lower in ["tiny_imagenet", "tinyimagenet"]:
                post_compression_ratio = 8.0  # Similar to CIFAR-100
            else:
                post_compression_ratio = 4.0  # Default for CIFAR10
            logger.info(f"Applying post-concatenation JL projection: {post_compression_ratio}x")
            logger.info(f"  Before compression: {train_features.shape}")
            
            from utils.compression_utils import SmartCompression
            compressor = SmartCompression(
                method='random_projection',
                compression_ratio=post_compression_ratio,
                random_state=seed
            )
            
            # Fit on train, transform all splits
            train_features_np = train_features.numpy() if torch.is_tensor(train_features) else train_features
            val_features_np = val_features.numpy() if torch.is_tensor(val_features) else val_features
            test_features_np = test_features.numpy() if torch.is_tensor(test_features) else test_features
            
            train_features = torch.from_numpy(compressor(train_features_np, train=True))
            val_features = torch.from_numpy(compressor(val_features_np, train=False))
            test_features = torch.from_numpy(compressor(test_features_np, train=False))
            
            logger.info(f"  After compression: {train_features.shape}")
        else:
            # Single layer - find its index
            layer_idx = all_layer_names.index(layer_names[0])
            train_features = train_features_all[layer_idx]
            val_features = val_features_all[layer_idx]
            test_features = test_features_all[layer_idx]
        
        extract_time = 0.0  # Already included in precomputed
        
    else:
        # Original extraction code
        logger.info(f"\n1. Extracting features from {len(layer_names)} layer(s)...")
        start_extract = time.perf_counter()
        
        # Create dataloaders
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
            batch_size=batch_size,
            shuffle=False
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
            batch_size=batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
            batch_size=batch_size,
            shuffle=False
        )
        
        if is_combined:
            # Use target_dimension if provided, otherwise fall back to compression_ratio
            if target_dimension is not None:
                compression_ratio = None
                logger.info(f"Using fixed target dimension: d={target_dimension}")
            else:
                # Determine compression ratio based on dataset
                dataset_lower = dataset_name.lower()
                if dataset_lower == "cifar100":
                    compression_ratio = 32.0
                elif dataset_lower in ["tiny_imagenet", "tinyimagenet"]:
                    compression_ratio = 32.0  # Similar to CIFAR-100
                else:
                    compression_ratio = 16.0  # Default for CIFAR10
                logger.info(f"Using compression ratio: {compression_ratio}x for {dataset_name}")
            # Extract from multiple layers and combine
            train_features_list = extract_features_from_multiple_layers(
                model, train_loader, layer_names, device, 
                compression_ratio=compression_ratio,
                target_dimension=target_dimension,
                pooling_mode=pooling_mode
            )
            val_features_list = extract_features_from_multiple_layers(
                model, val_loader, layer_names, device, 
                compression_ratio=compression_ratio,
                target_dimension=target_dimension,
                pooling_mode=pooling_mode
            )
            test_features_list = extract_features_from_multiple_layers(
                model, test_loader, layer_names, device, 
                compression_ratio=compression_ratio,
                target_dimension=target_dimension,
                pooling_mode=pooling_mode
            )
            
            train_features = combine_features(train_features_list)
            val_features = combine_features(val_features_list)
            test_features = combine_features(test_features_list)
            
            # Apply post-concatenation JL compression (consistent with SPP+JL pipeline)
            dataset_lower = dataset_name.lower()
            if dataset_lower == "cifar100":
                post_compression_ratio = 8.0
            elif dataset_lower in ["tiny_imagenet", "tinyimagenet"]:
                post_compression_ratio = 8.0  # Similar to CIFAR-100
            else:
                post_compression_ratio = 4.0  # Default for CIFAR10
            logger.info(f"Applying post-concatenation JL projection: {post_compression_ratio}x")
            logger.info(f"  Before compression: {train_features.shape}")
            
            from utils.compression_utils import SmartCompression
            compressor = SmartCompression(
                method='random_projection',
                compression_ratio=post_compression_ratio,
                random_state=seed
            )
            
            # Fit on train, transform all splits
            train_features_np = train_features.numpy() if torch.is_tensor(train_features) else train_features
            val_features_np = val_features.numpy() if torch.is_tensor(val_features) else val_features
            test_features_np = test_features.numpy() if torch.is_tensor(test_features) else test_features
            
            train_features = torch.from_numpy(compressor(train_features_np, train=True))
            val_features = torch.from_numpy(compressor(val_features_np, train=False))
            test_features = torch.from_numpy(compressor(test_features_np, train=False))
            
            logger.info(f"  After compression: {train_features.shape}")
        else:
            # Single layer extraction
            train_features, _ = extract_features_directly(model, train_loader, layer_names[0], device)
            val_features, _ = extract_features_directly(model, val_loader, layer_names[0], device)
            test_features, _ = extract_features_directly(model, test_loader, layer_names[0], device)
        
        extract_time = time.perf_counter() - start_extract
        logger.info(f"Feature extraction time: {extract_time:.2f}s")
    
    logger.info(f"  Train features shape: {train_features.shape}")
    logger.info(f"  Val features shape: {val_features.shape}")
    logger.info(f"  Test features shape: {test_features.shape}")
    
    # Fit geometric calibrator
    logger.info("\n2. Fitting geometric calibrator...")
    start_fit = time.perf_counter()
    
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features.numpy(),
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,  # We're using fixed layer(s)
        device=str(device),
        scoring_method=scoring_method  # Use separation or trust_score
    )
    
    geo_cal.fit(
        X_val_embed=val_features.numpy(),
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size
    )
    
    fit_time = time.perf_counter() - start_fit
    logger.info(f"Geometric fitting time: {fit_time:.2f}s")
    
    # Calibrate validation set if return_val_probs is True
    val_calibrated_probs = None
    if return_val_probs:
        logger.info("\n2.5. Calibrating validation set (for reuse in metrics)...")
        val_calibrated_probs = geo_cal.calibrate_batched(
            X_test_embed=val_features.numpy(),
            X_test_original=val_raw,
            batch_size=batch_size
        )
        logger.info(f"  Validation calibrated probabilities shape: {val_calibrated_probs.shape}")
    
    # Calibrate test set with timing
    logger.info("\n3. Calibrating test set...")
    
    # Reset GPU memory tracking
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    computed_model_probs = None
    
    # ========== FIXED TIMING: Separate forward pass from calibrator-only ==========
    # 1. Pre-compute model probabilities (forward pass) - time separately
    if precomputed_probs is not None:
        logger.info("   Using PRE-COMPUTED probabilities (skipping inference)...")
        model_probs = precomputed_probs
        forward_time = 0.0  # Already computed elsewhere
    else:
        logger.info("   Computing model probabilities (forward pass)...")
        # OPTIMIZATION: Use cached test_probs if available
        if precomputed_test_probs is not None:
            model_probs = precomputed_test_probs
            forward_time = 0.0  # No forward pass needed
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_forward = time.perf_counter()
            
            model_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_time = time.perf_counter() - start_forward
        logger.info(f"   Forward pass time: {forward_time:.4f}s")
    
    # 2. Calibrator-only timing (fair comparison with DAC - NO forward pass)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_calibrate = time.perf_counter()
    
    calibrated_probs = geo_cal.calibrate_batched_precomputed(
        X_test_embed=test_features.numpy(),
        X_test_original=test_raw,
        model_probs=model_probs,
        batch_size=batch_size
    )
    
    # Save per-sample data for reliability diagrams and risk-coverage curves
    if output_dir is not None:
        stability_scores = getattr(geo_cal, 'last_stability_scores', None)
        save_per_sample_data(
            calibrated_probs=calibrated_probs,
            test_labels=test_labels,
            method_name=method_name,
            output_dir=output_dir,
            stability_scores=stability_scores,
            model_name=model_name,
            dataset_name=dataset_name,
            training_method=training_method,
            seed=seed
        )
    
    if return_model_probs:
        computed_model_probs = model_probs
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    calibrator_only_time = time.perf_counter() - start_calibrate
    
    # 3. Calculate aggregate timings
    e2e_calibrate_time = forward_time + calibrator_only_time
    
    # For backward compatibility, calibrate_time_s = calibrator_only_time
    calibrate_time = calibrator_only_time
    
    logger.info(f"   Calibrator-only time: {calibrator_only_time:.4f}s")
    logger.info(f"   E2E time (forward + calibrator): {e2e_calibrate_time:.4f}s")
    # ==============================================================================
    
    # Calculate throughput
    n_test_samples = len(test_labels)
    calibrator_throughput = n_test_samples / calibrator_only_time if calibrator_only_time > 0 else 0.0
    e2e_throughput = n_test_samples / e2e_calibrate_time if e2e_calibrate_time > 0 else 0.0
    throughput = calibrator_throughput  # For backward compatibility
    
    # Memory usage
    peak_memory_mb = 0
    if device.type == 'cuda':
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    logger.info(f"Calibration time: {calibrate_time:.2f}s")
    logger.info(f"Throughput: {throughput:.2f} samples/sec")
    logger.info(f"Peak GPU memory: {peak_memory_mb:.2f} MB")
    
    # Calculate metrics
    logger.info("\n4. Computing metrics...")
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
    brier = calculate_brier_score(calibrated_probs, test_labels)
    top_label_brier = UncertaintyMetrics.calculate_top_label_brier_score(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)
    
    logger.info(f"Geometric Calibration Results:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Adaptive ECE: {adaptive_ece:.6f}")
    logger.info(f"  Brier: {brier:.6f}")
    logger.info(f"  Top-label Brier: {top_label_brier:.6f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    
    # OOD Evaluation
    ood_metrics = {}
    if ood_raw is not None:
        logger.info("\nComputing OOD metrics on SVHN...")
        
        # Extract OOD features using same pipeline as ID data
        ood_loader = DataLoader(
            TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
            batch_size=batch_size, shuffle=False
        )
        
        if precomputed_features is not None:
            # Use same feature extraction logic as ID but for OOD
            if is_combined:
                # Extract the specific layers we need from model
                ood_features_list = extract_features_from_multiple_layers(
                    model, ood_loader, layer_names, device,
                    compression_ratio=None if target_dimension is not None else (32.0 if dataset_name.lower() == "cifar100" or dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"] else 16.0),
                    target_dimension=target_dimension,
                    pooling_mode=pooling_mode
                )
                ood_features = combine_features(ood_features_list)
                
                # Reuse the same compressor that was used for ID features
                if compressor is None:
                    raise RuntimeError("compressor should have been created for combined layers with precomputed features")
                
                # Apply the same compression to OOD features (compressor is already fitted)
                ood_features_np = ood_features.numpy() if torch.is_tensor(ood_features) else ood_features
                ood_features = torch.from_numpy(compressor(ood_features_np, train=False))
            else:
                # Single layer extraction
                ood_features, _ = extract_features_directly(model, ood_loader, layer_names[0], device)
        else:
            # Not using precomputed - extract fresh
            if is_combined:
                # Determine compression ratio
                if target_dimension is not None:
                    ood_compression_ratio = None
                else:
                    dataset_lower = dataset_name.lower()
                    if dataset_lower == "cifar100":
                        ood_compression_ratio = 32.0
                    elif dataset_lower in ["tiny_imagenet", "tinyimagenet"]:
                        ood_compression_ratio = 32.0
                    else:
                        ood_compression_ratio = 16.0
                
                ood_features_list = extract_features_from_multiple_layers(
                    model, ood_loader, layer_names, device,
                    compression_ratio=ood_compression_ratio,
                    target_dimension=target_dimension,
                    pooling_mode=pooling_mode
                )
                ood_features = combine_features(ood_features_list)
                
                # Reuse the same compressor that was used for ID features
                if compressor is None:
                    raise RuntimeError("compressor should have been created for combined layers")
                
                # Apply the same compression to OOD features (compressor is already fitted)
                ood_features_np = ood_features.numpy() if torch.is_tensor(ood_features) else ood_features
                ood_features = torch.from_numpy(compressor(ood_features_np, train=False))
            else:
                # Single layer extraction
                ood_features, _ = extract_features_directly(model, ood_loader, layer_names[0], device)
        
        # Convert to numpy
        ood_features_np = ood_features.numpy() if torch.is_tensor(ood_features) else ood_features
        
        # OPTIMIZATION: Use cached OOD probs if available
        if precomputed_ood_probs is not None:
            ood_model_probs = precomputed_ood_probs
        else:
            ood_model_probs = model_adapter.predict_proba(ood_raw, batch_size=batch_size)
        
        # Calibrate OOD using fitted calibrator
        ood_calibrated_probs = geo_cal.calibrate_batched_precomputed(
            X_test_embed=ood_features_np,
            X_test_original=ood_raw,
            model_probs=ood_model_probs,
            batch_size=batch_size
        )
        
        # Compute metrics
        ood_metrics = compute_ood_metrics(
            calibrated_probs_id=calibrated_probs,
            calibrated_probs_ood=ood_calibrated_probs,
            id_labels=test_labels,
            ood_labels=ood_labels,
        )
        
        logger.info(f"  OOD AUROC: {ood_metrics['ood_auroc']:.4f}")
        logger.info(f"  OOD FPR@95: {ood_metrics['ood_fpr95']:.4f}")
        logger.info(f"  Confidence gap: {ood_metrics['confidence_gap']:.4f} (ID: {ood_metrics['id_confidence_mean']:.4f}, OOD: {ood_metrics['ood_confidence_mean']:.4f})")
    
    results = {
        'method': method_name,
        'ece': float(ece),
        'adaptive_ece': float(adaptive_ece),
        'calibration_mce': float(calibration_mce),
        'brier': float(brier),
        'top_label_brier': float(top_label_brier),
        'accuracy': float(accuracy),
        'extract_time_s': float(extract_time),
        'fit_time_s': float(fit_time),
        'forward_time_s': float(forward_time),
        'calibrator_only_time_s': float(calibrator_only_time),
        'calibrate_time_s': float(calibrate_time),  # Backward compat: = calibrator_only_time
        'e2e_calibrate_time_s': float(e2e_calibrate_time),
        'calibrator_throughput_samples_per_sec': float(calibrator_throughput),
        'e2e_throughput_samples_per_sec': float(e2e_throughput),
        'throughput_samples_per_sec': float(throughput),  # Backward compat: = calibrator_throughput
        'peak_memory_mb': float(peak_memory_mb),
        'layer_names': layer_names if is_combined else layer_names[0],
        'calibrator_params': geo_cal.get_params(),
        **ood_metrics,  # Add OOD metrics
    }
    
    if return_model_probs and computed_model_probs is not None:
        results['model_probs'] = computed_model_probs
    
    if return_val_probs:
        results['val_calibrated_probs'] = val_calibrated_probs
        results['calibrator'] = geo_cal
        
    return results




def extract_and_aggregate_sgc_features(
    model: torch.nn.Module,
    layer_names: List[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    target_dim: int = 256,
    seed: int = 42,
    pooling_mode: str = 'max'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Extract features using the SGC paper pipeline:
    1. SPP only per layer (no per-layer JL)
    2. Independent random projection per layer (different seeds)
    3. Project-then-sum aggregation
    4. L2 normalization
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to extract from
        train_loader, val_loader, test_loader: DataLoaders
        device: torch device
        target_dim: Target dimension d (paper uses 256)
        seed: Base random seed (each layer uses seed + layer_index)
        pooling_mode: 'max' or 'avg' for SPP
        
    Returns:
        train_features, val_features, test_features: L2-normalized aggregated features
        info: Dict with layer contributions and timing
    """
    import math
    import torch.nn.functional as F

    class TorchGaussianRandomProjection:
        """
        Deterministic Gaussian random projection implemented in PyTorch.

        This is intended to match scikit-learn's `GaussianRandomProjection` scaling:
          - `components_ ~ Normal(0, 1 / sqrt(n_components))`
        so that each entry has variance `1 / n_components`.

        For input `X` with shape (N, D_in), projection output is:
          `X_proj = X @ R^T` with `R` shape (D_out, D_in), where `D_out = n_components`.
        """

        def __init__(self, in_dim: int, out_dim: int, seed: int, device: torch.device):
            self.in_dim = int(in_dim)
            self.out_dim = int(out_dim)
            self.seed = int(seed)
            self.device = device

            # sklearn uses std = 1 / sqrt(n_components)
            scale = 1.0 / math.sqrt(self.out_dim)

            # Deterministic generator on the target device (CUDA/CPU)
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)

            # R: (out_dim, in_dim)
            self.R = torch.randn(
                (self.out_dim, self.in_dim),
                device=device,
                dtype=torch.float32,
                generator=gen,
            ) * scale

        def project(self, x: torch.Tensor) -> torch.Tensor:
            # x: (N, in_dim) -> (N, out_dim)
            if x.dtype != torch.float32:
                x = x.float()
            if x.device != self.device:
                x = x.to(self.device, non_blocking=True)
            return F.linear(x, self.R)
    
    logger.info(f"SGC Pipeline: L={len(layer_names)} layers, d={target_dim}")
    logger.info(f"Layers: {layer_names}")
    
    # Get yielded sample counts without iterating loaders (handle sampler-based splits).
    # CIFAR loaders keep a 50k dataset object and use SubsetRandomSampler for 45k/5k,
    # so len(loader.dataset) is not the number of rows extractors will return.
    def _safe_len_loader_samples(loader: Optional[DataLoader]) -> int:
        if loader is None:
            return 0
        sampler = getattr(loader, "sampler", None)
        try:
            return int(len(sampler)) if sampler is not None else 0
        except Exception:
            pass
        ds = getattr(loader, "dataset", None)
        try:
            return int(len(ds)) if ds is not None else 0
        except Exception:
            return 0

    n_train = _safe_len_loader_samples(train_loader)
    n_val = _safe_len_loader_samples(val_loader)
    n_test = _safe_len_loader_samples(test_loader)

    # Initialize accumulators ON DEVICE (GPU) (use empty tensors if loaders are None)
    train_sum = torch.zeros((n_train, target_dim), device=device, dtype=torch.float32) if n_train > 0 else torch.empty((0, target_dim), device=device, dtype=torch.float32)
    val_sum = torch.zeros((n_val, target_dim), device=device, dtype=torch.float32) if n_val > 0 else torch.empty((0, target_dim), device=device, dtype=torch.float32)
    test_sum = torch.zeros((n_test, target_dim), device=device, dtype=torch.float32) if n_test > 0 else torch.empty((0, target_dim), device=device, dtype=torch.float32)
    
    layer_dims = []
    layer_timing = {}

    use_cuda = (device.type == "cuda")
    if use_cuda:
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for i, lname in enumerate(layer_names):
        logger.info(f"Processing layer {i+1}/{len(layer_names)}: {lname}")
        
        # 1. Extract SPP-only features (no JL compression) - handle None loaders
        single_train = None
        single_val = None
        single_test = None

        # Timing: SPP extraction (includes forward + pooling; unchanged behavior)
        if use_cuda:
            torch.cuda.synchronize()
        spp_start = time.perf_counter()

        # Phase 1: Use return_on_device=True to keep features on GPU, avoiding CPU↔GPU transfers
        if train_loader is not None:
            single_train = extract_features_from_multiple_layers(
                model, train_loader, [lname], device, spp_only=True, pooling_mode=pooling_mode, return_on_device=True
            )[0]
        if val_loader is not None:
            single_val = extract_features_from_multiple_layers(
                model, val_loader, [lname], device, spp_only=True, pooling_mode=pooling_mode, return_on_device=True
            )[0]
        if test_loader is not None:
            single_test = extract_features_from_multiple_layers(
                model, test_loader, [lname], device, spp_only=True, pooling_mode=pooling_mode, return_on_device=True
            )[0]

        if use_cuda:
            torch.cuda.synchronize()
        spp_time = time.perf_counter() - spp_start
        
        # Determine layer dimension from first available feature
        # Phase 1: Features are now on GPU, so ensure they're on the correct device
        if single_train is not None:
            layer_dim = single_train.shape[1]
            # Ensure tensor is on device (should already be, but verify)
            if single_train.device != device:
                single_train = single_train.to(device, non_blocking=True)
        elif single_val is not None:
            layer_dim = single_val.shape[1]
            if single_val.device != device:
                single_val = single_val.to(device, non_blocking=True)
        elif single_test is not None:
            layer_dim = single_test.shape[1]
            if single_test.device != device:
                single_test = single_test.to(device, non_blocking=True)
        else:
            raise ValueError("All loaders are None - cannot determine layer dimension")
        
        layer_dims.append(layer_dim)
        logger.info(f"  SPP output dim: {layer_dim}")
        
        # 2. Deterministic Gaussian RP on GPU (independent per layer via seed + i)
        # Phase 1: All projection now runs on GPU via torch, eliminating sklearn CPU bottleneck
        # Note: Gaussian RP is data-independent. Equivalent to sklearn's fit() storing `components_`.
        rp = TorchGaussianRandomProjection(
            in_dim=int(layer_dim),
            out_dim=int(target_dim),
            seed=int(seed + i),
            device=device,
        )

        # Timing: projection + accumulation (GPU)
        if use_cuda:
            torch.cuda.synchronize()
        proj_start = time.perf_counter()

        # Phase 1: Features are already on GPU, projection runs entirely on GPU
        if single_train is not None and n_train > 0:
            train_sum += rp.project(single_train)
        if single_val is not None and n_val > 0:
            val_sum += rp.project(single_val)
        if single_test is not None and n_test > 0:
            test_sum += rp.project(single_test)

        if use_cuda:
            torch.cuda.synchronize()
        proj_time = time.perf_counter() - proj_start

        layer_timing[lname] = {
            "spp_time_s": float(spp_time),
            "projection_time_s": float(proj_time),
            "spp_dim": int(layer_dim),
            "seed": int(seed + i),
        }

        logger.info(f"  Timing: SPP={spp_time:.3f}s | RP+sum={proj_time:.3f}s")
    
    # 5. L2 normalization (only for non-empty arrays)
    logger.info("Applying L2 normalization...")

    if use_cuda:
        torch.cuda.synchronize()
    norm_start = time.perf_counter()

    train_t = F.normalize(train_sum, p=2, dim=1) if n_train > 0 else train_sum
    val_t = F.normalize(val_sum, p=2, dim=1) if n_val > 0 else val_sum
    test_t = F.normalize(test_sum, p=2, dim=1) if n_test > 0 else test_sum

    if use_cuda:
        torch.cuda.synchronize()
    norm_time = time.perf_counter() - norm_start

    total_time = time.perf_counter() - start_time

    # Convert once at the end (keep final outputs as NumPy float32)
    train_features = train_t.detach().cpu().numpy().astype(np.float32, copy=False)
    val_features = val_t.detach().cpu().numpy().astype(np.float32, copy=False)
    test_features = test_t.detach().cpu().numpy().astype(np.float32, copy=False)
    
    # Compute layer contributions
    total_dims = sum(layer_dims)
    layer_contributions = {
        layer_names[i]: {
            "dims": layer_dims[i],
            "percentage": 100.0 * layer_dims[i] / total_dims if total_dims > 0 else 0.0
        }
        for i in range(len(layer_names))
    }
    
    info = {
        "layer_names": layer_names,
        "layer_contributions": layer_contributions,
        "total_spp_dims": total_dims,
        "target_dim": target_dim,
        "extraction_time_s": total_time,
        "normalization_time_s": float(norm_time),
        "per_layer_timing": layer_timing,
    }
    
    logger.info(f"SGC feature extraction complete in {total_time:.2f}s")
    logger.info(f"  Total SPP dims: {total_dims} -> Target dim: {target_dim}")
    
    return train_features, val_features, test_features, info


def run_global_random_calibration(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    num_layers: int = 6,
    target_dim: int = 256,
    batch_size: int = 128,
    seed: int = 42,
    pooling_mode: str = 'max',
    output_file: str = None,  # NEW: Optional output file for early saving
    results_dict: Dict[str, Any] = None,  # NEW: Optional results dict for early saving
    ood_raw: np.ndarray = None,  # OOD data for evaluation
    ood_labels: np.ndarray = None,  # OOD labels for ECE calculation
    scoring_method: str = 'separation',  # Options: 'separation' or 'trust_score'
    normalization_method: str = 'rank',
    precomputed_train_features: np.ndarray = None,  # Optional pre-extracted features
    precomputed_val_features: np.ndarray = None,  # Optional pre-extracted features
    precomputed_test_features: np.ndarray = None,  # Optional pre-extracted features
    precomputed_extraction_info: Dict[str, Any] = None,  # Optional extraction metadata
    precomputed_selected_layers: List[str] = None,  # Optional pre-selected layers
    # OPTIMIZATION: Cached predictions and loaders
    precomputed_test_probs: np.ndarray = None,
    precomputed_ood_probs: np.ndarray = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
    ood_loader: DataLoader = None,
    ood_feature_cache: Dict[str, np.ndarray] = None,  # OPTIMIZATION: Cache for OOD features
    output_dir: str = None  # Optional output directory for per-sample data saving
) -> Dict[str, Any]:
    """
    Run Global Random layer selection calibration using the SGC paper pipeline.
    
    Paper specification:
    - L=6 layers sampled uniformly at random
    - d=256 target dimension
    - Project-then-sum with independent random matrices per layer
    - L2 normalization
    """
    score_label = "Trust Score" if scoring_method == 'trust_score' else "Separation"
    logger.info("\n" + "="*80)
    logger.info(f"RUNNING SGC (L={num_layers}, d={target_dim}) - {score_label}")
    logger.info("="*80)

    # OPTIMIZATION: Use cached train_loader if available, otherwise create new one
    # (needed for OOD evaluation even when using precomputed features)
    if train_loader is None:
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
            batch_size=batch_size, shuffle=False
        )

    # Check if precomputed features are provided
    use_precomputed = (
        precomputed_train_features is not None and
        precomputed_val_features is not None and
        precomputed_test_features is not None and
        precomputed_extraction_info is not None and
        precomputed_selected_layers is not None
    )
    
    if use_precomputed:
        logger.info("Using precomputed features (skipping extraction)")
        train_features = precomputed_train_features
        val_features = precomputed_val_features
        test_features = precomputed_test_features
        extraction_info = precomputed_extraction_info
        selected_layers = precomputed_selected_layers
        logger.info(f"Using pre-selected layers: {selected_layers}")
    else:
        # Discover ALL intermediate layers
        # Determine input shape for layer discovery based on dataset
        is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
        if is_dinov2:
            input_shape = (1, 3, 224, 224)  # DINOv2 requires 224x224
        elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
            input_shape = (1, 3, 64, 64)  # Tiny ImageNet uses 64x64
        else:
            input_shape = (1, 3, 32, 32)  # Default for CIFAR (32x32)
        discovered_layers = normalize_discovered_layers(
            model, discover_model_layers(model, device=device, input_shape=input_shape)
        )
        # Filter out activation and non-feature-bearing layers
        filtered_layers, excluded_layers = filter_non_feature_layers(discovered_layers, model)
        all_layer_names = [d["name"] for d in filtered_layers]
        logger.info(f"Discovered {len(discovered_layers)} total layers, {len(all_layer_names)} feature-bearing layers after filtering")

        # Randomly sample num_layers
        np.random.seed(seed)
        if num_layers > len(all_layer_names):
            logger.warning(f"Requested {num_layers} but only {len(all_layer_names)} available")
            selected_layers = all_layer_names
        else:
            selected_layers = list(np.random.choice(all_layer_names, size=num_layers, replace=False))

        logger.info(f"Selected {len(selected_layers)} layers: {selected_layers}")

        # Save checkpoint after layer selection (before expensive calibration)
        if output_file is not None and results_dict is not None:
            results_dict['global_random_layer_selection'] = {
                'selected_layers': selected_layers,
                'num_layers': num_layers,
                'target_dim': target_dim,
                'total_discovered_layers': len(all_layer_names),
                'seed': seed
            }
            save_results_incrementally(results_dict, output_file)
            logger.info("Checkpoint: layer selection saved (before calibration)")

        # OPTIMIZATION: Use cached loaders if available, otherwise create new ones
        if val_loader is None:
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                batch_size=batch_size, shuffle=False
            )
        if test_loader is None:
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
                batch_size=batch_size, shuffle=False
            )

        # Use the correct SGC pipeline
        logger.info("Extracting SGC features...")
        train_features, val_features, test_features, extraction_info = extract_and_aggregate_sgc_features(
            model=model,
            layer_names=selected_layers,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            target_dim=target_dim,
            seed=seed,
            pooling_mode=pooling_mode
        )

    logger.info(f"Final feature shape: {train_features.shape}")

    # Fit geometric calibrator
    logger.info("Fitting geometric calibrator...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_fit = time.perf_counter()

    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,
        device=str(device),
        scoring_method=scoring_method,  # Use separation or trust_score
        normalization_method=normalization_method
    )

    geo_cal.fit(
        X_val_embed=val_features,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fit_time = time.perf_counter() - start_fit

    # Calibrate test set with fixed timing
    logger.info("Calibrating test set...")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # 1. Pre-compute model probabilities (forward pass)
    # OPTIMIZATION: Use cached test_probs if available
    if precomputed_test_probs is not None:
        model_probs = precomputed_test_probs
        forward_time = 0.0  # No forward pass needed
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_forward = time.perf_counter()
        
        model_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - start_forward

    # 2. Calibrator-only timing (NO forward pass)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_calibrate = time.perf_counter()
    
    calibrated_probs = geo_cal.calibrate_batched_precomputed(
        X_test_embed=test_features,
        X_test_original=test_raw,
        model_probs=model_probs,
        batch_size=batch_size
    )
    
    # Save per-sample data for reliability diagrams and risk-coverage curves
    if output_dir is not None:
        stability_scores = getattr(geo_cal, 'last_stability_scores', None)
        save_per_sample_data(
            calibrated_probs=calibrated_probs,
            test_labels=test_labels,
            method_name='geometric_(combined_1_main_blocks)',
            output_dir=output_dir,
            stability_scores=stability_scores,
            model_name=model_name,
            dataset_name=dataset_name,
            training_method=None,  # training_method not available in this function
            seed=seed
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    calibrator_only_time = time.perf_counter() - start_calibrate

    # Aggregate timings
    e2e_calibrate_time = forward_time + calibrator_only_time
    calibrate_time = calibrator_only_time  # Backward compat
    
    # Calculate throughput
    n_test_samples = len(test_labels)
    calibrator_throughput = n_test_samples / calibrator_only_time if calibrator_only_time > 0 else 0.0
    e2e_throughput = n_test_samples / e2e_calibrate_time if e2e_calibrate_time > 0 else 0.0
    throughput = calibrator_throughput  # Backward compat
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0

    # Metrics
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
    brier = calculate_brier_score(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)

    logger.info(f"\nSGC Results:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Adaptive ECE: {adaptive_ece:.6f}")
    logger.info(f"  Brier: {brier:.6f}")
    logger.info(f"  Accuracy: {accuracy:.4f}%")

    # OOD Evaluation
    ood_metrics = {}
    if ood_raw is not None:
        logger.info("\nComputing OOD metrics on SVHN...")
        
        # OPTIMIZATION: Use cached OOD loader if available
        if ood_loader is None:
            ood_loader = DataLoader(
                TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
                batch_size=batch_size, shuffle=False
            )
        
        # Use same layer selection and compression as ID data
        # NOTE: train_loader is required so the compressor can be fitted with the same random projection
        # The compressor must see training data to fit its random projection matrix before transforming OOD data
        # OPTIMIZATION: Use cached train_loader if available
        if train_loader is None:
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=batch_size, shuffle=False
            )
        
        # OPTIMIZATION: Use OOD feature cache if available
        ood_cache_key = f"sgc_L{num_layers}_d{target_dim}_seed{seed}"
        if ood_feature_cache is not None and ood_cache_key in ood_feature_cache:
            logger.info(f"Using cached OOD features: {ood_cache_key}")
            ood_features = ood_feature_cache[ood_cache_key]
        else:
            _, _, ood_features, _ = extract_and_aggregate_sgc_features(
                model=model,
                layer_names=selected_layers,  # Same layers as ID
                train_loader=train_loader,  # FIX: pass train_loader for compressor fitting
                val_loader=None,
                test_loader=ood_loader,
                device=device,
                target_dim=target_dim,
                seed=seed,  # Same seed ensures same projection matrix as ID
                pooling_mode=pooling_mode
            )
            # Cache the extracted features
            if ood_feature_cache is not None:
                ood_feature_cache[ood_cache_key] = ood_features
        
        # Apply L2 normalization (same as ID)
        # ood_features is already a numpy array and L2-normalized from extract_and_aggregate_sgc_features
        # Normalizing again is idempotent and ensures consistency
        ood_features_np = normalize(ood_features, norm='l2', axis=1)
        
        # Calibrate OOD using fitted calibrator
        ood_calibrated_probs = geo_cal.calibrate_batched(
            X_test_embed=ood_features_np,
            X_test_original=ood_raw,
            batch_size=batch_size,
        )
        
        # Compute metrics (using SVHN labels for ECE calculation)
        ood_metrics = compute_ood_metrics(
            calibrated_probs_id=calibrated_probs,
            calibrated_probs_ood=ood_calibrated_probs,
            id_labels=test_labels,
            ood_labels=ood_labels,  # Use SVHN labels for ECE calculation
        )
        
        logger.info(f"  OOD AUROC: {ood_metrics['ood_auroc']:.4f}")
        logger.info(f"  OOD FPR@95: {ood_metrics['ood_fpr95']:.4f}")
        logger.info(f"  Confidence gap: {ood_metrics['confidence_gap']:.4f} (ID: {ood_metrics['id_confidence_mean']:.4f}, OOD: {ood_metrics['ood_confidence_mean']:.4f})")

    return {
        'method': f'SGC (L={num_layers}, d={target_dim})',
        'ece': float(ece),
        'adaptive_ece': float(adaptive_ece),
        'calibration_mce': float(calibration_mce),
        'brier': float(brier),
        'accuracy': float(accuracy),
        'num_layers': num_layers,
        'target_dim': target_dim,
        'selected_layers': selected_layers,
        'layer_contributions': extraction_info['layer_contributions'],
        'total_spp_dims': extraction_info['total_spp_dims'],
        'extraction_time_s': extraction_info['extraction_time_s'],
        'fit_time_s': float(fit_time),
        'forward_time_s': float(forward_time),
        'calibrator_only_time_s': float(calibrator_only_time),
        'calibrate_time_s': float(calibrate_time),  # Backward compat: = calibrator_only_time
        'e2e_calibrate_time_s': float(e2e_calibrate_time),
        'calibrator_throughput_samples_per_sec': float(calibrator_throughput),
        'e2e_throughput_samples_per_sec': float(e2e_throughput),
        'throughput_samples_per_sec': float(throughput),  # Backward compat: = calibrator_throughput
        'peak_memory_mb': float(peak_memory_mb),
        'calibrator_params': geo_cal.get_params(),
        **ood_metrics,  # Add OOD metrics
    }


def run_last_layer_only_baseline(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    target_dim: int = 256,
    batch_size: int = 128,
    seed: int = 42,
    pooling_mode: str = 'max',
    precomputed_features: Dict[str, Any] = None,
    ood_raw: np.ndarray = None,
    ood_labels: np.ndarray = None,
    scoring_method: str = 'separation',  # Options: 'separation' or 'trust_score'
    # OPTIMIZATION: Cached predictions and loaders
    precomputed_test_probs: np.ndarray = None,
    precomputed_ood_probs: np.ndarray = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
    ood_loader: DataLoader = None,
    ood_feature_cache: Dict[str, np.ndarray] = None,  # OPTIMIZATION: Cache for OOD features
    output_dir: str = None,  # Optional output directory for per-sample data saving
    training_method: str = None  # Training method for filename construction
) -> Dict[str, Any]:
    """
    Run baseline experiment using ONLY the penultimate layer with SGC's geometric scoring.
    
    This addresses the reviewer concern: "Does random layer sampling add value over 
    simply using the last layer?" This experiment shows whether multi-layer aggregation 
    provides meaningful improvement over using just the final feature layer.
    
    Uses the same SPP+JL compression pipeline as SGC for fair comparison.
    
    Args:
        model: PyTorch model
        model_name: Name of the model architecture
        dataset_name: Name of the dataset
        model_adapter: Model adapter for predictions
        train_raw, train_labels: Training data
        val_raw, val_labels: Validation data
        test_raw, test_labels: Test data
        device: Device to run on
        target_dim: Target dimension after compression (default: 256, same as SGC)
        batch_size: Batch size for processing
        seed: Random seed for reproducibility
        pooling_mode: SPP pooling mode ('max', 'avg', etc.)
        precomputed_features: Optional pre-extracted features to avoid recomputation
        ood_raw, ood_labels: Optional OOD data for evaluation
        scoring_method: Scoring method to use ('separation' or 'trust_score', default: 'separation')
        
    Returns:
        Dictionary with results including ECE, accuracy, timing, and layer information
    """
    logger.info("\n" + "="*80)
    logger.info("LAST LAYER ONLY BASELINE (Reviewer Request)")
    logger.info("="*80)
    logger.info("Testing if multi-layer aggregation adds value over single penultimate layer")
    
    # Discover all layers to identify penultimate layer
    is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
    if is_dinov2:
        input_shape = (1, 3, 224, 224)
    elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
        input_shape = (1, 3, 64, 64)
    else:
        input_shape = (1, 3, 32, 32)
    
    discovered_layers = normalize_discovered_layers(
        model, discover_model_layers(model, device=device, input_shape=input_shape)
    )
    filtered_layers, excluded_layers = filter_non_feature_layers(discovered_layers, model)
    all_layer_names = [d["name"] for d in filtered_layers]
    
    # Get penultimate layer
    penultimate_layer = get_penultimate_layer_name(model_name, all_layer_names)
    semantic_name = get_semantic_layer_name(penultimate_layer, model_name)
    
    logger.info(f"Identified penultimate layer: {penultimate_layer} ({semantic_name})")
    logger.info(f"Total available layers: {len(all_layer_names)}")
    
    # Create dataloaders
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
    
    # Extract features from penultimate layer only using SGC pipeline
    logger.info(f"\nExtracting features from penultimate layer: {penultimate_layer}")
    start_extract = time.perf_counter()
    
    # Use the same SGC pipeline (SPP + JL projection) but for single layer
    train_features, val_features, test_features, extraction_info = extract_and_aggregate_sgc_features(
        model=model,
        layer_names=[penultimate_layer],  # Single layer only
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        target_dim=target_dim,
        seed=seed,
        pooling_mode=pooling_mode
    )
    
    extract_time = time.perf_counter() - start_extract
    logger.info(f"Feature extraction time: {extract_time:.2f}s")
    logger.info(f"Final feature shape: {train_features.shape}")
    
    # Fit geometric calibrator (same as SGC)
    logger.info("\nFitting geometric calibrator...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_fit = time.perf_counter()
    
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,
        device=str(device),
        scoring_method=scoring_method  # Use separation or trust_score
    )
    
    geo_cal.fit(
        X_val_embed=val_features,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fit_time = time.perf_counter() - start_fit
    
    # Calibrate test set
    logger.info("Calibrating test set...")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Pre-compute model probabilities
    # OPTIMIZATION: Use cached test_probs if available
    if precomputed_test_probs is not None:
        model_probs = precomputed_test_probs
        forward_time = 0.0  # No forward pass needed
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_forward = time.perf_counter()
        
        model_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - start_forward
    
    # Calibrator-only timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_calibrate = time.perf_counter()
    
    calibrated_probs = geo_cal.calibrate_batched_precomputed(
        X_test_embed=test_features,
        X_test_original=test_raw,
        model_probs=model_probs,
        batch_size=batch_size
    )
    
    # Save per-sample data for reliability diagrams and risk-coverage curves
    if output_dir is not None:
        stability_scores = getattr(geo_cal, 'last_stability_scores', None)
        save_per_sample_data(
            calibrated_probs=calibrated_probs,
            test_labels=test_labels,
            method_name='geometric_(last_layer_only)',
            output_dir=output_dir,
            stability_scores=stability_scores,
            model_name=model_name,
            dataset_name=dataset_name,
            training_method=training_method,
            seed=seed
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    calibrator_only_time = time.perf_counter() - start_calibrate
    
    e2e_calibrate_time = forward_time + calibrator_only_time
    calibrate_time = calibrator_only_time
    
    # Calculate throughput
    n_test_samples = len(test_labels)
    calibrator_throughput = n_test_samples / calibrator_only_time if calibrator_only_time > 0 else 0.0
    e2e_throughput = n_test_samples / e2e_calibrate_time if e2e_calibrate_time > 0 else 0.0
    throughput = calibrator_throughput
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0
    
    # Metrics
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
    brier = calculate_brier_score(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)
    
    logger.info(f"\nLast Layer Only Baseline Results:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Adaptive ECE: {adaptive_ece:.6f}")
    logger.info(f"  Brier: {brier:.6f}")
    logger.info(f"  Accuracy: {accuracy:.4f}%")
    logger.info(f"  Layer used: {penultimate_layer} ({semantic_name})")
    
    # OOD Evaluation
    ood_metrics = {}
    if ood_raw is not None:
        logger.info("\nComputing OOD metrics on SVHN...")
        
        ood_loader = DataLoader(
            TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
            batch_size=batch_size, shuffle=False
        )
        
        # Extract OOD features using same pipeline
        _, _, ood_features, _ = extract_and_aggregate_sgc_features(
            model=model,
            layer_names=[penultimate_layer],
            train_loader=train_loader,
            val_loader=None,
            test_loader=ood_loader,
            device=device,
            target_dim=target_dim,
            seed=seed,
            pooling_mode=pooling_mode
        )
        
        ood_features_np = normalize(ood_features, norm='l2', axis=1)
        
        # Calibrate OOD
        ood_calibrated_probs = geo_cal.calibrate_batched(
            X_test_embed=ood_features_np,
            X_test_original=ood_raw,
            batch_size=batch_size,
        )
        
        ood_metrics = compute_ood_metrics(
            calibrated_probs_id=calibrated_probs,
            calibrated_probs_ood=ood_calibrated_probs,
            id_labels=test_labels,
            ood_labels=ood_labels,
        )
        
        logger.info(f"  OOD AUROC: {ood_metrics['ood_auroc']:.4f}")
        logger.info(f"  OOD FPR@95: {ood_metrics['ood_fpr95']:.4f}")
        logger.info(f"  Confidence gap: {ood_metrics['confidence_gap']:.4f}")
    
    return {
        'method': 'Geometric (Last Layer Only)',
        'layer_name': penultimate_layer,
        'semantic_layer_name': semantic_name,
        'ece': float(ece),
        'adaptive_ece': float(adaptive_ece),
        'calibration_mce': float(calibration_mce),
        'brier': float(brier),
        'accuracy': float(accuracy),
        'target_dim': target_dim,
        'extraction_time_s': extraction_info.get('extraction_time_s', extract_time),
        'fit_time_s': float(fit_time),
        'forward_time_s': float(forward_time),
        'calibrator_only_time_s': float(calibrator_only_time),
        'calibrate_time_s': float(calibrate_time),
        'e2e_calibrate_time_s': float(e2e_calibrate_time),
        'calibrator_throughput_samples_per_sec': float(calibrator_throughput),
        'e2e_throughput_samples_per_sec': float(e2e_throughput),
        'throughput_samples_per_sec': float(throughput),
        'peak_memory_mb': float(peak_memory_mb),
        'calibrator_params': geo_cal.get_params(),
        'comparison_note': 'Single penultimate layer vs SGC multi-layer aggregation',
        **ood_metrics,
    }


def run_sgc_with_dac_layers(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    target_dim: int = 256,
    batch_size: int = 128,
    seed: int = 42,
    pooling_mode: str = 'max',
    output_file: str = None,  # Optional output file for early saving
    results_dict: Dict[str, Any] = None,  # Optional results dict for early saving
    scoring_method: str = 'separation',  # Options: 'separation' or 'trust_score'
    precomputed_train_features: np.ndarray = None,  # Optional pre-extracted features
    precomputed_val_features: np.ndarray = None,  # Optional pre-extracted features
    precomputed_test_features: np.ndarray = None,  # Optional pre-extracted features
    precomputed_extraction_info: Dict[str, Any] = None,  # Optional extraction metadata
    ood_raw: np.ndarray = None,  # OOD data for evaluation
    ood_labels: np.ndarray = None,  # OOD labels for ECE calculation
    # OPTIMIZATION: Cached predictions and loaders
    precomputed_test_probs: np.ndarray = None,
    precomputed_ood_probs: np.ndarray = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
    ood_loader: DataLoader = None,
    ood_feature_cache: Dict[str, np.ndarray] = None,  # OPTIMIZATION: Cache for OOD features
    output_dir: str = None,  # Optional output directory for per-sample data saving
    training_method: str = None  # Training method for filename construction
) -> Dict[str, Any]:
    """
    SGC using DAC's layer selection instead of random selection.
    
    Pipeline is IDENTICAL to run_global_random_calibration:
    - SPP only per layer (no per-layer JL)
    - Independent random projection per layer
    - Project-then-sum aggregation
    - L2 normalization
    - Geometric calibration
    
    The ONLY difference: uses get_dac_target_layers() instead of random selection.
    
    Args:
        model: PyTorch model
        model_name: Model name (e.g., 'resnet18', 'densenet121')
        dataset_name: Dataset name
        model_adapter: Model adapter for geometric calibration
        train_raw, train_labels: Training data
        val_raw, val_labels: Validation data
        test_raw, test_labels: Test data
        device: torch device
        target_dim: Target dimension d (default 256)
        batch_size: Batch size for data loading
        seed: Random seed
        pooling_mode: 'max' or 'avg' for SPP
        output_file: Optional output file for incremental saving
        results_dict: Optional results dict for incremental saving
        
    Returns:
        Dict with calibration results (same format as run_global_random_calibration)
    """
    score_label = "Trust Score" if scoring_method == 'trust_score' else "Separation"
    logger.info("\n" + "="*80)
    logger.info(f"RUNNING SGC WITH DAC LAYER SELECTION (d={target_dim}) - {score_label}")
    logger.info("="*80)

    # Get DAC's layer selection (instead of random)
    try:
        selected_layers = get_dac_target_layers(model_name, model)
        logger.info(f"Using DAC layer selection: {selected_layers}")
        
        if not selected_layers or len(selected_layers) == 0:
            raise ValueError("DAC layer selection returned empty list")
            
    except Exception as e:
        logger.error(f"Failed to get DAC target layers: {e}")
        raise ValueError(f"Could not determine DAC layers for {model_name}: {e}")

    num_layers = len(selected_layers)
    logger.info(f"Selected {num_layers} layers: {selected_layers}")

    # Save checkpoint after layer selection (before expensive calibration)
    if output_file is not None and results_dict is not None:
        results_dict['sgc_with_dac_layers'] = {
            'selected_layers': selected_layers,
            'num_layers': num_layers,
            'target_dim': target_dim,
            'seed': seed
        }
        save_results_incrementally(results_dict, output_file)
        logger.info("Checkpoint: layer selection saved (before calibration)")

    # Check if precomputed features are provided
    use_precomputed = (
        precomputed_train_features is not None and
        precomputed_val_features is not None and
        precomputed_test_features is not None and
        precomputed_extraction_info is not None
    )
    
    if use_precomputed:
        logger.info("Using precomputed features (skipping extraction)")
        train_features = precomputed_train_features
        val_features = precomputed_val_features
        test_features = precomputed_test_features
        extraction_info = precomputed_extraction_info
    else:
        # OPTIMIZATION: Use cached loaders if available, otherwise create new ones
        if train_loader is None:
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=batch_size, shuffle=False
            )
        if val_loader is None:
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                batch_size=batch_size, shuffle=False
            )
        if test_loader is None:
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
                batch_size=batch_size, shuffle=False
            )

        # Use the standard SGC pipeline (same as run_global_random_calibration)
        logger.info("Extracting SGC features...")
        train_features, val_features, test_features, extraction_info = extract_and_aggregate_sgc_features(
            model=model,
            layer_names=selected_layers,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            target_dim=target_dim,
            seed=seed,
            pooling_mode=pooling_mode
        )

    logger.info(f"Final feature shape: {train_features.shape}")

    # Fit geometric calibrator (same as run_global_random_calibration)
    logger.info("Fitting geometric calibrator...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_fit = time.perf_counter()

    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,
        device=str(device),
        scoring_method=scoring_method  # Use separation or trust_score
    )

    geo_cal.fit(
        X_val_embed=val_features,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fit_time = time.perf_counter() - start_fit

    # Calibrate test set with fixed timing
    logger.info("Calibrating test set...")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # 1. Pre-compute model probabilities (forward pass)
    # OPTIMIZATION: Use cached test_probs if available
    if precomputed_test_probs is not None:
        model_probs = precomputed_test_probs
        forward_time = 0.0  # No forward pass needed
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_forward = time.perf_counter()
        
        model_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - start_forward

    # 2. Calibrator-only timing (NO forward pass)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_calibrate = time.perf_counter()
    
    calibrated_probs = geo_cal.calibrate_batched_precomputed(
        X_test_embed=test_features,
        X_test_original=test_raw,
        model_probs=model_probs,
        batch_size=batch_size
    )
    
    # Save per-sample data for reliability diagrams and risk-coverage curves
    if output_dir is not None:
        stability_scores = getattr(geo_cal, 'last_stability_scores', None)
        save_per_sample_data(
            calibrated_probs=calibrated_probs,
            test_labels=test_labels,
            method_name='sgc_dac_layers',
            output_dir=output_dir,
            stability_scores=stability_scores,
            model_name=model_name,
            dataset_name=dataset_name,
            training_method=training_method,
            seed=seed
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    calibrator_only_time = time.perf_counter() - start_calibrate

    # Aggregate timings
    e2e_calibrate_time = forward_time + calibrator_only_time
    calibrate_time = calibrator_only_time  # Backward compat
    
    # Calculate throughput
    n_test_samples = len(test_labels)
    calibrator_throughput = n_test_samples / calibrator_only_time if calibrator_only_time > 0 else 0.0
    e2e_throughput = n_test_samples / e2e_calibrate_time if e2e_calibrate_time > 0 else 0.0
    throughput = calibrator_throughput  # Backward compat
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0

    # Metrics
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
    brier = calculate_brier_score(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)
    
    logger.info(f"\nSGC (DAC Layers) Results:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Adaptive ECE: {adaptive_ece:.6f}")
    logger.info(f"  Brier: {brier:.6f}")
    logger.info(f"  Accuracy: {accuracy:.4f}%")

    # OOD Evaluation
    ood_metrics = {}
    if ood_raw is not None:
        logger.info("\nComputing OOD metrics on SVHN...")
        
        # OPTIMIZATION: Use cached OOD loader if available
        if ood_loader is None:
            ood_loader = DataLoader(
                TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
                batch_size=batch_size, shuffle=False
            )
        
        # Use same layer selection and compression as ID data
        # NOTE: train_loader is required so the compressor can be fitted with the same random projection
        # OPTIMIZATION: Use cached train_loader if available
        if train_loader is None:
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=batch_size, shuffle=False
            )
        
        # OPTIMIZATION: Use OOD feature cache if available
        ood_cache_key = f"sgc_dac_L{len(selected_layers)}_d{target_dim}_seed{seed}"
        if ood_feature_cache is not None and ood_cache_key in ood_feature_cache:
            logger.info(f"Using cached OOD features: {ood_cache_key}")
            ood_features = ood_feature_cache[ood_cache_key]
        else:
            _, _, ood_features, _ = extract_and_aggregate_sgc_features(
                model=model,
                layer_names=selected_layers,  # Same layers as ID
                train_loader=train_loader,  # Pass train_loader for compressor fitting
                val_loader=None,
                test_loader=ood_loader,
                device=device,
                target_dim=target_dim,
                seed=seed,  # Same seed ensures same projection matrix as ID
                pooling_mode=pooling_mode
            )
            # Cache the extracted features
            if ood_feature_cache is not None:
                ood_feature_cache[ood_cache_key] = ood_features
        
        # Apply L2 normalization (same as ID)
        from sklearn.preprocessing import normalize
        ood_features_np = normalize(ood_features, norm='l2', axis=1)
        
        # OPTIMIZATION: Use cached OOD probs if available
        if precomputed_ood_probs is not None:
            ood_model_probs = precomputed_ood_probs
        else:
            ood_model_probs = model_adapter.predict_proba(ood_raw, batch_size=batch_size)
        
        # Calibrate OOD using fitted calibrator
        ood_calibrated_probs = geo_cal.calibrate_batched_precomputed(
            X_test_embed=ood_features_np,
            X_test_original=ood_raw,
            model_probs=ood_model_probs,
            batch_size=batch_size
        )
        
        # Compute metrics
        ood_metrics = compute_ood_metrics(
            calibrated_probs_id=calibrated_probs,
            calibrated_probs_ood=ood_calibrated_probs,
            id_labels=test_labels,
            ood_labels=ood_labels,
        )
        
        logger.info(f"  OOD AUROC: {ood_metrics['ood_auroc']:.4f}")
        logger.info(f"  OOD FPR@95: {ood_metrics['ood_fpr95']:.4f}")
        logger.info(f"  Confidence gap: {ood_metrics['confidence_gap']:.4f} (ID: {ood_metrics['id_confidence_mean']:.4f}, OOD: {ood_metrics['ood_confidence_mean']:.4f})")

    return {
        'method': f'SGC (DAC Layers, L={num_layers}, d={target_dim})',
        'ece': float(ece),
        'adaptive_ece': float(adaptive_ece),
        'calibration_mce': float(calibration_mce),
        'brier': float(brier),
        'accuracy': float(accuracy),
        'num_layers': num_layers,
        'target_dim': target_dim,
        'selected_layers': selected_layers,
        'layer_contributions': extraction_info['layer_contributions'],
        'total_spp_dims': extraction_info['total_spp_dims'],
        'extraction_time_s': extraction_info['extraction_time_s'],
        'fit_time_s': float(fit_time),
        'forward_time_s': float(forward_time),
        'calibrator_only_time_s': float(calibrator_only_time),
        'calibrate_time_s': float(calibrate_time),  # Backward compat: = calibrator_only_time
        'e2e_calibrate_time_s': float(e2e_calibrate_time),
        'calibrator_throughput_samples_per_sec': float(calibrator_throughput),
        'e2e_throughput_samples_per_sec': float(e2e_throughput),
        'throughput_samples_per_sec': float(throughput),  # Backward compat: = calibrator_throughput
        'peak_memory_mb': float(peak_memory_mb),
        'calibrator_params': geo_cal.get_params(),
        **ood_metrics,  # Add OOD metrics
    }


def run_sgc_with_tulip_layers(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    target_dim: int = 256,
    batch_size: int = 128,
    seed: int = 42,
    pooling_mode: str = 'max',
    output_file: str = None,  # Optional output file for early saving
    results_dict: Dict[str, Any] = None,  # Optional results dict for early saving
    scoring_method: str = 'separation',  # Options: 'separation' or 'trust_score'
    precomputed_train_features: np.ndarray = None,  # Optional pre-extracted features
    precomputed_val_features: np.ndarray = None,  # Optional pre-extracted features
    precomputed_test_features: np.ndarray = None,  # Optional pre-extracted features
    precomputed_extraction_info: Dict[str, Any] = None,  # Optional extraction metadata
    ood_raw: np.ndarray = None,  # OOD data for evaluation
    ood_labels: np.ndarray = None,  # OOD labels for ECE calculation
    # OPTIMIZATION: Cached predictions and loaders
    precomputed_test_probs: np.ndarray = None,
    precomputed_ood_probs: np.ndarray = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
    ood_loader: DataLoader = None,
    ood_feature_cache: Dict[str, np.ndarray] = None,  # OPTIMIZATION: Cache for OOD features
    output_dir: str = None,  # Optional output directory for per-sample data saving
    training_method: str = None  # Training method for filename construction
) -> Dict[str, Any]:
    """
    SGC using TULIP's layer selection instead of random selection.
    
    Pipeline is IDENTICAL to run_global_random_calibration:
    - SPP only per layer (no per-layer JL)
    - Independent random projection per layer
    - Project-then-sum aggregation
    - L2 normalization
    - Geometric calibration
    
    The ONLY difference: uses select_tulip_layers() instead of random selection.
    
    Args:
        model: PyTorch model
        model_name: Model name (e.g., 'resnet18', 'resnet50')
        dataset_name: Dataset name
        model_adapter: Model adapter for geometric calibration
        train_raw, train_labels: Training data
        val_raw, val_labels: Validation data
        test_raw, test_labels: Test data
        device: torch device
        target_dim: Target dimension d (default 256)
        batch_size: Batch size for data loading
        seed: Random seed
        pooling_mode: 'max' or 'avg' for SPP
        output_file: Optional output file for incremental saving
        results_dict: Optional results dict for incremental saving
        scoring_method: 'separation' or 'trust_score'
        
    Returns:
        Dict with calibration results (same format as run_global_random_calibration)
    """
    score_label = "Trust Score" if scoring_method == 'trust_score' else "Separation"
    logger.info("\n" + "="*80)
    logger.info(f"RUNNING SGC WITH TULIP LAYER SELECTION (d={target_dim}) - {score_label}")
    logger.info("="*80)

    # Get TULIP's layer selection (instead of random)
    try:
        selected_layers = select_tulip_layers(model, model_name)
        logger.info(f"Using TULIP layer selection: {selected_layers}")
        
        if not selected_layers or len(selected_layers) == 0:
            raise ValueError("TULIP layer selection returned empty list")
            
    except Exception as e:
        logger.error(f"Failed to get TULIP target layers: {e}")
        raise ValueError(f"Could not determine TULIP layers for {model_name}: {e}")

    num_layers = len(selected_layers)
    logger.info(f"Selected {num_layers} layers: {selected_layers}")

    # Save checkpoint after layer selection (before expensive calibration)
    if output_file is not None and results_dict is not None:
        results_dict['sgc_with_tulip_layers'] = {
            'selected_layers': selected_layers,
            'num_layers': num_layers,
            'target_dim': target_dim,
            'seed': seed
        }
        save_results_incrementally(results_dict, output_file)
        logger.info("Checkpoint: layer selection saved (before calibration)")

    # Check if precomputed features are provided
    use_precomputed = (
        precomputed_train_features is not None and
        precomputed_val_features is not None and
        precomputed_test_features is not None and
        precomputed_extraction_info is not None
    )
    
    if use_precomputed:
        logger.info("Using precomputed features (skipping extraction)")
        train_features = precomputed_train_features
        val_features = precomputed_val_features
        test_features = precomputed_test_features
        extraction_info = precomputed_extraction_info
    else:
        # OPTIMIZATION: Use cached loaders if available, otherwise create new ones
        if train_loader is None:
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=batch_size, shuffle=False
            )
        if val_loader is None:
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                batch_size=batch_size, shuffle=False
            )
        if test_loader is None:
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
                batch_size=batch_size, shuffle=False
            )

        # Use the standard SGC pipeline (same as run_global_random_calibration)
        logger.info("Extracting SGC features...")
        train_features, val_features, test_features, extraction_info = extract_and_aggregate_sgc_features(
            model=model,
            layer_names=selected_layers,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            target_dim=target_dim,
            seed=seed,
            pooling_mode=pooling_mode
        )

    logger.info(f"Final feature shape: {train_features.shape}")

    # Fit geometric calibrator (same as run_global_random_calibration)
    logger.info("Fitting geometric calibrator...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_fit = time.perf_counter()

    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,
        device=str(device),
        scoring_method=scoring_method  # Use separation or trust_score
    )

    geo_cal.fit(
        X_val_embed=val_features,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fit_time = time.perf_counter() - start_fit

    # Calibrate test set with fixed timing
    logger.info("Calibrating test set...")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # 1. Pre-compute model probabilities (forward pass)
    # OPTIMIZATION: Use cached test_probs if available
    if precomputed_test_probs is not None:
        model_probs = precomputed_test_probs
        forward_time = 0.0  # No forward pass needed
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_forward = time.perf_counter()
        
        model_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - start_forward

    # 2. Calibrator-only timing (NO forward pass)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_calibrate = time.perf_counter()
    
    calibrated_probs = geo_cal.calibrate_batched_precomputed(
        X_test_embed=test_features,
        X_test_original=test_raw,
        model_probs=model_probs,
        batch_size=batch_size
    )
    
    # Save per-sample data for reliability diagrams and risk-coverage curves
    if output_dir is not None:
        stability_scores = getattr(geo_cal, 'last_stability_scores', None)
        save_per_sample_data(
            calibrated_probs=calibrated_probs,
            test_labels=test_labels,
            method_name='sgc_tulip_layers',
            output_dir=output_dir,
            stability_scores=stability_scores,
            model_name=model_name,
            dataset_name=dataset_name,
            training_method=training_method,
            seed=seed
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    calibrator_only_time = time.perf_counter() - start_calibrate

    # Aggregate timings
    e2e_calibrate_time = forward_time + calibrator_only_time
    calibrate_time = calibrator_only_time  # Backward compat
    
    # Calculate throughput
    n_test_samples = len(test_labels)
    calibrator_throughput = n_test_samples / calibrator_only_time if calibrator_only_time > 0 else 0.0
    e2e_throughput = n_test_samples / e2e_calibrate_time if e2e_calibrate_time > 0 else 0.0
    throughput = calibrator_throughput  # Backward compat
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0

    # Metrics
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
    brier = calculate_brier_score(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)
    
    logger.info(f"\nSGC (TULIP Layers) Results:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Adaptive ECE: {adaptive_ece:.6f}")
    logger.info(f"  Brier: {brier:.6f}")
    logger.info(f"  Accuracy: {accuracy:.4f}%")

    # OOD Evaluation
    ood_metrics = {}
    if ood_raw is not None:
        logger.info("\nComputing OOD metrics on SVHN...")
        
        # OPTIMIZATION: Use cached OOD loader if available
        if ood_loader is None:
            ood_loader = DataLoader(
                TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
                batch_size=batch_size, shuffle=False
            )
        
        # Use same layer selection and compression as ID data
        # NOTE: train_loader is required so the compressor can be fitted with the same random projection
        # OPTIMIZATION: Use cached train_loader if available
        if train_loader is None:
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=batch_size, shuffle=False
            )
        
        # OPTIMIZATION: Use OOD feature cache if available
        ood_cache_key = f"sgc_tulip_L{len(selected_layers)}_d{target_dim}_seed{seed}"
        if ood_feature_cache is not None and ood_cache_key in ood_feature_cache:
            logger.info(f"Using cached OOD features: {ood_cache_key}")
            ood_features = ood_feature_cache[ood_cache_key]
        else:
            _, _, ood_features, _ = extract_and_aggregate_sgc_features(
                model=model,
                layer_names=selected_layers,  # Same layers as ID
                train_loader=train_loader,  # Pass train_loader for compressor fitting
                val_loader=None,
                test_loader=ood_loader,
                device=device,
                target_dim=target_dim,
                seed=seed,  # Same seed ensures same projection matrix as ID
                pooling_mode=pooling_mode
            )
            # Cache the extracted features
            if ood_feature_cache is not None:
                ood_feature_cache[ood_cache_key] = ood_features
        
        # Apply L2 normalization (same as ID)
        from sklearn.preprocessing import normalize
        ood_features_np = normalize(ood_features, norm='l2', axis=1)
        
        # OPTIMIZATION: Use cached OOD probs if available
        if precomputed_ood_probs is not None:
            ood_model_probs = precomputed_ood_probs
        else:
            ood_model_probs = model_adapter.predict_proba(ood_raw, batch_size=batch_size)
        
        # Calibrate OOD using fitted calibrator
        ood_calibrated_probs = geo_cal.calibrate_batched_precomputed(
            X_test_embed=ood_features_np,
            X_test_original=ood_raw,
            model_probs=ood_model_probs,
            batch_size=batch_size
        )
        
        # Compute metrics
        ood_metrics = compute_ood_metrics(
            calibrated_probs_id=calibrated_probs,
            calibrated_probs_ood=ood_calibrated_probs,
            id_labels=test_labels,
            ood_labels=ood_labels,
        )
        
        logger.info(f"  OOD AUROC: {ood_metrics['ood_auroc']:.4f}")
        logger.info(f"  OOD FPR@95: {ood_metrics['ood_fpr95']:.4f}")
        logger.info(f"  Confidence gap: {ood_metrics['confidence_gap']:.4f} (ID: {ood_metrics['id_confidence_mean']:.4f}, OOD: {ood_metrics['ood_confidence_mean']:.4f})")

    return {
        'method': f'SGC (TULIP Layers, L={num_layers}, d={target_dim})',
        'ece': float(ece),
        'adaptive_ece': float(adaptive_ece),
        'calibration_mce': float(calibration_mce),
        'brier': float(brier),
        'accuracy': float(accuracy),
        'num_layers': num_layers,
        'target_dim': target_dim,
        'selected_layers': selected_layers,
        'layer_selection': 'tulip',
        'layer_contributions': extraction_info['layer_contributions'],
        'total_spp_dims': extraction_info['total_spp_dims'],
        'extraction_time_s': extraction_info['extraction_time_s'],
        'fit_time_s': float(fit_time),
        'forward_time_s': float(forward_time),
        'calibrator_only_time_s': float(calibrator_only_time),
        'calibrate_time_s': float(calibrate_time),  # Backward compat: = calibrator_only_time
        'e2e_calibrate_time_s': float(e2e_calibrate_time),
        'calibrator_throughput_samples_per_sec': float(calibrator_throughput),
        'e2e_throughput_samples_per_sec': float(e2e_throughput),
        'throughput_samples_per_sec': float(throughput),  # Backward compat: = calibrator_throughput
        'peak_memory_mb': float(peak_memory_mb),
        'calibrator_params': geo_cal.get_params(),
        **ood_metrics,  # Add OOD metrics
    }


def run_coordinate_calibration(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    num_coordinates: int = 256,  # K - matches SGC's target_dim
    batch_size: int = 128,
    seed: int = 42,
    ood_raw: np.ndarray = None,  # OOD data for evaluation
    ood_labels: np.ndarray = None,  # OOD labels for ECE calculation
    scoring_method: str = 'separation',  # Options: 'separation' or 'trust_score'
    normalization_method: str = 'rank',
    # OPTIMIZATION: Cached predictions and loaders
    precomputed_test_probs: np.ndarray = None,
    precomputed_ood_probs: np.ndarray = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
    ood_loader: DataLoader = None,
    ood_feature_cache: Dict[str, np.ndarray] = None,  # OPTIMIZATION: Cache for OOD features
    output_dir: str = None,  # Optional output directory for per-sample data saving
    training_method: str = None  # Training method for filename construction
) -> Dict[str, Any]:
    """
    Run calibration using Global Coordinate Sampling.
    
    This is the alternative to layer-based SGC:
    - SGC: Select L layers -> SPP per layer -> JL per layer -> project-then-sum -> L2 norm
    - Coordinate: Flatten ALL activations -> Sample K coordinates -> L2 norm
    
    The geometric calibration algorithm is identical - only the feature extraction differs.
    
    Args:
        num_coordinates: K - number of global coordinates to sample (default 256 to match SGC)
    
    Returns:
        Dict with ece, accuracy, timing, and comparison metadata
    """
    score_label = "Trust Score" if scoring_method == 'trust_score' else "Separation"
    logger.info("\n" + "="*80)
    logger.info(f"RUNNING COORDINATE SAMPLING CALIBRATION (K={num_coordinates}) - {score_label}")
    logger.info("="*80)
    
    # Determine input shape based on dataset
    is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
    if is_dinov2:
        input_shape = (1, 3, 224, 224)  # DINOv2 requires 224x224
    elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
        input_shape = (1, 3, 64, 64)  # Tiny ImageNet uses 64x64
    else:
        input_shape = (1, 3, 32, 32)  # Default for CIFAR (32x32)
    
    logger.info(f"Using input shape: {input_shape}")
    
    # 1. Discover coordinate space
    logger.info("\n1. Discovering coordinate space...")
    layer_map, total_size = discover_coordinate_space(
        model, input_shape=input_shape, device=str(device)
    )
    logger.info(f"Initial discovery: {total_size:,} scalars across {len(layer_map)} layers")
    
    # 2. Validate and correct coordinate space using a real batch
    logger.info("\n2. Validating coordinate space with real batch...")
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    validation_batch = next(iter(train_loader))
    if isinstance(validation_batch, (list, tuple)):
        real_batch = validation_batch[0]
    else:
        real_batch = validation_batch
    
    if not isinstance(real_batch, torch.Tensor):
        real_batch = torch.from_numpy(real_batch) if isinstance(real_batch, np.ndarray) else real_batch
    real_batch = real_batch.to(device)
    
    layer_map, total_size = validate_and_correct_coordinate_space(
        model=model,
        layer_map=layer_map,
        real_batch=real_batch,
        device=device,
    )
    
    logger.info(f"✓ Validated coordinate space: {total_size:,} scalars across {len(layer_map)} layers")
    
    # 3. Plan coordinate extraction
    logger.info(f"\n3. Planning coordinate extraction (K={num_coordinates})...")
    t_plan = time.perf_counter()
    sampling_plan, global_order = plan_coordinate_extraction(
        total_size=total_size,
        num_coordinates=num_coordinates,
        layer_map=layer_map,
        seed=seed,
    )
    plan_time = time.perf_counter() - t_plan
    
    layers_with_coords = list(sampling_plan.keys())
    coords_per_layer = {k: len(v) for k, v in sampling_plan.items()}
    logger.info(f"  Sampling plan: {len(layers_with_coords)} layers touched")
    logger.info(f"  Top 5 layers by coordinate count: {sorted(coords_per_layer.items(), key=lambda x: -x[1])[:5]}")
    
    # 4. Extract features using coordinate sampling
    logger.info("\n4. Extracting coordinate features...")
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
    
    # 5. L2 normalize features (same as SGC)
    logger.info("\n5. Applying L2 normalization...")
    train_features_np = normalize(train_features.numpy(), norm='l2', axis=1)
    val_features_np = normalize(val_features.numpy(), norm='l2', axis=1)
    test_features_np = normalize(test_features.numpy(), norm='l2', axis=1)
    
    # 6. Run GeometricCalibrator (identical to run_global_random_calibration)
    logger.info("\n6. Fitting geometric calibrator...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_fit = time.perf_counter()
    
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features_np,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,
        device=str(device),
        scoring_method=scoring_method,  # Use separation or trust_score
        normalization_method=normalization_method
    )
    
    geo_cal.fit(
        X_val_embed=val_features_np,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fit_time = time.perf_counter() - start_fit
    
    # Calibrate test set with fixed timing
    logger.info("7. Calibrating test set...")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # 1. Pre-compute model probabilities (forward pass)
    # OPTIMIZATION: Use cached test_probs if available
    if precomputed_test_probs is not None:
        model_probs = precomputed_test_probs
        forward_time = 0.0  # No forward pass needed
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_forward = time.perf_counter()
        
        model_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - start_forward

    # 2. Calibrator-only timing (NO forward pass)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_calibrate = time.perf_counter()
    
    calibrated_probs = geo_cal.calibrate_batched_precomputed(
        X_test_embed=test_features_np,
        X_test_original=test_raw,
        model_probs=model_probs,
        batch_size=batch_size
    )
    
    # Save per-sample data for reliability diagrams and risk-coverage curves
    if output_dir is not None:
        stability_scores = getattr(geo_cal, 'last_stability_scores', None)
        save_per_sample_data(
            calibrated_probs=calibrated_probs,
            test_labels=test_labels,
            method_name='coordinate',
            output_dir=output_dir,
            stability_scores=stability_scores,
            model_name=model_name,
            dataset_name=dataset_name,
            training_method=training_method,
            seed=seed
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    calibrator_only_time = time.perf_counter() - start_calibrate

    # Aggregate timings
    e2e_calibrate_time = forward_time + calibrator_only_time
    calibrate_time = calibrator_only_time  # Backward compat
    
    # Calculate throughput
    n_test_samples = len(test_labels)
    calibrator_throughput = n_test_samples / calibrator_only_time if calibrator_only_time > 0 else 0.0
    e2e_throughput = n_test_samples / e2e_calibrate_time if e2e_calibrate_time > 0 else 0.0
    throughput = calibrator_throughput  # Backward compat
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0
    
    # Metrics
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
    brier = calculate_brier_score(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)
    
    logger.info(f"\nCoordinate Sampling Results:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Adaptive ECE: {adaptive_ece:.6f}")
    logger.info(f"  Brier: {brier:.6f}")
    logger.info(f"  Accuracy: {accuracy:.4f}%")
    
    # OOD Evaluation
    ood_metrics = {}
    if ood_raw is not None:
        logger.info("\nComputing OOD metrics on SVHN...")
        
        ood_loader = DataLoader(
            TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
            batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Extract OOD features using same coordinate sampling plan
        ood_features = extract_coordinate_features(
            model, ood_loader, sampling_plan, layer_map, global_order, device=str(device)
        )
        
        # Apply L2 normalization (same as ID)
        ood_features_np = normalize(ood_features.numpy(), norm='l2', axis=1)
        
        # Calibrate OOD using fitted calibrator
        ood_calibrated_probs = geo_cal.calibrate_batched(
            X_test_embed=ood_features_np,
            X_test_original=ood_raw,
            batch_size=batch_size
        )
        
        # Compute metrics
        ood_metrics = compute_ood_metrics(
            calibrated_probs_id=calibrated_probs,
            calibrated_probs_ood=ood_calibrated_probs,
            id_labels=test_labels,
            ood_labels=ood_labels,
        )
        
        logger.info(f"  OOD AUROC: {ood_metrics['ood_auroc']:.4f}")
        logger.info(f"  OOD FPR@95: {ood_metrics['ood_fpr95']:.4f}")
        logger.info(f"  Confidence gap: {ood_metrics['confidence_gap']:.4f} (ID: {ood_metrics['id_confidence_mean']:.4f}, OOD: {ood_metrics['ood_confidence_mean']:.4f})")
    
    return {
        'method': f'Coordinate Sampling (K={num_coordinates})',
        'ece': float(ece),
        'adaptive_ece': float(adaptive_ece),
        'calibration_mce': float(calibration_mce),
        'brier': float(brier),
        'accuracy': float(accuracy),
        'num_coordinates': num_coordinates,
        'total_coordinate_space': total_size,
        'num_layers_touched': len(layers_with_coords),
        'coords_per_layer': coords_per_layer,
        'plan_time_s': float(plan_time),
        'extraction_time_s': float(extraction_time),
        'fit_time_s': float(fit_time),
        'forward_time_s': float(forward_time),
        'calibrator_only_time_s': float(calibrator_only_time),
        'calibrate_time_s': float(calibrate_time),  # Backward compat: = calibrator_only_time
        'e2e_calibrate_time_s': float(e2e_calibrate_time),
        'calibrator_throughput_samples_per_sec': float(calibrator_throughput),
        'e2e_throughput_samples_per_sec': float(e2e_throughput),
        'throughput_samples_per_sec': float(throughput),  # Backward compat: = calibrator_throughput
        'peak_memory_mb': float(peak_memory_mb),
        'calibrator_params': geo_cal.get_params(),
        **ood_metrics,  # Add OOD metrics
    }


def run_coordinate_spp_calibration(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    num_coordinates: int = 256,  # K - matches SGC's target_dim
    batch_size: int = 128,
    seed: int = 42,
    pooling_mode: str = 'max',
    ood_raw: np.ndarray = None,  # OOD data for evaluation
    ood_labels: np.ndarray = None,  # OOD labels for ECE calculation
    scoring_method: str = 'separation',  # Options: 'separation' or 'trust_score'
    # OPTIMIZATION: Cached predictions and loaders
    precomputed_test_probs: np.ndarray = None,
    precomputed_ood_probs: np.ndarray = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
    ood_loader: DataLoader = None,
    ood_feature_cache: Dict[str, np.ndarray] = None,  # OPTIMIZATION: Cache for OOD features
    output_dir: str = None  # Optional output directory for per-sample data saving
) -> Dict[str, Any]:
    """
    Run calibration using Coordinate Sampling with SPP-only preprocessing.
    
    This is a hybrid approach:
    - Coordinate: Flatten ALL activations -> Sample K coordinates
    - SPP-only: Reshape coordinates to spatial grid -> Apply SPP (no JL projection)
    - Geometric: Use geometric separation for calibration
    
    This tests whether SPP preprocessing helps coordinate-based features.
    
    Args:
        num_coordinates: K - number of global coordinates to sample (default 256 to match SGC)
        pooling_mode: 'max' or 'avg' for SPP pooling
    
    Returns:
        Dict with ece, accuracy, timing, and comparison metadata
    """
    score_label = "Trust Score" if scoring_method == 'trust_score' else "Separation"
    logger.info("\n" + "="*80)
    logger.info(f"RUNNING COORDINATE SPP CALIBRATION (K={num_coordinates}, SPP-only) - {score_label}")
    logger.info("="*80)
    
    # Determine input shape based on dataset
    is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
    if is_dinov2:
        input_shape = (1, 3, 224, 224)  # DINOv2 requires 224x224
    elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
        input_shape = (1, 3, 64, 64)  # Tiny ImageNet uses 64x64
    else:
        input_shape = (1, 3, 32, 32)  # Default for CIFAR (32x32)
    
    logger.info(f"Using input shape: {input_shape}")
    
    # 1. Discover coordinate space
    logger.info("\n1. Discovering coordinate space...")
    layer_map, total_size = discover_coordinate_space(
        model, input_shape=input_shape, device=str(device)
    )
    logger.info(f"Initial discovery: {total_size:,} scalars across {len(layer_map)} layers")
    
    # 2. Validate and correct coordinate space using a real batch
    logger.info("\n2. Validating coordinate space with real batch...")
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    validation_batch = next(iter(train_loader))
    if isinstance(validation_batch, (list, tuple)):
        real_batch = validation_batch[0]
    else:
        real_batch = validation_batch
    
    if not isinstance(real_batch, torch.Tensor):
        real_batch = torch.from_numpy(real_batch) if isinstance(real_batch, np.ndarray) else real_batch
    real_batch = real_batch.to(device)
    
    layer_map, total_size = validate_and_correct_coordinate_space(
        model=model,
        layer_map=layer_map,
        real_batch=real_batch,
        device=device,
    )
    
    logger.info(f"✓ Validated coordinate space: {total_size:,} scalars across {len(layer_map)} layers")
    
    # 3. Plan coordinate extraction
    logger.info(f"\n3. Planning coordinate extraction (K={num_coordinates})...")
    t_plan = time.perf_counter()
    sampling_plan, global_order = plan_coordinate_extraction(
        total_size=total_size,
        num_coordinates=num_coordinates,
        layer_map=layer_map,
        seed=seed,
    )
    plan_time = time.perf_counter() - t_plan
    
    layers_with_coords = list(sampling_plan.keys())
    coords_per_layer = {k: len(v) for k, v in sampling_plan.items()}
    logger.info(f"  Sampling plan: {len(layers_with_coords)} layers touched")
    logger.info(f"  Top 5 layers by coordinate count: {sorted(coords_per_layer.items(), key=lambda x: -x[1])[:5]}")
    
    # 4. Extract features using coordinate sampling
    logger.info("\n4. Extracting coordinate features...")
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
    train_coords = extract_coordinate_features(
        model, train_loader, sampling_plan, layer_map, global_order, device=str(device)
    )
    val_coords = extract_coordinate_features(
        model, val_loader, sampling_plan, layer_map, global_order, device=str(device)
    )
    test_coords = extract_coordinate_features(
        model, test_loader, sampling_plan, layer_map, global_order, device=str(device)
    )
    extraction_time = time.perf_counter() - t_extract
    
    logger.info(f"  Extracted coordinates: train={train_coords.shape}, val={val_coords.shape}, test={test_coords.shape}")
    
    # 5. Reshape coordinates to spatial grid and apply SPP-only
    logger.info("\n5. Reshaping coordinates to spatial grid and applying SPP-only...")
    import math
    from utils.compression_utils import SPP_Only
    
    # Reshape coordinates to a square grid
    # Find the largest square that fits in num_coordinates
    grid_size = int(math.isqrt(num_coordinates))
    actual_coords = grid_size * grid_size
    if actual_coords < num_coordinates:
        # Pad if needed
        logger.info(f"  Padding {num_coordinates} coordinates to {actual_coords} (grid_size={grid_size})")
    
    # Convert to torch tensors (extract_coordinate_features returns torch.Tensor)
    train_coords_torch = train_coords.float() if isinstance(train_coords, torch.Tensor) else torch.from_numpy(train_coords).float()
    val_coords_torch = val_coords.float() if isinstance(val_coords, torch.Tensor) else torch.from_numpy(val_coords).float()
    test_coords_torch = test_coords.float() if isinstance(test_coords, torch.Tensor) else torch.from_numpy(test_coords).float()
    
    # Truncate or pad to grid_size^2
    if train_coords_torch.shape[1] > actual_coords:
        train_coords_torch = train_coords_torch[:, :actual_coords]
        val_coords_torch = val_coords_torch[:, :actual_coords]
        test_coords_torch = test_coords_torch[:, :actual_coords]
    elif train_coords_torch.shape[1] < actual_coords:
        # Pad with zeros
        pad_size = actual_coords - train_coords_torch.shape[1]
        train_coords_torch = torch.cat([train_coords_torch, torch.zeros(train_coords_torch.shape[0], pad_size)], dim=1)
        val_coords_torch = torch.cat([val_coords_torch, torch.zeros(val_coords_torch.shape[0], pad_size)], dim=1)
        test_coords_torch = torch.cat([test_coords_torch, torch.zeros(test_coords_torch.shape[0], pad_size)], dim=1)
    
    # Reshape to (N, 1, grid_size, grid_size) - treat as single-channel feature map
    train_coords_4d = train_coords_torch.reshape(-1, 1, grid_size, grid_size)
    val_coords_4d = val_coords_torch.reshape(-1, 1, grid_size, grid_size)
    test_coords_4d = test_coords_torch.reshape(-1, 1, grid_size, grid_size)
    
    logger.info(f"  Reshaped to spatial grid: {train_coords_4d.shape}")
    
    # Apply SPP-only (no JL projection)
    spp = SPP_Only(
        pyramid_levels=[4, 2, 1],
        pooling_mode=pooling_mode,
    ).to(device)
    
    # Process in batches to avoid OOM
    def apply_spp_batched(coords_4d, batch_size_spp=512):
        spp_features_list = []
        for i in range(0, len(coords_4d), batch_size_spp):
            batch = coords_4d[i:i+batch_size_spp].to(device)
            with torch.no_grad():
                spp_batch = spp(batch)
            spp_features_list.append(spp_batch.cpu())
        return torch.cat(spp_features_list, dim=0)
    
    train_features_spp = apply_spp_batched(train_coords_4d, batch_size_spp=512)
    val_features_spp = apply_spp_batched(val_coords_4d, batch_size_spp=512)
    test_features_spp = apply_spp_batched(test_coords_4d, batch_size_spp=512)
    
    spp_output_dim = train_features_spp.shape[1]
    logger.info(f"  SPP output dimension: {spp_output_dim}")
    logger.info(f"  SPP features shape: {train_features_spp.shape}")
    
    # 6. L2 normalize features (same as SGC)
    logger.info("\n6. Applying L2 normalization...")
    train_features_np = normalize(train_features_spp.numpy(), norm='l2', axis=1)
    val_features_np = normalize(val_features_spp.numpy(), norm='l2', axis=1)
    test_features_np = normalize(test_features_spp.numpy(), norm='l2', axis=1)
    
    # 7. Run GeometricCalibrator
    logger.info("\n7. Fitting geometric calibrator...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_fit = time.perf_counter()
    
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features_np,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,
        device=str(device)
    )
    
    geo_cal.fit(
        X_val_embed=val_features_np,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fit_time = time.perf_counter() - start_fit
    
    # Calibrate test set with fixed timing
    logger.info("8. Calibrating test set...")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # 1. Pre-compute model probabilities (forward pass)
    # OPTIMIZATION: Use cached test_probs if available
    if precomputed_test_probs is not None:
        model_probs = precomputed_test_probs
        forward_time = 0.0  # No forward pass needed
    else:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_forward = time.perf_counter()
        
        model_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - start_forward

    # 2. Calibrator-only timing (NO forward pass)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_calibrate = time.perf_counter()
    
    calibrated_probs = geo_cal.calibrate_batched_precomputed(
        X_test_embed=test_features_np,
        X_test_original=test_raw,
        model_probs=model_probs,
        batch_size=batch_size
    )
    
    # Save per-sample data for reliability diagrams and risk-coverage curves
    if output_dir is not None:
        stability_scores = getattr(geo_cal, 'last_stability_scores', None)
        save_per_sample_data(
            calibrated_probs=calibrated_probs,
            test_labels=test_labels,
            method_name='coordinate_spp',
            output_dir=output_dir,
            stability_scores=stability_scores,
            model_name=model_name,
            dataset_name=dataset_name,
            training_method=None,  # training_method not available in this function
            seed=seed
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    calibrator_only_time = time.perf_counter() - start_calibrate

    # Aggregate timings
    e2e_calibrate_time = forward_time + calibrator_only_time
    calibrate_time = calibrator_only_time  # Backward compat
    
    # Calculate throughput
    n_test_samples = len(test_labels)
    calibrator_throughput = n_test_samples / calibrator_only_time if calibrator_only_time > 0 else 0.0
    e2e_throughput = n_test_samples / e2e_calibrate_time if e2e_calibrate_time > 0 else 0.0
    throughput = calibrator_throughput  # Backward compat
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0
    
    # Metrics
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
    brier = calculate_brier_score(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)
    
    logger.info(f"\nCoordinate SPP Results:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Adaptive ECE: {adaptive_ece:.6f}")
    logger.info(f"  Brier: {brier:.6f}")
    logger.info(f"  Accuracy: {accuracy:.4f}%")
    
    # OOD Evaluation
    ood_metrics = {}
    if ood_raw is not None:
        logger.info("\nComputing OOD metrics on SVHN...")
        
        ood_loader = DataLoader(
            TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
            batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # Extract OOD coordinates using same plan
        ood_coords = extract_coordinate_features(
            model, ood_loader, sampling_plan, layer_map, global_order, device=str(device)
        )
        
        # Apply SPP preprocessing (same as ID)
        def apply_spp_batched(coords_4d, batch_size_spp=512):
            # This is the same SPP logic from the function above
            from utils.compression_utils import SPP_Only
            spp = SPP_Only(
                pyramid_levels=[4, 2, 1],
                pooling_mode=pooling_mode,
            ).to(device)
            results = []
            for i in range(0, len(coords_4d), batch_size_spp):
                batch = coords_4d[i:i+batch_size_spp].to(device)
                with torch.no_grad():
                    batch_spp = spp(batch)
                results.append(batch_spp.cpu().numpy() if isinstance(batch_spp, torch.Tensor) else batch_spp)
            return np.concatenate(results, axis=0)
        
        # Reshape coordinates to spatial grid (same as ID)
        ood_coords_4d = ood_coords.view(len(ood_coords), 1, grid_size, grid_size)
        ood_features_spp = apply_spp_batched(ood_coords_4d)
        
        # Apply L2 normalization
        ood_features_np = normalize(ood_features_spp, norm='l2', axis=1)
        
        # Calibrate OOD using fitted calibrator
        ood_calibrated_probs = geo_cal.calibrate_batched(
            X_test_embed=ood_features_np,
            X_test_original=ood_raw,
            batch_size=batch_size
        )
        
        # Compute metrics
        ood_metrics = compute_ood_metrics(
            calibrated_probs_id=calibrated_probs,
            calibrated_probs_ood=ood_calibrated_probs,
            id_labels=test_labels,
            ood_labels=ood_labels,
        )
        
        logger.info(f"  OOD AUROC: {ood_metrics['ood_auroc']:.4f}")
        logger.info(f"  OOD FPR@95: {ood_metrics['ood_fpr95']:.4f}")
        logger.info(f"  Confidence gap: {ood_metrics['confidence_gap']:.4f} (ID: {ood_metrics['id_confidence_mean']:.4f}, OOD: {ood_metrics['ood_confidence_mean']:.4f})")
    
    return {
        'method': f'Coordinate SPP (K={num_coordinates})',
        'ece': float(ece),
        'adaptive_ece': float(adaptive_ece),
        'calibration_mce': float(calibration_mce),
        'brier': float(brier),
        'accuracy': float(accuracy),
        'num_coordinates': num_coordinates,
        'total_coordinate_space': total_size,
        'num_layers_touched': len(layers_with_coords),
        'coords_per_layer': coords_per_layer,
        'grid_size': grid_size,
        'spp_output_dim': int(spp_output_dim),
        'pooling_mode': pooling_mode,
        'plan_time_s': float(plan_time),
        'extraction_time_s': float(extraction_time),
        'fit_time_s': float(fit_time),
        'forward_time_s': float(forward_time),
        'calibrator_only_time_s': float(calibrator_only_time),
        'calibrate_time_s': float(calibrate_time),  # Backward compat: = calibrator_only_time
        'e2e_calibrate_time_s': float(e2e_calibrate_time),
        'calibrator_throughput_samples_per_sec': float(calibrator_throughput),
        'e2e_throughput_samples_per_sec': float(e2e_throughput),
        'throughput_samples_per_sec': float(throughput),  # Backward compat: = calibrator_throughput
        'peak_memory_mb': float(peak_memory_mb),
        'calibrator_params': geo_cal.get_params(),
        **ood_metrics,  # Add OOD metrics
    }


def run_coordinate_spp_dac_calibration(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    num_coordinates: int = 256,  # K - matches SGC's target_dim
    batch_size: int = 128,
    seed: int = 42,
    pooling_mode: str = 'max',
    ood_raw: np.ndarray = None,  # OOD data for evaluation
    ood_labels: np.ndarray = None,  # OOD labels for ECE calculation
    output_dir: str = None,  # Optional output directory for per-sample data saving
    training_method: str = None,  # Training method for filename construction
    dac_knn_backend: str = 'faiss'  # kNN backend: 'faiss', 'torch', or 'both'
) -> Dict[str, Any]:
    """
    Run DAC calibration using Coordinate Sampling with SPP-only preprocessing.
    
    This is the DAC version of coordinate_spp:
    - Coordinate: Flatten ALL activations -> Sample K coordinates
    - SPP-only: Reshape coordinates to spatial grid -> Apply SPP (no JL projection)
    - DAC: Use density-aware calibration with these features
    
    This tests whether DAC's algorithm works better with SPP-preprocessed coordinate features.
    
    Args:
        num_coordinates: K - number of global coordinates to sample (default 256 to match SGC)
        pooling_mode: 'max' or 'avg' for SPP pooling
    
    Returns:
        Dict with ece, accuracy, timing, and DAC weights
    """
    logger.info("\n" + "="*80)
    logger.info(f"RUNNING COORDINATE SPP DAC CALIBRATION (K={num_coordinates}, SPP-only + DAC)")
    logger.info("="*80)
    
    # Determine input shape based on dataset
    is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
    if is_dinov2:
        input_shape = (1, 3, 224, 224)  # DINOv2 requires 224x224
    elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
        input_shape = (1, 3, 64, 64)  # Tiny ImageNet uses 64x64
    else:
        input_shape = (1, 3, 32, 32)  # Default for CIFAR (32x32)
    
    logger.info(f"Using input shape: {input_shape}")
    
    # 1. Discover coordinate space
    logger.info("\n1. Discovering coordinate space...")
    layer_map, total_size = discover_coordinate_space(
        model, input_shape=input_shape, device=str(device)
    )
    logger.info(f"Initial discovery: {total_size:,} scalars across {len(layer_map)} layers")
    
    # 2. Validate and correct coordinate space using a real batch
    logger.info("\n2. Validating coordinate space with real batch...")
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    validation_batch = next(iter(train_loader))
    if isinstance(validation_batch, (list, tuple)):
        real_batch = validation_batch[0]
    else:
        real_batch = validation_batch
    
    if not isinstance(real_batch, torch.Tensor):
        real_batch = torch.from_numpy(real_batch) if isinstance(real_batch, np.ndarray) else real_batch
    real_batch = real_batch.to(device)
    
    layer_map, total_size = validate_and_correct_coordinate_space(
        model=model,
        layer_map=layer_map,
        real_batch=real_batch,
        device=device,
    )
    
    logger.info(f"✓ Validated coordinate space: {total_size:,} scalars across {len(layer_map)} layers")
    
    # 3. Plan coordinate extraction
    logger.info(f"\n3. Planning coordinate extraction (K={num_coordinates})...")
    t_plan = time.perf_counter()
    sampling_plan, global_order = plan_coordinate_extraction(
        total_size=total_size,
        num_coordinates=num_coordinates,
        layer_map=layer_map,
        seed=seed,
    )
    plan_time = time.perf_counter() - t_plan
    
    layers_with_coords = list(sampling_plan.keys())
    coords_per_layer = {k: len(v) for k, v in sampling_plan.items()}
    logger.info(f"  Sampling plan: {len(layers_with_coords)} layers touched")
    logger.info(f"  Top 5 layers by coordinate count: {sorted(coords_per_layer.items(), key=lambda x: -x[1])[:5]}")
    
    # 4. Extract features using coordinate sampling
    logger.info("\n4. Extracting coordinate features...")
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
    train_coords = extract_coordinate_features(
        model, train_loader, sampling_plan, layer_map, global_order, device=str(device)
    )
    val_coords = extract_coordinate_features(
        model, val_loader, sampling_plan, layer_map, global_order, device=str(device)
    )
    test_coords = extract_coordinate_features(
        model, test_loader, sampling_plan, layer_map, global_order, device=str(device)
    )
    extraction_time = time.perf_counter() - t_extract
    
    logger.info(f"  Extracted coordinates: train={train_coords.shape}, val={val_coords.shape}, test={test_coords.shape}")
    
    # 5. Reshape coordinates to spatial grid and apply SPP-only
    logger.info("\n5. Reshaping coordinates to spatial grid and applying SPP-only...")
    import math
    from utils.compression_utils import SPP_Only
    
    # Reshape coordinates to a square grid
    # Find the largest square that fits in num_coordinates
    grid_size = int(math.isqrt(num_coordinates))
    actual_coords = grid_size * grid_size
    if actual_coords < num_coordinates:
        # Pad if needed
        logger.info(f"  Padding {num_coordinates} coordinates to {actual_coords} (grid_size={grid_size})")
    
    # Convert to torch tensors (extract_coordinate_features returns torch.Tensor)
    train_coords_torch = train_coords.float() if isinstance(train_coords, torch.Tensor) else torch.from_numpy(train_coords).float()
    val_coords_torch = val_coords.float() if isinstance(val_coords, torch.Tensor) else torch.from_numpy(val_coords).float()
    test_coords_torch = test_coords.float() if isinstance(test_coords, torch.Tensor) else torch.from_numpy(test_coords).float()
    
    # Truncate or pad to grid_size^2
    if train_coords_torch.shape[1] > actual_coords:
        train_coords_torch = train_coords_torch[:, :actual_coords]
        val_coords_torch = val_coords_torch[:, :actual_coords]
        test_coords_torch = test_coords_torch[:, :actual_coords]
    elif train_coords_torch.shape[1] < actual_coords:
        # Pad with zeros
        pad_size = actual_coords - train_coords_torch.shape[1]
        train_coords_torch = torch.cat([train_coords_torch, torch.zeros(train_coords_torch.shape[0], pad_size)], dim=1)
        val_coords_torch = torch.cat([val_coords_torch, torch.zeros(val_coords_torch.shape[0], pad_size)], dim=1)
        test_coords_torch = torch.cat([test_coords_torch, torch.zeros(test_coords_torch.shape[0], pad_size)], dim=1)
    
    # Reshape to (N, 1, grid_size, grid_size) - treat as single-channel feature map
    train_coords_4d = train_coords_torch.reshape(-1, 1, grid_size, grid_size)
    val_coords_4d = val_coords_torch.reshape(-1, 1, grid_size, grid_size)
    test_coords_4d = test_coords_torch.reshape(-1, 1, grid_size, grid_size)
    
    logger.info(f"  Reshaped to spatial grid: {train_coords_4d.shape}")
    
    # Apply SPP-only (no JL projection)
    spp = SPP_Only(
        pyramid_levels=[4, 2, 1],
        pooling_mode=pooling_mode,
    ).to(device)
    
    # Process in batches to avoid OOM
    def apply_spp_batched(coords_4d, batch_size_spp=512):
        spp_features_list = []
        for i in range(0, len(coords_4d), batch_size_spp):
            batch = coords_4d[i:i+batch_size_spp].to(device)
            with torch.no_grad():
                spp_batch = spp(batch)
            spp_features_list.append(spp_batch.cpu())
        return torch.cat(spp_features_list, dim=0)
    
    train_features_spp = apply_spp_batched(train_coords_4d, batch_size_spp=512)
    val_features_spp = apply_spp_batched(val_coords_4d, batch_size_spp=512)
    test_features_spp = apply_spp_batched(test_coords_4d, batch_size_spp=512)
    
    spp_output_dim = train_features_spp.shape[1]
    logger.info(f"  SPP output dimension: {spp_output_dim}")
    logger.info(f"  SPP features shape: {train_features_spp.shape}")
    
    # 6. L2 normalize features (same as SGC and DAC preprocessing)
    logger.info("\n6. Applying L2 normalization...")
    train_features_np = normalize(train_features_spp.numpy(), norm='l2', axis=1)
    val_features_np = normalize(val_features_spp.numpy(), norm='l2', axis=1)
    test_features_np = normalize(test_features_spp.numpy(), norm='l2', axis=1)
    
    # 7. Get logits for DAC
    logger.info("\n7. Extracting logits...")
    model.eval()
    train_logits_list = []
    val_logits_list = []
    test_logits_list = []
    
    with torch.no_grad():
        if train_loader is not None:
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                data = data.to(device)
                logits = model(data)
                train_logits_list.append(logits.cpu())
        
        if val_loader is not None:
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                data = data.to(device)
                logits = model(data)
                val_logits_list.append(logits.cpu())
        
        if test_loader is not None:
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                data = data.to(device)
                logits = model(data)
                test_logits_list.append(logits.cpu())
    
    train_logits = torch.cat(train_logits_list, dim=0) if train_logits_list else torch.empty((0,), dtype=torch.float32)
    val_logits = torch.cat(val_logits_list, dim=0) if val_logits_list else torch.empty((0,), dtype=torch.float32)
    test_logits = torch.cat(test_logits_list, dim=0) if test_logits_list else torch.empty((0,), dtype=torch.float32)
    
    # 8. Prepare features for DAC (SPP features + logits)
    logger.info("\n8. Preparing features for DAC...")
    # Convert to torch tensors for DAC
    train_features_torch = torch.from_numpy(train_features_np).float()
    val_features_torch = torch.from_numpy(val_features_np).float()
    test_features_torch = torch.from_numpy(test_features_np).float()
    
    # DAC input: [spp_features, logits]
    train_features_for_dac = [train_features_torch, train_logits]
    val_features_for_dac = [val_features_torch, val_logits]
    test_features_for_dac = [test_features_torch, test_logits]
    
    logger.info(f"DAC input: {len(train_features_for_dac)} layers")
    logger.info(f"  Layer 0 (SPP features): {train_features_torch.shape}")
    logger.info(f"  Layer 1 (logits): {train_logits.shape}")
    
    # 9. Fit DAC
    logger.info("\n9. Fitting DAC calibrator...")
    k_value = get_dac_k_value(dataset_name)
    
    # Handle "both" mode: run both backends
    backends_to_run = ['faiss', 'torch'] if dac_knn_backend == 'both' else [dac_knn_backend]
    all_results = {}
    
    for backend in backends_to_run:
        logger.info(f"\n  Running DAC with {backend} backend...")
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_fit = time.perf_counter()
        
        dac = make_dac_calibrator(k=k_value, device=device, backend=backend)
        dac.fit(train_features_for_dac, val_features_for_dac, val_logits, val_labels)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        fit_time = time.perf_counter() - t_fit
        logger.info(f"  DAC fitting time ({backend}): {fit_time:.2f}s")
        
        # Extract DAC weights
        layer_weights = dac.weights[:-1].tolist() if dac.weights is not None else []
        bias = float(dac.weights[-1]) if dac.weights is not None else 0.0
        
        # 10. Calibrate test set
        logger.info(f"\n10. Calibrating test set ({backend})...")
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_cal = time.perf_counter()
        
        calibrated_probs = dac.calibrate(test_features_for_dac, test_logits)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        calibrate_time = time.perf_counter() - t_cal
        
        throughput = len(test_labels) / calibrate_time if calibrate_time > 0 else 0
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0
        
        logger.info(f"  Calibration time ({backend}): {calibrate_time:.2f}s")
        logger.info(f"  Throughput ({backend}): {throughput:.2f} samples/sec")
        
        # Metrics
        ece = calculate_ece(calibrated_probs, test_labels)
        adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
        calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
        brier = calculate_brier_score(calibrated_probs, test_labels)
        accuracy = calculate_accuracy(calibrated_probs, test_labels)
        
        logger.info(f"\n  Coordinate SPP DAC Results ({backend}):")
        logger.info(f"    ECE: {ece:.6f}")
        logger.info(f"    Adaptive ECE: {adaptive_ece:.6f}")
        logger.info(f"    Brier: {brier:.6f}")
        logger.info(f"    Accuracy: {accuracy:.4f}%")
        
        # OOD Evaluation
        ood_metrics = {}
        if ood_raw is not None:
            logger.info(f"\n  Computing OOD metrics on SVHN ({backend})...")
            
            ood_loader = DataLoader(
                TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
                batch_size=batch_size, shuffle=False, num_workers=0
            )
            
            # Extract OOD coordinates using same plan
            ood_coords = extract_coordinate_features(
                model, ood_loader, sampling_plan, layer_map, global_order, device=str(device)
            )
            
            # Reshape to spatial grid (same as ID)
            ood_coords_torch = ood_coords.float() if isinstance(ood_coords, torch.Tensor) else torch.from_numpy(ood_coords).float()
            
            # Truncate or pad to grid_size^2
            if ood_coords_torch.shape[1] > actual_coords:
                ood_coords_torch = ood_coords_torch[:, :actual_coords]
            elif ood_coords_torch.shape[1] < actual_coords:
                pad_size = actual_coords - ood_coords_torch.shape[1]
                ood_coords_torch = torch.cat([ood_coords_torch, torch.zeros(ood_coords_torch.shape[0], pad_size)], dim=1)
            
            # Reshape to (N, 1, grid_size, grid_size)
            ood_coords_4d = ood_coords_torch.reshape(-1, 1, grid_size, grid_size)
            
            # Apply SPP (same as ID)
            ood_features_spp = apply_spp_batched(ood_coords_4d, batch_size_spp=512)
            
            # L2 normalize
            ood_features_np = normalize(ood_features_spp.numpy(), norm='l2', axis=1)
            
            # Get OOD logits
            ood_logits_list = []
            model.eval()
            with torch.no_grad():
                for batch in ood_loader:
                    if isinstance(batch, (list, tuple)):
                        data = batch[0]
                    else:
                        data = batch
                    data = data.to(device)
                    logits = model(data)
                    ood_logits_list.append(logits.cpu())
            ood_logits = torch.cat(ood_logits_list, dim=0) if ood_logits_list else torch.empty((0,), dtype=torch.float32)
            
            # Prepare features for DAC
            ood_features_torch = torch.from_numpy(ood_features_np).float()
            ood_features_for_dac = [ood_features_torch, ood_logits]
            
            # Calibrate OOD using fitted DAC calibrator
            ood_calibrated_probs = dac.calibrate(ood_features_for_dac, ood_logits)
            
            # Compute metrics
            ood_metrics = compute_ood_metrics(
                calibrated_probs_id=calibrated_probs,
                calibrated_probs_ood=ood_calibrated_probs,
                id_labels=test_labels,
                ood_labels=ood_labels,
            )
            
            logger.info(f"    OOD AUROC ({backend}): {ood_metrics['ood_auroc']:.4f}")
            logger.info(f"    OOD FPR@95 ({backend}): {ood_metrics['ood_fpr95']:.4f}")
            logger.info(f"    Confidence gap ({backend}): {ood_metrics['confidence_gap']:.4f} (ID: {ood_metrics['id_confidence_mean']:.4f}, OOD: {ood_metrics['ood_confidence_mean']:.4f})")
        
        # Save per-sample data for reliability diagrams and risk-coverage curves
        if output_dir is not None:
            method_suffix = '_torch' if backend == 'torch' else ''
            save_per_sample_data(
                calibrated_probs=calibrated_probs,
                test_labels=test_labels,
                method_name=f'coordinate_spp_dac{method_suffix}',
                output_dir=output_dir,
                stability_scores=None,  # DAC doesn't use stability scores
                model_name=model_name,
                dataset_name=dataset_name,
                training_method=training_method,
                seed=seed
            )
        
        # Build result dict for this backend
        result_dict = {
            'method': f'Coordinate SPP DAC (K={num_coordinates})',
            'ece': float(ece),
            'adaptive_ece': float(adaptive_ece),
            'calibration_mce': float(calibration_mce),
            'brier': float(brier),
            'accuracy': float(accuracy),
            'num_coordinates': num_coordinates,
            'total_coordinate_space': total_size,
            'num_layers_touched': len(layers_with_coords),
            'coords_per_layer': coords_per_layer,
            'grid_size': grid_size,
            'spp_output_dim': int(spp_output_dim),
            'pooling_mode': pooling_mode,
            'plan_time_s': float(plan_time),
            'extraction_time_s': float(extraction_time),
            'fit_time_s': float(fit_time),
            'calibrate_time_s': float(calibrate_time),
            'throughput_samples_per_sec': float(throughput),
            'peak_memory_mb': float(peak_memory_mb),
            'dac_k': k_value,
            'layer_weights': layer_weights,
            'bias': bias,
            'knn_backend': backend,
            'knn_backend_impl': dac.__class__.__module__ + '.' + dac.__class__.__name__,
            **ood_metrics,  # Add OOD metrics
        }
        
        # Store result for this backend
        if backend == 'faiss':
            if dac_knn_backend == 'both':
                all_results['coordinate_spp_dac'] = result_dict
            else:
                all_results = result_dict  # Single backend mode (faiss)
        elif backend == 'torch':
            if dac_knn_backend == 'both':
                all_results['coordinate_spp_dac_torch'] = result_dict
            else:
                all_results = result_dict  # Single backend mode (torch)
    
    # Return appropriate result based on mode
    if dac_knn_backend == 'both':
        # Return dict with both results (caller will handle storing under appropriate keys)
        return all_results
    else:
        # Single backend mode - return result directly
        return all_results


def run_dac_with_coordinate_features(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    num_coordinates: int = 256,
    batch_size: int = 128,
    seed: int = 42,
    layer_map: List[Dict] = None,
    total_size: int = None,
    sampling_plan: Dict[str, List[int]] = None,
    global_order: List[int] = None,
    ood_raw: np.ndarray = None,  # OOD data for evaluation
    ood_labels: np.ndarray = None,  # OOD labels for ECE calculation
    # OPTIMIZATION: Cached predictions and loaders
    precomputed_test_probs: np.ndarray = None,
    precomputed_ood_probs: np.ndarray = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
    ood_loader: DataLoader = None,
    ood_feature_cache: Dict[str, np.ndarray] = None,  # OPTIMIZATION: Cache for OOD features
    output_dir: str = None,  # Optional output directory for per-sample data saving
    training_method: str = None,  # Training method for filename construction
    dac_knn_backend: str = 'faiss'  # kNN backend: 'faiss', 'torch', or 'both'
) -> Dict[str, Any]:
    """
    Run DAC calibration using coordinate-based features split into depth groups.
    
    This tests if DAC's multi-layer weighting helps coordinate features by:
    1. Using the same coordinate extraction as geometric calibration
    2. Splitting coordinates into 5 depth groups using build_depth_grouped_dac_features
    3. Running DAC calibration with these pseudo-layers + logits
    
    Args:
        layer_map, total_size, sampling_plan, global_order: Optional pre-computed values
            to avoid re-extraction if already computed in run_coordinate_calibration
    
    Returns:
        Dict with ece, accuracy, timing, and metadata
    """
    logger.info("\n" + "="*80)
    logger.info(f"RUNNING DAC WITH COORDINATE FEATURES (K={num_coordinates})")
    logger.info("="*80)
    
    # If not provided, discover and extract coordinates
    if layer_map is None or total_size is None or sampling_plan is None or global_order is None:
        # Determine input shape based on dataset
        is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
        if is_dinov2:
            input_shape = (1, 3, 224, 224)
        elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
            input_shape = (1, 3, 64, 64)
        else:
            input_shape = (1, 3, 32, 32)
        
        logger.info("Discovering coordinate space...")
        layer_map, total_size = discover_coordinate_space(
            model, input_shape=input_shape, device=str(device)
        )
        
        # Validate with real batch
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
            batch_size=batch_size, shuffle=False, num_workers=0
        )
        validation_batch = next(iter(train_loader))
        if isinstance(validation_batch, (list, tuple)):
            real_batch = validation_batch[0]
        else:
            real_batch = validation_batch
        
        if not isinstance(real_batch, torch.Tensor):
            real_batch = torch.from_numpy(real_batch) if isinstance(real_batch, np.ndarray) else real_batch
        real_batch = real_batch.to(device)
        
        layer_map, total_size = validate_and_correct_coordinate_space(
            model=model,
            layer_map=layer_map,
            real_batch=real_batch,
            device=device,
        )
        
        # Plan extraction
        sampling_plan, global_order = plan_coordinate_extraction(
            total_size=total_size,
            num_coordinates=num_coordinates,
            layer_map=layer_map,
            seed=seed,
        )
    
    # Extract coordinate features
    logger.info("Extracting coordinate features...")
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
    
    # Compute coordinate selection details (for comparison with geometric)
    layers_with_coords = list(sampling_plan.keys())
    coords_per_layer = {k: len(v) for k, v in sampling_plan.items()}
    logger.info(f"  Coordinate selection: {len(layers_with_coords)} layers touched")
    logger.info(f"  Top 5 layers by coordinate count: {sorted(coords_per_layer.items(), key=lambda x: -x[1])[:5]}")
    
    # Get logits
    logger.info("Extracting logits...")
    model.eval()
    train_logits_list = []
    val_logits_list = []
    test_logits_list = []
    
    with torch.no_grad():
        if train_loader is not None:
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                data = data.to(device)
                logits = model(data)
                train_logits_list.append(logits.cpu())
        
        if val_loader is not None:
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                data = data.to(device)
                logits = model(data)
                val_logits_list.append(logits.cpu())
        
        if test_loader is not None:
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                data = data.to(device)
                logits = model(data)
                test_logits_list.append(logits.cpu())
    
    train_logits = torch.cat(train_logits_list, dim=0) if train_logits_list else torch.empty((0,), dtype=torch.float32)
    val_logits = torch.cat(val_logits_list, dim=0) if val_logits_list else torch.empty((0,), dtype=torch.float32)
    test_logits = torch.cat(test_logits_list, dim=0) if test_logits_list else torch.empty((0,), dtype=torch.float32)
    
    # Split coordinates into depth groups
    logger.info("Splitting coordinates into depth groups...")
    train_depth_groups_np = build_depth_grouped_dac_features(
        sampling_plan, layer_map, train_features.numpy(), num_groups=5
    )
    val_depth_groups_np = build_depth_grouped_dac_features(
        sampling_plan, layer_map, val_features.numpy(), num_groups=5
    )
    test_depth_groups_np = build_depth_grouped_dac_features(
        sampling_plan, layer_map, test_features.numpy(), num_groups=5
    )
    
    # Convert to torch tensors (DAC expects torch tensors)
    train_dac_layers = [torch.from_numpy(g).float() for g in train_depth_groups_np]
    val_dac_layers = [torch.from_numpy(g).float() for g in val_depth_groups_np]
    test_dac_layers = [torch.from_numpy(g).float() for g in test_depth_groups_np]
    
    # Add logits as final layer
    train_features_dac = train_dac_layers + [train_logits]
    val_features_dac = val_dac_layers + [val_logits]
    test_features_dac = test_dac_layers + [test_logits]
    
    logger.info(f"DAC layer dims: {[f.shape[1] for f in train_features_dac]}")
    
    # Run DAC calibration
    logger.info("Fitting DAC calibrator...")
    k_value = get_dac_k_value(dataset_name)
    
    # Handle "both" mode: run both backends
    backends_to_run = ['faiss', 'torch'] if dac_knn_backend == 'both' else [dac_knn_backend]
    all_results = {}
    
    for backend in backends_to_run:
        logger.info(f"\n  Running DAC with {backend} backend...")
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_fit = time.perf_counter()
        
        dac = make_dac_calibrator(k=k_value, device=device, backend=backend)
        dac.fit(train_features_dac, val_features_dac, val_logits, val_labels)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        fit_time = time.perf_counter() - t_fit
        logger.info(f"  DAC fitting time ({backend}): {fit_time:.2f}s")
        
        logger.info(f"Calibrating test set ({backend})...")
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_cal = time.perf_counter()
        
        calibrated_probs = dac.calibrate(test_features_dac, test_logits)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        calibrate_time = time.perf_counter() - t_cal
        logger.info(f"  Calibration time ({backend}): {calibrate_time:.2f}s")
        
        # Extract DAC weights
        layer_weights = dac.weights[:-1].tolist() if dac.weights is not None else []
        bias = float(dac.weights[-1]) if dac.weights is not None else 0.0
        
        # Metrics
        ece = calculate_ece(calibrated_probs, test_labels)
        adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
        calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
        brier = calculate_brier_score(calibrated_probs, test_labels)
        accuracy = calculate_accuracy(calibrated_probs, test_labels)
        
        logger.info(f"\n  DAC with Coordinate Features Results ({backend}):")
        logger.info(f"    ECE: {ece:.6f}")
        logger.info(f"    Adaptive ECE: {adaptive_ece:.6f}")
        logger.info(f"    Brier: {brier:.6f}")
        logger.info(f"    Accuracy: {accuracy:.4f}%")
        
        # OOD Evaluation
        ood_metrics = {}
        if ood_raw is not None:
            logger.info(f"\n  Computing OOD metrics on SVHN ({backend})...")
            
            ood_loader = DataLoader(
                TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
                batch_size=batch_size, shuffle=False, num_workers=0
            )
            
            # Extract OOD coordinates using same plan
            ood_features = extract_coordinate_features(
                model, ood_loader, sampling_plan, layer_map, global_order, device=str(device)
            )
            
            # Extract OOD logits
            model.eval()
            ood_logits_list = []
            with torch.no_grad():
                for batch in ood_loader:
                    if isinstance(batch, (list, tuple)):
                        data = batch[0]
                    else:
                        data = batch
                    data = data.to(device)
                    logits = model(data)
                    ood_logits_list.append(logits.cpu())
            ood_logits = torch.cat(ood_logits_list, dim=0)
            
            # Split into depth groups (same as ID)
            ood_depth_groups_np = build_depth_grouped_dac_features(
                sampling_plan, layer_map, ood_features.numpy(), num_groups=5
            )
            ood_dac_layers = [torch.from_numpy(g).float() for g in ood_depth_groups_np]
            ood_features_dac = ood_dac_layers + [ood_logits]
            
            # Calibrate using fitted DAC
            ood_calibrated_probs = dac.calibrate(ood_features_dac, ood_logits)
            
            ood_metrics = compute_ood_metrics(
                calibrated_probs_id=calibrated_probs,
                calibrated_probs_ood=ood_calibrated_probs,
                id_labels=test_labels,
                ood_labels=ood_labels,
            )
            
            logger.info(f"    OOD AUROC ({backend}): {ood_metrics['ood_auroc']:.4f}")
            logger.info(f"    OOD FPR@95 ({backend}): {ood_metrics['ood_fpr95']:.4f}")
            logger.info(f"    Confidence gap ({backend}): {ood_metrics['confidence_gap']:.4f} (ID: {ood_metrics['id_confidence_mean']:.4f}, OOD: {ood_metrics['ood_confidence_mean']:.4f})")
        
        # Save per-sample data for reliability diagrams and risk-coverage curves
        if output_dir is not None:
            method_suffix = '_torch' if backend == 'torch' else ''
            save_per_sample_data(
                calibrated_probs=calibrated_probs,
                test_labels=test_labels,
                method_name=f'dac_with_coordinate_features{method_suffix}',
                output_dir=output_dir,
                stability_scores=None,  # DAC doesn't use stability scores
                model_name=model_name,
                dataset_name=dataset_name,
                training_method=training_method,
                seed=seed
            )
        
        # Build result dict for this backend
        result_dict = {
            'method': f'DAC with Coordinate Features (K={num_coordinates})',
            'ece': float(ece),
            'adaptive_ece': float(adaptive_ece),
            'calibration_mce': float(calibration_mce),
            'brier': float(brier),
            'accuracy': float(accuracy),
            'num_coordinates': num_coordinates,
            'total_coordinate_space': total_size,
            'num_layers_touched': len(layers_with_coords),
            'coords_per_layer': coords_per_layer,
            'num_pseudo_layers': len(train_features_dac),
            'extraction_time_s': float(extraction_time),
            'fit_time_s': float(fit_time),
            'calibrate_time_s': float(calibrate_time),
            'dac_k': k_value,
            'layer_weights': layer_weights,
            'bias': bias,
            'knn_backend': backend,
            'knn_backend_impl': dac.__class__.__module__ + '.' + dac.__class__.__name__,
            **ood_metrics,  # Add OOD metrics
        }
        
        # Store result for this backend
        if backend == 'faiss':
            if dac_knn_backend == 'both':
                all_results['dac_with_coordinate_features'] = result_dict
            else:
                all_results = result_dict  # Single backend mode (faiss)
        elif backend == 'torch':
            if dac_knn_backend == 'both':
                all_results['dac_with_coordinate_features_torch'] = result_dict
            else:
                all_results = result_dict  # Single backend mode (torch)
    
    # Return appropriate result based on mode
    if dac_knn_backend == 'both':
        # Return dict with both results (caller will handle storing under appropriate keys)
        return all_results
    else:
        # Single backend mode - return result directly
        return all_results


def run_geometric_with_dac_weighting(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    precomputed_features: Dict[str, Any] = None,
    precomputed_separations: Dict[str, Any] = None,
    pooling_mode: str = 'max',
    target_dimension: int = None,
    ood_raw: np.ndarray = None,  # OOD data for evaluation
    ood_labels: np.ndarray = None,  # OOD labels for ECE calculation
    # OPTIMIZATION: Cached predictions and loaders
    precomputed_test_probs: np.ndarray = None,
    precomputed_ood_probs: np.ndarray = None,
    precomputed_val_probs: np.ndarray = None,
    precomputed_val_preds: np.ndarray = None,
    precomputed_test_preds: np.ndarray = None,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
    ood_loader: DataLoader = None,
    output_dir: str = None,  # Optional output directory for per-sample data saving
    training_method: str = None  # Training method for filename construction
) -> Dict[str, Any]:
    """
    Run Geometric calibration with DAC's multi-layer weighting strategy.
    
    Instead of concatenating features, this computes geometric separation per layer,
    then learns optimal weights: combined_score = w1*sep1 + w2*sep2 + w3*sep3 + w4*sep4 + bias
    
    This directly tests if DAC's weighting approach improves geometric separation.
    
    Args:
        precomputed_features: Pre-extracted features to avoid recomputation
        precomputed_separations: Pre-computed separation scores to avoid recomputation
    """
    from scipy.optimize import minimize
    
    logger.info("\n" + "="*80)
    logger.info("GEOMETRIC WITH DAC-STYLE LAYER WEIGHTING")
    logger.info("="*80)
    
    all_layer_names = get_layer_names_for_model(model_name, model)
    n_layers = len(all_layer_names)
    logger.info(f"Using {n_layers} layers: {all_layer_names}")
    
    # Use precomputed features if available
    if precomputed_features is not None:
        logger.info("Using precomputed features")
        train_features_list = precomputed_features['train_features_per_layer']
        val_features_list = precomputed_features['val_features_per_layer']
        test_features_list = precomputed_features['test_features_per_layer']
        extract_time = 0.0
        
        logger.info("Feature shapes:")
        for i, layer_name in enumerate(all_layer_names):
            logger.info(f"  Layer {i+1} ({layer_name}): {train_features_list[i].shape}")
    else:
        # Original extraction code
        logger.info("\n1. Extracting features from all layers...")
        start_extract = time.perf_counter()
        
        # OPTIMIZATION: Use cached loaders if available, otherwise create new ones
        if train_loader is None:
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=batch_size, shuffle=False
            )
        if val_loader is None:
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                batch_size=batch_size, shuffle=False
            )
        if test_loader is None:
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
                batch_size=batch_size, shuffle=False
            )
        
        # Use target_dimension if provided, otherwise fall back to compression_ratio
        if target_dimension is not None:
            compression_ratio = None
            logger.info(f"Using fixed target dimension: d={target_dimension}")
        else:
            # Determine compression ratio based on dataset
            dataset_lower = dataset_name.lower()
            if dataset_lower == "cifar100":
                compression_ratio = 32.0
            elif dataset_lower in ["tiny_imagenet", "tinyimagenet"]:
                compression_ratio = 32.0  # Similar to CIFAR-100
            else:
                compression_ratio = 16.0  # Default for CIFAR10
            logger.info(f"Using compression ratio: {compression_ratio}x for {dataset_name}")
        
        train_features_list = extract_features_from_multiple_layers(
            model, train_loader, all_layer_names, device, 
            compression_ratio=compression_ratio,
            target_dimension=target_dimension,
            pooling_mode=pooling_mode
        )
        val_features_list = extract_features_from_multiple_layers(
            model, val_loader, all_layer_names, device, 
            compression_ratio=compression_ratio,
            target_dimension=target_dimension,
            pooling_mode=pooling_mode
        )
        test_features_list = extract_features_from_multiple_layers(
            model, test_loader, all_layer_names, device, 
            compression_ratio=compression_ratio,
            target_dimension=target_dimension,
            pooling_mode=pooling_mode
        )
        
        extract_time = time.perf_counter() - start_extract
        logger.info(f"Feature extraction time: {extract_time:.2f}s")
        for i, layer_name in enumerate(all_layer_names):
            logger.info(f"  Layer {i+1} ({layer_name}): {train_features_list[i].shape}")
    
    # Use precomputed separation scores if available
    # For calibration, we should use MODEL PREDICTIONS not true labels
    if precomputed_separations is not None:
        if 'val_calibration_separation_matrix' in precomputed_separations:
            logger.info("\nUsing precomputed CALIBRATION separation scores (model predictions)")
            val_sep_matrix = precomputed_separations['val_calibration_separation_matrix']
            test_sep_matrix = precomputed_separations['test_calibration_separation_matrix']
        else:
            logger.warning("\nCalibration separations not available, falling back to oracle separations")
            logger.warning("  (This uses true labels instead of model predictions)")
            val_sep_matrix = precomputed_separations['val_separation_matrix']
            test_sep_matrix = precomputed_separations['test_separation_matrix']
        compute_time = 0.0
    else:
        # Compute calibration separation using model predictions
        logger.info("\n2. Computing CALIBRATION separation scores per layer (using model predictions)...")
        start_compute = time.perf_counter()
        
        # OPTIMIZATION: Use cached predictions if available
        if precomputed_val_preds is not None:
            val_pred = precomputed_val_preds
        else:
            val_pred = np.argmax(model_adapter.predict_proba(val_raw), axis=1)
        
        if precomputed_test_preds is not None:
            test_pred = precomputed_test_preds
        else:
            test_pred = np.argmax(model_adapter.predict_proba(test_raw), axis=1)
        logger.info(f"  Val accuracy: {100.0 * np.mean(val_pred == val_labels):.2f}%")
        logger.info(f"  Test accuracy: {100.0 * np.mean(test_pred == test_labels):.2f}%")
        
        # Compute separation scores for each layer on validation and test sets
        val_separation_scores_per_layer = []
        test_separation_scores_per_layer = []
        
        for i, layer_name in enumerate(all_layer_names):
            logger.info(f"  Computing separation for layer {i+1}/{n_layers}: {layer_name}")
            
            # Convert to numpy
            train_np = train_features_list[i].numpy() if isinstance(train_features_list[i], torch.Tensor) else train_features_list[i]
            val_np = val_features_list[i].numpy() if isinstance(val_features_list[i], torch.Tensor) else val_features_list[i]
            test_np = test_features_list[i].numpy() if isinstance(test_features_list[i], torch.Tensor) else test_features_list[i]
            
            # L2 normalize features per layer (required for geometric separation, same as SGC)
            train_np = normalize(train_np, norm='l2', axis=1)
            val_np = normalize(val_np, norm='l2', axis=1)
            test_np = normalize(test_np, norm='l2', axis=1)
            
            # Use our existing separation utilities with MODEL PREDICTIONS
            val_sep = compute_fast_separation_scores(
                train_features=train_np,
                train_labels=train_labels,
                query_features=val_np,
                query_pred=val_pred,  # MODEL predictions, not true labels
                use_cuda=(device.type == 'cuda')
            )
            test_sep = compute_fast_separation_scores(
                train_features=train_np,
                train_labels=train_labels,
                query_features=test_np,
                query_pred=test_pred,  # MODEL predictions, not true labels
                use_cuda=(device.type == 'cuda')
            )
            
            val_separation_scores_per_layer.append(val_sep)
            test_separation_scores_per_layer.append(test_sep)
            
            logger.info(f"    Val separation range: [{val_sep.min():.4f}, {val_sep.max():.4f}]")
        
        compute_time = time.perf_counter() - start_compute
        logger.info(f"Separation computation time: {compute_time:.2f}s")
        
        # Convert to matrices: [n_samples, n_layers]
        val_sep_matrix = np.column_stack(val_separation_scores_per_layer)
        test_sep_matrix = np.column_stack(test_separation_scores_per_layer)
    
    logger.info(f"  Validation separation matrix shape: {val_sep_matrix.shape}")
    logger.info(f"  Test separation matrix shape: {test_sep_matrix.shape}")
    
    # Step 3: Learn optimal weights using DAC's optimization approach
    logger.info("\n3. Learning optimal layer weights (DAC-style optimization)...")
    start_fit = time.perf_counter()
    
    # Get model predictions for validation set ONCE (to determine correctness)
    # OPTIMIZATION: Use cached val_probs if available
    if precomputed_val_probs is not None:
        val_logits_precomputed = precomputed_val_probs
        val_preds = np.argmax(val_logits_precomputed, axis=1)
    else:
        val_logits_precomputed = model_adapter.predict_proba(val_raw)
        val_preds = np.argmax(val_logits_precomputed, axis=1)
    
    # Prepare data in one-hot format (like DAC does)
    n_classes = len(np.unique(train_labels))
    val_labels_onehot = np.zeros((len(val_labels), n_classes))
    val_labels_onehot[np.arange(len(val_labels)), val_labels] = 1
    
    # Define objective function (same as DAC's squared error loss)
    # OPTIMIZATION: val_logits_precomputed is captured in closure - no recomputation!
    def objective_function(params):
        """
        Minimize squared error between predicted confidences and true labels.
        params = [w1, w2, w3, w4, bias]
        """
        layer_weights = params[:-1]
        bias = params[-1]
        
        # Compute weighted separation scores: S(x,w) = w1*s1 + w2*s2 + ... + bias
        weighted_scores = val_sep_matrix @ layer_weights + bias
        
        # FIXED: Use pre-computed logits (captured in closure)
        # This saves ~500-1000 forward passes during optimization!
        
        # Apply temperature scaling: Q(x,w) = softmax(z_L / S(x,w))
        # Avoid division by zero
        temperatures = np.maximum(weighted_scores, 0.1)
        calibrated_logits = val_logits_precomputed / temperatures[:, np.newaxis]
        
        # Softmax
        max_logits = np.max(calibrated_logits, axis=1, keepdims=True)
        exp_logits = np.exp(calibrated_logits - max_logits)
        calibrated_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Squared error loss
        loss = np.sum((val_labels_onehot - calibrated_probs) ** 2)
        
        return loss
    
    # Initial weights (equal weighting)
    initial_params = np.ones(n_layers + 1)
    initial_params[-1] = 1.0  # bias
    
    # Bounds: weights >= 0, bias unconstrained
    bounds = [(0.0, None)] * n_layers + [(None, None)]
    
    logger.info(f"Optimizing {n_layers} layer weights + 1 bias using Squared Error Loss...")
    
    result = minimize(
        objective_function,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    optimal_weights = result.x[:-1]
    optimal_bias = result.x[-1]
    
    fit_time = time.perf_counter() - start_fit
    
    logger.info(f"Optimization finished. Loss: {result.fun:.6f}")
    logger.info(f"Learned Weights: {optimal_weights}")
    logger.info(f"Learned Bias: {optimal_bias:.4f}")
    logger.info(f"Fitting time: {fit_time:.2f}s")
    
    # Step 4: Apply to test set
    logger.info("\n4. Calibrating test set...")
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    start_calibrate = time.perf_counter()
    
    # Compute weighted separation scores for test
    test_weighted_scores = test_sep_matrix @ optimal_weights + optimal_bias
    
    # OPTIMIZATION: Use cached test_probs if available
    if precomputed_test_probs is not None:
        test_logits = precomputed_test_probs
    else:
        test_logits = model_adapter.predict_proba(test_raw)
    
    # Apply temperature scaling
    temperatures = np.maximum(test_weighted_scores, 0.1)
    calibrated_logits = test_logits / temperatures[:, np.newaxis]
    
    # Softmax to get calibrated probabilities
    max_logits = np.max(calibrated_logits, axis=1, keepdims=True)
    exp_logits = np.exp(calibrated_logits - max_logits)
    calibrated_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Save per-sample data for reliability diagrams and risk-coverage curves
    if output_dir is not None:
        save_per_sample_data(
            calibrated_probs=calibrated_probs,
            test_labels=test_labels,
            method_name='geometric_dac_weighting',
            output_dir=output_dir,
            stability_scores=None,  # This method doesn't use stability scores
            model_name=model_name,
            dataset_name=dataset_name,
            training_method=training_method,
            seed=None  # seed not available in this function
        )
    
    calibrate_time = time.perf_counter() - start_calibrate
    
    # Calculate throughput
    throughput = len(test_labels) / calibrate_time
    peak_memory_mb = 0
    if device.type == 'cuda':
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    logger.info(f"Calibration time: {calibrate_time:.2f}s")
    logger.info(f"Throughput: {throughput:.2f} samples/sec")
    
    # Calculate metrics
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
    brier = calculate_brier_score(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)
    
    logger.info(f"\nResults:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Adaptive ECE: {adaptive_ece:.6f}")
    logger.info(f"  Brier: {brier:.6f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    
    # OOD Evaluation
    ood_metrics = {}
    if ood_raw is not None:
        logger.info("\nComputing OOD metrics on SVHN...")
        
        # OPTIMIZATION: Use cached OOD loader if available
        if ood_loader is None:
            ood_loader = DataLoader(
                TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
                batch_size=batch_size, shuffle=False
            )
        
        # Extract OOD features per layer
        if precomputed_features is None:
            # Extract fresh
            ood_features_list = extract_features_from_multiple_layers(
                model, ood_loader, all_layer_names, device,
                compression_ratio=None if target_dimension is not None else (32.0 if dataset_name.lower() == "cifar100" or dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"] else 16.0),
                target_dimension=target_dimension,
                pooling_mode=pooling_mode
            )
        else:
            # For OOD, we need to extract fresh (can't use precomputed)
            ood_features_list = extract_features_from_multiple_layers(
                model, ood_loader, all_layer_names, device,
                compression_ratio=None if target_dimension is not None else (32.0 if dataset_name.lower() == "cifar100" or dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"] else 16.0),
                target_dimension=target_dimension,
                pooling_mode=pooling_mode
            )
        
        # Compute OOD separation scores per layer
        # OPTIMIZATION: Use cached OOD predictions if available
        if precomputed_ood_probs is not None:
            ood_pred = np.argmax(precomputed_ood_probs, axis=1)
        else:
            ood_pred = np.argmax(model_adapter.predict_proba(ood_raw), axis=1)
        ood_separation_scores_per_layer = []
        
        for i, layer_name in enumerate(all_layer_names):
            ood_np = ood_features_list[i].numpy() if isinstance(ood_features_list[i], torch.Tensor) else ood_features_list[i]
            ood_np = normalize(ood_np, norm='l2', axis=1)
            
            train_np = train_features_list[i].numpy() if isinstance(train_features_list[i], torch.Tensor) else train_features_list[i]
            train_np = normalize(train_np, norm='l2', axis=1)
            
            ood_sep = compute_fast_separation_scores(
                train_features=train_np,
                train_labels=train_labels,
                query_features=ood_np,
                query_pred=ood_pred,
                use_cuda=(device.type == 'cuda')
            )
            ood_separation_scores_per_layer.append(ood_sep)
        
        # Convert to matrix
        ood_sep_matrix = np.column_stack(ood_separation_scores_per_layer)
        
        # Compute weighted separation scores for OOD
        ood_weighted_scores = ood_sep_matrix @ optimal_weights + optimal_bias
        
        # Get OOD logits
        # OPTIMIZATION: Use cached OOD probs if available
        if precomputed_ood_probs is not None:
            ood_logits = precomputed_ood_probs
        else:
            ood_logits = model_adapter.predict_proba(ood_raw)
        
        # Apply temperature scaling (same as ID)
        ood_temperatures = np.maximum(ood_weighted_scores, 0.1)
        ood_calibrated_logits = ood_logits / ood_temperatures[:, np.newaxis]
        
        # Softmax to get calibrated probabilities
        ood_max_logits = np.max(ood_calibrated_logits, axis=1, keepdims=True)
        ood_exp_logits = np.exp(ood_calibrated_logits - ood_max_logits)
        ood_calibrated_probs = ood_exp_logits / np.sum(ood_exp_logits, axis=1, keepdims=True)
        
        # Compute metrics
        ood_metrics = compute_ood_metrics(
            calibrated_probs_id=calibrated_probs,
            calibrated_probs_ood=ood_calibrated_probs,
            id_labels=test_labels,
            ood_labels=ood_labels,
        )
        
        logger.info(f"  OOD AUROC: {ood_metrics['ood_auroc']:.4f}")
        logger.info(f"  OOD FPR@95: {ood_metrics['ood_fpr95']:.4f}")
        logger.info(f"  Confidence gap: {ood_metrics['confidence_gap']:.4f} (ID: {ood_metrics['id_confidence_mean']:.4f}, OOD: {ood_metrics['ood_confidence_mean']:.4f})")
    
    return {
        'method': 'Geometric + DAC Weighting',
        'ece': float(ece),
        'adaptive_ece': float(adaptive_ece),
        'calibration_mce': float(calibration_mce),
        'brier': float(brier),
        'accuracy': float(accuracy),
        'extract_time_s': float(extract_time),
        'compute_separation_time_s': float(compute_time),
        'fit_time_s': float(fit_time),
        'calibrate_time_s': float(calibrate_time),
        'throughput_samples_per_sec': float(throughput),
        'peak_memory_mb': float(peak_memory_mb),
        'num_layers': n_layers,
        'layer_names': all_layer_names,
        'layer_weights': optimal_weights.tolist(),
        'bias': float(optimal_bias),
        'note': 'Uses DAC-style weighted ensemble of per-layer geometric separation scores',
        **ood_metrics,  # Add OOD metrics
    }


def run_geometric_with_dac_features(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    train_raw: np.ndarray,
    val_raw: np.ndarray,
    test_raw: np.ndarray,
    device: torch.device,
    layer_index: int,
    batch_size: int = 128,
    target_dimension: int = 256
) -> Dict[str, Any]:
    """
    DEPRECATED: This is an ORACLE method that selects the best of 5 layers post-hoc.
    
    This function runs geometric calibration on each of DAC's 5 selected layers and
    picks the best one after seeing test performance. This is unfair to compare against
    methods that must commit to their configuration before seeing test performance.
    
    Use run_sgc_with_dac_preprocessing() instead for a fair comparison.
    
    Run Geometric calibration using DAC's feature extraction.
    
    This isolates whether DAC's preprocessing (spatial averaging, L2 norm) helps.
    """
    logger.info("\n" + "="*80)
    logger.info(f"GEOMETRIC WITH DAC FEATURES (Layer {layer_index+1})")
    logger.info("="*80)
    logger.info("Testing: Geometric algorithm with DAC's L2-normalized features")
    
    # Get DAC configuration
    target_layers = get_dac_target_layers(model_name, model)
    logger.info(f"Using DAC layer: {target_layers[layer_index]}")
    
    # Extract features using DAC's method (with its preprocessing)
    logger.info("\n1. Extracting features using DAC's method...")
    start_extract = time.perf_counter()
    
    train_features_all, _, _ = extract_dac_features_memory_efficient(
        model, train_loader, target_layers, device, max_layers_per_batch=4
    )
    val_features_all, _, _ = extract_dac_features_memory_efficient(
        model, val_loader, target_layers, device, max_layers_per_batch=4
    )
    test_features_all, _, _ = extract_dac_features_memory_efficient(
        model, test_loader, target_layers, device, max_layers_per_batch=4
    )
    
    # Select specific layer
    train_features = train_features_all[layer_index]
    val_features = val_features_all[layer_index]
    test_features = test_features_all[layer_index]
    
    extract_time = time.perf_counter() - start_extract
    logger.info(f"Feature extraction time: {extract_time:.2f}s")
    logger.info(f"  Features shape: {train_features.shape}")
    
    # Convert to numpy (DAC returns torch tensors)
    train_features_np = train_features.numpy() if isinstance(train_features, torch.Tensor) else train_features
    val_features_np = val_features.numpy() if isinstance(val_features, torch.Tensor) else val_features
    test_features_np = test_features.numpy() if isinstance(test_features, torch.Tensor) else test_features
    
    # Apply compression if needed to avoid OOM errors (especially for Tiny ImageNet)
    original_dim = train_features_np.shape[1]
    if target_dimension is not None and original_dim > target_dimension:
        logger.info(f"\nApplying random projection compression: {original_dim} -> {target_dimension} dims")
        logger.info(f"  Original shape: {train_features_np.shape}")
        
        from utils.compression_utils import SmartCompression
        compressor = SmartCompression(
            method='random_projection',
            target_dims=target_dimension,
            random_state=42 + layer_index  # Reproducible per layer
        )
        
        # Fit on train, transform all splits
        train_features_np = compressor(train_features_np, train=True)
        val_features_np = compressor(val_features_np, train=False)
        test_features_np = compressor(test_features_np, train=False)
        
        logger.info(f"  Compressed shape: {train_features_np.shape}")
    else:
        logger.info(f"  No compression needed (dim={original_dim}, target={target_dimension})")
    
    # Fit geometric calibrator (using DAC's features)
    logger.info("\n2. Fitting geometric calibrator...")
    start_fit = time.perf_counter()
    
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features_np,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,
        device=str(device)
    )
    
    geo_cal.fit(
        X_val_embed=val_features_np,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size
    )
    
    fit_time = time.perf_counter() - start_fit
    logger.info(f"Geometric fitting time: {fit_time:.2f}s")
    
    # Calibrate test set
    logger.info("\n3. Calibrating test set...")
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    start_calibrate = time.perf_counter()
    calibrated_probs = geo_cal.calibrate_batched(
        X_test_embed=test_features_np,
        X_test_original=test_raw,
        batch_size=batch_size
    )
    calibrate_time = time.perf_counter() - start_calibrate
    
    throughput = len(test_labels) / calibrate_time
    peak_memory_mb = 0
    if device.type == 'cuda':
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    logger.info(f"Calibration time: {calibrate_time:.2f}s")
    logger.info(f"Throughput: {throughput:.2f} samples/sec")
    
    # Calculate metrics
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
    brier = calculate_brier_score(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)
    
    logger.info(f"\nResults:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Adaptive ECE: {adaptive_ece:.6f}")
    logger.info(f"  Brier: {brier:.6f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    
    return {
        'method': f'Geometric + DAC Features (L{layer_index+1})',
        'ece': float(ece),
        'adaptive_ece': float(adaptive_ece),
        'calibration_mce': float(calibration_mce),
        'brier': float(brier),
        'accuracy': float(accuracy),
        'extract_time_s': float(extract_time),
        'fit_time_s': float(fit_time),
        'calibrate_time_s': float(calibrate_time),
        'throughput_samples_per_sec': float(throughput),
        'peak_memory_mb': float(peak_memory_mb),
        'layer_name': target_layers[layer_index],
        'note': 'Uses DAC feature extraction with Geometric calibration algorithm'
    }


def run_sgc_with_dac_preprocessing(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    num_layers: int = 6,
    target_dim: int = 256,
    batch_size: int = 128,
    seed: int = 42,
    ood_raw: np.ndarray = None,  # OOD data for evaluation
    ood_labels: np.ndarray = None,  # OOD labels for ECE calculation
    scoring_method: str = 'separation',  # Options: 'separation' or 'trust_score'
    precomputed_train_features: np.ndarray = None,  # Optional pre-extracted features
    precomputed_val_features: np.ndarray = None,  # Optional pre-extracted features
    precomputed_test_features: np.ndarray = None,  # Optional pre-extracted features
    precomputed_selected_layers: List[str] = None  # Optional pre-selected layers
) -> Dict[str, Any]:
    """
    SGC with DAC's preprocessing instead of SPP+JL.

    JL step now uses a deterministic Torch-based Gaussian projection on GPU
    instead of sklearn's GaussianRandomProjection.
    """
    """
    SGC with DAC's preprocessing instead of SPP+JL.
    
    Pipeline:
    1. Randomly select L layers (same as SGC)
    2. For each layer: extract with spatial average pooling (via extract_dac_features_memory_efficient)
    3. Apply L2 normalization per layer (DAC's preprocessing includes L2 normalization)
    4. Concatenate all layer features
    5. Apply JL projection to target_dim if needed
    6. L2 normalize final vector (after projection, in case projection changed norms)
    7. Geometric separation + isotonic regression (same as SGC)
    
    NOTE: extract_dac_features_memory_efficient only applies spatial pooling, not L2 normalization.
    We apply L2 normalization explicitly to match DAC's full preprocessing pipeline.
    
    This isolates: Does DAC's preprocessing help vs SPP?
    
    Args:
        model: PyTorch model
        model_name: Model architecture name
        dataset_name: Dataset name
        model_adapter: Model adapter for geometric calibrator
        train_raw, val_raw, test_raw: Raw input data
        train_labels, val_labels, test_labels: Labels
        device: Torch device
        num_layers: Number of random layers to select (L)
        target_dim: Target dimension after JL projection (d)
        batch_size: Batch size for data loading
        seed: Random seed for layer selection
        
    Returns:
        Dict with ECE, accuracy, timing, and metadata
    """
    logger.info("\n" + "="*80)
    logger.info(f"SGC WITH DAC PREPROCESSING (L={num_layers}, d={target_dim})")
    logger.info("="*80)
    logger.info("Testing: SGC pipeline with DAC preprocessing (spatial avg + L2) instead of SPP+JL")
    
    # Initialize loaders for later optional reuse (e.g., OOD compressor fitting)
    train_loader = None
    val_loader = None
    test_loader = None
    
    # Small helper: Torch-based Gaussian random projection (JL) on GPU/CPU
    class TorchGaussianRandomProjectionJL:
        """
        Deterministic Gaussian random projection implemented in PyTorch.

        Matches sklearn.GaussianRandomProjection scaling:
          components_ ~ N(0, 1 / sqrt(n_components))
        so each entry has variance 1 / n_components.
        """

        def __init__(self, in_dim: int, out_dim: int, seed: int, device: torch.device):
            self.in_dim = int(in_dim)
            self.out_dim = int(out_dim)
            self.seed = int(seed)
            self.device = device

            # std = 1 / sqrt(n_components)
            scale = 1.0 / (float(self.out_dim) ** 0.5)

            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)

            # (out_dim, in_dim) so that X @ R^T has shape (N, out_dim)
            self.R = torch.randn(
                (self.out_dim, self.in_dim),
                device=device,
                dtype=torch.float32,
                generator=gen,
            ) * scale

        def project_np(self, x_np: np.ndarray) -> np.ndarray:
            """Project a NumPy array (N, in_dim) -> (N, out_dim), return NumPy float32."""
            x_t = torch.from_numpy(x_np).to(self.device, dtype=torch.float32)
            y_t = torch.nn.functional.linear(x_t, self.R)
            return y_t.detach().cpu().numpy().astype(np.float32, copy=False)

    # Check if precomputed features are provided
    use_precomputed = (
        precomputed_train_features is not None and
        precomputed_val_features is not None and
        precomputed_test_features is not None and
        precomputed_selected_layers is not None
    )
    
    # Initialize train_features_concat to None - will be set if needed for OOD evaluation
    train_features_concat = None
    
    # JL projector (if needed) – shared between ID and OOD paths
    rp_jl: Optional[TorchGaussianRandomProjectionJL] = None

    if use_precomputed:
        logger.info("Using precomputed features (skipping extraction)")
        train_features = precomputed_train_features
        val_features = precomputed_val_features
        test_features = precomputed_test_features
        selected_layers = precomputed_selected_layers
        extract_time = 0.0  # Extraction time not available for precomputed features
        logger.info(f"Using pre-selected layers: {selected_layers}")
    else:
        # Discover ALL intermediate layers
        is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
        if is_dinov2:
            input_shape = (1, 3, 224, 224)
        elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
            input_shape = (1, 3, 64, 64)
        else:
            input_shape = (1, 3, 32, 32)
        
        discovered_layers = normalize_discovered_layers(
            model, discover_model_layers(model, device=device, input_shape=input_shape)
        )
        # Filter out activation and non-feature-bearing layers
        filtered_layers, excluded_layers = filter_non_feature_layers(discovered_layers, model)
        all_layer_names = [d["name"] for d in filtered_layers]
        logger.info(f"Discovered {len(discovered_layers)} total layers, {len(all_layer_names)} feature-bearing layers after filtering")
        
        # Randomly sample num_layers (same seed as SGC for fair comparison)
        np.random.seed(seed)
        if num_layers > len(all_layer_names):
            logger.warning(f"Requested {num_layers} but only {len(all_layer_names)} available")
            selected_layers = all_layer_names
        else:
            selected_layers = list(np.random.choice(all_layer_names, size=num_layers, replace=False))
        
        logger.info(f"Selected {len(selected_layers)} layers: {selected_layers}")
        
        # OPTIMIZATION: Use cached loaders if available, otherwise create new ones
        if train_loader is None:
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=batch_size, shuffle=False
            )
        if val_loader is None:
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                batch_size=batch_size, shuffle=False
            )
        if test_loader is None:
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
                batch_size=batch_size, shuffle=False
            )
        
        # Extract features using DAC preprocessing for each layer
        logger.info("\n1. Extracting features with DAC preprocessing (spatial avg + L2)...")
        start_extract = time.perf_counter()
        
        train_features_all, _, _ = extract_dac_features_memory_efficient(
            model, train_loader, selected_layers, device, max_layers_per_batch=4
        )
        val_features_all, _, _ = extract_dac_features_memory_efficient(
            model, val_loader, selected_layers, device, max_layers_per_batch=4
        )
        test_features_all, _, _ = extract_dac_features_memory_efficient(
            model, test_loader, selected_layers, device, max_layers_per_batch=4
        )
        
        # Apply L2 normalization per layer
        # NOTE: extract_dac_features_memory_efficient only applies spatial pooling (global avg),
        # NOT L2 normalization. We apply L2 normalization here to match DAC's preprocessing pipeline.
        logger.info("2. Applying L2 normalization per layer...")
        train_features_normalized = []
        val_features_normalized = []
        test_features_normalized = []
        
        for i in range(len(selected_layers)):
            train_feat = train_features_all[i].numpy() if isinstance(train_features_all[i], torch.Tensor) else train_features_all[i]
            val_feat = val_features_all[i].numpy() if isinstance(val_features_all[i], torch.Tensor) else val_features_all[i]
            test_feat = test_features_all[i].numpy() if isinstance(test_features_all[i], torch.Tensor) else test_features_all[i]
            
            train_features_normalized.append(l2_normalize_np(train_feat))
            val_features_normalized.append(l2_normalize_np(val_feat))
            test_features_normalized.append(l2_normalize_np(test_feat))
        
        # Concatenate all layers
        logger.info("3. Concatenating layer features...")
        train_features_concat = np.concatenate(train_features_normalized, axis=1)
        val_features_concat = np.concatenate(val_features_normalized, axis=1)
        test_features_concat = np.concatenate(test_features_normalized, axis=1)
        
        logger.info(f"  Concatenated shape: {train_features_concat.shape}")
        
        # Apply JL projection if needed
        original_dim = train_features_concat.shape[1]
        if target_dim is not None and original_dim > target_dim:
            logger.info(f"4. Applying JL projection (Torch, GPU): {original_dim} -> {target_dim} dims")
            rp_jl = TorchGaussianRandomProjectionJL(
                in_dim=original_dim,
                out_dim=target_dim,
                seed=seed,
                device=device,
            )
            train_features = rp_jl.project_np(train_features_concat)
            val_features = rp_jl.project_np(val_features_concat)
            test_features = rp_jl.project_np(test_features_concat)
        else:
            logger.info(f"  No compression needed (dim={original_dim}, target={target_dim})")
            train_features = train_features_concat
            val_features = val_features_concat
            test_features = test_features_concat
        
        # Final L2 normalization (after JL projection, in case projection changed norms)
        logger.info("5. Applying final L2 normalization...")
        train_features = l2_normalize_np(train_features)
        val_features = l2_normalize_np(val_features)
        test_features = l2_normalize_np(test_features)
        
        extract_time = time.perf_counter() - start_extract
        logger.info(f"Feature extraction time: {extract_time:.2f}s")
    
    logger.info(f"Final feature shape: {train_features.shape}")
    
    # Fit geometric calibrator
    logger.info("\n6. Fitting geometric calibrator...")
    start_fit = time.perf_counter()
    
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,
        device=str(device),
        scoring_method=scoring_method  # Use separation or trust_score
    )
    
    geo_cal.fit(
        X_val_embed=val_features,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size
    )
    
    fit_time = time.perf_counter() - start_fit
    
    # Calibrate test set
    logger.info("7. Calibrating test set...")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    start_calibrate = time.perf_counter()
    calibrated_probs = geo_cal.calibrate_batched(
        X_test_embed=test_features,
        X_test_original=test_raw,
        batch_size=batch_size
    )
    calibrate_time = time.perf_counter() - start_calibrate
    
    throughput = len(test_labels) / calibrate_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0
    
    # Metrics
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
    brier = calculate_brier_score(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)
    
    logger.info(f"\nSGC with DAC Preprocessing Results:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Adaptive ECE: {adaptive_ece:.6f}")
    logger.info(f"  Brier: {brier:.6f}")
    logger.info(f"  Accuracy: {accuracy:.4f}%")
    
    # OOD Evaluation
    ood_metrics = {}
    if ood_raw is not None:
        logger.info("\nComputing OOD metrics on SVHN...")
        
        ood_loader = DataLoader(
            TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
            batch_size=batch_size, shuffle=False
        )
        
        # Extract OOD features using DAC preprocessing (same as ID)
        ood_features_all, _, _ = extract_dac_features_memory_efficient(
            model, ood_loader, selected_layers, device, max_layers_per_batch=4
        )
        
        # Apply L2 normalization per layer
        ood_features_normalized = []
        for i in range(len(selected_layers)):
            ood_feat = ood_features_all[i].numpy() if isinstance(ood_features_all[i], torch.Tensor) else ood_features_all[i]
            ood_features_normalized.append(l2_normalize_np(ood_feat))
        
        # Concatenate all layers
        ood_features_concat = np.concatenate(ood_features_normalized, axis=1)
        
        # Apply JL projection if needed (same as ID)
        if target_dim is not None and ood_features_concat.shape[1] > target_dim:
            logger.info(f"  Applying JL projection to OOD features (Torch, GPU): {ood_features_concat.shape[1]} -> {target_dim} dims")
            # Reuse the same JL matrix as ID; create if it was not needed earlier
            if rp_jl is None:
                rp_jl = TorchGaussianRandomProjectionJL(
                    in_dim=ood_features_concat.shape[1],
                    out_dim=target_dim,
                    seed=seed,
                    device=device,
                )
            ood_features = rp_jl.project_np(ood_features_concat)
        else:
            ood_features = ood_features_concat
        
        # Final L2 normalization
        ood_features = l2_normalize_np(ood_features)
        
        # Calibrate OOD using fitted calibrator
        ood_calibrated_probs = geo_cal.calibrate_batched(
            X_test_embed=ood_features,
            X_test_original=ood_raw,
            batch_size=batch_size
        )
        
        # Compute metrics
        ood_metrics = compute_ood_metrics(
            calibrated_probs_id=calibrated_probs,
            calibrated_probs_ood=ood_calibrated_probs,
            id_labels=test_labels,
            ood_labels=ood_labels,
        )
        
        logger.info(f"  OOD AUROC: {ood_metrics['ood_auroc']:.4f}")
        logger.info(f"  OOD FPR@95: {ood_metrics['ood_fpr95']:.4f}")
        logger.info(f"  Confidence gap: {ood_metrics['confidence_gap']:.4f} (ID: {ood_metrics['id_confidence_mean']:.4f}, OOD: {ood_metrics['ood_confidence_mean']:.4f})")
    
    return {
        'method': f'SGC with DAC Preprocessing (L={num_layers}, d={target_dim})',
        'ece': float(ece),
        'adaptive_ece': float(adaptive_ece),
        'calibration_mce': float(calibration_mce),
        'brier': float(brier),
        'accuracy': float(accuracy),
        'num_layers': num_layers,
        'target_dim': target_dim,
        'selected_layers': selected_layers,
        'extraction_time_s': float(extract_time),
        'fit_time_s': float(fit_time),
        'calibrate_time_s': float(calibrate_time),
        'throughput_samples_per_sec': float(throughput),
        'peak_memory_mb': float(peak_memory_mb),
        'seed': seed,
        'note': 'SGC with DAC preprocessing (spatial avg + L2) instead of SPP+JL',
        'calibrator_params': geo_cal.get_params(),
        **ood_metrics,  # Add OOD metrics
    }


def run_random_single_layer_with_dac_preprocessing(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    target_dim: int = 256,
    batch_size: int = 128,
    seed: int = 42,
    ood_raw: np.ndarray = None,  # OOD data for evaluation
    ood_labels: np.ndarray = None  # OOD labels for ECE calculation
) -> Dict[str, Any]:
    """
    Single randomly selected layer with DAC preprocessing + Geometric scoring.
    
    Pipeline:
    1. Randomly select 1 layer from all available layers
    2. Extract features with spatial average pooling (via extract_dac_features_memory_efficient)
    3. Apply L2 normalization (DAC's preprocessing includes L2 normalization)
    4. Apply JL projection if needed to reduce to target_dim
    5. Apply final L2 normalization (after projection, in case projection changed norms)
    6. Geometric separation + isotonic regression
    
    NOTE: extract_dac_features_memory_efficient only applies spatial pooling, not L2 normalization.
    We apply L2 normalization explicitly to match DAC's full preprocessing pipeline.
    
    This isolates: Does multi-layer aggregation matter?
    Comparison: If this ≈ SGC, then aggregation doesn't help.
                If this << SGC, then aggregation is important.
    
    Args:
        model: PyTorch model
        model_name: Model architecture name
        dataset_name: Dataset name
        model_adapter: Model adapter for geometric calibrator
        train_raw, val_raw, test_raw: Raw input data
        train_labels, val_labels, test_labels: Labels
        device: Torch device
        target_dim: Target dimension after JL projection (d)
        batch_size: Batch size for data loading
        seed: Random seed for layer selection
        
    Returns:
        Dict with ECE, accuracy, timing, and metadata
    """
    logger.info("\n" + "="*80)
    logger.info(f"RANDOM SINGLE LAYER WITH DAC PREPROCESSING (d={target_dim})")
    logger.info("="*80)
    logger.info("Testing: Single random layer with DAC preprocessing + Geometric scoring")
    logger.info("This isolates: Does multi-layer aggregation matter?")
    
    # Small helper: Torch-based Gaussian random projection (JL) on GPU/CPU
    class TorchGaussianRandomProjectionJL:
        """
        Deterministic Gaussian random projection implemented in PyTorch.

        Matches sklearn.GaussianRandomProjection scaling:
          components_ ~ N(0, 1 / sqrt(n_components))
        so each entry has variance 1 / n_components.
        """

        def __init__(self, in_dim: int, out_dim: int, seed: int, device: torch.device):
            self.in_dim = int(in_dim)
            self.out_dim = int(out_dim)
            self.seed = int(seed)
            self.device = device

            scale = 1.0 / (float(self.out_dim) ** 0.5)
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)

            self.R = torch.randn(
                (self.out_dim, self.in_dim),
                device=device,
                dtype=torch.float32,
                generator=gen,
            ) * scale

        def project_np(self, x_np: np.ndarray) -> np.ndarray:
            x_t = torch.from_numpy(x_np).to(self.device, dtype=torch.float32)
            y_t = torch.nn.functional.linear(x_t, self.R)
            return y_t.detach().cpu().numpy().astype(np.float32, copy=False)

    # Used only in OOD evaluation; initialize to avoid UnboundLocalError
    ood_loader = None
    
    # Discover ALL intermediate layers
    is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
    if is_dinov2:
        input_shape = (1, 3, 224, 224)
    elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
        input_shape = (1, 3, 64, 64)
    else:
        input_shape = (1, 3, 32, 32)
    
    discovered_layers = normalize_discovered_layers(
        model, discover_model_layers(model, device=device, input_shape=input_shape)
    )
    # Filter out activation and non-feature-bearing layers
    filtered_layers, excluded_layers = filter_non_feature_layers(discovered_layers, model)
    all_layer_names = [d["name"] for d in filtered_layers]
    logger.info(f"Discovered {len(discovered_layers)} total layers, {len(all_layer_names)} feature-bearing layers after filtering")
    
    # Randomly sample 1 layer (same seed as SGC for fair comparison)
    np.random.seed(seed)
    selected_layer = [np.random.choice(all_layer_names)]
    
    logger.info(f"Selected layer: {selected_layer[0]}")
    
    # Create dataloaders
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
    
    # Extract features using DAC preprocessing
    logger.info("\n1. Extracting features with DAC preprocessing (spatial avg + L2)...")
    start_extract = time.perf_counter()
    
    train_features_all, _, _ = extract_dac_features_memory_efficient(
        model, train_loader, selected_layer, device, max_layers_per_batch=4
    )
    val_features_all, _, _ = extract_dac_features_memory_efficient(
        model, val_loader, selected_layer, device, max_layers_per_batch=4
    )
    test_features_all, _, _ = extract_dac_features_memory_efficient(
        model, test_loader, selected_layer, device, max_layers_per_batch=4
    )
    
    # Get single layer features
    # NOTE: extract_dac_features_memory_efficient only applies spatial pooling (global avg),
    # NOT L2 normalization. We apply L2 normalization here to match DAC's preprocessing pipeline.
    train_features = train_features_all[0].numpy() if isinstance(train_features_all[0], torch.Tensor) else train_features_all[0]
    val_features = val_features_all[0].numpy() if isinstance(val_features_all[0], torch.Tensor) else val_features_all[0]
    test_features = test_features_all[0].numpy() if isinstance(test_features_all[0], torch.Tensor) else test_features_all[0]
    
    logger.info("2. Applying L2 normalization...")
    train_features = l2_normalize_np(train_features)
    val_features = l2_normalize_np(val_features)
    test_features = l2_normalize_np(test_features)
    
    logger.info(f"  Feature shape: {train_features.shape}")
    
    # Apply JL projection if needed
    original_dim = train_features.shape[1]
    rp_jl: Optional[TorchGaussianRandomProjectionJL] = None  # JL projector reused for OOD
    if target_dim is not None and original_dim > target_dim:
        logger.info(f"3. Applying JL projection (Torch, GPU): {original_dim} -> {target_dim} dims")
        rp_jl = TorchGaussianRandomProjectionJL(
            in_dim=original_dim,
            out_dim=target_dim,
            seed=seed,
            device=device,
        )
        train_features = rp_jl.project_np(train_features)
        val_features = rp_jl.project_np(val_features)
        test_features = rp_jl.project_np(test_features)
    else:
        logger.info(f"  No compression needed (dim={original_dim}, target={target_dim})")
    
    # Final L2 normalization (in case compression changed norms)
    logger.info("4. Applying final L2 normalization...")
    train_features = l2_normalize_np(train_features)
    val_features = l2_normalize_np(val_features)
    test_features = l2_normalize_np(test_features)
    
    extract_time = time.perf_counter() - start_extract
    logger.info(f"Feature extraction time: {extract_time:.2f}s")
    logger.info(f"Final feature shape: {train_features.shape}")
    
    # Fit geometric calibrator
    logger.info("\n5. Fitting geometric calibrator...")
    start_fit = time.perf_counter()
    
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,
        device=str(device)
    )
    
    geo_cal.fit(
        X_val_embed=val_features,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=batch_size
    )
    
    fit_time = time.perf_counter() - start_fit
    
    # Calibrate test set
    logger.info("6. Calibrating test set...")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    start_calibrate = time.perf_counter()
    calibrated_probs = geo_cal.calibrate_batched(
        X_test_embed=test_features,
        X_test_original=test_raw,
        batch_size=batch_size
    )
    calibrate_time = time.perf_counter() - start_calibrate
    
    throughput = len(test_labels) / calibrate_time
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0
    
    # Metrics
    ece = calculate_ece(calibrated_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
    calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
    brier = calculate_brier_score(calibrated_probs, test_labels)
    accuracy = calculate_accuracy(calibrated_probs, test_labels)
    
    logger.info(f"\nRandom Single Layer with DAC Preprocessing Results:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Adaptive ECE: {adaptive_ece:.6f}")
    logger.info(f"  Brier: {brier:.6f}")
    logger.info(f"  Accuracy: {accuracy:.4f}%")
    
    # OOD Evaluation
    ood_metrics = {}
    if ood_raw is not None:
        logger.info("\nComputing OOD metrics on SVHN...")
        
        # OPTIMIZATION: Use cached OOD loader if available
        if ood_loader is None:
            ood_loader = DataLoader(
                TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
                batch_size=batch_size, shuffle=False
            )
        
        # Extract OOD features with DAC preprocessing
        ood_features_all, _, _ = extract_dac_features_memory_efficient(
            model, ood_loader, selected_layer, device, max_layers_per_batch=4
        )
        
        # Get single layer features
        ood_features = ood_features_all[0].numpy() if isinstance(ood_features_all[0], torch.Tensor) else ood_features_all[0]
        
        # Apply L2 normalization
        ood_features = l2_normalize_np(ood_features)
        
        # Apply JL projection if needed (same as ID)
        if target_dim is not None and ood_features.shape[1] > target_dim:
            # Reuse the same JL projector that was used for ID features
            if rp_jl is None:
                rp_jl = TorchGaussianRandomProjectionJL(
                    in_dim=ood_features.shape[1],
                    out_dim=target_dim,
                    seed=seed,
                    device=device,
                )
            ood_features = rp_jl.project_np(ood_features)
        
        # Final L2 normalization
        ood_features = l2_normalize_np(ood_features)
        
        # Calibrate OOD using fitted calibrator
        ood_calibrated_probs = geo_cal.calibrate_batched(
            X_test_embed=ood_features,
            X_test_original=ood_raw,
            batch_size=batch_size
        )
        
        # Compute metrics
        ood_metrics = compute_ood_metrics(
            calibrated_probs_id=calibrated_probs,
            calibrated_probs_ood=ood_calibrated_probs,
            id_labels=test_labels,
            ood_labels=ood_labels,
        )
        
        logger.info(f"  OOD AUROC: {ood_metrics['ood_auroc']:.4f}")
        logger.info(f"  OOD FPR@95: {ood_metrics['ood_fpr95']:.4f}")
        logger.info(f"  Confidence gap: {ood_metrics['confidence_gap']:.4f} (ID: {ood_metrics['id_confidence_mean']:.4f}, OOD: {ood_metrics['ood_confidence_mean']:.4f})")
    
    return {
        'method': f'Random Single Layer with DAC Preprocessing (d={target_dim})',
        'ece': float(ece),
        'adaptive_ece': float(adaptive_ece),
        'calibration_mce': float(calibration_mce),
        'brier': float(brier),
        'accuracy': float(accuracy),
        'target_dim': target_dim,
        'selected_layer': selected_layer[0],
        'extraction_time_s': float(extract_time),
        'fit_time_s': float(fit_time),
        'calibrate_time_s': float(calibrate_time),
        'throughput_samples_per_sec': float(throughput),
        'peak_memory_mb': float(peak_memory_mb),
        'seed': seed,
        'note': 'Single random layer with DAC preprocessing (spatial avg + L2) + Geometric scoring',
        'calibrator_params': geo_cal.get_params(),
        **ood_metrics,  # Add OOD metrics
    }


def run_dac_with_sgc_aggregated_features(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    target_dim: int = 256,
    seed: int = 42,
    pooling_mode: str = 'max',
    ood_raw: np.ndarray = None,  # OOD data for evaluation
    ood_labels: np.ndarray = None,  # OOD labels for ECE calculation
    output_dir: str = None,  # Optional output directory for per-sample data saving
    training_method: str = None,  # Training method for filename construction
    dac_knn_backend: str = 'faiss'  # kNN backend: 'faiss', 'torch', or 'both'
) -> Dict[str, Any]:
    """ 
    Run DAC calibration using SGC's feature extraction pipeline (project-then-sum aggregation).
    
    This tests: "What if we use SGC's aggregated features (SPP + project-then-sum + L2) 
    with DAC's density-based weighting algorithm?"
    
    IMPORTANT: This is NOT a fair layer selection comparison:
    - SGC: randomly select L layers -> SPP+JL per layer -> project-then-sum -> single 256d vector
    - This: uses SGC's aggregated 256d vector + logits -> DAC learns only 2 weights (aggregated + logits)
    
    This gives DAC only 2 inputs instead of multiple separate layers, so it's not testing
    whether DAC's algorithm works with random layer selection.
    
    For a fair comparison, see run_dac_with_random_layer_selection().
    
    This uses the SAME feature pipeline as standard SGC:
    - SPP only per layer (no per-layer JL)
    - Independent random projection per layer
    - Project-then-sum aggregation
    - L2 normalization
    """
    logger.info("\n" + "="*80)
    logger.info(f"DAC WITH SGC AGGREGATED FEATURES (d={target_dim})")
    logger.info("="*80)
    logger.info("Testing: DAC algorithm with SGC's aggregated (project-then-sum) features")
    logger.info("NOTE: This gives DAC only 2 inputs (aggregated vector + logits), not a fair comparison")
    
    # Get layer names (use same layers as standard SGC would use)
    all_layer_names = get_layer_names_for_model(model_name, model)
    logger.info(f"Using layers: {all_layer_names}")
    
    # Get DAC configuration
    k_value = get_dac_k_value(dataset_name)
    logger.info(f"DAC k-value: {k_value}")
    
    # Extract features using SGC pipeline (NOT the old per-layer SPP+JL)
    logger.info("\n1. Extracting features using SGC pipeline...")
    train_features, val_features, test_features, extraction_info = extract_and_aggregate_sgc_features(
        model=model,
        layer_names=all_layer_names,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        target_dim=target_dim,
        seed=seed,
        pooling_mode=pooling_mode
    )
    
    extract_time = extraction_info['extraction_time_s']
    logger.info(f"Feature extraction time: {extract_time:.2f}s")
    logger.info(f"Feature shape: {train_features.shape}")
    
    # Get logits for DAC (need these for calibration)
    logger.info("\n2. Extracting logits...")
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
    val_logits, val_labels_torch = get_logits_and_labels(val_loader)
    test_logits, test_labels_torch = get_logits_and_labels(test_loader)
    
    val_labels = val_labels_torch.numpy() if isinstance(val_labels_torch, torch.Tensor) else val_labels_torch
    test_labels = test_labels_torch.numpy() if isinstance(test_labels_torch, torch.Tensor) else test_labels_torch
    
    # DAC expects a LIST of feature arrays (one per "layer")
    # Since SGC produces a single aggregated vector, we pass it as a single-element list
    # Plus logits as another "layer" (matching DAC paper setup)
    logger.info("\n3. Preparing features for DAC...")
    
    # Convert to torch tensors for DAC
    train_features_torch = torch.from_numpy(train_features).float()
    val_features_torch = torch.from_numpy(val_features).float()
    test_features_torch = torch.from_numpy(test_features).float()
    
    # DAC input: [aggregated_sgc_features, logits]
    train_features_for_dac = [train_features_torch, train_logits]
    val_features_for_dac = [val_features_torch, val_logits]
    test_features_for_dac = [test_features_torch, test_logits]
    
    logger.info(f"DAC input: {len(train_features_for_dac)} layers")
    logger.info(f"  Layer 0 (SGC aggregated): {train_features_torch.shape}")
    logger.info(f"  Layer 1 (logits): {train_logits.shape}")
    
    # Fit DAC
    logger.info("\n4. Fitting DAC calibrator...")
    
    # Handle "both" mode: run both backends
    backends_to_run = ['faiss', 'torch'] if dac_knn_backend == 'both' else [dac_knn_backend]
    all_results = {}
    
    for backend in backends_to_run:
        logger.info(f"\n  Running DAC with {backend} backend...")
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_fit = time.perf_counter()
        
        dac = make_dac_calibrator(k=k_value, device=device, backend=backend)
        dac.fit(train_features_for_dac, val_features_for_dac, val_logits, val_labels)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        fit_time = time.perf_counter() - start_fit
        logger.info(f"  DAC fitting time ({backend}): {fit_time:.2f}s")
        
        # Calibrate test set
        logger.info(f"\n5. Calibrating test set ({backend})...")
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_calibrate = time.perf_counter()
        
        calibrated_probs = dac.calibrate(test_features_for_dac, test_logits)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        calibrate_time = time.perf_counter() - start_calibrate
        logger.info(f"  Calibration time ({backend}): {calibrate_time:.2f}s")
        
        throughput = len(test_labels) / calibrate_time
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0
        logger.info(f"  Throughput ({backend}): {throughput:.2f} samples/sec")
        
        # Extract DAC weights
        layer_weights = dac.weights[:-1].tolist() if dac.weights is not None else []
        bias = float(dac.weights[-1]) if dac.weights is not None else 0.0
        
        # Calculate metrics
        ece = calculate_ece(calibrated_probs, test_labels)
        adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
        calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
        brier = calculate_brier_score(calibrated_probs, test_labels)
        accuracy = calculate_accuracy(calibrated_probs, test_labels)
        
        logger.info(f"\n  Results ({backend}):")
        logger.info(f"    ECE: {ece:.6f}")
        logger.info(f"    Adaptive ECE: {adaptive_ece:.6f}")
        logger.info(f"    Brier: {brier:.6f}")
        logger.info(f"    Accuracy: {accuracy:.4f}")
        
        # OOD Evaluation
        ood_metrics = {}
        if ood_raw is not None:
            logger.info(f"\n  Computing OOD metrics on SVHN ({backend})...")
            
            # Create OOD loader
            ood_loader = DataLoader(
                TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
                batch_size=test_loader.batch_size, shuffle=False
            )
            
            # Extract OOD features using SGC pipeline (same as ID)
            _, _, ood_features, _ = extract_and_aggregate_sgc_features(
                model=model,
                layer_names=all_layer_names,
                train_loader=train_loader,  # Pass train_loader for compressor fitting
                val_loader=None,
                test_loader=ood_loader,
                device=device,
                target_dim=target_dim,
                seed=seed,  # Same seed ensures same projection matrix as ID
                pooling_mode=pooling_mode
            )
            
            # Get OOD logits
            ood_logits, _ = get_logits_and_labels(ood_loader)
            
            # Prepare features for DAC
            ood_features_torch = torch.from_numpy(ood_features).float()
            ood_features_for_dac = [ood_features_torch, ood_logits]
            
            # Calibrate OOD using fitted DAC calibrator
            ood_calibrated_probs = dac.calibrate(ood_features_for_dac, ood_logits)
            
            # Compute metrics
            ood_metrics = compute_ood_metrics(
                calibrated_probs_id=calibrated_probs,
                calibrated_probs_ood=ood_calibrated_probs,
                id_labels=test_labels,
                ood_labels=ood_labels,
            )
            
            logger.info(f"    OOD AUROC ({backend}): {ood_metrics['ood_auroc']:.4f}")
            logger.info(f"    OOD FPR@95 ({backend}): {ood_metrics['ood_fpr95']:.4f}")
            logger.info(f"    Confidence gap ({backend}): {ood_metrics['confidence_gap']:.4f} (ID: {ood_metrics['id_confidence_mean']:.4f}, OOD: {ood_metrics['ood_confidence_mean']:.4f})")
        
        # Save per-sample data for reliability diagrams and risk-coverage curves
        if output_dir is not None:
            method_suffix = '_torch' if backend == 'torch' else ''
            save_per_sample_data(
                calibrated_probs=calibrated_probs,
                test_labels=test_labels,
                method_name=f'dac_with_sgc_aggregated{method_suffix}',
                output_dir=output_dir,
                stability_scores=None,  # DAC doesn't use stability scores
                model_name=model_name,
                dataset_name=dataset_name,
                training_method=training_method,
                seed=seed
            )
        
        # Build result dict for this backend
        result_dict = {
            'method': f'DAC + SGC Aggregated Features (d={target_dim})',
            'ece': float(ece),
            'adaptive_ece': float(adaptive_ece),
            'calibration_mce': float(calibration_mce),
            'brier': float(brier),
            'accuracy': float(accuracy),
            'extract_time_s': float(extract_time),
            'fit_time_s': float(fit_time),
            'calibrate_time_s': float(calibrate_time),
            'throughput_samples_per_sec': float(throughput),
            'peak_memory_mb': float(peak_memory_mb),
            'k_value': int(k_value),
            'target_dim': target_dim,
            'num_layers': len(all_layer_names),
            'layer_names': all_layer_names,
            'layer_weights': layer_weights,
            'bias': bias,
            'knn_backend': backend,
            'knn_backend_impl': dac.__class__.__module__ + '.' + dac.__class__.__name__,
            'note': 'Uses SGC aggregated features (SPP + project-then-sum + L2) with DAC. NOT a fair layer selection comparison (only 2 inputs: aggregated + logits).',
            **ood_metrics,  # Add OOD metrics
        }
        
        # Store result for this backend
        if backend == 'faiss':
            if dac_knn_backend == 'both':
                all_results['dac_with_sgc_aggregated_features'] = result_dict
            else:
                all_results = result_dict  # Single backend mode (faiss)
        elif backend == 'torch':
            if dac_knn_backend == 'both':
                all_results['dac_with_sgc_aggregated_features_torch'] = result_dict
            else:
                all_results = result_dict  # Single backend mode (torch)
    
    # Return appropriate result based on mode
    if dac_knn_backend == 'both':
        # Return dict with both results (caller will handle storing under appropriate keys)
        return all_results
    else:
        # Single backend mode - return result directly
        return all_results


def run_dac_with_random_layer_selection(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_layers: int = 6,
    seed: int = 42,
    ood_raw: np.ndarray = None,  # OOD data for evaluation
    ood_labels: np.ndarray = None,  # OOD labels for ECE calculation
    batch_size: int = 128,  # For OOD loader
    output_dir: str = None,  # Optional output directory for per-sample data saving
    training_method: str = None,  # Training method for filename construction
    dac_knn_backend: str = 'faiss'  # kNN backend: 'faiss', 'torch', or 'both'
) -> Dict[str, Any]:
    """
    Run DAC calibration with RANDOMLY SELECTED layers instead of DAC's prescribed layers.
    
    This is a FAIR comparison to SGC's random layer selection (RLS):
    - SGC: randomly select L layers -> SPP+JL per layer -> project-then-sum -> geometric separation
    - This: randomly select L layers -> DAC preprocessing per layer -> DAC density weighting
    
    Both use the SAME random layer selection, but different:
    - Feature preprocessing (SGC: SPP+JL vs DAC: spatial avg + L2 norm)
    - Calibration algorithm (SGC: geometric separation + isotonic vs DAC: density-based weighting)
    
    If this works well: random layer selection is universally good
    If this fails: DAC's algorithm is fragile and requires its specific layer selection
    
    Args:
        model: PyTorch model
        model_name: Model architecture name
        dataset_name: Dataset name
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to run on
        num_layers: Number of random layers to select (default: 6, matching SGC)
        seed: Random seed for layer selection (should match SGC seed for fair comparison)
    
    Returns:
        Dict with ECE, accuracy, timing, and metadata
    """
    logger.info("\n" + "="*80)
    logger.info(f"DAC WITH RANDOM LAYER SELECTION (L={num_layers})")
    logger.info("="*80)
    logger.info("Testing: DAC algorithm with randomly selected layers (fair comparison to SGC)")
    
    # Discover ALL intermediate layers (same as SGC does)
    # Determine input shape for layer discovery based on dataset
    is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
    if is_dinov2:
        input_shape = (1, 3, 224, 224)  # DINOv2 requires 224x224
    elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
        input_shape = (1, 3, 64, 64)  # Tiny ImageNet uses 64x64
    else:
        input_shape = (1, 3, 32, 32)  # Default for CIFAR (32x32)
    
    discovered_layers = normalize_discovered_layers(
        model, discover_model_layers(model, device=device, input_shape=input_shape)
    )
    # Filter out activation and non-feature-bearing layers
    filtered_layers, excluded_layers = filter_non_feature_layers(discovered_layers, model)
    all_layer_names = [d["name"] for d in filtered_layers]
    logger.info(f"Discovered {len(discovered_layers)} total layers, {len(all_layer_names)} feature-bearing layers after filtering")
    
    # Randomly sample num_layers (SAME logic as run_global_random_calibration)
    np.random.seed(seed)
    if num_layers > len(all_layer_names):
        logger.warning(f"Requested {num_layers} but only {len(all_layer_names)} available")
        selected_layers = all_layer_names
    else:
        selected_layers = list(np.random.choice(all_layer_names, size=num_layers, replace=False))
    
    logger.info(f"Selected {len(selected_layers)} random layers: {selected_layers}")
    
    # Get DAC configuration
    k_value = get_dac_k_value(dataset_name)
    logger.info(f"DAC k-value: {k_value}")
    
    # Extract logits once (they're the same regardless of which layer we extract)
    logger.info("\n1. Extracting logits...")
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
    
    train_logits_final, train_labels_torch = get_logits_and_labels(train_loader)
    val_logits_final, val_labels_torch = get_logits_and_labels(val_loader)
    test_logits_final, test_labels_torch = get_logits_and_labels(test_loader)
    
    val_labels = val_labels_torch.numpy() if isinstance(val_labels_torch, torch.Tensor) else val_labels_torch
    test_labels = test_labels_torch.numpy() if isinstance(test_labels_torch, torch.Tensor) else test_labels_torch
    
    # Extract features for EACH selected layer separately using DAC's preprocessing
    # This gives spatial averaging + L2 normalization per layer (DAC's standard preprocessing)
    logger.info("\n2. Extracting features from each selected layer using DAC preprocessing...")
    t_extract = time.perf_counter()
    
    train_features_list = []
    val_features_list = []
    test_features_list = []
    
    # Extract features for each layer separately
    for layer_idx, layer_name in enumerate(selected_layers):
        logger.info(f"  Extracting from layer {layer_idx+1}/{len(selected_layers)}: {layer_name}")
        
        # Use DAC's memory-efficient extraction for a single layer
        # This applies: spatial avg pooling (if 4D) + L2 normalization
        train_feats, _, _ = extract_dac_features_memory_efficient(
            model, train_loader, [layer_name], device
        )
        val_feats, _, _ = extract_dac_features_memory_efficient(
            model, val_loader, [layer_name], device
        )
        test_feats, _, _ = extract_dac_features_memory_efficient(
            model, test_loader, [layer_name], device
        )
        
        # extract_dac_features_memory_efficient returns a list with one element per layer
        # Since we passed [layer_name], we get one feature tensor
        train_features_list.append(train_feats[0])
        val_features_list.append(val_feats[0])
        test_features_list.append(test_feats[0])
    
    extract_time = time.perf_counter() - t_extract
    logger.info(f"Feature extraction time: {extract_time:.2f}s")
    logger.info(f"Extracted {len(train_features_list)} separate feature arrays")
    for i, feat in enumerate(train_features_list):
        logger.info(f"  Layer {i+1} ({selected_layers[i]}): {feat.shape}")
    
    # Add logits as the final "layer" (matching DAC paper setup)
    logger.info("\n3. Preparing features for DAC (L separate layers + logits)...")
    train_features_for_dac = train_features_list + [train_logits_final]
    val_features_for_dac = val_features_list + [val_logits_final]
    test_features_for_dac = test_features_list + [test_logits_final]
    
    logger.info(f"DAC input: {len(train_features_for_dac)} layers ({len(selected_layers)} feature layers + 1 logits layer)")
    logger.info(f"  DAC will learn {len(train_features_for_dac)} weights (one per layer + bias)")
    
    # Fit DAC
    logger.info("\n4. Fitting DAC calibrator...")
    
    # Handle "both" mode: run both backends
    backends_to_run = ['faiss', 'torch'] if dac_knn_backend == 'both' else [dac_knn_backend]
    all_results = {}
    
    for backend in backends_to_run:
        logger.info(f"\n  Running DAC with {backend} backend...")
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_fit = time.perf_counter()
        
        dac = make_dac_calibrator(k=k_value, device=device, backend=backend)
        dac.fit(train_features_for_dac, val_features_for_dac, val_logits_final, val_labels)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        fit_time = time.perf_counter() - start_fit
        logger.info(f"  DAC fitting time ({backend}): {fit_time:.2f}s")
        
        # Calibrate test set
        logger.info(f"\n5. Calibrating test set ({backend})...")
        
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_calibrate = time.perf_counter()
        
        calibrated_probs = dac.calibrate(test_features_for_dac, test_logits_final)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        calibrate_time = time.perf_counter() - start_calibrate
        logger.info(f"  Calibration time ({backend}): {calibrate_time:.2f}s")
        
        throughput = len(test_labels) / calibrate_time if calibrate_time > 0 else 0
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0
        logger.info(f"  Throughput ({backend}): {throughput:.2f} samples/sec")
        
        # Extract DAC weights
        layer_weights = dac.weights[:-1].tolist() if dac.weights is not None else []
        bias = float(dac.weights[-1]) if dac.weights is not None else 0.0
        
        # Calculate metrics
        ece = calculate_ece(calibrated_probs, test_labels)
        adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
        calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
        brier = calculate_brier_score(calibrated_probs, test_labels)
        accuracy = calculate_accuracy(calibrated_probs, test_labels)
        
        logger.info(f"\n  Results ({backend}):")
        logger.info(f"    ECE: {ece:.6f}")
        logger.info(f"    Adaptive ECE: {adaptive_ece:.6f}")
        logger.info(f"    Brier: {brier:.6f}")
        logger.info(f"    Accuracy: {accuracy:.4f}")
        
        # OOD Evaluation
        ood_metrics = {}
        if ood_raw is not None:
            logger.info(f"\n  Computing OOD metrics on SVHN ({backend})...")
            
            ood_loader = DataLoader(
                TensorDataset(torch.from_numpy(ood_raw), torch.zeros(len(ood_raw), dtype=torch.long)),
                batch_size=batch_size, shuffle=False
            )
            
            # Extract features using same layers as ID
            ood_features_list, ood_logits, _ = extract_dac_features_memory_efficient(
                model, ood_loader, selected_layers, device
            )
            
            # Format for DAC (same as ID)
            ood_features_for_dac = [f.numpy() if isinstance(f, torch.Tensor) else f for f in ood_features_list] + [ood_logits.numpy()]
            
            # Calibrate using fitted DAC
            ood_calibrated_probs = dac.calibrate(ood_features_for_dac, ood_logits.numpy())
            
            # Get test labels as numpy
            test_labels_np = test_labels_torch.numpy() if isinstance(test_labels_torch, torch.Tensor) else test_labels
            
            ood_metrics = compute_ood_metrics(
                calibrated_probs_id=calibrated_probs,
                calibrated_probs_ood=ood_calibrated_probs,
                id_labels=test_labels_np,
                ood_labels=ood_labels,
            )
            
            logger.info(f"    OOD AUROC ({backend}): {ood_metrics['ood_auroc']:.4f}")
            logger.info(f"    OOD FPR@95 ({backend}): {ood_metrics['ood_fpr95']:.4f}")
            logger.info(f"    Confidence gap ({backend}): {ood_metrics['confidence_gap']:.4f} (ID: {ood_metrics['id_confidence_mean']:.4f}, OOD: {ood_metrics['ood_confidence_mean']:.4f})")
        
        # Save per-sample data for reliability diagrams and risk-coverage curves
        if output_dir is not None:
            method_suffix = '_torch' if backend == 'torch' else ''
            save_per_sample_data(
                calibrated_probs=calibrated_probs,
                test_labels=test_labels,
                method_name=f'dac_with_random_layers{method_suffix}',
                output_dir=output_dir,
                stability_scores=None,  # DAC doesn't use stability scores
                model_name=model_name,
                dataset_name=dataset_name,
                training_method=training_method,
                seed=seed
            )
        
        # Build result dict for this backend
        result_dict = {
            'method': f'DAC + Random Layers (L={num_layers})',
            'ece': float(ece),
            'adaptive_ece': float(adaptive_ece),
            'calibration_mce': float(calibration_mce),
            'brier': float(brier),
            'accuracy': float(accuracy),
            'extraction_time_s': float(extract_time),
            'fit_time_s': float(fit_time),
            'calibrate_time_s': float(calibrate_time),
            'throughput_samples_per_sec': float(throughput),
            'peak_memory_mb': float(peak_memory_mb),
            'k_value': int(k_value),
            'num_layers': num_layers,
            'selected_layers': selected_layers,
            'seed': seed,
            'layer_weights': layer_weights,
            'bias': bias,
            'knn_backend': backend,
            'knn_backend_impl': dac.__class__.__module__ + '.' + dac.__class__.__name__,
            'note': 'Fair comparison to SGC: uses same random layer selection, but DAC preprocessing (spatial avg + L2) and DAC density-based calibration instead of SGC preprocessing and geometric separation',
            **ood_metrics,  # Add OOD metrics
        }
        
        # Store result for this backend
        if backend == 'faiss':
            if dac_knn_backend == 'both':
                all_results['dac_with_random_layer_selection'] = result_dict
            else:
                all_results = result_dict  # Single backend mode (faiss)
        elif backend == 'torch':
            if dac_knn_backend == 'both':
                all_results['dac_with_random_layer_selection_torch'] = result_dict
            else:
                all_results = result_dict  # Single backend mode (torch)
    
    # Return appropriate result based on mode
    if dac_knn_backend == 'both':
        # Return dict with both results (caller will handle storing under appropriate keys)
        return all_results
    else:
        # Single backend mode - return result directly
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Compare DAC vs Geometric Calibration')

    # Model configuration
    parser.add_argument('--model-name', type=str, required=True,
                        help='Model architecture (e.g., resnet18, densenet121)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., cifar10, cifar100)')
    parser.add_argument('--training-method', type=str, default='baseline',
                        help='Training method (e.g., baseline, augmix)')
    parser.add_argument('--corruption-type', type=str, default=None,
                        help='Corruption type (e.g., gaussian_noise, motion_blur). If None, use clean data.')
    parser.add_argument('--corruption-severity', type=int, default=None, choices=[1, 2, 3, 4, 5],
                        help='Corruption severity level (1-5). Required if corruption-type is set.')
    parser.add_argument('--global-random-layers', type=int, default=6,
                        help='Number of random layers to sample for Global Random calibration (deprecated: use --num-layers)')
    parser.add_argument('--global-random-compression', type=float, default=4.0,
                        help='Compression ratio for Global Random calibration (deprecated: use --target-dimension)')
    parser.add_argument('--target-dimension', type=int, default=256,
                        help='Target dimension d after projection (paper default: 256)')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of random layers L to sample (paper default: 6)')
    parser.add_argument('--seed', type=int, default=11,
                        help='Random seed')
    parser.add_argument('--force-recompute', action='store_true',
                        help='Force recompute even if results JSON already exists')
    parser.add_argument('--eval-corruptions', action='store_true',
                        help='Enable corruption evaluation mode (CIFAR-10/100 only)')
    parser.add_argument('--mce-only', action='store_true',
                        help='Only compute mCE on corruptions, skip clean ECE evaluation')
    parser.add_argument('--cifar-c-dir', type=str, default=None,
                        help='Path to CIFAR-C data directory (defaults to dataset-specific path if not provided)')

    # Paths
    parser.add_argument('--results-base-dir', type=str,
                        default='aaai_full_experiments/results',
                        help='Base directory for experiment results')
    parser.add_argument('--output-dir', type=str,
                        default='calibration_comparison',
                        help='Output directory for comparison results')

    # Device and batch size
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for data loading')
    parser.add_argument('--spp-pooling-mode', type=str, default='max',
                        choices=['avg', 'max'],
                        help='SPP pooling mode: avg (average pooling) or max (max pooling). Default: avg')
    parser.add_argument('--skip-single-layers', action='store_true',
                        help='Skip single-layer geometric calibration experiments (geometric_original section)')
    parser.add_argument('--ood-only', action='store_true',
                        help='Only compute OOD metrics for existing results, skip all other experiments')
    parser.add_argument('--baselines-only', action='store_true',
                        help='Only run standard baseline calibrators, update standard_baselines, and exit')
    parser.add_argument('--compute-brier-top-label', action='store_true',
                        help='Compute and save top-label Brier for baseline calibrators as top_label_brier')
    parser.add_argument('--force-rerun-geometric', action='store_true',
                       help='Force re-run all geometric calibration experiments (uses rank-based normalization instead of buggy percentile normalization)')
    parser.add_argument('--normalization-method', type=str, default='rank',
                        choices=['rank', 'linear', 'percentile'],
                        help='Score normalization for geometric calibrator')
    parser.add_argument('--deep-ensemble-scan', action='store_true',
                        help='Run only the deep ensemble scan and skip all other experiments')
    parser.add_argument('--ensemble-size', type=int, default=3,
                        help='Number of models to include in the deep ensemble scan')
    parser.add_argument('--eval-ensemble', action='store_true',
                        help='Evaluate deep ensemble baseline (requires --ensemble-seeds)')
    parser.add_argument('--ensemble-seeds', type=str, default='11 12 13',
                        help='Space-separated list of seeds for ensemble evaluation (default: "11 12 13")')
    parser.add_argument('--method', type=str, action='append',
                       help='Specific geometric method to run (can be repeated). If provided, only these methods will run. Valid methods: sgc_with_dac_preprocessing_separation, sgc_with_dac_layers_separation, sgc_with_tulip_layers_separation, last_layer_only_baseline, global_random_separation, sgc_faiss, coordinate_sampling_separation, coordinate_spp_separation, coordinate_spp_faiss, coordinate_spp_dac, dac_with_coordinate_features')
    parser.add_argument('--dac-knn-backend', type=str, default='faiss',
                       choices=['faiss', 'torch', 'both'],
                       help='kNN backend for DAC calibration: "faiss" (default, FAISS-based), "torch" (PyTorch-based for fair timing with RGCL/RGCC), or "both" (run both and store separately)')

    args = parser.parse_args()
    
    # Set module-level flag for force rerun geometric
    global _FORCE_RERUN_GEOMETRIC
    _FORCE_RERUN_GEOMETRIC = args.force_rerun_geometric

    if args.corruption_type and args.corruption_severity is None:
        parser.error("--corruption-severity is required when --corruption-type is set")
    if args.baselines_only and args.method:
        parser.error("--baselines-only cannot be combined with --method")
    if args.baselines_only and (
        args.ood_only
        or args.deep_ensemble_scan
        or args.eval_corruptions
        or args.mce_only
        or args.eval_ensemble
    ):
        parser.error(
            "--baselines-only cannot be combined with --ood-only, --deep-ensemble-scan, "
            "--eval-corruptions, --mce-only, or --eval-ensemble"
        )

    # ===== METHOD SELECTION CONFIGURATION =====
    METHOD_CONFIGS = {
        'sgc_with_dac_preprocessing_separation': {
            'section': 'ablation.sgc_with_dac_preprocessing_separation',
            'requires_features': True,
            'requires_dac_layers': True
        },
        'sgc_with_dac_layers_separation': {
            'section': 'ablation.sgc_with_dac_layers_separation',
            'requires_features': True,
            'requires_dac_layers': True
        },
        'sgc_with_tulip_layers_separation': {
            'section': 'ablation.sgc_with_tulip_layers_separation',
            'requires_features': True,
            'requires_tulip_layers': True
        },
        'last_layer_only_baseline': {
            'section': 'last_layer_only_baseline',
            'requires_features': False
        },
        'global_random_separation': {
            'section': 'global_random_separation',
            'requires_features': True
        },
        'sgc_faiss': {
            'section': 'sgc_faiss',
            'requires_features': True
        },
        'coordinate_sampling_separation': {
            'section': 'coordinate_sampling_separation',
            'requires_features': False  # Uses coordinate extraction
        },
        'coordinate_spp_separation': {
            'section': 'coordinate_spp_separation',
            'requires_features': False  # Uses coordinate extraction
        },
        'coordinate_spp_faiss': {
            'section': 'coordinate_spp_faiss',
            'requires_features': False  # Uses coordinate extraction
        },
        'coordinate_spp_dac': {
            'section': 'coordinate_spp_dac',
            'requires_features': False  # Uses coordinate extraction
        },
        'dac_with_coordinate_features': {
            'section': 'dac_with_coordinate_features',
            'requires_features': False  # Uses coordinate extraction
        },
        'dac_orig': {
            'section': 'dac_orig',
            'requires_features': False,
            'requires_dac_layers': True  # Uses DAC's architecture-specific layers
        },
        'dac_orig_pytorch': {
            'section': 'dac_orig_pytorch',
            'requires_features': False,
            'requires_dac_layers': True  # Uses DAC's architecture-specific layers
        }
    }

    # Validate and set up method filtering
    methods_only_mode = False
    selected_methods = set()
    if args.method is not None and len(args.method) > 0:
        methods_only_mode = True
        # Validate method names
        invalid_methods = [m for m in args.method if m not in METHOD_CONFIGS]
        if invalid_methods:
            parser.error(f"Invalid method names: {invalid_methods}. Valid methods: {', '.join(METHOD_CONFIGS.keys())}")
        selected_methods = set(args.method)
        logger.info("="*80)
        logger.info(f"METHOD SELECTION MODE: Running only {len(selected_methods)} specified method(s)")
        logger.info(f"Selected methods: {', '.join(sorted(selected_methods))}")
        logger.info("="*80)
        
        # Determine which features/extractions are needed
        needs_features = any(METHOD_CONFIGS[m]['requires_features'] for m in selected_methods)
        needs_dac_layers = any(METHOD_CONFIGS[m].get('requires_dac_layers', False) for m in selected_methods)
        needs_tulip_layers = any(METHOD_CONFIGS[m].get('requires_tulip_layers', False) for m in selected_methods)
        needs_coordinate_extraction = any(m.startswith('coordinate') or m == 'dac_with_coordinate_features' for m in selected_methods)
        
        logger.info(f"Feature extraction needed: {needs_features}")
        logger.info(f"DAC layers needed: {needs_dac_layers}")
        logger.info(f"TULIP layers needed: {needs_tulip_layers}")
        logger.info(f"Coordinate extraction needed: {needs_coordinate_extraction}")
    else:
        # Normal mode - will be determined by normal logic
        needs_features = None
        needs_dac_layers = None
        needs_tulip_layers = None
        needs_coordinate_extraction = None
        selected_methods = set()  # Empty set for normal mode

    # ===== DINOv2 DETECTION =====
    # DINOv2 uses HuggingFace layer naming (backbone.encoder.layer.X) but DAC expects block_X
    # Skip DAC experiments that require architecture-specific layer selection
    is_dinov2 = 'dinov2' in args.model_name.lower()
    if is_dinov2:
        logger.info("="*80)
        logger.info("DINOv2 detected: Skipping DAC experiments requiring architecture-specific layer selection")
        logger.info("  (DAC paper did not specify layers for transformer architectures)")
        logger.info("  Keeping: DAC with random layers, DAC with coordinate features")
        logger.info("="*80)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Detect GPU hardware name
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU Hardware: {gpu_name}")
    else:
        gpu_name = "CPU"
        logger.info(f"GPU Hardware: {gpu_name}")

    # Setup output path
    os.makedirs(args.output_dir, exist_ok=True)
    if args.corruption_type:
        output_file = os.path.join(
            args.output_dir,
            f"ablation_{args.training_method}_{args.dataset}_{args.model_name}_seed{args.seed}_{args.corruption_type}_sev{args.corruption_severity}.json"
        )
    else:
        output_file = os.path.join(
            args.output_dir,
            f"ablation_{args.training_method}_{args.dataset}_{args.model_name}_seed{args.seed}.json"
        )

    # ===== BASELINES-ONLY MODE HANDLER =====
    if args.baselines_only:
        logger.info("=" * 80)
        logger.info("BASELINES-ONLY MODE: Running standard calibrators and top-label Brier")
        logger.info("Existing non-baseline sections will be preserved.")
        logger.info("=" * 80)

        existing_results = load_existing_results(output_file) if os.path.exists(output_file) else None
        results = existing_results if existing_results is not None else {}

        def standard_baselines_have_requested_metrics(section: Dict[str, Any]) -> bool:
            if not isinstance(section, dict) or not section:
                return False
            required_metric_fields = ["adaptive_ece", "calibration_mce"]
            if args.compute_brier_top_label:
                required_metric_fields.append("top_label_brier")

            def has_required_fields(method_result: Any) -> bool:
                if not isinstance(method_result, dict):
                    return False
                if method_result.get("skipped", False) or method_result.get("error"):
                    return True
                return all(method_result.get(field) is not None for field in required_metric_fields)

            required_methods = [
                "uncalibrated",
                "temperature_scaling",
                "isotonic_toplabel",
                "platt_scaling",
                "beta_calibration",
            ]
            for method_key in required_methods:
                if not has_required_fields(section.get(method_key)):
                    return False
            dac_result = section.get("density_aware_calibration")
            if isinstance(dac_result, dict) and not dac_result.get("skipped", False):
                if not has_required_fields(dac_result):
                    return False
            if not has_required_fields(section.get("geometric_physical_space")):
                return False
            raw_geometric_result = section.get("geometric_raw_images")
            if raw_geometric_result is not None and not has_required_fields(raw_geometric_result):
                return False
            return True

        if (
            not args.force_recompute
            and standard_baselines_have_requested_metrics(results.get("standard_baselines", {}))
        ):
            logger.info("✓ Standard baselines already include requested metrics; skipping")
            return

        model_path = construct_model_path(
            args.results_base_dir,
            args.training_method,
            args.dataset,
            args.model_name,
            args.seed,
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        logger.info(f"\nLoading model from: {model_path}")
        train_loader, val_loader, test_loader, num_classes = get_data_loaders(
            args.dataset,
            args.batch_size,
            seed=args.seed,
            corruption_type=args.corruption_type,
            corruption_severity=args.corruption_severity,
        )
        model = load_trained_model(
            model_path,
            args.model_name,
            num_classes,
            device,
            dataset=args.dataset,
        )

        train_raw, train_labels = get_all_data_as_numpy(train_loader)
        val_raw, val_labels = get_all_data_as_numpy(val_loader)
        test_raw, test_labels = get_all_data_as_numpy(test_loader)
        model_adapter = PyTorchModelAdapter(model, device, args.dataset)

        baseline_results = run_standard_baselines(
            model_adapter=model_adapter,
            val_raw=val_raw,
            val_labels=val_labels,
            test_raw=test_raw,
            test_labels=test_labels,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            train_raw=train_raw,
            train_labels=train_labels,
            seed=args.seed,
            device=str(device),
            include_dac=not is_dinov2,
        )

        if is_dinov2:
            baseline_results["density_aware_calibration"] = {
                "skipped": True,
                "reason": "DINOv2: DAC paper did not specify layers for transformer architectures",
            }

        results["standard_baselines"] = baseline_results
        if "experiment_config" not in results:
            results["experiment_config"] = {
                **vars(args),
                "corruption_type": args.corruption_type,
                "corruption_severity": args.corruption_severity,
                "is_corrupted": args.corruption_type is not None,
                "gpu_hardware": gpu_name,
            }
        else:
            results["experiment_config"]["compute_brier_top_label"] = bool(args.compute_brier_top_label)
            results["experiment_config"]["last_baselines_only_run"] = True
        results["top_label_brier_metadata"] = {
            "field": "standard_baselines.<method>.top_label_brier",
            "unit": "fraction",
            "definition": "mean((max_prob - correct)^2)",
        }
        save_results_incrementally(results, output_file)
        logger.info(f"\n✓ Baselines-only pass completed! Results saved to: {output_file}")
        return

    # ===== OOD-ONLY MODE HANDLER =====
    if args.ood_only:
        if not os.path.exists(output_file):
            logger.error(f"Cannot run --ood-only: results file not found at {output_file}")
            return
        
        logger.info("=" * 80)
        logger.info("OOD-ONLY MODE: Adding OOD metrics to existing results")
        logger.info("=" * 80)
        
        # Load existing results
        existing_results = load_existing_results(output_file)
        if existing_results is None:
            logger.error(f"Failed to load results from {output_file}")
            return
        results = existing_results
        
        # Load model and data for OOD evaluation
        model_path = construct_model_path(
            args.results_base_dir,
            args.training_method,
            args.dataset,
            args.model_name,
            args.seed
        )
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found at: {model_path}")
            return
        
        logger.info(f"\nLoading model from: {model_path}")
        train_loader, val_loader, test_loader, num_classes = get_data_loaders(
            args.dataset,
            args.batch_size,
            seed=args.seed,
            corruption_type=args.corruption_type,
            corruption_severity=args.corruption_severity
        )
        model = load_trained_model(
            model_path,
            args.model_name,
            num_classes,
            device,
            dataset=args.dataset
        )
        
        def extract_raw_data(loader):
            raw_data = []
            labels = []
            for data, target in loader:
                raw_data.append(data.numpy())
                labels.append(target.numpy())
            return np.concatenate(raw_data), np.concatenate(labels)
        
        val_raw, val_labels = extract_raw_data(val_loader)
        test_raw, test_labels = extract_raw_data(test_loader)
        model_adapter = PyTorchModelAdapter(model, device, args.dataset)
        
        # Load OOD data
        if args.dataset not in ["cifar10", "cifar100"]:
            logger.error("OOD evaluation only supported for CIFAR-10/100")
            return
        
        from Data.svhn import get_test_loader as svhn_test
        svhn_loader = svhn_test(batch_size=args.batch_size, shuffle=False)
        ood_raw, ood_labels = extract_raw_from_loader(svhn_loader)
        
        # Add OOD to baselines
        results = compute_baseline_ood_metrics(
            results, model_adapter,
            val_raw, val_labels, test_raw, test_labels,
            ood_raw, ood_labels, args.batch_size, str(device),
            train_raw=train_raw, train_labels=train_labels,
            dac_knn_backend=args.dac_knn_backend
        )
        
        # Save updated results
        save_results_incrementally(results, output_file)
        logger.info(f"Updated results with OOD metrics")
        logger.info(f"Note: For geometric/DAC methods, OOD metrics will be computed when those sections are rerun")
        return

    # ===== FORCE RERUN GEOMETRIC LOGGING =====
    if args.force_rerun_geometric:
        logger.info("=" * 80)
        logger.info("FORCE RE-RUN MODE: Re-running all geometric calibration experiments")
        logger.info("Reason: Previous runs used buggy percentile normalization (now fixed to rank-based)")
        logger.info("=" * 80)
    
    # ===== DEEP ENSEMBLE SCAN MODE HANDLER =====
    if args.deep_ensemble_scan:
        # This mode only runs ensemble scan, skips other experiments
        logger.info("=" * 80)
        logger.info("Running in DEEP ENSEMBLE SCAN mode")
        logger.info("=" * 80)
        
        # Load data (we need test/val data for evaluation)
        logger.info(f"Loading {args.dataset} dataset...")
        train_loader, val_loader, test_loader, num_classes = get_data_loaders(
            args.dataset,
            args.batch_size,
            seed=args.seed,
            corruption_type=args.corruption_type,
            corruption_severity=args.corruption_severity
        )
        
        def extract_raw_data(loader):
            raw_data = []
            labels = []
            for data, target in loader:
                raw_data.append(data.numpy())
                labels.append(target.numpy())
            return np.concatenate(raw_data), np.concatenate(labels)
        
        val_raw, val_labels = extract_raw_data(val_loader)
        test_raw, test_labels = extract_raw_data(test_loader)
        
        # Load OOD data (SVHN for CIFAR-10/100)
        ood_raw = None
        ood_labels = None
        if args.dataset in ["cifar10", "cifar100"]:
            try:
                from Data.svhn import get_test_loader as svhn_test
                logger.info("Loading SVHN test set as OOD dataset...")
                svhn_loader = svhn_test(batch_size=args.batch_size, shuffle=False)
                ood_raw, ood_labels = extract_raw_from_loader(svhn_loader)
                logger.info(f"Loaded SVHN OOD data: {ood_raw.shape}")
            except Exception as e:
                logger.warning(f"Failed to load SVHN OOD data: {e}")
                ood_raw = None
        
        # Run ensemble scan
        ensemble_results = run_deep_ensemble_scan(
            model_name=args.model_name,
            dataset=args.dataset,
            training_method=args.training_method,
            results_base_dir=args.results_base_dir,
            test_raw=test_raw,
            test_labels=test_labels,
            val_raw=val_raw,
            val_labels=val_labels,
            device=device,
            batch_size=args.batch_size,
            master_seed=args.seed,
            ensemble_size=args.ensemble_size,
            ood_raw=ood_raw,
            ood_labels=ood_labels,
        )
        
        # Load or create results dict
        if os.path.exists(output_file) and not args.force_recompute:
            existing_results = load_existing_results(output_file)
            results = existing_results if existing_results is not None else {}
        else:
            results = {}
        
        results['deep_ensemble_scan'] = ensemble_results
        save_results_incrementally(results, output_file)
        
        logger.info(f"\nDeep ensemble scan complete. Results saved to {output_file}")
        return  # Exit early, don't run other experiments

    # ===== CORRUPTION-FOCUSED MODE HANDLER =====
    # When --eval-corruptions or --mce-only is set, run minimal experiments for corruption evaluation
    if args.eval_corruptions or args.mce_only:
        # Check dataset compatibility first
        if args.dataset.lower() not in ['cifar10', 'cifar100']:
            logger.error(f"Corruption evaluation only supported for CIFAR-10/100, got {args.dataset}")
            return
        
        logger.info("=" * 80)
        logger.info("CORRUPTION-FOCUSED MODE")
        logger.info("=" * 80)
        
        # Load existing results if available
        if os.path.exists(output_file) and not args.force_recompute:
            logger.info(f"Found existing results file: {output_file}")
            existing_results = load_existing_results(output_file)
            results = existing_results if existing_results is not None else {}
            
            # Check if corruption evaluation already exists
            if 'corruption_evaluation' in results:
                logger.info("Corruption evaluation already computed. Use --force-recompute to rerun.")
                return
        else:
            results = {}
        
        # Load model and data
        model_path = construct_model_path(
            args.results_base_dir,
            args.training_method,
            args.dataset,
            args.model_name,
            args.seed
        )
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found at: {model_path}")
            return
        
        logger.info(f"\nLoading model from: {model_path}")
        train_loader, val_loader, test_loader, num_classes = get_data_loaders(
            args.dataset,
            args.batch_size,
            seed=args.seed,
            corruption_type=None,
            corruption_severity=None
        )
        model = load_trained_model(
            model_path,
            args.model_name,
            num_classes,
            device,
            dataset=args.dataset
        )
        
        def extract_raw_data(loader):
            raw_data = []
            labels = []
            for data, target in loader:
                raw_data.append(data.numpy())
                labels.append(target.numpy())
            return np.concatenate(raw_data), np.concatenate(labels)
        
        train_raw, train_labels = extract_raw_data(train_loader)
        val_raw, val_labels = extract_raw_data(val_loader)
        test_raw, test_labels = extract_raw_data(test_loader)
        model_adapter = PyTorchModelAdapter(model, device, args.dataset)
        
        # Save experiment config if not present
        if 'experiment_config' not in results:
            results['experiment_config'] = {
                **vars(args),
                'mode': 'corruption_focused',
                'gpu_hardware': gpu_name
            }
            save_results_incrementally(results, output_file)
        
        # ===== RUN MINIMAL METHODS IF NOT ALREADY COMPUTED =====
        
        # 1. Standard baselines (uncalibrated, temperature scaling)
        if 'standard_baselines' not in results or not results['standard_baselines']:
            logger.info("\n" + "="*80)
            logger.info("RUNNING MINIMAL STANDARD BASELINES")
            logger.info("="*80)
            
            baseline_results = run_standard_baselines(
                model_adapter=model_adapter,
                val_raw=val_raw,
                val_labels=val_labels,
                test_raw=test_raw,
                test_labels=test_labels,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                train_raw=train_raw,
                train_labels=train_labels,
                seed=args.seed,
                device=str(device),
                include_dac=not is_dinov2,
            )
            results['standard_baselines'] = baseline_results
            save_results_incrementally(results, output_file)
        else:
            logger.info("Standard baselines already computed")
        
        # 2. SGC-FAISS (fast geometric calibration)
        if 'sgc_faiss' not in results or results.get('sgc_faiss', {}).get('error'):
            logger.info("\n" + "="*80)
            logger.info("RUNNING SGC-FAISS (Fast Geometric Calibration)")
            logger.info("="*80)
            
            try:
                from Experiments.compare_sgc_backends import SGCCalibratorFast
                
                num_layers = args.num_layers
                target_dim = args.target_dimension
                
                # Discover layers
                is_dinov2 = args.model_name is not None and 'dinov2' in args.model_name.lower()
                if is_dinov2:
                    input_shape = (1, 3, 224, 224)
                elif args.dataset.lower() in ["tiny_imagenet", "tinyimagenet"]:
                    input_shape = (1, 3, 64, 64)
                else:
                    input_shape = (1, 3, 32, 32)
                
                discovered_layers = normalize_discovered_layers(
                    model, discover_model_layers(model, device=device, input_shape=input_shape)
                )
                filtered_layers, _ = filter_non_feature_layers(discovered_layers, model)
                all_layer_names = [d["name"] for d in filtered_layers]
                
                # Select random layers
                np.random.seed(args.seed)
                if num_layers > len(all_layer_names):
                    selected_layers = all_layer_names
                else:
                    selected_layers = list(np.random.choice(all_layer_names, size=num_layers, replace=False))
                
                logger.info(f"Selected {len(selected_layers)} layers: {selected_layers}")
                
                # Create dataloaders
                train_loader_sgc = DataLoader(
                    TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                    batch_size=args.batch_size, shuffle=False
                )
                val_loader_sgc = DataLoader(
                    TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                    batch_size=args.batch_size, shuffle=False
                )
                test_loader_sgc = DataLoader(
                    TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
                    batch_size=args.batch_size, shuffle=False
                )
                
                # Extract features
                start_extraction = time.perf_counter()
                train_features, val_features, test_features, extraction_info = extract_and_aggregate_sgc_features(
                    model=model,
                    layer_names=selected_layers,
                    train_loader=train_loader_sgc,
                    val_loader=val_loader_sgc,
                    test_loader=test_loader_sgc,
                    device=device,
                    target_dim=target_dim,
                    seed=args.seed,
                    pooling_mode=args.spp_pooling_mode
                )
                extraction_time = time.perf_counter() - start_extraction
                
                # Get predictions
                val_probs = model_adapter.predict_proba(val_raw, batch_size=args.batch_size)
                test_probs = model_adapter.predict_proba(test_raw, batch_size=args.batch_size)
                val_predictions = np.argmax(val_probs, axis=1)
                test_predictions = np.argmax(test_probs, axis=1)
                
                # Fit and calibrate
                sgc_cal = SGCCalibratorFast(use_gpu=(device.type == 'cuda'))
                
                start_fit = time.perf_counter()
                sgc_cal.fit(
                    X_train_features=train_features,
                    y_train=train_labels,
                    X_val_features=val_features,
                    y_val=val_labels,
                    val_predictions=val_predictions
                )
                fit_time = time.perf_counter() - start_fit
                
                start_cal = time.perf_counter()
                calibrated_probs = sgc_cal.calibrate(
                    X_test_features=test_features,
                    predictions=test_predictions,
                    logits=test_probs
                )
                calibrate_time = time.perf_counter() - start_cal
                
                # Calculate metrics
                ece = calculate_ece(calibrated_probs, test_labels)
                adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
                calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
                brier = calculate_brier_score(calibrated_probs, test_labels)
                accuracy = calculate_accuracy(calibrated_probs, test_labels)
                
                results['sgc_faiss'] = {
                    'ece': float(ece),
                    'adaptive_ece': float(adaptive_ece),
                    'calibration_mce': float(calibration_mce),
                    'brier': float(brier),
                    'accuracy': float(accuracy),
                    'extraction_time_s': extraction_time,
                    'fit_time_s': fit_time,
                    'calibrate_time_s': calibrate_time,
                    'selected_layers': selected_layers,
                    'num_layers': len(selected_layers),
                    'target_dim': target_dim,
                    'method': 'SGC-FAISS',
                    'calibrator_params': {'fitting_method': 'isotonic'}
                }
                
                logger.info(f"SGC-FAISS ECE: {ece:.6f}")
                save_results_incrementally(results, output_file)
                
            except Exception as e:
                logger.error(f"SGC-FAISS failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                results['sgc_faiss'] = {'error': str(e)}
                save_results_incrementally(results, output_file)
        else:
            logger.info("SGC-FAISS already computed")
        
        # ===== CORRUPTION EVALUATION =====
        logger.info("\n" + "="*80)
        logger.info("STARTING CIFAR-C CORRUPTION EVALUATION")
        logger.info("="*80)
        
        # Get test transform
        import torchvision.transforms as transforms
        if args.dataset.lower() == 'cifar10':
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])
        else:  # cifar100
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])
        
        corruption_results = evaluate_corruption_robustness_fast(
            results, model, model_adapter, args.dataset,
            test_transform, args.batch_size, device,
            train_raw=train_raw, train_labels=train_labels,
            val_raw=val_raw, val_labels=val_labels,
            cifar_c_dir=args.cifar_c_dir,
            pooling_mode=args.spp_pooling_mode,
            target_dimension=args.target_dimension,
            seed=args.seed
        )
        
        results['corruption_evaluation'] = corruption_results
        save_results_incrementally(results, output_file)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("CORRUPTION EVALUATION SUMMARY")
        logger.info("="*80)
        for method, data in corruption_results.items():
            if 'mce' in data and data['mce'] is not None:
                logger.info(f"  {method}: mCE = {data['mce']:.4f}")
        
        logger.info(f"\nResults saved to {output_file}")
        return  # Exit early

    # ===== LOAD EXISTING RESULTS OR START FRESH =====
    if os.path.exists(output_file) and not args.force_recompute:
        logger.info(f"Found existing results file: {output_file}")
        logger.info("Will skip completed sections and resume from where it left off")
        existing_results = load_existing_results(output_file)
        results = existing_results if existing_results is not None else {}
        
        # If force rerun geometric is set, delete geometric sections from loaded results
        if args.force_rerun_geometric:
            logger.info("Deleting geometric calibration sections from loaded results...")
            deleted_count = 0
            
            # Helper function to check if a section name matches any geometric section
            def is_geometric_section(section_name):
                """Check if a section name matches any geometric section pattern."""
                for geo_section in GEOMETRIC_SECTIONS_TO_RERUN:
                    # Exact match
                    if section_name == geo_section:
                        return True
                    # Match with suffix (e.g., global_random_separation matches global_random)
                    if section_name.startswith(geo_section + '_'):
                        return True
                    # Match nested (e.g., ablation.sgc_with_dac_preprocessing)
                    if geo_section.startswith('ablation.') and section_name.startswith('ablation.'):
                        ablation_key = geo_section.replace('ablation.', '')
                        if section_name == geo_section or section_name.endswith('.' + ablation_key) or section_name.endswith('.' + ablation_key + '_'):
                            return True
                return False
            
            # Helper function to recursively delete geometric sections from a dict
            def delete_geometric_from_dict(d, path=""):
                """Recursively delete geometric sections from a dictionary."""
                count = 0
                keys_to_delete = []
                
                for key in list(d.keys()):
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check if this key or path matches a geometric section
                    # Check both the key alone (for top-level matches) and the full path (for nested matches)
                    if is_geometric_section(key) or is_geometric_section(current_path):
                        keys_to_delete.append(key)
                        count += 1
                        logger.info(f"  Deleted: {current_path}")
                    # If it's a dict, recurse into it first
                    elif isinstance(d[key], dict):
                        sub_count = delete_geometric_from_dict(d[key], current_path)
                        count += sub_count
            
                for key in keys_to_delete:
                    del d[key]
                
                return count
            
            # Delete geometric sections recursively
            deleted_count = delete_geometric_from_dict(results)
            
            # Also handle top-level keys that might be variants
            top_level_keys_to_delete = []
            for key in list(results.keys()):
                if is_geometric_section(key):
                    if key not in top_level_keys_to_delete:
                        top_level_keys_to_delete.append(key)
            
            for key in top_level_keys_to_delete:
                if key in results:
                    del results[key]
                    deleted_count += 1
                    logger.info(f"  Deleted: {key}")
            
            logger.info(f"Deleted {deleted_count} geometric calibration section(s)")
            # Save the updated results (without geometric sections)
            save_results_incrementally(results, output_file)
    else:
        if args.force_recompute:
            logger.info("--force-recompute flag set, starting from scratch")
        else:
            logger.info("No existing results file found, starting fresh")
        results = {}

    # Save experiment config
    if 'experiment_config' not in results:
        results['experiment_config'] = {
            **vars(args),
            'corruption_type': args.corruption_type,
            'corruption_severity': args.corruption_severity,
            'is_corrupted': args.corruption_type is not None,
            'gpu_hardware': gpu_name
        }
        save_results_incrementally(results, output_file)

    # ===== LOAD MODEL AND DATA (ONLY IF NEEDED) =====
    # Check if we need to run ANY experiments
    # When --skip-single-layers is set, skip checks for single-layer dependent sections
    # When is_dinov2 is True, skip DAC experiments requiring architecture-specific layer selection
    if args.skip_single_layers:
        needs_model = (
            not is_section_complete(results, 'standard_baselines') or
            (not is_dinov2 and should_rerun_section(results, 'geometric_dac_weighted', DAC_REQUIRED_FIELDS)) or
            (not is_dinov2 and should_rerun_section(results, 'ablation.dac_with_sgc_aggregated_features', DAC_REQUIRED_FIELDS)) or
            should_rerun_section(results, 'ablation.dac_with_random_layer_selection', DAC_REQUIRED_FIELDS) or
            (not is_dinov2 and should_rerun_section(results, 'ablation.sgc_with_dac_preprocessing', GEO_REQUIRED_FIELDS)) or
            (not is_dinov2 and should_rerun_section(results, 'ablation.random_single_layer_dac_preprocessing', GEO_REQUIRED_FIELDS)) or
            (not is_dinov2 and should_rerun_section(results, 'sgc_with_dac_layers', GEO_REQUIRED_FIELDS)) or
            should_rerun_section(results, 'geometric_concatenated', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'global_random', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'coordinate_sampling', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'coordinate_spp', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'coordinate_spp_dac', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'dac_with_coordinate_features', DAC_REQUIRED_FIELDS) or
            should_rerun_section(results, 'dac_orig', DAC_REQUIRED_FIELDS) or
            not is_section_complete(results, 'analysis', ['best_overall_method'])
        )
    else:
        needs_model = (
            not is_section_complete(results, 'standard_baselines') or
            should_rerun_section(results, 'geometric_original', GEO_REQUIRED_FIELDS) or
            (not is_dinov2 and should_rerun_section(results, 'geometric_dac_weighted', DAC_REQUIRED_FIELDS)) or
            (not is_dinov2 and should_rerun_section(results, 'ablation.dac_with_sgc_aggregated_features', DAC_REQUIRED_FIELDS)) or
            should_rerun_section(results, 'ablation.dac_with_random_layer_selection', DAC_REQUIRED_FIELDS) or
            (not is_dinov2 and should_rerun_section(results, 'ablation.sgc_with_dac_preprocessing', GEO_REQUIRED_FIELDS)) or
            (not is_dinov2 and should_rerun_section(results, 'ablation.random_single_layer_dac_preprocessing', GEO_REQUIRED_FIELDS)) or
            (not is_dinov2 and should_rerun_section(results, 'sgc_with_dac_layers', GEO_REQUIRED_FIELDS)) or
            should_rerun_section(results, 'metric_guided_calibration', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'geometric_precomputed', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'geometric_concatenated', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'global_random', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'coordinate_sampling', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'coordinate_spp', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'coordinate_spp_dac', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'dac_with_coordinate_features', DAC_REQUIRED_FIELDS) or
            should_rerun_section(results, 'dac_orig', DAC_REQUIRED_FIELDS) or
            not is_section_complete(results, 'analysis', ['best_overall_method'])
        )

    if not needs_model:
        logger.info("\n" + "="*80)
        logger.info("ALL EXPERIMENTS COMPLETE - NOTHING TO DO")
        logger.info("="*80)
        logger.info(f"Results file: {output_file}")
        return

    # Load model and data
    model_path = construct_model_path(
        args.results_base_dir,
        args.training_method,
        args.dataset,
        args.model_name,
        args.seed
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    logger.info(f"\nLoading model from: {model_path}")

    # Load data
    logger.info(f"Loading {args.dataset} dataset...")
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset,
        args.batch_size,
        seed=args.seed,
        corruption_type=args.corruption_type,
        corruption_severity=args.corruption_severity
    )

    # Load model
    model = load_trained_model(
        model_path,
        args.model_name,
        num_classes,
        device,
        dataset=args.dataset
    )

    # Extract raw data for geometric calibrator
    logger.info("Extracting raw data...")

    def extract_raw_data(loader):
        raw_data = []
        labels = []
        for data, target in loader:
            raw_data.append(data.numpy())
            labels.append(target.numpy())
        return np.concatenate(raw_data), np.concatenate(labels)

    train_raw, train_labels = extract_raw_data(train_loader)
    val_raw, val_labels = extract_raw_data(val_loader)
    test_raw, test_labels = extract_raw_data(test_loader)

    # Load OOD data (SVHN for CIFAR-10/100)
    ood_raw = None
    ood_labels = None
    if args.dataset in ["cifar10", "cifar100"]:
        try:
            from Data.svhn import get_test_loader as svhn_test
            logger.info("Loading SVHN test set as OOD dataset...")
            svhn_loader = svhn_test(batch_size=args.batch_size, shuffle=False)
            ood_raw, ood_labels = extract_raw_from_loader(svhn_loader)
            logger.info(f"Loaded SVHN OOD data: {ood_raw.shape}")
        except Exception as e:
            logger.warning(f"Failed to load SVHN OOD data: {e}")
            ood_raw = None

    # Create model adapter
    model_adapter = PyTorchModelAdapter(model, device, args.dataset)

    # ===========================================================================
    # OPTIMIZATION: Precompute model predictions for all datasets
    # This eliminates redundant forward passes across all experiments
    # ===========================================================================
    logger.info("\nPre-computing model predictions for all datasets...")
    prediction_cache = {}
    
    # Training set
    prediction_cache['train_probs'] = model_adapter.predict_proba(train_raw, batch_size=args.batch_size)
    prediction_cache['train_preds'] = np.argmax(prediction_cache['train_probs'], axis=1)
    
    # Validation set
    prediction_cache['val_probs'] = model_adapter.predict_proba(val_raw, batch_size=args.batch_size)
    prediction_cache['val_preds'] = np.argmax(prediction_cache['val_probs'], axis=1)
    
    # Test set
    prediction_cache['test_probs'] = model_adapter.predict_proba(test_raw, batch_size=args.batch_size)
    prediction_cache['test_preds'] = np.argmax(prediction_cache['test_probs'], axis=1)
    
    # OOD set (if available)
    if ood_raw is not None:
        prediction_cache['ood_probs'] = model_adapter.predict_proba(ood_raw, batch_size=args.batch_size)
        prediction_cache['ood_preds'] = np.argmax(prediction_cache['ood_probs'], axis=1)
    
    logger.info("Pre-computation complete.")
    # ===========================================================================

    # ===========================================================================
    # OPTIMIZATION: Create DataLoaders once and reuse
    # ===========================================================================
    logger.info("Creating data loaders...")
    dataloader_cache = {}
    
    dataloader_cache['train'] = DataLoader(
        TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
        batch_size=args.batch_size, shuffle=False
    )
    dataloader_cache['val'] = DataLoader(
        TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
        batch_size=args.batch_size, shuffle=False
    )
    dataloader_cache['test'] = DataLoader(
        TensorDataset(torch.from_numpy(test_raw), torch.from_numpy(test_labels.astype(np.int64))),
        batch_size=args.batch_size, shuffle=False
    )
    if ood_raw is not None:
        dataloader_cache['ood'] = DataLoader(
            TensorDataset(torch.from_numpy(ood_raw), torch.from_numpy(np.zeros(len(ood_raw), dtype=np.int64))),
            batch_size=args.batch_size, shuffle=False
        )
    # ===========================================================================

    # ===========================================================================
    # OPTIMIZATION: OOD feature caching
    # Each unique feature extraction configuration stores its OOD features
    # ===========================================================================
    ood_feature_cache = {}
    
    def get_or_extract_ood_features(
        cache_key: str,
        extract_fn,
        *args,
        **kwargs
    ) -> np.ndarray:
        """Extract OOD features with caching to avoid redundant extraction."""
        if cache_key in ood_feature_cache:
            logger.info(f"Using cached OOD features: {cache_key}")
            return ood_feature_cache[cache_key]
        
        logger.info(f"Extracting OOD features for: {cache_key}")
        features = extract_fn(*args, **kwargs)
        ood_feature_cache[cache_key] = features
        return features
    # ===========================================================================

    # ===== STANDARD BASELINES (Uncalibrated, TS, Isotonic, Platt, Beta, DAC) =====
    if methods_only_mode:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING STANDARD BASELINES (methods-only mode)")
        logger.info("="*80)
        if 'standard_baselines' not in results:
            results['standard_baselines'] = {"skipped": True, "reason": "methods-only mode"}
        save_results_incrementally(results, output_file)
        baseline_results = results.get('standard_baselines', {})
        uncalibrated_data = baseline_results.get('uncalibrated', {})
        uncalibrated_results = {
            'ece': uncalibrated_data.get('ece'),
            'accuracy': uncalibrated_data.get('acc')
        }
        dac_results = baseline_results.get('density_aware_calibration', {})
        baseline_ts_results = baseline_results.get('temperature_scaling', {})
        ts_plus_dac_results = baseline_results.get('ts_plus_dac', None)
    elif not is_section_complete(results, 'standard_baselines'):
        logger.info("\n" + "="*80)
        logger.info("RUNNING STANDARD BASELINE CALIBRATORS")
        logger.info("="*80)
        
        baseline_results = run_standard_baselines(
            model_adapter=model_adapter,
            val_raw=val_raw,
            val_labels=val_labels,
            test_raw=test_raw,
            test_labels=test_labels,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            train_raw=train_raw,
            train_labels=train_labels,
            seed=args.seed,
            device=str(device),
            include_dac=not is_dinov2,
        )
        
        # Skip DAC for DINOv2 models (DAC requires architecture-specific layer selection)
        if is_dinov2:
            logger.info("Skipping DAC in standard baselines (DINOv2: DAC paper did not specify layers for transformer architectures)")
            baseline_results['density_aware_calibration'] = {
                "skipped": True,
                "reason": "DINOv2: DAC paper did not specify layers for transformer architectures"
            }
        
        results['standard_baselines'] = baseline_results
        
        # Add OOD metrics to baselines if OOD data is available
        if ood_raw is not None:
            logger.info("Adding OOD metrics to standard baselines...")
            results = compute_baseline_ood_metrics(
                results, model_adapter,
                val_raw, val_labels, test_raw, test_labels,
                ood_raw, ood_labels, args.batch_size, str(device),
                train_raw=train_raw, train_labels=train_labels
            )
        
        save_results_incrementally(results, output_file)
        
        # Extract key results for later use
        uncalibrated_data = baseline_results.get('uncalibrated', {})
        uncalibrated_results = {
            'ece': uncalibrated_data.get('ece'),
            'accuracy': uncalibrated_data.get('acc')
        }
        dac_results = baseline_results.get('density_aware_calibration', {})
        baseline_ts_results = baseline_results.get('temperature_scaling', {})
        
        # Extract TS+DAC if available (from DAC results)
        ts_plus_dac_results = None
        if 'density_aware_calibration' in baseline_results and not baseline_results['density_aware_calibration'].get('skipped', False):
            # Check if TS+DAC is available in the baseline results
            # Note: run_standard_baselines may not return ts_plus_dac, so we handle it gracefully
            if 'ts_plus_dac' in baseline_results:
                ts_plus_dac_results = baseline_results['ts_plus_dac']
    else:
        logger.info("✓ Standard baselines already computed, skipping")
        baseline_results = results['standard_baselines']
        uncalibrated_data = baseline_results.get('uncalibrated', {})
        uncalibrated_results = {
            'ece': uncalibrated_data.get('ece'),
            'accuracy': uncalibrated_data.get('acc')
        }
        dac_results = baseline_results.get('density_aware_calibration', {})
        baseline_ts_results = baseline_results.get('temperature_scaling', {})
        ts_plus_dac_results = baseline_results.get('ts_plus_dac', None)

    # ===== DEEP ENSEMBLE BASELINE =====
    if args.eval_ensemble:
        if 'deep_ensemble' not in results or should_rerun_section(results, 'deep_ensemble', ['ece']):
            logger.info("\n" + "="*80)
            logger.info("RUNNING DEEP ENSEMBLE BASELINE")
            logger.info("="*80)
            
            # Parse ensemble seeds
            ensemble_seeds = [int(s.strip()) for s in args.ensemble_seeds.split() if s.strip()]
            if not ensemble_seeds:
                logger.warning("No valid ensemble seeds provided, using default [11, 12, 13]")
                ensemble_seeds = [11, 12, 13]
            
            try:
                ensemble_results = run_deep_ensemble_baseline(
                    model_name=args.model_name,
                    dataset=args.dataset,
                    training_method=args.training_method,
                    ensemble_seeds=ensemble_seeds,
                    results_base_dir=args.results_base_dir,
                    test_raw=test_raw,
                    test_labels=test_labels,
                    val_raw=val_raw,
                    val_labels=val_labels,
                    device=device,
                    batch_size=args.batch_size,
                    ood_raw=ood_raw,
                    ood_labels=ood_labels,
                )
                results['deep_ensemble'] = ensemble_results
                save_results_incrementally(results, output_file)
            except Exception as e:
                logger.error(f"Failed to run deep ensemble baseline: {e}")
                import traceback
                logger.error(traceback.format_exc())
                results['deep_ensemble'] = {
                    'skipped': True,
                    'reason': f'Error: {str(e)}'
                }
                save_results_incrementally(results, output_file)
        else:
            logger.info("✓ Deep ensemble baseline already computed, skipping")

    # Check model accuracy (skip for corruption experiments)
    uncalibrated_accuracy = uncalibrated_results.get('accuracy')
    if args.corruption_type is None and uncalibrated_accuracy is not None and uncalibrated_accuracy < 0.5:
        logger.error(f"Model accuracy ({uncalibrated_accuracy:.2f}%) is below 50%")
        raise ValueError("Model accuracy too low")
    elif args.corruption_type is not None:
        logger.info(f"Corruption experiment: skipping accuracy check (accuracy may be lower on corrupted data)")

    if uncalibrated_accuracy is not None:
        logger.info(f"Model accuracy: {uncalibrated_accuracy:.2f}% - proceeding with calibration")
    else:
        logger.warning("Model accuracy not available - proceeding with calibration")

    # ===== FEATURE EXTRACTION (PREREQUISITE FOR GEOMETRIC METHODS) =====
    # Check if we need features for any geometric experiments
    # When --skip-single-layers is set, we still need features for:
    # - geometric_dac_weighted
    # - geometric_concatenated
    # But NOT for single-layer or metric-guided experiments
    if methods_only_mode:
        # In methods-only mode, only extract features if any selected method requires them
        # needs_features was already determined during method validation
        pass  # needs_features already set above
    elif args.skip_single_layers:
        needs_features = (
            should_rerun_section(results, 'geometric_dac_weighted', DAC_REQUIRED_FIELDS) or
            should_rerun_section(results, 'geometric_concatenated', GEO_REQUIRED_FIELDS)
        )
    else:
        needs_features = (
            should_rerun_section(results, 'geometric_original', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'geometric_dac_weighted', DAC_REQUIRED_FIELDS) or
            should_rerun_section(results, 'metric_guided_calibration', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'geometric_precomputed', GEO_REQUIRED_FIELDS) or
            should_rerun_section(results, 'geometric_concatenated', GEO_REQUIRED_FIELDS) or
            ('layer_quality_metrics' not in results or not results['layer_quality_metrics'])
        )

    # Extract features with normalization (default) for experiments
    if needs_features:
        logger.info("\n" + "="*80)
        logger.info("EXTRACTING FEATURES (PREREQUISITE FOR GEOMETRIC METHODS)")
        logger.info("="*80)
        precomputed_features = extract_all_features_once(
            model, args.model_name,
            train_raw, train_labels,
            val_raw, val_labels,
            test_raw, test_labels,
            device, args.batch_size,
            dataset=args.dataset,
            pooling_mode=args.spp_pooling_mode,
            target_dimension=args.target_dimension,
            normalize_spp_output=True  # Default: with normalization
        )
        all_layer_names = precomputed_features['layer_names']

        # Compute separation scores
        precomputed_separations = compute_all_separation_scores_once(
            precomputed_features['train_features_per_layer'], train_labels,
            precomputed_features['val_features_per_layer'], val_labels,
            precomputed_features['test_features_per_layer'], test_labels,
            all_layer_names,
            use_cuda=(device.type == 'cuda'),
            model_adapter=model_adapter,
            val_raw=val_raw,
            test_raw=test_raw
        )
    else:
        logger.info("✓ All geometric experiments already computed, skipping feature extraction")
        precomputed_features = None
        precomputed_separations = None
        all_layer_names = None

    # ===== GEOMETRIC SINGLE-LAYER CALIBRATIONS =====
    if methods_only_mode:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING GEOMETRIC SINGLE-LAYER CALIBRATIONS (methods-only mode)")
        logger.info("="*80)
        # Initialize empty results for skipped sections
        if 'geometric_original' not in results:
            results['geometric_original'] = {"skipped": True, "reason": "methods-only mode"}
        if 'layer_quality_metrics' not in results:
            results['layer_quality_metrics'] = {"skipped": True, "reason": "methods-only mode"}
        if 'metric_guided_calibration' not in results:
            results['metric_guided_calibration'] = {"skipped": True, "reason": "methods-only mode"}
        if 'geometric_precomputed' not in results:
            results['geometric_precomputed'] = {"skipped": True, "reason": "methods-only mode"}
        save_results_incrementally(results, output_file)
        geo_results_dict = {}
        layer_metrics = {}
        metric_selections = {}
        metric_guided_results = []
        # Ensure all_layer_names is available for later sections if needed
        if all_layer_names is None:
            all_layer_names = get_layer_names_for_model(args.model_name, model)
    elif args.skip_single_layers:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING GEOMETRIC SINGLE-LAYER CALIBRATIONS (--skip-single-layers flag set)")
        logger.info("="*80)
        logger.info("--skip-single-layers: Also skipping layer_quality_metrics, metric_guided_calibration, geometric_precomputed")
        # Initialize empty results for skipped sections
        if 'geometric_original' not in results:
            results['geometric_original'] = {"skipped": True, "reason": "--skip-single-layers flag"}
        if 'layer_quality_metrics' not in results:
            results['layer_quality_metrics'] = {"skipped": True}
        if 'metric_guided_calibration' not in results:
            results['metric_guided_calibration'] = {"skipped": True}
        if 'geometric_precomputed' not in results:
            results['geometric_precomputed'] = {"skipped": True}
        save_results_incrementally(results, output_file)
        geo_results_dict = {}
        layer_metrics = {}
        metric_selections = {}
        metric_guided_results = []  # Initialize empty list for skip case
        # Ensure all_layer_names is available for later sections
        if all_layer_names is None:
            all_layer_names = get_layer_names_for_model(args.model_name, model)
    elif should_rerun_section(results, "geometric_original", GEO_REQUIRED_FIELDS):
        logger.info("\n" + "="*80)
        logger.info("RUNNING (OR RESUMING) GEOMETRIC CALIBRATIONS FOR ALL LAYERS")
        logger.info("="*80)

        # 1) Reconstruct what you already have (even if partial)
        geo_results_dict: Dict[str, Dict[str, Any]] = {}
        if isinstance(results.get("geometric_original"), dict):
            for _, stored in results["geometric_original"].items():
                if not isinstance(stored, dict):
                    continue
                layer_name = stored.get("layer_names")

                # Single-layer results store string; be defensive if list
                if isinstance(layer_name, list) and len(layer_name) == 1:
                    layer_name = layer_name[0]

                if isinstance(layer_name, str):
                    geo_results_dict[layer_name] = stored

        # 2) Run only missing layers (missing 'calibrator_params', missing ece, or error)
        for i, layer_name in enumerate(all_layer_names, 1):
            semantic_name = get_semantic_layer_name(layer_name, args.model_name)

            existing = geo_results_dict.get(layer_name)
            if is_result_complete(existing, GEO_REQUIRED_FIELDS):
                logger.info(f"✓ Skipping layer {i}/{len(all_layer_names)}: {layer_name} (already has calibrator_params)")
                continue

            logger.info(f"\nCalibrating layer {i}/{len(all_layer_names)}: {layer_name} ({semantic_name})")
            result = run_geometric_calibration(
                model, args.model_name, args.dataset, model_adapter,
                train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
                device, layer_name, f"Geometric ({semantic_name})",
                args.batch_size, precomputed_features=precomputed_features,
                return_val_probs=True,
                seed=args.seed,
                pooling_mode=args.spp_pooling_mode,
                target_dimension=args.target_dimension,
                output_dir=args.output_dir,
                training_method=args.training_method
            )
            geo_results_dict[layer_name] = result

            # Persist after each layer so interruptions are safe
            results["geometric_original"] = {
                f"layer{j+1}": geo_results_dict[ln]
                for j, ln in enumerate(all_layer_names)
                if ln in geo_results_dict
            }
            save_results_incrementally(results, output_file)

        # 3) Finalize stable ordering
        results["geometric_original"] = {
            f"layer{i+1}": geo_results_dict[layer_name]
            for i, layer_name in enumerate(all_layer_names)
            if layer_name in geo_results_dict
        }
        save_results_incrementally(results, output_file)
    else:
        logger.info("✓ Geometric single-layer calibrations already computed, skipping")
        # Reconstruct geo_results_dict for use in metrics and later sections
        geo_results_dict = {}
        for _, result in results['geometric_original'].items():
            layer_name = result.get('layer_names')
            # Single-layer results store a string; be defensive if list
            if isinstance(layer_name, list) and len(layer_name) == 1:
                layer_name = layer_name[0]
            if isinstance(layer_name, str):
                geo_results_dict[layer_name] = result
        # Rebuild all_layer_names from keys if we didn't extract features this run
        if all_layer_names is None:
            all_layer_names = list(geo_results_dict.keys())

    # ===== LAYER QUALITY METRICS =====
    if methods_only_mode:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING LAYER QUALITY METRICS (methods-only mode)")
        logger.info("="*80)
        if 'layer_quality_metrics' not in results:
            results['layer_quality_metrics'] = {"skipped": True, "reason": "methods-only mode"}
        if 'metric_based_selections' not in results:
            results['metric_based_selections'] = {}
        layer_metrics = {}
        metric_selections = {}
    elif args.skip_single_layers:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING LAYER QUALITY METRICS (--skip-single-layers flag set)")
        logger.info("="*80)
        if 'layer_quality_metrics' not in results:
            results['layer_quality_metrics'] = {"skipped": True}
        if 'metric_based_selections' not in results:
            results['metric_based_selections'] = {}
        layer_metrics = {}
        metric_selections = {}
    else:
        needs_metrics = (
            'layer_quality_metrics' not in results or
            not results['layer_quality_metrics'] or
            should_rerun_section(results, 'metric_guided_calibration', GEO_REQUIRED_FIELDS)
        )

        if needs_metrics:
            logger.info("\n" + "="*80)
            logger.info("COMPUTING LAYER QUALITY METRICS")
            logger.info("="*80)

            # OPTIMIZATION: Use cached val_probs
            val_logits = prediction_cache['val_probs']
            layer_metrics = compute_all_layer_metrics_once(
                precomputed_features,
                train_labels,
                val_labels,
                val_logits,
                model_adapter=model_adapter,
                val_raw=val_raw,
                precomputed_calibrations=geo_results_dict
            )

            metric_selections = select_best_layer_by_metrics(layer_metrics)

            results['layer_quality_metrics'] = {
                layer_name: {k: float(v) if not np.isnan(v) else None for k, v in metrics.items()}
                for layer_name, metrics in layer_metrics.items()
            }
            results['metric_based_selections'] = metric_selections
            save_results_incrementally(results, output_file)
        else:
            logger.info("✓ Layer quality metrics already computed, skipping")
            layer_metrics = {
                layer_name: {
                    k: (np.nan if v is None else float(v))
                    for k, v in metrics.items()
                }
                for layer_name, metrics in results['layer_quality_metrics'].items()
            }
            metric_selections = results['metric_based_selections']

    # ===== DAC-STYLE WEIGHTED ENSEMBLE =====
    # DISABLED: Skip run_geometric_with_dac_weighting experiment
    if True:  # Always skip this experiment
        logger.info("\n" + "="*80)
        logger.info("SKIPPING DAC-STYLE WEIGHTED ENSEMBLE (DISABLED)")
        logger.info("="*80)
        logger.info("Reason: Experiment disabled by user request")
        results['geometric_dac_weighted'] = {
            "skipped": True,
            "reason": "Experiment disabled by user request"
        }
        save_results_incrementally(results, output_file)
        geo_with_dac_weighting = None
    elif is_dinov2:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING DAC-STYLE WEIGHTED ENSEMBLE (DINOv2)")
        logger.info("="*80)
        logger.info("Reason: DINOv2: DAC paper did not specify layers for transformer architectures")
        results['geometric_dac_weighted'] = {
            "skipped": True,
            "reason": "DINOv2: DAC paper did not specify layers for transformer architectures"
        }
        save_results_incrementally(results, output_file)
        geo_with_dac_weighting = None
    elif should_rerun_section(results, 'geometric_dac_weighted', DAC_REQUIRED_FIELDS):
        logger.info("\n" + "="*80)
        logger.info("RUNNING DAC-STYLE WEIGHTED ENSEMBLE")
        logger.info("="*80)

        geo_with_dac_weighting = run_geometric_with_dac_weighting(
            model, args.model_name, args.dataset, model_adapter,
            train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
            device, args.batch_size,
            precomputed_features=precomputed_features,
            precomputed_separations=precomputed_separations,
            pooling_mode=args.spp_pooling_mode,
            target_dimension=args.target_dimension,
            ood_raw=ood_raw,
            ood_labels=ood_labels,
            # OPTIMIZATION: Pass cached predictions and loaders
            precomputed_test_probs=prediction_cache['test_probs'],
            precomputed_ood_probs=prediction_cache.get('ood_probs'),
            precomputed_val_probs=prediction_cache['val_probs'],
            precomputed_val_preds=prediction_cache['val_preds'],
            precomputed_test_preds=prediction_cache['test_preds'],
            train_loader=dataloader_cache['train'],
            val_loader=dataloader_cache['val'],
            test_loader=dataloader_cache['test'],
            ood_loader=dataloader_cache.get('ood'),
            output_dir=args.output_dir,
            training_method=args.training_method
        )
        results['geometric_dac_weighted'] = geo_with_dac_weighting
        save_results_incrementally(results, output_file)
    else:
        logger.info("✓ DAC-style weighted ensemble already computed, skipping")
        geo_with_dac_weighting = results.get('geometric_dac_weighted')


    # ===== ABLATION STUDIES =====
    # Check if we need to run any ablation experiments (split into individual checks)
    if methods_only_mode:
        # In methods-only mode, only check for selected methods
        needs_sgc_dac_preprocessing = 'sgc_with_dac_preprocessing_separation' in selected_methods
        needs_sgc_dac_layers = 'sgc_with_dac_layers_separation' in selected_methods
        needs_sgc_tulip_layers = 'sgc_with_tulip_layers_separation' in selected_methods
        needs_dac_sgc_aggregated = False  # Not in selected methods
        needs_dac_random = False  # Not in selected methods
        needs_random_single_dac = False  # Not in selected methods
    else:
        needs_dac_sgc_aggregated = should_rerun_section(results, 'ablation.dac_with_sgc_aggregated_features', DAC_REQUIRED_FIELDS)
        needs_dac_random = should_rerun_section(results, 'ablation.dac_with_random_layer_selection', DAC_REQUIRED_FIELDS)
        needs_sgc_dac_preprocessing = should_rerun_section(results, 'ablation.sgc_with_dac_preprocessing', GEO_REQUIRED_FIELDS)
        needs_sgc_dac_layers = should_rerun_section(results, 'ablation.sgc_with_dac_layers', GEO_REQUIRED_FIELDS)
        needs_sgc_tulip_layers = should_rerun_section(results, 'ablation.sgc_with_tulip_layers', GEO_REQUIRED_FIELDS)
        needs_random_single_dac = should_rerun_section(results, 'ablation.random_single_layer_dac_preprocessing', GEO_REQUIRED_FIELDS)
    
    if needs_dac_sgc_aggregated or needs_dac_random or needs_sgc_dac_preprocessing or needs_sgc_dac_layers or needs_sgc_tulip_layers or needs_random_single_dac:
        logger.info("\n" + "="*80)
        logger.info("ABLATION STUDY: FEATURE EXTRACTION vs CALIBRATION ALGORITHM")
        logger.info("="*80)
        
        # Initialize ablation dict if needed
        if 'ablation' not in results:
            results['ablation'] = {}
        
        # DEPRECATED: Cross-comparison 1: Geometric algorithm + DAC features (ORACLE METHOD)
        # This experiment is deprecated because it selects the best of 5 layers post-hoc,
        # making it an unfair oracle baseline. Use run_sgc_with_dac_preprocessing instead.
        # if 'geometric_with_dac_features' not in results['ablation']:
        #     dac_target_layers = get_dac_target_layers(args.model_name, model)
        #     geo_with_dac_features_list = []
        #     for i in range(len(dac_target_layers)):
        #         result = run_geometric_with_dac_features(
        #             model, args.model_name, args.dataset, model_adapter,
        #             train_loader, val_loader, test_loader,
        #             train_labels, val_labels, test_labels,
        #             train_raw, val_raw, test_raw,
        #             device, i, args.batch_size,
        #             target_dimension=args.target_dimension
        #         )
        #         geo_with_dac_features_list.append(result)
        #     results['ablation']['geometric_with_dac_features'] = {
        #         f'layer{i+1}': geo_with_dac_features_list[i] for i in range(len(dac_target_layers))
        #     }
        #     save_results_incrementally(results, output_file)
        # else:
        #     logger.info("✓ Geometric with DAC features already computed, skipping")

        # Cross-comparison 2: DAC algorithm + SGC aggregated features (NOT a fair comparison)
        # Skip in methods-only mode (not in selected methods)
        if methods_only_mode:
            dac_with_sgc_aggregated = None
        elif is_dinov2 and needs_dac_sgc_aggregated:
            logger.info("\nSkipping: DAC with SGC aggregated features (DINOv2)")
            logger.info("Reason: DINOv2: DAC paper did not specify layers for transformer architectures")
            results['ablation']['dac_with_sgc_aggregated_features'] = {
                "skipped": True,
                "reason": "DINOv2: DAC paper did not specify layers for transformer architectures"
            }
            save_results_incrementally(results, output_file)
            dac_with_sgc_aggregated = None
        elif needs_dac_sgc_aggregated:
            logger.info("\nRunning: DAC with SGC aggregated features (NOT a fair layer selection comparison)")
            dac_with_sgc_aggregated = run_dac_with_sgc_aggregated_features(
                model, args.model_name, args.dataset,
                train_loader, val_loader, test_loader, device,
                target_dim=args.target_dimension,
                seed=args.seed,
                pooling_mode=args.spp_pooling_mode,
                ood_raw=ood_raw,
                ood_labels=ood_labels,
                dac_knn_backend=args.dac_knn_backend
            )
            # Handle "both" mode results
            if args.dac_knn_backend == 'both' and isinstance(dac_with_sgc_aggregated, dict) and 'dac_with_sgc_aggregated_features_torch' in dac_with_sgc_aggregated:
                results['ablation']['dac_with_sgc_aggregated_features'] = dac_with_sgc_aggregated.get('dac_with_sgc_aggregated_features', dac_with_sgc_aggregated)
                results['ablation']['dac_with_sgc_aggregated_features_torch'] = dac_with_sgc_aggregated.get('dac_with_sgc_aggregated_features_torch')
            else:
                results['ablation']['dac_with_sgc_aggregated_features'] = dac_with_sgc_aggregated
            save_results_incrementally(results, output_file)
        else:
            logger.info("✓ DAC with SGC aggregated features already computed, skipping")
            dac_with_sgc_aggregated = results['ablation'].get('dac_with_sgc_aggregated_features')

        # Cross-comparison 3: DAC algorithm + Random layer selection (FAIR comparison to SGC)
        # Skip in methods-only mode (not in selected methods)
        if methods_only_mode:
            dac_with_random_layers = None
        elif needs_dac_random:
            logger.info("\nRunning: DAC with random layer selection (FAIR comparison to SGC)")
            dac_with_random_layers = run_dac_with_random_layer_selection(
                model, args.model_name, args.dataset,
                train_loader, val_loader, test_loader, device,
                num_layers=args.num_layers,
                seed=args.seed,
                ood_raw=ood_raw,
                ood_labels=ood_labels,
                batch_size=args.batch_size,
                dac_knn_backend=args.dac_knn_backend
            )
            # Handle "both" mode results
            if args.dac_knn_backend == 'both' and isinstance(dac_with_random_layers, dict) and 'dac_with_random_layer_selection_torch' in dac_with_random_layers:
                results['ablation']['dac_with_random_layer_selection'] = dac_with_random_layers.get('dac_with_random_layer_selection', dac_with_random_layers)
                results['ablation']['dac_with_random_layer_selection_torch'] = dac_with_random_layers.get('dac_with_random_layer_selection_torch')
            else:
                results['ablation']['dac_with_random_layer_selection'] = dac_with_random_layers
            save_results_incrementally(results, output_file)
        else:
            logger.info("✓ DAC with random layer selection already computed, skipping")
            dac_with_random_layers = results['ablation'].get('dac_with_random_layer_selection')
        
        # ===== NEW ABLATION: SGC with DAC Preprocessing (SEPARATION & TRUST SCORE) =====
        # This section runs regardless of methods_only_mode because it's checked individually
        if 'sgc_with_dac_preprocessing' not in results['ablation']:
            results['ablation']['sgc_with_dac_preprocessing'] = {}
        
        if is_dinov2 and needs_sgc_dac_preprocessing:
            logger.info("\nSkipping: SGC with DAC preprocessing (DINOv2)")
            logger.info("Reason: DINOv2: DAC paper did not specify layers for transformer architectures")
            for scoring_method in ['separation', 'trust_score']:
                section_key = f'sgc_with_dac_preprocessing_{scoring_method}'
                results['ablation'][section_key] = {
                    "skipped": True,
                    "reason": "DINOv2: DAC paper did not specify layers for transformer architectures"
                }
            save_results_incrementally(results, output_file)
        elif needs_sgc_dac_preprocessing:
            # Check if any scoring method needs to run
            needs_any_dac_preprocessing = any(
                should_rerun_section(results, f'ablation.sgc_with_dac_preprocessing_{sm}', GEO_REQUIRED_FIELDS)
                for sm in ['separation', 'trust_score']
            )
            
            # Extract features once for both scoring methods (if needed)
            shared_dac_preprocessing_train_features = None
            shared_dac_preprocessing_val_features = None
            shared_dac_preprocessing_test_features = None
            shared_dac_preprocessing_selected_layers = None
            
            if needs_any_dac_preprocessing:
                # Discover layers and select randomly (same seed = same selection)
                is_dinov2 = args.model_name is not None and 'dinov2' in args.model_name.lower()
                if is_dinov2:
                    input_shape = (1, 3, 224, 224)
                elif args.dataset.lower() in ["tiny_imagenet", "tinyimagenet"]:
                    input_shape = (1, 3, 64, 64)
                else:
                    input_shape = (1, 3, 32, 32)
                
                discovered_layers = normalize_discovered_layers(
                    model, discover_model_layers(model, device=device, input_shape=input_shape)
                )
                filtered_layers, excluded_layers = filter_non_feature_layers(discovered_layers, model)
                all_layer_names = [d["name"] for d in filtered_layers]
                logger.info(f"Discovered {len(discovered_layers)} total layers, {len(all_layer_names)} feature-bearing layers after filtering")
                
                np.random.seed(args.seed)
                if args.num_layers > len(all_layer_names):
                    shared_dac_preprocessing_selected_layers = all_layer_names
                else:
                    shared_dac_preprocessing_selected_layers = list(np.random.choice(all_layer_names, size=args.num_layers, replace=False))
                
                logger.info(f"Selected {len(shared_dac_preprocessing_selected_layers)} layers: {shared_dac_preprocessing_selected_layers}")
                
                # Create dataloaders
                train_loader_dac = DataLoader(
                    TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                    batch_size=args.batch_size, shuffle=False
                )
                val_loader_dac = DataLoader(
                    TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                    batch_size=args.batch_size, shuffle=False
                )
                test_loader_dac = DataLoader(
                    TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
                    batch_size=args.batch_size, shuffle=False
                )
                
                # Extract features once (shared across scoring methods)
                logger.info("\n" + "="*80)
                logger.info("EXTRACTING SGC WITH DAC PREPROCESSING FEATURES (shared across scoring methods)")
                logger.info("="*80)
                logger.info("This extraction will be reused for both separation and trust_score methods")
                
                # Extract features using DAC preprocessing for each layer
                logger.info("1. Extracting features with DAC preprocessing (spatial avg + L2)...")
                start_extract = time.perf_counter()
                
                train_features_all, _, _ = extract_dac_features_memory_efficient(
                    model, train_loader_dac, shared_dac_preprocessing_selected_layers, device, max_layers_per_batch=4
                )
                val_features_all, _, _ = extract_dac_features_memory_efficient(
                    model, val_loader_dac, shared_dac_preprocessing_selected_layers, device, max_layers_per_batch=4
                )
                test_features_all, _, _ = extract_dac_features_memory_efficient(
                    model, test_loader_dac, shared_dac_preprocessing_selected_layers, device, max_layers_per_batch=4
                )

                # Apply L2 normalization per layer
                logger.info("2. Applying L2 normalization per layer...")
                train_features_normalized = []
                val_features_normalized = []
                test_features_normalized = []

                for i in range(len(shared_dac_preprocessing_selected_layers)):
                    train_feat = train_features_all[i].numpy() if isinstance(train_features_all[i], torch.Tensor) else train_features_all[i]
                    val_feat = val_features_all[i].numpy() if isinstance(val_features_all[i], torch.Tensor) else val_features_all[i]
                    test_feat = test_features_all[i].numpy() if isinstance(test_features_all[i], torch.Tensor) else test_features_all[i]

                    train_features_normalized.append(l2_normalize_np(train_feat))
                    val_features_normalized.append(l2_normalize_np(val_feat))
                    test_features_normalized.append(l2_normalize_np(test_feat))

                # Concatenate all layers
                logger.info("3. Concatenating layer features...")
                train_features_concat = np.concatenate(train_features_normalized, axis=1)
                val_features_concat = np.concatenate(val_features_normalized, axis=1)
                test_features_concat = np.concatenate(test_features_normalized, axis=1)

                logger.info(f"  Concatenated shape: {train_features_concat.shape}")

                # Apply JL projection if needed (Torch, GPU)
                original_dim = train_features_concat.shape[1]
                if args.target_dimension is not None and original_dim > args.target_dimension:
                    logger.info(f"4. Applying JL projection (Torch, GPU): {original_dim} -> {args.target_dimension} dims")

                    class TorchGaussianRandomProjectionJL:
                        """
                        Deterministic Gaussian random projection implemented in PyTorch.

                        Matches sklearn.GaussianRandomProjection scaling:
                          components_ ~ N(0, 1 / sqrt(n_components))
                        so each entry has variance 1 / n_components.
                        """

                        def __init__(self, in_dim: int, out_dim: int, seed: int, device: torch.device):
                            self.in_dim = int(in_dim)
                            self.out_dim = int(out_dim)
                            self.seed = int(seed)
                            self.device = device

                            scale = 1.0 / (float(self.out_dim) ** 0.5)
                            gen = torch.Generator(device=device)
                            gen.manual_seed(self.seed)

                            self.R = torch.randn(
                                (self.out_dim, self.in_dim),
                                device=device,
                                dtype=torch.float32,
                                generator=gen,
                            ) * scale

                        def project_np(self, x_np: np.ndarray) -> np.ndarray:
                            x_t = torch.from_numpy(x_np).to(self.device, dtype=torch.float32)
                            y_t = torch.nn.functional.linear(x_t, self.R)
                            return y_t.detach().cpu().numpy().astype(np.float32, copy=False)

                    rp = TorchGaussianRandomProjectionJL(
                        in_dim=original_dim,
                        out_dim=args.target_dimension,
                        seed=args.seed,
                        device=device,
                    )
                    shared_dac_preprocessing_train_features = rp.project_np(train_features_concat)
                    shared_dac_preprocessing_val_features = rp.project_np(val_features_concat)
                    shared_dac_preprocessing_test_features = rp.project_np(test_features_concat)
                else:
                    logger.info(f"  No compression needed (dim={original_dim}, target={args.target_dimension})")
                    shared_dac_preprocessing_train_features = train_features_concat
                    shared_dac_preprocessing_val_features = val_features_concat
                    shared_dac_preprocessing_test_features = test_features_concat

                # Final L2 normalization
                logger.info("5. Applying final L2 normalization...")
                shared_dac_preprocessing_train_features = l2_normalize_np(shared_dac_preprocessing_train_features)
                shared_dac_preprocessing_val_features = l2_normalize_np(shared_dac_preprocessing_val_features)
                shared_dac_preprocessing_test_features = l2_normalize_np(shared_dac_preprocessing_test_features)
                
                extract_time = time.perf_counter() - start_extract
                logger.info(f"Feature extraction completed. Time: {extract_time:.2f}s, Final shape: {shared_dac_preprocessing_train_features.shape}")
            
            # Run calibration for each scoring method
            for scoring_method in ['separation', 'trust_score']:
                section_key = f'sgc_with_dac_preprocessing_{scoring_method}'
                score_label = "Trust Score" if scoring_method == 'trust_score' else "Separation"
                
                # In methods-only mode, only run if this specific method is selected
                if methods_only_mode:
                    should_run = (section_key in selected_methods)
                else:
                    should_run = should_rerun_section(results, f'ablation.{section_key}', GEO_REQUIRED_FIELDS)
                
                if should_run:
                    logger.info(f"\nRunning: SGC with DAC preprocessing ({score_label}) - fair comparison")
                    sgc_dac_preprocess = run_sgc_with_dac_preprocessing(
                        model, args.model_name, args.dataset, model_adapter,
                        train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
                        device,
                        num_layers=args.num_layers,
                        target_dim=args.target_dimension,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        ood_raw=ood_raw,
                        ood_labels=ood_labels,
                        scoring_method=scoring_method,
                        precomputed_train_features=shared_dac_preprocessing_train_features,
                        precomputed_val_features=shared_dac_preprocessing_val_features,
                        precomputed_test_features=shared_dac_preprocessing_test_features,
                        precomputed_selected_layers=shared_dac_preprocessing_selected_layers
                    )
                    results['ablation'][section_key] = sgc_dac_preprocess
                    save_results_incrementally(results, output_file)
                else:
                    logger.info(f"✓ SGC with DAC preprocessing ({score_label}) already computed, skipping")
        else:
            logger.info("✓ SGC with DAC preprocessing already computed, skipping")
        
        # ===== NEW ABLATION: Random Single Layer with DAC Preprocessing =====
        # Skip in methods-only mode (not in selected methods)
        if methods_only_mode:
            random_single_dac = None
        elif is_dinov2 and needs_random_single_dac:
            logger.info("\nSkipping: Random single layer with DAC preprocessing (DINOv2)")
            logger.info("Reason: DINOv2: DAC paper did not specify layers for transformer architectures")
            results['ablation']['random_single_layer_dac_preprocessing'] = {
                "skipped": True,
                "reason": "DINOv2: DAC paper did not specify layers for transformer architectures"
            }
            save_results_incrementally(results, output_file)
        elif needs_random_single_dac:
            logger.info("\nRunning: Random single layer with DAC preprocessing")
            random_single_dac = run_random_single_layer_with_dac_preprocessing(
                model, args.model_name, args.dataset, model_adapter,
                train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
                device,
                target_dim=args.target_dimension,
                batch_size=args.batch_size,
                seed=args.seed,
                ood_raw=ood_raw,
                ood_labels=ood_labels
            )
            results['ablation']['random_single_layer_dac_preprocessing'] = random_single_dac
            save_results_incrementally(results, output_file)
        else:
            logger.info("✓ Random single layer with DAC preprocessing already computed, skipping")
    else:
        logger.info("✓ All ablation studies already computed, skipping")

    # ===== SGC WITH DAC LAYER SELECTION (SEPARATION & TRUST SCORE) =====
    # This uses DAC's layer selection with the standard SGC pipeline (project-then-sum + L2 norm)
    # Key difference from regular SGC: uses DAC's predetermined layers instead of random selection
    if methods_only_mode and 'sgc_with_dac_layers_separation' not in selected_methods:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING SGC WITH DAC LAYERS (not in selected methods)")
        logger.info("="*80)
        if 'sgc_with_dac_layers' not in results:
            results['sgc_with_dac_layers'] = {}
        for scoring_method in ['separation', 'trust_score']:
            section_key = f'sgc_with_dac_layers_{scoring_method}'
            results[section_key] = {"skipped": True, "reason": "methods-only mode: not selected"}
        save_results_incrementally(results, output_file)
    elif 'sgc_with_dac_layers' not in results:
        results['sgc_with_dac_layers'] = {}
    
    if is_dinov2:
        logger.info("Skipping: SGC with DAC layers (DINOv2: DAC paper did not specify layers)")
        for scoring_method in ['separation', 'trust_score']:
            section_key = f'sgc_with_dac_layers_{scoring_method}'
            results[section_key] = {
                "skipped": True,
                "reason": "DINOv2: DAC paper did not specify layers for transformer architectures"
            }
        save_results_incrementally(results, output_file)
    else:
        # Check if any scoring method needs to run
        if methods_only_mode:
            needs_extraction = 'sgc_with_dac_layers_separation' in selected_methods
        else:
            needs_extraction = any(
                should_rerun_section(results, f'sgc_with_dac_layers_{sm}', GEO_REQUIRED_FIELDS)
                for sm in ['separation', 'trust_score']
            )
        
        # Extract features once for both scoring methods (if needed)
        shared_train_features = None
        shared_val_features = None
        shared_test_features = None
        shared_extraction_info = None
        
        if needs_extraction:
            # Get DAC layers first
            try:
                selected_layers = get_dac_target_layers(args.model_name, model)
                logger.info(f"Using DAC layer selection: {selected_layers}")
                
                if not selected_layers or len(selected_layers) == 0:
                    raise ValueError("DAC layer selection returned empty list")
                    
            except Exception as e:
                logger.error(f"Failed to get DAC target layers: {e}")
                raise ValueError(f"Could not determine DAC layers for {args.model_name}: {e}")
            
            # Create dataloaders for feature extraction
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=args.batch_size, shuffle=False
            )
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                batch_size=args.batch_size, shuffle=False
            )
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
                batch_size=args.batch_size, shuffle=False
            )
            
            # Extract features once (shared across scoring methods)
            logger.info("\n" + "="*80)
            logger.info("EXTRACTING SGC FEATURES (shared across scoring methods)")
            logger.info("="*80)
            logger.info("This extraction will be reused for both separation and trust_score methods")
            
            shared_train_features, shared_val_features, shared_test_features, shared_extraction_info = extract_and_aggregate_sgc_features(
                model=model,
                layer_names=selected_layers,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                target_dim=args.target_dimension,
                seed=args.seed,
                pooling_mode=args.spp_pooling_mode
            )
            logger.info(f"Feature extraction completed. Final feature shape: {shared_train_features.shape}")
        
        # Run calibration for each scoring method
        for scoring_method in ['separation', 'trust_score']:
            section_key = f'sgc_with_dac_layers_{scoring_method}'
            score_label = "Trust Score" if scoring_method == 'trust_score' else "Separation"
            
            # In methods-only mode, only run if this specific method is selected
            if methods_only_mode:
                should_run = (section_key in selected_methods)
            else:
                should_run = should_rerun_section(results, section_key, GEO_REQUIRED_FIELDS)
            
            if should_run:
                logger.info("\n" + "="*80)
                logger.info(f"SGC WITH DAC LAYER SELECTION - {score_label}")
                logger.info("="*80)
                logger.info("Running SGC pipeline with DAC's selected layers")
                
                try:
                    # Use the new function that follows the standard SGC pipeline
                    # Pass precomputed features if available
                    sgc_dac_layers_result = run_sgc_with_dac_layers(
                        model=model,
                        model_name=args.model_name,
                        dataset_name=args.dataset,
                        model_adapter=model_adapter,
                        train_raw=train_raw,
                        train_labels=train_labels,
                        val_raw=val_raw,
                        val_labels=val_labels,
                        test_raw=test_raw,
                        test_labels=test_labels,
                        device=device,
                        target_dim=args.target_dimension,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        pooling_mode=args.spp_pooling_mode,
                        output_file=output_file,
                        results_dict=results,
                        scoring_method=scoring_method,
                        ood_raw=ood_raw,
                        ood_labels=ood_labels,
                        training_method=args.training_method,
                        precomputed_train_features=shared_train_features,
                        precomputed_val_features=shared_val_features,
                        precomputed_test_features=shared_test_features,
                        precomputed_extraction_info=shared_extraction_info,
                        # OPTIMIZATION: Pass cached predictions and loaders
                        precomputed_test_probs=prediction_cache['test_probs'],
                        precomputed_ood_probs=prediction_cache.get('ood_probs'),
                        train_loader=dataloader_cache['train'],
                        val_loader=dataloader_cache['val'],
                        test_loader=dataloader_cache['test'],
                        ood_loader=dataloader_cache.get('ood')
                    )
                    results[section_key] = sgc_dac_layers_result
                    save_results_incrementally(results, output_file)
                    logger.info(f"✅ SGC with DAC layers ({score_label}) completed: ECE={sgc_dac_layers_result.get('ece', 'N/A'):.4f}")
                except Exception as e:
                    logger.error(f"SGC with DAC layers ({score_label}) FAILED: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    results[section_key] = {
                        'method': f'SGC (DAC Layers) - {score_label} - FAILED',
                        'ece': float('inf'),
                        'error': str(e)
                    }
                    save_results_incrementally(results, output_file)
            else:
                logger.info(f"✓ SGC with DAC layer selection ({score_label}) already computed, skipping")

    # ===== SGC WITH TULIP LAYER SELECTION =====
    if methods_only_mode and 'sgc_with_tulip_layers_separation' not in selected_methods:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING SGC WITH TULIP LAYERS (not in selected methods)")
        logger.info("="*80)
        if 'sgc_with_tulip_layers' not in results:
            results['sgc_with_tulip_layers'] = {}
        for scoring_method in ['separation', 'trust_score']:
            section_key = f'sgc_with_tulip_layers_{scoring_method}'
            results[section_key] = {"skipped": True, "reason": "methods-only mode: not selected"}
        save_results_incrementally(results, output_file)
    elif 'sgc_with_tulip_layers' not in results:
        results['sgc_with_tulip_layers'] = {}
    
    if is_dinov2:
        logger.info("Skipping: SGC with TULIP layers (DINOv2: TULIP is designed for ResNets)")
        for scoring_method in ['separation', 'trust_score']:
            section_key = f'sgc_with_tulip_layers_{scoring_method}'
            results[section_key] = {
                "skipped": True,
                "reason": "DINOv2: TULIP layer selection is designed for ResNet architectures"
            }
        save_results_incrementally(results, output_file)
    elif 'densenet' in args.model_name.lower():
        logger.info("Skipping: SGC with TULIP layers (DenseNet: TULIP is designed for ResNets)")
        for scoring_method in ['separation', 'trust_score']:
            section_key = f'sgc_with_tulip_layers_{scoring_method}'
            results[section_key] = {
                "skipped": True,
                "reason": "DenseNet: TULIP layer selection is designed for ResNet architectures"
            }
        save_results_incrementally(results, output_file)
    else:
        # Check if any scoring method needs to run
        if methods_only_mode:
            needs_extraction = 'sgc_with_tulip_layers_separation' in selected_methods
        else:
            needs_extraction = any(
                should_rerun_section(results, f'sgc_with_tulip_layers_{sm}', GEO_REQUIRED_FIELDS)
                for sm in ['separation', 'trust_score']
            )
        
        # Extract features once for both scoring methods (if needed)
        tulip_train_features = None
        tulip_val_features = None
        tulip_test_features = None
        tulip_extraction_info = None
        tulip_layer_names = None
        
        if needs_extraction:
            # Get TULIP layers
            try:
                tulip_layer_names = select_tulip_layers(model, args.model_name)
                logger.info(f"Using TULIP layer selection: {tulip_layer_names}")
                
                if not tulip_layer_names or len(tulip_layer_names) == 0:
                    raise ValueError("TULIP layer selection returned empty list")
                    
            except Exception as e:
                logger.error(f"Failed to get TULIP target layers: {e}")
                raise ValueError(f"Could not determine TULIP layers for {args.model_name}: {e}")
            
            # Use cached loaders if available, otherwise create new ones
            train_loader_tulip = dataloader_cache.get('train') or DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=args.batch_size, shuffle=False
            )
            val_loader_tulip = dataloader_cache.get('val') or DataLoader(
                TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                batch_size=args.batch_size, shuffle=False
            )
            test_loader_tulip = dataloader_cache.get('test') or DataLoader(
                TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
                batch_size=args.batch_size, shuffle=False
            )
            
            # Extract features once (shared across scoring methods)
            logger.info("\n" + "="*80)
            logger.info("EXTRACTING SGC FEATURES WITH TULIP LAYERS")
            logger.info("="*80)
            
            tulip_train_features, tulip_val_features, tulip_test_features, tulip_extraction_info = extract_and_aggregate_sgc_features(
                model=model,
                layer_names=tulip_layer_names,
                train_loader=train_loader_tulip,
                val_loader=val_loader_tulip,
                test_loader=test_loader_tulip,
                device=device,
                target_dim=args.target_dimension,
                seed=args.seed,
                pooling_mode=args.spp_pooling_mode
            )
            logger.info(f"Feature extraction completed. Final feature shape: {tulip_train_features.shape}")
        
        # Run calibration for each scoring method
        for scoring_method in ['separation', 'trust_score']:
            section_key = f'sgc_with_tulip_layers_{scoring_method}'
            score_label = "Trust Score" if scoring_method == 'trust_score' else "Separation"
            
            # In methods-only mode, only run if this specific method is selected
            if methods_only_mode:
                should_run = (section_key in selected_methods)
            else:
                should_run = should_rerun_section(results, section_key, GEO_REQUIRED_FIELDS)
            
            if should_run:
                logger.info("\n" + "="*80)
                logger.info(f"SGC WITH TULIP LAYER SELECTION - {score_label}")
                logger.info("="*80)
                
                try:
                    # Run SGC with TULIP layers
                    sgc_tulip_result = run_sgc_with_tulip_layers(
                        model=model,
                        model_name=args.model_name,
                        dataset_name=args.dataset,
                        model_adapter=model_adapter,
                        train_raw=train_raw,
                        train_labels=train_labels,
                        val_raw=val_raw,
                        val_labels=val_labels,
                        test_raw=test_raw,
                        test_labels=test_labels,
                        device=device,
                        target_dim=args.target_dimension,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        pooling_mode=args.spp_pooling_mode,
                        output_file=output_file,
                        results_dict=results,
                        scoring_method=scoring_method,
                        ood_raw=ood_raw,
                        ood_labels=ood_labels,
                        training_method=args.training_method,
                        precomputed_train_features=tulip_train_features,
                        precomputed_val_features=tulip_val_features,
                        precomputed_test_features=tulip_test_features,
                        precomputed_extraction_info=tulip_extraction_info,
                        # OPTIMIZATION: Pass cached predictions and loaders
                        precomputed_test_probs=prediction_cache['test_probs'],
                        precomputed_ood_probs=prediction_cache.get('ood_probs'),
                        train_loader=dataloader_cache['train'],
                        val_loader=dataloader_cache['val'],
                        test_loader=dataloader_cache['test'],
                        ood_loader=dataloader_cache.get('ood')
                    )
                    
                    sgc_tulip_result['method'] = f'SGC (TULIP Layers) - {score_label}'
                    sgc_tulip_result['selected_layers'] = tulip_layer_names
                    sgc_tulip_result['layer_selection'] = 'tulip'
                    
                    results[section_key] = sgc_tulip_result
                    save_results_incrementally(results, output_file)
                    logger.info(f"✅ SGC with TULIP layers ({score_label}) completed: ECE={sgc_tulip_result.get('ece', 'N/A'):.4f}")
                    
                except Exception as e:
                    logger.error(f"SGC with TULIP layers ({score_label}) FAILED: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    results[section_key] = {
                        'method': f'SGC (TULIP Layers) - {score_label} - FAILED',
                        'ece': float('inf'),
                        'error': str(e)
                    }
                    save_results_incrementally(results, output_file)
            else:
                logger.info(f"✓ SGC with TULIP layer selection ({score_label}) already computed, skipping")

    # ===== METRIC-GUIDED LAYER SELECTION =====
    if methods_only_mode:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING METRIC-GUIDED LAYER SELECTION (methods-only mode)")
        logger.info("="*80)
        metric_guided_results = []
        if 'metric_guided_calibration' not in results:
            results['metric_guided_calibration'] = {"skipped": True, "reason": "methods-only mode"}
            save_results_incrementally(results, output_file)
    elif args.skip_single_layers:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING METRIC-GUIDED LAYER SELECTION (--skip-single-layers flag set)")
        logger.info("="*80)
        metric_guided_results = []
        if 'metric_guided_calibration' not in results:
            results['metric_guided_calibration'] = {"skipped": True}
            save_results_incrementally(results, output_file)
    elif should_rerun_section(results, 'metric_guided_calibration', GEO_REQUIRED_FIELDS):
        logger.info("\n" + "="*80)
        logger.info("RUNNING (OR RESUMING) METRIC-GUIDED LAYER SELECTION EXPERIMENTS")
        logger.info("="*80)

        # 1) Reconstruct what we already have (even if partial)
        existing_by_metric: Dict[str, Dict[str, Any]] = {}
        if isinstance(results.get('metric_guided_calibration'), list):
            for stored in results['metric_guided_calibration']:
                if isinstance(stored, dict) and 'selection_metric' in stored:
                    existing_by_metric[stored['selection_metric']] = stored

        metric_guided_results = []
        for metric_name_key, selected_layer in metric_selections.items():
            # Check if we already have a complete result for this metric
            existing = existing_by_metric.get(metric_name_key)
            if is_result_complete(existing, GEO_REQUIRED_FIELDS):
                logger.info(f"✓ Skipping metric {metric_name_key} (already has calibrator_params)")
                metric_guided_results.append(existing)
                continue

            logger.info(f"\nTesting layer selected by {metric_name_key}: {selected_layer}")

            if "PSC" in metric_name_key:
                base_metric_name = "nc4"
            else:
                base_metric_name = metric_name_key.replace(" (Max)", "").replace(" (Min)", "").replace(" (Med)", "")

            # Reuse precomputed result if available
            if selected_layer in geo_results_dict:
                logger.info(f"  Reusing precomputed calibration result for {selected_layer}")
                result = geo_results_dict[selected_layer].copy()
                result['method'] = f"Geometric ({metric_name_key} → {selected_layer})"
            else:
                result = run_geometric_calibration(
                    model, args.model_name, args.dataset, model_adapter,
                    train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
                    device, selected_layer, f"Geometric ({metric_name_key} → {selected_layer})",
                    args.batch_size, precomputed_features=precomputed_features,
                    seed=args.seed,
                    pooling_mode=args.spp_pooling_mode,
                    target_dimension=args.target_dimension,
                    ood_raw=ood_raw,
                    ood_labels=ood_labels,
                    training_method=args.training_method
                )

            result['selection_metric'] = metric_name_key
            result['metric_value'] = layer_metrics[selected_layer].get(base_metric_name, float('nan'))
            metric_guided_results.append(result)

            # Persist after each metric so interruptions are safe
            results['metric_guided_calibration'] = metric_guided_results
            save_results_incrementally(results, output_file)

        results['metric_guided_calibration'] = metric_guided_results
        save_results_incrementally(results, output_file)
    elif not args.skip_single_layers:
        # Only load from results if we didn't skip this section
        logger.info("✓ Metric-guided layer selection already computed, skipping")
        if 'metric_guided_calibration' in results:
            if isinstance(results['metric_guided_calibration'], list):
                metric_guided_results = results['metric_guided_calibration']
            else:
                metric_guided_results = []
        else:
            metric_guided_results = []

    # ===== LAST LAYER ONLY BASELINE =====
    if methods_only_mode and 'last_layer_only_baseline' not in selected_methods:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING LAST LAYER ONLY BASELINE (not in selected methods)")
        logger.info("="*80)
        if 'last_layer_only_baseline' not in results:
            results['last_layer_only_baseline'] = {"skipped": True, "reason": "methods-only mode: not selected"}
        save_results_incrementally(results, output_file)
    elif should_rerun_section(results, 'last_layer_only_baseline', GEO_REQUIRED_FIELDS):
        logger.info("\n" + "="*80)
        logger.info("LAST LAYER ONLY BASELINE (Reviewer Request)")
        logger.info("="*80)
        logger.info("Testing if multi-layer aggregation adds value over single penultimate layer")
        
        target_dim = args.target_dimension
        
        last_layer_results = run_last_layer_only_baseline(
            model, args.model_name, args.dataset, model_adapter,
            train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
            device,
            target_dim=target_dim,
            batch_size=args.batch_size,
            seed=args.seed,
            pooling_mode=args.spp_pooling_mode,
            precomputed_features=precomputed_features,
            ood_raw=ood_raw,
            ood_labels=ood_labels,
            # OPTIMIZATION: Pass cached predictions and loaders
            precomputed_test_probs=prediction_cache['test_probs'],
            precomputed_ood_probs=prediction_cache.get('ood_probs'),
            train_loader=dataloader_cache['train'],
            val_loader=dataloader_cache['val'],
            test_loader=dataloader_cache['test'],
            ood_loader=dataloader_cache.get('ood')
        )
        results['last_layer_only_baseline'] = last_layer_results
        save_results_incrementally(results, output_file)
    else:
        logger.info("✓ Last layer only baseline already computed, skipping")
        last_layer_results = results['last_layer_only_baseline']

    # ===== GLOBAL RANDOM CALIBRATION (SEPARATION & TRUST SCORE) =====
    if methods_only_mode and 'global_random_separation' not in selected_methods and 'sgc_faiss' not in selected_methods:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING GLOBAL RANDOM CALIBRATION (not in selected methods)")
        logger.info("="*80)
        if 'global_random' not in results:
            results['global_random'] = {}
        for key in ['global_random_separation', 'global_random_trust_score']:
            results[key] = {"skipped": True, "reason": "methods-only mode: not selected"}
        save_results_incrementally(results, output_file)
        # Set empty result for backward compatibility
        global_random_result = results.get('global_random_separation', results.get('global_random', {}))
    else:
        if 'global_random' not in results:
            results['global_random'] = {}
        
        # Use new arguments if provided, otherwise fall back to old ones for backward compatibility
        num_layers = getattr(args, 'num_layers', args.global_random_layers)
        target_dim = args.target_dimension
        
        # Check if any experiment in this group needs to run
        if methods_only_mode:
            needs_any_global_random = ('global_random_separation' in selected_methods) or ('sgc_faiss' in selected_methods)
        else:
            needs_any_global_random = any(
                should_rerun_section(results, key, GEO_REQUIRED_FIELDS)
                for key in ['global_random_separation', 'global_random_trust_score', 'sgc_faiss']
            )
        
        # Extract features once for all experiments in this group (if needed)
        shared_global_random_train_features = None
        shared_global_random_val_features = None
        shared_global_random_test_features = None
        shared_global_random_extraction_info = None
        shared_global_random_selected_layers = None
        
        if needs_any_global_random:
            # Discover layers and select randomly (same seed = same selection)
            is_dinov2 = args.model_name is not None and 'dinov2' in args.model_name.lower()
            if is_dinov2:
                input_shape = (1, 3, 224, 224)
            elif args.dataset.lower() in ["tiny_imagenet", "tinyimagenet"]:
                input_shape = (1, 3, 64, 64)
            else:
                input_shape = (1, 3, 32, 32)
            
            discovered_layers = normalize_discovered_layers(
                model, discover_model_layers(model, device=device, input_shape=input_shape)
            )
            filtered_layers, excluded_layers = filter_non_feature_layers(discovered_layers, model)
            all_layer_names = [d["name"] for d in filtered_layers]
            logger.info(f"Discovered {len(discovered_layers)} total layers, {len(all_layer_names)} feature-bearing layers after filtering")
            
            np.random.seed(args.seed)
            if num_layers > len(all_layer_names):
                shared_global_random_selected_layers = all_layer_names
            else:
                shared_global_random_selected_layers = list(np.random.choice(all_layer_names, size=num_layers, replace=False))
            
            logger.info(f"Selected {len(shared_global_random_selected_layers)} layers: {shared_global_random_selected_layers}")
            
            # Save layer selection for reuse
            if 'global_random_layer_selection' not in results:
                results['global_random_layer_selection'] = {
                    'selected_layers': shared_global_random_selected_layers,
                    'num_layers': num_layers,
                    'target_dim': target_dim,
                    'total_discovered_layers': len(all_layer_names),
                    'seed': args.seed
                }
                save_results_incrementally(results, output_file)
            
            # Create dataloaders for feature extraction
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=args.batch_size, shuffle=False
            )
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                batch_size=args.batch_size, shuffle=False
            )
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
                batch_size=args.batch_size, shuffle=False
            )
            
            # Extract features once (shared across all experiments in this group)
            logger.info("\n" + "="*80)
            logger.info("EXTRACTING SGC FEATURES (shared across global_random experiments)")
            logger.info("="*80)
            logger.info("This extraction will be reused for separation, trust_score, and sgc_faiss")
            
            shared_global_random_train_features, shared_global_random_val_features, shared_global_random_test_features, shared_global_random_extraction_info = extract_and_aggregate_sgc_features(
                model=model,
                layer_names=shared_global_random_selected_layers,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                target_dim=target_dim,
                seed=args.seed,
                pooling_mode=args.spp_pooling_mode
            )
            logger.info(f"Feature extraction completed. Final feature shape: {shared_global_random_train_features.shape}")
        
        # Run calibration for each scoring method
        for scoring_method in ['separation', 'trust_score']:
            section_key = f'global_random_{scoring_method}'
            score_label = "Trust Score" if scoring_method == 'trust_score' else "Separation"
            
            # In methods-only mode, only run if this specific method is selected
            if methods_only_mode:
                should_run = (section_key in selected_methods)
            else:
                should_run = should_rerun_section(results, section_key, GEO_REQUIRED_FIELDS)
            
            if should_run:
                logger.info("\n" + "="*80)
                logger.info(f"RUNNING GLOBAL RANDOM CALIBRATION - {score_label}")
                logger.info("="*80)
                
                global_random_result = run_global_random_calibration(
                    model, args.model_name, args.dataset, model_adapter,
                    train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
                    device,
                    num_layers=num_layers,
                    target_dim=target_dim,
                    output_dir=args.output_dir,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    pooling_mode=args.spp_pooling_mode,
                    output_file=output_file,
                    results_dict=results,
                    ood_raw=ood_raw,
                    ood_labels=ood_labels,
                    scoring_method=scoring_method,
                    normalization_method=args.normalization_method,
                    precomputed_train_features=shared_global_random_train_features,
                    precomputed_val_features=shared_global_random_val_features,
                    precomputed_test_features=shared_global_random_test_features,
                    precomputed_extraction_info=shared_global_random_extraction_info,
                    precomputed_selected_layers=shared_global_random_selected_layers,
                    # OPTIMIZATION: Pass cached predictions and loaders
                    precomputed_test_probs=prediction_cache['test_probs'],
                    precomputed_ood_probs=prediction_cache.get('ood_probs'),
                    train_loader=dataloader_cache['train'],
                    val_loader=dataloader_cache['val'],
                    test_loader=dataloader_cache['test'],
                    ood_loader=dataloader_cache.get('ood'),
                    ood_feature_cache=ood_feature_cache
                )
                results[section_key] = global_random_result
                save_results_incrementally(results, output_file)
            else:
                logger.info(f"✓ Global Random calibration ({score_label}) already computed, skipping")
        
        # For backward compatibility, set global_random_result to separation version
        global_random_result = results.get('global_random_separation', results.get('global_random', {}))

    # ===== SGC WITH FAISS BACKEND (FAIR TIMING COMPARISON) =====
    if methods_only_mode and 'sgc_faiss' not in selected_methods:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING SGC-FAISS (not in selected methods)")
        logger.info("="*80)
        if 'sgc_faiss' not in results:
            results['sgc_faiss'] = {"skipped": True, "reason": "methods-only mode: not selected"}
        save_results_incrementally(results, output_file)
    elif should_rerun_section(results, 'sgc_faiss', GEO_REQUIRED_FIELDS):
        logger.info("\n" + "="*80)
        logger.info("RUNNING SGC WITH FAISS BACKEND (FAIR TIMING COMPARISON)")
        logger.info("="*80)
        
        try:
            from Experiments.compare_sgc_backends import SGCCalibratorFast
            
            num_layers = getattr(args, 'num_layers', args.global_random_layers)
            target_dim = args.target_dimension
            
            # Reuse shared features if available (extracted for global_random)
            if shared_global_random_train_features is not None:
                logger.info("Reusing shared features from global_random experiments")
                train_sgc_features = shared_global_random_train_features
                val_sgc_features = shared_global_random_val_features
                test_sgc_features = shared_global_random_test_features
                extraction_time = shared_global_random_extraction_info.get('extraction_time_s', 0.0)
                logger.info(f"Using pre-extracted features (shape: {train_sgc_features.shape})")
            else:
                # Fallback: extract features if not already extracted
                logger.info("Extracting features for SGC-FAISS (shared features not available)")
                
                # Check if we have layer selection info from global_random
                if 'global_random_layer_selection' in results:
                    selected_layers = results['global_random_layer_selection']['selected_layers']
                    logger.info(f"Reusing layer selection from global_random: {len(selected_layers)} layers")
                else:
                    # Discover layers and select randomly (same as global_random)
                    is_dinov2 = args.model_name is not None and 'dinov2' in args.model_name.lower()
                    if is_dinov2:
                        input_shape = (1, 3, 224, 224)
                    elif args.dataset.lower() in ["tiny_imagenet", "tinyimagenet"]:
                        input_shape = (1, 3, 64, 64)
                    else:
                        input_shape = (1, 3, 32, 32)
                    discovered_layers = normalize_discovered_layers(
                        model, discover_model_layers(model, device=device, input_shape=input_shape)
                    )
                    filtered_layers, excluded_layers = filter_non_feature_layers(discovered_layers, model)
                    all_layer_names = [d["name"] for d in filtered_layers]
                    logger.info(f"Discovered {len(discovered_layers)} total layers, {len(all_layer_names)} feature-bearing layers after filtering")
                    np.random.seed(args.seed)
                    if num_layers > len(all_layer_names):
                        selected_layers = all_layer_names
                    else:
                        selected_layers = list(np.random.choice(all_layer_names, size=num_layers, replace=False))
                    logger.info(f"Selected {len(selected_layers)} layers for SGC-FAISS")
                
                # Extract features using same pipeline as global_random
                train_loader_sgc = DataLoader(
                    TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                    batch_size=args.batch_size, shuffle=False
                )
                val_loader_sgc = DataLoader(
                    TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                    batch_size=args.batch_size, shuffle=False
                )
                test_loader_sgc = DataLoader(
                    TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
                    batch_size=args.batch_size, shuffle=False
                )
                
                # Add timing around extraction
                start_extraction = time.perf_counter()
                train_sgc_features, val_sgc_features, test_sgc_features, _ = extract_and_aggregate_sgc_features(
                    model=model,
                    layer_names=selected_layers,
                    train_loader=train_loader_sgc,
                    val_loader=val_loader_sgc,
                    test_loader=test_loader_sgc,
                    device=device,
                    target_dim=target_dim,
                    seed=args.seed,
                    pooling_mode=args.spp_pooling_mode
                )
                extraction_time = time.perf_counter() - start_extraction
            
            # OPTIMIZATION: Use cached predictions
            val_logits = prediction_cache['val_probs']
            test_logits = prediction_cache['test_probs']
            val_predictions = prediction_cache['val_preds']
            test_predictions = prediction_cache['test_preds']
            
            # Run SGC-FAISS calibration
            sgc_faiss_cal = SGCCalibratorFast(use_gpu=(device.type == 'cuda'))
            
            # Fit timing
            start_fit = time.perf_counter()
            sgc_faiss_cal.fit(
                X_train_features=train_sgc_features,
                y_train=train_labels,
                X_val_features=val_sgc_features,
                y_val=val_labels,
                val_predictions=val_predictions
            )
            fit_time = time.perf_counter() - start_fit
            
            # Calibration timing
            start_cal = time.perf_counter()
            calibrated_probs = sgc_faiss_cal.calibrate(
                X_test_features=test_sgc_features,
                predictions=test_predictions,
                logits=test_logits
            )
            calibrate_time = time.perf_counter() - start_cal
            
            # Calculate metrics
            ece = calculate_ece(calibrated_probs, test_labels)
            adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
            calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
            brier = calculate_brier_score(calibrated_probs, test_labels)
            accuracy = calculate_accuracy(calibrated_probs, test_labels)
            throughput = len(test_labels) / calibrate_time
            
            # Extract isotonic regression parameters (for analysis / reconstruction)
            iso = sgc_faiss_cal.isotonic
            isotonic_params = {
                'X_thresholds': iso.X_thresholds_.tolist() if hasattr(iso, 'X_thresholds_') else None,
                'y_thresholds': iso.y_thresholds_.tolist() if hasattr(iso, 'y_thresholds_') else None,
                'X_min': float(iso.X_min_) if hasattr(iso, 'X_min_') else None,
                'X_max': float(iso.X_max_) if hasattr(iso, 'X_max_') else None,
                'y_min': float(iso.y_thresholds_[0]) if hasattr(iso, 'y_thresholds_') and len(iso.y_thresholds_) > 0 else None,
                'y_max': float(iso.y_thresholds_[-1]) if hasattr(iso, 'y_thresholds_') and len(iso.y_thresholds_) > 0 else None,
                'stability_p5': float(sgc_faiss_cal.stability_p5) if sgc_faiss_cal.stability_p5 is not None else None,
                'stability_p95': float(sgc_faiss_cal.stability_p95) if sgc_faiss_cal.stability_p95 is not None else None,
            }
            
            results['sgc_faiss'] = {
                'ece': float(ece),
                'adaptive_ece': float(adaptive_ece),
                'calibration_mce': float(calibration_mce),
                'brier': float(brier),
                'accuracy': float(accuracy),
                'extraction_time_s': extraction_time,
                'fit_time_s': fit_time,
                'calibrate_time_s': calibrate_time,
                'throughput_samples_per_sec': throughput,
                'method': 'SGC with FAISS backend (fair comparison)',
                'calibrator_params': {
                    'fitting_method': 'isotonic',
                    'isotonic_params': isotonic_params,
                },
            }
            
            logger.info(f"SGC-FAISS ECE: {ece:.6f}")
            logger.info(f"SGC-FAISS Adaptive ECE: {adaptive_ece:.6f}")
            logger.info(f"SGC-FAISS Brier: {brier:.6f}")
            logger.info(f"SGC-FAISS Accuracy: {accuracy:.4f}%")
            logger.info(f"SGC-FAISS Throughput: {throughput:.1f} samples/sec")
            logger.info(f"SGC-FAISS Fit time: {fit_time:.3f}s")
            logger.info(f"SGC-FAISS Calibrate time: {calibrate_time:.3f}s")
            
            save_results_incrementally(results, output_file)
            
        except Exception as e:
            logger.error(f"SGC-FAISS experiment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results['sgc_faiss'] = {'error': str(e)}
            save_results_incrementally(results, output_file)
    else:
        logger.info("✓ SGC-FAISS already computed, skipping")

    # ===== COORDINATE SAMPLING CALIBRATION (SEPARATION & TRUST SCORE) =====
    if methods_only_mode and 'coordinate_sampling_separation' not in selected_methods:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING COORDINATE SAMPLING (not in selected methods)")
        logger.info("="*80)
        if 'coordinate_sampling' not in results:
            results['coordinate_sampling'] = {}
        for scoring_method in ['separation', 'trust_score']:
            section_key = f'coordinate_sampling_{scoring_method}'
            results[section_key] = {"skipped": True, "reason": "methods-only mode: not selected"}
        save_results_incrementally(results, output_file)
    elif 'coordinate_sampling' not in results:
        results['coordinate_sampling'] = {}
    
    if not (methods_only_mode and 'coordinate_sampling_separation' not in selected_methods):
        for scoring_method in ['separation', 'trust_score']:
            section_key = f'coordinate_sampling_{scoring_method}'
            score_label = "Trust Score" if scoring_method == 'trust_score' else "Separation"
            
            # In methods-only mode, only run if this specific method is selected
            if methods_only_mode:
                should_run = (section_key in selected_methods)
            else:
                should_run = should_rerun_section(results, section_key, GEO_REQUIRED_FIELDS)
            
            if should_run:
                logger.info("\n" + "="*80)
                logger.info(f"RUNNING COORDINATE SAMPLING CALIBRATION - {score_label}")
                logger.info("="*80)
                
                coordinate_result = run_coordinate_calibration(
                    model, args.model_name, args.dataset, model_adapter,
                    train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
                    device,
                    num_coordinates=args.target_dimension,  # Use same K as SGC's d
                    batch_size=args.batch_size,
                    output_dir=args.output_dir,
                    seed=args.seed,
                    ood_raw=ood_raw,
                    ood_labels=ood_labels,
                    scoring_method=scoring_method,
                    normalization_method=args.normalization_method,
                    training_method=args.training_method,
                    # OPTIMIZATION: Pass cached predictions and loaders
                    precomputed_test_probs=prediction_cache['test_probs'],
                    precomputed_ood_probs=prediction_cache.get('ood_probs'),
                    train_loader=dataloader_cache['train'],
                    val_loader=dataloader_cache['val'],
                    test_loader=dataloader_cache['test'],
                    ood_loader=dataloader_cache.get('ood'),
                    ood_feature_cache=ood_feature_cache
                )
                results[section_key] = coordinate_result
                save_results_incrementally(results, output_file)
            else:
                logger.info(f"✓ Coordinate sampling ({score_label}) already computed, skipping")
    
    # For backward compatibility, set coordinate_result to separation version
    coordinate_result = results.get('coordinate_sampling_separation', results.get('coordinate_sampling', {}))

    # ===== COORDINATE SPP CALIBRATION (SEPARATION & TRUST SCORE) =====
    if methods_only_mode and 'coordinate_spp_separation' not in selected_methods:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING COORDINATE SPP (not in selected methods)")
        logger.info("="*80)
        if 'coordinate_spp' not in results:
            results['coordinate_spp'] = {}
        for scoring_method in ['separation', 'trust_score']:
            section_key = f'coordinate_spp_{scoring_method}'
            results[section_key] = {"skipped": True, "reason": "methods-only mode: not selected"}
        save_results_incrementally(results, output_file)
    elif 'coordinate_spp' not in results:
        results['coordinate_spp'] = {}
    
    if not (methods_only_mode and 'coordinate_spp_separation' not in selected_methods):
        for scoring_method in ['separation', 'trust_score']:
            section_key = f'coordinate_spp_{scoring_method}'
            score_label = "Trust Score" if scoring_method == 'trust_score' else "Separation"
            
            # In methods-only mode, only run if this specific method is selected
            if methods_only_mode:
                should_run = (section_key in selected_methods)
            else:
                should_run = should_rerun_section(results, section_key, GEO_REQUIRED_FIELDS)
            
            if should_run:
                logger.info("\n" + "="*80)
                logger.info(f"RUNNING COORDINATE SPP CALIBRATION - {score_label}")
                logger.info("="*80)
                
                coordinate_spp_result = run_coordinate_spp_calibration(
                    model, args.model_name, args.dataset, model_adapter,
                    train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
                    device,
                    num_coordinates=args.target_dimension,  # Use same K as SGC's d
                    batch_size=args.batch_size,
                    output_dir=args.output_dir,
                    seed=args.seed,
                    pooling_mode=args.spp_pooling_mode,
                    ood_raw=ood_raw,
                    ood_labels=ood_labels,
                    scoring_method=scoring_method,
                    # OPTIMIZATION: Pass cached predictions and loaders
                    precomputed_test_probs=prediction_cache['test_probs'],
                    precomputed_ood_probs=prediction_cache.get('ood_probs'),
                    train_loader=dataloader_cache['train'],
                    val_loader=dataloader_cache['val'],
                    test_loader=dataloader_cache['test'],
                    ood_loader=dataloader_cache.get('ood'),
                    ood_feature_cache=ood_feature_cache
                )
                results[section_key] = coordinate_spp_result
                save_results_incrementally(results, output_file)
            else:
                logger.info(f"✓ Coordinate SPP ({score_label}) already computed, skipping")
        
        # For backward compatibility, set coordinate_spp_result to separation version
        coordinate_spp_result = results.get('coordinate_spp_separation', results.get('coordinate_spp', {}))

    # ===== COORDINATE SAMPLING WITH FAISS BACKEND (FAIR TIMING COMPARISON) =====
    if methods_only_mode and 'coordinate_sampling_faiss' not in selected_methods:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING COORDINATE SAMPLING FAISS (not in selected methods)")
        logger.info("="*80)
        if 'coordinate_sampling_faiss' not in results:
            results['coordinate_sampling_faiss'] = {"skipped": True, "reason": "methods-only mode: not selected"}
        save_results_incrementally(results, output_file)
    elif should_rerun_section(results, 'coordinate_sampling_faiss', GEO_REQUIRED_FIELDS):
        logger.info("\n" + "="*80)
        logger.info("RUNNING COORDINATE SAMPLING WITH FAISS BACKEND (FAIR TIMING COMPARISON)")
        logger.info("="*80)
        
        try:
            from Experiments.compare_sgc_backends import SGCCalibratorFast
            
            num_coordinates = args.target_dimension
            
            # Reuse coordinate space discovery from coordinate_sampling if available
            # Otherwise, discover it fresh
            if 'coordinate_sampling' in results and 'num_coordinates' in results['coordinate_sampling']:
                logger.info("Reusing coordinate space from coordinate_sampling experiment")
                # We'll need to re-extract features, but can reuse the discovery logic
                needs_discovery = True
            else:
                needs_discovery = True
            
            if needs_discovery:
                # Discover coordinate space (same as run_coordinate_calibration)
                is_dinov2 = args.model_name is not None and 'dinov2' in args.model_name.lower()
                if is_dinov2:
                    input_shape = (1, 3, 224, 224)
                elif args.dataset.lower() in ["tiny_imagenet", "tinyimagenet"]:
                    input_shape = (1, 3, 64, 64)
                else:
                    input_shape = (1, 3, 32, 32)
                
                layer_map, total_size = discover_coordinate_space(
                    model, input_shape=input_shape, device=str(device)
                )
                
                # Validate with real batch
                train_loader_val = DataLoader(
                    TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                    batch_size=args.batch_size, shuffle=False, num_workers=0
                )
                validation_batch = next(iter(train_loader_val))
                if isinstance(validation_batch, (list, tuple)):
                    real_batch = validation_batch[0]
                else:
                    real_batch = validation_batch
                if not isinstance(real_batch, torch.Tensor):
                    real_batch = torch.from_numpy(real_batch) if isinstance(real_batch, np.ndarray) else real_batch
                real_batch = real_batch.to(device)
                
                layer_map, total_size = validate_and_correct_coordinate_space(
                    model=model,
                    layer_map=layer_map,
                    real_batch=real_batch,
                    device=device,
                )
                
                # Plan coordinate extraction
                sampling_plan, global_order = plan_coordinate_extraction(
                    total_size=total_size,
                    num_coordinates=num_coordinates,
                    layer_map=layer_map,
                    seed=args.seed,
                )
            
            # Extract features using same pipeline as coordinate_sampling
            train_loader_coord = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            val_loader_coord = DataLoader(
                TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            test_loader_coord = DataLoader(
                TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
                batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            
            # Add timing around extraction
            start_extraction = time.perf_counter()
            train_coord_features = extract_coordinate_features(
                model, train_loader_coord, sampling_plan, layer_map, global_order, device=str(device)
            )
            val_coord_features = extract_coordinate_features(
                model, val_loader_coord, sampling_plan, layer_map, global_order, device=str(device)
            )
            test_coord_features = extract_coordinate_features(
                model, test_loader_coord, sampling_plan, layer_map, global_order, device=str(device)
            )
            extraction_time = time.perf_counter() - start_extraction
            
            # L2 normalize (same as coordinate_sampling)
            train_coord_features_np = normalize(train_coord_features.numpy(), norm='l2', axis=1)
            val_coord_features_np = normalize(val_coord_features.numpy(), norm='l2', axis=1)
            test_coord_features_np = normalize(test_coord_features.numpy(), norm='l2', axis=1)
            
            # OPTIMIZATION: Use cached predictions
            val_logits = prediction_cache['val_probs']
            test_logits = prediction_cache['test_probs']
            val_predictions = prediction_cache['val_preds']
            test_predictions = prediction_cache['test_preds']
            
            # Run SGC-FAISS calibration on coordinate features
            coord_faiss_cal = SGCCalibratorFast(use_gpu=(device.type == 'cuda'))
            
            # Fit timing
            start_fit = time.perf_counter()
            coord_faiss_cal.fit(
                X_train_features=train_coord_features_np,
                y_train=train_labels,
                X_val_features=val_coord_features_np,
                y_val=val_labels,
                val_predictions=val_predictions
            )
            fit_time = time.perf_counter() - start_fit
            
            # Calibration timing
            start_cal = time.perf_counter()
            calibrated_probs = coord_faiss_cal.calibrate(
                X_test_features=test_coord_features_np,
                predictions=test_predictions,
                logits=test_logits
            )
            calibrate_time = time.perf_counter() - start_cal
            
            # Calculate metrics
            ece = calculate_ece(calibrated_probs, test_labels)
            adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
            calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
            brier = calculate_brier_score(calibrated_probs, test_labels)
            accuracy = calculate_accuracy(calibrated_probs, test_labels)
            throughput = len(test_labels) / calibrate_time
            
            # Extract isotonic regression parameters
            iso = coord_faiss_cal.isotonic
            isotonic_params = {
                'X_thresholds': iso.X_thresholds_.tolist() if hasattr(iso, 'X_thresholds_') else None,
                'y_thresholds': iso.y_thresholds_.tolist() if hasattr(iso, 'y_thresholds_') else None,
                'X_min': float(iso.X_min_) if hasattr(iso, 'X_min_') else None,
                'X_max': float(iso.X_max_) if hasattr(iso, 'X_max_') else None,
                'y_min': float(iso.y_thresholds_[0]) if hasattr(iso, 'y_thresholds_') and len(iso.y_thresholds_) > 0 else None,
                'y_max': float(iso.y_thresholds_[-1]) if hasattr(iso, 'y_thresholds_') and len(iso.y_thresholds_) > 0 else None,
                'stability_p5': float(coord_faiss_cal.stability_p5) if coord_faiss_cal.stability_p5 is not None else None,
                'stability_p95': float(coord_faiss_cal.stability_p95) if coord_faiss_cal.stability_p95 is not None else None,
            }
            
            results['coordinate_sampling_faiss'] = {
                'ece': float(ece),
                'adaptive_ece': float(adaptive_ece),
                'calibration_mce': float(calibration_mce),
                'brier': float(brier),
                'accuracy': float(accuracy),
                'extraction_time_s': extraction_time,
                'fit_time_s': fit_time,
                'calibrate_time_s': calibrate_time,
                'throughput_samples_per_sec': throughput,
                'num_coordinates': num_coordinates,
                'method': 'Coordinate Sampling with FAISS backend (fair comparison)',
                'calibrator_params': {
                    'fitting_method': 'isotonic',
                    'isotonic_params': isotonic_params,
                },
            }
            
            logger.info(f"Coordinate-FAISS ECE: {ece:.6f}")
            logger.info(f"Coordinate-FAISS Adaptive ECE: {adaptive_ece:.6f}")
            logger.info(f"Coordinate-FAISS Brier: {brier:.6f}")
            logger.info(f"Coordinate-FAISS Accuracy: {accuracy:.4f}%")
            logger.info(f"Coordinate-FAISS Throughput: {throughput:.1f} samples/sec")
            logger.info(f"Coordinate-FAISS Fit time: {fit_time:.3f}s")
            logger.info(f"Coordinate-FAISS Calibrate time: {calibrate_time:.3f}s")
            
            save_results_incrementally(results, output_file)
            
        except Exception as e:
            logger.error(f"Coordinate-FAISS experiment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results['coordinate_sampling_faiss'] = {'error': str(e)}
            save_results_incrementally(results, output_file)
    else:
        logger.info("✓ Coordinate-FAISS already computed, skipping")

    # ===== COORDINATE SPP WITH FAISS BACKEND (FAIR TIMING COMPARISON) =====
    if methods_only_mode and 'coordinate_spp_faiss' not in selected_methods:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING COORDINATE SPP FAISS (not in selected methods)")
        logger.info("="*80)
        if 'coordinate_spp_faiss' not in results:
            results['coordinate_spp_faiss'] = {"skipped": True, "reason": "methods-only mode: not selected"}
        save_results_incrementally(results, output_file)
    elif should_rerun_section(results, 'coordinate_spp_faiss', GEO_REQUIRED_FIELDS):
        logger.info("\n" + "="*80)
        logger.info("RUNNING COORDINATE SPP WITH FAISS BACKEND (FAIR TIMING COMPARISON)")
        logger.info("="*80)
        
        try:
            from Experiments.compare_sgc_backends import SGCCalibratorFast
            import math
            from utils.compression_utils import SPP_Only
            
            num_coordinates = args.target_dimension
            
            # Discover coordinate space (same as run_coordinate_spp_calibration)
            is_dinov2 = args.model_name is not None and 'dinov2' in args.model_name.lower()
            if is_dinov2:
                input_shape = (1, 3, 224, 224)
            elif args.dataset.lower() in ["tiny_imagenet", "tinyimagenet"]:
                input_shape = (1, 3, 64, 64)
            else:
                input_shape = (1, 3, 32, 32)
            
            layer_map, total_size = discover_coordinate_space(
                model, input_shape=input_shape, device=str(device)
            )
            
            # Validate with real batch
            train_loader_val = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            validation_batch = next(iter(train_loader_val))
            if isinstance(validation_batch, (list, tuple)):
                real_batch = validation_batch[0]
            else:
                real_batch = validation_batch
            if not isinstance(real_batch, torch.Tensor):
                real_batch = torch.from_numpy(real_batch) if isinstance(real_batch, np.ndarray) else real_batch
            real_batch = real_batch.to(device)
            
            layer_map, total_size = validate_and_correct_coordinate_space(
                model=model,
                layer_map=layer_map,
                real_batch=real_batch,
                device=device,
            )
            
            # Plan coordinate extraction
            sampling_plan, global_order = plan_coordinate_extraction(
                total_size=total_size,
                num_coordinates=num_coordinates,
                layer_map=layer_map,
                seed=args.seed,
            )
            
            # Extract coordinate features
            train_loader_coord = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            val_loader_coord = DataLoader(
                TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            test_loader_coord = DataLoader(
                TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
                batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            
            # Add timing around extraction (includes coordinate extraction + SPP application)
            start_extraction = time.perf_counter()
            train_coords = extract_coordinate_features(
                model, train_loader_coord, sampling_plan, layer_map, global_order, device=str(device)
            )
            val_coords = extract_coordinate_features(
                model, val_loader_coord, sampling_plan, layer_map, global_order, device=str(device)
            )
            test_coords = extract_coordinate_features(
                model, test_loader_coord, sampling_plan, layer_map, global_order, device=str(device)
            )
            
            # Reshape coordinates to spatial grid and apply SPP-only (same as run_coordinate_spp_calibration)
            grid_size = int(math.isqrt(num_coordinates))
            actual_coords = grid_size * grid_size
            
            train_coords_torch = train_coords.float() if isinstance(train_coords, torch.Tensor) else torch.from_numpy(train_coords).float()
            val_coords_torch = val_coords.float() if isinstance(val_coords, torch.Tensor) else torch.from_numpy(val_coords).float()
            test_coords_torch = test_coords.float() if isinstance(test_coords, torch.Tensor) else torch.from_numpy(test_coords).float()
            
            # Truncate or pad to grid_size^2
            if train_coords_torch.shape[1] > actual_coords:
                train_coords_torch = train_coords_torch[:, :actual_coords]
                val_coords_torch = val_coords_torch[:, :actual_coords]
                test_coords_torch = test_coords_torch[:, :actual_coords]
            elif train_coords_torch.shape[1] < actual_coords:
                pad_size = actual_coords - train_coords_torch.shape[1]
                train_coords_torch = torch.cat([train_coords_torch, torch.zeros(train_coords_torch.shape[0], pad_size)], dim=1)
                val_coords_torch = torch.cat([val_coords_torch, torch.zeros(val_coords_torch.shape[0], pad_size)], dim=1)
                test_coords_torch = torch.cat([test_coords_torch, torch.zeros(test_coords_torch.shape[0], pad_size)], dim=1)
            
            # Reshape to (N, 1, grid_size, grid_size)
            train_coords_4d = train_coords_torch.reshape(-1, 1, grid_size, grid_size)
            val_coords_4d = val_coords_torch.reshape(-1, 1, grid_size, grid_size)
            test_coords_4d = test_coords_torch.reshape(-1, 1, grid_size, grid_size)
            
            # Apply SPP-only
            spp = SPP_Only(
                pyramid_levels=[4, 2, 1],
                pooling_mode=args.spp_pooling_mode,
            ).to(device)
            
            def apply_spp_batched(coords_4d, batch_size_spp=512):
                spp_features_list = []
                for i in range(0, len(coords_4d), batch_size_spp):
                    batch = coords_4d[i:i+batch_size_spp].to(device)
                    with torch.no_grad():
                        spp_batch = spp(batch)
                    spp_features_list.append(spp_batch.cpu())
                return torch.cat(spp_features_list, dim=0)
            
            train_features_spp = apply_spp_batched(train_coords_4d, batch_size_spp=512)
            val_features_spp = apply_spp_batched(val_coords_4d, batch_size_spp=512)
            test_features_spp = apply_spp_batched(test_coords_4d, batch_size_spp=512)
            extraction_time = time.perf_counter() - start_extraction
            
            # L2 normalize
            train_features_np = normalize(train_features_spp.numpy(), norm='l2', axis=1)
            val_features_np = normalize(val_features_spp.numpy(), norm='l2', axis=1)
            test_features_np = normalize(test_features_spp.numpy(), norm='l2', axis=1)
            
            # OPTIMIZATION: Use cached predictions
            val_logits = prediction_cache['val_probs']
            test_logits = prediction_cache['test_probs']
            val_predictions = prediction_cache['val_preds']
            test_predictions = prediction_cache['test_preds']
            
            # Run SGC-FAISS calibration on Coordinate+SPP features
            coord_spp_faiss_cal = SGCCalibratorFast(use_gpu=(device.type == 'cuda'))
            
            # Fit timing
            start_fit = time.perf_counter()
            coord_spp_faiss_cal.fit(
                X_train_features=train_features_np,
                y_train=train_labels,
                X_val_features=val_features_np,
                y_val=val_labels,
                val_predictions=val_predictions
            )
            fit_time = time.perf_counter() - start_fit
            
            # Calibration timing
            start_cal = time.perf_counter()
            calibrated_probs = coord_spp_faiss_cal.calibrate(
                X_test_features=test_features_np,
                predictions=test_predictions,
                logits=test_logits
            )
            calibrate_time = time.perf_counter() - start_cal
            
            # Calculate metrics
            ece = calculate_ece(calibrated_probs, test_labels)
            adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
            calibration_mce = calculate_calibration_mce(calibrated_probs, test_labels)
            brier = calculate_brier_score(calibrated_probs, test_labels)
            accuracy = calculate_accuracy(calibrated_probs, test_labels)
            throughput = len(test_labels) / calibrate_time
            
            # Extract isotonic regression parameters
            iso = coord_spp_faiss_cal.isotonic
            isotonic_params = {
                'X_thresholds': iso.X_thresholds_.tolist() if hasattr(iso, 'X_thresholds_') else None,
                'y_thresholds': iso.y_thresholds_.tolist() if hasattr(iso, 'y_thresholds_') else None,
                'X_min': float(iso.X_min_) if hasattr(iso, 'X_min_') else None,
                'X_max': float(iso.X_max_) if hasattr(iso, 'X_max_') else None,
                'y_min': float(iso.y_thresholds_[0]) if hasattr(iso, 'y_thresholds_') and len(iso.y_thresholds_) > 0 else None,
                'y_max': float(iso.y_thresholds_[-1]) if hasattr(iso, 'y_thresholds_') and len(iso.y_thresholds_) > 0 else None,
                'stability_p5': float(coord_spp_faiss_cal.stability_p5) if coord_spp_faiss_cal.stability_p5 is not None else None,
                'stability_p95': float(coord_spp_faiss_cal.stability_p95) if coord_spp_faiss_cal.stability_p95 is not None else None,
            }
            
            results['coordinate_spp_faiss'] = {
                'ece': float(ece),
                'adaptive_ece': float(adaptive_ece),
                'calibration_mce': float(calibration_mce),
                'brier': float(brier),
                'accuracy': float(accuracy),
                'extraction_time_s': extraction_time,
                'fit_time_s': fit_time,
                'calibrate_time_s': calibrate_time,
                'throughput_samples_per_sec': throughput,
                'num_coordinates': num_coordinates,
                'method': 'Coordinate SPP with FAISS backend (fair comparison)',
                'calibrator_params': {
                    'fitting_method': 'isotonic',
                    'isotonic_params': isotonic_params,
                },
            }
            
            logger.info(f"Coordinate-SPP-FAISS ECE: {ece:.6f}")
            logger.info(f"Coordinate-SPP-FAISS Adaptive ECE: {adaptive_ece:.6f}")
            logger.info(f"Coordinate-SPP-FAISS Brier: {brier:.6f}")
            logger.info(f"Coordinate-SPP-FAISS Accuracy: {accuracy:.4f}%")
            logger.info(f"Coordinate-SPP-FAISS Throughput: {throughput:.1f} samples/sec")
            logger.info(f"Coordinate-SPP-FAISS Fit time: {fit_time:.3f}s")
            logger.info(f"Coordinate-SPP-FAISS Calibrate time: {calibrate_time:.3f}s")
            
            save_results_incrementally(results, output_file)
            
        except Exception as e:
            logger.error(f"Coordinate-SPP-FAISS experiment failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results['coordinate_spp_faiss'] = {'error': str(e)}
            save_results_incrementally(results, output_file)
    else:
        logger.info("✓ Coordinate-SPP-FAISS already computed, skipping")

    # ===== COORDINATE SPP DAC CALIBRATION =====
    if methods_only_mode and 'coordinate_spp_dac' not in selected_methods:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING COORDINATE SPP DAC (not in selected methods)")
        logger.info("="*80)
        if 'coordinate_spp_dac' not in results:
            results['coordinate_spp_dac'] = {"skipped": True, "reason": "methods-only mode: not selected"}
        save_results_incrementally(results, output_file)
        coordinate_spp_dac_result = results.get('coordinate_spp_dac', {})
    # Skip this experiment for tiny_imagenet dataset
    elif args.dataset.lower() == 'tiny_imagenet':
        logger.info("✓ Skipping Coordinate SPP DAC calibration for tiny_imagenet dataset")
        if 'coordinate_spp_dac' not in results:
            results['coordinate_spp_dac'] = {'skipped': True, 'reason': 'tiny_imagenet dataset'}
        coordinate_spp_dac_result = results.get('coordinate_spp_dac', {})
    elif should_rerun_section(results, 'coordinate_spp_dac', GEO_REQUIRED_FIELDS):
        logger.info("\n" + "="*80)
        logger.info("RUNNING COORDINATE SPP DAC CALIBRATION")
        logger.info("="*80)
        
        coordinate_spp_dac_result = run_coordinate_spp_dac_calibration(
            model, args.model_name, args.dataset,
            train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
            device,
            num_coordinates=args.target_dimension,  # Use same K as SGC's d
            batch_size=args.batch_size,
            seed=args.seed,
            pooling_mode=args.spp_pooling_mode,
            ood_raw=ood_raw,
            ood_labels=ood_labels,
            output_dir=args.output_dir,
            training_method=args.training_method,
            dac_knn_backend=args.dac_knn_backend
        )
        # Handle "both" mode results
        if args.dac_knn_backend == 'both' and isinstance(coordinate_spp_dac_result, dict) and 'coordinate_spp_dac_torch' in coordinate_spp_dac_result:
            results['coordinate_spp_dac'] = coordinate_spp_dac_result.get('coordinate_spp_dac', coordinate_spp_dac_result)
            results['coordinate_spp_dac_torch'] = coordinate_spp_dac_result.get('coordinate_spp_dac_torch')
        else:
            results['coordinate_spp_dac'] = coordinate_spp_dac_result
        save_results_incrementally(results, output_file)
    else:
        logger.info("✓ Coordinate SPP DAC already computed, skipping")
        coordinate_spp_dac_result = results.get('coordinate_spp_dac', {})

    # ===== ORIGINAL DAC (DENSITY-AWARE CALIBRATION) =====
    if methods_only_mode and 'dac_orig' not in selected_methods:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING ORIGINAL DAC (not in selected methods)")
        logger.info("="*80)
        if 'dac_orig' not in results:
            results['dac_orig'] = {"skipped": True, "reason": "methods-only mode: not selected"}
        save_results_incrementally(results, output_file)
    elif is_dinov2:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING ORIGINAL DAC (DINOv2 not supported)")
        logger.info("="*80)
        results['dac_orig'] = {
            "skipped": True,
            "reason": "DINOv2: DAC paper did not specify layers for transformer architectures",
        }
        save_results_incrementally(results, output_file)
    elif should_rerun_section(results, 'dac_orig', DAC_REQUIRED_FIELDS):
        logger.info("\n" + "="*80)
        logger.info("RUNNING ORIGINAL DAC (DENSITY-AWARE CALIBRATION)")
        logger.info("="*80)

        dac_start = time.perf_counter()
        try:
            raw_model = model
            dev = device

            # Get model name from model class (fallbacks to align with DAC layer presets)
            model_name_dac = raw_model.__class__.__name__.lower()
            if 'resnet' in model_name_dac:
                if hasattr(raw_model, 'layer1'):
                    num_blocks = [
                        len(raw_model.layer1), len(raw_model.layer2),
                        len(raw_model.layer3), len(raw_model.layer4)
                    ]
                    if num_blocks == [2, 2, 2, 2]:
                        model_name_dac = 'resnet18'
                    elif num_blocks == [3, 4, 6, 3]:
                        model_name_dac = 'resnet50'
                    elif num_blocks == [3, 4, 23, 3]:
                        model_name_dac = 'resnet101'
                    elif num_blocks == [3, 8, 36, 3]:
                        model_name_dac = 'resnet152'
                    else:
                        model_name_dac = 'resnet50'
            elif 'densenet' in model_name_dac:
                model_name_dac = 'densenet121'

            dataset_name_dac = args.dataset

            # Get intermediate layer names
            layer_names = get_dac_target_layers(model_name_dac, raw_model)
            logger.info(f"DAC target layers for {model_name_dac}: {layer_names}")

            # Create data loaders (reuse cached)
            tr_loader = dataloader_cache['train']
            va_loader = dataloader_cache['val']
            te_loader = dataloader_cache['test']

            # Extract intermediate features and logits
            # TIMING: Feature extraction (for fair comparison with RGCL/RGCC)
            logger.info("Extracting DAC features from intermediate layers...")
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dac_extract_start = time.perf_counter()
            
            train_feats_list, _, train_labels_dac = extract_dac_features(raw_model, tr_loader, layer_names, dev)
            val_feats_list, logits_val_dac, val_labels_dac = extract_dac_features(raw_model, va_loader, layer_names, dev)
            test_feats_list, logits_test_dac, _ = extract_dac_features(raw_model, te_loader, layer_names, dev)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dac_extract_elapsed = time.perf_counter() - dac_extract_start
            logger.info(f"  DAC feature extraction time: {dac_extract_elapsed:.2f}s")

            # Convert to numpy arrays
            train_feats_list_np = [feat.detach().cpu().numpy() for feat in train_feats_list]
            val_feats_list_np = [feat.detach().cpu().numpy() for feat in val_feats_list]
            test_feats_list_np = [feat.detach().cpu().numpy() for feat in test_feats_list]
            val_logits_np = logits_val_dac.detach().cpu().numpy()
            test_logits_np = logits_test_dac.detach().cpu().numpy()
            val_labels_np = val_labels_dac.detach().cpu().numpy()

            # Get k value based on dataset
            k_value = get_dac_k_value(dataset_name_dac)
            logger.info(f"Using k={k_value} for {dataset_name_dac}")

            # Handle "both" mode: run both backends
            backends_to_run = ['faiss', 'torch'] if args.dac_knn_backend == 'both' else [args.dac_knn_backend]
            dac_results = {}
            
            for backend in backends_to_run:
                logger.info(f"\n  Running DAC with {backend} backend...")
                
                # Fit DAC
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                dac_fit_start = time.perf_counter()
                
                dac = make_dac_calibrator(k=k_value, device=device, backend=backend)
                dac.fit(
                    train_features_list=train_feats_list_np,
                    val_features_list=val_feats_list_np,
                    val_logits=val_logits_np,
                    val_labels=val_labels_np
                )
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                dac_fit_elapsed = time.perf_counter() - dac_fit_start

                # Calibrate test set
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                dac_calibrate_start = time.perf_counter()
                
                probs_dac = dac.calibrate(
                    test_features_list=test_feats_list_np,
                    test_logits=test_logits_np
                )
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                dac_calibrate_elapsed = time.perf_counter() - dac_calibrate_start
                
                logger.info(f"  DAC fitting time ({backend}): {dac_fit_elapsed:.2f}s")
                logger.info(f"  DAC calibration time ({backend}): {dac_calibrate_elapsed:.2f}s")
                
                # Compute end-to-end timing (for fair comparison with RGCL/RGCC)
                dac_offline_time = dac_extract_elapsed + dac_fit_elapsed  # extraction + fit
                dac_end_to_end_time = dac_offline_time + dac_calibrate_elapsed  # extraction + fit + calibrate
                logger.info(f"  DAC offline time ({backend}): {dac_offline_time:.2f}s (extract + fit)")
                logger.info(f"  DAC end-to-end time ({backend}): {dac_end_to_end_time:.2f}s (extract + fit + calibrate)")

                # Calculate metrics
                ece = calculate_ece(probs_dac, test_labels)
                adaptive_ece = calculate_adaptive_ece(probs_dac, test_labels)
                calibration_mce = calculate_calibration_mce(probs_dac, test_labels)
                brier = calculate_brier_score(probs_dac, test_labels)
                accuracy = calculate_accuracy(probs_dac, test_labels)
                throughput = len(test_labels) / dac_calibrate_elapsed if dac_calibrate_elapsed > 0 else float('inf')

                result_dict = {
                    'ece': float(ece),
                    'adaptive_ece': float(adaptive_ece),
                    'calibration_mce': float(calibration_mce),
                    'brier': float(brier),
                    'accuracy': float(accuracy),
                    'extraction_time_s': dac_extract_elapsed,
                    'fit_time_s': dac_fit_elapsed,
                    'calibrate_time_s': dac_calibrate_elapsed,
                    'offline_time_s': dac_offline_time,  # extraction + fit
                    'end_to_end_time_s': dac_end_to_end_time,  # extraction + fit + calibrate
                    'throughput_samples_per_sec': throughput,
                    'selected_layers': layer_names,
                    'k_value': k_value,
                    # DAC_REQUIRED_FIELDS expects 'layer_weights' to exist and be non-None.
                    # Original DAC doesn't produce layer weights, so we store uniform weights
                    # over the selected layers for completeness/reproducibility.
                    'layer_weights': [1.0 / max(1, len(layer_names))] * len(layer_names),
                    'method': 'Original DAC (Density-Aware Calibration)',
                    'knn_backend': backend,
                    'knn_backend_impl': dac.__class__.__module__ + '.' + dac.__class__.__name__,
                }

                logger.info(f"  DAC Original ECE ({backend}): {ece:.6f}")
                logger.info(f"  DAC Original Adaptive ECE ({backend}): {adaptive_ece:.6f}")
                logger.info(f"  DAC Original Brier ({backend}): {brier:.6f}")
                logger.info(f"  DAC Original Accuracy ({backend}): {accuracy:.4f}%")
                logger.info(f"  DAC Original Throughput ({backend}): {throughput:.1f} samples/sec")
                
                # Store result for this backend
                if backend == 'faiss':
                    if args.dac_knn_backend == 'both':
                        dac_results['dac_orig'] = result_dict
                    else:
                        dac_results = result_dict  # Single backend mode (faiss)
                elif backend == 'torch':
                    if args.dac_knn_backend == 'both':
                        dac_results['dac_orig_torch'] = result_dict
                    else:
                        dac_results = result_dict  # Single backend mode (torch)

            dac_elapsed = time.perf_counter() - dac_start
            
            # Store results (handle both single and "both" mode)
            if args.dac_knn_backend == 'both':
                results.update(dac_results)
            else:
                results['dac_orig'] = dac_results

            save_results_incrementally(results, output_file)

        except Exception as e:
            dac_elapsed = time.perf_counter() - dac_start
            logger.error(f"Original DAC failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            results['dac_orig'] = {
                'ece': None,
                'brier': None,
                'accuracy': None,
                'error': str(e),
                'total_time_s': dac_elapsed
            }
            save_results_incrementally(results, output_file)
    else:
        logger.info("Original DAC already computed, skipping")

    # ===== DAC WITH COORDINATE FEATURES =====
    if methods_only_mode and 'dac_with_coordinate_features' not in selected_methods:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING DAC WITH COORDINATE FEATURES (not in selected methods)")
        logger.info("="*80)
        if 'dac_with_coordinate_features' not in results:
            results['dac_with_coordinate_features'] = {"skipped": True, "reason": "methods-only mode: not selected"}
        save_results_incrementally(results, output_file)
    elif should_rerun_section(results, 'dac_with_coordinate_features', DAC_REQUIRED_FIELDS):
        logger.info("\n" + "="*80)
        logger.info("RUNNING DAC WITH COORDINATE FEATURES")
        logger.info("="*80)
        
        dac_coord_result = run_dac_with_coordinate_features(
            model, args.model_name, args.dataset,
            train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
            device,
            num_coordinates=args.target_dimension,
            batch_size=args.batch_size,
            seed=args.seed,
            ood_raw=ood_raw,
            ood_labels=ood_labels,
            # OPTIMIZATION: Pass cached predictions and loaders
            precomputed_test_probs=prediction_cache['test_probs'],
            precomputed_ood_probs=prediction_cache.get('ood_probs'),
            train_loader=dataloader_cache['train'],
            val_loader=dataloader_cache['val'],
            test_loader=dataloader_cache['test'],
            ood_loader=dataloader_cache.get('ood'),
            output_dir=args.output_dir,
            training_method=args.training_method,
            dac_knn_backend=args.dac_knn_backend
        )
        # Handle "both" mode results
        if args.dac_knn_backend == 'both' and isinstance(dac_coord_result, dict) and 'dac_with_coordinate_features_torch' in dac_coord_result:
            results['dac_with_coordinate_features'] = dac_coord_result.get('dac_with_coordinate_features', dac_coord_result)
            results['dac_with_coordinate_features_torch'] = dac_coord_result.get('dac_with_coordinate_features_torch')
        else:
            results['dac_with_coordinate_features'] = dac_coord_result
        save_results_incrementally(results, output_file)
    else:
        logger.info("Skipping DAC with coordinate features (already computed)")

    # ===== PRE-COMPUTED THROUGHPUT EXPERIMENT =====
    if args.skip_single_layers:
        logger.info("\n" + "="*80)
        logger.info("SKIPPING PRE-COMPUTED THROUGHPUT EXPERIMENT (--skip-single-layers flag set)")
        logger.info("="*80)
        if 'geometric_precomputed' not in results:
            results['geometric_precomputed'] = {"skipped": True}
            save_results_incrementally(results, output_file)
    elif should_rerun_section(results, 'geometric_precomputed', GEO_REQUIRED_FIELDS):
        logger.info("\n" + "="*80)
        logger.info("RUNNING PRE-COMPUTED THROUGHPUT EXPERIMENT")
        logger.info("="*80)

        geo_results_list = [geo_results_dict[ln] for ln in all_layer_names]
        best_geo_single = min(geo_results_list, key=lambda x: x['ece'])
        best_layer_name = best_geo_single['layer_names']

        logger.info("Pre-computing model probabilities...")
        test_probs = model_adapter.predict_proba(test_raw)

        geo_precomputed = run_geometric_calibration(
            model, args.model_name, args.dataset, model_adapter,
            train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
            device, best_layer_name, f"Geometric Pre-computed ({get_semantic_layer_name(best_layer_name, args.model_name)})",
            args.batch_size, precomputed_features=precomputed_features,
            precomputed_probs=test_probs,
            seed=args.seed,
            pooling_mode=args.spp_pooling_mode,
            target_dimension=args.target_dimension,
            ood_raw=ood_raw,
            ood_labels=ood_labels,
            output_dir=args.output_dir,
            training_method=args.training_method
        )

        results['geometric_precomputed'] = geo_precomputed
        save_results_incrementally(results, output_file)
    else:
        logger.info("✓ Pre-computed throughput experiment already computed, skipping")

    # ===== COMBINED LAYERS EXPERIMENT =====
    if should_rerun_section(results, 'geometric_concatenated', GEO_REQUIRED_FIELDS):
        logger.info("\n" + "="*80)
        logger.info("COMBINED LAYERS EXPERIMENT (RUNNING LAST)")
        logger.info("="*80)

        if 'resnet' in args.model_name.lower():
            main_blocks = [name for name in all_layer_names if '.' not in name]
        elif 'densenet' in args.model_name.lower():
            main_blocks = [name for name in all_layer_names if 'dense' in name.lower() and 'trans' not in name.lower()]
            main_blocks = main_blocks[:min(5, len(main_blocks))]
        else:
            main_blocks = all_layer_names[:min(6, len(all_layer_names))]

        # Filter main_blocks to only include layers that exist in precomputed_features
        geo_combined = None
        if precomputed_features is not None:
            available_layers = set(precomputed_features['layer_names'])
            main_blocks = [name for name in main_blocks if name in available_layers]
            if not main_blocks:
                logger.warning(f"No matching layers found in precomputed features. Available: {precomputed_features['layer_names'][:10]}...")
                geo_combined = {
                    'method': 'Geometric (Combined) - FAILED',
                    'ece': float('inf'),
                    'error': 'No matching layers found in precomputed features'
                }
                results['geometric_concatenated'] = geo_combined
                save_results_incrementally(results, output_file)
            else:
                logger.info(f"Filtered main_blocks to {len(main_blocks)} layers that exist in precomputed features: {main_blocks}")

        if geo_combined is None:
            try:
                geo_combined = run_geometric_calibration(
                    model, args.model_name, args.dataset, model_adapter,
                    train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
                    device, main_blocks, f"Geometric (Combined {len(main_blocks)} Main Blocks)",
                    args.batch_size, precomputed_features=precomputed_features,
                    seed=args.seed,
                    pooling_mode=args.spp_pooling_mode,
                    target_dimension=args.target_dimension,
                    ood_raw=ood_raw,
                    ood_labels=ood_labels,
                    output_dir=args.output_dir,
                    training_method=args.training_method
                )
                results['geometric_concatenated'] = geo_combined
                save_results_incrementally(results, output_file)
            except Exception as e:
                logger.error(f"Combined layers experiment FAILED: {e}")
                geo_combined = {
                    'method': 'Geometric (Combined) - FAILED',
                    'ece': float('inf'),
                    'error': str(e)
                }
                results['geometric_concatenated'] = geo_combined
                save_results_incrementally(results, output_file)
    else:
        logger.info("✓ Combined layers experiment already computed, skipping")
        geo_combined = results['geometric_concatenated']

    # ===== ANALYSIS SECTION =====
    if not is_section_complete(results, 'analysis', ['best_overall_method']):
        logger.info("\n" + "="*80)
        logger.info("COMPUTING FINAL ANALYSIS")
        logger.info("="*80)

        # Collect all results for analysis
        # Handle missing single-layer results when --skip-single-layers is set
        if args.skip_single_layers or not geo_results_dict:
            geo_results_list = []
            best_geo_single = None
        else:
            geo_results_list = [geo_results_dict[ln] for ln in all_layer_names]
            best_geo_single = min(geo_results_list, key=lambda x: x['ece']) if geo_results_list else None

        # Add baselines with consistent format
        all_results_for_best = []
        if 'standard_baselines' in results:
            baselines = results['standard_baselines']
            if baselines.get('temperature_scaling', {}).get('ece') is not None:
                all_results_for_best.append({
                    'method': 'Temperature Scaling',
                    'ece': baselines['temperature_scaling']['ece'],
                    'accuracy': baselines['temperature_scaling'].get('acc', 0)
                })
            if baselines.get('density_aware_calibration', {}).get('ece') is not None:
                all_results_for_best.append({
                    'method': 'DAC',
                    'ece': baselines['density_aware_calibration']['ece'],
                    'accuracy': baselines['density_aware_calibration'].get('acc', 0)
                })
            if baselines.get('isotonic_toplabel', {}).get('ece') is not None:
                all_results_for_best.append({
                    'method': 'Isotonic',
                    'ece': baselines['isotonic_toplabel']['ece'],
                    'accuracy': baselines['isotonic_toplabel'].get('acc', 0)
                })
            if baselines.get('platt_scaling', {}).get('ece') is not None:
                all_results_for_best.append({
                    'method': 'Platt Scaling',
                    'ece': baselines['platt_scaling']['ece'],
                    'accuracy': baselines['platt_scaling'].get('acc', 0)
                })
            if baselines.get('beta_calibration', {}).get('ece') is not None:
                all_results_for_best.append({
                    'method': 'Beta Calibration',
                    'ece': baselines['beta_calibration']['ece'],
                    'accuracy': baselines['beta_calibration'].get('acc', 0)
                })
            # Add TS+DAC if available
            if ts_plus_dac_results and ts_plus_dac_results.get('ece') is not None:
                all_results_for_best.append(ts_plus_dac_results)
        
        if best_geo_single and best_geo_single.get('ece') is not None:
            all_results_for_best.append(best_geo_single)
        if geo_combined and geo_combined.get('ece', float('inf')) != float('inf'):
            all_results_for_best.append(geo_combined)
        if 'global_random' in results and results['global_random'].get('ece') is not None:
            all_results_for_best.append(results['global_random'])
        if 'coordinate_sampling' in results and results['coordinate_sampling'].get('ece') is not None:
            all_results_for_best.append(results['coordinate_sampling'])
        if 'coordinate_spp' in results and results['coordinate_spp'].get('ece') is not None:
            all_results_for_best.append(results['coordinate_spp'])
        if 'coordinate_spp_dac' in results and results['coordinate_spp_dac'].get('ece') is not None:
            all_results_for_best.append(results['coordinate_spp_dac'])
        if 'dac_with_coordinate_features' in results and results['dac_with_coordinate_features'].get('ece') is not None:
            all_results_for_best.append(results['dac_with_coordinate_features'])
        if geo_with_dac_weighting and geo_with_dac_weighting.get('ece') is not None:
            all_results_for_best.append(geo_with_dac_weighting)
        if metric_guided_results:
            # Filter metric_guided_results to only include those with 'ece' key
            metric_guided_with_ece = [r for r in metric_guided_results if r.get('ece') is not None]
            all_results_for_best.extend(metric_guided_with_ece)

        # Filter out any results without 'ece' key before finding minimum
        all_results_for_best = [r for r in all_results_for_best if r.get('ece') is not None]
        
        if not all_results_for_best:
            logger.error("No results with 'ece' key found for best_overall calculation")
            raise ValueError("No valid results with 'ece' metric available for comparison")
        
        best_overall = min(all_results_for_best, key=lambda x: x['ece'])
        # Filter metric_guided_results to only include those with 'ece' key
        metric_guided_with_ece = [r for r in metric_guided_results if r.get('ece') is not None] if metric_guided_results else []
        best_metric_guided = min(metric_guided_with_ece, key=lambda x: x['ece']) if metric_guided_with_ece else None

        # Calculate improvements (handle case where dac_results might not have 'ece' key)
        dac_ece_improvement = None
        if dac_results and dac_results.get('ece') is not None and uncalibrated_results.get('ece') is not None:
            dac_ece_improvement = ((uncalibrated_results['ece'] - dac_results['ece']) / uncalibrated_results['ece']) * 100
        best_overall_improvement = None
        if uncalibrated_results.get('ece') is not None:
            best_overall_improvement = ((uncalibrated_results['ece'] - best_overall['ece']) / uncalibrated_results['ece']) * 100

        # Last Layer vs SGC Random comparison
        last_layer_vs_random_comparison = None
        if 'last_layer_only_baseline' in results and 'global_random' in results:
            last_layer_ece = results['last_layer_only_baseline'].get('ece')
            sgc_random_ece = results['global_random'].get('ece')
            if last_layer_ece is not None and sgc_random_ece is not None:
                random_improves = sgc_random_ece < last_layer_ece
                improvement_percent = ((last_layer_ece - sgc_random_ece) / last_layer_ece) * 100 if last_layer_ece > 0 else 0.0
                last_layer_vs_random_comparison = {
                    'last_layer_ece': float(last_layer_ece),
                    'sgc_random_ece': float(sgc_random_ece),
                    'random_improves_over_last_layer': random_improves,
                    'improvement_percent': float(improvement_percent),
                    'conclusion': 'Multi-layer aggregation provides value' if random_improves else 'Last layer sufficient'
                }
                logger.info(f"\nLast Layer vs SGC Random Comparison:")
                logger.info(f"  Last Layer Only ECE: {last_layer_ece:.6f}")
                logger.info(f"  SGC Random (L=6) ECE: {sgc_random_ece:.6f}")
                logger.info(f"  SGC improves: {random_improves} ({improvement_percent:.2f}%)")
                logger.info(f"  Conclusion: {last_layer_vs_random_comparison['conclusion']}")
        
        # SGC vs Coordinate comparison
        sgc_vs_coordinate_comparison = None
        if 'global_random' in results and 'coordinate_sampling' in results:
            sgc_ece = results['global_random'].get('ece')
            coordinate_ece = results['coordinate_sampling'].get('ece')
            if sgc_ece is not None and coordinate_ece is not None:
                sgc_vs_coordinate_comparison = {
                    'sgc_ece': float(sgc_ece),
                    'coordinate_ece': float(coordinate_ece),
                    'sgc_wins': sgc_ece < coordinate_ece,
                    'ece_difference': float(abs(sgc_ece - coordinate_ece)),
                    'note': 'Both use K=256 dimensions. SGC uses layer structure, Coordinate uses raw activations.'
                }
        
        # Coordinate methods comparison (Geometric vs DAC)
        coordinate_methods_comparison = None
        if 'coordinate_sampling' in results and 'dac_with_coordinate_features' in results:
            geometric_ece = results['coordinate_sampling'].get('ece')
            dac_ece = results['dac_with_coordinate_features'].get('ece')
            if geometric_ece is not None and dac_ece is not None:
                coordinate_methods_comparison = {
                    'geometric_coordinate_ece': float(geometric_ece),
                    'dac_coordinate_ece': float(dac_ece),
                    'geometric_wins': geometric_ece < dac_ece,
                    'ece_difference': float(abs(geometric_ece - dac_ece)),
                    'note': 'Both use same K=256 coordinate features. Geometric uses single embedding, DAC uses depth-grouped layers + logits.'
                }

        # Calculate improvement over DAC if available
        improvement_over_dac_percent = None
        if dac_results and dac_results.get('ece') is not None:
            improvement_over_dac_percent = float(((dac_results['ece'] - best_overall['ece']) / dac_results['ece']) * 100)
        
        results['analysis'] = {
            'best_overall_method': best_overall['method'],
            'best_overall_ece': float(best_overall['ece']),
            'improvement_over_dac_percent': improvement_over_dac_percent,
            'dac_ece_improvement_over_uncalibrated_percent': float(dac_ece_improvement) if dac_ece_improvement is not None else None,
            'best_overall_ece_improvement_over_uncalibrated_percent': float(best_overall_improvement) if best_overall_improvement is not None else None,
            'best_metric_guided_method': best_metric_guided['method'] if best_metric_guided else None,
            'best_metric_guided_ece': float(best_metric_guided['ece']) if best_metric_guided else None,
            'best_metric_guided_selection_metric': best_metric_guided['selection_metric'] if best_metric_guided else None,
            'last_layer_vs_random_comparison': last_layer_vs_random_comparison,
            'sgc_vs_coordinate_comparison': sgc_vs_coordinate_comparison,
            'coordinate_methods_comparison': coordinate_methods_comparison,
            'note': 'Full ablation: features (DAC vs Geo) × algorithms (single-layer, concat, weighted-ensemble) + metric-guided selection + last-layer baseline'
        }
        save_results_incrementally(results, output_file)

        # Log summary
        logger.info(f"\n{'='*80}")
        logger.info("FINAL SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Best Overall Method: {best_overall['method']}")
        logger.info(f"Best Overall ECE: {best_overall['ece']:.6f}")
        if best_overall_improvement is not None:
            logger.info(f"Improvement over Uncalibrated: {best_overall_improvement:.2f}%")
        else:
            logger.info("Improvement over Uncalibrated: N/A (uncalibrated ECE not available)")
        improvement_over_dac = results['analysis'].get('improvement_over_dac_percent')
        if improvement_over_dac is not None:
            logger.info(f"Improvement over DAC: {improvement_over_dac:.2f}%")
        else:
            logger.info("Improvement over DAC: N/A")
    else:
        logger.info("✓ Analysis section already computed, skipping")

    # ===== CIFAR-C CORRUPTION EVALUATION =====
    if args.eval_corruptions and args.dataset.lower() in ['cifar10', 'cifar100']:
        if args.mce_only:
            logger.info("\n" + "="*80)
            logger.info("MCE-ONLY MODE: Skipping clean ECE evaluation")
            logger.info("="*80)
        
        # Get test transform from the test loader
        # We need to extract it from the test_loader's dataset
        test_transform = None
        if hasattr(test_loader, 'dataset') and hasattr(test_loader.dataset, 'transform'):
            test_transform = test_loader.dataset.transform
        else:
            # Fallback: create standard CIFAR test transform
            import torchvision.transforms as transforms
            if args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'cifar100':
                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465] if args.dataset.lower() == 'cifar10' else [0.5071, 0.4867, 0.4408],
                        std=[0.2023, 0.1994, 0.2010] if args.dataset.lower() == 'cifar10' else [0.2675, 0.2565, 0.2761]
                    )
                ])
            else:
                logger.warning(f"Cannot determine test_transform for dataset {args.dataset}, skipping corruption evaluation")
                test_transform = None
        
        if test_transform is not None:
            # Check if corruption evaluation already exists
            if 'corruption_evaluation' not in results or args.force_recompute:
                logger.info("\n" + "="*80)
                logger.info("STARTING CIFAR-C CORRUPTION EVALUATION")
                logger.info("="*80)
                
                corruption_results = evaluate_corruption_robustness(
                    results, model, model_adapter, args.dataset,
                    test_transform, args.batch_size, device,
                    train_raw=train_raw, train_labels=train_labels,
                    val_raw=val_raw, val_labels=val_labels,
                    cifar_c_dir=args.cifar_c_dir,
                    pooling_mode=args.spp_pooling_mode,
                    target_dimension=args.target_dimension,
                    seed=args.seed
                )
                
                results['corruption_evaluation'] = corruption_results
                save_results_incrementally(results, output_file)
                
                # Add mCE to analysis section
                if 'analysis' in results:
                    mce_summary = {}
                    for method, data in corruption_results.items():
                        if 'mce' in data and data['mce'] is not None:
                            mce_summary[method] = float(data['mce'])
                    results['analysis']['corruption_mce'] = mce_summary
                    save_results_incrementally(results, output_file)
            else:
                logger.info("✓ Corruption evaluation already computed, skipping")
        else:
            logger.warning("Could not determine test_transform, skipping corruption evaluation")
    elif args.eval_corruptions:
        logger.warning(f"Corruption evaluation only supported for CIFAR-10 and CIFAR-100, got {args.dataset}")

    logger.info(f"\n✓ All experiments completed! Results saved to: {output_file}")


if __name__ == "__main__":
    main()
