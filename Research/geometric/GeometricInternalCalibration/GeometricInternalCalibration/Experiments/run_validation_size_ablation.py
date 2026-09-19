"""
Validation size ablation study for RGCC vs SGC.
Tests the implicit regularization hypothesis: RGCC's lower complexity feature space
acts as implicit regularization when calibration data is limited.

This script runs experiments across:
- Validation sizes: [50, 100, 200, 500, 1000, 2000, 5000]
- Feature modes: sgc (SPP+JL), coordinate (RGCC)
- Multiple trials per size (random subsamples)
"""

import numpy as np
import torch
import logging
import argparse
import json
import os
import traceback
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import normalize

# Import functions from run_post_hoc_calibration.py
from Experiments.run_post_hoc_calibration import (
    get_data_loaders,
    load_trained_model,
    construct_model_path,
    PyTorchModelAdapter,
)

# Import functions from compare_dac_geometric.py
from Experiments.compare_dac_geometric import (
    # Feature extraction
    extract_and_aggregate_sgc_features,
    discover_coordinate_space,
    extract_coordinate_features,
    validate_and_correct_coordinate_space,
    
    # Layer selection
    get_dac_target_layers,
    normalize_discovered_layers,
    discover_model_layers,
    filter_non_feature_layers,
    
    # Metrics
    calculate_ece,
    calculate_adaptive_ece,
    calculate_brier_score,
    calculate_accuracy,
    
    # Utilities  
    save_results_incrementally,
    make_json_serializable,
)

from utils.coordinate_extraction import plan_coordinate_extraction
from Calibrators.geometric_calibrator_new import GeometricCalibrator

logger = logging.getLogger(__name__)


def select_layers(
    model: torch.nn.Module,
    model_name: str,
    mode: str,
    seed: int = 42,
    device: torch.device = None,
    dataset_name: str = "cifar10"
) -> List[str]:
    """
    Select layers based on the specified mode.
    
    Args:
        model: PyTorch model
        model_name: Model name (e.g., 'resnet18')
        mode: One of 'random_6', 'random_1', 'dac_layers', 'last_layer'
        seed: Random seed for random selection
        device: torch device
        dataset_name: Dataset name (for input shape determination)
    
    Returns:
        List of layer names
    """
    if mode == 'dac_layers':
        return get_dac_target_layers(model_name, model)
    
    # Determine input shape
    is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
    if is_dinov2:
        input_shape = (1, 3, 224, 224)
    elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
        input_shape = (1, 3, 64, 64)
    else:
        input_shape = (1, 3, 32, 32)
    
    # Discover all layers
    discovered_layers = normalize_discovered_layers(
        model, discover_model_layers(model, device=device, input_shape=input_shape)
    )
    filtered_layers, _ = filter_non_feature_layers(discovered_layers, model)
    all_layer_names = [d["name"] for d in filtered_layers]
    
    if mode == 'last_layer':
        # Return only the last layer
        return [all_layer_names[-1]]
    elif mode == 'random_1':
        np.random.seed(seed)
        return [np.random.choice(all_layer_names)]
    elif mode == 'random_6':
        np.random.seed(seed)
        num_layers = min(6, len(all_layer_names))
        return list(np.random.choice(all_layer_names, size=num_layers, replace=False))
    else:
        raise ValueError(f"Unknown layer selection mode: {mode}")


def extract_features_for_mode(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    layer_names: List[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    feature_mode: str,
    target_dim: int = 256,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Extract features based on the specified feature mode.
    
    Args:
        model: PyTorch model
        model_name: Model name
        dataset_name: Dataset name
        layer_names: List of layer names to extract from (for SGC mode)
        train_loader, val_loader, test_loader: DataLoaders
        device: torch device
        feature_mode: 'sgc' or 'coordinate'
        target_dim: Target dimension (for both modes)
        seed: Random seed
    
    Returns:
        train_features, val_features, test_features, info_dict
    """
    if feature_mode == 'sgc':
        # SGC: SPP pooling → random projection → project-then-sum → L2 norm
        train_features, val_features, test_features, info = extract_and_aggregate_sgc_features(
            model=model,
            layer_names=layer_names,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            target_dim=target_dim,
            seed=seed,
            pooling_mode='max'
        )
        
        return train_features, val_features, test_features, info
    
    elif feature_mode == 'coordinate':
        # Coordinate: Raw coordinate sampling (RGCC)
        # Determine input shape
        is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
        if is_dinov2:
            input_shape = (1, 3, 224, 224)
        elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
            input_shape = (1, 3, 64, 64)
        else:
            input_shape = (1, 3, 32, 32)
        
        logger.info(f"Coordinate mode: discovering global coordinate space")
        
        # Discover coordinate space (initial)
        layer_map, total_size = discover_coordinate_space(
            model, input_shape=input_shape, device=str(device)
        )
        
        # Validate and correct coordinate space with a real batch
        if train_loader is not None:
            try:
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
                logger.info(f"Validated coordinate space: {total_size:,} scalars across {len(layer_map)} layers")
            except Exception as e:
                logger.warning(f"Failed to validate coordinate space, using initial discovery only: {e}")
        
        # Plan extraction
        sampling_plan, global_order = plan_coordinate_extraction(
            total_size=total_size,
            num_coordinates=target_dim,
            layer_map=layer_map,
            seed=seed,
        )
        
        # Extract features
        train_features = extract_coordinate_features(
            model, train_loader, sampling_plan, layer_map, global_order, device=str(device)
        )
        val_features = extract_coordinate_features(
            model, val_loader, sampling_plan, layer_map, global_order, device=str(device)
        )
        test_features = extract_coordinate_features(
            model, test_loader, sampling_plan, layer_map, global_order, device=str(device)
        )
        
        # L2 normalize
        train_features = normalize(train_features.numpy(), norm='l2', axis=1)
        val_features = normalize(val_features.numpy(), norm='l2', axis=1)
        test_features = normalize(test_features.numpy(), norm='l2', axis=1)
        
        info = {
            'feature_mode': 'coordinate',
            'num_coordinates': target_dim,
            'layer_map_size': total_size
        }
        
        return train_features, val_features, test_features, info
    
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")


def subsample_validation(
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_raw: np.ndarray,
    val_size: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subsample validation set to specified size.
    
    Args:
        val_features: Full validation features
        val_labels: Full validation labels
        val_raw: Full validation raw data
        val_size: Target size for subsample
        seed: Random seed for reproducibility
    
    Returns:
        Subsampled val_features, val_labels, val_raw
    """
    np.random.seed(seed)
    n_val = len(val_labels)
    if val_size >= n_val:
        return val_features, val_labels, val_raw
    
    indices = np.random.choice(n_val, val_size, replace=False)
    return val_features[indices], val_labels[indices], val_raw[indices]


def run_validation_size_ablation(
    model, model_name, dataset_name, model_adapter,
    train_raw, train_labels,
    val_raw, val_labels,  # Full validation set
    test_raw, test_labels,
    device,
    val_sizes=[50, 100, 200, 500, 1000, 2000, 5000],
    feature_modes=['sgc', 'coordinate'],
    target_dim=256,
    layer_mode='random_6',
    batch_size=128,
    seed=42,
    num_trials=3,
    output_dir='results/validation_size_ablation',
) -> Dict[str, Any]:
    """
    Run validation size ablation experiment.
    
    Tests the hypothesis that RGCC (coordinate) performs better than SGC
    when calibration data is limited due to implicit regularization.
    
    Args:
        model: PyTorch model
        model_name: Model name
        dataset_name: Dataset name
        model_adapter: Model adapter for predictions
        train_raw, train_labels: Training data
        val_raw, val_labels: Full validation set
        test_raw, test_labels: Test data
        device: torch device
        val_sizes: List of validation sizes to test
        feature_modes: List of feature modes ('sgc', 'coordinate')
        target_dim: Target dimension K (same for both methods)
        layer_mode: Layer selection mode ('random_6', 'dac_layers', etc.)
        batch_size: Batch size
        seed: Random seed
        num_trials: Number of random subsamples per validation size
        output_dir: Output directory for results
    
    Returns:
        Results dictionary organized by (feature_mode, val_size)
    """
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SIZE ABLATION STUDY")
    logger.info("="*80)
    logger.info(f"Model: {model_name}, Dataset: {dataset_name}")
    logger.info(f"Validation sizes: {val_sizes}")
    logger.info(f"Feature modes: {feature_modes}")
    logger.info(f"Target dimension: {target_dim}")
    logger.info(f"Layer mode: {layer_mode}")
    logger.info(f"Number of trials per size: {num_trials}")
    
    results = {
        "experiment_config": {
            "val_sizes": val_sizes,
            "feature_modes": feature_modes,
            "target_dim": target_dim,
            "layer_mode": layer_mode,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "seed": seed,
            "num_trials": num_trials
        },
        "results_by_val_size": {},
        "summary": {}
    }
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_raw), torch.from_numpy(test_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Get model predictions and logits
    logger.info("Computing model predictions...")
    val_logits = model_adapter.predict_proba(val_raw)
    test_logits = model_adapter.predict_proba(test_raw)
    val_predictions = np.argmax(val_logits, axis=1)
    test_predictions = np.argmax(test_logits, axis=1)
    
    # Extract features ONCE for each feature mode (using full train/val/test)
    features_cache = {}
    
    for feature_mode in feature_modes:
        logger.info("\n" + "-"*80)
        logger.info(f"Extracting features for mode: {feature_mode}")
        logger.info("-"*80)
        
        # Select layers only for SGC feature mode
        if feature_mode == 'sgc':
            try:
                layer_names = select_layers(
                    model, model_name, layer_mode, seed=seed, device=device, dataset_name=dataset_name
                )
                logger.info(f"Selected {len(layer_names)} layers: {layer_names}")
            except Exception as e:
                logger.error(f"Failed to select layers for {feature_mode}: {e}")
                continue
        else:
            layer_names = []
            logger.info("Coordinate feature mode: using global coordinate sampling")
        
        # Extract features
        try:
            train_features, val_features, test_features, extraction_info = extract_features_for_mode(
                model=model,
                model_name=model_name,
                dataset_name=dataset_name,
                layer_names=layer_names,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                feature_mode=feature_mode,
                target_dim=target_dim,
                seed=seed
            )
            logger.info(f"Extracted features: train={train_features.shape}, val={val_features.shape}, test={test_features.shape}")
            
            features_cache[feature_mode] = {
                'train_features': train_features,
                'val_features': val_features,
                'test_features': test_features,
                'extraction_info': extraction_info
            }
        except Exception as e:
            logger.error(f"Failed to extract features for {feature_mode}: {e}")
            traceback.print_exc()
            continue
    
    # Run experiments for each validation size
    for val_size in val_sizes:
        logger.info("\n" + "="*80)
        logger.info(f"Validation size: {val_size}")
        logger.info("="*80)
        
        # Skip if validation size exceeds available data
        if val_size > len(val_labels):
            logger.warning(f"Skipping val_size={val_size} (exceeds available {len(val_labels)} samples)")
            continue
        
        results["results_by_val_size"][val_size] = {}
        
        for feature_mode in feature_modes:
            if feature_mode not in features_cache:
                continue
            
            logger.info(f"\n  Feature mode: {feature_mode}")
            
            train_features = features_cache[feature_mode]['train_features']
            val_features_full = features_cache[feature_mode]['val_features']
            test_features = features_cache[feature_mode]['test_features']
            
            # Initialize results for this feature mode
            results["results_by_val_size"][val_size][feature_mode] = {
                'trials': []
            }
            
            # Run multiple trials (random subsamples)
            trial_eces = []
            trial_accuracies = []
            trial_fit_times = []
            
            for trial in range(num_trials):
                logger.info(f"    Trial {trial + 1}/{num_trials}...")
                
                try:
                    # Subsample validation set
                    trial_seed = seed + trial * 1000 + val_size  # Unique seed per trial/size
                    val_features_sub, val_labels_sub, val_raw_sub = subsample_validation(
                        val_features_full, val_labels, val_raw, val_size, trial_seed
                    )
                    
                    # Get predictions for subsampled validation set
                    val_logits_sub = model_adapter.predict_proba(val_raw_sub)
                    val_predictions_sub = np.argmax(val_logits_sub, axis=1)
                    
                    # Fit GeometricCalibrator with subsampled validation
                    start_fit = time.perf_counter()
                    
                    geo_cal = GeometricCalibrator(
                        model=model_adapter,
                        X_train_embed=train_features,
                        y_train=train_labels,
                        library="fast_separation",
                        auto_select_layer=False,
                        device=str(device),
                        scoring_method='separation'  # Use separation score (SGC-style)
                    )
                    
                    geo_cal.fit(
                        X_val_embed=val_features_sub,
                        y_val=val_labels_sub,
                        X_val_original=val_raw_sub,
                        fit_batch_size=batch_size
                    )
                    
                    fit_time = time.perf_counter() - start_fit
                    
                    # Calibrate test set
                    calibrated_probs = geo_cal.calibrate_batched(
                        X_test_embed=test_features,
                        X_test_original=test_raw,
                        batch_size=batch_size
                    )
                    
                    # Compute metrics
                    ece = calculate_ece(calibrated_probs, test_labels)
                    accuracy = calculate_accuracy(calibrated_probs, test_labels)
                    
                    trial_eces.append(ece)
                    trial_accuracies.append(accuracy)
                    trial_fit_times.append(fit_time)
                    
                    logger.info(f"      ECE: {ece:.4f}, Accuracy: {accuracy:.4f}, Fit time: {fit_time:.2f}s")
                    
                    # Store trial results
                    results["results_by_val_size"][val_size][feature_mode]['trials'].append({
                        'ece': float(ece),
                        'accuracy': float(accuracy),
                        'fit_time': float(fit_time)
                    })
                    
                except Exception as e:
                    logger.error(f"      Trial {trial + 1} failed: {e}")
                    traceback.print_exc()
                    continue
            
            # Compute statistics across trials
            if trial_eces:
                results["results_by_val_size"][val_size][feature_mode]['ece_mean'] = float(np.mean(trial_eces))
                results["results_by_val_size"][val_size][feature_mode]['ece_std'] = float(np.std(trial_eces))
                results["results_by_val_size"][val_size][feature_mode]['accuracy_mean'] = float(np.mean(trial_accuracies))
                results["results_by_val_size"][val_size][feature_mode]['accuracy_std'] = float(np.std(trial_accuracies))
                results["results_by_val_size"][val_size][feature_mode]['fit_time_mean'] = float(np.mean(trial_fit_times))
                results["results_by_val_size"][val_size][feature_mode]['fit_time_std'] = float(np.std(trial_fit_times))
                
                logger.info(f"    Mean ECE: {np.mean(trial_eces):.4f} ± {np.std(trial_eces):.4f}")
                logger.info(f"    Mean Accuracy: {np.mean(trial_accuracies):.4f} ± {np.std(trial_accuracies):.4f}")
        
        # Save checkpoint after each validation size
        output_file = os.path.join(output_dir, f"{model_name}_{dataset_name}_seed{seed}_validation_size_ablation.json")
        save_results_incrementally(results, output_file)
        logger.info(f"Checkpoint saved to {output_file}")
    
    # Compute summary statistics
    logger.info("\n" + "="*80)
    logger.info("COMPUTING SUMMARY STATISTICS")
    logger.info("="*80)
    
    # Find crossover point (where SGC catches up to coordinate)
    crossover_point = None
    hypothesis_supported = False
    
    if 'coordinate' in feature_modes and 'sgc' in feature_modes:
        # Compare ECE at different validation sizes
        coordinate_better_at_small = []
        
        for val_size in sorted(val_sizes):
            if val_size not in results["results_by_val_size"]:
                continue
            
            coord_ece = results["results_by_val_size"][val_size].get('coordinate', {}).get('ece_mean')
            sgc_ece = results["results_by_val_size"][val_size].get('sgc', {}).get('ece_mean')
            
            if coord_ece is not None and sgc_ece is not None:
                coord_better = coord_ece < sgc_ece
                coordinate_better_at_small.append((val_size, coord_better))
                
                # Find first point where SGC becomes better or equal
                if not coord_better and crossover_point is None:
                    crossover_point = val_size
        
        # Hypothesis supported if coordinate is better at small sizes
        if coordinate_better_at_small:
            small_sizes = [v for v, better in coordinate_better_at_small[:3] if better]
            hypothesis_supported = len(small_sizes) >= 2
    
    results["summary"] = {
        "hypothesis_supported": hypothesis_supported,
        "crossover_point": crossover_point,
        "interpretation": (
            "RGCC (coordinate) shows better calibration at small validation sizes, "
            "supporting implicit regularization hypothesis"
            if hypothesis_supported
            else "No clear evidence of implicit regularization advantage"
        )
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validation size ablation study for RGCC vs SGC")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., cifar10)')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g., resnet18)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--results_base_dir', type=str, required=True,
                        help='Base directory with pre-trained models')
    parser.add_argument('--training_method', type=str, default='baseline_cross_entropy',
                        help='Training method (e.g., baseline_cross_entropy, focal)')
    parser.add_argument('--val_sizes', type=int, nargs='+', 
                        default=[50, 100, 200, 500, 1000, 2000, 5000],
                        help='Validation sizes to test')
    parser.add_argument('--num_trials', type=int, default=3,
                        help='Number of random subsamples per validation size')
    parser.add_argument('--target_dim', type=int, default=256,
                        help='Target dimension K (same for both methods)')
    parser.add_argument('--layer_mode', type=str, default='random_6',
                        help='Layer selection mode (random_6, dac_layers, etc.)')
    parser.add_argument('--output_dir', type=str, default='results/validation_size_ablation',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading data for {args.dataset}...")
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset, batch_size=args.batch_size, seed=args.seed
    )
    
    # Extract raw data from loaders
    def extract_from_loader(loader):
        raw_list, label_list = [], []
        for batch in loader:
            data, labels = batch[0], batch[1]
            raw_list.append(data.numpy())
            label_list.append(labels.numpy())
        return np.concatenate(raw_list), np.concatenate(label_list)
    
    train_raw, train_labels = extract_from_loader(train_loader)
    val_raw, val_labels = extract_from_loader(val_loader)
    test_raw, test_labels = extract_from_loader(test_loader)
    
    logger.info(f"Data loaded: train={train_raw.shape}, val={val_raw.shape}, test={test_raw.shape}")
    
    # Load model
    logger.info(f"Loading model {args.model_name}...")
    model_path = construct_model_path(
        args.results_base_dir,
        args.training_method,
        args.dataset,
        args.model_name,
        args.seed
    )
    model = load_trained_model(model_path, args.model_name, num_classes, device, dataset=args.dataset)
    model_adapter = PyTorchModelAdapter(model, device, args.dataset)
    
    # Run experiment
    results = run_validation_size_ablation(
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
        val_sizes=args.val_sizes,
        feature_modes=['sgc', 'coordinate'],
        target_dim=args.target_dim,
        layer_mode=args.layer_mode,
        batch_size=args.batch_size,
        seed=args.seed,
        num_trials=args.num_trials,
        output_dir=args.output_dir
    )
    
    # Save final results
    output_file = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset}_seed{args.seed}_validation_size_ablation.json")
    cleaned_results = make_json_serializable(results)
    with open(output_file, 'w') as f:
        json.dump(cleaned_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Hypothesis supported: {results['summary']['hypothesis_supported']}")
    logger.info(f"Crossover point: {results['summary']['crossover_point']}")
    logger.info(f"Interpretation: {results['summary']['interpretation']}")


if __name__ == '__main__':
    main()

