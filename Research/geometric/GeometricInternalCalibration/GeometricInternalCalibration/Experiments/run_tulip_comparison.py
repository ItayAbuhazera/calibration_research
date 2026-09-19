#!/usr/bin/env python3
"""
TULIP comparison experiments with two configurations:
1. Original TULIP (no compression) - baseline
2. Compressed TULIP (with target dims) - fair comparison with random ablation

Implements incremental saving to avoid data loss on crashes.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Experiments.compare_dac_geometric import (
    calculate_accuracy,
    calculate_ece,
)
from Experiments.run_post_hoc_calibration import (
    PyTorchModelAdapter,
    construct_model_path,
    get_data_loaders,
    load_trained_model,
)
from Experiments.multi_layer_ensemble import discover_model_layers
from Experiments.layer_selection import TULIPWrapper
from utils.compression_utils import SmartCompression
from Calibrators.tulip import select_tulip_layers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
from utils.logging_config import get_logger
logger = get_logger(__name__)


def make_json_serializable(obj: Any):
    """Convert numpy types to Python types for JSON serialization."""
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
    """Save results incrementally with atomic file replacement."""
    temp_path = output_path + ".tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(make_json_serializable(results), f, indent=2)
        os.replace(temp_path, output_path)
        logger.info(f"✓ Saved incremental results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save incremental results: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise


def extract_raw_from_loader(loader):
    """Extract raw numpy arrays from DataLoader."""
    xs, ys = [], []
    for batch in loader:
        data, labels = batch[:2]
        xs.append(data.numpy())
        ys.append(labels.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def normalize_discovered_layers(model, discovered):
    """Ensure each discovered layer has a descriptive name."""
    idx_to_name = {i: name for i, (name, _) in enumerate(model.named_modules())}
    normalized = []
    for entry in discovered:
        idx = entry.get("idx")
        entry = dict(entry)
        entry["name"] = idx_to_name.get(idx, entry.get("name", f"layer_{idx}"))
        normalized.append(entry)
    return normalized


def extract_single_layer_features(model_adapter, layer_name, data_raw, batch_size=128, device='cuda'):
    """
    Extract features from a single layer (memory-efficient).
    
    Args:
        model_adapter: PyTorchModelAdapter instance
        layer_name: Name of the layer to extract features from
        data_raw: Raw numpy array of input data
        batch_size: Batch size for extraction
        device: Device to run on
    
    Returns:
        numpy array of features (N, feature_dim)
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    dataset = TensorDataset(
        torch.from_numpy(data_raw),
        torch.zeros(len(data_raw), dtype=torch.long)  # Dummy labels
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features = []
    hook_handle = None
    
    def hook_fn(module, input, output):
        # Global average pooling if spatial dims exist
        if len(output.shape) == 4:
            pooled = output.mean(dim=(2, 3))  # (N, C, H, W) -> (N, C)
        else:
            pooled = output
        features.append(pooled.detach().cpu())
    
    # Find and register hook on specific layer
    model_adapter.model.eval()
    model_adapter.model.to(device)
    
    for name, module in model_adapter.model.named_modules():
        if name == layer_name:
            hook_handle = module.register_forward_hook(hook_fn)
            break
    
    if hook_handle is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Extract features
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            _ = model_adapter.model(batch_x)
    
    hook_handle.remove()
    
    # Concatenate and convert to numpy
    if features:
        return torch.cat(features).numpy()
    else:
        raise RuntimeError(f"No features extracted from layer {layer_name}")


def extract_tulip_intermediate_features(tulip, data_raw, batch_size=128):
    """
    Extract intermediate features from TULIP's internal classifiers.
    Returns: List of feature arrays, one per internal classifier.
    
    NOTE: This function is kept for backward compatibility but is memory-intensive.
    For compressed TULIP, use extract_single_layer_features instead.
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dataloader
    dataset = TensorDataset(
        torch.from_numpy(data_raw),
        torch.zeros(len(data_raw), dtype=torch.long)  # Dummy labels
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Extract features from each internal classifier
    features_by_layer = {name: [] for name in tulip.layer_names}
    
    tulip.tulip.model.eval()
    tulip.tulip.model.to(tulip.device)
    
    # Register hooks to capture intermediate activations
    hooks = []
    def get_activation(name):
        def hook(model, input, output):
            features_by_layer[name].append(output.detach().cpu())
        return hook
    
    for name, module in tulip.tulip.model.named_modules():
        if name in tulip.layer_names:
            hooks.append(module.register_forward_hook(get_activation(name)))
    
    # Run forward pass
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(tulip.device)
            _ = tulip.tulip.model(batch_x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Concatenate and convert to numpy
    features_list = []
    for name in tulip.layer_names:
        if features_by_layer[name]:
            feats = torch.cat(features_by_layer[name]).numpy()
            # Flatten spatial dimensions if needed (N, C, H, W) -> (N, C)
            if len(feats.shape) == 4:
                feats = feats.mean(axis=(2, 3))  # Global average pooling
            features_list.append(feats)
    
    return features_list


def run_tulip_original(
    model_adapter,
    tulip_wrapper,
    train_raw, train_labels,
    val_raw, val_labels,
    test_raw, test_labels,
    batch_size=128
):
    """Run original TULIP without compression."""
    
    logger.info("\n" + "="*80)
    logger.info("RUNNING ORIGINAL TULIP (NO COMPRESSION)")
    logger.info("="*80)
    
    t_start = time.perf_counter()
    
    # Train TULIP
    logger.info("Training TULIP internal classifiers...")
    t_train = time.perf_counter()
    tulip_wrapper.fit(
        train_raw=train_raw,
        train_labels=train_labels,
        val_raw=val_raw,
        val_labels=val_labels,
        batch_size=batch_size
    )
    train_time = time.perf_counter() - t_train
    
    # Calibrate test set
    logger.info("Running calibration on test set...")
    t_cal = time.perf_counter()
    calibrated_probs = tulip_wrapper.calibrate(test_raw, batch_size=batch_size)
    calibrate_time = time.perf_counter() - t_cal
    
    # Compute metrics
    ece = float(calculate_ece(calibrated_probs, test_labels))
    acc = float(calculate_accuracy(calibrated_probs, test_labels))
    
    total_time = time.perf_counter() - t_start
    samples_per_second = len(test_raw) / total_time if total_time > 0 else 0.0
    
    # Get combination weights
    combination_weights = tulip_wrapper.combination_weights.tolist() if tulip_wrapper.combination_weights is not None else None
    
    logger.info(f"Results:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  Throughput: {samples_per_second:.1f} samples/sec")
    
    result = {
        "num_layers": len(tulip_wrapper.layer_names),
        "candidate_layer_names": tulip_wrapper.layer_names,
        "ece": ece,
        "accuracy": acc,
        "train_time_s": train_time,
        "calibrate_time_s": calibrate_time,
        "total_time_s": total_time,
        "samples_per_second": samples_per_second,
        "combination_weights": combination_weights,
        "compression": "none",
    }
    
    return result


def run_tulip_compressed(
    model_adapter,
    candidate_indices,
    discovered_layers,
    num_classes,
    device,
    train_raw, train_labels,
    val_raw, val_labels,
    test_raw, test_labels,
    target_dim,
    batch_size=128,
    seed=42
):
    """
    Run TULIP with compression to target dimension for fair comparison.
    This is NOT the original TULIP - it's a modified version for comparison.
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING COMPRESSED TULIP: target_dim={target_dim}")
    logger.info(f"{'='*80}")
    logger.info("NOTE: This is a modified version with compression for fair comparison")
    
    t_start = time.perf_counter()
    
    # Step 1: Initialize TULIP
    logger.info("Initializing TULIP...")
    tulip = TULIPWrapper(
        model_adapter=model_adapter,
        candidate_layers_indices=candidate_indices,
        device=device,
        num_classes=num_classes
    )
    
    num_layers = len(tulip.layer_names)
    logger.info(f"Processing {num_layers} layers with layer-by-layer extraction and compression")
    
    # Step 2: Extract and compress layer-by-layer (memory-efficient)
    logger.info("Extracting and compressing features layer-by-layer...")
    logger.info(f"Target total dimension: {target_dim}")
    
    # Allocate dimensions per layer (ensure minimum of 32 dims per layer)
    per_layer_dim = max(target_dim // num_layers, 32)
    logger.info(f"Per-layer target dimension: {per_layer_dim} (total will be ~{per_layer_dim * num_layers})")
    
    t_compress = time.perf_counter()
    
    # Store compressors for each layer (needed for val/test)
    layer_compressors = []
    compressed_features_list = []
    total_concatenated_dim = 0
    
    # Process training data layer-by-layer
    for i, layer_name in enumerate(tulip.layer_names):
        logger.info(f"Processing layer {i+1}/{num_layers}: {layer_name}")
        
        # Extract this layer's features
        train_feats = extract_single_layer_features(
            model_adapter, layer_name, train_raw, batch_size, device
        )
        logger.info(f"  Extracted shape: {train_feats.shape}")
        total_concatenated_dim += train_feats.shape[1]
        
        # Compress this layer individually
        layer_compressor = SmartCompression(
            method="random_projection",
            target_dims=per_layer_dim,
            random_state=seed + i  # Different seed per layer for diversity
        )
        
        train_compressed = layer_compressor(train_feats, train=True)
        compressed_features_list.append(train_compressed)
        layer_compressors.append(layer_compressor)
        
        # Clear memory immediately
        del train_feats
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Concatenate compressed features
    train_features_compressed = np.concatenate(compressed_features_list, axis=1)
    actual_output_dim = train_features_compressed.shape[1]
    compression_ratio = total_concatenated_dim / actual_output_dim if actual_output_dim > 0 else 1.0
    
    logger.info(f"Train compressed shape: {train_features_compressed.shape}")
    logger.info(f"  Total concatenated dim: {total_concatenated_dim}")
    logger.info(f"  Final compressed dim: {actual_output_dim}")
    logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
    
    # Process validation data with stored compressors
    logger.info("Processing validation data...")
    val_compressed_list = []
    for i, (layer_name, compressor) in enumerate(zip(tulip.layer_names, layer_compressors)):
        val_feats = extract_single_layer_features(
            model_adapter, layer_name, val_raw, batch_size, device
        )
        val_compressed = compressor(val_feats, train=False)
        val_compressed_list.append(val_compressed)
        del val_feats
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    val_features_compressed = np.concatenate(val_compressed_list, axis=1)
    logger.info(f"Val compressed shape: {val_features_compressed.shape}")
    
    # Process test data with stored compressors
    logger.info("Processing test data...")
    test_compressed_list = []
    for i, (layer_name, compressor) in enumerate(zip(tulip.layer_names, layer_compressors)):
        test_feats = extract_single_layer_features(
            model_adapter, layer_name, test_raw, batch_size, device
        )
        test_compressed = compressor(test_feats, train=False)
        test_compressed_list.append(test_compressed)
        del test_feats
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    test_features_compressed = np.concatenate(test_compressed_list, axis=1)
    logger.info(f"Test compressed shape: {test_features_compressed.shape}")
    
    compression_time = time.perf_counter() - t_compress
    logger.info(f"Compression completed in {compression_time:.2f}s")
    
    # Step 4: Train TULIP on compressed features using new fit_from_features method
    logger.info("Training TULIP internal classifiers on compressed features...")
    t_train = time.perf_counter()
    tulip.fit_from_features(
        train_features_list=compressed_features_list,
        train_labels=train_labels,
        val_features_list=val_compressed_list,
        val_labels=val_labels,
        batch_size=batch_size
    )
    train_time = time.perf_counter() - t_train
    
    # Step 5: Get calibrated probabilities using compressed features
    logger.info("Running calibration on test set using compressed features...")
    t_cal = time.perf_counter()
    calibrated_probs = tulip.calibrate_from_features(test_compressed_list, batch_size=batch_size)
    calibrate_time = time.perf_counter() - t_cal
    
    # Clean up compressed feature lists to free memory (after training/calibration)
    del compressed_features_list, val_compressed_list, test_compressed_list
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Compute metrics
    ece = float(calculate_ece(calibrated_probs, test_labels))
    acc = float(calculate_accuracy(calibrated_probs, test_labels))
    
    total_time = time.perf_counter() - t_start
    samples_per_second = len(test_raw) / total_time if total_time > 0 else 0.0
    
    # Get combination weights
    combination_weights = tulip.combination_weights.tolist() if tulip.combination_weights is not None else None
    
    logger.info(f"Results:")
    logger.info(f"  ECE: {ece:.6f}")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  Throughput: {samples_per_second:.1f} samples/sec")
    
    result = {
        "num_layers": len(tulip.layer_names),
        "candidate_layer_names": tulip.layer_names,
        "concatenated_dim": int(total_concatenated_dim),
        "target_dim": int(target_dim),
        "actual_feature_dim": int(actual_output_dim),
        "post_concatenation_compression_ratio": float(compression_ratio),
        "ece": ece,
        "accuracy": acc,
        "compression_time_s": compression_time,
        "train_time_s": train_time,
        "calibrate_time_s": calibrate_time,
        "total_time_s": total_time,
        "samples_per_second": samples_per_second,
        "combination_weights": combination_weights,
        "compression": "random_projection",
    }
    
    return result


def run_experiments(args):
    """Main experiment runner with incremental saving."""
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"tulip_{args.training_method}_{args.dataset}_{args.model_name}_seed{args.seed}.json",
    )
    
    # Load existing results if available
    existing_results = {}
    if os.path.exists(output_path):
        logger.info(f"Found existing results file: {output_path}")
        try:
            with open(output_path, "r") as f:
                existing_results = json.load(f)
            logger.info("Loaded existing results")
        except Exception as e:
            logger.error(f"Failed to load existing results: {e}")
    
    device = torch.device(args.device)
    
    # Load data
    logger.info(f"Loading {args.dataset} dataset...")
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset, args.batch_size, seed=args.seed
    )
    train_raw, train_labels = extract_raw_from_loader(train_loader)
    val_raw, val_labels = extract_raw_from_loader(val_loader)
    test_raw, test_labels = extract_raw_from_loader(test_loader)
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model_path = construct_model_path(
        args.results_base_dir, args.training_method, args.dataset, args.model_name, args.seed
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = load_trained_model(model_path, args.model_name, num_classes, device, dataset=args.dataset)
    model_adapter = PyTorchModelAdapter(model, device, args.dataset)
    
    # Discover candidate layers
    logger.info("Discovering candidate layers...")
    discovered_layers = normalize_discovered_layers(
        model, discover_model_layers(model, device=device)
    )

    # Strategic layer selection (per paper) to avoid exhaustive ICs
    candidate_layer_names = select_tulip_layers(model, args.model_name)
    name_to_idx = {d["name"]: d["idx"] for d in discovered_layers}
    candidate_indices = [name_to_idx[n] for n in candidate_layer_names if n in name_to_idx]

    if not candidate_indices:
        logger.warning(
            "Strategic layer selection produced no matching layers; falling back to all discovered layers"
        )
        candidate_indices = [d["idx"] for d in discovered_layers]
        candidate_layer_names = [d["name"] for d in discovered_layers]

    logger.info(f"Selected {len(candidate_indices)} candidate layers for TULIP")
    for idx, name in zip(candidate_indices, candidate_layer_names):
        logger.info(f"  - {name} (idx {idx})")
    
    # Initialize results structure
    results = existing_results.get("tulip_results", {
        "experiment_config": {
            "dataset": args.dataset,
            "model_name": args.model_name,
            "training_method": args.training_method,
            "seed": args.seed,
            "num_candidate_layers": len(candidate_indices),
            "candidate_layer_names": candidate_layer_names,
            "note": "Original TULIP only. Note: TULIP has high memory requirements - DenseNet and larger models may be impractical.",
        },
        "original_tulip": None,
    })
    
    # Wrapper results structure for compatibility
    full_results = {
        "experiment_info": {
            "dataset": args.dataset,
            "model_name": args.model_name,
            "training_method": args.training_method,
            "seed": args.seed,
        },
        "tulip_results": results
    }
    
    # Merge with existing results
    if existing_results:
        existing_results.update(full_results)
        full_results = existing_results
    
    # =========================================================================
    # Original TULIP (no compression)
    # =========================================================================
    
    # Check model memory requirements
    model_size_warning = ""
    if "densenet" in args.model_name.lower():
        model_size_warning = "WARNING: DenseNet has very high memory requirements for TULIP. This may fail with OOM."
        logger.warning(model_size_warning)
    elif "resnet50" in args.model_name.lower():
        model_size_warning = "NOTE: ResNet50 may require significant memory for TULIP."
        logger.info(model_size_warning)
    
    if results.get("original_tulip") is None:
        logger.info("\n" + "="*80)
        logger.info("RUNNING ORIGINAL TULIP")
        logger.info("="*80)
        if model_size_warning:
            logger.info(model_size_warning)
        
        # Initialize TULIP
        tulip_wrapper = TULIPWrapper(
            model_adapter=model_adapter,
            candidate_layers_indices=candidate_indices,
            device=args.device,
            num_classes=num_classes
        )
        
        # Run original TULIP
        original_result = run_tulip_original(
            model_adapter=model_adapter,
            tulip_wrapper=tulip_wrapper,
            train_raw=train_raw,
            train_labels=train_labels,
            val_raw=val_raw,
            val_labels=val_labels,
            test_raw=test_raw,
            test_labels=test_labels,
            batch_size=args.batch_size
        )
        
        results["original_tulip"] = original_result
        full_results["tulip_results"] = results
        
        # Save incrementally
        save_incremental_results(output_path, full_results)
    else:
        logger.info("✓ Original TULIP results already exist, skipping")
    
    # Final save
    logger.info("\n" + "="*80)
    logger.info("TULIP EXPERIMENTS COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_path}")
    
    # Print summary
    if results.get("original_tulip"):
        orig = results["original_tulip"]
        logger.info(f"\nOriginal TULIP Results:")
        logger.info(f"  ECE: {orig['ece']:.6f}")
        logger.info(f"  Accuracy: {orig['accuracy']:.4f}")
        logger.info(f"  Throughput: {orig['samples_per_second']:.1f} samples/sec")
        logger.info(f"  Num Layers: {orig['num_layers']}")
    
    return full_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="TULIP experiments (original only). "
        "Note: TULIP has high memory requirements - recommended for ResNet18/ResNet50 only. "
        "DenseNet and larger models may be impractical."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--training-method", type=str, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--results-base-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(args)


