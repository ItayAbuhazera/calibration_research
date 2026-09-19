"""
Collect NC1 and NC4 metrics for all layers

This script computes Neural Collapse metrics (NC1 and NC4) for all intermediate 
layers of trained models. Results are saved for later analysis and filtering.

Usage:
    python collect_nc_metrics.py --dataset cifar10 --model resnet18 --training_loss baseline_cross_entropy --seed 11
"""

import torch
import numpy as np
import os
import logging
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import sys

# Add the NC metrics
# Ensure project root on path for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Calibrators.nc_metrics import NC1CollapseMetricCalculator, NC4SeparabilityCalculator

from torch.utils.data import TensorDataset, DataLoader
from Experiments.run_post_hoc_calibration import PyTorchModelAdapter, get_data_loaders, load_trained_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from utils.logging_config import get_logger
logger = get_logger(__name__)


def _safe_to_tensor(out):
    """Safely convert output to tensor, handling various output types"""
    if torch.is_tensor(out):
        return out
    elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
        return out[0]
    return None


def discover_model_layers(
    model: torch.nn.Module, 
    device: torch.device,
    input_shape: tuple = (1, 3, 32, 32),
    keep_kinds: tuple = ('Conv', 'BatchNorm', 'ReLU', 'SiLU', 'GELU', 'Bottleneck', 'BasicBlock', 'Linear', 'Transition', 'DenseBlock'),
    always_include_last: bool = True
) -> List[int]:
    """
    Discover all extractable layers in the model by doing a forward pass.
    Returns list of layer indices that produce valid tensor outputs.
    
    Uses single-pass forward with all hooks registered at once for efficiency.
    """
    model = model.to(device).eval()
    modules = list(model.modules())
    dummy = torch.randn(*input_shape, device=device)

    # Register all hooks at once
    acts = [None] * len(modules)
    hooks = []
    
    def make_hook(i):
        def _h(mod, inp, out):
            t = _safe_to_tensor(out)
            acts[i] = t.detach() if torch.is_tensor(t) else None
        return _h

    for i, m in enumerate(modules):
        hooks.append(m.register_forward_hook(make_hook(i)))

    # Single forward pass
    with torch.no_grad():
        try:
            _ = model(dummy)
        except Exception as e:
            logger.warning(f"Forward pass failed during discovery: {e}")

    # Remove all hooks
    for h in hooks:
        h.remove()

    # Build discovered layer indices
    discovered_indices = []
    for idx, (m, t) in enumerate(zip(modules, acts)):
        mtype = m.__class__.__name__
        
        # Keep "semantic" modules by type hint
        keep = any(k in mtype for k in keep_kinds)
        if keep and t is not None:
            discovered_indices.append(idx)
    
    logger.info(f"Discovered {len(discovered_indices)} valid layers")
    return discovered_indices


def extract_features_at_layer(
    model: torch.nn.Module,
    layer_idx: int,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features at a specific layer"""
    
    model.eval()
    layers = list(model.modules())
    target_layer = layers[layer_idx]
    
    features_list = []
    labels_list = []
    
    def hook_fn(module, input, output):
        # Handle different output types
        if isinstance(output, torch.Tensor):
            feat = output
        elif isinstance(output, (tuple, list)):
            feat = output[0]
        else:
            return
        
        # Flatten spatial dimensions if needed
        if len(feat.shape) > 2:
            # Conv layer: [B, C, H, W] -> [B, C]
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            feat = feat.view(feat.size(0), -1)
        
        features_list.append(feat.detach().cpu())
    
    handle = target_layer.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                _ = model(batch_x)
                labels_list.append(batch_y)
    finally:
        handle.remove()
    
    if not features_list:
        raise ValueError(f"No features extracted at layer {layer_idx}")
    
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    return features, labels


def evaluate_model_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """Evaluate model accuracy on the given dataloader"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Evaluating accuracy", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def compute_nc_metrics_for_all_layers(
    model: torch.nn.Module,
    candidate_layers: List[int],
    dataloader: DataLoader,
    device: torch.device
) -> Dict[int, Dict[str, float]]:
    """Compute NC1 and NC4 for all candidate layers"""
    
    nc1_calc = NC1CollapseMetricCalculator()
    nc4_calc = NC4SeparabilityCalculator()
    
    results = {}
    
    for layer_idx in tqdm(candidate_layers, desc="Computing NC metrics"):
        try:
            # Extract features
            features, labels = extract_features_at_layer(
                model, layer_idx, dataloader, device
            )
            
            # Compute NC1 and NC4
            nc1 = nc1_calc.compute(features, labels)
            nc4 = nc4_calc.compute(features, labels)
            
            results[layer_idx] = {
                'nc1': float(nc1),
                'nc4': float(nc4),
                'feature_dim': int(features.shape[1]) if len(features.shape) > 1 else 1
            }
            
            logger.info(f"Layer {layer_idx}: NC1={nc1:.4f}, NC4={nc4:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to compute NC metrics for layer {layer_idx}: {e}")
            results[layer_idx] = {
                'nc1': 0.0,
                'nc4': 0.0,
                'feature_dim': 0,
                'error': str(e)
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Collect NC1 and NC4 metrics for all layers")
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['cifar10', 'cifar100', 'svhn'],
                       help='Dataset name')
    parser.add_argument('--model', type=str, required=True,
                       help='Model architecture (e.g., resnet18, densenet121)')
    parser.add_argument('--training_loss', type=str, required=True,
                       help='Training loss type (e.g., baseline_cross_entropy, augmix, constellation)')
    parser.add_argument('--seed', type=int, required=True,
                       help='Random seed for the trained model')
    parser.add_argument('--models_dir', type=str, default='aaai_full_experiments/results',
                       help='Base directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='nc_metrics_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for feature extraction')
    parser.add_argument('--use_validation', action='store_true',
                       help='Use validation split instead of test')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Build model path (matching compression_experiment.py pattern)
    logger.info(f"Loading model: {args.model} trained on {args.dataset} with {args.training_loss}")
    dynamic_folder_name = f"{args.training_loss}_{args.dataset}_{args.model}_seed{args.seed}"
    model_path = (
        Path(args.models_dir) 
        / "baseline"
        / args.training_loss 
        / args.dataset 
        / args.model 
        / f"seed{args.seed}"
        / dynamic_folder_name
        / "best_model.pth"
    )
    
    if not model_path.exists():
        logger.error(f"Model checkpoint not found at {model_path}")
        sys.exit(1)
    
    # Load data to get num_classes
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Load model (matching compression_experiment.py signature)
    model = load_trained_model(
        str(model_path), 
        args.model, 
        num_classes, 
        device, 
        dataset=args.dataset
    )
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    
    dataloader = val_loader if args.use_validation else test_loader
    split_name = 'validation' if args.use_validation else 'test'
    logger.info(f"Using {split_name} split with {len(dataloader.dataset)} samples")
    
    # Check model accuracy - skip if below 60%
    logger.info("Evaluating model accuracy...")
    accuracy = evaluate_model_accuracy(model, dataloader, device)
    logger.info(f"Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if accuracy < 0.6:
        logger.error(f"Model accuracy ({accuracy:.4f}) is below 60% threshold. Skipping NC metrics computation.")
        logger.error("This model appears to be poorly trained. Please check the training configuration.")
        sys.exit(1)
    
    logger.info("Model accuracy check passed. Proceeding with NC metrics computation...")
    
    # Determine input shape based on dataset
    if args.dataset in ['cifar10', 'cifar100']:
        input_shape = (1, 3, 32, 32)
    elif args.dataset == 'svhn':
        input_shape = (1, 3, 32, 32)
    elif args.dataset == 'tiny_imagenet':
        input_shape = (1, 3, 64, 64)
    else:
        input_shape = (1, 3, 32, 32)  # Default
    
    # Discover all valid layers
    logger.info("Discovering model layers...")
    candidate_layers = discover_model_layers(
        model=model,
        device=device,
        input_shape=input_shape
    )
    
    # Compute NC metrics
    logger.info(f"Computing NC1 and NC4 for {len(candidate_layers)} layers...")
    nc_metrics = compute_nc_metrics_for_all_layers(
        model, candidate_layers, dataloader, device
    )
    
    # Save results with enhanced analysis
    output_dir = Path(args.output_dir) / args.training_loss / args.dataset / args.model / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"nc_metrics_{split_name}.json"
    
    # Compute Pareto front for saving
    valid_layers = [(idx, m['nc1'], m['nc4']) for idx, m in nc_metrics.items() if 'error' not in m]
    pareto_front = []
    for idx1, nc1_1, nc4_1 in valid_layers:
        is_dominated = False
        for idx2, nc1_2, nc4_2 in valid_layers:
            if idx1 != idx2:
                if (nc1_2 >= nc1_1 and nc4_2 >= nc4_1) and (nc1_2 > nc1_1 or nc4_2 > nc4_1):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_front.append({'layer_idx': int(idx1), 'nc1': float(nc1_1), 'nc4': float(nc4_1)})
    
    # Sort by NC4 descending
    pareto_front.sort(key=lambda x: x['nc4'], reverse=True)
    
    # Find best layers by different criteria
    nc_metrics_list = [(idx, m) for idx, m in nc_metrics.items() if 'error' not in m]
    
    best_nc4_layer = max(nc_metrics_list, key=lambda x: x[1]['nc4']) if nc_metrics_list else (None, {})
    best_nc1_layer = max(nc_metrics_list, key=lambda x: x[1]['nc1']) if nc_metrics_list else (None, {})
    
    # PSC recommended layer (best NC4 among NC1 > 0.3)
    psc_viable = [(idx, m) for idx, m in nc_metrics.items() if 'error' not in m and m['nc1'] > 0.3]
    psc_recommended = max(psc_viable, key=lambda x: x[1]['nc4']) if psc_viable else (None, {})
    
    # Strict criteria layers
    strict_layers = [
        int(idx) for idx, m in nc_metrics.items()
        if 'error' not in m and m['nc1'] > 0.3 and m['nc4'] > 0.8
    ]
    
    results = {
        'dataset': args.dataset,
        'model': args.model,
        'training_loss': args.training_loss,
        'seed': args.seed,
        'split': split_name,
        'candidate_layers': candidate_layers,
        'nc_metrics': {str(k): v for k, v in nc_metrics.items()},
        'analysis': {
            'total_layers': len(nc_metrics),
            'layers_with_nc1_above_0.3': sum(1 for m in nc_metrics.values() if 'error' not in m and m['nc1'] > 0.3),
            'layers_with_nc4_above_0.8': sum(1 for m in nc_metrics.values() if 'error' not in m and m['nc4'] > 0.8),
            'strict_criteria_layers': strict_layers,
            'psc_recommended_layer': {
                'layer_idx': int(psc_recommended[0]) if psc_recommended[0] is not None else None,
                'nc1': float(psc_recommended[1].get('nc1', 0)) if psc_recommended[1] else None,
                'nc4': float(psc_recommended[1].get('nc4', 0)) if psc_recommended[1] else None,
            },
            'best_nc4_layer': {
                'layer_idx': int(best_nc4_layer[0]) if best_nc4_layer[0] is not None else None,
                'nc1': float(best_nc4_layer[1].get('nc1', 0)) if best_nc4_layer[1] else None,
                'nc4': float(best_nc4_layer[1].get('nc4', 0)) if best_nc4_layer[1] else None,
            },
            'best_nc1_layer': {
                'layer_idx': int(best_nc1_layer[0]) if best_nc1_layer[0] is not None else None,
                'nc1': float(best_nc1_layer[1].get('nc1', 0)) if best_nc1_layer[1] else None,
                'nc4': float(best_nc1_layer[1].get('nc4', 0)) if best_nc1_layer[1] else None,
            },
            'pareto_front': pareto_front,
            'pareto_recommended': pareto_front[0] if pareto_front else None,
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary statistics
    nc1_values = [m['nc1'] for m in nc_metrics.values() if 'error' not in m]
    nc4_values = [m['nc4'] for m in nc_metrics.values() if 'error' not in m]
    
    if nc1_values and nc4_values:
        logger.info("\n" + "="*60)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*60)
        logger.info(f"NC1 (within-class variability):")
        logger.info(f"  Mean: {np.mean(nc1_values):.4f}")
        logger.info(f"  Std:  {np.std(nc1_values):.4f}")
        logger.info(f"  Min:  {np.min(nc1_values):.4f}")
        logger.info(f"  Max:  {np.max(nc1_values):.4f}")
        logger.info(f"  Layers with NC1 > 0.3: {sum(1 for v in nc1_values if v > 0.3)}/{len(nc1_values)}")
        
        logger.info(f"\nNC4 (NCC accuracy):")
        logger.info(f"  Mean: {np.mean(nc4_values):.4f}")
        logger.info(f"  Std:  {np.std(nc4_values):.4f}")
        logger.info(f"  Min:  {np.min(nc4_values):.4f}")
        logger.info(f"  Max:  {np.max(nc4_values):.4f}")
        logger.info(f"  Layers with NC4 > 0.8: {sum(1 for v in nc4_values if v > 0.8)}/{len(nc4_values)}")
        
        # PSC approach: Select layers where NC1 > 0.3, then pick highest NC4
        viable_layers = [
            idx for idx, metrics in nc_metrics.items()
            if 'error' not in metrics and metrics['nc1'] > 0.3
        ]
        
        logger.info(f"\nPSC-STYLE FILTERING (NC1 > 0.3):")
        logger.info(f"  Viable layers: {len(viable_layers)}/{len(nc_metrics)}")
        if viable_layers:
            # Find best NC4 among viable layers
            best_viable = max(viable_layers, key=lambda idx: nc_metrics[idx]['nc4'])
            logger.info(f"  Best layer: {best_viable} (NC1={nc_metrics[best_viable]['nc1']:.4f}, NC4={nc_metrics[best_viable]['nc4']:.4f})")
        
        # High-quality layers (both criteria)
        high_quality_layers = [
            idx for idx, metrics in nc_metrics.items()
            if 'error' not in metrics and metrics['nc1'] > 0.3 and metrics['nc4'] > 0.8
        ]
        
        logger.info(f"\nSTRICT CRITERIA (NC1 > 0.3 AND NC4 > 0.8):")
        logger.info(f"  Count: {len(high_quality_layers)}/{len(nc_metrics)}")
        if high_quality_layers:
            logger.info(f"  Layer indices: {high_quality_layers}")
        
        # ENHANCED ANALYSIS: When strict criteria produce 0 results
        if len(high_quality_layers) == 0:
            logger.info("\n" + "="*60)
            logger.info("RELAXED CRITERIA ANALYSIS (no layers meet strict criteria)")
            logger.info("="*60)
            
            # Top layers by NC4 (regardless of NC1)
            logger.info(f"\nTOP-5 LAYERS BY NC4 (best class separation):")
            top_nc4 = sorted(
                [(idx, m['nc1'], m['nc4']) for idx, m in nc_metrics.items() if 'error' not in m],
                key=lambda x: x[2], reverse=True
            )[:5]
            for rank, (idx, nc1, nc4) in enumerate(top_nc4, 1):
                logger.info(f"  {rank}. Layer {idx}: NC1={nc1:.4f}, NC4={nc4:.4f}")
            
            # Top layers by NC1 (most variability)
            logger.info(f"\nTOP-5 LAYERS BY NC1 (most within-class variability):")
            top_nc1 = sorted(
                [(idx, m['nc1'], m['nc4']) for idx, m in nc_metrics.items() if 'error' not in m],
                key=lambda x: x[1], reverse=True
            )[:5]
            for rank, (idx, nc1, nc4) in enumerate(top_nc1, 1):
                logger.info(f"  {rank}. Layer {idx}: NC1={nc1:.4f}, NC4={nc4:.4f}")
            
            # Relaxed NC4 threshold
            relaxed_nc4_threshold = 0.7
            relaxed_layers = [
                idx for idx, metrics in nc_metrics.items()
                if 'error' not in metrics and metrics['nc1'] > 0.3 and metrics['nc4'] > relaxed_nc4_threshold
            ]
            logger.info(f"\nRELAXED THRESHOLD (NC1 > 0.3 AND NC4 > 0.7):")
            logger.info(f"  Count: {len(relaxed_layers)}/{len(nc_metrics)}")
            if relaxed_layers:
                best_relaxed = max(relaxed_layers, key=lambda idx: nc_metrics[idx]['nc4'])
                logger.info(f"  Best layer: {best_relaxed} (NC1={nc_metrics[best_relaxed]['nc1']:.4f}, NC4={nc_metrics[best_relaxed]['nc4']:.4f})")
            
            # Even more relaxed if needed
            if len(relaxed_layers) == 0:
                relaxed_nc4_threshold = 0.6
                relaxed_layers = [
                    idx for idx, metrics in nc_metrics.items()
                    if 'error' not in metrics and metrics['nc1'] > 0.3 and metrics['nc4'] > relaxed_nc4_threshold
                ]
                logger.info(f"\nFURTHER RELAXED (NC1 > 0.3 AND NC4 > 0.6):")
                logger.info(f"  Count: {len(relaxed_layers)}/{len(nc_metrics)}")
                if relaxed_layers:
                    best_relaxed = max(relaxed_layers, key=lambda idx: nc_metrics[idx]['nc4'])
                    logger.info(f"  Best layer: {best_relaxed} (NC1={nc_metrics[best_relaxed]['nc1']:.4f}, NC4={nc_metrics[best_relaxed]['nc4']:.4f})")
            
            # Pareto-optimal layers (non-dominated solutions)
            logger.info(f"\nPARETO-OPTIMAL LAYERS (best trade-offs):")
            valid_layers = [(idx, m['nc1'], m['nc4']) for idx, m in nc_metrics.items() if 'error' not in m]
            
            # Compute Pareto front (maximize both NC1 and NC4)
            pareto_front = []
            for idx1, nc1_1, nc4_1 in valid_layers:
                is_dominated = False
                for idx2, nc1_2, nc4_2 in valid_layers:
                    if idx1 != idx2:
                        # idx2 dominates idx1 if it's better or equal in both, and strictly better in at least one
                        if (nc1_2 >= nc1_1 and nc4_2 >= nc4_1) and (nc1_2 > nc1_1 or nc4_2 > nc4_1):
                            is_dominated = True
                            break
                if not is_dominated:
                    pareto_front.append((idx1, nc1_1, nc4_1))
            
            # Sort Pareto front by NC4 (descending)
            pareto_front.sort(key=lambda x: x[2], reverse=True)
            
            logger.info(f"  Found {len(pareto_front)} Pareto-optimal layers:")
            for idx, nc1, nc4 in pareto_front[:10]:  # Show top 10
                logger.info(f"    Layer {idx}: NC1={nc1:.4f}, NC4={nc4:.4f}")
            if len(pareto_front) > 10:
                logger.info(f"    ... and {len(pareto_front) - 10} more")
            
            # Recommend best layer from Pareto front (highest NC4)
            if pareto_front:
                recommended = pareto_front[0]
                logger.info(f"\n  RECOMMENDED: Layer {recommended[0]} (NC1={recommended[1]:.4f}, NC4={recommended[2]:.4f})")
                logger.info(f"    This layer maximizes NC4 among non-dominated solutions")
        else:
            # When we have strict matches, still show Pareto analysis
            logger.info(f"\n" + "="*60)
            logger.info("PARETO-OPTIMAL ANALYSIS")
            logger.info("="*60)
            
            valid_layers = [(idx, m['nc1'], m['nc4']) for idx, m in nc_metrics.items() if 'error' not in m]
            
            # Compute Pareto front
            pareto_front = []
            for idx1, nc1_1, nc4_1 in valid_layers:
                is_dominated = False
                for idx2, nc1_2, nc4_2 in valid_layers:
                    if idx1 != idx2:
                        if (nc1_2 >= nc1_1 and nc4_2 >= nc4_1) and (nc1_2 > nc1_1 or nc4_2 > nc4_1):
                            is_dominated = True
                            break
                if not is_dominated:
                    pareto_front.append((idx1, nc1_1, nc4_1))
            
            pareto_front.sort(key=lambda x: x[2], reverse=True)
            
            logger.info(f"Found {len(pareto_front)} Pareto-optimal layers (showing top 10):")
            for idx, nc1, nc4 in pareto_front[:10]:
                meets_strict = "✓ STRICT" if nc1 > 0.3 and nc4 > 0.8 else ""
                logger.info(f"  Layer {idx}: NC1={nc1:.4f}, NC4={nc4:.4f} {meets_strict}")
            if len(pareto_front) > 10:
                logger.info(f"  ... and {len(pareto_front) - 10} more")
    
    logger.info("="*60)
    logger.info("Done!")


if __name__ == '__main__':
    main()