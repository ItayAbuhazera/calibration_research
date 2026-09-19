"""
Feature Extraction Script for SGC Timing Comparison

Extracts and saves features for use with sgc_timing_comparison.py
This keeps timing clean and lets you reuse features across multiple experiments.
"""

import numpy as np
import torch
import argparse
import json
from pathlib import Path
import sys
import logging
import time

# Add parent directory for project imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_and_save_features(
    dataset: str,
    model_name: str,
    seed: int,
    num_layers: int = 6,
    target_dim: int = 256,
    output_dir: str = 'Data/extracted_features',
    batch_size: int = 128,
    pooling_mode: str = 'max',
    device: str = 'cuda',
    results_dir: str = 'checkpoints',
    training_method: str = 'standard',
):
    """
    Extract SGC features and save to .npy files for timing experiments.
    
    Saves:
        - {output_dir}/{dataset}_{model_name}_seed{seed}/
            - train_features.npy
            - train_labels.npy
            - val_features.npy
            - val_labels.npy
            - val_logits.npy
            - test_features.npy
            - test_labels.npy
            - test_logits.npy
            - extraction_info.json
    """
    output_path = Path(output_dir) / f'{dataset}_{model_name}_seed{seed}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    if (output_path / 'test_features.npy').exists():
        logger.info(f"Features already extracted at {output_path}, skipping...")
        return output_path
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Import project modules
    try:
        from Experiments.run_post_hoc_calibration import (
            PyTorchModelAdapter,
            get_data_loaders,
            load_trained_model,
            construct_model_path
        )
        from Experiments.compare_dac_geometric import (
            extract_and_aggregate_sgc_features,
            normalize_discovered_layers,
            filter_non_feature_layers
        )
        from Experiments.multi_layer_ensemble import discover_model_layers
    except ImportError as e:
        logger.error(f"Failed to import project modules: {e}")
        logger.error("Make sure you're running from the project root directory.")
        raise
    
    # Determine number of classes
    num_classes_map = {'cifar10': 10, 'cifar100': 100, 'tiny_imagenet': 200, 'tinyimagenet': 200}
    num_classes = num_classes_map.get(dataset.lower(), 100)
    
    # Construct model path
    model_path = construct_model_path(
        base_dir=results_dir,
        method=training_method,
        dataset=dataset,
        model=model_name,
        seed=seed
    )
    
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load model
    model = load_trained_model(model_path, model_name, num_classes, device, dataset)
    model.eval()
    
    # Load data
    logger.info(f"Loading dataset {dataset}...")
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        dataset=dataset,
        batch_size=batch_size,
        seed=seed,
    )
    
    # Extract raw data
    logger.info("Extracting raw data...")
    def extract_raw(loader):
        xs, ys = [], []
        for batch in loader:
            data, labels = batch[:2]
            xs.append(data.numpy())
            ys.append(labels.numpy())
        return np.concatenate(xs), np.concatenate(ys)
    
    train_raw, train_labels = extract_raw(train_loader)
    val_raw, val_labels = extract_raw(val_loader)
    test_raw, test_labels = extract_raw(test_loader)
    
    logger.info(f"Data shapes: train={train_raw.shape}, val={val_raw.shape}, test={test_raw.shape}")
    
    # Discover and select layers
    logger.info("Discovering model layers...")
    is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
    if is_dinov2:
        input_shape = (1, 3, 224, 224)
    elif dataset.lower() in ["tiny_imagenet", "tinyimagenet"]:
        input_shape = (1, 3, 64, 64)
    else:
        input_shape = (1, 3, 32, 32)
    
    discovered_layers = normalize_discovered_layers(
        model, discover_model_layers(model, device=device, input_shape=input_shape)
    )
    filtered_layers, _ = filter_non_feature_layers(discovered_layers, model)
    all_layer_names = [d["name"] for d in filtered_layers]
    
    logger.info(f"Discovered {len(all_layer_names)} feature-bearing layers")
    
    # Random layer selection
    np.random.seed(seed)
    if num_layers > len(all_layer_names):
        selected_layers = all_layer_names
    else:
        selected_layers = list(np.random.choice(all_layer_names, size=num_layers, replace=False))
    
    logger.info(f"Selected {len(selected_layers)} layers: {selected_layers}")
    
    # Create data loaders for feature extraction
    from torch.utils.data import DataLoader, TensorDataset
    
    train_loader_feat = DataLoader(
        TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
        batch_size=batch_size, shuffle=False
    )
    val_loader_feat = DataLoader(
        TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
        batch_size=batch_size, shuffle=False
    )
    test_loader_feat = DataLoader(
        TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )
    
    # Extract features
    logger.info("Extracting features...")
    start_extraction = time.perf_counter()
    
    train_features, val_features, test_features, extraction_info = extract_and_aggregate_sgc_features(
        model=model,
        layer_names=selected_layers,
        train_loader=train_loader_feat,
        val_loader=val_loader_feat,
        test_loader=test_loader_feat,
        device=device,
        target_dim=target_dim,
        seed=seed,
        pooling_mode=pooling_mode
    )
    
    extraction_time = time.perf_counter() - start_extraction
    logger.info(f"Feature extraction took {extraction_time:.2f}s")
    logger.info(f"Feature shapes: train={train_features.shape}, val={val_features.shape}, test={test_features.shape}")
    
    # Get model predictions (logits)
    logger.info("Getting model predictions...")
    model_adapter = PyTorchModelAdapter(model, device, dataset)
    
    with torch.no_grad():
        # Validation logits
        val_logits = model_adapter.predict_logits(val_raw, batch_size=batch_size)
        
        # Test logits
        test_logits = model_adapter.predict_logits(test_raw, batch_size=batch_size)
    
    # Save everything
    logger.info(f"Saving to {output_path}...")
    np.save(output_path / 'train_features.npy', train_features)
    np.save(output_path / 'train_labels.npy', train_labels)
    np.save(output_path / 'val_features.npy', val_features)
    np.save(output_path / 'val_labels.npy', val_labels)
    np.save(output_path / 'val_logits.npy', val_logits)
    np.save(output_path / 'test_features.npy', test_features)
    np.save(output_path / 'test_labels.npy', test_labels)
    np.save(output_path / 'test_logits.npy', test_logits)
    
    # Save metadata
    metadata = {
        'dataset': dataset,
        'model_name': model_name,
        'seed': seed,
        'num_layers': num_layers,
        'target_dim': target_dim,
        'selected_layers': selected_layers,
        'extraction_time_s': extraction_time,
        'train_shape': list(train_features.shape),
        'val_shape': list(val_features.shape),
        'test_shape': list(test_features.shape),
        'pooling_mode': pooling_mode,
        'model_path': str(model_path),
        'training_method': training_method,
        'extraction_info': {k: v for k, v in extraction_info.items() if not isinstance(v, np.ndarray)},
    }
    with open(output_path / 'extraction_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Done!")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Extract SGC features for timing comparison')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset name')
    parser.add_argument('--model', type=str, nargs='+', default=['resnet50'], help='Model name(s) - can specify multiple')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456], help='Random seeds')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers to sample')
    parser.add_argument('--target_dim', type=int, default=256, help='Target dimension after projection')
    parser.add_argument('--output_dir', type=str, default='Data/extracted_features', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--pooling_mode', type=str, default='max', help='Pooling mode (max or avg)')
    parser.add_argument('--results_dir', type=str, default='checkpoints', help='Directory containing trained models')
    parser.add_argument('--training_method', type=str, default='standard', help='Training method name for model path')
    args = parser.parse_args()
    
    # Ensure model is a list
    if isinstance(args.model, str):
        models = [args.model]
    else:
        models = args.model
    
    total_combinations = len(models) * len(args.seeds)
    current = 0
    
    for model_name in models:
        for seed in args.seeds:
            current += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Extracting features [{current}/{total_combinations}]")
            logger.info(f"Model: {model_name}, Seed: {seed}")
            logger.info(f"{'='*60}")
            
            try:
                output_path = extract_and_save_features(
                    dataset=args.dataset,
                    model_name=model_name,
                    seed=seed,
                    num_layers=args.num_layers,
                    target_dim=args.target_dim,
                    output_dir=args.output_dir,
                    batch_size=args.batch_size,
                    pooling_mode=args.pooling_mode,
                    results_dir=args.results_dir,
                    training_method=args.training_method,
                )
                
                logger.info(f"✓ Features saved to: {output_path}")
            except Exception as e:
                logger.error(f"✗ Failed to extract features for {model_name} seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                continue


if __name__ == '__main__':
    main()
