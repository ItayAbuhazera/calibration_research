"""
Geometric Quality Analysis Script

Analyzes which layers are suitable for geometric calibration by computing
meta-predictors that estimate calibration success without running full experiments.

Usage:
    # Analyze all layers for a specific checkpoint
    python Experiments/analyze_geometric_quality.py \
        --checkpoint path/to/checkpoint.pth \
        --dataset cifar10 \
        --output-dir analysis/geometric_quality

    # Compare multiple checkpoints
    python Experiments/analyze_geometric_quality.py \
        --checkpoint-dir aaai_full_experiments/results/baseline/cifar10 \
        --compare-models resnet18 resnet50 densenet121
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Calibrators.geometric_quality_predictors import GeometricQualityAnalyzer
from Calibrators.metrics import CrossModalLayerMapper
from Experiments.run_post_hoc_calibration import PyTorchModelAdapter, get_data_loaders, load_trained_model

from utils.logging_config import get_logger
logger = get_logger(__name__)


def extract_layer_features(
    model: nn.Module,
    dataloader: DataLoader,
    layer_idx: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from a specific layer.

    Args:
        model: PyTorch model
        dataloader: DataLoader with input data
        layer_idx: Index of layer to extract from
        device: Device to run on

    Returns:
        features: Extracted features (N, D)
        labels: Corresponding labels (N,)
    """
    model.eval()
    layers = list(model.modules())
    target_layer = layers[layer_idx]

    features_list = []
    labels_list = []

    # Register hook to capture activations
    activations = {}

    def hook_fn(module, input, output):
        activations['output'] = output

    hook = target_layer.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)

                # Forward pass
                _ = model(batch_x)

                # Get activations
                act = activations.get('output')
                if act is None:
                    raise ValueError(f"No activations captured for layer {layer_idx}")

                # Convert to numpy
                if isinstance(act, torch.Tensor):
                    # Flatten spatial dimensions if needed
                    if len(act.shape) > 2:
                        act = act.view(act.size(0), -1)
                    feat = act.cpu().numpy()
                else:
                    raise TypeError(f"Unexpected activation type: {type(act)}")

                features_list.append(feat)
                labels_list.append(batch_y.numpy())

                activations.clear()

    finally:
        hook.remove()

    # Concatenate all batches
    features = np.concatenate(features_list, axis=0).astype(np.float32)
    labels = np.concatenate(labels_list, axis=0).astype(np.int64)

    return features, labels


def analyze_layer(
    layer_idx: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    model_adapter: PyTorchModelAdapter,
    val_original: np.ndarray
) -> Dict:
    """
    Analyze a single layer using GeometricQualityAnalyzer.

    Returns:
        Dictionary with layer analysis results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing Layer {layer_idx}")
    logger.info(f"{'='*80}")

    # Extract features
    logger.info("Extracting features from training set...")
    train_features, train_labels = extract_layer_features(model, train_loader, layer_idx, device)

    logger.info("Extracting features from validation set...")
    val_features, val_labels = extract_layer_features(model, val_loader, layer_idx, device)

    # Get model predictions on validation set
    logger.info("Getting model predictions...")
    val_probs = model_adapter.predict_proba(val_original, batch_size=256)
    val_predictions = np.argmax(val_probs, axis=1)

    # Create analyzer
    analyzer = GeometricQualityAnalyzer(
        features=val_features,
        labels=val_labels,
        train_features=train_features,
        train_labels=train_labels,
        predictions=val_predictions
    )

    # Compute all predictors and make prediction
    prediction_result = analyzer.predict_calibration_success()

    # Package results
    result = {
        'layer_idx': layer_idx,
        'feature_dim': train_features.shape[1],
        'train_samples': len(train_features),
        'val_samples': len(val_features),
        **prediction_result
    }

    return result


def generate_visualizations(
    results: List[Dict],
    output_dir: Path,
    dataset: str,
    model_name: str
):
    """
    Generate visualizations for layer analysis results.

    Creates:
        1. Radar charts for each layer
        2. Heatmap across all layers
        3. Layer progression line plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data for plotting
    layer_indices = [r['layer_idx'] for r in results]
    predictors = results[0]['predictors'].keys()

    # Prepare dataframe
    data = []
    for r in results:
        row = {'layer_idx': r['layer_idx'], 'will_work': r['will_work']}
        row.update(r['predictors'])
        data.append(row)

    df = pd.DataFrame(data)

    # ========================================================================
    # 1. HEATMAP: Layers × Predictors
    # ========================================================================
    logger.info("Generating heatmap...")

    fig, ax = plt.subplots(figsize=(14, max(8, len(layer_indices) * 0.4)))

    # Prepare matrix
    predictor_names = list(predictors)
    matrix = df[predictor_names].values

    # Create heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0.5,
        vmin=0,
        vmax=1,
        xticklabels=[p.replace('_', '\n') for p in predictor_names],
        yticklabels=[f"Layer {idx}" for idx in layer_indices],
        cbar_kws={'label': 'Metric Value'},
        ax=ax
    )

    ax.set_title(f'Geometric Quality Predictors Heatmap\n{model_name} on {dataset.upper()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predictor', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_layers_predictors.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {output_dir / 'heatmap_layers_predictors.png'}")

    # ========================================================================
    # 2. LAYER PROGRESSION: Line plots
    # ========================================================================
    logger.info("Generating layer progression plots...")

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, predictor in enumerate(predictor_names):
        ax = axes[idx]

        values = df[predictor].values
        ax.plot(layer_indices, values, marker='o', linewidth=2, markersize=8)

        # Highlight best layer
        best_idx = np.argmax(values)
        ax.scatter([layer_indices[best_idx]], [values[best_idx]],
                  color='red', s=200, zorder=5, marker='*', label='Best')

        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(predictor.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # Overall title
    fig.suptitle(f'Layer Progression of Geometric Quality Predictors\n{model_name} on {dataset.upper()}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'layer_progression.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {output_dir / 'layer_progression.png'}")

    # ========================================================================
    # 3. RADAR CHARTS: Top 3 layers
    # ========================================================================
    logger.info("Generating radar charts for top layers...")

    # Find top 3 layers by composite score
    # Create composite score: average of key metrics
    key_metrics = [
        'separation_accuracy_correlation',
        'safe_dangerous_gap',
        'distance_ratio',
        'nn_class_purity'
    ]

    df['composite'] = df[key_metrics].mean(axis=1)
    top_layers = df.nlargest(min(3, len(df)), 'composite')

    for _, row in top_layers.iterrows():
        layer_idx = int(row['layer_idx'])

        # Prepare data for radar chart
        categories = [p.replace('_', '\n') for p in predictor_names]
        values = [row[p] for p in predictor_names]

        # Close the plot
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        ax.plot(angles, values, 'o-', linewidth=2, label=f'Layer {layer_idx}')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True)

        prediction = 'WILL WORK ✅' if row['will_work'] else 'UNLIKELY ❌'
        ax.set_title(f'Layer {layer_idx} Geometric Quality Analysis\n{model_name} on {dataset.upper()}\nPrediction: {prediction}',
                    fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_dir / f'radar_layer_{layer_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved: {output_dir / f'radar_layer_{layer_idx}.png'}")


def generate_markdown_report(
    results: List[Dict],
    output_file: Path,
    dataset: str,
    model_name: str,
    training_method: str = "unknown"
):
    """
    Generate markdown report summarizing analysis.
    """
    logger.info("Generating markdown report...")

    with open(output_file, 'w') as f:
        f.write(f"# Geometric Quality Analysis Report\n\n")
        f.write(f"**Model:** {model_name}  \n")
        f.write(f"**Dataset:** {dataset.upper()}  \n")
        f.write(f"**Training Method:** {training_method}  \n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")

        f.write("---\n\n")
        f.write("## Executive Summary\n\n")

        # Count predictions
        will_work = sum(1 for r in results if r['will_work'])
        total = len(results)

        f.write(f"Analyzed **{total}** layers:\n")
        f.write(f"- ✅ **{will_work}** layers predicted to work well\n")
        f.write(f"- ❌ **{total - will_work}** layers unlikely to work well\n\n")

        # Find best layer
        best_result = max(results, key=lambda r: r['n_conditions_satisfied'])
        f.write(f"**Recommended Layer:** {best_result['layer_idx']} ({best_result['n_conditions_satisfied']}/4 conditions met)\n\n")

        f.write("---\n\n")
        f.write("## Layer Recommendations\n\n")

        # Sort by number of conditions satisfied
        sorted_results = sorted(results, key=lambda r: r['n_conditions_satisfied'], reverse=True)

        for r in sorted_results:
            layer_idx = r['layer_idx']
            confidence = r['confidence'].upper()
            will_work = r['will_work']
            n_cond = r['n_conditions_satisfied']

            # Determine emoji
            if will_work and confidence == 'HIGH':
                emoji = '✅'
            elif will_work:
                emoji = '⚠️'
            else:
                emoji = '❌'

            f.write(f"### {emoji} Layer {layer_idx} - {confidence} CONFIDENCE\n\n")
            f.write(f"**Prediction:** {'WILL WORK' if will_work else 'UNLIKELY'}\n\n")
            f.write(f"**Reason:** {r['reason']}\n\n")
            f.write(f"**Conditions Met:** {n_cond}/4\n\n")

            # Show conditions
            f.write("**Conditions:**\n")
            for cond, satisfied in r['conditions_met'].items():
                check = '✅' if satisfied else '❌'
                f.write(f"- {check} {cond}\n")
            f.write("\n")

            # Show key metrics
            f.write("**Key Metrics:**\n")
            p = r['predictors']
            f.write(f"- Separation-Accuracy Correlation: **{p['separation_accuracy_correlation']:.4f}**\n")
            f.write(f"- Safe/Dangerous Gap: **{p['safe_dangerous_gap']:.4f}**\n")
            f.write(f"- Distance Ratio: **{p['distance_ratio']:.4f}**\n")
            f.write(f"- Impostor Rate: **{p['impostor_rate']:.4f}**\n")
            f.write(f"- NN Class Purity: **{p['nn_class_purity']:.4f}**\n")
            f.write("\n")

            # Show recommendations
            f.write("**Recommendations:**\n")
            for rec in r['recommendations']:
                f.write(f"- {rec}\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("## Dataset Pattern Explanation\n\n")

        # Heuristic explanations based on dataset
        if 'cifar10' in dataset.lower():
            f.write("**CIFAR-10 (Coarse Classes):** Geometric separation typically emerges early because classes\n")
            f.write("are semantically distinct. Best layers are usually in the middle (before neural collapse reduces\n")
            f.write("dimensionality too much).\n\n")
        elif 'cifar100' in dataset.lower():
            f.write("**CIFAR-100 (Fine-Grained Classes):** Requires deeper layers to achieve class separation\n")
            f.write("due to fine-grained distinctions. Best layers are usually later in the network.\n\n")
        elif 'svhn' in dataset.lower():
            f.write("**SVHN (Digits):** High baseline accuracy means geometric separation is strong throughout.\n")
            f.write("Earlier layers may work well due to simpler task.\n\n")

        f.write("---\n\n")
        f.write("## Visualizations\n\n")
        f.write("See the following generated plots:\n\n")
        f.write("1. `heatmap_layers_predictors.png` - Overview of all layers and predictors\n")
        f.write("2. `layer_progression.png` - How each predictor changes across layers\n")
        f.write("3. `radar_layer_*.png` - Detailed analysis for top layers\n")

    logger.info(f"  Saved: {output_file}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze geometric calibration quality for model layers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and data
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "cifar100", "svhn"],
        help="Dataset name"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Model architecture"
    )
    parser.add_argument(
        "--training-method",
        type=str,
        default="baseline",
        help="Training method used"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Training seed"
    )

    # Alternative: checkpoint directory
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory containing checkpoints (alternative to --checkpoint)"
    )

    # Layer selection
    parser.add_argument(
        "--layers",
        type=int,
        nargs='+',
        help="Specific layer indices to analyze (default: auto-detect candidates)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/geometric_quality_analysis"),
        help="Output directory for results"
    )

    # Compute settings
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for feature extraction"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load model
    logger.info("Loading model...")
    try:
        if args.checkpoint:
            # Load from specific checkpoint
            model = torch.load(args.checkpoint, map_location=args.device)
            checkpoint_path = args.checkpoint
        elif args.checkpoint_dir:
            # Load using helper function
            model, checkpoint_path = load_trained_model(
                dataset=args.dataset,
                model_name=args.model,
                training_method=args.training_method,
                seed=args.seed,
                checkpoint_base_dir=args.checkpoint_dir,
                device=args.device
            )
        else:
            raise ValueError("Must provide either --checkpoint or --checkpoint-dir")

        logger.info(f"Loaded model from: {checkpoint_path}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Load data
    logger.info("Loading data...")
    try:
        train_loader, val_loader, test_loader, num_classes = get_data_loaders(
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_workers=0
        )
        logger.info(f"Loaded {args.dataset} dataset with {num_classes} classes")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    # Convert validation set to numpy for predictions
    logger.info("Preparing validation data...")
    val_original_list = []
    val_labels_list = []
    for batch_x, batch_y in val_loader:
        val_original_list.append(batch_x.numpy())
        val_labels_list.append(batch_y.numpy())

    val_original = np.concatenate(val_original_list, axis=0)
    val_labels_concat = np.concatenate(val_labels_list, axis=0)

    # Get candidate layers
    if args.layers:
        candidate_layers = args.layers
    else:
        candidate_layers = CrossModalLayerMapper.get_candidate_indices(model)

    logger.info(f"Analyzing {len(candidate_layers)} candidate layers: {candidate_layers}")

    # Create model adapter
    model_adapter = PyTorchModelAdapter(model, device=args.device)

    # Move model to device
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # Analyze each layer
    results = []

    for layer_idx in tqdm(candidate_layers, desc="Analyzing layers"):
        try:
            result = analyze_layer(
                layer_idx=layer_idx,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                model_adapter=model_adapter,
                val_original=val_original
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to analyze layer {layer_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    if not results:
        logger.error("No layers analyzed successfully")
        return 1

    # Create output directory
    output_dir = args.output_dir / args.dataset / args.model / args.training_method / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results as JSON
    logger.info("Saving results...")
    with open(output_dir / "layer_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"  Saved: {output_dir / 'layer_analysis.json'}")

    # Save as CSV
    csv_data = []
    for r in results:
        row = {
            'layer_idx': r['layer_idx'],
            'feature_dim': r['feature_dim'],
            'will_work': r['will_work'],
            'confidence': r['confidence'],
            'n_conditions_satisfied': r['n_conditions_satisfied'],
        }
        row.update(r['predictors'])
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv(output_dir / "layer_comparison.csv", index=False)
    logger.info(f"  Saved: {output_dir / 'layer_comparison.csv'}")

    # Generate visualizations
    logger.info("Generating visualizations...")
    generate_visualizations(
        results=results,
        output_dir=output_dir,
        dataset=args.dataset,
        model_name=args.model
    )

    # Generate markdown report
    generate_markdown_report(
        results=results,
        output_file=output_dir / "REPORT.md",
        dataset=args.dataset,
        model_name=args.model,
        training_method=args.training_method
    )

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - layer_analysis.json")
    logger.info(f"  - layer_comparison.csv")
    logger.info(f"  - REPORT.md")
    logger.info(f"  - Visualizations (PNG files)")
    logger.info("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
