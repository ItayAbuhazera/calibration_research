#!/usr/bin/env python3
"""
Training CLI for Bradley-Terry chooser experiments.

This script trains a Bradley-Terry chooser model on experiment runs using
grouped cross-validation and saves the trained model.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Set

import numpy as np

from .data_io import load_runs, available_metrics
from .bt_model import BradleyTerryChooser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from utils.logging_config import get_logger
logger = get_logger(__name__)


def infer_metric_names(runs: List, threshold: float = 0.8) -> List[str]:
    """
    Infer metric names as intersection over all runs of metrics available on ≥threshold of layers.
    
    Args:
        runs: List of RunRecord objects
        threshold: Minimum fraction of layers that must have each metric
        
    Returns:
        List of metric names that meet the threshold
    """
    if not runs:
        return []
    
    # Collect all available metrics across all runs
    all_metrics = set()
    for run in runs:
        run_metrics = available_metrics(run)
        all_metrics.update(run_metrics)
    
    if not all_metrics:
        return []
    
    # Count how many layers have each metric
    metric_counts = {}
    total_layers = 0
    
    for run in runs:
        for layer in run.layers:
            total_layers += 1
            layer_metrics = set(layer["metrics"].keys())
            for metric in all_metrics:
                if metric in layer_metrics:
                    metric_counts[metric] = metric_counts.get(metric, 0) + 1
    
    # Filter metrics that meet the threshold
    valid_metrics = []
    for metric, count in metric_counts.items():
        if count >= threshold * total_layers:
            valid_metrics.append(metric)
    
    # Sort for consistent ordering
    valid_metrics.sort()
    
    logger.info(f"Found {len(valid_metrics)} metrics available on ≥{threshold:.0%} of layers")
    logger.info(f"Valid metrics: {valid_metrics}")
    
    return valid_metrics


def print_dataset_counts(runs: List) -> None:
    """Print dataset and backbone counts."""
    if not runs:
        logger.warning("No runs provided")
        return
    
    # Count by dataset
    dataset_counts = {}
    backbone_counts = {}
    dataset_backbone_counts = {}
    
    for run in runs:
        dataset_counts[run.dataset] = dataset_counts.get(run.dataset, 0) + 1
        backbone_counts[run.backbone] = backbone_counts.get(run.backbone, 0) + 1
        key = f"{run.dataset}_{run.backbone}"
        dataset_backbone_counts[key] = dataset_backbone_counts.get(key, 0) + 1
    
    print("\n" + "="*50)
    print("DATASET AND BACKBONE COUNTS")
    print("="*50)
    
    print(f"\nTotal runs: {len(runs)}")
    
    print(f"\nBy dataset:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"  {dataset}: {count}")
    
    print(f"\nBy backbone:")
    for backbone, count in sorted(backbone_counts.items()):
        print(f"  {backbone}: {count}")
    
    print(f"\nBy dataset_backbone:")
    for key, count in sorted(dataset_backbone_counts.items()):
        print(f"  {key}: {count}")
    
    print("="*50)


def print_training_results(report: dict) -> None:
    """Print training results including hit rate, regret, and feature weights."""
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    # CV results
    cv_results = report.get('cv_results', [])
    if cv_results:
        best_result = cv_results[0]  # Already sorted by performance
        print(f"\nBest C: {report['best_C']}")
        print(f"CV Hit Rate: {best_result['hit_rate_mean']:.3f} ± {best_result['hit_rate_std']:.3f}")
        print(f"CV Regret: {best_result['regret_mean']:.3f} ± {best_result['regret_std']:.3f}")
        print(f"CV Folds: {best_result['n_folds']}")
    
    # Feature weights
    feature_weights = report.get('feature_weights', [])
    feature_names = report.get('feature_names', [])
    
    if feature_weights and feature_names:
        print(f"\nFeature Weights (Top 5):")
        # Create list of (name, weight) tuples and sort by absolute weight
        weight_pairs = list(zip(feature_names, feature_weights))
        weight_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for i, (name, weight) in enumerate(weight_pairs[:5]):
            print(f"  {i+1}. {name}: {weight:.4f}")
        
        if len(weight_pairs) > 5:
            print(f"  ... and {len(weight_pairs) - 5} more features")
    
    # Training statistics
    print(f"\nTraining Statistics:")
    print(f"  Runs: {report.get('n_runs', 0)}")
    print(f"  Pairs: {report.get('n_pairs', 0)}")
    print(f"  Metrics: {len(report.get('metric_names', []))}")
    print(f"  Use Ranks: {report.get('use_ranks', False)}")
    print(f"  Standardize: {report.get('standardize', False)}")
    
    print("="*50)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Bradley-Terry chooser for layer selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--runs_dir",
        type=str,
        required=True,
        help="Directory containing experiment runs"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="bt_chooser.json",
        help="Output path for trained model"
    )
    
    parser.add_argument(
        "--group_by",
        type=str,
        choices=["dataset", "dataset_backbone"],
        default="dataset",
        help="How to group runs for cross-validation"
    )
    
    parser.add_argument(
        "--eps",
        type=float,
        default=0.003,
        help="Minimum ECE difference threshold for keeping pairs"
    )
    
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=400,
        help="Maximum pairs per run (None for no limit)"
    )
    
    parser.add_argument(
        "--metric_names",
        type=str,
        default="auto",
        help="Comma-separated metric names or 'auto' to infer from data"
    )
    
    parser.add_argument(
        "--use_ranks",
        action="store_true",
        help="Include rank features"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        logger.error(f"Runs directory does not exist: {runs_dir}")
        sys.exit(1)
    
    # Load runs
    logger.info(f"Loading runs from {runs_dir}")
    runs = load_runs(str(runs_dir))
    
    if not runs:
        logger.error("No runs found in directory")
        sys.exit(1)
    
    logger.info(f"Loaded {len(runs)} runs")
    
    # Print dataset counts
    print_dataset_counts(runs)
    
    # Determine metric names
    if args.metric_names == "auto":
        logger.info("Auto-detecting metric names...")
        metric_names = infer_metric_names(runs)
        if not metric_names:
            logger.error("No valid metrics found with 80% availability threshold")
            sys.exit(1)
    else:
        metric_names = [name.strip() for name in args.metric_names.split(",")]
        logger.info(f"Using specified metrics: {metric_names}")
    
    # Create chooser
    chooser = BradleyTerryChooser(
        metric_names=metric_names,
        use_ranks=args.use_ranks,
        standardize=True,
        penalty="l2",
        C_grid=[0.1, 0.3, 1, 3, 10],
        class_weight="balanced",
        eps_tie=args.eps,
        max_pairs_per_run=args.max_pairs,
        random_state=args.seed
    )
    
    # Train model
    logger.info("Training Bradley-Terry chooser...")
    try:
        report = chooser.fit(runs, group_by=args.group_by)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
    
    # Print results
    print_training_results(report)
    
    # Save model
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        chooser.save(str(output_path))
        logger.info(f"Model saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        sys.exit(1)
    
    print(f"\n✅ Training completed successfully!")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
