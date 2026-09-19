#!/usr/bin/env python3
"""
Evaluation harness for Bradley-Terry chooser on historical runs.

This script evaluates a trained BT chooser on historical experiment runs,
computing hit rates, regret, tie statistics, and Borda resolution rates.
"""

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from .data_io import load_runs, RunRecord
from .infer import choose_layer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from utils.logging_config import get_logger
logger = get_logger(__name__)


def bootstrap_confidence_interval(
    data: List[float], 
    confidence: float = 0.95, 
    n_bootstrap: int = 1000
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a dataset.
    
    Args:
        data: List of values to compute CI for
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if not data:
        return 0.0, 0.0, 0.0
    
    data = np.array(data)
    n = len(data)
    
    if n == 0:
        return 0.0, 0.0, 0.0
    
    # Bootstrap sampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    mean_val = np.mean(data)
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return mean_val, lower_bound, upper_bound


def evaluate_run(
    run: RunRecord, 
    bt_path: str, 
    apply_pareto: bool = True, 
    tie_delta: float = 1e-4
) -> Dict:
    """
    Evaluate a single run with the BT chooser.
    
    Args:
        run: RunRecord to evaluate
        bt_path: Path to trained Bradley-Terry model
        apply_pareto: Whether to apply Pareto filtering
        tie_delta: Threshold for tie detection
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Choose layer using BT chooser
        result = choose_layer(bt_path, run, apply_pareto=apply_pareto, tie_delta=tie_delta)
        
        # Find true best layer by target ECE
        layer_to_ece = {layer["layer_idx"]: layer["target_ece"] for layer in run.layers}
        true_best_layer = min(layer_to_ece.keys(), key=lambda x: layer_to_ece[x])
        true_best_ece = layer_to_ece[true_best_layer]
        
        # Compute metrics
        if result["strategy"] == "single":
            chosen_layer = result["layer_idx"]
            chosen_ece = layer_to_ece.get(chosen_layer, float('inf'))
            
            hit1 = 1 if chosen_layer == true_best_layer else 0
            regret = chosen_ece - true_best_ece
            ties = 0
            borda_resolved = 0
            
        elif result["strategy"] == "tie":
            # Check if any of the tied layers is the true best
            tied_layers = [item[0] for item in result["top2"]]
            hit1 = 1 if true_best_layer in tied_layers else 0
            
            # Regret is the difference between best tied layer and true best
            tied_eces = [layer_to_ece.get(layer, float('inf')) for layer in tied_layers]
            best_tied_ece = min(tied_eces)
            regret = best_tied_ece - true_best_ece
            
            ties = 1
            borda_resolved = 1 if result.get("borda_used", False) else 0
            
        else:  # error case
            hit1 = 0
            regret = float('inf')
            ties = 0
            borda_resolved = 0
        
        return {
            "run_id": run.run_id,
            "dataset": run.dataset,
            "backbone": run.backbone,
            "hit1": hit1,
            "regret": regret,
            "ties": ties,
            "borda_resolved": borda_resolved,
            "strategy": result["strategy"],
            "n_layers": len(run.layers)
        }
        
    except Exception as e:
        logger.error(f"Error evaluating run {run.run_id}: {e}")
        return {
            "run_id": run.run_id,
            "dataset": run.dataset,
            "backbone": run.backbone,
            "hit1": 0,
            "regret": float('inf'),
            "ties": 0,
            "borda_resolved": 0,
            "strategy": "error",
            "n_layers": len(run.layers),
            "error": str(e)
        }


def evaluate_all_runs(
    runs: List[RunRecord], 
    bt_path: str, 
    apply_pareto: bool = True, 
    tie_delta: float = 1e-4
) -> List[Dict]:
    """
    Evaluate all runs with the BT chooser.
    
    Args:
        runs: List of RunRecords to evaluate
        bt_path: Path to trained Bradley-Terry model
        apply_pareto: Whether to apply Pareto filtering
        tie_delta: Threshold for tie detection
        
    Returns:
        List of evaluation results
    """
    logger.info(f"Evaluating {len(runs)} runs")
    
    results = []
    for i, run in enumerate(runs):
        logger.debug(f"Evaluating run {i+1}/{len(runs)}: {run.run_id}")
        result = evaluate_run(run, bt_path, apply_pareto, tie_delta)
        results.append(result)
    
    return results


def compute_aggregates(results: List[Dict]) -> Dict:
    """
    Compute aggregate statistics from evaluation results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary with aggregate statistics
    """
    if not results:
        return {}
    
    # Extract metrics
    hit1_values = [r["hit1"] for r in results if r["hit1"] is not None]
    regret_values = [r["regret"] for r in results if r["regret"] != float('inf')]
    tie_values = [r["ties"] for r in results]
    borda_values = [r["borda_resolved"] for r in results]
    
    # Compute basic statistics
    n_runs = len(results)
    n_hits = sum(hit1_values)
    n_ties = sum(tie_values)
    n_borda_resolved = sum(borda_values)
    
    hit1_rate = n_hits / n_runs if n_runs > 0 else 0.0
    tie_rate = n_ties / n_runs if n_runs > 0 else 0.0
    borda_resolution_rate = n_borda_resolved / n_ties if n_ties > 0 else 0.0
    
    # Compute confidence intervals
    hit1_mean, hit1_lower, hit1_upper = bootstrap_confidence_interval(hit1_values)
    
    if regret_values:
        regret_mean, regret_lower, regret_upper = bootstrap_confidence_interval(regret_values)
    else:
        regret_mean = regret_lower = regret_upper = 0.0
    
    return {
        "n_runs": n_runs,
        "hit1_rate": hit1_rate,
        "hit1_mean": hit1_mean,
        "hit1_ci": (hit1_lower, hit1_upper),
        "regret_mean": regret_mean,
        "regret_ci": (regret_lower, regret_upper),
        "tie_rate": tie_rate,
        "borda_resolution_rate": borda_resolution_rate,
        "n_ties": n_ties,
        "n_borda_resolved": n_borda_resolved
    }


def compute_group_aggregates(results: List[Dict], group_by: str) -> Dict[str, Dict]:
    """
    Compute aggregate statistics by group.
    
    Args:
        results: List of evaluation results
        group_by: How to group results ("dataset" or "dataset_backbone")
        
    Returns:
        Dictionary mapping group names to aggregate statistics
    """
    # Group results
    groups = defaultdict(list)
    for result in results:
        if group_by == "dataset":
            group_key = result["dataset"]
        elif group_by == "dataset_backbone":
            group_key = f"{result['dataset']}_{result['backbone']}"
        else:
            group_key = "all"
        
        groups[group_key].append(result)
    
    # Compute aggregates for each group
    group_aggregates = {}
    for group_key, group_results in groups.items():
        group_aggregates[group_key] = compute_aggregates(group_results)
    
    return group_aggregates


def print_evaluation_results(
    overall_aggregates: Dict, 
    group_aggregates: Dict[str, Dict],
    group_by: str
) -> None:
    """
    Print evaluation results in a formatted table.
    
    Args:
        overall_aggregates: Overall aggregate statistics
        group_aggregates: Per-group aggregate statistics
        group_by: How results are grouped
    """
    print("\n" + "="*80)
    print("BRADLEY-TERRY CHOOSER EVALUATION RESULTS")
    print("="*80)
    
    # Overall results
    print(f"\nOVERALL RESULTS ({overall_aggregates['n_runs']} runs)")
    print("-" * 50)
    print(f"Hit Rate:     {overall_aggregates['hit1_rate']:.3f} ({overall_aggregates['hit1_ci'][0]:.3f}, {overall_aggregates['hit1_ci'][1]:.3f})")
    print(f"Regret:       {overall_aggregates['regret_mean']:.4f} ({overall_aggregates['regret_ci'][0]:.4f}, {overall_aggregates['regret_ci'][1]:.4f})")
    print(f"Tie Rate:     {overall_aggregates['tie_rate']:.3f} ({overall_aggregates['n_ties']} ties)")
    print(f"Borda Res.:   {overall_aggregates['borda_resolution_rate']:.3f} ({overall_aggregates['n_borda_resolved']}/{overall_aggregates['n_ties']})")
    
    # Per-group results
    print(f"\nPER-{group_by.upper()} RESULTS")
    print("-" * 50)
    
    # Sort groups by hit rate (descending)
    sorted_groups = sorted(group_aggregates.items(), 
                          key=lambda x: x[1]['hit1_rate'], reverse=True)
    
    print(f"{'Group':<20} {'Runs':<6} {'Hit Rate':<12} {'Regret':<12} {'Ties':<8} {'Borda':<8}")
    print("-" * 80)
    
    for group_key, stats in sorted_groups:
        hit_ci = stats['hit1_ci']
        regret_ci = stats['regret_ci']
        
        print(f"{group_key:<20} {stats['n_runs']:<6} "
              f"{stats['hit1_rate']:.3f}±{max(hit_ci[1]-stats['hit1_rate'], stats['hit1_rate']-hit_ci[0]):.3f} "
              f"{stats['regret_mean']:.4f}±{max(regret_ci[1]-stats['regret_mean'], stats['regret_mean']-regret_ci[0]):.4f} "
              f"{stats['tie_rate']:.3f} "
              f"{stats['borda_resolution_rate']:.3f}")
    
    print("="*80)


def print_leaderboard(group_aggregates: Dict[str, Dict]) -> None:
    """
    Print a short leaderboard per dataset.
    
    Args:
        group_aggregates: Per-group aggregate statistics
    """
    # Group by dataset only
    dataset_groups = defaultdict(list)
    for group_key, stats in group_aggregates.items():
        if '_' in group_key:
            dataset = group_key.split('_')[0]
        else:
            dataset = group_key
        dataset_groups[dataset].append((group_key, stats))
    
    print("\n" + "="*60)
    print("DATASET LEADERBOARDS")
    print("="*60)
    
    for dataset, groups in dataset_groups.items():
        if len(groups) <= 1:
            continue  # Skip datasets with only one group
        
        print(f"\n{dataset.upper()}")
        print("-" * 30)
        
        # Sort by hit rate
        sorted_groups = sorted(groups, key=lambda x: x[1]['hit1_rate'], reverse=True)
        
        for i, (group_key, stats) in enumerate(sorted_groups[:5]):  # Top 5
            rank = i + 1
            backbone = group_key.split('_', 1)[1] if '_' in group_key else group_key
            print(f"{rank}. {backbone:<15} {stats['hit1_rate']:.3f} ({stats['n_runs']} runs)")
    
    print("="*60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate Bradley-Terry chooser on historical runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--bt",
        type=str,
        required=True,
        help="Path to trained Bradley-Terry model JSON file"
    )
    
    parser.add_argument(
        "--runs_dir",
        type=str,
        required=True,
        help="Directory containing experiment runs"
    )
    
    parser.add_argument(
        "--group_by",
        type=str,
        choices=["dataset", "dataset_backbone"],
        default="dataset",
        help="How to group runs for analysis"
    )
    
    parser.add_argument(
        "--apply_pareto",
        action="store_true",
        help="Apply Pareto filtering during inference"
    )
    
    parser.add_argument(
        "--tie-delta",
        type=float,
        default=1e-4,
        help="Threshold for detecting ties in top scores"
    )
    
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence intervals"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    bt_path = Path(args.bt)
    if not bt_path.exists():
        logger.error(f"Model file does not exist: {bt_path}")
        return
    
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        logger.error(f"Runs directory does not exist: {runs_dir}")
        return
    
    # Load runs
    logger.info(f"Loading runs from {runs_dir}")
    runs = load_runs(str(runs_dir))
    
    if not runs:
        logger.error("No runs found in directory")
        return
    
    logger.info(f"Loaded {len(runs)} runs")
    
    # Evaluate all runs
    logger.info("Evaluating runs...")
    results = evaluate_all_runs(
        runs, 
        str(bt_path), 
        apply_pareto=args.apply_pareto, 
        tie_delta=args.tie_delta
    )
    
    # Compute aggregates
    logger.info("Computing aggregates...")
    overall_aggregates = compute_aggregates(results)
    group_aggregates = compute_group_aggregates(results, args.group_by)
    
    # Print results
    print_evaluation_results(overall_aggregates, group_aggregates, args.group_by)
    print_leaderboard(group_aggregates)
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
