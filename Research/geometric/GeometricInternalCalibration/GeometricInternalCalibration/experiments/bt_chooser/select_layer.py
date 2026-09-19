#!/usr/bin/env python3
"""
CLI for running Bradley-Terry chooser on a single run.

This script loads a trained model and applies it to a single experiment run
to select the best layer, displaying results in a formatted table.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .data_io import RunRecord, load_run_file
from .infer import choose_layer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from utils.logging_config import get_logger
logger = get_logger(__name__)


def load_single_run(run_json_path: str) -> RunRecord:
    """
    Load a single run from JSON file.
    
    Args:
        run_json_path: Path to the run JSON file
        
    Returns:
        RunRecord object
        
    Raises:
        ValueError: If no valid run is found
    """
    logger.info(f"Loading run from {run_json_path}")
    
    run = load_run_file(run_json_path)
    if not run:
        raise ValueError(f"No valid run in {run_json_path}")
    return run


def format_results_table(
    run: RunRecord, 
    selection_result: Dict, 
    all_scores: List[Tuple[int, float]]
) -> None:
    """
    Print a formatted table of layer results.
    
    Args:
        run: RunRecord containing the experiment data
        selection_result: Result from choose_layer
        all_scores: List of (layer_idx, score) tuples for all layers
    """
    print("\n" + "="*80)
    print("LAYER SELECTION RESULTS")
    print("="*80)
    
    # Create mapping from layer_idx to target_ece
    layer_to_ece = {layer["layer_idx"]: layer["target_ece"] for layer in run.layers}
    
    # Sort all scores by score (descending)
    all_scores_sorted = sorted(all_scores, key=lambda x: x[1], reverse=True)
    
    # Find best ECE for delta calculation
    all_eces = [layer_to_ece.get(layer_idx, float('inf')) for layer_idx, _ in all_scores_sorted]
    best_ece = min(ece for ece in all_eces if ece != float('inf'))
    
    # Print table header
    print(f"{'Layer':<8} {'Score':<12} {'Target ECE':<12} {'Delta to Best':<15} {'Status'}")
    print("-" * 80)
    
    # Print each layer
    for i, (layer_idx, score) in enumerate(all_scores_sorted):
        target_ece = layer_to_ece.get(layer_idx, None)
        delta_to_best = target_ece - best_ece if target_ece is not None else None
        
        # Determine status
        if selection_result["strategy"] == "single" and layer_idx == selection_result["layer_idx"]:
            status = "SELECTED"
        elif selection_result["strategy"] == "tie" and layer_idx in [item[0] for item in selection_result["top2"]]:
            status = "TIED"
        else:
            status = ""
        
        # Format values
        score_str = f"{score:.6f}"
        ece_str = f"{target_ece:.6f}" if target_ece is not None else "N/A"
        delta_str = f"{delta_to_best:.6f}" if delta_to_best is not None else "N/A"
        
        print(f"{layer_idx:<8} {score_str:<12} {ece_str:<12} {delta_str:<15} {status}")
    
    print("-" * 80)
    
    # Print selection summary
    print(f"\nSelection Strategy: {selection_result['strategy']}")
    
    if selection_result["strategy"] == "single":
        print(f"Selected Layer: {selection_result['layer_idx']}")
        print(f"Score: {selection_result['score']:.6f}")
    elif selection_result["strategy"] == "tie":
        print(f"Tied Layers: {[item[0] for item in selection_result['top2']]}")
        print(f"Suggestion: {selection_result['suggestion']}")
    elif selection_result["strategy"] == "error":
        print(f"Error: {selection_result.get('message', 'Unknown error')}")
    
    print("="*80)


def get_all_layer_scores(run: RunRecord, bt_path: str) -> List[Tuple[int, float]]:
    """
    Get scores for all layers in the run.
    
    Args:
        run: RunRecord containing the experiment data
        bt_path: Path to trained Bradley-Terry model
        
    Returns:
        List of (layer_idx, score) tuples for all layers
    """
    from .infer import prepare_layer_matrix
    from .bt_model import BradleyTerryChooser
    
    # Load model
    chooser = BradleyTerryChooser(
        metric_names=[],  # Will be loaded from file
        use_ranks=True,   # Will be loaded from file
        random_state=0
    )
    chooser.load(bt_path)
    
    # Prepare layer matrix
    X, layer_order = prepare_layer_matrix(run, chooser.metric_names, chooser.use_ranks)
    
    if X.size == 0:
        return []
    
    # Score all layers
    scores = chooser.score_layers(X)
    
    # Return as list of (layer_idx, score) tuples
    return list(zip(layer_order, scores))


def main():
    """Main selection function."""
    parser = argparse.ArgumentParser(
        description="Select best layer using trained Bradley-Terry chooser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--bt",
        type=str,
        required=True,
        help="Path to trained Bradley-Terry model JSON file"
    )
    
    parser.add_argument(
        "--run_json",
        type=str,
        required=True,
        help="Path to run JSON file (per_layer_ground_truth.json or run_summary.json)"
    )
    
    parser.add_argument(
        "--no-pareto",
        action="store_true",
        help="Disable Pareto filtering"
    )
    
    parser.add_argument(
        "--tie-delta",
        type=float,
        default=1e-4,
        help="Threshold for detecting ties in top scores"
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
    bt_path = Path(args.bt)
    if not bt_path.exists():
        logger.error(f"Model file does not exist: {bt_path}")
        sys.exit(0)  # Exit code 0 for analysis
    
    run_json_path = Path(args.run_json)
    if not run_json_path.exists():
        logger.error(f"Run JSON file does not exist: {run_json_path}")
        sys.exit(0)  # Exit code 0 for analysis
    
    # Load run
    try:
        run = load_single_run(str(run_json_path))
        logger.info(f"Loaded run {run.run_id} with {len(run.layers)} layers")
    except Exception as e:
        logger.error(f"Failed to load run: {e}")
        sys.exit(0)  # Exit code 0 for analysis
    
    # Get all layer scores for table
    try:
        all_scores = get_all_layer_scores(run, str(bt_path))
        logger.info(f"Computed scores for {len(all_scores)} layers")
    except Exception as e:
        logger.error(f"Failed to compute layer scores: {e}")
        sys.exit(0)  # Exit code 0 for analysis
    
    # Choose best layer
    try:
        apply_pareto = not args.no_pareto
        selection_result = choose_layer(
            str(bt_path), 
            run, 
            apply_pareto=apply_pareto, 
            tie_delta=args.tie_delta
        )
        logger.info(f"Layer selection completed: {selection_result['strategy']}")
    except Exception as e:
        logger.error(f"Failed to select layer: {e}")
        selection_result = {"strategy": "error", "message": str(e)}
    
    # Print results table
    try:
        format_results_table(run, selection_result, all_scores)
    except Exception as e:
        logger.error(f"Failed to format results: {e}")
        print(f"Error formatting results: {e}")
    
    # Always exit with code 0 (for analysis)
    sys.exit(0)


if __name__ == "__main__":
    main()
