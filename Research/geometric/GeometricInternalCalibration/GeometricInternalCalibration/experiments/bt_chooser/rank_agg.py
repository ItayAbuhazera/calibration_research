"""
Rank aggregation module for BT chooser experiments.

This module provides Borda count ranking as a fallback tie-breaking method
when Bradley-Terry chooser results in ties.
"""

import logging
from typing import Dict, List

import numpy as np

from .data_io import RunRecord
from .features import make_layer_features, add_within_run_ranks

# Configure logging
from utils.logging_config import get_logger
logger = get_logger(__name__)


def borda_from_metrics(run: RunRecord, metric_names: List[str]) -> Dict[int, float]:
    """
    Compute Borda count scores for layers based on metric rankings.
    
    For each metric, ranks layers in descending order (higher is better).
    Each layer gets a normalized rank score, and the final Borda score is the
    average normalized rank across all metrics.
    
    Args:
        run: RunRecord containing the experiment data
        metric_names: List of metric names to use for ranking
        
    Returns:
        Dictionary mapping layer_idx to Borda score
    """
    logger.debug(f"Computing Borda scores for run {run.run_id} with metrics {metric_names}")
    
    if not run.layers:
        logger.warning("No layers found in run")
        return {}
    
    # Extract layer features
    layer_features = make_layer_features(run, metric_names)
    if not layer_features:
        logger.warning("No valid layer features found")
        return {}
    
    # Get layer order and build feature matrix
    layer_order = sorted(layer_features.keys())
    n_layers = len(layer_order)
    n_metrics = len(metric_names)
    
    if n_layers == 0:
        return {}
    
    # Build feature matrix
    X = np.full((n_layers, n_metrics), np.nan)
    
    for i, layer_idx in enumerate(layer_order):
        layer_data = layer_features[layer_idx]
        for j, metric in enumerate(metric_names):
            if metric in layer_data:
                X[i, j] = layer_data[metric]
    
    # Impute missing values with median across layers
    for j in range(n_metrics):
        col_data = X[:, j]
        valid_mask = ~np.isnan(col_data)
        
        if np.any(valid_mask):
            median_val = np.median(col_data[valid_mask])
            X[np.isnan(col_data), j] = median_val
        else:
            X[:, j] = 0.0
    
    # Compute Borda scores
    borda_scores = {}
    
    if n_layers == 1:
        # Single layer gets score 1.0
        borda_scores[layer_order[0]] = 1.0
    else:
        # For each metric, rank layers (descending order)
        metric_ranks = np.zeros((n_layers, n_metrics))
        
        for j in range(n_metrics):
            # Get values for this metric
            values = X[:, j]
            
            # Rank in descending order (higher values get better ranks)
            # Use 'average' method to handle ties
            ranks = np.argsort(np.argsort(values, kind='mergesort'), kind='mergesort')
            
            # Normalize ranks to [0, 1] where 1 is best
            if n_layers > 1:
                normalized_ranks = ranks / (n_layers - 1)
            else:
                normalized_ranks = np.array([1.0])
            
            metric_ranks[:, j] = normalized_ranks
        
        # Average normalized ranks across metrics
        avg_ranks = np.mean(metric_ranks, axis=1)
        
        # Map back to layer indices
        for i, layer_idx in enumerate(layer_order):
            borda_scores[layer_idx] = float(avg_ranks[i])
    
    logger.debug(f"Computed Borda scores: {borda_scores}")
    return borda_scores


def test_borda_ranking():
    """Test the Borda ranking function with synthetic data."""
    from .data_io import RunRecord
    
    # Create test run with clear ranking
    test_run = RunRecord(
        run_id="test_borda",
        dataset="cifar10",
        backbone="resnet50",
        seed=42,
        layers=[
            {"layer_idx": 0, "metrics": {"metric1": 0.8, "metric2": 0.6}, "target_ece": 0.15},
            {"layer_idx": 1, "metrics": {"metric1": 0.7, "metric2": 0.8}, "target_ece": 0.12},
            {"layer_idx": 2, "metrics": {"metric1": 0.9, "metric2": 0.5}, "target_ece": 0.18},
        ]
    )
    
    print("Testing Borda ranking...")
    
    # Test with two metrics
    borda_scores = borda_from_metrics(test_run, ["metric1", "metric2"])
    
    print(f"Borda scores: {borda_scores}")
    
    # Check that scores are in [0, 1] range
    for layer_idx, score in borda_scores.items():
        assert 0 <= score <= 1, f"Score {score} not in [0, 1] range"
    
    # Check that we have scores for all layers
    assert len(borda_scores) == 3, f"Expected 3 scores, got {len(borda_scores)}"
    
    print("✓ Borda ranking test passed!")
    
    # Test with single layer
    single_layer_run = RunRecord(
        run_id="test_single",
        dataset="cifar10",
        backbone="resnet50",
        seed=42,
        layers=[
            {"layer_idx": 0, "metrics": {"metric1": 0.8, "metric2": 0.6}, "target_ece": 0.15},
        ]
    )
    
    single_scores = borda_from_metrics(single_layer_run, ["metric1", "metric2"])
    assert len(single_scores) == 1, f"Expected 1 score, got {len(single_scores)}"
    assert single_scores[0] == 1.0, f"Single layer should have score 1.0, got {single_scores[0]}"
    
    print("✓ Single layer test passed!")
    
    print("\n🎉 All Borda ranking tests passed!")


if __name__ == "__main__":
    test_borda_ranking()
