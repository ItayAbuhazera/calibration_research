"""
Pairwise dataset builder module for BT chooser experiments.

This module provides functions for building pairwise training datasets from experiment runs,
with feature extraction, pair formation, and optional downsampling.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .data_io import RunRecord
from .features import make_layer_features, add_within_run_ranks

# Configure logging
from utils.logging_config import get_logger
logger = get_logger(__name__)


def build_pairs(
    run: RunRecord, 
    metric_names: List[str], 
    eps: float = 0.003, 
    max_pairs: Optional[int] = 400, 
    use_ranks: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Build pairwise dataset from a single run.
    
    Args:
        run: RunRecord containing the experiment data
        metric_names: List of metric names to use for features
        eps: Minimum ECE difference threshold for keeping pairs
        max_pairs: Maximum number of pairs to return (None for no limit)
        use_ranks: Whether to include rank features
        
    Returns:
        Tuple of (X, y, w, layer_order) where:
        - X: Feature matrix of pair differences (n_pairs, n_features)
        - y: Binary labels (1 if target_ece[i] < target_ece[j], 0 otherwise)
        - w: Pair weights (absolute ECE differences)
        - layer_order: List of layer_idx values in the order used for features
    """
    logger.info(f"Building pairs for run {run.run_id} with {len(metric_names)} metrics")
    
    # Extract layer features
    if use_ranks:
        layer_features = make_layer_features(run, metric_names)
        layer_features = add_within_run_ranks(layer_features)
    else:
        layer_features = make_layer_features(run, metric_names)
    
    if not layer_features:
        logger.warning(f"No valid layer features found for run {run.run_id}")
        return np.array([]).reshape(0, 0), np.array([]), np.array([]), []
    
    # Build feature matrix per layer
    X_per_layer, layer_order = _build_layer_feature_matrix(layer_features, metric_names, use_ranks)
    
    if X_per_layer.size == 0:
        logger.warning(f"Empty feature matrix for run {run.run_id}")
        return np.array([]).reshape(0, 0), np.array([]), np.array([]), layer_order
    
    # Form pairs and labels
    X_pairs, y, w = _form_pairs(X_per_layer, run, layer_order, eps)
    
    if X_pairs.size == 0:
        logger.warning(f"No valid pairs found for run {run.run_id}")
        return np.array([]).reshape(0, X_per_layer.shape[1]), np.array([]), np.array([]), layer_order
    
    # Optional downsampling
    if max_pairs is not None and len(y) > max_pairs:
        orig = len(y)
        X_pairs, y, w = _downsample_pairs(X_pairs, y, w, max_pairs)
        logger.info(f"Downsampled to {len(y)} pairs (from {orig})")
    
    logger.info(f"Built {len(y)} pairs with {X_pairs.shape[1]} features")
    return X_pairs, y, w, layer_order


def _build_layer_feature_matrix(
    layer_features: Dict[int, Dict[str, float]], 
    metric_names: List[str], 
    use_ranks: bool
) -> Tuple[np.ndarray, List[int]]:
    """
    Build feature matrix per layer with fixed column order and imputation.
    
    Args:
        layer_features: Dictionary mapping layer_idx to metric values
        metric_names: List of metric names to use
        use_ranks: Whether rank features are included
        
    Returns:
        Tuple of (feature_matrix, layer_order) where feature_matrix has shape (n_layers, n_features)
    """
    layer_order = sorted(layer_features.keys())
    n_layers = len(layer_order)
    
    # Determine column order: [metric_names..., metric_names+"_rank"...]
    columns = []
    for metric in metric_names:
        columns.append(metric)
    if use_ranks:
        for metric in metric_names:
            columns.append(f"{metric}_rank")
    
    n_features = len(columns)
    X = np.full((n_layers, n_features), np.nan)
    
    # Fill in available values
    for i, layer_idx in enumerate(layer_order):
        layer_data = layer_features[layer_idx]
        for j, col_name in enumerate(columns):
            if col_name in layer_data:
                X[i, j] = layer_data[col_name]
    
    # Impute missing values with neutral values
    for j in range(n_features):
        col_data = X[:, j]
        valid_mask = ~np.isnan(col_data)
        
        if np.any(valid_mask):
            median_val = np.median(col_data[valid_mask])
            X[np.isnan(col_data), j] = median_val
            logger.debug(f"Imputed {np.sum(np.isnan(col_data))} missing values for {columns[j]} with median {median_val:.4f}")
        else:
            # Use neutral imputation: 0.5 for rank features, 0.0 for raw features
            if columns[j].endswith("_rank"):
                X[:, j] = 0.5  # Neutral rank
                logger.warning(f"No valid values for {columns[j]}, setting to neutral rank 0.5")
            else:
                X[:, j] = 0.0  # Neutral raw value
                logger.warning(f"No valid values for {columns[j]}, setting to 0")
    
    return X, layer_order


def _form_pairs(
    X_per_layer: np.ndarray, 
    run: RunRecord, 
    layer_order: List[int], 
    eps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Form all i≠j pairs with labels and weights.
    
    Args:
        X_per_layer: Feature matrix per layer (n_layers, n_features)
        run: RunRecord for accessing target ECE values
        layer_order: List of layer_idx values in order
        eps: Minimum ECE difference threshold
        
    Returns:
        Tuple of (X_pairs, y, w) where:
        - X_pairs: Feature differences (n_pairs, n_features)
        - y: Binary labels
        - w: Pair weights (absolute ECE differences)
    """
    n_layers = len(layer_order)
    if n_layers < 2:
        return np.array([]).reshape(0, X_per_layer.shape[1]), np.array([]), np.array([])
    
    # Create mapping from layer_idx to target_ece
    layer_to_ece = {}
    for layer in run.layers:
        layer_to_ece[layer["layer_idx"]] = layer["target_ece"]
    
    # Form all pairs
    pairs = []
    labels = []
    weights = []
    
    for i in range(n_layers):
        for j in range(n_layers):
            if i == j:
                continue
                
            layer_i = layer_order[i]
            layer_j = layer_order[j]
            
            # Get target ECE values
            ece_i = layer_to_ece.get(layer_i, 0.0)
            ece_j = layer_to_ece.get(layer_j, 0.0)
            
            # Calculate ECE difference
            ece_diff = ece_i - ece_j
            abs_ece_diff = abs(ece_diff)
            
            # Skip pairs with small ECE difference
            if abs_ece_diff < eps:
                continue
            
            # Create feature difference
            feature_diff = X_per_layer[i] - X_per_layer[j]
            
            # Label: 1 if layer_i is better (lower ECE), 0 otherwise
            label = 1 if ece_diff < 0 else 0
            
            pairs.append(feature_diff)
            labels.append(label)
            weights.append(abs_ece_diff)
    
    if not pairs:
        logger.warning(f"No pairs found with ECE difference >= {eps}")
        return np.array([]).reshape(0, X_per_layer.shape[1]), np.array([]), np.array([])
    
    X_pairs = np.array(pairs)
    y = np.array(labels)
    w = np.array(weights)
    
    logger.debug(f"Formed {len(pairs)} pairs with ECE threshold {eps}")
    return X_pairs, y, w


def _downsample_pairs(
    X_pairs: np.ndarray, 
    y: np.ndarray, 
    w: np.ndarray, 
    max_pairs: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample pairs by keeping those with largest margins (weights) first.
    
    Args:
        X_pairs: Feature matrix of pairs
        y: Binary labels
        w: Pair weights
        max_pairs: Maximum number of pairs to keep
        
    Returns:
        Downsampled (X_pairs, y, w)
    """
    if len(y) <= max_pairs:
        return X_pairs, y, w
    
    # Sort by weight (largest first)
    sorted_indices = np.argsort(w)[::-1]
    
    # Keep top max_pairs
    keep_indices = sorted_indices[:max_pairs]
    
    return X_pairs[keep_indices], y[keep_indices], w[keep_indices]


def test_build_pairs():
    """Test the build_pairs function with synthetic data."""
    
    # Create test run
    test_run = RunRecord(
        run_id="test_run_001",
        dataset="cifar10",
        backbone="resnet50", 
        seed=42,
        layers=[
            {
                "layer_idx": 0,
                "metrics": {
                    "prototype_softmax_ece": 0.12,
                    "reliability_curve_quality": 0.85,
                    "confidence_distance_correlation": 0.72,
                },
                "target_ece": 0.15
            },
            {
                "layer_idx": 1,
                "metrics": {
                    "prototype_softmax_ece": 0.14,
                    "reliability_curve_quality": 0.82,
                    "confidence_distance_correlation": 0.68,
                },
                "target_ece": 0.18
            },
            {
                "layer_idx": 2,
                "metrics": {
                    "prototype_softmax_ece": 0.10,
                    "reliability_curve_quality": 0.90,
                    "confidence_distance_correlation": 0.75,
                },
                "target_ece": 0.12
            }
        ]
    )
    
    metric_names = ["prototype_softmax_ece", "reliability_curve_quality", "confidence_distance_correlation"]
    
    print("Testing build_pairs...")
    
    # Test with ranks
    X, y, w, layer_order = build_pairs(test_run, metric_names, eps=0.01, use_ranks=True)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels: {y}")
    print(f"Weights: {w}")
    print(f"Layer order: {layer_order}")
    
    # Should have 3 layers, so 6 pairs (3 choose 2 * 2)
    expected_pairs = 6
    assert len(y) == expected_pairs, f"Expected {expected_pairs} pairs, got {len(y)}"
    
    # Check that all pairs have positive weights (above threshold)
    assert np.all(w > 0), "All weights should be positive"
    
    # Check feature matrix has correct number of columns (3 metrics + 3 ranks = 6)
    expected_features = len(metric_names) * 2  # metrics + ranks
    assert X.shape[1] == expected_features, f"Expected {expected_features} features, got {X.shape[1]}"
    
    print("✓ build_pairs test passed!")
    
    # Test without ranks
    X_no_ranks, y_no_ranks, w_no_ranks, layer_order_no_ranks = build_pairs(
        test_run, metric_names, eps=0.01, use_ranks=False
    )
    
    # Should have same number of pairs but fewer features
    assert len(y_no_ranks) == expected_pairs
    assert X_no_ranks.shape[1] == len(metric_names)  # Only metrics, no ranks
    
    print("✓ build_pairs without ranks test passed!")
    
    # Test downsampling
    X_down, y_down, w_down, _ = build_pairs(test_run, metric_names, eps=0.01, max_pairs=3)
    
    assert len(y_down) <= 3, f"Expected at most 3 pairs, got {len(y_down)}"
    assert len(y_down) > 0, "Should have at least some pairs"
    
    print("✓ build_pairs downsampling test passed!")
    
    print("\n🎉 All build_pairs tests passed!")


if __name__ == "__main__":
    test_build_pairs()
