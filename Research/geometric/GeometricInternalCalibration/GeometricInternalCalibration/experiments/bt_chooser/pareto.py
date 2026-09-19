"""
Pareto front utility module for BT chooser experiments.

This module provides functions for finding Pareto-optimal solutions when maximizing
multiple objectives simultaneously.
"""

import logging
from typing import Dict, List

import numpy as np

# Configure logging
from utils.logging_config import get_logger
logger = get_logger(__name__)


def pareto_front(X: np.ndarray) -> np.ndarray:
    """
    Find Pareto-optimal points (non-dominated rows) when maximizing all columns.
    
    A point is Pareto-optimal if there is no other point that is better in at least
    one objective and not worse in any other objective.
    
    Args:
        X: 2D array where rows are points and columns are objectives to maximize
        
    Returns:
        Boolean mask indicating which rows are on the Pareto front
        
    Raises:
        ValueError: If X is not 2D or has no data
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D array")
    
    if X.size == 0:
        return np.array([], dtype=bool)
    
    n_points, n_objectives = X.shape
    
    if n_points == 1:
        return np.array([True])
    
    # Initialize all points as potentially Pareto-optimal
    is_pareto = np.ones(n_points, dtype=bool)
    
    # O(n^2) algorithm: for each point, check if it's dominated by any other point
    for i in range(n_points):
        if not is_pareto[i]:
            continue  # Already determined to be dominated
            
        for j in range(n_points):
            if i == j:
                continue
                
            # Check if point j dominates point i
            # j dominates i if: j >= i in all objectives AND j > i in at least one objective
            j_better_or_equal = np.all(X[j] >= X[i])
            j_strictly_better = np.any(X[j] > X[i])
            
            if j_better_or_equal and j_strictly_better:
                is_pareto[i] = False
                break  # i is dominated, no need to check other points
    
    n_pareto = np.sum(is_pareto)
    logger.debug(f"Found {n_pareto}/{n_points} Pareto-optimal points")
    
    return is_pareto


def filter_pareto(feats: Dict[int, Dict[str, float]], metric_names: List[str]) -> List[int]:
    """
    Find layers on the Pareto front based on specified metrics.
    
    Args:
        feats: Dictionary mapping layer_idx to metric values
        metric_names: List of metric names to use for Pareto comparison
        
    Returns:
        List of layer_idx values that are on the Pareto front
        
    Raises:
        ValueError: If no valid layers found or metrics are missing
    """
    if not feats:
        logger.warning("No features provided")
        return []
    
    if not metric_names:
        logger.warning("No metric names provided")
        return []
    
    # Find layers that have all required metrics
    valid_layers = []
    for layer_idx, layer_feats in feats.items():
        if all(metric in layer_feats for metric in metric_names):
            valid_layers.append(layer_idx)
    
    if not valid_layers:
        logger.warning(f"No layers found with all metrics: {metric_names}")
        return []
    
    if len(valid_layers) == 1:
        logger.debug("Only one valid layer, it's trivially Pareto-optimal")
        return valid_layers
    
    # Extract metric values for valid layers
    X = np.zeros((len(valid_layers), len(metric_names)))
    for i, layer_idx in enumerate(valid_layers):
        for j, metric in enumerate(metric_names):
            X[i, j] = feats[layer_idx][metric]
    
    # Find Pareto-optimal points
    is_pareto = pareto_front(X)
    
    # Return layer indices that are Pareto-optimal
    pareto_layers = [valid_layers[i] for i in range(len(valid_layers)) if is_pareto[i]]
    
    logger.info(f"Found {len(pareto_layers)}/{len(valid_layers)} Pareto-optimal layers")
    return pareto_layers


def test_pareto_front():
    """Test the pareto_front function with synthetic cases."""
    
    # Test case 1: Simple 2D case
    print("Test 1: Simple 2D case")
    X1 = np.array([
        [1, 2],  # Dominated by [2, 2]
        [2, 2],  # Pareto-optimal
        [1, 3],  # Pareto-optimal
        [0, 4],  # Pareto-optimal
    ])
    result1 = pareto_front(X1)
    expected1 = np.array([False, True, True, True])
    assert np.array_equal(result1, expected1), f"Expected {expected1}, got {result1}"
    print(f"✓ Test 1 passed: {result1}")
    
    # Test case 2: All points Pareto-optimal (no dominance)
    print("\nTest 2: All points Pareto-optimal")
    X2 = np.array([
        [1, 0],
        [0, 1],
        [0.5, 0.5],
    ])
    result2 = pareto_front(X2)
    expected2 = np.array([True, True, True])
    assert np.array_equal(result2, expected2), f"Expected {expected2}, got {result2}"
    print(f"✓ Test 2 passed: {result2}")
    
    # Test case 3: Single point
    print("\nTest 3: Single point")
    X3 = np.array([[1, 2]])
    result3 = pareto_front(X3)
    expected3 = np.array([True])
    assert np.array_equal(result3, expected3), f"Expected {expected3}, got {result3}"
    print(f"✓ Test 3 passed: {result3}")
    
    # Test case 4: Empty array
    print("\nTest 4: Empty array")
    X4 = np.array([]).reshape(0, 2)
    result4 = pareto_front(X4)
    expected4 = np.array([], dtype=bool)
    assert np.array_equal(result4, expected4), f"Expected {expected4}, got {result4}"
    print(f"✓ Test 4 passed: {result4}")
    
    # Test case 5: 3D case with clear dominance
    print("\nTest 5: 3D case with clear dominance")
    X5 = np.array([
        [1, 1, 1],  # Dominated by [2, 1, 1]
        [2, 1, 1],  # Dominated by [2, 2, 1]
        [2, 2, 1],  # Dominated by [2, 2, 2]
        [2, 2, 2],  # Pareto-optimal
        [1, 2, 2],  # Pareto-optimal
        [2, 1, 2],  # Pareto-optimal
    ])
    result5 = pareto_front(X5)
    expected5 = np.array([False, False, False, True, True, True])
    assert np.array_equal(result5, expected5), f"Expected {expected5}, got {result5}"
    print(f"✓ Test 5 passed: {result5}")
    
    print("\n🎉 All pareto_front tests passed!")


def test_filter_pareto():
    """Test the filter_pareto function with synthetic cases."""
    
    # Test case 1: Simple case with clear Pareto front
    print("Test 1: Simple case with clear Pareto front")
    feats1 = {
        0: {"metric1": 1.0, "metric2": 2.0},  # Dominated by layer 1
        1: {"metric1": 2.0, "metric2": 2.0},  # Pareto-optimal
        2: {"metric1": 1.0, "metric2": 3.0},  # Pareto-optimal
        3: {"metric1": 0.0, "metric2": 4.0},  # Pareto-optimal
    }
    result1 = filter_pareto(feats1, ["metric1", "metric2"])
    expected1 = [1, 2, 3]  # Layer 0 is dominated
    assert set(result1) == set(expected1), f"Expected {expected1}, got {result1}"
    print(f"✓ Test 1 passed: {result1}")
    
    # Test case 2: Missing metrics
    print("\nTest 2: Missing metrics")
    feats2 = {
        0: {"metric1": 1.0, "metric2": 2.0},
        1: {"metric1": 2.0},  # Missing metric2
        2: {"metric1": 1.0, "metric2": 3.0},
    }
    result2 = filter_pareto(feats2, ["metric1", "metric2"])
    expected2 = [0, 2]  # Layer 1 excluded due to missing metric2
    assert set(result2) == set(expected2), f"Expected {expected2}, got {result2}"
    print(f"✓ Test 2 passed: {result2}")
    
    # Test case 3: Single valid layer
    print("\nTest 3: Single valid layer")
    feats3 = {
        0: {"metric1": 1.0, "metric2": 2.0},
    }
    result3 = filter_pareto(feats3, ["metric1", "metric2"])
    expected3 = [0]
    assert result3 == expected3, f"Expected {expected3}, got {result3}"
    print(f"✓ Test 3 passed: {result3}")
    
    # Test case 4: Empty features
    print("\nTest 4: Empty features")
    feats4 = {}
    result4 = filter_pareto(feats4, ["metric1", "metric2"])
    expected4 = []
    assert result4 == expected4, f"Expected {expected4}, got {result4}"
    print(f"✓ Test 4 passed: {result4}")
    
    print("\n🎉 All filter_pareto tests passed!")


if __name__ == "__main__":
    test_pareto_front()
    test_filter_pareto()
