"""
Feature preprocessing module for BT chooser experiments.

This module provides functions for canonicalizing metrics, extracting layer features,
and adding within-run ranking features for model training.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from .data_io import RunRecord

# Configure logging
from utils.logging_config import get_logger
logger = get_logger(__name__)

# Direction overrides for all standard metrics
# All metrics are maximized (higher is better)
DIRECTION_OVERRIDES = {
    "prototype_softmax_ece": "max",
    "reliability_curve_quality": "max", 
    "confidence_distance_correlation": "max",
    "boundary_proximity_correlation": "max",
    "uncertainty_geometry_alignment": "max",
    "avg_class_separation_ratio": "max",
    "class_separation_uniformity": "max",
    "local_intrinsic_dimensionality": "max",
    "label_cka": "max",
    "label_cka_hsic": "max",
    "margin_tail_cvar": "max",
    "multiscale_separation": "max",
    "geometry_error_concordance": "max",
    "kfold_ece_utility": "max",
    "logistic_calibratability": "max",
    "spearman_stability_accuracy": "max"
}


def canonicalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Canonicalize metrics by converting negative ECE metrics to utilities and clipping values.
    
    For negative ECE metrics (like prototype_softmax_ece), converts them to utilities:
    - If value ≤ 0: val = 1 - clip(-val, 0, 1)
    - Otherwise: leave untouched
    
    All values are clipped to [0, 1] range and only metrics in DIRECTION_OVERRIDES are kept.
    
    Args:
        metrics: Dictionary of metric names to values
        
    Returns:
        Canonicalized dictionary with only keys in DIRECTION_OVERRIDES
    """
    canonicalized = {}
    
    for metric_name, value in metrics.items():
        if metric_name not in DIRECTION_OVERRIDES:
            continue
            
        # Handle negative ECE metrics (convert to utility)
        if metric_name == "prototype_softmax_ece" and value <= 0:
            # Convert negative ECE to utility: 1 - clip(-val, 0, 1)
            utility = 1 - np.clip(-value, 0, 1)
            canonicalized[metric_name] = float(utility)
        else:
            # For other metrics, just clip to [0, 1] range
            clipped_value = np.clip(value, 0, 1)
            canonicalized[metric_name] = float(clipped_value)
    
    return canonicalized


def make_layer_features(run: RunRecord, metric_names: List[str]) -> Dict[int, Dict[str, float]]:
    """
    Build per-layer dictionary of raw features (post-canonicalization) for specified metrics.
    
    Args:
        run: RunRecord containing the experiment data
        metric_names: List of metric names to extract
        
    Returns:
        Dictionary mapping layer_idx to canonicalized metric values
    """
    layer_features = {}
    
    for layer in run.layers:
        layer_idx = layer["layer_idx"]
        layer_metrics = layer["metrics"]
        
        # Filter to only requested metrics
        filtered_metrics = {
            name: value for name, value in layer_metrics.items() 
            if name in metric_names
        }
        
        # Canonicalize the metrics
        canonicalized = canonicalize_metrics(filtered_metrics)
        
        if canonicalized:  # Only include layers with valid metrics
            layer_features[layer_idx] = canonicalized
    
    logger.debug(f"Extracted features for {len(layer_features)} layers with metrics: {metric_names}")
    return layer_features


def add_within_run_ranks(layer_features: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    """
    Add within-run rank features for each metric present across layers.
    
    For each metric present across layers, adds a rank feature `metric+"_rank"` 
    scaled to [0, 1] where 0=worst and 1=best. Ignores metrics missing on some layers.
    
    Args:
        layer_features: Dictionary mapping layer_idx to metric values
        
    Returns:
        Dictionary with original features plus rank features
    """
    if not layer_features:
        return layer_features
    
    # Find metrics that are present across all layers
    all_metrics = set()
    for layer_data in layer_features.values():
        all_metrics.update(layer_data.keys())
    
    # Find metrics present in all layers (no missing values)
    common_metrics = set()
    for metric in all_metrics:
        if all(metric in layer_data for layer_data in layer_features.values()):
            common_metrics.add(metric)
    
    logger.debug(f"Adding ranks for {len(common_metrics)} common metrics: {common_metrics}")
    
    # Create result dictionary with original features
    result = {}
    for layer_idx, layer_data in layer_features.items():
        result[layer_idx] = layer_data.copy()
    
    # Add rank features for common metrics
    for metric in common_metrics:
        # Extract values for this metric across all layers
        metric_values = []
        layer_indices = []
        
        for layer_idx, layer_data in layer_features.items():
            if metric in layer_data:
                metric_values.append(layer_data[metric])
                layer_indices.append(layer_idx)
        
        if len(metric_values) < 2:
            continue  # Need at least 2 values to rank
        
        # Convert to numpy array for easier ranking
        values_array = np.array(metric_values)
        
        # Calculate ranks (higher values get higher ranks)
        # Use 'average' method to handle ties
        ranks = np.argsort(np.argsort(values_array, kind='mergesort'), kind='mergesort')
        
        # Scale ranks to [0, 1] where 0=worst, 1=best
        if len(ranks) > 1:
            scaled_ranks = ranks / (len(ranks) - 1)
        else:
            scaled_ranks = np.array([0.5])  # Single value gets middle rank
        
        # Add rank features
        rank_metric_name = f"{metric}_rank"
        for i, layer_idx in enumerate(layer_indices):
            result[layer_idx][rank_metric_name] = float(scaled_ranks[i])
    
    logger.debug(f"Added rank features. Result has {len(result)} layers")
    return result


def extract_features_for_modeling(run: RunRecord, metric_names: List[str]) -> Dict[int, Dict[str, float]]:
    """
    Extract and preprocess features ready for modeling.
    
    This is a convenience function that combines make_layer_features and add_within_run_ranks.
    
    Args:
        run: RunRecord containing the experiment data
        metric_names: List of metric names to extract
        
    Returns:
        Dictionary mapping layer_idx to features (raw + ranks) ready for modeling
    """
    # Extract raw features
    layer_features = make_layer_features(run, metric_names)
    
    # Add within-run ranks
    features_with_ranks = add_within_run_ranks(layer_features)
    
    logger.info(f"Extracted {len(features_with_ranks)} layer features with {len(metric_names)} metrics")
    return features_with_ranks
