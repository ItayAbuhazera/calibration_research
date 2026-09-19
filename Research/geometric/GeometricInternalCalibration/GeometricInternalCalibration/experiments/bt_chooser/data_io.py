"""
Data I/O module for BT chooser experiments.

This module provides data structures and loading functions for experiment runs,
with robust parsing and normalization of metrics across different experiment formats.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
from utils.logging_config import get_logger
logger = get_logger(__name__)


def _infer_meta_from_path(p: Path):
    # Expect: .../<dataset>/<backbone>/<seed>/per_layer_ground_truth.json
    try:
        dataset, backbone, seed_dir = p.parts[-4], p.parts[-3], p.parts[-2]
        seed = int(seed_dir.replace("seed", "")) if seed_dir.startswith("seed") else None
    except Exception:
        dataset, backbone, seed = "unknown", "unknown", None
    run_id = f"{dataset}_{backbone}_{seed if seed is not None else 'x'}"
    return run_id, dataset, backbone, seed


def _read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _parse_split_candidate_format(gt_path_str: str) -> Optional['RunRecord']:
    """
    Support format:
      per_layer_ground_truth.json:
        {
          "candidate_layers": [5, 10, ...],
          "val": {"ece": {...}} or "test": {"ece": {...}}
        }
      all_metrics_per_layer.json:
        {
          "candidate_layers": [...],
          "metrics": [{"layer_idx": 5, "<metric>": <float>, ...}, ...]
        }
    """
    p = Path(gt_path_str)
    if p.name != "per_layer_ground_truth.json":
        return None

    try:
        gt = _read_json(p)
    except Exception as e:
        logger.error(f"Error reading {p}: {e}")
        return None

    candidates = gt.get("candidate_layers", [])
    if not isinstance(candidates, list) or not all(isinstance(x, int) for x in candidates):
        # Not the format we're looking for
        return None

    # Choose ECE source preference: test, then val
    ece_map = {}
    for split_key in ("test", "val"):
        split = gt.get(split_key)
        if isinstance(split, dict) and isinstance(split.get("ece"), dict):
            # keys might be strings
            for k, v in split["ece"].items():
                try:
                    ece_map[int(k)] = float(v)
                except Exception:
                    pass
            break

    # Read sibling metrics file
    metrics_path = p.with_name("all_metrics_per_layer.json")
    if not metrics_path.exists():
        logger.warning(f"No all_metrics_per_layer.json next to {p}, cannot merge split format")
        return None

    try:
        metrics_blob = _read_json(metrics_path)
    except Exception as e:
        logger.error(f"Error reading {metrics_path}: {e}")
        return None

    metrics_list = metrics_blob.get("metrics", [])
    if not isinstance(metrics_list, list) or not metrics_list:
        logger.warning(f"'metrics' missing/empty in {metrics_path}")
        return None

    # Build layer_idx -> metrics dict (dropping non-feature fields)
    metrics_by_layer = {}
    for row in metrics_list:
        if not isinstance(row, dict) or "layer_idx" not in row:
            continue
        lid = row["layer_idx"]
        try:
            lid = int(lid)
        except Exception:
            continue
        feats = {k: v for k, v in row.items()
                 if k not in ("layer_idx", "computation_times")
                 and isinstance(v, (int, float))}
        metrics_by_layer[lid] = feats

    # Construct layers
    layers = []
    missing_metrics = 0
    for lid in candidates:
        feats = metrics_by_layer.get(lid)
        if feats is None:
            missing_metrics += 1
            continue
        target_ece = ece_map.get(lid)
        if target_ece is None:
            # ok if absent for some layers; trainer may filter later
            pass
        layers.append({
            "layer_idx": lid,              # preserve real layer id from your files
            "metrics": feats,              # dense feature dict
            "target_ece": target_ece,      # float or None
        })

    if not layers:
        logger.warning(f"Split-format merge produced 0 layers for {p}")
        return None
    if missing_metrics:
        logger.warning(f"{missing_metrics} candidate layers had no metrics in {metrics_path}")

    run_id, dataset, backbone, seed = _infer_meta_from_path(p)
    return RunRecord(
        run_id=run_id,
        dataset=dataset,
        backbone=backbone,
        seed=seed,
        layers=layers
    )


# Standardized metric names that will be normalized to
STANDARD_METRICS = [
    "prototype_softmax_ece",
    "reliability_curve_quality", 
    "confidence_distance_correlation",
    "boundary_proximity_correlation",
    "uncertainty_geometry_alignment",
    "avg_class_separation_ratio",
    "class_separation_uniformity",
    "local_intrinsic_dimensionality",
    "label_cka",
    "label_cka_hsic",
    "margin_tail_cvar",
    "multiscale_separation",
    "geometry_error_concordance",
    "kfold_ece_utility",
    "logistic_calibratability",
    "spearman_stability_accuracy"
]


@dataclass
class RunRecord:
    """
    Data class capturing one experiment run.
    
    Attributes:
        run_id: Unique identifier for the run
        dataset: Name of the dataset used
        backbone: Architecture/backbone model name
        seed: Random seed used (optional)
        layers: List of layer dictionaries with metrics and target ECE
    """
    run_id: str
    dataset: str
    backbone: str
    seed: Optional[int]
    layers: List[Dict[str, Any]]


def _normalize_metric_name(metric_name: str) -> Optional[str]:
    """
    Normalize metric names to standard format.
    
    Args:
        metric_name: Original metric name
        
    Returns:
        Normalized metric name if it matches a standard metric, None otherwise
    """
    # Direct match
    if metric_name in STANDARD_METRICS:
        return metric_name
    
    # Common variations and mappings
    metric_mappings = {
        "ece": "prototype_softmax_ece",
        "final_ece": "prototype_softmax_ece", 
        "target_ece": "prototype_softmax_ece",
        "reliability_curve": "reliability_curve_quality",
        "confidence_distance": "confidence_distance_correlation",
        "boundary_proximity": "boundary_proximity_correlation",
        "uncertainty_geometry": "uncertainty_geometry_alignment",
        "class_separation": "avg_class_separation_ratio",
        "separation_uniformity": "class_separation_uniformity",
        "intrinsic_dimensionality": "local_intrinsic_dimensionality",
        "cka": "label_cka",
        "cka_hsic": "label_cka_hsic",
        "tail_cvar": "margin_tail_cvar",
        "multiscale": "multiscale_separation",
        "geometry_concordance": "geometry_error_concordance",
        "kfold_ece": "kfold_ece_utility",
        "logistic_cal": "logistic_calibratability",
        "spearman_stability": "spearman_stability_accuracy"
    }
    
    # Try exact match first
    if metric_name in metric_mappings:
        return metric_mappings[metric_name]
    
    # Try case-insensitive match
    for key, value in metric_mappings.items():
        if metric_name.lower() == key.lower():
            return value
    
    # Try partial matches
    for key, value in metric_mappings.items():
        if key in metric_name.lower() or metric_name.lower() in key:
            return value
    
    return None


def _clean_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Clean metrics by removing NaN and infinite values.
    
    Args:
        metrics: Dictionary of metric names to values
        
    Returns:
        Cleaned dictionary with only valid float values
    """
    cleaned = {}
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            if np.isfinite(value):
                cleaned[name] = float(value)
            else:
                logger.debug(f"Dropping non-finite metric {name}: {value}")
        else:
            logger.debug(f"Dropping non-numeric metric {name}: {value}")
    return cleaned


def _extract_target_ece(layer_data: Dict[str, Any]) -> float:
    """
    Extract target ECE from layer data using tolerant key mapping.
    
    Args:
        layer_data: Dictionary containing layer information
        
    Returns:
        Target ECE value, 0.0 if not found
    """
    ece_keys = ["ece", "final_ece", "target_ece"]
    
    for key in ece_keys:
        if key in layer_data:
            value = layer_data[key]
            if isinstance(value, (int, float)) and np.isfinite(value):
                return float(value)
    
    logger.warning(f"No valid target ECE found in layer data: {layer_data}")
    return 0.0


def _parse_layer_data(layer_data: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
    """
    Parse and normalize layer data.
    
    Args:
        layer_data: Raw layer data from JSON
        layer_idx: Index of the layer
        
    Returns:
        Normalized layer dictionary
    """
    # Extract target ECE
    target_ece = _extract_target_ece(layer_data)
    
    # Clean and normalize metrics
    metrics = {}
    for key, value in layer_data.items():
        if key not in ["ece", "final_ece", "target_ece", "layer_idx"]:
            normalized_name = _normalize_metric_name(key)
            if normalized_name:
                if isinstance(value, (int, float)) and np.isfinite(value):
                    metrics[normalized_name] = float(value)
                else:
                    logger.debug(f"Skipping non-finite metric {key}: {value}")
            else:
                logger.debug(f"Skipping unrecognized metric {key}")
    
    return {
        "layer_idx": layer_idx,
        "metrics": metrics,
        "target_ece": target_ece
    }


def _parse_run_data(data: Dict[str, Any], file_path: str) -> Optional[RunRecord]:
    """
    Parse run data from JSON into RunRecord.
    
    Args:
        data: Parsed JSON data
        file_path: Path to the JSON file for error reporting
        
    Returns:
        RunRecord if parsing successful, None otherwise
    """
    try:
        # Extract run metadata with tolerant key mapping
        run_id = data.get("run_id", data.get("id", os.path.basename(file_path)))
        dataset = data.get("dataset", data.get("dataset_name", "unknown"))
        backbone = data.get("backbone", data.get("arch", "unknown"))
        seed = data.get("seed", data.get("random_seed"))
        
        # Parse layers
        layers = []
        if "layers" in data:
            for i, layer_data in enumerate(data["layers"]):
                parsed_layer = _parse_layer_data(layer_data, i)
                layers.append(parsed_layer)
        elif "per_layer_metrics" in data:
            for i, layer_data in enumerate(data["per_layer_metrics"]):
                parsed_layer = _parse_layer_data(layer_data, i)
                layers.append(parsed_layer)
        else:
            # First: try our split-candidate format (your files)
            split = _parse_split_candidate_format(file_path)
            if split:
                return split

            # Last-resort: only treat as layers if it's a list of dicts (not candidate_layers)
            for key, val in data.items():
                if ("layer" in key.lower()
                    and key.lower() != "candidate_layers"
                    and isinstance(val, list)
                    and len(val) > 0
                    and isinstance(val[0], dict)):
                    for i, layer_data in enumerate(val):
                        parsed_layer = _parse_layer_data(layer_data, i)
                        layers.append(parsed_layer)
                    break
        
        if not layers:
            logger.warning(f"No layer data found in {file_path}")
            return None
        
        return RunRecord(
            run_id=run_id,
            dataset=dataset,
            backbone=backbone,
            seed=seed,
            layers=layers
        )
        
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return None


def load_runs(runs_dir: str) -> List[RunRecord]:
    """
    Recursively scan runs_dir for JSON files and parse into RunRecord objects.
    
    Looks for files named either 'per_layer_ground_truth.json' or 'run_summary.json'.
    Uses tolerant key mapping for different experiment formats.
    
    Args:
        runs_dir: Directory to scan for run data
        
    Returns:
        List of RunRecord objects parsed from found JSON files
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        logger.error(f"Runs directory does not exist: {runs_dir}")
        return []
    
    target_files = ["per_layer_ground_truth.json", "run_summary.json"]
    runs = []
    
    logger.info(f"Scanning {runs_dir} for run data...")
    
    # Recursively find target JSON files
    for file_path in runs_dir.rglob("*.json"):
        if file_path.name in target_files:
            logger.info(f"Found run data file: {file_path}")
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                run_record = _parse_run_data(data, str(file_path))
                if run_record:
                    runs.append(run_record)
                    logger.info(f"Successfully parsed run: {run_record.run_id}")
                else:
                    logger.warning(f"Failed to parse run from {file_path}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {file_path}: {e}")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
    
    logger.info(f"Loaded {len(runs)} runs from {runs_dir}")
    return runs


def load_run_file(file_path: str) -> Optional[RunRecord]:
    """Load exactly one run JSON file into a RunRecord."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return _parse_run_data(data, file_path)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def available_metrics(run: RunRecord) -> Set[str]:
    """
    Get the set of all available metrics across all layers in a run.
    
    Args:
        run: RunRecord to analyze
        
    Returns:
        Set of all metric names present in the run
    """
    metrics = set()
    for layer in run.layers:
        metrics.update(layer["metrics"].keys())
    return metrics


def count_pairs(run: RunRecord, eps: float = 0.003) -> int:
    """
    Count the number of valid non-tie pairs across all layers in a run.
    
    A pair is considered valid if both layers have the same metrics available,
    and non-tie if the difference in target ECE is greater than eps.
    
    Args:
        run: RunRecord to analyze
        eps: Minimum difference threshold for non-tie pairs
        
    Returns:
        Number of valid non-tie pairs
    """
    if len(run.layers) < 2:
        return 0
    
    # Find common metrics across all layers
    if not run.layers:
        return 0
    
    common_metrics = set(run.layers[0]["metrics"].keys())
    for layer in run.layers[1:]:
        common_metrics &= set(layer["metrics"].keys())
    
    if not common_metrics:
        return 0
    
    # Count valid non-tie pairs
    valid_pairs = 0
    for i in range(len(run.layers)):
        for j in range(i + 1, len(run.layers)):
            layer_i = run.layers[i]
            layer_j = run.layers[j]
            
            # Check if both layers have all common metrics
            if (all(metric in layer_i["metrics"] for metric in common_metrics) and
                all(metric in layer_j["metrics"] for metric in common_metrics)):
                
                # Check if it's a non-tie pair
                ece_diff = abs(layer_i["target_ece"] - layer_j["target_ece"])
                if ece_diff > eps:
                    valid_pairs += 1
    
    return valid_pairs
