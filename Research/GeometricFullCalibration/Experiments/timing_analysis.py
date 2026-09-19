#!/usr/bin/env python3
"""
Timing analysis for calibration experiments.

Reads JSON files from `calibration_comparison/` and extracts timing metrics
for multiple calibration methods. Produces:

1. Raw CSV with per-run timing (per seed, per config, per method)
2. Aggregated CSV/LaTeX tables:
   - Table A: Offline vs Online Time Comparison
   - Table B: Offline Time Breakdown
   - Table C: Speed vs Accuracy Trade-off
3. JSON with aggregated timing statistics
4. Markdown summary for quick inspection
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class SimpleLogger:
    def info(self, msg: str) -> None:
        print(f"[INFO] {msg}")

    def warning(self, msg: str) -> None:
        print(f"[WARNING] {msg}")

    def debug(self, msg: str) -> None:
        # Intentionally quiet by default; enable/replace with a real logger if needed.
        # We keep this method to avoid crashing when debug messages are emitted.
        pass


logger = SimpleLogger()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_SET_SIZES: Dict[str, int] = {
    "cifar10": 10000,
    "cifar100": 10000,
    "svhn": 26032,
    "tiny_imagenet": 10000,
    "tinyimagenet": 10000,  # Alternative naming variation
    "tiny-imagenet": 10000,  # Hyphenated variation
}

# Mapping from display name to JSON key path
# NOTE: Keys are dot-paths into the JSON, traversed by `_traverse_path`.
METHOD_CONFIG: Dict[str, str] = {
    # --- Methods requested for timing comparison (display name -> JSON key path) ---
    # RGCL: SGC global random (separation variant)
    "RGCL (SGC global random)": "global_random_separation",
    # SGC variants with layer selection / preprocessing
    "SGC + DAC layers": "sgc_with_dac_layers_separation",
    "SGC + DAC preprocessing": "ablation.sgc_with_dac_preprocessing_separation",
    "SGC + TULIP": "sgc_with_tulip_layers_separation",
    # DAC baselines
    # NOTE: "DAC_Orig" is the original DAC baseline timing node we want, which can be
    # missing/skipped in methods-only JSONs. With --dac-input-dir, we source it from that directory.
    "DAC_Orig": "dac_orig",
    "DAC random layers": "ablation.dac_with_random_layer_selection",
    # RGCC: coordinate sampling (separation variant)
    "RGCC (coordinate sampling)": "coordinate_sampling_separation",
    # --- FAISS variants ---
    "RGCL (FAISS)": "sgc_faiss",
    "RGCC (FAISS)": "coordinate_sampling_faiss",
    "Coord SPP (FAISS)": "coordinate_spp_faiss",
    # Geometric pixel calibration (physical space)
    "Geometric Pixel": "standard_baselines.geometric_physical_space",
}

# Timing field mappings by JSON key path
TIMING_FIELDS: Dict[str, Dict[str, Optional[str]]] = {
    "dac_standalone": {
        "extract": "extract_time_s",
        "plan": None,
        "fit": "fit_time_s",
        "calibrate": "calibrate_time_s",
        "throughput": "throughput_samples_per_sec",
    },
    # Original DAC baseline (dac_orig method).
    # Uses the standardized timing keys for extraction/fit/calibration.
    "dac_orig": {
        "extract": "extraction_time_s",
        "plan": None,
        "fit": "fit_time_s",
        "calibrate": "calibrate_time_s",
        "throughput": "throughput_samples_per_sec",
    },
    # Legacy: Original DAC baseline under standard baselines (for backward compatibility).
    "standard_baselines.density_aware_calibration": {
        "extract": "extraction_time_s",
        "plan": None,
        "fit": "fit_time_s",
        "calibrate": "calibrate_time_s",
        "throughput": "throughput_samples_per_sec",
    },
    # RGCL (SGC global random): feature extraction + fit + calibrate
    "global_random_separation": {
        "extract": "extraction_time_s",
        "plan": None,
        "fit": "fit_time_s",
        "calibrate": "calibrate_time_s",
        "throughput": "throughput_samples_per_sec",
    },
    # SGC variants (all use the same timing schema in our outputs)
    "sgc_with_dac_layers_separation": {
        "extract": "extraction_time_s",
        "plan": None,
        "fit": "fit_time_s",
        "calibrate": "calibrate_time_s",
        "throughput": "throughput_samples_per_sec",
    },
    "ablation.sgc_with_dac_preprocessing_separation": {
        "extract": "extraction_time_s",
        "plan": None,
        "fit": "fit_time_s",
        "calibrate": "calibrate_time_s",
        "throughput": "throughput_samples_per_sec",
    },
    "sgc_with_tulip_layers_separation": {
        "extract": "extraction_time_s",
        "plan": None,
        "fit": "fit_time_s",
        "calibrate": "calibrate_time_s",
        "throughput": "throughput_samples_per_sec",
    },
    # RGCC (coordinate sampling): includes planning step in addition to extraction
    "coordinate_sampling_separation": {
        "extract": "extraction_time_s",
        "plan": "plan_time_s",
        "fit": "fit_time_s",
        "calibrate": "calibrate_time_s",
        "throughput": "throughput_samples_per_sec",
    },
    # DAC random layers ablation (DAC schema)
    "ablation.dac_with_random_layer_selection": {
        "extract": "extract_time_s",
        "plan": None,
        "fit": "fit_time_s",
        "calibrate": "calibrate_time_s",
        "throughput": "throughput_samples_per_sec",
    },
    # --- FAISS variants ---
    "sgc_faiss": {
        "extract": "extraction_time_s",
        "plan": None,
        "fit": "fit_time_s",
        "calibrate": "calibrate_time_s",
        "throughput": "throughput_samples_per_sec",
    },
    "coordinate_sampling_faiss": {
        "extract": "extraction_time_s",
        "plan": "plan_time_s",
        "fit": "fit_time_s",
        "calibrate": "calibrate_time_s",
        "throughput": "throughput_samples_per_sec",
    },
    "coordinate_spp_faiss": {
        "extract": "extraction_time_s",
        "plan": "plan_time_s",
        "fit": "fit_time_s",
        "calibrate": "calibrate_time_s",
        "throughput": "throughput_samples_per_sec",
    },
    # Geometric pixel calibration (physical space)
    "standard_baselines.geometric_physical_space": {
        "extract": "extraction_time_s",
        "plan": None,
        "fit": "fit_time_s",
        "calibrate": "calibrate_time_s",
        "throughput": "throughput_samples_per_sec",
    },
    # Alternative path (if not under standard_baselines)
    "geometric_physical_space": {
        "extract": "extraction_time_s",
        "plan": None,
        "fit": "fit_time_s",
        "calibrate": "calibrate_time_s",
        "throughput": "throughput_samples_per_sec",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_dataset_name(dataset: str) -> str:
    """
    Normalize dataset name to handle variations (e.g., tiny_imagenet, tinyimagenet, tiny-imagenet).
    Returns the canonical form for lookup in TEST_SET_SIZES.
    """
    if not dataset:
        return dataset
    normalized = str(dataset).lower().replace('-', '_').replace(' ', '_')
    # Map common variations to canonical form
    if normalized in ['tiny_imagenet', 'tinyimagenet']:
        return 'tiny_imagenet'
    return normalized


def _traverse_path(json_data: Dict[str, Any], path: str) -> Optional[Any]:
    current: Any = json_data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def compute_online_ms_per_sample(calibrate_time_s: Optional[float], n_test: int) -> Optional[float]:
    """Convert calibrate_time (s) to milliseconds per sample."""
    if calibrate_time_s is None or calibrate_time_s <= 0 or n_test <= 0:
        return None
    return float(calibrate_time_s / n_test * 1000.0)


def compute_throughput_from_time(calibrate_time_s: Optional[float], n_test: int) -> Optional[float]:
    """Compute throughput if not provided."""
    if calibrate_time_s is None or calibrate_time_s <= 0 or n_test <= 0:
        return None
    return float(n_test / calibrate_time_s)


def extract_timing_for_node(node: Any, field_map: Dict[str, Optional[str]]) -> Dict[str, Optional[float]]:
    """
    Extract raw timing fields from a node.

    node can be:
      - dict with timing keys directly
      - dict of dicts (e.g. geometric_with_dac_features per layer)
    Returns dict with keys: extract, plan, fit, calibrate, throughput

    Timing definitions (as used for outputs/tables):
      - feature extraction time: time to compute method-specific features from logits/features (offline)
      - calibration time: time to apply the fitted calibrator over the test set (online)
      - end-to-end time: offline + online (feature extraction + optional planning + fitting + calibration)

    IMPORTANT: Values that are exactly 0.0 are treated as missing (None).
    Rationale: some runs can record cached/no-op timings as 0.0, which breaks log-scale plots and can
    misleadingly imply "free" computation. We treat only exactly-zero as missing; positive values remain.
    """
    def zero_to_none(x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        return None if x == 0.0 else x

    def read_from_single(d: Dict[str, Any]) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {
            "extract": None,
            "plan": None,
            "fit": None,
            "calibrate": None,
            "throughput": None,
        }
        for key in ("extract", "plan", "fit", "calibrate", "throughput"):
            field_name = field_map.get(key)
            if field_name and isinstance(d, dict) and field_name in d and d[field_name] is not None:
                try:
                    out[key] = zero_to_none(float(d[field_name]))
                except (TypeError, ValueError):
                    out[key] = None
        return out

    # Direct dict
    # NOTE: Guard dict membership checks with isinstance(node, dict) to avoid operator-precedence bugs.
    if isinstance(node, dict) and (
        ("ece" in node)
        or ("method" in node)
        or any(k in node for k in ("extract_time_s", "extraction_time_s", "fit_time_s", "calibrate_time_s"))
    ):
        return read_from_single(node)

    # Dict of dicts (e.g., per-layer results)
    if isinstance(node, dict):
        accum: Dict[str, List[float]] = {
            "extract": [],
            "plan": [],
            "fit": [],
            "calibrate": [],
            "throughput": [],
        }
        for v in node.values():
            if not isinstance(v, dict):
                continue
            single = read_from_single(v)
            for k, val in single.items():
                if val is not None:
                    accum[k].append(val)
        out2: Dict[str, Optional[float]] = {}
        for k, vals in accum.items():
            out2[k] = float(np.mean(vals)) if vals else None
        return out2

    # Fallback: nothing
    return {"extract": None, "plan": None, "fit": None, "calibrate": None, "throughput": None}


def extract_ece_for_method(json_data: Dict[str, Any], json_key: str) -> Optional[float]:
    """Extract ECE for a method (direct dict or nested under ablation.*)."""
    node = _traverse_path(json_data, json_key)
    if node is None:
        return None
    if isinstance(node, dict) and node.get("skipped", False):
        return None

    # direct
    if isinstance(node, dict) and "ece" in node:
        try:
            return float(node["ece"])
        except (TypeError, ValueError):
            return None

    # dict of dicts: take min ECE
    if isinstance(node, dict):
        eces: List[float] = []
        for v in node.values():
            if isinstance(v, dict) and "ece" in v:
                try:
                    eces.append(float(v["ece"]))
                except (TypeError, ValueError):
                    continue
        if eces:
            return float(min(eces))

    return None


def extract_timing_for_method(
    json_data: Dict[str, Any],
    display_name: str,
    dataset: str,
) -> Optional[Dict[str, Any]]:
    """
    Extract timing metrics for a method.

    Returns dict with:
      - offline_time_s
      - online_time_s
      - total_time_s
      - throughput
      - breakdown: {extract, plan, fit, calibrate}
      - ece
    or None if method not present.
    """
    json_key = METHOD_CONFIG[display_name]
    field_map = TIMING_FIELDS.get(json_key)
    if field_map is None:
        logger.warning(f"No timing field mapping for method key '{json_key}'")
        return None

    node = _traverse_path(json_data, json_key)
    # For geometric_physical_space, try alternative path if primary path fails
    if node is None and json_key == "standard_baselines.geometric_physical_space":
        alt_key = "geometric_physical_space"
        alt_field_map = TIMING_FIELDS.get(alt_key)
        if alt_field_map is not None:
            node = _traverse_path(json_data, alt_key)
            if node is not None:
                field_map = alt_field_map
                json_key = alt_key
    
    if node is None or (isinstance(node, dict) and node.get("skipped", False)):
        return None

    raw = extract_timing_for_node(node, field_map)

    extract_s = raw["extract"]
    plan_s = raw["plan"]
    fit_s = raw["fit"]
    calibrate_s = raw["calibrate"]
    throughput = raw["throughput"]

    normalized_dataset = _normalize_dataset_name(dataset)
    n_test = TEST_SET_SIZES.get(normalized_dataset, 10000)

    # Offline time = feature extraction (+ optional planning) + fit
    offline_parts = [extract_s, plan_s, fit_s]
    offline_time_s = sum(v for v in offline_parts if v is not None) if any(v is not None for v in offline_parts) else None
    # Online time = calibration application over test set
    online_time_s = calibrate_s
    # End-to-end time = offline + online (only if at least one component is present)
    total_time_s = (
        (offline_time_s or 0.0) + (online_time_s or 0.0)
        if (offline_time_s is not None or online_time_s is not None)
        else None
    )

    if throughput is None:
        throughput = compute_throughput_from_time(calibrate_s, n_test)

    online_ms_per_sample = compute_online_ms_per_sample(calibrate_s, n_test)
    ece = extract_ece_for_method(json_data, json_key)

    return {
        "method": display_name,
        "offline_time_s": offline_time_s,
        "online_time_s": online_time_s,
        "total_time_s": total_time_s,
        "throughput": throughput,
        "online_ms_per_sample": online_ms_per_sample,
        "breakdown": {
            "extract_s": extract_s,
            "plan_s": plan_s,
            "fit_s": fit_s,
            "calibrate_s": calibrate_s,
        },
        "ece": ece,
        "n_test": n_test,
    }


# ---------------------------------------------------------------------------
# Outlier Removal
# ---------------------------------------------------------------------------

def remove_timing_outliers(
    df: pd.DataFrame,
    time_cols: List[str] = None,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Remove runs where timing exceeds threshold * median for that config group.
    
    Filters outliers per (dataset, model, method) group. A run is removed if
    ANY of the specified timing columns exceeds the threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw timing DataFrame with columns: dataset, model, method, and timing columns
    time_cols : List[str], optional
        List of timing column names to check. If None, defaults to:
        ['offline_s', 'online_s', 'total_s']
    threshold : float
        Multiplier for median (default: 1.5). Runs exceeding median * threshold are removed.
    
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame with outliers removed
    """
    if df.empty:
        return df
    
    if time_cols is None:
        # Check both online_s and online_ms since online_ms is what's plotted/aggregated
        # and can have outliers even when online_s doesn't exceed threshold
        time_cols = ['offline_s', 'online_s', 'online_ms', 'total_s']
    
    # Only use columns that exist in the DataFrame
    available_cols = [col for col in time_cols if col in df.columns]
    if not available_cols:
        logger.warning("No timing columns found for outlier removal")
        return df
    
    initial_count = len(df)
    
    def filter_group(group: pd.DataFrame) -> pd.DataFrame:
        """Filter outliers within a single (dataset, model, method) group."""
        if len(group) <= 1:
            # Need at least 2 points to compute median meaningfully
            return group
        
        mask = pd.Series(True, index=group.index)
        
        for col in available_cols:
            if col not in group.columns:
                continue
            
            # Get non-NaN values for this column
            col_vals = group[col].dropna()
            n_vals = len(col_vals)
            
            if n_vals < 4:
                # For small sample sizes, use a more aggressive approach
                # With N < 4, IQR is unreliable, so use a different method
                if n_vals <= 2:
                    # With only 2 samples, don't remove outliers (need at least 2 for stats)
                    continue
                else:
                    # N=3: Check if max value is outlier compared to the other two
                    sorted_vals = np.sort(col_vals.values)
                    # If max is more than 1.3x the median of the two smaller values, remove it
                    median_of_two = np.median(sorted_vals[:2])
                    max_val = sorted_vals[-1]
                    if median_of_two > 0 and max_val > median_of_two * 1.3:
                        # Remove the max value (keep values <= max_val, but actually we want < max_val)
                        col_mask = (group[col].isna()) | (group[col] < max_val)
                    else:
                        # No clear outlier, keep all (including NaN)
                        col_mask = pd.Series(True, index=group.index)
            else:
                # Use IQR (Interquartile Range) method for better outlier detection
                # This is more robust than simple threshold * median
                Q1 = col_vals.quantile(0.25)
                Q3 = col_vals.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR <= 0:
                    # If IQR is 0 or negative, fall back to threshold method
                    median_val = col_vals.median()
                    if pd.isna(median_val) or median_val <= 0:
                        continue
                    col_mask = (group[col].isna()) | (group[col] <= median_val * threshold)
                else:
                    # Outliers are values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
                    # But we only remove upper outliers (too high), not lower ones
                    # For small N (4-5), use slightly stricter threshold
                    if n_vals <= 5:
                        effective_threshold = min(threshold, 1.3)
                    else:
                        effective_threshold = threshold
                    upper_bound = Q3 + effective_threshold * IQR
                    col_mask = (group[col].isna()) | (group[col] <= upper_bound)
            
            mask = mask & col_mask
        
        return group[mask]
    
    df_filtered = df.groupby(['dataset', 'model', 'method'], group_keys=False).apply(filter_group)
    
    removed_count = initial_count - len(df_filtered)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} outlier runs (IQR method, threshold={threshold}x)")
    else:
        logger.info(f"No outliers removed (IQR method, threshold={threshold}x)")
    
    return df_filtered


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

ConfigKey = Tuple[Optional[str], Optional[str], Optional[str], Any, Optional[str], Any]


def _config_key_from_experiment_config(cfg: Dict[str, Any]) -> ConfigKey:
    """
    Build a stable matching key for runs across directories.

    Key:
      (training_method, dataset, model_name, seed, corruption_type, corruption_severity)
    """
    return (
        cfg.get("training_method", "unknown"),
        cfg.get("dataset", "unknown"),
        cfg.get("model_name", "unknown"),
        cfg.get("seed", None),
        cfg.get("corruption_type", None),
        cfg.get("corruption_severity", None),
    )


def build_dac_index(dac_directory: Path) -> Dict[ConfigKey, Dict[str, Any]]:
    """
    Index DAC-original baseline JSONs by config tuple, so we can pull DAC timings even when the main
    --input-dir JSONs were generated in "methods-only" mode and have standard baselines skipped.

    We only index files where:
      - experiment_config exists
      - dac_orig exists (or legacy standard_baselines.density_aware_calibration for backward compatibility)
      - that node is not marked {"skipped": true}
    """
    index: Dict[ConfigKey, Dict[str, Any]] = {}
    json_files = sorted(dac_directory.glob("*.json"))
    logger.info(f"[DAC index] Scanning {len(json_files)} JSON files in {dac_directory}")

    indexed = 0
    skipped = 0
    for path in json_files:
        try:
            with path.open("r") as f:
                j = json.load(f)
        except Exception as e:
            logger.warning(f"[DAC index] Failed to read {path.name}: {e}")
            continue

        cfg = j.get("experiment_config", {})
        key = _config_key_from_experiment_config(cfg)

        # Check new dac_orig path first, then fall back to legacy path for backward compatibility
        node = _traverse_path(j, "dac_orig")
        if node is None:
            node = _traverse_path(j, "standard_baselines.density_aware_calibration")
        
        if node is None or (isinstance(node, dict) and node.get("skipped", False)):
            skipped += 1
            continue

        index[key] = j  # store full JSON so we can reuse existing extraction logic (timing + ECE)
        indexed += 1

    logger.info(f"[DAC index] Indexed {indexed} DAC files (skipped {skipped} without usable DAC node)")
    return index


def build_geometric_pixel_index(geometric_pixel_directory: Path) -> Dict[ConfigKey, Dict[str, Any]]:
    """
    Index geometric pixel calibration JSONs by config tuple, so we can pull geometric pixel timings
    even when the main --input-dir JSONs were generated in "methods-only" mode and have standard baselines skipped.

    We only index files where:
      - experiment_config exists
      - geometric_physical_space exists (under standard_baselines or directly)
      - that node is not marked {"skipped": true}
    """
    index: Dict[ConfigKey, Dict[str, Any]] = {}
    json_files = sorted(geometric_pixel_directory.glob("*.json"))
    logger.info(f"[Geometric Pixel index] Scanning {len(json_files)} JSON files in {geometric_pixel_directory}")

    indexed = 0
    skipped = 0
    for path in json_files:
        try:
            with path.open("r") as f:
                j = json.load(f)
        except Exception as e:
            logger.warning(f"[Geometric Pixel index] Failed to read {path.name}: {e}")
            continue

        cfg = j.get("experiment_config", {})
        key = _config_key_from_experiment_config(cfg)

        # Check standard_baselines.geometric_physical_space first, then fall back to geometric_physical_space
        node = _traverse_path(j, "standard_baselines.geometric_physical_space")
        if node is None:
            node = _traverse_path(j, "geometric_physical_space")
        
        if node is None or (isinstance(node, dict) and node.get("skipped", False)):
            skipped += 1
            continue

        index[key] = j  # store full JSON so we can reuse existing extraction logic (timing + ECE)
        indexed += 1

    logger.info(f"[Geometric Pixel index] Indexed {indexed} Geometric Pixel files (skipped {skipped} without usable Geometric Pixel node)")
    return index


def collect_timing_from_directory(
    directory: Path, 
    dac_index: Optional[Dict[ConfigKey, Dict[str, Any]]] = None,
    geometric_pixel_index: Optional[Dict[ConfigKey, Dict[str, Any]]] = None
) -> pd.DataFrame:
    """
    Scan all JSONs, extract timing+ECE per method, and return raw DataFrame.
    Columns: training, dataset, model, seed, file, method, offline_s, online_s,
    total_s, throughput, online_ms, extract_s, plan_s, fit_s, calibrate_s, ece.
    """
    rows: List[Dict[str, Any]] = []

    json_files = sorted(directory.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {directory}")

    dac_matches = 0
    dac_misses = 0
    geo_pixel_matches = 0
    geo_pixel_misses = 0

    for idx, path in enumerate(json_files, 1):
        try:
            with path.open("r") as f:
                j = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {path.name}: {e}")
            continue

        cfg = j.get("experiment_config", {})
        # Skip files without experiment_config or with empty/invalid config
        if not cfg or not isinstance(cfg, dict):
            logger.debug(f"Skipping {path.name}: missing or invalid experiment_config")
            continue
        
        training = cfg.get("training_method", "unknown")
        dataset = cfg.get("dataset", "unknown")
        model = cfg.get("model_name", "unknown")
        seed = cfg.get("seed", None)
        cfg_key = _config_key_from_experiment_config(cfg)
        
        # Skip files where all key fields are unknown/missing (likely not a valid experiment file)
        if training == "unknown" and dataset == "unknown" and model == "unknown":
            logger.debug(f"Skipping {path.name}: experiment_config has no recognizable fields")
            continue

        for display in METHOD_CONFIG.keys():
            # If a DAC index was provided, source DAC-original timings from that directory instead
            # of the current JSON (which may have standard baselines skipped).
            if display == "DAC_Orig" and dac_index is not None:
                dac_json = dac_index.get(cfg_key)
                if dac_json is None:
                    # Try fallback: use any available seed for same dataset/model/training
                    available_keys = [k for k in dac_index.keys() 
                                   if k[1] == dataset and k[2] == model and k[0] == training]
                    if available_keys:
                        # Use the first available seed as fallback
                        fallback_key = available_keys[0]
                        dac_json = dac_index[fallback_key]
                        available_seeds = [str(k[3]) for k in available_keys if k[3] is not None]
                        logger.info(
                            f"[DAC index] No exact match for DAC_Orig: training={training}, dataset={dataset}, model={model}, seed={seed}. "
                            f"Using fallback seed={fallback_key[3]} (available seeds: {', '.join(available_seeds[:10])})"
                        )
                        dac_matches += 1
                        timing = extract_timing_for_method(dac_json, display, dataset)
                    else:
                        dac_misses += 1
                        logger.warning(
                            f"[DAC index] No match for DAC_Orig: training={training}, dataset={dataset}, model={model}, seed={seed} "
                            f"(corruption_type={cfg.get('corruption_type')}, severity={cfg.get('corruption_severity')}). "
                            f"No entries found for this dataset/model combination in DAC index."
                        )
                        continue
                else:
                    dac_matches += 1
                    timing = extract_timing_for_method(dac_json, display, dataset)
            # If a Geometric Pixel index was provided, source Geometric Pixel timings from that directory
            elif display == "Geometric Pixel" and geometric_pixel_index is not None:
                geo_pixel_json = geometric_pixel_index.get(cfg_key)
                if geo_pixel_json is None:
                    # Try fallback: use any available seed for same dataset/model/training
                    available_keys = [k for k in geometric_pixel_index.keys() 
                                   if k[1] == dataset and k[2] == model and k[0] == training]
                    if available_keys:
                        # Use the first available seed as fallback
                        fallback_key = available_keys[0]
                        geo_pixel_json = geometric_pixel_index[fallback_key]
                        available_seeds = [str(k[3]) for k in available_keys if k[3] is not None]
                        logger.info(
                            f"[Geometric Pixel index] No exact match for Geometric Pixel: training={training}, dataset={dataset}, model={model}, seed={seed}. "
                            f"Using fallback seed={fallback_key[3]} (available seeds: {', '.join(available_seeds[:10])})"
                        )
                        geo_pixel_matches += 1
                        timing = extract_timing_for_method(geo_pixel_json, display, dataset)
                    else:
                        geo_pixel_misses += 1
                        logger.warning(
                            f"[Geometric Pixel index] No match for Geometric Pixel: training={training}, dataset={dataset}, model={model}, seed={seed} "
                            f"(corruption_type={cfg.get('corruption_type')}, severity={cfg.get('corruption_severity')}). "
                            f"No entries found for this dataset/model combination in Geometric Pixel index."
                        )
                        continue
                else:
                    geo_pixel_matches += 1
                    timing = extract_timing_for_method(geo_pixel_json, display, dataset)
            else:
                timing = extract_timing_for_method(j, display, dataset)

            if timing is None:
                continue

            br = timing["breakdown"]
            row: Dict[str, Any] = {
                "training": training,
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "file": path.name,
                "method": display,
                "offline_s": timing["offline_time_s"],
                "online_s": timing["online_time_s"],
                "total_s": timing["total_time_s"],
                "throughput": timing["throughput"],
                "online_ms": timing["online_ms_per_sample"],
                "extract_s": br["extract_s"],
                "plan_s": br["plan_s"],
                "fit_s": br["fit_s"],
                "calibrate_s": br["calibrate_s"],
                "ece": timing["ece"],
                "n_test": timing["n_test"],
            }
            rows.append(row)

        if idx % 25 == 0:
            logger.info(f"Processed {idx}/{len(json_files)} files")

    if dac_index is not None:
        logger.info(f"[DAC index] DAC_Orig matches: {dac_matches}, misses: {dac_misses}")
    if geometric_pixel_index is not None:
        logger.info(f"[Geometric Pixel index] Geometric Pixel matches: {geo_pixel_matches}, misses: {geo_pixel_misses}")

    df = pd.DataFrame(rows)
    logger.info(f"Extracted timing for {len(df)} method-runs")
    return df


def aggregate_timing_by_config(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate timing across seeds, grouped by (dataset, model, method).

    Returns DataFrame with:
      Dataset, Model, Method, N,
      offline_mean_s, offline_std_s,
      online_mean_ms, online_std_ms,
      throughput_mean, throughput_std,
      extract_mean_s, plan_mean_s, fit_mean_s, total_offline_mean_s,
      ece_mean, ece_std
    """
    if df_raw.empty:
        return pd.DataFrame()

    groups = df_raw.groupby(["dataset", "model", "method"], dropna=False)
    rows: List[Dict[str, Any]] = []

    for (dataset, model, method), g in groups:
        n = len(g)
        row: Dict[str, Any] = {
            "Dataset": dataset,
            "Model": model,
            "Method": method,
            "N": n,
        }

        def mean_std(col: str) -> Tuple[Optional[float], Optional[float]]:
            vals = g[col].dropna().values
            if len(vals) == 0:
                return None, None
            if len(vals) == 1:
                return float(vals[0]), 0.0
            return float(np.mean(vals)), float(np.std(vals, ddof=1))

        offline_mean, offline_std = mean_std("offline_s")
        online_mean_ms, online_std_ms = mean_std("online_ms")
        thr_mean, thr_std = mean_std("throughput")
        extract_mean, _ = mean_std("extract_s")
        plan_mean, _ = mean_std("plan_s")
        fit_mean, _ = mean_std("fit_s")
        total_mean, _ = mean_std("total_s")
        ece_mean, ece_std = mean_std("ece")

        row.update(
            {
                "offline_mean_s": offline_mean,
                "offline_std_s": offline_std,
                "online_mean_ms": online_mean_ms,
                "online_std_ms": online_std_ms,
                "throughput_mean": thr_mean,
                "throughput_std": thr_std,
                "extract_mean_s": extract_mean,
                "plan_mean_s": plan_mean,
                "fit_mean_s": fit_mean,
                "total_offline_mean_s": None
                if extract_mean is None and plan_mean is None and fit_mean is None
                else sum(
                    v or 0.0 for v in (extract_mean, plan_mean, fit_mean)
                ),
                "total_mean_s": total_mean,
                "ece_mean": ece_mean,
                "ece_std": ece_std,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def generate_offline_online_table(df_agg: pd.DataFrame, output_csv: Path, output_tex: Path) -> None:
    """Generate Table A: Offline vs Online Time Comparison."""
    if df_agg.empty:
        logger.warning("No aggregated data for Table A")
        return

    cols = [
        "Dataset",
        "Model",
        "Method",
        "N",
        "offline_mean_s",
        "online_mean_ms",
        "throughput_mean",
    ]
    df = df_agg[cols].copy()
    df.to_csv(output_csv, index=False)
    logger.info(f"Table A (offline vs online) CSV saved to {output_csv}")
    df.to_latex(output_tex, index=False, float_format="%.3f", escape=False)
    logger.info(f"Table A (offline vs online) LaTeX saved to {output_tex}")


def generate_breakdown_table(df_agg: pd.DataFrame, output_csv: Path, output_tex: Path) -> None:
    """Generate Table B: Offline time breakdown."""
    if df_agg.empty:
        logger.warning("No aggregated data for Table B")
        return

    cols = [
        "Dataset",
        "Model",
        "Method",
        "N",
        "extract_mean_s",
        "plan_mean_s",
        "fit_mean_s",
        "total_offline_mean_s",
    ]
    df = df_agg[cols].copy()
    df.to_csv(output_csv, index=False)
    logger.info(f"Table B (offline breakdown) CSV saved to {output_csv}")
    df.to_latex(output_tex, index=False, float_format="%.3f", escape=False)
    logger.info(f"Table B (offline breakdown) LaTeX saved to {output_tex}")


def generate_tradeoff_table(df_agg: pd.DataFrame, output_csv: Path, output_tex: Path) -> None:
    """
    Generate Table C: Speed vs Accuracy trade-off.

    For each (Dataset, Model, Method):
      - ECE (%)
      - Throughput (samples/s)
      - Relative speed vs DAC baseline within same Dataset+Model
    """
    if df_agg.empty:
        logger.warning("No aggregated data for Table C")
        return

    df = df_agg.copy()

    # Convert ECE to %
    df["ece_pct"] = df["ece_mean"] * 100.0

    # Compute relative speed vs DAC per dataset+model
    rel_speeds: List[Optional[float]] = []
    for _, row in df.iterrows():
        ds = row["Dataset"]
        md = row["Model"]
        thr = row["throughput_mean"]
        if thr is None or not np.isfinite(thr):
            rel_speeds.append(None)
            continue
        # Find DAC throughput for same dataset+model
        mask = (df["Dataset"] == ds) & (df["Model"] == md) & (df["Method"] == "DAC")
        dac_rows = df[mask]
        if dac_rows.empty or dac_rows["throughput_mean"].isna().all():
            rel_speeds.append(None)
            continue
        dac_thr = float(dac_rows["throughput_mean"].iloc[0])
        if dac_thr <= 0:
            rel_speeds.append(None)
        else:
            rel_speeds.append(float(thr / dac_thr))

    df["relative_speed_vs_DAC"] = rel_speeds

    cols = [
        "Dataset",
        "Model",
        "Method",
        "N",
        "ece_pct",
        "throughput_mean",
        "relative_speed_vs_DAC",
    ]
    table = df[cols].copy()
    table.to_csv(output_csv, index=False)
    logger.info(f"Table C (speed vs accuracy) CSV saved to {output_csv}")
    table.to_latex(output_tex, index=False, float_format="%.3f", escape=False)
    logger.info(f"Table C (speed vs accuracy) LaTeX saved to {output_tex}")


def save_aggregated_json(df_agg: pd.DataFrame, output_json: Path) -> None:
    """Save aggregated timing stats as JSON."""
    data: Dict[str, Any] = defaultdict(dict)
    for _, row in df_agg.iterrows():
        ds = row["Dataset"]
        md = row["Model"]
        mt = row["Method"]
        key = f"{ds}__{md}"
        if key not in data:
            data[key] = {"Dataset": ds, "Model": md, "methods": {}}
        data[key]["methods"][mt] = {
            "N": int(row["N"]),
            "offline_mean_s": row["offline_mean_s"],
            "offline_std_s": row["offline_std_s"],
            "online_mean_ms": row["online_mean_ms"],
            "online_std_ms": row["online_std_ms"],
            "throughput_mean": row["throughput_mean"],
            "throughput_std": row["throughput_std"],
            "extract_mean_s": row["extract_mean_s"],
            "plan_mean_s": row["plan_mean_s"],
            "fit_mean_s": row["fit_mean_s"],
            "total_offline_mean_s": row["total_offline_mean_s"],
            "total_mean_s": row["total_mean_s"],
            "ece_mean": row["ece_mean"],
            "ece_std": row["ece_std"],
        }
    with output_json.open("w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Aggregated timing JSON saved to {output_json}")


def save_markdown_summary(df_agg: pd.DataFrame, output_md: Path) -> None:
    """Write a short Markdown summary of timing vs accuracy by method."""
    if df_agg.empty:
        logger.warning("No aggregated data for Markdown summary")
        return

    # Aggregate over all datasets/models per method
    groups = df_agg.groupby("Method")
    lines: List[str] = []
    lines.append("# Timing Summary\n")
    lines.append("| Method | Offline (s, mean) | Online (ms, mean) | Throughput (samples/s) | ECE (%) |")
    lines.append("|--------|-------------------|-------------------|------------------------|--------|")

    for method, g in groups:
        def mean_ignore(col: str) -> Optional[float]:
            vals = g[col].dropna().values
            if len(vals) == 0:
                return None
            return float(np.mean(vals))

        off = mean_ignore("offline_mean_s")
        on = mean_ignore("online_mean_ms")
        thr = mean_ignore("throughput_mean")
        ece = mean_ignore("ece_mean")

        def fmt(x: Optional[float], scale: float = 1.0, prec: int = 2) -> str:
            if x is None or not np.isfinite(x):
                return "N/A"
            return f"{x * scale:.{prec}f}"

        lines.append(
            f"| {method} | {fmt(off)} | {fmt(on)} | {fmt(thr, 1.0, 1)} | {fmt(ece, 100.0, 2)} |"
        )

    output_md.write_text("\n".join(lines))
    logger.info(f"Markdown summary saved to {output_md}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and analyze timing metrics from calibration JSON files.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing calibration JSONs (e.g. calibration_comparison/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="timing_analysis",
        help="Directory where timing outputs (CSV/TeX/JSON/MD) will be written.",
    )
    parser.add_argument(
        "--dac-input-dir",
        type=str,
        default=None,
        help=(
            "Optional directory of JSONs to source the DAC_Orig baseline "
            "(standard_baselines.density_aware_calibration). Useful when --input-dir "
            "JSONs were generated in methods-only mode with baselines skipped."
        ),
    )
    parser.add_argument(
        "--geometric-pixel-input-dir",
        type=str,
        default=None,
        help=(
            "Optional directory of JSONs to source the Geometric Pixel baseline "
            "(standard_baselines.geometric_physical_space). Useful when --input-dir "
            "JSONs were generated in methods-only mode with baselines skipped."
        ),
    )
    parser.add_argument(
        "--remove-outliers",
        action="store_true",
        default=True,
        help="Remove timing outliers (>1.5x median per config group). Recommended for paper. (default: True)",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=1.5,
        help="Outlier threshold multiplier (default: 1.5). Runs exceeding median * threshold are removed.",
    )
    parser.add_argument(
        "--no-remove-outliers",
        dest="remove_outliers",
        action="store_false",
        help="Disable outlier removal (keep all runs).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dac_index: Optional[Dict[ConfigKey, Dict[str, Any]]] = None
    if args.dac_input_dir:
        dac_dir = Path(args.dac_input_dir)
        if not dac_dir.exists():
            raise SystemExit(f"DAC input directory does not exist: {dac_dir}")
        dac_index = build_dac_index(dac_dir)

    geometric_pixel_index: Optional[Dict[ConfigKey, Dict[str, Any]]] = None
    if args.geometric_pixel_input_dir:
        geo_pixel_dir = Path(args.geometric_pixel_input_dir)
        if not geo_pixel_dir.exists():
            raise SystemExit(f"Geometric Pixel input directory does not exist: {geo_pixel_dir}")
        geometric_pixel_index = build_geometric_pixel_index(geo_pixel_dir)

    # 1) Raw timing extraction
    df_raw = collect_timing_from_directory(input_dir, dac_index=dac_index, geometric_pixel_index=geometric_pixel_index)
    raw_csv = out_dir / "timing_raw.csv"
    df_raw.to_csv(raw_csv, index=False)
    logger.info(f"Raw timing CSV saved to {raw_csv}")

    # 2) Outlier removal (recommended for paper)
    if args.remove_outliers:
        logger.info(f"Removing timing outliers (threshold={args.outlier_threshold}x median)")
        df_clean = remove_timing_outliers(df_raw, threshold=args.outlier_threshold)
        clean_csv = out_dir / "timing_raw_cleaned.csv"
        df_clean.to_csv(clean_csv, index=False)
        logger.info(f"Cleaned timing CSV saved to {clean_csv}")
        df_for_agg = df_clean
    else:
        logger.info("Outlier removal disabled (keeping all runs)")
        df_for_agg = df_raw

    # 3) Aggregation by (dataset, model, method)
    df_agg = aggregate_timing_by_config(df_for_agg)
    agg_csv = out_dir / "timing_aggregated.csv"
    df_agg.to_csv(agg_csv, index=False)
    logger.info(f"Aggregated timing CSV saved to {agg_csv}")

    # 3) Tables
    generate_offline_online_table(
        df_agg,
        output_csv=out_dir / "tableA_offline_online.csv",
        output_tex=out_dir / "tableA_offline_online.tex",
    )
    generate_breakdown_table(
        df_agg,
        output_csv=out_dir / "tableB_offline_breakdown.csv",
        output_tex=out_dir / "tableB_offline_breakdown.tex",
    )
    generate_tradeoff_table(
        df_agg,
        output_csv=out_dir / "tableC_speed_accuracy.csv",
        output_tex=out_dir / "tableC_speed_accuracy.tex",
    )

    # 4) JSON + Markdown summary
    save_aggregated_json(df_agg, out_dir / "timing_aggregated.json")
    save_markdown_summary(df_agg, out_dir / "timing_summary.md")


if __name__ == "__main__":
    main()




