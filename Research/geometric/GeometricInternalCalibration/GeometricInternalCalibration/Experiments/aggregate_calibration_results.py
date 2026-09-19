#!/usr/bin/env python3
"""
Aggregate calibration experiment results from JSON files.

This module is the single source of truth for:
- Parsing result JSONs from `calibration_comparison/`
- Extracting ECE values for all methods (using a configurable METHOD registry)
- Computing basic statistics (mean, 95% CI)
- Producing a "raw" CSV for downstream analysis

Other scripts (`generate_paper_tables.py`, `statistical_analysis.py`) import
helpers from here instead of re‑implementing parsing/aggregation logic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd


class SimpleLogger:
    def info(self, msg: str) -> None:
        print(f"[INFO] {msg}")

    def warning(self, msg: str) -> None:
        print(f"[WARNING] {msg}")


logger = SimpleLogger()


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodSpec:
    code: str          # short internal code
    path: str          # JSON key path (with dots for nesting)
    mtype: str         # 'direct' or 'min_dict'
    display: str       # column/display name


METHODS: Dict[str, MethodSpec] = {
    # Main / baseline methods
    "uncal": MethodSpec("uncal", "uncalibrated", "direct", "Uncal"),
    "ts": MethodSpec("ts", "baseline_ts", "direct", "TS"),
    "dac": MethodSpec("dac", "dac_standalone", "direct", "DAC"),
    "ts_dac": MethodSpec("ts_dac", "ts_plus_dac", "direct", "TS+DAC"),
    "sgc": MethodSpec("sgc", "global_random", "direct", "SGC"),
    "geo_comb": MethodSpec("geo_comb", "geometric_concatenated", "direct", "GeoComb"),
    # Coordinate / feature variants
    "rand_coord": MethodSpec("rand_coord", "coordinate_sampling", "direct", "RandCoord"),
    "coord_spp": MethodSpec("coord_spp", "coordinate_spp", "direct", "CoordSPP"),
    "coord_spp_dac": MethodSpec("coord_spp_dac", "coordinate_spp_dac", "direct", "CoordSPP+DAC"),
    "dac_coord": MethodSpec("dac_coord", "dac_with_coordinate_features", "direct", "DAC+Coord"),
    # Geometric / DAC hybrids
    "geo_dac_w": MethodSpec("geo_dac_w", "geometric_dac_weighted", "direct", "GeoDAC-W"),
    "geo_dac_feat": MethodSpec(
        "geo_dac_feat",
        "ablation.geometric_with_dac_features",
        "min_dict",
        "Geo+DACFeat",
    ),
    "dac_rand_l": MethodSpec(
        "dac_rand_l",
        "ablation.dac_with_random_layer_selection",
        "direct",
        "DAC+RandL",
    ),
    "dac_sgc_feat": MethodSpec(
        "dac_sgc_feat",
        "ablation.dac_with_sgc_aggregated_features",
        "direct",
        "DAC+SGCFeat",
    ),
    # Normalization ablation
    "norm_post": MethodSpec(
        "norm_post",
        "normalization_ablation.geometric__l2_norm_post_agg",
        "direct",
        "NormPostAgg",
    ),
}


def _traverse_path(json_data: Dict[str, Any], path: str) -> Optional[Any]:
    """Traverse a dotted key path like 'ablation.geometric_with_dac_features'."""
    current: Any = json_data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def extract_metric(json_data: Dict[str, Any], spec: MethodSpec, metric_field: str = "ece") -> Optional[float]:
    """
    Extract a scalar metric from JSON data following a MethodSpec.

    Handles:
    - Direct keys with the requested metric field
    - Nested keys (a.b.c)
    - Dict/list results where we choose the minimum requested metric (min_dict)
    - Skipped sections (return None, not error)
    """
    node = _traverse_path(json_data, spec.path)
    if node is None:
        return None

    # Handle "skipped" marker at this node
    if isinstance(node, dict) and node.get("skipped", False):
        return None

    # Direct scalar
    if isinstance(node, (int, float)):
        return float(node)

    # Direct dict: expect 'ece'
    if spec.mtype == "direct":
        if isinstance(node, dict) and metric_field in node:
            try:
                return float(node[metric_field])
            except (TypeError, ValueError):
                return None
        return None

    # min_dict: choose minimum ECE across dict or list of dicts
    if spec.mtype == "min_dict":
        values: List[float] = []
        if isinstance(node, list):
            for item in node:
                if isinstance(item, dict) and not item.get("skipped", False) and metric_field in item:
                    try:
                        values.append(float(item[metric_field]))
                    except (TypeError, ValueError):
                        continue
        elif isinstance(node, dict):
            for value in node.values():
                if isinstance(value, dict) and not value.get("skipped", False) and metric_field in value:
                    try:
                        values.append(float(value[metric_field]))
                    except (TypeError, ValueError):
                        continue

        if values:
            return float(min(values))
        return None

    # Unknown type
    logger.warning(f"Unknown method type '{spec.mtype}' for {spec.code}")
    return None


def extract_ece(json_data: Dict[str, Any], spec: MethodSpec) -> Optional[float]:
    """Backward-compatible ECE extractor."""
    return extract_metric(json_data, spec, metric_field="ece")


def calculate_statistics(values: List[float]) -> Tuple[int, float, float]:
    """Return (n, mean, 95% CI margin) for a list of ECE values."""
    if not values:
        return 0, np.nan, np.nan
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    if n < 2:
        return n, mean, 0.0
    std = float(arr.std(ddof=1))
    ci = 1.96 * std / np.sqrt(n)
    return n, mean, float(ci)


def format_statistic(mean: float, ci: float, n: int) -> str:
    """Format as 'mean ± ci' in percent, or 'N/A' if no data."""
    if n == 0 or np.isnan(mean):
        return "N/A"
    return f"{mean * 100:.2f} ± {ci * 100:.2f}"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def collect_results_from_dir(
    directory: Path,
    metric_field: str = "ece",
) -> Tuple[
    Dict[Tuple[str, str, str], Dict[str, List[float]]],
    List[Dict[str, Any]],
]:
    """
    Parse all JSONs in directory and collect ECE values.

    Returns:
        config_data:
            (training_method, dataset, model) -> {method_code: [ece_values_across_seeds]}
        raw_rows:
            list of per‑file rows for the raw CSV
    """
    config_data: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}
    raw_rows: List[Dict[str, Any]] = []

    directory = Path(directory)
    json_files = sorted(directory.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {directory}")
    logger.info(f"Extracting metric field: {metric_field}")

    for idx, path in enumerate(json_files, 1):
        try:
            with path.open("r") as f:
                j = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {path.name}: {e}")
            continue

        cfg = j.get("experiment_config", {})
        training = cfg.get("training_method", "unknown")
        dataset = cfg.get("dataset", "unknown")
        model = cfg.get("model_name", "unknown")
        seed = cfg.get("seed", None)
        key = (training, dataset, model)

        if key not in config_data:
            config_data[key] = {m.code: [] for m in METHODS.values()}

        row: Dict[str, Any] = {
            "training": training,
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "file": path.name,
        }

        any_found = False
        for m in METHODS.values():
            metric_value = extract_metric(j, m, metric_field=metric_field)
            if metric_value is not None:
                config_data[key][m.code].append(metric_value)
                row[m.display] = metric_value
                any_found = True
            else:
                row[m.display] = np.nan

        if not any_found:
            logger.warning(f"No {metric_field} values extracted from {path.name}")
        raw_rows.append(row)

        if idx % 25 == 0:
            logger.info(f"Processed {idx}/{len(json_files)} files")

    logger.info(f"Finished parsing: {len(json_files)} files")
    return config_data, raw_rows


def save_raw_csv(raw_rows: List[Dict[str, Any]], output_csv: Path) -> None:
    """Save the per‑file raw results to CSV."""
    df = pd.DataFrame(raw_rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved raw results to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate calibration ECE results from JSON files.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing ablation_*.json files (e.g. calibration_comparison/).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="calibration_results_raw.csv",
        help="Path for raw per‑file CSV output.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="calibration_results_aggregated.json",
        help="Path for aggregated JSON (per configuration).",
    )
    parser.add_argument(
        "--metric-field",
        type=str,
        default="ece",
        choices=["ece", "adaptive_ece", "calibration_mce"],
        help="Metric field to aggregate from each method result.",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    config_data, raw_rows = collect_results_from_dir(input_dir, metric_field=args.metric_field)

    # Save raw CSV
    save_raw_csv(raw_rows, Path(args.output_csv))

    # Save aggregated config_data as JSON (means + all samples for each method)
    aggregated_for_json: Dict[str, Any] = {}
    for (training, dataset, model), methods in config_data.items():
        cfg_key = f"{training}__{dataset}__{model}"
        aggregated_for_json[cfg_key] = {
            "training_method": training,
            "dataset": dataset,
            "model": model,
            "methods": {
                code: values for code, values in methods.items() if any(np.isfinite(values))
            },
        }

    out_json_path = Path(args.output_json)
    with out_json_path.open("w") as f:
        json.dump(aggregated_for_json, f, indent=2)
    logger.info(f"Saved aggregated config data to {out_json_path}")


if __name__ == "__main__":
    main()



