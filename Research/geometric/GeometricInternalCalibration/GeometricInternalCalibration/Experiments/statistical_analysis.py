#!/usr/bin/env python3
"""
Paired statistical tests for calibration methods.

This script runs paired t‑tests for each configuration (training, dataset,
model) using the aggregated results from `aggregate_calibration_results.py`.

Pairs tested (ECE, lower is better):
- SGC vs DAC           (our main claim: SGC vs original DAC)
- SGC vs TS            (baseline comparison vs Temperature Scaling)
- SGC vs RandCoord     (layer selection vs coordinate selection)
- GeoComb vs SGC       (combined vs random layer selection)
- SGC vs Geo+DAC Feat  (SPP+JL vs spatial avg+L2 feature extraction)

Outputs:
- JSON file with per‑comparison and per‑configuration statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .aggregate_calibration_results import (
    collect_results_from_dir,
    logger,
)

try:
    from scipy.stats import ttest_rel

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not available - statistical tests will be skipped")


def _paired_t(values1: List[float], values2: List[float]) -> Tuple[float, float]:
    """Return (p, d) where d is Cohen's d for paired samples."""
    if not HAS_SCIPY:
        return float("nan"), float("nan")
    if len(values1) != len(values2) or len(values1) < 2:
        return float("nan"), float("nan")

    a = np.asarray(values1, dtype=float)
    b = np.asarray(values2, dtype=float)
    diffs = a - b
    try:
        _t, p = ttest_rel(a, b)
    except Exception:
        return float("nan"), float("nan")

    std = diffs.std(ddof=1)
    d = diffs.mean() / std if std > 0 else 0.0
    return float(p), float(d)


def _round_floats(obj: Any, ndigits: int = 5) -> Any:
    """Recursively round floats for nicer JSON output."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def run_comparisons(
    config_data: Dict[Tuple[str, str, str], Dict[str, List[float]]]
) -> Dict[str, Any]:
    """
    Run all requested comparisons.

    config_data:
      (training, dataset, model) -> {method_code: [ece_values]}
    """
    comparisons = [
        ("sgc", "dac", "SGC_vs_DAC"),
        ("sgc", "ts", "SGC_vs_TS"),
        ("sgc", "rand_coord", "SGC_vs_RandCoord"),
        ("geo_comb", "sgc", "GeoComb_vs_SGC"),
        ("sgc", "geo_dac_feat", "SGC_vs_GeoDACFeat"),
    ]

    results: Dict[str, Any] = {}
    for m1, m2, name in comparisons:
        per_cfg: List[Dict[str, Any]] = []
        p_vals: List[float] = []
        ds: List[float] = []
        sig = 0
        total = 0

        for (training, dataset, model), methods in sorted(config_data.items()):
            v1 = methods.get(m1, [])
            v2 = methods.get(m2, [])
            if len(v1) != len(v2) or len(v1) < 2:
                continue

            p, d = _paired_t(v1, v2)
            if not np.isfinite(p):
                continue

            total += 1
            if p < 0.05:
                sig += 1
            p_vals.append(p)
            ds.append(d)

            per_cfg.append(
                {
                    "config": f"{training}_{dataset}_{model}",
                    "n": len(v1),
                    "p_value": p,
                    "effect_size_d": d,
                    "mean_ece_1": float(np.mean(v1)),
                    "mean_ece_2": float(np.mean(v2)),
                    "mean_diff": float(np.mean(np.array(v1) - np.array(v2))),
                    "method1": m1,
                    "method2": m2,
                }
            )

        if p_vals:
            results[name] = {
                "method1": m1,
                "method2": m2,
                "total_comparisons": total,
                "significant_count": sig,
                "significant_rate": sig / total if total > 0 else 0.0,
                "mean_p_value": float(np.mean(p_vals)),
                "median_p_value": float(np.median(p_vals)),
                "mean_effect_size_d": float(np.mean(ds)),
                "median_effect_size_d": float(np.median(ds)),
                "per_config": per_cfg,
            }
        else:
            results[name] = {
                "method1": m1,
                "method2": m2,
                "total_comparisons": 0,
                "significant_count": 0,
                "significant_rate": 0.0,
                "note": "No valid paired comparisons (need >=2 seeds with both methods).",
            }

    return _round_floats(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired statistical tests between calibration methods.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing ablation_*.json files (e.g. calibration_comparison/).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="calibration_statistical_tests.json",
        help="Where to write JSON with statistical test results.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    config_data, _raw_rows = collect_results_from_dir(input_dir)
    if not config_data:
        raise SystemExit("No configurations found; check input directory and JSON structure.")

    results = run_comparisons(config_data)
    out_path = Path(args.output_json)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Statistical test results written to {out_path}")


if __name__ == "__main__":
    main()


