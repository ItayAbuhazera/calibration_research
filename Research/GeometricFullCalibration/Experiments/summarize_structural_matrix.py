"""
Summarize calibration methods by their structural axes.

Reads summary_metrics.json, groups methods by structural-argument axes, and
emits a CSV with one row per method containing all structural dimensions plus
key decision-improvement metrics.

Usage:
    python Experiments/summarize_structural_matrix.py \\
        --summary_json path/to/summary_metrics.json \\
        --output_csv path/to/structural_matrix.csv

Columns in output:
    method_name, sample_dependent, class_dependent, joint_sample_class_dependent,
    argmax_invariant_by_construction, can_change_argmax, structural_notes,
    accuracy, nll, top_label_ece, net_flips, net_flip_rate, argmax_change_rate,
    changed_to_correct_count, changed_to_wrong_count
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

STRUCTURAL_KEYS = [
    "sample_dependent",
    "class_dependent",
    "joint_sample_class_dependent",
    "argmax_invariant_by_construction",
    "can_change_argmax",
]

METRIC_KEYS = [
    "accuracy",
    "nll",
    "top_label_ece",
]

DA_KEYS = [
    "net_flips",
    "net_flip_rate",
    "argmax_change_rate",
    "changed_to_correct_count",
    "changed_to_wrong_count",
    "accuracy_delta",
]

ALL_OUTPUT_COLS = (
    ["method_name"]
    + STRUCTURAL_KEYS
    + ["structural_notes"]
    + METRIC_KEYS
    + DA_KEYS
)


def _get_nested(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _method_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"method_name": entry.get("method_name", "?")}

    axes = entry.get("structural_axes", {})
    for k in STRUCTURAL_KEYS:
        row[k] = axes.get(k, None) if isinstance(axes, dict) else None
    row["structural_notes"] = axes.get("notes", "") if isinstance(axes, dict) else ""

    metrics = entry.get("metrics", {})
    for k in METRIC_KEYS:
        row[k] = metrics.get(k, None)

    da = entry.get("decision_audit", {})
    for k in DA_KEYS:
        row[k] = da.get(k, None) if isinstance(da, dict) else None

    return row


def _sort_key(row: Dict[str, Any]) -> tuple:
    # Argmax-invariant methods first, then by can_change_argmax, then method name
    return (
        not bool(row.get("argmax_invariant_by_construction")),
        bool(row.get("can_change_argmax")),
        str(row.get("method_name", "")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Structural matrix summary from summary_metrics.json")
    parser.add_argument(
        "--summary_json",
        required=True,
        help="Path to summary_metrics.json produced by run_unified_benchmark.py",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        default=True,
        help="Sort methods: argmax-invariant first, then by can_change_argmax and name.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        print(f"Error: {summary_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(summary_path, "r") as f:
        summary = json.load(f)

    methods: List[Dict[str, Any]] = summary.get("methods", [])
    if not methods:
        print("Warning: no methods found in summary JSON.", file=sys.stderr)

    rows = [_method_row(m) for m in methods]
    if args.sort:
        rows.sort(key=_sort_key)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_OUTPUT_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Structural matrix written: {output_path} ({len(rows)} methods)")

    # Print a compact summary table to stdout
    print(f"\n{'Method':<45} {'argmax_inv':>10} {'can_change':>10} {'net_flips':>10} {'accuracy':>9}")
    print("-" * 90)
    for row in rows:
        print(
            f"{str(row['method_name']):<45}"
            f" {str(row.get('argmax_invariant_by_construction','')):>10}"
            f" {str(row.get('can_change_argmax','')):>10}"
            f" {str(row.get('net_flips','')):>10}"
            f" {str(row.get('accuracy','')):>9}"
        )


if __name__ == "__main__":
    main()
