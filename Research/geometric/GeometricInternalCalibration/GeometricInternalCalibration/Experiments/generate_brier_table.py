#!/usr/bin/env python3
"""Aggregate full-distribution Brier scores from per-seed JSON results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


METHODS: List[Tuple[str, str, List[str]]] = [
    ("uncal", "Uncal", ["standard_baselines.uncalibrated.brier"]),
    ("ts", "TS", ["standard_baselines.temperature_scaling.brier"]),
    ("platt", "Platt", ["standard_baselines.platt_scaling.brier"]),
    ("isotonic", "Isotonic", ["standard_baselines.isotonic_toplabel.brier"]),
    ("beta", "Beta", ["standard_baselines.beta_calibration.brier"]),
    (
        "dac",
        "DAC",
        [
            "dac_orig.brier",
            "standard_baselines.density_aware_calibration.brier",
        ],
    ),
    ("fast_separation", "Fast Separation", ["pixel_geometric.brier"]),
    ("tulip", "TULIP", ["sgc_with_tulip_layers_separation.brier"]),
    ("deep_ensemble", "D.Ens", ["deep_ensemble.brier"]),
    ("rgcl", "RGCL", ["global_random_separation.brier"]),
    ("rgcc", "RGCC", ["coordinate_sampling_separation.brier"]),
]

ROW_GROUPS: List[Tuple[str, List[str]]] = [
    ("cifar10", ["densenet121", "resnet101", "resnet152", "resnet18", "resnet50"]),
    (
        "cifar100",
        [
            "densenet121",
            "dinov2_giant",
            "dinov2_large",
            "resnet101",
            "resnet152",
            "resnet18",
            "resnet50",
        ],
    ),
    ("tiny_imagenet", ["resnet101", "resnet152", "resnet50"]),
]

DATASET_DISPLAY = {
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "tiny_imagenet": "TINY",
}

MODEL_DISPLAY = {
    "dinov2_large": "dinov2(L)",
    "dinov2_giant": "dinov2(G)",
}

FILENAME_RE = re.compile(
    r"^ablation_(?P<training>.+)_(?P<dataset>cifar10|cifar100|tiny_imagenet)_(?P<model>.+)_seed(?P<seed>\d+)\.json$"
)


def traverse(data: Dict[str, Any], dotted_path: str) -> Optional[Any]:
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def first_available_brier(data: Dict[str, Any], paths: Iterable[str]) -> Tuple[Optional[float], Optional[str]]:
    for path in paths:
        value = finite_float(traverse(data, path))
        if value is not None:
            return value, path
    return None, None


def parse_metadata(path: Path, data: Dict[str, Any]) -> Optional[Tuple[str, str, str, int]]:
    match = FILENAME_RE.match(path.name)
    if match:
        return (
            match.group("training"),
            match.group("dataset"),
            match.group("model"),
            int(match.group("seed")),
        )

    cfg = data.get("experiment_config", {})
    training = cfg.get("training_method")
    dataset = cfg.get("dataset")
    model = cfg.get("model_name")
    seed = cfg.get("seed")
    if training is None or dataset is None or model is None or seed is None:
        return None
    try:
        seed_int = int(seed)
    except (TypeError, ValueError):
        return None
    return str(training), str(dataset), str(model), seed_int


def collect(input_dir: Path, pattern: str, training_method: str) -> Tuple[
    Dict[Tuple[str, str], Dict[str, List[float]]],
    Dict[Tuple[str, str], Dict[str, set]],
    List[str],
]:
    values: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    seeds: Dict[Tuple[str, str], Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    warnings: List[str] = []

    files = sorted(input_dir.glob(pattern))
    print(f"[INFO] Found {len(files)} JSON files in {input_dir}")

    for path in files:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            warnings.append(f"unreadable JSON {path.name}: {exc}")
            continue

        metadata = parse_metadata(path, data)
        if metadata is None:
            warnings.append(f"could not parse metadata for {path.name}")
            continue

        training, dataset, model, seed = metadata
        if training != training_method:
            continue

        key = (dataset, model)
        for code, _display, paths in METHODS:
            value, used_path = first_available_brier(data, paths)
            if value is None:
                continue
            values[key][code].append(value)
            seeds[key][code].add(seed)
            if code == "dac" and used_path == "standard_baselines.density_aware_calibration.brier":
                if traverse(data, "dac_orig.brier") is not None:
                    warnings.append(f"DAC fallback used despite dac_orig presence in {path.name}")

    return values, seeds, warnings


def stats(raw_values: List[float]) -> Tuple[int, float, float]:
    arr = np.asarray([v for v in raw_values if math.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return 0, math.nan, math.nan
    mean = float(arr.mean())
    if n < 2:
        return n, mean, 0.0
    ci95 = 1.96 * float(arr.std(ddof=1)) / math.sqrt(n)
    return n, mean, ci95


def display_dataset(dataset: str) -> str:
    return DATASET_DISPLAY.get(dataset, dataset.upper().replace("_", "\\_"))


def display_model(model: str) -> str:
    return MODEL_DISPLAY.get(model, model.replace("_", "\\_"))


def metric_cell(n: int, mean: float, ci95: float) -> str:
    if n == 0 or not math.isfinite(mean):
        return "---"
    return f"{mean * 100:.2f}$\\pm${ci95 * 100:.2f}"


def color_cell(cell: str, rank: int) -> str:
    colors = ["blue", "teal", "olive"]
    if cell == "---" or rank >= len(colors):
        return cell
    return f"\\textcolor{{{colors[rank]}}}{{\\textbf{{{cell}}}}}"


def ordered_row_keys() -> List[Tuple[str, str]]:
    return [(dataset, model) for dataset, models in ROW_GROUPS for model in models]


def build_long_rows(values: Dict[Tuple[str, str], Dict[str, List[float]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset, model in ordered_row_keys():
        method_values = values.get((dataset, model), {})
        for code, display, _paths in METHODS:
            n, mean, ci95 = stats(method_values.get(code, []))
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "method": display,
                    "n_seeds": n,
                    "mean_pct": mean * 100 if math.isfinite(mean) else "",
                    "ci95_pct": ci95 * 100 if math.isfinite(ci95) else "",
                }
            )
    return rows


def validate_scale(long_rows: List[Dict[str, Any]]) -> None:
    finite_means = [
        float(row["mean_pct"])
        for row in long_rows
        if row["mean_pct"] != "" and math.isfinite(float(row["mean_pct"]))
    ]
    if not finite_means:
        raise SystemExit("No finite Brier means collected; refusing to write table.")
    over_100 = [value for value in finite_means if value > 100.0]
    if over_100:
        raise SystemExit(
            f"Brier scaling looks wrong: found mean_pct > 100, max={max(over_100):.2f}."
        )
    if max(finite_means) < 5.0:
        raise SystemExit(
            "Brier scaling looks like ECE-scale values: every mean is below 5 pp."
        )


def write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "model", "method", "n_seeds", "mean_pct", "ci95_pct"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_latex(values: Dict[Tuple[str, str], Dict[str, List[float]]], output_tex: Path) -> None:
    lines = [
        "% Auto-generated by Experiments/generate_brier_table.py",
        "\\begin{table*}[htbp]",
        "\\centering",
        (
            "\\caption{Brier score (\\%) comparison across datasets and models. "
            "Values shown as mean $\\pm$ 95\\% CI. "
            "\\textcolor{blue}{\\textbf{Best}}, \\textcolor{teal}{\\textbf{2nd}}, "
            "\\textcolor{olive}{\\textbf{3rd}}.}"
        ),
        "\\label{tab:unified_brier}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\small",
        "\\resizebox{\\linewidth}{!}{",
        "\\begin{tabular}{>{\\columncolor{gray!15}}l>{\\columncolor{gray!15}}lccccccccccc}",
        "\\toprule",
        "Dataset & Model & Uncal & TS & Platt & Isotonic & Beta & DAC & Fast Separation & TULIP & D.Ens & $\\RGCL$ & $\\RGCC$ \\\\",
        "\\midrule",
    ]

    for group_idx, (dataset, models) in enumerate(ROW_GROUPS):
        if group_idx > 0:
            lines.append("\\midrule")
        for model in models:
            row_values = values.get((dataset, model), {})
            formatted: Dict[str, str] = {}
            ranking: List[Tuple[float, int, str]] = []

            for method_idx, (code, _display, _paths) in enumerate(METHODS):
                n, mean, ci95 = stats(row_values.get(code, []))
                formatted[code] = metric_cell(n, mean, ci95)
                if code != "uncal" and n > 0 and math.isfinite(mean):
                    ranking.append((mean, method_idx, code))

            for rank, (_mean, _method_idx, code) in enumerate(sorted(ranking)[:3]):
                formatted[code] = color_cell(formatted[code], rank)

            cells = [
                display_dataset(dataset),
                display_model(model),
                formatted["uncal"],
                formatted["ts"],
                formatted["platt"],
                formatted["isotonic"],
                formatted["beta"],
                formatted["dac"],
                formatted["fast_separation"],
                formatted["tulip"],
                formatted["deep_ensemble"],
                formatted["rgcl"],
                formatted["rgcc"],
            ]
            lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table*}"])
    output_tex.write_text("\n".join(lines) + "\n")


def print_summary(
    values: Dict[Tuple[str, str], Dict[str, List[float]]],
    warnings: List[str],
) -> None:
    print("\n[SUMMARY] n_seeds per requested cell")
    missing_cells: List[str] = []
    for dataset, models in ROW_GROUPS:
        for model in models:
            row_values = values.get((dataset, model), {})
            populated = []
            for code, display, _paths in METHODS:
                n = len(row_values.get(code, []))
                if n > 0:
                    populated.append(f"{display}={n}")
                else:
                    missing_cells.append(f"{dataset}/{model}/{display}")
            print(f"  {dataset}/{model}: " + (", ".join(populated) if populated else "none"))

    print(f"\n[SUMMARY] Cells reported as ---: {len(missing_cells)}")
    for item in missing_cells[:80]:
        print(f"  --- {item}")
    if len(missing_cells) > 80:
        print(f"  ... {len(missing_cells) - 80} more")

    if warnings:
        print(f"\n[WARNINGS] {len(warnings)}")
        for warning in warnings[:80]:
            print(f"  {warning}")
        if len(warnings) > 80:
            print(f"  ... {len(warnings) - 80} more")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a Brier score paper table from JSON results.")
    parser.add_argument("--input-dir", type=Path, default=Path("calibration_comparison_mce_adaptive"))
    parser.add_argument("--pattern", default="ablation_*_seed*.json")
    parser.add_argument("--training-method", default="baseline_cross_entropy")
    parser.add_argument("--output-csv", type=Path, default=Path("brier_long.csv"))
    parser.add_argument("--output-tex", type=Path, default=Path("brier_table.tex"))
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    values, _seeds, warnings = collect(args.input_dir, args.pattern, args.training_method)
    long_rows = build_long_rows(values)
    validate_scale(long_rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    write_csv(long_rows, args.output_csv)
    write_latex(values, args.output_tex)
    print_summary(values, warnings)

    print(f"\nWrote CSV: {args.output_csv}")
    print(f"Wrote LaTeX: {args.output_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
