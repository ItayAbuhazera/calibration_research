#!/usr/bin/env python3
"""
Generate the unified paper comparison table for ECE-like metrics.

This is intentionally focused on the paper table shape:
Dataset, Model (Acc), Uncal, TS, Platt, Isotonic, Beta, DAC,
Fast Separation, GC(TULIP), D.Ens, RGCL, GC(DAC), RGCC.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


METHODS = [
    ("uncal", "Uncal", ["standard_baselines.uncalibrated", "baselines.uncalibrated", "uncalibrated"]),
    ("ts", "TS", ["standard_baselines.temperature_scaling", "baselines.temperature_scaling", "baseline_ts"]),
    ("platt", "Platt", ["standard_baselines.platt_scaling", "baselines.platt_scaling"]),
    ("isotonic", "Isotonic", ["standard_baselines.isotonic_toplabel", "baselines.isotonic_toplabel"]),
    ("beta", "Beta", ["standard_baselines.beta_calibration", "baselines.beta_calibration"]),
    (
        "dac",
        "DAC",
        ["standard_baselines.density_aware_calibration", "baselines.density_aware_calibration", "dac_standalone"],
    ),
    (
        "fast_separation",
        "Fast Separation",
        ["standard_baselines.geometric_physical_space", "baselines.geometric_physical_space", "geometric_physical_space"],
    ),
    ("tulip", "GC(TULIP)", ["sgc_with_tulip_layers_separation", "sgc_with_tulip_layers"]),
    ("deep_ensemble", "D.Ens", ["deep_ensemble", "deep_ensemble_scan"]),
    ("rgcl", "$\\RGCL$", ["global_random_separation", "sgc_faiss", "global_random"]),
    (
        "gc_dac",
        "GC(DAC)",
        ["sgc_with_dac_layers_separation", "sgc_with_dac_layers_trust_score", "sgc_with_dac_layers"],
    ),
    ("rgcc", "$\\RGCC$", ["coordinate_sampling_separation", "coordinate_sampling_faiss", "coordinate_sampling"]),
]

DATASET_ORDER = {"cifar10": 0, "cifar100": 1, "tiny_imagenet": 2, "tinyimagenet": 2}
DATASET_DISPLAY = {
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "tiny_imagenet": "TINY",
    "tinyimagenet": "TINY",
}
MODEL_DISPLAY = {
    "dinov2_large": "dinov2(L)",
    "dinov2_giant": "dinov2(G)",
}
METRIC_LABEL = {
    "ece": "ECE",
    "adaptive_ece": "Adaptive ECE",
    "calibration_mce": "MCE",
}


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


def extract_metric(data: Dict[str, Any], paths: Iterable[str], metric_field: str) -> Optional[float]:
    candidate_fields = [metric_field, f"{metric_field}_mean"]
    if metric_field == "ece":
        candidate_fields.append("ece_mean")
    elif metric_field == "adaptive_ece":
        candidate_fields.append("adaptive_ece_mean")
    elif metric_field == "calibration_mce":
        candidate_fields.extend(["calibration_mce_mean", "mce", "mce_mean"])

    for path in paths:
        node = traverse(data, path)
        if node is None:
            continue
        scalar = finite_float(node)
        if scalar is not None:
            return scalar
        if not isinstance(node, dict) or node.get("skipped") or node.get("error"):
            continue
        for field in candidate_fields:
            value = finite_float(node.get(field))
            if value is not None:
                return value
    return None


def extract_accuracy(data: Dict[str, Any]) -> Optional[float]:
    for path in ["standard_baselines.uncalibrated", "baselines.uncalibrated", "uncalibrated"]:
        node = traverse(data, path)
        if not isinstance(node, dict):
            continue
        acc = finite_float(node.get("acc", node.get("accuracy")))
        if acc is None:
            continue
        return acc * 100.0 if acc <= 1.0 else acc
    return None


def stats(values: List[float]) -> Tuple[int, float, float]:
    clean = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    n = int(clean.size)
    if n == 0:
        return 0, math.nan, math.nan
    mean = float(clean.mean())
    if n < 2:
        return n, mean, 0.0
    ci = 1.96 * float(clean.std(ddof=1)) / math.sqrt(n)
    return n, mean, ci


def metric_cell(mean: float, ci: float, n: int) -> str:
    if n == 0 or not math.isfinite(mean):
        return "---"
    return f"{mean * 100:.2f}$\\pm${ci * 100:.2f}"


def accuracy_cell(mean: float, ci: float, n: int) -> str:
    if n == 0 or not math.isfinite(mean):
        return "---"
    return f"{mean:.2f}$\\pm${ci:.2f}"


def color_cell(cell: str, rank: int) -> str:
    colors = ["blue", "teal", "olive"]
    if cell == "---" or rank >= len(colors):
        return cell
    return f"\\textcolor{{{colors[rank]}}}{{\\textbf{{{cell}}}}}"


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def display_dataset(dataset: str) -> str:
    return DATASET_DISPLAY.get(dataset, dataset.upper().replace("_", "\\_"))


def display_model(model: str) -> str:
    return MODEL_DISPLAY.get(model, latex_escape(model))


def sort_key(row: Dict[str, Any]) -> Tuple[int, str, str]:
    dataset = row["dataset"]
    return (DATASET_ORDER.get(dataset, 99), dataset, row["model"])


def collect(input_dir: Path, pattern: str, training_method: Optional[str], metric_field: str) -> Dict[Tuple[str, str, str], Dict[str, List[float]]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(input_dir.glob(pattern)):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"[WARNING] Skipping unreadable JSON {path.name}: {exc}")
            continue

        cfg = data.get("experiment_config", data.get("experiment_info", {}))
        training = cfg.get("training_method", "unknown")
        if training_method and training != training_method:
            continue
        dataset = cfg.get("dataset", "unknown")
        model = cfg.get("model_name", "unknown")
        key = (training, dataset, model)

        for code, _display, paths in METHODS:
            value = extract_metric(data, paths, metric_field)
            if value is not None:
                grouped[key][code].append(value)

        acc = extract_accuracy(data)
        if acc is not None:
            grouped[key]["accuracy"].append(acc)

    return grouped


def build_rows(grouped: Dict[Tuple[str, str, str], Dict[str, List[float]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (training, dataset, model), values in sorted(grouped.items(), key=lambda item: sort_key({
        "dataset": item[0][1],
        "model": item[0][2],
    })):
        row: Dict[str, Any] = {"training": training, "dataset": dataset, "model": model}
        acc_n, acc_mean, acc_ci = stats(values.get("accuracy", []))
        row["accuracy_n"] = acc_n
        row["accuracy_mean"] = acc_mean
        row["accuracy_ci"] = acc_ci
        for code, _display, _paths in METHODS:
            n, mean, ci = stats(values.get(code, []))
            row[f"{code}_n"] = n
            row[f"{code}_mean"] = mean
            row[f"{code}_ci"] = ci
        rows.append(row)
    return rows


def write_stats_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    fieldnames = ["training", "dataset", "model", "accuracy_n", "accuracy_mean", "accuracy_ci"]
    for code, _display, _paths in METHODS:
        fieldnames.extend([f"{code}_n", f"{code}_mean", f"{code}_ci"])
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows: List[Dict[str, Any]], output_tex: Path, metric_field: str, label: str) -> None:
    metric_label = METRIC_LABEL.get(metric_field, metric_field.replace("_", " ").title())
    lines = [
        "% Auto-generated by Experiments/generate_unified_metric_table.py",
        "\\begin{table*}[htbp]",
        "\\centering",
        (
            f"\\caption{{{metric_label} (\\%) comparison across datasets and models. "
            "Values shown as mean $\\pm$ 95\\% CI. "
            "\\textcolor{blue}{\\textbf{Best}}, \\textcolor{teal}{\\textbf{2nd}}, "
            "\\textcolor{olive}{\\textbf{3rd}}.}}"
        ),
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\small",
        "\\resizebox{\\linewidth}{!}{",
        "\\begin{tabular}{>{\\columncolor{gray!15}}l>{\\columncolor{gray!15}}lcccccccccccc}",
        "\\toprule",
        "Dataset & Model {\\tiny (Acc)} & Uncal & TS & Platt & Isotonic & Beta & DAC & Fast Separation & GC(TULIP) & D.Ens & $\\RGCL$ & GC(DAC) & $\\RGCC$ \\\\",
        "\\midrule",
    ]

    current_dataset = None
    for row in rows:
        dataset = row["dataset"]
        if current_dataset is not None and dataset != current_dataset:
            lines.append("\\midrule")
        current_dataset = dataset

        formatted: Dict[str, str] = {}
        ranking: List[Tuple[float, str]] = []
        for code, _display, _paths in METHODS:
            n = int(row[f"{code}_n"])
            mean = float(row[f"{code}_mean"])
            ci = float(row[f"{code}_ci"])
            formatted[code] = metric_cell(mean, ci, n)
            if code != "uncal" and n > 0 and math.isfinite(mean):
                ranking.append((mean, code))

        for rank, (_mean, code) in enumerate(sorted(ranking)[:3]):
            formatted[code] = color_cell(formatted[code], rank)

        acc = accuracy_cell(float(row["accuracy_mean"]), float(row["accuracy_ci"]), int(row["accuracy_n"]))
        model_with_acc = f"{display_model(row['model'])} {{\\tiny ({acc})}}" if acc != "---" else display_model(row["model"])

        cells = [
            display_dataset(dataset),
            model_with_acc,
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
            formatted["gc_dac"],
            formatted["rgcc"],
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table*}",
    ])
    output_tex.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate unified paper table for ECE-like metrics.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing ablation_*.json files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV and LaTeX outputs.")
    parser.add_argument("--metric-field", choices=["ece", "adaptive_ece", "calibration_mce"], default="ece")
    parser.add_argument("--training-method", default="baseline_cross_entropy")
    parser.add_argument("--pattern", default="ablation_*.json")
    parser.add_argument("--label", default=None, help="LaTeX label. Defaults to tab:unified_<metric-field>.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or f"tab:unified_{args.metric_field}"

    grouped = collect(args.input_dir, args.pattern, args.training_method, args.metric_field)
    rows = build_rows(grouped)
    if not rows:
        raise SystemExit("No rows collected. Check --input-dir, --pattern, and --training-method.")

    csv_path = args.output_dir / f"unified_{args.metric_field}_stats.csv"
    tex_path = args.output_dir / f"unified_{args.metric_field}_table.tex"
    write_stats_csv(rows, csv_path)
    write_latex(rows, tex_path, args.metric_field, label)

    print(f"Wrote stats CSV: {csv_path}")
    print(f"Wrote LaTeX table: {tex_path}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
