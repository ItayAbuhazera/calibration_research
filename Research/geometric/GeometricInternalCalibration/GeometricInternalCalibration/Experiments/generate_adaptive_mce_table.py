#!/usr/bin/env python3
"""
Generate a final calibration table containing both Adaptive ECE and MCE.

The table uses the same method extraction rules as generate_unified_metric_table.py
and reports metric values as percentages with mean +/- 95% CI.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from generate_unified_metric_table import (
    DATASET_ORDER,
    METHODS,
    collect,
    display_dataset,
    display_model,
    stats,
)


MetricRows = Dict[Tuple[str, str, str], Dict[str, List[float]]]


def _row_sort_key(key: Tuple[str, str, str]) -> Tuple[int, str, str, str]:
    training, dataset, model = key
    return (DATASET_ORDER.get(dataset, 99), dataset, model, training)


def _metric_stats(values: List[float]) -> Dict[str, Any]:
    n, mean, ci = stats(values)
    return {"n": n, "mean": mean, "ci": ci}


def _fmt_percent(mean: float, ci: float, n: int) -> str:
    if n == 0 or not math.isfinite(mean):
        return "N/A"
    return f"{mean * 100:.2f}+/-{ci * 100:.2f}"


def _fmt_percent_latex(mean: float, ci: float, n: int) -> str:
    if n == 0 or not math.isfinite(mean):
        return "---"
    return f"{mean * 100:.2f}$\\pm${ci * 100:.2f}"


def _fmt_accuracy(mean: float, ci: float, n: int) -> str:
    if n == 0 or not math.isfinite(mean):
        return "N/A"
    return f"{mean:.2f}+/-{ci:.2f}"


def _fmt_accuracy_latex(mean: float, ci: float, n: int) -> str:
    if n == 0 or not math.isfinite(mean):
        return "---"
    return f"{mean:.2f}$\\pm${ci:.2f}"


def _combined_csv_cell(adaptive: Dict[str, Any], mce: Dict[str, Any]) -> str:
    return (
        f"A-ECE: {_fmt_percent(adaptive['mean'], adaptive['ci'], adaptive['n'])}; "
        f"MCE: {_fmt_percent(mce['mean'], mce['ci'], mce['n'])}"
    )


def _combined_latex_cell(adaptive: Dict[str, Any], mce: Dict[str, Any]) -> str:
    adaptive_text = _fmt_percent_latex(adaptive["mean"], adaptive["ci"], adaptive["n"])
    mce_text = _fmt_percent_latex(mce["mean"], mce["ci"], mce["n"])
    if adaptive_text == "---" and mce_text == "---":
        return "---"
    return f"\\shortstack{{A: {adaptive_text}\\\\M: {mce_text}}}"


def _accuracy_values(adaptive_values: Dict[str, List[float]], mce_values: Dict[str, List[float]]) -> List[float]:
    values = adaptive_values.get("accuracy") or mce_values.get("accuracy") or []
    return values


def _all_keys(adaptive: MetricRows, mce: MetricRows) -> List[Tuple[str, str, str]]:
    return sorted(set(adaptive.keys()) | set(mce.keys()), key=_row_sort_key)


def write_long_stats_csv(adaptive: MetricRows, mce: MetricRows, output_csv: Path) -> None:
    fieldnames = [
        "training",
        "dataset",
        "model",
        "method",
        "method_display",
        "adaptive_ece_n",
        "adaptive_ece_mean",
        "adaptive_ece_ci95",
        "adaptive_ece_percent",
        "mce_n",
        "mce_mean",
        "mce_ci95",
        "mce_percent",
        "accuracy_n",
        "accuracy_mean_percent",
        "accuracy_ci95_percent",
        "accuracy_percent",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key in _all_keys(adaptive, mce):
            training, dataset, model = key
            adaptive_values = adaptive.get(key, {})
            mce_values = mce.get(key, {})
            acc_n, acc_mean, acc_ci = stats(_accuracy_values(adaptive_values, mce_values))

            for code, display, _paths in METHODS:
                adaptive_stats = _metric_stats(adaptive_values.get(code, []))
                mce_stats = _metric_stats(mce_values.get(code, []))
                writer.writerow(
                    {
                        "training": training,
                        "dataset": dataset,
                        "model": model,
                        "method": code,
                        "method_display": display,
                        "adaptive_ece_n": adaptive_stats["n"],
                        "adaptive_ece_mean": adaptive_stats["mean"],
                        "adaptive_ece_ci95": adaptive_stats["ci"],
                        "adaptive_ece_percent": _fmt_percent(
                            adaptive_stats["mean"], adaptive_stats["ci"], adaptive_stats["n"]
                        ),
                        "mce_n": mce_stats["n"],
                        "mce_mean": mce_stats["mean"],
                        "mce_ci95": mce_stats["ci"],
                        "mce_percent": _fmt_percent(mce_stats["mean"], mce_stats["ci"], mce_stats["n"]),
                        "accuracy_n": acc_n,
                        "accuracy_mean_percent": acc_mean,
                        "accuracy_ci95_percent": acc_ci,
                        "accuracy_percent": _fmt_accuracy(acc_mean, acc_ci, acc_n),
                    }
                )


def write_wide_csv(adaptive: MetricRows, mce: MetricRows, output_csv: Path) -> None:
    fieldnames = ["Training", "Dataset", "Model", "Accuracy", "Accuracy N"]
    fieldnames.extend(display for _code, display, _paths in METHODS)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key in _all_keys(adaptive, mce):
            training, dataset, model = key
            adaptive_values = adaptive.get(key, {})
            mce_values = mce.get(key, {})
            acc_n, acc_mean, acc_ci = stats(_accuracy_values(adaptive_values, mce_values))
            row = {
                "Training": training,
                "Dataset": dataset,
                "Model": model,
                "Accuracy": _fmt_accuracy(acc_mean, acc_ci, acc_n),
                "Accuracy N": acc_n,
            }

            for code, display, _paths in METHODS:
                adaptive_stats = _metric_stats(adaptive_values.get(code, []))
                mce_stats = _metric_stats(mce_values.get(code, []))
                row[display] = _combined_csv_cell(adaptive_stats, mce_stats)

            writer.writerow(row)


def write_latex(adaptive: MetricRows, mce: MetricRows, output_tex: Path, label: str) -> None:
    lines = [
        "% Auto-generated by Experiments/generate_adaptive_mce_table.py",
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Final calibration comparison with Adaptive ECE and MCE (\\%). "
        "Each method cell reports Adaptive ECE (A) and MCE (M) as mean $\\pm$ 95\\% CI.}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{",
        "\\begin{tabular}{>{\\columncolor{gray!15}}l>{\\columncolor{gray!15}}lcccccccccccc}",
        "\\toprule",
        "Dataset & Model {\\tiny (Acc)} & Uncal & TS & Platt & Isotonic & Beta & DAC & "
        "Fast Separation & GC(TULIP) & D.Ens & $\\RGCL$ & GC(DAC) & $\\RGCC$ \\\\",
        "\\midrule",
    ]

    current_dataset = None
    for key in _all_keys(adaptive, mce):
        _training, dataset, model = key
        if current_dataset is not None and dataset != current_dataset:
            lines.append("\\midrule")
        current_dataset = dataset

        adaptive_values = adaptive.get(key, {})
        mce_values = mce.get(key, {})
        acc_n, acc_mean, acc_ci = stats(_accuracy_values(adaptive_values, mce_values))
        acc = _fmt_accuracy_latex(acc_mean, acc_ci, acc_n)
        model_cell = f"{display_model(model)} {{\\tiny ({acc})}}" if acc != "---" else display_model(model)

        cells = [display_dataset(dataset), model_cell]
        for code, _display, _paths in METHODS:
            adaptive_stats = _metric_stats(adaptive_values.get(code, []))
            mce_stats = _metric_stats(mce_values.get(code, []))
            cells.append(_combined_latex_cell(adaptive_stats, mce_stats))

        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table*}",
        ]
    )
    output_tex.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate final Adaptive ECE + MCE calibration tables.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing ablation_*.json files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV and LaTeX outputs.")
    parser.add_argument("--training-method", default="baseline_cross_entropy")
    parser.add_argument("--pattern", default="ablation_*.json")
    parser.add_argument("--label", default="tab:final_adaptive_ece_mce")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    adaptive = collect(args.input_dir, args.pattern, args.training_method, "adaptive_ece")
    mce = collect(args.input_dir, args.pattern, args.training_method, "calibration_mce")
    if not adaptive and not mce:
        raise SystemExit("No rows collected. Check --input-dir, --pattern, and --training-method.")

    stats_csv = args.output_dir / "final_adaptive_ece_mce_stats.csv"
    table_csv = args.output_dir / "final_adaptive_ece_mce_table.csv"
    table_tex = args.output_dir / "final_adaptive_ece_mce_table.tex"

    write_long_stats_csv(adaptive, mce, stats_csv)
    write_wide_csv(adaptive, mce, table_csv)
    write_latex(adaptive, mce, table_tex, args.label)

    print(f"Wrote long stats CSV: {stats_csv}")
    print(f"Wrote final table CSV: {table_csv}")
    print(f"Wrote final LaTeX table: {table_tex}")
    print(f"Rows: {len(_all_keys(adaptive, mce))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
