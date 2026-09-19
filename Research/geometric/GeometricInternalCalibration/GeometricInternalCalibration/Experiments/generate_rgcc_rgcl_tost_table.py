#!/usr/bin/env python3
"""
Generate paired TOST equivalence tables comparing RGCC and RGCL.

The script reads the same ablation_*.json files used by
generate_unified_metric_table.py and pairs RGCC/RGCL values within each
experiment seed. The reported difference is RGCC - RGCL, so positive values
mean RGCC has higher error for lower-is-better metrics.
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

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except Exception:
    scipy_stats = None
    HAS_SCIPY = False


RGCL_PATHS = ["global_random_separation", "sgc_faiss", "global_random"]
RGCC_PATHS = ["coordinate_sampling_separation", "coordinate_sampling_faiss", "coordinate_sampling"]

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


PairKey = Tuple[str, str, str]


def _betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


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


def metric_fields(metric_field: str) -> List[str]:
    fields = [metric_field, f"{metric_field}_mean"]
    if metric_field == "ece":
        fields.append("ece_mean")
    elif metric_field == "adaptive_ece":
        fields.append("adaptive_ece_mean")
    elif metric_field == "calibration_mce":
        fields.extend(["calibration_mce_mean", "mce", "mce_mean"])
    return fields


def extract_metric(data: Dict[str, Any], paths: Iterable[str], metric_field: str) -> Optional[float]:
    for path in paths:
        node = traverse(data, path)
        if node is None:
            continue

        scalar = finite_float(node)
        if scalar is not None:
            return scalar

        if not isinstance(node, dict) or node.get("skipped") or node.get("error"):
            continue

        for field in metric_fields(metric_field):
            value = finite_float(node.get(field))
            if value is not None:
                return value

    return None


def extract_base_accuracy(data: Dict[str, Any]) -> Optional[float]:
    """Extract base model accuracy as a fraction in [0, 1] when available."""
    for path in ["standard_baselines.uncalibrated", "baselines.uncalibrated", "uncalibrated"]:
        node = traverse(data, path)
        if not isinstance(node, dict) or node.get("skipped") or node.get("error"):
            continue
        value = finite_float(node.get("acc", node.get("accuracy")))
        if value is None:
            continue
        return value / 100.0 if value > 1.0 else value
    return None


def display_dataset(dataset: str) -> str:
    return DATASET_DISPLAY.get(dataset, dataset.upper().replace("_", "\\_"))


def display_model(model: str) -> str:
    return MODEL_DISPLAY.get(model, model.replace("_", "\\_"))


def t_cdf(value: float, df: int) -> float:
    if HAS_SCIPY:
        return float(scipy_stats.t.cdf(value, df))
    if value == 0.0:
        return 0.5
    x = df / (df + value * value)
    ib = _regularized_incomplete_beta(df / 2.0, 0.5, x)
    if value > 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def t_crit_95(df: int) -> float:
    if HAS_SCIPY:
        return float(scipy_stats.t.ppf(0.975, df))
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < 0.975:
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < 0.975:
            lo = mid
        else:
            hi = mid
    return hi


def format_pm(mean: float, ci: float) -> str:
    return f"{mean * 100:.2f}$\\pm${ci * 100:.2f}"


def format_p(p_value: float) -> str:
    if not math.isfinite(p_value):
        return "---"
    if p_value < 0.001:
        return "$<0.001$"
    return f"{p_value:.3f}"


def mean_ci(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, 0.0
    se = float(arr.std(ddof=1)) / math.sqrt(arr.size)
    return mean, t_crit_95(arr.size - 1) * se


def collect_pairs(
    input_dir: Path,
    pattern: str,
    training_method: Optional[str],
    metric_field: str,
) -> Tuple[Dict[PairKey, List[Dict[str, Any]]], Dict[str, int]]:
    grouped: Dict[PairKey, List[Dict[str, Any]]] = defaultdict(list)
    skipped = {
        "low_accuracy": 0,
        "missing_pair": 0,
        "unreadable": 0,
        "wrong_training_method": 0,
    }
    for path in sorted(input_dir.glob(pattern)):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"[WARNING] Skipping unreadable JSON {path.name}: {exc}")
            skipped["unreadable"] += 1
            continue

        cfg = data.get("experiment_config", data.get("experiment_info", {}))
        training = cfg.get("training_method", "unknown")
        if training_method and training != training_method:
            skipped["wrong_training_method"] += 1
            continue

        accuracy = extract_base_accuracy(data)
        if accuracy is not None and accuracy < collect_pairs.min_accuracy:
            skipped["low_accuracy"] += 1
            continue

        rgcl = extract_metric(data, RGCL_PATHS, metric_field)
        rgcc = extract_metric(data, RGCC_PATHS, metric_field)
        if rgcl is None or rgcc is None:
            skipped["missing_pair"] += 1
            continue

        dataset = cfg.get("dataset", "unknown")
        model = cfg.get("model_name", "unknown")
        seed = cfg.get("seed", path.stem)
        grouped[(training, dataset, model)].append(
            {
                "seed": seed,
                "file": path.name,
                "rgcl": rgcl,
                "rgcc": rgcc,
                "accuracy": accuracy,
            }
        )

    for pairs in grouped.values():
        pairs.sort(key=lambda item: str(item["seed"]))
    return grouped, skipped


collect_pairs.min_accuracy = 0.5


def tost(values_rgcc: List[float], values_rgcl: List[float], margin: float, alpha: float) -> Dict[str, Any]:
    diffs = np.asarray(values_rgcc, dtype=float) - np.asarray(values_rgcl, dtype=float)
    n = int(diffs.size)
    mean_diff = float(diffs.mean())

    if n < 2:
        return {
            "n": n,
            "mean_diff": mean_diff,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
            "tost_p": math.nan,
            "dz": math.nan,
            "equivalent": False,
            "result": "Need more pairs",
        }

    sd = float(diffs.std(ddof=1))
    if sd == 0.0:
        equivalent = abs(mean_diff) < margin
        p_value = 0.0 if equivalent else 1.0
        return {
            "n": n,
            "mean_diff": mean_diff,
            "ci_lower": mean_diff,
            "ci_upper": mean_diff,
            "tost_p": p_value,
            "dz": 0.0,
            "equivalent": equivalent,
            "result": "Equivalent" if equivalent else "Different",
        }

    se = sd / math.sqrt(n)
    df = n - 1
    ci_half = t_crit_95(df) * se
    ci_lower = mean_diff - ci_half
    ci_upper = mean_diff + ci_half

    t_upper = (mean_diff - margin) / se
    t_lower = (mean_diff + margin) / se
    p_upper = t_cdf(t_upper, df)
    p_lower = 1.0 - t_cdf(t_lower, df)
    p_value = max(p_upper, p_lower)
    equivalent = p_value < alpha

    if equivalent:
        result = "Equivalent"
    elif ci_upper < 0:
        result = "RGCC lower"
    elif ci_lower > 0:
        result = "RGCL lower"
    else:
        result = "Inconclusive"

    return {
        "n": n,
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "tost_p": float(p_value),
        "dz": float(mean_diff / sd),
        "equivalent": equivalent,
        "result": result,
    }


def build_rows(
    grouped: Dict[PairKey, List[Dict[str, Any]]],
    relative_margin: float,
    absolute_margin: float,
    alpha: float,
    min_pairs: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for (training, dataset, model), pairs in sorted(
        grouped.items(),
        key=lambda item: (DATASET_ORDER.get(item[0][1], 99), item[0][1], item[0][2], item[0][0]),
    ):
        if len(pairs) < min_pairs:
            continue

        rgcl_values = [float(item["rgcl"]) for item in pairs]
        rgcc_values = [float(item["rgcc"]) for item in pairs]
        accuracies = [float(item["accuracy"]) for item in pairs if item.get("accuracy") is not None]
        rgcl_mean, rgcl_ci = mean_ci(rgcl_values)
        rgcc_mean, rgcc_ci = mean_ci(rgcc_values)
        accuracy_mean, accuracy_ci = mean_ci(accuracies) if accuracies else (math.nan, math.nan)
        baseline = (rgcl_mean + rgcc_mean) / 2.0
        margin = max(relative_margin * baseline, absolute_margin)
        test = tost(rgcc_values, rgcl_values, margin, alpha)

        rows.append(
            {
                "training": training,
                "dataset": dataset,
                "model": model,
                "n": len(pairs),
                "rgcl_mean": rgcl_mean,
                "rgcl_ci": rgcl_ci,
                "rgcc_mean": rgcc_mean,
                "rgcc_ci": rgcc_ci,
                "accuracy_mean": accuracy_mean,
                "accuracy_ci": accuracy_ci,
                "margin": margin,
                **test,
            }
        )
    return rows


def write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    fieldnames = [
        "training",
        "dataset",
        "model",
        "n",
        "rgcl_mean",
        "rgcl_ci95",
        "rgcc_mean",
        "rgcc_ci95",
        "accuracy_mean",
        "accuracy_ci95",
        "mean_diff",
        "ci_lower",
        "ci_upper",
        "equivalence_margin",
        "tost_p",
        "cohen_dz",
        "equivalent",
        "result",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "training": row["training"],
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "n": row["n"],
                    "rgcl_mean": row["rgcl_mean"],
                    "rgcl_ci95": row["rgcl_ci"],
                    "rgcc_mean": row["rgcc_mean"],
                    "rgcc_ci95": row["rgcc_ci"],
                    "accuracy_mean": row["accuracy_mean"],
                    "accuracy_ci95": row["accuracy_ci"],
                    "mean_diff": row["mean_diff"],
                    "ci_lower": row["ci_lower"],
                    "ci_upper": row["ci_upper"],
                    "equivalence_margin": row["margin"],
                    "tost_p": row["tost_p"],
                    "cohen_dz": row["dz"],
                    "equivalent": row["equivalent"],
                    "result": row["result"],
                }
            )


def write_latex(rows: List[Dict[str, Any]], output_tex: Path, metric_field: str, label: str) -> None:
    metric_label = METRIC_LABEL.get(metric_field, metric_field.replace("_", " ").title())
    margins = [row["margin"] for row in rows]
    fixed_margin = margins and all(abs(margin - margins[0]) < 1e-12 for margin in margins)
    margin_note = (
        f"Equivalence margin is fixed at $\\delta={margins[0] * 100:.1f}$ percentage points."
        if fixed_margin
        else "Equivalence margins are shown per row."
    )
    lines = [
        "% Auto-generated by Experiments/generate_rgcc_rgcl_tost_table.py",
        "\\begin{table*}[htbp]",
        "\\centering",
        (
            f"\\caption{{Paired TOST equivalence test for $\\RGCC$ vs $\\RGCL$ using {metric_label}. "
            f"Values are percentages. Difference is $\\RGCC-\\RGCL$; lower is better. "
            f"Runs with base accuracy below 50\\% are excluded. {margin_note}}}"
        ),
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\small",
        "\\resizebox{\\linewidth}{!}{",
        "\\begin{tabular}{llrrrrrrrl}",
        "\\toprule",
        "Dataset & Model & $N$ & $\\RGCL$ & $\\RGCC$ & Diff & 95\\% CI & $\\delta$ & TOST $p$ & Result \\\\",
        "\\midrule",
    ]

    current_dataset = None
    for row in rows:
        dataset = row["dataset"]
        if current_dataset is not None and dataset != current_dataset:
            lines.append("\\midrule")
        current_dataset = dataset

        ci_text = f"[{row['ci_lower'] * 100:.2f}, {row['ci_upper'] * 100:.2f}]"
        cells = [
            display_dataset(dataset),
            display_model(row["model"]),
            str(row["n"]),
            format_pm(row["rgcl_mean"], row["rgcl_ci"]),
            format_pm(row["rgcc_mean"], row["rgcc_ci"]),
            f"{row['mean_diff'] * 100:+.2f}",
            ci_text,
            f"{row['margin'] * 100:.2f}",
            format_p(row["tost_p"]),
            row["result"],
        ]
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
    parser = argparse.ArgumentParser(description="Generate RGCC vs RGCL paired TOST tables.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing ablation_*.json files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV and LaTeX outputs.")
    parser.add_argument("--metric-field", choices=["ece", "adaptive_ece", "calibration_mce"], default="ece")
    parser.add_argument("--training-method", default="baseline_cross_entropy")
    parser.add_argument("--pattern", default="ablation_*.json")
    parser.add_argument("--relative-margin", type=float, default=0.0, help="Relative TOST margin multiplier.")
    parser.add_argument(
        "--absolute-margin",
        type=float,
        default=0.005,
        help="Minimum/fixed TOST margin in metric units. Default 0.005 = 0.5 percentage points.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-pairs", type=int, default=2)
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.5,
        help="Minimum base accuracy required for a seed to be included. Values >1 are treated as percentages.",
    )
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or f"tab:rgcc_rgcl_tost_{args.metric_field}"
    min_accuracy = args.min_accuracy / 100.0 if args.min_accuracy > 1.0 else args.min_accuracy
    collect_pairs.min_accuracy = min_accuracy

    grouped, skipped_counts = collect_pairs(args.input_dir, args.pattern, args.training_method, args.metric_field)
    rows = build_rows(grouped, args.relative_margin, args.absolute_margin, args.alpha, args.min_pairs)
    if not rows:
        raise SystemExit("No paired RGCC/RGCL rows found. Check input directory, metric field, and min-pairs.")

    csv_path = args.output_dir / f"tost_rgcc_vs_rgcl_{args.metric_field}.csv"
    tex_path = args.output_dir / f"tost_rgcc_vs_rgcl_{args.metric_field}.tex"
    write_csv(rows, csv_path)
    write_latex(rows, tex_path, args.metric_field, label)

    total_pairs = sum(row["n"] for row in rows)
    skipped_configs = sum(1 for pairs in grouped.values() if len(pairs) < args.min_pairs)
    print(f"Wrote TOST CSV: {csv_path}")
    print(f"Wrote TOST LaTeX table: {tex_path}")
    print(
        f"Rows: {len(rows)}; paired seeds used: {total_pairs}; "
        f"skipped configs below min-pairs: {skipped_configs}; min accuracy: {min_accuracy * 100:.1f}%"
    )
    print(
        "Skipped files: "
        f"low_accuracy={skipped_counts['low_accuracy']}, "
        f"missing_pair={skipped_counts['missing_pair']}, "
        f"wrong_training_method={skipped_counts['wrong_training_method']}, "
        f"unreadable={skipped_counts['unreadable']}"
    )
    if not HAS_SCIPY:
        print("[INFO] scipy not available; used built-in Student-t fallback for TOST p-values and CIs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
