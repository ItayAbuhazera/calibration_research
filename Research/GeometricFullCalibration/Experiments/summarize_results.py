"""
Aggregate summary_metrics.csv across seeds and print a ranked table
with 95% confidence intervals (t-distribution, two-tailed).

Usage:
    python Experiments/summarize_results.py \
        --dataset cifar100 --model resnet18 --run_name decision_run_v1 \
        [--seeds 20 21 22 25] [--min_seeds 2] [--sort_by nll]
"""

import argparse
import glob
import json
import os
import sys

import math

import numpy as np
import pandas as pd

# ── Metrics to report (name, display, lower_is_better) ────────────────────────
METRICS = [
    ("nll",              "NLL",          True),
    ("top_label_ece",    "ECE",          True),
    ("adaptive_ece",     "AdaECE",       True),
    ("brier",            "Brier",        True),
    ("accuracy",         "Acc",          False),
    ("da_accuracy_delta","DA-delta",     False),
]

REQUIRED_STAGES = {"stage1_splits", "stage2_early", "stage3_features_fusion", "stage4_late_outputs"}

# Methods to always exclude from the display table (internal / diagnostic)
EXCLUDE = {
    "gc_dac",           # anchor, not a calibrator
    "trust_score_original_diagnostic",
}


def _is_complete(output_dir: str) -> bool:
    p = os.path.join(output_dir, "intermediates", "progress.json")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        progress = json.load(f)
    return REQUIRED_STAGES.issubset(set(progress.get("completed_stages", [])))


def _t_ppf_975(df: int) -> float:
    """Two-tailed 97.5th percentile of t(df) via a lookup table for small df,
    falling back to the normal approximation (1.96) for df >= 30."""
    table = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,
             6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,
             11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,
             16:2.120,17:2.110,18:2.101,19:2.093,20:2.086,
             25:2.060,29:2.045}
    if df in table:
        return table[df]
    if df >= 30:
        return 1.960
    # linear interpolation between nearest known values
    keys = sorted(table)
    for i in range(len(keys) - 1):
        if keys[i] < df < keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            w = (df - lo) / (hi - lo)
            return table[lo] + w * (table[hi] - table[lo])
    return 1.960


def _ci95(values: np.ndarray):
    """Return (mean, half-width) using the t-distribution."""
    n = len(values)
    if n < 2:
        return float(values[0]), float("nan")
    se = float(np.std(values, ddof=1) / math.sqrt(n))
    t = _t_ppf_975(n - 1)
    return float(np.mean(values)), float(t * se)


def load_all(base_dir: str, dataset: str, model: str, run_name: str,
             seeds: list[int] | None, min_seeds: int) -> pd.DataFrame:
    pattern = os.path.join(
        base_dir, "results", "unified_benchmark", dataset, model,
        "seed*", run_name, "summary_metrics.csv",
    )
    files = sorted(glob.glob(pattern))

    rows = []
    used_seeds = []
    for f in files:
        seed_str = f.split(os.sep)[-3].replace("seed", "")
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        if seeds is not None and seed not in seeds:
            continue
        run_dir = os.path.dirname(f)
        if not _is_complete(run_dir):
            continue
        df = pd.read_csv(f, low_memory=False)
        df["seed"] = seed
        rows.append(df)
        used_seeds.append(seed)

    if not rows:
        sys.exit("No complete seed runs found.")

    print(f"Using {len(used_seeds)} complete seeds: {sorted(used_seeds)}")
    return pd.concat(rows, ignore_index=True), sorted(used_seeds)


def build_table(df: pd.DataFrame, min_seeds: int) -> pd.DataFrame:
    metric_cols = [m[0] for m in METRICS]
    keep = ["method_name", "seed"] + [c for c in metric_cols if c in df.columns]
    sub = df[keep].copy()

    # coerce metrics to numeric
    for c in metric_cols:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

    records = []
    for method, grp in sub.groupby("method_name"):
        if method in EXCLUDE:
            continue
        n_seeds = grp["seed"].nunique()
        if n_seeds < min_seeds:
            continue
        rec = {"method": method, "n_seeds": n_seeds}
        for col, label, _ in METRICS:
            if col not in grp.columns:
                rec[f"{label}_mean"] = float("nan")
                rec[f"{label}_ci"]   = float("nan")
                continue
            vals = grp[col].dropna().values
            if len(vals) == 0:
                rec[f"{label}_mean"] = float("nan")
                rec[f"{label}_ci"]   = float("nan")
            else:
                mean, ci = _ci95(vals)
                rec[f"{label}_mean"] = mean
                rec[f"{label}_ci"]   = ci
        records.append(rec)

    return pd.DataFrame(records)


def format_cell(mean, ci, lower_is_better: bool, best_mean: float) -> str:
    if np.isnan(mean):
        return "   —   "
    arrow = "↓" if lower_is_better else "↑"
    ci_str = f"±{ci:.4f}" if not np.isnan(ci) else ""
    mark = " *" if abs(mean - best_mean) < 1e-9 else ""
    return f"{mean:.4f} {ci_str}{mark}"


def print_table(tbl: pd.DataFrame, sort_by: str, top_n: int) -> None:
    sort_col_mean = f"{sort_by}_mean"
    if sort_col_mean not in tbl.columns:
        sys.exit(f"Sort column '{sort_by}' not found.")

    lower_is_better_map = {label: lib for _, label, lib in METRICS}
    sort_ascending = lower_is_better_map.get(sort_by, True)
    tbl = tbl.sort_values(sort_col_mean, ascending=sort_ascending).reset_index(drop=True)

    if top_n > 0:
        tbl = tbl.head(top_n)

    # Find best per metric
    best = {}
    for _, label, lib in METRICS:
        col = f"{label}_mean"
        if col in tbl.columns:
            valid = tbl[col].dropna()
            if not valid.empty:
                best[label] = valid.min() if lib else valid.max()

    # Build display
    header_metrics = [label for _, label, _ in METRICS]
    col_w = 22
    name_w = 42

    sep = "-" * (name_w + col_w * len(header_metrics) + 10)
    header = f"{'Method':<{name_w}}" + "".join(f"{h:^{col_w}}" for h in header_metrics)
    print()
    print(sep)
    print(header)
    print(sep)

    for _, row in tbl.iterrows():
        cells = []
        for _, label, lib in METRICS:
            mean = row.get(f"{label}_mean", float("nan"))
            ci   = row.get(f"{label}_ci",   float("nan"))
            cell = format_cell(mean, ci, lib, best.get(label, float("nan")))
            cells.append(f"{cell:^{col_w}}")
        name = str(row["method"])[:name_w]
        print(f"{name:<{name_w}}" + "".join(cells))

    print(sep)
    print("* = best across displayed methods   ↓ = lower is better   ↑ = higher is better")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",   default="cifar100")
    ap.add_argument("--model",     default="resnet18")
    ap.add_argument("--run_name",  default="decision_run_v1")
    ap.add_argument("--seeds",     type=int, nargs="*", default=None,
                    help="Specific seeds to include (default: all complete)")
    ap.add_argument("--min_seeds", type=int, default=2,
                    help="Minimum seeds a method must appear in to be shown")
    ap.add_argument("--sort_by",   default="nll",
                    choices=["NLL","ECE","AdaECE","Brier","Acc","DA-delta",
                             "nll","top_label_ece","adaptive_ece","brier","accuracy","da_accuracy_delta"])
    ap.add_argument("--top_n",     type=int, default=0,
                    help="Show only top N methods (0 = all)")
    ap.add_argument("--base_dir",  default=".")
    args = ap.parse_args()

    # normalise sort_by to label form
    alias = {
        "nll": "NLL", "top_label_ece": "ECE", "adaptive_ece": "AdaECE",
        "brier": "Brier", "accuracy": "Acc", "da_accuracy_delta": "DA-delta",
    }
    sort_label = alias.get(args.sort_by, args.sort_by)

    df, used_seeds = load_all(
        args.base_dir, args.dataset, args.model, args.run_name,
        args.seeds, args.min_seeds,
    )

    tbl = build_table(df, args.min_seeds)
    print(f"\nDataset: {args.dataset}  Model: {args.model}  Run: {args.run_name}")
    print(f"Sorted by: {sort_label}   Min seeds: {args.min_seeds}")
    print_table(tbl, sort_label, args.top_n)


if __name__ == "__main__":
    main()
