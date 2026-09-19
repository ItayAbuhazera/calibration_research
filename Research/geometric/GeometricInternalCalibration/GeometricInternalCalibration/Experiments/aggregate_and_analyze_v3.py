#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_and_analyze_v3.py

Adds the missing P0 analyses:
  - Oracle comparison (ensemble vs. best single layer per run)
  - Learned-weight stability analysis (entropy/effective size, Gini, per-layer stats)
  - Metric failure diagnosis (gap over oracle; stratified)

Also keeps v2 improvements:
  - Safe pairing by (dataset, model, seed, training_method)
  - Keeps metric variants (__min/__max)
  - Extracts weighting hyper-params (e.g., _a1.5)
  - Canonical labels and richer effect sizes (Cliff's delta)
  - Optional multiple-testing correction (Holm) on baseline comparisons
  - Publication-ready CSVs + (optional) basic plots

USAGE
-----
python aggregate_and_analyze_v3.py \
  --results-dir /home/ptamar/geometric-internal-calibration/aaai_full_experiments/results/multi_layer_ensemble_v1 \
  --output-dir  /home/ptamar/geometric-internal-calibration/aaai_full_experiments/analysis_summary_v3 \
  --oracle-threshold 0.01 \
  --plot
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import kendalltau

# Optional: multiple testing correction (Holm)
try:
    from statsmodels.stats.multitest import multipletests
    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAVE_PLOTTING = True
except Exception:
    HAVE_PLOTTING = False


# -----------------------------
# Utilities
# -----------------------------
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's delta effect size for two independent samples a, b.
    Returns delta in [-1, 1]. Positive means 'a' > 'b' more often.
    Here we typically compute delta(oracle, best) or delta(baseline, best).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    # Pairwise comparisons
    gt = 0
    lt = 0
    for ai in a:
        gt += np.sum(ai > b)
        lt += np.sum(ai < b)
    n = len(a) * len(b)
    if n == 0:
        return np.nan
    return round((gt - lt) / n, 7)


def mean_ci(x: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Mean and (lo, hi) confidence interval via t-distribution."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        return round(x[0], 7), np.nan, np.nan
    mean = float(np.mean(x))
    se = stats.sem(x)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return round(mean, 7), round(mean - h, 7), round(mean + h, 7)


def parse_weighting_param_from_key(key: str) -> Optional[float]:
    """
    Extracts a numeric parameter from keys like 'top3_deep_a1.5' -> 1.5
    """
    m = re.search(r"_a([0-9]*\.?[0-9]+)", key)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def parse_context_from_path(fp: Path) -> Dict[str, Any]:
    """
    Given a path .../k_3/<training_method>/<dataset>/<model>/seed17/multi_layer_ensemble/ensemble_results.json
    returns dict of context fields + directories.
    """
    parts = list(fp.parts)
    # Find seed index
    seed_idx = None
    for i, p in enumerate(parts):
        if p.startswith("seed"):
            seed_idx = i
            break
    if seed_idx is None or seed_idx < 3:
        return {}
    seed_str = parts[seed_idx]
    try:
        seed = int(seed_str.replace("seed", ""))
    except Exception:
        seed = None

    # k_* is somewhere before training_method
    k_val = None
    k_idx = None
    for i, p in enumerate(parts):
        if re.fullmatch(r"k_\d+", p):
            k_idx = i
            try:
                k_val = int(p.replace("k_", ""))
            except Exception:
                k_val = None
            break

    training_method = parts[seed_idx - 3] if seed_idx - 3 >= 0 else None
    dataset = parts[seed_idx - 2] if seed_idx - 2 >= 0 else None
    model = parts[seed_idx - 1] if seed_idx - 1 >= 0 else None

    seed_dir = Path(*parts[: seed_idx + 1])  # include 'seedXX'
    run_dir = fp.parent

    return dict(
        training_method=training_method,
        dataset=dataset,
        model=model,
        seed=seed,
        k=k_val,
        seed_dir=str(seed_dir),
        run_dir=str(run_dir),
    )


# -----------------------------
# Data ingestion
# -----------------------------
def _find_metric_docs_dir(results_json_path: Path, seed_dir: Path) -> Optional[Path]:
    """
    Given .../multi_layer_ensemble/k_3/a_1.5/ensemble_results.json, return
    .../multi_layer_ensemble/metric_docs if it exists. Also check a few
    compatible legacy locations.
    """
    # climb up to .../multi_layer_ensemble
    multi_dir = None
    for p in results_json_path.parents:
        if p.name == "multi_layer_ensemble":
            multi_dir = p
            break
    candidates = []
    if multi_dir is not None:
        candidates.append(multi_dir / "metric_docs")               # new layout
    candidates.append(Path(seed_dir) / "multi_layer_ensemble" / "metric_docs")  # explicit
    candidates.append(Path(seed_dir) / "metric_docs")                          # ultra-legacy
    # Also try a couple of relative jumps just in case
    candidates += [results_json_path.parent.parent / "metric_docs",
                   results_json_path.parent.parent.parent / "metric_docs"]
    for d in candidates:
        if d.exists():
            return d
    return None


def find_and_parse_results(root_dir: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    # Normalize baseline names to a canonical set
    BASELINE_RENAMES = {
        "none": "uncalibrated",
        "no_calibration": "uncalibrated",
    }

    result_files = list(root_dir.rglob("ensemble_results.json"))
    print(f"Found {len(result_files)} 'ensemble_results.json' files.")

    for f in result_files:
        try:
            ctx = parse_context_from_path(f)
            if not ctx or ctx.get("seed") is None:
                continue

            data = json.loads(f.read_text())

            # ---- baselines ----
            if "baselines" in data and isinstance(data["baselines"], dict):
                for name, metrics in data["baselines"].items():
                    if not metrics:
                        continue
                    # tolerant baseline naming
                    name = BASELINE_RENAMES.get(name, name)
                    ece = metrics.get("ece", None)
                    if ece is None:
                        continue
                    records.append({
                        **ctx,
                        "result_type": "baseline",
                        "selection": "baseline",
                        "method_name": name,
                        "full_method": name,  # canonical for baselines
                        "weighting": None,
                        "weight_param": None,
                        "metric_name": None,
                        "metric_variant": None,
                        "ece": round(float(ece), 7),
                        "brier": round(float(metrics.get("brier")), 7) if metrics.get("brier") is not None else None,
                        "accuracy": round(float(metrics.get("acc")), 7) if metrics.get("acc") is not None else None,
                        "selected_layers": None,
                        "weights_json": None,
                    })

            # ---- topK ensembles (e.g., 'top3_deep_a1.5', 'top3_learned', 'top3_uniform', ...)
            #      also include K=0 keys written as 'K0_*' ----
            for key, result in data.items():
                if not isinstance(result, dict):
                    continue
                if key.startswith(("top", "all", "K0")):
                    ece = result.get("test_ece", None)
                    if ece is None:
                        continue
                    weighting = result.get("weighting_method", None)
                    weight_param = parse_weighting_param_from_key(key)
                    # prefer canonical depth order if present
                    selected_layers = (result.get("selected_layers_canonical")
                                       or result.get("selected_layers"))
                    if selected_layers:
                        try:
                            selected_layers = tuple(int(x) for x in selected_layers)
                        except Exception:
                            selected_layers = tuple(selected_layers)
                    weights = result.get("weights", None)
                    records.append({
                        **ctx,
                        "result_type": "ensemble",
                        "selection": "topk",
                        "method_name": key,
                        "full_method": None,  # set later with canon_label
                        "weighting": weighting,
                        "weight_param": weight_param,
                        "metric_name": None,
                        "metric_variant": None,
                        "ece": round(float(ece), 7),
                        "brier": round(float(result.get("test_brier")), 7) if result.get("test_brier") is not None else None,
                        "accuracy": round(float(result.get("test_accuracy")), 7) if result.get("test_accuracy") is not None else None,
                        "selected_layers": tuple(selected_layers) if selected_layers else None,
                        "ees": round(float(result.get("ees")), 7) if result.get("ees") is not None else None,
                        "weights_json": json.dumps(weights) if weights is not None else None,
                    })

            # ---- metric picks file (robust path) ----
            md_dir = _find_metric_docs_dir(f, Path(ctx["seed_dir"]))
            if md_dir is not None:
                metric_picks_file = md_dir / "metric_picks_test_eval.json"
                if metric_picks_file.exists():
                    md = json.loads(metric_picks_file.read_text())
                    by_metric = md.get("by_metric", {})
                    for metric_name, k_dict in by_metric.items():
                        base_metric = metric_name
                        variant = None
                        if metric_name.endswith("__min"):
                            base_metric = metric_name.replace("__min", "")
                            variant = "min"
                        elif metric_name.endswith("__max"):
                            base_metric = metric_name.replace("__max", "")
                            variant = "max"
                        for k_key, w_dict in k_dict.items():  # e.g., "K3"
                            try:
                                current_k = int(k_key.replace("K", ""))
                            except Exception:
                                continue
                            for weighting_name, w_res in w_dict.items():
                                if not isinstance(w_res, dict):
                                    continue
                                ece = w_res.get("test_ece", None)
                                if ece is None:
                                    continue
                                # prefer canonical order if present
                                selected_layers = (w_res.get("selected_layers_canonical")
                                                   or w_res.get("selected_layers"))
                                if selected_layers:
                                    try:
                                        selected_layers = tuple(int(x) for x in selected_layers)
                                    except Exception:
                                        selected_layers = tuple(selected_layers)
                                weights = w_res.get("weights", None)
                                records.append({
                                    **ctx,
                                    "k": current_k,
                                    "result_type": "ensemble",
                                    "selection": "metric",
                                    "method_name": f"metric:{metric_name}_{weighting_name}",
                                    "full_method": None,  # set later
                                    "weighting": weighting_name,
                                    "weight_param": None,
                                    "metric_name": base_metric,
                                    "metric_variant": variant,
                                    "ece": round(float(ece), 7),
                                    "brier": round(float(w_res.get("test_brier")), 7) if w_res.get("test_brier") is not None else None,
                                    "accuracy": round(float(w_res.get("test_accuracy")), 7) if w_res.get("test_accuracy") is not None else None,
                                    "selected_layers": tuple(selected_layers) if selected_layers else None,
                                    "ees": round(float(w_res.get("ees")), 7) if w_res.get("ees") is not None else None,
                                    "weights_json": json.dumps(weights) if weights is not None else None,
                                })

        except Exception as e:
            print(f"[WARN] Failed to parse {f}: {e}")

    if not records:
        raise RuntimeError("No records parsed. Check the --results-dir and file structure.")

    df = pd.DataFrame.from_records(records)

    # Canonical label
    def canon_label(r):
        if r["selection"] == "metric":
            suf = f"__{r['metric_variant']}" if pd.notna(r["metric_variant"]) and r["metric_variant"] else ""
            return f"metric:{r['metric_name']}{suf}_{r['weighting']}"
        elif r["selection"] == "topk":
            wp = "" if pd.isna(r["weight_param"]) or r["weight_param"] is None else f"_{r['weight_param']}"
            return f"topk:{r['weighting']}{wp}"
        else:
            return r["method_name"]

    df["full_method"] = df.apply(canon_label, axis=1)
    return df


# -----------------------------
# Oracle loader
# -----------------------------
def load_oracle_best_ece(seed_dir: Path) -> Optional[float]:
    """
    More forgiving oracle loader. Returns float test ECE or None.
    
    This version is MODIFIED to check for 'multi_composite_analysis.json'
    from the layer_selection_analysis experiments, which may be symlinked.
    """
    seed_dir = Path(seed_dir)

    def _maybe(v):
        try:
            v = float(v)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    # (0) NEW: Check for the linked 'multi_composite_analysis.json' file first
    # This is the file you are linking from layer_selection_analysis_baselines_4
    for c in seed_dir.rglob("multi_composite_analysis.json"):
        try:
            data = json.loads(c.read_text())
            # This file uses 'empirical_best_ece'
            v = data.get("empirical_best_ece")
            v = _maybe(v)
            if v is not None:
                return round(v, 7)
        except Exception:
            pass # Failed to parse, try next method

    # (1) direct oracle files (several common names)
    for name in ["oracle_best_layer.json", "best_single_layer.json", "best_layer.json"]:
        for c in seed_dir.rglob(name):
            try:
                data = json.loads(c.read_text())
                # accept both test_ece and ece fields
                v = data.get("test_ece", data.get("ece"))
                v = _maybe(v)
                if v is not None:
                    return round(v, 7)
            except Exception:
                pass

    # (2) per-layer ground-truth with multiple shapes
    for c in seed_dir.rglob("per_layer_ground_truth.json"):
        try:
            data = json.loads(c.read_text())
            if isinstance(data, list):  # [{"layer_idx":..., "test_ece":...}, ...]
                vals = [_maybe(d.get("test_ece", d.get("ece"))) for d in data if isinstance(d, dict)]
                vals = [v for v in vals if v is not None]
                if vals:
                    return round(min(vals), 7)
            elif isinstance(data, dict):
                # try nested test.ece map OR a flat "ece" dict
                test_ece_map = (data.get("test", {}) or {}).get("ece", {})
                if isinstance(test_ece_map, dict) and test_ece_map:
                    vals = [_maybe(v) for v in test_ece_map.values()]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        return round(min(vals), 7)
                if isinstance(data.get("ece"), dict):
                    vals = [_maybe(v) for v in data["ece"].values()]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        return round(min(vals), 7)
        except Exception:
            pass

    # (3) embedded oracle in any ensemble_results.json under the seed
    for c in seed_dir.rglob("ensemble_results.json"):
        try:
            data = json.loads(c.read_text())
            # This is the key your 'inject' script was trying to add
            if isinstance(data, dict) and "oracle_best_ece" in data:
                v = _maybe(data.get("oracle_best_ece"))
                if v is not None:
                    return round(v, 7)
            # This is the fallback logic from the original script
            if isinstance(data, dict) and "oracle_best_layer" in data:
                cand = data["oracle_best_layer"] or {}
                v = _maybe(cand.get("test_ece", cand.get("ece")))
                if v is not None:
                    return round(v, 7)
        except Exception:
            pass

    return None


# -----------------------------
# Analyses
# -----------------------------
def analyze_best_metrics(df: pd.DataFrame, k_val: float) -> pd.DataFrame:
    """
    For K=k_val, rank metric-selected ensembles by mean ECE (+ 95% CI).
    """
    sub = df[(df["k"] == k_val) & (df["selection"] == "metric")].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for m, g in sub.groupby("full_method"):
        mu, lo, hi = mean_ci(g["ece"].values)
        rows.append({
            "Method": m,
            "Mean ECE": round(mu, 7),
            "95% CI Lower": round(lo, 7),
            "95% CI Upper": round(hi, 7),
            "Std Dev": round(float(np.std(g["ece"].values)), 7),
            "Runs": int(len(g)),
        })
    out = pd.DataFrame(rows).sort_values("Mean ECE", ascending=True).reset_index(drop=True)
    return out


def compare_to_baselines(df: pd.DataFrame, best_method_name: str, k_val: float, mtest_correction: bool = True) -> pd.DataFrame:
    """
    Compare a chosen ensemble method (full_method == best_method_name) to all baselines
    using paired runs on (dataset, model, seed, training_method).
    Reports win-rate, median ΔECE, Wilcoxon p-value, and Cliff's delta.
    """
    on_cols = ["dataset", "model", "seed", "training_method"]

    best_rows = df[(df["k"] == k_val) & (df["full_method"] == best_method_name)].copy()
    base = df[df["result_type"] == "baseline"].copy()

    if best_rows.empty or base.empty:
        return pd.DataFrame(columns=["Baseline", "Win Rate vs Best Ensemble", "Median ΔECE (base-best)", "Wilcoxon P-Value", "Cliff's delta", "N"])

    merged = pd.merge(best_rows[on_cols + ["ece"]],
                      base[on_cols + ["method_name", "ece"]],
                      on=on_cols, how="inner", suffixes=("_best", "_base"))

    comps = []
    for bname, g in merged.groupby("method_name"):
        # lower is better
        wins = int((g["ece_best"] < g["ece_base"]).sum())
        ties = int((g["ece_best"] == g["ece_base"]).sum())
        total = int(len(g))
        win_rate = (wins + 0.5 * ties) / total if total > 0 else np.nan
        diff = g["ece_base"].values - g["ece_best"].values  # positive => best better

        # Wilcoxon: is best < baseline? i.e., base - best > 0
        try:
            if len(diff) > 0 and np.any(diff != 0):
                wstat, p = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
            else:
                p = np.nan
        except Exception:
            p = np.nan

        cd = cliffs_delta(g["ece_base"].values, g["ece_best"].values)  # positive => baseline > best
        comps.append({
            "Baseline": bname,
            "Win Rate vs Best Ensemble": round(win_rate, 7),
            "Median ΔECE (base-best)": round(float(np.median(diff)), 7) if len(diff) else np.nan,
            "Wilcoxon P-Value": round(p, 7) if not np.isnan(p) else np.nan,
            "Cliff's delta": round(cd, 7) if not np.isnan(cd) else np.nan,
            "N": total,
        })

    out = pd.DataFrame(comps).sort_values("Win Rate vs Best Ensemble", ascending=False).reset_index(drop=True)

    if mtest_correction and HAVE_STATSMODELS and len(out) > 0:
        try:
            pvals = out["Wilcoxon P-Value"].values
            mask = np.isfinite(pvals)
            corrected = np.full_like(pvals, np.nan, dtype=float)
            if np.any(mask):
                _, p_corr, _, _ = multipletests(pvals[mask], method="holm")
                corrected[mask] = p_corr
            out["Wilcoxon P-Value (Holm)"] = corrected
        except Exception:
            pass

    return out


def analyze_layer_combinations(df: pd.DataFrame, method_name: str, k_val: float) -> Counter:
    sub = df[(df["k"] == k_val) & (df["full_method"] == method_name)]
    combos = [c for c in sub["selected_layers"].dropna().tolist() if isinstance(c, tuple)]
    return Counter(combos)


def compare_ensemble_vs_oracle(df: pd.DataFrame, k: float, method_name: str) -> Dict[str, Any]:
    """
    Compare selected ensemble method to oracle 'best single layer' for each run.
    Returns summary statistics.
    """
    ens = df[(df["k"] == k) & (df["full_method"] == method_name)].copy()
    if ens.empty or "oracle_ece" not in ens.columns:
        return {"error": "No runs or missing oracle_ece column."}

    valid = ens.dropna(subset=["ece", "oracle_ece"]).copy()
    if valid.empty:
        return {"error": "No valid oracle rows."}

    improvements = valid["oracle_ece"].values - valid["ece"].values  # >0 => ensemble better than oracle
    wins = int((valid["ece"].values < valid["oracle_ece"].values).sum())
    ties = int((valid["ece"].values == valid["oracle_ece"].values).sum())
    total = int(len(valid))

    mu, lo, hi = mean_ci(improvements)
    try:
        _, p = stats.wilcoxon(improvements, alternative="greater", zero_method="wilcox")
    except Exception:
        p = np.nan

    cd = cliffs_delta(valid["oracle_ece"].values, valid["ece"].values)
    return {
        "K": k,
        "Method": method_name,
        "N": total,
        "WinRate_vs_Oracle": round((wins + 0.5 * ties) / total, 7),
        "MeanImprovement": round(mu, 7),
        "CI95_lo": round(lo, 7),
        "CI95_hi": round(hi, 7),
        "MedianImprovement": round(float(np.median(improvements)), 7),
        "WilcoxonP_greater": round(p, 7) if not np.isnan(p) else np.nan,
        "CliffsDelta (oracle-best)": round(cd, 7) if not np.isnan(cd) else np.nan,
    }


def analyze_weight_distributions(df: pd.DataFrame, k: float, weighting: str = "learned") -> Dict[str, Any]:
    """
    Examine learned weights across runs. Computes per-layer stats, effective size, and Gini.
    Returns a dict with summary and a layer_stats dataframe.
    """
    learned = df[(df["k"] == k) & (df["selection"].isin(["metric", "topk"])) & (df["weighting"] == weighting) & df["weights_json"].notna()].copy()
    if learned.empty:
        return {"error": "No learned weights found."}

    weight_records = []
    eff_sizes = []
    ginis = []

    def eff_size_from_weights(w_dict: Dict[str, float]) -> float:
        w = np.array(list(w_dict.values()), dtype=float)
        w = w / (w.sum() + 1e-12)
        entropy = -np.sum(w * np.log(w + 1e-12))
        return round(float(np.exp(entropy)), 7)

    def gini_from_weights(w_dict: Dict[str, float]) -> float:
        w = np.array(sorted(w_dict.values()), dtype=float)
        if np.sum(w) <= 0:
            return np.nan
        n = len(w)
        # Gini for positive weights
        cumw = np.cumsum(w)
        # Using a standard discrete Gini formula with 1-indexing adjustment
        g = (2 * np.sum((np.arange(1, n + 1) * w)) / (n * np.sum(w))) - (n + 1) / n
        return round(float(g), 7)

    for _, row in learned.iterrows():
        try:
            weights = json.loads(row["weights_json"])
            eff_sizes.append(eff_size_from_weights(weights))
            ginis.append(gini_from_weights(weights))
            for layer_idx, w in weights.items():
                weight_records.append({
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "seed": row["seed"],
                    "layer_idx": int(layer_idx),
                    "weight": round(float(w), 7),
                    "run_ece": round(float(row["ece"]), 7),
                    "full_method": row["full_method"],
                })
        except Exception:
            continue

    if not weight_records:
        return {"error": "No parseable weight records."}

    wdf = pd.DataFrame(weight_records)
    layer_stats = (
        wdf.groupby("layer_idx")["weight"]
           .agg(["mean", "std", "min", "max", "count"])
           .sort_index()
    )
    # Round the numeric columns to 7 decimal places
    for col in ["mean", "std", "min", "max"]:
        if col in layer_stats.columns:
            layer_stats[col] = layer_stats[col].round(7)

    return {
        "K": k,
        "N_runs": int(len(learned)),
        "mean_effective_size": round(float(np.nanmean(eff_sizes)), 7) if eff_sizes else np.nan,
        "std_effective_size": round(float(np.nanstd(eff_sizes)), 7) if eff_sizes else np.nan,
        "mean_gini": round(float(np.nanmean(ginis)), 7) if ginis else np.nan,
        "median_gini": round(float(np.nanmedian(ginis)), 7) if ginis else np.nan,
        "layer_stats": layer_stats,
        "weights_long": wdf,
    }


def diagnose_metric_failures(df: pd.DataFrame, k: float, threshold: float = 0.01) -> pd.DataFrame:
    """
    For metric-selected ensembles at K, compute failure rate vs oracle:
      failure = 1 if ECE > oracle + threshold
    Produces both overall metric stats and dataset-stratified rows.
    """
    sub = df[(df["k"] == k) & (df["selection"] == "metric")].copy()
    if sub.empty or "oracle_ece" not in sub.columns:
        return pd.DataFrame()

    valid = sub.dropna(subset=["ece", "oracle_ece"]).copy()
    if valid.empty:
        return pd.DataFrame()

    valid["failure"] = (valid["ece"] > valid["oracle_ece"] + float(threshold)).astype(int)
    valid["gap"] = valid["ece"] - valid["oracle_ece"]

    rows = []
    # Aggregate by metric (with variant) overall
    for (mname, mvar), g in valid.groupby(["metric_name", "metric_variant"]):
        fullname = f"{mname}__{mvar}" if pd.notna(mvar) and mvar else f"{mname}"
        rows.append({
            "Metric": fullname,
            "Level": "overall",
            "Group": "all",
            "FailureRate": round(float(g["failure"].mean()), 7),
            "MeanGap": round(float(g["gap"].mean()), 7),
            "MedianGap": round(float(g["gap"].median()), 7),
            "N": int(len(g)),
        })
        # Per dataset
        for dname, dg in g.groupby("dataset"):
            rows.append({
                "Metric": fullname,
                "Level": "dataset",
                "Group": dname,
                "FailureRate": round(float(dg["failure"].mean()), 7),
                "MeanGap": round(float(dg["gap"].mean()), 7),
                "MedianGap": round(float(dg["gap"].median()), 7),
                "N": int(len(dg)),
            })
    if not rows:
        return pd.DataFrame(columns=["Metric", "Level", "Group", "FailureRate", "MeanGap", "MedianGap", "N"])
    out = pd.DataFrame(rows).sort_values(["Metric", "Level", "Group"]).reset_index(drop=True)
    return out


# -----------------------------
# New: Metrics vs Baselines (win-rates, deltas, means) + accuracy bins
# -----------------------------
def _paired_vs_baseline_table(lhs: pd.DataFrame, base: pd.DataFrame, on_cols, lhs_label_col: str) -> pd.DataFrame:
    """
    Generic paired comparison of a left-hand set of rows (lhs) against all baselines on shared runs.
    Returns long-form rows per (lhs_label, baseline).
    """
    rows = []
    merged = pd.merge(lhs[on_cols + [lhs_label_col, "ece"]],
                      base[on_cols + ["method_name", "ece"]],
                      on=on_cols, how="inner", suffixes=("_lhs", "_base"))
    if merged.empty:
        return pd.DataFrame(columns=[lhs_label_col, "Baseline", "Win Rate", "Median ΔECE (base-lhs)",
                                     "Mean ΔECE", "CI95_lo", "CI95_hi", "Wilcoxon P (base>lhs)", 
                                     "Mean ECE (lhs)", "Mean ECE (base)", "N", "Mean(lhs)≤Mean(base)"])
    for (label, bname), g in merged.groupby([lhs_label_col, "method_name"]):
        lhs = g["ece_lhs"].values
        bse = g["ece_base"].values
        diff = bse - lhs  # positive => lhs better (lower ECE)
        wins = int((lhs < bse).sum())
        ties = int((lhs == bse).sum())
        total = int(len(g))
        win_rate = (wins + 0.5 * ties) / total if total > 0 else np.nan
        mu, lo, hi = mean_ci(diff)
        try:
            # H1: baseline - lhs > 0 (lhs better)
            _, p = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
        except Exception:
            p = np.nan
        rows.append({
            lhs_label_col: label,
            "Baseline": bname,
            "Win Rate": round(win_rate, 7),
            "Median ΔECE (base-lhs)": round(float(np.median(diff)), 7) if total else np.nan,
            "Mean ΔECE": round(mu, 7),
            "CI95_lo": round(lo, 7), "CI95_hi": round(hi, 7),
            "Wilcoxon P (base>lhs)": round(p, 7) if not np.isnan(p) else np.nan,
            "Mean ECE (lhs)": round(float(np.mean(lhs)), 7) if total else np.nan,
            "Mean ECE (base)": round(float(np.mean(bse)), 7) if total else np.nan,
            "N": total,
            "Mean(lhs)≤Mean(base)": round(float(np.mean(lhs)), 7) <= round(float(np.mean(bse)), 7) if total else np.nan,
        })
    return pd.DataFrame(rows).sort_values([lhs_label_col, "Baseline"]).reset_index(drop=True)


def metric_winrates_vs_baselines(df: pd.DataFrame, k: Optional[float] = None,
                                 by: Optional[str] = None) -> pd.DataFrame:
    """
    For ALL metric-selected methods (every metric/variant/weighting), compute win rates vs each baseline.
    Optional stratification by 'by' in {'dataset','model'}.
    """
    on_cols = ["dataset", "model", "seed", "training_method"]
    subm = df[df["selection"] == "metric"].copy()
    base = df[df["result_type"] == "baseline"].copy()
    if k is not None:
        subm = subm[subm["k"] == k]
    if subm.empty or base.empty:
        return pd.DataFrame()

    # Construct a compact label for the metric selector itself
    subm["metric_label"] = subm.apply(
        lambda r: f"{r['metric_name']}{'__'+r['metric_variant'] if pd.notna(r['metric_variant']) and r['metric_variant'] else ''}_{r['weighting']}", axis=1
    )

    if by is None:
        out = _paired_vs_baseline_table(subm.rename(columns={"metric_label": "Method"}),
                                        base, on_cols, lhs_label_col="Method")
        return out
    else:
        rows = []
        for key, g in subm.groupby(by):
            tab = _paired_vs_baseline_table(
                g.rename(columns={"metric_label": "Method"}), base[base[by] == key], on_cols, lhs_label_col="Method"
            )
            if not tab.empty:
                tab.insert(0, by, key)
                rows.append(tab)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_accuracy_bins(df: pd.DataFrame, edges: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Adds 'acc_bin' categorical column using accuracy in [0,1] with default edges:
    [0.70,0.75,0.80,0.85,0.90,0.95,1.01] -> labels '70–75', '75–80', ..., '95–100'
    Drops rows with missing/NaN accuracy.
    """
    if edges is None:
        edges = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
    labels = [f"{int(100*edges[i]):d}–{int(100*edges[i+1]):d}" for i in range(len(edges)-1)]
    df = df.copy()
    df = df[pd.notna(df["accuracy"])]
    df["acc_bin"] = pd.cut(df["accuracy"].astype(float), bins=edges, labels=labels, include_lowest=True, right=False)
    return df


def summarize_by_accuracy_bins(df: pd.DataFrame, k: float, methods: List[str]) -> pd.DataFrame:
    """
    For a fixed K and a set of 'methods' (strings in df['full_method'] for ensembles or baseline names in df['full_method']),
    compute mean/CI ECE per accuracy bin. Returns long-form table: acc_bin, method, mean_ece, ci_lo, ci_hi, N.
    """
    sub = df.copy()
    sub = add_accuracy_bins(sub)
    sub = sub[(sub["k"] == k) & sub["full_method"].isin(methods)]
    rows = []
    for (ab, m), g in sub.groupby(["acc_bin", "full_method"]):
        mu, lo, hi = mean_ci(g["ece"].values)
        rows.append({
            "K": k, "acc_bin": ab, "Method": m,
            "Mean ECE": round(mu, 7), "CI95_lo": round(lo, 7), "CI95_hi": round(hi, 7), "N": int(len(g))
        })
    return pd.DataFrame(rows).sort_values(["acc_bin", "Mean ECE"])


def plot_ece_vs_accbins(df: pd.DataFrame, k: float, methods: List[str], outpng: Path, title: Optional[str] = None):
    """Small line+errorbar plot of ECE vs accuracy bins for the chosen methods."""
    if not HAVE_PLOTTING:
        return
    tab = summarize_by_accuracy_bins(df, k, methods)
    if tab.empty:
        return
    try:
        plt.figure(figsize=(10, 6))
        # order bins numerically by their left edge:
        bin_order = sorted(tab["acc_bin"].dropna().unique(),
                           key=lambda s: float(str(s).split('–')[0]))  # crude but works for '70–75' labels
        for m, g in tab.groupby("Method"):
            g = g.sort_values("acc_bin")
            x = [bin_order.index(ab) for ab in g["acc_bin"]]
            plt.errorbar(x, g["Mean ECE"], yerr=[g["Mean ECE"]-g["CI95_lo"], g["CI95_hi"]-g["Mean ECE"]],
                         marker="o", linewidth=2, capsize=3, label=m)
        plt.xticks(ticks=range(len(bin_order)), labels=bin_order, rotation=0)
        plt.xlabel("Accuracy bin (%)")
        plt.ylabel("Mean ECE")
        plt.title(title or f"ECE vs Accuracy bins (K={k})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpng)
        plt.close()
    except Exception as e:
        print(f"[WARN] acc-bin plot failed: {e}")


# -----------------------------
# Plotting
# -----------------------------
def plot_weight_stability(weights_long: pd.DataFrame, outpath: Path, title: str):
    if not HAVE_PLOTTING:
        return
    if weights_long.empty:
        return

    # Create a matrix: rows=runs, columns=sorted unique layers
    # We'll pivot by (seed, full_method) to keep a consistent ordering per run
    try:
        piv = weights_long.pivot_table(index=["dataset", "model", "seed", "full_method"],
                                       columns="layer_idx", values="weight", fill_value=0.0)
        # Sort columns (layers)
        piv = piv.reindex(sorted(piv.columns), axis=1)
        plt.figure(figsize=(10, max(4, 0.25 * len(piv))))
        sns.heatmap(piv.values, cmap="viridis", cbar=True)
        plt.title(title)
        plt.xlabel("Layer index")
        plt.ylabel("Run")
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
    except Exception as e:
        print(f"[WARN] weight stability plot failed: {e}")


def plot_ece_distribution(df: pd.DataFrame, k: float, best_methods: List[str], outpath: Path, oracle_ref: Optional[float] = None):
    if not HAVE_PLOTTING:
        return
    sub = df[(df["k"] == k) & (df["full_method"].isin(best_methods))].copy()
    baselines = df[df["result_type"] == "baseline"]["full_method"].unique().tolist()
    add = df[df["full_method"].isin(baselines)].copy()
    plot_df = pd.concat([sub, add], ignore_index=True)

    if plot_df.empty:
        return

    plot_df["plot_name"] = plot_df["full_method"]

    plt.figure(figsize=(12, 8))
    sns.violinplot(data=plot_df, x="ece", y="plot_name", inner=None, cut=0, color=".85")
    sns.boxplot(data=plot_df, x="ece", y="plot_name", width=0.3)
    sns.stripplot(data=plot_df, x="ece", y="plot_name", jitter=0.12, alpha=0.55)

    if oracle_ref is not None and np.isfinite(oracle_ref):
        plt.axvline(oracle_ref, linestyle="--", linewidth=2, label="Oracle (mean)", color="red")
        plt.legend()

    plt.title(f"ECE distribution (K={k}) – top metric methods vs. baselines")
    plt.xlabel("Expected Calibration Error (ECE)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Additional analyses (A–F)
# -----------------------------
def _collect_metric_selector_diagnostics(
    root_df: pd.DataFrame,
    full_df: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Enhanced function to collect and analyze metric selector diagnostics.
    """
    diag_rows: List[Dict[str, Any]] = []
    metric_topk_data: Dict[str, Any] = {}

    try:
        # Get unique seed directories and context (dataset, model, seed)
        seed_contexts = (
            full_df[["seed_dir", "dataset", "model", "seed", "training_method", "accuracy"]]
            .drop_duplicates()
            .dropna(subset=["seed_dir"])
        )
        if seed_contexts.empty:
            print("[WARN] No valid seed directories found in full_df for diagnostics.")
            return None
        print(f"[INFO] Analyzing diagnostics for {len(seed_contexts)} unique runs.")

    except Exception as e:
        print(f"[WARN] Failed to extract seed contexts: {e}")
        return None

    # Mapping from seed_dir to its context
    seed_dir_to_context = {row['seed_dir']: row for _, row in seed_contexts.iterrows()}

    for sd_path_str in seed_contexts["seed_dir"].unique():
        sd_path = Path(sd_path_str)
        context = seed_dir_to_context.get(sd_path_str)
        if not context:
            continue

        # --- Load Diagnostics File ---
        probe = sd_path / "multi_layer_ensemble" / "ensemble_results.json"
        md_dir = _find_metric_docs_dir(probe, sd_path)
        if md_dir is None:
            continue
        diag_json_path = md_dir / "metric_selector_diagnostics.json"
        if not diag_json_path.exists():
            continue

        # --- Load metric_docs_and_topk.json for Top-K Hit Rate ---
        # Cache this data as it's the same per seed_dir
        topk_json_path = md_dir / "metric_docs_and_topk.json"
        if sd_path_str not in metric_topk_data:
            if topk_json_path.exists():
                try:
                    with open(topk_json_path, 'r') as f:
                        metric_topk_data[sd_path_str] = json.load(f)
                except Exception:
                    metric_topk_data[sd_path_str] = None
            else:
                metric_topk_data[sd_path_str] = None

        run_topk_data = metric_topk_data.get(sd_path_str)

        # --- Load Oracle Layer for Hit Rate ---
        try:
            run_oracle_ece = full_df[full_df['seed_dir'] == sd_path_str]['oracle_ece'].iloc[0]
            oracle_layer = None
            pl_ground_truth_path = md_dir / "per_layer_ground_truth.json"
            if pd.notna(run_oracle_ece) and pl_ground_truth_path.exists():
                try:
                    with open(pl_ground_truth_path, 'r') as f:
                        pl_data = json.load(f)
                    # Find the layer matching the oracle ECE
                    if isinstance(pl_data, list):
                        for layer_info in pl_data:
                            if np.isclose(layer_info.get('test_ece', np.inf), run_oracle_ece, atol=1e-6):
                                oracle_layer = layer_info.get('layer_idx')
                                break
                except Exception:
                    pass
        except Exception:
            oracle_layer = None

        # --- Process Diagnostics JSON ---
        try:
            j = json.loads(diag_json_path.read_text())
            per_metric_diag = j.get("per_metric", {})
            depth_bias = j.get("depth_bias_topK5", {})

            # --- NEW: Load Raw Metric Values for Kendall Tau ---
            all_metric_values = run_topk_data.get("all_metric_values", {}) if run_topk_data else {}
            holdout_ece_map = run_topk_data.get("holdout_ece_per_layer", {}) if run_topk_data else {}

            for m, v in per_metric_diag.items():
                base_metric = m.replace('__min','').replace('__max','')

                # --- NEW: Calculate Kendall Tau ---
                kendall_tau_val = np.nan
                kendall_p = np.nan
                metric_vals = all_metric_values.get(m)
                if metric_vals and holdout_ece_map:
                    layers = list(metric_vals.keys())
                    metric_scores = np.array([metric_vals[str(l)] for l in layers if str(l) in holdout_ece_map], dtype=float)
                    oracle_scores = np.array([holdout_ece_map[str(l)] for l in layers if str(l) in holdout_ece_map], dtype=float)

                    valid_mask = np.isfinite(metric_scores) & np.isfinite(oracle_scores)
                    if valid_mask.sum() >= 2:
                        try:
                            tau, p_val = kendalltau(metric_scores[valid_mask], oracle_scores[valid_mask])
                            kendall_tau_val = tau
                            kendall_p = p_val
                        except Exception:
                            pass

                # --- NEW: Calculate Top-K Hit Rate ---
                topk_layers: List[int] = []
                if run_topk_data and 'metrics_topk' in run_topk_data and m in run_topk_data['metrics_topk']:
                    topk_layers = run_topk_data['metrics_topk'][m].get('topk_layers', [])
                    try:
                        topk_layers = [int(l) for l in topk_layers]
                    except Exception:
                        topk_layers = []

                hit_rate_k1 = int(oracle_layer == topk_layers[0]) if oracle_layer is not None and len(topk_layers) >= 1 else 0
                hit_rate_k3 = int(oracle_layer in topk_layers[:3]) if oracle_layer is not None and len(topk_layers) >= 3 else 0
                hit_rate_k5 = int(oracle_layer in topk_layers[:5]) if oracle_layer is not None and len(topk_layers) >= 5 else 0

                diag_rows.append({
                    # Context
                    "seed_dir": sd_path_str,
                    "dataset": context['dataset'],
                    "model": context['model'],
                    "seed": context['seed'],
                    "training_method": context['training_method'],
                    "raw_model_accuracy": context['accuracy'],
                    # Metric Info
                    "metric": m,
                    "metric_base": base_metric,
                    # Original Diagnostics
                    "spearman_vs_holdout_ece": v.get("spearman_rho_vs_holdout_ece"),
                    "inferred_direction": v.get("inferred_direction"),
                    "gap_from_best_holdout_top1": v.get("gap_from_best_holdout_top1"),
                    "depth_bias_norm0shallow1deep": (depth_bias.get(m, {}) or {}).get("normalized_0_shallow_1_deep"),
                    # --- NEW Analyses ---
                    "kendall_tau_vs_holdout_ece": float(kendall_tau_val) if np.isfinite(kendall_tau_val) else None,
                    "kendall_p_value": float(kendall_p) if np.isfinite(kendall_p) else None,
                    "oracle_hit_rate_K1": hit_rate_k1,
                    "oracle_hit_rate_K3": hit_rate_k3,
                    "oracle_hit_rate_K5": hit_rate_k5,
                })
        except Exception as e:
            print(f"[WARN] Failed processing diagnostics for {sd_path_str}: {e}")
            continue

    if not diag_rows:
        print("[WARN] No diagnostic rows collected.")
        return None

    diag_df = pd.DataFrame(diag_rows)

    # --- NEW: Perform Stability, Stratified, and Depth Bias Analyses ---

    print("\n--- Metric Correlation Stability (Std Dev across Seeds) ---")
    stability = (
        diag_df.groupby(['metric', 'dataset', 'model'])['spearman_vs_holdout_ece']
        .std()
        .reset_index()
        .rename(columns={'spearman_vs_holdout_ece': 'spearman_std_dev'})
    )
    print(stability.sort_values(['dataset', 'model', 'spearman_std_dev']))

    print("\n--- Stratified Correlation (Mean Abs Spearman by Dataset/Model) ---")
    diag_df['abs_spearman'] = diag_df['spearman_vs_holdout_ece'].abs()
    stratified_mean = (
        diag_df.groupby(['metric', 'dataset', 'model'])['abs_spearman']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'mean_abs_spearman'})
    )
    print(stratified_mean.sort_values(['dataset', 'model', 'mean_abs_spearman'], ascending=[True, True, False]))

    print("\n--- Depth Bias vs. Correlation ---")
    bins = [0, 0.25, 0.75, 1.01]
    labels = ['Shallow (0-0.25)', 'Middle (0.25-0.75)', 'Deep (0.75-1.0)']
    diag_df['depth_bin'] = pd.cut(
        diag_df['depth_bias_norm0shallow1deep'],
        bins=bins, labels=labels, right=False, include_lowest=True
    )
    depth_vs_corr = (
        diag_df.groupby(['metric', 'depth_bin'])['abs_spearman']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'mean_abs_spearman'})
    )
    print(depth_vs_corr.sort_values(['metric', 'depth_bin']))

    print("\n--- Correlation vs. Raw Model Accuracy ---")
    corr_vs_acc = (
        diag_df.groupby('metric')[['abs_spearman', 'raw_model_accuracy']]
        .corr(method='spearman')
        .unstack()
        .iloc[:, 1]
        .reset_index()
        .rename(columns={0: 'spearman_corr_vs_accuracy'})
    )
    print(corr_vs_acc.sort_values('spearman_corr_vs_accuracy', ascending=False))

    print("\n--- Average Top-K Oracle Hit Rates ---")
    avg_hit_rates = (
        diag_df.groupby('metric')[[
            'oracle_hit_rate_K1', 'oracle_hit_rate_K3', 'oracle_hit_rate_K5']]
        .mean()
        .reset_index()
    )
    print(avg_hit_rates.sort_values('oracle_hit_rate_K5', ascending=False))

    # Save the enriched diagnostics dataframe
    return diag_df


def weighting_wins(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df["result_type"] == "ensemble") & (df["selection"].isin(["topk", "metric"]))].copy()
    if sub.empty:
        return pd.DataFrame()
    on = ["dataset", "model", "seed", "training_method", "k", "selection"]
    best = (sub.sort_values("ece")
              .groupby(on, as_index=False).first()[on + ["full_method", "weighting", "ece"]])
    out = (best.groupby(["k", "weighting"])["ece"]
              .agg(["count", "mean"]).reset_index()
              .rename(columns={"count": "wins", "mean": "mean_ece"}))
    if out.empty:
        return out
    out["wins"] = out["wins"].astype(int)
    out["mean_ece"] = out["mean_ece"].round(7)
    return out.sort_values(["k", "wins"], ascending=[True, False])


def ees_vs_ece(df: pd.DataFrame) -> pd.DataFrame:
    if "ees" not in df.columns:
        return pd.DataFrame()
    sub = df[(df["result_type"] == "ensemble") & df["ees"].notna()]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for (k, wt), g in sub.groupby(["k", "weighting"]):
        if len(g) < 3:
            continue
        try:
            r = stats.spearmanr(g["ees"].values, g["ece"].values, nan_policy="omit")
            corr = float(r.correlation) if np.isfinite(r.correlation) else np.nan
            pval = float(r.pvalue) if np.isfinite(r.pvalue) else np.nan
        except Exception:
            corr, pval = np.nan, np.nan
        rows.append({"k": k, "weighting": wt,
                     "spearman_ees_vs_ece": round(corr, 7) if not np.isnan(corr) else np.nan,
                     "pvalue": round(pval, 7) if not np.isnan(pval) else np.nan,
                     "N": int(len(g))})
    return pd.DataFrame(rows)


def layer_cooccurrence(df: pd.DataFrame, k: int) -> pd.DataFrame:
    sub = df[(df["result_type"] == "ensemble") & (df["k"] == k) & df["selected_layers"].notna()]
    if sub.empty:
        return pd.DataFrame()
    from itertools import combinations
    counts = Counter()
    for layers in sub["selected_layers"]:
        try:
            L = [int(x) for x in list(layers)]
        except Exception:
            L = list(layers)
        for a, b in combinations(L, 2):
            counts[tuple(sorted((int(a), int(b))))] += 1
    if not counts:
        return pd.DataFrame()
    data = [{"layer_a": a, "layer_b": b, "count": c} for (a, b), c in counts.items()]
    return pd.DataFrame(data).sort_values("count", ascending=False)


def topk_vs_oracle(df: pd.DataFrame, k: int, threshold: float) -> pd.DataFrame:
    sub = df[(df["k"] == k) & (df["selection"] == "topk") & df["ece"].notna() & df["oracle_ece"].notna()]
    if sub.empty:
        return pd.DataFrame()
    sub = sub.assign(gap=sub["ece"] - sub["oracle_ece"],
                     failure=((sub["ece"] > sub["oracle_ece"] + float(threshold)).astype(int)))
    out = (sub.groupby("full_method")[ ["gap", "failure"] ]
             .agg(gap_mean=("gap", "mean"),
                  gap_median=("gap", "median"),
                  failure_rate=("failure", "mean"),
                  N=("gap", "count"))
             .reset_index())
    for c in ["gap_mean", "gap_median", "failure_rate"]:
        out[c] = out[c].round(7)
    return out.sort_values("gap_mean")


def k_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df["result_type"] == "ensemble") & (df["selection"] == "topk") & df["k"].notna()]
    if sub.empty:
        return pd.DataFrame()
    return (sub.groupby(["dataset", "model", "training_method", "weighting", "k"])["ece"]
               .mean().reset_index()
               .rename(columns={"ece": "mean_ece"}).sort_values(["weighting", "k"]))

def summarize_alpha_sweep(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      alpha_grid: mean/std/count ECE for every (dataset, model, train, K, weighting, alpha)
      alpha_best: the best alpha per (dataset, model, train, K, weighting)
    """
    sub = df[
        (df["selection"] == "topk")
        & (df["weighting"].isin(["deep", "shallow"]))
        & df["weight_param"].notna()
    ].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    sub["alpha"] = sub["weight_param"].astype(float)

    group_cols = ["dataset", "model", "training_method", "k", "weighting", "alpha"]
    alpha_grid = (
        sub.groupby(group_cols)["ece"]
           .agg(["mean", "std", "count"])
           .reset_index()
           .rename(columns={"mean": "mean_ece", "std": "std_ece", "count": "n"})
           .sort_values(group_cols)
    )
    # Round numeric columns to 7 decimal places
    alpha_grid["mean_ece"] = alpha_grid["mean_ece"].round(7)
    alpha_grid["std_ece"] = alpha_grid["std_ece"].round(7)

    # best α per group (ties -> first by lowest mean, then lowest alpha)
    base_cols = ["dataset", "model", "training_method", "k", "weighting"]
    alpha_best = (
        alpha_grid.sort_values(base_cols + ["mean_ece", "alpha"])
                  .groupby(base_cols, as_index=False)
                  .first()
                  .rename(columns={"alpha": "best_alpha", "mean_ece": "best_mean_ece"})
    )
    # Round the best mean ECE to 7 decimal places
    alpha_best["best_mean_ece"] = alpha_best["best_mean_ece"].round(7)
    return alpha_grid, alpha_best


def _paired_merge(dfA: pd.DataFrame, dfB: pd.DataFrame, on_cols=None):
    if on_cols is None:
        on_cols = ["dataset", "model", "seed", "training_method"]
    return pd.merge(
        dfA[on_cols + ["ece"]],
        dfB[on_cols + ["ece"]],
        on=on_cols,
        how="inner",
        suffixes=("_A", "_B"),
    )


def head2head_winrates(df: pd.DataFrame, methods: List[str], k: Optional[float] = None) -> pd.DataFrame:
    """
    Pairwise winrates between ensemble methods (lower ECE wins).
    If k is not None, restrict to that K; otherwise compare across Ks.
    Returns long-form table: A, B, win_rate, median_delta(B-A), N.
    """
    sub = df[(df["selection"].isin(["metric", "topk"])) & (df["full_method"].isin(methods))].copy()
    if k is not None:
        sub = sub[sub["k"] == k]
    rows = []
    on_cols = ["dataset", "model", "seed", "training_method"]
    for i, a in enumerate(methods):
        A = sub[sub["full_method"] == a]
        for b in methods:
            if a == b: 
                continue
            B = sub[sub["full_method"] == b]
            M = _paired_merge(A, B, on_cols)
            if len(M) == 0:
                rows.append({"A": a, "B": b, "win_rate": np.nan, "median_delta(B-A)": np.nan, "N": 0})
                continue
            wins = int((M["ece_A"] < M["ece_B"]).sum())
            ties = int((M["ece_A"] == M["ece_B"]).sum())
            total = int(len(M))
            win_rate = (wins + 0.5 * ties) / total
            rows.append({
                "A": a, "B": b, "win_rate": round(float(win_rate), 7),
                "median_delta(B-A)": round(float(np.median(M["ece_B"] - M["ece_A"])), 7),
                "N": total
            })
    return pd.DataFrame(rows)


def head2head_matrix(h2h_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot to an A×B winrate matrix."""
    if h2h_long.empty:
        return pd.DataFrame()
    return h2h_long.pivot_table(index="A", columns="B", values="win_rate")


def compare_to_baselines_by_dataset(df: pd.DataFrame, best_method_name: str, k_val: float) -> pd.DataFrame:
    """
    Same as compare_to_baselines, but stratified by dataset.
    """
    on_cols = ["dataset", "model", "seed", "training_method"]
    cols = ["Dataset", "Baseline", "Win Rate vs Best Ensemble",
            "Median ΔECE (base-best)", "Wilcoxon P-Value", "N"]

    best_rows = df[(df["k"] == k_val) & (df["full_method"] == best_method_name)].copy()
    base = df[df["result_type"] == "baseline"].copy()

    # Guard: if either side is empty, return an empty, well-formed table
    if best_rows.empty or base.empty:
        return pd.DataFrame(columns=cols)

    merged = pd.merge(
        best_rows[on_cols + ["ece"]],
        base[on_cols + ["method_name", "ece"]],
        on=on_cols, how="inner", suffixes=("_best", "_base")
    )

    # Guard: no paired runs → return empty, well-formed table
    if merged.empty:
        return pd.DataFrame(columns=cols)

    out_rows = []
    for (dset, bname), g in merged.groupby(["dataset", "method_name"]):
        wins = int((g["ece_best"] < g["ece_base"]).sum())
        ties = int((g["ece_best"] == g["ece_base"]).sum())
        total = int(len(g))
        win_rate = (wins + 0.5 * ties) / total if total > 0 else np.nan
        diff = g["ece_base"].values - g["ece_best"].values
        try:
            if len(diff) > 0 and np.any(diff != 0):
                _, p = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
            else:
                p = np.nan
        except Exception:
            p = np.nan
        out_rows.append({
            "Dataset": dset,
            "Baseline": bname,
            "Win Rate vs Best Ensemble": round(win_rate, 7),
            "Median ΔECE (base-best)": round(float(np.median(diff)), 7) if len(diff) else np.nan,
            "Wilcoxon P-Value": round(p, 7) if not np.isnan(p) else np.nan,
            "N": total,
        })

    out_df = pd.DataFrame(out_rows, columns=cols)
    if out_df.empty:
        return out_df
    return out_df.sort_values(["Dataset", "Win Rate vs Best Ensemble"], ascending=[True, False]).reset_index(drop=True)


# -----------------------------
# Insight add-ons: worst gaps & selection stability
# -----------------------------
def worst_oracle_gaps(df: pd.DataFrame, method: str, k: float, top: int = 20) -> pd.DataFrame:
    sub = df[(df["k"] == k) & (df["full_method"] == method) & df["oracle_ece"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.assign(gap=sub["ece"] - sub["oracle_ece"]).sort_values("gap", ascending=False)
    cols = ["dataset", "model", "training_method", "seed", "ece", "oracle_ece", "gap", "selected_layers", "weights_json"]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].head(top)


from itertools import combinations as _it_combinations
def selection_stability(df: pd.DataFrame, method: str, k: float) -> pd.DataFrame:
    sub = df[(df["k"] == k) & (df["full_method"] == method) & df["selected_layers"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    by = sub.groupby(["dataset", "model", "training_method"])
    rows = []
    for key, g in by:
        sets = [set(t) for t in g["selected_layers"]]
        if len(sets) < 2:
            continue
        pairs = list(_it_combinations(sets, 2))
        j = [len(a & b) / len(a | b) if len(a | b) > 0 else np.nan for a, b in pairs]
        rows.append({
            "dataset": key[0],
            "model": key[1],
            "training_method": key[2],
            "k": k,
            "method": method,
            "mean_jaccard": float(np.nanmean(j)),
            "Npairs": len(pairs)
        })
    return pd.DataFrame(rows)


# -----------------------------
# New additions: Oracle/baseline and non-inferiority analyses
# -----------------------------
def oracle_vs_baselines(df: pd.DataFrame, k: int) -> pd.DataFrame:
    on = ["dataset", "model", "seed", "training_method"]
    runs = df[on + ["oracle_ece"]].drop_duplicates()
    base = df[df["result_type"] == "baseline"][on + ["method_name", "ece"]].copy()
    M = pd.merge(runs, base, on=on, how="inner")
    rows = []
    if M.empty:
        return pd.DataFrame(columns=[
            "Baseline", "Win Rate (Oracle < Baseline)", "Median ΔECE (baseline - oracle)", "N"
        ])
    for bname, g in M.groupby("method_name"):
        wins = int((g["oracle_ece"] < g["ece"]).sum())
        ties = int((g["oracle_ece"] == g["ece"]).sum())
        total = int(len(g))
        win_rate = (wins + 0.5 * ties) / total if total else float("nan")
        diff = g["ece"].values - g["oracle_ece"].values  # positive => oracle better
        rows.append({
            "Baseline": bname,
            "Win Rate (Oracle < Baseline)": round(win_rate, 7),
            "Median ΔECE (baseline - oracle)": round(float(np.median(diff)), 7) if total else np.nan,
            "N": total,
        })
    return pd.DataFrame(rows).sort_values("Win Rate (Oracle < Baseline)", ascending=False)


def best_vs_best_baseline(df: pd.DataFrame, best_method_name: str, k: int) -> pd.DataFrame:
    on = ["dataset", "model", "seed", "training_method"]
    best = (
        df[(df["k"] == k) & (df["full_method"] == best_method_name)][on + ["ece"]]
        .rename(columns={"ece": "ece_best"})
    )
    base = df[df["result_type"] == "baseline"][on + ["method_name", "ece"]]
    # pick the min-ECE baseline per run
    base_min = (
        base.sort_values("ece")
        .groupby(on, as_index=False)
        .first()
        .rename(columns={"method_name": "best_baseline", "ece": "ece_base_min"})
    )
    M = pd.merge(best, base_min, on=on, how="inner")
    if M.empty:
        return pd.DataFrame(
            columns=["K", "Method", "Win Rate vs best-baseline", "Median ΔECE (base-best)", "N"]
        )
    wins = int((M["ece_best"] < M["ece_base_min"]).sum())
    ties = int((M["ece_best"] == M["ece_base_min"]).sum())
    total = int(len(M))
    win_rate = (wins + 0.5 * ties) / total if total else float("nan")
    diff = M["ece_base_min"].values - M["ece_best"].values
    return pd.DataFrame(
        [
            {
                "K": k,
                "Method": best_method_name,
                "Win Rate vs best-baseline": round(float(win_rate), 7) if np.isfinite(win_rate) else np.nan,
                "Median ΔECE (base-best)": round(float(np.median(diff)), 7) if total else np.nan,
                "N": total,
            }
        ]
    )


def within_delta_of_oracle(
    df: pd.DataFrame, k: int, method: str, delta: float = 0.01
) -> pd.DataFrame:
    on = ["dataset", "model", "seed", "training_method"]
    sub = df[
        (df["k"] == k)
        & (df["full_method"] == method)
        & df["oracle_ece"].notna()
    ][on + ["ece", "oracle_ece"]].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "K",
                "Method",
                "delta",
                "Frac within δ",
                "Median gap (best - oracle)",
                "N",
            ]
        )
    sub["gap"] = sub["ece"] - sub["oracle_ece"]
    frac = float((sub["gap"] <= float(delta)).mean())
    return pd.DataFrame(
        [
            {
                "K": k,
                "Method": method,
                "delta": float(delta),
                "Frac within δ": round(frac, 7),
                "Median gap (best - oracle)": round(float(sub["gap"].median()), 7),
                "N": int(len(sub)),
            }
        ]
    )

# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate and analyze multi-layer ensemble results (v3, with P0 analyses).")
    ap.add_argument("--results-dir", type=str, required=True, help="Root results directory (contains k_*/...).")
    ap.add_argument("--output-dir", type=str, default="./analysis_summary_v3", help="Where to save CSVs/plots/report.")
    ap.add_argument("--oracle-threshold", type=float, default=0.01, help="Failure threshold for ECE vs. oracle.")
    ap.add_argument("--plot", action="store_true", help="Generate basic plots.")
    ap.add_argument("--include-k0", action="store_true", help="Include K=0 in per-K analysis.")
    # Minimal switches to control compute and outputs
    ap.add_argument("--k", type=int, nargs="*", help="Only analyze these K values, e.g., --k 1 3")
    ap.add_argument(
        "--profile", choices=["core", "paper", "all"], default="paper",
        help="core = fastest P0; paper = P0+some P1; all = everything"
    )
    ap.add_argument("--skip", nargs="*", default=[], help="Extra blocks to skip by name")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_dir)
    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    # Feature gates
    PROFILE_BLOCKS = {
        "core": {
            "leaderboard", "best_vs_baselines", "by_dataset_baselines",
            "oracle_vs_baselines", "best_vs_bestbaseline", "within_delta_oracle",
            "accbin_summary"
        },
        "paper": {
            # core + selected P1
            "leaderboard", "best_vs_baselines", "by_dataset_baselines",
            "oracle_vs_baselines", "best_vs_bestbaseline", "within_delta_oracle",
            "accbin_summary", "ece_dist_plot",
            "metric_vs_baselines_all", "weight_stats"
        },
        "all": {
            # everything
            "leaderboard", "best_vs_baselines", "by_dataset_baselines",
            "oracle_vs_baselines", "best_vs_bestbaseline", "within_delta_oracle",
            "accbin_summary", "accbin_plot", "metric_vs_baselines_all",
            "metric_vs_baselines_by_dataset", "metric_vs_baselines_by_model",
            "head2head_top", "alpha_summaries", "alpha_head2head",
            "weight_stats", "weight_heatmap", "weight_long", "topk_vs_oracle",
            "layer_cooccurrence", "worst_gaps", "selection_stability",
            "ees_vs_ece", "metric_diagnostics", "weighting_winshares",
            "crossK_head2head", "ece_dist_plot"
        }
    }
    ENABLED = set(PROFILE_BLOCKS.get(args.profile, set())) - set(args.skip)
    DO = lambda name: (name in ENABLED)

    # 1) Aggregate
    df = find_and_parse_results(root)
    
    # Ensure full_method always exists
    if "full_method" not in df.columns:
        if "selection" in df.columns:
            # conservative fallback
            df["full_method"] = df["method_name"].astype(str)
            if "weighting" in df.columns:
                df["full_method"] = df["full_method"] + "_" + df["weighting"].astype(str)
        else:
            df["full_method"] = df["method_name"].astype(str)
    
    # --- NEW: compute oracle once per run and attach to ALL rows ---
    # 1) build a cache: seed_dir -> oracle_ece (computed once)
    seed_dirs = df["seed_dir"].dropna().unique().tolist()
    oracle_cache = {}

    for sd in seed_dirs:
        try:
            v = load_oracle_best_ece(Path(sd))
            if v is not None and np.isfinite(v):
                oracle_cache[sd] = float(v)
        except Exception:
            pass

    # --- Fallback: approximate oracle from existing results (K=1) ---
    on_cols = ["dataset", "model", "seed", "training_method"]

    # Build quick index from (dataset, model, seed, training_method) -> seed_dir
    run2seed = (df.drop_duplicates(subset=on_cols + ["seed_dir"])
                  .set_index(on_cols)["seed_dir"].to_dict())

    # For runs without a file-backed oracle, synthesize from K=1 rows
    missing = [sd for sd in seed_dirs if sd not in oracle_cache]
    if missing:
        # group by run identity
        for key, g in df.groupby(on_cols):
            sd = run2seed.get(key)
            if sd in oracle_cache or sd is None:
                continue
            k1 = g[(g["k"] == 1) & pd.notna(g["ece"])]
            if k1.empty:
                continue
            # Prefer true single-layer selections when available
            def is_one_layer(x):
                return isinstance(x, tuple) and len(x) == 1
            k1_one = k1[k1["selected_layers"].apply(is_one_layer) if "selected_layers" in k1.columns else False]
            src = k1_one if not k1_one.empty else k1
            # use the lowest ECE observed for that run at K=1
            approx = round(float(src["ece"].min()), 7)
            oracle_cache[sd] = approx  # mark as proxy

    # 2) add a column with the oracle value for every row (works across all K's)
    df["oracle_ece"] = df["seed_dir"].map(oracle_cache).astype(float)
    print(f"[OK] Oracle available for {df['oracle_ece'].notna().groupby(df['seed_dir']).any().sum()}/{len(seed_dirs)} seed runs (files+fallback)")

    # 3) optional: save a quick audit of what we found
    oracle_map_csv = outdir / "oracle_per_seed.csv"
    pd.DataFrame({"seed_dir": list(oracle_cache.keys()),
                  "oracle_ece": list(oracle_cache.values())}).to_csv(oracle_map_csv, index=False)
    print(f"[OK] Cached oracle ECE for {len(oracle_cache)}/{len(seed_dirs)} seed runs -> {oracle_map_csv}")

    # Optional: write a CSV that tells you which seeds were file-backed vs fallback
    rows = []
    for sd in seed_dirs:
        rows.append({"seed_dir": sd,
                     "oracle_ece": oracle_cache.get(sd),
                     "source": "fallback_k1_min" if sd in missing and sd in oracle_cache else
                               ("file" if sd in oracle_cache else "missing")})
    pd.DataFrame(rows).to_csv(outdir / "oracle_per_seed_detailed.csv", index=False)
    print(f"[OK] Detailed oracle source info -> {outdir / 'oracle_per_seed_detailed.csv'}")

    # Show missing seeds for debugging
    missing_after = [sd for sd in seed_dirs if sd not in oracle_cache]
    if missing_after[:5]:
        print("[WARN] Still missing oracle for these seeds (showing up to 5):")
        for sd in missing_after[:5]:
            print("   ", sd)
    
    agg_csv = outdir / "aggregated_results.csv"
    df.to_csv(agg_csv, index=False)
    print(f"[OK] Aggregated: {len(df)} rows -> {agg_csv}")

    # Methods catalog for readability/repro
    try:
        methods_catalog_csv = outdir / "methods_catalog.csv"
        (df[["full_method","selection","metric_name","metric_variant","weighting","weight_param"]]
           .drop_duplicates()
           .sort_values(["selection","full_method"])
           .to_csv(methods_catalog_csv, index=False))
        print(f"[OK] Wrote methods catalog → {methods_catalog_csv}")
    except Exception:
        pass

    # α sweep summaries (deep/shallow), per dataset/model/…/K
    if DO("alpha_summaries"):
        alpha_grid, alpha_best = summarize_alpha_sweep(df)
        alpha_dir = outdir / "alpha_sweeps"
        ensure_dir(alpha_dir)
        if not alpha_grid.empty:
            alpha_grid.to_csv(alpha_dir / "alpha_grid.csv", index=False)
            alpha_best.to_csv(alpha_dir / "alpha_best_by_group.csv", index=False)
            print(f"[OK] Wrote α grid:  {alpha_dir / 'alpha_grid.csv'}")
            print(f"[OK] Wrote α best:  {alpha_dir / 'alpha_best_by_group.csv'}")
        else:
            print("[WARN] No α-sweep rows found (need topk deep/shallow with _a{val} keys).")

    # 2) Rank metric-selected methods per K
    # Handle both int and float K values
    ks = sorted([k for k in df["k"].dropna().unique() if np.isfinite(k) and (k > 0 or args.include_k0)])
    if args.k:
        try:
            kset = set(int(x) for x in args.k)
            ks = [k for k in ks if int(k) in kset]
        except Exception:
            pass
    best_tables: Dict[float, pd.DataFrame] = {}
    winrate_tables: Dict[float, pd.DataFrame] = {}
    oracle_summaries: Dict[float, Dict[str, Any]] = {}
    weight_summaries: Dict[float, Dict[str, Any]] = {}
    failure_tables: Dict[float, pd.DataFrame] = {}

    report_lines: List[str] = []
    report_lines.append("=" * 88)
    report_lines.append("COMPREHENSIVE EXPERIMENT ANALYSIS (v3)")
    report_lines.append("=" * 88)
    report_lines.append(f"Results dir: {root}")
    report_lines.append("")
    
    # Data snapshot for debugging
    report_lines.append("\nDATA SNAPSHOT")
    report_lines.append(f"  Total rows: {len(df)}")
    if "selection" in df.columns:
        vc = df["selection"].value_counts(dropna=False).to_dict()
        report_lines.append(f"  selection counts: {vc}")
    if "k" in df.columns:
        vc = df["k"].value_counts(dropna=False).to_dict()
        report_lines.append(f"  k counts: {vc}")
    report_lines.append(f"  datasets: {sorted(df['dataset'].dropna().unique().tolist())}")
    report_lines.append("")

    for k in ks:
        report_lines.append("\n" + "-" * 88)
        report_lines.append(f"K = {k}")
        report_lines.append("-" * 88)

        best_df = analyze_best_metrics(df, k)
        best_tables[k] = best_df
        if best_df.empty:
            # Fallback: analyze top-K ensembles (so the report isn't empty)
            topk_df = df[(df["k"] == k) & (df["selection"] == "topk")].copy()
            if not topk_df.empty:
                # build "best method" table from topk_df instead of metric df
                if "full_method" not in topk_df.columns:
                    topk_df["full_method"] = topk_df["method_name"].astype(str)
                    if "weighting" in topk_df.columns:
                        topk_df["full_method"] = topk_df["full_method"] + "_" + topk_df["weighting"].astype(str)
                
                # summarize by method
                summary = (topk_df
                           .groupby("full_method")["ece"]
                           .agg(["mean","std","count"])
                           .sort_values("mean"))
                
                # Convert to the expected format
                rows = []
                for method, row in summary.iterrows():
                    rows.append({
                        "Method": method,
                        "Mean ECE": round(row["mean"], 7),
                        "95% CI Lower": np.nan,  # Not computed for fallback
                        "95% CI Upper": np.nan,  # Not computed for fallback
                        "Std Dev": round(row["std"], 7),
                        "Runs": int(row["count"]),
                    })
                best_df = pd.DataFrame(rows).sort_values("Mean ECE", ascending=True).reset_index(drop=True)
                best_tables[k] = best_df
                report_lines.append(f"[FALLBACK] Using top-K ensembles (no metric results found)")
            else:
                report_lines.append("No metric-selected or top-K ensemble results found.")
                continue

        # Save per-K method stats
        if DO("leaderboard"):
            best_csv = outdir / f"method_stats_K{k}.csv"
            best_df.to_csv(best_csv, index=False)
            report_lines.append(f"[OK] Saved method leaderboard: {best_csv}")
        # Choose top 1 as "best"
        best_method_name = best_df.iloc[0]["Method"]
        # QoL: print weighting embedded in best method (if present)
        try:
            if isinstance(best_method_name, str) and ":" in best_method_name:
                wtok = best_method_name.split(":", 1)[1]
                if "_" in wtok:
                    wname = wtok.split("_")[0]
                else:
                    wname = wtok
                report_lines.append(f"Best weighting (heuristic parse): {wname}")
        except Exception:
            pass

        # Baseline comparison
        if DO("best_vs_baselines"):
            win_df = compare_to_baselines(df, best_method_name, k, mtest_correction=True)
            winrate_tables[k] = win_df
            win_csv = outdir / f"winrates_vs_baselines_K{k}.csv"
            win_df.to_csv(win_csv, index=False)
            report_lines.append(f"[OK] Saved baseline comparisons: {win_csv}")

        # Short textual summary of baseline wins (top 3 by win rate)
        top3 = win_df.sort_values("Win Rate vs Best Ensemble", ascending=False).head(3)
        report_lines.append("Top baseline win-rates:")
        for _, r in top3.iterrows():
            report_lines.append(f"  vs {r['Baseline']}: win={r['Win Rate vs Best Ensemble']:.7f}, "
                                f"median ΔECE={r['Median ΔECE (base-best)']:.7f}, "
                                f"N={int(r['N'])}")

        # Dataset-stratified baseline comparison
        if DO("by_dataset_baselines"):
            win_by_ds = compare_to_baselines_by_dataset(df, best_method_name, k)
            if not win_by_ds.empty:
                byds_csv = outdir / f"winrates_vs_baselines_K{k}_by_dataset.csv"
                win_by_ds.to_csv(byds_csv, index=False)
                report_lines.append(f"[OK] Saved baseline comparisons by dataset: {byds_csv}")
                # Add one line per dataset for the top baseline
                for dset, g in win_by_ds.groupby("Dataset"):
                    r = g.iloc[0]
                    report_lines.append(f"  [{dset}] best vs {r['Baseline']}: win={r['Win Rate vs Best Ensemble']:.7f}, "
                                        f"median ΔECE={r['Median ΔECE (base-best)']:.7f}, N={int(r['N'])}")
            else:
                report_lines.append(f"[WARN] No dataset-stratified baseline pairs for K={k}.")
            # Quick sanity checks (printed once per K)
            try:
                print("best_rows:", len(df[(df["k"]==k) & (df["full_method"]==best_method_name)]))
                print("baselines:", len(df[df["result_type"]=="baseline"]))
                print("paired merge:",
                      len(pd.merge(
                          df[(df["k"]==k) & (df["full_method"]==best_method_name)][["dataset","model","seed","training_method","ece"]],
                          df[df["result_type"]=="baseline"][ ["dataset","model","seed","training_method","method_name","ece"] ],
                          on=["dataset","model","seed","training_method"], how="inner")))
            except Exception:
                pass

        # Within-K head-to-head among the top few ensemble methods (keeps report readable)
        if DO("head2head_top"):
            top_methods = (df[(df["k"] == k) & (df["selection"].isin(["metric", "topk"]))]
                             .groupby("full_method")["ece"].mean().sort_values().head(4).index.tolist())
            h2h = head2head_winrates(df, top_methods, k=k)
            h2h_csv = outdir / f"head2head_K{k}.csv"
            h2h.to_csv(h2h_csv, index=False)
            h2h_mat = head2head_matrix(h2h)
            h2h_mat.to_csv(outdir / f"head2head_K{k}_matrix.csv")
            report_lines.append(f"[OK] Saved head-to-head (K={k}) vs ensembles: {h2h_csv}")
            if not h2h.empty and len(top_methods) > 1:
                second = [m for m in top_methods if m != best_method_name][0]
                row = h2h[(h2h["A"] == best_method_name) & (h2h["B"] == second)].head(1)
                if not row.empty:
                    report_lines.append(f"Head-to-head: {best_method_name} vs {second} — "
                                        f"win={row.iloc[0]['win_rate']:.7f}, "
                                        f"median ΔECE={row.iloc[0]['median_delta(B-A)']:.7f}, "
                                        f"N={int(row.iloc[0]['N'])}")

        # Layer combinations
        combos = analyze_layer_combinations(df, best_method_name, k)
        top_combos = ", ".join([f"{c}:{n}" for c, n in combos.most_common(5)])
        report_lines.append(f"Top layer combos (first 5): {top_combos if top_combos else 'n/a'}")

        # Oracle comparison
        oracle_sum = compare_ensemble_vs_oracle(df, k, best_method_name)
        oracle_summaries[k] = oracle_sum
        if "error" in oracle_sum:
            report_lines.append(f"[WARN] Oracle comparison: {oracle_sum['error']}")
            oracle_ref = None
        else:
            report_lines.append(f"Oracle comparison: win-rate={oracle_sum['WinRate_vs_Oracle']:.7f}, "
                                f"ΔECE mean={oracle_sum['MeanImprovement']:.7f} [{oracle_sum['CI95_lo']:.7f}, {oracle_sum['CI95_hi']:.7f}], "
                                f"Wilcoxon p={oracle_sum['WilcoxonP_greater']:.7f}, "
                                f"Cliff's δ={oracle_sum['CliffsDelta (oracle-best)']:.7f}")
            # For plots, use mean oracle as reference
            oracle_ref = None  # we'll compute separately if plotting

        # Weight stability (learned)
        wstats = analyze_weight_distributions(df, k, weighting="learned")
        weight_summaries[k] = wstats
        if "error" in wstats:
            report_lines.append(f"[WARN] Weight analysis: {wstats['error']}")
        else:
            # Save layer stats and optional long-form + heatmap
            if DO("weight_stats"):
                layer_stats_csv = outdir / f"learned_weights_layer_stats_K{k}.csv"
                wstats["layer_stats"].to_csv(layer_stats_csv)
                report_lines.append(f"[OK] Saved learned-weight layer stats: {layer_stats_csv}")

            if DO("weight_long"):
                weights_long_csv = outdir / f"learned_weights_long_K{k}.csv"
                wstats["weights_long"].to_csv(weights_long_csv, index=False)
                report_lines.append(f"[OK] Saved learned-weight (long-form): {weights_long_csv}")

            report_lines.append(f"Learned weights: eff.size μ={wstats['mean_effective_size']:.7f}±{wstats['std_effective_size']:.7f}, "
                                f"Gini μ={wstats['mean_gini']:.7f} (↓ is more uniform)")

            if DO("weight_heatmap") and args.plot:
                out_png = outdir / f"weight_stability_K{k}.png"
                plot_weight_stability(wstats["weights_long"], out_png, title=f"Learned Weights Stability (K={k})")
                report_lines.append(f"[OK] Saved heatmap: {out_png}")

        # Metric failure diagnosis
        if DO("failure_tables"):
            failures = diagnose_metric_failures(df, k, threshold=args.oracle_threshold)
            failure_tables[k] = failures
            if failures.empty:
                report_lines.append("[WARN] Metric failure diagnosis: no valid rows.")
            else:
                fail_csv = outdir / f"metric_failures_K{k}.csv"
                failures.to_csv(fail_csv, index=False)
                report_lines.append(f"[OK] Saved metric failure stats: {fail_csv}")

        # E. Top-K vs oracle diagnostics (gap/failure)
        if DO("topk_vs_oracle"):
            tk = topk_vs_oracle(df, k, args.oracle_threshold)
            if not tk.empty:
                tk_csv = outdir / f"topk_vs_oracle_K{k}.csv"
                tk.to_csv(tk_csv, index=False)
                report_lines.append(f"[OK] Saved top-K vs oracle diagnostics: {tk_csv}")

        # --- NEW: all metrics vs baselines (overall) ---
        if DO("metric_vs_baselines_all"):
            met_vs_base = metric_winrates_vs_baselines(df, k=k, by=None)
            if not met_vs_base.empty:
                out_csv = outdir / f"metric_winrates_vs_baselines_K{k}.csv"
                met_vs_base.to_csv(out_csv, index=False)
                report_lines.append(f"[OK] Saved ALL-metrics vs baselines: {out_csv}")

        # Stratified by dataset & by model
        if DO("metric_vs_baselines_by_dataset"):
            met_vs_base_ds = metric_winrates_vs_baselines(df, k=k, by="dataset")
            if not met_vs_base_ds.empty:
                out_csv = outdir / f"metric_winrates_vs_baselines_K{k}_by_dataset.csv"
                met_vs_base_ds.to_csv(out_csv, index=False)
                report_lines.append(f"[OK] Saved ALL-metrics vs baselines (by dataset): {out_csv}")

        if DO("metric_vs_baselines_by_model"):
            met_vs_base_md = metric_winrates_vs_baselines(df, k=k, by="model")
            if not met_vs_base_md.empty:
                out_csv = outdir / f"metric_winrates_vs_baselines_K{k}_by_model.csv"
                met_vs_base_md.to_csv(out_csv, index=False)
                report_lines.append(f"[OK] Saved ALL-metrics vs baselines (by model): {out_csv}")

        # --- NEW: Accuracy-binned summaries & (optional) plot for "best vs key baselines" ---
        # Choose the "best" method you already picked above
        best_method_name = best_df.iloc[0]["Method"]
        # Baselines to show alongside (customize if you like)
        key_bases = ["temperature_scaling", "beta_calibration", "uncalibrated"]
        # Map baselines to full_method for joining in the same column:
        base_fulls = df[df["result_type"]=="baseline"]["full_method"].dropna().unique().tolist()
        bases_present = [b for b in key_bases if b in base_fulls]
        methods_for_bins = [best_method_name] + bases_present

        if DO("accbin_summary"):
            accbin_csv = outdir / f"accbin_summary_K{k}.csv"
            summarize_by_accuracy_bins(df, k, methods_for_bins).to_csv(accbin_csv, index=False)
            report_lines.append(f"[OK] Saved accuracy-bin summary (best vs baselines): {accbin_csv}")

        if DO("accbin_plot") and args.plot:
            accbin_png = outdir / f"ece_vs_accbins_K{k}.png"
            plot_ece_vs_accbins(df, k, methods_for_bins, accbin_png,
                                title=f"ECE vs Accuracy bins (K={k}) – best method vs baselines")
            report_lines.append(f"[OK] Saved accuracy-bin plot: {accbin_png}")

        # Optional ECE distribution plot with oracle mean reference (if available)
        if DO("ece_dist_plot") and args.plot:
            # compute oracle mean across runs of best method
            try:
                ens = df[(df["k"] == k) & (df["full_method"] == best_method_name)].copy()
                om = ens["oracle_ece"].dropna()
                oracle_mean = float(om.mean()) if not om.empty else None
            except Exception:
                oracle_mean = None

            dist_png = outdir / f"ece_distribution_K{k}.png"
            plot_ece_distribution(df, k, [best_method_name], dist_png, oracle_ref=oracle_mean)
            report_lines.append(f"[OK] Saved ECE distribution: {dist_png}")

        # Add report line: best mean ECE vs mean oracle for this K
        try:
            ens = df[(df["k"] == k) & (df["full_method"] == best_method_name)]
            if not ens.empty and ens["oracle_ece"].notna().any():
                best_mean = float(ens["ece"].mean())
                oracle_mean = float(ens["oracle_ece"].dropna().mean())
                report_lines.append(f"Best vs Oracle means (K={k}): best μECE={best_mean:.7f}, oracle μECE={oracle_mean:.7f}")
        except Exception:
            pass

        # D. Layer co-occurrence map
        if DO("layer_cooccurrence"):
            co = layer_cooccurrence(df, k)
            if not co.empty:
                co_csv = outdir / f"layer_cooccurrence_K{k}.csv"
                co.to_csv(co_csv, index=False)
                report_lines.append(f"[OK] Saved layer co-occurrence: {co_csv}")

        # Failure exemplars (worst oracle gaps)
        if DO("worst_gaps"):
            try:
                bad = worst_oracle_gaps(df, best_method_name, k)
                if not bad.empty:
                    bad_csv = outdir / f"worst_gaps_vs_oracle_K{k}.csv"
                    bad.to_csv(bad_csv, index=False)
                    report_lines.append(f"[OK] Saved worst gaps vs oracle: {bad_csv}")
            except Exception:
                pass

        # Selection stability by Jaccard
        if DO("selection_stability"):
            try:
                stab = selection_stability(df, best_method_name, k)
                if not stab.empty:
                    stab_csv = outdir / f"selection_stability_K{k}.csv"
                    stab.to_csv(stab_csv, index=False)
                    report_lines.append(f"[OK] Saved selection stability (Jaccard): {stab_csv}")
            except Exception:
                pass

        # New: Oracle vs Baselines table (per K)
        try:
            orb_csv = outdir / f"oracle_vs_baselines_K{k}.csv"
            oracle_vs_baselines(df, k).to_csv(orb_csv, index=False)
            report_lines.append(f"[OK] Saved oracle vs baselines: {orb_csv}")
        except Exception as e:
            report_lines.append(f"[WARN] Oracle vs baselines failed: {e}")

        # New: Best vs Best-Baseline-per-run (per K)
        try:
            bvb_csv = outdir / f"best_vs_bestbaseline_K{k}.csv"
            best_vs_best_baseline(df, best_method_name, k).to_csv(bvb_csv, index=False)
            report_lines.append(f"[OK] Saved best vs best-baseline: {bvb_csv}")
        except Exception as e:
            report_lines.append(f"[WARN] Best vs best-baseline failed: {e}")

        # New: Within-δ of Oracle (δ=0.01, 0.005)
        try:
            for dlt in [0.01, 0.005]:
                wdo_csv = outdir / f"within_delta_oracle_K{k}_d{str(dlt).replace('.', 'p')}.csv"
                within_delta_of_oracle(df, k, best_method_name, dlt).to_csv(wdo_csv, index=False)
                report_lines.append(f"[OK] Saved within-δ-of-oracle (δ={dlt}): {wdo_csv}")
        except Exception as e:
            report_lines.append(f"[WARN] Within-δ-of-oracle failed: {e}")

    # Cross-K head-to-head between the K-winners
    best_per_k = {k: best_tables[k].iloc[0]["Method"] for k in best_tables if not best_tables[k].empty}
    if DO("crossK_head2head") and len(best_per_k) >= 2:
        methods = list(best_per_k.values())
        h2h_cross = head2head_winrates(df, methods, k=None)  # across Ks, paired by (dataset,model,seed,training_method)
        cross_csv = outdir / "head2head_best_by_K.csv"
        h2h_cross.to_csv(cross_csv, index=False)
        h2h_cross_mat = head2head_matrix(h2h_cross)
        h2h_cross_mat.to_csv(outdir / "head2head_best_by_K_matrix.csv")
        report_lines.append(f"[OK] Saved cross-K head-to-head of K-winners: {cross_csv}")

    # (Optional) Within-K α head-to-head (deep and shallow separately)
    if DO("alpha_head2head"):
        for k in ks:
            for w in ["deep", "shallow"]:
                mlist = (df[(df["k"] == k) & (df["selection"]=="topk") & (df["weighting"]==w) & df["weight_param"].notna()]
                           .assign(alpha=lambda x: x["weight_param"].astype(float))
                           .sort_values(["alpha"])
                           .apply(lambda r: f"topk:{w}_{r['alpha']}", axis=1).unique().tolist())
                if len(mlist) >= 2:
                    hh = head2head_winrates(df, mlist, k=k)
                    hh.to_csv(outdir / f"alpha_head2head_K{k}_{w}.csv", index=False)
                    report_lines.append(f"[OK] Saved α head-to-head (K={k}, {w}): {outdir / f'alpha_head2head_K{k}_{w}.csv'}")

    # After the per-K loop: one big table across K
    if DO("metric_vs_baselines_all"):
        met_vs_base_all = metric_winrates_vs_baselines(df, k=None, by=None)
        if not met_vs_base_all.empty:
            out_csv = outdir / "metric_winrates_vs_baselines_ALLK.csv"
            met_vs_base_all.to_csv(out_csv, index=False)
            report_lines.append(f"[OK] Saved ALL-K metrics vs baselines: {out_csv}")

    # A. Metric selector diagnostics (once per seed) - ENHANCED
    if DO("metric_diagnostics"):
        diag_df = _collect_metric_selector_diagnostics(df, full_df=df)
        if diag_df is not None and not diag_df.empty:
            diag_csv = outdir / "metric_selector_diagnostics_summary_ENHANCED.csv"
            diag_df.to_csv(diag_csv, index=False)
            print("[OK] Wrote ENHANCED metric selector diagnostics → metric_selector_diagnostics_summary_ENHANCED.csv")

    # B. Weighting win-shares
    if DO("weighting_winshares"):
        ww = weighting_wins(df)
        if not ww.empty:
            ww_csv = outdir / "weighting_winshares.csv"
            ww.to_csv(ww_csv, index=False)
            print("[OK] Wrote weighting win shares → weighting_winshares.csv")

    # C. EES vs ECE relationship
    if DO("ees_vs_ece"):
        ees_tab = ees_vs_ece(df)
        if not ees_tab.empty:
            ees_csv = outdir / "ees_vs_ece.csv"
            ees_tab.to_csv(ees_csv, index=False)
            print("[OK] Saved EES vs ECE correlations → ees_vs_ece.csv")

    # Write report
    report_path = outdir / "summary_report.txt"
    with open(report_path, "w") as fh:
        fh.write("\n".join(report_lines))
    print(f"[OK] Wrote report: {report_path}")


if __name__ == "__main__":
    main()