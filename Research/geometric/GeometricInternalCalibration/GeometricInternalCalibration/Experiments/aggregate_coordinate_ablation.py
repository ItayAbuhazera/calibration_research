#!/usr/bin/env python3
"""
Aggregate Coordinate Ablation Results

Compares:
1. Nested vs Independent sampling strategies
2. Coordinate approach vs traditional baselines
3. Coordinate approach vs L=6 random layer approach
"""

import argparse
import json
import glob
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def load_coordinate_results(results_dir: Path) -> List[Dict]:
    """Load all coordinate ablation results (both nested and independent)."""
    results: List[Dict] = []

    for strategy in ["nested", "independent"]:
        pattern = str(results_dir / f"coordinate_ablation_*_{strategy}.json")
        for fpath in glob.glob(pattern):
            with open(fpath, "r") as f:
                data = json.load(f)

            config = data.get("experiment_config", {})
            dataset = config.get("dataset")
            model_name = config.get("model_name")
            training_method = config.get("training_method")
            seed = config.get("seed")
            total_coordinate_space = config.get("total_coordinate_space")
            num_layers_discovered = config.get("num_layers_discovered")
            method = config.get("method", "global_coordinate_sampling")
            sampling_strategy = config.get("sampling_strategy", strategy)
            coordinate_counts = config.get("coordinate_counts", [])
            num_random_trials = config.get("num_random_trials")

            results_by_k = data.get("results_by_num_coordinates", {})
            for k_str, trials in results_by_k.items():
                try:
                    k = int(k_str)
                except (ValueError, TypeError):
                    continue

                for trial in trials:
                    results.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "training_method": training_method,
                            "seed": seed,
                            "method": method,
                            "sampling_strategy": trial.get(
                                "sampling_strategy", sampling_strategy
                            ),
                            "num_coordinates": int(trial.get("num_coordinates", k)),
                            "total_coordinate_space": trial.get(
                                "total_coordinate_space", total_coordinate_space
                            ),
                            "num_layers_discovered": num_layers_discovered,
                            "num_layers_touched": trial.get("num_layers_touched"),
                            "coords_per_layer": trial.get("coords_per_layer"),
                            "ece_geometric": trial.get("ece_geometric"),
                            "accuracy_geometric": trial.get("accuracy_geometric"),
                            "ece_dac": trial.get("ece_dac"),
                            "accuracy_dac": trial.get("accuracy_dac"),
                            "extraction_time_s": trial.get("extraction_time_s"),
                            "fit_time_s": trial.get(
                                "fit_time_geometric_s", trial.get("fit_time_s")
                            ),
                            "calibrate_time_s": trial.get(
                                "calibrate_time_geometric_s",
                                trial.get("calibrate_time_s"),
                            ),
                            "total_time_s": trial.get("total_time_s"),
                            "plan_time_s": trial.get("plan_time_s"),
                            "dac_fit_time_s": trial.get("fit_time_dac_s"),
                            "dac_calibrate_time_s": trial.get("calibrate_time_dac_s"),
                            "dac_total_coordinates": trial.get("dac_total_coordinates"),
                            "dac_num_pseudo_layers": trial.get(
                                "dac_num_pseudo_layers"
                            ),
                            "coordinate_counts_list": coordinate_counts,
                            "num_random_trials": num_random_trials,
                            "trial_idx": trial.get("trial_idx", 0),
                            "source_file": os.path.basename(fpath),
                        }
                    )

    return results


def load_baselines_and_l6(
    baseline_dir: Path, dataset: str, model: str, method: str, seed: int
) -> Dict:
    """Load baselines and L=6 layer results for a config."""
    fname = f"ablation_{method}_{dataset}_{model}_seed{seed}.json"
    fpath = baseline_dir / fname

    if not fpath.exists():
        return {}

    with open(fpath, "r") as f:
        data = json.load(f)

    baselines = data.get("baselines", {})

    # Extract L=6 results from random_layer_ablation at target_dim=256 (preferred) then 512
    l6_results: List[float] = []
    rla = data.get("random_layer_ablation", {})
    results_by_td = rla.get("results_by_target_dim", {})

    # keys may be strings ("256") or ints (256); normalize
    normalized = {}
    for td_key, td_val in results_by_td.items():
        try:
            td_int = int(td_key)
        except (TypeError, ValueError):
            continue
        normalized[td_int] = td_val

    for td in [256, 512]:
        if td in normalized:
            td_block = normalized[td]
            for trial in td_block.get("global_random", []):
                if trial.get("num_layers") == 6:
                    ece_val = trial.get("ece")
                    if ece_val is not None:
                        l6_results.append(ece_val)

    l6_ece = float(np.mean(l6_results)) if l6_results else None

    return {
        "uncalibrated": baselines.get("uncalibrated", {}).get("ece"),
        "temp_scaling": baselines.get("temperature_scaling", {}).get("ece"),
        "isotonic": baselines.get("isotonic_toplabel", {}).get("ece"),
        "dac": baselines.get("density_aware_calibration", {}).get("ece"),
        "geo_physical": baselines.get("geometric_physical_space", {}).get("ece"),
        "l6_layer_ece": l6_ece,
    }


def merge_with_baselines(coord_results: List[Dict], baseline_dir: Path) -> pd.DataFrame:
    """Merge coordinate results with baselines."""
    baseline_cache: Dict = {}

    for r in coord_results:
        key = (r["dataset"], r["model"], r["training_method"], r["seed"])
        if key not in baseline_cache:
            baseline_cache[key] = load_baselines_and_l6(
                baseline_dir, r["dataset"], r["model"], r["training_method"], r["seed"]
            )

        bl = baseline_cache[key]
        r["baseline_uncalibrated"] = bl.get("uncalibrated")
        r["baseline_temp_scaling"] = bl.get("temp_scaling")
        r["baseline_isotonic"] = bl.get("isotonic")
        r["baseline_dac"] = bl.get("dac")
        r["baseline_geo_physical"] = bl.get("geo_physical")
        r["baseline_l6_layer"] = bl.get("l6_layer_ece")

    df = pd.DataFrame(coord_results)

    # Standardize metric column names to what we care about
    df.rename(
        columns={
            "accuracy_geometric": "accuracy",
        },
        inplace=True,
    )

    # Ensure numeric types where possible
    numeric_cols = [
        "num_coordinates",
        "total_coordinate_space",
        "num_layers_discovered",
        "num_layers_touched",
        "ece_geometric",
        "ece_dac",
        "accuracy",
        "accuracy_dac",
        "extraction_time_s",
        "fit_time_s",
        "calibrate_time_s",
        "total_time_s",
        "plan_time_s",
        "dac_fit_time_s",
        "dac_calibrate_time_s",
        "dac_total_coordinates",
        "dac_num_pseudo_layers",
        "baseline_uncalibrated",
        "baseline_temp_scaling",
        "baseline_isotonic",
        "baseline_dac",
        "baseline_geo_physical",
        "baseline_l6_layer",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_improvements(df: pd.DataFrame) -> pd.DataFrame:
    """Compute improvement percentages vs baselines."""
    df = df.copy()

    for baseline_col, imp_col in [
        ("baseline_uncalibrated", "imp_vs_uncalibrated"),
        ("baseline_temp_scaling", "imp_vs_temp_scaling"),
        ("baseline_isotonic", "imp_vs_isotonic"),
        ("baseline_dac", "imp_vs_dac"),
        ("baseline_geo_physical", "imp_vs_geo_physical"),
        ("baseline_l6_layer", "imp_vs_l6_layer"),
    ]:
        if baseline_col not in df.columns:
            df[imp_col] = np.nan
            continue

        baseline_vals = df[baseline_col].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            improvement = (baseline_vals - df["ece_geometric"]) / baseline_vals * 100.0
        improvement[~np.isfinite(improvement)] = np.nan
        df[imp_col] = improvement

    return df


def analyze_nested_vs_independent(df: pd.DataFrame) -> pd.DataFrame:
    """Compare nested vs independent sampling head-to-head."""
    records: List[Dict] = []

    # Head-to-head at (dataset, model, training_method, seed, num_coordinates)
    grouped = df.groupby(
        ["dataset", "model", "training_method", "seed", "num_coordinates"]
    )

    for (dataset, model, method, seed, num_coords), group in grouped:
        nested = group[group["sampling_strategy"] == "nested"]
        indep = group[group["sampling_strategy"] == "independent"]

        if nested.empty or indep.empty:
            continue

        nested_ece = nested["ece_geometric"].mean()
        indep_ece = indep["ece_geometric"].mean()
        nested_std = nested["ece_geometric"].std()
        indep_std = indep["ece_geometric"].std()

        if nested_ece < indep_ece:
            winner = "nested"
        elif indep_ece < nested_ece:
            winner = "independent"
        else:
            winner = "tie"

        pct_diff = (
            (nested_ece - indep_ece) / indep_ece * 100.0 if indep_ece > 0 else np.nan
        )

        records.append(
            {
                "dataset": dataset,
                "model": model,
                "training_method": method,
                "seed": seed,
                "num_coordinates": num_coords,
                "nested_ece": nested_ece,
                "independent_ece": indep_ece,
                "nested_ece_std": nested_std,
                "independent_ece_std": indep_std,
                "ece_diff": nested_ece - indep_ece,
                "pct_diff": pct_diff,
                "winner": winner,
                "nested_layers_touched_mean": nested["num_layers_touched"].mean(),
                "indep_layers_touched_mean": indep["num_layers_touched"].mean(),
                "nested_extraction_time_mean": nested["extraction_time_s"].mean(),
                "indep_extraction_time_mean": indep["extraction_time_s"].mean(),
            }
        )

    return pd.DataFrame(records)


def summarize_by_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics by num_coordinates and sampling strategy."""
    agg = (
        df.groupby(["num_coordinates", "sampling_strategy"])
        .agg(
            ece_geometric_mean=("ece_geometric", "mean"),
            ece_geometric_std=("ece_geometric", "std"),
            ece_geometric_min=("ece_geometric", "min"),
            ece_geometric_max=("ece_geometric", "max"),
            ece_dac_mean=("ece_dac", "mean"),
            ece_dac_std=("ece_dac", "std"),
            num_layers_touched_mean=("num_layers_touched", "mean"),
            num_layers_touched_std=("num_layers_touched", "std"),
            extraction_time_mean=("extraction_time_s", "mean"),
            extraction_time_std=("extraction_time_s", "std"),
            fit_time_mean=("fit_time_s", "mean"),
            fit_time_std=("fit_time_s", "std"),
            calibrate_time_mean=("calibrate_time_s", "mean"),
            calibrate_time_std=("calibrate_time_s", "std"),
            imp_vs_uncalibrated_mean=("imp_vs_uncalibrated", "mean"),
            imp_vs_temp_scaling_mean=("imp_vs_temp_scaling", "mean"),
            imp_vs_isotonic_mean=("imp_vs_isotonic", "mean"),
            imp_vs_dac_mean=("imp_vs_dac", "mean"),
            imp_vs_l6_layer_mean=("imp_vs_l6_layer", "mean"),
            count=("ece_geometric", "count"),
        )
        .reset_index()
    )
    return agg


def identify_sweet_spots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify 'sweet spot' num_coordinates per dataset/model/training_method:
    best tradeoff between ECE and compute (here: mean ECE, break ties by total time).
    """
    df = df.copy()
    df["total_compute_time_s"] = (
        df["extraction_time_s"].fillna(0)
        + df["fit_time_s"].fillna(0)
        + df["calibrate_time_s"].fillna(0)
    )

    sweet_records: List[Dict] = []
    grouped = df.groupby(["dataset", "model", "training_method", "sampling_strategy"])
    for key, group in grouped:
        # Aggregate by num_coordinates
        by_k = (
            group.groupby("num_coordinates")
            .agg(
                ece_mean=("ece_geometric", "mean"),
                ece_std=("ece_geometric", "std"),
                time_mean=("total_compute_time_s", "mean"),
                layers_mean=("num_layers_touched", "mean"),
                count=("ece_geometric", "count"),
            )
            .reset_index()
        )
        if by_k.empty:
            continue

        # Choose K with lowest mean ECE; tie-break by time
        best_idx = by_k["ece_mean"].idxmin()
        best = by_k.loc[best_idx]

        sweet_records.append(
            {
                "dataset": key[0],
                "model": key[1],
                "training_method": key[2],
                "sampling_strategy": key[3],
                "best_num_coordinates": int(best["num_coordinates"]),
                "best_ece_mean": best["ece_mean"],
                "best_ece_std": best["ece_std"],
                "best_time_mean": best["time_mean"],
                "best_layers_mean": best["layers_mean"],
                "num_k_values": int(by_k.shape[0]),
            }
        )

    return pd.DataFrame(sweet_records)


def statistical_comparison_coord_vs_l6(
    df: pd.DataFrame, k_value: int = 256, equivalence_margin: float = 0.005
) -> Dict:
    """
    Statistical comparison of coordinate approach vs L=6 layer selection.
    
    Tests:
    1. Paired t-test (H0: means are equal)
    2. Wilcoxon signed-rank (non-parametric)
    3. Equivalence test (TOST) with specified margin
    
    Args:
        df: DataFrame with coordinate ablation results merged with baselines
        k_value: Number of coordinates to use for comparison (default: 256)
        equivalence_margin: Equivalence margin in ECE units (default: 0.005 = 0.5%)
    
    Returns:
        Dictionary with statistical test results
    """
    # Filter to specific K and aggregate by config
    subset = df[df["num_coordinates"] == k_value].copy()
    
    if subset.empty:
        return {
            "n_pairs": 0,
            "error": f"No data found for K={k_value}",
        }
    
    # Aggregate coordinate ECE per (dataset, model, training_method, seed)
    # This averages across trials for each config
    coord_agg = (
        subset.groupby(["dataset", "model", "training_method", "seed"])
        .agg(
            coord_ece=("ece_geometric", "mean"),
            l6_ece=("baseline_l6_layer", "first"),  # L6 is same across trials
        )
        .dropna()  # Remove rows where either coord_ece or l6_ece is missing
        .reset_index()
    )
    
    if coord_agg.empty:
        return {
            "n_pairs": 0,
            "error": f"No paired data found for K={k_value} (missing L6 baselines)",
        }
    
    coord_vals = coord_agg["coord_ece"].values
    l6_vals = coord_agg["l6_ece"].values
    diffs = coord_vals - l6_vals
    
    n = len(diffs)
    if n < 2:
        return {
            "n_pairs": n,
            "error": f"Insufficient data for statistical tests (n={n}, need >= 2)",
        }
    
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    
    # 95% CI for mean difference
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    # Paired t-test (H0: mean difference = 0)
    t_stat, t_pval = stats.ttest_rel(coord_vals, l6_vals)
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    w_stat, w_pval = stats.wilcoxon(coord_vals, l6_vals, alternative="two-sided")
    
    # TOST equivalence test
    # H0: |mean_diff| >= margin, H1: |mean_diff| < margin
    # We test two one-sided hypotheses:
    # H01: mean_diff >= margin (reject if p_upper < 0.05)
    # H02: mean_diff <= -margin (reject if p_lower < 0.05)
    # Equivalence established if both are rejected (max(p_upper, p_lower) < 0.05)
    t_upper = (mean_diff - equivalence_margin) / se_diff
    t_lower = (mean_diff + equivalence_margin) / se_diff
    p_upper = stats.t.cdf(t_upper, df=n - 1)
    p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)
    tost_pval = max(p_upper, p_lower)
    
    equivalence_established = tost_pval < 0.05
    
    return {
        "k_value": k_value,
        "n_pairs": n,
        "coord_ece_mean": float(np.mean(coord_vals)),
        "coord_ece_std": float(np.std(coord_vals, ddof=1)),
        "l6_ece_mean": float(np.mean(l6_vals)),
        "l6_ece_std": float(np.std(l6_vals, ddof=1)),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pval),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pval),
        "equivalence_margin": equivalence_margin,
        "tost_pvalue": float(tost_pval),
        "equivalence_established": equivalence_established,
        "paired_data": coord_agg,
    }


def print_statistical_comparison(results: Dict) -> None:
    """
    Format and display statistical comparison results.
    
    Args:
        results: Dictionary returned by statistical_comparison_coord_vs_l6
    """
    if "error" in results:
        print(f"\n⚠️  Statistical Comparison Error: {results['error']}")
        return
    
    k = results.get("k_value", "?")
    n = results.get("n_pairs", 0)
    
    print("\n" + "=" * 80)
    print(f"STATISTICAL COMPARISON: Coordinate (K={k}) vs L=6 Layer Selection")
    print("=" * 80)
    
    print(f"\nSample Size: {n} paired configurations")
    print(f"  (matched on: dataset, model, training_method, seed)")
    
    print(f"\nMean ECE:")
    print(f"  Coordinate approach: {results['coord_ece_mean']:.6f} ± {results['coord_ece_std']:.6f}")
    print(f"  L=6 layer selection:  {results['l6_ece_mean']:.6f} ± {results['l6_ece_std']:.6f}")
    
    mean_diff = results["mean_diff"]
    ci_lower = results["ci_95_lower"]
    ci_upper = results["ci_95_upper"]
    
    print(f"\nMean Difference (Coord - L6): {mean_diff:+.6f}")
    print(f"95% Confidence Interval: [{ci_lower:+.6f}, {ci_upper:+.6f}]")
    
    print(f"\nStatistical Tests:")
    print(f"  1. Paired t-test:")
    print(f"     t-statistic = {results['t_statistic']:.4f}")
    print(f"     p-value    = {results['t_pvalue']:.6f}")
    if results['t_pvalue'] < 0.05:
        print(f"     → Reject H0: means are significantly different (p < 0.05)")
    else:
        print(f"     → Cannot reject H0: no significant difference (p >= 0.05)")
    
    print(f"\n  2. Wilcoxon signed-rank test (non-parametric):")
    print(f"     statistic = {results['wilcoxon_statistic']:.4f}")
    print(f"     p-value   = {results['wilcoxon_pvalue']:.6f}")
    if results['wilcoxon_pvalue'] < 0.05:
        print(f"     → Reject H0: distributions are significantly different (p < 0.05)")
    else:
        print(f"     → Cannot reject H0: no significant difference (p >= 0.05)")
    
    margin = results["equivalence_margin"]
    tost_pval = results["tost_pvalue"]
    equiv_est = results["equivalence_established"]
    
    print(f"\n  3. Equivalence Test (TOST) with margin = ±{margin:.4f} ({margin*100:.2f}% ECE):")
    print(f"     p-value = {tost_pval:.6f}")
    if equiv_est:
        print(f"     ✓ EQUIVALENCE ESTABLISHED (p < 0.05)")
        print(f"     → Mean difference is within ±{margin:.4f} ECE with 95% confidence")
    else:
        print(f"     ✗ Equivalence NOT established (p >= 0.05)")
        print(f"     → Cannot conclude that difference is within ±{margin:.4f} ECE")
    
    print("\n" + "-" * 80)
    print("Interpretation:")
    if equiv_est:
        print(f"  ✓ The coordinate approach (K={k}) and L=6 layer selection are")
        print(f"    statistically equivalent within ±{margin:.4f} ECE (p = {tost_pval:.6f})")
    else:
        if abs(mean_diff) < margin:
            print(f"  ⚠️  Mean difference ({mean_diff:+.6f}) is smaller than margin ({margin:.4f}),")
            print(f"     but equivalence test did not reach significance (p = {tost_pval:.6f})")
            print(f"     → May need more samples to establish equivalence")
        else:
            print(f"  ✗ Mean difference ({mean_diff:+.6f}) exceeds equivalence margin ({margin:.4f})")
            print(f"     → Methods are NOT equivalent within the specified margin")
    
    print("=" * 80 + "\n")


def summarize_by_dataset_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics by (dataset, model, training_method, num_coordinates).
    This is the main view the user wants for model-by-dataset breakdown.
    """
    group_cols = ["dataset", "model", "training_method", "num_coordinates"]
    agg = (
        df.groupby(group_cols)
        .agg(
            ece_geometric_mean=("ece_geometric", "mean"),
            ece_geometric_std=("ece_geometric", "std"),
            ece_geometric_min=("ece_geometric", "min"),
            ece_geometric_max=("ece_geometric", "max"),
            count=("ece_geometric", "count"),
            imp_vs_uncalibrated_mean=("imp_vs_uncalibrated", "mean"),
            imp_vs_temp_scaling_mean=("imp_vs_temp_scaling", "mean"),
            imp_vs_isotonic_mean=("imp_vs_isotonic", "mean"),
            imp_vs_dac_mean=("imp_vs_dac", "mean"),
            imp_vs_l6_layer_mean=("imp_vs_l6_layer", "mean"),
        )
        .reset_index()
    )
    return agg


def export_latex_by_dataset_model(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Export LaTeX tables breaking results down by dataset and model,
    loosely following the style of generate_sgc_tables.py.
    """
    if summary_df.empty:
        return

    latex_dir = output_dir / "latex_tables"
    latex_dir.mkdir(parents=True, exist_ok=True)

    for dataset in sorted(summary_df["dataset"].unique()):
        ds_df = summary_df[summary_df["dataset"] == dataset].copy()
        if ds_df.empty:
            continue

        ds_upper = str(dataset).upper()
        table_lines = []
        table_lines.append("% Coordinate ablation summary by model for dataset " + ds_upper)
        table_lines.append("\\begin{table}[htbp]")
        table_lines.append("\\centering")
        caption = (
            "Coordinate ablation ECE and improvement vs baselines on "
            f"{ds_upper}. Lower is better."
        )
        table_lines.append(f"\\caption{{{caption}}}")
        safe_dataset = str(dataset).replace(" ", "_")
        table_lines.append(f"\\label{{tab:coord_ablation_{safe_dataset}}}")
        table_lines.append("\\small")
        table_lines.append(
            "\\begin{tabular}{llrcccccc}"
        )
        table_lines.append("\\toprule")
        table_lines.append(
            "Training & Model & $K$ & ECE & $\\Delta$Uncal (\\%) & "
            "$\\Delta$TS (\\%) & $\\Delta$Iso (\\%) & $\\Delta$DAC (\\%) & $\\Delta$L6 (\\%) \\\\"
        )
        table_lines.append("\\midrule")

        # Sort for readability
        ds_df = ds_df.sort_values(
            ["training_method", "model", "num_coordinates"]
        )

        for _, row in ds_df.iterrows():
            training = str(row["training_method"])
            model = str(row["model"])
            k_val = int(row["num_coordinates"])
            ece_mean = row["ece_geometric_mean"]

            def fmt_pct(x):
                return "---" if pd.isna(x) else f"{x:.1f}"

            d_uncal = fmt_pct(row.get("imp_vs_uncalibrated_mean"))
            d_ts = fmt_pct(row.get("imp_vs_temp_scaling_mean"))
            d_iso = fmt_pct(row.get("imp_vs_isotonic_mean"))
            d_dac = fmt_pct(row.get("imp_vs_dac_mean"))
            d_l6 = fmt_pct(row.get("imp_vs_l6_layer_mean"))

            cells = [
                training.replace("_", "\\_"),
                model.replace("_", "\\_"),
                str(k_val),
                f"{ece_mean:.4f}" if not pd.isna(ece_mean) else "---",
                d_uncal,
                d_ts,
                d_iso,
                d_dac,
                d_l6,
            ]
            table_lines.append(" & ".join(cells) + " \\\\")

        table_lines.append("\\bottomrule")
        table_lines.append("\\end{tabular}")
        table_lines.append("\\end{table}")

        tex_path = latex_dir / f"coordinate_ablation_{safe_dataset}.tex"
        with open(tex_path, "w") as f:
            f.write("\n".join(table_lines))


def print_summary_tables(
    df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    sweet_df: pd.DataFrame,
) -> None:
    """Print formatted summary tables."""
    print("\n" + "=" * 80)
    print("COORDINATE ABLATION SUMMARY")
    print("=" * 80)

    # A. Nested vs Independent winner counts and consistency
    print("\n--- Nested vs Independent Comparison ---")
    if not comparison_df.empty:
        winner_counts = comparison_df["winner"].value_counts()
        print(f"Winner counts (per dataset/model/method/seed/K): {dict(winner_counts)}")
        print(
            f"Mean ECE diff (nested - independent): {comparison_df['ece_diff'].mean():.6f}"
        )
        print(f"Mean % diff: {comparison_df['pct_diff'].mean():.2f}%")
        print(
            "Consistency (ECE std across trials; lower = more consistent, averaged over all groups):"
        )
        print(
            f"  Nested   mean std: {comparison_df['nested_ece_std'].mean():.6f}"
        )
        print(
            f"  Indep.   mean std: {comparison_df['independent_ece_std'].mean():.6f}"
        )

    # B. ECE by num_coordinates and strategy + layers touched
    print("\n--- ECE and Layers by Coordinate Count and Strategy ---")
    if not summary_df.empty:
        display_cols = [
            "num_coordinates",
            "sampling_strategy",
            "ece_geometric_mean",
            "ece_geometric_std",
            "ece_geometric_min",
            "ece_geometric_max",
            "num_layers_touched_mean",
            "num_layers_touched_std",
            "extraction_time_mean",
            "extraction_time_std",
            "count",
        ]
        print(summary_df[display_cols].round(6).to_string(index=False))

    # C. Comparison with baselines
    print("\n--- Improvement vs Baselines (Geometric ECE) ---")
    for k in sorted(df["num_coordinates"].dropna().unique()):
        subset = df[df["num_coordinates"] == k]
        if subset.empty:
            continue
        print(f"\nK={int(k)}:")
        print(f"  Mean ECE: {subset['ece_geometric'].mean():.6f}")
        for col, label in [
            ("imp_vs_uncalibrated", "vs Uncalibrated"),
            ("imp_vs_temp_scaling", "vs Temp Scaling"),
            ("imp_vs_isotonic", "vs Isotonic"),
            ("imp_vs_dac", "vs DAC"),
            ("imp_vs_l6_layer", "vs L=6 Layers"),
        ]:
            if col in subset.columns:
                print(f"  {label}: {subset[col].mean():+.1f}%")

    # C.2. Per-dataset/model breakdown for the same quantities
    print("\n--- Improvement vs Baselines by Dataset and Model (Geometric ECE) ---")
    for (dataset, model), g in df.groupby(["dataset", "model"]):
        print(f"\nDataset={dataset}, Model={model}:")
        for k in sorted(g["num_coordinates"].dropna().unique()):
            subset = g[g["num_coordinates"] == k]
            if subset.empty:
                continue
            print(f"  K={int(k)}:")
            print(f"    Mean ECE: {subset['ece_geometric'].mean():.6f}")
            for col, label in [
                ("imp_vs_uncalibrated", "vs Uncalibrated"),
                ("imp_vs_temp_scaling", "vs Temp Scaling"),
                ("imp_vs_isotonic", "vs Isotonic"),
                ("imp_vs_dac", "vs DAC"),
                ("imp_vs_l6_layer", "vs L=6 Layers"),
            ]:
                if col in subset.columns:
                    print(f"    {label}: {subset[col].mean():+.1f}%")

    # D. Coordinate vs layer approach (K=256 vs L=6@d=256)
    print("\n--- Coordinate vs Layer Approach (K=256 vs L=6@d=256) ---")
    k256 = df[df["num_coordinates"] == 256]
    if not k256.empty and k256["baseline_l6_layer"].notna().any():
        coord_wins = (k256["ece_geometric"] < k256["baseline_l6_layer"]).sum()
        layer_wins = (k256["ece_geometric"] > k256["baseline_l6_layer"]).sum()
        ties = (k256["ece_geometric"] == k256["baseline_l6_layer"]).sum()
        print(f"  Coordinate wins: {coord_wins}")
        print(f"  Layer wins:      {layer_wins}")
        print(f"  Ties:            {ties}")
        print(f"  Mean coord ECE:  {k256['ece_geometric'].mean():.6f}")
        print(f"  Mean L=6 ECE:    {k256['baseline_l6_layer'].mean():.6f}")

    # E. Efficiency analysis: extraction time vs ECE
    print("\n--- Efficiency: Extraction Time vs ECE ---")
    per_group = []
    for (k, strat), group in df.groupby(
        ["num_coordinates", "sampling_strategy"]
    ):
        total_time = (
            group["extraction_time_s"].fillna(0)
            + group["fit_time_s"].fillna(0)
            + group["calibrate_time_s"].fillna(0)
        )
        per_group.append(
            {
                "num_coordinates": k,
                "sampling_strategy": strat,
                "ece_mean": group["ece_geometric"].mean(),
                "extraction_time_mean": group["extraction_time_s"].mean(),
                "total_time_mean": total_time.mean(),
                "count": group.shape[0],
            }
        )
    eff_df = pd.DataFrame(per_group)
    if not eff_df.empty:
        print(eff_df.round(6).to_string(index=False))

    # DAC comparison note
    print("\n--- DAC Comparison (Coordinate-based DAC ECE) ---")
    if "ece_dac" in df.columns and df["ece_dac"].notna().any():
        dac_summary = (
            df.groupby(["num_coordinates", "sampling_strategy"])
            .agg(
                ece_dac_mean=("ece_dac", "mean"),
                ece_dac_std=("ece_dac", "std"),
            )
            .reset_index()
        )
        print(dac_summary.round(6).to_string(index=False))
        overall_dac_mean = df["ece_dac"].mean()
        print(f"\nOverall DAC ECE (coordinate features): {overall_dac_mean:.6f}")
        print(
            "Note: High DAC ECE values here indicate DAC struggles with sparse coordinate embeddings."
        )

    # Sweet spot summary
    print("\n--- Sweet Spot Identification (Best K per config/strategy) ---")
    if not sweet_df.empty:
        print(
            sweet_df[
                [
                    "dataset",
                    "model",
                    "training_method",
                    "sampling_strategy",
                    "best_num_coordinates",
                    "best_ece_mean",
                    "best_time_mean",
                    "best_layers_mean",
                ]
            ]
            .round(6)
            .to_string(index=False)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate coordinate ablation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--coord-dir",
        type=str,
        default="/home/ptamar/geometric-internal-calibration/calibration_comparison_results_coori",
        help="Directory with coordinate ablation results",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="/home/ptamar/geometric-internal-calibration/calibration_comparison_results/random_ablation",
        help="Directory with baseline/random ablation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/ptamar/geometric-internal-calibration/calibration_comparison_results_coori/aggregated",
        help="Output directory for aggregated results",
    )
    parser.add_argument(
        "--stat-k-value",
        type=int,
        default=256,
        help="K value to use for statistical comparison vs L=6 (default: 256)",
    )
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=0.005,
        help="Equivalence margin for TOST test in ECE units (default: 0.005 = 0.5%%)",
    )

    args = parser.parse_args()

    coord_dir = Path(args.coord_dir)
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading coordinate ablation results...")
    coord_results = load_coordinate_results(coord_dir)
    print(f"Loaded {len(coord_results)} coordinate ablation trials")

    if not coord_results:
        print("No coordinate ablation results found. Exiting.")
        return

    print("Merging with baselines...")
    df = merge_with_baselines(coord_results, baseline_dir)

    print("Computing improvements vs baselines...")
    df = compute_improvements(df)

    print("Analyzing nested vs independent strategies...")
    comparison_df = analyze_nested_vs_independent(df)

    print("Summarizing by coordinate count...")
    summary_df = summarize_by_coordinates(df)

    print("Identifying sweet spots...")
    sweet_df = identify_sweet_spots(df)

    # Statistical comparison: Coordinate vs L=6
    print(f"\nRunning statistical comparison (K={args.stat_k_value} vs L=6)...")
    stat_results = statistical_comparison_coord_vs_l6(
        df, k_value=args.stat_k_value, equivalence_margin=args.equivalence_margin
    )
    print_statistical_comparison(stat_results)

    # Print summaries to stdout
    print_summary_tables(df, comparison_df, summary_df, sweet_df)

    # Save outputs
    full_path = output_dir / "coordinate_ablation_full.csv"
    compare_path = output_dir / "nested_vs_independent_comparison.csv"
    summary_path = output_dir / "summary_by_coordinates.csv"
    sweet_path = output_dir / "sweet_spots_by_config.csv"
    by_ds_model_path = output_dir / "summary_by_dataset_model.csv"

    # Per-dataset/model summary and LaTeX export
    by_ds_model_df = summarize_by_dataset_model(df)
    by_ds_model_df.to_csv(by_ds_model_path, index=False)
    export_latex_by_dataset_model(by_ds_model_df, output_dir)

    df.to_csv(full_path, index=False)
    comparison_df.to_csv(compare_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    sweet_df.to_csv(sweet_path, index=False)

    # Save statistical comparison results
    if "error" not in stat_results:
        # Convert paired_data DataFrame to dict for JSON serialization
        stat_results_save = stat_results.copy()
        if "paired_data" in stat_results_save and isinstance(
            stat_results_save["paired_data"], pd.DataFrame
        ):
            stat_results_save["paired_data"] = stat_results_save["paired_data"].to_dict(
                orient="records"
            )
        stat_path = output_dir / f"statistical_comparison_k{args.stat_k_value}.json"
        with open(stat_path, "w") as f:
            json.dump(stat_results_save, f, indent=2, default=str)
        print(f"Statistical comparison saved to: {stat_path}")

    print("\n" + "-" * 80)
    print(f"Full results saved to: {full_path}")
    print(f"Nested vs independent comparison saved to: {compare_path}")
    print(f"Summary by coordinates saved to: {summary_path}")
    print(f"Sweet spots by config saved to: {sweet_path}")
    print(f"Summary by dataset/model saved to: {by_ds_model_path}")
    print(f"LaTeX tables saved under: {output_dir / 'latex_tables'}")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    main()


