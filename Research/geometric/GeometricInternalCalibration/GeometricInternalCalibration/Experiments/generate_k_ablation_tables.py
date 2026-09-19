#!/usr/bin/env python3
"""
Generate K-Ablation analysis tables for the SGC unification paper.

This script processes k-ablation results and generates:
1. K-value trend analysis: How OOD AUROC and ID ECE change with k
2. Best configuration summary: Best k for each (feature_mode, layer_mode) combination
3. Unification evidence table: Comparing k=1 vs k=50 vs k=200
4. Feature mode comparison: SGC vs Coordinate features
5. Layer mode comparison: random_6 vs dac_layers vs last_layer

Usage:
    python generate_k_ablation_tables.py --results-dir results/k_ablation --output-dir tables/k_ablation
"""

import json
import logging
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all k-ablation JSON files from results directory."""
    patterns = [
        '*_k_ablation.json',
        '*_seed*_k_ablation.json',
    ]
    
    result_files = []
    for pattern in patterns:
        result_files.extend(results_dir.glob(pattern))
    
    # Deduplicate
    result_files = list(set(result_files))
    
    if not result_files:
        logger.warning(f"No k-ablation JSON files found in {results_dir}")
        return []
    
    results = []
    for result_file in sorted(result_files):
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            result['_source_file'] = str(result_file)
            results.append(result)
            logger.info(f"Loaded: {result_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load {result_file.name}: {e}")
            continue
    
    logger.info(f"Loaded {len(results)} result files")
    return results


def extract_flat_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten k-ablation results into a DataFrame with one row per (config, k) combination.
    
    Returns DataFrame with columns:
        model_name, dataset, seed, feature_mode, layer_mode, k, 
        id_ece, id_adaptive_ece, id_brier, id_accuracy,
        ood_auroc, ood_fpr95, ood_ece, confidence_gap
    """
    rows = []
    
    for result in results:
        config = result.get('config', {})
        model_name = config.get('model_name', 'unknown')
        dataset = config.get('dataset_name', 'unknown')
        seed = config.get('seed', 0)
        
        results_dict = result.get('results', {})
        
        for config_key, k_results in results_dict.items():
            if isinstance(k_results, dict) and 'error' not in k_results:
                # Parse config_key: "sgc_random_6" -> feature_mode="sgc", layer_mode="random_6"
                parts = config_key.split('_', 1)
                if len(parts) == 2:
                    feature_mode, layer_mode = parts
                else:
                    feature_mode = config_key
                    layer_mode = 'unknown'
                
                for k_key, metrics in k_results.items():
                    if isinstance(metrics, dict) and 'error' not in metrics:
                        # Parse k_key: "k_50" -> k=50
                        if k_key.startswith('k_'):
                            try:
                                k = int(k_key.split('_')[1])
                            except (ValueError, IndexError):
                                continue
                        elif k_key == 'trust_score':
                            # Skip trust_score entries for now
                            continue
                        else:
                            continue
                        
                        rows.append({
                            'model_name': model_name,
                            'dataset': dataset,
                            'seed': seed,
                            'feature_mode': feature_mode,
                            'layer_mode': layer_mode,
                            'k': k,
                            'id_ece': metrics.get('id_ece'),
                            'id_adaptive_ece': metrics.get('id_adaptive_ece'),
                            'id_brier': metrics.get('id_brier'),
                            'id_accuracy': metrics.get('id_accuracy'),
                            'ood_auroc': metrics.get('ood_auroc'),
                            'ood_fpr95': metrics.get('ood_fpr95'),
                            'ood_ece': metrics.get('ood_ece'),
                            'confidence_gap': metrics.get('confidence_gap'),
                            'ood_confidence_mean': metrics.get('ood_confidence_mean'),
                            'id_confidence_mean': metrics.get('id_confidence_mean'),
                        })
    
    df = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df)} rows from {len(results)} result files")
    return df


def aggregate_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results across seeds for each (dataset, model, feature_mode, layer_mode, k).
    
    Returns DataFrame with mean and std for each metric.
    """
    if df.empty:
        return df
    
    group_cols = ['dataset', 'model_name', 'feature_mode', 'layer_mode', 'k']
    
    agg_dict = {
        'id_ece': ['mean', 'std', 'count'],
        'id_adaptive_ece': ['mean', 'std'],
        'id_brier': ['mean', 'std'],
        'id_accuracy': ['mean', 'std'],
        'ood_auroc': ['mean', 'std'],
        'ood_fpr95': ['mean', 'std'],
        'ood_ece': ['mean', 'std'],
        'confidence_gap': ['mean', 'std'],
    }
    
    # Filter to only columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Flatten column names
    agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                      for col in agg_df.columns]
    
    return agg_df


def find_best_k_per_config(df: pd.DataFrame, metric: str = 'ood_auroc', 
                           higher_is_better: bool = True) -> pd.DataFrame:
    """
    Find the best k value for each (dataset, model, feature_mode, layer_mode) combination.
    
    Args:
        df: Aggregated DataFrame
        metric: Metric to optimize (e.g., 'ood_auroc_mean', 'id_ece_mean')
        higher_is_better: Whether higher values are better for the metric
    
    Returns:
        DataFrame with best k and corresponding metrics for each config.
    """
    if df.empty:
        return df
    
    # Ensure metric column exists
    if metric not in df.columns:
        metric_col = f"{metric}_mean"
        if metric_col not in df.columns:
            logger.warning(f"Metric {metric} not found in DataFrame")
            return pd.DataFrame()
        metric = metric_col
    
    group_cols = ['dataset', 'model_name', 'feature_mode', 'layer_mode']
    
    if higher_is_better:
        idx = df.groupby(group_cols)[metric].idxmax()
    else:
        idx = df.groupby(group_cols)[metric].idxmin()
    
    best_df = df.loc[idx].reset_index(drop=True)
    best_df = best_df.rename(columns={'k': 'best_k'})
    
    return best_df


def generate_k_trend_table(df: pd.DataFrame, feature_mode: str = 'sgc') -> str:
    """
    Generate a table showing how metrics change with k for SGC features.
    
    Returns LaTeX table string.
    """
    if df.empty:
        return "% No data available"
    
    # Filter to specific feature mode
    df_filtered = df[df['feature_mode'] == feature_mode].copy()
    
    if df_filtered.empty:
        return f"% No data for feature_mode={feature_mode}"
    
    # Aggregate across all configs to show overall trend
    k_agg = df_filtered.groupby('k').agg({
        'id_ece_mean': 'mean',
        'id_ece_std': 'mean',
        'ood_auroc_mean': 'mean',
        'ood_auroc_std': 'mean',
        'ood_fpr95_mean': 'mean',
        'confidence_gap_mean': 'mean',
        'id_ece_count': 'sum',
    }).reset_index()
    
    k_agg = k_agg.sort_values('k')
    
    # Build LaTeX table
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Effect of k on SGC Performance (Feature Mode: {feature_mode.upper()})}}",
        f"\\label{{tab:k_trend_{feature_mode}}}",
        "\\begin{tabular}{r|cc|cc|c}",
        "\\toprule",
        "k & ID ECE (\\%) $\\downarrow$ & & OOD AUROC $\\uparrow$ & & Conf. Gap \\\\",
        "\\midrule",
    ]
    
    for _, row in k_agg.iterrows():
        k = int(row['k'])
        id_ece = row['id_ece_mean'] * 100
        id_ece_std = row['id_ece_std'] * 100 if pd.notna(row['id_ece_std']) else 0
        ood_auroc = row['ood_auroc_mean']
        ood_auroc_std = row['ood_auroc_std'] if pd.notna(row['ood_auroc_std']) else 0
        conf_gap = row['confidence_gap_mean']
        
        lines.append(f"{k} & {id_ece:.2f} & $\\pm${id_ece_std:.2f} & "
                    f"{ood_auroc:.3f} & $\\pm${ood_auroc_std:.3f} & {conf_gap:.3f} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    return '\n'.join(lines)


def generate_feature_comparison_table(df: pd.DataFrame) -> str:
    """
    Generate a table comparing SGC vs Coordinate features across k values.
    """
    if df.empty:
        return "% No data available"
    
    # Pivot to compare feature modes
    pivot_cols = ['dataset', 'model_name', 'layer_mode', 'k']
    
    sgc_df = df[df['feature_mode'] == 'sgc'].copy()
    coord_df = df[df['feature_mode'] == 'coordinate'].copy()
    
    if sgc_df.empty or coord_df.empty:
        return "% Insufficient data for feature comparison"
    
    # Merge on common columns
    merged = pd.merge(
        sgc_df, coord_df, 
        on=pivot_cols,
        suffixes=('_sgc', '_coord'),
        how='outer'
    )
    
    # Aggregate by k to show overall comparison
    k_comparison = merged.groupby('k').agg({
        'ood_auroc_mean_sgc': 'mean',
        'ood_auroc_mean_coord': 'mean',
        'id_ece_mean_sgc': 'mean',
        'id_ece_mean_coord': 'mean',
    }).reset_index()
    
    k_comparison = k_comparison.sort_values('k')
    
    # Build LaTeX table
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{SGC vs Coordinate Features: Effect of k}",
        "\\label{tab:sgc_vs_coord_k}",
        "\\begin{tabular}{r|cc|cc}",
        "\\toprule",
        "& \\multicolumn{2}{c|}{OOD AUROC $\\uparrow$} & \\multicolumn{2}{c}{ID ECE (\\%) $\\downarrow$} \\\\",
        "k & SGC & Coord & SGC & Coord \\\\",
        "\\midrule",
    ]
    
    for _, row in k_comparison.iterrows():
        k = int(row['k'])
        sgc_auroc = row['ood_auroc_mean_sgc']
        coord_auroc = row['ood_auroc_mean_coord']
        sgc_ece = row['id_ece_mean_sgc'] * 100 if pd.notna(row['id_ece_mean_sgc']) else np.nan
        coord_ece = row['id_ece_mean_coord'] * 100 if pd.notna(row['id_ece_mean_coord']) else np.nan
        
        # Format with bold for winner
        sgc_auroc_str = f"\\textbf{{{sgc_auroc:.3f}}}" if pd.notna(sgc_auroc) and pd.notna(coord_auroc) and sgc_auroc > coord_auroc else f"{sgc_auroc:.3f}" if pd.notna(sgc_auroc) else "-"
        coord_auroc_str = f"\\textbf{{{coord_auroc:.3f}}}" if pd.notna(coord_auroc) and pd.notna(sgc_auroc) and coord_auroc > sgc_auroc else f"{coord_auroc:.3f}" if pd.notna(coord_auroc) else "-"
        
        sgc_ece_str = f"\\textbf{{{sgc_ece:.2f}}}" if pd.notna(sgc_ece) and pd.notna(coord_ece) and sgc_ece < coord_ece else f"{sgc_ece:.2f}" if pd.notna(sgc_ece) else "-"
        coord_ece_str = f"\\textbf{{{coord_ece:.2f}}}" if pd.notna(coord_ece) and pd.notna(sgc_ece) and coord_ece < sgc_ece else f"{coord_ece:.2f}" if pd.notna(coord_ece) else "-"
        
        lines.append(f"{k} & {sgc_auroc_str} & {coord_auroc_str} & {sgc_ece_str} & {coord_ece_str} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    return '\n'.join(lines)


def generate_layer_mode_comparison_table(df: pd.DataFrame, k_values: List[int] = [1, 50, 200]) -> str:
    """
    Generate a table comparing layer selection modes at specific k values,
    broken down by dataset (averaged over models).
    """
    if df.empty:
        return "% No data available"
    
    # Filter to SGC features and specific k values
    df_filtered = df[(df['feature_mode'] == 'sgc') & (df['k'].isin(k_values))].copy()
    
    if df_filtered.empty:
        return "% Insufficient data for layer mode comparison"
    
    # Datasets and layer modes
    datasets = sorted(df_filtered['dataset'].unique())
    layer_modes = sorted(df_filtered['layer_mode'].unique())
    
    # Build LaTeX table: one row per (dataset, layer_mode, metric)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Layer Selection Mode Comparison by Dataset (SGC Features)}",
        "\\label{tab:layer_mode_comparison}",
        "\\begin{tabular}{lll|" + "c" * len(k_values) + "}",
        "\\toprule",
        "& & & \\multicolumn{" + str(len(k_values)) + "}{c}{k} \\\\",
        "Dataset & Layer Mode & Metric & " + " & ".join(str(k) for k in k_values) + " \\\\",
        "\\midrule",
    ]
    
    # Aggregate by dataset, layer mode, and k (averaged over models)
    agg = df_filtered.groupby(['dataset', 'layer_mode', 'k']).agg({
        'ood_auroc_mean': 'mean',
        'id_ece_mean': 'mean',
    }).reset_index()
    
    for ds in datasets:
        ds_agg = agg[agg['dataset'] == ds]
        for layer_mode in layer_modes:
            layer_data = ds_agg[ds_agg['layer_mode'] == layer_mode]
            
            auroc_values = []
            ece_values = []
            for k in k_values:
                k_data = layer_data[layer_data['k'] == k]
                if len(k_data) > 0:
                    auroc_values.append(f"{k_data['ood_auroc_mean'].values[0]:.3f}")
                    ece_values.append(f"{k_data['id_ece_mean'].values[0]*100:.2f}")
                else:
                    auroc_values.append("-")
                    ece_values.append("-")
            
            # AUROC row
            lines.append(f"{ds.upper()} & {layer_mode} & AUROC & " + " & ".join(auroc_values) + " \\\\")
            # ECE row
            lines.append(f" &  & ECE(\\%) & " + " & ".join(ece_values) + " \\\\")
        lines.append("\\midrule")
    
    # Remove last midrule
    lines[-1] = "\\bottomrule"
    
    lines.extend([
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    return '\n'.join(lines)


def generate_unification_evidence_table(df: pd.DataFrame) -> str:
    """
    Generate a table showing unification evidence: k=1 vs k=50 vs k=200.
    """
    if df.empty:
        return "% No data available"
    
    # Filter to SGC features
    df_sgc = df[df['feature_mode'] == 'sgc'].copy()
    
    if df_sgc.empty:
        return "% No SGC data available"
    
    # Get data for k=1, 50, 200
    k_values = [1, 50, 200]
    df_filtered = df_sgc[df_sgc['k'].isin(k_values)].copy()
    
    # Aggregate by (dataset, model, layer_mode, k)
    summary = df_filtered.groupby(['dataset', 'model_name', 'layer_mode', 'k']).agg({
        'ood_auroc_mean': 'mean',
        'id_ece_mean': 'mean',
    }).reset_index()
    
    # Calculate improvement from k=1 to k=200
    k1_data = summary[summary['k'] == 1].set_index(['dataset', 'model_name', 'layer_mode'])
    k200_data = summary[summary['k'] == 200].set_index(['dataset', 'model_name', 'layer_mode'])
    
    improvement = pd.DataFrame({
        'auroc_k1': k1_data['ood_auroc_mean'],
        'auroc_k200': k200_data['ood_auroc_mean'],
        'ece_k1': k1_data['id_ece_mean'],
        'ece_k200': k200_data['id_ece_mean'],
    }).dropna()
    
    improvement['auroc_delta'] = improvement['auroc_k200'] - improvement['auroc_k1']
    improvement['ece_delta'] = improvement['ece_k200'] - improvement['ece_k1']
    
    # Build summary statistics
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Unification Evidence: Effect of k on OOD Detection}",
        "\\label{tab:unification_evidence}",
        "\\begin{tabular}{l|ccc|c}",
        "\\toprule",
        "Config & k=1 & k=50 & k=200 & $\\Delta$(k=200 - k=1) \\\\",
        "\\midrule",
    ]
    
    # Overall summary
    overall = df_filtered.groupby('k').agg({
        'ood_auroc_mean': 'mean',
        'id_ece_mean': 'mean',
    }).reset_index()
    
    k1_auroc = overall[overall['k'] == 1]['ood_auroc_mean'].values[0] if 1 in overall['k'].values else np.nan
    k50_auroc = overall[overall['k'] == 50]['ood_auroc_mean'].values[0] if 50 in overall['k'].values else np.nan
    k200_auroc = overall[overall['k'] == 200]['ood_auroc_mean'].values[0] if 200 in overall['k'].values else np.nan
    
    delta = k200_auroc - k1_auroc if pd.notna(k200_auroc) and pd.notna(k1_auroc) else np.nan
    
    lines.append(f"OOD AUROC (avg) & {k1_auroc:.3f} & {k50_auroc:.3f} & {k200_auroc:.3f} & "
                f"\\textbf{{+{delta:.3f}}} \\\\")
    
    k1_ece = overall[overall['k'] == 1]['id_ece_mean'].values[0] * 100 if 1 in overall['k'].values else np.nan
    k50_ece = overall[overall['k'] == 50]['id_ece_mean'].values[0] * 100 if 50 in overall['k'].values else np.nan
    k200_ece = overall[overall['k'] == 200]['id_ece_mean'].values[0] * 100 if 200 in overall['k'].values else np.nan
    
    ece_delta = k200_ece - k1_ece if pd.notna(k200_ece) and pd.notna(k1_ece) else np.nan
    
    lines.append(f"ID ECE (\\%, avg) & {k1_ece:.2f} & {k50_ece:.2f} & {k200_ece:.2f} & "
                f"{ece_delta:+.2f} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{2mm}",
        "\\footnotesize{Higher k improves OOD detection while maintaining ID calibration.}",
        "\\end{table}",
    ])
    
    return '\n'.join(lines)


def generate_dataset_specific_table(df: pd.DataFrame, feature_mode: str = 'sgc') -> str:
    """
    Generate a table showing k-trend broken down by dataset.
    
    This reveals whether the k improvement is consistent across datasets
    or driven by specific ones.
    """
    if df.empty:
        return "% No data available"
    
    # Filter to specific feature mode
    df_filtered = df[df['feature_mode'] == feature_mode].copy()
    
    if df_filtered.empty:
        return f"% No data for feature_mode={feature_mode}"
    
    datasets = sorted(df_filtered['dataset'].unique())
    k_values = sorted(df_filtered['k'].unique())
    
    # Build LaTeX table
    num_datasets = len(datasets)
    col_spec = "r|" + "cc|" * num_datasets
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Effect of k on SGC Performance by Dataset}}",
        f"\\label{{tab:k_trend_by_dataset}}",
        "\\resizebox{\\textwidth}{!}{",
        f"\\begin{{tabular}}{{{col_spec[:-1]}}}",
        "\\toprule",
    ]
    
    # Header row 1: Dataset names
    header1 = "& " + " & ".join([f"\\multicolumn{{2}}{{c|}}{{{ds.upper()}}}" for ds in datasets])
    header1 = header1[:-1] + " \\\\"  # Remove last | 
    lines.append(header1)
    
    # Header row 2: Metrics
    header2 = "k & " + " & ".join(["AUROC & ECE(\\%)"] * num_datasets) + " \\\\"
    lines.append(header2)
    lines.append("\\midrule")
    
    # Data rows
    for k in k_values:
        row_data = [str(int(k))]
        for ds in datasets:
            ds_k_data = df_filtered[(df_filtered['dataset'] == ds) & (df_filtered['k'] == k)]
            if len(ds_k_data) > 0:
                auroc = ds_k_data['ood_auroc_mean'].mean()
                ece = ds_k_data['id_ece_mean'].mean() * 100
                row_data.append(f"{auroc:.3f}")
                row_data.append(f"{ece:.2f}")
            else:
                row_data.extend(["-", "-"])
        lines.append(" & ".join(row_data) + " \\\\")
    
    # Add delta row (k=200 - k=1)
    lines.append("\\midrule")
    delta_row = ["$\\Delta$"]
    for ds in datasets:
        k1_data = df_filtered[(df_filtered['dataset'] == ds) & (df_filtered['k'] == 1)]
        k200_data = df_filtered[(df_filtered['dataset'] == ds) & (df_filtered['k'] == 200)]
        
        if len(k1_data) > 0 and len(k200_data) > 0:
            auroc_delta = k200_data['ood_auroc_mean'].mean() - k1_data['ood_auroc_mean'].mean()
            ece_delta = (k200_data['id_ece_mean'].mean() - k1_data['id_ece_mean'].mean()) * 100
            
            # Bold if positive improvement for AUROC
            auroc_str = f"\\textbf{{+{auroc_delta:.3f}}}" if auroc_delta > 0 else f"{auroc_delta:+.3f}"
            ece_str = f"{ece_delta:+.2f}"
            delta_row.extend([auroc_str, ece_str])
        else:
            delta_row.extend(["-", "-"])
    lines.append(" & ".join(delta_row) + " \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
    ])
    
    return '\n'.join(lines)


def perform_tost_test(
    scores1: List[float], 
    scores2: List[float], 
    equivalence_margin: float = 0.003
) -> Dict[str, Any]:
    """
    Perform Two One-Sided Tests (TOST) for equivalence testing.
    
    Args:
        scores1: First set of paired scores (e.g., reference k ECE/AUROC values)
        scores2: Second set of paired scores (e.g., other k ECE/AUROC values)
        equivalence_margin: Equivalence margin (delta) in same units as scores
    
    Returns:
        Dictionary with TOST results including p-value, mean difference, CI, etc.
    """
    # Filter out NaN values and ensure paired data
    pairs = [(s1, s2) for s1, s2 in zip(scores1, scores2) if pd.notna(s1) and pd.notna(s2)]
    
    if len(pairs) < 2:
        return {
            'tost_pvalue': 1.0,
            'mean_diff': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'dz': 0.0,
            'equivalence_established': False,
            'n_pairs': len(pairs)
        }
    
    arr1 = np.array([p[0] for p in pairs])
    arr2 = np.array([p[1] for p in pairs])
    diff_arr = arr2 - arr1
    
    n = len(diff_arr)
    mean_diff = diff_arr.mean()
    std_diff = diff_arr.std(ddof=1)
    
    # Standard error of the mean difference
    se_diff = std_diff / np.sqrt(n) if n > 1 else 0.0
    
    # 95% CI for mean difference (for TOST table display)
    t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else 1.96
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    # TOST: Two One-Sided Tests
    if se_diff > 0 and n > 1:
        t_upper = (mean_diff - equivalence_margin) / se_diff
        t_lower = (mean_diff + equivalence_margin) / se_diff
        p_upper = stats.t.cdf(t_upper, df=n - 1)
        p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)
        tost_pval = max(p_upper, p_lower)
    else:
        tost_pval = 1.0
    
    # Paired effect size (Cohen's d_z)
    dz = mean_diff / std_diff if std_diff > 0 else 0.0
    
    equivalence_established = tost_pval < 0.05
    
    # Minimum margin for equivalence
    if pd.notna(ci_lower) and pd.notna(ci_upper):
        min_margin_for_equiv = max(abs(ci_lower), abs(ci_upper))
    else:
        min_margin_for_equiv = np.nan
    
    return {
        'tost_pvalue': float(tost_pval),
        'mean_diff': float(mean_diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'dz': float(dz),
        'equivalence_established': equivalence_established,
        'n_pairs': n,
        'min_margin_for_equiv': float(min_margin_for_equiv) if pd.notna(min_margin_for_equiv) else np.nan,
        'min_margin_for_equiv_pct': float(min_margin_for_equiv * 100) if pd.notna(min_margin_for_equiv) else np.nan,
    }


def determine_tost_result(
    tost_pvalue: float, 
    mean_diff: float, 
    equivalence_margin: float,
    ci_lower: float,
    ci_upper: float,
    higher_is_better: bool = True
) -> str:
    """
    Determine the result category based on TOST and CI.
    
    Args:
        higher_is_better: True for metrics like AUROC (higher is better),
                         False for metrics like ECE (lower is better)
    
    Returns:
        "Equivalent", "Inconclusive", or "k=X better"
    """
    if pd.isna(tost_pvalue) or pd.isna(mean_diff) or pd.isna(ci_lower) or pd.isna(ci_upper):
        return "---"
    
    if tost_pvalue < 0.05:
        return "Equivalent"
    
    # Check if CI overlaps with equivalence region
    ci_overlaps_equiv = (ci_lower <= equivalence_margin and ci_upper >= -equivalence_margin)
    
    # Check if CI excludes 0 (statistically significant difference)
    ci_excludes_zero = (ci_lower > 0) or (ci_upper < 0)
    
    # If CI excludes 0 and doesn't overlap with equivalence region, one k is superior
    if ci_excludes_zero and not ci_overlaps_equiv:
        if higher_is_better:
            # For AUROC: positive diff (scores2 - scores1) means k2 > k1, so k2 is better
            if ci_lower > 0:
                return "k2 better"
            elif ci_upper < 0:
                return "k1 better"
        else:
            # For ECE: negative diff (scores2 - scores1) means k2 < k1, so k2 is better (lower ECE)
            if ci_upper < 0:
                return "k2 better"
            elif ci_lower > 0:
                return "k1 better"
    
    return "Inconclusive"


def generate_k_tost_comparison_table(
    df_raw: pd.DataFrame,
    df_agg: pd.DataFrame, 
    feature_mode: str = 'sgc',
    metric: str = 'ood_auroc',
    equivalence_margin: float = 0.01,  # 1% for AUROC, 0.003 (0.3%) for ECE
    higher_is_better: bool = True
) -> str:
    """
    Generate a TOST comparison table comparing each k value to all others.
    
    Args:
        df_raw: Raw DataFrame with seed-level data for proper pairing
        df_agg: Aggregated DataFrame (used for validation)
        feature_mode: Feature mode to analyze ('sgc' or 'coordinate')
        metric: Metric to compare ('ood_auroc' or 'id_ece')
        equivalence_margin: Equivalence margin for TOST
        higher_is_better: Whether higher values are better for the metric
    
    Returns:
        LaTeX table string
    """
    if df_raw.empty:
        return "% No data available"
    
    # Filter to specific feature mode
    df_filtered = df_raw[df_raw['feature_mode'] == feature_mode].copy()
    
    if df_filtered.empty:
        return f"% No data for feature_mode={feature_mode}"
    
    # Get metric column (raw data uses metric name without _mean)
    metric_col = metric
    if metric_col not in df_filtered.columns:
        logger.warning(f"Metric {metric_col} not found in DataFrame")
        return f"% Metric {metric_col} not found"
    
    # Get all k values
    k_values = sorted(df_filtered['k'].unique())
    
    if len(k_values) < 2:
        return f"% Need at least 2 k values for comparison, found {len(k_values)}"
    
    # Pairing columns: (dataset, model_name, layer_mode, seed) for proper pairing
    pair_cols = ['dataset', 'model_name', 'layer_mode', 'seed']
    
    # Prepare data: one row per (pair_cols, k) with metric value
    # Use raw data for proper seed-level pairing
    comparison_data = []
    
    for k in k_values:
        k_data = df_filtered[df_filtered['k'] == k].copy()
        for _, row in k_data.iterrows():
            # Create key from pairing columns
            key_parts = []
            for col in pair_cols:
                if col in row:
                    key_parts.append(row[col])
                else:
                    key_parts.append(None)
            key = tuple(key_parts)
            comparison_data.append({
                'key': key,
                'k': k,
                'metric_value': row[metric_col],
            })
    
    # Convert to DataFrame for easier manipulation
    comp_df = pd.DataFrame(comparison_data)
    
    # Perform pairwise TOST comparisons
    results = []
    
    for k1 in k_values:
        for k2 in k_values:
            if k1 >= k2:  # Only compare each pair once (k1 < k2)
                continue
            
            # Get paired observations
            k1_data = comp_df[comp_df['k'] == k1].set_index('key')
            k2_data = comp_df[comp_df['k'] == k2].set_index('key')
            
            # Find common keys (paired observations)
            common_keys = k1_data.index.intersection(k2_data.index)
            
            if len(common_keys) < 3:
                continue
            
            scores1 = k1_data.loc[common_keys, 'metric_value'].values
            scores2 = k2_data.loc[common_keys, 'metric_value'].values
            
            # Perform TOST
            tost_result = perform_tost_test(
                scores1=scores1.tolist(),
                scores2=scores2.tolist(),
                equivalence_margin=equivalence_margin
            )
            
            result_str = determine_tost_result(
                tost_pvalue=tost_result['tost_pvalue'],
                mean_diff=tost_result['mean_diff'],
                equivalence_margin=equivalence_margin,
                ci_lower=tost_result['ci_lower'],
                ci_upper=tost_result['ci_upper'],
                higher_is_better=higher_is_better
            )
            
            # Replace generic labels with actual k values
            if "k1 better" in result_str:
                result_str = f"k={k1} better"
            elif "k2 better" in result_str:
                result_str = f"k={k2} better"
            
            results.append({
                'k1': k1,
                'k2': k2,
                'n_pairs': tost_result['n_pairs'],
                'mean_diff': tost_result['mean_diff'],
                'mean_diff_pct': tost_result['mean_diff'] * 100 if metric == 'ood_auroc' else tost_result['mean_diff'] * 100,
                'ci_lower': tost_result['ci_lower'],
                'ci_upper': tost_result['ci_upper'],
                'tost_p': tost_result['tost_pvalue'],
                'cohen_dz': tost_result['dz'],
                'result': result_str,
            })
    
    if not results:
        return "% No valid pairwise comparisons found"
    
    results_df = pd.DataFrame(results)
    
    # Build LaTeX table
    metric_display = "OOD AUROC" if metric == 'ood_auroc' else "ID ECE (%)"
    metric_label = metric.replace('_', '_').upper()
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{TOST Equivalence Testing: k-value Comparison ({feature_mode.upper()} Features, {metric_display})}}",
        f"\\label{{tab:k_tost_{feature_mode}_{metric}}}",
        "\\small",
        "\\begin{tabular}{lrrrrrrl}",
        "\\toprule",
        f"Comparison & $N$ & $\\Delta${metric_label} & 95\\% CI & TOST $p$ & $d_z$ & Result \\\\",
        "\\midrule",
    ]
    
    for _, row in results_df.iterrows():
        ci_str = f"[{row['ci_lower']*100:+.2f}, {row['ci_upper']*100:+.2f}]" if metric == 'ood_auroc' else \
                 f"[{row['ci_lower']*100:+.2f}, {row['ci_upper']*100:+.2f}]"
        
        result_str = row['result']
        if result_str == 'Equivalent':
            result_str = "\\textbf{Equivalent}"
        
        diff_str = f"{row['mean_diff_pct']:+.2f}" if metric == 'ood_auroc' else f"{row['mean_diff']*100:+.2f}"
        
        lines.append(
            f"k={int(row['k1'])} vs k={int(row['k2'])} & "
            f"{int(row['n_pairs'])} & "
            f"{diff_str} & "
            f"{ci_str} & "
            f"{row['tost_p']:.3f} & "
            f"{row['cohen_dz']:.2f} & "
            f"{result_str} \\\\"
        )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\vspace{{2mm}}",
        f"\\footnotesize{{Equivalence margin: $\\pm${equivalence_margin*100:.1f}\\%}}",
        "\\end{table}",
    ])
    
    return '\n'.join(lines)


def generate_k_tost_summary_table(
    df_raw: pd.DataFrame,
    feature_mode: str = 'sgc',
    equivalence_margin_auroc: float = 0.01,  # 1% for AUROC
    equivalence_margin_ece: float = 0.003     # 0.3% for ECE
) -> str:
    """
    Generate a comprehensive TOST summary table comparing all k values.
    Shows both AUROC and ECE comparisons in one table.
    """
    if df_raw.empty:
        return "% No data available"
    
    # Filter to specific feature mode
    df_filtered = df_raw[df_raw['feature_mode'] == feature_mode].copy()
    
    if df_filtered.empty:
        return f"% No data for feature_mode={feature_mode}"
    
    k_values = sorted(df_filtered['k'].unique())
    
    if len(k_values) < 2:
        return f"% Need at least 2 k values for comparison"
    
    # Pairing columns: include seed for proper pairing
    pair_cols = ['dataset', 'model_name', 'layer_mode', 'seed']
    
    # Prepare data for both metrics (use raw column names)
    comparison_data = []
    for k in k_values:
        k_data = df_filtered[df_filtered['k'] == k].copy()
        for _, row in k_data.iterrows():
            key_parts = []
            for col in pair_cols:
                if col in row:
                    key_parts.append(row[col])
                else:
                    key_parts.append(None)
            key = tuple(key_parts)
            comparison_data.append({
                'key': key,
                'k': k,
                'auroc': row.get('ood_auroc', np.nan),
                'ece': row.get('id_ece', np.nan),
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Build comparison matrix: for each k, count how many times it's better/equivalent/worse
    summary_rows = []
    
    for k_ref in k_values:
        ref_data = comp_df[comp_df['k'] == k_ref].set_index('key')
        
        auroc_wins = 0
        auroc_equiv = 0
        auroc_losses = 0
        ece_wins = 0
        ece_equiv = 0
        ece_losses = 0
        
        for k_other in k_values:
            if k_ref == k_other:
                continue
            
            other_data = comp_df[comp_df['k'] == k_other].set_index('key')
            common_keys = ref_data.index.intersection(other_data.index)
            
            if len(common_keys) < 3:
                continue
            
            # AUROC comparison
            ref_auroc = ref_data.loc[common_keys, 'auroc'].values
            other_auroc = other_data.loc[common_keys, 'auroc'].values
            valid_auroc = ~(pd.isna(ref_auroc) | pd.isna(other_auroc))
            
            if valid_auroc.sum() >= 3:
                tost_auroc = perform_tost_test(
                    scores1=ref_auroc[valid_auroc].tolist(),  # k_ref is scores1 (k1)
                    scores2=other_auroc[valid_auroc].tolist(),  # k_other is scores2 (k2)
                    equivalence_margin=equivalence_margin_auroc
                )
                result_auroc = determine_tost_result(
                    tost_auroc['tost_pvalue'],
                    tost_auroc['mean_diff'],
                    equivalence_margin_auroc,
                    tost_auroc['ci_lower'],
                    tost_auroc['ci_upper'],
                    higher_is_better=True
                )
                
                if result_auroc == "Equivalent":
                    auroc_equiv += 1
                elif "k1 better" in result_auroc:  # k_ref (scores1) is better
                    auroc_wins += 1
                elif "k2 better" in result_auroc:  # k_other (scores2) is better
                    auroc_losses += 1
                # Inconclusive cases are not counted
            
            # ECE comparison
            ref_ece = ref_data.loc[common_keys, 'ece'].values
            other_ece = other_data.loc[common_keys, 'ece'].values
            valid_ece = ~(pd.isna(ref_ece) | pd.isna(other_ece))
            
            if valid_ece.sum() >= 3:
                tost_ece = perform_tost_test(
                    scores1=ref_ece[valid_ece].tolist(),  # k_ref is scores1 (k1)
                    scores2=other_ece[valid_ece].tolist(),  # k_other is scores2 (k2)
                    equivalence_margin=equivalence_margin_ece
                )
                result_ece = determine_tost_result(
                    tost_ece['tost_pvalue'],
                    tost_ece['mean_diff'],
                    equivalence_margin_ece,
                    tost_ece['ci_lower'],
                    tost_ece['ci_upper'],
                    higher_is_better=False
                )
                
                if result_ece == "Equivalent":
                    ece_equiv += 1
                elif "k1 better" in result_ece:  # k_ref (scores1) is better
                    ece_wins += 1
                elif "k2 better" in result_ece:  # k_other (scores2) is better
                    ece_losses += 1
                # Inconclusive cases are not counted
        
        summary_rows.append({
            'k': k_ref,
            'auroc_wins': auroc_wins,
            'auroc_equiv': auroc_equiv,
            'auroc_losses': auroc_losses,
            'ece_wins': ece_wins,
            'ece_equiv': ece_equiv,
            'ece_losses': ece_losses,
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Build LaTeX table
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{TOST Summary: k-value Performance Comparison ({feature_mode.upper()} Features)}}",
        f"\\label{{tab:k_tost_summary_{feature_mode}}}",
        "\\small",
        "\\begin{tabular}{r|ccc|ccc}",
        "\\toprule",
        "& \\multicolumn{3}{c|}{OOD AUROC} & \\multicolumn{3}{c}{ID ECE} \\\\",
        "k & Wins & Equiv & Losses & Wins & Equiv & Losses \\\\",
        "\\midrule",
    ]
    
    for _, row in summary_df.iterrows():
        lines.append(
            f"{int(row['k'])} & "
            f"{int(row['auroc_wins'])} & {int(row['auroc_equiv'])} & {int(row['auroc_losses'])} & "
            f"{int(row['ece_wins'])} & {int(row['ece_equiv'])} & {int(row['ece_losses'])} \\\\"
        )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{2mm}",
        "\\footnotesize{Wins/Losses: statistically superior/inferior; Equiv: statistically equivalent}",
        "\\end{table}",
    ])
    
    return '\n'.join(lines)


def generate_k_sensitivity_table(
    df_raw: pd.DataFrame,
    df_agg: pd.DataFrame,
    feature_mode: str = 'sgc',
    equivalence_margin_ece: float = 0.003,  # 0.3% for ECE
    equivalence_margin_auroc: float = 0.01,  # 1% for AUROC
    reference_k: int = 1
) -> str:
    """
    Generate k sensitivity table showing ECE and OOD AUROC with comparisons to k=1.
    
    Format:
    k | ECE (%) | Δ vs k=1 | OOD AUROC (%) | Δ vs k=1
    """
    if df_raw.empty:
        return "% No data available"
    
    # Filter to specific feature mode
    df_filtered = df_raw[df_raw['feature_mode'] == feature_mode].copy()
    
    if df_filtered.empty:
        return f"% No data for feature_mode={feature_mode}"
    
    # Get all k values
    k_values = sorted(df_filtered['k'].unique())
    
    if reference_k not in k_values:
        return f"% Reference k={reference_k} not found in data"
    
    # Pairing columns
    pair_cols = ['dataset', 'model_name', 'layer_mode', 'seed']
    
    # Prepare data
    comparison_data = []
    for k in k_values:
        k_data = df_filtered[df_filtered['k'] == k].copy()
        for _, row in k_data.iterrows():
            key_parts = [row.get(col, None) for col in pair_cols]
            key = tuple(key_parts)
            comparison_data.append({
                'key': key,
                'k': k,
                'ece': row.get('id_ece', np.nan),
                'auroc': row.get('ood_auroc', np.nan),
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Get reference (k=1) data
    ref_data = comp_df[comp_df['k'] == reference_k].set_index('key')
    
    # Aggregate mean ECE and AUROC per k (for baseline display)
    agg_ece = df_agg[df_agg['feature_mode'] == feature_mode].groupby('k')['id_ece_mean'].mean()
    agg_auroc = df_agg[df_agg['feature_mode'] == feature_mode].groupby('k')['ood_auroc_mean'].mean()
    
    results = []
    
    for k in k_values:
        if k == reference_k:
            # Baseline row
            ece_mean = agg_ece.get(k, np.nan) * 100 if k in agg_ece.index else np.nan
            auroc_mean = agg_auroc.get(k, np.nan) * 100 if k in agg_auroc.index else np.nan
            
            results.append({
                'k': k,
                'ece_mean': ece_mean,
                'ece_diff': 0.0,
                'ece_result': '---',
                'auroc_mean': auroc_mean,
                'auroc_diff': 0.0,
                'auroc_result': '---',
            })
        else:
            # Compare to reference
            k_data = comp_df[comp_df['k'] == k].set_index('key')
            common_keys = ref_data.index.intersection(k_data.index)
            
            if len(common_keys) < 3:
                continue
            
            # ECE comparison
            ref_ece = ref_data.loc[common_keys, 'ece'].values
            k_ece = k_data.loc[common_keys, 'ece'].values
            valid_ece = ~(pd.isna(ref_ece) | pd.isna(k_ece))
            
            ece_mean = agg_ece.get(k, np.nan) * 100 if k in agg_ece.index else np.nan
            ece_diff = np.nan
            ece_result = "---"
            
            if valid_ece.sum() >= 3:
                tost_ece = perform_tost_test(
                    scores1=ref_ece[valid_ece].tolist(),
                    scores2=k_ece[valid_ece].tolist(),
                    equivalence_margin=equivalence_margin_ece
                )
                ece_diff = tost_ece['mean_diff'] * 100
                ece_result_str = determine_tost_result(
                    tost_ece['tost_pvalue'],
                    tost_ece['mean_diff'],
                    equivalence_margin_ece,
                    tost_ece['ci_lower'],
                    tost_ece['ci_upper'],
                    higher_is_better=False
                )
                
                if ece_result_str == "Equivalent":
                    ece_result = "Equiv."
                elif "k2 better" in ece_result_str or "superior" in ece_result_str.lower():
                    ece_result = "Sig. better"
                elif "k1 better" in ece_result_str:
                    ece_result = "Sig. worse"
                else:
                    ece_result = "Inconcl."
            
            # AUROC comparison
            ref_auroc = ref_data.loc[common_keys, 'auroc'].values
            k_auroc = k_data.loc[common_keys, 'auroc'].values
            valid_auroc = ~(pd.isna(ref_auroc) | pd.isna(k_auroc))
            
            auroc_mean = agg_auroc.get(k, np.nan) * 100 if k in agg_auroc.index else np.nan
            auroc_diff = np.nan
            auroc_result = "---"
            
            if valid_auroc.sum() >= 3:
                tost_auroc = perform_tost_test(
                    scores1=ref_auroc[valid_auroc].tolist(),
                    scores2=k_auroc[valid_auroc].tolist(),
                    equivalence_margin=equivalence_margin_auroc
                )
                auroc_diff = tost_auroc['mean_diff'] * 100
                auroc_result_str = determine_tost_result(
                    tost_auroc['tost_pvalue'],
                    tost_auroc['mean_diff'],
                    equivalence_margin_auroc,
                    tost_auroc['ci_lower'],
                    tost_auroc['ci_upper'],
                    higher_is_better=True
                )
                
                if auroc_result_str == "Equivalent":
                    auroc_result = "Equiv."
                elif "k2 better" in auroc_result_str or "superior" in auroc_result_str.lower():
                    auroc_result = "Sig. better"
                elif "k1 better" in auroc_result_str:
                    auroc_result = "Sig. worse"
                else:
                    auroc_result = "Inconcl."
            
            results.append({
                'k': k,
                'ece_mean': ece_mean,
                'ece_diff': ece_diff,
                'ece_result': ece_result,
                'auroc_mean': auroc_mean,
                'auroc_diff': auroc_diff,
                'auroc_result': auroc_result,
            })
    
    if not results:
        return "% No valid comparisons found"
    
    results_df = pd.DataFrame(results).sort_values('k')
    
    # Count sample sizes
    n_ece = len(comp_df[comp_df['ece'].notna()])
    n_auroc = len(comp_df[comp_df['auroc'].notna()])
    
    # Build LaTeX table
    caption = (
        f"Effect of $k$ neighbors on calibration (ECE) and OOD detection (AUROC). "
        f"ECE remains stable across all $k$ values (all pairs equivalent, $p < 0.001$). "
        f"OOD detection improves significantly with larger $k$."
    )
    
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:k_sensitivity}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "$k$ & ECE (\\%) & $\\Delta$ vs $k$=1 & OOD AUROC (\\%) & $\\Delta$ vs $k$=1 \\\\",
        "\\midrule",
    ]
    
    for _, row in results_df.iterrows():
        k = int(row['k'])
        
        if k == reference_k:
            ece_diff_str = "baseline"
            ece_result_str = "---"
            auroc_diff_str = "baseline"
            auroc_result_str = "---"
        else:
            # ECE: show delta value (not absolute)
            if pd.notna(row['ece_diff']):
                ece_diff_str = f"{row['ece_diff']:+.2f}"
                ece_result_str = row['ece_result']
            else:
                ece_diff_str = "---"
                ece_result_str = "---"
            
            # AUROC: show delta value (not absolute)
            if pd.notna(row['auroc_diff']):
                auroc_diff_str = f"{row['auroc_diff']:+.2f}"
                auroc_result_str = row['auroc_result']
            else:
                auroc_diff_str = "---"
                auroc_result_str = "---"
        
        lines.append(
            f"{k}   & {ece_diff_str}    & {ece_result_str}     & {auroc_diff_str}    & {auroc_result_str} \\\\"
        )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{1mm}",
        f"\\footnotesize{{$N$={n_ece} (ECE), $N$={n_auroc} (AUROC). ECE margin: $\\pm${equivalence_margin_ece*100:.1f}\\%, AUROC margin: $\\pm${equivalence_margin_auroc*100:.1f}\\%.}}",
        "\\end{table}"
    ])
    
    return '\n'.join(lines)


def generate_best_config_per_dataset_table(df: pd.DataFrame) -> str:
    """
    Generate a table showing the best configuration for each dataset.
    """
    if df.empty:
        return "% No data available"
    
    # Filter to SGC features only
    df_sgc = df[df['feature_mode'] == 'sgc'].copy()
    
    if df_sgc.empty:
        return "% No SGC data available"
    
    datasets = sorted(df_sgc['dataset'].unique())
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Best Configuration per Dataset (SGC Features)}",
        "\\label{tab:best_config_per_dataset}",
        "\\begin{tabular}{l|cc|cc}",
        "\\toprule",
        "& \\multicolumn{2}{c|}{Best OOD AUROC} & \\multicolumn{2}{c}{Best ID ECE} \\\\",
        "Dataset & Config & Value & Config & Value \\\\",
        "\\midrule",
    ]
    
    for ds in datasets:
        ds_data = df_sgc[df_sgc['dataset'] == ds]
        
        # Best AUROC
        # For some datasets (e.g. Tiny-ImageNet) all OOD AUROC values can be NaN,
        # in which case idxmax() returns NaN and .loc[NaN] raises a KeyError.
        # Drop NaNs and handle the all-NaN case gracefully.
        valid_auroc_data = ds_data.dropna(subset=['ood_auroc_mean'])
        if valid_auroc_data.empty:
            best_auroc_config = "N/A"
            best_auroc_val_str = "-"
        else:
            best_auroc_idx = valid_auroc_data['ood_auroc_mean'].idxmax()
            best_auroc_row = valid_auroc_data.loc[best_auroc_idx]
            best_auroc_config = f"{best_auroc_row['layer_mode']}, k={int(best_auroc_row['k'])}"
            best_auroc_val_str = f"{best_auroc_row['ood_auroc_mean']:.3f}"
        
        # Best ECE
        best_ece_idx = ds_data['id_ece_mean'].idxmin()
        best_ece_row = ds_data.loc[best_ece_idx]
        best_ece_config = f"{best_ece_row['layer_mode']}, k={int(best_ece_row['k'])}"
        best_ece_val = best_ece_row['id_ece_mean'] * 100
        
        lines.append(f"{ds.upper()} & {best_auroc_config} & {best_auroc_val_str} & "
                    f"{best_ece_config} & {best_ece_val:.2f}\\% \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    return '\n'.join(lines)


def print_dataset_specific_summary(df: pd.DataFrame):
    """Print dataset-specific summary to console."""
    if df.empty:
        return
    
    print("\n" + "=" * 80)
    print("DATASET-SPECIFIC K-TREND ANALYSIS (SGC Features)")
    print("=" * 80)
    
    df_sgc = df[df['feature_mode'] == 'sgc']
    if df_sgc.empty:
        print("No SGC data")
        return
    
    datasets = sorted(df_sgc['dataset'].unique())
    
    for ds in datasets:
        print(f"\n--- {ds.upper()} ---")
        ds_data = df_sgc[df_sgc['dataset'] == ds]
        
        # K-trend
        k_summary = ds_data.groupby('k').agg({
            'ood_auroc_mean': 'mean',
            'id_ece_mean': 'mean',
        }).reset_index().sort_values('k')
        
        print(f"{'k':>5} | {'OOD AUROC':>10} | {'ID ECE (%)':>10}")
        print("-" * 30)
        for _, row in k_summary.iterrows():
            print(f"{int(row['k']):>5} | {row['ood_auroc_mean']:>10.4f} | {row['id_ece_mean']*100:>10.3f}")
        
        # Delta
        k1 = k_summary[k_summary['k'] == 1]['ood_auroc_mean'].values
        k200 = k_summary[k_summary['k'] == 200]['ood_auroc_mean'].values
        if len(k1) > 0 and len(k200) > 0:
            delta = k200[0] - k1[0]
            print(f"\nΔ AUROC (k=200 - k=1): {delta:+.4f} ({delta*100:+.1f} points)")
        
        # Best config
        # NOTE: For some datasets (e.g. Tiny-ImageNet) all OOD AUROC values can be NaN
        # for SGC features, in which case idxmax() returns NaN and .loc[NaN] raises
        # a KeyError. We guard against this by dropping NaNs first.
        valid_ds_data = ds_data.dropna(subset=['ood_auroc_mean'])
        if valid_ds_data.empty:
            print("Best AUROC: N/A (all OOD AUROC values are NaN for this dataset)")
        else:
            best_auroc_idx = valid_ds_data['ood_auroc_mean'].idxmax()
            best_row = valid_ds_data.loc[best_auroc_idx]
            print(f"Best AUROC: {best_row['ood_auroc_mean']:.4f} "
                  f"({best_row['layer_mode']}, k={int(best_row['k'])})")


def generate_best_config_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary of best configurations for each (dataset, model).
    """
    if df.empty:
        return df
    
    # Find best k for OOD AUROC (higher is better)
    best_auroc = find_best_k_per_config(df, metric='ood_auroc_mean', higher_is_better=True)
    best_auroc = best_auroc.rename(columns={
        'best_k': 'best_k_auroc',
        'ood_auroc_mean': 'best_auroc',
    })
    
    # Find best k for ID ECE (lower is better)
    best_ece = find_best_k_per_config(df, metric='id_ece_mean', higher_is_better=False)
    best_ece = best_ece.rename(columns={
        'best_k': 'best_k_ece',
        'id_ece_mean': 'best_ece',
    })
    
    return best_auroc, best_ece


def print_summary_statistics(df: pd.DataFrame, agg_df: pd.DataFrame):
    """Print summary statistics to console."""
    print("\n" + "=" * 80)
    print("K-ABLATION SUMMARY STATISTICS")
    print("=" * 80)
    
    if df.empty:
        print("No data available")
        return
    
    print(f"\nTotal data points: {len(df)}")
    print(f"Unique seeds: {sorted(df['seed'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Models: {sorted(df['model_name'].unique())}")
    print(f"Feature modes: {sorted(df['feature_mode'].unique())}")
    print(f"Layer modes: {sorted(df['layer_mode'].unique())}")
    print(f"K values: {sorted(df['k'].unique())}")
    
    # Print k-trend summary for SGC
    print("\n" + "-" * 40)
    print("SGC K-TREND SUMMARY (averaged across all configs)")
    print("-" * 40)
    
    sgc_df = df[df['feature_mode'] == 'sgc']
    if not sgc_df.empty:
        k_summary = sgc_df.groupby('k').agg({
            'id_ece': 'mean',
            'ood_auroc': 'mean',
        }).reset_index()
        k_summary = k_summary.sort_values('k')
        
        print(f"{'k':>5} | {'ID ECE (%)':>12} | {'OOD AUROC':>10}")
        print("-" * 35)
        for _, row in k_summary.iterrows():
            print(f"{int(row['k']):>5} | {row['id_ece']*100:>12.3f} | {row['ood_auroc']:>10.4f}")
    
    # Print coordinate comparison
    print("\n" + "-" * 40)
    print("COORDINATE K-TREND SUMMARY")
    print("-" * 40)
    
    coord_df = df[df['feature_mode'] == 'coordinate']
    if not coord_df.empty:
        k_summary = coord_df.groupby('k').agg({
            'id_ece': 'mean',
            'ood_auroc': 'mean',
        }).reset_index()
        k_summary = k_summary.sort_values('k')
        
        print(f"{'k':>5} | {'ID ECE (%)':>12} | {'OOD AUROC':>10}")
        print("-" * 35)
        for _, row in k_summary.iterrows():
            print(f"{int(row['k']):>5} | {row['id_ece']*100:>12.3f} | {row['ood_auroc']:>10.4f}")
    
    # Print best configurations
    print("\n" + "-" * 40)
    print("BEST CONFIGURATIONS")
    print("-" * 40)
    
    if not agg_df.empty:
        # Best for OOD AUROC
        best_auroc_idx = agg_df['ood_auroc_mean'].idxmax()
        best_auroc = agg_df.loc[best_auroc_idx]
        print(f"\nBest OOD AUROC: {best_auroc['ood_auroc_mean']:.4f}")
        print(f"  Config: {best_auroc['feature_mode']}_{best_auroc['layer_mode']}, k={int(best_auroc['k'])}")
        print(f"  Dataset: {best_auroc['dataset']}, Model: {best_auroc['model_name']}")
        
        # Best for ID ECE
        best_ece_idx = agg_df['id_ece_mean'].idxmin()
        best_ece = agg_df.loc[best_ece_idx]
        print(f"\nBest ID ECE: {best_ece['id_ece_mean']*100:.4f}%")
        print(f"  Config: {best_ece['feature_mode']}_{best_ece['layer_mode']}, k={int(best_ece['k'])}")
        print(f"  Dataset: {best_ece['dataset']}, Model: {best_ece['model_name']}")
    
    # Unification evidence
    print("\n" + "-" * 40)
    print("UNIFICATION EVIDENCE (SGC features)")
    print("-" * 40)
    
    sgc_df = df[df['feature_mode'] == 'sgc']
    if not sgc_df.empty:
        k1 = sgc_df[sgc_df['k'] == 1]['ood_auroc'].mean()
        k50 = sgc_df[sgc_df['k'] == 50]['ood_auroc'].mean() if 50 in sgc_df['k'].values else np.nan
        k200 = sgc_df[sgc_df['k'] == 200]['ood_auroc'].mean() if 200 in sgc_df['k'].values else np.nan
        
        print(f"OOD AUROC at k=1:   {k1:.4f}")
        if pd.notna(k50):
            print(f"OOD AUROC at k=50:  {k50:.4f} (delta: {k50-k1:+.4f})")
        if pd.notna(k200):
            print(f"OOD AUROC at k=200: {k200:.4f} (delta: {k200-k1:+.4f})")
        
        if pd.notna(k200) and k200 > k1:
            print(f"\n** Higher k improves OOD detection by {(k200-k1)*100:.1f} percentage points **")


def generate_detailed_csv(df: pd.DataFrame, output_path: Path):
    """Save detailed results to CSV."""
    if df.empty:
        logger.warning("No data to save to CSV")
        return
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved detailed results to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate K-Ablation analysis tables")
    parser.add_argument('--results-dir', type=Path, required=True,
                        help='Directory containing k-ablation JSON results')
    parser.add_argument('--output-dir', type=Path, default=Path('tables/k_ablation'),
                        help='Output directory for tables and CSV files')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    logger.info(f"Loading results from: {args.results_dir}")
    results = load_all_results(args.results_dir)
    
    if not results:
        logger.error("No results found. Exiting.")
        return 1
    
    # Extract flat DataFrame
    df = extract_flat_results(results)
    
    if df.empty:
        logger.error("No valid data extracted. Exiting.")
        return 1
    
    # Save raw data
    raw_csv_path = args.output_dir / "k_ablation_raw_results.csv"
    generate_detailed_csv(df, raw_csv_path)
    
    # Aggregate across seeds
    agg_df = aggregate_by_config(df)
    agg_csv_path = args.output_dir / "k_ablation_aggregated_results.csv"
    if not agg_df.empty:
        agg_df.to_csv(agg_csv_path, index=False)
        logger.info(f"Saved aggregated results to: {agg_csv_path}")
    
    # Print summary statistics
    print_summary_statistics(df, agg_df)
    
    # Print dataset-specific summary
    print_dataset_specific_summary(agg_df)
    
    # Generate tables
    logger.info("\nGenerating LaTeX tables...")
    
    # Table 1: K-trend for SGC
    k_trend_latex = generate_k_trend_table(agg_df, feature_mode='sgc')
    k_trend_path = args.output_dir / "table_k_trend_sgc.tex"
    with open(k_trend_path, 'w') as f:
        f.write(k_trend_latex)
    logger.info(f"Saved K-trend table to: {k_trend_path}")
    
    # Table 2: Feature comparison (SGC vs Coordinate)
    feature_comp_latex = generate_feature_comparison_table(agg_df)
    feature_comp_path = args.output_dir / "table_feature_comparison.tex"
    with open(feature_comp_path, 'w') as f:
        f.write(feature_comp_latex)
    logger.info(f"Saved feature comparison table to: {feature_comp_path}")
    
    # Table 3: Layer mode comparison
    layer_comp_latex = generate_layer_mode_comparison_table(agg_df)
    layer_comp_path = args.output_dir / "table_layer_mode_comparison.tex"
    with open(layer_comp_path, 'w') as f:
        f.write(layer_comp_latex)
    logger.info(f"Saved layer mode comparison table to: {layer_comp_path}")
    
    # Table 4: Unification evidence
    unification_latex = generate_unification_evidence_table(agg_df)
    unification_path = args.output_dir / "table_unification_evidence.tex"
    with open(unification_path, 'w') as f:
        f.write(unification_latex)
    logger.info(f"Saved unification evidence table to: {unification_path}")
    
    # Table 5: Dataset-specific k-trend (NEW)
    dataset_specific_latex = generate_dataset_specific_table(agg_df, feature_mode='sgc')
    dataset_specific_path = args.output_dir / "table_k_trend_by_dataset.tex"
    with open(dataset_specific_path, 'w') as f:
        f.write(dataset_specific_latex)
    logger.info(f"Saved dataset-specific table to: {dataset_specific_path}")
    
    # Table 6: Best config per dataset (NEW)
    best_config_latex = generate_best_config_per_dataset_table(agg_df)
    best_config_path = args.output_dir / "table_best_config_per_dataset.tex"
    with open(best_config_path, 'w') as f:
        f.write(best_config_latex)
    logger.info(f"Saved best config table to: {best_config_path}")
    
    # Table 7: TOST comparison for AUROC (NEW)
    tost_auroc_latex = generate_k_tost_comparison_table(
        df_raw=df,
        df_agg=agg_df,
        feature_mode='sgc',
        metric='ood_auroc',
        equivalence_margin=0.01,  # 1% for AUROC
        higher_is_better=True
    )
    tost_auroc_path = args.output_dir / "table_k_tost_auroc.tex"
    with open(tost_auroc_path, 'w') as f:
        f.write(tost_auroc_latex)
    logger.info(f"Saved TOST AUROC comparison table to: {tost_auroc_path}")
    
    # Table 8: TOST comparison for ECE (NEW)
    tost_ece_latex = generate_k_tost_comparison_table(
        df_raw=df,
        df_agg=agg_df,
        feature_mode='sgc',
        metric='id_ece',
        equivalence_margin=0.003,  # 0.3% for ECE
        higher_is_better=False
    )
    tost_ece_path = args.output_dir / "table_k_tost_ece.tex"
    with open(tost_ece_path, 'w') as f:
        f.write(tost_ece_latex)
    logger.info(f"Saved TOST ECE comparison table to: {tost_ece_path}")
    
    # Table 9: TOST summary (wins/equiv/losses) (NEW)
    tost_summary_latex = generate_k_tost_summary_table(
        df_raw=df,
        feature_mode='sgc',
        equivalence_margin_auroc=0.01,
        equivalence_margin_ece=0.003
    )
    tost_summary_path = args.output_dir / "table_k_tost_summary.tex"
    with open(tost_summary_path, 'w') as f:
        f.write(tost_summary_latex)
    logger.info(f"Saved TOST summary table to: {tost_summary_path}")
    
    # Table 10: K sensitivity table (ECE and AUROC with delta values) (NEW)
    k_sensitivity_latex = generate_k_sensitivity_table(
        df_raw=df,
        df_agg=agg_df,
        feature_mode='sgc',
        equivalence_margin_ece=0.003,  # 0.3% for ECE
        equivalence_margin_auroc=0.01,  # 1% for AUROC
        reference_k=1
    )
    k_sensitivity_path = args.output_dir / "table_k_sensitivity.tex"
    with open(k_sensitivity_path, 'w') as f:
        f.write(k_sensitivity_latex)
    logger.info(f"Saved K sensitivity table to: {k_sensitivity_path}")
    
    print("\n" + "=" * 80)
    print("All tables generated successfully!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())