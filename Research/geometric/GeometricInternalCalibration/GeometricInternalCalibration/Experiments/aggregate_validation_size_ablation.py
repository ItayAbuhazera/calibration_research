#!/usr/bin/env python3
"""
Aggregate validation size ablation results into summary tables.

This script processes validation size ablation JSON files and generates:
1. Summary table: ECE and accuracy by validation size and feature mode
2. Crossover analysis: When SGC catches up to coordinate
3. Hypothesis validation: Whether RGCC shows better performance at small sizes

Usage:
    python aggregate_validation_size_ablation.py \
        --results-dir results/validation_size_ablation \
        --output-dir tables/validation_size_ablation
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all validation size ablation JSON files."""
    pattern = '*_validation_size_ablation.json'
    result_files = list(results_dir.glob(pattern))
    
    if not result_files:
        logger.warning(f"No validation size ablation JSON files found in {results_dir}")
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
    Flatten validation size ablation results into a DataFrame.
    
    Returns DataFrame with columns:
        model_name, dataset, seed, val_size, feature_mode,
        ece_mean, ece_std, accuracy_mean, accuracy_std, fit_time_mean
    """
    rows = []
    
    for result in results:
        config = result.get('experiment_config', {})
        model_name = config.get('model_name', 'unknown')
        dataset = config.get('dataset_name', 'unknown')
        seed = config.get('seed', 0)
        
        results_by_val_size = result.get('results_by_val_size', {})
        
        for val_size_str, val_size_results in results_by_val_size.items():
            try:
                val_size = int(val_size_str)
            except (ValueError, TypeError):
                continue
            
            for feature_mode, mode_results in val_size_results.items():
                if isinstance(mode_results, dict) and 'ece_mean' in mode_results:
                    rows.append({
                        'model_name': model_name,
                        'dataset': dataset,
                        'seed': seed,
                        'val_size': val_size,
                        'feature_mode': feature_mode,
                        'ece_mean': mode_results.get('ece_mean'),
                        'ece_std': mode_results.get('ece_std'),
                        'accuracy_mean': mode_results.get('accuracy_mean'),
                        'accuracy_std': mode_results.get('accuracy_std'),
                        'fit_time_mean': mode_results.get('fit_time_mean'),
                        'fit_time_std': mode_results.get('fit_time_std'),
                        'num_trials': len(mode_results.get('trials', [])),
                    })
    
    df = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df)} rows from {len(results)} result files")
    return df


def aggregate_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results across seeds for each (dataset, model, val_size, feature_mode).
    
    Returns DataFrame with mean and std across seeds.
    """
    if df.empty:
        return df
    
    group_cols = ['dataset', 'model_name', 'val_size', 'feature_mode']
    
    agg_dict = {
        'ece_mean': ['mean', 'std', 'count'],
        'ece_std': ['mean'],
        'accuracy_mean': ['mean', 'std'],
        'accuracy_std': ['mean'],
        'fit_time_mean': ['mean', 'std'],
        'num_trials': ['mean'],
    }
    
    # Filter to only columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Flatten column names
    agg_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                      for col in agg_df.columns]
    
    return agg_df


def create_summary_table(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table showing ECE by validation size and feature mode.
    
    Returns a pivoted DataFrame with val_size as index and feature_mode as columns.
    """
    if agg_df.empty:
        return pd.DataFrame()
    
    # Create formatted ECE strings (mean ± std)
    agg_df['ece_formatted'] = agg_df.apply(
        lambda row: f"{row['ece_mean_mean']:.4f} ± {row['ece_mean_std']:.4f}",
        axis=1
    )
    
    # Pivot table
    summary = agg_df.pivot_table(
        index=['dataset', 'model_name', 'val_size'],
        columns='feature_mode',
        values='ece_formatted',
        aggfunc='first'
    ).reset_index()
    
    return summary


def analyze_crossover_points(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze crossover points where SGC catches up to coordinate.
    
    Returns DataFrame with crossover analysis.
    """
    if agg_df.empty:
        return pd.DataFrame()
    
    rows = []
    
    for (dataset, model_name), group_df in agg_df.groupby(['dataset', 'model_name']):
        # Separate by feature mode
        sgc_df = group_df[group_df['feature_mode'] == 'sgc'].sort_values('val_size')
        coord_df = group_df[group_df['feature_mode'] == 'coordinate'].sort_values('val_size')
        
        if sgc_df.empty or coord_df.empty:
            continue
        
        # Find crossover point (where SGC ECE <= coordinate ECE)
        crossover = None
        for val_size in sorted(group_df['val_size'].unique()):
            sgc_row = sgc_df[sgc_df['val_size'] == val_size]
            coord_row = coord_df[coord_df['val_size'] == val_size]
            
            if not sgc_row.empty and not coord_row.empty:
                sgc_ece = sgc_row['ece_mean_mean'].iloc[0]
                coord_ece = coord_row['ece_mean_mean'].iloc[0]
                
                if sgc_ece <= coord_ece:
                    crossover = val_size
                    break
        
        # Compute advantage at small sizes (first 3 sizes)
        small_sizes = sorted(group_df['val_size'].unique())[:3]
        coord_better_count = 0
        for val_size in small_sizes:
            sgc_row = sgc_df[sgc_df['val_size'] == val_size]
            coord_row = coord_df[coord_df['val_size'] == val_size]
            
            if not sgc_row.empty and not coord_row.empty:
                sgc_ece = sgc_row['ece_mean_mean'].iloc[0]
                coord_ece = coord_row['ece_mean_mean'].iloc[0]
                if coord_ece < sgc_ece:
                    coord_better_count += 1
        
        hypothesis_supported = coord_better_count >= 2
        
        rows.append({
            'dataset': dataset,
            'model_name': model_name,
            'crossover_point': crossover,
            'coord_better_at_small': coord_better_count,
            'hypothesis_supported': hypothesis_supported,
        })
    
    return pd.DataFrame(rows)


def generate_latex_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Generate LaTeX table from summary DataFrame."""
    if summary_df.empty:
        logger.warning("Empty summary DataFrame, skipping LaTeX generation")
        return
    
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Validation Size Ablation: ECE by Validation Size and Feature Mode}",
        "\\label{tab:validation_size_ablation}",
        "\\begin{tabular}{lcc" + "c" * len([c for c in summary_df.columns if c not in ['dataset', 'model_name', 'val_size']]) + "}",
        "\\toprule",
    ]
    
    # Header
    header_cols = ['Dataset', 'Model', 'Val Size']
    for col in summary_df.columns:
        if col not in ['dataset', 'model_name', 'val_size']:
            header_cols.append(col.replace('_', ' ').title())
    
    lines.append(" & ".join(header_cols) + " \\\\")
    lines.append("\\midrule")
    
    # Rows
    for _, row in summary_df.iterrows():
        row_values = [
            str(row['dataset']),
            str(row['model_name']),
            str(int(row['val_size'])),
        ]
        for col in summary_df.columns:
            if col not in ['dataset', 'model_name', 'val_size']:
                val = row[col]
                if pd.isna(val):
                    row_values.append("---")
                else:
                    row_values.append(str(val))
        
        lines.append(" & ".join(row_values) + " \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    
    logger.info(f"Saved LaTeX table to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate validation size ablation results into summary tables"
    )
    parser.add_argument('--results-dir', type=Path, required=True,
                        help='Directory containing validation_size_ablation JSON files')
    parser.add_argument('--output-dir', type=Path, default=Path('tables/validation_size_ablation'),
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
    raw_csv_path = args.output_dir / "validation_size_ablation_raw.csv"
    df.to_csv(raw_csv_path, index=False)
    logger.info(f"Saved raw results to: {raw_csv_path}")
    
    # Aggregate across seeds
    agg_df = aggregate_by_config(df)
    agg_csv_path = args.output_dir / "validation_size_ablation_aggregated.csv"
    if not agg_df.empty:
        agg_df.to_csv(agg_csv_path, index=False)
        logger.info(f"Saved aggregated results to: {agg_csv_path}")
    
    # Create summary table
    summary_df = create_summary_table(agg_df)
    if not summary_df.empty:
        summary_csv_path = args.output_dir / "validation_size_ablation_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Saved summary table to: {summary_csv_path}")
        
        # Generate LaTeX table
        latex_path = args.output_dir / "table_validation_size_ablation.tex"
        generate_latex_table(summary_df, latex_path)
    
    # Analyze crossover points
    crossover_df = analyze_crossover_points(agg_df)
    if not crossover_df.empty:
        crossover_csv_path = args.output_dir / "validation_size_ablation_crossover.csv"
        crossover_df.to_csv(crossover_csv_path, index=False)
        logger.info(f"Saved crossover analysis to: {crossover_csv_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("CROSSOVER ANALYSIS")
        print("="*80)
        print(crossover_df.to_string(index=False))
    
    logger.info(f"\nAll tables saved to: {args.output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())
