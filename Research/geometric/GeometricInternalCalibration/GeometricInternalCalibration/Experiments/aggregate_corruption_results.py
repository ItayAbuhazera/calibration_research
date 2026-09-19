#!/usr/bin/env python3
"""
aggregate_corruption_results.py

Aggregates results from multiple corruption ablation JSON files.
Computes mean/std mCE and per-corruption ECE statistics across seeds/models.

Usage:
    python aggregate_corruption_results.py --input_dir corruption_ablation_results
    python aggregate_corruption_results.py --input_dir corruption_ablation_results --group_by both --output_csv summary.csv
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np


# CIFAR-C corruption types (15 total)
CIFAR_C_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


def parse_filename(filename: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Parse corruption ablation filename to extract dataset, model, and seed.
    
    Format: corruption_ablation_{dataset}_{model}_s{seed}.json
    
    Returns:
        (dataset, model, seed) or (None, None, None) if parsing fails
    """
    pattern = r"corruption_ablation_(cifar10|cifar100)_([^_]+)_s(\d+)\.json"
    match = re.match(pattern, filename)
    if match:
        dataset = match.group(1)
        model = match.group(2)
        seed = int(match.group(3))
        return dataset, model, seed
    return None, None, None


def load_result_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load a single JSON result file, return None if invalid."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load {filepath}: {e}")
        return None


def extract_method_data(
    result: Dict[str, Any],
    section_name: str
) -> Dict[str, Dict[str, Any]]:
    """
    Extract method data from a section (geometric_methods or coordinate_methods).
    
    Returns:
        Dict mapping method_name -> {mce: float, corruption_eces: dict, error: str}
    """
    methods = {}
    section = result.get(section_name, {})
    
    if not isinstance(section, dict):
        return methods
    
    for method_name, method_data in section.items():
        if not isinstance(method_data, dict):
            continue
        
        # Skip if error exists and mce is None
        if method_data.get('error') and method_data.get('mce') is None:
            continue
        
        methods[method_name] = {
            'mce': method_data.get('mce'),
            'corruption_eces': method_data.get('corruption_eces', {}),
            'error': method_data.get('error'),
        }
    
    return methods


def collect_all_results(
    input_dir: str,
    group_by: str = 'dataset'
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Load all result files and group them.
    
    Args:
        input_dir: Directory containing JSON files
        group_by: 'dataset', 'architecture', or 'both'
    
    Returns:
        Nested dict: group_key -> method_name -> list of method results
        For group_by='dataset': key is dataset name
        For group_by='architecture': key is model name
        For group_by='both': key is f"{dataset}_{model}"
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Collect all results
    all_results = defaultdict(lambda: defaultdict(list))
    
    # Find all matching JSON files
    json_files = list(input_path.glob("corruption_ablation_*.json"))
    json_files = [f for f in json_files if f.name != "best_model.json"]
    
    print(f"Found {len(json_files)} result files")
    
    for json_file in json_files:
        # Parse filename
        dataset, model, seed = parse_filename(json_file.name)
        if dataset is None:
            print(f"Warning: Could not parse filename: {json_file.name}")
            # Try to extract from config
            data = load_result_file(json_file)
            if data and 'config' in data:
                config = data['config']
                dataset = config.get('dataset')
                # Try to extract model from model_path
                model_path = config.get('model_path', '')
                model_match = re.search(r'/(resnet\d+|densenet\d+)/', model_path)
                if model_match:
                    model = model_match.group(1)
        
        if dataset is None or model is None:
            print(f"Warning: Skipping {json_file.name} (could not determine dataset/model)")
            continue
        
        # Load result file
        data = load_result_file(json_file)
        if data is None:
            continue
        
        # Determine group key
        if group_by == 'dataset':
            group_key = dataset
        elif group_by == 'architecture':
            group_key = model
        elif group_by == 'both':
            group_key = f"{dataset}_{model}"
        else:
            raise ValueError(f"Invalid group_by: {group_by}")
        
        # Extract methods from geometric_methods
        geometric_methods = extract_method_data(data, 'geometric_methods')
        for method_name, method_data in geometric_methods.items():
            all_results[group_key][method_name].append({
                'dataset': dataset,
                'model': model,
                'seed': seed,
                'mce': method_data['mce'],
                'corruption_eces': method_data['corruption_eces'],
                'error': method_data.get('error'),
            })
        
        # Extract methods from coordinate_methods
        coordinate_methods = extract_method_data(data, 'coordinate_methods')
        for method_name, method_data in coordinate_methods.items():
            all_results[group_key][method_name].append({
                'dataset': dataset,
                'model': model,
                'seed': seed,
                'mce': method_data['mce'],
                'corruption_eces': method_data['corruption_eces'],
                'error': method_data.get('error'),
            })
    
    return all_results


def compute_statistics(
    method_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute statistics for a method across multiple runs.
    
    Args:
        method_results: List of method result dicts
    
    Returns:
        Dict with mean_mce, std_mce, mean_corruption_eces, n_samples
    """
    # Filter out None mce values
    valid_results = [r for r in method_results if r['mce'] is not None]
    
    if len(valid_results) == 0:
        return {
            'mean_mce': None,
            'std_mce': None,
            'mean_corruption_eces': {},
            'n_samples': 0,
        }
    
    # Compute mCE statistics
    mces = [r['mce'] for r in valid_results]
    mean_mce = np.mean(mces)
    std_mce = np.std(mces, ddof=1) if len(mces) > 1 else 0.0
    
    # Compute per-corruption ECE means
    mean_corruption_eces = {}
    for corruption in CIFAR_C_CORRUPTIONS:
        eces = []
        for r in valid_results:
            corruption_eces = r.get('corruption_eces', {})
            if corruption in corruption_eces and corruption_eces[corruption] is not None:
                eces.append(corruption_eces[corruption])
        
        if len(eces) > 0:
            mean_corruption_eces[corruption] = np.mean(eces)
        else:
            mean_corruption_eces[corruption] = None
    
    return {
        'mean_mce': mean_mce,
        'std_mce': std_mce,
        'mean_corruption_eces': mean_corruption_eces,
        'n_samples': len(valid_results),
    }


def create_summary_table(
    all_results: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> pd.DataFrame:
    """
    Create a summary DataFrame with aggregated statistics.
    
    Returns:
        DataFrame with columns: group, method, mean_mce, std_mce, n_samples, 
                                 and one column per corruption type
    """
    rows = []
    
    for group_key, methods in sorted(all_results.items()):
        for method_name, method_results in sorted(methods.items()):
            stats = compute_statistics(method_results)
            
            if stats['n_samples'] == 0:
                continue  # Skip methods with no valid results
            
            row = {
                'group': group_key,
                'method': method_name,
                'mean_mce': stats['mean_mce'],
                'std_mce': stats['std_mce'],
                'n_samples': stats['n_samples'],
            }
            
            # Add per-corruption means
            for corruption in CIFAR_C_CORRUPTIONS:
                row[corruption] = stats['mean_corruption_eces'].get(corruption)
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def print_formatted_table(df: pd.DataFrame, group_by: str):
    """Print a nicely formatted table to console."""
    if len(df) == 0:
        print("No results to display.")
        return
    
    print("\n" + "="*120)
    print(f"CORRUPTION ABLATION RESULTS SUMMARY (grouped by {group_by})")
    print("="*120)
    
    # Group by group_key for display
    for group_key in sorted(df['group'].unique()):
        group_df = df[df['group'] == group_key].copy()
        
        print(f"\n{'='*120}")
        print(f"Group: {group_key}")
        print(f"{'='*120}")
        
        # Sort by mean_mce (ascending, best first)
        # Handle NaN values by filling with a large number for sorting
        group_df = group_df.copy()
        group_df['_sort_key'] = group_df['mean_mce'].fillna(float('inf'))
        group_df = group_df.sort_values('_sort_key').drop('_sort_key', axis=1)
        
        # Print header
        header = f"{'Method':<40} {'N':<5} {'Mean mCE':<12} {'Std mCE':<12}"
        corruption_header = " | ".join([c[:8] for c in CIFAR_C_CORRUPTIONS])
        print(header + " | " + corruption_header)
        print("-" * 120)
        
        # Print rows
        for _, row in group_df.iterrows():
            method = row['method']
            n = int(row['n_samples'])
            mean_mce = row['mean_mce']
            std_mce = row['std_mce']
            
            # Format mCE values
            if pd.isna(mean_mce):
                mce_str = "N/A"
                std_str = "N/A"
            else:
                mce_str = f"{mean_mce:.4f}"
                std_str = f"{std_mce:.4f}"
            
            # Format corruption ECEs
            corruption_strs = []
            for corruption in CIFAR_C_CORRUPTIONS:
                val = row[corruption]
                if pd.isna(val):
                    corruption_strs.append("  N/A  ")
                else:
                    corruption_strs.append(f"{val:.4f}")
            
            corruption_line = " | ".join(corruption_strs)
            
            print(f"{method:<40} {n:<5} {mce_str:<12} {std_str:<12} | {corruption_line}")
        
        print()
    
    print("="*120)
    print(f"\nTotal methods: {len(df)}")
    print(f"Total groups: {len(df['group'].unique())}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate corruption ablation results from multiple JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default='corruption_ablation_results',
        help='Directory containing corruption_ablation_*.json files'
    )
    
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Optional path to save results as CSV'
    )
    
    parser.add_argument(
        '--group_by',
        type=str,
        choices=['dataset', 'architecture', 'both'],
        default='dataset',
        help='How to group results: by dataset, architecture, or both'
    )
    
    args = parser.parse_args()
    
    # Load and aggregate results
    print(f"Loading results from: {args.input_dir}")
    print(f"Grouping by: {args.group_by}")
    
    all_results = collect_all_results(args.input_dir, group_by=args.group_by)
    
    if not all_results:
        print("No results found!")
        return
    
    # Create summary table
    df = create_summary_table(all_results)
    
    # Print formatted table
    print_formatted_table(df, args.group_by)
    
    # Save to CSV if requested
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)
    print(f"Total groups: {len(df['group'].unique())}")
    print(f"Total methods: {len(df['method'].unique())}")
    print(f"Total method-group combinations: {len(df)}")
    
    # Count valid results
    valid_mce = df['mean_mce'].notna().sum()
    print(f"Methods with valid mCE: {valid_mce} / {len(df)}")
    
    # Show best methods per group
    print("\nBest method (lowest mCE) per group:")
    for group_key in sorted(df['group'].unique()):
        group_df = df[df['group'] == group_key]
        group_df_valid = group_df[group_df['mean_mce'].notna()]
        if len(group_df_valid) > 0:
            best = group_df_valid.loc[group_df_valid['mean_mce'].idxmin()]
            print(f"  {group_key}: {best['method']} (mCE={best['mean_mce']:.4f} ± {best['std_mce']:.4f}, N={best['n_samples']})")


if __name__ == '__main__':
    main()

