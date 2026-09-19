#!/usr/bin/env python3
"""
Analyze layer count ablation: How does performance change with 2, 4, 6, 8 layers?
SEPARATELY for global_random (all layers) vs block_random (structured)

Research questions from advisor meeting:
1. What's the optimal number of layers?
2. Does global random (all layers) vs block random (structured) matter?
3. How does compression ratio affect optimal layer count?
4. Is there a trade-off between layers and compression?
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from utils.logging_config import get_logger
logger = get_logger(__name__)


def load_ablation_result(json_path: Path) -> Dict[str, Any]:
    """Load a single ablation result JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_layer_count_metrics(result: Dict[str, Any]) -> pd.DataFrame:
    """Extract metrics grouped by layer count - global_random, block_random, and geo_comb (fixed)."""
    
    config = result['experiment_info']
    rows = []
    
    # Extract from random_layer_ablation (global_random and block_random)
    if 'random_layer_ablation' in result:
        ablation = result['random_layer_ablation']
        
        # Extract from BOTH strategies
        for strategy in ['global_random', 'block_random']:
            if ablation.get(strategy):
                for trial in ablation[strategy]:
                    rows.append({
                        'model': config['model_name'],
                        'dataset': config['dataset'],
                        'training_method': config['training_method'],
                        'seed': config['seed'],
                        'strategy': strategy,
                        'num_layers': trial['num_layers'],
                        'compression_ratio': trial.get('post_concatenation_compression_ratio', trial.get('compression_ratio')),
                        'ece': trial['ece'] * 100,
                        'accuracy': trial['accuracy'],
                        'trial_idx': trial.get('trial_idx', 0),
                    })
    
    # Extract from geo_comb (fixed choice)
    if 'geo_comb' in result:
        for trial in result['geo_comb']['results']:
            rows.append({
                'model': config['model_name'],
                'dataset': config['dataset'],
                'training_method': config['training_method'],
                'seed': config['seed'],
                'strategy': 'geo_comb_fixed',
                'num_layers': trial.get('num_layers', len(trial.get('layer_names', []))),
                'compression_ratio': trial.get('post_concatenation_compression_ratio', trial.get('compression_ratio')),
                'ece': trial['ece'] * 100,
                'accuracy': trial['accuracy'],
                'trial_idx': 0,  # Fixed choice, no trials
            })
    
    return pd.DataFrame(rows)


def load_baseline_geo_comb(result: Dict[str, Any]) -> pd.DataFrame:
    """Load geo_comb (fixed layers) as baseline."""
    
    config = result['experiment_info']
    rows = []
    
    if 'geo_comb' not in result:
        return pd.DataFrame()
    
    for trial in result['geo_comb']['results']:
        rows.append({
            'model': config['model_name'],
            'dataset': config['dataset'],
            'training_method': config['training_method'],
            'seed': config['seed'],
            'compression_ratio': trial.get('post_concatenation_compression_ratio', trial.get('compression_ratio')),
            'ece': trial['ece'] * 100,
            'accuracy': trial['accuracy'],
            'num_layers': trial['num_layers'],
        })
    
    return pd.DataFrame(rows)


def analyze_single_strategy(strategy_data: pd.DataFrame, baseline_df: pd.DataFrame, 
                            strategy_name: str) -> tuple:
    """Analyze layer count effect for a single strategy (global or block random)."""
    
    groupby_cols = ['training_method', 'dataset', 'model', 'compression_ratio', 'num_layers']
    
    # Aggregate across seeds and trials
    agg_stats = strategy_data.groupby(groupby_cols)['ece'].agg(['mean', 'std', 'count']).reset_index()
    
    # Get unique configs
    configs = strategy_data[['training_method', 'dataset', 'model', 'compression_ratio']].drop_duplicates()
    
    comparison_rows = []
    
    for _, config in configs.iterrows():
        mask = (
            (agg_stats['training_method'] == config['training_method']) &
            (agg_stats['dataset'] == config['dataset']) &
            (agg_stats['model'] == config['model']) &
            (agg_stats['compression_ratio'] == config['compression_ratio'])
        )
        
        config_data = agg_stats[mask].sort_values('num_layers')
        
        if config_data.empty:
            continue
        
        row = {
            'training_method': config['training_method'],
            'dataset': config['dataset'],
            'model': config['model'],
            'compression_ratio': config['compression_ratio'],
            'strategy': strategy_name,
        }
        
        # Extract ECE for each layer count
        for _, layer_row in config_data.iterrows():
            num_layers = int(layer_row['num_layers'])
            row[f'layers_{num_layers}_ece'] = layer_row['mean']
            row[f'layers_{num_layers}_std'] = layer_row['std'] if pd.notna(layer_row['std']) else 0.0
            # Coefficient of variation (stability indicator)
            if layer_row['mean'] and not np.isclose(layer_row['mean'], 0):
                row[f'layers_{num_layers}_cv'] = (layer_row['std'] / layer_row['mean']) * 100
            else:
                row[f'layers_{num_layers}_cv'] = np.nan
            row[f'layers_{num_layers}_count'] = int(layer_row['count'])
        
        # Find best layer count
        layer_counts = [col.split('_')[1] for col in row.keys() if col.startswith('layers_') and col.endswith('_ece')]
        if layer_counts:
            ece_values = {int(lc): row[f'layers_{lc}_ece'] for lc in layer_counts if pd.notna(row.get(f'layers_{lc}_ece'))}
            if ece_values:
                best_layer_count = min(ece_values, key=ece_values.get)
                worst_layer_count = max(ece_values, key=ece_values.get)
                
                row['best_layer_count'] = best_layer_count
                row['best_ece'] = ece_values[best_layer_count]
                row['worst_layer_count'] = worst_layer_count
                row['worst_ece'] = ece_values[worst_layer_count]
                row['ece_range'] = ece_values[worst_layer_count] - ece_values[best_layer_count]
                # Average CV across available layers for quick stability summary
                cv_values = [row.get(f'layers_{lc}_cv') for lc in ece_values.keys() if pd.notna(row.get(f'layers_{lc}_cv'))]
                row['avg_cv'] = np.mean(cv_values) if cv_values else np.nan
        
        # Compare to baseline
        if not baseline_df.empty:
            baseline_match = baseline_df[
                (baseline_df['model'] == config['model']) &
                (baseline_df['dataset'] == config['dataset']) &
                (baseline_df['training_method'] == config['training_method']) &
                (baseline_df['compression_ratio'] == config['compression_ratio'])
            ]
            
            if not baseline_match.empty:
                row['baseline_fixed_ece'] = baseline_match['ece'].mean()
                row['baseline_fixed_std'] = baseline_match['ece'].std() if len(baseline_match) > 1 else 0.0
        
        comparison_rows.append(row)
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Lightweight per-CR summary (kept for downstream comparisons; printing handled elsewhere)
    cr_summary_rows = []
    for cr in sorted(comparison_df['compression_ratio'].unique()):
        cr_data = comparison_df[comparison_df['compression_ratio'] == cr]
        cr_summary_rows.append({
            'compression_ratio': cr,
            'n_experiments': len(cr_data),
            'avg_range': cr_data['ece_range'].mean() if 'ece_range' in cr_data.columns else np.nan,
            'common_best': cr_data['best_layer_count'].mode()[0] if 'best_layer_count' in cr_data.columns and not cr_data['best_layer_count'].mode().empty else np.nan,
            'avg_cv': cr_data['avg_cv'].mean() if 'avg_cv' in cr_data.columns else np.nan,
        })
    cr_summary_df = pd.DataFrame(cr_summary_rows)
    
    return comparison_df, cr_summary_df


def print_layer_count_table(comparison_df: pd.DataFrame, strategy_name: str):
    """Print detailed layer count comparison table."""
    
    print(f"\n{strategy_name.upper()} - Detailed Results:")
    print(f"{'Config':<45} {'CR':<6} {'L=2':<12} {'L=4':<12} {'L=6':<12} {'L=8':<12} {'Best':<8} {'Range':<8}")
    print("-"*140)
    
    for _, row in comparison_df.iterrows():
        config_str = f"{row['training_method']}_{row['dataset']}_{row['model']}"[:43]
        cr = f"{row['compression_ratio']:.0f}x"
        
        l2 = f"{row.get('layers_2_ece', np.nan):.2f}" if pd.notna(row.get('layers_2_ece')) else "---"
        l4 = f"{row.get('layers_4_ece', np.nan):.2f}" if pd.notna(row.get('layers_4_ece')) else "---"
        l6 = f"{row.get('layers_6_ece', np.nan):.2f}" if pd.notna(row.get('layers_6_ece')) else "---"
        l8 = f"{row.get('layers_8_ece', np.nan):.2f}" if pd.notna(row.get('layers_8_ece')) else "---"
        
        best = f"L={int(row['best_layer_count'])}" if pd.notna(row.get('best_layer_count')) else "---"
        ece_range = f"{row.get('ece_range', np.nan):.2f}" if pd.notna(row.get('ece_range')) else "---"
        
        print(f"{config_str:<45} {cr:<6} {l2:<12} {l4:<12} {l6:<12} {l8:<12} {best:<8} {ece_range:<8}")


def print_per_cr_summary(comparison_df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """Print per-CR summary."""
    
    print(f"\n{strategy_name.upper()} - Per-CR Summary:")
    
    cr_summary_rows = []
    
    for cr in sorted(comparison_df['compression_ratio'].unique()):
        cr_data = comparison_df[comparison_df['compression_ratio'] == cr]
        
        if cr_data.empty:
            continue
        
        # Check if columns exist before accessing them (block_random may only have L=4)
        avg_l2 = cr_data['layers_2_ece'].mean() if 'layers_2_ece' in cr_data.columns else np.nan
        avg_l4 = cr_data['layers_4_ece'].mean() if 'layers_4_ece' in cr_data.columns else np.nan
        avg_l6 = cr_data['layers_6_ece'].mean() if 'layers_6_ece' in cr_data.columns else np.nan
        avg_l8 = cr_data['layers_8_ece'].mean() if 'layers_8_ece' in cr_data.columns else np.nan
        avg_range = cr_data['ece_range'].mean() if 'ece_range' in cr_data.columns else np.nan
        
        # Handle best_layer_count
        if 'best_layer_count' in cr_data.columns:
            layer_count_wins = cr_data['best_layer_count'].dropna().value_counts()
            if not layer_count_wins.empty:
                most_common_best = int(layer_count_wins.idxmax())
                win_rate = layer_count_wins.max() / len(cr_data) * 100
            else:
                most_common_best = np.nan
                win_rate = np.nan
        else:
            most_common_best = np.nan
            win_rate = np.nan
        
        cr_summary_rows.append({
            'compression_ratio': cr,
            'n_experiments': len(cr_data),
            'avg_l2_ece': avg_l2,
            'avg_l4_ece': avg_l4,
            'avg_l6_ece': avg_l6,
            'avg_l8_ece': avg_l8,
            'avg_range': avg_range,
            'most_common_best': most_common_best,
            'win_rate': win_rate,
        })
    
    cr_summary_df = pd.DataFrame(cr_summary_rows)
    
    print(f"{'CR':<8} {'N':<6} {'Avg L=2':<12} {'Avg L=4':<12} {'Avg L=6':<12} {'Avg L=8':<12} {'Range':<10} {'Best':<10} {'Win %':<10}")
    print("-"*120)
    
    for _, row in cr_summary_df.iterrows():
        cr_str = f"{row['compression_ratio']:.0f}x"
        n = int(row['n_experiments'])
        
        l2 = f"{row['avg_l2_ece']:.2f}" if pd.notna(row['avg_l2_ece']) else "---"
        l4 = f"{row['avg_l4_ece']:.2f}" if pd.notna(row['avg_l4_ece']) else "---"
        l6 = f"{row['avg_l6_ece']:.2f}" if pd.notna(row['avg_l6_ece']) else "---"
        l8 = f"{row['avg_l8_ece']:.2f}" if pd.notna(row['avg_l8_ece']) else "---"
        
        range_str = f"{row['avg_range']:.2f}" if pd.notna(row['avg_range']) else "---"
        best = f"L={int(row['most_common_best'])}" if pd.notna(row['most_common_best']) else "---"
        win_pct = f"{row['win_rate']:.1f}%" if pd.notna(row['win_rate']) else "---"
        
        print(f"{cr_str:<8} {n:<6} {l2:<12} {l4:<12} {l6:<12} {l8:<12} {range_str:<10} {best:<10} {win_pct:<10}")
    
    return cr_summary_df


def print_strategy_insights(comparison_df: pd.DataFrame, strategy_name: str):
    """Print insights for specific strategy."""
    
    print(f"\n{strategy_name.upper()} - Key Insights:")
    print("-" * 80)
    
    if 'best_layer_count' in comparison_df.columns:
        layer_dist = comparison_df['best_layer_count'].value_counts()
        
        print(f"\nOptimal layer count distribution:")
        for lc, count in layer_dist.sort_index().items():
            pct = count / len(comparison_df) * 100
            print(f"  L={int(lc)}: {count}/{len(comparison_df)} ({pct:.1f}%)")
        
        most_common = int(layer_dist.idxmax())
        print(f"  → Most common: L={most_common}")
    
    if 'ece_range' in comparison_df.columns:
        avg_range = comparison_df['ece_range'].mean()
        print(f"\nSensitivity: Avg ECE range = {avg_range:.2f}%")
        
        if avg_range < 0.5:
            print(f"  → LOW sensitivity (robust)")
        elif avg_range < 1.0:
            print(f"  → MODERATE sensitivity")
        else:
            print(f"  → HIGH sensitivity (layer count matters)")


def compare_strategies(global_df: pd.DataFrame, block_df: pd.DataFrame, fixed_df: pd.DataFrame = None):
    """Compare global_random vs block_random AT L=4 ONLY (where both have data), optionally including fixed."""
    
    print("\n" + "="*140)
    print("GLOBAL vs BLOCK RANDOM COMPARISON (L=4 HEAD-TO-HEAD)")
    if fixed_df is not None:
        print("(with Fixed geo_comb for reference)")
    print("="*140)
    print("NOTE: Comparing ONLY at 4 layers where both strategies have data")
    print("="*140)
    
    # Determine fixed layer count if fixed_df is provided
    fixed_layer_count = None
    if fixed_df is not None:
        for col in fixed_df.columns:
            if col.startswith('layers_') and col.endswith('_ece'):
                fixed_layer_count = int(col.split('_')[1])
                break
    
    if fixed_df is not None and fixed_layer_count:
        print(f"\n{'Config':<45} {'CR':<6} {'Global L=4':<12} {'Block L=4':<12} {'Fixed L=' + str(fixed_layer_count):<15} {'Diff':<10} {'Winner':<10}")
    else:
        print(f"\n{'Config':<45} {'CR':<6} {'Global L=4':<12} {'Block L=4':<12} {'Diff':<10} {'Winner':<10}")
    print("-"*140)
    
    merge_cols = ['training_method', 'dataset', 'model', 'compression_ratio']
    comparison_rows = []
    
    for _, config in global_df[merge_cols].drop_duplicates().iterrows():
        global_row = global_df[
            (global_df['training_method'] == config['training_method']) &
            (global_df['dataset'] == config['dataset']) &
            (global_df['model'] == config['model']) &
            (global_df['compression_ratio'] == config['compression_ratio'])
        ]
        
        block_row = block_df[
            (block_df['training_method'] == config['training_method']) &
            (block_df['dataset'] == config['dataset']) &
            (block_df['model'] == config['model']) &
            (block_df['compression_ratio'] == config['compression_ratio'])
        ]
        
        if global_row.empty or block_row.empty:
            continue
        
        # CRITICAL: Only compare at L=4 where both have data
        global_l4_ece = global_row.iloc[0].get('layers_4_ece')
        block_l4_ece = block_row.iloc[0].get('layers_4_ece')
        
        # Skip if either doesn't have L=4 data
        if pd.isna(global_l4_ece) or pd.isna(block_l4_ece):
            continue
        
        # Get fixed ECE if available
        fixed_ece = None
        if fixed_df is not None and fixed_layer_count:
            fixed_row = fixed_df[
                (fixed_df['training_method'] == config['training_method']) &
                (fixed_df['dataset'] == config['dataset']) &
                (fixed_df['model'] == config['model']) &
                (fixed_df['compression_ratio'] == config['compression_ratio'])
            ]
            if not fixed_row.empty:
                fixed_layer_col = f'layers_{fixed_layer_count}_ece'
                fixed_ece = fixed_row.iloc[0].get(fixed_layer_col)
        
        config_str = f"{config['training_method']}_{config['dataset']}_{config['model']}"[:43]
        cr = f"{config['compression_ratio']:.0f}x"
        
        diff = global_l4_ece - block_l4_ece
        winner = "Block" if diff > 0 else "Global" if diff < 0 else "Tie"
        
        if fixed_df is not None and fixed_layer_count and fixed_ece is not None:
            fixed_str = f"{fixed_ece:.2f}"
            print(f"{config_str:<45} {cr:<6} {global_l4_ece:<12.2f} {block_l4_ece:<12.2f} {fixed_str:<15} {diff:<10.2f} {winner:<10}")
        else:
            print(f"{config_str:<45} {cr:<6} {global_l4_ece:<12.2f} {block_l4_ece:<12.2f} {diff:<10.2f} {winner:<10}")
        
        row_data = {
            'training_method': config['training_method'],
            'dataset': config['dataset'],
            'model': config['model'],
            'compression_ratio': config['compression_ratio'],
            'global_l4_ece': global_l4_ece,
            'block_l4_ece': block_l4_ece,
            'diff': diff,
            'winner': winner,
        }
        
        if fixed_ece is not None:
            row_data['fixed_ece'] = fixed_ece
        
        comparison_rows.append(row_data)
    
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        
        global_wins = (comp_df['winner'] == 'Global').sum()
        block_wins = (comp_df['winner'] == 'Block').sum()
        ties = (comp_df['winner'] == 'Tie').sum()
        avg_diff = comp_df['diff'].mean()
        
        print(f"\nHEAD-TO-HEAD SUMMARY (L=4 ONLY):")
        print(f"  Total comparisons: {len(comp_df)}")
        print(f"  Global Random wins: {global_wins}/{len(comp_df)} ({global_wins/len(comp_df)*100:.1f}%)")
        print(f"  Block Random wins: {block_wins}/{len(comp_df)} ({block_wins/len(comp_df)*100:.1f}%)")
        print(f"  Ties: {ties}")
        print(f"  Average difference: {avg_diff:+.2f}% (positive = block better)")
        
        if abs(avg_diff) < 0.1:
            print(f"\n  → STRATEGIES ARE EQUIVALENT at L=4 (|diff| < 0.1%)")
        elif abs(avg_diff) < 0.3:
            print(f"\n  → STRATEGIES ARE VERY SIMILAR at L=4 (|diff| < 0.3%)")
        elif avg_diff > 0:
            print(f"\n  → BLOCK RANDOM IS BETTER at L=4 by {avg_diff:.2f}%")
        else:
            print(f"\n  → GLOBAL RANDOM IS BETTER at L=4 by {abs(avg_diff):.2f}%")
        
        # Additional insight: Check if difference is consistent across CRs
        per_cr_avg = comp_df.groupby('compression_ratio')['diff'].mean()
        print(f"\n  Per-CR breakdown:")
        for cr, avg in per_cr_avg.items():
            cr_data = comp_df[comp_df['compression_ratio'] == cr]
            n = len(cr_data)
            g_wins = (cr_data['winner'] == 'Global').sum()
            b_wins = (cr_data['winner'] == 'Block').sum()
            print(f"    CR={cr:.0f}x: avg_diff={avg:+.2f}%, Global wins {g_wins}/{n}, Block wins {b_wins}/{n}")
    
    return comparison_rows


def compare_fixed_vs_random(fixed_df: pd.DataFrame, random_df: pd.DataFrame, random_strategy_name: str):
    """Compare fixed geo_comb (5 layers) vs random strategy - compare with best random result."""
    
    print("\n" + "="*140)
    print(f"FIXED (geo_comb, 5 layers) vs {random_strategy_name.upper()} COMPARISON")
    print("="*140)
    print("Comparing fixed 5-layer selection with random strategy's BEST result")
    print("="*140)
    
    # Get the layer count from fixed by looking at the columns (e.g., layers_5_ece)
    fixed_layer_count = None
    for col in fixed_df.columns:
        if col.startswith('layers_') and col.endswith('_ece'):
            layer_num = int(col.split('_')[1])
            if fixed_layer_count is None:
                fixed_layer_count = layer_num
            # If there are multiple, use the first one found (should be only one for fixed)
            break
    
    if fixed_layer_count is None:
        print("  No fixed layer data available (no layers_X_ece columns found)")
        return
    
    print(f"\nFixed geo_comb uses {fixed_layer_count} layers")
    print(f"Comparing with {random_strategy_name}'s BEST result across all layer counts\n")
    
    print(f"{'Config':<45} {'CR':<6} {'Fixed (L={fixed_layer_count})':<18} {'Random Best':<15} {'Random L':<10} {'Diff':<10} {'Winner':<10}")
    print("-"*140)
    
    merge_cols = ['training_method', 'dataset', 'model', 'compression_ratio']
    comparison_rows = []
    
    for _, config in fixed_df[merge_cols].drop_duplicates().iterrows():
        fixed_row = fixed_df[
            (fixed_df['training_method'] == config['training_method']) &
            (fixed_df['dataset'] == config['dataset']) &
            (fixed_df['model'] == config['model']) &
            (fixed_df['compression_ratio'] == config['compression_ratio'])
        ]
        
        if fixed_row.empty:
            continue
        
        # For fixed, get ECE from layer-specific column (e.g., layers_5_ece)
        fixed_ece = None
        fixed_layer_col = f'layers_{fixed_layer_count}_ece'
        if fixed_layer_col in fixed_row.columns:
            fixed_ece = fixed_row.iloc[0].get(fixed_layer_col)
        
        # Fallback: try best_ece or any layer column
        if fixed_ece is None or pd.isna(fixed_ece):
            if 'best_ece' in fixed_row.columns and pd.notna(fixed_row.iloc[0].get('best_ece')):
                fixed_ece = fixed_row.iloc[0]['best_ece']
            else:
                # Try any layer column
                for col in fixed_row.columns:
                    if col.startswith('layers_') and col.endswith('_ece'):
                        val = fixed_row.iloc[0].get(col)
                        if pd.notna(val):
                            fixed_ece = val
                            break
        
        if fixed_ece is None or pd.isna(fixed_ece):
            continue
        
        # Get random's best result
        random_row = random_df[
            (random_df['training_method'] == config['training_method']) &
            (random_df['dataset'] == config['dataset']) &
            (random_df['model'] == config['model']) &
            (random_df['compression_ratio'] == config['compression_ratio'])
        ]
        
        if random_row.empty:
            continue
        
        # Get random's best ECE and which layer count achieved it
        random_best_ece = None
        random_best_layers = None
        
        if 'best_ece' in random_row.columns and pd.notna(random_row.iloc[0].get('best_ece')):
            random_best_ece = random_row.iloc[0]['best_ece']
            if 'best_layer_count' in random_row.columns:
                random_best_layers = int(random_row.iloc[0]['best_layer_count'])
        
        # If best_ece not available, find best from layer columns
        if random_best_ece is None or pd.isna(random_best_ece):
            best_val = None
            best_layers = None
            for col in random_row.columns:
                if col.startswith('layers_') and col.endswith('_ece'):
                    val = random_row.iloc[0].get(col)
                    if pd.notna(val):
                        layers = int(col.split('_')[1])
                        if best_val is None or val < best_val:
                            best_val = val
                            best_layers = layers
            if best_val is not None:
                random_best_ece = best_val
                random_best_layers = best_layers
        
        if random_best_ece is None or pd.isna(random_best_ece):
            continue
        
        config_str = f"{config['training_method']}_{config['dataset']}_{config['model']}"[:43]
        cr = f"{config['compression_ratio']:.0f}x"
        diff = fixed_ece - random_best_ece
        winner = random_strategy_name if diff > 0 else "Fixed" if diff < 0 else "Tie"
        random_layers_str = f"L={random_best_layers}" if random_best_layers else "---"
        
        print(f"{config_str:<45} {cr:<6} {fixed_ece:<18.2f} {random_best_ece:<15.2f} {random_layers_str:<10} {diff:<10.2f} {winner:<10}")
        
        comparison_rows.append({
            'training_method': config['training_method'],
            'dataset': config['dataset'],
            'model': config['model'],
            'compression_ratio': config['compression_ratio'],
            'fixed_ece': fixed_ece,
            'fixed_layers': fixed_layer_count,
            'random_ece': random_best_ece,
            'random_layers': random_best_layers,
            'diff': diff,
            'winner': winner,
        })
    
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        
        fixed_wins = (comp_df['winner'] == 'Fixed').sum()
        random_wins = (comp_df['winner'] == random_strategy_name).sum()
        ties = (comp_df['winner'] == 'Tie').sum()
        avg_diff = comp_df['diff'].mean()
        
        print(f"\nSUMMARY:")
        print(f"  Total comparisons: {len(comp_df)}")
        print(f"  Fixed (geo_comb, {fixed_layer_count} layers) wins: {fixed_wins}/{len(comp_df)} ({fixed_wins/len(comp_df)*100:.1f}%)")
        print(f"  {random_strategy_name} (best across all layer counts) wins: {random_wins}/{len(comp_df)} ({random_wins/len(comp_df)*100:.1f}%)")
        print(f"  Ties: {ties}")
        print(f"  Average difference: {avg_diff:+.2f}% (positive = random better)")
        
        # Show which layer counts were best for random
        if 'random_layers' in comp_df.columns:
            layer_dist = comp_df['random_layers'].value_counts()
            if not layer_dist.empty:
                print(f"\n  Random strategy's best layer counts:")
                for layers, count in layer_dist.sort_index().items():
                    pct = count / len(comp_df) * 100
                    print(f"    L={int(layers)}: {count}/{len(comp_df)} ({pct:.1f}%)")
        
        if abs(avg_diff) < 0.1:
            print(f"\n  → STRATEGIES ARE EQUIVALENT (|diff| < 0.1%)")
        elif abs(avg_diff) < 0.3:
            print(f"\n  → STRATEGIES ARE VERY SIMILAR (|diff| < 0.3%)")
        elif avg_diff > 0:
            print(f"\n  → {random_strategy_name.upper()} IS BETTER by {avg_diff:.2f}%")
        else:
            print(f"\n  → FIXED (geo_comb, {fixed_layer_count} layers) IS BETTER by {abs(avg_diff):.2f}%")
        
        # Also show specific comparison: Fixed 5 vs Random 4 (if L=4 exists)
        if fixed_layer_count == 5 and 'layers_4_ece' in random_df.columns:
            print(f"\n" + "="*140)
            print(f"SPECIFIC COMPARISON: Fixed L={fixed_layer_count} vs {random_strategy_name.upper()} L=4")
            print("="*140)
            print(f"{'Config':<45} {'CR':<6} {'Fixed L={fixed_layer_count}':<18} {'Random L=4':<15} {'Diff':<10} {'Winner':<10}")
            print("-"*140)
            
            l4_comparisons = []
            for _, config in fixed_df[merge_cols].drop_duplicates().iterrows():
                fixed_row = fixed_df[
                    (fixed_df['training_method'] == config['training_method']) &
                    (fixed_df['dataset'] == config['dataset']) &
                    (fixed_df['model'] == config['model']) &
                    (fixed_df['compression_ratio'] == config['compression_ratio'])
                ]
                
                random_row = random_df[
                    (random_df['training_method'] == config['training_method']) &
                    (random_df['dataset'] == config['dataset']) &
                    (random_df['model'] == config['model']) &
                    (random_df['compression_ratio'] == config['compression_ratio'])
                ]
                
                if fixed_row.empty or random_row.empty:
                    continue
                
                fixed_ece_l5 = fixed_row.iloc[0].get(f'layers_{fixed_layer_count}_ece')
                random_ece_l4 = random_row.iloc[0].get('layers_4_ece')
                
                if pd.isna(fixed_ece_l5) or pd.isna(random_ece_l4):
                    continue
                
                diff_l4 = fixed_ece_l5 - random_ece_l4
                winner_l4 = random_strategy_name if diff_l4 > 0 else "Fixed" if diff_l4 < 0 else "Tie"
                
                config_str = f"{config['training_method']}_{config['dataset']}_{config['model']}"[:43]
                cr = f"{config['compression_ratio']:.0f}x"
                print(f"{config_str:<45} {cr:<6} {fixed_ece_l5:<18.2f} {random_ece_l4:<15.2f} {diff_l4:<10.2f} {winner_l4:<10}")
                
                l4_comparisons.append({
                    'fixed_ece': fixed_ece_l5,
                    'random_ece': random_ece_l4,
                    'diff': diff_l4,
                    'winner': winner_l4,
                })
            
            if l4_comparisons:
                l4_df = pd.DataFrame(l4_comparisons)
                fixed_wins_l4 = (l4_df['winner'] == 'Fixed').sum()
                random_wins_l4 = (l4_df['winner'] == random_strategy_name).sum()
                avg_diff_l4 = l4_df['diff'].mean()
                
                print(f"\n  Fixed L={fixed_layer_count} vs Random L=4:")
                print(f"    Fixed wins: {fixed_wins_l4}/{len(l4_df)} ({fixed_wins_l4/len(l4_df)*100:.1f}%)")
                print(f"    Random L=4 wins: {random_wins_l4}/{len(l4_df)} ({random_wins_l4/len(l4_df)*100:.1f}%)")
                print(f"    Average difference: {avg_diff_l4:+.2f}% (positive = random L=4 better)")
    
    return comparison_rows


# ---------------------------------------------------------------------------
# New focused summary helpers
# ---------------------------------------------------------------------------

def compute_experiment_overview(all_data: pd.DataFrame) -> Dict[str, Any]:
    """Compute experiment overview statistics."""
    overview: Dict[str, Any] = {}
    
    if all_data.empty:
        return overview
    
    # Count unique configurations (training_method, dataset, model)
    config_cols = ['training_method', 'dataset', 'model']
    if all(col in all_data.columns for col in config_cols):
        unique_configs = all_data[config_cols].drop_duplicates()
        overview["total_configurations"] = len(unique_configs)
    else:
        overview["total_configurations"] = 0
    
    # Count total experiments (including seeds and trials)
    overview["total_experiments"] = len(all_data)
    
    # Extract unique values
    if 'training_method' in all_data.columns:
        overview["training_methods"] = sorted(all_data["training_method"].unique().tolist())
    else:
        overview["training_methods"] = []
    
    if 'dataset' in all_data.columns:
        overview["datasets"] = sorted(all_data["dataset"].unique().tolist())
    else:
        overview["datasets"] = []
    
    if 'model' in all_data.columns:
        overview["models"] = sorted(all_data["model"].unique().tolist())
    else:
        overview["models"] = []
    
    # Extract compression ratios
    if 'compression_ratio' in all_data.columns:
        compression_ratios = sorted(all_data["compression_ratio"].dropna().unique().tolist())
        overview["compression_ratios"] = [f"{int(cr)}x" if cr == int(cr) else f"{cr:.1f}x" for cr in compression_ratios]
    else:
        overview["compression_ratios"] = []
    
    # Extract layer counts
    if 'num_layers' in all_data.columns:
        layer_counts = sorted(all_data["num_layers"].dropna().unique().tolist())
        overview["layer_counts"] = [int(lc) for lc in layer_counts]
    else:
        overview["layer_counts"] = []
    
    return overview


def print_experiment_overview(overview: Dict[str, Any]):
    """Print experiment overview statistics."""
    if not overview:
        logger.warning("No overview data to display.")
        return
    
    print("\nEXPERIMENT OVERVIEW")
    print("-" * 80)
    print(f"Total configurations: {overview.get('total_configurations', 0)}")
    print(f"Total experiments (including seeds): {overview.get('total_experiments', 0)}")
    
    training_methods = overview.get("training_methods", [])
    if training_methods:
        methods_str = ", ".join(training_methods)
        print(f"Training methods: {methods_str} ({len(training_methods)} total)")
    
    datasets = overview.get("datasets", [])
    if datasets:
        datasets_str = ", ".join(datasets)
        print(f"Datasets: {datasets_str} ({len(datasets)} total)")
    
    models = overview.get("models", [])
    if models:
        models_str = ", ".join(models)
        print(f"Models: {models_str} ({len(models)} total)")
    
    compression_ratios = overview.get("compression_ratios", [])
    if compression_ratios:
        ratios_str = ", ".join(compression_ratios)
        print(f"Compression ratios: {ratios_str} ({len(compression_ratios)} total)")
    
    layer_counts = overview.get("layer_counts", [])
    if layer_counts:
        counts_str = ", ".join(str(lc) for lc in layer_counts)
        print(f"Layer counts: {counts_str} ({len(layer_counts)} total)")


def print_header_block(title: str):
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")


def head_to_head_all_layers(global_df: pd.DataFrame, block_df: pd.DataFrame):
    """Head-to-head summary across all layer counts (2,4,6,8)."""
    print_header_block("QUESTION 1: Global vs Block Random - Who Wins?")
    if global_df.empty or block_df.empty:
        print("No global/block data for comparison.")
        return {}
    
    layer_counts = [2, 4, 6, 8]
    total = 0
    skipped = 0
    global_wins = 0
    block_wins = 0
    diffs = []
    
    for lc in layer_counts:
        col = f'layers_{lc}_ece'
        for _, cfg in global_df[['training_method', 'dataset', 'model', 'compression_ratio']].drop_duplicates().iterrows():
            g_row = global_df[
                (global_df['training_method'] == cfg['training_method']) &
                (global_df['dataset'] == cfg['dataset']) &
                (global_df['model'] == cfg['model']) &
                (global_df['compression_ratio'] == cfg['compression_ratio'])
            ]
            b_row = block_df[
                (block_df['training_method'] == cfg['training_method']) &
                (block_df['dataset'] == cfg['dataset']) &
                (block_df['model'] == cfg['model']) &
                (block_df['compression_ratio'] == cfg['compression_ratio'])
            ]
            if g_row.empty or b_row.empty:
                skipped += 1
                continue
            g_val = g_row.iloc[0].get(col)
            b_val = b_row.iloc[0].get(col)
            if pd.isna(g_val) or pd.isna(b_val):
                skipped += 1
                continue
            diff = g_val - b_val  # positive means block better
            diffs.append(diff)
            total += 1
            if diff < 0:
                global_wins += 1
            elif diff > 0:
                block_wins += 1
    
    if total == 0:
        print("No valid comparisons (all missing/NaN).")
        return {}
    
    avg_diff = np.mean(diffs) if diffs else 0.0
    print(f"Compared {total} configs (skipped {skipped} due to incomplete data).")
    print(f"Global wins: {global_wins} ({global_wins/total*100:.1f}%) | Block wins: {block_wins} ({block_wins/total*100:.1f}%)")
    print(f"Average difference: {avg_diff:+.2f}% (positive = Block better)")
    if abs(avg_diff) < 0.05:
        verdict = "No difference - Structure doesn't matter"
    elif abs(avg_diff) < 0.10:
        verdict = "Minimal difference - Structure barely matters"
    else:
        verdict = "Structure matters (Block better)" if avg_diff > 0 else "Structure matters (Global better)"
    print(f"Verdict: {verdict}")
    return {
        'total': total,
        'skipped': skipped,
        'global_wins': global_wins,
        'block_wins': block_wins,
        'avg_diff': avg_diff,
        'verdict': verdict,
    }


# ---------------------------------------------------------------------------
# Architecture-aware fixed layer counts and matched comparisons
# ---------------------------------------------------------------------------

def architecture_layer_counts(geo_comb_data: pd.DataFrame):
    """Print per-config layer counts for fixed geo_comb."""
    print_header_block("ARCHITECTURE LAYER COUNTS (Fixed geo_comb)")
    if geo_comb_data.empty or 'num_layers' not in geo_comb_data.columns:
        print("No geo_comb fixed data available.")
        return []
    config_cols = ['training_method', 'dataset', 'model']
    rows = []
    for _, row in geo_comb_data[config_cols + ['num_layers']].drop_duplicates().iterrows():
        cfg = f"{row['training_method']}_{row['dataset']}_{row['model']}"
        layers = int(row['num_layers']) if pd.notna(row['num_layers']) else None
        rows.append({'config': cfg, 'layers': layers})
        print(f"{cfg}: {layers} layers")
    return rows


def best_layer_count_by_strategy(strategy_df: pd.DataFrame, name: str):
    """Compute best layer counts: avg ECE per layer and win counts."""
    if strategy_df.empty:
        return {}
    layer_counts = [2, 4, 6, 8]
    layer_avgs = {}
    layer_wins = {lc: 0 for lc in layer_counts}
    valid_rows = 0
    for lc in layer_counts:
        col = f"layers_{lc}_ece"
        if col in strategy_df.columns:
            vals = strategy_df[col].dropna()
            layer_avgs[lc] = vals.mean() if not vals.empty else np.nan
        else:
            layer_avgs[lc] = np.nan
    for _, row in strategy_df.iterrows():
        vals = {lc: row.get(f"layers_{lc}_ece") for lc in layer_counts if pd.notna(row.get(f"layers_{lc}_ece"))}
        if not vals:
            continue
        best_lc = min(vals, key=vals.get)
        layer_wins[best_lc] += 1
        valid_rows += 1
    summary = {lc: {'avg_ece': layer_avgs[lc], 'wins': layer_wins[lc]} for lc in layer_counts}
    summary['valid_rows'] = valid_rows
    return summary


def compression_ratio_effect(strategy_df: pd.DataFrame, name: str):
    """Find best/worst CR per layer based on mean ECE."""
    if strategy_df.empty:
        return {}
    layer_counts = [2, 4, 6, 8, 5]  # include 5 for fixed if present
    result = {}
    for lc in layer_counts:
        col = f"layers_{lc}_ece"
        if col not in strategy_df.columns:
            continue
        by_cr = strategy_df[['compression_ratio', col]].dropna()
        if by_cr.empty:
            continue
        grouped = by_cr.groupby('compression_ratio')[col].mean().reset_index()
        best_row = grouped.loc[grouped[col].idxmin()]
        worst_row = grouped.loc[grouped[col].idxmax()]
        result[lc] = {
            'best_cr': best_row['compression_ratio'],
            'best_ece': best_row[col],
            'worst_cr': worst_row['compression_ratio'],
            'worst_ece': worst_row[col],
        }
    return result


def fixed_vs_random(fixed_df: pd.DataFrame, random_df: pd.DataFrame, random_name: str):
    """Compare fixed vs random using random's BEST layer per config."""
    if fixed_df.empty or random_df.empty:
        return {}
    config_cols = ['training_method', 'dataset', 'model', 'compression_ratio']
    rows = []
    skipped = 0
    for _, cfg in fixed_df[config_cols].drop_duplicates().iterrows():
        f_row = fixed_df[
            (fixed_df['training_method'] == cfg['training_method']) &
            (fixed_df['dataset'] == cfg['dataset']) &
            (fixed_df['model'] == cfg['model']) &
            (fixed_df['compression_ratio'] == cfg['compression_ratio'])
        ]
        r_row = random_df[
            (random_df['training_method'] == cfg['training_method']) &
            (random_df['dataset'] == cfg['dataset']) &
            (random_df['model'] == cfg['model']) &
            (random_df['compression_ratio'] == cfg['compression_ratio'])
        ]
        if f_row.empty or r_row.empty:
            skipped += 1
            continue
        # fixed best (min across available layer cols)
        f_vals = {col: f_row.iloc[0].get(col) for col in f_row.columns if col.startswith('layers_') and col.endswith('_ece')}
        f_vals = {k: v for k, v in f_vals.items() if pd.notna(v)}
        if not f_vals:
            skipped += 1
            continue
        fixed_ece = min(f_vals.values())
        # random best
        r_vals = {col: r_row.iloc[0].get(col) for col in r_row.columns if col.startswith('layers_') and col.endswith('_ece')}
        r_vals = {k: v for k, v in r_vals.items() if pd.notna(v)}
        if not r_vals:
            skipped += 1
            continue
        random_best = min(r_vals.values())
        diff = fixed_ece - random_best  # positive = fixed better
        rows.append(diff)
    if not rows:
        print(f"{random_name}: No valid fixed vs random comparisons (skipped {skipped}).")
        return {}
    total = len(rows)
    random_better = sum(1 for d in rows if d < 0)
    fixed_better = sum(1 for d in rows if d > 0)
    within_02 = sum(1 for d in rows if abs(d) <= 0.2)
    avg_diff = np.mean(rows)
    within_02_pct = within_02 / total if total else 0
    print_header_block(f"QUESTION 4: Fixed vs {random_name.replace('_',' ').title()}")
    print(f"Compared {total} configs (skipped {skipped}).")
    print(f"{random_name} better: {random_better}/{total} ({random_better/total*100:.1f}%)")
    print(f"Fixed better: {fixed_better}/{total} ({fixed_better/total*100:.1f}%)")
    print(f"Within ±0.2%: {within_02}/{total} ({within_02/total*100:.1f}%)")
    print(f"Average diff: {avg_diff:+.2f}% (positive = fixed better)")
    # New verdict logic
    if avg_diff < 0.1:
        verdict_text = "✓ Random works! Algorithm matters, not layer selection"
        explanation = f"Avg diff only {avg_diff:.3f}% - essentially equivalent"
    elif within_02_pct > 0.65:
        verdict_text = "✓ Random acceptable - Algorithm is key factor"
        explanation = f"{within_02_pct:.1%} configs within ±0.2%"
    elif within_02_pct > 0.50:
        verdict_text = "~ Random mostly works - Mixed results"
        explanation = f"{within_02_pct:.1%} configs within ±0.2%, avg diff {avg_diff:.3f}%"
    else:
        verdict_text = "❌ Fixed layer choice critical"
        explanation = f"Only {within_02_pct:.1%} within ±0.2%, avg diff {avg_diff:.3f}%"
    print(f"Verdict: {verdict_text}")
    print(f"Reason: {explanation}")
    return {
        'total': total,
        'skipped': skipped,
        'random_better': random_better,
        'fixed_better': fixed_better,
        'within_02': within_02,
        'avg_diff': avg_diff,
        'within_02_pct': within_02_pct,
        'verdict': verdict_text,
        'explanation': explanation,
    }


def export_latex_tables(results: Dict, output_dir: Path):
    """Export all LaTeX tables."""
    
    latex_dir = output_dir / "latex_tables"
    latex_dir.mkdir(parents=True, exist_ok=True)
    
    # Tables for each strategy
    for strategy_name in ['global_random', 'block_random', 'geo_comb_fixed']:
        if strategy_name not in results:
            continue
        
        comparison_df = results[strategy_name]['comparison']
        if strategy_name == 'global_random':
            strategy_label = "Global Random"
        elif strategy_name == 'block_random':
            strategy_label = "Block Random"
        else:
            strategy_label = "Fixed (geo_comb)"
        
        # Detailed table
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        
        # Determine layer counts to show
        if strategy_name == 'geo_comb_fixed':
            # For fixed, find what layer counts exist
            layer_cols = [col for col in comparison_df.columns if col.startswith('layers_') and col.endswith('_ece')]
            layer_counts = sorted([int(col.split('_')[1]) for col in layer_cols])
            layer_counts_str = ", ".join([f"L={lc}" for lc in layer_counts])
            lines.append(f"\\caption{{{strategy_label}: Fixed layer selection ({layer_counts_str}).}}")
        else:
            lines.append(f"\\caption{{{strategy_label}: Layer count ablation (2, 4, 6, 8 layers). Best is bolded.}}")
        
        lines.append(f"\\label{{tab:layer_{strategy_name}}}")
        lines.append("\\small")
        
        # Build table header based on layer counts
        if strategy_name == 'geo_comb_fixed':
            layer_counts_to_show = layer_counts
            col_spec = "llll" + "r" * len(layer_counts_to_show) + "rr"
            header = "Training & Dataset & Model & CR & " + " & ".join([f"L={lc}" for lc in layer_counts_to_show]) + " & ECE & Range \\\\"
        else:
            layer_counts_to_show = [2, 4, 6, 8]
            col_spec = "llllrrrrrr"
            header = "Training & Dataset & Model & CR & L=2 & L=4 & L=6 & L=8 & Best & Range \\\\"
        
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        lines.append(header)
        lines.append("\\midrule")
        
        for _, row in comparison_df.iterrows():
            cells = [
                str(row['training_method']),
                str(row['dataset']),
                str(row['model']),
                f"{row['compression_ratio']:.0f}x",
            ]
            
            best_lc = int(row['best_layer_count']) if pd.notna(row.get('best_layer_count')) else None
            
            for lc in layer_counts_to_show:
                ece_val = row.get(f'layers_{lc}_ece')
                if pd.notna(ece_val):
                    ece_str = f"{ece_val:.2f}"
                    if lc == best_lc:
                        ece_str = f"\\textbf{{{ece_str}}}"
                    cells.append(ece_str)
                else:
                    cells.append("---")
            
            if strategy_name == 'geo_comb_fixed':
                # For fixed, show the ECE value and range if available
                if pd.notna(row.get('best_ece')):
                    cells.append(f"{row['best_ece']:.2f}")
                else:
                    # Get first available layer ECE
                    for lc in layer_counts_to_show:
                        ece_val = row.get(f'layers_{lc}_ece')
                        if pd.notna(ece_val):
                            cells.append(f"{ece_val:.2f}")
                            break
                    else:
                        cells.append("---")
                
                if pd.notna(row.get('ece_range')):
                    cells.append(f"{row['ece_range']:.2f}")
                else:
                    cells.append("---")
            else:
                if pd.notna(row.get('best_layer_count')):
                    cells.append(f"L={int(row['best_layer_count'])}")
                    cells.append(f"{row.get('ece_range', 0):.2f}")
                else:
                    cells.append("---")
                    cells.append("---")
            
            lines.append(" & ".join(cells) + " \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        latex_file = latex_dir / f"layer_count_{strategy_name}.tex"
        with open(latex_file, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"  ✓ {latex_file.name}")


def export_summary_tables(global_df: pd.DataFrame, block_df: pd.DataFrame, fixed_df: pd.DataFrame,
                          head_stats: Dict, fixed_vs_global: Dict, fixed_vs_block: Dict,
                          layer_stats: Dict[str, Dict], cr_effects: Dict[str, Dict], output_dir: Path):
    """Export concise summary LaTeX tables for the paper."""
    latex_dir = output_dir / "latex_tables"
    latex_dir.mkdir(parents=True, exist_ok=True)
    
    # Table 1: Layer count comparison (avg ECE across CRs)
    table1_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Layer count ablation: Average ECE (\\%) across all compression ratios}",
        "\\label{tab:layer_count_comparison}",
        "\\small",
        "\\begin{tabular}{lccccc c}",
        "\\toprule",
        "Strategy & L=2 & L=4 & L=5 & L=6 & L=8 & Best \\\\",
        "\\midrule",
    ]
    def avg_by_layer(df, layers):
        return {lc: df[f"layers_{lc}_ece"].mean() if f"layers_{lc}_ece" in df.columns and not df[f"layers_{lc}_ece"].dropna().empty else np.nan for lc in layers}
    layers_full = [2,4,5,6,8]
    for name, df in [('Global Random', global_df), ('Block Random', block_df), ('Fixed (geo\\_comb)', fixed_df)]:
        avgs = avg_by_layer(df, layers_full)
        best_lc = min([lc for lc,v in avgs.items() if not np.isnan(v)], key=lambda k: avgs[k]) if any(not np.isnan(v) for v in avgs.values()) else None
        cells = []
        for lc in layers_full:
            val = avgs[lc]
            if np.isnan(val):
                cells.append("---")
            else:
                sval = f"{val:.2f}"
                if lc == best_lc:
                    sval = f"\\\\textbf{{{sval}}}"
                cells.append(sval)
        best_str = f"L={best_lc}" if best_lc else "---"
        table1_lines.append(f"{name} & " + " & ".join(cells) + f" & {best_str} \\\\")
    table1_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    with open(latex_dir / "layer_count_summary.tex", "w") as f:
        f.write("\n".join(table1_lines))
    
    # Table 2: Compression ratio effects at optimal layer count per strategy
    cr_columns = [1,4,8,16,32,64]
    table2_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Compression ratio effects on ECE (\\%) for each strategy at optimal layer count}",
        "\\label{tab:compression_effects}",
        "\\small",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Strategy (Layers) & 1x & 4x & 8x & 16x & 32x & 64x & Best CR \\\\",
        "\\midrule",
    ]
    def optimal_layer_and_cr(df):
        layer_cols = [c for c in df.columns if c.startswith("layers_") and c.endswith("_ece")]
        if not layer_cols:
            return None, {}
        layer_nums = [int(c.split('_')[1]) for c in layer_cols]
        layer_means = {ln: df[f"layers_{ln}_ece"].mean() for ln in layer_nums if not df[f"layers_{ln}_ece"].dropna().empty}
        if not layer_means:
            return None, {}
        best_layer = min(layer_means, key=layer_means.get)
        cr_map = {}
        by_cr = df[['compression_ratio', f"layers_{best_layer}_ece"]].dropna()
        if not by_cr.empty:
            grouped = by_cr.groupby('compression_ratio')[f"layers_{best_layer}_ece"].mean()
            cr_map = grouped.to_dict()
        return best_layer, cr_map
    for label, df in [('Global Random', global_df), ('Block Random', block_df), ('Fixed', fixed_df)]:
        best_layer, cr_map = optimal_layer_and_cr(df)
        if best_layer is None:
            continue
        cells = []
        for cr in cr_columns:
            val = cr_map.get(cr)
            cells.append(f"{val:.2f}" if val is not None else "---")
        if cr_map:
            best_cr = min(cr_map, key=cr_map.get)
            best_cr_str = f"{best_cr:.0f}x"
        else:
            best_cr_str = "---"
        table2_lines.append(f"{label} (L={best_layer}) & " + " & ".join(cells) + f" & {best_cr_str} \\\\")
    table2_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    with open(latex_dir / "compression_ratio_summary.tex", "w") as f:
        f.write("\n".join(table2_lines))
    
    # Table 3: Head-to-head summary
    table3_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Head-to-head comparison: Fixed vs Random at matched configurations}",
        "\\label{tab:fixed_vs_random}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Comparison & Avg ECE Diff & Within $\\pm$0.2\\% & Winner & Verdict \\\\",
        "\\midrule",
    ]
    def add_row(name, stats):
        if not stats:
            return
        avg = stats.get('avg_diff', np.nan)
        within = stats.get('within_02_pct', 0) * 100 if 'within_02_pct' in stats else (stats.get('within_02', 0) / stats.get('total', 1) * 100 if stats.get('total') else 0)
        verdict = stats.get('verdict', 'Tie')
        winner = "Tie"
        if 'avg_diff' in stats and not np.isnan(avg):
            if abs(avg) < 0.1:
                winner = "Tie"
            else:
                winner = "Fixed" if avg > 0 else "Random"
        table3_lines.append(f"{name} & {avg:+.2f}\\% & {within:.1f}\\% & {winner} & {verdict} \\\\")
    add_row("Fixed vs Global Random", fixed_vs_global)
    add_row("Fixed vs Block Random", fixed_vs_block)
    if head_stats:
        avg = head_stats.get('avg_diff', np.nan)
        verdict = "No difference" if abs(avg) < 0.05 else "Structure matters (Block)" if avg > 0 else "Structure matters (Global)"
        table3_lines.append(f"Global vs Block Random & {avg:+.2f}\\% & --- & Tie & {verdict} \\\\")
    table3_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\small",
        "\\item Avg diff < 0.1\\% indicates practical equivalence. Positive diff means fixed is better.",
        "\\end{tablenotes}",
        "\\end{table}",
    ])
    with open(latex_dir / "head_to_head_summary.tex", "w") as f:
        f.write("\n".join(table3_lines))
    


def main():
    parser = argparse.ArgumentParser(description="Analyze layer count ablation (global vs block)")
    parser.add_argument('--results-dir', type=Path, default=Path('calibration_comparison_results_full'))
    parser.add_argument('--output-dir', type=Path, default=Path('calibration_comparison_results_full/layer_count_analysis'))
    parser.add_argument('--pattern', type=str, default='ablation_*.json')
    
    args = parser.parse_args()
    
    result_files = sorted(args.results_dir.glob(args.pattern))
    
    if not result_files:
        logger.error(f"No files found matching {args.pattern}")
        return 1
    
    logger.info(f"Found {len(result_files)} files")
    
    # Load data
    all_layer_data = []
    all_baseline_data = []
    
    for result_file in result_files:
        try:
            result = load_ablation_result(result_file)
            
            layer_df = extract_layer_count_metrics(result)
            if not layer_df.empty:
                all_layer_data.append(layer_df)
            
            baseline_df = load_baseline_geo_comb(result)
            if not baseline_df.empty:
                all_baseline_data.append(baseline_df)
            
            # logger.info(f"Loaded: {result_file.name}")
        except Exception as e:
            # logger.error(f"Failed: {result_file}: {e}")
            continue
    
    if not all_layer_data:
        logger.error("No data extracted")
        return 1
    
    all_data = pd.concat(all_layer_data, ignore_index=True)
    baseline_df = pd.concat(all_baseline_data, ignore_index=True) if all_baseline_data else pd.DataFrame()
    
    logger.info(f"Total experiments: {len(all_data)}")
    logger.info(f"Strategies: {sorted(all_data['strategy'].unique())}")
    logger.info(f"Layer counts: {sorted(all_data['num_layers'].unique())}")
    
    # Compute and print experiment overview
    overview = compute_experiment_overview(all_data)
    print_experiment_overview(overview)
    
    # Create output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw data
    raw_file = args.output_dir / "layer_count_raw.csv"
    all_data.to_csv(raw_file, index=False)
    
    # Analyze
    print("\n" + "="*140)
    print("LAYER COUNT ABLATION: Global Random vs Block Random vs Fixed (geo_comb)")
    print("="*140)
    
    global_data = all_data[all_data['strategy'] == 'global_random']
    block_data = all_data[all_data['strategy'] == 'block_random']
    geo_comb_data = all_data[all_data['strategy'] == 'geo_comb_fixed']
    
    results = {}
    geo_comb_comp = pd.DataFrame()
    geo_comb_cr = pd.DataFrame()
    
    if not global_data.empty:
        print("\n" + "="*140)
        print("GLOBAL RANDOM: Random selection from ALL layers")
        print("="*140)
        global_comp, global_cr = analyze_single_strategy(global_data, baseline_df, 'global_random')
        results['global_random'] = {'comparison': global_comp, 'cr_summary': global_cr}
    else:
        global_comp = pd.DataFrame()
        global_cr = pd.DataFrame()
    
    if not block_data.empty:
        print("\n" + "="*140)
        print("BLOCK RANDOM: Structured selection (one per block)")
        print("="*140)
        block_comp, block_cr = analyze_single_strategy(block_data, baseline_df, 'block_random')
        results['block_random'] = {'comparison': block_comp, 'cr_summary': block_cr}
    else:
        block_comp = pd.DataFrame()
        block_cr = pd.DataFrame()
    
    if not geo_comb_data.empty:
        print("\n" + "="*140)
        print("GEO_COMB FIXED: Fixed layer selection (DAC paper baseline)")
        print("="*140)
        geo_comb_comp, geo_comb_cr = analyze_single_strategy(geo_comb_data, baseline_df, 'geo_comb_fixed')
        results['geo_comb_fixed'] = {'comparison': geo_comb_comp, 'cr_summary': geo_comb_cr}
    
    # Focused summaries (skip matched-layer since L=5 missing for random)
    arch_layers = architecture_layer_counts(geo_comb_data)
    head_stats = head_to_head_all_layers(global_comp, block_comp)
    global_best_layers = best_layer_count_by_strategy(global_comp, 'global_random')
    block_best_layers = best_layer_count_by_strategy(block_comp, 'block_random')
    fixed_best_layers = best_layer_count_by_strategy(geo_comb_comp, 'geo_comb_fixed')
    global_cr_effect = compression_ratio_effect(global_comp, 'global_random')
    block_cr_effect = compression_ratio_effect(block_comp, 'block_random')
    fixed_cr_effect = compression_ratio_effect(geo_comb_comp, 'fixed (geo_comb)')
    fixed_vs_global = fixed_vs_random(geo_comb_comp, global_comp, 'global_random')
    fixed_vs_block = fixed_vs_random(geo_comb_comp, block_comp, 'block_random')
    
    # Export
    print("\nExporting LaTeX tables...")
    export_latex_tables(results, args.output_dir)
    
    # Save results
    for strategy_name, data in results.items():
        comp_file = args.output_dir / f"{strategy_name}_comparison.csv"
        data['comparison'].to_csv(comp_file, index=False)
        
        if data['cr_summary'] is not None and not data['cr_summary'].empty:
            cr_file = args.output_dir / f"{strategy_name}_per_cr.csv"
            data['cr_summary'].to_csv(cr_file, index=False)
    
    export_summary_tables(global_comp, block_comp, geo_comb_comp, head_stats, fixed_vs_global, fixed_vs_block,
                          {'global': global_best_layers, 'block': block_best_layers, 'fixed': fixed_best_layers},
                          {'global': global_cr_effect, 'block': block_cr_effect, 'fixed': fixed_cr_effect},
                          args.output_dir)
    
    print_header_block("KEY FINDINGS FOR ADVISORS")
    # Architecture breakdown
    if arch_layers:
        print("Architecture layer counts (fixed):")
        for item in arch_layers:
            print(f"  {item['config']}: {item['layers']} layers")
    # Global vs Block summary
    if head_stats:
        print(f"Global vs Block: Global wins {head_stats['global_wins']}/{head_stats['total']} ({head_stats['global_wins']/head_stats['total']*100:.1f}%), "
              f"Block wins {head_stats['block_wins']}/{head_stats['total']} ({head_stats['block_wins']/head_stats['total']*100:.1f}%), "
              f"Avg diff {head_stats['avg_diff']:+.2f}% (positive=Block better)")
        print(f"Skipped: {head_stats['skipped']} due to NaNs/missing.")
    # Best layer counts
    print("Best layer count (Global):")
    for lc in [2,4,6,8]:
        info = global_best_layers.get(lc, {})
        if info:
            print(f"  L={lc}: avg {info.get('avg_ece', float('nan')):.2f}% wins={info.get('wins',0)}")
    print("Best layer count (Block):")
    for lc in [2,4,6,8]:
        info = block_best_layers.get(lc, {})
        if info:
            print(f"  L={lc}: avg {info.get('avg_ece', float('nan')):.2f}% wins={info.get('wins',0)}")
    # Fixed vs random verdicts
    if fixed_vs_global:
        print(f"Fixed vs Global: within ±0.2% = {fixed_vs_global['within_02']}/{fixed_vs_global['total']} ({fixed_vs_global['within_02']/fixed_vs_global['total']*100:.1f}%), "
              f"avg diff {fixed_vs_global['avg_diff']:+.2f}% (positive=fixed better)")
    if fixed_vs_block:
        print(f"Fixed vs Block: within ±0.2% = {fixed_vs_block['within_02']}/{fixed_vs_block['total']} ({fixed_vs_block['within_02']/fixed_vs_block['total']*100:.1f}%), "
              f"avg diff {fixed_vs_block['avg_diff']:+.2f}% (positive=fixed better)")
    # Final verdicts
    random_ok = False
    if fixed_vs_global and fixed_vs_global.get('within_02') is not None:
        random_ok = random_ok or (fixed_vs_global['within_02']/fixed_vs_global['total']*100 >= 70)
    if fixed_vs_block and fixed_vs_block.get('within_02') is not None:
        random_ok = random_ok or (fixed_vs_block['within_02']/fixed_vs_block['total']*100 >= 70)
    block_matters = False
    if head_stats:
        block_matters = head_stats['avg_diff'] > 0.1  # block meaningfully better
    print("\nFINAL VERDICT")
    print(f"Random selection works: {'✓ Yes' if random_ok else '❌ No'}")
    print(f"Block structure matters: {'✓ Yes' if block_matters else 'No/Tie'}")
    if random_ok:
        print("Conclusion: Random works; performance comes from algorithm, not layer choice.")
    else:
        print("Conclusion: Fixed layer choice matters; success tied to picking layers.")
    
    print("\n" + "="*140)
    print("Analysis complete!")
    print(f"Results: {args.output_dir}")
    print(f"LaTeX tables: {args.output_dir / 'latex_tables'}")
    print("="*140)
    
    # Interpretation block
    print_header_block("INTERPRETATION")
    if fixed_vs_global:
        print(f"Q1: Does random layer selection work? -> {fixed_vs_global.get('verdict', '')}")
        print(f"    Reason: {fixed_vs_global.get('explanation', '')}")
    if head_stats:
        hd = head_stats
        if abs(hd.get('avg_diff', 0)) < 0.05:
            q2 = "✓ NO - Global and Block are equivalent (<0.05%)"
        elif abs(hd.get('avg_diff', 0)) < 0.10:
            q2 = "Minimal difference (<0.1%)"
        else:
            q2 = "Structure matters"
        print(f"Q2: Does block structure matter? -> {q2}")
    # Simple layer guidance from best-layer summaries
    def best_layer_line(summary, label):
        if not summary:
            return
        # pick best by lowest avg_ece
        best_lc = None
        best_val = None
        for lc in [2,4,6,8]:
            val = summary.get(lc, {}).get('avg_ece', np.nan)
            if np.isnan(val):
                continue
            if best_val is None or val < best_val:
                best_val = val
                best_lc = lc
        if best_lc is not None:
            print(f"{label}: L={best_lc} best ({best_val:.2f}% ECE)")
    best_layer_line(global_best_layers, "Global")
    best_layer_line(block_best_layers, "Block")
    best_layer_line(fixed_best_layers, "Fixed")
    print("Recommendation: Use 6-8 layers; CR 4x-16x balance performance vs speed.")
    if random_ok:
        print("Conclusion: Success comes from the algorithm; layer selection is not critical.")
    else:
        print("Conclusion: Layer selection contributes materially; prefer fixed if equivalence not met.")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())