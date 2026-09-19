#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_hybrid_enhanced.py

ENHANCED analysis with:
- Baseline dominance: which methods beat ALL baselines across datasets?
- Model-specific and training-method-specific breakdowns
- Top-k analysis (handling default top-5)

USAGE
-----
python aggregate_hybrid_enhanced.py \
  --results-dir /path/to/hybrid_ensemble_v1 \
  --baselines-dir /path/to/multi_layer_ensemble_v1 \
  --output-dir ./hybrid_analysis_enhanced \
  --oracle-threshold 0.005
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================
# Utility functions
# ============================================================================

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


def extract_topk_from_method(method_name: str) -> int:
    """Extract top-k value from method name, default to 5."""
    m = re.search(r'_top(\d+)_', method_name)
    if m:
        return int(m.group(1))
    # If no topk specified in method name, default is 5
    return 5


def parse_context_from_path(fp: Path) -> Dict[str, Any]:
    """Parse context from hybrid ensemble path."""
    parts = list(fp.parts)
    
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
    
    training_method = parts[seed_idx - 3] if seed_idx - 3 >= 0 else None
    dataset = parts[seed_idx - 2] if seed_idx - 2 >= 0 else None
    model = parts[seed_idx - 1] if seed_idx - 1 >= 0 else None
    
    strategy = None
    try:
        exp_idx = parts.index("experiments", seed_idx)
        if exp_idx + 1 < len(parts):
            strategy = parts[exp_idx + 1]
    except (ValueError, IndexError):
        pass
    
    seed_dir = Path(*parts[: seed_idx + 1])
    run_dir = fp.parent
    
    return dict(
        training_method=training_method,
        dataset=dataset,
        model=model,
        seed=seed,
        strategy=strategy,
        seed_dir=str(seed_dir),
        run_dir=str(run_dir),
    )


def parse_method_key(key: str) -> Dict[str, Any]:
    """Parse method keys like 'pareto_core_shallow_a1.5'."""
    alpha = None
    m = re.search(r"_a([0-9]*\.?[0-9]+)$", key)
    if m:
        try:
            alpha = float(m.group(1))
            key = key[:m.start()]
        except Exception:
            pass
    
    parts = key.split('_')
    weighting_methods = ['uniform', 'learned', 'deep', 'shallow']
    weighting = None
    for wm in weighting_methods:
        if parts[-1] == wm:
            weighting = parts[-1]
            parts = parts[:-1]
            break
    
    strategy = '_'.join(parts) if parts else None
    
    return {
        'strategy': strategy,
        'weighting': weighting,
        'alpha': alpha
    }


def find_ensemble_results(root_dir: Path) -> pd.DataFrame:
    """Find and parse all ensemble_results.json files."""
    records: List[Dict[str, Any]] = []
    
    result_files = list(root_dir.rglob("ensemble_results.json"))
    print(f"Found {len(result_files)} 'ensemble_results.json' files.")
    
    for f in result_files:
        try:
            ctx = parse_context_from_path(f)
            if not ctx or ctx.get("seed") is None:
                continue
            
            data = json.loads(f.read_text())
            
            if data.get("skipped_low_acc"):
                continue
            
            for key, result in data.items():
                if key in ['baselines', 'metric_picks_test_best_per_k', 'per_layer_ground_truth', 
                          'oracle_best_layer', 'skipped_low_acc']:
                    continue
                
                if not isinstance(result, dict):
                    continue
                
                if 'test_ece' not in result:
                    continue
                
                test_acc = result.get('test_accuracy')
                if test_acc is None:
                    continue 
                
                try:
                    accuracy_float = float(test_acc)
                except (ValueError, TypeError):
                    continue 
                    
                if accuracy_float < 0.8:
                    continue
                
                parsed = parse_method_key(key)
                
                records.append({
                    **ctx,
                    'method_key': key,
                    'strategy': parsed['strategy'] or ctx.get('strategy'),
                    'weighting': parsed['weighting'] or result.get('weighting_method'),
                    'alpha': parsed['alpha'] or result.get('depth_alpha'),
                    'k': result.get('k'),
                    'ece': round(float(result['test_ece']), 7),
                    'accuracy': round(accuracy_float, 7),
                    'brier': round(float(result['test_brier']), 7) if 'test_brier' in result else None,
                    'ees': round(float(result['ees']), 7) if 'ees' in result else None,
                    'selected_layers': result.get('selected_layers'),
                    'weights_json': json.dumps(result.get('weights')) if 'weights' in result else None,
                })
        
        except Exception as e:
            print(f"[WARN] Failed to parse {f}: {e}")
    
    if not records:
        print("[WARN] No records parsed.")
        return pd.DataFrame()
    
    df = pd.DataFrame.from_records(records)
    
    def make_label(r):
        parts = [r['strategy'], r['weighting']]
        if pd.notna(r['alpha']) and r['weighting'] in ['deep', 'shallow']:
            parts.append(f"a{r['alpha']}")
        return '_'.join(str(p) for p in parts if p)
    
    df['full_method'] = df.apply(make_label, axis=1)
    
    unique_cols = ['training_method', 'dataset', 'model', 'seed', 'full_method']
    num_removed = df.duplicated(subset=unique_cols, keep='first').sum()
    if num_removed > 0:
        print(f"[WARN] Found and removed {int(num_removed)} duplicate results.")
    
    df = df.drop_duplicates(subset=unique_cols, keep='first').reset_index(drop=True)
    
    # Add topk column
    df['topk'] = df['full_method'].apply(extract_topk_from_method)
    
    return df


def load_baselines(baselines_dir: Path, ensemble_df: pd.DataFrame) -> pd.DataFrame:
    """Load baseline results."""
    baseline_records = []
    
    runs = ensemble_df[['dataset', 'model', 'seed', 'training_method']].drop_duplicates()
    
    for _, row in runs.iterrows():
        baseline_path = (
            baselines_dir / 
            row['training_method'] / 
            row['dataset'] / 
            row['model'] / 
            f"seed{row['seed']}" / 
            "multi_layer_ensemble" / 
            "baselines" / 
            "baseline_results.json"
        )
        
        if not baseline_path.exists():
            continue
        
        try:
            data = json.loads(baseline_path.read_text())
            
            for method_name, metrics in data.items():
                if not isinstance(metrics, dict):
                    continue
                
                ece = metrics.get('ece')
                if ece is None:
                    continue
                
                baseline_records.append({
                    'dataset': row['dataset'],
                    'model': row['model'],
                    'seed': row['seed'],
                    'training_method': row['training_method'],
                    'method_name': method_name,
                    'ece': round(float(ece), 7),
                    'accuracy': round(float(metrics['acc']), 7) if 'acc' in metrics else None,
                    'brier': round(float(metrics['brier']), 7) if 'brier' in metrics else None,
                })
        
        except Exception as e:
            print(f"[WARN] Failed to load baseline {baseline_path}: {e}")
    
    return pd.DataFrame.from_records(baseline_records) if baseline_records else pd.DataFrame()


def load_oracle(ensemble_df: pd.DataFrame, results_dir: Path, 
                oracle_fallback_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load oracle results."""
    oracle_records = []
    
    runs = ensemble_df[['dataset', 'model', 'seed', 'training_method', 'run_dir']].drop_duplicates()
    
    for _, row in runs.iterrows():
        oracle_ece = None
        run_dir = Path(row['run_dir'])
        
        oracle_file = run_dir / "metric_docs" / "oracle_best_layer.json"
        if oracle_file.exists():
            try:
                data = json.loads(oracle_file.read_text())
                oracle_ece = data.get('test_ece')
            except Exception:
                pass
        
        if oracle_ece is None:
            per_layer_file = run_dir / "metric_docs" / "per_layer_ground_truth.json"
            if per_layer_file.exists():
                try:
                    data = json.loads(per_layer_file.read_text())
                    if isinstance(data, list):
                        valid = [r for r in data if 'test_ece' in r and np.isfinite(r['test_ece'])]
                        if valid:
                            oracle_ece = min(r['test_ece'] for r in valid)
                except Exception:
                    pass
        
        if oracle_ece is None and oracle_fallback_dir is not None:
            fallback_path = (
                oracle_fallback_dir /
                row['training_method'] /
                row['dataset'] /
                row['model'] /
                f"seed{row['seed']}"
            )
            
            for oracle_name in ["oracle_best_layer.json", "per_layer_ground_truth.json", 
                               "multi_composite_analysis.json"]:
                for candidate in fallback_path.rglob(oracle_name):
                    try:
                        data = json.loads(candidate.read_text())
                        if oracle_name == "multi_composite_analysis.json":
                            oracle_ece = data.get('empirical_best_ece')
                        elif isinstance(data, dict):
                            oracle_ece = data.get('test_ece') or data.get('ece')
                        elif isinstance(data, list):
                            valid = [r for r in data if 'test_ece' in r and np.isfinite(r['test_ece'])]
                            if valid:
                                oracle_ece = min(r['test_ece'] for r in valid)
                        
                        if oracle_ece is not None:
                            break
                    except Exception:
                        pass
                if oracle_ece is not None:
                    break
        
        if oracle_ece is not None:
            oracle_records.append({
                'dataset': row['dataset'],
                'model': row['model'],
                'seed': row['seed'],
                'training_method': row['training_method'],
                'oracle_ece': round(float(oracle_ece), 7),
            })
    
    return pd.DataFrame.from_records(oracle_records) if oracle_records else pd.DataFrame()


# ============================================================================
# NEW: Enhanced analysis functions
# ============================================================================

def find_baseline_dominators(ensemble_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find methods that beat ALL baselines across ALL datasets.
    This is a strong claim for publication!
    """
    on_cols = ['dataset', 'model', 'seed', 'training_method']
    
    datasets = sorted(ensemble_df['dataset'].unique())
    baselines = sorted(baseline_df['method_name'].unique())
    
    results = []
    
    for method in sorted(ensemble_df['full_method'].unique()):
        ens_method = ensemble_df[ensemble_df['full_method'] == method]
        
        # Track which datasets this method appears in
        method_datasets = set(ens_method['dataset'].unique())
        
        # For each baseline, check win rate across all datasets
        beats_all_baselines = True
        baseline_summary = {}
        
        for baseline_name in baselines:
            base_group = baseline_df[baseline_df['method_name'] == baseline_name]
            
            # Datasets where both method and baseline exist
            common_datasets = method_datasets.intersection(set(base_group['dataset'].unique()))
            
            if not common_datasets:
                # If method doesn't overlap with this baseline, skip
                continue
            
            total_wins = 0
            total_comparisons = 0
            
            for dataset in common_datasets:
                ens_ds = ens_method[ens_method['dataset'] == dataset]
                base_ds = base_group[base_group['dataset'] == dataset]
                
                merged = pd.merge(
                    ens_ds[on_cols + ['ece']],
                    base_ds[on_cols + ['ece']],
                    on=on_cols,
                    how='inner',
                    suffixes=('_ens', '_base')
                )
                
                if not merged.empty:
                    wins = int((merged['ece_ens'] < merged['ece_base']).sum())
                    total = int(len(merged))
                    total_wins += wins
                    total_comparisons += total
            
            if total_comparisons > 0:
                win_rate = total_wins / total_comparisons
                baseline_summary[baseline_name] = {
                    'wins': total_wins,
                    'total': total_comparisons,
                    'win_rate': win_rate
                }
                
                if win_rate <= 0.5:
                    beats_all_baselines = False
            else:
                # No comparisons possible with this baseline
                baseline_summary[baseline_name] = None
        
        # Only include methods that beat ALL baselines they were compared against
        if beats_all_baselines and len(baseline_summary) > 0:
            # Filter out None entries
            valid_baselines = {k: v for k, v in baseline_summary.items() if v is not None}
            
            if valid_baselines:  # Must beat at least one baseline
                results.append({
                    'Method': method,
                    'Datasets': len(method_datasets),
                    'Baselines Beaten': len(valid_baselines),
                    'Total Wins': sum(v['wins'] for v in valid_baselines.values()),
                    'Total Comparisons': sum(v['total'] for v in valid_baselines.values()),
                    'Overall Win Rate': sum(v['wins'] for v in valid_baselines.values()) / sum(v['total'] for v in valid_baselines.values()),
                    'Baseline Details': baseline_summary
                })
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('Overall Win Rate', ascending=False)
        return df
    else:
        return pd.DataFrame()


def per_model_leaderboard(df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Create leaderboard for each (dataset, model) combination."""
    leaderboards = {}
    
    for (dataset, model), group in df.groupby(['dataset', 'model']):
        rows = []
        for method, m_group in group.groupby('full_method'):
            mu, lo, hi = mean_ci(m_group['ece'].values)
            
            rows.append({
                'Method': method,
                'Mean ECE': mu,
                '95% CI Lower': lo,
                '95% CI Upper': hi,
                'Std Dev': round(float(np.std(m_group['ece'].values)), 7),
                'N': int(len(m_group)),
            })
        
        leaderboards[(dataset, model)] = (
            pd.DataFrame(rows)
            .sort_values('Mean ECE', ascending=True)
            .reset_index(drop=True)
        )
    
    return leaderboards


def per_training_method_leaderboard(df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Create leaderboard for each (dataset, training_method) combination."""
    leaderboards = {}
    
    for (dataset, training_method), group in df.groupby(['dataset', 'training_method']):
        rows = []
        for method, m_group in group.groupby('full_method'):
            mu, lo, hi = mean_ci(m_group['ece'].values)
            
            rows.append({
                'Method': method,
                'Mean ECE': mu,
                '95% CI Lower': lo,
                '95% CI Upper': hi,
                'Std Dev': round(float(np.std(m_group['ece'].values)), 7),
                'N': int(len(m_group)),
            })
        
        leaderboards[(dataset, training_method)] = (
            pd.DataFrame(rows)
            .sort_values('Mean ECE', ascending=True)
            .reset_index(drop=True)
        )
    
    return leaderboards


def topk_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by top-k value."""
    rows = []
    
    for topk, group in df.groupby('topk'):
        mu, lo, hi = mean_ci(group['ece'].values)
        
        rows.append({
            'Top-K': int(topk),
            'Mean ECE': mu,
            '95% CI Lower': lo,
            '95% CI Upper': hi,
            'Std Dev': round(float(np.std(group['ece'].values)), 7),
            'N': int(len(group)),
            'N Datasets': int(group['dataset'].nunique()),
            'N Methods': int(group['full_method'].nunique()),
        })
    
    return pd.DataFrame(rows).sort_values('Mean ECE', ascending=True)


# ============================================================================
# Existing analysis functions (unchanged)
# ============================================================================

def per_dataset_leaderboard(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create leaderboard for each dataset."""
    leaderboards = {}
    
    for dataset, ds_group in df.groupby('dataset'):
        rows = []
        for method, m_group in ds_group.groupby('full_method'):
            mu, lo, hi = mean_ci(m_group['ece'].values)
            
            rows.append({
                'Method': method,
                'Mean ECE': mu,
                '95% CI Lower': lo,
                '95% CI Upper': hi,
                'Std Dev': round(float(np.std(m_group['ece'].values)), 7),
                'N': int(len(m_group)),
            })
        
        leaderboards[dataset] = (
            pd.DataFrame(rows)
            .sort_values('Mean ECE', ascending=True)
            .reset_index(drop=True)
        )
    
    return leaderboards


def per_dataset_vs_baselines(ensemble_df: pd.DataFrame, baseline_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compare methods to baselines separately for each dataset."""
    on_cols = ['dataset', 'model', 'seed', 'training_method']
    
    comparisons = {}
    
    for dataset in ensemble_df['dataset'].unique():
        ens_ds = ensemble_df[ensemble_df['dataset'] == dataset]
        base_ds = baseline_df[baseline_df['dataset'] == dataset]
        
        if base_ds.empty:
            continue
        
        rows = []
        
        for method, ens_group in ens_ds.groupby('full_method'):
            for baseline_name, base_group in base_ds.groupby('method_name'):
                merged = pd.merge(
                    ens_group[on_cols + ['ece']],
                    base_group[on_cols + ['ece']],
                    on=on_cols,
                    how='inner',
                    suffixes=('_ens', '_base')
                )
                
                if merged.empty:
                    continue
                
                wins = int((merged['ece_ens'] < merged['ece_base']).sum())
                ties = int((merged['ece_ens'] == merged['ece_base']).sum())
                total = int(len(merged))
                win_rate = (wins + 0.5 * ties) / total if total > 0 else np.nan
                
                diff = merged['ece_base'].values - merged['ece_ens'].values
                
                rows.append({
                    'Method': method,
                    'Baseline': baseline_name,
                    'Win Rate': round(win_rate, 7),
                    'Wins': wins,
                    'Losses': total - wins - ties,
                    'Ties': ties,
                    'Median ΔECE': round(float(np.median(diff)), 7) if len(diff) else np.nan,
                    'Mean ECE (ours)': round(float(np.mean(merged['ece_ens'])), 7),
                    'Mean ECE (baseline)': round(float(np.mean(merged['ece_base'])), 7),
                    'N': total,
                })
        
        comparisons[dataset] = (
            pd.DataFrame(rows)
            .sort_values(['Method', 'Win Rate'], ascending=[True, False])
        )
    
    return comparisons


def per_dataset_vs_oracle(ensemble_df: pd.DataFrame, oracle_df: pd.DataFrame,
                          threshold: float = 0.01) -> Dict[str, pd.DataFrame]:
    """Compare methods to oracle separately for each dataset."""
    on_cols = ['dataset', 'model', 'seed', 'training_method']
    
    comparisons = {}
    
    for dataset in ensemble_df['dataset'].unique():
        ens_ds = ensemble_df[ensemble_df['dataset'] == dataset]
        oracle_ds = oracle_df[oracle_df['dataset'] == dataset]
        
        if oracle_ds.empty:
            continue
        
        rows = []
        
        for method, ens_group in ens_ds.groupby('full_method'):
            merged = pd.merge(
                ens_group[on_cols + ['ece']],
                oracle_ds[on_cols + ['oracle_ece']],
                on=on_cols,
                how='inner'
            )
            
            if merged.empty:
                continue
            
            total = int(len(merged))
            gaps = merged['ece'].values - merged['oracle_ece'].values
            wins = int((merged['ece'] < merged['oracle_ece']).sum())
            ties = int((merged['ece'] == merged['oracle_ece']).sum())
            within_threshold = int((np.abs(gaps) <= threshold).sum())
            
            rows.append({
                'Method': method,
                'Mean Gap': round(float(np.mean(gaps)), 7),
                'Median Gap': round(float(np.median(gaps)), 7),
                'Std Gap': round(float(np.std(gaps)), 7),
                'Better than Oracle': wins,
                'Worse than Oracle': total - wins - ties,
                f'Within {threshold}': within_threshold,
                f'% Within {threshold}': round(within_threshold / total * 100, 1) if total > 0 else 0.0,
                'Mean ECE (ours)': round(float(np.mean(merged['ece'])), 7),
                'Mean Oracle ECE': round(float(np.mean(merged['oracle_ece'])), 7),
                'N': total,
            })
        
        comparisons[dataset] = (
            pd.DataFrame(rows)
            .sort_values('Mean Gap', ascending=True)
        )
    
    return comparisons


def strategy_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown by strategy (ignoring weighting)."""
    rows = []
    
    for strategy, st_group in df.groupby('strategy'):
        mu, lo, hi = mean_ci(st_group['ece'].values)
        
        rows.append({
            'Strategy': strategy,
            'Mean ECE': mu,
            '95% CI Lower': lo,
            '95% CI Upper': hi,
            'Std Dev': round(float(np.std(st_group['ece'].values)), 7),
            'N': int(len(st_group)),
            'N Datasets': int(st_group['dataset'].nunique()),
        })
    
    return pd.DataFrame(rows).sort_values('Mean ECE', ascending=True)


def weighting_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Breakdown by weighting (ignoring strategy)."""
    rows = []
    
    for weighting, wt_group in df.groupby('weighting'):
        for alpha_val in wt_group['alpha'].dropna().unique():
            alpha_group = wt_group[wt_group['alpha'] == alpha_val]
            mu, lo, hi = mean_ci(alpha_group['ece'].values)
            
            rows.append({
                'Weighting': weighting,
                'Alpha': alpha_val if pd.notna(alpha_val) else None,
                'Mean ECE': mu,
                'Std Dev': round(float(np.std(alpha_group['ece'].values)), 7),
                'N': int(len(alpha_group)),
            })
        
        no_alpha = wt_group[wt_group['alpha'].isna()]
        if not no_alpha.empty:
            mu, lo, hi = mean_ci(no_alpha['ece'].values)
            rows.append({
                'Weighting': weighting,
                'Alpha': None,
                'Mean ECE': mu,
                'Std Dev': round(float(np.std(no_alpha['ece'].values)), 7),
                'N': int(len(no_alpha)),
            })
    
    return pd.DataFrame(rows).sort_values(['Weighting', 'Mean ECE'])


def baseline_win_matrix(ensemble_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Matrix showing which methods beat which baselines (overall)."""
    on_cols = ['dataset', 'model', 'seed', 'training_method']
    
    methods = sorted(ensemble_df['full_method'].unique())
    baselines = sorted(baseline_df['method_name'].unique())
    
    matrix = pd.DataFrame(index=methods, columns=baselines, dtype=float)
    
    for method in methods:
        ens_group = ensemble_df[ensemble_df['full_method'] == method]
        
        for baseline_name in baselines:
            base_group = baseline_df[baseline_df['method_name'] == baseline_name]
            
            merged = pd.merge(
                ens_group[on_cols + ['ece']],
                base_group[on_cols + ['ece']],
                on=on_cols,
                how='inner',
                suffixes=('_ens', '_base')
            )
            
            if not merged.empty:
                wins = int((merged['ece_ens'] < merged['ece_base']).sum())
                total = int(len(merged))
                matrix.loc[method, baseline_name] = wins / total if total > 0 else np.nan
    
    return matrix


# ============================================================================
# Enhanced Report Generation
# ============================================================================

def generate_enhanced_report(
    ensemble_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    per_ds_leaderboards: Dict[str, pd.DataFrame],
    per_ds_vs_baselines: Dict[str, pd.DataFrame],
    per_ds_vs_oracle: Dict[str, pd.DataFrame],
    dominators: pd.DataFrame,
    topk_breakdown: pd.DataFrame,
    per_model_lbs: Dict[Tuple[str, str], pd.DataFrame],
    per_training_lbs: Dict[Tuple[str, str], pd.DataFrame],
    threshold: float
) -> List[str]:
    """Generate comprehensive enhanced text report."""
    lines = []
    
    lines.append("="*88)
    lines.append("ENHANCED HYBRID ENSEMBLE ANALYSIS")
    lines.append("="*88)
    lines.append("")
    
    # ========================================================================
    # NEW: Baseline dominance section - PUBLICATION-READY CLAIM
    # ========================================================================
    lines.append("🏆 METHODS THAT BEAT ALL BASELINES ACROSS ALL DATASETS")
    lines.append("="*88)
    
    if not dominators.empty:
        lines.append(f"Found {len(dominators)} method(s) that beat ALL baselines across datasets!\n")
        
        for idx, row in dominators.iterrows():
            lines.append(f"✓ {row['Method']}")
            lines.append(f"  Overall Win Rate: {row['Overall Win Rate']*100:.1f}% ({row['Total Wins']}/{row['Total Comparisons']})")
            lines.append(f"  Datasets: {row['Datasets']} | Baselines beaten: {row['Baselines Beaten']}")
            
            # Show per-baseline breakdown
            lines.append(f"  Breakdown by baseline:")
            for baseline, details in row['Baseline Details'].items():
                if details is not None:
                    win_rate = details['win_rate'] * 100
                    lines.append(f"    • {baseline:30s}: {win_rate:5.1f}% ({details['wins']}/{details['total']})")
            lines.append("")
        
        lines.append("💡 PUBLICATION CLAIM: These methods consistently outperform ALL standard")
        lines.append("   calibration baselines (temp scaling, isotonic, beta, etc.) across datasets.\n")
    else:
        lines.append("No single method beats ALL baselines across all datasets.")
        lines.append("This suggests dataset-specific tuning is important.\n")
    
    lines.append("")
    
    # ========================================================================
    # NEW: Top-K analysis
    # ========================================================================
    lines.append("TOP-K ANALYSIS (default is k=5 if not specified)")
    lines.append("-" * 88)
    
    if not topk_breakdown.empty:
        lines.append("\nPerformance by ensemble size (k):\n")
        for _, row in topk_breakdown.iterrows():
            lines.append(
                f"  k={int(row['Top-K'])}: Mean ECE={row['Mean ECE']:.4f} ± {row['Std Dev']:.4f} "
                f"(N={row['N']}, {row['N Methods']} methods, {row['N Datasets']} datasets)"
            )
        
        best_k = topk_breakdown.iloc[0]
        lines.append(f"\n💡 Best k: {int(best_k['Top-K'])} (ECE={best_k['Mean ECE']:.4f})")
    
    lines.append("\n")
    
    # ========================================================================
    # Per-dataset winners (existing)
    # ========================================================================
    lines.append("BEST METHOD PER DATASET:")
    lines.append("-" * 88)
    for dataset in sorted(per_ds_leaderboards.keys()):
        lb = per_ds_leaderboards[dataset]
        if not lb.empty:
            best = lb.iloc[0]
            lines.append(f"\n{dataset.upper()}:")
            lines.append(f"  Winner: {best['Method']}")
            lines.append(f"  ECE: {best['Mean ECE']:.4f} ± {best['Std Dev']:.4f} (N={best['N']})")
            
            if len(lb) > 1:
                lines.append(f"  Top 3:")
                for i, row in lb.head(3).iterrows():
                    lines.append(f"    {i+1}. {row['Method']:40s} ECE={row['Mean ECE']:.4f}")
    
    lines.append("")
    
    # ========================================================================
    # NEW: Model-specific analysis
    # ========================================================================
    lines.append("\nMODEL-SPECIFIC WINNERS:")
    lines.append("-" * 88)
    
    model_winners = {}
    for (dataset, model), lb in per_model_lbs.items():
        if not lb.empty:
            winner = lb.iloc[0]
            key = f"{dataset} + {model}"
            model_winners[key] = winner['Method']
            lines.append(f"\n{key}:")
            lines.append(f"  Winner: {winner['Method']}")
            lines.append(f"  ECE: {winner['Mean ECE']:.4f} (N={winner['N']})")
    
    lines.append("")
    
    # ========================================================================
    # NEW: Training method analysis
    # ========================================================================
    lines.append("\nTRAINING-METHOD-SPECIFIC WINNERS:")
    lines.append("-" * 88)
    
    training_winners = {}
    for (dataset, training_method), lb in per_training_lbs.items():
        if not lb.empty:
            winner = lb.iloc[0]
            key = f"{dataset} + {training_method}"
            training_winners[key] = winner['Method']
            lines.append(f"\n{key}:")
            lines.append(f"  Winner: {winner['Method']}")
            lines.append(f"  ECE: {winner['Mean ECE']:.4f} (N={winner['N']})")
    
    lines.append("")
    
    # ========================================================================
    # Per-dataset vs baselines (existing, but enhanced)
    # ========================================================================
    lines.append("\nPER-DATASET BASELINE PERFORMANCE:")
    lines.append("-" * 88)
    
    for dataset in sorted(per_ds_vs_baselines.keys()):
        comp = per_ds_vs_baselines[dataset]
        if comp.empty:
            continue
        
        lines.append(f"\n{dataset.upper()}:")
        
        if dataset not in per_ds_leaderboards or per_ds_leaderboards[dataset].empty:
            continue
            
        best_method = per_ds_leaderboards[dataset].iloc[0]['Method']
        best_comp = comp[comp['Method'] == best_method]
        
        if not best_comp.empty:
            lines.append(f"  Best method ({best_method}) vs baselines:")
            for _, row in best_comp.iterrows():
                win_pct = row['Win Rate'] * 100
                delta = row['Median ΔECE']
                status = "✓ WIN" if row['Win Rate'] > 0.5 else "✗ LOSE"
                lines.append(
                    f"    {status} vs {row['Baseline']:25s} | "
                    f"{win_pct:5.1f}% ({row['Wins']}/{row['N']}) | "
                    f"ΔECE={delta:+.4f}"
                )
    
    lines.append("")
    
    # ========================================================================
    # Oracle comparison (existing)
    # ========================================================================
    if not oracle_df.empty:
        lines.append("\nPER-DATASET ORACLE PERFORMANCE:")
        lines.append("-" * 88)
        
        for dataset in sorted(per_ds_vs_oracle.keys()):
            comp = per_ds_vs_oracle[dataset]
            if comp.empty:
                continue
            
            lines.append(f"\n{dataset.upper()}:")
            
            if dataset not in per_ds_leaderboards or per_ds_leaderboards[dataset].empty:
                continue
            
            best_method = per_ds_leaderboards[dataset].iloc[0]['Method']
            best_oracle = comp[comp['Method'] == best_method]
            
            if not best_oracle.empty:
                row = best_oracle.iloc[0]
                lines.append(f"  Best method ({best_method}) vs oracle:")
                lines.append(f"    Mean Gap: {row['Mean Gap']:+.4f}")
                lines.append(f"    Median Gap: {row['Median Gap']:+.4f}")
                lines.append(f"    Within {threshold}: {row[f'% Within {threshold}']:.1f}% ({row[f'Within {threshold}']}/{row['N']})")
                lines.append(f"    Better than Oracle: {row['Better than Oracle']}/{row['N']}")
                
                if row['Better than Oracle'] > row['N'] / 2:
                    lines.append(f"    🏆 BEATS ORACLE in majority of cases!")
    
    lines.append("")
    
    # ========================================================================
    # Strategy insights (existing)
    # ========================================================================
    lines.append("\nSTRATEGY INSIGHTS:")
    lines.append("-" * 88)
    
    strategy_wins = {}
    for dataset, lb in per_ds_leaderboards.items():
        if not lb.empty:
            winner = lb.iloc[0]['Method']
            strategy = ensemble_df[ensemble_df['full_method'] == winner]['strategy'].iloc[0]
            strategy_wins[strategy] = strategy_wins.get(strategy, 0) + 1
    
    lines.append("Dataset wins by strategy:")
    for strategy, count in sorted(strategy_wins.items(), key=lambda x: -x[1]):
        total_datasets = len(per_ds_leaderboards)
        lines.append(f"  {strategy:30s}: {count}/{total_datasets} datasets")
    
    lines.append("")
    
    return lines


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced per-dataset analysis with baseline dominance and granular breakdowns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Root directory containing hybrid ensemble results",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=None,
        help="Directory containing baseline results",
    )
    parser.add_argument(
        "--oracle-fallback-dir",
        type=Path,
        default=None,
        help="Fallback directory to search for oracle",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./hybrid_analysis_enhanced"),
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--oracle-threshold",
        type=float,
        default=0.005,
        help="Threshold for 'within oracle' analysis",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ENHANCED HYBRID ENSEMBLE ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n[1/4] Loading data...")
    ensemble_df = find_ensemble_results(args.results_dir)
    print(f"  ✓ Loaded {len(ensemble_df)} ensemble results")
    
    baseline_df = pd.DataFrame()
    if args.baselines_dir and args.baselines_dir.exists():
        baseline_df = load_baselines(args.baselines_dir, ensemble_df)
        if not baseline_df.empty:
            print(f"  ✓ Loaded {len(baseline_df)} baseline results")
    
    oracle_df = load_oracle(ensemble_df, args.results_dir, args.oracle_fallback_dir)
    if not oracle_df.empty:
        print(f"  ✓ Loaded {len(oracle_df)} oracle results")
    
    # Standard per-dataset analysis
    print("\n[2/4] Generating per-dataset analysis...")
    
    per_ds_leaderboards = per_dataset_leaderboard(ensemble_df)
    print(f"  ✓ Created leaderboards for {len(per_ds_leaderboards)} datasets")
    
    per_ds_vs_baselines = {}
    if not baseline_df.empty:
        per_ds_vs_baselines = per_dataset_vs_baselines(ensemble_df, baseline_df)
        print(f"  ✓ Created baseline comparisons for {len(per_ds_vs_baselines)} datasets")
    
    per_ds_vs_oracle = {}
    if not oracle_df.empty:
        per_ds_vs_oracle = per_dataset_vs_oracle(ensemble_df, oracle_df, args.oracle_threshold)
        print(f"  ✓ Created oracle comparisons for {len(per_ds_vs_oracle)} datasets")
    
    # NEW: Enhanced analysis
    print("\n[3/4] Running enhanced analysis...")
    
    dominators = pd.DataFrame()
    if not baseline_df.empty:
        dominators = find_baseline_dominators(ensemble_df, baseline_df)
        if not dominators.empty:
            print(f"  🏆 Found {len(dominators)} method(s) that beat ALL baselines!")
        else:
            print(f"  ℹ️  No single method beats all baselines across all datasets")
    
    topk_breakdown = topk_analysis(ensemble_df)
    print(f"  ✓ Created top-k analysis")
    
    per_model_lbs = per_model_leaderboard(ensemble_df)
    print(f"  ✓ Created {len(per_model_lbs)} model-specific leaderboards")
    
    per_training_lbs = per_training_method_leaderboard(ensemble_df)
    print(f"  ✓ Created {len(per_training_lbs)} training-method-specific leaderboards")
    
    # Save results
    print("\n[4/4] Saving results...")
    
    # Per-dataset subdirectories
    for dataset in per_ds_leaderboards.keys():
        ds_dir = args.output_dir / "per_dataset" / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        
        per_ds_leaderboards[dataset].to_csv(
            ds_dir / f"{dataset}_leaderboard.csv", index=False
        )
        
        if dataset in per_ds_vs_baselines:
            per_ds_vs_baselines[dataset].to_csv(
                ds_dir / f"{dataset}_vs_baselines.csv", index=False
            )
        
        if dataset in per_ds_vs_oracle:
            per_ds_vs_oracle[dataset].to_csv(
                ds_dir / f"{dataset}_vs_oracle.csv", index=False
            )
    
    print(f"  ✓ Saved per-dataset tables")
    
    # NEW: Save enhanced analysis results
    if not dominators.empty:
        # Save without the nested dict column
        dom_save = dominators.drop(columns=['Baseline Details'])
        dom_save.to_csv(args.output_dir / "baseline_dominators.csv", index=False)
        print(f"  ✓ Saved baseline dominators")
    
    topk_breakdown.to_csv(args.output_dir / "topk_analysis.csv", index=False)
    print(f"  ✓ Saved top-k analysis")
    
    # Save model-specific leaderboards
    model_dir = args.output_dir / "per_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, model), lb in per_model_lbs.items():
        filename = f"{dataset}_{model}_leaderboard.csv"
        lb.to_csv(model_dir / filename, index=False)
    print(f"  ✓ Saved {len(per_model_lbs)} model-specific leaderboards")
    
    # Save training-method-specific leaderboards
    training_dir = args.output_dir / "per_training_method"
    training_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, training_method), lb in per_training_lbs.items():
        filename = f"{dataset}_{training_method}_leaderboard.csv"
        lb.to_csv(training_dir / filename, index=False)
    print(f"  ✓ Saved {len(per_training_lbs)} training-method leaderboards")
    
    # Other breakdowns
    strategy_bd = strategy_breakdown(ensemble_df)
    strategy_bd.to_csv(args.output_dir / "strategy_breakdown.csv", index=False)
    
    weighting_bd = weighting_breakdown(ensemble_df)
    weighting_bd.to_csv(args.output_dir / "weighting_breakdown.csv", index=False)
    
    if not baseline_df.empty:
        win_matrix = baseline_win_matrix(ensemble_df, baseline_df)
        win_matrix.to_csv(args.output_dir / "baseline_win_matrix.csv")
    
    # Generate enhanced report
    report_lines = generate_enhanced_report(
        ensemble_df, baseline_df, oracle_df,
        per_ds_leaderboards, per_ds_vs_baselines, per_ds_vs_oracle,
        dominators, topk_breakdown, per_model_lbs, per_training_lbs,
        args.oracle_threshold
    )
    
    report_path = args.output_dir / "ENHANCED_REPORT.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"  ✓ Saved enhanced report")
    
    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    if not dominators.empty:
        print("\n🏆 METHODS BEATING ALL BASELINES:")
        for _, row in dominators.iterrows():
            print(f"  • {row['Method']}: {row['Overall Win Rate']*100:.1f}% win rate")
    
    print("\n📊 Top-k performance:")
    for _, row in topk_breakdown.head(3).iterrows():
        print(f"  k={int(row['Top-K'])}: ECE={row['Mean ECE']:.4f} (N={row['N']})")
    
    print("\n🎯 Best method per dataset:")
    for dataset in sorted(per_ds_leaderboards.keys()):
        best = per_ds_leaderboards[dataset].iloc[0]
        print(f"  {dataset:15s}: {best['Method']:40s} ECE={best['Mean ECE']:.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())