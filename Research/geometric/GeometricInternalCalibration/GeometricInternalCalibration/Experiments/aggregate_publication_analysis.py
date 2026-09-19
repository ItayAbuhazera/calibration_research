#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_publication_analysis.py

A MASTER script to aggregate and analyze results from MULTIPLE experiment types,
incorporating detailed feedback for a publication-quality analysis.

1. Hybrid Ensembles (from aggregate_hybrid_ensemble_results.py)
2. NC Ensembles (from aggregate_nc_ensemble_results.py)
3. Multi-Layer Baselines (Original baselines from hybrid script)
4. Post-Hoc Baselines (Temp, Platt, etc., from NC script)
5. Oracle (Best Layer)

This script includes:
- Unified data loading with consistent accuracy thresholds.
- Robust method categorization.
- Granular, per-configuration win-rate analysis against baselines.
- Direct NC vs. Hybrid comparison (statistical and visual).
- Publication-ready visualizations (boxplots, scatter plots).

USAGE
-----
python aggregate_publication_analysis.py \
  --hybrid-results-dir /path/to/hybrid_ensemble_v1 \
  --nc-results-dir /path/to/nc_ensemble_results \
  --multi-layer-baselines-dir /path/to/multi_layer_ensemble_v1 \
  --post-hoc-baselines-dir /path/to/layer_selection_analysis_baselines_4 \
  --output-dir ./publication_master_analysis \
  --oracle-threshold 0.005 \
  --min-accuracy-threshold 0.6
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import warnings

# NEW: Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# NEW: Method Categorization (More Precise)
# ============================================================================

LAYER_SELECTION_CATEGORIES = {
    # 'nc_metrics' is handled by method_source
    'pareto_methods': ['pareto_'],
    'calibration_proxies': [
        'kfold_ece', 'prototype_softmax_ece', 'reliability_curve',
        'logistic_calibratability'
    ],
    'uncertainty_geometry': [
        'uncertainty_geometry', 'confidence_distance',
        'boundary_proximity', 'geometry_error'
    ],
    'class_separation': [
        'class_separation', 'margin_tail', 'class_balanced'
    ],
    'composite': ['composite_score', 'psc', 'hybrid_'],
}

CATEGORY_DISPLAY_NAMES = {
    'nc_metrics': 'NC Metrics Only',
    'pareto_methods': 'Pareto Front',
    'calibration_proxies': 'Calibration Proxies',
    'uncertainty_geometry': 'Uncertainty-Geometry',
    'class_separation': 'Class Separation',
    'composite': 'Composite/Hybrid',
    'dac_predetermined': 'DAC Predetermined',
    'single_optimal_layer': 'Single Optimal Layer',
    'topk_ensemble': 'Top-K Ensemble',
    'other': 'Other',
}

def categorize_method(full_method_name: str, strategy: str, source: str) -> str:
    """Categorizes a method based on its name, strategy, and source."""
    
    # FIX: Use method_source for the most reliable classification
    if source == 'nc_ensemble':
        return 'nc_metrics'
        
    if pd.isna(full_method_name):
        return 'unknown'
    
    fm_lower = full_method_name.lower()
    s_lower = str(strategy).lower()

    if 'dac_predetermined' in s_lower:
        return 'dac_predetermined'

    if s_lower.startswith('optimal_'):
        return 'single_optimal_layer'

    if s_lower.startswith('top2_') or s_lower.startswith('top3_'):
        return 'topk_ensemble'
    
    for category, keywords in LAYER_SELECTION_CATEGORIES.items():
        for keyword in keywords:
            if keyword in fm_lower or keyword in s_lower:
                return category
    
    if 'hybrid' in fm_lower:
        return 'composite'
        
    return 'other'

# ============================================================================
# NEW: Dataset & Baseline Unification Helpers
# ============================================================================

def get_dataset_variants(name: str) -> Set[str]:
    """Generate common variations of dataset names."""
    base = name.lower()
    variants = {name, base}
    variants.add(base.replace('-', ''))
    variants.add(base.replace('_', '-'))
    if 'cifar' in base and base.endswith('c') and '-' not in base:
        variants.add(base[:-1] + '-c')
    return variants

def check_path_variants(base_path: Path, target_dataset: str) -> Optional[Path]:
    """Check if path exists, trying dataset name variants if needed."""
    if base_path.exists():
        return base_path
    path_str = str(base_path)
    variants = get_dataset_variants(target_dataset)
    for v in variants:
        if v == target_dataset: continue
        new_path_str = path_str.replace(target_dataset, v)
        if Path(new_path_str).exists():
            return Path(new_path_str)
    return None

def unify_dataset_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize dataset names (cifar10c -> cifar10-c)."""
    if df.empty:
        return df
    mapping = {
        'cifar10c': 'cifar10-c',
        'cifar100c': 'cifar100-c',
    }
    df['dataset'] = df['dataset'].replace(mapping)
    return df

def unify_baseline_names(df: pd.DataFrame) -> pd.DataFrame:
    """Merge MultiLayer_ and PostHoc_ into Baseline_, warn on conflicts."""
    if df.empty:
        return df
    
    def _rename(name):
        return re.sub(r'^(MultiLayer_|PostHoc_)', 'Baseline_', name)
    
    df['method_name'] = df['method_name'].apply(_rename)
    
    # Check for conflicting duplicates
    dup_cols = ['dataset', 'model', 'seed', 'training_method', 'method_name']
    duplicates = df[df.duplicated(subset=dup_cols, keep=False)]
    
    if not duplicates.empty:
        grouped = duplicates.groupby(dup_cols)['ece']
        ranges = grouped.max() - grouped.min()
        bad_dups = ranges[ranges > 1e-5]
        
        if not bad_dups.empty:
            print("\n" + "!"*80)
            print(f"[WARNING] Found {len(bad_dups)} baseline conflicts!")
            print(bad_dups.head(5))
            print("!"*80 + "\n")
    
    df = df.sort_values('ece')
    df = df.drop_duplicates(subset=dup_cols, keep='first')
    return df.reset_index(drop=True)

# ============================================================================
# Utility functions (from hybrid script)
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

# ============================================================================
# DATA LOADER 1: Hybrid Ensemble Results
# ============================================================================

def find_hybrid_ensemble_results(root_dir: Path, min_acc_threshold: float) -> pd.DataFrame:
    """Find and parse all hybrid ensemble_results.json files."""
    records: List[Dict[str, Any]] = []
    
    result_files = list(root_dir.rglob("ensemble_results.json"))
    print(f"[Hybrid Loader] Found {len(result_files)} 'ensemble_results.json' files in {root_dir}")
    
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
                
                # *** FIX ***: Use unified threshold
                if accuracy_float < min_acc_threshold:
                    continue
                
                parsed = parse_method_key(key)
                
                dataset_lower = (ctx.get('dataset') or '').lower()

                if 'cifar100' in dataset_lower and accuracy_float > 0.85:
                    continue

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
                    'method_source': 'hybrid' # Flag for unified DF
                })
        
        except Exception as e:
            print(f"[WARN] Failed to parse {f}: {e}")
    
    if not records:
        print("[WARN] No hybrid ensemble records parsed.")
        return pd.DataFrame()
    
    df = pd.DataFrame.from_records(records)
    
    def make_label(r):
        parts = [r['strategy'], r['weighting']]
        if pd.notna(r['alpha']) and r['weighting'] in ['deep', 'shallow']:
            parts.append(f"a{r['alpha']}")
        label = '_'.join(str(p) for p in parts if p)
        return f"Hybrid_{label}" # Add prefix to avoid name collision
    
    df['full_method'] = df.apply(make_label, axis=1)
    
    unique_cols = ['training_method', 'dataset', 'model', 'seed', 'full_method']
    num_removed = df.duplicated(subset=unique_cols, keep='first').sum()
    if num_removed > 0:
        print(f"[WARN] Found and removed {int(num_removed)} duplicate hybrid results.")
    
    df = df.drop_duplicates(subset=unique_cols, keep='first').reset_index(drop=True)
    
    # Add topk column
    df['topk'] = df['full_method'].apply(extract_topk_from_method)
    
    # *** NEW ***: Add category
    df['category'] = df.apply(lambda row: categorize_method(row['full_method'], row['strategy'], row['method_source']), axis=1)

    return df

# ============================================================================
# DATA LOADER 2: NC Ensemble Results
# ============================================================================

def find_nc_ensemble_results(root_dir: Path, min_acc_threshold: float) -> pd.DataFrame:
    """Find and parse all NC ensemble_results.json files."""
    records: List[Dict[str, Any]] = []
    
    result_files = list(root_dir.rglob("ensemble_results.json"))
    print(f"[NC Loader] Found {len(result_files)} 'ensemble_results.json' files in {root_dir}")
    
    loaded = 0
    failed = 0
    
    for f in result_files:
        try:
            # Expected structure: {base}/{training_loss}/{dataset}/{model}/{seed}/{layer_strategy}/{weighting_method}/ensemble_results.json
            parts = f.relative_to(root_dir).parts
            
            if len(parts) < 7:
                failed += 1
                continue
            
            training_loss = parts[0]
            dataset_from_path = parts[1]
            model = parts[2]
            seed_str = parts[3]  # e.g., 'seed19'
            layer_strategy = parts[4]
            weighting_method = parts[5]
            
            seed = int(seed_str.replace('seed', ''))
            
            data = json.loads(f.read_text())
            
            for config_name, result in data.items():
                if not isinstance(result, dict):
                    continue
                
                # NORMALIZE column names to match hybrid script
                if 'test_ece' not in result:
                    continue
                
                ece = result.get('test_ece')
                acc = result.get('test_accuracy')
                brier = result.get('test_brier')

                dataset_meta = (
                    result.get('dataset')
                    or result.get('dataset_name')
                    or result.get('dataset_id')
                    or dataset_from_path
                )
                dataset = str(dataset_meta).lower() if dataset_meta else dataset_from_path
                dataset_lower = dataset.lower() if isinstance(dataset, str) else dataset
                
                if ece is None or acc is None:
                    continue
                
                if dataset_lower and 'cifar100' in dataset_lower and float(acc) > 0.85:
                    continue
                
                # *** FIX ***: Use unified threshold
                if float(acc) < min_acc_threshold:
                    continue
                
                # Create a unique, prefixed method name
                full_method = f"NC_{layer_strategy}_{weighting_method}"
                
                # *** FIX ***: Correctly parse k/topk
                k_val = result.get('num_layers') or len(result.get('selected_layers', []))
                if k_val == 0: # Fallback
                     k_val = extract_topk_from_method(config_name)

                record = {
                    'training_method': training_loss, # Normalize name
                    'dataset': dataset,
                    'model': model,
                    'seed': seed,
                    'strategy': layer_strategy,
                    'weighting': weighting_method,
                    'k': k_val, # This is the 'k'
                    'topk': k_val, # This is for the 'topk_analysis' function
                    'ece': round(float(ece), 7),
                    'accuracy': round(float(acc), 7),
                    'brier': round(float(brier), 7) if brier is not None else None,
                    'selected_layers': result.get('selected_layers'),
                    'method_source': 'nc_ensemble', # Flag for unified DF
                    'full_method': full_method,
                    'method_key': config_name
                }
                records.append(record)
                loaded += 1
        
        except Exception as e:
            print(f"[WARN] Failed to parse {f}: {e}")
            failed += 1
            
    if not records:
        print("[WARN] No NC ensemble records parsed.")
        return pd.DataFrame()
        
    df = pd.DataFrame.from_records(records)
    
    unique_cols = ['training_method', 'dataset', 'model', 'seed', 'full_method']
    num_removed = df.duplicated(subset=unique_cols, keep='first').sum()
    if num_removed > 0:
        print(f"[WARN] Found and removed {int(num_removed)} duplicate NC results.")
    
    df = df.drop_duplicates(subset=unique_cols, keep='first').reset_index(drop=True)

    # *** NEW ***: Add category
    df['category'] = df.apply(lambda row: categorize_method(row['full_method'], row['strategy'], row['method_source']), axis=1)
    
    return df

# ============================================================================
# DATA LOADER 3: Multi-Layer Baselines
# ============================================================================

def load_multi_layer_baselines(baselines_dir: Path, ensemble_df: pd.DataFrame) -> pd.DataFrame:
    """Load 'old' multi-layer baseline results."""
    baseline_records = []
    
    runs = ensemble_df[['dataset', 'model', 'seed', 'training_method']].drop_duplicates()
    
    print(f"[Multi-Layer Baseline Loader] Probing for {len(runs)} unique runs in {baselines_dir}")

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
        
        real_path = check_path_variants(baseline_path, row['dataset'])
        if not real_path:
            continue
        
        try:
            data = json.loads(real_path.read_text())
            
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
                    'method_name': f"MultiLayer_{method_name}", # Add prefix
                    'ece': round(float(ece), 7),
                    'accuracy': round(float(metrics['acc']), 7) if 'acc' in metrics else None,
                    'brier': round(float(metrics['brier']), 7) if 'brier' in metrics else None,
                    'method_source': 'multi_layer'
                })
        
        except Exception as e:
            print(f"[WARN] Failed to load baseline {baseline_path}: {e}")
    
    return pd.DataFrame.from_records(baseline_records) if baseline_records else pd.DataFrame()


# ============================================================================
# DATA LOADER 4: Post-Hoc Baselines
# ============================================================================

def load_post_hoc_baselines(baselines_dir: Path, ensemble_df: pd.DataFrame) -> pd.DataFrame:
    """Load post-hoc baseline results (Temp, Platt, etc.)."""
    baseline_records = []
    
    runs = ensemble_df[['dataset', 'model', 'seed', 'training_method']].drop_duplicates()
    
    print(f"[Post-Hoc Baseline Loader] Probing for {len(runs)} unique runs in {baselines_dir}")

    # Use rglob, as path structure might vary slightly
    result_files = list(baselines_dir.rglob("baseline_results.json"))
    
    if not result_files:
        print("[WARN] No post-hoc baseline_results.json files found.")
        return pd.DataFrame()

    # Create a lookup map from the found files for efficiency
    file_index = defaultdict(list)
    for f in baselines_dir.rglob("baseline_results.json"):
        try:
            parts = f.relative_to(baselines_dir).parts
            if len(parts) < 5: continue
            train_m, ds_disk, mod, seed_s = parts[0], parts[1], parts[2], parts[3]
            if not seed_s.startswith("seed"): continue
            seed = int(seed_s.replace("seed", ""))
            file_index[(train_m, mod, seed)].append((ds_disk, f))
        except Exception:
            pass
            
    loaded_count = 0
    for _, row in runs.iterrows():
        key = (row['training_method'], row['model'], row['seed'])
        candidates = file_index.get(key, [])
        if not candidates: continue
        
        valid_variants = get_dataset_variants(row['dataset'])
        matched_file = None
        for ds_disk, fpath in candidates:
            if ds_disk in valid_variants:
                matched_file = fpath
                break
        
        if matched_file:
            baseline_path = matched_file
            try:
                data = json.loads(baseline_path.read_text())
                
                for method_name, metrics in data.items():
                    if not isinstance(metrics, dict):
                        continue
                    
                    # NORMALIZE column names
                    ece = metrics.get('ece')
                    if ece is None:
                        continue
                    
                    baseline_records.append({
                        'dataset': row['dataset'],
                        'model': row['model'],
                        'seed': row['seed'],
                        'training_method': row['training_method'],
                        'method_name': f"PostHoc_{method_name}", # Add prefix
                        'ece': round(float(ece), 7),
                        'accuracy': round(float(metrics['acc']), 7) if 'acc' in metrics else None,
                        'brier': round(float(metrics['brier']), 7) if 'brier' in metrics else None,
                        'method_source': 'post_hoc'
                    })
                loaded_count += 1
            except Exception as e:
                print(f"[WARN] Failed to load baseline {baseline_path}: {e}")

    print(f"[Post-Hoc Baseline Loader] Successfully loaded {len(baseline_records)} records from {loaded_count} files.")
    return pd.DataFrame.from_records(baseline_records) if baseline_records else pd.DataFrame()


# ============================================================================
# DATA LOADER 5: Oracle Results
# ============================================================================

def load_oracle(ensemble_df: pd.DataFrame, 
                hybrid_dir: Path, nc_dir: Optional[Path], 
                oracle_fallback_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load oracle results, searching in all provided dirs."""
    oracle_records = []
    
    # Combine run info
    all_runs = ensemble_df[['dataset', 'model', 'seed', 'training_method']].drop_duplicates()

    print(f"[Oracle Loader] Probing for {len(all_runs)} unique runs")
    
    for _, row in all_runs.iterrows():
        oracle_ece = None
        
        # 1. Check Hybrid Dirs - construct seed_dir path from scratch
        if hybrid_dir:
            seed_dir = (
                hybrid_dir /
                row['training_method'] /
                row['dataset'] /
                row['model'] /
                f"seed{row['seed']}"
            )
            # Debug: see which seed dirs we are probing
            # print(f"[Oracle Loader][DEBUG] Checking hybrid seed_dir: {seed_dir}, exists: {seed_dir.exists()}")
            
            # Check if seed_dir exists before trying to read files
            real_path = check_path_variants(seed_dir, row['dataset'])
            if real_path:
                seed_dir = real_path
                # Check hybrid oracle files in seed_dir/metric_docs/
                for fname in ["oracle_best_layer.json", "per_layer_ground_truth.json"]:
                    oracle_file = seed_dir / "metric_docs" / fname
                    if oracle_file.exists():
                        try:
                            data = json.loads(oracle_file.read_text())
                            if isinstance(data, dict):
                                oracle_ece = data.get('test_ece')
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
        
        # 2. Check NC Dirs (if not found)
        if oracle_ece is None and nc_dir:
            nc_seed_path = (
                nc_dir /
                row['training_method'] /
                row['dataset'] /
                row['model'] /
                f"seed{row['seed']}"
            )
            # print(f"[Oracle Loader][DEBUG] Checking NC seed_dir: {nc_seed_path}, exists: {nc_seed_path.exists()}")
            # Check if nc_seed_path exists before trying to read files
            real_path = check_path_variants(nc_seed_path, row['dataset'])
            if real_path:
                nc_seed_path = real_path
                # Check in seed_dir/metric_docs/ first
                for oracle_name in ["oracle_best_layer.json", "per_layer_ground_truth.json"]:
                    oracle_file = nc_seed_path / "metric_docs" / oracle_name
                    if oracle_file.exists():
                        try:
                            data = json.loads(oracle_file.read_text())
                            if isinstance(data, dict):
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
                
                # If not found in metric_docs, search recursively
                if oracle_ece is None:
                    for oracle_name in ["oracle_best_layer.json", "per_layer_ground_truth.json"]:
                        for candidate in nc_seed_path.rglob(oracle_name):
                            try:
                                data = json.loads(candidate.read_text())
                                if isinstance(data, dict):
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

        # 3. Check Fallback Dir (if still not found)
        if oracle_ece is None and oracle_fallback_dir is not None:
            fallback_seed_path = (
                oracle_fallback_dir /
                row['training_method'] /
                row['dataset'] /
                row['model'] /
                f"seed{row['seed']}"
            )
            # print(f"[Oracle Loader][DEBUG] Checking fallback seed_dir: {fallback_seed_path}, exists: {fallback_seed_path.exists()}")
            
            # Check if fallback_seed_path exists before trying to read files
            real_path = check_path_variants(fallback_seed_path, row['dataset'])
            if real_path:
                fallback_seed_path = real_path
                # Check in seed_dir/metric_docs/ first
                for oracle_name in ["oracle_best_layer.json", "per_layer_ground_truth.json", 
                                   "multi_composite_analysis.json"]:
                    oracle_file = fallback_seed_path / "metric_docs" / oracle_name
                    if oracle_file.exists():
                        try:
                            data = json.loads(oracle_file.read_text())
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
                
                # If not found in metric_docs, search recursively
                if oracle_ece is None:
                    for oracle_name in ["oracle_best_layer.json", "per_layer_ground_truth.json", 
                                       "multi_composite_analysis.json"]:
                        for candidate in fallback_seed_path.rglob(oracle_name):
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
    
    return pd.DataFrame.from_records(oracle_records).drop_duplicates()


# ============================================================================
# ANALYSIS FUNCTIONS
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
                baseline_summary[baseline_name] = None
        
        if beats_all_baselines and len(baseline_summary) > 0:
            valid_baselines = {k: v for k, v in baseline_summary.items() if v is not None}
            
            if valid_baselines:
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
    
    # *** FIX ***: Use the 'topk' column which is now correctly populated
    for topk_val, group in df.groupby('topk'):
        if pd.isna(topk_val):
            continue
        
        mu, lo, hi = mean_ci(group['ece'].values)
        
        rows.append({
            'Top-K': int(topk_val),
            'Mean ECE': mu,
            '95% CI Lower': lo,
            '95% CI Upper': hi,
            'Std Dev': round(float(np.std(group['ece'].values)), 7),
            'N': int(len(group)),
            'N Datasets': int(group['dataset'].nunique()),
            'N Methods': int(group['full_method'].nunique()),
        })
    
    return pd.DataFrame(rows).sort_values('Mean ECE', ascending=True)


def per_dataset_leaderboard(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create leaderboard for each dataset."""
    leaderboards = {}
    
    for dataset, ds_group in df.groupby('dataset'):
        rows = []
        for method, m_group in ds_group.groupby('full_method'):
            mu, lo, hi = mean_ci(m_group['ece'].values)
            
            rows.append({
                'Method': method,
                'Category': m_group['category'].iloc[0],
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
        
        print(f"  [Oracle Comparison] Analyzing {dataset}...")
        
        for method, ens_group in ens_ds.groupby('full_method'):
            total_method_configs = len(ens_group)
            
            # Ensure k is preserved in merge
            merged = pd.merge(
                ens_group[on_cols + ['ece', 'k']],
                oracle_ds[on_cols + ['oracle_ece']],
                on=on_cols,
                how='inner'
            )
            
            if merged.empty:
                continue
            
            n_paired = int(len(merged))
            
            # Conditional print
            if len(rows) == 0:
                print(f"    (Sample) Method: {method} | Method Configs: {total_method_configs} | "
                      f"Oracle Configs: {len(oracle_ds)} | Paired: {n_paired}")
            
            total = n_paired
            gaps = merged['ece'].values - merged['oracle_ece'].values
            
            # Wins analysis
            better_mask = merged['ece'] < merged['oracle_ece']
            wins = int(better_mask.sum())
            ties = int((merged['ece'] == merged['oracle_ece']).sum())
            
            # Split wins by k
            wins_k1 = int((better_mask & (merged['k'] == 1)).sum())
            wins_k_gt1 = int((better_mask & (merged['k'] > 1)).sum())
            
            within_threshold = int((np.abs(gaps) <= threshold).sum())
            
            rows.append({
                'Method': method,
                'Total Method Configs': total_method_configs,
                'Total Oracle Configs': len(oracle_ds),
                'Paired Comparisons': n_paired,
                'N_paired': n_paired,
                'Mean Gap (paired samples only)': round(float(np.mean(gaps)), 7),
                'Median Gap': round(float(np.median(gaps)), 7),
                'Std Gap': round(float(np.std(gaps)), 7),
                'Better than Oracle': wins,
                'Better (k=1)': wins_k1,
                'Better (k>1)': wins_k_gt1,
                'Worse than Oracle': total - wins - ties,
                f'Within {threshold}': within_threshold,
                f'% Within {threshold}': round(within_threshold / total * 100, 1) if total > 0 else 0.0,
                'Mean ECE (ours)': round(float(np.mean(merged['ece'])), 7),
                'Mean Oracle ECE': round(float(np.mean(merged['oracle_ece'])), 7),
                'N': total,
            })
        
        if rows:
            comparisons[dataset] = (
                pd.DataFrame(rows)
                .sort_values('Mean Gap (paired samples only)', ascending=True)
            )
    
    return comparisons


def analyze_ensemble_vs_oracle(ensemble_df: pd.DataFrame, oracle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance of ENSEMBLE methods (k>1) vs Oracle.
    Returns dataframe sorted by win rate.
    """
    # Filter to k > 1
    ens_k_gt1 = ensemble_df[ensemble_df['k'] > 1].copy()
    
    if ens_k_gt1.empty:
        return pd.DataFrame()
        
    on_cols = ['dataset', 'model', 'seed', 'training_method']
    rows = []
    
    for method, group in ens_k_gt1.groupby('full_method'):
        merged = pd.merge(
            group[on_cols + ['ece', 'k']],
            oracle_df[on_cols + ['oracle_ece']],
            on=on_cols,
            how='inner'
        )
        
        if merged.empty:
            continue
            
        total = len(merged)
        wins_mask = merged['ece'] < merged['oracle_ece']
        wins = int(wins_mask.sum())
        losses = int((merged['ece'] >= merged['oracle_ece']).sum())
        
        # Improvement when winning
        winning_samples = merged[wins_mask]
        if not winning_samples.empty:
            imp = (winning_samples['oracle_ece'] - winning_samples['ece']) / winning_samples['oracle_ece'] * 100
            mean_imp = float(imp.mean())
        else:
            mean_imp = 0.0
            
        # Degradation when losing
        losing_samples = merged[~wins_mask]
        if not losing_samples.empty:
            deg = (losing_samples['ece'] - losing_samples['oracle_ece']) / losing_samples['oracle_ece'] * 100
            mean_deg = float(deg.mean())
        else:
            mean_deg = 0.0
            
        # Overall gap
        gap = merged['ece'] - merged['oracle_ece']
        
        rows.append({
            'method': method,
            'k': int(merged['k'].iloc[0]),
            'total_paired': total,
            'wins': wins,
            'losses': losses,
            'win_rate_pct': (wins / total) * 100,
            'mean_improvement_pct': mean_imp,
            'mean_degradation_pct': mean_deg,
            'overall_mean_gap': float(gap.mean()),
            'mean_ensemble_ece': float(merged['ece'].mean()),
            'mean_oracle_ece': float(merged['oracle_ece'].mean())
        })
        
    return pd.DataFrame(rows).sort_values('win_rate_pct', ascending=False)


def analyze_single_layer_vs_oracle(ensemble_df: pd.DataFrame, oracle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze performance of SINGLE LAYER methods (k=1) vs Oracle.
    Serves as validation - win rates should be near 0.
    """
    # Filter to k = 1
    ens_k1 = ensemble_df[ensemble_df['k'] == 1].copy()
    
    if ens_k1.empty:
        return pd.DataFrame()
        
    on_cols = ['dataset', 'model', 'seed', 'training_method']
    rows = []
    
    for method, group in ens_k1.groupby('full_method'):
        merged = pd.merge(
            group[on_cols + ['ece', 'k']],
            oracle_df[on_cols + ['oracle_ece']],
            on=on_cols,
            how='inner'
        )
        
        if merged.empty:
            continue
            
        total = len(merged)
        wins = int((merged['ece'] < merged['oracle_ece']).sum())
        losses = int((merged['ece'] >= merged['oracle_ece']).sum())
        
        rows.append({
            'method': method,
            'k': 1,
            'total_paired': total,
            'wins': wins,
            'losses': losses,
            'win_rate_pct': (wins / total) * 100,
            'mean_ensemble_ece': float(merged['ece'].mean()),
            'mean_oracle_ece': float(merged['oracle_ece'].mean())
        })
        
    return pd.DataFrame(rows).sort_values('win_rate_pct', ascending=False)



def validate_oracle_comparisons(ensemble_df: pd.DataFrame, oracle_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Validate pairing rates between ensemble methods and oracle data.
    """
    print("\n[Oracle Validation] Running comprehensive pairing analysis...")
    on_cols = ['dataset', 'model', 'seed', 'training_method']
    
    results = []
    
    # Group by method to check coverage
    for method, group in ensemble_df.groupby('full_method'):
        total_configs = len(group)
        
        # Inner merge with oracle to find matches
        merged = pd.merge(
            group[on_cols + ['ece', 'k']],
            oracle_df[on_cols + ['oracle_ece']],
            on=on_cols,
            how='inner'
        )
        
        oracle_matched = len(merged)
        pairing_rate = oracle_matched / total_configs if total_configs > 0 else 0.0
        
        mean_ece_paired = merged['ece'].mean() if not merged.empty else np.nan
        mean_oracle_ece_paired = merged['oracle_ece'].mean() if not merged.empty else np.nan
        
        k_val = group['k'].iloc[0] if 'k' in group.columns else np.nan
        
        coverage_status = "LOW_COVERAGE" if pairing_rate < 0.5 else "OK"
        
        results.append({
            'method': method,
            'total_configs': total_configs,
            'oracle_matched': oracle_matched,
            'pairing_rate': pairing_rate,
            'mean_ece_paired': mean_ece_paired,
            'mean_oracle_ece_paired': mean_oracle_ece_paired,
            'k': k_val,
            'coverage_status': coverage_status
        })
        
    val_df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = output_dir / "oracle_pairing_validation.csv"
    val_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved oracle pairing validation to {output_path}")
    
    # Print warnings
    low_cov = val_df[val_df['coverage_status'] == 'LOW_COVERAGE']
    if not low_cov.empty:
        print(f"  ⚠️  WARNING: {len(low_cov)} methods have < 50% oracle pairing rate!")
        
    return val_df


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
# NEW: Granular and Comparative Analysis
# ============================================================================

def category_performance_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare performance across layer selection categories."""
    rows = []
    
    for category, cat_data in df.groupby('category'):
        mu, lo, hi = mean_ci(cat_data['ece'].values)
        
        rows.append({
            'Category': category,
            'Mean ECE': mu,
            'Std ECE': round(float(np.std(cat_data['ece'].values)), 7),
            '95% CI Lower': lo,
            '95% CI Upper': hi,
            'Median ECE': round(float(np.median(cat_data['ece'].values)), 7),
            'Min ECE': round(float(np.min(cat_data['ece'].values)), 7),
            'N': int(len(cat_data)),
            'N Datasets': int(cat_data['dataset'].nunique()),
        })
    
    return pd.DataFrame(rows).sort_values('Mean ECE')


def nc_vs_hybrid_comparison(ensemble_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Direct comparison between hybrid/old metrics and NC methods."""
    on_cols = ['dataset', 'model', 'seed', 'training_method']
    
    old_df = ensemble_df[ensemble_df['method_source'] == 'hybrid']
    nc_df = ensemble_df[ensemble_df['method_source'] == 'nc_ensemble']

    if old_df.empty or nc_df.empty:
        return pd.DataFrame(), pd.DataFrame() # Not enough data to compare

    # Aggregate to best per method type
    old_best = old_df.loc[old_df.groupby(on_cols)['ece'].idxmin()].copy()
    nc_best = nc_df.loc[nc_df.groupby(on_cols)['ece'].idxmin()].copy()
    
    rows = []
    
    # Compare where both exist
    merged = pd.merge(
        old_best[on_cols + ['ece', 'full_method']],
        nc_best[on_cols + ['ece', 'full_method']],
        on=on_cols,
        how='inner',
        suffixes=('_hybrid', '_nc')
    )
        
    if not merged.empty:
        nc_wins = int((merged['ece_nc'] < merged['ece_hybrid']).sum())
        nc_ties = int((merged['ece_nc'] == merged['ece_hybrid']).sum())
        total = int(len(merged))
        
        improvement = ((merged['ece_hybrid'] - merged['ece_nc']) / merged['ece_hybrid']) * 100
        
        rows.append({
            'Comparison': 'NC vs Hybrid',
            'NC Wins': nc_wins,
            'Hybrid Wins': total - nc_wins - nc_ties,
            'Ties': nc_ties,
            'Total': total,
            'NC Win Rate %': round(nc_wins / total * 100, 2) if total > 0 else 0,
            'Mean Improvement %': round(float(np.mean(improvement[improvement > 0])), 2),
            'Median Improvement %': round(float(np.median(improvement)), 2),
            'Mean ECE (NC)': round(float(merged['ece_nc'].mean()), 7),
            'Mean ECE (Hybrid)': round(float(merged['ece_hybrid'].mean()), 7),
        })
    
    return pd.DataFrame(rows), merged # Return merged_df for plotting


def configuration_win_rates_vs_baselines(ensemble_df: pd.DataFrame, 
                                         baseline_best: pd.DataFrame) -> pd.DataFrame:
    """Calculate win rates for each configuration against best baseline."""
    on_cols = ['dataset', 'model', 'seed', 'training_method']
    
    rows = []
    
    for method_key, method_data in ensemble_df.groupby('full_method'):
        
        merged = pd.merge(
            method_data[on_cols + ['ece', 'category', 'strategy']],
            baseline_best[on_cols + ['ece']],
            on=on_cols,
            how='inner',
            suffixes=('_ens', '_base')
        )
        
        if not merged.empty:
            wins = int((merged['ece_ens'] < merged['ece_base']).sum())
            total = int(len(merged))
            
            improvement = ((merged['ece_base'] - merged['ece_ens']) / merged['ece_base']) * 100
            
            rows.append({
                'Configuration': method_key,
                'Category': merged['category'].iloc[0],
                'Layer Strategy': merged['strategy'].iloc[0],
                'Wins': wins,
                'Total': total,
                'Win Rate %': round(wins / total * 100, 2),
                'Mean Improvement %': round(float(np.mean(improvement)), 2),
                'Mean ECE': round(float(merged['ece_ens'].mean()), 7),
                'Mean Baseline ECE': round(float(merged['ece_base'].mean()), 7),
            })
    
    return pd.DataFrame(rows).sort_values('Win Rate %', ascending=False)


def single_layer_selection_accuracy(ensemble_df: pd.DataFrame, 
                                    hybrid_dir: Path, 
                                    nc_dir: Optional[Path],
                                    oracle_fallback_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Analyze how often PSC/NC methods select the actual oracle best layer.
    
    Returns DataFrame with accuracy metrics per method.
    """
    on_cols = ['dataset', 'model', 'seed', 'training_method']
    
    # Filter for methods that are intended to select a single layer (k=1)
    single_layer_methods = ensemble_df[ensemble_df['k'] == 1].copy()
    
    if single_layer_methods.empty:
        print("  ℹ️  Skipping single_layer_selection_accuracy (no k=1 methods found).")
        return pd.DataFrame()
    
    results = []
    
    for _, row in single_layer_methods.iterrows():
        # Try to locate per-layer oracle data in multiple roots (hybrid, NC, fallback),
        # mirroring the logic in load_oracle.
        candidate_roots: List[Path] = []
        if hybrid_dir is not None:
            candidate_roots.append(hybrid_dir)
        if nc_dir is not None:
            candidate_roots.append(nc_dir)
        if oracle_fallback_dir is not None:
            candidate_roots.append(oracle_fallback_dir)

        oracle_file = None
        for base in candidate_roots:
            seed_dir = (
                base / row['training_method'] / row['dataset'] /
                row['model'] / f"seed{row['seed']}"
            )
            if not seed_dir.exists():
                continue

            # First, try the standard metric_docs path
            candidate = seed_dir / "metric_docs" / "per_layer_ground_truth.json"
            if candidate.exists():
                oracle_file = candidate
                break

            # If not present there (e.g., in baseline analysis dirs), search recursively
            try:
                for c in seed_dir.rglob("per_layer_ground_truth.json"):
                    oracle_file = c
                    break
            except Exception:
                oracle_file = None

            if oracle_file is not None:
                break

        if oracle_file is None:
            continue
        
        try:
            # Load per-layer ECEs
            data = json.loads(oracle_file.read_text())
            
            # Get ECE values per layer
            layer_eces = {}
            if isinstance(data, dict):
                # Try different possible structures
                if 'test' in data and isinstance(data['test'], dict) and 'ece' in data['test']:
                    layer_eces = data['test']['ece']
                elif 'layers' in data:
                    for layer_info in data['layers']:
                        if isinstance(layer_info, dict) and 'layer' in layer_info and 'test_ece' in layer_info:
                            layer_eces[layer_info['layer']] = layer_info['test_ece']
            elif isinstance(data, list):
                layer_eces = {r['layer']: r['test_ece'] for r in data if 'test_ece' in r and 'layer' in r}
            
            if not layer_eces:
                continue
            
            # Sort layers by ECE (ascending) and get the top 3
            sorted_oracle_layers = sorted(layer_eces, key=layer_eces.get)
            oracle_best_layer = sorted_oracle_layers[0] if sorted_oracle_layers else None
            oracle_top3_layers = sorted_oracle_layers[:3]

            if oracle_best_layer is None:
                continue
            
            # Get PSC's selected layers
            selected_layers = row.get('selected_layers', [])
            if not selected_layers:
                continue
            
            # Get the single layer this method selected
            selected_layer = selected_layers[0]

            # CAST TO STRING for robust comparison (int vs str)
            sel_str = str(selected_layer)
            best_str = str(oracle_best_layer)
            top3_strs = [str(x) for x in oracle_top3_layers]

            # Check hits
            hit_as_top1 = (sel_str == best_str)
            hit_in_top3 = (sel_str in top3_strs)

            # Lightweight debug for a few rows to inspect structure
            if len(results) < 3:
                print("[Single-Layer Accuracy][DEBUG] Sample row:")
                print(f"  method: {row['full_method']}")
                print(f"  selected_layers type: {type(selected_layers)}")
                print(f"  selected_layers value: {selected_layers}")
                print(f"  oracle_best_layer: {oracle_best_layer}")
            
            results.append({
                'method': row['full_method'],
                'dataset': row['dataset'],
                'model': row['model'],
                'seed': row['seed'],
                'oracle_best_layer': oracle_best_layer,
                'selected_layers': selected_layers,
                'hit_in_top3': hit_in_top3,
                'hit_as_top1': hit_as_top1,
                'oracle_ece': layer_eces[oracle_best_layer],
                'method_ece': row['ece']
            })
            
        except Exception as e:
            continue
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Aggregate by method
    summary = df.groupby('method').agg({
        'hit_in_top3': 'mean',
        'hit_as_top1': 'mean',
        'oracle_ece': 'mean',
        'method_ece': 'mean',
        'seed': 'count'
    }).rename(columns={
        'hit_in_top3': 'Top-3 Hit Rate',
        'hit_as_top1': 'Top-1 Hit Rate',
        'seed': 'N',
        'oracle_ece': 'Mean Oracle ECE',
        'method_ece': 'Mean Method ECE'
    })
    
    summary['Top-3 Hit Rate'] = summary['Top-3 Hit Rate'] * 100
    summary['Top-1 Hit Rate'] = summary['Top-1 Hit Rate'] * 100
    
    return summary.reset_index().sort_values('Top-3 Hit Rate', ascending=False)

# ============================================================================
# NEW: Visualization Functions
# ============================================================================

def plot_category_boxplots(ensemble_df: pd.DataFrame, output_dir: Path):
    """Boxplots showing ECE distribution by category."""
    try:
        plt.figure(figsize=(12, 8))
        
        # Sort categories by median ECE
        order = ensemble_df.groupby('category')['ece'].median().sort_values().index
        
        sns.boxplot(data=ensemble_df, x='ece', y='category', order=order)
        plt.title('ECE Distribution by Method Category', fontsize=16)
        plt.xlabel('Expected Calibration Error (ECE)', fontsize=12)
        plt.ylabel('Category', fontsize=12)
        plt.xscale('log') # ECE is often better viewed on a log scale
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        
        plot_path = output_dir / "category_performance_boxplot.png"
        plt.savefig(plot_path, dpi=300)
        print(f"  ✓ Saved category boxplot to {plot_path}")
        plt.close()
    
    except Exception as e:
        print(f"[WARN] Failed to generate category boxplot: {e}")


def plot_nc_vs_hybrid_scatter(merged_df: pd.DataFrame, output_dir: Path):
    """Scatter plot: Hybrid ECE vs NC ECE."""
    if merged_df.empty:
        print("  ℹ️  Skipping NC vs Hybrid scatter plot (no overlapping data).")
        return
        
    try:
        plt.figure(figsize=(10, 10))
        
        sns.scatterplot(data=merged_df, x='ece_hybrid', y='ece_nc', s=50, alpha=0.7)
        
        # Add diagonal line
        min_val = min(merged_df['ece_hybrid'].min(), merged_df['ece_nc'].min()) * 0.9
        max_val = max(merged_df['ece_hybrid'].max(), merged_df['ece_nc'].max()) * 1.1
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Equal Performance')
        
        plt.title('NC Ensemble (Best) vs. Hybrid Ensemble (Best)', fontsize=16)
        plt.xlabel('Hybrid Ensemble ECE (Lower is better)', fontsize=12)
        plt.ylabel('NC Ensemble ECE (Lower is better)', fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        
        plot_path = output_dir / "nc_vs_hybrid_scatter.png"
        plt.savefig(plot_path, dpi=300)
        print(f"  ✓ Saved NC vs Hybrid scatter plot to {plot_path}")
        plt.close()

    except Exception as e:
        print(f"[WARN] Failed to generate NC vs Hybrid scatter plot: {e}")


# ============================================================================
# NEW: SOTA Competitor Comparison Module (Model-Specific)
# ============================================================================

# Best reported ECE values extracted directly from the provided PDFs.
# Keys are (dataset_name, model_name).
# Values are normalized to [0, 1] range.
SOTA_DATA_MODEL_SPECIFIC = [
    # --- CIFAR-10-C (Corrupted) ---
    {
        'dataset': 'cifar10c', 'model': 'resnet18',
        'Paper': 'DAC (Tomani et al. 2023)', 'Method': 'TS+DAC', 'ECE': 0.0439
    },
    {
        'dataset': 'cifar10c', 'model': 'densenet121',
        'Paper': 'DAC (Tomani et al. 2023)', 'Method': 'SPL+DAC', 'ECE': 0.0442
    },
    # --- CIFAR-100-C (Corrupted) ---
    {
        'dataset': 'cifar100c', 'model': 'resnet18',
        'Paper': 'DAC (Tomani et al. 2023)', 'Method': 'SPL+DAC', 'ECE': 0.0847
    },
    {
        'dataset': 'cifar100c', 'model': 'densenet121',
        'Paper': 'DAC (Tomani et al. 2023)', 'Method': 'ETS+DAC', 'ECE': 0.0840
    },
    # --- CIFAR-10 (Clean) ---
    {
        'dataset': 'cifar10', 'model': 'resnet50',
        'Paper': 'SNGP (Liu et al. 2020)', 'Method': 'SNGP (via TULIP)', 'ECE': 0.0100
    },
    {
        'dataset': 'cifar10', 'model': 'resnet50',
        'Paper': 'TULIP (Benkert et al. 2024)', 'Method': 'TULIP', 'ECE': 0.0300
    },
    # --- CIFAR-100 (Clean) ---
    {
        'dataset': 'cifar100', 'model': 'resnet50',
        'Paper': 'TULIP (Benkert et al. 2024)', 'Method': 'TULIP', 'ECE': 0.658
    },
    {
        'dataset': 'cifar100', 'model': 'resnet50',
        'Paper': 'SNGP (Liu et al. 2020)', 'Method': 'SNGP (via TULIP)', 'ECE': 0.720
    },
    # --- ImageNet (Corrupted) ---
    {
        'dataset': 'imagenet-c', 'model': 'resnet152',
        'Paper': 'DAC (Tomani et al. 2023)', 'Method': 'TS+DAC', 'ECE': 0.0348
    },
    {
        'dataset': 'imagenet-c', 'model': 'densenet169',
        'Paper': 'DAC (Tomani et al. 2023)', 'Method': 'ETS+DAC', 'ECE': 0.0387
    },
]

def normalize_model_name(name: str) -> str:
    """Normalize model names for matching (e.g., 'ResNet18' -> 'resnet18')."""
    return name.lower().replace('-', '').replace('_', '')

def generate_sota_comparison_table_v2(
    ensemble_df: pd.DataFrame,
    output_dir: Path,
    allowed_training_methods: Optional[Set[str]] = None,
):
    """
    Generates comparison using per-model leaderboards and optionally filters to
    specific training losses (e.g., cross-entropy only).
    """
    print("\n[SOTA Comparison] Generating model-specific SOTA competitor analysis...")

    if allowed_training_methods:
        filtered_df = ensemble_df[
            ensemble_df["training_method"].isin(allowed_training_methods)
        ].copy()
        if filtered_df.empty:
            print(
                f"  [WARN] Skipping SOTA comparison; no runs match training methods: "
                f"{sorted(allowed_training_methods)}"
            )
            return
    else:
        filtered_df = ensemble_df

    per_model_lbs = per_model_leaderboard(filtered_df)
    if not per_model_lbs:
        print("  [WARN] Skipping SOTA comparison; no per-model leaderboards available.")
        return

    rows = []

    for (dataset, model), lb in per_model_lbs.items():
        if lb.empty:
            continue
            
        # Normalize names
        ds_norm = dataset.lower().replace('_', '-').replace('cifar10c', 'cifar10-c').replace('cifar100c', 'cifar100-c')
        mod_norm = normalize_model_name(model)
        
        # Find matching SOTA competitors
        competitors = [entry for entry in SOTA_DATA_MODEL_SPECIFIC 
                      if entry['dataset'] == ds_norm and entry['model'] == mod_norm]
        
        if not competitors:
            continue
            
        # Get Your Best Result for this specific model
        best_ours = lb.iloc[0]
        our_ece = best_ours['Mean ECE']
        seeds_used = int(best_ours.get('N', 0))
        
        # Clean up method name
        our_method_name = str(best_ours['Method']).replace('Hybrid_', '').replace('NC_', '').replace('single_', '')
        if len(our_method_name) > 25:
            our_method_name = our_method_name[:23] + "..."
        
        for comp in competitors:
            sota_ece = comp['ECE']
            improvement_pct = ((sota_ece - our_ece) / sota_ece) * 100.0
            
            status = "WIN" if improvement_pct > 0 else "LOSE"
            if abs(improvement_pct) < 5.0: status = "TIE"
            if improvement_pct > 20.0: status = "**WIN**"
            
            rows.append({
                'Dataset': dataset.upper(),
                'Model': model,
                'Seeds': seeds_used,
                'Our Method': our_method_name,
                'Our ECE': our_ece,
                'Competitor': comp['Paper'],
                'SOTA Method': comp['Method'],
                'SOTA ECE': sota_ece,
                'Imp. (%)': improvement_pct,
                'Result': status
            })
            
    if not rows:
        print("[WARN] No matching (Dataset, Model) pairs found in SOTA database.")
        return

    df = pd.DataFrame(rows)
    
    # Sort by Dataset then Model
    df.sort_values(by=['Dataset', 'Model'], inplace=True)
    
    # Save CSV
    csv_path = output_dir / "sota_model_specific_comparison.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  ✓ Saved Model-Specific SOTA comparison CSV to {csv_path}")
    
    # Generate LaTeX
    latex_df = df.copy()
    latex_df['Our ECE'] = latex_df['Our ECE'].apply(lambda x: f"{x:.4f}")
    latex_df['SOTA ECE'] = latex_df['SOTA ECE'].apply(lambda x: f"{x:.4f}")
    latex_df['Imp. (%)'] = latex_df['Imp. (%)'].apply(lambda x: f"{x:+.1f}\\%")
    
    def bold_winner(row):
        our = float(row['Our ECE'])
        sota = float(row['SOTA ECE'])
        if our < sota:
            row['Our ECE'] = f"\\textbf{{{row['Our ECE']}}}"
        elif sota < our:
            row['SOTA ECE'] = f"\\textbf{{{row['SOTA ECE']}}}"
        return row
        
    latex_df = latex_df.apply(bold_winner, axis=1)
    
    # Select columns
    display_cols = ['Dataset', 'Model', 'Seeds', 'Our Method', 'Our ECE', 'SOTA Method', 'SOTA ECE', 'Imp. (%)', 'Competitor']
    latex_df = latex_df[display_cols]
    
    latex_str = latex_df.to_latex(
        index=False,
        caption="Direct Comparison vs SOTA on Matching Architectures",
        label="tab:sota_model_spec",
        column_format="l l c l c l c r l",
        position="htbp",
        escape=False
    )
    
    tex_path = output_dir / "latex_tables" / "sota_model_specific.tex"
    tex_path.parent.mkdir(exist_ok=True)
    
    with open(tex_path, 'w') as f:
        f.write("% Required packages: booktabs\n")
        f.write(latex_str.replace('\\toprule', '\\toprule\n').replace('\\midrule', '\\midrule\n').replace('\\bottomrule', '\\bottomrule\n'))
        
    print(f"  ✓ Saved Model-Specific SOTA LaTeX to {tex_path}")
    
    # Print Summary
    print("\n--- SOTA MODEL-SPECIFIC SUMMARY ---")
    for _, row in df.iterrows():
        print(f"{row['Dataset']} ({row['Model']}, {int(row['Seeds'])} seeds): "
              f"Ours ({row['Our ECE']:.4f}) vs {row['SOTA Method']} ({row['SOTA ECE']:.4f}) "
              f"-> {row['Result']} ({row['Imp. (%)']:.1f}%)")


# ============================================================================
# LaTeX Export Helpers
# ============================================================================

def export_table_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    filename: Path,
    float_format: str = '%.4f',
    footnote: Optional[str] = None
) -> None:
    """Export DataFrame to LaTeX table with professional formatting."""
    if df is None or df.empty:
        print(f"[WARN] Skipping LaTeX export for {filename.name} (empty DataFrame).")
        return

    latex_str = df.to_latex(
        index=False,
        float_format=float_format,
        caption=caption,
        label=label,
        position='htbp',
        column_format='l' + 'r' * (len(df.columns) - 1),
        escape=False,
        bold_rows=False,
    )

    # Add booktabs for professional appearance (to be used with \usepackage{booktabs})
    latex_str = latex_str.replace('\\toprule', '\\toprule\n')
    latex_str = latex_str.replace('\\midrule', '\\midrule\n')
    latex_str = latex_str.replace('\\bottomrule', '\\bottomrule\n')
    
    if footnote:
        # Insert footnote before end of table environment
        latex_str = latex_str.replace('\\end{table}', f'\\vspace{{0.5em}}\n\\footnotesize\n{footnote}\n\\end{{table}}')

    preamble = (
        "% Required packages:\n"
        "% \\usepackage{booktabs}\n"
        "% \\usepackage{multirow}\n"
        "% \\usepackage{array}\n\n"
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        f.write(preamble + latex_str)

    print(f"  ✓ Exported LaTeX table: {filename.name}")


# ============================================================================
# Enhanced Report Generation
# ============================================================================

def systematic_metric_analysis(ensemble_df: pd.DataFrame, oracle_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Systematically analyze each base metric across:
    1. Different k values (top1, top2, top3, etc.)
    2. For best k, compare weighting methods and alphas
    
    Returns dict of dataframes with results per base metric.
    """
    
    # Extract base metric name (before _top or depth suffix)
    def extract_base_metric(full_method: str) -> str:
        # Remove prefixes
        method = full_method.replace('Hybrid_single_', '').replace('NC_', '')
        
        # Remove _topX suffix
        if '_top' in method:
            method = method.split('_top')[0]
        
        # Remove weighting suffixes (_deep, _shallow, _learned, _uniform)
        for suffix in ['_deep_a', '_shallow_a', '_learned', '_uniform']:
            if suffix in method:
                method = method.split(suffix)[0]
                break
        
        return method
    
    # Extract k value
    def extract_k(row):
        if pd.notna(row.get('k')):
            return int(row['k'])
        # Fallback: parse from method name
        full = row['full_method']
        if '_top' in full:
            match = re.search(r'_top(\d+)_', full)
            if match:
                return int(match.group(1))
        return 1  # Default to k=1
    
    # Extract weighting method
    def extract_weighting(full_method: str) -> str:
        if '_learned' in full_method:
            return 'learned'
        elif '_uniform' in full_method:
            return 'uniform'
        elif '_deep_a' in full_method:
            # Extract alpha
            match = re.search(r'_deep_a([\d.]+)', full_method)
            alpha = match.group(1) if match else 'unknown'
            return f'deep_a{alpha}'
        elif '_shallow_a' in full_method:
            match = re.search(r'_shallow_a([\d.]+)', full_method)
            alpha = match.group(1) if match else 'unknown'
            return f'shallow_a{alpha}'
        return 'none'
    
    # Add extracted columns
    ensemble_df = ensemble_df.copy()
    ensemble_df['base_metric'] = ensemble_df['full_method'].apply(extract_base_metric)
    ensemble_df['k_val'] = ensemble_df.apply(extract_k, axis=1)
    ensemble_df['weighting_method'] = ensemble_df['full_method'].apply(extract_weighting)
    
    # Merge with oracle for gap analysis
    on_cols = ['dataset', 'model', 'seed', 'training_method']
    merged = pd.merge(
        ensemble_df[on_cols + ['ece', 'base_metric', 'k_val', 'weighting_method', 'full_method']],
        oracle_df[on_cols + ['oracle_ece']],
        on=on_cols,
        how='inner'
    )
    
    results = {}
    
    # For each base metric
    for base_metric in sorted(merged['base_metric'].unique()):
        metric_data = merged[merged['base_metric'] == base_metric]
        
        if len(metric_data) < 10:  # Skip metrics with too few samples
            continue
        
        # STEP 1: Compare k values
        k_comparison = []
        for k_val in sorted(metric_data['k_val'].unique()):
            k_data = metric_data[metric_data['k_val'] == k_val]
            
            # Best method for this k
            best_method = k_data.loc[k_data['ece'].idxmin()]
            
            # Aggregate stats
            mean_ece = k_data['ece'].mean()
            std_ece = k_data['ece'].std()
            mean_gap = (k_data['ece'] - k_data['oracle_ece']).mean()
            n_samples = len(k_data)
            n_beat_oracle = (k_data['ece'] < k_data['oracle_ece']).sum()
            
            k_comparison.append({
                'k': k_val,
                'mean_ece': mean_ece,
                'std_ece': std_ece,
                'mean_gap_to_oracle': mean_gap,
                'n_samples': n_samples,
                'n_beat_oracle': n_beat_oracle,
                'oracle_win_rate': n_beat_oracle / n_samples * 100,
                'best_method_name': best_method['full_method'],
                'best_method_ece': best_method['ece']
            })
        
        k_df = pd.DataFrame(k_comparison).sort_values('mean_ece')
        
        # STEP 2: For best k>1, compare weighting methods
        # Filter to only k>1 for weighting comparison (k=1 is irrelevant for weighting)
        k_gt1_data = metric_data[metric_data['k_val'] > 1]
        
        if not k_gt1_data.empty:
            # Find best k among k>1 values (smallest mean_ece)
            k_gt1_df = k_df[k_df['k'] > 1]
            if not k_gt1_df.empty:
                best_k = k_gt1_df.iloc[0]['k']  # Already sorted by mean_ece, so first is best
                best_k_data = metric_data[metric_data['k_val'] == best_k]
            else:
                # Fallback: use smallest k>1 available
                best_k = int(k_gt1_data['k_val'].min())
                best_k_data = metric_data[metric_data['k_val'] == best_k]
        else:
            # No k>1 data available, skip weighting comparison
            best_k = None
            best_k_data = pd.DataFrame()
        
        weighting_comparison = []
        if not best_k_data.empty:
            for weighting in sorted(best_k_data['weighting_method'].unique()):
                w_data = best_k_data[best_k_data['weighting_method'] == weighting]
                
                mean_ece = w_data['ece'].mean()
                std_ece = w_data['ece'].std()
                mean_gap = (w_data['ece'] - w_data['oracle_ece']).mean()
                n_samples = len(w_data)
                
                weighting_comparison.append({
                    'weighting': weighting,
                    'mean_ece': mean_ece,
                    'std_ece': std_ece,
                    'mean_gap_to_oracle': mean_gap,
                    'n_samples': n_samples,
                })
        
        # Create DataFrame - handle empty case
        if weighting_comparison:
            w_df = pd.DataFrame(weighting_comparison).sort_values('mean_ece')
        else:
            # Create empty DataFrame with proper columns
            w_df = pd.DataFrame(columns=['weighting', 'mean_ece', 'std_ece', 'mean_gap_to_oracle', 'n_samples'])
        
        # Store results
        # For best_overall, use best k>1 if available, otherwise use overall best
        if best_k is not None:
            best_overall_method = k_df[k_df['k'] == best_k].iloc[0]['best_method_name']
            best_overall_ece = k_df[k_df['k'] == best_k].iloc[0]['mean_ece']
        else:
            # Fallback to overall best (might be k=1)
            best_overall_method = k_df.iloc[0]['best_method_name']
            best_overall_ece = k_df.iloc[0]['mean_ece']
            best_k = k_df.iloc[0]['k']  # Set best_k for consistency
        
        results[base_metric] = {
            'k_comparison': k_df,
            'weighting_comparison': w_df,
            'best_k': best_k,
            'best_overall_method': best_overall_method,
            'best_overall_ece': best_overall_ece
        }
    
    return results


def create_ensemble_oracle_table(
    ensemble_vs_oracle_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Create LaTeX table for Top 10 Ensemble Methods vs Oracle."""
    if ensemble_vs_oracle_df.empty:
        return
        
    top10 = ensemble_vs_oracle_df.head(10).copy()
    
    table_data = []
    for _, row in top10.iterrows():
        method_clean = row['method'].replace('Hybrid_', '').replace('NC_', '')
        table_data.append({
            'Method': method_clean,
            'k': row['k'],
            'Win Rate': f"{row['win_rate_pct']:.1f}\\%",
            'Wins/Total': f"{row['wins']}/{row['total_paired']}",
            'Mean $\\Delta$ ECE': f"{row['overall_mean_gap']:.4f}",
            'Mean ECE': f"{row['mean_ensemble_ece']:.4f}"
        })
        
    df = pd.DataFrame(table_data)
    
    export_table_to_latex(
        df,
        caption="Top 10 Ensemble Methods (k$>$1) vs Oracle Best Single Layer",
        label="tab:ensemble_vs_oracle",
        filename=output_dir / "ensemble_vs_oracle.tex",
        float_format="%.4f",
        footnote="Sorted by Win Rate against Oracle. $\\Delta$ ECE = Ensemble ECE - Oracle ECE (negative is better)."
    )


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
    category_perf: pd.DataFrame,
    nc_vs_hybrid: pd.DataFrame,
    config_wins: pd.DataFrame,
    single_layer_accuracy: pd.DataFrame,
    oracle_validation_df: pd.DataFrame,
    ensemble_vs_oracle_df: pd.DataFrame, # NEW
    single_layer_vs_oracle_df: pd.DataFrame, # NEW
    threshold: float
) -> List[str]:
    """Generate comprehensive enhanced text report."""
    lines = []
    
    lines.append("="*88)
    lines.append("PUBLICATION-READY MASTER ANALYSIS REPORT")
    lines.append(f"(Hybrid, NC, Multi-Layer, Post-Hoc, Oracle)")
    lines.append("="*88)
    lines.append("")
    lines.append(f"TOTAL ENSEMBLE CONFIGS LOADED: {len(ensemble_df)}")
    lines.append(f"  - Hybrid: {(ensemble_df['method_source'] == 'hybrid').sum()}")
    lines.append(f"  - NC:     {(ensemble_df['method_source'] == 'nc_ensemble').sum()}")
    lines.append("")
    lines.append(f"TOTAL BASELINE CONFIGS LOADED: {len(baseline_df)}")
    lines.append(f"  - Multi-Layer: {(baseline_df['method_source'] == 'multi_layer').sum()}")
    lines.append(f"  - Post-Hoc:    {(baseline_df['method_source'] == 'post_hoc').sum()}")
    lines.append("")
    lines.append(f"TOTAL ORACLE RECORDS LOADED: {len(oracle_df)}")
    lines.append("")
    
    # ========================================================================
    # Category performance
    # ========================================================================
    lines.append("🎯 LAYER SELECTION CATEGORY PERFORMANCE")
    lines.append("="*88)
    
    if not category_perf.empty:
        lines.append("\nRanked by mean ECE (lower is better):\n")
        for _, row in category_perf.iterrows():
            lines.append(f"  {row['Category']:25s}: {row['Mean ECE']:.4f} ± {row['Std ECE']:.4f} "
                         f"(N={row['N']}, {row['N Datasets']} datasets)")
        
        best_cat = category_perf.iloc[0]
        lines.append(f"\n💡 Best category: {best_cat['Category']} "
                     f"(ECE={best_cat['Mean ECE']:.4f})")
    else:
        lines.append("No category performance data available.")
    
    lines.append("")
    
    # ========================================================================
    # NC vs Hybrid comparison
    # ========================================================================
    lines.append("\n⚖️  NC ENSEMBLE vs HYBRID ENSEMBLE COMPARISON")
    lines.append("=" * 88)
    
    if not nc_vs_hybrid.empty:
        row = nc_vs_hybrid.iloc[0]
        lines.append(f"\nDirect head-to-head (best config from each method type):")
        lines.append(f"  NC wins: {row['NC Wins']}/{row['Total']} ({row['NC Win Rate %']:.1f}%)")
        lines.append(f"  Hybrid wins: {row['Hybrid Wins']}/{row['Total']}")
        lines.append(f"  Ties: {row['Ties']}")
        
        lines.append(f"\n  Mean improvement when NC wins: {row['Mean Improvement %']:.2f}%")
        lines.append(f"  Median improvement (all): {row['Median Improvement %']:.2f}%")
        lines.append(f"\n  Mean ECE comparison:")
        lines.append(f"    NC Ensemble: {row['Mean ECE (NC)']:.4f}")
        lines.append(f"    Hybrid Ensemble: {row['Mean ECE (Hybrid)']:.4f}")
            
        if row['NC Win Rate %'] > 50:
            lines.append(f"\n💡 NC methods win in {row['NC Win Rate %']:.0f}% of cases!")
        else:
            lines.append(f"\n💡 Hybrid methods win in {100 - row['NC Win Rate %']:.0f}% of cases.")
    else:
        lines.append("Not enough data for a direct NC vs Hybrid comparison.")
        
    lines.append("")

    # ========================================================================
    # NEW: Single-layer (k=1) selection accuracy
    # ========================================================================
    if not single_layer_accuracy.empty:
        lines.append("\n🎯 SINGLE-LAYER SELECTION ACCURACY")
        lines.append("=" * 88)
        lines.append("\nHow often do k=1 methods select near-oracle layers (Top-1 / Top-3)?\n")
        
        for _, row in single_layer_accuracy.head(10).iterrows():
            lines.append(f"  {row['method']:50s}")
            lines.append(f"    Top-3 Hit Rate:       {row['Top-3 Hit Rate']:.1f}%")
            lines.append(f"    Top-1 Hit Rate:       {row['Top-1 Hit Rate']:.1f}%")
            lines.append(f"    N: {int(row['N'])}\n")
    else:
        lines.append("\n🎯 SINGLE-LAYER SELECTION ACCURACY")
        lines.append("=" * 88)
        lines.append("\nNo single-layer selection accuracy data available.\n")

    lines.append("")

    # ========================================================================
    # NEW: Granular Win Rates
    # ========================================================================
    lines.append("\n📈 TOP 10 CONFIGURATIONS by WIN-RATE vs. BEST BASELINE")
    lines.append("="*88)
    
    if not config_wins.empty:
        lines.append("\nTop 10 configurations that most consistently beat the *best* baseline:\n")
        for i, row in config_wins.head(10).iterrows():
            lines.append(f"  {i+1}. {row['Configuration']}")
            lines.append(f"     Category: {row['Category']}")
            lines.append(f"     Win Rate: {row['Win Rate %']:.1f}% ({row['Wins']}/{row['Total']})")
            lines.append(f"     Mean ΔECE: {row['Mean Improvement %']:.2f}%")
            lines.append(f"     Mean ECE: {row['Mean ECE']:.4f}\n")
    else:
        lines.append("No configuration win-rate data available.")

    lines.append("")

    # ========================================================================
    # Baseline dominance section
    # ========================================================================
    lines.append("\n🏆 METHODS THAT BEAT ALL BASELINES ACROSS ALL DATASETS")
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
    # Top-K analysis
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
    # Per-dataset winners
    # ========================================================================
    lines.append("BEST METHOD PER DATASET:")
    lines.append("-" * 88)
    for dataset in sorted(per_ds_leaderboards.keys()):
        lb = per_ds_leaderboards[dataset]
        if not lb.empty:
            best = lb.iloc[0]
            lines.append(f"\n{dataset.upper()}:")
            lines.append(f"  Winner: {best['Method']}")
            lines.append(f"  Category: {best['Category']}")
            lines.append(f"  ECE: {best['Mean ECE']:.4f} ± {best['Std Dev']:.4f} (N={best['N']})")
            
            if len(lb) > 1:
                lines.append(f"  Top 3:")
                for i, row in lb.head(3).iterrows():
                    lines.append(f"    {i+1}. {row['Method']:50s} (Cat: {row['Category']}) ECE={row['Mean ECE']:.4f}")
    
    lines.append("")
    
    # ========================================================================
    # Model-specific analysis
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
    # Training method analysis
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
    # Per-dataset vs baselines
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
                    f"    {status} vs {row['Baseline']:35s} | "
                    f"{win_pct:5.1f}% ({row['Wins']}/{row['N']}) | "
                    f"ΔECE={delta:+.4f}"
                )
    
    lines.append("")
    
    # ========================================================================
    # ORACLE COMPARISON VALIDATION
    # ========================================================================
    if not oracle_validation_df.empty:
        lines.append("\n🔍 ORACLE COMPARISON VALIDATION")
        lines.append("="*88)
        
        # Global stats
        total_configs = oracle_validation_df['total_configs'].sum()
        total_matched = oracle_validation_df['oracle_matched'].sum()
        global_rate = (total_matched / total_configs * 100) if total_configs > 0 else 0
        
        lines.append(f"\nGlobal Pairing Statistics:")
        lines.append(f"  Total Method Configs: {total_configs}")
        lines.append(f"  Matched with Oracle: {total_matched}")
        lines.append(f"  Global Pairing Rate: {global_rate:.1f}%")
        
        # Low coverage warning
        low_cov = oracle_validation_df[oracle_validation_df['coverage_status'] == 'LOW_COVERAGE']
        if not low_cov.empty:
            lines.append(f"\n  ⚠️ WARNING: {len(low_cov)} methods have < 50% pairing rate.")
            low_cov = low_cov.sort_values('pairing_rate')
            for _, row in low_cov.head(5).iterrows():
                lines.append(f"    - {row['method']}: {row['pairing_rate']*100:.1f}% ({row['oracle_matched']}/{row['total_configs']})")
            if len(low_cov) > 5:
                lines.append(f"    ... and {len(low_cov)-5} more.")
        else:
            lines.append("\n  ✓ All methods have > 50% pairing rate.")
            
        lines.append("")

    # ========================================================================
    # ENSEMBLE vs ORACLE (k>1)
    # ========================================================================
    if not ensemble_vs_oracle_df.empty:
        lines.append("\n🎯 ENSEMBLE METHODS vs ORACLE (k>1 only)")
        lines.append("="*88)
        
        total_paired = ensemble_vs_oracle_df['total_paired'].sum()
        total_wins = ensemble_vs_oracle_df['wins'].sum()
        if total_paired > 0:
            agg_win_rate = (total_wins / total_paired) * 100
            mean_imp = ensemble_vs_oracle_df[ensemble_vs_oracle_df['mean_improvement_pct'] > 0]['mean_improvement_pct'].mean()
        else:
            agg_win_rate = 0
            mean_imp = 0
            
        lines.append(f"\nOverall Statistics (Aggregated across methods):")
        lines.append(f"  Total Ensemble Configs with Oracle: {total_paired}")
        lines.append(f"  Ensemble Beats Oracle: {total_wins} ({agg_win_rate:.1f}%)")
        lines.append(f"  Mean Improvement (when winning): +{mean_imp:.2f}%")
        
        lines.append(f"\nTop 10 Ensemble Methods by Win Rate against Oracle:")
        for i, row in ensemble_vs_oracle_df.head(10).iterrows():
            lines.append(f"  {i+1}. {row['method']} (k={row['k']})")
            lines.append(f"       Win Rate: {row['win_rate_pct']:.1f}% ({row['wins']}/{row['total_paired']})")
            lines.append(f"       Mean Improvement: +{row['mean_improvement_pct']:.1f}%")
            lines.append(f"       Mean ECE: {row['mean_ensemble_ece']:.4f} vs Oracle: {row['mean_oracle_ece']:.4f}")
            lines.append("")
            
        if not single_layer_vs_oracle_df.empty:
            val_win_rate = single_layer_vs_oracle_df['win_rate_pct'].mean()
            lines.append(f"  ✓ Validation: k=1 methods beat oracle in {val_win_rate:.1f}% of cases (should be ~0%)")
            if val_win_rate > 5.0:
                lines.append(f"  ⚠️  WARNING: k=1 win rate > 5% indicates potential data mismatch or noise.")
        
        lines.append("")

    # ========================================================================
    # Oracle comparison
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
                lines.append(f"    Mean Gap (paired): {row['Mean Gap (paired samples only)']:+.4f}")
                lines.append(f"    N Paired: {int(row['N_paired'])}")
                lines.append(f"    Median Gap: {row['Median Gap']:+.4f}")
                lines.append(f"    Within {threshold}: {row[f'% Within {threshold}']:.1f}% ({row[f'Within {threshold}']}/{row['N']})")
                lines.append(f"    Better than Oracle: {row['Better than Oracle']}/{row['N']}")
                
                if row['Better than Oracle'] > row['N'] / 2:
                    lines.append(f"    🏆 BEATS ORACLE in majority of cases!")
    
    lines.append("")
    
    # ========================================================================
    # Strategy insights
    # ========================================================================
    lines.append("\nSTRATEGY INSIGHTS:")
    lines.append("-" * 88)
    
    strategy_wins = {}
    for dataset, lb in per_ds_leaderboards.items():
        if not lb.empty:
            winner = lb.iloc[0]['Method']
            # Find the original strategy string, not the full_method
            strategy = ensemble_df[ensemble_df['full_method'] == winner]['strategy'].iloc[0]
            if pd.notna(strategy):
                strategy_wins[strategy] = strategy_wins.get(strategy, 0) + 1
    
    lines.append("Dataset wins by strategy:")
    for strategy, count in sorted(strategy_wins.items(), key=lambda x: -x[1]):
        total_datasets = len(per_ds_leaderboards)
        lines.append(f"  {strategy:30s}: {count}/{total_datasets} datasets")
    
    lines.append("")
    
    return lines


# ============================================================================
# LaTeX Table Builders
# ============================================================================

def create_executive_summary_table(
    ensemble_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    category_perf: pd.DataFrame,
    nc_vs_hybrid: pd.DataFrame,
    dominators: pd.DataFrame,
) -> pd.DataFrame:
    """Create a high-level summary table of all experiments."""

    best_cat = category_perf.iloc[0] if not category_perf.empty else None
    best_cat_name = best_cat['Category'] if best_cat is not None else 'N/A'
    best_cat_ece = f"{best_cat['Mean ECE']:.4f}" if best_cat is not None else 'N/A'

    hybrid_win_rate = 'N/A'
    if not nc_vs_hybrid.empty:
        row = nc_vs_hybrid.iloc[0]
        hybrid_win_rate = f"{100 - row['NC Win Rate %']:.1f}%"

    num_dominators = len(dominators) if dominators is not None and not dominators.empty else 0

    summary = {
        'Metric': [
            'Total Ensemble Configurations',
            'Total Baseline Methods',
            'Total Oracle Records',
            'Datasets Tested',
            'Models Tested',
            'Seeds per Configuration',
            'Best Method Category',
            'Best Method Mean ECE',
            'Hybrid Win Rate vs NC',
            'Methods Beating All Baselines',
        ],
        'Value': [
            len(ensemble_df),
            len(baseline_df),
            len(oracle_df),
            ensemble_df['dataset'].nunique(),
            ensemble_df['model'].nunique(),
            '11-20 (varies)',
            best_cat_name,
            best_cat_ece,
            hybrid_win_rate,
            f"{num_dominators} methods",
        ],
    }

    return pd.DataFrame(summary)


def create_method_comparison_table(
    per_ds_leaderboards: Dict[str, pd.DataFrame],
    nc_vs_hybrid: pd.DataFrame,
) -> pd.DataFrame:
    """Create table comparing top methods across datasets."""

    rows: List[Dict[str, Any]] = []

    for dataset in sorted(per_ds_leaderboards.keys()):
        lb = per_ds_leaderboards[dataset]
        if lb.empty:
            continue

        # Get top 3 methods
        for rank, (_, row) in enumerate(lb.head(3).iterrows(), 1):
            rows.append({
                'Dataset': dataset.upper(),
                'Rank': rank,
                'Method': str(row['Method']).replace('Hybrid_', '').replace('NC_', ''),
                'Category': row['Category'],
                'ECE': row['Mean ECE'],
                'Std': row['Std Dev'],
                'N': row['N'],
            })

    return pd.DataFrame(rows)


def create_summary_statistics_table(
    ensemble_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create summary statistics across all configurations for selected methods.

    Configuration = (dataset, model, training_method).
    For each configuration:
      - Use mean ECE across seeds for each method.
      - Rank methods by ECE (1 = best) among the listed methods (excluding Uncalib).
      - Compute improvement vs PostHoc_uncalibrated.

    Outputs per method:
      - Average Rank (mean rank position, lower is better).
      - Win Rate (%): percentage of configurations where method is in top-3.
      - Mean Improvement (%): mean percentage reduction in ECE vs Uncalib.
    """

    on_cols = ["dataset", "model", "training_method"]

    # Map LaTeX display names to internal identifiers
    method_map_ensemble: Dict[str, str] = {
        "PSC": "NC_psc_learned",
        "Margin CVaR": "Hybrid_single_margin_tail_cvar_top2_shallow_a3.0",
        "Multiscale Sep": "Hybrid_single_multiscale_separation_top2_shallow_a3.0",
        "Calib. Decis.": "Hybrid_single_calibration_decisiveness_top2_shallow_a3.0",
        "LID": "Hybrid_single_local_intrinsic_dimensionality_top2_shallow_a3.0",
    }

    method_map_baseline: Dict[str, str] = {
        "Beta": "Baseline_beta_calibration",
        "Geo.": "Baseline_geometric_physical_space",
        "Iso.": "Baseline_isotonic_toplabel",
        "Platt": "Baseline_platt_scaling",
        "Temp.": "Baseline_temperature_scaling",
    }

    uncalib_name = "Baseline_uncalibrated"

    # Pre-aggregate to one ECE per configuration & method
    ens_agg = (
        ensemble_df.groupby(on_cols + ["full_method"])["ece"]
        .mean()
        .reset_index()
    )
    base_agg = (
        baseline_df.groupby(on_cols + ["method_name"])["ece"]
        .mean()
        .reset_index()
    )

    # Build configuration index where uncalibrated is available
    uncalib = base_agg[base_agg["method_name"] == uncalib_name].copy()
    if uncalib.empty:
        return pd.DataFrame()

    uncalib = uncalib.rename(columns={"ece": "ece_uncalib"})
    config_keys = uncalib[on_cols].drop_duplicates()
    total_configs = len(config_keys)
    if total_configs == 0:
        return pd.DataFrame()

    # Helper to get ECE for a given method and config
    def get_ece_for_ensemble(row_cfg, full_method: str) -> Optional[float]:
        sub = ens_agg[
            (ens_agg["dataset"] == row_cfg["dataset"])
            & (ens_agg["model"] == row_cfg["model"])
            & (ens_agg["training_method"] == row_cfg["training_method"])
            & (ens_agg["full_method"] == full_method)
        ]["ece"]
        if sub.empty:
            return None
        return float(sub.iloc[0])

    def get_ece_for_baseline(row_cfg, method_name: str) -> Optional[float]:
        sub = base_agg[
            (base_agg["dataset"] == row_cfg["dataset"])
            & (base_agg["model"] == row_cfg["model"])
            & (base_agg["training_method"] == row_cfg["training_method"])
            & (base_agg["method_name"] == method_name)
        ]["ece"]
        if sub.empty:
            return None
        return float(sub.iloc[0])

    # Accumulators
    methods = [
        "PSC",
        "Margin CVaR",
        "Multiscale Sep",
        "Calib. Decis.",
        "LID",
        "Beta",
        "Geo.",
        "Iso.",
        "Platt",
        "Temp.",
    ]

    rank_sum = {m: 0.0 for m in methods}
    rank_count = {m: 0 for m in methods}
    top3_count = {m: 0 for m in methods}
    improvement_sum = {m: 0.0 for m in methods}
    improvement_count = {m: 0 for m in methods}

    # Iterate over configurations with uncalibrated baseline
    for _, cfg in config_keys.iterrows():
        # Uncalibrated ECE
        unc_sub = uncalib[
            (uncalib["dataset"] == cfg["dataset"])
            & (uncalib["model"] == cfg["model"])
            & (uncalib["training_method"] == cfg["training_method"])
        ]
        if unc_sub.empty:
            continue
        unc_ece = float(unc_sub["ece_uncalib"].iloc[0])
        if not np.isfinite(unc_ece) or unc_ece <= 0:
            continue

        # Collect method ECEs for this configuration
        eces_for_rank = []
        per_method_ece: Dict[str, float] = {}

        # Ensemble-based methods
        for disp_name, full_method in method_map_ensemble.items():
            val = get_ece_for_ensemble(cfg, full_method)
            if val is None or not np.isfinite(val):
                continue
            per_method_ece[disp_name] = val
            eces_for_rank.append((disp_name, val))

        # Baseline methods
        for disp_name, method_name in method_map_baseline.items():
            val = get_ece_for_baseline(cfg, method_name)
            if val is None or not np.isfinite(val):
                continue
            per_method_ece[disp_name] = val
            eces_for_rank.append((disp_name, val))

        if not eces_for_rank:
            continue

        # Rank methods (1 = best)
        eces_for_rank.sort(key=lambda x: x[1])
        for rank_idx, (m_name, _) in enumerate(eces_for_rank, start=1):
            rank_sum[m_name] += rank_idx
            rank_count[m_name] += 1
            if rank_idx <= 3:
                top3_count[m_name] += 1

        # Improvements vs Uncalib.
        for m_name, val in per_method_ece.items():
            improvement = (unc_ece - val) / unc_ece * 100.0
            improvement_sum[m_name] += improvement
            improvement_count[m_name] += 1

    # Build summary rows
    rows: List[Dict[str, Any]] = []
    for m in methods:
        if rank_count[m] > 0:
            avg_rank = rank_sum[m] / rank_count[m]
        else:
            avg_rank = np.nan

        win_rate = (top3_count[m] / total_configs * 100.0) if total_configs > 0 else np.nan

        if improvement_count[m] > 0:
            mean_impr = improvement_sum[m] / improvement_count[m]
        else:
            mean_impr = np.nan

        rows.append(
            {
                "Method": m,
                "Average Rank": avg_rank,
                "Win Rate (%)": win_rate,
                "Mean Improvement (%)": mean_impr,
            }
        )

    df = pd.DataFrame(rows)
    return df

def create_baseline_comparison_table(dominators: pd.DataFrame) -> pd.DataFrame:
    """Create clean table of methods that beat all baselines."""

    if dominators is None or dominators.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, row in dominators.iterrows():
        method_name = str(row['Method']).replace('Hybrid_', '').replace('NC_', '')
        rows.append({
            'Method': method_name,
            'Overall Win Rate': f"{row['Overall Win Rate']*100:.1f}%",
            'Total Wins': row['Total Wins'],
            'Total Comparisons': row['Total Comparisons'],
            'Datasets': row['Datasets'],
            'Baselines Beaten': row['Baselines Beaten'],
        })

    return pd.DataFrame(rows)


def create_category_breakdown_table(category_perf: pd.DataFrame) -> pd.DataFrame:
    """Format category performance for LaTeX."""

    if category_perf is None or category_perf.empty:
        return pd.DataFrame()

    cols = ['Category', 'Mean ECE', 'Std ECE', 'Median ECE', 'N', 'N Datasets']
    return category_perf[cols].copy()


def create_oracle_gap_analysis(
    ensemble_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze gap between best methods and oracle performance."""

    if oracle_df is None or oracle_df.empty:
        return pd.DataFrame()

    on_cols = ['dataset', 'model', 'seed', 'training_method']

    # Best method per configuration
    best_methods = ensemble_df.loc[ensemble_df.groupby(on_cols)['ece'].idxmin()]

    merged = pd.merge(
        best_methods[on_cols + ['ece', 'full_method', 'category']],
        oracle_df[on_cols + ['oracle_ece']],
        on=on_cols,
        how='inner',
    )

    if merged.empty:
        return pd.DataFrame()

    merged['gap'] = merged['ece'] - merged['oracle_ece']
    merged['gap_pct'] = (merged['gap'] / merged['oracle_ece']) * 100

    summary = merged.groupby('dataset').agg({
        'gap': ['count', 'mean', 'std', 'min', 'max'],
        'gap_pct': ['mean', 'median'],
        'oracle_ece': 'mean',
        'ece': 'mean',
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'gap_count': 'N'})
    return summary.reset_index()


def generate_baseline_comparison_tables(ensemble_df, baseline_df, oracle_df, output_dir):
    """
    Generate clean LaTeX tables with top 5 baseline dominators vs all baselines.
    Format: Single line per model with mean±std, bold for minimum ECE in each row.
    """

    # Filter to only PostHoc baselines (exclude MultiLayer)
    # if not baseline_df.empty and "method_source" in baseline_df.columns:
    #     baseline_df = baseline_df[baseline_df["method_source"] == "post_hoc"].copy()

    # Top 5 methods by baseline domination win rate
    TOP_METHODS = {
        'PSC': 'NC_psc_learned',
        'Margin CVaR': 'Hybrid_single_margin_tail_cvar_top2_shallow_a3.0',
        'Multiscale Sep': 'Hybrid_single_multiscale_separation_top2_shallow_a3.0',
        'Calib. Decis.': 'Hybrid_single_calibration_decisiveness_top2_shallow_a3.0',
        'LID': 'Hybrid_single_local_intrinsic_dimensionality_top2_shallow_a3.0',
    }

    # Baseline name mappings (short forms)
    BASELINE_NAMES = {
        'beta_calibration': 'Beta',
        'geometric_physical_space': 'Geo.',
        'isotonic_toplabel': 'Iso.',
        'platt_scaling': 'Platt',
        'temperature_scaling': 'Temp.',
        'uncalibrated': 'Uncalib.',
    }

    def short_baseline_name(method_name: str) -> str:
        """
        Convert a baseline method name from baseline_df['method_name']
        (e.g. 'PostHoc_beta_calibration') into a short display name (e.g. 'Beta').
        """
        base = method_name
        if base.startswith('Baseline_'):
            base = base[len('Baseline_') :]
        elif base.startswith('PostHoc_'): # Keep as fallback just in case
            base = base[len('PostHoc_') :]
        elif base.startswith('MultiLayer_'):
            base = base[len('MultiLayer_') :]
        return BASELINE_NAMES.get(base, base[:8])

    # Save directly in output_dir
    results_dir = output_dir

    # Get all (dataset, training_method) combinations
    combos = ensemble_df[['dataset', 'training_method']].drop_duplicates()

    for _, combo in combos.iterrows():
        dataset = combo['dataset']
        training_method = combo['training_method']

        # Filter data for this combination
        ens_subset = ensemble_df[
            (ensemble_df['dataset'] == dataset)
            & (ensemble_df['training_method'] == training_method)
        ]
        base_subset = baseline_df[
            (baseline_df['dataset'] == dataset)
            & (baseline_df['training_method'] == training_method)
        ]

        if ens_subset.empty:
            continue

        # Get all unique models
        models = sorted(ens_subset['model'].unique())

        # Get all baseline methods present
        baseline_methods = sorted(base_subset['method_name'].unique())

        # Build table data
        table_data = []

        for model in models:
            row_data = {'Model': model}
            all_values = []  # Track all values to find minimum

            # Add data for top 5 methods
            for display_name, method_full_name in TOP_METHODS.items():
                method_data = ens_subset[
                    (ens_subset['model'] == model)
                    & (ens_subset['full_method'] == method_full_name)
                ]['ece']

                if len(method_data) > 0:
                    mean = np.mean(method_data.values)
                    std = np.std(method_data.values)
                    row_data[display_name] = {
                        'mean': mean,
                        'std': std,
                    }
                    all_values.append(mean)
                else:
                    row_data[display_name] = None

            # Add baseline methods (post-hoc and multi-layer baselines)
            for baseline in baseline_methods:
                base_data = base_subset[
                    (base_subset['model'] == model)
                    & (base_subset['method_name'] == baseline)
                ]['ece']

                col_name = short_baseline_name(baseline)  # e.g. 'Beta', 'Geo.', ...

                if len(base_data) > 0:
                    mean = float(np.mean(base_data.values))
                    std = float(np.std(base_data.values))
                    row_data[col_name] = {
                        'mean': mean,
                        'std': std,
                    }
                    all_values.append(mean)
                else:
                    if col_name not in row_data:
                        row_data[col_name] = None

            # Add Oracle column (mean±std across seeds for this config)
            oracle_vals = None
            if oracle_df is not None and not oracle_df.empty:
                oracle_vals = oracle_df[
                    (oracle_df['dataset'] == dataset)
                    & (oracle_df['model'] == model)
                    & (oracle_df['training_method'] == training_method)
                ]['oracle_ece'].values

            if oracle_vals is not None and oracle_vals.size > 0:
                mean = float(np.mean(oracle_vals))
                std = float(np.std(oracle_vals))
                row_data['Oracle'] = {'mean': mean, 'std': std}
                # Oracle is ground truth, do not include in min/all_values for ranking
            else:
                row_data['Oracle'] = None

            table_data.append(row_data)

        # Generate LaTeX table
        baseline_display_names = [short_baseline_name(b) for b in baseline_methods]

        generate_clean_latex_table(
            table_data,
            list(TOP_METHODS.keys()),
            baseline_display_names,
            dataset,
            training_method,
            results_dir,
        )


def generate_clean_latex_table(
    table_data,
    top_method_names,
    baseline_names,
    dataset,
    training_method,
    output_dir,
):
    """Generate clean LaTeX table with single line per model and bold minimum."""

    # All column names in order (Oracle appended at the end)
    all_columns = top_method_names + baseline_names + ['Oracle']

    # Column specification with vertical separator between novel methods and baselines
    left_count = len(top_method_names)
    right_count = len(baseline_names)
    col_spec = 'l' + ('c' * left_count) + '|' + ('c' * right_count) + 'c'

    # Build LaTeX
    lines = []
    lines.append("% Required packages: booktabs")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")

    # Format training method name nicely
    training_nice = training_method.replace('baseline_', '').replace('_', ' ').title()
    if training_method == 'augmix':
        training_nice = 'Augmix'

    lines.append(f"\\caption{{ECE (\\%) on {dataset.upper()} with {training_nice}}}")
    lines.append("{\\footnotesize \\textcolor{blue}{\\textbf{Best}}, "
                 "\\textcolor{teal}{\\textbf{2nd}}, "
                 "\\textcolor{olive}{\\textbf{3rd}}}")
    lines.append(f"\\label{{tab:{dataset}_{training_method}}}")
    lines.append("% Adjust column spacing and font size")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row
    header = "Model"
    for col in all_columns:
        header += f" & {col}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows
    for row in table_data:
        model_name = row['Model']

        # Determine top-3 ranks per row (excluding Uncalib. and Oracle)
        metric_values = []
        for col in all_columns:
            if col in ('Uncalib.', 'Oracle'):
                continue
            if col in row and row[col] is not None:
                metric_values.append((col, row[col]['mean']))
        metric_values.sort(key=lambda x: x[1])
        rank_map = {}
        for rank_idx, (col_name, _) in enumerate(metric_values[:3], start=1):
            rank_map[col_name] = rank_idx

        # Build row
        line = f"{model_name}"
        for col in all_columns:
            if col in row and row[col] is not None:
                data = row[col]
                mean = data['mean'] * 100.0  # Convert to percentage
                std = data['std'] * 100.0

                # Base formatted value
                value_str = f"{mean:.2f}$\\pm${std:.2f}"

                # Apply color + bolding based on rank (Oracle and Uncalib. never in rank_map)
                rank = rank_map.get(col)
                if rank == 1:
                    value_str = f"\\textcolor{{blue}}{{\\textbf{{{value_str}}}}}"
                elif rank == 2:
                    value_str = f"\\textcolor{{teal}}{{\\textbf{{{value_str}}}}}"
                elif rank == 3:
                    value_str = f"\\textcolor{{olive}}{{\\textbf{{{value_str}}}}}"

                line += f" & {value_str}"
            else:
                line += " & ---"
        line += " \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    # Save to file
    filename = output_dir / f"{dataset}_{training_method}_comparison.tex"
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  ✓ Generated: {filename.name}")


def export_summary_statistics_to_latex(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Export the summary statistics table to LaTeX.

    Title:  Summary Statistics Across All Configurations
    Label:  tab:summary_stats
    Environment: regular table (not sideways)
    """
    if summary_df is None or summary_df.empty:
        print("[WARN] Summary statistics table is empty; skipping LaTeX export.")
        return

    # Determine top-3 by Average Rank (lower is better)
    df = summary_df.copy()
    df_sorted = df.sort_values("Average Rank", ascending=True)
    top3_methods = set(df_sorted["Method"].head(3).tolist())

    # Column order
    cols = ["Method", "Average Rank", "Win Rate (%)", "Mean Improvement (%)"]

    lines: List[str] = []
    lines.append("% Required packages: booktabs")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Summary Statistics Across All Configurations}")
    lines.append("\\label{tab:summary_stats}")
    lines.append("% Adjust column spacing and font size")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Method & Average Rank & Win Rate (\\%) & Mean Improvement (\\%) \\\\")
    lines.append("\\midrule")

    # Methods in desired order with midrule before Beta
    method_order = [
        "PSC",
        "Margin CVaR",
        "Multiscale Sep",
        "Calib. Decis.",
        "LID",
        "Beta",
        "Geo.",
        "Iso.",
        "Platt",
        "Temp.",
    ]

    for idx, m in enumerate(method_order):
        row = df[df["Method"] == m]
        if row.empty:
            # If no data, print dashes
            avg_rank_str = "---"
            win_rate_str = "---"
            mean_impr_str = "---"
        else:
            r = row.iloc[0]
            avg_rank = r["Average Rank"]
            win_rate = r["Win Rate (%)"]
            mean_impr = r["Mean Improvement (%)"]

            avg_rank_str = f"{avg_rank:.2f}" if np.isfinite(avg_rank) else "---"
            win_rate_str = f"{win_rate:.2f}" if np.isfinite(win_rate) else "---"
            mean_impr_str = f"{mean_impr:.2f}" if np.isfinite(mean_impr) else "---"

        # Insert midrule before Beta to separate proposed methods and baselines
        if m == "Beta":
            lines.append("\\midrule")

        # Bold entire row if method is in top-3 by average rank
        if m in top3_methods:
            line = (
                f"\\textbf{{{m}}} & "
                f"\\textbf{{{avg_rank_str}}} & "
                f"\\textbf{{{win_rate_str}}} & "
                f"\\textbf{{{mean_impr_str}}} \\\\"
            )
        else:
            line = f"{m} & {avg_rank_str} & {win_rate_str} & {mean_impr_str} \\\\"

        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    filename = output_dir / "summary_statistics.tex"
    with open(filename, "w") as f:
        f.write("\n".join(lines))

    print(f"  ✓ Exported summary statistics table: {filename.name}")

def generate_mean_ece_across_training_tables(
    ensemble_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate mean ECE tables across training methods (Augmix, Brier, Cross Entropy)
    for CIFAR10 and CIFAR100.

    For each model and metric column, we:
      - average the per-training-method means directly
      - propagate uncertainty via sqrt(sum sigma_i^2 / n) with n=3

    The resulting LaTeX tables use a sidewaystable environment with the
    same column headers as the existing baseline comparison tables.
    """

    # Filter to only PostHoc baselines (exclude MultiLayer)
    # if not baseline_df.empty and "method_source" in baseline_df.columns:
    #     baseline_df = baseline_df[baseline_df["method_source"] == "post_hoc"].copy()

    # Datasets and training methods of interest
    target_datasets = ["cifar10", "cifar100"]
    target_training_methods = ["augmix", "baseline_brier", "baseline_cross_entropy"]

    # Mapping from display names to ensemble full_method keys
    top_methods_map: Dict[str, str] = {
        "PSC": "NC_psc_learned",
        "Margin CVaR": "Hybrid_single_margin_tail_cvar_top2_shallow_a3.0",
        "Multiscale Sep": "Hybrid_single_multiscale_separation_top2_shallow_a3.0",
        "Calib. Decis.": "Hybrid_single_calibration_decisiveness_top2_shallow_a3.0",
        "LID": "Hybrid_single_local_intrinsic_dimensionality_top2_shallow_a3.0",
    }

    # Baseline name mappings (short forms) – only post-hoc baselines are averaged
    baseline_methods_map: Dict[str, str] = {
        "Beta": "Baseline_beta_calibration",
        "Geo.": "Baseline_geometric_physical_space",
        "Iso.": "Baseline_isotonic_toplabel",
        "Platt": "Baseline_platt_scaling",
        "Temp.": "Baseline_temperature_scaling",
        "Uncalib.": "Baseline_uncalibrated",
    }

    def collect_means_stds_for_method(
        dataset: str,
        model: str,
        full_method: Optional[str] = None,
        baseline_method_name: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Collect mean and std per training method for a given metric and then
        aggregate across training methods.
        """
        means: List[float] = []
        sigmas: List[float] = []

        for t_method in target_training_methods:
            if full_method is not None:
                # Ensemble-based method (PSC, Margin CVaR, etc.)
                sub = ensemble_df[
                    (ensemble_df["dataset"] == dataset)
                    & (ensemble_df["model"] == model)
                    & (ensemble_df["training_method"] == t_method)
                    & (ensemble_df["full_method"] == full_method)
                ]["ece"].values
            else:
                # Post-hoc baseline method
                if baseline_method_name is None:
                    continue
                sub = baseline_df[
                    (baseline_df["dataset"] == dataset)
                    & (baseline_df["model"] == model)
                    & (baseline_df["training_method"] == t_method)
                    & (baseline_df["method_name"] == baseline_method_name)
                ]["ece"].values

            if sub.size == 0:
                continue

            m = float(np.mean(sub))
            s = float(np.std(sub))
            means.append(m)
            sigmas.append(s)

        if not means or not sigmas:
            return None, None

        n = len(means)
        mean_across = float(np.mean(means))
        # Propagation of uncertainty across training methods
        std_across = float(np.sqrt(np.sum(np.square(sigmas)) / n))
        return mean_across, std_across

    def collect_oracle_means_stds(dataset: str, model: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Collect oracle mean/std across training methods for a dataset/model pair.
        """
        if oracle_df is None or oracle_df.empty:
            return None, None

        means: List[float] = []
        sigmas: List[float] = []

        for t_method in target_training_methods:
            sub = oracle_df[
                (oracle_df["dataset"] == dataset)
                & (oracle_df["model"] == model)
                & (oracle_df["training_method"] == t_method)
            ]["oracle_ece"].values
            if sub.size > 0:
                means.append(float(np.mean(sub)))
                sigmas.append(float(np.std(sub)))

        if not means:
            return None, None

        mean_across = float(np.mean(means))
        std_across = float(np.sqrt(np.sum(np.square(sigmas)) / len(means)))
        return mean_across, std_across

    def write_sideways_table(dataset: str, table_data: List[Dict[str, Any]]) -> None:
        # Order and names of columns
        all_columns = [
            "PSC",
            "Margin CVaR",
            "Multiscale Sep",
            "Calib. Decis.",
            "LID",
            "Beta",
            "Geo.",
            "Iso.",
            "Platt",
            "Temp.",
            "Uncalib.",
            "Oracle",
        ]

        # Vertical separator between novel methods (left) and baselines (right, including Oracle)
        left_methods = ["PSC", "Margin CVaR", "Multiscale Sep", "Calib. Decis.", "LID"]
        left_count = len(left_methods)
        right_count = len(all_columns) - left_count
        col_spec = "l" + ("c" * left_count) + "|" + ("c" * right_count)

        lines: List[str] = []
        lines.append("% Required packages: booktabs, rotating")
        lines.append("\\begin{sidewaystable}[htbp]")
        lines.append("\\centering")

        ds_upper = dataset.upper()
        caption = f"Mean ECE (\\%) on {ds_upper} across training methods"
        label = f"tab:{dataset}_mean"

        lines.append(f"\\caption{{{caption}}}")
        lines.append("{\\footnotesize \\textcolor{blue}{\\textbf{Best}}, "
                     "\\textcolor{teal}{\\textbf{2nd}}, "
                     "\\textcolor{olive}{\\textbf{3rd}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append("% Adjust column spacing and font size")
        lines.append("\\setlength{\\tabcolsep}{4pt}")
        lines.append("\\small")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Header row
        header = "Model"
        for col in all_columns:
            header += f" & {col}"
        header += " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        # Data rows
        for row in table_data:
            model_name = row["Model"]

            # Determine top-3 ranks per row (excluding Uncalib. and Oracle)
            metric_values = []
            for col in all_columns:
                if col in ("Uncalib.", "Oracle"):
                    continue
                if col in row and row[col] is not None:
                    metric_values.append((col, row[col]["mean"]))
            metric_values.sort(key=lambda x: x[1])
            rank_map = {}
            for rank_idx, (col_name, _) in enumerate(metric_values[:3], start=1):
                rank_map[col_name] = rank_idx

            line = f"{model_name}"
            for col in all_columns:
                if col in row and row[col] is not None:
                    data = row[col]
                    mean_pct = data["mean"] * 100.0
                    std_pct = data["std"] * 100.0
                    value_str = f"{mean_pct:.2f}$\\pm${std_pct:.2f}"

                    rank = rank_map.get(col)
                    if rank == 1:
                        value_str = f"\\textcolor{{blue}}{{\\textbf{{{value_str}}}}}"
                    elif rank == 2:
                        value_str = f"\\textcolor{{teal}}{{\\textbf{{{value_str}}}}}"
                    elif rank == 3:
                        value_str = f"\\textcolor{{olive}}{{\\textbf{{{value_str}}}}}"

                    line += f" & {value_str}"
                else:
                    line += " & ---"
            line += " \\\\"
            lines.append(line)

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{sidewaystable}")

        filename = output_dir / f"{dataset}_mean_across_training.tex"
        with open(filename, "w") as f:
            f.write("\n".join(lines))

        print(f"  ✓ Generated mean-across-training table: {filename.name}")

    # Build tables for each dataset
    for ds in target_datasets:
        ds_models = sorted(
            ensemble_df[
                (ensemble_df["dataset"] == ds)
                & (ensemble_df["training_method"].isin(target_training_methods))
            ]["model"].unique()
        )

        if not ds_models:
            continue

        table_data: List[Dict[str, Any]] = []

        for model in ds_models:
            row: Dict[str, Any] = {"Model": model}

            # Ensemble-based methods
            for display_name, full_method in top_methods_map.items():
                mean_val, std_val = collect_means_stds_for_method(
                    ds, model, full_method=full_method
                )
                if mean_val is not None and std_val is not None:
                    row[display_name] = {"mean": mean_val, "std": std_val}
                else:
                    row[display_name] = None

            # Post-hoc baselines – ignore MultiLayer_* methods
            for display_name, method_name in baseline_methods_map.items():
                mean_val, std_val = collect_means_stds_for_method(
                    ds,
                    model,
                    full_method=None,
                    baseline_method_name=method_name,
                )
                if mean_val is not None and std_val is not None:
                    row[display_name] = {"mean": mean_val, "std": std_val}
                else:
                    row[display_name] = None

            # Oracle (mean±std across training methods)
            oracle_mean, oracle_std = collect_oracle_means_stds(ds, model)
            if oracle_mean is not None and oracle_std is not None:
                row["Oracle"] = {"mean": oracle_mean, "std": oracle_std}
            else:
                row["Oracle"] = None

            table_data.append(row)

        write_sideways_table(ds, table_data)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified analysis for Hybrid, NC, Multi-Layer, and Post-Hoc results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Data Dirs ---
    parser.add_argument(
        "--hybrid-results-dir",
        type=Path,
        required=True,
        help="Root directory for HYBRID ensemble results",
    )
    parser.add_argument(
        "--nc-results-dir",
        type=Path,
        default=None,
        help="Root directory for NC ensemble results (optional)",
    )
    parser.add_argument(
        "--multi-layer-baselines-dir",
        type=Path,
        default=None,
        help="Directory for MULTI-LAYER (old) baseline results",
    )
    parser.add_argument(
        "--post-hoc-baselines-dir",
        type=Path,
        default=None,
        help="Directory for POST-HOC (Temp, Platt) baseline results",
    )
    parser.add_argument(
        "--oracle-fallback-dir",
        type=Path,
        default=None,
        help="Fallback directory to search for oracle",
    )
    # --- Output ---
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./publication_master_analysis"),
        help="Output directory for analysis results",
    )
    # --- Analysis Params ---
    parser.add_argument(
        "--oracle-threshold",
        type=float,
        default=0.005,
        help="Threshold for 'within oracle' analysis",
    )
    parser.add_argument(
        "--min-accuracy-threshold",
        type=float,
        default=0.6,
        help="Unified minimum accuracy to include a result in the analysis",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
    warnings.filterwarnings('ignore', category=RuntimeWarning) # For mean_ci

    # ------------------ LOAD THE DATA (UNIFY, MINIMAL) ------------------
    # We must actually run and load/unify so the variables are available here
    print("="*80)
    print("PUBLICATION-READY MASTER ANALYSIS [LaTeX only]")
    print(f"Unified accuracy threshold: {args.min_accuracy_threshold}")
    print("="*80)

    print("\n[1/2] Loading and unifying data...")

    # Load Hybrid results
    hybrid_df = pd.DataFrame()
    if args.hybrid_results_dir and args.hybrid_results_dir.exists():
        hybrid_df = find_hybrid_ensemble_results(args.hybrid_results_dir, args.min_accuracy_threshold)
        print(f"  ✓ Loaded {len(hybrid_df)} HYBRID ensemble results")
    else:
        print("  [ERROR] Could not find --hybrid-results-dir")
        return 1

    # Load NC results
    nc_df = pd.DataFrame()
    nc_dir_path = args.nc_results_dir if args.nc_results_dir and args.nc_results_dir.exists() else None
    if nc_dir_path:
        nc_df = find_nc_ensemble_results(nc_dir_path, args.min_accuracy_threshold)
        print(f"  ✓ Loaded {len(nc_df)} NC ensemble results")

    # Merge ensemble results
    ensemble_df = pd.concat([hybrid_df, nc_df], ignore_index=True)
    if ensemble_df.empty:
        print("[ERROR] No ensemble results found. Exiting.")
        return 1
    print(f"  ► TOTAL {len(ensemble_df)} ENSEMBLE results loaded.")

    # Load baselines
    multi_layer_base_df = pd.DataFrame()
    if args.multi_layer_baselines_dir and args.multi_layer_baselines_dir.exists():
        multi_layer_base_df = load_multi_layer_baselines(args.multi_layer_baselines_dir, ensemble_df)
        print(f"  ✓ Loaded {len(multi_layer_base_df)} MULTI-LAYER baseline results")
    post_hoc_base_df = pd.DataFrame()
    if args.post_hoc_baselines_dir and args.post_hoc_baselines_dir.exists():
        post_hoc_base_df = load_post_hoc_baselines(args.post_hoc_baselines_dir, ensemble_df)
        if not post_hoc_base_df.empty:
            print(f"  ✓ Loaded {len(post_hoc_base_df)} POST-HOC baseline results")
    baseline_df = pd.concat([multi_layer_base_df, post_hoc_base_df], ignore_index=True)

    # Oracle fallback logic
    oracle_fallback_dir = args.oracle_fallback_dir
    if oracle_fallback_dir is None:
        if args.post_hoc_baselines_dir and args.post_hoc_baselines_dir.exists():
            oracle_fallback_dir = args.post_hoc_baselines_dir
        elif args.multi_layer_baselines_dir and args.multi_layer_baselines_dir.exists():
            oracle_fallback_dir = args.multi_layer_baselines_dir

    oracle_df = pd.DataFrame()
    try:
        oracle_df = load_oracle(ensemble_df, args.hybrid_results_dir, nc_dir_path, oracle_fallback_dir)
    except Exception as e:
        print("  [WARN] Could not load oracle df:", e)
        oracle_df = pd.DataFrame()

    # Unification
    print("[UNIFICATION] Normalizing datasets and baselines...")
    ensemble_df = unify_dataset_names(ensemble_df)
    baseline_df = unify_dataset_names(baseline_df)
    oracle_df = unify_dataset_names(oracle_df)
    baseline_df = unify_baseline_names(baseline_df)

    # CATEGORY PERFORMANCE
    category_perf = category_performance_comparison(ensemble_df)
    if category_perf is None:
        category_perf = pd.DataFrame()
    # NC vs Hybrid
    nc_vs_hybrid, nc_hybrid_merged_df = nc_vs_hybrid_comparison(ensemble_df)
    # Dominators
    dominators = pd.DataFrame()
    if not baseline_df.empty:
        dominators = find_baseline_dominators(ensemble_df, baseline_df)
    # Top-K analysis
    topk_breakdown = topk_analysis(ensemble_df)
    # Per-dataset leaderboards
    per_ds_leaderboards = per_dataset_leaderboard(ensemble_df)

    print("\n[2/2] Exporting LaTeX tables...")

    # ========================================================================
    # Export LaTeX Tables
    # ========================================================================
    latex_dir = args.output_dir / "latex_tables"
    latex_dir.mkdir(exist_ok=True)

    # 1. Executive Summary
    exec_summary = create_executive_summary_table(
        ensemble_df, baseline_df, oracle_df, category_perf, nc_vs_hybrid, dominators
    )
    export_table_to_latex(
        exec_summary,
        caption="Executive Summary of All Experiments",
        label="tab:exec_summary",
        filename=latex_dir / "executive_summary.tex",
    )

    # 2. Category Performance
    if not category_perf.empty:
        cat_table = create_category_breakdown_table(category_perf)
        export_table_to_latex(
            cat_table,
            caption="Layer Selection Category Performance Comparison",
            label="tab:category_perf",
            filename=latex_dir / "category_performance.tex",
        )

    # 3. Top Methods per Dataset
    method_comp = create_method_comparison_table(per_ds_leaderboards, nc_vs_hybrid)
    if not method_comp.empty:
        export_table_to_latex(
            method_comp,
            caption="Top 3 Methods Per Dataset",
            label="tab:top_methods",
            filename=latex_dir / "top_methods_per_dataset.tex",
        )

    # 4. Baseline Dominators
    if not dominators.empty:
        baseline_comp = create_baseline_comparison_table(dominators)
        export_table_to_latex(
            baseline_comp,
            caption="Methods That Beat All Baselines",
            label="tab:baseline_dominators",
            filename=latex_dir / "baseline_dominators.tex",
        )

    # 5. Oracle Gap Analysis
    if not oracle_df.empty:
        oracle_gap = create_oracle_gap_analysis(ensemble_df, oracle_df)
        if not oracle_gap.empty:
            export_table_to_latex(
                oracle_gap,
                caption="Gap Between Best Methods and Oracle Per Dataset",
                label="tab:oracle_gap",
                filename=latex_dir / "oracle_gap_analysis.tex",
                float_format='%.5f',
                footnote="All comparisons use paired samples only - methods and oracle evaluated on identical (dataset, model, seed, training_method) tuples. N indicates number of paired samples."
            )

    # 6. NC vs Hybrid Direct Comparison
    if not nc_vs_hybrid.empty:
        export_table_to_latex(
            nc_vs_hybrid,
            caption="NC Ensemble vs Hybrid Ensemble Head-to-Head Comparison",
            label="tab:nc_vs_hybrid",
            filename=latex_dir / "nc_vs_hybrid.tex",
        )

    # 7. Top-K Analysis
    export_table_to_latex(
        topk_breakdown,
        caption="Performance by Ensemble Size (k)",
        label="tab:topk_analysis",
        filename=latex_dir / "topk_analysis.tex",
    )

    # 8. Model-specific SOTA comparison (cross-entropy only)
    generate_sota_comparison_table_v2(
        ensemble_df,
        args.output_dir,
        allowed_training_methods={"baseline_brier"},
    )

    print(f"  ✓ All LaTeX tables saved to {latex_dir}/")

    print("\nDone (LaTeX tables only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())