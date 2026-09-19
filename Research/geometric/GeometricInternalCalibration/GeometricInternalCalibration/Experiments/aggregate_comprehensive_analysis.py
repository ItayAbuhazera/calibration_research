#!/usr/bin/env python3
"""
Comprehensive aggregation and analysis script for calibration experiment results.

This script:
1. Reads JSON experiment results from a directory
2. Groups by (Training Method, Dataset, Model)
3. Extracts ECE values for multiple methods across seeds
4. Calculates Mean ± 95% CI for each method
5. Outputs CSV and LaTeX tables
6. Provides win rate analysis
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

# Simple logger for compatibility
class SimpleLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    
    def warning(self, msg):
        print(f"[WARNING] {msg}")


logger = SimpleLogger()

# Try to import scipy for statistical tests
try:
    from scipy.stats import ttest_rel, wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not available - statistical tests will be skipped")


def extract_ece_from_json(json_data: Dict[str, Any], method_key: str) -> Optional[float]:
    """
    Extract ECE value from JSON data for a given method key.
    Handles skipped sections gracefully.
    
    Args:
        json_data: Parsed JSON dictionary
        method_key: Key path to the method (e.g., 'uncalibrated', 'baseline_ts')
    
    Returns:
        ECE value if found, None otherwise
    """
    try:
        # Direct keys
        if method_key in json_data:
            result = json_data[method_key]
            # Check if section was skipped
            if isinstance(result, dict):
                if result.get('skipped', False):
                    return None
                if 'ece' in result:
                    return float(result['ece'])
            elif isinstance(result, (int, float)):
                return float(result)
        
        # Nested keys (e.g., 'ablation.geometric_with_dac_features')
        if '.' in method_key:
            parts = method_key.split('.')
            current = json_data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            # Check if section was skipped
            if isinstance(current, dict):
                if current.get('skipped', False):
                    return None
                if 'ece' in current:
                    return float(current['ece'])
            elif isinstance(current, (int, float)):
                return float(current)
        
        return None
    except (KeyError, TypeError, ValueError) as e:
        return None


def extract_lowest_ece_from_list(json_data: Dict[str, Any], key_path: str) -> Optional[float]:
    """
    Extract the lowest ECE from a list of results (e.g., metric_guided_calibration).
    Handles skipped sections gracefully.
    
    Args:
        json_data: Parsed JSON dictionary
        key_path: Key path to the list (e.g., 'metric_guided_calibration')
    
    Returns:
        Lowest ECE value if found, None otherwise
    """
    try:
        parts = key_path.split('.')
        current = json_data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        # Check if section was skipped
        if isinstance(current, dict) and current.get('skipped', False):
            return None
        
        if isinstance(current, list):
            ece_values = []
            for item in current:
                if isinstance(item, dict) and 'ece' in item:
                    ece_values.append(float(item['ece']))
            if ece_values:
                return min(ece_values)
        
        # Handle dict of results (e.g., ablation.geometric_with_dac_features)
        if isinstance(current, dict):
            ece_values = []
            for key, value in current.items():
                if isinstance(value, dict):
                    # Skip if this entry was skipped
                    if value.get('skipped', False):
                        continue
                    if 'ece' in value:
                        ece_values.append(float(value['ece']))
            if ece_values:
                return min(ece_values)
        
        return None
    except (KeyError, TypeError, ValueError):
        return None


def extract_coordinate_ece(json_data: Dict[str, Any], method: str) -> Optional[float]:
    """
    Extract ECE for coordinate-based methods.
    Prioritizes direct method keys over analysis section to ensure we use raw ECE values.
    
    Args:
        json_data: Parsed JSON dictionary
        method: One of 'random_coord', 'gradient_coord', 'coordinate_sampling', 'coordinate_spp'
    
    Returns:
        ECE value if found, None otherwise
    """
    # Try direct keys first (preferred - these are raw ECE values)
    if method == 'coordinate_sampling' and 'coordinate_sampling' in json_data:
        result = json_data['coordinate_sampling']
        if isinstance(result, dict) and 'ece' in result:
            return float(result['ece'])
    
    if method == 'coordinate_spp' and 'coordinate_spp' in json_data:
        result = json_data['coordinate_spp']
        if isinstance(result, dict) and 'ece' in result:
            return float(result['ece'])
    
    # Fallback: Try analysis section (only if direct keys not found)
    # Note: We only use this as fallback since user wants raw ECE values
    if 'analysis' in json_data and isinstance(json_data['analysis'], dict):
        analysis = json_data['analysis']
        
        # Check sgc_vs_coordinate_comparison
        if 'sgc_vs_coordinate_comparison' in analysis:
            comp = analysis['sgc_vs_coordinate_comparison']
            if method == 'random_coord' and 'random_coord_ece' in comp:
                return float(comp['random_coord_ece'])
            elif method == 'gradient_coord' and 'gradient_coord_ece' in comp:
                return float(comp['gradient_coord_ece'])
            elif method == 'coordinate_sampling' and 'coordinate_ece' in comp:
                return float(comp['coordinate_ece'])
        
        # Check coordinate_methods_comparison
        if 'coordinate_methods_comparison' in analysis:
            comp = analysis['coordinate_methods_comparison']
            if method == 'coordinate_sampling' and 'geometric_coordinate_ece' in comp:
                return float(comp['geometric_coordinate_ece'])
    
    return None


def calculate_statistics(values: List[float]) -> Tuple[int, float, float]:
    """
    Calculate count, mean, and 95% confidence interval for a list of values.
    
    Args:
        values: List of ECE values
    
    Returns:
        Tuple of (count, mean, ci_margin)
    """
    if not values:
        return 0, np.nan, np.nan
    
    values_array = np.array(values)
    n = len(values_array)
    mean = np.mean(values_array)
    
    if n < 2:
        return n, mean, 0.0
    
    std_dev = np.std(values_array, ddof=1)  # Sample standard deviation
    ci_margin = 1.96 * (std_dev / np.sqrt(n))
    
    return n, mean, ci_margin


def format_statistic(mean: float, ci_margin: float, n: int) -> str:
    """
    Format statistic as "mean ± ci" percentage.
    
    Args:
        mean: Mean value
        ci_margin: 95% CI margin
        n: Count
    
    Returns:
        Formatted string (e.g., "0.52 ± 0.04" or "N/A" if n=0)
    """
    if n == 0 or np.isnan(mean):
        return "N/A"
    # Convert to percentage (multiply by 100)
    mean_pct = mean * 100
    ci_pct = ci_margin * 100
    return f"{mean_pct:.2f} ± {ci_pct:.2f}"


def perform_paired_test(values1: List[float], values2: List[float], test_type: str = 't-test') -> Tuple[float, float, bool]:
    """
    Perform paired statistical test between two sets of values.
    
    Args:
        values1: First set of ECE values (paired with values2)
        values2: Second set of ECE values (paired with values1)
        test_type: 't-test' for paired t-test, 'wilcoxon' for Wilcoxon signed-rank
    
    Returns:
        Tuple of (p-value, effect_size, is_significant)
        effect_size: Cohen's d for paired samples
        is_significant: True if p < 0.05
    """
    if not HAS_SCIPY:
        return np.nan, np.nan, False
    
    if len(values1) != len(values2) or len(values1) < 2:
        return np.nan, np.nan, False
    
    values1_array = np.array(values1)
    values2_array = np.array(values2)
    
    # Calculate differences
    differences = values1_array - values2_array
    
    # Perform test
    if test_type == 't-test':
        try:
            t_stat, p_value = ttest_rel(values1_array, values2_array)
        except Exception:
            return np.nan, np.nan, False
    elif test_type == 'wilcoxon':
        try:
            stat, p_value = wilcoxon(differences)
        except Exception:
            return np.nan, np.nan, False
    else:
        raise ValueError(f"Unknown test_type: {test_type}")
    
    # Calculate Cohen's d (effect size for paired samples)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    if std_diff > 0:
        cohens_d = mean_diff / std_diff
    else:
        cohens_d = 0.0
    
    is_significant = p_value < 0.05
    
    return float(p_value), float(cohens_d), is_significant


def round_floats_to_5_decimals(obj: Any) -> Any:
    """
    Recursively round all float values in a nested structure to 5 decimal places.
    
    Args:
        obj: Any object (dict, list, float, etc.)
    
    Returns:
        Object with all floats rounded to 5 decimal places
    """
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 5)
    elif isinstance(obj, dict):
        return {key: round_floats_to_5_decimals(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [round_floats_to_5_decimals(item) for item in obj]
    else:
        return obj


def perform_statistical_comparisons(
    data: Dict[Tuple[str, str, str], Dict[str, List[float]]],
    comparison_groups: List[Tuple[str, str, str]]
) -> Dict[str, Dict[str, Any]]:
    """
    Perform statistical comparisons between method pairs.
    
    Args:
        data: Collected data dictionary
        comparison_groups: List of (method1, method2, description) tuples
    
    Returns:
        Dictionary with comparison results
    """
    if not HAS_SCIPY:
        return {"error": "scipy not available for statistical tests"}
    
    results = {}
    
    for method1, method2, description in comparison_groups:
        all_p_values = []
        all_effect_sizes = []
        significant_count = 0
        total_comparisons = 0
        config_results = []
        
        for (training, dataset, model), method_data in sorted(data.items()):
            if method1 in method_data and method2 in method_data:
                values1 = method_data[method1]
                values2 = method_data[method2]
                
                # Only compare if we have paired data (same number of seeds)
                if len(values1) == len(values2) and len(values1) >= 2:
                    # Match pairs by index (assuming same seed order)
                    p_value, effect_size, is_sig = perform_paired_test(values1, values2, test_type='t-test')
                    
                    if not np.isnan(p_value):
                        all_p_values.append(p_value)
                        all_effect_sizes.append(effect_size)
                        total_comparisons += 1
                        if is_sig:
                            significant_count += 1
                        
                        config_results.append({
                            'config': f"{training}_{dataset}_{model}",
                            'p_value': p_value,
                            'effect_size': effect_size,
                            'significant': is_sig,
                            'mean_diff': float(np.mean(np.array(values1) - np.array(values2))),
                            'method1_mean': float(np.mean(values1)),
                            'method2_mean': float(np.mean(values2))
                        })
        
        # Aggregate statistics
        if all_p_values:
            results[description] = {
                'total_comparisons': total_comparisons,
                'significant_count': significant_count,
                'significant_rate': float(significant_count / total_comparisons) if total_comparisons > 0 else 0.0,
                'mean_p_value': float(np.mean(all_p_values)),
                'median_p_value': float(np.median(all_p_values)),
                'mean_effect_size': float(np.mean(all_effect_sizes)),
                'median_effect_size': float(np.median(all_effect_sizes)),
                'config_results': config_results
            }
        else:
            results[description] = {
                'total_comparisons': 0,
                'significant_count': 0,
                'significant_rate': 0.0,
                'note': 'No valid paired comparisons found'
            }
    
    # Round all float values to 5 decimal places
    results = round_floats_to_5_decimals(results)
    
    return results


def print_statistical_comparisons(comparison_results: Dict[str, Dict[str, Any]]):
    """
    Print statistical comparison results in a readable format.
    
    Args:
        comparison_results: Results from perform_statistical_comparisons
    """
    print("\n" + "="*80)
    print("STATISTICAL COMPARISONS (Same Algorithm, Different Features)")
    print("="*80)
    
    if 'error' in comparison_results:
        print(f"\nError: {comparison_results['error']}")
        return
    
    for description, result in comparison_results.items():
        print(f"\n{description}")
        print("-" * 80)
        
        if 'note' in result:
            print(f"  {result['note']}")
            continue
        
        print(f"  Total Comparisons: {result['total_comparisons']}")
        print(f"  Significant (p < 0.05): {result['significant_count']} ({result['significant_rate']*100:.1f}%)")
        print(f"  Mean p-value: {result['mean_p_value']:.4f}")
        print(f"  Median p-value: {result['median_p_value']:.4f}")
        print(f"  Mean Effect Size (Cohen's d): {result['mean_effect_size']:.3f}")
        print(f"  Median Effect Size: {result['median_effect_size']:.3f}")
        
        # Show per-configuration results
        if result['config_results']:
            print(f"\n  Per-Configuration Results:")
            for config_result in result['config_results']:
                sig_marker = "***" if config_result['significant'] else ""
                print(f"    {config_result['config']}:")
                print(f"      p = {config_result['p_value']:.4f} {sig_marker}")
                print(f"      Effect Size = {config_result['effect_size']:.3f}")
                print(f"      Mean Diff = {config_result['mean_diff']*100:.3f}% "
                      f"({config_result['method1_mean']*100:.2f}% vs {config_result['method2_mean']*100:.2f}%)")
    
    print("="*80)
    print("Note: *** indicates p < 0.05 (significant difference)")


def extract_baseline_ece(json_data: Dict[str, Any], baseline_key: str) -> Optional[float]:
    """Extract ECE from standard_baselines or baselines section."""
    baselines = json_data.get('standard_baselines', json_data.get('baselines', {}))
    if baseline_key in baselines:
        result = baselines[baseline_key]
        if isinstance(result, dict):
            if result.get('skipped', False):
                return None
            ece = result.get('ece')
            if ece is not None:
                return float(ece)
    return None


def extract_baseline_acc(json_data: Dict[str, Any], baseline_key: str) -> Optional[float]:
    """Extract accuracy from standard_baselines or baselines section."""
    baselines = json_data.get('standard_baselines', json_data.get('baselines', {}))
    if baseline_key in baselines:
        result = baselines[baseline_key]
        if isinstance(result, dict):
            if result.get('skipped', False):
                return None
            acc = result.get('acc')
            if acc is not None:
                # Accuracy is already in [0,1] format
                return float(acc)
    return None


def extract_sgc_acc_fixed(json_data: Dict[str, Any], L: int = 6, d: int = 256) -> Optional[float]:
    """Extract SGC accuracy for specific (L, d) configuration."""
    # Try new structure first: global_random at top level
    global_random = json_data.get('global_random', {})
    if global_random and isinstance(global_random, dict):
        if global_random.get('skipped', False):
            return None
        num_layers = global_random.get('num_layers')
        target_dim = global_random.get('target_dim')
        if num_layers == L and target_dim == d:
            acc = global_random.get('accuracy')
            if acc is not None:
                # Accuracy is in percentage, convert to [0,1]
                return float(acc) / 100.0
    
    # Fallback to old structure: random_layer_ablation
    random_ablation = json_data.get('random_layer_ablation', {})
    if 'results_by_target_dim' not in random_ablation:
        return None
    
    results_by_target_dim = random_ablation['results_by_target_dim']
    target_dim_str = str(d)
    
    if target_dim_str not in results_by_target_dim:
        return None
    
    dim_data = results_by_target_dim[target_dim_str]
    global_random = dim_data.get('global_random', [])
    
    # Find entries with num_layers == L
    matching_entries = [r for r in global_random if r.get('num_layers') == L]
    
    if not matching_entries:
        return None
    
    # Return mean accuracy across all matching entries
    accs = [r.get('accuracy') for r in matching_entries if 'accuracy' in r]
    if not accs:
        return None
    
    # Convert from percentage to [0,1] and return mean
    return float(np.mean([acc / 100.0 for acc in accs]))


def extract_acc_from_json(json_data: Dict[str, Any], method_key: str) -> Optional[float]:
    """Extract accuracy value from JSON data for a given method key."""
    try:
        # Direct keys
        if method_key in json_data:
            result = json_data[method_key]
            if isinstance(result, dict):
                if result.get('skipped', False):
                    return None
                acc = result.get('accuracy')
                if acc is not None:
                    # Accuracy is in percentage, convert to [0,1]
                    return float(acc) / 100.0
                # Also try 'acc' key (for baselines)
                acc = result.get('acc')
                if acc is not None:
                    return float(acc)
        
        # Nested keys
        if '.' in method_key:
            parts = method_key.split('.')
            current = json_data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            if isinstance(current, dict):
                if current.get('skipped', False):
                    return None
                acc = current.get('accuracy')
                if acc is not None:
                    return float(acc) / 100.0
                acc = current.get('acc')
                if acc is not None:
                    return float(acc)
        
        return None
    except (KeyError, TypeError, ValueError):
        return None


def extract_sgc_ece_fixed(json_data: Dict[str, Any], L: int = 6, d: int = 256) -> Optional[float]:
    """Extract SGC ECE for specific (L, d) configuration, matching generate_sgc_tables.py logic."""
    # Try new structure first: global_random at top level
    global_random = json_data.get('global_random', {})
    if global_random and isinstance(global_random, dict):
        if global_random.get('skipped', False):
            return None
        num_layers = global_random.get('num_layers')
        target_dim = global_random.get('target_dim')
        if num_layers == L and target_dim == d:
            ece = global_random.get('ece')
            if ece is not None:
                return float(ece)
    
    # Fallback to old structure: random_layer_ablation
    random_ablation = json_data.get('random_layer_ablation', {})
    if 'results_by_target_dim' not in random_ablation:
        return None
    
    results_by_target_dim = random_ablation['results_by_target_dim']
    target_dim_str = str(d)
    
    if target_dim_str not in results_by_target_dim:
        return None
    
    dim_data = results_by_target_dim[target_dim_str]
    global_random = dim_data.get('global_random', [])
    
    # Find entries with num_layers == L
    matching_entries = [r for r in global_random if r.get('num_layers') == L]
    
    if not matching_entries:
        return None
    
    # Return mean ECE across all matching entries
    eces = [r.get('ece') for r in matching_entries if 'ece' in r]
    if not eces:
        return None
    
    return float(np.mean(eces))


def extract_geo_comb_ece_fixed(json_data: Dict[str, Any], target_dim: int = 256) -> Optional[float]:
    """Extract Geo Comb (geometric_concatenated) ECE, matching generate_sgc_tables.py logic."""
    # Try geometric_concatenated first (new structure)
    geo_concatenated = json_data.get('geometric_concatenated', {})
    if geo_concatenated and isinstance(geo_concatenated, dict):
        if geo_concatenated.get('skipped', False):
            return None
        ece = geo_concatenated.get('ece')
        if ece is not None:
            return float(ece)
    
    # Fallback to old structure: geo_comb with results list
    geo_comb = json_data.get('geo_comb', {})
    results = geo_comb.get('results', [])
    
    # Find best ECE among entries with target_dim
    matching_results = [r for r in results if r.get('target_dim') == target_dim]
    
    if not matching_results:
        return None
    
    # Return best (minimum) ECE
    eces = [r.get('ece') for r in matching_results if 'ece' in r]
    if not eces:
        return None
    
    return float(min(eces))


def collect_data_from_directory(directory: Path) -> Dict[Tuple[str, str, str], Dict[str, List[float]]]:
    """
    Collect ECE and accuracy values from all JSON files in directory.
    Uses correct extraction logic matching generate_sgc_tables.py
    
    Args:
        directory: Path to directory containing JSON files
    
    Returns:
        Dictionary mapping (training_method, dataset, model) -> {method_name: [ece_values], 'accuracy': [acc_values]}
    """
    data = defaultdict(lambda: defaultdict(list))
    
    # Method extraction mapping - using correct paths matching generate_sgc_tables.py
    method_extractors = {
        'uncalibrated': lambda j: extract_baseline_ece(j, 'uncalibrated'),
        'ts': lambda j: extract_baseline_ece(j, 'temperature_scaling'),
        'platt': lambda j: extract_baseline_ece(j, 'platt_scaling'),
        'isotonic': lambda j: extract_baseline_ece(j, 'isotonic_toplabel'),
        'beta': lambda j: extract_baseline_ece(j, 'beta_calibration'),
        'dac': lambda j: extract_baseline_ece(j, 'density_aware_calibration'),
        'pixel_geo': lambda j: extract_baseline_ece(j, 'geometric_physical_space'),
        'sgc': lambda j: extract_sgc_ece_fixed(j, L=6, d=256),
        'sgc_dac_layers': lambda j: extract_ece_from_json(j, 'sgc_with_dac_layers'),
        'geo_comb': lambda j: extract_ece_from_json(j, 'geometric_concatenated') or extract_geo_comb_ece_fixed(j, target_dim=256),
        'rand_coords': lambda j: extract_ece_from_json(j, 'coordinate_sampling'),
        'grad_coords': lambda j: extract_ece_from_json(j, 'coordinate_spp'),
        'dac_coord_feat': lambda j: extract_ece_from_json(j, 'dac_with_coordinate_features'),
        'geo_dac_feat': lambda j: extract_lowest_ece_from_list(j, 'ablation.geometric_with_dac_features'),
        'dac_geo_feat': lambda j: (
            extract_lowest_ece_from_list(j, 'ablation.dac_with_geometric_features') or
            extract_ece_from_json(j, 'ablation.dac_with_sgc_aggregated_features') or
            extract_ece_from_json(j, 'ablation.dac_with_random_layer_selection')
        ),
        'metric_guided': lambda j: extract_lowest_ece_from_list(j, 'metric_guided_calibration'),
    }
    
    json_files = list(directory.glob('*.json'))
    logger.info(f"Found {len(json_files)} JSON files in {directory}")
    
    processed_count = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            
            # Extract configuration
            config = json_data.get('experiment_config', {})
            training_method = config.get('training_method', 'unknown')
            dataset = config.get('dataset', 'unknown')
            model = config.get('model_name', 'unknown')
            
            key = (training_method, dataset, model)
            
            # Extract ECE values for each method
            methods_found = 0
            for method_name, extractor in method_extractors.items():
                ece_value = extractor(json_data)
                if ece_value is not None:
                    data[key][method_name].append(ece_value)
                    methods_found += 1
            
            # Extract accuracy from uncalibrated baseline (same across all methods)
            acc_value = extract_baseline_acc(json_data, 'uncalibrated')
            if acc_value is not None:
                data[key]['accuracy'].append(acc_value)
            
            if methods_found > 0:
                processed_count += 1
            else:
                logger.warning(f"No methods found in {json_file.name}")
        
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in {json_file}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Error processing {json_file}: {e}")
            continue
    
    logger.info(f"Successfully processed {processed_count}/{len(json_files)} files")
    
    return data


def create_summary_table(data: Dict[Tuple[str, str, str], Dict[str, List[float]]]) -> pd.DataFrame:
    """
    Create summary table with statistics for each configuration.
    
    Args:
        data: Collected data dictionary
    
    Returns:
        DataFrame with columns: Training, Dataset, Model, N, and method columns (with formatted values)
        Also stores raw mean values in _mean columns for sorting/coloring
    """
    rows = []
    method_names = ['uncalibrated', 'ts', 'platt', 'isotonic', 'beta', 'dac', 'pixel_geo',
                    'sgc_dac_layers', 'sgc', 'geo_comb', 'rand_coords', 'grad_coords',
                    'dac_coord_feat', 'geo_dac_feat', 'dac_geo_feat', 'metric_guided']
    
    for (training, dataset, model), method_data in sorted(data.items()):
        row = {
            'Training': training,
            'Dataset': dataset,
            'Model': model,
        }
        
        # Determine N (use the method with most samples)
        max_n = 0
        for method_name in method_names:
            if method_name in method_data:
                max_n = max(max_n, len(method_data[method_name]))
        row['N'] = max_n
        
        # Calculate accuracy statistics (from uncalibrated baseline)
        # Accuracy is in [0,1] format, format_statistic already multiplies by 100 for percentages
        if 'accuracy' in method_data and method_data['accuracy']:
            n_acc, mean_acc, ci_acc = calculate_statistics(method_data['accuracy'])
            # format_statistic expects values in [0,1] and converts to percentage
            # But since we want percentage display, we need to multiply by 100 first
            # Actually, format_statistic multiplies by 100, so we should pass values as-is
            row['accuracy'] = format_statistic(mean_acc, ci_acc, n_acc)
            row['accuracy_mean'] = mean_acc
        else:
            row['accuracy'] = "N/A"
            row['accuracy_mean'] = np.nan
        
        # Calculate statistics for each method
        for method_name in method_names:
            if method_name in method_data:
                n, mean, ci = calculate_statistics(method_data[method_name])
                row[method_name] = format_statistic(mean, ci, n)
                # Store raw mean for sorting/coloring (as hidden column)
                row[f'{method_name}_mean'] = mean
            else:
                row[method_name] = "N/A"
                row[f'{method_name}_mean'] = np.nan
        
        rows.append(row)
    
    # Create DataFrame - include both formatted and mean columns, plus accuracy
    columns = ['Training', 'Dataset', 'Model', 'N'] + method_names + [f'{m}_mean' for m in method_names] + ['accuracy', 'accuracy_mean']
    df = pd.DataFrame(rows)
    
    # Ensure all columns exist (fill missing with NaN)
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    
    # Reorder columns - only include columns that exist
    existing_columns = [col for col in columns if col in df.columns]
    df = df[existing_columns]
    
    return df


def find_best_method_per_row(data: Dict[Tuple[str, str, str], Dict[str, List[float]]]) -> Dict[int, str]:
    """
    Find the method with lowest mean ECE for each configuration.
    
    Args:
        data: Collected data dictionary
    
    Returns:
        Dictionary mapping row index to best method name
    """
    best_methods = {}
    method_names = ['uncalibrated', 'ts', 'dac', 'sgc', 'geo_comb', 'rand_coords', 
                    'grad_coords', 'geo_dac_feat', 'dac_geo_feat', 'dac_coord_feat', 'metric_guided']
    
    for idx, ((training, dataset, model), method_data) in enumerate(sorted(data.items())):
        best_method = None
        best_mean = float('inf')
        
        for method_name in method_names:
            if method_name in method_data and method_data[method_name]:
                n, mean, _ = calculate_statistics(method_data[method_name])
                if not np.isnan(mean) and mean < best_mean:
                    best_mean = mean
                    best_method = method_name
        
        if best_method:
            best_methods[idx] = best_method
    
    return best_methods


def generate_latex_table(df: pd.DataFrame, best_methods: Dict[int, str], output_file: Path):
    """
    Generate LaTeX table with best method highlighted.
    
    Args:
        df: Summary DataFrame
        best_methods: Dictionary mapping row index to best method name
        output_file: Path to output LaTeX file
    """
    # Method name mapping for display
    method_display_names = {
        'uncalibrated': 'Uncalibrated',
        'ts': 'TS',
        'dac': 'DAC',
        'sgc': 'SGC (Standard)',
        'geo_comb': 'Geo Comb',
        'rand_coords': 'Rand Coords',
        'grad_coords': 'Grad Coords',
        'geo_dac_feat': 'Geo+DAC Feat',
        'dac_geo_feat': 'DAC+Geo Feat',
        'dac_coord_feat': 'DAC+Coord Feat',
        'metric_guided': 'Metric Guided',
    }
    
    with open(output_file, 'w') as f:
        f.write("\\begin{table*}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{l l l c")
        
        # Add column specifiers for each method
        for _ in range(len(df.columns) - 4):  # -4 for Training, Dataset, Model, N
            f.write(" c")
        f.write("}\n")
        f.write("\\toprule\n")
        
        # Header row
        headers = ['Training', 'Dataset', 'Model', 'N'] + [method_display_names.get(col, col) 
                                                           for col in df.columns[4:]]
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        
        # Data rows
        for idx, row in df.iterrows():
            row_data = []
            row_data.append(str(row['Training']))
            row_data.append(str(row['Dataset']))
            row_data.append(str(row['Model']))
            row_data.append(str(int(row['N'])))
            
            # Add method columns with bold for best method
            for col in df.columns[4:]:
                value = str(row[col])
                if idx in best_methods and best_methods[idx] == col:
                    # Bold the best method
                    value = f"\\textbf{{{value}}}"
                row_data.append(value)
            
            f.write(" & ".join(row_data) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Comprehensive calibration comparison across methods. Values show Mean ECE (\\%) ± 95\\% CI. Best method per row is bolded.}\n")
        f.write("\\label{tab:comprehensive_comparison}\n")
        f.write("\\end{table*}\n")


def calculate_win_rates(data: Dict[Tuple[str, str, str], Dict[str, List[float]]]) -> Dict[str, Tuple[int, float]]:
    """
    Calculate win rates (how often each method achieves lowest ECE).
    
    Args:
        data: Collected data dictionary
    
    Returns:
        Dictionary mapping method name to (win_count, win_rate_percent)
    """
    method_names = ['uncalibrated', 'ts', 'dac', 'sgc', 'geo_comb', 'rand_coords', 
                    'grad_coords', 'geo_dac_feat', 'dac_geo_feat', 'dac_coord_feat', 'metric_guided']
    
    win_counts = {method: 0 for method in method_names}
    total_configs = len(data)
    
    for (training, dataset, model), method_data in data.items():
        best_method = None
        best_mean = float('inf')
        
        for method_name in method_names:
            if method_name in method_data and method_data[method_name]:
                n, mean, _ = calculate_statistics(method_data[method_name])
                if not np.isnan(mean) and mean < best_mean:
                    best_mean = mean
                    best_method = method_name
        
        if best_method:
            win_counts[best_method] += 1
    
    win_rates = {
        method: (win_counts[method], (win_counts[method] / total_configs * 100) if total_configs > 0 else 0.0)
        for method in method_names
    }
    
    return win_rates


def print_win_rate_summary(win_rates: Dict[str, Tuple[int, float]]):
    """
    Print textual summary of win rates.
    
    Args:
        win_rates: Dictionary from calculate_win_rates
    """
    method_display_names = {
        'uncalibrated': 'Uncalibrated',
        'ts': 'Temperature Scaling',
        'dac': 'DAC',
        'sgc': 'SGC (Standard)',
        'geo_comb': 'Geometric Combined',
        'rand_coords': 'Random Coordinates',
        'grad_coords': 'Gradient Coordinates',
        'geo_dac_feat': 'Geometric + DAC Features',
        'dac_geo_feat': 'DAC + Geometric Features',
        'dac_coord_feat': 'DAC + Coordinate Features',
        'metric_guided': 'Metric Guided',
    }
    
    # Sort by win count (descending)
    sorted_methods = sorted(win_rates.items(), key=lambda x: x[1][0], reverse=True)
    
    print("\n" + "="*80)
    print("WIN RATE ANALYSIS")
    print("="*80)
    print(f"{'Method':<30} {'Wins':<10} {'Win Rate (%)':<15}")
    print("-"*80)
    
    for method, (wins, rate) in sorted_methods:
        display_name = method_display_names.get(method, method)
        print(f"{display_name:<30} {wins:<10} {rate:.2f}%")
    
    print("="*80)


def generate_cross_entropy_table(df: pd.DataFrame, best_methods: Dict[int, str], output_file: Path):
    """
    Generate LaTeX table filtered to baseline_cross_entropy training method only.
    Similar to generate_main_results_table in generate_sgc_tables.py
    
    Args:
        df: Summary DataFrame
        best_methods: Dictionary mapping row index to best method name
        output_file: Path to output LaTeX file
    """
    # Filter to baseline_cross_entropy only and exclude SVHN
    df_filtered = df[(df['Training'] == 'baseline_cross_entropy') & (df['Dataset'] != 'svhn')].copy()
    
    if df_filtered.empty:
        logger.warning("No baseline_cross_entropy data found!")
        return
    
    # Method name mapping for display (matching table_1_main_results.tex format)
    method_display_names = {
        'uncalibrated': 'Uncal',
        'ts': 'TS',
        'platt': 'Platt',
        'isotonic': 'Isotonic',
        'beta': 'Beta',
        'dac': 'DAC',
        'pixel_geo': 'Pixel Geo',
        'sgc_dac_layers': 'SGC(DAC)',
        'sgc': 'SGC',
        'geo_comb': 'Geo Comb',
        'rand_coords': 'Coord',
        'grad_coords': 'Grad Coords',
        'dac_coord_feat': 'DAC+Coord',
        'geo_dac_feat': 'Geo+DAC Feat',
        'dac_geo_feat': 'DAC+Geo Feat',
        'metric_guided': 'Metric Guided',
    }
    
    # Define column order matching table_1_main_results.tex
    column_order = ['uncalibrated', 'ts', 'platt', 'isotonic', 'beta', 'dac', 'pixel_geo', 
                    'sgc_dac_layers', 'sgc', 'geo_comb', 'rand_coords']
    
    # Filter to only columns that exist in the dataframe and have at least one non-N/A value
    available_columns = []
    for col in column_order:
        if col in df_filtered.columns:
            # Check if at least one row has non-N/A data
            if df_filtered[col].notna().any():
                available_columns.append(col)
    
    with open(output_file, 'w') as f:
        f.write("% Table: Main Results Comparison (baseline_cross_entropy only)\n")
        f.write("% Required packages: \\usepackage{booktabs}, \\usepackage{xcolor}\n")
        f.write("\\begin{table*}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        
        # Build column specifiers: Dataset, Model, then methods, then Acc, Seeds
        num_method_cols = len(available_columns)
        f.write("\\begin{tabular}{ll")
        for _ in range(num_method_cols):
            f.write("c")
        f.write("cc}\n")
        f.write("\\toprule\n")
        
        # Header row
        headers = ['Dataset', 'Model'] + [method_display_names.get(col, col) for col in available_columns] + ['Acc', 'Seeds']
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        
        # Sort by dataset, then model
        df_filtered = df_filtered.sort_values(['Dataset', 'Model'])
        
        current_dataset = None
        for idx, row in df_filtered.iterrows():
            dataset = row['Dataset']
            model = row['Model']
            
            # Group rows by dataset
            is_new_dataset = (current_dataset != dataset)
            if is_new_dataset:
                if current_dataset is not None:
                    f.write("\\midrule\n")
                current_dataset = dataset
            
            # Format dataset name (uppercase, only show on first row of group)
            dataset_display = dataset.upper() if is_new_dataset else ""
            
            # Find best, 2nd, 3rd methods for this row (using raw mean values)
            method_values = {}
            for col in available_columns:
                mean_col = f'{col}_mean'
                if mean_col in row and pd.notna(row[mean_col]):
                    method_values[col] = float(row[mean_col])
            
            # Sort methods by value
            sorted_methods = sorted(method_values.items(), key=lambda x: x[1])
            best_method = sorted_methods[0][0] if len(sorted_methods) > 0 else None
            second_method = sorted_methods[1][0] if len(sorted_methods) > 1 else None
            third_method = sorted_methods[2][0] if len(sorted_methods) > 2 else None
            
            # Collect method values
            row_data = [dataset_display, model]
            
            # Add method columns with coloring
            for col in available_columns:
                value = str(row[col])
                # Color code: best=blue, 2nd=teal, 3rd=olive
                if col == best_method:
                    value = f"\\textcolor{{blue}}{{\\textbf{{{value}}}}}"
                elif col == second_method:
                    value = f"\\textcolor{{teal}}{{\\textbf{{{value}}}}}"
                elif col == third_method:
                    value = f"\\textcolor{{olive}}{{\\textbf{{{value}}}}}"
                row_data.append(value)
            
            # Add accuracy and seeds (if available)
            if 'accuracy' in df_filtered.columns and pd.notna(row.get('accuracy')):
                acc_value = str(row['accuracy'])
            else:
                acc_value = 'N/A'
            seeds_value = str(int(row['N'])) if 'N' in df_filtered.columns and pd.notna(row['N']) else 'N/A'
            row_data.extend([acc_value, seeds_value])
            
            f.write(" & ".join(row_data) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Main calibration comparison for baseline\\_cross\\_entropy training method. Values show Mean ECE (\\%) ± 95\\% CI. Best method per row is bolded.}\n")
        f.write("\\label{tab:cross_entropy_results}\n")
        f.write("\\end{table*}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate and analyze calibration experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing JSON result files'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default='comprehensive_analysis.csv',
        help='Output CSV file path (default: comprehensive_analysis.csv)'
    )
    parser.add_argument(
        '--output-latex',
        type=str,
        default='comprehensive_analysis.tex',
        help='Output LaTeX file path (default: comprehensive_analysis.tex)'
    )
    parser.add_argument(
        '--output-latex-cross-entropy',
        type=str,
        default=None,
        help='Output LaTeX file path for baseline_cross_entropy filtered table (default: None)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Collect data
    print(f"Collecting data from {input_dir}...")
    data = collect_data_from_directory(input_dir)
    
    if not data:
        print("No data collected. Check that JSON files exist and have the expected structure.")
        return
    
    print(f"Collected data for {len(data)} unique configurations")
    
    # Create summary table
    print("Creating summary table...")
    df = create_summary_table(data)
    
    # Find best methods per row
    best_methods = find_best_method_per_row(data)
    
    # Save CSV
    csv_path = Path(args.output_csv)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")
    
    # Generate LaTeX
    latex_path = Path(args.output_latex)
    generate_latex_table(df, best_methods, latex_path)
    print(f"Saved LaTeX table to {latex_path}")
    
    # Generate filtered LaTeX table for baseline_cross_entropy
    if args.output_latex_cross_entropy:
        cross_entropy_latex_path = Path(args.output_latex_cross_entropy)
        generate_cross_entropy_table(df, best_methods, cross_entropy_latex_path)
        print(f"Saved baseline_cross_entropy LaTeX table to {cross_entropy_latex_path}")
    
    # Calculate and print win rates
    win_rates = calculate_win_rates(data)
    print_win_rate_summary(win_rates)
    
    # ===== STATISTICAL COMPARISONS =====
    print("\n" + "="*80)
    print("PERFORMING STATISTICAL COMPARISONS")
    print("="*80)
    
    # Define comparison groups: (method1, method2, description)
    # Same algorithm, different features
    comparison_groups = [
        # Geometric algorithm comparisons
        ('sgc', 'geo_comb', 'Geometric: SGC vs Geo Comb (different feature extraction)'),
        ('sgc', 'rand_coords', 'Geometric: SGC vs Random Coords (layer-based vs coordinate-based)'),
        ('geo_comb', 'rand_coords', 'Geometric: Geo Comb vs Random Coords'),
        
        # DAC algorithm comparisons
        ('dac', 'dac_geo_feat', 'DAC: Original vs DAC with Geometric Features'),
        ('dac', 'dac_coord_feat', 'DAC: Original vs DAC with Random Coordinates'),
        
        # Cross-algorithm feature comparisons
        ('sgc', 'geo_dac_feat', 'Geometric: SGC vs Geo+DAC Features (SPP+JL vs spatial avg+L2)'),
    ]
    
    comparison_results = perform_statistical_comparisons(data, comparison_groups)
    print_statistical_comparisons(comparison_results)
    
    # Save comparison results to JSON
    comparison_json_path = Path(args.output_csv).with_suffix('.comparisons.json')
    with open(comparison_json_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    print(f"\nSaved comparison results to {comparison_json_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total configurations: {len(data)}")
    print(f"Methods analyzed: {len([m for m in df.columns if m not in ['Training', 'Dataset', 'Model', 'N']])}")
    print("="*80)


if __name__ == "__main__":
    main()

