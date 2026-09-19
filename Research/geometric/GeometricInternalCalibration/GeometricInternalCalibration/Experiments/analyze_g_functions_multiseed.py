import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import interp1d
from scipy import stats


KNOWN_DATASETS = {"cifar10", "cifar100", "tiny_imagenet", "tinyimagenet"}


def reconstruct_isotonic_func(params):
    """Reconstructs the isotonic function from saved parameters."""
    if not params:
        return None
    
    # Handle both structures:
    # 1. Direct: params['isotonic_params'] (from regular geometric calibrator)
    # 2. Nested: params['calibrator_params']['isotonic_params'] (from FAISS methods, if nested)
    iso = None
    if isinstance(params, dict):
        if 'isotonic_params' in params:
            iso = params['isotonic_params']
        elif 'calibrator_params' in params:
            cal_params = params['calibrator_params']
            if isinstance(cal_params, dict) and 'isotonic_params' in cal_params:
                iso = cal_params['isotonic_params']
    
    if not iso or not isinstance(iso, dict) or 'X_thresholds' not in iso or 'y_thresholds' not in iso:
        return None

    x_thresholds = np.array(iso['X_thresholds'])
    y_thresholds = np.array(iso['y_thresholds'])
    y_min = iso.get('y_min', y_thresholds[0] if len(y_thresholds) > 0 else 0.0)
    y_max = iso.get('y_max', y_thresholds[-1] if len(y_thresholds) > 0 else 1.0)

    # Create interpolator
    func = interp1d(
        x_thresholds,
        y_thresholds,
        kind='linear',
        bounds_error=False,
        fill_value=(y_min, y_max)
    )
    return func


def parse_config_from_filename(filename):
    """
    Parse (training_method, dataset, model_name) from filenames like:
    ablation_{training_method}_{dataset}_{model_name}_seed{seed}.json

    Handles underscores in training_method/dataset/model_name by using
    a known dataset list; falls back to rsplit if no dataset is found.
    """
    base = os.path.basename(filename)
    if not base.startswith("ablation_") or "_seed" not in base:
        return None

    core = base[len("ablation_"):]          # remove 'ablation_'
    core = core.split("_seed")[0]           # drop everything from '_seed' onwards

    parts = core.split("_")

    # Try to locate dataset token using KNOWN_DATASETS
    for i in range(1, len(parts) - 1):
        # Try single-token dataset
        if parts[i] in KNOWN_DATASETS:
            training_method = "_".join(parts[:i])
            dataset = parts[i]
            model_name = "_".join(parts[i + 1:])
            return training_method, dataset, model_name

        # Try two-token dataset like "tiny_imagenet"
        if i + 1 < len(parts):
            candidate = "_".join(parts[i:i + 2])
            if candidate in KNOWN_DATASETS:
                training_method = "_".join(parts[:i])
                dataset = candidate
                model_name = "_".join(parts[i + 2:])
                return training_method, dataset, model_name

    # Fallback: original rsplit behavior (may mislabel but still groups consistently)
    fallback = core.rsplit("_", 2)
    if len(fallback) == 3:
        return tuple(fallback)
    return None


def load_all_runs(results_dir):
    """
    Loads all JSON results and groups them by
    (training_method, dataset, model_name).
    Returns both calibrator_params and timing data.
    """
    pattern = os.path.join(results_dir, "ablation_*.json")
    files = glob.glob(pattern)
    print(f"📂 Found {len(files)} result files in {results_dir}")

    # groups[(training_method, dataset, model_name)] = {
    #     'SGC (Layers)': {'params': [...], 'timing': [...]},
    #     'Coordinate': {'params': [...], 'timing': [...]},
    #     'SGC (DAC Layers)': {'params': [...], 'timing': [...]}
    # }
    groups = {}

    for filepath in files:
        cfg = parse_config_from_filename(filepath)
        if cfg is None:
            print(f"⚠️ Could not parse config from filename: {filepath}")
            continue

        key = cfg
        if key not in groups:
            groups[key] = {
                'SGC (Layers)': {'params': [], 'timing': [], 'extraction_timing': []},
                'Coordinate': {'params': [], 'timing': [], 'extraction_timing': []},
                'SGC (DAC Layers)': {'params': [], 'timing': [], 'extraction_timing': []},
                'SGC-FAISS': {'params': [], 'timing': [], 'extraction_timing': []},
                'Coordinate-FAISS': {'params': [], 'timing': [], 'extraction_timing': []},
                'Coordinate-SPP-FAISS': {'params': [], 'timing': [], 'extraction_timing': []},
                'DAC': {'params': [], 'timing': [], 'extraction_timing': []},
                'DAC (Coord)': {'params': [], 'timing': [], 'extraction_timing': []}
            }

        methods = groups[key]

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Helper function to extract data with ECE
            def extract_method_data(method_key, json_key):
                if json_key in data:
                    method_data = data[json_key]
                    
                    # Skip if this method was skipped
                    if isinstance(method_data, dict) and method_data.get('skipped', False):
                        return
                    
                    params = None
                    ece_val = method_data.get('ece', None)
                    
                    # Extract params - handle both structures
                    if 'calibrator_params' in method_data:
                        params = method_data['calibrator_params']
                    elif 'isotonic_params' in method_data:
                        # Old format: isotonic_params directly in the method dict
                        params = method_data
                    
                    if params is not None:
                        methods[method_key]['params'].append({
                            'params': params,
                            'ece': ece_val
                        })
                    
                    # Extract calibration timing
                    if 'calibrate_time_s' in method_data:
                        methods[method_key]['timing'].append(method_data['calibrate_time_s'])
                    
                    # Extract extraction timing (handle missing values)
                    extraction_time = method_data.get('extraction_time_s', None)
                    if extraction_time is not None:
                        methods[method_key]['extraction_timing'].append(extraction_time)
                    else:
                        # Default to 0.0 if missing
                        methods[method_key]['extraction_timing'].append(0.0)

            # Extract all methods using helper
            extract_method_data('SGC (Layers)', 'global_random')
            extract_method_data('Coordinate', 'coordinate_sampling')
            extract_method_data('SGC (DAC Layers)', 'sgc_with_dac_layers')
            extract_method_data('SGC-FAISS', 'sgc_faiss')
            extract_method_data('Coordinate-FAISS', 'coordinate_sampling_faiss')
            extract_method_data('Coordinate-SPP-FAISS', 'coordinate_spp_faiss')
            
            # Extract DAC - check multiple possible locations
            # Try standard_baselines.density_aware_calibration first
            dac_data = None
            dac_source = None
            if 'standard_baselines' in data and isinstance(data['standard_baselines'], dict):
                if 'density_aware_calibration' in data['standard_baselines']:
                    dac_data = data['standard_baselines']['density_aware_calibration']
                    dac_source = 'standard_baselines'
            # Fallback to old format baselines.density_aware_calibration
            if dac_data is None and 'baselines' in data and isinstance(data['baselines'], dict):
                if 'density_aware_calibration' in data['baselines']:
                    dac_data = data['baselines']['density_aware_calibration']
                    dac_source = 'baselines'
            # Fallback to top-level dac_standalone
            if dac_data is None and 'dac_standalone' in data:
                dac_data = data['dac_standalone']
                dac_source = 'dac_standalone'
            
            # Extract DAC timing data if found
            if dac_data and isinstance(dac_data, dict) and not dac_data.get('skipped', False):
                # Extract calibration timing - prioritize calibrate_time_s, fallback to timing_seconds
                calibration_time = None
                if 'calibrate_time_s' in dac_data:
                    calibration_time = dac_data['calibrate_time_s']
                elif 'timing_seconds' in dac_data:
                    # timing_seconds is used in old format (this is total time, not just calibration)
                    # For old format, we'll use timing_seconds as calibration time and assume extraction is 0
                    calibration_time = dac_data['timing_seconds']
                
                # Always append timing - must have a value for the method to appear in plots
                if calibration_time is not None and calibration_time > 0:
                    methods['DAC']['timing'].append(calibration_time)
                    
                    # Extract extraction timing (must keep lists in sync)
                    extraction_time = dac_data.get('extraction_time_s', None)
                    if extraction_time is not None and extraction_time >= 0:
                        methods['DAC']['extraction_timing'].append(extraction_time)
                    else:
                        # Default to 0.0 if missing (old format doesn't have extraction_time_s)
                        methods['DAC']['extraction_timing'].append(0.0)
            
            # Extract DAC with Coordinate Features
            if 'dac_with_coordinate_features' in data:
                dac_coord_data = data['dac_with_coordinate_features']
                if isinstance(dac_coord_data, dict) and not dac_coord_data.get('skipped', False):
                    # Extract calibration timing
                    if 'calibrate_time_s' in dac_coord_data:
                        methods['DAC (Coord)']['timing'].append(dac_coord_data['calibrate_time_s'])
                    # Extract extraction timing
                    extraction_time = dac_coord_data.get('extraction_time_s', None)
                    if extraction_time is not None:
                        methods['DAC (Coord)']['extraction_timing'].append(extraction_time)
                    else:
                        # Default to 0.0 if missing
                        methods['DAC (Coord)']['extraction_timing'].append(0.0)

        except Exception as e:
            print(f"⚠️ Error reading {filepath}: {e}")

    return groups


def compute_function_distances(curves_a, curves_b, x_grid):
    """
    Compute multiple distance metrics between two sets of curves.
    Returns dict with mean, std, and CI for each metric.
    """
    n_pairs = min(curves_a.shape[0], curves_b.shape[0])
    if n_pairs == 0:
        return None, 0
    
    curves_a = curves_a[:n_pairs]
    curves_b = curves_b[:n_pairs]
    
    results = {}
    
    # 1. RMSE (you already have this)
    diffs = curves_a - curves_b
    rmse_per_seed = np.sqrt(np.mean(diffs ** 2, axis=1))
    results['rmse'] = {
        'mean': float(np.mean(rmse_per_seed)),
        'std': float(np.std(rmse_per_seed)),
        'ci_95': float(1.96 * np.std(rmse_per_seed) / np.sqrt(n_pairs))
    }
    
    # 2. Maximum Absolute Deviation (supremum norm)
    max_dev_per_seed = np.max(np.abs(diffs), axis=1)
    results['max_deviation'] = {
        'mean': float(np.mean(max_dev_per_seed)),
        'std': float(np.std(max_dev_per_seed)),
        'ci_95': float(1.96 * np.std(max_dev_per_seed) / np.sqrt(n_pairs))
    }
    
    # 3. Area between curves (L1 distance, integral approximation)
    dx = x_grid[1] - x_grid[0]
    area_per_seed = np.sum(np.abs(diffs), axis=1) * dx
    results['area_between'] = {
        'mean': float(np.mean(area_per_seed)),
        'std': float(np.std(area_per_seed)),
        'ci_95': float(1.96 * np.std(area_per_seed) / np.sqrt(n_pairs))
    }
    
    # 4. Correlation between curves (should be high if similar shape)
    correlations = []
    for i in range(n_pairs):
        corr = np.corrcoef(curves_a[i], curves_b[i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    if correlations:
        correlations = np.array(correlations)
        results['correlation'] = {
            'mean': float(np.mean(correlations)),
            'std': float(np.std(correlations)),
            'ci_95': float(1.96 * np.std(correlations) / np.sqrt(len(correlations)))
        }
    else:
        results['correlation'] = {
            'mean': float('nan'),
            'std': float('nan'),
            'ci_95': float('nan')
        }
    
    return results, n_pairs


def perform_significance_tests(curves_a, curves_b):
    """
    Statistical tests to determine if methods produce significantly different functions.
    """
    n_pairs = min(curves_a.shape[0], curves_b.shape[0])
    if n_pairs < 3:
        return {'error': 'Not enough samples for significance test'}
    
    curves_a = curves_a[:n_pairs]
    curves_b = curves_b[:n_pairs]
    
    # Compute per-seed mean difference
    mean_diff_per_seed = np.mean(curves_a - curves_b, axis=1)
    
    results = {}
    
    # Paired t-test: is the mean difference significantly different from 0?
    t_stat, p_value = stats.ttest_1samp(mean_diff_per_seed, 0)
    results['paired_ttest'] = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant_at_0.05': p_value < 0.05
    }
    
    # Wilcoxon signed-rank test (non-parametric alternative)
    if n_pairs >= 6:
        try:
            w_stat, w_pvalue = stats.wilcoxon(mean_diff_per_seed)
            results['wilcoxon'] = {
                'statistic': float(w_stat),
                'p_value': float(w_pvalue),
                'significant_at_0.05': w_pvalue < 0.05
            }
        except ValueError:
            results['wilcoxon'] = {
                'error': 'All differences are zero'
            }
    
    return results


def check_normalization_consistency(params_list):
    """
    Check if normalization parameters are consistent across seeds.
    If not, the G functions are not directly comparable.
    """
    def extract_stability_value(p, key):
        """Extract stability_p5 or stability_p95 from params, checking both direct and nested locations."""
        # Handle new format: dict with 'params' key
        if isinstance(p, dict) and 'params' in p:
            p = p['params']
        
        if not p or not isinstance(p, dict):
            return None
        # Check direct location
        if key in p:
            return p[key]
        # Check inside isotonic_params
        if 'isotonic_params' in p and isinstance(p['isotonic_params'], dict):
            return p['isotonic_params'].get(key)
        # Check inside calibrator_params -> isotonic_params
        if 'calibrator_params' in p and isinstance(p['calibrator_params'], dict):
            cal_params = p['calibrator_params']
            if 'isotonic_params' in cal_params and isinstance(cal_params['isotonic_params'], dict):
                return cal_params['isotonic_params'].get(key)
        return None
    
    p5_values = [extract_stability_value(p, 'stability_p5') for p in params_list]
    p95_values = [extract_stability_value(p, 'stability_p95') for p in params_list]
    
    # Filter out None values
    p5_values = [v for v in p5_values if v is not None]
    p95_values = [v for v in p95_values if v is not None]
    
    # Use defaults if no values found
    if not p5_values:
        p5_values = [0.0]
    if not p95_values:
        p95_values = [1.0]
    
    p5_std = np.std(p5_values) if len(p5_values) > 1 else 0
    p95_std = np.std(p95_values) if len(p95_values) > 1 else 0
    
    return {
        'p5_mean': float(np.mean(p5_values)) if p5_values else None,
        'p5_std': float(p5_std),
        'p95_mean': float(np.mean(p95_values)) if p95_values else None,
        'p95_std': float(p95_std),
        'consistent': p5_std < 0.01 and p95_std < 0.01
    }


def plot_latency_comparison(timing_stats, output_base_filename):
    """
    Generates FOUR separate latency plots:
    1. Standard Implementations - End-to-End (extraction + calibration)
    2. Standard Implementations - Online Only (calibration only)
    3. FAISS Implementations - End-to-End
    4. FAISS Implementations - Online Only
    
    Args:
        timing_stats: Dictionary with timing statistics (in seconds, will be converted to ms)
                      Must contain both 'total_timing' and 'calibration_timing'
        output_base_filename: Base filename for output (will be modified with suffixes)
    """
    # Extract both total timing and calibration timing data
    total_timing = timing_stats.get('total_timing', {})
    calibration_timing = timing_stats.get('calibration_timing', {})
    extraction_timing_stats = timing_stats.get('extraction_timing', {})
    
    if not total_timing and not calibration_timing:
        print("⚠️  No timing data available for latency plot")
        return
    
    # Define groups
    standard_methods = ['SGC (Layers)', 'Coordinate', 'SGC (DAC Layers)']
    faiss_methods = ['SGC-FAISS', 'Coordinate-FAISS', 'Coordinate-SPP-FAISS', 'DAC', 'DAC (Coord)']
    
    # Helper to plot a single group
    def plot_group(methods_list, timing_data, title_suffix, phase_name, filename_suffix, use_throughput=False):
        names = []
        means = []
        stds = []
        speedups = []
        
        # Collect data for methods that exist in timing_data
        available_methods = [m for m in methods_list if m in timing_data]
        if not available_methods:
            print(f"⚠️  No data available for {title_suffix} methods ({phase_name})")
            return
        
        # Convert units based on use_throughput flag
        if use_throughput:
            # Assume 10000 test samples (CIFAR test set size)
            test_set_size = 10000
            # timing_data is in seconds, convert to samples/sec
            converted_data = {}
            for m in available_methods:
                if timing_data[m]['mean'] > 0:
                    converted_data[m] = {
                        'mean': test_set_size / timing_data[m]['mean'],
                        'std': test_set_size * timing_data[m]['std'] / (timing_data[m]['mean'] ** 2)  # Error propagation
                    }
                else:
                    converted_data[m] = {'mean': 0, 'std': 0}
            # Sort by throughput (highest to lowest)
            available_methods = sorted(available_methods, key=lambda m: converted_data[m]['mean'], reverse=True)
            xlabel = 'Throughput (samples/second)'
        else:
            # For e2e, use seconds (not milliseconds)
            converted_data = {m: timing_data[m] for m in available_methods}
            # Sort by time (fastest to slowest)
            available_methods = sorted(available_methods, key=lambda m: converted_data[m]['mean'])
            xlabel = 'Total Time (seconds)'
        
        # Find reference method for speedup calculation
        if use_throughput:
            # For throughput, compare vs slowest (lowest throughput)
            reference_method = available_methods[-1] if available_methods else None
            reference_value = converted_data[reference_method]['mean'] if reference_method else 1.0
        else:
            # For time, compare vs slowest (highest time)
            reference_method = available_methods[-1] if available_methods else None
            reference_value = converted_data[reference_method]['mean'] if reference_method else 1.0
        
        # Collect data
        for m in available_methods:
            names.append(m)
            means.append(converted_data[m]['mean'])
            stds.append(converted_data[m]['std'])
            
            # Speedup relative to reference method
            current_mean = converted_data[m]['mean']
            if current_mean > 0 and reference_value > 0:
                if use_throughput:
                    # For throughput, higher is better: current / slowest
                    speedup = current_mean / reference_value
                    speedups.append(f"{speedup:.1f}x vs slowest")
                else:
                    # For time, lower is better: slowest / current
                    speedup = reference_value / current_mean
                    speedups.append(f"{speedup:.1f}x vs slowest")
            else:
                speedups.append("N/A")
        
        if not names:
            return
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.8)))
        y_pos = np.arange(len(names))
        
        # Create horizontal bars with error bars
        bars = ax.barh(y_pos, means, xerr=stds, capsize=5, align='center', 
                       color='steelblue', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=12)
        
        ax.set_title(f'Latency Comparison: {title_suffix}\n({phase_name})', fontsize=14, fontweight='bold')
        
        # Add labels to bars
        max_mean = max(means) if means else 1
        spacing = max_mean * 0.02  # Spacing between labels
        for i, bar in enumerate(bars):
            bar_width = bar.get_width()
            bar_y = bar.get_y() + bar.get_height() / 2
            
            # Format value text based on unit
            if use_throughput:
                value_text = f"{means[i]:.0f} samples/s"
            else:
                value_text = f"{means[i]:.2f} s"
            
            if bar_width > max_mean * 0.15:  # If bar is wide enough, put text inside
                ax.text(bar_width * 0.95, bar_y, 
                        value_text, 
                        va='center', ha='right', color='white', fontsize=10, fontweight='bold')
                # Speedup text outside bar (above)
                ax.text(bar_width + spacing, bar_y, 
                        f" {speedups[i]}", 
                        va='center', fontweight='bold', color='black', fontsize=10)
            else:  # Otherwise, put both outside
                # Absolute value text outside (below)
                ax.text(bar_width + spacing, bar_y - 0.15, 
                        value_text, 
                        va='center', ha='left', color='black', fontsize=9)
                # Speedup text outside (above)
                ax.text(bar_width + spacing, bar_y + 0.15, 
                        f" {speedups[i]}", 
                        va='center', fontweight='bold', color='black', fontsize=10)
        
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_base_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save
        save_path = output_base_filename.replace('.png', f'_{filename_suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Latency plot saved to: {save_path}")
        plt.close()
    
    # Temporary fix for missing FAISS extraction times (until experiments are rerun)
    # Borrow extraction time from non-FAISS equivalents if missing
    faiss_to_standard = {
        'SGC-FAISS': 'SGC (Layers)',
        'Coordinate-FAISS': 'Coordinate',
        'Coordinate-SPP-FAISS': 'Coordinate'  # Approximate
    }
    
    # Compute extraction timing from total - calibration for each method
    # Use passed extraction_timing_stats if available, otherwise compute from difference
    extraction_means = {}
    extraction_stds = {}
    for method in list(total_timing.keys()) + list(calibration_timing.keys()):
        if method in extraction_timing_stats:
            # Use provided extraction timing if available
            extraction_means[method] = extraction_timing_stats[method].get('mean', 0.0)
            extraction_stds[method] = extraction_timing_stats[method].get('std', 0.0)
        elif method in total_timing and method in calibration_timing:
            # Compute from difference if not provided
            extraction_means[method] = total_timing[method]['mean'] - calibration_timing[method]['mean']
            # Approximate std as sqrt of sum of variances (assuming independence)
            extraction_stds[method] = np.sqrt(
                max(0, total_timing[method]['std']**2 - calibration_timing[method]['std']**2)
            )
        elif method in total_timing:
            # If only total timing available, assume extraction is 0
            extraction_means[method] = 0.0
            extraction_stds[method] = 0.0
    
    # Apply temporary fix for FAISS methods
    for faiss_method, standard_method in faiss_to_standard.items():
        if faiss_method in extraction_means and standard_method in extraction_means:
            if extraction_means[faiss_method] < 1.0:  # Less than 1 second means likely missing
                extraction_means[faiss_method] = extraction_means[standard_method]
                extraction_stds[faiss_method] = extraction_stds.get(standard_method, 0.0)
                # Update total_timing if it exists
                if faiss_method in total_timing and standard_method in total_timing:
                    # Recompute total time with borrowed extraction time
                    if faiss_method in calibration_timing:
                        total_timing[faiss_method] = {
                            'mean': extraction_means[faiss_method] + calibration_timing[faiss_method]['mean'],
                            'std': np.sqrt(extraction_stds[faiss_method]**2 + calibration_timing[faiss_method]['std']**2)
                        }
    
    # Generate the four plots
    # End-to-end plots (seconds)
    if total_timing:
        plot_group(standard_methods, total_timing, "Standard Implementations", 
                   "End-to-End: Extraction + Calibration", "standard_e2e", use_throughput=False)
        plot_group(faiss_methods, total_timing, "FAISS Accelerated", 
                   "End-to-End: Extraction + Calibration", "faiss_e2e", use_throughput=False)
    
    # Online-only plots (samples/second)
    if calibration_timing:
        plot_group(standard_methods, calibration_timing, "Standard Implementations", 
                   "Online Phase Only", "standard_online", use_throughput=True)
        plot_group(faiss_methods, calibration_timing, "FAISS Accelerated", 
                   "Online Phase Only", "faiss_online", use_throughput=True)


def generate_timing_comparison_table_faiss(groups, training_method_filter='baseline_cross_entropy', output_file="latex_tables_statistics/table_timing_sgc_faiss_vs_sgc_lite_faiss.tex"):
    """
    Generate a LaTeX table comparing timing between SGC-FAISS and SGC Lite-FAISS (Coordinate-FAISS).
    
    Args:
        groups: Dictionary of grouped results from load_all_runs
        training_method_filter: Filter to specific training method (default: 'baseline_cross_entropy')
        output_file: Path to output LaTeX file
    """
    # Filter groups
    filtered_groups = {
        (tm, ds, mn): methods_data
        for (tm, ds, mn), methods_data in groups.items()
        if tm == training_method_filter and ds != 'svhn'
    }
    
    if not filtered_groups:
        print("No groups found for FAISS timing comparison table")
        return
    
    # Collect timing data for each (dataset, model) combination
    timing_data = []
    
    for (training_method, dataset, model_name), methods_data in filtered_groups.items():
        sgc_data = methods_data.get('SGC-FAISS', {})
        coord_data = methods_data.get('Coordinate-FAISS', {})
        
        # Extract timing lists
        sgc_extraction = sgc_data.get('extraction_timing', [])
        sgc_calibration = sgc_data.get('timing', [])
        coord_extraction = coord_data.get('extraction_timing', [])
        coord_calibration = coord_data.get('timing', [])
        
        # Filter valid values
        sgc_extraction = [t for t in sgc_extraction if t is not None and t >= 0]
        sgc_calibration = [t for t in sgc_calibration if t is not None and t > 0]
        coord_extraction = [t for t in coord_extraction if t is not None and t >= 0]
        coord_calibration = [t for t in coord_calibration if t is not None and t > 0]
        
        if not sgc_calibration or not coord_calibration:
            continue  # Skip if no timing data
        
        # Compute statistics
        def compute_stats(values):
            if not values:
                return None, None, 0
            arr = np.array(values)
            return float(np.mean(arr)), float(np.std(arr)), len(arr)
        
        sgc_ext_mean, sgc_ext_std, sgc_ext_n = compute_stats(sgc_extraction)
        sgc_cal_mean, sgc_cal_std, sgc_cal_n = compute_stats(sgc_calibration)
        coord_ext_mean, coord_ext_std, coord_ext_n = compute_stats(coord_extraction)
        coord_cal_mean, coord_cal_std, coord_cal_n = compute_stats(coord_calibration)
        
        # Total time = extraction + calibration
        sgc_total_mean = (sgc_ext_mean or 0.0) + (sgc_cal_mean or 0.0)
        sgc_total_std = np.sqrt((sgc_ext_std or 0.0)**2 + (sgc_cal_std or 0.0)**2) if sgc_ext_mean is not None and sgc_cal_mean is not None else (sgc_cal_std or 0.0)
        
        coord_total_mean = (coord_ext_mean or 0.0) + (coord_cal_mean or 0.0)
        coord_total_std = np.sqrt((coord_ext_std or 0.0)**2 + (coord_cal_std or 0.0)**2) if coord_ext_mean is not None and coord_cal_mean is not None else (coord_cal_std or 0.0)
        
        # Use minimum count for consistency
        n_seeds = min(sgc_cal_n, coord_cal_n) if sgc_cal_n > 0 and coord_cal_n > 0 else max(sgc_cal_n, coord_cal_n)
        
        timing_data.append({
            'dataset': dataset,
            'model': model_name,
            'sgc_extraction_mean': sgc_ext_mean,
            'sgc_extraction_std': sgc_ext_std,
            'sgc_calibration_mean': sgc_cal_mean,
            'sgc_calibration_std': sgc_cal_std,
            'sgc_total_mean': sgc_total_mean,
            'sgc_total_std': sgc_total_std,
            'coord_extraction_mean': coord_ext_mean,
            'coord_extraction_std': coord_ext_std,
            'coord_calibration_mean': coord_cal_mean,
            'coord_calibration_std': coord_cal_std,
            'coord_total_mean': coord_total_mean,
            'coord_total_std': coord_total_std,
            'n_seeds': n_seeds
        })
    
    if not timing_data:
        print("No timing data available for FAISS table generation")
        return
    
    # Sort by dataset, then model
    timing_data.sort(key=lambda x: (x['dataset'], x['model']))
    
    # Generate LaTeX table
    lines = []
    lines.append("% Timing comparison table: SGC-FAISS vs SGC Lite-FAISS (Coordinate-FAISS)")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{xcolor}")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Timing comparison between SGC-FAISS and SGC Lite-FAISS (Coordinate-FAISS) across datasets and models. Times are in seconds.}")
    lines.append("\\label{tab:timing_sgc_faiss_vs_sgc_lite_faiss}")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llccccccccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & \\multicolumn{3}{c}{SGC-FAISS} & \\multicolumn{3}{c}{SGC Lite-FAISS} & \\multicolumn{2}{c}{Speedup} & Seeds \\\\")
    lines.append("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-10}")
    lines.append(" & & Extract & Calib & Total & Extract & Calib & Total & Total & Calib & \\\\")
    lines.append("\\midrule")
    
    for row in timing_data:
        dataset = row['dataset'].upper().replace('_', '\\_')
        model = row['model']
        
        # Format times
        def format_time(mean_val, std_val, unit='s'):
            if mean_val is None or mean_val == 0:
                return "---"
            if unit == 'ms':
                mean_val *= 1000
                std_val *= 1000 if std_val else 0
            if std_val and std_val > 0:
                return f"{mean_val:.2f}$\\pm${std_val:.2f}"
            else:
                return f"{mean_val:.2f}"
        
        sgc_ext = format_time(row['sgc_extraction_mean'], row['sgc_extraction_std'], 's')
        sgc_cal = format_time(row['sgc_calibration_mean'], row['sgc_calibration_std'], 's')
        sgc_total = format_time(row['sgc_total_mean'], row['sgc_total_std'], 's')
        
        coord_ext = format_time(row['coord_extraction_mean'], row['coord_extraction_std'], 's')
        coord_cal = format_time(row['coord_calibration_mean'], row['coord_calibration_std'], 's')
        coord_total = format_time(row['coord_total_mean'], row['coord_total_std'], 's')
        
        # Calculate speedups (SGC Lite vs SGC)
        def format_speedup(coord_time, sgc_time):
            """Calculate speedup of SGC Lite relative to SGC."""
            if coord_time is None or sgc_time is None or coord_time == 0 or sgc_time == 0:
                return "---"
            if coord_time < sgc_time:
                # SGC Lite is faster
                speedup = sgc_time / coord_time
                return f"{speedup:.2f}x"
            else:
                # SGC Lite is slower
                slowdown = coord_time / sgc_time
                return f"{slowdown:.2f}x slower"
        
        total_speedup = format_speedup(row['coord_total_mean'], row['sgc_total_mean'])
        calib_speedup = format_speedup(row['coord_calibration_mean'], row['sgc_calibration_mean'])
        
        n_seeds = row['n_seeds']
        
        lines.append(f"{dataset} & {model} & {sgc_ext} & {sgc_cal} & {sgc_total} & {coord_ext} & {coord_cal} & {coord_total} & {total_speedup} & {calib_speedup} & {n_seeds} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    # Write to file
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\n📊 FAISS timing comparison table saved to: {output_file}")
    return '\n'.join(lines)


def generate_timing_comparison_table(groups, training_method_filter='baseline_cross_entropy', output_file="latex_tables_statistics/table_timing_sgc_vs_sgc_lite.tex"):
    """
    Generate a LaTeX table comparing timing between SGC and SGC Lite (Coordinate).
    
    Args:
        groups: Dictionary of grouped results from load_all_runs
        training_method_filter: Filter to specific training method (default: 'baseline_cross_entropy')
        output_file: Path to output LaTeX file
    """
    # Filter groups
    filtered_groups = {
        (tm, ds, mn): methods_data
        for (tm, ds, mn), methods_data in groups.items()
        if tm == training_method_filter and ds != 'svhn'
    }
    
    if not filtered_groups:
        print("No groups found for timing comparison table")
        return
    
    # Collect timing data for each (dataset, model) combination
    timing_data = []
    
    for (training_method, dataset, model_name), methods_data in filtered_groups.items():
        sgc_data = methods_data.get('SGC (Layers)', {})
        coord_data = methods_data.get('Coordinate', {})
        
        # Extract timing lists
        sgc_extraction = sgc_data.get('extraction_timing', [])
        sgc_calibration = sgc_data.get('timing', [])
        coord_extraction = coord_data.get('extraction_timing', [])
        coord_calibration = coord_data.get('timing', [])
        
        # Filter valid values
        sgc_extraction = [t for t in sgc_extraction if t is not None and t >= 0]
        sgc_calibration = [t for t in sgc_calibration if t is not None and t > 0]
        coord_extraction = [t for t in coord_extraction if t is not None and t >= 0]
        coord_calibration = [t for t in coord_calibration if t is not None and t > 0]
        
        if not sgc_calibration or not coord_calibration:
            continue  # Skip if no timing data
        
        # Compute statistics
        def compute_stats(values):
            if not values:
                return None, None, 0
            arr = np.array(values)
            return float(np.mean(arr)), float(np.std(arr)), len(arr)
        
        sgc_ext_mean, sgc_ext_std, sgc_ext_n = compute_stats(sgc_extraction)
        sgc_cal_mean, sgc_cal_std, sgc_cal_n = compute_stats(sgc_calibration)
        coord_ext_mean, coord_ext_std, coord_ext_n = compute_stats(coord_extraction)
        coord_cal_mean, coord_cal_std, coord_cal_n = compute_stats(coord_calibration)
        
        # Total time = extraction + calibration
        sgc_total_mean = (sgc_ext_mean or 0.0) + (sgc_cal_mean or 0.0)
        sgc_total_std = np.sqrt((sgc_ext_std or 0.0)**2 + (sgc_cal_std or 0.0)**2) if sgc_ext_mean is not None and sgc_cal_mean is not None else (sgc_cal_std or 0.0)
        
        coord_total_mean = (coord_ext_mean or 0.0) + (coord_cal_mean or 0.0)
        coord_total_std = np.sqrt((coord_ext_std or 0.0)**2 + (coord_cal_std or 0.0)**2) if coord_ext_mean is not None and coord_cal_mean is not None else (coord_cal_std or 0.0)
        
        # Use minimum count for consistency
        n_seeds = min(sgc_cal_n, coord_cal_n) if sgc_cal_n > 0 and coord_cal_n > 0 else max(sgc_cal_n, coord_cal_n)
        
        timing_data.append({
            'dataset': dataset,
            'model': model_name,
            'sgc_extraction_mean': sgc_ext_mean,
            'sgc_extraction_std': sgc_ext_std,
            'sgc_calibration_mean': sgc_cal_mean,
            'sgc_calibration_std': sgc_cal_std,
            'sgc_total_mean': sgc_total_mean,
            'sgc_total_std': sgc_total_std,
            'coord_extraction_mean': coord_ext_mean,
            'coord_extraction_std': coord_ext_std,
            'coord_calibration_mean': coord_cal_mean,
            'coord_calibration_std': coord_cal_std,
            'coord_total_mean': coord_total_mean,
            'coord_total_std': coord_total_std,
            'n_seeds': n_seeds
        })
    
    if not timing_data:
        print("No timing data available for table generation")
        return
    
    # Sort by dataset, then model
    timing_data.sort(key=lambda x: (x['dataset'], x['model']))
    
    # Generate LaTeX table
    lines = []
    lines.append("% Timing comparison table: SGC vs SGC Lite (Coordinate)")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{xcolor}")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Timing comparison between SGC and SGC Lite (Coordinate) across datasets and models. Times are in seconds.}")
    lines.append("\\label{tab:timing_sgc_vs_sgc_lite}")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llccccccccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & \\multicolumn{3}{c}{SGC} & \\multicolumn{3}{c}{SGC Lite} & \\multicolumn{2}{c}{Speedup} & Seeds \\\\")
    lines.append("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-10}")
    lines.append(" & & Extract & Calib & Total & Extract & Calib & Total & Total & Calib & \\\\")
    lines.append("\\midrule")
    
    for row in timing_data:
        dataset = row['dataset'].upper().replace('_', '\\_')
        model = row['model']
        
        # Format SGC times (convert to milliseconds for display)
        def format_time(mean_val, std_val, unit='s'):
            if mean_val is None or mean_val == 0:
                return "---"
            if unit == 'ms':
                mean_val *= 1000
                std_val *= 1000 if std_val else 0
            if std_val and std_val > 0:
                return f"{mean_val:.2f}$\\pm${std_val:.2f}"
            else:
                return f"{mean_val:.2f}"
        
        sgc_ext = format_time(row['sgc_extraction_mean'], row['sgc_extraction_std'], 's')
        sgc_cal = format_time(row['sgc_calibration_mean'], row['sgc_calibration_std'], 's')
        sgc_total = format_time(row['sgc_total_mean'], row['sgc_total_std'], 's')
        
        coord_ext = format_time(row['coord_extraction_mean'], row['coord_extraction_std'], 's')
        coord_cal = format_time(row['coord_calibration_mean'], row['coord_calibration_std'], 's')
        coord_total = format_time(row['coord_total_mean'], row['coord_total_std'], 's')
        
        # Calculate speedups (SGC Lite vs SGC)
        # If SGC Lite is faster, show "X.XXx" speedup
        # If SGC Lite is slower, show "X.XXx slower"
        def format_speedup(coord_time, sgc_time):
            """Calculate speedup of SGC Lite relative to SGC."""
            if coord_time is None or sgc_time is None or coord_time == 0 or sgc_time == 0:
                return "---"
            if coord_time < sgc_time:
                # SGC Lite is faster
                speedup = sgc_time / coord_time
                return f"{speedup:.2f}x"
            else:
                # SGC Lite is slower
                slowdown = coord_time / sgc_time
                return f"{slowdown:.2f}x slower"
        
        total_speedup = format_speedup(row['coord_total_mean'], row['sgc_total_mean'])
        calib_speedup = format_speedup(row['coord_calibration_mean'], row['sgc_calibration_mean'])
        
        n_seeds = row['n_seeds']
        
        lines.append(f"{dataset} & {model} & {sgc_ext} & {sgc_cal} & {sgc_total} & {coord_ext} & {coord_cal} & {coord_total} & {total_speedup} & {calib_speedup} & {n_seeds} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    # Write to file
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\n📊 Timing comparison table saved to: {output_file}")
    return '\n'.join(lines)


def plot_multiseed_comparison(methods_data, title_suffix="", output_file="functions_comparison/g_function_multiseed_comparison.png", method_filter=None):
    """
    Plot calibration functions for specified methods.
    
    Args:
        methods_data: Dictionary of method data
        title_suffix: Title suffix for the plot
        output_file: Output file path
        method_filter: Optional list of method names to include (None = all methods)
    """
    # Evaluation grid
    x_grid = np.linspace(0, 1, 1000)

    plt.figure(figsize=(12, 8))

    any_curve_plotted = False
    all_y_curves = {}  # Store per-method curves for RMSE statistics

    # Colors for each method
    colors = {
        'SGC (Layers)': 'blue',
        'Coordinate': 'orange',
        'SGC (DAC Layers)': 'green',
        'SGC-FAISS': 'cyan',
        'Coordinate-FAISS': 'red',
        'Coordinate-SPP-FAISS': 'purple'
    }

    linestyles = {
        'SGC (Layers)': '-',
        'Coordinate': '--',
        'SGC (DAC Layers)': ':',
        'SGC-FAISS': '-.',
        'Coordinate-FAISS': '--',
        'Coordinate-SPP-FAISS': ':'
    }

    print("\n📊 Statistical Comparison:")
    print(f"{'Method':<25} | {'N_Seeds':<10} | {'Mean Pred @ 0.5':<15} | {'Std Dev':<10}")
    print("-" * 70)

    # Extract params_list, ECE, and timing data
    params_dict = {}
    timing_dict = {}
    extraction_timing_dict = {}
    for name, data in methods_data.items():
        # Apply filter if specified
        if method_filter is not None and name not in method_filter:
            continue
            
        if isinstance(data, dict) and 'params' in data:
            # New format: list of dicts with 'params' and 'ece'
            params_dict[name] = data['params']
            timing_dict[name] = data.get('timing', [])
            extraction_timing_dict[name] = data.get('extraction_timing', [])
        else:
            # Backward compatibility: if methods_data is old format
            params_dict[name] = data
            timing_dict[name] = []
            extraction_timing_dict[name] = []

    for name, data_list in params_dict.items():
        if not data_list:
            continue

        # Handle both old format (list of params) and new format (list of dicts with params and ece)
        if isinstance(data_list[0], dict) and 'params' in data_list[0]:
            # New format: extract params and ECE separately
            params_list = [d['params'] for d in data_list]
            ece_list = [d.get('ece') for d in data_list if d.get('ece') is not None]
            mean_ece = np.mean(ece_list) if ece_list else None
        else:
            # Old format: just params
            params_list = data_list
            ece_list = []
            mean_ece = None

        # Check normalization consistency (pass the original data_list to preserve structure)
        norm_check = check_normalization_consistency(data_list)
        if not norm_check['consistent']:
            print(f"\n⚠️  WARNING: Normalization parameters vary for {name}:")
            print(f"   stability_p5: {norm_check['p5_mean']:.4f} ± {norm_check['p5_std']:.4f}")
            print(f"   stability_p95: {norm_check['p95_mean']:.4f} ± {norm_check['p95_std']:.4f}")
            print(f"   Functions may not be directly comparable!")

        # Evaluate all seeds on the grid
        y_curves = []
        for params in params_list:
            func = reconstruct_isotonic_func(params)
            if func is not None:
                y_curves.append(func(x_grid))

        if not y_curves:
            continue

        y_curves = np.array(y_curves)  # Shape: (n_seeds, 1000)
        all_y_curves[name] = y_curves

        # Calculate statistics
        mean_curve = np.mean(y_curves, axis=0)
        std_curve = np.std(y_curves, axis=0)

        # 95% Confidence Interval (1.96 * stderr)
        n = len(y_curves)
        ci = 1.96 * std_curve / np.sqrt(n)

        # Create label with ECE if available
        if mean_ece is not None:
            label_text = f"{name}\n(n={n}, ECE={mean_ece:.4f})"
        else:
            label_text = f"{name} (n={n})"

        # Plot Mean
        plt.plot(
            x_grid,
            mean_curve,
            label=label_text,
            color=colors[name],
            linestyle=linestyles[name],
            linewidth=2
        )

        # Plot Confidence Interval
        plt.fill_between(
            x_grid,
            mean_curve - ci,
            mean_curve + ci,
            color=colors[name],
            alpha=0.2
        )

        # Print stats
        mid_idx = 500  # Index for x=0.5
        ece_str = f", ECE={mean_ece:.4f}" if mean_ece is not None else ""
        print(f"{name:<25} | {n:<10} | {mean_curve[mid_idx]:.4f}          | {std_curve[mid_idx]:.4f}{ece_str}")
        any_curve_plotted = True

    # If no curves were plotted at all, don't create an empty plot
    if not any_curve_plotted:
        plt.close()
        print("No valid curves for this configuration – skipping plot.")
        # Return empty timing_stats to avoid errors in main loop
        return {
            'title': title_suffix,
            'extraction_timing_s': {},
            'calibration_timing_s': {},
            'total_timing_s': {},
            'extraction_timing_ms': {},
            'calibration_timing_ms': {},
            'total_timing_ms': {},
            'speedup_vs_sgc': {}
        }

    plt.plot([0, 1], [0, 1], 'k:', alpha=0.3, label='Identity')

    # Compute comprehensive distance metrics and significance tests
    pair_names = [
        ('SGC (Layers)', 'Coordinate'),
        ('SGC (Layers)', 'SGC (DAC Layers)'),
        ('Coordinate', 'SGC (DAC Layers)'),
        ('SGC (Layers)', 'SGC-FAISS'),
        ('Coordinate', 'Coordinate-FAISS'),
        ('Coordinate-SPP-FAISS', 'Coordinate-FAISS'),
        ('SGC-FAISS', 'Coordinate-FAISS'),
    ]
    
    print("\n" + "="*80)
    print("📏 FUNCTION DISTANCE METRICS")
    print("="*80)
    
    distance_lines = []
    significance_lines = []
    
    # Initialize list to store comparison statistics for CSV
    comparison_stats = []
    
    for a, b in pair_names:
        if a in all_y_curves and b in all_y_curves:
            # Compute distance metrics
            distances, n_pairs = compute_function_distances(all_y_curves[a], all_y_curves[b], x_grid)
            if distances:
                print(f"\n{a} vs {b} (n={n_pairs}):")
                print(f"  RMSE:           {distances['rmse']['mean']:.6f} ± {distances['rmse']['ci_95']:.6f}")
                print(f"  Max Deviation:  {distances['max_deviation']['mean']:.6f} ± {distances['max_deviation']['ci_95']:.6f}")
                print(f"  Area Between:   {distances['area_between']['mean']:.6f} ± {distances['area_between']['ci_95']:.6f}")
                if not np.isnan(distances['correlation']['mean']):
                    print(f"  Correlation:    {distances['correlation']['mean']:.4f} ± {distances['correlation']['ci_95']:.4f}")
                
                distance_lines.append(f"{a} vs {b}: RMSE={distances['rmse']['mean']:.4f}, MaxDev={distances['max_deviation']['mean']:.4f}")
                
                # Perform significance tests
                sig_results = perform_significance_tests(all_y_curves[a], all_y_curves[b])
                
                # Initialize comparison dictionary
                comparison_dict = {
                    'Comparison': f"{a} vs {b}",
                    'N': n_pairs,
                    'RMSE_Mean': distances['rmse']['mean'],
                    'RMSE_CI_95': distances['rmse']['ci_95'],
                    'Max_Deviation_Mean': distances['max_deviation']['mean'],
                    'Max_Deviation_CI_95': distances['max_deviation']['ci_95'],
                    'Correlation_Mean': distances['correlation']['mean'] if not np.isnan(distances['correlation']['mean']) else None,
                    'Correlation_CI_95': distances['correlation']['ci_95'] if not np.isnan(distances['correlation']['ci_95']) else None,
                }
                
                if 'error' not in sig_results:
                    print(f"\n  Statistical Tests:")
                    ttest = sig_results.get('paired_ttest', {})
                    if ttest:
                        sig_str = "***" if ttest.get('significant_at_0.05', False) else ""
                        print(f"    Paired t-test: t={ttest.get('t_statistic', 0):.4f}, p={ttest.get('p_value', 1):.4f} {sig_str}")
                        significance_lines.append(f"{a} vs {b}: p={ttest.get('p_value', 1):.4f} {sig_str}")
                        
                        # Extract paired t-test statistics
                        comparison_dict['Paired_TTest_t_statistic'] = ttest.get('t_statistic', None)
                        comparison_dict['Paired_TTest_p_value'] = ttest.get('p_value', None)
                    else:
                        comparison_dict['Paired_TTest_t_statistic'] = None
                        comparison_dict['Paired_TTest_p_value'] = None
                    
                    wilcoxon = sig_results.get('wilcoxon', {})
                    if wilcoxon and 'error' not in wilcoxon:
                        sig_str = "***" if wilcoxon.get('significant_at_0.05', False) else ""
                        print(f"    Wilcoxon:      W={wilcoxon.get('statistic', 0):.4f}, p={wilcoxon.get('p_value', 1):.4f} {sig_str}")
                        
                        # Extract Wilcoxon statistics
                        comparison_dict['Wilcoxon_statistic'] = wilcoxon.get('statistic', None)
                        comparison_dict['Wilcoxon_p_value'] = wilcoxon.get('p_value', None)
                    else:
                        comparison_dict['Wilcoxon_statistic'] = None
                        comparison_dict['Wilcoxon_p_value'] = None
                else:
                    # No significance tests available
                    comparison_dict['Paired_TTest_t_statistic'] = None
                    comparison_dict['Paired_TTest_p_value'] = None
                    comparison_dict['Wilcoxon_statistic'] = None
                    comparison_dict['Wilcoxon_p_value'] = None
                
                # Add to comparison stats list
                comparison_stats.append(comparison_dict)
    
    # Save comparison statistics to CSV
    if comparison_stats:
        df_stats = pd.DataFrame(comparison_stats)
        # Create CSV filename from plot output filename
        csv_filename = os.path.splitext(output_file)[0] + "_statistics.csv"
        df_stats.to_csv(csv_filename, index=False)
        print(f"\n💾 Statistical comparison results saved to: {csv_filename}")
    
    # Latency Benchmark Analysis (in milliseconds)
    print("\n" + "="*80)
    print("🚀 LATENCY BENCHMARK (ms)")
    print("="*80)
    
    extraction_means = {}
    extraction_stds = {}
    calibration_means = {}
    calibration_stds = {}
    total_time_means = {}
    total_time_stds = {}
    speedup_factors = {}
    
    for name in params_dict.keys():
        extraction_list = extraction_timing_dict.get(name, [])
        calibration_list = timing_dict.get(name, [])
        
        # Filter and compute extraction time (in seconds)
        if extraction_list:
            extraction_array = np.array([t for t in extraction_list if t is not None and t >= 0])
            if len(extraction_array) > 0:
                extraction_means[name] = np.mean(extraction_array)
                extraction_stds[name] = np.std(extraction_array) if len(extraction_array) > 1 else 0.0
        
        # Filter and compute calibration time (in seconds)
        if calibration_list:
            calibration_array = np.array([t for t in calibration_list if t is not None and t > 0])
            if len(calibration_array) > 0:
                calibration_mean = np.mean(calibration_array)
                calibration_std = np.std(calibration_array) if len(calibration_array) > 1 else 0.0
                calibration_means[name] = calibration_mean
                calibration_stds[name] = calibration_std
                
                # Total time = extraction + calibration
                if name in extraction_means:
                    total_time_means[name] = extraction_means[name] + calibration_mean
                    # For total std, we approximate by summing variances (assuming independence)
                    total_time_stds[name] = np.sqrt(extraction_stds[name]**2 + calibration_std**2)
                else:
                    total_time_means[name] = calibration_mean
                    total_time_stds[name] = calibration_std
    
    # Fair comparison: SGC (DAC Layers) reuses precomputed features, so borrow extraction time from SGC
    if 'SGC (DAC Layers)' in extraction_means and 'SGC (Layers)' in extraction_means:
        if extraction_means['SGC (DAC Layers)'] < 1.0:  # Less than 1 second means it reused features
            extraction_means['SGC (DAC Layers)'] = extraction_means['SGC (Layers)']
            extraction_stds['SGC (DAC Layers)'] = extraction_stds.get('SGC (Layers)', 0.0)
            # Recompute total time for SGC (DAC Layers) if it was already computed
            if 'SGC (DAC Layers)' in calibration_means:
                total_time_means['SGC (DAC Layers)'] = extraction_means['SGC (DAC Layers)'] + calibration_means['SGC (DAC Layers)']
                total_time_stds['SGC (DAC Layers)'] = np.sqrt(extraction_stds['SGC (DAC Layers)']**2 + calibration_stds.get('SGC (DAC Layers)', 0.0)**2)
    
    # Find SGC (Layers) total time for speedup calculation
    sgc_total_time = total_time_means.get('SGC (Layers)', None)
    
    if total_time_means:
        print(f"\n{'Method':<30} | {'Extraction (ms)':<20} | {'Calibration (ms)':<20} | {'Total (ms)':<20} | {'Speedup (vs SGC)':<20}")
        print("-" * 120)
        
        for name in sorted(total_time_means.keys(), key=lambda x: total_time_means[x]):
            extraction_mean_s = extraction_means.get(name, 0.0)
            extraction_std_s = extraction_stds.get(name, 0.0)
            calibration_mean_s = calibration_means.get(name, 0.0)
            calibration_std_s = calibration_stds.get(name, 0.0)
            total_mean_s = total_time_means[name]
            total_std_s = total_time_stds.get(name, 0.0)
            
            # Convert to milliseconds
            extraction_mean_ms = extraction_mean_s * 1000
            extraction_std_ms = extraction_std_s * 1000
            calibration_mean_ms = calibration_mean_s * 1000
            calibration_std_ms = calibration_std_s * 1000
            total_mean_ms = total_mean_s * 1000
            total_std_ms = total_std_s * 1000
            
            # Calculate speedup relative to SGC (Layers)
            if sgc_total_time and sgc_total_time > 0:
                speedup = sgc_total_time / total_mean_s
                speedup_factors[name] = float(speedup)
                if speedup >= 1:
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = f"{1/speedup:.2f}x slower"
            else:
                speedup_factors[name] = None
                speedup_str = "N/A"
            
            # Format strings with mean ± std
            extraction_str = f"{extraction_mean_ms:.2f} ± {extraction_std_ms:.2f}" if extraction_mean_ms > 0 else "0.00 ± 0.00"
            calibration_str = f"{calibration_mean_ms:.2f} ± {calibration_std_ms:.2f}"
            total_str = f"{total_mean_ms:.2f} ± {total_std_ms:.2f}"
            
            print(f"  {name:<30} | {extraction_str:<20} | {calibration_str:<20} | {total_str:<20} | {speedup_str:<20}")
    else:
        print("  No timing data available")
    
    # Save timing statistics to JSON (both seconds and milliseconds)
    timing_stats = {
        'title': title_suffix,
        'extraction_timing_s': {},
        'calibration_timing_s': {},
        'total_timing_s': {},
        'extraction_timing_ms': {},
        'calibration_timing_ms': {},
        'total_timing_ms': {},
        'speedup_vs_sgc': {}
    }
    
    for name in params_dict.keys():
        if name in extraction_means:
            # Seconds
            timing_stats['extraction_timing_s'][name] = {
                'mean': float(extraction_means[name]),
                'std': float(extraction_stds.get(name, 0.0))
            }
            # Milliseconds
            timing_stats['extraction_timing_ms'][name] = {
                'mean': float(extraction_means[name] * 1000),
                'std': float(extraction_stds.get(name, 0.0) * 1000)
            }
        
        if name in calibration_means:
            # Seconds
            timing_stats['calibration_timing_s'][name] = {
                'mean': float(calibration_means[name]),
                'std': float(calibration_stds.get(name, 0.0))
            }
            # Milliseconds
            timing_stats['calibration_timing_ms'][name] = {
                'mean': float(calibration_means[name] * 1000),
                'std': float(calibration_stds.get(name, 0.0) * 1000)
            }
        
        if name in total_time_means:
            # Seconds
            timing_stats['total_timing_s'][name] = {
                'mean': float(total_time_means[name]),
                'std': float(total_time_stds.get(name, 0.0))
            }
            # Milliseconds
            timing_stats['total_timing_ms'][name] = {
                'mean': float(total_time_means[name] * 1000),
                'std': float(total_time_stds.get(name, 0.0) * 1000)
            }
        
        if name in speedup_factors:
            timing_stats['speedup_vs_sgc'][name] = speedup_factors[name]
    
    # Save to JSON file in same directory as plot
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create JSON filename from plot filename
    json_filename = os.path.splitext(output_file)[0] + "_timing.json"
    with open(json_filename, 'w') as f:
        json.dump(timing_stats, f, indent=2)
    print(f"\n💾 Timing statistics saved to: {json_filename}")

    full_title = (
        "Statistical Comparison of Calibration Functions g(x)\n"
        "(Mean ± 95% CI over multiple seeds)"
    )
    if title_suffix:
        full_title += f"\n{title_suffix}"

    plt.title(full_title, fontsize=14)
    plt.xlabel("Normalized Stability Score", fontsize=12)
    plt.ylabel("Calibrated Probability", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add summary text inside the plot (bottom-left corner) if available
    summary_lines = []
    if distance_lines:
        summary_lines.extend(distance_lines[:2])  # Show first 2 distance comparisons
    # if timing_lines and len(timing_lines) <= 3:
    #     summary_lines.append("Timing: " + ", ".join(timing_lines))
    
    if summary_lines:
        summary_text = "\n".join(summary_lines)
        plt.gcf().text(
            0.02,
            0.02,
            summary_text,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_file, dpi=300)
    print(f"\n📈 Plot saved to: {output_file}")
    plt.close()
    
    # Return timing_stats for use in latency plot
    return timing_stats


if __name__ == "__main__":
    results_dir = "calibration_comparison"
    groups = load_all_runs(results_dir)

    # Filter to only process baseline_cross_entropy training method and exclude SVHN dataset
    filtered_groups = {
        (training_method, dataset, model_name): methods_data
        for (training_method, dataset, model_name), methods_data in groups.items()
        if training_method == 'baseline_cross_entropy' and dataset != 'svhn'
    }
    
    print(f"Filtered to {len(filtered_groups)} groups with baseline_cross_entropy training method (excluding SVHN) (out of {len(groups)} total groups)")
    
    # Generate timing comparison table
    print("\n" + "="*80)
    print("===== Generating Timing Comparison Table: SGC vs SGC Lite =====")
    print("="*80)
    generate_timing_comparison_table(groups, training_method_filter='baseline_cross_entropy')
    
    # Generate FAISS timing comparison table
    print("\n" + "="*80)
    print("===== Generating Timing Comparison Table: SGC-FAISS vs SGC Lite-FAISS =====")
    print("="*80)
    generate_timing_comparison_table_faiss(groups, training_method_filter='baseline_cross_entropy')

    # Define method groups
    standard_methods = ['SGC (Layers)', 'Coordinate', 'SGC (DAC Layers)']
    faiss_methods = ['SGC-FAISS', 'Coordinate-FAISS', 'Coordinate-SPP-FAISS', 'DAC', 'DAC (Coord)']

    if not filtered_groups:
        print("No groups found with baseline_cross_entropy training method – check results_dir and filename pattern.")
    else:
        for (training_method, dataset, model_name), methods_data in filtered_groups.items():
            title_suffix = f"{training_method}, {dataset}, {model_name}"
            
            # For directory name: use underscores (e.g., "cifar10_resnet18_baseline_cross_entropy")
            config_dir = f"{dataset}_{model_name}_{training_method}"
            config_output_dir = os.path.join("functions_comparison", config_dir)
            os.makedirs(config_output_dir, exist_ok=True)
            
            # For filenames: use hyphens for safety
            safe_tm = training_method.replace(" ", "-").replace("_", "-")
            safe_dataset = dataset.replace("_", "-")
            safe_model = model_name.replace("_", "-")
            
            print(f"\n{'='*80}")
            print(f"===== Plotting group: {title_suffix} =====")
            print(f"===== Output directory: {config_output_dir} =====")
            print(f"{'='*80}")
            
            # Plot 1: Standard methods only
            print("\n--- Creating Standard Methods Plot ---")
            standard_output = os.path.join(
                config_output_dir,
                f"g_function_multiseed_{safe_tm}_{safe_dataset}_{safe_model}_standard.png"
            )
            timing_stats_standard = plot_multiseed_comparison(
                methods_data,
                title_suffix=f"{title_suffix} (Standard Methods)",
                output_file=standard_output,
                method_filter=standard_methods
            )
            
            # Plot 2: FAISS methods only
            print("\n--- Creating FAISS Methods Plot ---")
            faiss_output = os.path.join(
                config_output_dir,
                f"g_function_multiseed_{safe_tm}_{safe_dataset}_{safe_model}_faiss.png"
            )
            timing_stats_faiss = plot_multiseed_comparison(
                methods_data,
                title_suffix=f"{title_suffix} (FAISS Methods)",
                output_file=faiss_output,
                method_filter=faiss_methods
            )
            
            # Plot 3: Combined (all methods)
            print("\n--- Creating Combined Plot ---")
            combined_output = os.path.join(
                config_output_dir,
                f"g_function_multiseed_{safe_tm}_{safe_dataset}_{safe_model}_combined.png"
            )
            timing_stats_combined = plot_multiseed_comparison(
                methods_data,
                title_suffix=title_suffix,
                output_file=combined_output,
                method_filter=None  # All methods
            )
            
            # Use combined timing stats for latency plots (has all methods)
            timing_stats = timing_stats_combined
            
            # Create latency comparison plot
            if timing_stats and 'total_timing_s' in timing_stats and timing_stats['total_timing_s']:
                # Convert to format expected by plot_latency_comparison
                latency_stats = {
                    'total_timing': timing_stats['total_timing_s'],
                    'calibration_timing': timing_stats.get('calibration_timing_s', {}),
                    'extraction_timing': timing_stats.get('extraction_timing_s', {}),
                    'speedup_vs_sgc': timing_stats.get('speedup_vs_sgc', {})
                }
                latency_output_file = os.path.join(
                    config_output_dir,
                    f"latency_comparison_{safe_tm}_{safe_dataset}_{safe_model}.png"
                )
                plot_latency_comparison(
                    latency_stats,
                    output_base_filename=latency_output_file
                )


