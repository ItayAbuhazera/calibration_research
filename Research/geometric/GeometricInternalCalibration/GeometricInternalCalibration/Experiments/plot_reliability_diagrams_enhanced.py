import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
from collections import defaultdict
from scipy import stats

# =============================================================================
# CONFIGURATION - Edit these to control output
# =============================================================================
CONFIG = {
    # Data settings
    'base_dir': 'reliability_data',
    'output_dir': 'reliability_diagrams',
    'n_bins': 15,
    
    # Which visualization style to use: 'bars', 'curves', 'weighted_alpha'
    'style': 'bars',
    
    # Binning type: 'equal_width' or 'adaptive'
    'bin_type': 'equal_width',
    
    # Show histogram below reliability diagram
    'show_histogram': True,
    
    # Generate individual method plots (in addition to comparisons)
    'generate_individual': False,
    
    # Figure dimensions
    'figsize': (7, 3.5),  # Width, height for comparison plots
    
    # Methods to include and their order (None = auto-detect all)
    'methods_order': ['uncalibrated', 'temperature_scaling', 'top_label_isotonic', 'sgc', 'sgc_trust_score', 'sgc_lite', 'dac'],
    
    # Which metrics to display in title (choose 1-3 for readability)
    # Options: 'ece', 'mce', 'brier', 'nll', 'accuracy', 'overconfidence', 'sharpness', 'auroc'
    'display_metrics': ['ece'],
    
    # Generate separate metrics summary table
    'generate_metrics_table': True,
}

# Presets for common use cases
PRESETS = {
    'paper_main': {
        'style': 'bars',
        'bin_type': 'equal_width',
        'show_histogram': True,
        'figsize': (7, 3.5),
        'display_metrics': ['ece'],
    },
    'paper_supplementary': {
        'style': 'bars', 
        'bin_type': 'equal_width',
        'show_histogram': True,
        'figsize': (7, 3.5),
        'generate_individual': True,
        'display_metrics': ['ece', 'mce'],
    },
    'full_analysis': {
        'style': 'bars',
        'bin_type': 'equal_width',
        'show_histogram': True,
        'figsize': (7, 3.5),
        'display_metrics': ['ece', 'sharpness'],
        'generate_metrics_table': True,
    },
    'adaptive_comparison': {
        'style': 'bars',
        'bin_type': 'adaptive',
        'show_histogram': True,
        'figsize': (7, 3.5),
    },
    'curves_clean': {
        'style': 'curves',
        'bin_type': 'equal_width', 
        'show_histogram': False,
        'figsize': (7, 2.8),
    },
}

# =============================================================================
# STYLING
# =============================================================================
import matplotlib.font_manager as fm

available_fonts = [f.name for f in fm.fontManager.ttflist]
serif_fonts = ['Times New Roman', 'Times', 'Palatino', 'DejaVu Serif']
has_serif = any(font in available_fonts for font in serif_fonts)

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 10,
    'font.family': 'serif' if has_serif else 'sans-serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

COLORS = {
    'uncalibrated': '#1f77b4',
    'temperature_scaling': '#ff7f0e',
    'temperature': '#ff7f0e',
    'sgc': '#d62728',
    'sgc_lite': '#9467bd',
    'dac': '#2ca02c',
    'geometric': '#8c564b',
    'coordinate': '#e377c2',
    'sgc_trust_score': '#17becf',
    'top_label_isotonic': '#e6194B',
}

METHOD_NAMES = {
    'uncalibrated': 'Uncalibrated',
    'temperature_scaling': 'Temp. Scaling',
    'temperature': 'Temp. Scaling',
    'sgc': 'RGC-L',
    'sgc_lite': 'RGC-C',
    'dac': 'DAC',
    'geometric': 'Geometric',
    'coordinate': 'Coordinate',
    'sgc_trust_score': 'RGC-L (Trust Score)',
    'top_label_isotonic': 'Iso'
}

# Metric display names and formatting
METRIC_INFO = {
    'ece': {'name': 'ECE', 'format': '{:.2f}%', 'multiply': 100, 'lower_better': True},
    'mce': {'name': 'MCE', 'format': '{:.2f}%', 'multiply': 100, 'lower_better': True},
    'brier': {'name': 'Brier', 'format': '{:.4f}', 'multiply': 1, 'lower_better': True},
    'nll': {'name': 'NLL', 'format': '{:.3f}', 'multiply': 1, 'lower_better': True},
    'accuracy': {'name': 'Acc', 'format': '{:.2f}%', 'multiply': 100, 'lower_better': False},
    'overconfidence': {'name': 'OvConf', 'format': '{:+.2f}%', 'multiply': 100, 'lower_better': True},
    'sharpness': {'name': 'Sharp', 'format': '{:.3f}', 'multiply': 1, 'lower_better': False},
    'auroc': {'name': 'AUROC', 'format': '{:.3f}', 'multiply': 1, 'lower_better': False},
    'conf_range': {'name': 'Conf Range', 'format': '{:.2f}', 'multiply': 1, 'lower_better': False},
}

# =============================================================================
# DATA LOADING
# =============================================================================
def extract_metadata_from_filename(filepath):
    """Extract metadata from filename."""
    filename = Path(filepath).stem
    path_parts = Path(filepath).parts
    
    metadata = {
        'model': None,
        'dataset': None,
        'training_method': None,
        'seed': None,
        'method': None,
    }
    
    # Extract seed
    seed_match = re.search(r'seed(\d+)', filename)
    if seed_match:
        metadata['seed'] = int(seed_match.group(1))
    
    # Parse filename
    parts = filename.replace('_per_sample', '').split('_')
    seed_idx = next((i for i, p in enumerate(parts) if p.startswith('seed')), None)
    
    if seed_idx is not None:
        model_parts = parts[:seed_idx]
        method_parts = parts[seed_idx+1:] if seed_idx + 1 < len(parts) else []
        
        if model_parts:
            metadata['model'] = model_parts[0]
        
        known_datasets = ['tiny_imagenet', 'tinyimagenet', 'cifar100', 'cifar10']
        for ds in known_datasets:
            if ds in '_'.join(model_parts):
                metadata['dataset'] = ds
                break
        
        if method_parts:
            metadata['method'] = '_'.join(method_parts)
    
    # Fallback extraction
    method_patterns = [
        ('sgc_lite', 'sgc_lite'),
        ('sgc_trust_score', 'sgc_trust_score'),
        ('sgc', 'sgc'),
        ('temperature_scaling', 'temperature_scaling'),
        ('temperature', 'temperature_scaling'),
        ('uncalibrated', 'uncalibrated'),
        ('dac', 'dac'),
        ('geometric', 'geometric'),
        ('coordinate', 'coordinate'),
        ('top_label_isotonic', 'top_label_isotonic'),
    ]
    
    if not metadata['method']:
        for pattern, method in method_patterns:
            if pattern in filename:
                metadata['method'] = method
                break
    
    # Extract model/dataset from path
    full_path = '_'.join(path_parts) + '_' + filename
    for model in ['resnet18', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'dinov2']:
        if model in full_path.lower():
            metadata['model'] = metadata['model'] or model
            break
    
    for dataset in ['cifar100', 'cifar10', 'tiny_imagenet', 'tinyimagenet']:
        if dataset in full_path.lower():
            metadata['dataset'] = metadata['dataset'] or dataset
            break
    
    return metadata


def load_and_group_data(base_dir):
    """Load and group .npz files by (model, dataset, method)."""
    base_dir = Path(base_dir)
    grouped_data = defaultdict(list)
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.npz') and 'per_sample' in file:
                filepath = os.path.join(root, file)
                try:
                    metadata = extract_metadata_from_filename(filepath)
                    
                    if not all([metadata['model'], metadata['dataset'], metadata['method']]):
                        print(f"Warning: Incomplete metadata for {filepath}, skipping")
                        continue
                    
                    data = np.load(filepath)
                    group_key = (metadata['model'], metadata['dataset'], metadata['method'])
                    grouped_data[group_key].append({
                        'seed': metadata['seed'],
                        'max_probs': data['max_probs'],
                        'correct': data['correct'],
                    })
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
    
    return grouped_data

# =============================================================================
# METRICS COMPUTATION
# =============================================================================
def compute_ece(max_probs, correct, n_bins=15):
    """Compute ECE (Expected Calibration Error) with equal-width bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(max_probs)
    
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (max_probs >= bin_edges[i]) & (max_probs <= bin_edges[i+1])
        else:
            in_bin = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i+1])
        
        prop_in_bin = in_bin.sum() / n_samples
        if prop_in_bin > 0:
            acc = correct[in_bin].mean()
            conf = max_probs[in_bin].mean()
            ece += np.abs(conf - acc) * prop_in_bin
    
    return ece


def compute_mce(max_probs, correct, n_bins=15):
    """Compute MCE (Maximum Calibration Error) - worst-case bin error."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    max_error = 0.0
    
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (max_probs >= bin_edges[i]) & (max_probs <= bin_edges[i+1])
        else:
            in_bin = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i+1])
        
        if in_bin.sum() > 0:
            acc = correct[in_bin].mean()
            conf = max_probs[in_bin].mean()
            max_error = max(max_error, np.abs(conf - acc))
    
    return max_error


def compute_brier_score(max_probs, correct):
    """Compute Brier Score - proper scoring rule (lower is better)."""
    # Brier score for binary correctness: (confidence - correct)^2
    return np.mean((max_probs - correct) ** 2)


def compute_nll(max_probs, correct, eps=1e-15):
    """Compute Negative Log Likelihood (lower is better)."""
    # Clip probabilities to avoid log(0)
    probs_clipped = np.clip(max_probs, eps, 1 - eps)
    # NLL = -mean(correct * log(p) + (1-correct) * log(1-p))
    nll = -np.mean(correct * np.log(probs_clipped) + (1 - correct) * np.log(1 - probs_clipped))
    return nll


def compute_overconfidence(max_probs, correct, n_bins=15):
    """
    Compute average signed calibration error (positive = overconfident).
    Overconfidence means confidence > accuracy.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    signed_error = 0.0
    n_samples = len(max_probs)
    
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (max_probs >= bin_edges[i]) & (max_probs <= bin_edges[i+1])
        else:
            in_bin = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i+1])
        
        prop_in_bin = in_bin.sum() / n_samples
        if prop_in_bin > 0:
            acc = correct[in_bin].mean()
            conf = max_probs[in_bin].mean()
            signed_error += (conf - acc) * prop_in_bin  # positive if overconfident
    
    return signed_error


def compute_sharpness(max_probs):
    """
    Compute sharpness - standard deviation of confidence scores.
    Higher sharpness means more spread in confidence values.
    A method that assigns similar confidence to all samples has low sharpness.
    """
    return np.std(max_probs)


def compute_confidence_range(max_probs):
    """
    Compute the interquartile range (IQR) of confidence scores.
    More robust measure of spread than std.
    """
    q75, q25 = np.percentile(max_probs, [75, 25])
    return q75 - q25


def compute_auroc_confidence(max_probs, correct):
    """
    Compute AUROC of using confidence to predict correctness.
    Higher is better - measures how well confidence separates correct/incorrect.
    """
    from sklearn.metrics import roc_auc_score
    try:
        # Handle edge case where all predictions are correct or incorrect
        if len(np.unique(correct)) < 2:
            return np.nan
        return roc_auc_score(correct, max_probs)
    except Exception:
        return np.nan


def compute_all_metrics(max_probs, correct, n_bins=15):
    """Compute all calibration metrics for a single run."""
    return {
        'ece': compute_ece(max_probs, correct, n_bins),
        'mce': compute_mce(max_probs, correct, n_bins),
        'brier': compute_brier_score(max_probs, correct),
        'nll': compute_nll(max_probs, correct),
        'accuracy': np.mean(correct),
        'overconfidence': compute_overconfidence(max_probs, correct, n_bins),
        'sharpness': compute_sharpness(max_probs),
        'conf_range': compute_confidence_range(max_probs),
        'auroc': compute_auroc_confidence(max_probs, correct),
        'conf_mean': np.mean(max_probs),
        'conf_min': np.min(max_probs),
        'conf_max': np.max(max_probs),
    }


def compute_bin_statistics(data_list, n_bins=15, bin_type='equal_width'):
    """Compute per-bin accuracy with uncertainty across seeds, plus all metrics."""
    if bin_type == 'equal_width':
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        all_probs = np.concatenate([d['max_probs'] for d in data_list])
        sorted_probs = np.sort(all_probs)
        bin_edges = np.interp(
            np.linspace(0, len(sorted_probs), n_bins + 1),
            np.arange(len(sorted_probs)),
            sorted_probs
        )
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    per_seed_accuracies = []
    per_seed_counts = []
    
    # Collect all metrics per seed
    all_metrics = defaultdict(list)
    
    for data_dict in data_list:
        max_probs = data_dict['max_probs']
        correct = data_dict['correct']
        
        seed_accuracies = []
        seed_counts = []
        
        for i in range(n_bins):
            if i == n_bins - 1:
                in_bin = (max_probs >= bin_edges[i]) & (max_probs <= bin_edges[i+1])
            else:
                in_bin = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i+1])
            
            bin_count = in_bin.sum()
            seed_counts.append(bin_count)
            seed_accuracies.append(correct[in_bin].mean() if bin_count > 0 else np.nan)
        
        per_seed_accuracies.append(seed_accuracies)
        per_seed_counts.append(seed_counts)
        
        # Compute all metrics for this seed
        metrics = compute_all_metrics(max_probs, correct, n_bins)
        for key, value in metrics.items():
            all_metrics[key].append(value)
    
    per_seed_accuracies = np.array(per_seed_accuracies)
    per_seed_counts = np.array(per_seed_counts)
    n_seeds = len(data_list)
    
    mean_accuracies = np.nanmean(per_seed_accuracies, axis=0)
    std_accuracies = np.nanstd(per_seed_accuracies, axis=0)
    mean_counts = np.nanmean(per_seed_counts, axis=0).astype(int)
    
    if n_seeds > 1:
        t_val = stats.t.ppf(0.975, n_seeds - 1)
        ci_accuracies = t_val * std_accuracies / np.sqrt(n_seeds)
    else:
        ci_accuracies = np.zeros_like(std_accuracies)
    
    # Aggregate metrics across seeds
    metrics_summary = {}
    for key, values in all_metrics.items():
        values = np.array(values)
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            metrics_summary[f'{key}_mean'] = np.mean(valid_values)
            metrics_summary[f'{key}_std'] = np.std(valid_values) if len(valid_values) > 1 else 0.0
        else:
            metrics_summary[f'{key}_mean'] = np.nan
            metrics_summary[f'{key}_std'] = np.nan
    
    result = {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'mean_accuracies': mean_accuracies,
        'ci_accuracies': ci_accuracies,
        'mean_counts': mean_counts,
        'n_seeds': n_seeds,
    }
    
    # Add all metrics to result
    result.update(metrics_summary)
    
    # Keep backward compatibility
    result['ece_mean'] = metrics_summary.get('ece_mean', np.nan)
    result['ece_std'] = metrics_summary.get('ece_std', 0.0)
    
    return result

# =============================================================================
# PLOTTING
# =============================================================================
def format_metric_string(bin_stats, metric_name, include_std=True):
    """Format a metric value for display."""
    info = METRIC_INFO.get(metric_name, {'name': metric_name, 'format': '{:.3f}', 'multiply': 1})
    
    mean_val = bin_stats.get(f'{metric_name}_mean', np.nan)
    std_val = bin_stats.get(f'{metric_name}_std', 0.0)
    
    if np.isnan(mean_val):
        return f"{info['name']}: N/A"
    
    display_val = mean_val * info['multiply']
    formatted = info['format'].format(display_val)
    
    if include_std and bin_stats['n_seeds'] > 1 and std_val > 0:
        std_display = std_val * info['multiply']
        # Use same decimal places as mean
        if '%' in info['format']:
            formatted += f"+/-{std_display:.2f}%"
        else:
            formatted += f"+/-{std_display:.3f}"
    
    return formatted


def build_title_metrics(bin_stats, display_metrics, method_name):
    """Build title string with method name and selected metrics."""
    display_name = METHOD_NAMES.get(method_name, method_name.upper())
    
    if not display_metrics:
        return display_name
    
    metric_strings = []
    for metric in display_metrics:
        metric_str = format_metric_string(bin_stats, metric, include_std=(bin_stats['n_seeds'] > 1))
        # Remove metric name prefix for cleaner display
        info = METRIC_INFO.get(metric, {'name': metric})
        val_only = metric_str.replace(f"{info['name']}: ", "")
        metric_strings.append(val_only)
    
    metrics_line = ", ".join(metric_strings)
    return f"{display_name}\n{metrics_line}"


def plot_comparison(all_methods_stats, model, dataset, save_path, config):
    """
    Unified plotting function supporting all visualization styles.
    """
    methods_order = config.get('methods_order') or ['uncalibrated', 'temperature_scaling', 'sgc', 'sgc_lite']
    methods_order = [m for m in methods_order if m in all_methods_stats]
    for m in all_methods_stats.keys():
        if m not in methods_order:
            methods_order.append(m)
    
    n_methods = len(methods_order)
    style = config.get('style', 'bars')
    show_histogram = config.get('show_histogram', True)
    figsize = config.get('figsize', (7, 3.5))
    display_metrics = config.get('display_metrics', ['ece'])
    
    # Adjust figure width based on number of methods
    fig_width = max(figsize[0], n_methods * 1.8)
    
    # Create figure
    if show_histogram:
        fig_height = figsize[1] * 1.4
        fig, axes = plt.subplots(2, n_methods, figsize=(fig_width, fig_height),
                                 gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})
        axes_main = axes[0] if n_methods > 1 else [axes[0]]
        axes_hist = axes[1] if n_methods > 1 else [axes[1]]
    else:
        fig, axes_main = plt.subplots(1, n_methods, figsize=(fig_width, figsize[1]), sharey=True)
        axes_main = axes_main if n_methods > 1 else [axes_main]
        axes_hist = None
    
    for idx, method_name in enumerate(methods_order):
        ax = axes_main[idx]
        bin_stats = all_methods_stats[method_name]
        
        bin_centers = bin_stats['bin_centers']
        mean_accuracies = bin_stats['mean_accuracies']
        mean_counts = bin_stats['mean_counts']
        bin_width = bin_stats['bin_edges'][1] - bin_stats['bin_edges'][0]
        
        color = COLORS.get(method_name, '#1f77b4')
        errors = bin_stats['ci_accuracies'] if bin_stats['n_seeds'] > 1 else None
        valid_mask = ~np.isnan(mean_accuracies)
        
        # Plot based on style
        if style == 'curves':
            ax.plot(bin_centers[valid_mask], mean_accuracies[valid_mask],
                    'o-', color=color, linewidth=2, markersize=4)
            if errors is not None:
                ax.fill_between(bin_centers[valid_mask],
                               mean_accuracies[valid_mask] - errors[valid_mask],
                               mean_accuracies[valid_mask] + errors[valid_mask],
                               alpha=0.2, color=color)
        
        elif style == 'weighted_alpha':
            valid_counts = mean_counts[valid_mask]
            max_count = valid_counts.max() if len(valid_counts) > 0 and valid_counts.max() > 0 else 1
            alphas = 0.3 + 0.6 * (valid_counts / max_count)
            
            bars = ax.bar(bin_centers[valid_mask], mean_accuracies[valid_mask],
                          width=bin_width*0.8, color=color, edgecolor='black', linewidth=0.5)
            for bar, alpha in zip(bars, alphas):
                bar.set_alpha(alpha)
            
            if errors is not None:
                ax.errorbar(bin_centers[valid_mask], mean_accuracies[valid_mask],
                           yerr=errors[valid_mask], fmt='none', color='black',
                           capsize=2, elinewidth=0.6, capthick=0.6)
        
        else:  # 'bars' (default)
            ax.bar(bin_centers[valid_mask], mean_accuracies[valid_mask],
                   width=bin_width*0.8, alpha=0.7, color=color,
                   edgecolor='black', linewidth=0.5,
                   yerr=errors[valid_mask] if errors is not None else None,
                   capsize=2, error_kw={'elinewidth': 0.6, 'capthick': 0.6})
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, zorder=10)
        
        # Title with selected metrics
        title = build_title_metrics(bin_stats, display_metrics, method_name)
        ax.set_title(title, fontsize=9)
        
        # Axis formatting
        if idx == 0:
            ax.set_ylabel('Accuracy', fontsize=9)
        if not show_histogram:
            ax.set_xlabel('Confidence', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.grid(False)
        
        # Histogram
        if axes_hist is not None:
            ax_hist = axes_hist[idx]
            ax_hist.bar(bin_centers[valid_mask], mean_counts[valid_mask],
                        width=bin_width*0.8, color='gray', alpha=0.5,
                        edgecolor='black', linewidth=0.5)
            ax_hist.set_xlabel('Confidence', fontsize=9)
            if idx == 0:
                ax_hist.set_ylabel('Count', fontsize=9)
            ax_hist.set_xlim(0, 1)
            ax_hist.set_xticks([0, 0.5, 1.0])
            ax_hist.grid(False)
    
    # Title
    dataset_display = dataset.upper().replace('_', '-')
    model_display = model.upper().replace('RESNET', 'ResNet-').replace('DENSENET', 'DenseNet-')
    fig.suptitle(f"{model_display} on {dataset_display}", fontsize=10, y=0.98)
    
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    print(f"Saved: {save_path}.pdf")
    plt.close()


def generate_metrics_table(all_methods_by_config, output_dir, config):
    """Generate a comprehensive metrics comparison table as CSV and LaTeX."""
    
    all_metrics = ['ece', 'mce', 'brier', 'nll', 'accuracy', 'overconfidence', 'sharpness', 'conf_range', 'auroc']
    
    # Collect data for table
    rows = []
    for (model, dataset), methods_stats in all_methods_by_config.items():
        for method_name, bin_stats in methods_stats.items():
            row = {
                'Model': model,
                'Dataset': dataset,
                'Method': METHOD_NAMES.get(method_name, method_name),
                'N_Seeds': bin_stats['n_seeds'],
            }
            
            for metric in all_metrics:
                mean_val = bin_stats.get(f'{metric}_mean', np.nan)
                std_val = bin_stats.get(f'{metric}_std', 0.0)
                
                if not np.isnan(mean_val):
                    info = METRIC_INFO.get(metric, {'multiply': 1})
                    row[f'{metric.upper()}_mean'] = mean_val * info['multiply']
                    row[f'{metric.upper()}_std'] = std_val * info['multiply']
                else:
                    row[f'{metric.upper()}_mean'] = np.nan
                    row[f'{metric.upper()}_std'] = np.nan
            
            rows.append(row)
    
    # Save as CSV
    import csv
    csv_path = output_dir / "metrics_summary.csv"
    if rows:
        fieldnames = rows[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved metrics table: {csv_path}")
    
    # Generate LaTeX table (grouped by model/dataset)
    latex_path = output_dir / "metrics_summary.tex"
    with open(latex_path, 'w') as f:
        f.write("% Auto-generated metrics summary table\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Calibration Metrics Summary}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{llcccccccc}\n")
        f.write("\\toprule\n")
        f.write("Config & Method & ECE$\\downarrow$ & MCE$\\downarrow$ & Brier$\\downarrow$ & Acc$\\uparrow$ & OvConf & Sharp & AUROC$\\uparrow$ \\\\\n")
        f.write("\\midrule\n")
        
        current_config = None
        for row in rows:
            config_str = f"{row['Model']}/{row['Dataset']}"
            if config_str != current_config:
                if current_config is not None:
                    f.write("\\midrule\n")
                current_config = config_str
                config_display = config_str
            else:
                config_display = ""
            
            method = row['Method']
            ece = f"{row['ECE_mean']:.2f}" if not np.isnan(row['ECE_mean']) else "N/A"
            mce = f"{row['MCE_mean']:.2f}" if not np.isnan(row['MCE_mean']) else "N/A"
            brier = f"{row['BRIER_mean']:.4f}" if not np.isnan(row['BRIER_mean']) else "N/A"
            acc = f"{row['ACCURACY_mean']:.1f}" if not np.isnan(row['ACCURACY_mean']) else "N/A"
            ovconf = f"{row['OVERCONFIDENCE_mean']:+.2f}" if not np.isnan(row['OVERCONFIDENCE_mean']) else "N/A"
            sharp = f"{row['SHARPNESS_mean']:.3f}" if not np.isnan(row['SHARPNESS_mean']) else "N/A"
            auroc = f"{row['AUROC_mean']:.3f}" if not np.isnan(row['AUROC_mean']) else "N/A"
            
            f.write(f"{config_display} & {method} & {ece} & {mce} & {brier} & {acc} & {ovconf} & {sharp} & {auroc} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved LaTeX table: {latex_path}")
    
    return rows


def print_metrics_summary(all_methods_by_config):
    """Print a nicely formatted metrics summary to console."""
    print("\n" + "="*100)
    print("METRICS SUMMARY")
    print("="*100)
    
    for (model, dataset), methods_stats in sorted(all_methods_by_config.items()):
        print(f"\n{model.upper()} on {dataset.upper()}")
        print("-"*90)
        
        # Header
        header = f"{'Method':<20} {'ECE':>8} {'MCE':>8} {'Brier':>8} {'Acc':>8} {'OvConf':>8} {'Sharp':>8} {'AUROC':>8}"
        print(header)
        print("-"*90)
        
        for method_name in ['uncalibrated', 'temperature_scaling', 'top_label_isotonic', 'sgc', 'sgc_trust_score', 'sgc_lite', 'dac']:
            if method_name not in methods_stats:
                continue
            
            bin_stats = methods_stats[method_name]
            display_name = METHOD_NAMES.get(method_name, method_name)
            
            ece = bin_stats.get('ece_mean', np.nan) * 100
            mce = bin_stats.get('mce_mean', np.nan) * 100
            brier = bin_stats.get('brier_mean', np.nan)
            acc = bin_stats.get('accuracy_mean', np.nan) * 100
            ovconf = bin_stats.get('overconfidence_mean', np.nan) * 100
            sharp = bin_stats.get('sharpness_mean', np.nan)
            auroc = bin_stats.get('auroc_mean', np.nan)
            
            row = f"{display_name:<20} {ece:>7.2f}% {mce:>7.2f}% {brier:>8.4f} {acc:>7.1f}% {ovconf:>+7.2f}% {sharp:>8.3f} {auroc:>8.3f}"
            print(row)
    
    print("\n" + "="*100)


# =============================================================================
# MAIN
# =============================================================================
def main(config=None, preset=None):
    """
    Main function to generate reliability diagrams.
    
    Args:
        config: Configuration dict (overrides CONFIG)
        preset: Name of preset to use (overrides config)
    """
    # Build final config
    final_config = CONFIG.copy()
    if preset and preset in PRESETS:
        final_config.update(PRESETS[preset])
        print(f"Using preset: {preset}")
    if config:
        final_config.update(config)
    
    base_dir = final_config['base_dir']
    output_dir = Path(final_config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print(f"Loading data from {base_dir}...")
    grouped_data = load_and_group_data(base_dir)
    print(f"Found {len(grouped_data)} (model, dataset, method) combinations")
    
    # Group by (model, dataset) for comparison plots
    all_methods_by_config = defaultdict(dict)
    
    for (model, dataset, method), data_list in grouped_data.items():
        print(f"  Processing: {model} / {dataset} / {method} ({len(data_list)} seeds)")
        bin_stats = compute_bin_statistics(
            data_list, 
            n_bins=final_config['n_bins'],
            bin_type=final_config['bin_type']
        )
        all_methods_by_config[(model, dataset)][method] = bin_stats
    
    # Print metrics summary to console
    print_metrics_summary(all_methods_by_config)
    
    # Generate metrics table if requested
    if final_config.get('generate_metrics_table', False):
        generate_metrics_table(all_methods_by_config, output_dir, final_config)
    
    # Generate comparison plots
    print(f"\nGenerating comparison plots...")
    print(f"  Style: {final_config['style']}")
    print(f"  Binning: {final_config['bin_type']}")
    print(f"  Histogram: {final_config['show_histogram']}")
    print(f"  Display metrics: {final_config.get('display_metrics', ['ece'])}")
    
    for (model, dataset), methods_stats in all_methods_by_config.items():
        if len(methods_stats) > 1:
            save_path = output_dir / f"{model}_{dataset}_comparison"
            plot_comparison(methods_stats, model, dataset, str(save_path), final_config)
    
    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate reliability diagrams with comprehensive metrics')
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='Use a preset configuration')
    parser.add_argument('--style', type=str, choices=['bars', 'curves', 'weighted_alpha'],
                        help='Visualization style')
    parser.add_argument('--bin-type', type=str, choices=['equal_width', 'adaptive'],
                        help='Binning type')
    parser.add_argument('--no-histogram', action='store_true',
                        help='Disable histogram subplot')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--base-dir', type=str, help='Input data directory')
    parser.add_argument('--metrics', type=str, nargs='+', 
                        choices=['ece', 'mce', 'brier', 'nll', 'accuracy', 'overconfidence', 'sharpness', 'auroc'],
                        help='Metrics to display in plot titles')
    parser.add_argument('--generate-table', action='store_true',
                        help='Generate CSV and LaTeX metrics summary table')
    
    args = parser.parse_args()
    
    # Build config from args
    config_overrides = {}
    if args.style:
        config_overrides['style'] = args.style
    if args.bin_type:
        config_overrides['bin_type'] = args.bin_type
    if args.no_histogram:
        config_overrides['show_histogram'] = False
    if args.output_dir:
        config_overrides['output_dir'] = args.output_dir
    if args.base_dir:
        config_overrides['base_dir'] = args.base_dir
    if args.metrics:
        config_overrides['display_metrics'] = args.metrics
    if args.generate_table:
        config_overrides['generate_metrics_table'] = True
    
    main(config=config_overrides, preset=args.preset)