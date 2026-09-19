"""
Compression Diagnostic Analysis
Deep dive into when/why compression succeeds or fails.
Helps answer: "Which configurations are most robust to compression?"
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List


ORIGINAL_DIM_BINS = [
    ("Very Low (<512)", 0, 511),
    ("Low (512-2K)", 512, 2047),
    ("Medium (2K-8K)", 2048, 8191),
    ("High (8K-16K)", 8192, 16383),
    ("Very High (16K-32K)", 16384, 32767),
    ("Extremely High (>32K)", 32768, float("inf")),
]


def assign_original_dim_bin(value: float) -> str:
    """Categorize an original dimensionality into predefined bins."""
    if value is None or pd.isna(value):
        return None
    try:
        dim = float(value)
    except (TypeError, ValueError):
        return None
    
    for label, lower, upper in ORIGINAL_DIM_BINS:
        if lower <= dim <= upper:
            return label
    return None


def format_compression_ratio(value: float) -> str:
    """Format a compression ratio (×) for presentation."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if ratio <= 0:
        return "N/A"
    if np.isclose(ratio, round(ratio)):
        return f"{int(round(ratio))}×"
    if ratio >= 10:
        return f"{ratio:.1f}×"
    return f"{ratio:.2f}×"


def analyze_compression_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify patterns in compression robustness:
    - Which datasets handle compression best?
    - Which models are most compression-robust?
    - Does training method matter?
    """
    pattern_analysis: List[Dict] = []
    
    df_with_bins = df.copy()
    df_with_bins['original_dim_bin'] = df_with_bins['original_dim'].apply(assign_original_dim_bin)
    
    for layer_type in df_with_bins['layer_type'].dropna().unique():
        layer_df = df_with_bins[df_with_bins['layer_type'] == layer_type]
        
        for dim_bin in layer_df['original_dim_bin'].dropna().unique():
            dim_bin_df = layer_df[layer_df['original_dim_bin'] == dim_bin]
            
            for compression_ratio, ratio_df in dim_bin_df.groupby('compression_ratio'):
                avg_original_dim = ratio_df['original_dim'].dropna().mean()
                avg_compression_ratio = ratio_df['compression_ratio'].dropna().astype(float).mean()
                
                by_dataset = ratio_df.groupby('dataset').agg({
                    'rel_ece_mean': 'mean',
                    'frac_within_5pct': 'mean'
                }).reset_index()
                by_model = ratio_df.groupby('model').agg({
                    'rel_ece_mean': 'mean',
                    'frac_within_5pct': 'mean'
                }).reset_index()
                
                dataset_variance = by_dataset['rel_ece_mean'].std() if not by_dataset.empty else np.nan
                model_variance = by_model['rel_ece_mean'].std() if not by_model.empty else np.nan
                
                best_dataset = worst_dataset = None
                if not by_dataset.empty:
                    dataset_non_na = by_dataset.dropna(subset=['rel_ece_mean'])
                    if not dataset_non_na.empty:
                        best_idx = (dataset_non_na['rel_ece_mean'].abs()).idxmin()
                        worst_idx = (dataset_non_na['rel_ece_mean'].abs()).idxmax()
                        best_dataset = dataset_non_na.loc[best_idx, 'dataset']
                        worst_dataset = dataset_non_na.loc[worst_idx, 'dataset']
                
                best_model = worst_model = None
                if not by_model.empty:
                    model_non_na = by_model.dropna(subset=['rel_ece_mean'])
                    if not model_non_na.empty:
                        best_idx = (model_non_na['rel_ece_mean'].abs()).idxmin()
                        worst_idx = (model_non_na['rel_ece_mean'].abs()).idxmax()
                        best_model = model_non_na.loc[best_idx, 'model']
                        worst_model = model_non_na.loc[worst_idx, 'model']
                
                pattern_analysis.append({
                    'layer_type': layer_type,
                    'original_dim_bin': dim_bin,
                    'compression_ratio': compression_ratio,
                    'avg_original_dim': avg_original_dim,
                    'avg_compression_ratio': avg_compression_ratio,
                    'dataset_variance': dataset_variance,
                    'model_variance': model_variance,
                    'best_dataset': best_dataset,
                    'worst_dataset': worst_dataset,
                    'best_model': best_model,
                    'worst_model': worst_model
                })
    
    return pd.DataFrame(pattern_analysis)


def identify_failure_modes(df: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """
    Identify configurations where compression fails badly (>threshold% ECE increase).
    """
    failures = df[df['rel_ece_mean'] > threshold].copy()
    
    if len(failures) == 0:
        print(f"No failures found (threshold: {threshold}%)")
        return pd.DataFrame()
    
    failures['compression_ratio_value'] = pd.to_numeric(failures['compression_ratio'], errors='coerce')
    failures['compression_ratio_label'] = failures['compression_ratio_value'].apply(format_compression_ratio)
    
    failures['severity'] = pd.cut(
        failures['rel_ece_mean'],
        bins=[10, 20, 50, 100, float('inf')],
        labels=['Moderate', 'Severe', 'Critical', 'Catastrophic']
    )
    
    failure_summary = failures.groupby(['layer_type', 'compression_ratio_label']).agg({
        'rel_ece_mean': ['count', 'mean', 'max'],
        'dataset': lambda x: x.mode()[0] if len(x) > 0 else None,
        'model': lambda x: x.mode()[0] if len(x) > 0 else None
    }).reset_index()
    
    return failures, failure_summary


def compute_compression_curves(df: pd.DataFrame) -> Dict:
    """
    Fit curves to understand compression-ECE relationship.
    Helps predict safe compression levels.
    """
    curves = {}
    
    for layer_type in df['layer_type'].unique():
        layer_df = df[df['layer_type'] == layer_type].copy()
        layer_df['compression_ratio_value'] = pd.to_numeric(layer_df['compression_ratio'], errors='coerce')
        layer_df = layer_df.dropna(subset=['compression_ratio_value'])
        if layer_df.empty:
            continue
        
        grouped = layer_df.groupby('compression_ratio_value').agg({
            'rel_ece_mean': 'mean'
        }).reset_index()
        
        x = np.log2(grouped['compression_ratio_value'].values)
        y = grouped['rel_ece_mean'].values
        
        valid_mask = np.isfinite(x) & np.isfinite(y)
        if valid_mask.sum() > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x[valid_mask], y[valid_mask]
            )
            
            curves[layer_type] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'interpretation': 'degrades' if slope > 0 else 'improves',
                'safe_dim_for_5pct': int(2 ** ((5 - intercept) / slope)) if slope != 0 else None
            }
    
    return curves


def plot_compression_heatmap(df: pd.DataFrame, output_path: Path,
                            layer_type: str = 'best_layer') -> None:
    """
    Heatmap: configurations × dimensions showing relative ECE change.
    """
    filtered = df[df['layer_type'] == layer_type].copy()
    filtered['compression_ratio_value'] = pd.to_numeric(filtered['compression_ratio'], errors='coerce')
    filtered = filtered.dropna(subset=['compression_ratio_value'])
    if filtered.empty:
        print(f"No valid compression ratios for heatmap ({layer_type}), skipping.")
        return
    filtered['config'] = (filtered['dataset'] + '_' + 
                         filtered['model'] + '_' + 
                         filtered['training_loss'])
    filtered['compression_ratio_display'] = filtered['compression_ratio_value'].apply(format_compression_ratio)
    
    pivot = filtered.pivot_table(
        values='rel_ece_mean',
        index='config',
        columns='compression_ratio_display',
        aggfunc='mean'
    )
    
    ratio_order = (
        filtered[['compression_ratio_display', 'compression_ratio_value']]
        .drop_duplicates()
        .sort_values('compression_ratio_value')
    )
    pivot = pivot.reindex(columns=ratio_order['compression_ratio_display'])
    
    fig, ax = plt.subplots(figsize=(12, len(pivot) * 0.4 + 2))
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        center=0,
        vmin=-10,
        vmax=20,
        cbar_kws={'label': 'Relative ECE Change (%)'},
        ax=ax
    )
    
    ax.set_xlabel('Compression Ratio (×)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_title(f'Compression Impact Heatmap ({layer_type.replace("_", " ").title()})',
                fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap: {output_path}")


def plot_robustness_by_factor(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Separate plots showing how each factor affects compression robustness.
    """
    factors = ['dataset', 'model', 'training_loss']
    
    for factor in factors:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for layer_idx, layer_type in enumerate(['best_layer', 'data_layer']):
            ax = axes[layer_idx]
            layer_df = df[df['layer_type'] == layer_type].copy()
            layer_df['compression_ratio_value'] = pd.to_numeric(layer_df['compression_ratio'], errors='coerce')
            layer_df = layer_df.dropna(subset=['compression_ratio_value'])
            if layer_df.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
                continue
            
            grouped = layer_df.groupby([factor, 'compression_ratio_value']).agg({
                'rel_ece_mean': 'mean'
            }).reset_index()
            
            for factor_value in grouped[factor].unique():
                factor_data = grouped[grouped[factor] == factor_value]
                factor_data = factor_data.sort_values('compression_ratio_value')
                if factor_data.empty:
                    continue
                
                ax.plot(
                    factor_data['compression_ratio_value'],
                    factor_data['rel_ece_mean'],
                    marker='o',
                    label=factor_value,
                    linewidth=2,
                    markersize=6
                )
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax.set_xscale('log', base=2)
            ax.set_xlabel('Compression Ratio (×)', fontsize=11)
            ax.set_ylabel('Relative ECE Change (%)', fontsize=11)
            ax.set_title(f'{layer_type.replace("_", " ").title()}', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)
        
        plt.suptitle(f'Compression Robustness by {factor.title()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = output_dir / f"robustness_by_{factor}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved factor analysis: {output_path}")


def generate_diagnostic_report(df: pd.DataFrame, 
                              patterns: pd.DataFrame,
                              curves: Dict,
                              output_path: Path) -> None:
    """
    Generate comprehensive diagnostic text report using actual layer dimensions.
    """
    lines = [
        "COMPRESSION DIAGNOSTIC ANALYSIS",
        "=" * 70,
        "NOTE: All analyses use actual layer original_dim values",
        "=" * 70,
        "",
        "1. LAYER DIMENSIONALITY DISTRIBUTION",
        "-" * 70
    ]
    
    if 'original_dim' in df.columns:
        dim_stats = df.groupby('layer_type')['original_dim'].describe()
        lines.append("\nOriginal dimension statistics by layer type:")
        lines.append(dim_stats.to_string())
        lines.append("")
    
    lines.extend([
        "",
        "2. COMPRESSION CURVE CHARACTERISTICS BY DIMENSION",
        "-" * 70
    ])
    
    if curves:
        for layer_type, curve_data in curves.items():
            lines.append(f"\n{layer_type.replace('_', ' ').title()}:")
            lines.append(f"  Trend: {curve_data['interpretation']} with compression")
            lines.append(f"  R² = {curve_data['r_squared']:.3f}")
            lines.append(f"  Slope = {curve_data['slope']:.3f} % per log₂(dim)")
            safe_dim = curve_data.get('safe_dim_for_5pct')
            if safe_dim:
                lines.append(f"  Predicted safe dim (5% threshold): {safe_dim}")
    
    df_with_bins = df.copy()
    df_with_bins['original_dim_bin'] = df_with_bins['original_dim'].apply(assign_original_dim_bin)
    
    def compute_compression_factors(frame: pd.DataFrame) -> pd.Series:
        original = pd.to_numeric(frame['original_dim'], errors='coerce')
        target = pd.to_numeric(frame['target_dim'], errors='coerce')
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where((original > 0) & (target > 0), original / target, np.nan)
        return pd.Series(ratios, index=frame.index)
    
    for layer_type in df_with_bins['layer_type'].dropna().unique():
        lines.append(f"\n{layer_type.replace('_', ' ').title()}:")
        
        for bin_label, _, _ in ORIGINAL_DIM_BINS:
            bin_df = df_with_bins[
                (df_with_bins['layer_type'] == layer_type) &
                (df_with_bins['original_dim_bin'] == bin_label)
            ]
            if bin_df.empty:
                continue
            
            compression_factors = compute_compression_factors(bin_df)
            avg_compression_factor = float(np.nanmean(compression_factors)) if len(compression_factors) > 0 else np.nan
            n_layers = bin_df['layer_name'].nunique()
            
            safe_configs = bin_df[bin_df['frac_within_5pct'] > 0.8]
            safe_layers = safe_configs['layer_name'].nunique()
            layer_safe_factors: List[float] = []
            for _, layer_group in safe_configs.groupby('layer_name'):
                layer_original = layer_group['original_dim'].dropna().unique()
                if len(layer_original) == 0:
                    continue
                original_dim = float(layer_original[0])
                targets = layer_group['target_dim'].dropna()
                if targets.empty:
                    continue
                min_target = targets.min()
                if min_target > 0:
                    layer_safe_factors.append(original_dim / min_target)
            avg_max_safe_factor = float(np.nanmean(layer_safe_factors)) if layer_safe_factors else np.nan
            
            lines.append(f"  {bin_label}:")
            lines.append(f"    Layers covered: {n_layers}")
            if not np.isnan(avg_compression_factor):
                lines.append(f"    Avg compression factor tested: {avg_compression_factor:.2f}×")
            if not np.isnan(avg_max_safe_factor):
                lines.append(f"    Avg max safe factor (Good≤5%>0.8): {avg_max_safe_factor:.2f}× "
                             f"across {safe_layers} layers")
            else:
                lines.append("    No safe compression observed (Good≤5%>0.8)")
            
            if not patterns.empty:
                bin_patterns = patterns[
                    (patterns['layer_type'] == layer_type) &
                    (patterns['original_dim_bin'] == bin_label)
                ]
                if not bin_patterns.empty:
                    dataset_sigma = bin_patterns['dataset_variance'].dropna().mean()
                    model_sigma = bin_patterns['model_variance'].dropna().mean()
                    lines.append(
                        f"    Avg dataset σ: {dataset_sigma:.2f}% | Avg model σ: {model_sigma:.2f}%"
                        if not np.isnan(dataset_sigma) and not np.isnan(model_sigma)
                        else "    Variance data insufficient"
                    )
                    common_best_dataset = bin_patterns['best_dataset'].dropna()
                    if not common_best_dataset.empty:
                        lines.append(f"    Most stable dataset(s): {common_best_dataset.mode().iat[0]}")
                    common_best_model = bin_patterns['best_model'].dropna()
                    if not common_best_model.empty:
                        lines.append(f"    Most stable model(s): {common_best_model.mode().iat[0]}")
    
    lines.extend([
        "",
        "3. COMPRESSION SAFETY RECOMMENDATIONS BY ORIGINAL DIM",
        "-" * 70
    ])
    
    for layer_type in df_with_bins['layer_type'].dropna().unique():
        lines.append(f"\n{layer_type.replace('_', ' ').title()}:")
        for bin_label, _, _ in ORIGINAL_DIM_BINS:
            bin_df = df_with_bins[
                (df_with_bins['layer_type'] == layer_type) &
                (df_with_bins['original_dim_bin'] == bin_label)
            ]
            if bin_df.empty:
                continue
            
            avg_orig_dim = bin_df['original_dim'].dropna().mean()
            safe_ratios = bin_df[bin_df['frac_within_5pct'] >= 0.80]['compression_ratio'].dropna().unique()
            safe_ratios = np.sort(safe_ratios) if len(safe_ratios) > 0 else []
            
            header = f"  {bin_label}"
            if not np.isnan(avg_orig_dim):
                header += f" (avg {int(round(avg_orig_dim))}D):"
            else:
                header += ":"
            lines.append(header)
            
            if len(safe_ratios) > 0:
                safe_ratio_list = ", ".join(format_compression_ratio(ratio) for ratio in safe_ratios)
                lines.append(f"    Safe compression ratios: {safe_ratio_list}")
                lines.append(f"    Recommended max safe ratio: {format_compression_ratio(safe_ratios[-1])}")
            else:
                lines.append("    No safe compression ratios (Good≤5% ≥ 0.80) identified")
    
    output_path.write_text("\n".join(lines))
    print(f"Saved diagnostic report: {output_path}")


def analyze_compression_methods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare different compression methods and configurations.
    """
    cols = ['compression_method', 'pyramid_levels_str', 'mid_channels']
    available_cols = [c for c in cols if c in df.columns]
    
    if not available_cols:
        print("Warning: No compression method columns found in dataframe")
        return pd.DataFrame()
    
    # Filter for valid compression results (ignore uncompressed baselines if marked as such, 
    # though they might be useful for comparison. Usually ratio ~ 1.0 is uncompressed)
    # We want to see how methods perform at significant compression.
    # Let's filter for ratio >= 2.0 to see impact.
    
    compressed_df = df[pd.to_numeric(df['compression_ratio'], errors='coerce') >= 1.5].copy()
    if compressed_df.empty:
        print("No compressed results (ratio >= 1.5) found for method analysis")
        return pd.DataFrame()

    grouped = compressed_df.groupby(available_cols, dropna=False).agg({
        'rel_ece_mean': ['mean', 'std', 'count'],
        'frac_within_5pct': 'mean',
        'feature_time_mean': 'mean',
        'compression_ratio': 'mean'
    }).reset_index()
    
    # Flatten columns
    grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]
    
    # Sort by performance (lower ECE change is better, or higher frac within 5%)
    grouped = grouped.sort_values('rel_ece_mean_mean')
    
    return grouped


def main():
    parser = argparse.ArgumentParser(
        description="Deep diagnostic analysis of compression experiments"
    )
    parser.add_argument(
        '--aggregated_csv',
        type=str,
        required=True,
        help='Path to aggregated_results.csv from aggregate_compression_paper.py'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='aaai_full_experiments/results/compression_diagnostics',
        help='Directory for diagnostic outputs'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading aggregated data...")
    df = pd.read_csv(args.aggregated_csv)
    
    print("\nAnalyzing compression patterns...")
    patterns = analyze_compression_patterns(df)
    patterns.to_csv(output_dir / "compression_patterns.csv", index=False)
    
    print("\nAnalyzing compression methods...")
    method_analysis = analyze_compression_methods(df)
    if not method_analysis.empty:
        method_analysis.to_csv(output_dir / "method_comparison.csv", index=False)
        print("Method comparison:")
        print(method_analysis.to_string())
    
    print("Fitting compression curves...")
    curves = compute_compression_curves(df)
    
    print("Identifying failure modes...")
    failures, failure_summary = identify_failure_modes(df, threshold=10.0)
    if len(failures) > 0:
        failures.to_csv(output_dir / "compression_failures.csv", index=False)
        failure_summary.to_csv(output_dir / "failure_summary.csv", index=False)
    
    print("\nGenerating visualizations...")
    plot_compression_heatmap(df, output_dir / "compression_heatmap_best.png", 'best_layer')
    plot_compression_heatmap(df, output_dir / "compression_heatmap_data.png", 'data_layer')
    plot_robustness_by_factor(df, output_dir)
    
    print("\nGenerating diagnostic report...")
    generate_diagnostic_report(
        df, patterns, curves,
        output_dir / "diagnostic_report.txt"
    )
    
    print(f"\n{'='*70}")
    print(f"Diagnostics saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()