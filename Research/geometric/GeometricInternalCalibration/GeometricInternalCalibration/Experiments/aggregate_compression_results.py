import json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_DIR = Path("/home/ptamar/geometric-internal-calibration/aaai_full_experiments/results/compression_experiments")
OUTPUT_DIR = Path("compression_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

def collect_all_results():
    """Recursively collect all JSON result files"""
    all_results = []
    
    for json_file in BASE_DIR.rglob("*_results.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Parse path components
            parts = json_file.relative_to(BASE_DIR).parts
            if len(parts) >= 5:
                data['training_loss'] = parts[0]
                data['dataset'] = parts[1]
                data['model'] = parts[2]
                # seed is already in data
            
            data['file_path'] = str(json_file)
            all_results.append(data)
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return pd.DataFrame(all_results)

def analyze_compression_tradeoffs(df):
    """Analyze ECE vs compression ratio"""
    print("\n" + "="*80)
    print("COMPRESSION VS PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Group by compression ratio
    compression_summary = df.groupby('compression_ratio').agg({
        'ece': ['mean', 'std', 'min', 'max'],
        'accuracy': ['mean', 'std'],
        'target_dim': 'first',
        'original_dim': 'first'
    }).round(6)
    
    print("\n📊 Performance by Compression Ratio:")
    print(compression_summary)
    
    # Best compression ratio (lowest ECE)
    best_compression = df.loc[df.groupby('compression_ratio')['ece'].idxmin()]
    print("\n🏆 Best ECE for each compression ratio:")
    print(best_compression[['compression_ratio', 'target_dim', 'ece', 'accuracy', 
                           'dataset', 'model', 'seed']].to_string(index=False))

def analyze_timing(df):
    """Analyze timing metrics"""
    print("\n" + "="*80)
    print("TIMING ANALYSIS")
    print("="*80)
    
    timing_cols = ['feature_extract_time_s', 'fit_time_s', 'calibrate_time_s', 
                   'calibrate_time_per_image_s', 'calibrate_fps']
    
    # Overall timing statistics
    print("\n⏱️  Overall Timing Statistics:")
    timing_stats = df[timing_cols].describe().round(3)
    print(timing_stats)
    
    # Timing by compression ratio
    print("\n⏱️  Timing by Compression Ratio:")
    timing_by_compression = df.groupby('compression_ratio')[timing_cols].mean().round(3)
    print(timing_by_compression)
    
    # Timing by dataset and model
    print("\n⏱️  Timing by Dataset and Model:")
    timing_by_config = df.groupby(['dataset', 'model'])[timing_cols].mean().round(3)
    print(timing_by_config)
    
    # Total time analysis
    df['total_time_s'] = df['feature_extract_time_s'] + df['fit_time_s'] + df['calibrate_time_s']
    print("\n⏱️  Total Time Statistics:")
    print(df.groupby('compression_ratio')['total_time_s'].describe().round(2))

def analyze_by_dataset_model(df):
    """Analyze results stratified by dataset and model"""
    print("\n" + "="*80)
    print("DATASET & MODEL ANALYSIS")
    print("="*80)
    
    for dataset in df['dataset'].unique():
        for model in df['model'].unique():
            subset = df[(df['dataset'] == dataset) & (df['model'] == model)]
            if len(subset) == 0:
                continue
                
            print(f"\n📁 {dataset.upper()} - {model}")
            print("-" * 60)
            
            summary = subset.groupby('compression_ratio').agg({
                'ece': ['mean', 'std', 'count'],
                'accuracy': 'mean',
                'calibrate_time_s': 'mean'
            }).round(6)
            print(summary)

def find_optimal_compression(df):
    """Find optimal compression ratios based on different criteria"""
    print("\n" + "="*80)
    print("OPTIMAL COMPRESSION RATIOS")
    print("="*80)
    
    # For each dataset-model combination
    for (dataset, model), group in df.groupby(['dataset', 'model']):
        print(f"\n🎯 {dataset} - {model}:")
        
        # Average across seeds for each compression ratio
        avg_by_compression = group.groupby('compression_ratio').agg({
            'ece': 'mean',
            'accuracy': 'mean',
            'calibrate_time_s': 'mean',
            'target_dim': 'first'
        }).reset_index()
        
        # Best ECE
        best_ece_idx = avg_by_compression['ece'].idxmin()
        best_ece = avg_by_compression.loc[best_ece_idx]
        print(f"  • Best ECE: {best_ece['compression_ratio']:.1f}x "
              f"(dim={best_ece['target_dim']:.0f}, ECE={best_ece['ece']:.6f})")
        
        # Best speed (if time matters)
        fastest_idx = avg_by_compression['calibrate_time_s'].idxmin()
        fastest = avg_by_compression.loc[fastest_idx]
        print(f"  • Fastest: {fastest['compression_ratio']:.1f}x "
              f"(dim={fastest['target_dim']:.0f}, time={fastest['calibrate_time_s']:.2f}s)")
        
        # Best tradeoff (e.g., ECE < 1.5x best, fastest time)
        threshold_ece = best_ece['ece'] * 1.5
        candidates = avg_by_compression[avg_by_compression['ece'] <= threshold_ece]
        if len(candidates) > 0:
            tradeoff_idx = candidates['calibrate_time_s'].idxmin()
            tradeoff = candidates.loc[tradeoff_idx]
            print(f"  • Best tradeoff: {tradeoff['compression_ratio']:.1f}x "
                  f"(dim={tradeoff['target_dim']:.0f}, ECE={tradeoff['ece']:.6f}, "
                  f"time={tradeoff['calibrate_time_s']:.2f}s)")

def calculate_relative_ece_change(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate relative ECE change vs baseline (compression_ratio=1.0 or closest)."""
    df_with_relative = df.copy()
    if not {'dataset', 'model', 'seed', 'compression_ratio', 'ece'}.issubset(df_with_relative.columns):
        return df_with_relative
    for (dataset, model, seed), group in df_with_relative.groupby(['dataset', 'model', 'seed']):
        baseline_rows = group[group['compression_ratio'] == 1.0]
        if len(baseline_rows) > 0:
            baseline_ece = baseline_rows['ece'].values[0]
        else:
            closest_idx = (group['compression_ratio'] - 1.0).abs().idxmin()
            baseline_ece = group.loc[closest_idx, 'ece']
        mask = (
            (df_with_relative['dataset'] == dataset) &
            (df_with_relative['model'] == model) &
            (df_with_relative['seed'] == seed)
        )
        df_with_relative.loc[mask, 'baseline_ece'] = baseline_ece
        df_with_relative.loc[mask, 'relative_ece_change'] = (
            (df_with_relative.loc[mask, 'ece'] - baseline_ece) / baseline_ece * 100.0
        )
    return df_with_relative

def analyze_by_target_dimension(df_with_relative: pd.DataFrame) -> None:
    """Analyze results by target dimension (advisor-focused view)."""
    print("\n" + "="*80)
    print("TARGET DIMENSION ANALYSIS")
    print("="*80)
    if 'target_dim' not in df_with_relative.columns:
        print("target_dim column missing; skipping.")
        return
    dim_summary = df_with_relative.groupby('target_dim').agg({
        'ece': ['mean', 'std', 'min'],
        'relative_ece_change': ['mean', 'std'],
        'calibrate_time_s': ['mean'],
        'original_dim': lambda x: f"{x.min():.0f}-{x.max():.0f}"
    }).round(4)
    print("\n📏 Performance by Target Dimension:")
    print(dim_summary)
    print("\n🎯 Optimal Target Dimensions:")
    if not {'dataset', 'model'}.issubset(df_with_relative.columns):
        return
    for (dataset, model), group in df_with_relative.groupby(['dataset', 'model']):
        valid = group.dropna(subset=['target_dim', 'relative_ece_change'])
        if len(valid) == 0:
            continue
        avg_by_dim = valid.groupby('target_dim').agg({
            'relative_ece_change': 'mean',
            'ece': 'mean',
            'calibrate_time_s': 'mean'
        }).reset_index()
        best_idx = avg_by_dim['relative_ece_change'].idxmin()
        best = avg_by_dim.loc[best_idx]
        print(f"\n  {dataset}-{model}:")
        print(f"    • Best dimension: {best['target_dim']:.0f} "
              f"(rel_change={best['relative_ece_change']:.2f}%, "
              f"ECE={best['ece']:.6f}, time={best['calibrate_time_s']:.1f}s)")
        acceptable = avg_by_dim[avg_by_dim['relative_ece_change'] < 20]
        if len(acceptable) > 0:
            smallest_dim = acceptable.loc[acceptable['target_dim'].idxmin()]
            print(f"    • Smallest acceptable dim: {smallest_dim['target_dim']:.0f} "
                  f"(rel_change={smallest_dim['relative_ece_change']:.2f}%, "
                  f"time={smallest_dim['calibrate_time_s']:.1f}s)")

def plot_compression_analysis(df):
    """Create visualization plots - both compression ratio and target dimension views"""
    print("\n📈 Creating visualizations...")
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    # Styling by layer type and dataset-model
    layer_styles = {
        'best_layer': {'marker': 'o', 'linestyle': '-', 'linewidth': 2.5},
        'worst_layer': {'marker': 's', 'linestyle': '--', 'linewidth': 2},
        'data_layer': {'marker': '^', 'linestyle': ':', 'linewidth': 2},
    }
    color_map = {
        'cifar10-densenet121': '#1f77b4',
        'cifar10-resnet18': '#ff7f0e',
        'cifar10-resnet50': '#2ca02c',
        'cifar100-resnet18': '#d62728',
        'cifar100-densenet121': '#9467bd',
        'cifar100-resnet50': '#8c564b',
    }
    # ----------------------- PART 1: COMPRESSION RATIO VIEW --------------------
    fig1, axes = plt.subplots(2, 2, figsize=(16, 11))
    # ECE vs Compression Ratio - with layer type
    ax = axes[0, 0]
    for (dataset, model, layer_type), group in df.groupby(['dataset', 'model', 'layer_type']):
        valid_group = group.dropna(subset=['compression_ratio', 'ece'])
        if len(valid_group) == 0:
            continue
        avg_data = valid_group.groupby('compression_ratio').agg({'ece': ['mean', 'std']}).reset_index()
        avg_data.columns = ['compression_ratio', 'ece_mean', 'ece_std']
        color = color_map.get(f'{dataset}-{model}', 'gray')
        style = layer_styles.get(layer_type, {'marker': 'o', 'linestyle': '-', 'linewidth': 2})
        ax.errorbar(avg_data['compression_ratio'], avg_data['ece_mean'],
                    yerr=avg_data['ece_std'],
                    color=color,
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    label=f"{dataset}-{model}-{layer_type.replace('_', ' ')}",
                    capsize=5, markersize=6, alpha=0.8)
    ax.set_xlabel('Compression Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('ECE', fontsize=12, fontweight='bold')
    ax.set_title('ECE vs Compression Ratio (by Layer Type)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.005, 0.08)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    # Timing vs Compression Ratio - with layer type
    ax = axes[0, 1]
    for (dataset, model, layer_type), group in df.groupby(['dataset', 'model', 'layer_type']):
        valid_group = group.dropna(subset=['compression_ratio', 'calibrate_time_s'])
        if len(valid_group) == 0:
            continue
        avg_data = valid_group.groupby('compression_ratio')['calibrate_time_s'].mean().reset_index()
        color = color_map.get(f'{dataset}-{model}', 'gray')
        style = layer_styles.get(layer_type, {'marker': 'o', 'linestyle': '-', 'linewidth': 2})
        ax.plot(avg_data['compression_ratio'], avg_data['calibrate_time_s'],
                color=color,
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                label=f"{dataset}-{model}-{layer_type.replace('_', ' ')}",
                markersize=6, alpha=0.8)
    ax.set_xlabel('Compression Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Calibration Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Timing vs Compression Ratio (by Layer Type)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_yscale('log')
    # Time Breakdown - by layer type
    ax = axes[1, 0]
    time_cols = ['feature_extract_time_s', 'fit_time_s', 'calibrate_time_s']
    layer_types = df['layer_type'].dropna().unique().tolist() if 'layer_type' in df.columns else []
    x = np.arange(len(time_cols))
    width = 0.25 if len(layer_types) > 0 else 0.6
    for i, layer_type in enumerate(layer_types):
        layer_data = df[df['layer_type'] == layer_type]
        time_means = layer_data[time_cols].mean()
        offset = (i - len(layer_types) / 2 + 0.5) * width
        bars = ax.bar(x + offset, time_means, width,
                      label=layer_type.replace('_', ' '), edgecolor='black', linewidth=1)
        for bar, val in zip(bars, time_means):
            height = bar.get_height()
            if height > 5:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{val:.0f}s', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['Feature Extract', 'Fit', 'Calibrate'], fontsize=11)
    ax.set_ylabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Average Time Breakdown by Layer Type', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    # FPS vs Compression Ratio - with layer type
    ax = axes[1, 1]
    for (dataset, model, layer_type), group in df.groupby(['dataset', 'model', 'layer_type']):
        valid_group = group.dropna(subset=['compression_ratio', 'calibrate_fps'])
        if len(valid_group) == 0:
            continue
        avg_data = valid_group.groupby('compression_ratio')['calibrate_fps'].mean().reset_index()
        color = color_map.get(f'{dataset}-{model}', 'gray')
        style = layer_styles.get(layer_type, {'marker': 'o', 'linestyle': '-', 'linewidth': 2})
        ax.plot(avg_data['compression_ratio'], avg_data['calibrate_fps'],
                color=color,
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                label=f"{dataset}-{model}-{layer_type.replace('_', ' ')}",
                markersize=6, alpha=0.7)
    ax.set_xlabel('Compression Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Calibration FPS', fontsize=12, fontweight='bold')
    ax.set_title('Throughput vs Compression Ratio (by Layer Type)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    ratio_path = OUTPUT_DIR / 'compression_analysis_by_ratio.png'
    plt.savefig(ratio_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved compression ratio view to {ratio_path}")
    # ----------------------- PART 2: TARGET DIMENSION VIEW ---------------------
    df_with_relative = calculate_relative_ece_change(df)
    fig2, axes = plt.subplots(2, 2, figsize=(16, 11))
    # Relative ECE Change vs Target Dimension - with layer type
    ax = axes[0, 0]
    for (dataset, model, layer_type), group in df_with_relative.groupby(['dataset', 'model', 'layer_type']):
        valid_group = group.dropna(subset=['target_dim', 'relative_ece_change'])
        valid_group = valid_group[valid_group['target_dim'] > 0]
        if len(valid_group) == 0:
            continue
        avg_data = valid_group.groupby('target_dim').agg({'relative_ece_change': ['mean', 'std']}).reset_index()
        avg_data.columns = ['target_dim', 'rel_ece_mean', 'rel_ece_std']
        avg_data = avg_data.sort_values('target_dim')
        color = color_map.get(f'{dataset}-{model}', 'gray')
        style = layer_styles.get(layer_type, {'marker': 'o', 'linestyle': '-', 'linewidth': 2})
        ax.errorbar(avg_data['target_dim'], avg_data['rel_ece_mean'],
                    yerr=avg_data['rel_ece_std'],
                    color=color,
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    label=f"{dataset}-{model}-{layer_type.replace('_', ' ')}",
                    capsize=5, markersize=6, alpha=0.8)
    ax.set_xlabel('Target Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative ECE Change (%)', fontsize=12, fontweight='bold')
    ax.set_title('Relative ECE Change vs Target Dimension (by Layer Type)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax.set_xscale('log', base=2)
    # Absolute ECE vs Target Dimension - with layer type
    ax = axes[0, 1]
    for (dataset, model, layer_type), group in df_with_relative.groupby(['dataset', 'model', 'layer_type']):
        valid_group = group.dropna(subset=['target_dim', 'ece'])
        valid_group = valid_group[valid_group['target_dim'] > 0]
        if len(valid_group) == 0:
            continue
        avg_data = valid_group.groupby('target_dim').agg({'ece': ['mean', 'std']}).reset_index()
        avg_data.columns = ['target_dim', 'ece_mean', 'ece_std']
        avg_data = avg_data.sort_values('target_dim')
        color = color_map.get(f'{dataset}-{model}', 'gray')
        style = layer_styles.get(layer_type, {'marker': 'o', 'linestyle': '-', 'linewidth': 2})
        ax.errorbar(avg_data['target_dim'], avg_data['ece_mean'],
                    yerr=avg_data['ece_std'],
                    color=color,
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    label=f"{dataset}-{model}-{layer_type.replace('_', ' ')}",
                    capsize=5, markersize=6, alpha=0.8)
    ax.set_xlabel('Target Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('ECE', fontsize=12, fontweight='bold')
    ax.set_title('Absolute ECE vs Target Dimension (by Layer Type)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 0.08)
    # Calibration Time vs Target Dimension - with layer type
    ax = axes[1, 0]
    for (dataset, model, layer_type), group in df_with_relative.groupby(['dataset', 'model', 'layer_type']):
        valid_group = group.dropna(subset=['target_dim', 'calibrate_time_s'])
        valid_group = valid_group[valid_group['target_dim'] > 0]
        if len(valid_group) == 0:
            continue
        avg_data = valid_group.groupby('target_dim')['calibrate_time_s'].mean().reset_index()
        avg_data = avg_data.sort_values('target_dim')
        color = color_map.get(f'{dataset}-{model}', 'gray')
        style = layer_styles.get(layer_type, {'marker': 'o', 'linestyle': '-', 'linewidth': 2})
        ax.plot(avg_data['target_dim'], avg_data['calibrate_time_s'],
                color=color,
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                label=f"{dataset}-{model}-{layer_type.replace('_', ' ')}",
                markersize=6, alpha=0.8)
    ax.set_xlabel('Target Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Calibration Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Timing vs Target Dimension (by Layer Type)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    # Pareto Front: ECE vs Time - colored by layer type
    ax = axes[1, 1]
    layer_colors = {
        'best_layer': '#2ca02c',
        'worst_layer': '#d62728',
        'data_layer': '#9467bd',
    }
    for (dataset, model, layer_type), group in df_with_relative.groupby(['dataset', 'model', 'layer_type']):
        valid_group = group.dropna(subset=['calibrate_time_s', 'ece', 'target_dim'])
        valid_group = valid_group[valid_group['target_dim'] > 0]
        if len(valid_group) == 0:
            continue
        avg_data = valid_group.groupby('target_dim').agg({
            'ece': 'mean',
            'calibrate_time_s': 'mean'
        }).reset_index()
        color = layer_colors.get(layer_type, 'gray')
        style = layer_styles.get(layer_type, {'marker': 'o', 'linestyle': '-', 'linewidth': 2})
        ax.scatter(avg_data['calibrate_time_s'], avg_data['ece'],
                   c=color, marker=style['marker'], s=100, alpha=0.7,
                   edgecolors='black', linewidth=1.5,
                   label=f"{dataset}-{model}-{layer_type.replace('_', ' ')}")
        avg_sorted = avg_data.sort_values('target_dim')
        ax.plot(avg_sorted['calibrate_time_s'], avg_sorted['ece'],
                color=color, linestyle=style['linestyle'],
                alpha=0.3, linewidth=1.5)
    ax.set_xlabel('Calibration Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ECE', fontsize=12, fontweight='bold')
    ax.set_title('Quality-Speed Tradeoff by Layer Type', fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    dim_path = OUTPUT_DIR / 'compression_analysis_by_dimension.png'
    plt.savefig(dim_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved target dimension view to {dim_path}")
    # ------------------- PART 3: PAPER-READY PLOTS -----------------------------
    # (a) Relative ECE vs dimension (aggregated)
    fig3a, ax = plt.subplots(1, 1, figsize=(10, 7))
    for (dataset, model), group in df_with_relative.groupby(['dataset', 'model']):
        valid_group = group.dropna(subset=['target_dim', 'relative_ece_change'])
        valid_group = valid_group[valid_group['target_dim'] > 0]
        if len(valid_group) == 0:
            continue
        avg_data = valid_group.groupby('target_dim').agg({
            'relative_ece_change': ['mean', 'std', 'count']
        }).reset_index()
        avg_data.columns = ['target_dim', 'rel_ece_mean', 'rel_ece_std', 'count']
        avg_data = avg_data.sort_values('target_dim')
        ax.errorbar(avg_data['target_dim'], avg_data['rel_ece_mean'],
                    yerr=avg_data['rel_ece_std'], marker='o', label=f"{dataset}-{model}",
                    capsize=5, linewidth=2.5, markersize=8, alpha=0.8)
    ax.set_xlabel('Target Feature Dimension', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative ECE Change (%)', fontsize=14, fontweight='bold')
    ax.set_title('Impact of Dimensionality Reduction on Calibration Quality',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
    ax.set_xscale('log', base=2)
    ax.axhspan(-100, 20, alpha=0.1, color='green')
    plt.tight_layout()
    paper_path = OUTPUT_DIR / 'relative_ece_vs_dimension_paper.png'
    plt.savefig(paper_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved paper-ready plot to {paper_path}")
    # (b) Layer type comparison paper figure
    fig3b, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    for (dataset, model, layer_type), group in df_with_relative.groupby(['dataset', 'model', 'layer_type']):
        valid_group = group.dropna(subset=['target_dim', 'relative_ece_change'])
        valid_group = valid_group[valid_group['target_dim'] > 0]
        if len(valid_group) == 0:
            continue
        avg_data = valid_group.groupby('target_dim').agg({'relative_ece_change': ['mean', 'std']}).reset_index()
        avg_data.columns = ['target_dim', 'rel_ece_mean', 'rel_ece_std']
        avg_data = avg_data.sort_values('target_dim')
        color = color_map.get(f'{dataset}-{model}', 'gray')
        style = layer_styles.get(layer_type, {'marker': 'o', 'linestyle': '-', 'linewidth': 2})
        ax.errorbar(avg_data['target_dim'], avg_data['rel_ece_mean'],
                    yerr=avg_data['rel_ece_std'],
                    color=color, marker=style['marker'], linestyle=style['linestyle'],
                    linewidth=2.5,
                    label=f"{dataset}-{model}-{layer_type.replace('_', ' ')}",
                    capsize=5, markersize=8, alpha=0.8)
    ax.set_xlabel('Target Feature Dimension', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative ECE Change (%)', fontsize=14, fontweight='bold')
    ax.set_title('Compression Impact by Layer Type', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best', ncol=1)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
    ax.set_xscale('log', base=2)
    ax.axhspan(-100, 20, alpha=0.1, color='green')
    ax = axes[1]
    for layer_type in ['best_layer', 'worst_layer', 'data_layer']:
        subset = df_with_relative[
            (df_with_relative.get('dataset') == 'cifar10') &
            (df_with_relative.get('model') == 'resnet18') &
            (df_with_relative.get('layer_type') == layer_type)
        ]
        if len(subset) == 0:
            continue
        valid_group = subset.dropna(subset=['target_dim', 'ece'])
        valid_group = valid_group[valid_group['target_dim'] > 0]
        if len(valid_group) == 0:
            continue
        avg_data = valid_group.groupby('target_dim').agg({'ece': ['mean', 'std']}).reset_index()
        avg_data.columns = ['target_dim', 'ece_mean', 'ece_std']
        avg_data = avg_data.sort_values('target_dim')
        style = layer_styles.get(layer_type, {'marker': 'o', 'linestyle': '-', 'linewidth': 2})
        ax.errorbar(avg_data['target_dim'], avg_data['ece_mean'],
                    yerr=avg_data['ece_std'],
                    marker=style['marker'], linestyle=style['linestyle'],
                    linewidth=2.5,
                    label=layer_type.replace('_', ' ').title(),
                    capsize=5, markersize=10, alpha=0.8)
    ax.set_xlabel('Target Feature Dimension', fontsize=14, fontweight='bold')
    ax.set_ylabel('ECE', fontsize=14, fontweight='bold')
    ax.set_title('Layer Type Comparison (CIFAR-10 ResNet18)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_xscale('log', base=2)
    ax.set_ylim(0, 0.08)
    plt.tight_layout()
    comp_path = OUTPUT_DIR / 'layer_type_comparison_paper.png'
    plt.savefig(comp_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved layer type comparison to {comp_path}")
    # Save data with relative changes
    rel_csv = OUTPUT_DIR / 'compression_results_with_relative_ece.csv'
    try:
        df_with_relative.to_csv(rel_csv, index=False)
        print(f"  ✓ Saved data with relative ECE to {rel_csv}")
    except Exception as e:
        print(f"Failed to save relative ECE CSV: {e}")

def analyze_by_layer_type(df: pd.DataFrame) -> None:
    """Analyze differences between best, worst, and data layers."""
    print("\n" + "="*80)
    print("LAYER TYPE COMPARISON")
    print("="*80)
    if 'layer_type' not in df.columns:
        print("layer_type column missing; skipping.")
        return
    layer_summary = df.groupby('layer_type').agg({
        'ece': ['mean', 'std', 'min', 'max'],
        'accuracy': ['mean', 'std'],
        'calibrate_time_s': ['mean', 'std']
    }).round(6)
    print("\n📊 Performance by Layer Type:")
    print(layer_summary)
    # Compare layer types within same configuration
    print("\n🔍 Layer Type Comparison (same dataset/model/compression):")
    if not {'dataset', 'model', 'compression_ratio'}.issubset(df.columns):
        print("Required columns missing for per-config comparison.")
        return
    for (dataset, model, compression_ratio), group in df.groupby(['dataset', 'model', 'compression_ratio']):
        layer_types_present = group['layer_type'].dropna().unique()
        if len(layer_types_present) < 2:
            continue
        print(f"\n  {dataset}-{model} @ {compression_ratio}x:")
        for layer_type in layer_types_present:
            subset = group[group['layer_type'] == layer_type]
            print(f"    {layer_type}: ECE={subset['ece'].mean():.6f} ± {subset['ece'].std():.6f}")

def main():
    print("🔍 Collecting compression experiment results...")
    df = collect_all_results()
    
    if len(df) == 0:
        print("❌ No results found!")
        return
    
    print(f"✓ Found {len(df)} results")
    print(f"  • Datasets: {df['dataset'].dropna().unique()}")
    print(f"  • Models: {df['model'].dropna().unique()}")
    if 'layer_type' in df.columns:
        print(f"  • Layer types: {df['layer_type'].dropna().unique()}")
    print(f"  • Compression ratios: {sorted(df['compression_ratio'].dropna().unique())}")
    if 'target_dim' in df.columns:
        print(f"  • Target dimensions: {sorted(df['target_dim'].dropna().unique())}")
    if 'seed' in df.columns:
        print(f"  • Seeds: {sorted(df['seed'].dropna().unique())}")
    
    # Save raw data
    csv_path = OUTPUT_DIR / 'compression_results_raw.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved raw data to {csv_path}")
    
    # Run analyses
    analyze_compression_tradeoffs(df)
    analyze_timing(df)
    analyze_by_dataset_model(df)
    analyze_by_layer_type(df)
    find_optimal_compression(df)
    
    # Calculate relative ECE and analyze by dimension
    df_with_relative = calculate_relative_ece_change(df)
    analyze_by_target_dimension(df_with_relative)
    
    # Create all plots (both ratio and dimension views, plus paper plot)
    plot_compression_analysis(df)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()