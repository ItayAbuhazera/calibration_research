"""
Aggregate Compression Results - Paper-Focused Version
Generates publication-ready analyses following advisor's specifications:
X-axis: compression dimension/ratio
Y-axis: relative ECE change
Key insight: compression preserves (or improves) calibration quality
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

sns.set_style("whitegrid")


def load_results(results_dir: Path) -> List[Dict]:
    """Load all compression experiment results."""
    all_results = []
    
    for json_file in results_dir.rglob("*_results.json"):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                all_results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    
    print(f"Loaded {len(all_results)} result files")
    return all_results


def compute_baseline_per_config(results: List[Dict]) -> Dict[Tuple, float]:
    """
    Compute baseline ECE for each configuration.
    Baseline = uncompressed (compression_ratio ~= 1.0) or highest target_dim.
    
    Returns:
        Dict mapping (dataset, model, training_loss, layer_name, seed) -> baseline_ece
    """
    baselines = {}
    
    grouped = defaultdict(list)
    config_keys = set()
    for r in results:
        layer_name = r.get('layer_name')
        if layer_name is None:
            layer_name = r.get('layer_type')
            print(f"Warning: Result missing layer_name; falling back to layer_type for baseline key: "
                  f"{(r.get('dataset'), r.get('model'), r.get('training_loss'), layer_name, r.get('seed'))}")
        key = (r['dataset'], r['model'], r['training_loss'], 
               layer_name, r['seed'])
        grouped[key].append(r)
        config_keys.add((r['dataset'], r['model'], r['training_loss'], layer_name))
    
    for key, group in grouped.items():
        key_without_seed = key[:-1]
        original_dims = {g.get('original_dim') for g in group if g.get('original_dim') is not None}
        if len(original_dims) > 1:
            print(f"Warning: Inconsistent original_dim values for {key_without_seed}: {sorted(original_dims)}")
        
        baseline_candidates = []
        for r in group:
            ratio = r.get('compression_ratio')
            if ratio is None:
                continue
            try:
                ratio_value = float(ratio)
                # Baseline is uncompressed (1.0× compression with 5% tolerance)
                if abs(ratio_value - 1.0) <= 0.05:
                    baseline_candidates.append(r)
            except (TypeError, ValueError):
                continue
        
        if not baseline_candidates:
            baseline_candidates = sorted(
                group, 
                key=lambda x: x.get('target_dim', 0), 
                reverse=True
            )
        
        if baseline_candidates:
            baselines[key] = baseline_candidates[0]['ece']
    
    baseline_config_keys = {(k[0], k[1], k[2], k[3]) for k in baselines.keys()}
    print(f"Found {len(config_keys)} unique (dataset, model, training_loss, layer_name) combinations")
    print(f"{len(baseline_config_keys)} configurations have valid baselines")
    
    missing_configs = config_keys - baseline_config_keys
    if missing_configs:
        for cfg in sorted(missing_configs):
            print(f"Warning: Missing baseline for configuration {cfg}")
    else:
        print("All configurations have valid baselines")
    
    return baselines


def aggregate_with_relative_change(results: List[Dict],
                                   baselines: Dict[Tuple, float] = None) -> pd.DataFrame:
    """
    Aggregate results with proper relative ECE change computation.
    
    For each configuration, compute:
    - relative_ece_change = (ece - baseline_ece) / baseline_ece * 100
    - Aggregate across seeds
    
    Args:
        results: Raw experiment result dictionaries.
        baselines: Optional precomputed baseline ECE values keyed by
            (dataset, model, training_loss, layer_name, seed). If not provided,
            they will be computed internally.
    """
    if baselines is None:
        baselines = compute_baseline_per_config(results)
    
    enriched_results = []
    for r in results:
        layer_name = r.get('layer_name')
        if layer_name is None:
            layer_name = r.get('layer_type')
            print(f"Warning: Result missing layer_name during aggregation; "
                  f"falling back to layer_type for baseline lookup: "
                  f"{(r.get('dataset'), r.get('model'), r.get('training_loss'), layer_name, r.get('seed'))}")
        
        baseline_key = (r['dataset'], r['model'], r['training_loss'], 
                        layer_name, r['seed'])
        
        baseline_ece = baselines.get(baseline_key)
        if baseline_ece is not None and baseline_ece > 0:
            rel_change = (r['ece'] - baseline_ece) / baseline_ece * 100
        else:
            rel_change = None
        
        enriched_results.append({
            **r,
            'baseline_ece_computed': baseline_ece,
            'relative_ece_change_computed': rel_change
        })
    
    df = pd.DataFrame(enriched_results)
    
    df['compression_ratio_requested'] = pd.to_numeric(
        df.get('requested_compression_ratio'), errors='coerce'
    )
    df['compression_ratio_actual'] = pd.to_numeric(
        df.get('compression_ratio'), errors='coerce'
    )
    df['compression_ratio_effective'] = df['compression_ratio_requested'].fillna(df['compression_ratio_actual'])
    df['compression_ratio_key'] = df['compression_ratio_effective'].round(6)
    
    # Validate original_dim consistency per physical layer
    for key, group in df.groupby(['dataset', 'model', 'training_loss', 'layer_name']):
        original_dims = group['original_dim'].dropna().unique()
        if len(original_dims) > 1:
            print(f"Warning: Inconsistent original_dim values for "
                  f"{(key[0], key[1], key[2], key[3])}: {sorted(original_dims)}")
    
    # Ensure missing fields are filled for grouping
    df['compression_method'] = df.get('compression_method').fillna('fixed_spp_jl')
    # Convert pyramid levels list to string for grouping
    df['pyramid_levels_str'] = df['pyramid_levels'].apply(lambda x: str(x) if isinstance(x, list) else str([4, 2, 1]))
    df['mid_channels'] = df.get('mid_channels') # Keep as float/nan

    grouped = df.groupby([
        'dataset', 'model', 'training_loss',
        'layer_name', 'compression_ratio_key', 'target_dim', 'original_dim',
        'compression_method', 'pyramid_levels_str', 'mid_channels'
    ], dropna=False)
    
    def safe_mean(x):
        valid = x.dropna()
        return valid.mean() if len(valid) > 0 else np.nan
    
    def safe_std(x):
        valid = x.dropna()
        return valid.std() if len(valid) > 0 else np.nan
    
    def fraction_within_threshold(x, threshold=5.0):
        valid = x.dropna()
        if len(valid) == 0:
            return np.nan
        return (valid <= threshold).mean()
    
    layer_type_summary = grouped['layer_type'].agg(
        lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]
    ).reset_index(name='layer_type')
    
    aggregated = grouped.agg(
        ece_mean=('ece', 'mean'),
        ece_std=('ece', 'std'),
        n_seeds=('ece', 'count'),
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        rel_ece_mean=('relative_ece_change_computed', safe_mean),
        rel_ece_std=('relative_ece_change_computed', safe_std),
        frac_within_5pct=('relative_ece_change_computed', lambda x: fraction_within_threshold(x, 5.0)),
        frac_within_10pct=('relative_ece_change_computed', lambda x: fraction_within_threshold(x, 10.0)),
        compression_ratio_mean=('compression_ratio_effective', safe_mean),
        compression_ratio_actual_mean=('compression_ratio_actual', safe_mean),
        compression_ratio_requested_mean=('compression_ratio_requested', safe_mean),
        feature_time_mean=('feature_extract_time_s', 'mean'),
        fit_time_mean=('fit_time_s', 'mean'),
        calibrate_time_mean=('calibrate_time_s', 'mean'),
        time_per_image_mean=('calibrate_time_per_image_s', 'mean')
    ).reset_index()
    
    aggregated = aggregated.merge(
        layer_type_summary,
        on=['dataset', 'model', 'training_loss', 'layer_name', 'compression_ratio_key', 'target_dim', 'original_dim',
            'compression_method', 'pyramid_levels_str', 'mid_channels'],
        how='left'
    )
    
    aggregated['compression_ratio'] = aggregated['compression_ratio_mean'].fillna(
        aggregated['compression_ratio_requested_mean']
    ).fillna(
        aggregated['compression_ratio_actual_mean']
    ).fillna(
        aggregated['compression_ratio_key']
    )
    
    aggregated.rename(columns={'compression_ratio_key': 'compression_ratio_rounded'}, inplace=True)
    aggregated['compression_ratio_requested_mean'] = aggregated['compression_ratio_requested_mean'].fillna(aggregated['compression_ratio'])
    aggregated['compression_ratio_actual_mean'] = aggregated['compression_ratio_actual_mean'].fillna(aggregated['compression_ratio'])
    
    aggregated['compression_ratio_value'] = pd.to_numeric(aggregated['compression_ratio'], errors='coerce')
    aggregated['compression_ratio_label'] = aggregated['compression_ratio_value'].apply(format_compression_ratio)
    
    aggregated = aggregated[
        [
            'dataset', 'model', 'training_loss', 'layer_name', 'layer_type',
            'compression_ratio', 'compression_ratio_value', 'compression_ratio_label',
            'compression_ratio_rounded', 'compression_ratio_actual_mean', 'compression_ratio_requested_mean',
            'target_dim', 'original_dim',
            'compression_method', 'pyramid_levels_str', 'mid_channels',
            'ece_mean', 'ece_std', 'n_seeds',
            'accuracy_mean', 'accuracy_std',
            'rel_ece_mean', 'rel_ece_std', 'frac_within_5pct', 'frac_within_10pct',
            'feature_time_mean', 'fit_time_mean',
            'calibrate_time_mean', 'time_per_image_mean'
        ]
    ]
    
    return aggregated


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


COMPRESSION_RATIO_BINS = [
    ("1×", 1.0, 0.05),
    ("2×", 2.0, 0.3),
    ("4×", 4.0, 0.6),
    ("8×", 8.0, 1.2),
    ("16×", 16.0, 2.5),
    ("32×", 32.0, 5.0),
]

RATIO_LABEL_TO_CENTER = {label: center for label, center, _ in COMPRESSION_RATIO_BINS}


def assign_compression_ratio_bin(factor: float) -> str:
    """Map a compression factor to a named bin."""
    if factor is None or pd.isna(factor):
        return None
    try:
        value = float(factor)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    for label, center, tolerance in COMPRESSION_RATIO_BINS:
        if abs(value - center) <= tolerance:
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


def compute_compression_factor(df: pd.DataFrame) -> pd.Series:
    """Compute compression factor (original_dim / target_dim)."""
    if 'compression_ratio' in df.columns:
        return pd.to_numeric(df['compression_ratio'], errors='coerce')
    if 'compression_ratio_effective' in df.columns:
        return pd.to_numeric(df['compression_ratio_effective'], errors='coerce')
    if 'original_dim' not in df.columns or 'target_dim' not in df.columns:
        return pd.Series(np.nan, index=df.index)
    original = pd.to_numeric(df['original_dim'], errors='coerce')
    target = pd.to_numeric(df['target_dim'], errors='coerce')
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.where((target > 0) & (original > 0), original / target, np.nan)
    return pd.Series(factor, index=df.index)


def aggregate_by_compression_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate calibration outcomes by compression ratio bins.
    
    Groups by (layer_type, compression_ratio_bin) where compression ratio is
    defined as original_dim / target_dim. Bins are centered on powers of two
    with specified tolerances to account for discrete dimensionalities.
    """
    ratio_df = df.copy()
    ratio_df['compression_factor'] = compute_compression_factor(ratio_df)
    ratio_df['compression_ratio_bin'] = ratio_df['compression_factor'].apply(assign_compression_ratio_bin)
    ratio_df = ratio_df.dropna(subset=['compression_ratio_bin', 'rel_ece_mean'])
    
    if ratio_df.empty:
        print("No data available after assigning compression ratio bins.")
        return pd.DataFrame(columns=[
            'layer_type', 'compression_ratio_bin', 'compression_ratio_center',
            'rel_ece_mean', 'rel_ece_std', 'frac_within_5pct', 'n_instances'
        ])
    
    ratio_df['compression_ratio_center'] = ratio_df['compression_ratio_bin'].map(RATIO_LABEL_TO_CENTER)
    
    def frac_within_threshold(series: pd.Series, threshold: float = 5.0) -> float:
        valid = series.dropna()
        if len(valid) == 0:
            return np.nan
        return (valid <= threshold).mean()
    
    aggregated = ratio_df.groupby(['layer_type', 'compression_ratio_bin']).agg({
        'compression_ratio_center': 'mean',
        'rel_ece_mean': ['mean', 'std', frac_within_threshold],
        'layer_name': 'count'
    }).reset_index()
    
    aggregated.columns = [
        'layer_type',
        'compression_ratio_bin',
        'compression_ratio_center',
        'rel_ece_mean',
        'rel_ece_std',
        'frac_within_5pct',
        'n_instances'
    ]
    
    aggregated['rel_ece_std'] = aggregated['rel_ece_std'].fillna(0.0)
    aggregated['frac_within_5pct'] = aggregated['frac_within_5pct'].clip(0, 1)
    
    aggregated = aggregated.sort_values(['layer_type', 'compression_ratio_center']).reset_index(drop=True)
    
    baseline_entries = ratio_df[ratio_df['compression_ratio_bin'] == '1×']
    if baseline_entries.empty:
        print("ERROR: No entries found for 1× compression ratio; baselines may be missing.")
    else:
        baseline_deviation = baseline_entries[np.abs(baseline_entries['rel_ece_mean']) > 0.5]
        if not baseline_deviation.empty:
            print("ERROR: Baseline (1×) relative ECE deviates from 0% for the following layers:")
            for _, row in baseline_deviation.iterrows():
                print(
                    f"  {row['dataset']} / {row['model']} / Layer {row['layer_name']} "
                    f"(type={row['layer_type']}): ΔECE={row['rel_ece_mean']:+.2f}%"
                )
    
    baseline_mask = aggregated['compression_ratio_bin'] == '1×'
    if baseline_mask.any():
        aggregated.loc[baseline_mask, 'rel_ece_mean'] = 0.0
        aggregated.loc[baseline_mask, 'rel_ece_std'] = 0.0
        aggregated.loc[baseline_mask, 'frac_within_5pct'] = 1.0
    
    return aggregated


def make_paper_table(df: pd.DataFrame, output_path: Path, 
                     layer_types: List[str] = ['best_layer', 'data_layer'],
                     dimensions: List[int] = None) -> None:
    """
    Create compact LaTeX table for paper.
    
    Table shows: Layer | Compression Ratio | ΔECE (%) | Good≤5% | N
    """
    ratio_summary = aggregate_by_compression_ratio(df)
    if ratio_summary.empty:
        output_path.write_text("No data available for paper table.\n")
        print("Skipping paper table generation due to empty ratio summary.")
        return
    
    filtered = ratio_summary[ratio_summary['layer_type'].isin(layer_types)].copy()
    
    if dimensions:
        ratio_labels = [f"{int(d)}×" for d in dimensions]
        filtered = filtered[filtered['compression_ratio_bin'].isin(ratio_labels)]
    
    filtered = filtered.sort_values(['layer_type', 'compression_ratio_center'])
    
    layer_labels = {
        'best_layer': 'Best',
        'data_layer': 'Data'
    }
    
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Impact of compression ratio on calibration quality. "
        r"Values show mean $\pm$ std of relative ECE change (\%) versus uncompressed baseline, "
        r"aggregated across datasets, models, layers within each role, and seeds. "
        r"$\mathrm{Good}_{\le 5\%}$ indicates fraction of runs with $\Delta\mathrm{ECE} \le 5\%$ (improvement or small drift).}",
        r"\label{tab:compression-impact}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Layer & Compression & $\Delta$ECE (\%) & $\mathrm{Good}_{\le 5\%}$ & $N$ \\",
        r"\midrule"
    ]
    
    for _, row in filtered.iterrows():
        layer = layer_labels.get(row['layer_type'], row['layer_type'])
        ratio_label = row['compression_ratio_bin']
        mean_rel = row['rel_ece_mean']
        std_rel = row['rel_ece_std']
        frac5 = row['frac_within_5pct']
        n = int(row['n_instances'])
        
        lines.append(
            f"{layer} & {ratio_label} & "
            f"{mean_rel:+.1f} $\\pm$ {std_rel:.1f} & "
            f"{frac5:.2f} & {n} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    output_path.write_text("\n".join(lines))
    print(f"Saved paper table: {output_path}")


def make_stratified_table(df: pd.DataFrame, output_path: Path,
                         layer_type: str = 'best_layer') -> None:
    """
    Create table stratified by dataset/model showing compression robustness.
    Useful for understanding which configurations benefit most.
    """
    filtered = df[df['layer_type'] == layer_type].copy()
    if filtered.empty:
        output_path.write_text("No data available for stratified table.\n")
        print(f"No data for layer_type={layer_type}, skipping stratified table.")
        return
    
    filtered['compression_ratio_value'] = pd.to_numeric(filtered['compression_ratio'], errors='coerce')
    filtered = filtered.dropna(subset=['compression_ratio_value'])
    if filtered.empty:
        output_path.write_text("No valid compression ratios for stratified table.\n")
        print(f"No valid compression ratios for layer_type={layer_type}, skipping stratified table.")
        return
    
    filtered['compression_ratio_display'] = filtered['compression_ratio_value'].apply(format_compression_ratio)
    
    pivot = filtered.pivot_table(
        values='rel_ece_mean',
        index=['dataset', 'model'],
        columns='compression_ratio_display',
        aggfunc='mean'
    )
    
    ratio_order = (
        filtered[['compression_ratio_display', 'compression_ratio_value']]
        .drop_duplicates()
        .sort_values('compression_ratio_value')
    )
    pivot = pivot.reindex(columns=ratio_order['compression_ratio_display'])
    
    with open(output_path, 'w') as f:
        f.write("% Table S1: Detailed Per-Configuration Results\n")
        f.write(r"\begin{table}[h]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(
            r"\caption{Table S1. Detailed per-configuration relative ECE change (\%) for "
            + layer_type.replace('_', ' ')
            + r" across compression ratios.}"
            + "\n"
        )
        
        cols = list(ratio_order['compression_ratio_display'])
        header = "Dataset & Model & " + " & ".join(cols) + r" \\" + "\n"
        
        f.write(r"\begin{tabular}{ll" + "r" * len(cols) + "}\n")
        f.write(r"\toprule" + "\n")
        f.write(header)
        f.write(r"\midrule" + "\n")
        
        for (dataset, model), row in pivot.iterrows():
            values = [
                f"{row[c]:+.1f}" if c in row.index and pd.notna(row[c]) else "-"
                for c in cols
            ]
            f.write(f"{dataset} & {model} & " + " & ".join(values) + r" \\" + "\n")
        
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\label{tab:table-s1-compression}" + "\n")
        f.write(r"\end{table}" + "\n")
    
    print(f"Saved stratified table: {output_path}")


def plot_main_figure(df: pd.DataFrame, output_path: Path,
                    layer_types: List[str] = ['best_layer', 'data_layer']) -> None:
    """
    Main figure: Relative ECE change vs target dimension.
    Clean, publication-ready plot.
    """
    filtered = df[df['layer_type'].isin(layer_types)].copy()
    filtered['compression_ratio_value'] = pd.to_numeric(filtered['compression_ratio'], errors='coerce')
    filtered = filtered.dropna(subset=['compression_ratio_value'])
    if filtered.empty:
        print("No valid compression ratios available for main figure; skipping plot.")
        return
    
    grouped = filtered.groupby(['layer_type', 'compression_ratio_value']).agg({
        'rel_ece_mean': 'mean',
        'rel_ece_std': 'mean'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = {'best_layer': '#2ecc71', 'data_layer': '#3498db'}
    labels = {'best_layer': 'Best Layer', 'data_layer': 'Data Layer'}
    markers = {'best_layer': 'o', 'data_layer': 's'}
    
    for layer_type in layer_types:
        layer_df = grouped[grouped['layer_type'] == layer_type].sort_values('compression_ratio_value')
        if layer_df.empty:
            continue
        
        x = layer_df['compression_ratio_value'].values
        y = layer_df['rel_ece_mean'].values
        y_err = layer_df['rel_ece_std'].values
        
        ax.plot(x, y, 
                marker=markers[layer_type], 
                color=colors[layer_type],
                label=labels[layer_type],
                linewidth=2.5, 
                markersize=8,
                alpha=0.9)
        
        ax.fill_between(x, y - y_err, y + y_err, 
                        color=colors[layer_type], 
                        alpha=0.15)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.4)
    
    ratio_min = filtered['compression_ratio_value'].min()
    ratio_max = filtered['compression_ratio_value'].max()
    ax.fill_between([ratio_min, ratio_max],
                    -5, 5, color='lightgreen', alpha=0.1, 
                    label='±5% tolerance')
    
    ax.set_xlabel('Compression Ratio (×)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Relative ECE Change (%)', fontsize=13, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_title('Calibration Quality Under Feature Compression', 
                fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved main figure: {output_path}")


def plot_compression_optimization(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create the main optimization figure showing compression ratio vs ECE change.
    This is the key figure the advisor requested.
    """
    working = df.copy()
    working['compression_ratio_value'] = pd.to_numeric(working['compression_ratio'], errors='coerce')
    working = working.dropna(subset=['compression_ratio_value', 'rel_ece_mean'])
    if working.empty:
        print("No valid data available for compression optimization plot; skipping.")
        return
    
    aggregated = working.groupby(['layer_type', 'compression_ratio_value']).agg({
        'rel_ece_mean': 'mean',
        'rel_ece_std': 'mean',
        'layer_name': 'count'
    }).reset_index()
    
    aggregated.rename(columns={
        'compression_ratio_value': 'compression_ratio',
        'rel_ece_std': 'rel_ece_std',
        'rel_ece_mean': 'rel_ece_mean',
        'layer_name': 'n_instances'
    }, inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layer_styles = {
        'best_layer': {'color': '#2E86AB', 'marker': 'o', 'label': 'Best Layer (Semantic)'},
        'data_layer': {'color': '#A23B72', 'marker': 's', 'label': 'Data Layer (Pixel Space)'}
    }
    
    for layer_type, style in layer_styles.items():
        layer_data = aggregated[aggregated['layer_type'] == layer_type].sort_values('compression_ratio')
        if layer_data.empty:
            continue
        
        x = layer_data['compression_ratio'].values
        y = layer_data['rel_ece_mean'].values
        yerr = layer_data['rel_ece_std'].values
        
        ax.errorbar(
            x, y, yerr=yerr,
            marker=style['marker'],
            color=style['color'],
            label=style['label'],
            linewidth=2.5,
            markersize=8,
            capsize=4,
            capthick=1.5,
            alpha=0.9
        )
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Baseline (no change)')
    ax.axhspan(-5, 5, color='lightgreen', alpha=0.2, label='Acceptable region (±5%)')
    
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Compression Ratio (×)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative ECE Change (%)', fontsize=14, fontweight='bold')
    ax.set_title('Optimization: Dimension Reduction vs Calibration Quality',
                 fontsize=16, fontweight='bold', pad=20)
    
    xticks = [1, 2, 4, 8, 16, 32]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t}×" for t in xticks])
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved optimization figure: {output_path}")


def plot_compression_optimization_stratified(df: pd.DataFrame, output_path: Path) -> None:
    """
    Stratified optimization plot showing compression behavior by original dimension range.
    Focuses on best layers to highlight representation depth effects.
    """
    working = df.copy()
    working['compression_ratio_value'] = pd.to_numeric(working['compression_ratio'], errors='coerce')
    working['original_dim_bin'] = working['original_dim'].apply(assign_original_dim_bin)
    working = working.dropna(subset=['compression_ratio_value', 'rel_ece_mean', 'original_dim_bin'])
    
    semantic_df = working[working['layer_type'] == 'best_layer']
    if semantic_df.empty:
        print("No semantic layer data available for stratified optimization plot; skipping.")
        return
    
    dim_bins = sorted(semantic_df['original_dim_bin'].dropna().unique())
    if not dim_bins:
        print("No dimension bins available for stratified optimization plot; skipping.")
        return
    
    fig, axes = plt.subplots(1, len(dim_bins), figsize=(5 * len(dim_bins), 5), sharey=True)
    if len(dim_bins) == 1:
        axes = [axes]
    
    for ax, dim_bin in zip(axes, dim_bins):
        bin_df = semantic_df[semantic_df['original_dim_bin'] == dim_bin]
        if bin_df.empty:
            ax.set_visible(False)
            continue
        
        aggregated = bin_df.groupby('compression_ratio_value').agg({
            'rel_ece_mean': ['mean', 'std', 'count']
        }).reset_index()
        aggregated.columns = ['compression_ratio', 'rel_ece_mean', 'rel_ece_std', 'n_instances']
        aggregated = aggregated.sort_values('compression_ratio')
        
        x = aggregated['compression_ratio'].values
        y = aggregated['rel_ece_mean'].values
        yerr = aggregated['rel_ece_std'].values
        
        ax.errorbar(
            x, y, yerr=yerr,
            marker='o',
            linewidth=2.5,
            markersize=8,
            capsize=4,
            color='#2E86AB',
            alpha=0.9
        )
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
        ax.axhspan(-5, 5, color='lightgreen', alpha=0.2)
        
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Compression Ratio (×)', fontsize=12, fontweight='bold')
        ax.set_title(f'{dim_bin}\n(n={len(bin_df)} runs)', fontsize=11, fontweight='bold')
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(['1×', '2×', '4×', '8×', '16×', '32×'])
        ax.grid(True, alpha=0.3)
        
        for xi, yi, ni in zip(x, y, aggregated['n_instances']):
            ax.text(xi, yi - 8, f'n={int(ni)}', ha='center', fontsize=8, color='gray')
    
    axes[0].set_ylabel('Relative ECE Change (%)', fontsize=12, fontweight='bold')
    plt.suptitle('Compression Robustness by Layer Dimensionality (Best Layers)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved stratified optimization figure: {output_path}")


def plot_compression_optimization_robust(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create an optimization figure using median & IQR statistics for robustness.
    """
    working = df.copy()
    working['compression_ratio_value'] = pd.to_numeric(working['compression_ratio'], errors='coerce')
    working = working.dropna(subset=['compression_ratio_value', 'rel_ece_mean'])
    if working.empty:
        print("No valid data available for robust optimization plot; skipping.")
        return
    
    aggregated = working.groupby(['layer_type', 'compression_ratio_value']).agg({
        'rel_ece_mean': ['median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)],
        'layer_name': 'count'
    }).reset_index()
    
    aggregated.columns = ['layer_type', 'compression_ratio', 'rel_ece_median', 'q25', 'q75', 'n_instances']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layer_styles = {
        'best_layer': {'color': '#2E86AB', 'marker': 'o', 'label': 'Best Layer (Semantic)'},
        'data_layer': {'color': '#A23B72', 'marker': 's', 'label': 'Data Layer (Pixel Space)'}
    }
    
    for layer_type, style in layer_styles.items():
        layer_data = aggregated[aggregated['layer_type'] == layer_type].sort_values('compression_ratio')
        if layer_data.empty:
            continue
        
        x = layer_data['compression_ratio'].values
        y = layer_data['rel_ece_median'].values
        yerr_lower = y - layer_data['q25'].values
        yerr_upper = layer_data['q75'].values - y
        
        ax.errorbar(
            x, y, yerr=[yerr_lower, yerr_upper],
            marker=style['marker'],
            color=style['color'],
            label=style['label'],
            linewidth=2.5,
            markersize=8,
            capsize=4,
            alpha=0.9
        )
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axhspan(-5, 5, color='lightgreen', alpha=0.2)
    
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Compression Ratio (×)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative ECE Change (%, median)', fontsize=14, fontweight='bold')
    ax.set_title('Compression Optimization (Robust Aggregation)', fontsize=16, fontweight='bold')
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(['1×', '2×', '4×', '8×', '16×', '32×'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved robust optimization figure: {output_path}")


def plot_compression_best_case(df: pd.DataFrame, output_path: Path) -> None:
    """
    Show success rates and best-case performance for semantic layers.
    """
    working = df.copy()
    working['compression_ratio_value'] = pd.to_numeric(working['compression_ratio'], errors='coerce')
    working = working.dropna(subset=['compression_ratio_value', 'rel_ece_mean'])
    
    semantic_df = working[working['layer_type'] == 'best_layer']
    if semantic_df.empty:
        print("No semantic layer data available for best-case plot; skipping.")
        return
    
    success_rate = semantic_df.groupby('compression_ratio_value').apply(
        lambda g: (g['rel_ece_mean'] <= 5).sum() / len(g) if len(g) > 0 else np.nan
    ).reset_index(name='success_rate')
    
    best_case = semantic_df.groupby('compression_ratio_value')['rel_ece_mean'].quantile(0.25).reset_index()
    best_case.columns = ['compression_ratio', 'rel_ece_25th_percentile']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    
    ax1.plot(
        success_rate['compression_ratio_value'],
        success_rate['success_rate'] * 100,
        marker='o',
        linewidth=2.5,
        markersize=8,
        color='#2E86AB'
    )
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Compression Ratio (×)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)\n(ΔECE ≤ 5%)', fontsize=12, fontweight='bold')
    ax1.set_title('Compression Success Rate', fontsize=13, fontweight='bold')
    ax1.set_xticks([1, 2, 4, 8, 16, 32])
    ax1.set_xticklabels(['1×', '2×', '4×', '8×', '16×', '32×'])
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    ax2.plot(
        best_case['compression_ratio'],
        best_case['rel_ece_25th_percentile'],
        marker='o',
        linewidth=2.5,
        markersize=8,
        color='#2E86AB',
        label='25th percentile (best quartile)'
    )
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.axhspan(-5, 5, color='lightgreen', alpha=0.2, label='Target zone')
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Compression Ratio (×)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Relative ECE Change (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Best-Case Performance', fontsize=13, fontweight='bold')
    ax2.set_xticks([1, 2, 4, 8, 16, 32])
    ax2.set_xticklabels(['1×', '2×', '4×', '8×', '16×', '32×'])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle('Compression Optimization: What’s Achievable (Semantic Layers)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved best-case optimization figure: {output_path}")


def plot_compression_heatmap_per_config(df: pd.DataFrame, output_path: Path,
                                        layer_type: str = 'best_layer') -> None:
    """
    Create a heatmap showing compression behavior across configurations.
    Visual alternative to the per-configuration table.
    """
    semantic_df = df[df['layer_type'] == layer_type].copy()
    if semantic_df.empty:
        print(f"No data available for layer_type={layer_type}, skipping configuration heatmap.")
        return

    semantic_df['compression_ratio_value'] = pd.to_numeric(
        semantic_df['compression_ratio'], errors='coerce'
    )
    semantic_df = semantic_df.dropna(
        subset=['compression_ratio_value', 'rel_ece_mean', 'dataset', 'model']
    )
    if semantic_df.empty:
        print("No valid entries for configuration heatmap after filtering.")
        return

    semantic_df['config'] = semantic_df['dataset'].astype(str) + "\n" + semantic_df['model'].astype(str)
    semantic_df['compression_ratio_round'] = semantic_df['compression_ratio_value'].apply(lambda x: round(x, 3))

    pivot = semantic_df.pivot_table(
        values='rel_ece_mean',
        index='config',
        columns='compression_ratio_round',
        aggfunc='mean'
    )
    if pivot.empty:
        print("Configuration heatmap pivot is empty; skipping plot.")
        return

    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    column_labels = [format_compression_ratio(col) for col in pivot.columns]

    fig_width = max(8.0, 1.2 * len(column_labels))
    fig_height = max(6.0, 0.5 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        center=0,
        vmin=-30,
        vmax=30,
        cbar_kws={'label': 'Relative ECE Change (%)'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    ax.set_xlabel('Compression Ratio (×)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_title(
        'Compression Impact by Configuration\n(Negative = Improvement, Positive = Degradation)',
        fontsize=13, fontweight='bold', pad=15
    )
    ax.set_xticks(np.arange(len(column_labels)) + 0.5)
    ax.set_xticklabels(column_labels, rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved configuration heatmap: {output_path}")


def plot_timing_analysis(df: pd.DataFrame, output_path: Path,
                        layer_type: str = 'best_layer') -> None:
    """
    Timing analysis using FPS for practical intuition.
    Three panels: (1) Absolute FPS throughput, (2) Throughput gain, (3) Quality-throughput tradeoff.
    """
    filtered = df[df['layer_type'] == layer_type].copy()
    required_cols = {
        'compression_ratio', 'time_per_image_mean', 'rel_ece_mean'
    }
    missing_cols = required_cols - set(filtered.columns)
    if filtered.empty or missing_cols:
        print(f"Timing data unavailable for layer_type={layer_type}: missing {sorted(missing_cols)}")
        return
    
    filtered['compression_ratio_value'] = pd.to_numeric(filtered['compression_ratio'], errors='coerce')
    filtered['time_per_image_mean'] = pd.to_numeric(filtered['time_per_image_mean'], errors='coerce')
    filtered['rel_ece_mean'] = pd.to_numeric(filtered['rel_ece_mean'], errors='coerce')
    filtered = filtered.dropna(subset=['compression_ratio_value', 'time_per_image_mean', 'rel_ece_mean'])
    if filtered.empty:
        print("No valid timing entries after filtering; skipping timing analysis plot.")
        return
    
    filtered = filtered[filtered['time_per_image_mean'] > 0]
    if filtered.empty:
        print("All timing entries have non-positive time_per_image_mean; skipping timing analysis plot.")
        return
    
    filtered['calibrate_fps'] = 1.0 / filtered['time_per_image_mean']
    
    grouped = filtered.groupby('compression_ratio_value').agg({
        'calibrate_fps': ['mean', 'std'],
        'rel_ece_mean': 'mean',
        'layer_name': 'count'
    }).reset_index()
    
    grouped.columns = [
        'compression_ratio',
        'fps_mean', 'fps_std',
        'rel_ece_mean', 'n'
    ]
    grouped = grouped.sort_values('compression_ratio')
    grouped = grouped[grouped['compression_ratio'] > 0]
    if grouped.empty:
        print("Timing aggregation produced no valid rows; skipping plot.")
        return
    
    baseline_row = grouped[np.isclose(grouped['compression_ratio'], 1.0, atol=1e-6)]
    if baseline_row.empty:
        print("Baseline FPS (1× compression) missing; cannot compute throughput gain.")
        return
    baseline_fps = baseline_row['fps_mean'].iloc[0]
    if baseline_fps <= 0:
        print("Invalid baseline FPS encountered; skipping timing plot.")
        return
    
    grouped['throughput_gain'] = grouped['fps_mean'] / baseline_fps
    grouped['compression_ratio_label'] = grouped['compression_ratio'].apply(format_compression_ratio)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Panel 1: Absolute FPS Throughput
    ax1 = axes[0]
    x = grouped['compression_ratio'].values
    y = grouped['fps_mean'].values
    yerr = grouped['fps_std'].fillna(0).values
    
    ax1.errorbar(
        x, y, yerr=yerr,
        marker='o',
        linewidth=2.5,
        markersize=8,
        capsize=4,
        color='#2E86AB',
        alpha=0.9
    )
    
    for xi, yi in zip(x, y):
        ax1.text(
            xi,
            yi + max(y) * 0.03,
            f'{yi:.0f}',
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='#2E86AB'
        )
    
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Compression Ratio (×)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Calibration Throughput\n(images / second)', fontsize=12, fontweight='bold')
    ax1.set_title('Absolute Throughput (FPS)', fontsize=13, fontweight='bold')
    ax1.set_xticks([1, 2, 4, 8, 16, 32])
    ax1.set_xticklabels(['1×', '2×', '4×', '8×', '16×', '32×'])
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Panel 2: Throughput Gain
    ax2 = axes[1]
    gain = grouped['throughput_gain'].values
    
    ax2.plot(
        x, gain,
        marker='o',
        linewidth=2.5,
        markersize=8,
        color='#A23B72',
        alpha=0.9
    )
    ax2.axhline(
        y=1.0,
        color='black',
        linestyle='--',
        linewidth=1.5,
        alpha=0.6,
        label='Baseline (no speedup)'
    )
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Compression Ratio (×)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax2.set_title('Computational Speedup', fontsize=13, fontweight='bold')
    ax2.set_xticks([1, 2, 4, 8, 16, 32])
    ax2.set_xticklabels(['1×', '2×', '4×', '8×', '16×', '32×'])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    for xi, yi in zip(x, gain):
        if xi > 1:
            ax2.text(
                xi,
                yi + max(gain) * 0.04,
                f'{yi:.1f}×',
                ha='center',
                fontsize=9,
                fontweight='bold',
                color='#A23B72'
            )
    ax2.set_ylim(bottom=0)
    
    # Panel 3: Quality-Throughput Tradeoff
    ax3 = axes[2]
    color_values = np.where(
        grouped['compression_ratio'] > 0,
        np.log2(grouped['compression_ratio']),
        np.nan
    )
    
    scatter = ax3.scatter(
        grouped['fps_mean'],
        grouped['rel_ece_mean'],
        c=color_values,
        s=200,
        cmap='viridis',
        alpha=0.8,
        edgecolors='black',
        linewidth=1.5
    )
    
    for _, row in grouped.iterrows():
        ratio_label = format_compression_ratio(row['compression_ratio'])
        ax3.text(
            row['fps_mean'],
            row['rel_ece_mean'] - 1.2,
            ratio_label,
            ha='center',
            fontsize=9,
            fontweight='bold'
        )
    
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
    ax3.axhspan(-5, 5, color='lightgreen', alpha=0.15)
    
    sweet_spot_ratios = [4, 8]
    highlight_present = False
    for ratio in sweet_spot_ratios:
        ratio_row = grouped[np.isclose(grouped['compression_ratio'], ratio, atol=1e-6)]
        if ratio_row.empty:
            continue
        highlight_present = True
        row = ratio_row.iloc[0]
        ax3.scatter(
            row['fps_mean'],
            row['rel_ece_mean'],
            s=250,
            facecolors='none',
            edgecolors='red',
            linewidth=2.5,
            alpha=0.7
        )
    
    if highlight_present:
        ax3.text(
            0.95,
            0.05,
            'Red circles:\nSweet spot\n(4–8×)',
            transform=ax3.transAxes,
            ha='right',
            va='bottom',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    ax3.set_xlabel('Calibration Throughput (images / second)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Relative ECE Change (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Quality-Throughput Tradeoff', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Compression (log₂×)', fontsize=10)
    
    plt.suptitle(
        f'Compression Efficiency: Throughput Analysis ({layer_type.replace("_", " ").title()})',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved timing analysis: {output_path}")
    print("\n" + "=" * 60)
    print("FPS THROUGHPUT SUMMARY")
    print("=" * 60)
    for _, row in grouped.iterrows():
        ratio_label = format_compression_ratio(row['compression_ratio'])
        fps_val = row['fps_mean']
        gain_val = row['throughput_gain']
        ece_val = row['rel_ece_mean']
        print(f"{ratio_label:>6}: {fps_val:6.1f} FPS ({gain_val:.1f}× gain) | ΔECE = {ece_val:+6.1f}%")
    print("=" * 60)


def plot_per_layer_curves(df: pd.DataFrame, output_dir: Path,
                          layer_type_filter: str = 'best_layer') -> None:
    """
    Plot relative ECE change curves for individual layers to diagnose
    compression robustness per layer.
    """
    filtered = df[df['layer_type'] == layer_type_filter].copy()
    filtered['compression_ratio_value'] = pd.to_numeric(filtered['compression_ratio'], errors='coerce')
    filtered = filtered.dropna(subset=['compression_ratio_value'])
    
    if filtered.empty:
        print(f"No data found for layer_type={layer_type_filter}, skipping per-layer plots.")
        return
    
    combos = filtered[['dataset', 'model']].drop_duplicates().sort_values(['dataset', 'model'])
    color_palette = sns.color_palette("viridis", n_colors=max(3, filtered['layer_name'].nunique()))
    
    for _, row in combos.iterrows():
        dataset = row['dataset']
        model = row['model']
        
        combo_df = filtered[(filtered['dataset'] == dataset) & (filtered['model'] == model)]
        layer_groups = combo_df.groupby('layer_name')
        
        valid_layers = []
        for layer_name, layer_df in layer_groups:
            if layer_df['compression_ratio_value'].nunique() >= 3:
                valid_layers.append((layer_name, layer_df.sort_values('compression_ratio_value')))
        
        if not valid_layers:
            print(f"Skipping {dataset} / {model}: fewer than 3 compression ratios per layer.")
            continue
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for idx, (layer_name, layer_df) in enumerate(valid_layers):
            color = color_palette[idx % len(color_palette)]
            x = layer_df['compression_ratio_value'].values
            y = layer_df['rel_ece_mean'].values
            
            ax.plot(
                x, y,
                marker='o',
                linewidth=2,
                markersize=6,
                label=f"Layer {layer_name}",
                color=color,
                alpha=0.9
            )
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Compression Ratio (×)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Relative ECE Change (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Per-Layer Compression Curves\n{dataset} / {model} ({layer_type_filter})',
                     fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle='--')
        
        plt.tight_layout()
        
        dataset_slug = str(dataset).replace(" ", "_").replace("/", "-")
        model_slug = str(model).replace(" ", "_").replace("/", "-")
        filename = f"per_layer_curves_{dataset_slug}_{model_slug}.png"
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved per-layer curves: {output_path}")


def plot_compression_by_original_dim(df: pd.DataFrame, output_path: Path,
                                     layer_type_filter: str = 'best_layer') -> None:
    """
    Visualize how compression robustness varies with original layer dimensionality.
    Produces a two-panel figure highlighting aggregate trends and per-layer safe factors.
    """
    filtered = df[df['layer_type'] == layer_type_filter].copy()
    if filtered.empty:
        print(f"No data found for layer_type={layer_type_filter}, skipping compression-by-depth plot.")
        return
    
    filtered['original_dim_bin'] = filtered['original_dim'].apply(assign_original_dim_bin)
    filtered = filtered.dropna(subset=['original_dim_bin', 'target_dim', 'rel_ece_mean'])
    
    if filtered.empty:
        print(f"No valid entries after binning original dimensions for {layer_type_filter}, skipping plot.")
        return
    
    filtered['compression_factor'] = compute_compression_factor(filtered)
    filtered = filtered.dropna(subset=['compression_factor'])
    filtered = filtered[filtered['compression_factor'] >= 1.0]
    filtered = filtered[filtered['compression_factor'] <= 64.0]
    if filtered.empty:
        print("No compression factors within [1×, 64×] for compression-by-depth plot.")
        return
    
    filtered['compression_factor_round'] = filtered['compression_factor'].apply(lambda x: round(x, 3))
    
    curve_stats = filtered.groupby(['original_dim_bin', 'compression_factor_round']).agg(
        rel_mean=('rel_ece_mean', 'mean'),
        rel_std=('rel_ece_mean', 'std'),
        count=('rel_ece_mean', 'count')
    ).reset_index()
    
    layer_counts = filtered.groupby('original_dim_bin')['layer_name'].nunique().to_dict()
    
    layer_info_records = []
    for (dataset, model, layer_name, original_dim, bin_label), group in filtered.groupby(
        ['dataset', 'model', 'layer_name', 'original_dim', 'original_dim_bin']
    ):
        safe_rows = group[group['rel_ece_mean'] <= 5].sort_values('target_dim')
        has_safe = not safe_rows.empty
        if has_safe:
            min_target = safe_rows['target_dim'].min()
            if min_target and original_dim:
                max_safe_factor = float(original_dim) / float(min_target)
            else:
                max_safe_factor = np.nan
        else:
            max_safe_factor = 1.0
        layer_info_records.append({
            'dataset': dataset,
            'model': model,
            'layer_name': layer_name,
            'original_dim': original_dim,
            'original_dim_bin': bin_label,
            'max_safe_factor': max_safe_factor,
            'has_safe': has_safe
        })
    
    layer_info = pd.DataFrame(layer_info_records)
    if layer_info.empty:
        print("No layer-level information available for compression-by-depth plot.")
        return
    
    palette = sns.color_palette("colorblind", n_colors=max(4, len(ORIGINAL_DIM_BINS)))
    safe_palette = sns.color_palette("colorblind", n_colors=2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_left, ax_right = axes
    
    # Left panel: compression curves by depth bins
    for idx, (label, _, _) in enumerate(ORIGINAL_DIM_BINS):
        bin_df = curve_stats[curve_stats['original_dim_bin'] == label].sort_values('compression_factor_round')
        if bin_df.empty:
            continue
        mean_vals = bin_df['rel_mean'].values
        std_vals = bin_df['rel_std'].fillna(0).values
        comp_factors = bin_df['compression_factor_round'].values
        count = layer_counts.get(label, 0)
        
        baseline_idx = np.where(np.isclose(comp_factors, 1.0, atol=1e-3))[0]
        if baseline_idx.size > 0:
            mean_vals[baseline_idx] = 0.0
            std_vals[baseline_idx] = 0.0
        
        ax_left.plot(
            comp_factors,
            mean_vals,
            label=f"{label} (n={count})",
            color=palette[idx % len(palette)],
            linewidth=2.5,
            alpha=0.9
        )
        ax_left.fill_between(
            comp_factors,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=palette[idx % len(palette)],
            alpha=0.15
        )
    
    ax_left.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
    ax_left.fill_between(
        [filtered['compression_factor'].min(), filtered['compression_factor'].max()],
        -5, 5,
        color='lightgreen',
        alpha=0.15
    )
    ax_left.set_xscale('log', base=2)
    ax_left.set_xlim(1, 64)
    ax_left.set_xlabel('Compression Ratio (×)', fontsize=12, fontweight='bold')
    ax_left.set_ylabel('Mean Relative ECE Change (%)', fontsize=12, fontweight='bold')
    ax_left.set_title('Compression Curves by Layer Depth', fontsize=13, fontweight='bold')
    ax_left.grid(True, alpha=0.25, linestyle='--')
    ax_left.legend(fontsize=10, title='Original Dim Bin', title_fontsize=10)
    
    # Right panel: safe compression factor vs original dim
    safe_df = layer_info.dropna(subset=['original_dim', 'max_safe_factor'])
    if safe_df.empty:
        print("No safe compression data available for scatter plot; skipping right panel.")
    else:
        safe_mask = safe_df['has_safe']
        ax_right.scatter(
            safe_df.loc[safe_mask, 'original_dim'],
            safe_df.loc[safe_mask, 'max_safe_factor'],
            color=safe_palette[0],
            label='Has safe range',
            s=80,
            alpha=0.85,
            edgecolors='black',
            linewidth=0.5
        )
        if (~safe_mask).any():
            ax_right.scatter(
                safe_df.loc[~safe_mask, 'original_dim'],
                safe_df.loc[~safe_mask, 'max_safe_factor'],
                color=safe_palette[1],
                label='No safe range',
                s=80,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
        
        reg_df = safe_df[safe_mask].dropna(subset=['original_dim', 'max_safe_factor'])
        if len(reg_df) >= 2:
            x_log = np.log2(reg_df['original_dim'].astype(float))
            y = reg_df['max_safe_factor'].astype(float)
            coeffs = np.polyfit(x_log, y, 1)
            x_vals = np.linspace(x_log.min(), x_log.max(), 200)
            y_pred = coeffs[0] * x_vals + coeffs[1]
            x_vals_linear = np.power(2, x_vals)
            ax_right.plot(
                x_vals_linear,
                y_pred,
                color='gray',
                linestyle='--',
                linewidth=2,
                alpha=0.8,
                label='Linear fit'
            )
            ss_res = np.sum((y - (coeffs[0] * x_log + coeffs[1])) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            ax_right.text(
                0.05,
                0.95,
                f"$R^2 = {r_squared:.2f}$",
                transform=ax_right.transAxes,
                fontsize=11,
                fontweight='bold',
                ha='left',
                va='top'
            )
        
        ax_right.set_xscale('log', base=2)
        ax_right.set_xlabel('Original Dimensionality', fontsize=12, fontweight='bold')
        ax_right.set_ylabel('Max Safe Compression Factor', fontsize=12, fontweight='bold')
        ax_right.set_title('Safe Compression Factor vs Layer Depth', fontsize=13, fontweight='bold')
        ax_right.grid(True, alpha=0.25, linestyle='--')
        ax_right.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved compression-by-depth figure: {output_path}")


def generate_compression_guidelines_table(df: pd.DataFrame, output_path: Path,
                                          layer_type_filter: str = 'best_layer') -> None:
    """
    Create a LaTeX table summarizing recommended compression targets by original dim bin.
    """
    filtered = df[df['layer_type'] == layer_type_filter].copy()
    if filtered.empty:
        output_path.write_text("No data available to generate guidelines.\n")
        print(f"No data for layer_type={layer_type_filter}, skipping guidelines table.")
        return
    
    filtered['original_dim_bin'] = filtered['original_dim'].apply(assign_original_dim_bin)
    filtered = filtered.dropna(subset=['original_dim_bin'])
    if filtered.empty:
        output_path.write_text("No valid entries after binning original dimensions.\n")
        print("No valid entries after binning original dimensions; skipping guidelines table.")
        return
    
    layer_records = []
    for (dataset, model, layer_name, original_dim, bin_label), group in filtered.groupby(
        ['dataset', 'model', 'layer_name', 'original_dim', 'original_dim_bin']
    ):
        safe_rows = group[group['rel_ece_mean'] <= 5].sort_values('target_dim')
        if not safe_rows.empty:
            best_target = float(safe_rows['target_dim'].min())
            ratio_values = safe_rows['compression_ratio'].dropna().astype(float)
            if not ratio_values.empty:
                ratio = float(ratio_values.max())
            elif original_dim and best_target > 0:
                ratio = float(original_dim) / best_target
            else:
                ratio = np.nan
        else:
            best_target = np.nan
            ratio = np.nan
        layer_records.append({
            'bin': bin_label,
            'dataset': dataset,
            'model': model,
            'layer_name': layer_name,
            'original_dim': float(original_dim),
            'best_target': best_target,
            'compression_ratio': ratio
        })
    
    if not layer_records:
        output_path.write_text("No layer records available for guidelines.\n")
        print("No layer records were collected; skipping guidelines table.")
        return
    
    df_layers = pd.DataFrame(layer_records)
    
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Compression guidelines by layer depth. Recommended compression factors "
        r"are the median ratios achieving $<5\%$ relative ECE change.}",
        r"\label{tab:compression-guidelines}",
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"Original Dim Range & Rec. Compression & \# Layers & Example Layers \\",
        r"\midrule"
    ]
    
    for label, _, _ in ORIGINAL_DIM_BINS:
        bin_df = df_layers[df_layers['bin'] == label]
        if bin_df.empty:
            continue
        
        best_targets = bin_df['best_target'].dropna()
        ratios = bin_df['compression_ratio'].dropna()
        layer_count = bin_df['layer_name'].nunique()
        
        if not ratios.empty:
            median_ratio = float(np.median(ratios))
            ratio_str = format_compression_ratio(median_ratio).replace("×", r"$\times$")
        else:
            ratio_str = "--"
        
        examples = bin_df['layer_name'].unique()
        example_list = ", ".join(examples[:3]) if len(examples) > 0 else "-"
        
        lines.append(
            f"{label} & {ratio_str} & {layer_count} & {example_list} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    output_path.write_text("\n".join(lines))
    print(f"Saved compression guidelines table: {output_path}")


def generate_layer_compression_report(df: pd.DataFrame, output_path: Path,
                                      layer_type_filter: str = 'best_layer') -> None:
    """
    Produce a detailed text report describing compression robustness by layer,
    broader patterns, and actionable recommendations using actual layer dimensions.
    """
    filtered = df[df['layer_type'] == layer_type_filter].copy()
    
    if filtered.empty:
        message = (f"No data available for layer_type={layer_type_filter}. "
                   "Skipping layer compression report.")
        print(message)
        output_path.write_text(message + "\n")
        return
    
    lines: List[str] = []
    summary_records = []
    
    lines.append("COMPRESSION ROBUSTNESS BY LAYER")
    lines.append("=" * 60)
    lines.append("NOTE: All compression ratios computed using actual layer original_dim")
    lines.append("=" * 60)
    
    group_cols = ['dataset', 'model', 'layer_name']
    grouped = filtered.groupby(group_cols)
    
    for (dataset, model, layer_name), group in grouped:
        original_dims = group['original_dim'].dropna().unique()
        if len(original_dims) == 0:
            print(f"Warning: No original_dim found for {dataset}/{model}/Layer {layer_name}, skipping")
            continue
        
        original_dim = float(original_dims[0])
        if len(original_dims) > 1:
            print(
                f"Warning: Layer {layer_name} in {dataset}/{model} has inconsistent original_dim values: "
                f"{sorted(original_dims)}. Using first value: {original_dim}"
            )
        
        num_levels = group['target_dim'].nunique()
        mean_rel = group['rel_ece_mean'].mean()
        
        safe_rows = group[group['frac_within_5pct'] > 0.8].sort_values('target_dim')
        if not safe_rows.empty:
            safe_min_dim = float(safe_rows['target_dim'].min())
            safe_max_dim = float(safe_rows['target_dim'].max())
            safe_ratio_series = safe_rows['compression_ratio'].dropna().astype(float)
            if not safe_ratio_series.empty:
                safe_ratio_min_val = safe_ratio_series.min()
                safe_ratio_max_val = safe_ratio_series.max()
                safe_range = f"{format_compression_ratio(safe_ratio_min_val)}–{format_compression_ratio(safe_ratio_max_val)}"
            else:
                safe_ratio_min_val = safe_ratio_max_val = np.nan
                safe_range = "None"
            if (
                safe_min_dim is not None
                and not np.isnan(safe_min_dim)
                and not np.isnan(original_dim)
            ):
                max_compression_factor = original_dim / safe_min_dim
            else:
                max_compression_factor = np.nan
        else:
            safe_min_dim = safe_max_dim = None
            safe_range = "None"
            safe_ratio_min_val = safe_ratio_max_val = np.nan
            max_compression_factor = np.nan
        
        within_5pct_rows = group[group['rel_ece_mean'] <= 5].sort_values('compression_ratio')
        if not within_5pct_rows.empty:
            best_row = within_5pct_rows.iloc[0]
            best_ratio = best_row['compression_ratio']
            best_dim = best_row['target_dim']
            if pd.notna(best_ratio) and pd.notna(best_dim):
                best_ratio_str = format_compression_ratio(best_ratio)
            else:
                best_ratio_str = "Not achieved (<5% increase)"
                best_ratio = np.nan
                best_dim = np.nan
        else:
            best_ratio = np.nan
            best_dim = np.nan
            best_ratio_str = "Not achieved (<5% increase)"
        
        lines.extend([
            f"{dataset} / {model} / Layer {layer_name}",
            f"  Original dimensionality : {int(original_dim)}",
            f"  Compression levels tested : {num_levels}",
            f"  Mean ΔECE across compressions : {mean_rel:+.2f}%",
            f"  Safe compression range (Good≤5% > 0.8) : {safe_range}",
            f"  Best compression ratio with <5% increase : {best_ratio_str}",
        ])
        
        if not np.isnan(max_compression_factor):
            lines.append(
                f"  Max safe compression factor : {format_compression_ratio(max_compression_factor)}"
            )
        else:
            lines.append("  Max safe compression factor : N/A")
        lines.append("")
        
        summary_records.append({
            'dataset': dataset,
            'model': model,
            'layer_name': layer_name,
            'original_dim': original_dim,
            'n_levels': num_levels,
            'mean_rel': mean_rel,
            'safe_min_dim': safe_min_dim,
            'safe_max_dim': safe_max_dim,
            'safe_count': len(safe_rows),
            'safe_ratio_min': safe_ratio_min_val,
            'safe_ratio_max': safe_ratio_max_val,
            'best_ratio': best_ratio,
            'best_dim': best_dim,
            'max_compression_factor': max_compression_factor
        })
    
    lines.append("")
    lines.append("PATTERN ANALYSIS")
    lines.append("=" * 60)
    lines.append("NOTE: Analysis uses actual layer original_dim, not normalized values")
    lines.append("")
    
    summary_df = pd.DataFrame(summary_records)
    
    if summary_df.empty:
        lines.append("Insufficient data to compute pattern analysis.")
    else:
        summary_df['original_dim_bin'] = summary_df['original_dim'].apply(assign_original_dim_bin)
        
        valid_factor = summary_df.dropna(subset=['original_dim', 'max_compression_factor'])
        if len(valid_factor) > 1:
            corr = np.corrcoef(valid_factor['original_dim'], valid_factor['max_compression_factor'])[0, 1]
            trend = "positive" if corr > 0.2 else "negative" if corr < -0.2 else "weak"
            lines.append(f"- Correlation(original_dim, max safe compression factor): {corr:+.2f} ({trend} association)")
            if trend == "positive":
                lines.append("  Higher-dimensional layers generally retain calibration under stronger compression.")
            elif trend == "negative":
                lines.append("  Lower-dimensional layers appear to tolerate proportionally more compression.")
            else:
                lines.append("  Compression headroom does not clearly depend on original dimensionality.")
            
            available_bins = [
                label for label, _, _ in ORIGINAL_DIM_BINS
                if label in summary_df['original_dim_bin'].dropna().unique()
            ]
            if available_bins:
                lines.append("\n- Compression tolerance by original dimension range:")
                for bin_label in available_bins:
                    bin_data = summary_df[summary_df['original_dim_bin'] == bin_label]
                    if bin_data.empty:
                        continue
                    avg_max_factor = bin_data['max_compression_factor'].dropna().mean()
                    layer_count = len(bin_data)
                    lines.append(f"  {bin_label}: {layer_count} layers, avg max factor {avg_max_factor:.2f}×")
        else:
            lines.append("- Not enough layers with safe compression data to evaluate dimensionality trends.")
        
        valid_best = summary_df.dropna(subset=['original_dim', 'best_ratio'])
        if len(valid_best) > 1:
            corr_best = np.corrcoef(valid_best['original_dim'], valid_best['best_ratio'])[0, 1]
            lines.append(f"- Correlation(original_dim, best <5% ratio): {corr_best:+.2f}")
        else:
            lines.append("- Insufficient data to correlate original_dim with best <5% ratio.")
        
        summary_df['has_safe_range'] = summary_df['safe_count'] > 0
        robustness_df = summary_df.groupby(['dataset', 'model']).agg({
            'has_safe_range': 'sum',
            'safe_count': 'sum',
            'max_compression_factor': lambda x: np.nanmean(x) if np.any(~np.isnan(x)) else np.nan
        }).reset_index()
        robustness_df = robustness_df.sort_values(
            ['has_safe_range', 'safe_count', 'max_compression_factor'],
            ascending=[False, False, False]
        )
        
        top_robust = robustness_df.head(3)
        if len(top_robust) > 0:
            lines.append("- Most compression-robust configurations:")
            for _, row in top_robust.iterrows():
                lines.append(
                    f"  • {row['dataset']} / {row['model']}: "
                    f"{int(row['has_safe_range'])} layers with safe ranges, "
                    f"avg max compression factor {row['max_compression_factor']:.2f}"
                )
        else:
            lines.append("- No configurations exhibit safe compression ranges (>0.8 Good≤5%).")
    
    lines.append("")
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 60)
    
    if summary_df.empty:
        lines.append("Unable to form recommendations due to missing data.")
    else:
        target_original_dims = [2048, 4096, 8192]
        for target_dim in target_original_dims:
            tolerance = max(256, target_dim * 0.15)
            subset = summary_df[
                summary_df['original_dim'].notna() &
                (np.abs(summary_df['original_dim'] - target_dim) <= tolerance)
            ]
            if subset.empty:
                lines.append(f"- Original dim ≈ {target_dim}: no data available.")
                continue
            
            candidate_ratios = subset['best_ratio'].dropna()
            if candidate_ratios.empty:
                candidate_ratios = subset['safe_ratio_min'].dropna()
            
            if candidate_ratios.empty:
                lines.append(f"- Original dim ≈ {target_dim}: no stable compression observed.")
                continue
            
            recommended_ratio = float(np.median(candidate_ratios))
            lines.append(
                f"- Original dim ≈ {target_dim}: recommend compression ≈ "
                f"{format_compression_ratio(recommended_ratio)} based on median stable runs."
            )
        
        high_variance = filtered[filtered['rel_ece_std'].notna()].sort_values('rel_ece_std', ascending=False)
        high_variance = high_variance[high_variance['rel_ece_std'] > 5].head(5)
        if not high_variance.empty:
            lines.append("- Warning: High calibration variance observed in:")
            for _, row in high_variance.iterrows():
                lines.append(
                    f"  • {row['dataset']} / {row['model']} / Layer {row['layer_name']} "
                    f"@ {format_compression_ratio(row['compression_ratio'])} "
                    f"(σΔECE={row['rel_ece_std']:.1f}%)"
                )
        else:
            lines.append("- No high-variance configurations (σΔECE > 5%) detected.")
        
        suggested_ratios = summary_df.dropna(subset=['best_ratio'])
        if not suggested_ratios.empty:
            suggestion = suggested_ratios.groupby('original_dim')['best_ratio'].median().reset_index()
            lines.append("- Suggested production compression factors (median best <5% ratios):")
            for _, row in suggestion.iterrows():
                lines.append(
                    f"  • Original {int(row['original_dim'])} → {format_compression_ratio(row['best_ratio'])}"
                )
        else:
            lines.append("- No stable <5% ratios identified for production guidance.")
    
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved layer compression report: {output_path}")


def generate_summary_statistics(df: pd.DataFrame, output_path: Path) -> None:
    """Generate text summary of key findings."""
    lines = ["COMPRESSION EXPERIMENT SUMMARY", "=" * 60, ""]
    
    for layer_type in ['best_layer', 'data_layer']:
        layer_df = df[df['layer_type'] == layer_type].copy()
        layer_df['compression_ratio_value'] = pd.to_numeric(layer_df['compression_ratio'], errors='coerce')
        layer_df = layer_df.dropna(subset=['compression_ratio_value'])
        if layer_df.empty:
            lines.append(f"\n{layer_type.replace('_', ' ').title()}:")
            lines.append("-" * 40)
            lines.append("  No data available.")
            continue
        
        lines.append(f"\n{layer_type.replace('_', ' ').title()}:")
        lines.append("-" * 40)
        
        for ratio_value in sorted(layer_df['compression_ratio_value'].unique()):
            ratio_df = layer_df[layer_df['compression_ratio_value'] == ratio_value]
            
            mean_rel = ratio_df['rel_ece_mean'].mean()
            good_share = (ratio_df['rel_ece_mean'] <= 5).mean()
            n_configs = len(ratio_df)
            
            lines.append(
                f"  Ratio {format_compression_ratio(ratio_value):>6}: ΔECE = {mean_rel:+6.1f}%, "
                f"Good≤5% = {good_share:.2f}, N = {n_configs}"
            )
        
        mean_by_ratio = layer_df.groupby('compression_ratio_value')['rel_ece_mean'].mean()
        if not mean_by_ratio.empty:
            best_ratio_value = mean_by_ratio.idxmin()
            best_ratio_mean = mean_by_ratio.loc[best_ratio_value]
            lines.append(
                f"\n  → Best compression: {format_compression_ratio(best_ratio_value)} "
                f"(ΔECE {best_ratio_mean:+.1f}%)"
            )
        else:
            lines.append("\n  → Best compression: N/A")
        
        safe_ratio_share = layer_df.groupby('compression_ratio_value').apply(
            lambda g: (g['rel_ece_mean'] <= 5).mean()
        ).rename('good_share')
        safe_ratios = safe_ratio_share[safe_ratio_share >= 0.8]
        
        if len(safe_ratios) > 0:
            safe_range = (
                f"{format_compression_ratio(safe_ratios.index.min())}-"
                f"{format_compression_ratio(safe_ratios.index.max())}"
            )
            lines.append(f"  → Safe range (≥80% within 5%): {safe_range}")
    
    output_path.write_text("\n".join(lines))
    print(f"Saved summary statistics: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate compression results for paper"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='aaai_full_experiments/results/compression_experiments_ratio',
        help='Directory containing compression experiment results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='aaai_full_experiments/results/compression_paper_analysis',
        help='Directory for paper-ready outputs'
    )
    parser.add_argument(
        '--dimensions',
        type=int,
        nargs='+',
        default=None,
        help='Specific dimensions to include in table (default: all)'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    results = load_results(results_dir)
    
    if len(results) == 0:
        print("No results found!")
        return
    
    required_columns = {
        'dataset', 'model', 'training_loss', 'layer_type', 'layer_name',
        'target_dim', 'original_dim', 'ece', 'compression_ratio', 'seed'
    }
    
    missing_columns = set()
    missing_layer_name_count = 0
    original_dim_by_layer = defaultdict(set)
    
    for r in results:
        missing_columns.update(required_columns - set(r.keys()))
        if r.get('layer_name') is None:
            missing_layer_name_count += 1
        else:
            original_dim_by_layer[(r.get('dataset'), r.get('model'), r.get('layer_name'))].add(r.get('original_dim'))
    
    if missing_columns:
        print(f"Warning: Missing required columns in results: {sorted(missing_columns)}")
    
    total_records = len(results)
    missing_layer_name_pct = (missing_layer_name_count / total_records * 100) if total_records > 0 else 0
    if missing_layer_name_count > 0:
        print(f"Warning: {missing_layer_name_count}/{total_records} results "
              f"({missing_layer_name_pct:.2f}%) missing layer_name")
        if missing_layer_name_pct >= 1.0:
            print("Error: More than 1% of records lack layer_name; please fix the input data.")
    
    inconsistent_layers = [
        (dataset, model, layer_name, sorted(values))
        for (dataset, model, layer_name), values in original_dim_by_layer.items()
        if len({v for v in values if v is not None}) > 1
    ]
    if inconsistent_layers:
        for dataset, model, layer_name, dims in inconsistent_layers:
            print(f"Warning: Inconsistent original_dim for {dataset}/{model}/Layer {layer_name}: {dims}")
    
    unique_layers = {
        key for key in {(r['dataset'], r['model'], r['layer_name']) for r in results if r.get('layer_name') is not None}
    }
    print(f"Found {len(unique_layers)} unique (dataset, model, layer_name) combinations")
    
    role_counts = defaultdict(int)
    for r in results:
        role_counts[r.get('layer_type')] += 1
    
    print("Layer role distribution:")
    for role in ['best_layer', 'data_layer', 'worst_layer']:
        print(f"  {role}: {role_counts.get(role, 0)} instances")
    
    print("\nComputing baselines and aggregations...")
    baselines = compute_baseline_per_config(results)
    baseline_configs = {(k[0], k[1], k[2], k[3]) for k in baselines.keys()}
    total_configs = len({(r['dataset'], r['model'], r['training_loss'], r.get('layer_name'))
                        for r in results if r.get('layer_name') is not None})
    coverage_pct = (len(baseline_configs) / total_configs * 100) if total_configs > 0 else 0
    print(f"Configurations with baselines: {len(baseline_configs)}/{total_configs} ({coverage_pct:.1f}%)")
    
    df = aggregate_with_relative_change(results, baselines=baselines)
    
    csv_path = output_dir / "aggregated_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved aggregated data: {csv_path}")
    
    print("\nGenerating paper artifacts...")
    
    plot_compression_optimization(
        df,
        output_dir / "compression_optimization.png"
    )
    
    plot_compression_optimization_stratified(
        df,
        output_dir / "compression_optimization_stratified.png"
    )
    
    plot_compression_optimization_robust(
        df,
        output_dir / "compression_optimization_robust.png"
    )
    
    plot_compression_best_case(
        df,
        output_dir / "compression_best_case.png"
    )
    
    plot_compression_heatmap_per_config(
        df,
        output_dir / "compression_configuration_heatmap.png",
        layer_type='best_layer'
    )
    
    make_paper_table(
        df,
        output_dir / "paper_compression_table.tex",
        dimensions=args.dimensions
    )
    
    make_stratified_table(
        df,
        output_dir / "table_s1_compression_by_config.tex",
        layer_type='best_layer'
    )
    
    plot_main_figure(
        df,
        output_dir / "main_compression_figure.png"
    )
    
    plot_compression_by_original_dim(
        df,
        output_dir / "compression_by_layer_depth.png",
        layer_type_filter='best_layer'
    )
    
    plot_per_layer_curves(
        df,
        output_dir,
        layer_type_filter='best_layer'
    )
    
    plot_timing_analysis(
        df,
        output_dir / "compression_timing_analysis.png",
        layer_type='best_layer'
    )
    
    generate_layer_compression_report(
        df,
        output_dir / "layer_compression_report.txt",
        layer_type_filter='best_layer'
    )
    
    generate_compression_guidelines_table(
        df,
        output_dir / "compression_guidelines_table.tex",
        layer_type_filter='best_layer'
    )
    
    generate_summary_statistics(
        df,
        output_dir / "summary.txt"
    )
    
    print(f"\n{'='*60}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()