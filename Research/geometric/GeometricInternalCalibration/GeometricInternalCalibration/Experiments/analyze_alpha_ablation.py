"""
Analysis Script for Alpha Ablation Study

This script analyzes the results from the alpha ablation experiment and generates:
1. Statistical summaries (mean ECE per alpha, best alpha per dataset)
2. Statistical significance tests (paired t-tests comparing each alpha to baseline α=0.20)
3. Visualizations (line plots, box plots, heatmaps)
4. A markdown report with findings and recommendations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import functions needed for accuracy computation
from Experiments.run_post_hoc_calibration import get_data_loaders, load_trained_model


def compute_model_accuracy(
    checkpoint_base_dir: Path,
    dataset: str,
    model: str,
    training_method: str,
    seed: int,
    device: str = 'cuda',
    batch_size: int = 256,
    accuracy_cache: Optional[Dict] = None
) -> Optional[float]:
    """
    Compute model accuracy on test set.
    
    Returns:
        Accuracy as a float (0.0 to 1.0), or None if model not found or error occurred.
    """
    if accuracy_cache is None:
        accuracy_cache = {}
    
    # Create cache key
    cache_key = (dataset, model, training_method, seed)
    if cache_key in accuracy_cache:
        return accuracy_cache[cache_key]
    
    try:
        # Construct model path (same pattern as in alpha_ablation_margin_cvar.py)
        dynamic_folder_name = f"{training_method}_{dataset}_{model}_seed{seed}"
        model_path = (
            checkpoint_base_dir
            / training_method
            / dataset
            / model
            / f"seed{seed}"
            / dynamic_folder_name
            / "best_model.pth"
        )
        
        # Try alternative paths
        possible_paths = [
            model_path,
            checkpoint_base_dir / training_method / dataset / model / f"seed{seed}" / "best_model.pth",
            checkpoint_base_dir / dataset / model / training_method / f"seed{seed}" / "best_model.pth",
        ]
        
        found_path = None
        for path in possible_paths:
            if path.exists():
                found_path = path
                break
        
        if found_path is None:
            accuracy_cache[cache_key] = None
            return None
        
        # Determine num_classes
        dataset_num_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'svhn': 10,
        }
        num_classes = dataset_num_classes.get(dataset)
        if num_classes is None:
            accuracy_cache[cache_key] = None
            return None
        
        # Load model
        device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
        model_obj = load_trained_model(
            model_path=str(found_path),
            model_name=model,
            num_classes=num_classes,
            device=device_obj,
            dataset=dataset
        )
        model_obj = model_obj.to(device_obj)
        model_obj.eval()
        
        # Load test data
        _, _, test_loader, _ = get_data_loaders(
            dataset=dataset,
            batch_size=batch_size,
        )
        
        # Compute accuracy
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device_obj)
                batch_y = batch_y.to(device_obj)
                
                outputs = model_obj(batch_x)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        accuracy_cache[cache_key] = accuracy
        return accuracy
        
    except Exception as e:
        print(f"Warning: Failed to compute accuracy for {dataset}/{model}/{training_method}/seed{seed}: {e}")
        accuracy_cache[cache_key] = None
        return None


def load_results(results_dir: Path, min_accuracy: float = 0.6, checkpoint_base_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load all result JSON files from the experiment directory.
    
    New format: Each JSON file contains multiple selection strategies in 'selected_layer_per_config'.
    We create one row per strategy.
    
    Args:
        results_dir: Directory containing result JSON files
        min_accuracy: Minimum model accuracy threshold (default: 0.6 = 60%)
        checkpoint_base_dir: Base directory for model checkpoints (default: inferred from results_dir)

    Returns:
        DataFrame with columns: dataset, model, training_method, seed, alpha, strategy, test_ece, selected_layer, etc.
    """
    # Infer checkpoint base dir if not provided
    if checkpoint_base_dir is None:
        # Default location based on alpha_ablation_margin_cvar.py
        checkpoint_base_dir = Path("aaai_full_experiments/results/baseline")
    
    results = []
    accuracy_cache = {}  # Cache accuracy computations
    filtered_count = 0
    total_count = 0

    for json_file in results_dir.rglob("*.json"):
        try:
            with json_file.open('r') as f:
                data = json.load(f)

            # Extract metadata (same for all strategies in this file)
            metadata = data.get('metadata', {})
            alpha = data.get('alpha')
            dataset = metadata.get('dataset')
            model = metadata.get('model')
            training_method = metadata.get('training_method')
            seed = metadata.get('seed')
            
            # Check model accuracy (only once per file, not per strategy)
            total_count += 1
            model_accuracy = compute_model_accuracy(
                checkpoint_base_dir=checkpoint_base_dir,
                dataset=dataset,
                model=model,
                training_method=training_method,
                seed=seed,
                accuracy_cache=accuracy_cache
            )
            
            # Filter by accuracy threshold
            if model_accuracy is None:
                print(f"  ⚠ Skipping {dataset}/{model}/{training_method}/seed{seed}: Could not compute accuracy")
                filtered_count += 1
                continue
            
            if model_accuracy < min_accuracy:
                print(f"  ⚠ Skipping {dataset}/{model}/{training_method}/seed{seed}: Accuracy {model_accuracy:.2%} < {min_accuracy:.0%}")
                filtered_count += 1
                continue
            
            # Get uncalibrated ECE (should be same for all strategies, get from first)
            uncalibrated_ece = None
            selected_layer_per_config = data.get('selected_layer_per_config', {})
            
            if selected_layer_per_config:
                # Get uncalibrated_ece from first strategy (all should be the same)
                first_strategy = list(selected_layer_per_config.values())[0]
                uncalibrated_ece = first_strategy.get('uncalibrated_ece')
            
            # Create one row per selection strategy
            for strategy_name, strategy_result in selected_layer_per_config.items():
                result = {
                    'dataset': metadata.get('dataset'),
                    'model': metadata.get('model'),
                    'training_method': metadata.get('training_method'),
                    'seed': metadata.get('seed'),
                    'alpha': alpha,
                    'strategy': strategy_name,
                    'test_ece': strategy_result.get('test_ece'),
                    'uncalibrated_ece': strategy_result.get('uncalibrated_ece', uncalibrated_ece),
                    'ece_improvement': strategy_result.get('ece_improvement'),
                    'selected_layer': strategy_result.get('layer'),
                    'num_classes': metadata.get('num_classes'),
                    'model_accuracy': model_accuracy,  # Add accuracy to results
                    'file_path': str(json_file),
                }
                
                # Add strategy-specific metrics if available
                if 'normalized_cvar_score' in strategy_result:
                    result['normalized_cvar_score'] = strategy_result['normalized_cvar_score']
                if 'raw_cvar' in strategy_result:
                    result['raw_cvar'] = strategy_result['raw_cvar']
                if 'nc1_collapse' in strategy_result:
                    result['nc1_collapse'] = strategy_result['nc1_collapse']
                if 'nc4_separability' in strategy_result:
                    result['nc4_separability'] = strategy_result['nc4_separability']
                if 'calibration_decisiveness' in strategy_result:
                    result['calibration_decisiveness'] = strategy_result['calibration_decisiveness']
                if 'ensemble_score' in strategy_result:
                    result['ensemble_score'] = strategy_result['ensemble_score']
                
                results.append(result)

        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            import traceback
            print(traceback.format_exc())
            continue

    if not results:
        raise ValueError(f"No valid results found in {results_dir}")

    df = pd.DataFrame(results)
    
    if len(df) == 0:
        raise ValueError(f"No valid results found in {results_dir} (after filtering by accuracy >= {min_accuracy:.0%})")
    
    print(f"Loaded {len(df)} results (rows) from {results_dir}")
    print(f"  Total files processed: {total_count}")
    print(f"  Files filtered out (accuracy < {min_accuracy:.0%}): {filtered_count}")
    print(f"  Files included: {total_count - filtered_count}")
    print(f"  Unique experiments: {len(df.groupby(['dataset', 'model', 'training_method', 'seed', 'alpha']))}")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Training methods: {sorted(df['training_method'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    print(f"  Alphas: {sorted(df['alpha'].unique())}")
    print(f"  Strategies: {sorted(df['strategy'].unique())}")
    if 'model_accuracy' in df.columns:
        print(f"  Accuracy range: {df['model_accuracy'].min():.2%} - {df['model_accuracy'].max():.2%} (mean: {df['model_accuracy'].mean():.2%})")

    return df


def compute_summary_statistics(df: pd.DataFrame, group_by_strategy: bool = False) -> pd.DataFrame:
    """
    Compute summary statistics: mean ECE per (dataset, alpha) or (dataset, alpha, strategy).
    
    Args:
        df: DataFrame with results
        group_by_strategy: If True, include strategy in grouping
    """
    if group_by_strategy:
        group_cols = ['dataset', 'alpha', 'strategy']
    else:
        group_cols = ['dataset', 'alpha']
    
    summary = df.groupby(group_cols).agg({
        'test_ece': ['mean', 'std', 'min', 'max', 'count'],
        'ece_improvement': ['mean', 'std'],
        'selected_layer': ['mean', 'std'],
    }).reset_index()

    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in summary.columns.values]

    return summary


def find_best_alpha_per_dataset(df: pd.DataFrame, strategy: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
    """
    Find the alpha value with lowest mean ECE for each dataset.
    
    Args:
        df: DataFrame with results
        strategy: If provided, filter to this strategy only. If None, aggregate across all strategies.

    Returns:
        Dict mapping dataset -> (best_alpha, mean_ece)
    """
    df_filtered = df.copy()
    if strategy:
        df_filtered = df_filtered[df_filtered['strategy'] == strategy]
    
    summary = df_filtered.groupby(['dataset', 'alpha'])['test_ece'].mean().reset_index()

    best_alphas = {}
    for dataset in summary['dataset'].unique():
        dataset_data = summary[summary['dataset'] == dataset]
        best_row = dataset_data.loc[dataset_data['test_ece'].idxmin()]
        best_alphas[dataset] = (best_row['alpha'], best_row['test_ece'])

    return best_alphas


def perform_statistical_tests(df: pd.DataFrame, baseline_alpha: float = 0.20, strategy: Optional[str] = None) -> pd.DataFrame:
    """
    Perform paired t-tests comparing each alpha to the baseline (α=0.20).

    For each (dataset, alpha) pair, test if ECE is significantly different from baseline.

    Args:
        df: DataFrame with results
        baseline_alpha: Baseline alpha value for comparison
        strategy: If provided, filter to this strategy only. If None, aggregate across all strategies.

    Returns:
        DataFrame with columns: dataset, alpha, mean_ece, baseline_mean_ece, t_statistic, p_value, significant
    """
    df_filtered = df.copy()
    if strategy:
        df_filtered = df_filtered[df_filtered['strategy'] == strategy]
    
    results = []

    for dataset in df_filtered['dataset'].unique():
        dataset_df = df_filtered[df_filtered['dataset'] == dataset]

        # Get baseline data
        baseline_data = dataset_df[dataset_df['alpha'] == baseline_alpha]['test_ece'].values

        if len(baseline_data) == 0:
            print(f"Warning: No baseline data (α={baseline_alpha}) for {dataset}")
            continue

        for alpha in df_filtered['alpha'].unique():
            if alpha == baseline_alpha:
                continue  # Skip comparing baseline to itself

            alpha_data = dataset_df[dataset_df['alpha'] == alpha]['test_ece'].values

            if len(alpha_data) == 0:
                print(f"Warning: No data for α={alpha} on {dataset}")
                continue

            # Paired t-test (requires same number of samples)
            # If different sample sizes, use independent t-test instead
            if len(alpha_data) == len(baseline_data):
                t_stat, p_value = stats.ttest_rel(alpha_data, baseline_data)
                test_type = "paired"
            else:
                t_stat, p_value = stats.ttest_ind(alpha_data, baseline_data)
                test_type = "independent"

            results.append({
                'dataset': dataset,
                'alpha': alpha,
                'mean_ece': alpha_data.mean(),
                'std_ece': alpha_data.std(),
                'baseline_mean_ece': baseline_data.mean(),
                'baseline_std_ece': baseline_data.std(),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'test_type': test_type,
                'n_samples': len(alpha_data),
            })

    return pd.DataFrame(results)


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """
    Create visualizations:
    1. Line plot: ECE vs alpha, one line per dataset
    2. Box plot: ECE distribution per alpha, faceted by dataset
    3. Heatmap: dataset × alpha, color = mean ECE
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # 1. Line plot: Mean ECE vs Alpha per dataset
    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]
        summary = dataset_df.groupby('alpha')['test_ece'].agg(['mean', 'std']).reset_index()

        ax.plot(summary['alpha'], summary['mean'], marker='o', label=dataset.upper(), linewidth=2)
        ax.fill_between(
            summary['alpha'],
            summary['mean'] - summary['std'],
            summary['mean'] + summary['std'],
            alpha=0.2
        )

    ax.set_xlabel('Alpha (α) - Tail Percentage', fontsize=12)
    ax.set_ylabel('Mean Test ECE', fontsize=12)
    ax.set_title('Effect of Alpha on Calibration Performance', fontsize=14, fontweight='bold')
    ax.legend(title='Dataset', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add vertical line at baseline α=0.20
    ax.axvline(x=0.20, color='gray', linestyle='--', alpha=0.5, label='Baseline (α=0.20)')

    plt.tight_layout()
    plt.savefig(output_dir / 'ece_vs_alpha_lineplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved line plot: {output_dir / 'ece_vs_alpha_lineplot.png'}")

    # 2. Box plot: ECE distribution per alpha, faceted by dataset
    datasets = sorted(df['dataset'].unique())
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 6), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        dataset_df = df[df['dataset'] == dataset]

        sns.boxplot(
            data=dataset_df,
            x='alpha',
            y='test_ece',
            ax=ax,
            palette='Set2'
        )

        ax.set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Alpha (α)', fontsize=12)
        ax.set_ylabel('Test ECE' if dataset == datasets[0] else '', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight baseline
        baseline_idx = list(dataset_df['alpha'].unique()).index(0.20) if 0.20 in dataset_df['alpha'].unique() else None
        if baseline_idx is not None:
            ax.axvline(x=baseline_idx, color='red', linestyle='--', alpha=0.5, linewidth=2)

    plt.tight_layout()
    plt.savefig(output_dir / 'ece_boxplot_by_dataset.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved box plot: {output_dir / 'ece_boxplot_by_dataset.png'}")

    # 3. Heatmap: Dataset × Alpha (mean ECE)
    heatmap_data = df.groupby(['dataset', 'alpha'])['test_ece'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='dataset', columns='alpha', values='test_ece')

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        heatmap_pivot,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn_r',  # Red = high ECE (bad), Green = low ECE (good)
        cbar_kws={'label': 'Mean Test ECE'},
        ax=ax,
        linewidths=0.5
    )

    ax.set_title('Mean Test ECE: Dataset × Alpha', fontsize=14, fontweight='bold')
    ax.set_xlabel('Alpha (α)', fontsize=12)
    ax.set_ylabel('Dataset', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'ece_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved heatmap: {output_dir / 'ece_heatmap.png'}")

    # 4. Layer selection consistency: How often does each alpha pick the same layer?
    fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 6), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        dataset_df = df[df['dataset'] == dataset]

        # Count layer selections per alpha
        layer_counts = dataset_df.groupby(['alpha', 'selected_layer']).size().reset_index(name='count')

        # Get most common layer per alpha
        most_common_layers = layer_counts.loc[layer_counts.groupby('alpha')['count'].idxmax()]

        ax.bar(most_common_layers['alpha'].astype(str), most_common_layers['count'])
        ax.set_title(f'{dataset.upper()} - Layer Selection Consistency', fontsize=12, fontweight='bold')
        ax.set_xlabel('Alpha (α)', fontsize=10)
        ax.set_ylabel('Count (Most Common Layer)' if dataset == datasets[0] else '', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'layer_selection_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved layer selection plot: {output_dir / 'layer_selection_consistency.png'}")


def generate_report(
    df: pd.DataFrame,
    summary_stats: pd.DataFrame,
    best_alphas: Dict[str, Tuple[float, float]],
    stat_tests: pd.DataFrame,
    output_file: Path
):
    """
    Generate a markdown report summarizing the findings.
    """
    report_lines = [
        "# Alpha Ablation Study for MarginTailCVaRCalculator",
        "",
        "## Executive Summary",
        "",
        "This report summarizes the results of testing different alpha (α) values for the MarginTailCVaRCalculator metric.",
        "The goal was to determine the optimal tail percentage for layer selection on different datasets.",
        "",
        "### Key Findings",
        "",
    ]

    # Best alpha per dataset
    report_lines.append("**Best Alpha per Dataset:**")
    report_lines.append("")
    for dataset, (alpha, ece) in sorted(best_alphas.items()):
        report_lines.append(f"- **{dataset.upper()}**: α = {alpha:.2f} (Mean ECE = {ece:.4f})")
    report_lines.append("")

    # Baseline comparison
    baseline_alpha = 0.20
    report_lines.append(f"**Comparison to Baseline (α = {baseline_alpha}):**")
    report_lines.append("")

    baseline_summary = df[df['alpha'] == baseline_alpha].groupby('dataset')['test_ece'].mean()
    for dataset, (best_alpha, best_ece) in sorted(best_alphas.items()):
        if dataset in baseline_summary.index:
            baseline_ece = baseline_summary[dataset]
            improvement = baseline_ece - best_ece
            improvement_pct = (improvement / baseline_ece) * 100
            report_lines.append(
                f"- **{dataset.upper()}**: Best α={best_alpha:.2f} improves ECE by {improvement:.4f} "
                f"({improvement_pct:+.2f}%) compared to baseline α={baseline_alpha}"
            )
    report_lines.append("")

    # Statistical significance
    report_lines.extend([
        "## Statistical Analysis",
        "",
        "### Hypothesis Tests",
        "",
        "We performed statistical tests comparing each alpha to the baseline (α = 0.20):",
        "",
    ])

    for dataset in sorted(df['dataset'].unique()):
        dataset_tests = stat_tests[stat_tests['dataset'] == dataset]

        if len(dataset_tests) == 0:
            continue

        report_lines.append(f"**{dataset.upper()}:**")
        report_lines.append("")

        for _, row in dataset_tests.iterrows():
            significance = "✓ Significant" if row['significant'] else "✗ Not significant"
            direction = "better" if row['mean_ece'] < row['baseline_mean_ece'] else "worse"

            report_lines.append(
                f"- α = {row['alpha']:.2f}: Mean ECE = {row['mean_ece']:.4f} vs baseline {row['baseline_mean_ece']:.4f} "
                f"({direction}), p = {row['p_value']:.4f} - {significance}"
            )

        report_lines.append("")

    # Summary statistics table
    report_lines.extend([
        "## Summary Statistics",
        "",
        "### Mean ECE by Dataset and Alpha",
        "",
    ])

    # Create a formatted table
    report_lines.append("| Dataset | Alpha | Mean ECE | Std ECE | Min ECE | Max ECE | N |")
    report_lines.append("|---------|-------|----------|---------|---------|---------|---|")

    for _, row in summary_stats.iterrows():
        report_lines.append(
            f"| {row['dataset'].upper()} | {row['alpha']:.2f} | "
            f"{row['test_ece_mean']:.4f} | {row['test_ece_std']:.4f} | "
            f"{row['test_ece_min']:.4f} | {row['test_ece_max']:.4f} | "
            f"{int(row['test_ece_count'])} |"
        )

    report_lines.append("")

    # Interpretation
    report_lines.extend([
        "## Interpretation",
        "",
        "### Dataset-Specific Patterns",
        "",
    ])

    for dataset in sorted(df['dataset'].unique()):
        best_alpha, best_ece = best_alphas[dataset]

        # Determine tail size interpretation
        if best_alpha <= 0.10:
            tail_size = "very tight (5-10%)"
        elif best_alpha <= 0.15:
            tail_size = "tight (10-15%)"
        elif best_alpha <= 0.20:
            tail_size = "moderate (15-20%)"
        elif best_alpha <= 0.25:
            tail_size = "broad (20-25%)"
        else:
            tail_size = "very broad (25-30%)"

        report_lines.append(f"**{dataset.upper()}**: Best performance with α = {best_alpha:.2f} ({tail_size} tail)")

        # Check if hypothesis was supported
        if dataset == "cifar100":
            hypothesis = "0.10-0.15 (tight tail)"
            supported = 0.10 <= best_alpha <= 0.15
        elif dataset == "cifar10":
            hypothesis = "0.20-0.30 (broad tail)"
            supported = 0.20 <= best_alpha <= 0.30
        else:
            hypothesis = "unclear"
            supported = None

        if hypothesis != "unclear":
            status = "✓ Supported" if supported else "✗ Not supported"
            report_lines.append(f"- Hypothesis: {hypothesis} - {status}")

        report_lines.append("")

    # Recommendations
    report_lines.extend([
        "## Recommendations",
        "",
        "Based on the experimental results:",
        "",
    ])

    # Check if one alpha dominates across all datasets
    alpha_wins = defaultdict(int)
    for dataset, (alpha, _) in best_alphas.items():
        alpha_wins[alpha] += 1

    if max(alpha_wins.values()) == len(best_alphas):
        # Same alpha is best for all datasets
        universal_alpha = max(alpha_wins, key=alpha_wins.get)
        report_lines.append(
            f"1. **Universal Setting**: α = {universal_alpha:.2f} is optimal across all tested datasets. "
            f"Use this as the default value."
        )
    else:
        # Dataset-specific alphas are better
        report_lines.append("1. **Dataset-Adaptive Setting**: Use dataset-specific alpha values for optimal performance:")
        for dataset, (alpha, _) in sorted(best_alphas.items()):
            report_lines.append(f"   - {dataset.upper()}: α = {alpha:.2f}")

    report_lines.extend([
        "",
        f"2. **Baseline Comparison**: The current baseline (α = {baseline_alpha}) is "
        + ("adequate for most datasets" if baseline_alpha in alpha_wins else "suboptimal - consider updating"),
        "",
        "3. **Future Work**: ",
        "   - Test additional alpha values for fine-grained optimization",
        "   - Extend study to other datasets (ImageNet, CIFAR-10-C, etc.)",
        "   - Investigate interaction between alpha and other hyperparameters (compression ratio, etc.)",
        "",
        "## Visualizations",
        "",
        "See the following figures for detailed analysis:",
        "- `ece_vs_alpha_lineplot.png`: Mean ECE vs alpha for each dataset",
        "- `ece_boxplot_by_dataset.png`: ECE distribution per alpha (boxplots)",
        "- `ece_heatmap.png`: Heatmap of mean ECE across datasets and alphas",
        "- `layer_selection_consistency.png`: Layer selection consistency per alpha",
        "",
        "---",
        f"*Report generated from {len(df)} experiments*",
    ])

    # Write report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w') as f:
        f.write('\n'.join(report_lines))

    print(f"  ✓ Generated report: {output_file}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze alpha ablation study results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/alpha_ablation_margin_cvar"),
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for analysis (default: same as results-dir)"
    )
    parser.add_argument(
        "--baseline-alpha",
        type=float,
        default=0.20,
        help="Baseline alpha value for comparison"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Filter analysis to specific strategy (e.g., 'margin_cvar_tanh2.0', 'nc4_separability'). If None, aggregate across all strategies."
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir

    print("="*80)
    print("Alpha Ablation Analysis")
    print("="*80)

    # Load results
    print("\n1. Loading results...")
    print(f"  Filtering models with accuracy >= {args.min_accuracy:.0%}")
    df = load_results(
        args.results_dir,
        min_accuracy=args.min_accuracy,
        checkpoint_base_dir=args.checkpoint_base_dir
    )
    
    # Filter by strategy if specified
    if args.strategy:
        if args.strategy not in df['strategy'].unique():
            print(f"Error: Strategy '{args.strategy}' not found in results.")
            print(f"Available strategies: {sorted(df['strategy'].unique())}")
            return 1
        df = df[df['strategy'] == args.strategy]
        print(f"  Filtered to strategy: {args.strategy}")
        print(f"  Remaining rows: {len(df)}")

    # Compute summary statistics
    print("\n2. Computing summary statistics...")
    summary_stats = compute_summary_statistics(df, group_by_strategy=False)
    print(f"  ✓ Computed statistics for {len(summary_stats)} (dataset, alpha) combinations")
    
    # Also compute per-strategy statistics if not filtering
    summary_stats_by_strategy = None
    if not args.strategy:
        summary_stats_by_strategy = compute_summary_statistics(df, group_by_strategy=True)
        print(f"  ✓ Per-strategy statistics: {len(summary_stats_by_strategy)} combinations")

    # Find best alphas
    print("\n3. Finding best alpha per dataset...")
    best_alphas = find_best_alpha_per_dataset(df, strategy=args.strategy)
    for dataset, (alpha, ece) in sorted(best_alphas.items()):
        print(f"  ✓ {dataset.upper()}: α = {alpha:.2f} (ECE = {ece:.4f})")

    # Perform statistical tests
    print(f"\n4. Performing statistical tests (baseline α = {args.baseline_alpha})...")
    stat_tests = perform_statistical_tests(df, baseline_alpha=args.baseline_alpha, strategy=args.strategy)
    print(f"  ✓ Completed {len(stat_tests)} statistical tests")

    # Create visualizations
    print("\n5. Creating visualizations...")
    create_visualizations(df, args.output_dir)

    # Generate report
    print("\n6. Generating report...")
    report_file = args.output_dir / "REPORT.md"
    generate_report(df, summary_stats, best_alphas, stat_tests, report_file)

    # Save summary data
    print("\n7. Saving summary data...")
    summary_stats.to_csv(args.output_dir / "summary_statistics.csv", index=False)
    stat_tests.to_csv(args.output_dir / "statistical_tests.csv", index=False)
    print(f"  ✓ Saved summary_statistics.csv")
    print(f"  ✓ Saved statistical_tests.csv")
    
    # Save per-strategy statistics if not filtering
    if not args.strategy:
        summary_stats_by_strategy.to_csv(args.output_dir / "summary_statistics_by_strategy.csv", index=False)
        print(f"  ✓ Saved summary_statistics_by_strategy.csv")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - REPORT.md: Comprehensive analysis report")
    print(f"  - summary_statistics.csv: Summary stats by dataset/alpha")
    print(f"  - statistical_tests.csv: Statistical test results")
    print(f"  - *.png: Visualization figures")

    return 0


if __name__ == "__main__":
    sys.exit(main())
