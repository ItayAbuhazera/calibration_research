#!/usr/bin/env python3
"""
NC-Based Multi-Layer Ensemble Results Aggregator with Baseline Comparison

This script aggregates and analyzes results from NC-guided ensemble calibration experiments
and compares them against baseline post-hoc calibration methods (Temperature Scaling,
Platt Scaling, Isotonic Regression, Beta Calibration, etc.).

It computes statistics, compares different strategies, and identifies best configurations
across both NC ensemble methods and baseline calibration approaches.

Usage:
    # Without baseline comparison
    python aggregate_nc_ensemble_results.py --results-dir nc_ensemble_results --output-dir nc_ensemble_analysis

    # With baseline comparison
    python aggregate_nc_ensemble_results.py --results-dir nc_ensemble_results \\
           --output-dir nc_ensemble_analysis \\
           --baselines-dir aaai_full_experiments/results/layer_selection_analysis_baselines_4
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class NCEnsembleResultsAggregator:
    """Aggregator for NC-based ensemble calibration results"""

    def __init__(self, results_dir: str, output_dir: str, baselines_dir: Optional[str] = None):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Baseline directory (optional)
        self.baselines_dir = Path(baselines_dir) if baselines_dir else None

        self.all_results = []
        self.baseline_results = []
        self.df = None

    def load_all_results(self) -> int:
        """
        Recursively find and load all ensemble_results.json files.
        Returns the number of results loaded.
        """
        print("=" * 80)
        print("Loading NC Ensemble Results")
        print("=" * 80)

        result_files = list(self.results_dir.rglob("ensemble_results.json"))
        print(f"Found {len(result_files)} result files")

        loaded = 0
        failed = 0

        for result_file in result_files:
            try:
                # Parse path to extract metadata
                # Expected structure: {base}/{training_loss}/{dataset}/{model}/{seed}/{layer_strategy}/{weighting_method}/ensemble_results.json
                parts = result_file.relative_to(self.results_dir).parts

                if len(parts) < 7:
                    print(f"Skipping {result_file}: unexpected path structure")
                    failed += 1
                    continue

                training_loss = parts[0]
                dataset = parts[1]
                model = parts[2]
                seed_str = parts[3]  # e.g., 'seed19'
                layer_strategy = parts[4]
                weighting_method = parts[5]

                # Extract seed number
                seed = int(seed_str.replace('seed', ''))

                # Load results
                with open(result_file, 'r') as f:
                    results = json.load(f)

                # Process each result configuration
                for config_name, config_results in results.items():
                    if not isinstance(config_results, dict):
                        continue

                    # Create record
                    record = {
                        'training_loss': training_loss,
                        'dataset': dataset,
                        'model': model,
                        'seed': seed,
                        'layer_strategy': layer_strategy,
                        'weighting_method': weighting_method,
                        'config_name': config_name,
                        'file_path': str(result_file),
                    }

                    # Add all metrics
                    for key, value in config_results.items():
                        if key not in ['selected_layers', 'weights', 'selected_layers_canonical']:
                            record[key] = value
                        elif key == 'selected_layers':
                            record['selected_layers'] = str(value)
                            record['num_layers'] = len(value) if value else 0

                    self.all_results.append(record)
                    loaded += 1

            except Exception as e:
                print(f"Failed to load {result_file}: {e}")
                failed += 1
                continue

        print(f"\nLoaded: {loaded} configurations")
        print(f"Failed: {failed}")

        if loaded > 0:
            # Add method_type to distinguish from baselines
            for record in self.all_results:
                record['method_type'] = 'nc_ensemble'

            self.df = pd.DataFrame(self.all_results)
            print(f"\nDataFrame shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")

        return loaded

    def load_baseline_results(self) -> int:
        """
        Recursively find and load all baseline_results.json files.
        Returns the number of baseline configurations loaded.
        """
        if self.baselines_dir is None or not self.baselines_dir.exists():
            print("\nNo baselines directory specified or directory does not exist. Skipping baseline loading.")
            return 0

        print("\n" + "=" * 80)
        print("Loading Baseline Results")
        print("=" * 80)

        result_files = list(self.baselines_dir.rglob("baseline_results.json"))
        print(f"Found {len(result_files)} baseline result files")

        loaded = 0
        failed = 0

        for result_file in result_files:
            try:
                # Parse path to extract metadata
                # Expected structure: {base}/{training_loss}/{dataset}/{model}/{seed}/baseline_results.json
                parts = result_file.relative_to(self.baselines_dir).parts

                if len(parts) < 5:
                    print(f"Skipping {result_file}: unexpected path structure")
                    failed += 1
                    continue

                training_loss = parts[0]
                dataset = parts[1]
                model = parts[2]
                seed_str = parts[3]  # e.g., 'seed15'

                # Extract seed number
                seed = int(seed_str.replace('seed', ''))

                # Load results
                with open(result_file, 'r') as f:
                    results = json.load(f)

                # Process each baseline method
                for method_name, method_results in results.items():
                    if not isinstance(method_results, dict):
                        continue

                    # Create record
                    record = {
                        'training_loss': training_loss,
                        'dataset': dataset,
                        'model': model,
                        'seed': seed,
                        'method_type': 'baseline',
                        'calibration_method': method_name,
                        'file_path': str(result_file),
                    }

                    # Map baseline metrics to standard column names
                    # Baseline uses: ece, brier, acc
                    # We map to: test_ece, test_brier, test_accuracy
                    if 'ece' in method_results:
                        record['test_ece'] = method_results['ece']
                    if 'brier' in method_results:
                        record['test_brier'] = method_results['brier']
                    if 'acc' in method_results:
                        record['test_accuracy'] = method_results['acc']

                    # Store additional parameters (e.g., T for temperature scaling)
                    for key, value in method_results.items():
                        if key not in ['ece', 'brier', 'acc']:
                            record[f'param_{key}'] = value

                    self.baseline_results.append(record)
                    loaded += 1

            except Exception as e:
                print(f"Failed to load {result_file}: {e}")
                failed += 1
                continue

        print(f"\nLoaded: {loaded} baseline configurations")
        print(f"Failed: {failed}")

        return loaded

    def compute_summary_statistics(self) -> pd.DataFrame:
        """Compute mean and std for metrics grouped by different factors"""

        if self.df is None or len(self.df) == 0:
            print("No data to analyze")
            return None

        print("\n" + "=" * 80)
        print("Computing Summary Statistics")
        print("=" * 80)

        # Metrics to aggregate
        metrics = ['test_ece', 'test_accuracy', 'test_brier', 'mce']

        # Group by different factors
        groupby_configs = [
            {
                'name': 'by_layer_strategy',
                'groupby': ['layer_strategy'],
                'description': 'Layer Selection Strategy'
            },
            {
                'name': 'by_weighting_method',
                'groupby': ['weighting_method'],
                'description': 'Weighting Method'
            },
            {
                'name': 'by_strategy_and_weighting',
                'groupby': ['layer_strategy', 'weighting_method'],
                'description': 'Layer Strategy + Weighting Method'
            },
            {
                'name': 'by_dataset',
                'groupby': ['dataset'],
                'description': 'Dataset'
            },
            {
                'name': 'by_model',
                'groupby': ['model'],
                'description': 'Model'
            },
            {
                'name': 'by_dataset_model',
                'groupby': ['dataset', 'model'],
                'description': 'Dataset + Model'
            },
            {
                'name': 'by_dataset_model_strategy_weighting',
                'groupby': ['dataset', 'model', 'layer_strategy', 'weighting_method'],
                'description': 'Full Configuration'
            },
        ]

        summaries = {}

        for config in groupby_configs:
            print(f"\n{config['description']}:")
            print("-" * 80)

            groupby_cols = config['groupby']

            # Compute statistics
            agg_dict = {}
            for metric in metrics:
                if metric in self.df.columns:
                    agg_dict[f'{metric}_mean'] = (metric, 'mean')
                    agg_dict[f'{metric}_std'] = (metric, 'std')
                    agg_dict[f'{metric}_min'] = (metric, 'min')
                    agg_dict[f'{metric}_max'] = (metric, 'max')

            agg_dict['count'] = (groupby_cols[0], 'count')

            summary = self.df.groupby(groupby_cols).agg(**agg_dict).reset_index()

            # Round for readability
            for col in summary.columns:
                # Handle MultiIndex columns (from agg with tuples)
                col_name = col if isinstance(col, str) else str(col)
                
                # Check if this is a metric column that should be rounded
                if '_mean' in col_name or '_std' in col_name or '_min' in col_name or '_max' in col_name:
                    # Only round if the column is numeric
                    if pd.api.types.is_numeric_dtype(summary[col]):
                        summary[col] = summary[col].round(6)

            summaries[config['name']] = summary

            # Print top 10 by ECE (if available)
            if 'test_ece_mean' in summary.columns:
                print(f"\nTop 10 by ECE (lower is better):")
                top_10 = summary.nsmallest(10, 'test_ece_mean')
                print(top_10.to_string(index=False))

            # Save to CSV
            output_file = self.output_dir / f"summary_{config['name']}.csv"
            summary.to_csv(output_file, index=False)
            print(f"\nSaved to: {output_file}")

        return summaries

    def analyze_best_configurations(self) -> Dict[str, pd.DataFrame]:
        """Identify best configurations for different metrics"""

        if self.df is None or len(self.df) == 0:
            return {}

        print("\n" + "=" * 80)
        print("Best Configurations Analysis")
        print("=" * 80)

        # Compute mean performance across seeds for each configuration
        config_cols = ['dataset', 'model', 'layer_strategy', 'weighting_method']
        metrics = ['test_ece', 'test_accuracy', 'test_brier', 'mce']

        # Group by configuration
        agg_dict = {}
        for metric in metrics:
            if metric in self.df.columns:
                agg_dict[f'{metric}_mean'] = (metric, 'mean')
                agg_dict[f'{metric}_std'] = (metric, 'std')
        agg_dict['count'] = ('seed', 'count')

        config_summary = self.df.groupby(config_cols).agg(**agg_dict).reset_index()

        best_configs = {}

        # Best by ECE (lower is better)
        if 'test_ece_mean' in config_summary.columns:
            print("\n" + "-" * 80)
            print("TOP 10 CONFIGURATIONS BY ECE (Lower is Better)")
            print("-" * 80)
            best_ece = config_summary.nsmallest(10, 'test_ece_mean')
            print(best_ece.to_string(index=False))
            best_configs['best_by_ece'] = best_ece

            # Save
            best_ece.to_csv(self.output_dir / "best_by_ece.csv", index=False)

        # Best by accuracy (higher is better)
        if 'test_accuracy_mean' in config_summary.columns:
            print("\n" + "-" * 80)
            print("TOP 10 CONFIGURATIONS BY ACCURACY (Higher is Better)")
            print("-" * 80)
            best_acc = config_summary.nlargest(10, 'test_accuracy_mean')
            print(best_acc.to_string(index=False))
            best_configs['best_by_accuracy'] = best_acc

            # Save
            best_acc.to_csv(self.output_dir / "best_by_accuracy.csv", index=False)

        # Best by Brier score (lower is better)
        if 'test_brier_mean' in config_summary.columns:
            print("\n" + "-" * 80)
            print("TOP 10 CONFIGURATIONS BY BRIER SCORE (Lower is Better)")
            print("-" * 80)
            best_brier = config_summary.nsmallest(10, 'test_brier_mean')
            print(best_brier.to_string(index=False))
            best_configs['best_by_brier'] = best_brier

            # Save
            best_brier.to_csv(self.output_dir / "best_by_brier.csv", index=False)

        return best_configs

    def compare_strategies(self) -> pd.DataFrame:
        """Compare layer selection strategies"""

        if self.df is None or len(self.df) == 0:
            return None

        print("\n" + "=" * 80)
        print("Layer Selection Strategy Comparison")
        print("=" * 80)

        metrics = ['test_ece', 'test_accuracy', 'test_brier']

        strategy_comparison = self.df.groupby('layer_strategy').agg({
            'test_ece': ['mean', 'std', 'min'],
            'test_accuracy': ['mean', 'std', 'max'],
            'test_brier': ['mean', 'std', 'min'],
            'seed': 'count'
        })
        
        # Round only numeric columns
        for col in strategy_comparison.columns:
            if pd.api.types.is_numeric_dtype(strategy_comparison[col]):
                strategy_comparison[col] = strategy_comparison[col].round(6)

        print(strategy_comparison)

        # Save
        strategy_comparison.to_csv(self.output_dir / "strategy_comparison.csv")

        return strategy_comparison

    def compare_weighting_methods(self) -> pd.DataFrame:
        """Compare weighting methods"""

        if self.df is None or len(self.df) == 0:
            return None

        print("\n" + "=" * 80)
        print("Weighting Method Comparison")
        print("=" * 80)

        weighting_comparison = self.df.groupby('weighting_method').agg({
            'test_ece': ['mean', 'std', 'min'],
            'test_accuracy': ['mean', 'std', 'max'],
            'test_brier': ['mean', 'std', 'min'],
            'seed': 'count'
        })
        
        # Round only numeric columns
        for col in weighting_comparison.columns:
            if pd.api.types.is_numeric_dtype(weighting_comparison[col]):
                weighting_comparison[col] = weighting_comparison[col].round(6)

        print(weighting_comparison)

        # Save
        weighting_comparison.to_csv(self.output_dir / "weighting_comparison.csv")

        return weighting_comparison

    def plot_visualizations(self):
        """Generate visualization plots"""

        if self.df is None or len(self.df) == 0:
            return

        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80)

        # 1. ECE by layer strategy and weighting method
        if 'test_ece' in self.df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # ECE by layer strategy
            sns.boxplot(data=self.df, x='layer_strategy', y='test_ece', ax=axes[0, 0])
            axes[0, 0].set_title('ECE by Layer Selection Strategy')
            axes[0, 0].set_xlabel('Layer Selection Strategy')
            axes[0, 0].set_ylabel('ECE')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # ECE by weighting method
            sns.boxplot(data=self.df, x='weighting_method', y='test_ece', ax=axes[0, 1])
            axes[0, 1].set_title('ECE by Weighting Method')
            axes[0, 1].set_xlabel('Weighting Method')
            axes[0, 1].set_ylabel('ECE')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # Heatmap: strategy x weighting
            pivot_ece = self.df.pivot_table(
                values='test_ece',
                index='layer_strategy',
                columns='weighting_method',
                aggfunc='mean'
            )
            sns.heatmap(pivot_ece, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=axes[1, 0])
            axes[1, 0].set_title('Mean ECE: Strategy × Weighting')
            axes[1, 0].set_xlabel('Weighting Method')
            axes[1, 0].set_ylabel('Layer Strategy')

            # Accuracy vs ECE scatter
            if 'test_accuracy' in self.df.columns:
                for strategy in self.df['layer_strategy'].unique():
                    subset = self.df[self.df['layer_strategy'] == strategy]
                    axes[1, 1].scatter(subset['test_ece'], subset['test_accuracy'],
                                     label=strategy, alpha=0.6, s=50)
                axes[1, 1].set_xlabel('ECE')
                axes[1, 1].set_ylabel('Accuracy')
                axes[1, 1].set_title('Accuracy vs ECE by Strategy')
                axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.savefig(self.output_dir / "ece_analysis.png", dpi=300, bbox_inches='tight')
            print(f"Saved: {self.output_dir / 'ece_analysis.png'}")
            plt.close()

        # 2. Performance comparison across datasets and models
        if 'test_ece' in self.df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # By dataset
            if 'dataset' in self.df.columns:
                sns.boxplot(data=self.df, x='dataset', y='test_ece', hue='weighting_method', ax=axes[0])
                axes[0].set_title('ECE by Dataset and Weighting Method')
                axes[0].set_xlabel('Dataset')
                axes[0].set_ylabel('ECE')
                axes[0].legend(title='Weighting', bbox_to_anchor=(1.05, 1), loc='upper left')

            # By model
            if 'model' in self.df.columns:
                sns.boxplot(data=self.df, x='model', y='test_ece', hue='weighting_method', ax=axes[1])
                axes[1].set_title('ECE by Model and Weighting Method')
                axes[1].set_xlabel('Model')
                axes[1].set_ylabel('ECE')
                axes[1].tick_params(axis='x', rotation=45)
                axes[1].legend(title='Weighting', bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.savefig(self.output_dir / "dataset_model_comparison.png", dpi=300, bbox_inches='tight')
            print(f"Saved: {self.output_dir / 'dataset_model_comparison.png'}")
            plt.close()

        # 3. Detailed heatmaps for each dataset
        for dataset in self.df['dataset'].unique():
            dataset_df = self.df[self.df['dataset'] == dataset]

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # ECE heatmap
            pivot_ece = dataset_df.pivot_table(
                values='test_ece',
                index='layer_strategy',
                columns='weighting_method',
                aggfunc='mean'
            )
            sns.heatmap(pivot_ece, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=axes[0])
            axes[0].set_title(f'{dataset.upper()} - Mean ECE')

            # Accuracy heatmap
            if 'test_accuracy' in dataset_df.columns:
                pivot_acc = dataset_df.pivot_table(
                    values='test_accuracy',
                    index='layer_strategy',
                    columns='weighting_method',
                    aggfunc='mean'
                )
                sns.heatmap(pivot_acc, annot=True, fmt='.4f', cmap='RdYlGn', ax=axes[1])
                axes[1].set_title(f'{dataset.upper()} - Mean Accuracy')

            plt.tight_layout()
            plt.savefig(self.output_dir / f"heatmap_{dataset}.png", dpi=300, bbox_inches='tight')
            print(f"Saved: {self.output_dir / f'heatmap_{dataset}.png'}")
            plt.close()

        # 4. Baseline vs NC Ensemble comparison plots
        if 'method_type' in self.df.columns and len(self.df['method_type'].unique()) > 1:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # ECE comparison
            if 'test_ece' in self.df.columns:
                # Prepare data for baseline methods
                baseline_df = self.df[self.df['method_type'] == 'baseline']
                nc_df = self.df[self.df['method_type'] == 'nc_ensemble']

                if len(baseline_df) > 0 and len(nc_df) > 0:
                    # ECE boxplot
                    plot_data = []
                    labels = []

                    # Add baseline methods
                    if 'calibration_method' in baseline_df.columns:
                        for method in baseline_df['calibration_method'].unique():
                            method_data = baseline_df[baseline_df['calibration_method'] == method]['test_ece']
                            plot_data.append(method_data)
                            labels.append(f"B: {method}")

                    # Add top NC ensemble configurations
                    if 'layer_strategy' in nc_df.columns:
                        nc_summary = nc_df.groupby(['layer_strategy', 'weighting_method'])['test_ece'].mean().nsmallest(5)
                        for (strategy, weighting), _ in nc_summary.items():
                            config_data = nc_df[
                                (nc_df['layer_strategy'] == strategy) &
                                (nc_df['weighting_method'] == weighting)
                            ]['test_ece']
                            if len(config_data) > 0:
                                plot_data.append(config_data)
                                labels.append(f"NC: {strategy[:10]}")

                    bp = axes[0].boxplot(plot_data, labels=labels)
                    axes[0].set_title('ECE Comparison: Baseline vs NC Ensemble')
                    axes[0].set_ylabel('ECE')
                    axes[0].tick_params(axis='x', rotation=45)
                    axes[0].grid(True, alpha=0.3)

                    # Mean ECE bar plot
                    mean_data = []
                    mean_labels = []

                    # Baseline mean
                    if len(baseline_df) > 0:
                        mean_data.append(baseline_df['test_ece'].mean())
                        mean_labels.append('Baseline\n(mean)')

                    # NC ensemble mean
                    if len(nc_df) > 0:
                        mean_data.append(nc_df['test_ece'].mean())
                        mean_labels.append('NC Ensemble\n(mean)')

                    # Best of each
                    if len(baseline_df) > 0:
                        mean_data.append(baseline_df['test_ece'].min())
                        mean_labels.append('Baseline\n(best)')

                    if len(nc_df) > 0:
                        mean_data.append(nc_df['test_ece'].min())
                        mean_labels.append('NC Ensemble\n(best)')

                    bars = axes[1].bar(mean_labels, mean_data, color=['#ff7f0e', '#1f77b4', '#ff7f0e', '#1f77b4'], alpha=0.7)
                    axes[1].set_title('Mean and Best ECE Comparison')
                    axes[1].set_ylabel('ECE')
                    axes[1].grid(True, alpha=0.3, axis='y')

                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.4f}',
                                   ha='center', va='bottom', fontsize=9)

                    # Scatter plot: ECE vs Accuracy
                    if 'test_accuracy' in self.df.columns:
                        # Baseline points
                        if 'calibration_method' in baseline_df.columns:
                            for method in baseline_df['calibration_method'].unique():
                                method_data = baseline_df[baseline_df['calibration_method'] == method]
                                axes[2].scatter(method_data['test_ece'], method_data['test_accuracy'],
                                              label=f'B: {method}', alpha=0.6, s=50, marker='o')

                        # NC ensemble points (sample)
                        nc_sample = nc_df.sample(n=min(100, len(nc_df)))
                        axes[2].scatter(nc_sample['test_ece'], nc_sample['test_accuracy'],
                                      label='NC Ensemble', alpha=0.4, s=30, marker='^', color='blue')

                        axes[2].set_xlabel('ECE')
                        axes[2].set_ylabel('Accuracy')
                        axes[2].set_title('Accuracy vs ECE')
                        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                        axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / "baseline_vs_nc_comparison.png", dpi=300, bbox_inches='tight')
            print(f"Saved: {self.output_dir / 'baseline_vs_nc_comparison.png'}")
            plt.close()

    def combine_with_baselines(self):
        """Combine NC ensemble results with baseline results into a single DataFrame"""

        if len(self.baseline_results) == 0:
            print("\nNo baseline results to combine")
            return

        print("\n" + "=" * 80)
        print("Combining NC Ensemble and Baseline Results")
        print("=" * 80)

        # Create baseline DataFrame
        baseline_df = pd.DataFrame(self.baseline_results)

        # Combine with existing results
        if self.df is not None and len(self.df) > 0:
            # Ensure compatible columns - fill missing ones with None
            all_columns = set(self.df.columns).union(set(baseline_df.columns))

            # Fill missing columns in both DataFrames
            for col in all_columns:
                if col not in self.df.columns:
                    self.df[col] = None
                if col not in baseline_df.columns:
                    baseline_df[col] = None

            # Combine
            self.df = pd.concat([self.df, baseline_df], ignore_index=True)
        else:
            self.df = baseline_df

        print(f"Combined DataFrame shape: {self.df.shape}")
        print(f"Method types: {self.df['method_type'].value_counts().to_dict()}")

        # Filter out low accuracy models (< 0.6)
        if 'test_accuracy' in self.df.columns:
            before_count = len(self.df)
            self.df = self.df[self.df['test_accuracy'] >= 0.6]
            after_count = len(self.df)
            filtered_count = before_count - after_count
            if filtered_count > 0:
                print(f"\nFiltered out {filtered_count} results with accuracy < 0.6")
                print(f"Remaining results: {after_count}")

    def compare_with_baselines(self) -> Dict[str, pd.DataFrame]:
        """Compare NC ensemble methods against baseline calibration methods"""

        if self.df is None or len(self.df) == 0:
            return {}

        if 'method_type' not in self.df.columns:
            print("\nNo method_type column - cannot compare with baselines")
            return {}

        # Check if we have both types
        method_types = self.df['method_type'].unique()
        if len(method_types) < 2:
            print(f"\nOnly one method type found: {method_types}. Need both 'nc_ensemble' and 'baseline' for comparison.")
            return {}

        print("\n" + "=" * 80)
        print("Comparing NC Ensemble vs Baseline Calibration Methods")
        print("=" * 80)

        comparisons = {}

        # Prepare method names column
        # For baselines: use calibration_method
        # For NC ensemble: create a combined name from layer_strategy + weighting_method
        method_name_col = []
        for idx, row in self.df.iterrows():
            if row['method_type'] == 'baseline':
                method_name_col.append(row.get('calibration_method', 'unknown'))
            else:
                # NC ensemble - use best configuration for each dataset/model/training_loss
                method_name_col.append('NC_Ensemble')

        self.df['method_name'] = method_name_col

        # 1. Create pivot tables by dataset, model, training_loss
        print("\n" + "-" * 80)
        print("ECE COMPARISON BY DATASET/MODEL/TRAINING_LOSS")
        print("-" * 80)

        # Group by dataset, model, training_loss and find best method in each
        grouping_cols = ['dataset', 'model', 'training_loss']

        # For each combination, we want to show ECE for each method
        pivot_results = []

        for (dataset, model, training_loss), group in self.df.groupby(grouping_cols):
            if len(group) == 0:
                continue

            result_row = {
                'Dataset': dataset,
                'Model': model,
                'Training_Loss': training_loss,
            }

            # Get baseline methods
            baseline_group = group[group['method_type'] == 'baseline']
            for method_name in baseline_group['calibration_method'].unique():
                method_data = baseline_group[baseline_group['calibration_method'] == method_name]
                # Use mean ECE across seeds
                mean_ece = method_data['test_ece'].mean()
                result_row[method_name] = mean_ece

            # Get best NC ensemble configuration
            nc_group = group[group['method_type'] == 'nc_ensemble']
            if len(nc_group) > 0:
                # Find best NC ensemble by minimum ECE
                best_nc = nc_group.loc[nc_group['test_ece'].idxmin()]
                result_row['NC_Ensemble_Best'] = best_nc['test_ece']
                result_row['NC_Best_Config'] = f"{best_nc.get('layer_strategy', 'N/A')}_{best_nc.get('weighting_method', 'N/A')}"

                # Also include mean across all NC configurations
                result_row['NC_Ensemble_Mean'] = nc_group['test_ece'].mean()

            pivot_results.append(result_row)

        if pivot_results:
            comparison_df = pd.DataFrame(pivot_results)

            # Round numeric columns
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
            comparison_df[numeric_cols] = comparison_df[numeric_cols].round(4)

            print("\nECE Comparison Table:")
            print(comparison_df.to_string(index=False))

            # Save full comparison
            comparison_df.to_csv(self.output_dir / "ece_comparison_by_config.csv", index=False)
            comparisons['ece_by_config'] = comparison_df

            # 2. Create summary by dataset/model (averaged over training_loss)
            print("\n" + "-" * 80)
            print("ECE COMPARISON BY DATASET/MODEL (averaged over training_loss)")
            print("-" * 80)

            summary_by_dataset_model = []
            for (dataset, model), group_df in comparison_df.groupby(['Dataset', 'Model']):
                result_row = {
                    'Dataset': dataset,
                    'Model': model,
                }

                # Average each method's ECE across training losses
                for col in comparison_df.columns:
                    if col not in ['Dataset', 'Model', 'Training_Loss', 'NC_Best_Config']:
                        if col in group_df.columns:
                            mean_val = group_df[col].mean()
                            if pd.notna(mean_val):
                                result_row[col] = mean_val

                summary_by_dataset_model.append(result_row)

            if summary_by_dataset_model:
                summary_df = pd.DataFrame(summary_by_dataset_model)
                numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
                summary_df[numeric_cols] = summary_df[numeric_cols].round(4)

                print("\nSummary Table:")
                print(summary_df.to_string(index=False))

                summary_df.to_csv(self.output_dir / "ece_comparison_by_dataset_model.csv", index=False)
                comparisons['ece_by_dataset_model'] = summary_df

            # 3. Create improvement table (comparing NC_Ensemble_Best vs best baseline)
            print("\n" + "-" * 80)
            print("IMPROVEMENT ANALYSIS")
            print("-" * 80)

            improvement_results = []
            for idx, row in comparison_df.iterrows():
                # Find best baseline method
                baseline_cols = [col for col in row.index if col not in
                                ['Dataset', 'Model', 'Training_Loss', 'NC_Ensemble_Best',
                                 'NC_Best_Config', 'NC_Ensemble_Mean']]

                baseline_eces = {col: row[col] for col in baseline_cols if pd.notna(row.get(col))}

                if baseline_eces and 'NC_Ensemble_Best' in row and pd.notna(row['NC_Ensemble_Best']):
                    best_baseline_method = min(baseline_eces, key=baseline_eces.get)
                    best_baseline_ece = baseline_eces[best_baseline_method]
                    nc_ece = row['NC_Ensemble_Best']

                    improvement = ((best_baseline_ece - nc_ece) / best_baseline_ece) * 100

                    improvement_results.append({
                        'Dataset': row['Dataset'],
                        'Model': row['Model'],
                        'Training_Loss': row['Training_Loss'],
                        'Best_Baseline': best_baseline_method,
                        'Best_Baseline_ECE': best_baseline_ece,
                        'NC_Ensemble_ECE': nc_ece,
                        'NC_Config': row.get('NC_Best_Config', 'N/A'),
                        'Improvement_%': improvement,
                        'Winner': 'NC_Ensemble' if nc_ece < best_baseline_ece else 'Baseline'
                    })

            if improvement_results:
                improvement_df = pd.DataFrame(improvement_results)
                improvement_df = improvement_df.round(4)

                print("\nImprovement Analysis:")
                print(improvement_df.to_string(index=False))

                # Summary statistics
                print("\n" + "-" * 40)
                print("Summary Statistics:")
                print(f"NC Ensemble wins: {(improvement_df['Winner'] == 'NC_Ensemble').sum()}")
                print(f"Baseline wins: {(improvement_df['Winner'] == 'Baseline').sum()}")
                print(f"Mean improvement: {improvement_df['Improvement_%'].mean():.2f}%")
                print(f"Median improvement: {improvement_df['Improvement_%'].median():.2f}%")

                improvement_df.to_csv(self.output_dir / "improvement_analysis.csv", index=False)
                comparisons['improvement'] = improvement_df

        return comparisons

    def generate_full_report(self):
        """Generate a comprehensive markdown report"""

        if self.df is None or len(self.df) == 0:
            return

        print("\n" + "=" * 80)
        print("Generating Full Report")
        print("=" * 80)

        report_lines = []
        report_lines.append("# NC-Based Multi-Layer Ensemble Results Report\n")
        report_lines.append(f"**Generated:** {pd.Timestamp.now()}\n")
        report_lines.append(f"**Total Configurations:** {len(self.df)}\n")
        report_lines.append(f"**Datasets:** {', '.join(self.df['dataset'].unique())}\n")
        report_lines.append(f"**Models:** {', '.join(self.df['model'].unique())}\n")
        report_lines.append(f"**Seeds:** {sorted(self.df['seed'].unique())}\n")
        report_lines.append("\n---\n\n")

        # Overall statistics
        report_lines.append("## Overall Statistics\n\n")
        if 'test_ece' in self.df.columns:
            report_lines.append(f"- **Mean ECE:** {self.df['test_ece'].mean():.6f} ± {self.df['test_ece'].std():.6f}\n")
            report_lines.append(f"- **Min ECE:** {self.df['test_ece'].min():.6f}\n")
            report_lines.append(f"- **Max ECE:** {self.df['test_ece'].max():.6f}\n")
        if 'test_accuracy' in self.df.columns:
            report_lines.append(f"- **Mean Accuracy:** {self.df['test_accuracy'].mean():.4f} ± {self.df['test_accuracy'].std():.4f}\n")
        if 'test_brier' in self.df.columns:
            report_lines.append(f"- **Mean Brier:** {self.df['test_brier'].mean():.6f} ± {self.df['test_brier'].std():.6f}\n")
        report_lines.append("\n")

        # Best configurations
        report_lines.append("## Best Configurations\n\n")

        # Best ECE
        if 'test_ece' in self.df.columns:
            config_cols = ['dataset', 'model', 'layer_strategy', 'weighting_method']
            config_summary = self.df.groupby(config_cols).agg({
                'test_ece': ['mean', 'std'],
                'test_accuracy': ['mean', 'std'],
                'seed': 'count'
            }).reset_index()

            best_ece_row = config_summary.loc[config_summary[('test_ece', 'mean')].idxmin()]
            report_lines.append("### Best by ECE\n\n")
            report_lines.append(f"- **Dataset:** {best_ece_row[('dataset', '')]}\n")
            report_lines.append(f"- **Model:** {best_ece_row[('model', '')]}\n")
            report_lines.append(f"- **Layer Strategy:** {best_ece_row[('layer_strategy', '')]}\n")
            report_lines.append(f"- **Weighting Method:** {best_ece_row[('weighting_method', '')]}\n")
            report_lines.append(f"- **ECE:** {best_ece_row[('test_ece', 'mean')]:.6f} ± {best_ece_row[('test_ece', 'std')]:.6f}\n")
            report_lines.append(f"- **Accuracy:** {best_ece_row[('test_accuracy', 'mean')]:.4f} ± {best_ece_row[('test_accuracy', 'std')]:.4f}\n")
            report_lines.append("\n")

        # Strategy comparison
        report_lines.append("## Layer Selection Strategy Comparison\n\n")
        strategy_stats = self.df.groupby('layer_strategy').agg({
            'test_ece': ['mean', 'std'],
            'test_accuracy': ['mean', 'std'],
            'seed': 'count'
        }).round(6)
        report_lines.append(strategy_stats.to_markdown())
        report_lines.append("\n\n")

        # Weighting comparison
        report_lines.append("## Weighting Method Comparison\n\n")
        weighting_stats = self.df.groupby('weighting_method').agg({
            'test_ece': ['mean', 'std'],
            'test_accuracy': ['mean', 'std'],
            'seed': 'count'
        }).round(6)
        report_lines.append(weighting_stats.to_markdown())
        report_lines.append("\n\n")

        # Dataset-specific analysis
        report_lines.append("## Dataset-Specific Analysis\n\n")
        for dataset in sorted(self.df['dataset'].unique()):
            dataset_df = self.df[self.df['dataset'] == dataset]
            report_lines.append(f"### {dataset.upper()}\n\n")
            report_lines.append(f"- **Configurations:** {len(dataset_df)}\n")
            if 'test_ece' in dataset_df.columns:
                report_lines.append(f"- **Mean ECE:** {dataset_df['test_ece'].mean():.6f} ± {dataset_df['test_ece'].std():.6f}\n")
            if 'test_accuracy' in dataset_df.columns:
                report_lines.append(f"- **Mean Accuracy:** {dataset_df['test_accuracy'].mean():.4f} ± {dataset_df['test_accuracy'].std():.4f}\n")
            report_lines.append("\n")

        # Save report
        report_path = self.output_dir / "full_report.md"
        with open(report_path, 'w') as f:
            f.writelines(report_lines)

        print(f"Saved: {report_path}")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""

        print("\n" + "=" * 80)
        print("NC ENSEMBLE RESULTS ANALYSIS")
        print("=" * 80)
        print(f"Results Directory: {self.results_dir}")
        if self.baselines_dir:
            print(f"Baselines Directory: {self.baselines_dir}")
        print(f"Output Directory: {self.output_dir}")
        print("=" * 80)

        # Load NC ensemble results
        num_loaded = self.load_all_results()

        if num_loaded == 0:
            print("\nNo NC ensemble results found. Exiting.")
            return

        # Load baseline results if directory provided
        if self.baselines_dir:
            num_baselines = self.load_baseline_results()

            # Combine results
            if num_baselines > 0:
                self.combine_with_baselines()

        # Save raw data
        if self.df is not None:
            self.df.to_csv(self.output_dir / "all_results.csv", index=False)
            print(f"\nSaved raw data: {self.output_dir / 'all_results.csv'}")

        # Compute summaries
        self.compute_summary_statistics()

        # Analyze best configurations
        self.analyze_best_configurations()

        # Compare strategies and methods (NC ensemble only)
        self.compare_strategies()
        self.compare_weighting_methods()

        # Compare with baselines if we have them
        if self.baselines_dir and len(self.baseline_results) > 0:
            self.compare_with_baselines()

        # Generate visualizations
        self.plot_visualizations()

        # Generate full report
        self.generate_full_report()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"All outputs saved to: {self.output_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and analyze NC-based ensemble calibration results with baseline comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='nc_ensemble_results',
        help='Directory containing ensemble results (default: nc_ensemble_results)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='nc_ensemble_analysis',
        help='Output directory for analysis (default: nc_ensemble_analysis)'
    )

    parser.add_argument(
        '--baselines-dir',
        type=str,
        default=None,
        help='Directory containing baseline results (optional)'
    )

    args = parser.parse_args()

    # Create aggregator
    aggregator = NCEnsembleResultsAggregator(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        baselines_dir=args.baselines_dir
    )

    # Run analysis
    aggregator.run_full_analysis()


if __name__ == '__main__':
    main()
