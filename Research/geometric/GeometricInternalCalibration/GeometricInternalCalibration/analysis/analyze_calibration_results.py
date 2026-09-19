#!/usr/bin/env python3
"""
Statistical Analysis of Post-Hoc Calibration Results
Aggregates results across seeds and performs statistical significance testing
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_rel, friedmanchisquare
import warnings
warnings.filterwarnings('ignore')

class CalibrationResultsAnalyzer:
    """Comprehensive analysis of post-hoc calibration results"""
    
    def __init__(self, base_results_dir: str, accuracy_threshold: float = 0.5):
        """
        Initialize analyzer
        
        Args:
            base_results_dir: Base directory containing phase2 calibration results
            accuracy_threshold: Minimum accuracy to include results (default 50%)
        """
        self.base_dir = Path(base_results_dir)
        self.accuracy_threshold = accuracy_threshold
        self.results_df = None
        self.aggregated_stats = None
        
    def load_all_results(self):
        """Load all calibration result files and create master DataFrame"""
        all_results = []
        
        # Find all JSON result files
        json_files = list(self.base_dir.rglob("calibration_*.json"))
        print(f"📊 Found {len(json_files)} result files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract experiment info
                config = data['experiment_config']
                method = config['method']
                dataset = config['dataset']
                model = config['model']
                seed = config['seed']
                
                # Extract calibration results
                results = data['results']
                
                for calibration_method, metrics in results.items():
                    # Skip if error occurred or accuracy too low
                    if 'error' in metrics:
                        continue
                    
                    accuracy = metrics['accuracy']
                    if accuracy < self.accuracy_threshold * 100:  # Convert to percentage
                        print(f"⚠️  Skipping {method}|{dataset}|{model}|seed{seed}|{calibration_method}: "
                              f"Low accuracy ({accuracy:.1f}%)")
                        continue
                    
                    # Add to results list
                    all_results.append({
                        'training_method': method,
                        'dataset': dataset,
                        'model': model,
                        'seed': seed,
                        'calibration_method': calibration_method,
                        'accuracy': accuracy,
                        'ece': metrics['ece'],
                        'num_samples': metrics['num_samples']
                    })
                    
            except Exception as e:
                print(f"❌ Error loading {json_file}: {e}")
                continue
        
        self.results_df = pd.DataFrame(all_results)
        print(f"✅ Loaded {len(all_results)} valid results")
        
        return self.results_df
    
    def calculate_statistics(self):
        """Calculate descriptive statistics with confidence intervals"""
        if self.results_df is None:
            self.load_all_results()
        
        # Group by training method, dataset, model, and calibration method
        grouped = self.results_df.groupby([
            'training_method', 'dataset', 'model', 'calibration_method'
        ])
        
        stats_list = []
        
        for (training_method, dataset, model, cal_method), group in grouped:
            if len(group) < 2:  # Need at least 2 samples for CI
                continue
            
            # Calculate statistics for accuracy and ECE
            for metric in ['accuracy', 'ece']:
                values = group[metric].values
                n = len(values)
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                se_val = std_val / np.sqrt(n)
                
                # 95% confidence interval using t-distribution
                t_val = stats.t.ppf(0.975, df=n-1)
                ci_lower = mean_val - t_val * se_val
                ci_upper = mean_val + t_val * se_val
                
                stats_list.append({
                    'training_method': training_method,
                    'dataset': dataset,
                    'model': model,
                    'calibration_method': cal_method,
                    'metric': metric,
                    'n_seeds': n,
                    'mean': mean_val,
                    'std': std_val,
                    'se': se_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })
        
        self.aggregated_stats = pd.DataFrame(stats_list)
        return self.aggregated_stats
    
    def perform_significance_tests(self):
        """Perform statistical significance tests between calibration methods"""
        if self.results_df is None:
            self.load_all_results()
        
        significance_results = []
        
        # Group by training method, dataset, model
        for (training_method, dataset, model), group in self.results_df.groupby([
            'training_method', 'dataset', 'model'
        ]):
            
            # Get data for each calibration method
            cal_methods = group['calibration_method'].unique()
            
            if len(cal_methods) < 2:
                continue
            
            # Prepare data for pairwise comparisons
            method_data = {}
            for cal_method in cal_methods:
                method_group = group[group['calibration_method'] == cal_method]
                if len(method_group) >= 2:  # Need at least 2 seeds
                    method_data[cal_method] = {
                        'accuracy': method_group['accuracy'].values,
                        'ece': method_group['ece'].values
                    }
            
            # Pairwise t-tests for ECE (lower is better)
            methods = list(method_data.keys())
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    if method1 in method_data and method2 in method_data:
                        # Test for ECE (paired t-test if same seeds)
                        ece1 = method_data[method1]['ece']
                        ece2 = method_data[method2]['ece']
                        
                        if len(ece1) == len(ece2):
                            # Paired t-test
                            t_stat, p_val = stats.ttest_rel(ece1, ece2)
                            test_type = 'paired'
                        else:
                            # Independent t-test
                            t_stat, p_val = stats.ttest_ind(ece1, ece2)
                            test_type = 'independent'
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(ece1)-1)*np.var(ece1, ddof=1) + 
                                            (len(ece2)-1)*np.var(ece2, ddof=1)) / 
                                           (len(ece1) + len(ece2) - 2))
                        cohens_d = (np.mean(ece1) - np.mean(ece2)) / pooled_std
                        
                        significance_results.append({
                            'training_method': training_method,
                            'dataset': dataset,
                            'model': model,
                            'method1': method1,
                            'method2': method2,
                            'metric': 'ece',
                            'test_type': test_type,
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'cohens_d': cohens_d,
                            'method1_mean': np.mean(ece1),
                            'method2_mean': np.mean(ece2),
                            'significant': p_val < 0.05
                        })
        
        return pd.DataFrame(significance_results)
    
    def create_summary_table(self):
        """Create publication-ready summary table"""
        if self.aggregated_stats is None:
            self.calculate_statistics()
        
        # Filter for ECE results
        ece_stats = self.aggregated_stats[self.aggregated_stats['metric'] == 'ece'].copy()
        
        # Create formatted results string
        ece_stats['result_str'] = ece_stats.apply(
            lambda row: f"{row['mean']:.4f} ± {1.96*row['se']:.4f}", axis=1
        )
        
        # Pivot table for nice display
        pivot_table = ece_stats.pivot_table(
            index=['training_method', 'dataset', 'model'],
            columns='calibration_method',
            values='result_str',
            aggfunc='first'
        )
        
        return pivot_table
    
    def create_visualizations(self, save_dir: str = "./analysis_plots"):
        """Create comprehensive visualizations"""
        if self.results_df is None:
            self.load_all_results()
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. ECE Comparison Box Plot
        plt.figure(figsize=(14, 8))
        
        # Create subplot for each training method
        training_methods = self.results_df['training_method'].unique()
        n_methods = len(training_methods)
        
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 6), sharey=True)
        if n_methods == 1:
            axes = [axes]
        
        for i, method in enumerate(training_methods):
            method_data = self.results_df[self.results_df['training_method'] == method]
            
            sns.boxplot(
                data=method_data,
                x='calibration_method',
                y='ece',
                ax=axes[i]
            )
            
            axes[i].set_title(f'{method}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Calibration Method')
            if i == 0:
                axes[i].set_ylabel('Expected Calibration Error (ECE)')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add mean markers
            for j, cal_method in enumerate(method_data['calibration_method'].unique()):
                cal_data = method_data[method_data['calibration_method'] == cal_method]
                mean_ece = cal_data['ece'].mean()
                axes[i].scatter(j, mean_ece, color='red', s=100, marker='D', zorder=5)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/ece_comparison_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ECE vs Accuracy Scatter Plot
        plt.figure(figsize=(12, 8))
        
        for cal_method in self.results_df['calibration_method'].unique():
            method_data = self.results_df[self.results_df['calibration_method'] == cal_method]
            plt.scatter(
                method_data['accuracy'],
                method_data['ece'],
                label=cal_method,
                alpha=0.7,
                s=60
            )
        
        plt.xlabel('Accuracy (%)')
        plt.ylabel('Expected Calibration Error (ECE)')
        plt.title('ECE vs Accuracy: Post-Hoc Calibration Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/ece_vs_accuracy_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Mean ECE Bar Plot with Error Bars
        if self.aggregated_stats is not None:
            ece_stats = self.aggregated_stats[self.aggregated_stats['metric'] == 'ece']
            
            plt.figure(figsize=(14, 8))
            
            # Group by training method and calibration method
            for i, training_method in enumerate(ece_stats['training_method'].unique()):
                method_stats = ece_stats[ece_stats['training_method'] == training_method]
                
                x_positions = np.arange(len(method_stats)) + i * 0.8
                
                plt.bar(
                    x_positions,
                    method_stats['mean'],
                    yerr=1.96 * method_stats['se'],  # 95% CI
                    label=training_method,
                    alpha=0.7,
                    capsize=5
                )
            
            plt.xlabel('Calibration Method')
            plt.ylabel('Expected Calibration Error (ECE)')
            plt.title('Mean ECE by Training and Calibration Method (95% CI)')
            plt.legend()
            plt.xticks(
                range(len(ece_stats['calibration_method'].unique())),
                ece_stats['calibration_method'].unique(),
                rotation=45
            )
            plt.tight_layout()
            plt.savefig(f"{save_dir}/mean_ece_barplot.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Visualizations saved to {save_dir}/")
    
    def generate_report(self, output_file: str = "calibration_analysis_report.md"):
        """Generate comprehensive analysis report"""
        if self.results_df is None:
            self.load_all_results()
        
        if self.aggregated_stats is None:
            self.calculate_statistics()
        
        significance_df = self.perform_significance_tests()
        summary_table = self.create_summary_table()
        
        # Generate report
        report_lines = [
            "# Post-Hoc Calibration Analysis Report",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"- **Total experiments analyzed:** {len(self.results_df)}",
            f"- **Training methods:** {', '.join(self.results_df['training_method'].unique())}",
            f"- **Datasets:** {', '.join(self.results_df['dataset'].unique())}",
            f"- **Models:** {', '.join(self.results_df['model'].unique())}",
            f"- **Seeds analyzed:** {', '.join(map(str, sorted(self.results_df['seed'].unique())))}",
            f"- **Accuracy threshold:** {self.accuracy_threshold*100:.1f}%",
            "",
            "## Statistical Summary",
            "",
            "### Mean ECE ± 95% Confidence Interval",
            "```",
            str(summary_table),
            "```",
            "",
            "## Statistical Significance Tests",
            "",
            "### Pairwise Comparisons (ECE - Lower is Better)",
        ]
        
        # Add significance test results
        for _, row in significance_df.iterrows():
            significance = "✅ SIGNIFICANT" if row['significant'] else "❌ Not significant"
            effect_size = "Large" if abs(row['cohens_d']) > 0.8 else "Medium" if abs(row['cohens_d']) > 0.5 else "Small"
            
            report_lines.extend([
                f"**{row['training_method']} | {row['dataset']} | {row['model']}**",
                f"- {row['method1']} vs {row['method2']}",
                f"- Mean ECE: {row['method1_mean']:.4f} vs {row['method2_mean']:.4f}",
                f"- t-statistic: {row['t_statistic']:.3f}, p-value: {row['p_value']:.4f}",
                f"- Effect size (Cohen's d): {row['cohens_d']:.3f} ({effect_size})",
                f"- Result: {significance}",
                ""
            ])
        
        # Best performing methods
        report_lines.extend([
            "",
            "## Best Performing Methods",
            "",
            "### By Lowest Mean ECE:",
        ])
        
        ece_stats = self.aggregated_stats[self.aggregated_stats['metric'] == 'ece']
        best_methods = ece_stats.nsmallest(5, 'mean')[['training_method', 'calibration_method', 'mean', 'ci_lower', 'ci_upper']]
        
        for _, row in best_methods.iterrows():
            report_lines.append(
                f"1. **{row['training_method']} + {row['calibration_method']}**: "
                f"ECE = {row['mean']:.4f} (95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}])"
            )
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Analysis report saved to {output_file}")
        
        return {
            'summary_table': summary_table,
            'significance_tests': significance_df,
            'best_methods': best_methods
        }

def main():
    """Main analysis function"""
    
    # Initialize analyzer
    analyzer = CalibrationResultsAnalyzer(
        base_results_dir="/home/ptamar/geometric-internal-calibration/aaai_full_experiments/phase2_calibration/results",
        accuracy_threshold=0.5  # 50% accuracy threshold
    )
    
    # Load and analyze results
    print("🔍 Loading calibration results...")
    df = analyzer.load_all_results()
    
    print("📊 Calculating statistics...")
    stats_df = analyzer.calculate_statistics()
    
    print("🔬 Performing significance tests...")
    significance_df = analyzer.perform_significance_tests()
    
    print("📈 Creating visualizations...")
    analyzer.create_visualizations("./analysis/calibration_analysis_plots")
    
    print("📄 Generating comprehensive report...")
    results = analyzer.generate_report("analysis/calibration_analysis_report.md")
    
    # Print quick summary
    print("\n" + "="*60)
    print("🏆 QUICK SUMMARY")
    print("="*60)
    
    # Show best methods
    ece_stats = stats_df[stats_df['metric'] == 'ece']
    best_overall = ece_stats.loc[ece_stats['mean'].idxmin()]
    
    print(f"🥇 Best Overall: {best_overall['training_method']} + {best_overall['calibration_method']}")
    print(f"   ECE: {best_overall['mean']:.4f} ± {1.96*best_overall['se']:.4f}")
    
    # Count significant improvements
    sig_geometric = significance_df[
        (significance_df['method2'] == 'Geometric') & 
        (significance_df['significant'] == True) &
        (significance_df['method1_mean'] > significance_df['method2_mean'])  # Geometric is better
    ]
    
    print(f"🔬 Geometric calibration significantly outperformed others in {len(sig_geometric)} comparisons")
    
    print("\n📊 Full results saved to:")
    print("   - analysis/calibration_analysis_report.md")
    print("   - analysis/calibration_analysis_plots/")

if __name__ == "__main__":
    main() 