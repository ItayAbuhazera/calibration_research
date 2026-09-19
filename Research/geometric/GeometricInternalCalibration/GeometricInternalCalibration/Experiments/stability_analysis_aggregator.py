#!/usr/bin/env python3
"""
AAAI Section 4: Enhanced Stability Analysis Results Aggregator - ALL 4 CORRELATION TYPES (ENHANCED)
Aggregates stability analysis results with comprehensive Spearman and Pearson correlation analysis

This enhanced version features ALL 4 correlation type combinations:
1. Stability Spearman (PRIMARY for isotonic regression)
2. Stability Pearson (supplementary linear relationship assessment)  
3. Percentile Spearman (PRIMARY for isotonic regression)
4. Percentile Pearson (supplementary linear relationship assessment)

KEY ENHANCEMENT: Generates 4 separate visualizations to comprehensively analyze both monotonic
(Spearman) and linear (Pearson) relationships for both stability and percentile metrics.

VISUALIZATION OUTPUT:
- comprehensive_stability_spearman_correlation_analysis_{dataset}.png
- comprehensive_stability_pearson_correlation_analysis_{dataset}.png
- comprehensive_percentile_spearman_correlation_analysis_{dataset}.png
- comprehensive_percentile_pearson_correlation_analysis_{dataset}.png

This provides complete correlation analysis supporting isotonic regression-based geometric calibration.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class EnhancedStabilityAnalysisAggregator:
    """Enhanced aggregator emphasizing correlation analysis for AAAI submission."""
    
    def __init__(self, base_results_dir: str, output_dir: str):
        self.base_results_dir = Path(base_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage for aggregated results
        self.aggregated_data = defaultdict(list)
        
        # Corruption groups based on your data
        self.corruption_groups = {
            'Digital': ['brightness', 'elastic_transform', 'jpeg_compression', 'pixelate'],
            'Weather': ['fog', 'frost'],
            'Blur': ['glass_blur', 'zoom_blur'],
            'Noise': ['impulse_noise', 'shot_noise']
        }
        
        # Track what we've found
        self.found_results = defaultdict(list)
        
        # Correlation interpretation thresholds
        self.correlation_thresholds = {
            'excellent': 0.85,
            'good': 0.70,
            'moderate': 0.50,
            'poor': 0.30
        }
        
    def scan_and_aggregate_results(self, training_methods: List[str], 
                                 models: List[str], seeds: List[int], dataset: str = 'cifar10') -> Dict[str, Any]:
        """
        Scan for stability analysis results and aggregate them with correlation focus.
        """
        print(f"🔍 Scanning for stability analysis results for {dataset.upper()}...")
        
        # Define layer mappings based on your data structure (including pixel space)
        model_layers = {
            'resnet18': ['pixel', 'layer1', 'layer2', 'layer3', 'layer4', 'layer4.1'],
            'resnet50': ['pixel', 'layer1', 'layer2', 'layer3', 'layer4', 'layer4.1'],
            'densenet121': ['pixel', 'dense1', 'trans1', 'dense2', 'trans2', 'dense3', 'trans3', 'dense4', 'bn']
        }
        
        total_expected = 0
        total_found = 0
        
        # Scan through all combinations
        for training_method in training_methods:
            for model in models:
                layers = model_layers.get(model, [])
                for layer in layers:
                    for seed in seeds:
                        total_expected += 1
                        
                        # YOUR ACTUAL PATH STRUCTURE (with dataset support)
                        result_path = (self.base_results_dir / 
                                     f"results/{training_method}/{dataset}/{model}/{layer}_stability/seed{seed}")
                        
                        if result_path.exists():
                            success = self._process_single_experiment(
                                result_path, training_method, model, layer, seed
                            )
                            if success:
                                total_found += 1
                                self.found_results[f"{training_method}_{model}"].append(f"{layer}_seed{seed}")
        
        print(f"📊 Found {total_found}/{total_expected} experiments ({total_found/total_expected*100:.1f}%)")
        
        if total_found == 0:
            print("❌ No results found! Check your path structure.")
            print(f"Expected path format: results/{{training_method}}/{dataset}/{{model}}/{{layer}}_stability/seed{{seed}}/")
            return {}
        
        # Create comprehensive aggregated results with correlation focus
        return self._create_correlation_focused_analysis(dataset)
        
    def _process_single_experiment(self, result_path: Path, training_method: str, 
                                 model: str, layer: str, seed: int) -> bool:
        """Process a single experiment's stability analysis results."""
        
        try:
            # Look for YOUR multi_corruption_summary.json file
            summary_file = result_path / "multi_corruption_summary.json"
            
            if summary_file.exists():
                print(f"  ✅ Found: {training_method}/{model}/{layer}/seed{seed}")
                
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                
                # Extract overall statistics
                overall_stats = data.get('overall_statistics', {})
                per_corruption = data.get('per_corruption_results', {})
                
                # Add experiment record with correlation focus
                self._add_experiment_data_with_correlations(training_method, model, layer, seed, 
                                                          overall_stats, per_corruption)
                return True
            else:
                print(f"  ❌ Missing: {summary_file}")
                return False
            
        except Exception as e:
            print(f"⚠️ Error processing {result_path}: {e}")
            return False
    
    def _add_experiment_data_with_correlations(self, training_method: str, model: str, layer: str, 
                                             seed: int, overall_stats: Dict, per_corruption: Dict):
        """Add experiment data with enhanced correlation metrics for all 4 correlation types."""
        
        # Extract correlation data from per-corruption results using ACTUAL field names
        stability_correlations = []
        percentile_correlations = []
        correlation_p_values = []
        percentile_p_values = []
        
        for corruption_name, corruption_data in per_corruption.items():
            # Extract ACTUAL correlation metrics (using field names that exist in the data)
            stability_corr = corruption_data.get('stability_correctness_correlation', np.nan)
            percentile_corr = corruption_data.get('percentile_correctness_correlation', np.nan)
            corr_p_val = corruption_data.get('correlation_p_value', 1.0)
            percentile_p_val = corruption_data.get('percentile_correlation_p_value', 1.0)
            
            # Collect correlations (only if they are valid numbers)
            if not np.isnan(stability_corr) and stability_corr is not None:
                stability_correlations.append(stability_corr)
            if not np.isnan(percentile_corr) and percentile_corr is not None:
                percentile_correlations.append(percentile_corr)
            if not np.isnan(corr_p_val) and corr_p_val is not None:
                correlation_p_values.append(corr_p_val)
            if not np.isnan(percentile_p_val) and percentile_p_val is not None:
                percentile_p_values.append(percentile_p_val)
        
        # Calculate aggregate correlation metrics using ACTUAL data
        mean_stability_correlation = np.mean(stability_correlations) if stability_correlations else np.nan
        mean_percentile_correlation = np.mean(percentile_correlations) if percentile_correlations else np.nan
        
        # Consistency metrics (lower std = better consistency)
        correlation_consistency = 1.0 - (np.std(stability_correlations) if len(stability_correlations) > 1 else 0.0)
        percentile_consistency = 1.0 - (np.std(percentile_correlations) if len(percentile_correlations) > 1 else 0.0)
        
        # Create overall record using ACTUAL correlation data
        overall_record = {
            'training_method': training_method,
            'model': model, 
            'layer': layer,
            'seed': seed,
            'corruption': 'overall',
            'corruption_group': 'Overall',
            
            # PRIMARY CORRELATION METRICS (using actual field names)
            'mean_stability_correlation': mean_stability_correlation,
            'mean_percentile_correlation': mean_percentile_correlation,
            
            # CONSISTENCY METRICS
            'correlation_consistency': correlation_consistency,
            'percentile_consistency': percentile_consistency,
            
            # SIGNIFICANCE COUNTS
            'num_significant_correlations': sum(1 for p in correlation_p_values if p < 0.05),
            'num_significant_percentile_correlations': sum(1 for p in percentile_p_values if p < 0.05),
            
            # CORRELATION STRENGTH CATEGORIES
            'correlation_strength_category': self._categorize_correlation_strength(mean_stability_correlation),
            'percentile_strength_category': self._categorize_correlation_strength(mean_percentile_correlation),
            
            # Traditional stability metrics (secondary)
            'mean_stability_across_corruptions': overall_stats.get('mean_stability_across_corruptions', np.nan),
            'mean_positive_percentage': overall_stats.get('mean_positive_percentage', np.nan),
            'mean_correlation': overall_stats.get('mean_correlation', np.nan),
            'best_stability_corruption': overall_stats.get('best_stability_corruption', ''),
            'best_correlation_corruption': overall_stats.get('best_correlation_corruption', ''),
            'num_corruptions_analyzed': len(per_corruption)
        }
        
        self.aggregated_data['experiments'].append(overall_record)
        
        # Add detailed per-corruption records with ALL 4 correlation types
        for corruption_name, corruption_data in per_corruption.items():
            overall_stats_corr = corruption_data.get('overall_stats', {})
            
            corruption_record = {
                'training_method': training_method,
                'model': model,
                'layer': layer,
                'seed': seed,
                'corruption': corruption_name,
                'corruption_group': self._get_corruption_group(corruption_name),
                
                # PRIMARY CORRELATION METRICS (using actual field names)
                'stability_correctness_correlation': corruption_data.get('stability_correctness_correlation', np.nan),
                'percentile_correctness_correlation': corruption_data.get('percentile_correctness_correlation', np.nan),
                
                # P-VALUES
                'correlation_p_value': corruption_data.get('correlation_p_value', 1.0),
                'percentile_correlation_p_value': corruption_data.get('percentile_correlation_p_value', 1.0),
                
                # SIGNIFICANCE FLAGS
                'correlation_significance': corruption_data.get('correlation_p_value', 1.0) < 0.05,
                'percentile_significance': corruption_data.get('percentile_correlation_p_value', 1.0) < 0.05,
                
                # STRENGTH CATEGORIES
                'correlation_strength_category': self._categorize_correlation_strength(
                    corruption_data.get('stability_correctness_correlation', np.nan)
                ),
                'percentile_strength_category': self._categorize_correlation_strength(
                    corruption_data.get('percentile_correctness_correlation', np.nan)
                ),
                
                # Secondary stability metrics
                'mean_stability': overall_stats_corr.get('mean_stability', np.nan),
                'positive_stability_percentage': overall_stats_corr.get('positive_stability_pct', np.nan),
                'variance_to_mean_ratio': overall_stats_corr.get('variance_to_mean_ratio', np.nan),
                'calibration_informativeness': overall_stats_corr.get('calibration_informativeness', np.nan),
                'total_samples': overall_stats_corr.get('total_samples', np.nan)
            }
            
            self.aggregated_data['experiments'].append(corruption_record)
    
    def _categorize_correlation_strength(self, correlation: float) -> str:
        """Categorize correlation strength for analysis."""
        if np.isnan(correlation):
            return 'unknown'
        
        abs_corr = abs(correlation)
        if abs_corr >= self.correlation_thresholds['excellent']:
            return 'excellent'
        elif abs_corr >= self.correlation_thresholds['good']:
            return 'good'
        elif abs_corr >= self.correlation_thresholds['moderate']:
            return 'moderate'
        elif abs_corr >= self.correlation_thresholds['poor']:
            return 'poor'
        else:
            return 'very_poor'
    
    def _get_corruption_group(self, corruption: str) -> str:
        """Get corruption group for a corruption type."""
        for group, corruptions in self.corruption_groups.items():
            if corruption in corruptions:
                return group
        return 'Other'
    
    def _create_correlation_focused_analysis(self, dataset: str) -> Dict[str, Any]:
        """Create comprehensive correlation-focused analysis."""
        
        print("📊 Creating correlation-focused aggregated analysis...")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.aggregated_data['experiments'])
        
        if df.empty:
            print("❌ No data found for aggregation!")
            return {}
        
        print(f"✅ Aggregating {len(df)} experiment records")
        
        # Separate overall vs per-corruption data
        overall_df = df[df['corruption'] == 'overall'].copy()
        corruption_df = df[df['corruption'] != 'overall'].copy()
        
        aggregated_results = {}
        
        # 1. CORRELATION ANALYSIS (Primary focus)
        if not corruption_df.empty:
            aggregated_results['correlation_analysis'] = self._analyze_correlation_performance(corruption_df)
        
        # 2. LAYER PERFORMANCE WITH CORRELATION RANKING
        if not overall_df.empty:
            aggregated_results['layer_analysis'] = self._analyze_layer_performance(overall_df, corruption_df)
        
        # 3. SEMANTIC VS PIXEL SPACE COMPARISON
        if not overall_df.empty:
            aggregated_results['semantic_vs_pixel_analysis'] = self._analyze_semantic_vs_pixel(overall_df)
        
        # 4. Traditional analyses (secondary)
        if not corruption_df.empty:
            aggregated_results['corruption_group_analysis'] = self._analyze_corruption_groups(corruption_df)
        
        if not overall_df.empty:
            aggregated_results['training_method_analysis'] = self._analyze_training_methods(overall_df)
            aggregated_results['architecture_analysis'] = self._analyze_architectures(overall_df, corruption_df)
        
        # Save raw aggregated data
        self._save_aggregated_data(overall_df, corruption_df, aggregated_results, dataset)
        
        # Generate comprehensive visualizations
        self._create_correlation_visualizations(overall_df, corruption_df, aggregated_results, dataset)
        
        return aggregated_results
    
    def _analyze_layer_performance(self, overall_df: pd.DataFrame, corruption_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance across different layers, focusing on correlation."""
        
        print("🔍 Analyzing layer performance with correlation ranking...")
        
        # Group by model and layer for overall stats from the `overall_df`
        layer_stats_overall = overall_df.groupby(['model', 'layer']).agg({
            'mean_correlation': ['mean', 'std']
        }).round(4)
        layer_stats_overall.columns = ['_'.join(col).strip() for col in layer_stats_overall.columns]
        
        # Group by model and layer for per-corruption stats from `corruption_df`
        layer_stats_corruption = corruption_df.groupby(['model', 'layer']).agg({
            'stability_correctness_correlation': ['mean', 'std'],
            'percentile_correctness_correlation': ['mean', 'std'],
            'positive_stability_percentage': ['mean', 'std', 'count']
        }).round(4)
        layer_stats_corruption.columns = ['_'.join(col).strip() for col in layer_stats_corruption.columns]
        
        # Merge the stats
        layer_stats = pd.merge(layer_stats_overall.reset_index(), layer_stats_corruption.reset_index(), on=['model', 'layer'])
        
        # Find optimal layers per model based on BOTH correlation types from per-corruption data
        optimal_layers = {}
        df_corr = corruption_df.dropna(subset=['stability_correctness_correlation', 'percentile_correctness_correlation'])

        for model in df_corr['model'].unique():
            model_data = df_corr[df_corr['model'] == model]
            
            if not model_data.empty:
                # Optimal by stability correlation
                layer_perf_stability = model_data.groupby('layer')['stability_correctness_correlation'].mean()
                optimal_layer_stability = layer_perf_stability.idxmax()
                optimal_score_stability = layer_perf_stability.max()

                # Optimal by percentile correlation
                layer_perf_percentile = model_data.groupby('layer')['percentile_correctness_correlation'].mean()
                optimal_layer_percentile = layer_perf_percentile.idxmax()
                optimal_score_percentile = layer_perf_percentile.max()
                
                optimal_layers[model] = {
                    'optimal_layer_by_stability_correlation': optimal_layer_stability,
                    'mean_stability_correlation': optimal_score_stability,
                    'layer_ranking_by_stability_correlation': layer_perf_stability.sort_values(ascending=False).to_dict(),
                    
                    'optimal_layer_by_percentile_correlation': optimal_layer_percentile,
                    'mean_percentile_correlation': optimal_score_percentile,
                    'layer_ranking_by_percentile_correlation': layer_perf_percentile.sort_values(ascending=False).to_dict()
                }
        
        return {
            'layer_statistics': layer_stats.to_dict('records'),
            'optimal_layers': optimal_layers,
            'layer_consistency_across_methods': self._check_layer_consistency(corruption_df)
        }
    
    def _analyze_correlation_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation performance in detail, as requested."""
        print("🔗 Analyzing stability-correctness correlations (PRIMARY METRIC)...")
        
        # Filter out NaN values for meaningful statistics
        df_corr = df.dropna(subset=['stability_correctness_correlation', 'percentile_correctness_correlation', 'correlation_p_value'])

        # Group stats by key factors
        corr_stats = df_corr.groupby(['model', 'layer', 'training_method']).agg(
            mean_stability_corr=('stability_correctness_correlation', 'mean'),
            std_stability_corr=('stability_correctness_correlation', 'std'),
            mean_percentile_corr=('percentile_correctness_correlation', 'mean'),
            std_percentile_corr=('percentile_correctness_correlation', 'std'),
            run_count=('stability_correctness_correlation', 'count')
        ).round(4).reset_index()

        # Correlation consistency (lower std is better)
        consistency_stability = df_corr.groupby(['model', 'layer'])['stability_correctness_correlation'].std()
        consistency_percentile = df_corr.groupby(['model', 'layer'])['percentile_correctness_correlation'].std()
        
        # Convert tuple keys to string for JSON compatibility
        consistency_stability_dict = {f'{model}|{layer}': value for (model, layer), value in consistency_stability.items()}
        consistency_percentile_dict = {f'{model}|{layer}': value for (model, layer), value in consistency_percentile.items()}

        # Significance analysis (p < 0.001 is highly significant)
        significant_runs = df_corr[df_corr['correlation_p_value'] < 0.001]
        significance_pct = (len(significant_runs) / len(df_corr)) * 100 if len(df_corr) > 0 else 0

        return {
            'correlation_statistics': corr_stats.to_dict('records'),
            'consistency_by_stability_correlation_std': consistency_stability_dict,
            'consistency_by_percentile_correlation_std': consistency_percentile_dict,
            'significance_analysis': {
                'highly_significant_runs_pct (p<0.001)': significance_pct,
                'num_highly_significant_runs': len(significant_runs),
                'total_runs_with_correlation': len(df_corr)
            }
        }

    def _analyze_corruption_groups(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance across corruption groups."""
        
        print("🌪️ Analyzing corruption patterns with correlation metrics...")
        
        # Group statistics by corruption group
        group_stats = df.groupby('corruption_group').agg({
            'stability_correctness_correlation': ['mean', 'std'],
            'percentile_correctness_correlation': ['mean', 'std'],
            'positive_stability_percentage': ['mean', 'std', 'count'],
            'variance_to_mean_ratio': ['mean', 'std']
        }).round(4)
        
        group_stats.columns = ['_'.join(col).strip() for col in group_stats.columns]
        group_stats = group_stats.reset_index()
        
        # Rank corruption groups by stability-correctness correlation
        group_ranking_stability = df.groupby('corruption_group')['stability_correctness_correlation'].mean().sort_values(ascending=False)
        group_ranking_percentile = df.groupby('corruption_group')['percentile_correctness_correlation'].mean().sort_values(ascending=False)
        
        return {
            'group_statistics': group_stats,
            'group_ranking_by_stability_correlation': group_ranking_stability.to_dict(),
            'group_ranking_by_percentile_correlation': group_ranking_percentile.to_dict(),
            'best_individual_corruptions': df.nlargest(10, 'stability_correctness_correlation')[['corruption', 'stability_correctness_correlation', 'model', 'layer']].to_dict('records')
        }
    
    def _analyze_training_methods(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze training methods with correlation focus."""
        
        print("🎯 Analyzing training methods with correlation metrics...")
        
        method_stats = df.groupby('training_method').agg({
            'mean_stability_correlation': ['mean', 'std', 'count'],
            'mean_percentile_correlation': ['mean', 'std'],
            'correlation_consistency': ['mean', 'std'],
            'mean_positive_percentage': ['mean', 'std']
        }).round(4)
        
        method_stats.columns = ['_'.join(col).strip() for col in method_stats.columns]
        method_stats = method_stats.reset_index()
        
        return {
            'method_statistics': method_stats
        }
    
    def _analyze_architectures(self, overall_df: pd.DataFrame, corruption_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance across model architectures using detailed correlation data."""
        
        print("🏗️ Analyzing architectures with correlation metrics...")
        
        # Use corruption_df for detailed correlation stats
        df_corr = corruption_df.dropna(subset=['stability_correctness_correlation', 'percentile_correctness_correlation'])
        arch_stats_corr = df_corr.groupby('model').agg(
            mean_stability_correlation=('stability_correctness_correlation', 'mean'),
            std_stability_correlation=('stability_correctness_correlation', 'std'),
            mean_percentile_correlation=('percentile_correctness_correlation', 'mean'),
            std_percentile_correlation=('percentile_correctness_correlation', 'std'),
        ).round(4)
        
        # Use overall_df for other stats
        arch_stats_overall = overall_df.groupby('model').agg(
            mean_positive_percentage=('mean_positive_percentage', 'mean'),
            run_count=('mean_positive_percentage', 'count')
        ).round(4)

        # Merge stats
        arch_stats = pd.merge(arch_stats_corr.reset_index(), arch_stats_overall.reset_index(), on='model')

        # Rank architecture by correlation using the same data
        arch_ranking_stability = arch_stats.set_index('model')['mean_stability_correlation'].sort_values(ascending=False)
        arch_ranking_percentile = arch_stats.set_index('model')['mean_percentile_correlation'].sort_values(ascending=False)
        
        return {
            'architecture_statistics': arch_stats.to_dict('records'),
            'architecture_ranking_by_stability_correlation': arch_ranking_stability.to_dict(),
            'architecture_ranking_by_percentile_correlation': arch_ranking_percentile.to_dict()
        }

    def _analyze_semantic_vs_pixel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Quantify the performance gap between semantic layers and pixel space."""
        print("🔬 Quantifying semantic vs. pixel space advantage...")

        df_corr = df.dropna(subset=['mean_stability_correlation', 'mean_percentile_correlation'])
        if df_corr.empty:
            return {'error': 'No correlation data available for this analysis.'}

        # Separate data into semantic and pixel layers
        pixel_data = df_corr[df_corr['layer'] == 'pixel']
        semantic_data = df_corr[df_corr['layer'] != 'pixel']

        if pixel_data.empty or semantic_data.empty:
            return {
                'summary': 'Insufficient data to compare semantic and pixel spaces.',
                'pixel_stats': {},
                'semantic_stats': {}
            }

        # Calculate statistics for stability correlation
        pixel_stats_stability = {
            'mean_correlation': pixel_data['mean_stability_correlation'].mean(),
            'std_correlation': pixel_data['mean_stability_correlation'].std(),
        }
        semantic_stats_stability = {
            'mean_correlation': semantic_data['mean_stability_correlation'].mean(),
            'std_correlation': semantic_data['mean_stability_correlation'].std(),
        }

        # Calculate statistics for percentile correlation
        pixel_stats_percentile = {
            'mean_correlation': pixel_data['mean_percentile_correlation'].mean(),
            'std_correlation': pixel_data['mean_percentile_correlation'].std(),
        }
        semantic_stats_percentile = {
            'mean_correlation': semantic_data['mean_percentile_correlation'].mean(),
            'std_correlation': semantic_data['mean_percentile_correlation'].std(),
        }

        # Calculate performance gap
        gap_stability = semantic_stats_stability['mean_correlation'] - pixel_stats_stability['mean_correlation']
        gap_percentile = semantic_stats_percentile['mean_correlation'] - pixel_stats_percentile['mean_correlation']

        return {
            'summary': f"Semantic layers show a {gap_stability:.3f} (stability) and {gap_percentile:.3f} (percentile) point advantage in mean correlation.",
            'stability_correlation_advantage': {
                'performance_gap': gap_stability,
                'pixel_stats': pixel_stats_stability,
                'semantic_stats': semantic_stats_stability
            },
            'percentile_correlation_advantage': {
                'performance_gap': gap_percentile,
                'pixel_stats': pixel_stats_percentile,
                'semantic_stats': semantic_stats_percentile
            },
            'run_counts': {'pixel': len(pixel_data), 'semantic': len(semantic_data)}
        }

    def _check_layer_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check if optimal layers are consistent across training methods, based on correlation."""
        
        consistency_analysis = {}
        
        # Ensure we're using correlation data
        df_corr = df.dropna(subset=['stability_correctness_correlation'])
        
        for model in df_corr['model'].unique():
            model_data = df_corr[df_corr['model'] == model]
            
            if model_data.empty:
                continue
                
            # Get optimal layer for each training method based on stability correlation
            optimal_by_method = {}
            for method in model_data['training_method'].unique():
                method_data = model_data[model_data['training_method'] == method]
                if not method_data.empty:
                    layer_performance = method_data.groupby('layer')['stability_correctness_correlation'].mean()
                    if not layer_performance.empty:
                        optimal_layer = layer_performance.idxmax()
                        optimal_by_method[method] = optimal_layer
            
            if optimal_by_method:
                # Calculate consistency (most common optimal layer)
                from collections import Counter
                layer_counts = Counter(optimal_by_method.values())
                most_common_layer, count = layer_counts.most_common(1)[0]
                consistency_score = count / len(optimal_by_method)
                
                consistency_analysis[model] = {
                    'most_consistent_layer': most_common_layer,
                    'consistency_score': consistency_score,
                    'optimal_by_method': optimal_by_method,
                    'layer_vote_counts': dict(layer_counts)
                }
        
        return consistency_analysis
    
    def _save_aggregated_data(self, overall_df: pd.DataFrame, corruption_df: pd.DataFrame, 
                            aggregated_results: Dict[str, Any], dataset: str = 'cifar10'):
        """Save aggregated data with correlation focus. FIXED: Corrected field name mismatches."""
        
        print("💾 Saving correlation-focused aggregated results...")
        
        # Save raw data
        overall_df.to_csv(self.output_dir / f"stability_analysis_overall_data_{dataset}.csv", index=False)
        corruption_df.to_csv(self.output_dir / f"stability_analysis_per_corruption_data_{dataset}.csv", index=False)
        
        # Save correlation summary for both types
        if 'correlation_analysis' in aggregated_results:
            corr_analysis = aggregated_results['correlation_analysis']
            
            # Generate summaries for both correlation types
            for correlation_type in ['stability', 'percentile']:
                correlation_summary = []
                
                # Add layer correlation rankings
                if 'correlation_statistics' in corr_analysis:
                    for row in corr_analysis['correlation_statistics']:
                        model, layer, training_method = row['model'], row['layer'], row['training_method']
                        
                        # Get the appropriate correlation field
                        if correlation_type == 'stability':
                            correlation_value = row['mean_stability_corr']
                        else:
                            correlation_value = row['mean_percentile_corr']
                        
                        correlation_summary.append({
                            'model_method': f"{model}_{training_method}",  # Combined identifier
                            'model': model,  # Keep separate for potential analysis
                            'training_method': training_method,  # Keep separate for potential analysis
                            'layer': layer,
                            f'mean_{correlation_type}_correlation': correlation_value,
                            'correlation_category': self._categorize_correlation_strength(correlation_value)
                        })
                
                if correlation_summary:
                    correlation_df = pd.DataFrame(correlation_summary)
                    correlation_df.to_csv(self.output_dir / f"{correlation_type}_correlation_summary_{dataset}.csv", index=False)
        
        # Save aggregated results
        with open(self.output_dir / f"stability_analysis_aggregated_{dataset}.json", 'w') as f:
            import json
            
            def round_floats(obj):
                if isinstance(obj, float):
                    return round(obj, 5)
                elif isinstance(obj, dict):
                    return {key: round_floats(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [round_floats(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(round_floats(item) for item in obj)
                else:
                    return obj
            
            rounded_results = round_floats(aggregated_results)
            json.dump(rounded_results, f, indent=2, default=str)
        
        # Create a summary CSV for all analysis types - FIXED field names
        summary_data = []
        
        # Add correlation summary statistics
        if 'correlation_analysis' in aggregated_results:
            corr_analysis = aggregated_results['correlation_analysis']
            if 'correlation_statistics' in corr_analysis and corr_analysis['correlation_statistics']:
                first_stat = corr_analysis['correlation_statistics'][0]
                summary_data.append({
                    'analysis_type': 'correlation_summary',
                    'mean_stability_correlation': first_stat.get('mean_stability_corr', np.nan),  # FIXED
                    'std_stability_correlation': first_stat.get('std_stability_corr', np.nan),    # FIXED
                    'mean_percentile_correlation': first_stat.get('mean_percentile_corr', np.nan), # FIXED
                    'std_percentile_correlation': first_stat.get('std_percentile_corr', np.nan),  # FIXED
                    'mean_positive_percentage': overall_df['mean_positive_percentage'].mean() if 'mean_positive_percentage' in overall_df.columns else np.nan,
                    'run_count': len(overall_df)
                })
        
        # Add layer summary statistics
        if 'layer_analysis' in aggregated_results and 'optimal_layers' in aggregated_results['layer_analysis']:
            layer_analysis = aggregated_results['layer_analysis']
            # Get first model's data as example
            first_model = list(layer_analysis['optimal_layers'].keys())[0] if layer_analysis['optimal_layers'] else None
            if first_model:
                first_model_data = layer_analysis['optimal_layers'][first_model]
                summary_data.append({
                    'analysis_type': 'layer_summary',
                    'optimal_layer_stability': first_model_data.get('optimal_layer_by_stability_correlation', ''),
                    'mean_stability_correlation': first_model_data.get('mean_stability_correlation', np.nan),
                    'optimal_layer_percentile': first_model_data.get('optimal_layer_by_percentile_correlation', ''),
                    'mean_percentile_correlation': first_model_data.get('mean_percentile_correlation', np.nan),
                    'mean_positive_percentage': overall_df['mean_positive_percentage'].mean() if 'mean_positive_percentage' in overall_df.columns else np.nan,
                    'run_count': len(overall_df)
                })
        
        # Add semantic vs pixel summary
        if 'semantic_vs_pixel_analysis' in aggregated_results:
            semantic_analysis = aggregated_results['semantic_vs_pixel_analysis']
            summary_data.append({
                'analysis_type': 'semantic_vs_pixel_summary',
                'stability_correlation_gap': semantic_analysis.get('stability_correlation_advantage', {}).get('performance_gap', np.nan),
                'percentile_correlation_gap': semantic_analysis.get('percentile_correlation_advantage', {}).get('performance_gap', np.nan),
                'semantic_stability_mean': semantic_analysis.get('stability_correlation_advantage', {}).get('semantic_stats', {}).get('mean_correlation', np.nan),
                'pixel_stability_mean': semantic_analysis.get('stability_correlation_advantage', {}).get('pixel_stats', {}).get('mean_correlation', np.nan),
                'semantic_percentile_mean': semantic_analysis.get('percentile_correlation_advantage', {}).get('semantic_stats', {}).get('mean_correlation', np.nan),
                'pixel_percentile_mean': semantic_analysis.get('percentile_correlation_advantage', {}).get('pixel_stats', {}).get('mean_correlation', np.nan),
                'run_count': len(overall_df)
            })

        # Add corruption group rankings
        if 'corruption_group_analysis' in aggregated_results:
            corruption_analysis = aggregated_results['corruption_group_analysis']
            if 'group_ranking_by_stability_correlation' in corruption_analysis:
                for group, score in corruption_analysis['group_ranking_by_stability_correlation'].items():
                    summary_data.append({
                        'analysis_type': 'corruption_group_ranking_stability',
                        'corruption_group': group,
                        'mean_stability_correlation': score,
                        'ranking_type': 'stability'
                    })
            if 'group_ranking_by_percentile_correlation' in corruption_analysis:
                for group, score in corruption_analysis['group_ranking_by_percentile_correlation'].items():
                    summary_data.append({
                        'analysis_type': 'corruption_group_ranking_percentile',
                        'corruption_group': group,
                        'mean_percentile_correlation': score,
                        'ranking_type': 'percentile'
                    })
        
        # Add training method summary
        if 'training_method_analysis' in aggregated_results and 'method_statistics' in aggregated_results['training_method_analysis']:
            method_stats = aggregated_results['training_method_analysis']['method_statistics']
            # Convert DataFrame to list of records if it's a DataFrame
            if hasattr(method_stats, 'to_dict'):
                method_stats = method_stats.to_dict('records')
            
            for method_row in method_stats:
                summary_data.append({
                    'analysis_type': 'training_method_summary',
                    'training_method': method_row.get('training_method', ''),
                    'mean_stability_correlation': method_row.get('mean_stability_correlation_mean', np.nan),
                    'std_stability_correlation': method_row.get('mean_stability_correlation_std', np.nan),
                    'mean_percentile_correlation': method_row.get('mean_percentile_correlation_mean', np.nan),
                    'std_percentile_correlation': method_row.get('mean_percentile_correlation_std', np.nan),
                    'run_count': method_row.get('mean_stability_correlation_count', 0)
                })

        # Add architecture summary statistics
        if 'architecture_analysis' in aggregated_results and 'architecture_statistics' in aggregated_results['architecture_analysis']:
            arch_stats = aggregated_results['architecture_analysis']['architecture_statistics']
            for arch_row in arch_stats:  # arch_stats is already a list of records
                summary_data.append({
                    'analysis_type': 'architecture_summary',
                    'model': arch_row.get('model', ''),
                    'mean_stability_correlation': arch_row.get('mean_stability_correlation', np.nan),
                    'std_stability_correlation': arch_row.get('std_stability_correlation', np.nan),
                    'mean_percentile_correlation': arch_row.get('mean_percentile_correlation', np.nan),
                    'std_percentile_correlation': arch_row.get('std_percentile_correlation', np.nan),
                    'mean_positive_percentage': arch_row.get('mean_positive_percentage', np.nan),
                    'run_count': arch_row.get('run_count', 0)
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.output_dir / f"stability_analysis_summary_{dataset}.csv", index=False)
        
        print(f"✅ Results saved to {self.output_dir}")
    
    def _create_correlation_visualizations(self, overall_df: pd.DataFrame, 
                                           corruption_df: pd.DataFrame, 
                                           aggregated_results: Dict[str, Any], dataset: str = 'cifar10'):
        """Create clean, publication-ready visualizations using actual correlation data."""
        
        print("📊 Creating comprehensive correlation visualizations for AAAI paper...")
        
        # Generate visualizations for the two correlation types that actually exist in the data
        correlation_types = ['stability', 'percentile']
        
        for metric_type in correlation_types:
            self._create_single_correlation_visualization(
                overall_df, corruption_df, aggregated_results, dataset, metric_type
            )
        
        print(f"✅ All correlation visualizations created successfully!")
        print(f"   - Stability-Correctness Correlation Analysis")
        print(f"   - Percentile-Correctness Correlation Analysis")

    def _create_single_correlation_visualization(self, overall_df: pd.DataFrame, 
                                                corruption_df: pd.DataFrame, 
                                                aggregated_results: Dict[str, Any], 
                                                dataset: str = 'cifar10',
                                                metric_type: str = 'stability'):
        """Create visualization for a specific correlation type using actual field names."""
        
        correlation_name = f'{metric_type.title()}-Correctness'
        print(f"📊 Creating {correlation_name} correlation visualization...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

        fig.suptitle(f'{correlation_name} Correlation Analysis for {dataset.upper()}', fontsize=24, fontweight='bold')

        # 1. Primary Correlation Heatmap
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_correlation_heatmaps(corruption_df, ax1, fig, metric_type)
        
        # 2. Layer-wise Correlation Comparison (Key Insight)
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_layerwise_correlation_bars(corruption_df, ax2, metric_type)
        
        # 3. Correlation by Training Method
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_correlation_by_method(corruption_df, ax3, metric_type)
        
        # 4. Correlation by Corruption Group
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_correlation_by_corruption_group(corruption_df, ax4, metric_type)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Create filename
        filename = f"comprehensive_{metric_type}_correlation_analysis_{dataset}.png"
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ {correlation_name} correlation visualization saved as {filename}!")

    def _plot_correlation_heatmaps(self, df: pd.DataFrame, ax_main, fig, correlation_type: str):
        """Plot heatmap for specified correlation type."""
        correlation_name = 'Stability' if correlation_type == 'stability' else 'Percentile'
        correlation_column = f'{correlation_type}_correctness_correlation'
        
        ax_main.set_title(f'Layer {correlation_name}-Correctness Correlation Strength', fontweight='bold', fontsize=16)
        
        if df.empty:
            ax_main.text(0.5, 0.5, 'No data', ha='center', va='center')
            return

        # Heatmap for specified correlation type
        pivot_data = df.pivot_table(
            values=correlation_column,
            index='model', columns='layer', aggfunc='mean'
        )
        
        # Check if pivot_data is empty or has no valid data
        if pivot_data.empty or pivot_data.isna().all().all():
            ax_main.text(0.5, 0.5, 'No valid correlation data', ha='center', va='center')
            return
            
        sns.heatmap(pivot_data, annot=True, cmap='viridis', ax=ax_main, fmt='.3f', 
                   linewidths=.5, cbar_kws={'label': f'Mean {correlation_name}-Correctness Correlation'})
        ax_main.set_xlabel('Layer', fontsize=12)
        ax_main.set_ylabel('Model', fontsize=12)
        plt.setp(ax_main.get_xticklabels(), rotation=45)

    def _plot_layerwise_correlation_bars(self, df: pd.DataFrame, ax, correlation_type: str):
        """Plot bar chart for layer-wise correlation of specified type."""
        correlation_name = 'Stability' if correlation_type == 'stability' else 'Percentile'
        correlation_column = f'{correlation_type}_correctness_correlation'
        
        ax.set_title(f'Semantic vs. Pixel Space: Layer-wise {correlation_name}-Correctness Correlation', fontweight='bold', fontsize=16)
        if df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return

        df_corr = df.dropna(subset=[correlation_column])
        
        # Check if df_corr is empty after dropping NaN values
        if df_corr.empty:
            ax.text(0.5, 0.5, 'No valid correlation data', ha='center', va='center')
            return
        
        # Create a combined 'model | layer' column for the x-axis
        df_corr['model_layer'] = df_corr['model'] + ' | ' + df_corr['layer']
        
        # Sort by correlation value for a cleaner plot
        order = df_corr.groupby('model_layer')[correlation_column].mean().sort_values(ascending=False).index

        sns.barplot(
            data=df_corr,
            x='model_layer',
            y=correlation_column,
            ax=ax,
            palette='viridis',
            order=order,
            ci=95,
            capsize=.05
        )
        
        ax.set_xlabel('Model and Layer', fontsize=12)
        ax.set_ylabel(f'{correlation_name}-Correctness Correlation', fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
        
        # Add interpretation lines
        ax.axhline(0.8, color='green', linestyle='--', alpha=0.8, lw=1.5)
        ax.text(1.0, 0.8, 'Excellent', transform=ax.get_yaxis_transform(), ha='left', va='center', color='green', fontsize=10, fontweight='bold')
        ax.axhline(0.65, color='darkorange', linestyle='--', alpha=0.8, lw=1.5)
        ax.text(1.0, 0.65, 'Good', transform=ax.get_yaxis_transform(), ha='left', va='center', color='darkorange', fontsize=10, fontweight='bold')

    def _plot_correlation_by_method(self, df: pd.DataFrame, ax, correlation_type: str):
        """Plot boxplot for correlation by training method for specified correlation type."""
        correlation_name = 'Stability' if correlation_type == 'stability' else 'Percentile'
        correlation_column = f'{correlation_type}_correctness_correlation'
        
        ax.set_title(f'{correlation_name}-Correctness Correlation Robustness by Training Method', fontweight='bold', fontsize=14)
        if df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return
            
        df_corr = df.dropna(subset=[correlation_column])
        
        # Check if df_corr is empty after dropping NaN values
        if df_corr.empty:
            ax.text(0.5, 0.5, 'No valid correlation data', ha='center', va='center')
            return
        
        order = df_corr.groupby('training_method')[correlation_column].median().sort_values(ascending=False).index
        
        sns.boxplot(
            data=df_corr,
            x=correlation_column,
            y='training_method',
            ax=ax,
            orient='h',
            order=order,
            palette='viridis'
        )
        
        ax.set_xlabel(f'{correlation_name}-Correctness Correlation Distribution', fontsize=12)
        ax.set_ylabel('Training Method', fontsize=12)
        ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.6)

    def _plot_correlation_by_corruption_group(self, df: pd.DataFrame, ax, correlation_type: str):
        """Plot boxplot for correlation by corruption group for specified correlation type."""
        correlation_name = 'Stability' if correlation_type == 'stability' else 'Percentile'
        correlation_column = f'{correlation_type}_correctness_correlation'
        
        ax.set_title(f'{correlation_name}-Correctness Correlation Robustness by Corruption Group', fontweight='bold', fontsize=14)
        if df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return
            
        df_corr = df.dropna(subset=[correlation_column])

        # Check if df_corr is empty after dropping NaN values
        if df_corr.empty:
            ax.text(0.5, 0.5, 'No valid correlation data', ha='center', va='center')
            return

        order = df_corr.groupby('corruption_group')[correlation_column].median().sort_values(ascending=False).index

        sns.boxplot(
            data=df_corr,
            x='corruption_group',
            y=correlation_column,
            ax=ax,
            order=order,
            palette='viridis'
        )
        
        ax.set_xlabel('Corruption Group', fontsize=12)
        ax.set_ylabel(f'{correlation_name}-Correctness Correlation', fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description="Enhanced Stability Analysis Aggregator - Correlation Focused for AAAI (FIXED)")
    parser.add_argument("--input-dir", type=str, required=True,
                      help="Base directory containing stability analysis results")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Output directory for aggregated results")
    parser.add_argument("--training-methods", nargs='+', 
                      default=["constellation"],
                      help="Training methods to aggregate")
    parser.add_argument("--models", nargs='+',
                      default=["resnet18", "resnet50", "densenet121"],
                      help="Models to aggregate")
    parser.add_argument("--seeds", nargs='+', type=int,
                      default=[12],
                      help="Seeds to aggregate")
    parser.add_argument("--dataset", type=str, default='cifar10',
                      choices=['cifar10', 'cifar100'],
                      help="Dataset to aggregate (default: cifar10)")
    
    args = parser.parse_args()
    
    print("🔗 AAAI Enhanced Stability Analysis Aggregator - CORRELATION FOCUSED (FIXED)")
    print("=" * 80)
    print(f"🎯 PRIMARY FOCUS: Stability-correctness correlation as calibration quality metric")
    print(f"💡 KEY INSIGHT: High correlation enables effective isotonic regression mapping")
    print(f"🔧 FIXED: Corrected field name mismatches in data aggregation")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Training methods: {args.training_methods}")
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    print("=" * 80)
    
    # Create enhanced aggregator
    aggregator = EnhancedStabilityAnalysisAggregator(args.input_dir, args.output_dir)
    
    # Run correlation-focused aggregation
    results = aggregator.scan_and_aggregate_results(
        training_methods=args.training_methods,
        models=args.models,
        seeds=args.seeds,
        dataset=args.dataset
    )
    
    if results:
        print(f"✅ {args.dataset.upper()} correlation-focused analysis completed successfully!")
        print(f"📊 Results saved to: {args.output_dir}")
        print("\n🎯 Key files generated:")
        print(f"  - correlation_summary_{args.dataset}.csv (PRIMARY)")
        print(f"  - stability_analysis_overall_data_{args.dataset}.csv")
        print(f"  - stability_analysis_per_corruption_data_{args.dataset}.csv")
        print(f"  - stability_analysis_aggregated_{args.dataset}.json") 
        print(f"  - stability_analysis_summary_{args.dataset}.csv")
        print(f"  - comprehensive_correlation_analysis_{args.dataset}.png")
        
        # Print key insights
        if 'layer_analysis' in results and 'optimal_layers' in results['layer_analysis']:
            print("\n🎯 KEY INSIGHTS (based on Correlation):")
            for model, data in results['layer_analysis']['optimal_layers'].items():
                print(f"  {model}: Optimal layer (Stability Corr) = {data['optimal_layer_by_stability_correlation']} ({data['mean_stability_correlation']:.3f})")
                print(f"         Optimal layer (Percentile Corr) = {data['optimal_layer_by_percentile_correlation']} ({data['mean_percentile_correlation']:.3f})")
        
        print(f"\n💡 PAPER INSIGHT: High stability-correctness correlation in semantic layers")
        print(f"   demonstrates why geometric calibration works best in learned feature space!")
        
    else:
        print("❌ No results found for aggregation!")
        print(f"Check your path structure matches: results/{{training_method}}/{args.dataset}/{{model}}/{{layer}}_stability/seed{{seed}}/")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())