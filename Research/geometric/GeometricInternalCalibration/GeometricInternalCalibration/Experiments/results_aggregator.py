#!/usr/bin/env python3
"""
Enhanced Results Aggregator for Multi-Corruption Section 4 Statistical Analysis
Updated to handle comprehensive analysis across:
- All severity levels (1-5)
- All classes (full dataset classes)
- All corruption types (15 types)
- Training methods and layers
- Statistical significance testing

Compatible with new aggregated_results.json structure from robust_statistical_analysis_multi_corruption.py
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
from collections import defaultdict
from typing import Dict, List, Any, Tuple
warnings.filterwarnings('ignore')

# Corruption groupings for analysis
CORRUPTION_GROUPS = {
    'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
    'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'weather': ['snow', 'frost', 'fog', 'brightness'],
    'digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
}

ALL_CORRUPTION_TYPES = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

# Dataset configurations
DATASET_CONFIGS = {
    'cifar10': {
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'num_classes': 10
    },
    'cifar100': {
        'classes': [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
            'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
            'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
            'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
            'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ],
        'num_classes': 100
    }
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced aggregator for multi-corruption Section 4 experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Base directory containing results (e.g., section4_multi_corruption_analysis)')
    
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save aggregated results')
    
    parser.add_argument('--training-methods', type=str, nargs='+', 
                       default=['constellation_original', 'baseline_focal_adaptive', 'baseline_cross_entropy', 'augmix', 'baseline_mmce_adaptive', 'baseline_focal', 'baseline_mmce'],
                       help='Training methods to aggregate')
    
    parser.add_argument('--models', type=str, nargs='+',
                       default=['resnet18', 'resnet50', 'densenet121'],
                       help='Models to aggregate')
    
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[12, 13, 14, 11],
                       help='Seeds to aggregate')
    
    parser.add_argument('--layers', type=str, nargs='+',
                       default=None,
                       help='Specific layers to analyze (default: all available layers)')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset name (default: cifar10)')
    
    return parser.parse_args()

def load_enhanced_multi_corruption_results(input_dir, training_methods, models, seeds, layers=None, dataset='cifar10'):
    """Load enhanced multi-corruption results with severity and class analysis."""
    
    input_path = Path(input_dir)
    all_results = {}
    
    print(f"📁 Scanning {input_path} for enhanced multi-corruption results...")
    print(f"   Training methods: {training_methods}")
    print(f"   Models: {models}")
    print(f"   Seeds: {seeds}")
    print(f"   Dataset: {dataset}")
    
    # Define layer mappings for each model
    model_layers = {
        'resnet18': ['layer1', 'layer2', 'layer3', 'layer4', 'layer4.1'],
        'resnet50': ['layer1', 'layer2', 'layer3', 'layer4', 'layer4.1'],
        'densenet121': ['dense1', 'trans1', 'dense2', 'trans2', 'dense3', 'trans3', 'dense4', 'bn'],
        'dinov2_small_scratch': ['block_0', 'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'block_6', 'block_7', 'block_8', 'block_9', 'block_10', 'block_11'],
        'dinov2base_scratch': ['block_0', 'block_1', 'block_2', 'block_3', 'block_4', 'block_5', 'block_6', 'block_7', 'block_8', 'block_9', 'block_10', 'block_11']
    }
    
    # If specific layers provided, use them; otherwise use all layers for each model
    if layers:
        for model in models:
            model_layers[model] = layers
    
    total_expected = 0
    total_found = 0
    
    for training_method in training_methods:
        for model in models:
            model_layer_list = model_layers.get(model, [])
            for layer in model_layer_list:
                for seed in seeds:
                    total_expected += 1
                    
                    # Look for comprehensive, then legacy analysis directories
                    result_dir_comp = input_path / "results" / training_method / dataset / model / f"{layer}_analysis_comprehensive" / f"seed{seed}"
                    result_dir_legacy = input_path / "results" / training_method / dataset / model / f"{layer}_analysis" / f"seed{seed}"
                    
                    aggregated_file = None
                    if (result_dir_comp / 'aggregated_results.json').exists():
                        aggregated_file = result_dir_comp / 'aggregated_results.json'
                    elif (result_dir_legacy / 'aggregated_results.json').exists():
                        aggregated_file = result_dir_legacy / 'aggregated_results.json'
                    
                    result_key = (training_method, model, layer, seed)
                    
                    if aggregated_file:
                        print(f"   📄 Loading {aggregated_file.relative_to(input_path)}...")
                        try:
                            with open(aggregated_file, 'r') as f:
                                data = json.load(f)
                            
                            agg_stats = data.get('aggregated_statistics', {})
                            per_corruption_results = data.get('per_corruption_results', {})
                                
                            all_results[result_key] = {
                                'training_method': training_method,
                                'model': model,
                                'layer': layer,
                                'seed': seed,
                                'data': agg_stats,
                                'per_corruption_data': per_corruption_results,
                                'multi_corruption': True,
                                'k_values': agg_stats.get('k_values', [1, 3, 5]),
                                'has_dimensional_data': agg_stats.get('has_dimensional_data', False)
                            }
                            
                            print(f"      ✅ {model}-{layer} (seed{seed}) - Loaded with k_values: {all_results[result_key]['k_values']}")
                            total_found += 1
                            
                        except Exception as e:
                            print(f"      ❌ Error loading {aggregated_file}: {e}")
                    else:
                        print(f"      ⏭️ Missing: {training_method}/{model}/{layer}/seed{seed}")
    
    print(f"✅ Loaded {total_found}/{total_expected} enhanced experiment results")
    return all_results

def create_severity_analysis(all_results, output_dir, dataset='cifar10'):
    """Analyze performance across severity levels (1-5) for each layer and method, aggregated by k."""
    
    print("📊 Creating severity-level analysis by k...")
    
    severity_data = []
    
    for (training_method, model, layer, seed), result in all_results.items():
        if result['has_dimensional_data']:
            severity_analysis = result['data'].get('by_severity_analysis', {})
            k_values = result.get('k_values', [1, 3, 5])
            
            for severity_key, severity_stats in severity_analysis.items():
                for k in k_values:
                    k_str = str(k)
                    if 'k_metrics' in severity_stats and k_str in severity_stats['k_metrics']:
                        k_data = severity_stats['k_metrics'][k_str]
                        severity_level = severity_stats.get('severity_level', 0)
                        
                        severity_data.append({
                            'k': k,
                            'Training Method': training_method,
                            'Model': model,
                            'Layer': layer,
                            'Seed': seed,
                            'Severity Level': severity_level,
                            'Mean Advantage': k_data.get('mean_advantage', 0.0),
                            'Std Advantage': k_data.get('std_advantage', 0.0),
                            'Positive Advantage %': k_data.get('positive_advantage_pct', 0.0),
                            'Sample Count': severity_stats.get('sample_count', 0),
                            'Semantic Accuracy': k_data.get('mean_semantic_accuracy', 0.0),
                            'Pixel Accuracy': k_data.get('mean_pixel_accuracy', 0.0),
                            'Advantage Magnitude': k_data.get('semantic_advantage_magnitude', 0.0),
                            'Consistency Score': k_data.get('consistency_score', 0.0)
                        })
    
    if not severity_data:
        print("⚠️ No severity analysis data found")
        return None
    
    df = pd.DataFrame(severity_data)
    
    # Create comprehensive severity analysis visualizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Mean Advantage by Severity Level and k
    ax = axes[0, 0]
    sns.lineplot(data=df, x='Severity Level', y='Mean Advantage', hue='k', marker='o', ax=ax, palette='viridis', ci='sd')
    ax.set_title('Mean Advantage by Severity Level and k-NN', fontweight='bold')
    ax.set_xlabel('Severity Level')
    ax.set_ylabel('Mean Advantage')
    ax.grid(True, alpha=0.3)
    
    # 2. Best performing layer for each severity and k
    ax = axes[0, 1]
    best_layer_data = df.loc[df.groupby(['Severity Level', 'k'])['Mean Advantage'].idxmax()]
    sns.scatterplot(data=best_layer_data, x='Severity Level', y='Mean Advantage', hue='Layer', style='k', s=150, ax=ax)
    ax.set_title('Best Layer Performance by Severity and k-NN', fontweight='bold')
    ax.set_xlabel('Severity Level')
    ax.set_ylabel('Mean Advantage')
    ax.grid(True, alpha=0.3)
    
    # 3. Semantic vs Pixel Accuracy by Severity and k
    ax = axes[1, 0]
    df_melt = df.melt(id_vars=['Severity Level', 'k'], value_vars=['Semantic Accuracy', 'Pixel Accuracy'], 
                      var_name='Space', value_name='Accuracy')
    sns.barplot(data=df_melt, x='Severity Level', y='Accuracy', hue='Space', ci='sd', ax=ax)
    ax.set_title('Accuracy by Severity Level', fontweight='bold')
    ax.set_xlabel('Severity Level')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Training method comparison across severities for k=5
    ax = axes[1, 1]
    df_k5 = df[df['k'] == df['k'].max()]
    method_severity = df_k5.groupby(['Training Method', 'Severity Level'])['Mean Advantage'].mean().reset_index()
    method_pivot = method_severity.pivot(index='Training Method', columns='Severity Level', values='Mean Advantage')
    sns.heatmap(method_pivot, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax)
    ax.set_title(f'Training Method Performance by Severity (k={df["k"].max()})', fontweight='bold')
    ax.set_xlabel('Severity Level')
    ax.set_ylabel('Training Method')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'severity_level_analysis_{dataset}_by_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed severity analysis
    df.to_csv(output_dir / f'severity_analysis_detailed_{dataset}_by_k.csv', index=False)
    
    print(f"✅ Severity analysis by k saved to {output_dir}/severity_level_analysis_{dataset}_by_k.png")
    
    return df

def create_class_analysis(all_results, output_dir, dataset='cifar10'):
    """Analyze performance across all classes for each layer and method, aggregated by k."""
    
    print("📊 Creating class-specific analysis by k...")
    
    class_data = []
    class_names = DATASET_CONFIGS[dataset]['classes']
    
    for (training_method, model, layer, seed), result in all_results.items():
        if result['has_dimensional_data']:
            class_analysis = result['data'].get('by_class_analysis', {})
            k_values = result.get('k_values', [1, 3, 5])

            for class_key, class_stats in class_analysis.items():
                for k in k_values:
                    k_str = str(k)
                    if 'k_metrics' in class_stats and k_str in class_stats['k_metrics']:
                        k_data = class_stats['k_metrics'][k_str]
                        class_id = class_stats.get('class_id', 0)
                        class_name = class_stats.get('class_name', f'class_{class_id}')
                        
                        class_data.append({
                            'k': k,
                            'Training Method': training_method,
                            'Model': model,
                            'Layer': layer,
                            'Seed': seed,
                            'Class ID': class_id,
                            'Class Name': class_name,
                            'Mean Advantage': k_data.get('mean_advantage', 0.0),
                            'Std Advantage': k_data.get('std_advantage', 0.0),
                            'Positive Advantage %': k_data.get('positive_advantage_pct', 0.0),
                            'Sample Count': class_stats.get('sample_count', 0),
                            'Semantic Accuracy': k_data.get('mean_semantic_accuracy', 0.0),
                            'Pixel Accuracy': k_data.get('mean_pixel_accuracy', 0.0),
                            'Advantage Magnitude': k_data.get('semantic_advantage_magnitude', 0.0),
                            'Consistency Score': k_data.get('consistency_score', 0.0)
                        })
    
    if not class_data:
        print("⚠️ No class analysis data found")
        return None
    
    df = pd.DataFrame(class_data)
    
    # Create comprehensive class analysis visualizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Mean Advantage by Class and k (top 15 classes)
    ax = axes[0, 0]
    class_summary = df.groupby(['Class Name', 'k'])['Mean Advantage'].mean().reset_index()
    top_classes = class_summary.groupby('Class Name')['Mean Advantage'].mean().nlargest(15).index
    class_summary_top = class_summary[class_summary['Class Name'].isin(top_classes)]
    
    sns.barplot(data=class_summary_top, x='Class Name', y='Mean Advantage', hue='k', ax=ax)
    ax.set_title(f'Mean Advantage by Class and k-NN (Top {len(top_classes)} Classes)', fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean Advantage')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 2. Best layer for each class (at k=5)
    ax = axes[0, 1]
    df_k5 = df[df['k'] == df['k'].max()]
    best_layers_by_class = df_k5.loc[df_k5.groupby('Class Name')['Mean Advantage'].idxmax()]
    
    top_classes_k5 = best_layers_by_class.nlargest(15, 'Mean Advantage')
    
    layer_colors = {'layer1': 'red', 'layer2': 'blue', 'layer3': 'green', 'layer4': 'orange', 'layer4.1': 'purple',
                   'dense1': 'red', 'trans1': 'blue', 'dense2': 'green', 'trans2': 'orange', 'dense3': 'purple', 'trans3': 'brown', 'dense4': 'pink', 'bn': 'gray'}
    
    bars = ax.bar(range(len(top_classes_k5)), top_classes_k5['Mean Advantage'], 
                  color=[layer_colors.get(layer, 'gray') for layer in top_classes_k5['Layer']], alpha=0.7)
    ax.set_title(f'Best Layer Performance by Class (k={df["k"].max()})', fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean Advantage')
    ax.set_xticks(range(len(top_classes_k5)))
    ax.set_xticklabels(top_classes_k5['Class Name'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 3. Semantic vs Pixel accuracy by k
    ax = axes[1, 0]
    acc_by_k = df.groupby('k')[['Semantic Accuracy', 'Pixel Accuracy']].mean().reset_index()
    acc_melt = acc_by_k.melt(id_vars='k', var_name='Space', value_name='Accuracy')
    sns.barplot(data=acc_melt, x='k', y='Accuracy', hue='Space', ax=ax)
    ax.set_title('Overall Accuracy by k-NN', fontweight='bold')
    ax.set_xlabel('k-NN Value')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Class difficulty ranking (at k=5)
    ax = axes[1, 1]
    difficulty_ranking = df_k5.groupby('Class Name')['Mean Advantage'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    
    # Show top 20 classes
    if len(difficulty_ranking) > 20:
        difficulty_ranking = difficulty_ranking.head(20)
    
    bars = ax.bar(range(len(difficulty_ranking)), difficulty_ranking['mean'], 
                  yerr=difficulty_ranking['std'], alpha=0.7, capsize=5,
                  color=plt.cm.RdYlBu_r(np.linspace(0, 1, len(difficulty_ranking))))
    ax.set_title(f'Class Difficulty Ranking (k={df["k"].max()})', fontweight='bold')
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean Advantage ± Std')
    ax.set_xticks(range(len(difficulty_ranking)))
    ax.set_xticklabels(difficulty_ranking.index, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'class_analysis_{dataset}_by_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed class analysis
    df.to_csv(output_dir / f'class_analysis_detailed_{dataset}_by_k.csv', index=False)
    
    print(f"✅ Class analysis by k saved to {output_dir}/class_analysis_{dataset}_by_k.png")
    
    return df

def create_comprehensive_training_method_analysis(all_results, output_dir, dataset='cifar10'):
    """Analyze training method effectiveness across all conditions, aggregated by k."""
    
    print("📊 Creating comprehensive training method analysis by k...")
    
    method_data = []
    
    for (training_method, model, layer, seed), result in all_results.items():
        k_values = result.get('k_values', [1, 3, 5])
        
        # This handles both dimensional and legacy data structures
        if result['has_dimensional_data']:
            overall_analysis = result['data'].get('overall_analysis', {})
            overall_stats_by_k = overall_analysis.get('overall', {})
        else: # Legacy
            overall_stats_by_k = result['data'].get('overall', {})

        for k in k_values:
            k_str = str(k)
            if k_str in overall_stats_by_k:
                overall_stats = overall_stats_by_k[k_str]
                
                method_data.append({
                    'k': k,
                    'Training Method': training_method,
                    'Model': model,
                    'Layer': layer,
                    'Seed': seed,
                    'Overall Mean Advantage': overall_stats.get('mean_advantage', 0.0),
                    'Overall Std Advantage': overall_stats.get('std_advantage', 0.0),
                    'Overall Positive %': overall_stats.get('positive_advantage_pct', 0.0),
                    'Semantic Accuracy': overall_stats.get('mean_semantic_target_accuracy', 0.0),
                    'Pixel Accuracy': overall_stats.get('mean_pixel_target_accuracy', 0.0),
                    'Corruption Count': overall_stats.get('corruption_count', 15),
                    'Total Analyzed': overall_stats.get('total_analyzed', 0)
                })
    
    if not method_data:
        print("⚠️ No method analysis data found")
        return None
        
    df = pd.DataFrame(method_data)
    
    # Create training method comparison visualizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Overall performance by training method and k
    ax = axes[0, 0]
    sns.barplot(data=df, x='Training Method', y='Overall Mean Advantage', hue='k', ax=ax, ci='sd')
    ax.set_title('Training Method Performance by k-NN', fontweight='bold')
    ax.set_xlabel('Training Method')
    ax.set_ylabel('Mean Advantage')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 2. Semantic vs Pixel accuracy by training method for k=1
    ax = axes[0, 1]
    df_k5 = df[df['k'] == df['k'].min()]
    method_acc = df_k5.groupby('Training Method')[['Semantic Accuracy', 'Pixel Accuracy']].mean().reset_index()
    method_acc_melt = method_acc.melt(id_vars='Training Method', var_name='Space', value_name='Accuracy')
    
    sns.barplot(data=method_acc_melt, x='Training Method', y='Accuracy', hue='Space', ax=ax)
    ax.set_title(f'Accuracy by Training Method (k={df["k"].min()})', fontweight='bold')
    ax.set_xlabel('Training Method')
    ax.set_ylabel('Accuracy (%)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 3. Training method vs Model performance for k=1
    ax = axes[1, 0]
    method_model = df_k5.groupby(['Training Method', 'Model'])['Overall Mean Advantage'].mean().reset_index()
    method_model_pivot = method_model.pivot(index='Training Method', columns='Model', values='Overall Mean Advantage')
    sns.heatmap(method_model_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax)
    ax.set_title(f'Training Method vs Model Performance (k={df["k"].min()})', fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Training Method')
    
    # 4. Statistical significance testing between methods for k=5
    ax = axes[1, 1]
    methods = df_k5['Training Method'].unique()
    significance_matrix = pd.DataFrame(np.ones((len(methods), len(methods))), index=methods, columns=methods)
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                data1 = df_k5[df_k5['Training Method'] == method1]['Overall Mean Advantage'].values
                data2 = df_k5[df_k5['Training Method'] == method2]['Overall Mean Advantage'].values
                
                if len(data1) > 1 and len(data2) > 1:
                    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False) # Welch's t-test
                    significance_matrix.iloc[i, j] = p_value
                    significance_matrix.iloc[j, i] = p_value
    
    sns.heatmap(significance_matrix, annot=True, fmt='.3f', cmap='coolwarm_r', ax=ax, vmin=0, vmax=0.1)
    ax.set_title(f'P-values for Method Comparisons (k={df["k"].max()})', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_method_analysis_{dataset}_by_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed analysis
    df.to_csv(output_dir / f'training_method_detailed_{dataset}_by_k.csv', index=False)
    
    print(f"✅ Training method analysis by k saved to {output_dir}/training_method_analysis_{dataset}_by_k.png")
    
    return df

def create_optimal_configuration_analysis(all_results, output_dir, dataset='cifar10'):
    """Find optimal layer-method combinations for different scenarios, considering k."""
    
    print("📊 Creating optimal configuration analysis by k...")
    
    config_data = []
    
    for (training_method, model, layer, seed), result in all_results.items():
        k_values = result.get('k_values', [1, 3, 5])
        
        # This handles both dimensional and legacy data structures
        if result['has_dimensional_data']:
            overall_analysis = result['data'].get('overall_analysis', {})
            overall_stats_by_k = overall_analysis.get('overall', {})
        else: # Legacy
            overall_stats_by_k = result['data'].get('overall', {})

        for k in k_values:
            k_str = str(k)
            if k_str in overall_stats_by_k:
                overall_stats = overall_stats_by_k[k_str]
                
                config_data.append({
                    'k': k,
                    'Config': f"{training_method}-{model}-{layer}",
                    'Training Method': training_method,
                    'Model': model,
                    'Layer': layer,
                    'Seed': seed,
                    'Mean Advantage': overall_stats.get('mean_advantage', 0.0),
                    'Std Advantage': overall_stats.get('std_advantage', 0.0),
                    'Positive %': overall_stats.get('positive_advantage_pct', 0.0),
                    'Semantic Accuracy': overall_stats.get('mean_semantic_target_accuracy', 0.0),
                    'Pixel Accuracy': overall_stats.get('mean_pixel_target_accuracy', 0.0)
                })
    
    if not config_data:
        print("⚠️ No configuration analysis data found")
        return None, None

    df = pd.DataFrame(config_data)
    
    # Find optimal configurations for k=5
    df_k5 = df[df['k'] == df['k'].max()]
    optimal_configs = {
        'best_overall': df_k5.loc[df_k5['Mean Advantage'].idxmax()],
        'most_stable': df_k5.loc[(df_k5['Mean Advantage'] - df_k5['Std Advantage']).idxmax()],
        'best_semantic': df_k5.loc[df_k5['Semantic Accuracy'].idxmax()]
    }
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Top 10 configurations by mean advantage for different k
    ax = axes[0]
    top_configs = df.groupby('k').apply(lambda x: x.nlargest(4, 'Mean Advantage')).reset_index(drop=True)
    sns.barplot(data=top_configs, x='Config', y='Mean Advantage', hue='k', ax=ax, dodge=False)
    ax.set_title('Top Configurations by Mean Advantage and k-NN', fontweight='bold')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Mean Advantage')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 2. Layer effectiveness summary by k
    ax = axes[1]
    sns.boxplot(data=df, x='Layer', y='Mean Advantage', hue='k', ax=ax)
    ax.set_title('Layer Effectiveness Summary by k-NN', fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Advantage Distribution')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'optimal_configuration_analysis_{dataset}_by_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    df.to_csv(output_dir / f'optimal_configuration_detailed_{dataset}_by_k.csv', index=False)
    
    # Convert optimal configs to serializable format
    optimal_configs_serializable = {}
    for key, config in optimal_configs.items():
        if not config.empty:
            optimal_configs_serializable[key] = config.to_dict()

    with open(output_dir / f'optimal_configurations_{dataset}_by_k.json', 'w') as f:
        json.dump(optimal_configs_serializable, f, indent=2)
    
    print(f"✅ Optimal configuration analysis by k saved to {output_dir}/optimal_configuration_analysis_{dataset}_by_k.png")
    
    return df, optimal_configs_serializable

def create_advantage_distribution_plot(all_results: Dict[Tuple, Dict], output_dir: Path, dataset: str):
    """Create a violin plot of the semantic advantage distribution by k."""
    print("📊 Creating semantic advantage distribution plot...")
    
    distribution_data = []
    
    for result in all_results.values():
        k_values = result.get('k_values', [1, 3, 5])
        per_corruption_data = result.get('per_corruption_data', {})
        
        for corruption_results in per_corruption_data.values():
            samples = corruption_results.get('all_results', [])
            for sample in samples:
                for k in k_values:
                    score_key = f'advantage_score_k{k}'
                    if score_key in sample:
                        distribution_data.append({
                            'k': f'k={k}',
                            'Advantage Score': sample[score_key]
                        })
    
    if not distribution_data:
        print("⚠️ No data found for semantic advantage distribution plot.")
        return

    df = pd.DataFrame(distribution_data)
    
    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(data=df, x='k', y='Advantage Score', hue='k', dodge=False, legend=False, palette='viridis')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Semantic Advantage Distribution by k', fontweight='bold', fontsize=14)
    ax.set_xlabel('k-NN Value')
    ax.set_ylabel('Semantic Advantage Score\n(# correct class in semantic - # in pixel)')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    save_path = output_dir / f'semantic_advantage_distribution_{dataset}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Semantic advantage distribution plot saved to {save_path}")

def print_comprehensive_summary(all_results, severity_df, class_df, method_df, optimal_configs, dataset='cifar10'):
    """Print comprehensive summary of all analyses, with focus on k."""
    
    print("\n" + "="*80)
    print(f"COMPREHENSIVE ANALYSIS SUMMARY (Dataset: {dataset.upper()})")
    print("="*80)
    
    if method_df is not None:
        print(f"\n🔧 TRAINING METHOD INSIGHTS (Aggregated across k):")
        method_summary = method_df.groupby('Training Method')['Overall Mean Advantage'].mean().sort_values(ascending=False)
        print(f"   Best method overall: {method_summary.index[0]} (avg advantage = {method_summary.iloc[0]:.3f})")

        print("\n   Performance by k-NN value:")
        k_summary = method_df.groupby('k')['Overall Mean Advantage'].mean()
        for k, adv in k_summary.items():
            print(f"   - k={k}: Average advantage = {adv:.3f}")

    if optimal_configs:
        max_k = max(c['k'] for c in optimal_configs.values() if 'k' in c)
        print(f"\n🏆 OPTIMAL CONFIGURATIONS (for k={max_k}):")
        for config_type, config_data in optimal_configs.items():
            if config_data:
                print(f"   {config_type.replace('_', ' ').title()}: {config_data['Config']} "
                      f"(advantage = {config_data['Mean Advantage']:.3f})")
    
    print("\n" + "="*80)

def main():
    """Main aggregation function with enhanced multi-corruption analysis."""
    args = parse_args()
    
    print("="*80)
    print("ENHANCED MULTI-CORRUPTION RESULTS AGGREGATOR")
    print("Section 4 Statistical Analysis - Comprehensive Insights")
    print("="*80)
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Dataset: {args.dataset}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load all experiment results
    all_results = load_enhanced_multi_corruption_results(
        args.input_dir, 
        args.training_methods, 
        args.models, 
        args.seeds, 
        args.layers, 
        args.dataset
    )
    
    if not all_results:
        print("❌ No results found to aggregate.")
        return
    
    print(f"✅ Found results for {len(all_results)} experiment configurations")
    
    # Create comprehensive analyses
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE ANALYSES")
    print("="*60)
    
    # 1. Severity Analysis
    severity_df = create_severity_analysis(all_results, output_dir, args.dataset)
    
    # 2. Class Analysis
    class_df = create_class_analysis(all_results, output_dir, args.dataset)
    
    # 3. Training Method Analysis
    method_df = create_comprehensive_training_method_analysis(all_results, output_dir, args.dataset)
    
    # 4. Optimal Configuration Analysis
    config_df, optimal_configs = create_optimal_configuration_analysis(all_results, output_dir, args.dataset)
    
    # 5. Advantage Distribution Plot
    create_advantage_distribution_plot(all_results, output_dir, args.dataset)

    # 6. Print comprehensive summary
    print_comprehensive_summary(all_results, severity_df, class_df, method_df, optimal_configs, args.dataset)
    
    print(f"\n🚀 Enhanced multi-corruption analysis complete!")
    print(f"   All results saved to: {output_dir}")
    print(f"   Ready for AAAI submission insights!")

if __name__ == "__main__":
    main()