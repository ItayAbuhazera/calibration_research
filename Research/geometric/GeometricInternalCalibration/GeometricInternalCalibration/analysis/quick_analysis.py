#!/usr/bin/env python3
"""
Quick calibration results analysis - simplified version
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

def quick_analysis(results_dir: str, accuracy_threshold: float = 0.5):
    """Quick analysis of calibration results"""
    
    results_dir = Path(results_dir)
    all_results = []
    
    # Load all results
    for json_file in results_dir.rglob("calibration_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            config = data['experiment_config']
            for cal_method, metrics in data['results'].items():
                if 'error' not in metrics and metrics['accuracy'] >= accuracy_threshold * 100:
                    all_results.append({
                        'training_method': config['method'],
                        'dataset': config['dataset'],
                        'model': config['model'],
                        'seed': config['seed'],
                        'calibration_method': cal_method,
                        'accuracy': metrics['accuracy'],
                        'ece': metrics['ece']
                    })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    df = pd.DataFrame(all_results)
    
    # Calculate statistics by calibration method
    stats_summary = df.groupby(['training_method', 'calibration_method']).agg({
        'ece': ['count', 'mean', 'std'],
        'accuracy': 'mean'
    }).round(4)
    
    print("📊 CALIBRATION RESULTS SUMMARY")
    print("="*50)
    print(stats_summary)
    
    # Statistical tests
    print("\n🔬 STATISTICAL SIGNIFICANCE TESTS")
    print("="*50)
    
    for training_method in df['training_method'].unique():
        method_data = df[df['training_method'] == training_method]
        
        # Get ECE values for each calibration method
        cal_methods = ['Uncalibrated', 'Temperature', 'Isotonic', 'Geometric']
        method_eces = {}
        
        for cal_method in cal_methods:
            cal_data = method_data[method_data['calibration_method'] == cal_method]
            if len(cal_data) > 0:
                method_eces[cal_method] = cal_data['ece'].values
        
        print(f"\n{training_method}:")
        print("-" * 30)
        
        # Pairwise comparisons with Geometric
        if 'Geometric' in method_eces:
            geometric_ece = method_eces['Geometric']
            
            for other_method, other_ece in method_eces.items():
                if other_method != 'Geometric' and len(other_ece) > 1:
                    if len(geometric_ece) == len(other_ece):
                        t_stat, p_val = stats.ttest_rel(other_ece, geometric_ece)
                        test_type = "paired"
                    else:
                        t_stat, p_val = stats.ttest_ind(other_ece, geometric_ece)
                        test_type = "independent"
                    
                    improvement = (np.mean(other_ece) - np.mean(geometric_ece)) / np.mean(other_ece) * 100
                    significance = "✅ SIGNIFICANT" if p_val < 0.05 else "❌ Not significant"
                    
                    print(f"  {other_method} vs Geometric ({test_type}):")
                    print(f"    Mean ECE: {np.mean(other_ece):.4f} vs {np.mean(geometric_ece):.4f}")
                    print(f"    Improvement: {improvement:+.1f}%")
                    print(f"    p-value: {p_val:.4f} - {significance}")
    
    return df

# Usage
if __name__ == "__main__":
    results = quick_analysis(
        "/home/ptamar/geometric-internal-calibration/aaai_full_experiments/phase2_calibration/results"
    ) 