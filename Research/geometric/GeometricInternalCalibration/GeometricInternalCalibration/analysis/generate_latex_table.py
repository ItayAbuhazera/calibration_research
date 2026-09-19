#!/usr/bin/env python3
"""
Generate publication-ready LaTeX tables
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_latex_table(results_df, output_file="calibration_table.tex"):
    """Create publication-ready LaTeX table"""
    
    # Calculate statistics
    stats = results_df.groupby(['training_method', 'calibration_method']).agg({
        'ece': ['mean', 'std', 'count'],
        'accuracy': 'mean'
    }).round(4)
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns]
    stats = stats.reset_index()
    
    # Create table string
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Post-Hoc Calibration Results: Expected Calibration Error (ECE)}",
        "\\label{tab:calibration_results}",
        "\\begin{tabular}{l|cccc}",
        "\\toprule",
        "Training Method & Uncalibrated & Temperature & Isotonic & Geometric \\\\",
        "\\midrule"
    ]
    
    # Group by training method
    for training_method in stats['training_method'].unique():
        method_stats = stats[stats['training_method'] == training_method]
        
        row_data = [training_method.replace('_', '\\_')]
        
        for cal_method in ['Uncalibrated', 'Temperature', 'Isotonic', 'Geometric']:
            cal_stats = method_stats[method_stats['calibration_method'] == cal_method]
            
            if len(cal_stats) > 0:
                mean_ece = cal_stats['ece_mean'].iloc[0]
                std_ece = cal_stats['ece_std'].iloc[0]
                n_seeds = int(cal_stats['ece_count'].iloc[0])
                
                # Calculate 95% CI
                se = std_ece / np.sqrt(n_seeds)
                ci_95 = 1.96 * se
                
                # Format: mean ± CI
                result_str = f"{mean_ece:.3f} ± {ci_95:.3f}"
                row_data.append(result_str)
            else:
                row_data.append("--")
        
        latex_lines.append(" & ".join(row_data) + " \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"📄 LaTeX table saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    from quick_analysis import quick_analysis
    
    # Load results
    results_df = quick_analysis(
        "/home/ptamar/geometric-internal-calibration/aaai_full_experiments/phase2_calibration/results"
    )
    
    # Generate LaTeX table
    create_latex_table(results_df, "analysis/calibration_table.tex") 