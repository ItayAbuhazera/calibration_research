"""Generate LaTeX timing table from experiment results."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse


def generate_timing_table(results_dir: str) -> str:
    """Generate LaTeX table for paper."""
    
    # Load all results
    results = []
    results_path = Path(results_dir)
    
    if results_path.is_file():
        # Single file
        with open(results_path) as f:
            results.extend(json.load(f))
    elif results_path.is_dir():
        # Directory - load all JSON files
        for f in results_path.glob('*_timing.json'):
            with open(f) as fp:
                results.extend(json.load(fp))
    else:
        raise ValueError(f"Results path does not exist: {results_dir}")
    
    if not results:
        raise ValueError(f"No results found in {results_dir}")
    
    # Aggregate by dataset/model
    aggregated = {}
    for r in results:
        key = (r['dataset'], r['model'])
        if key not in aggregated:
            aggregated[key] = {'sklearn': [], 'faiss': [], 'speedup': []}
        aggregated[key]['sklearn'].append(r['sklearn_timing'])
        aggregated[key]['faiss'].append(r['faiss_timing'])
        aggregated[key]['speedup'].append(r['speedup'])
    
    # Generate LaTeX
    latex = r"""\begin{table}[t]
\centering
\caption{Timing comparison: Original SGC vs FAISS-accelerated SGC. 
Offline includes index building and isotonic fitting. 
Online is per-batch kNN query + isotonic transform.
Times in seconds, averaged over multiple runs.}
\label{tab:timing_comparison}
\small
\begin{tabular}{llcccccc}
\toprule
& & \multicolumn{2}{c}{Offline (s)} & \multicolumn{2}{c}{Online (s)} & \multicolumn{2}{c}{Speedup} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}
Dataset & Model & sklearn & FAISS & sklearn & FAISS & Offline & Online \\
\midrule
"""
    
    for (dataset, model), data in sorted(aggregated.items()):
        # Calculate means and standard deviations
        sklearn_offline = [d['total_offline_time_s'] for d in data['sklearn']]
        faiss_offline = [d['total_offline_time_s'] for d in data['faiss']]
        sklearn_online = [d['total_online_time_s'] for d in data['sklearn']]
        faiss_online = [d['total_online_time_s'] for d in data['faiss']]
        speedup_offline = [d['offline'] for d in data['speedup']]
        speedup_online = [d['online'] for d in data['speedup']]
        
        sklearn_offline_mean = np.mean(sklearn_offline)
        faiss_offline_mean = np.mean(faiss_offline)
        sklearn_online_mean = np.mean(sklearn_online)
        faiss_online_mean = np.mean(faiss_online)
        speedup_offline_mean = np.mean(speedup_offline)
        speedup_online_mean = np.mean(speedup_online)
        
        # Format with appropriate precision
        latex += f"{dataset} & {model} & "
        latex += f"{sklearn_offline_mean:.2f} & {faiss_offline_mean:.2f} & "
        latex += f"{sklearn_online_mean:.3f} & {faiss_online_mean:.3f} & "
        latex += f"{speedup_offline_mean:.1f}x & {speedup_online_mean:.1f}x \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_detailed_timing_table(results_dir: str) -> str:
    """Generate detailed LaTeX table with breakdown of offline/online components."""
    
    # Load all results
    results = []
    results_path = Path(results_dir)
    
    if results_path.is_file():
        with open(results_path) as f:
            results.extend(json.load(f))
    elif results_path.is_dir():
        for f in results_path.glob('*_timing.json'):
            with open(f) as fp:
                results.extend(json.load(fp))
    else:
        raise ValueError(f"Results path does not exist: {results_dir}")
    
    if not results:
        raise ValueError(f"No results found in {results_dir}")
    
    # Aggregate by dataset/model
    aggregated = {}
    for r in results:
        key = (r['dataset'], r['model'])
        if key not in aggregated:
            aggregated[key] = {'sklearn': [], 'faiss': [], 'speedup': []}
        aggregated[key]['sklearn'].append(r['sklearn_timing'])
        aggregated[key]['faiss'].append(r['faiss_timing'])
        aggregated[key]['speedup'].append(r['speedup'])
    
    # Generate LaTeX
    latex = r"""\begin{table}[t]
\centering
\caption{Detailed timing breakdown: Original SGC vs FAISS-accelerated SGC.}
\label{tab:timing_detailed}
\small
\begin{tabular}{llcccccccc}
\toprule
& & \multicolumn{4}{c}{Offline (s)} & \multicolumn{4}{c}{Online (s)} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-10}
Dataset & Model & \multicolumn{2}{c}{Index Build} & \multicolumn{2}{c}{Isotonic Fit} & \multicolumn{2}{c}{kNN Query} & \multicolumn{2}{c}{Isotonic Transform} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}
& & sklearn & FAISS & sklearn & FAISS & sklearn & FAISS & sklearn & FAISS \\
\midrule
"""
    
    for (dataset, model), data in sorted(aggregated.items()):
        sklearn_index = np.mean([d['index_build_time_s'] for d in data['sklearn']])
        faiss_index = np.mean([d['index_build_time_s'] for d in data['faiss']])
        sklearn_isotonic_fit = np.mean([d['isotonic_fit_time_s'] for d in data['sklearn']])
        faiss_isotonic_fit = np.mean([d['isotonic_fit_time_s'] for d in data['faiss']])
        sklearn_knn = np.mean([d['knn_query_time_s'] for d in data['sklearn']])
        faiss_knn = np.mean([d['knn_query_time_s'] for d in data['faiss']])
        sklearn_isotonic_transform = np.mean([d['isotonic_transform_time_s'] for d in data['sklearn']])
        faiss_isotonic_transform = np.mean([d['isotonic_transform_time_s'] for d in data['faiss']])
        
        latex += f"{dataset} & {model} & "
        latex += f"{sklearn_index:.2f} & {faiss_index:.2f} & "
        latex += f"{sklearn_isotonic_fit:.3f} & {faiss_isotonic_fit:.3f} & "
        latex += f"{sklearn_knn:.3f} & {faiss_knn:.3f} & "
        latex += f"{sklearn_isotonic_transform:.4f} & {faiss_isotonic_transform:.4f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX timing tables')
    parser.add_argument('--results_dir', type=str, default='Results/timing_comparison',
                        help='Directory or file containing timing results')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path (default: print to stdout)')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate detailed breakdown table instead of summary')
    args = parser.parse_args()
    
    if args.detailed:
        table = generate_detailed_timing_table(args.results_dir)
    else:
        table = generate_timing_table(args.results_dir)
    
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(table)
        print(f"Table written to {output_path}")
    else:
        print(table)


if __name__ == '__main__':
    main()
