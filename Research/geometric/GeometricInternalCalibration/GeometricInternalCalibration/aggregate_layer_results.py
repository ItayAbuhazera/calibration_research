import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys

def aggregate_layer_selection_results(base_results_dir):
    """
    Aggregates layer selection ECE results from a nested directory structure.
    
    Searches for all 'per_layer_ground_truth.json' files, finds the
    best and worst layer ECEs, and pairs them with the 'geometric_physical_space'
    (data layer) ECE from 'baseline_results.json'.
    
    Calculates mean and std across all seeds for each 
    (loss, dataset, model) group.
    """
    
    # Use Pathlib for easy path manipulation
    base_path = Path(base_results_dir)
    if not base_path.is_dir():
        print(f"Error: Base directory not found: {base_path}")
        return
        
    # This will store { (group_key): { metric: [val1, val2, ...] } }
    # e.g., { ('cross_entropy', 'cifar10', 'resnet18'): { 'best': [0.1, 0.2], ... } }
    results_store = defaultdict(lambda: defaultdict(list))
    
    # Recursively find all 'per_layer_ground_truth.json' files
    # The glob pattern '*/' ensures we go deep enough for all components
    search_pattern = '*/*/*/*/per_layer_ground_truth.json'
    
    print(f"Searching in: {base_path / search_pattern}\n")
    
    found_files = list(base_path.glob(search_pattern))
    
    if not found_files:
        print("No 'per_layer_ground_truth.json' files found with the expected structure.")
        print("Please check your `base_results_dir` and directory structure.")
        return

    print(f"Found {len(found_files)} result sets to aggregate...")

    for layer_file in found_files:
        try:
            # Get the parent directory (which is the 'seedXX' directory)
            seed_dir = layer_file.parent
            baseline_file = seed_dir / 'baseline_results.json'
            
            # Extract keys from the path
            # The structure is: base / loss / dataset / model / seed / file
            parts = layer_file.parts
            loss_fn = parts[-5]
            dataset = parts[-4]
            model = parts[-3]
            # seed = parts[-2] # We don't need the seed name, just its existence
            
            group_key = (loss_fn, dataset, model)

            # 1. Process Layer ECEs (Best & Worst)
            with open(layer_file, 'r') as f:
                layer_data = json.load(f)
            
            # Get all ECE values from the 'test' dictionary
            ece_values = layer_data.get('test', {}).get('ece', {}).values()
            
            if not ece_values:
                print(f"Warning: No 'test' or 'ece' data in {layer_file}. Skipping.")
                continue
                
            best_layer_ece = min(ece_values)
            worst_layer_ece = max(ece_values)
            
            # 2. Process Baseline ECE (Data Layer)
            if not baseline_file.exists():
                print(f"Warning: 'baseline_results.json' not found in {seed_dir}. Skipping.")
                continue

            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
            
            # Find the 'data layer' ECE
            data_layer_ece = baseline_data.get('geometric_physical_space', {}).get('ece')
            
            if data_layer_ece is None:
                print(f"Warning: 'geometric_physical_space' ECE not found in {baseline_file}. Skipping.")
                continue

            # 3. Store the results for this seed
            results_store[group_key]['best_layer'].append(best_layer_ece)
            results_store[group_key]['worst_layer'].append(worst_layer_ece)
            results_store[group_key]['data_layer'].append(data_layer_ece)

        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON in {layer_file}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred for {layer_file}: {e}. Skipping.")

    if not results_store:
        print("No valid results were collected.")
        return

    # 4. Process collected data into a final report (list of dicts)
    final_report = []
    
    for group_key, metrics in results_store.items():
        loss_fn, dataset, model = group_key
        
        for metric_name, values in metrics.items():
            if values:
                mean = np.mean(values)
                std = np.std(values)
                n_seeds = len(values)
                
                final_report.append({
                    'Loss': loss_fn,
                    'Dataset': dataset,
                    'Model': model,
                    'Metric': metric_name,
                    'Mean ECE': mean,
                    'Std ECE': std,
                    'N Seeds': n_seeds
                })

    # 5. Convert to DataFrame for easy viewing
    df = pd.DataFrame(final_report)
    df = df.sort_values(by=['Loss', 'Dataset', 'Model', 'Metric'])
    
    return df

# --- SCRIPT EXECUTION ---

if __name__ == "__main__":
    # Define the base directory containing all your results
    # This path is relative to where you run the script
    RESULTS_DIR = 'aaai_full_experiments/results/layer_selection_analysis_baselines_4'
    
    # Run the aggregation
    results_df = aggregate_layer_selection_results(RESULTS_DIR)
    
    if results_df is not None:
        print("\n--- Aggregated Results ---")
        
        # Set display options for clear output
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.6f}'.format)
        
        print(results_df)
        
        # Optional: Save to CSV
        output_csv = 'layer_selection_summary.csv'
        results_df.to_csv(output_csv, index=False, float_format='%.6f')
        print(f"\nResults also saved to {output_csv}")