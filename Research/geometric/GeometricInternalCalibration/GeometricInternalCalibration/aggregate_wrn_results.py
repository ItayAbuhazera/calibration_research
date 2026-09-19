import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# Base path
base_path = Path("/home/ptamar/geometric-internal-calibration/aaai_full_experiments/results/layer_selection_analysis_baselines_4")

# Pattern: acc{75,85,95}/baseline_cross_entropy/cifar10/wide-resnet28-10/seed*/multi_composite_analysis.json
results_by_acc = defaultdict(lambda: {
    'empirical_best_ece': [],
    'empirical_best_layer': [],
    'geometric_physical_ece': [],
    'geometric_physical_acc': [],
    'geometric_physical_brier': []
})

# Collect results
for acc_level in ['acc75', 'acc85', 'acc95']:
    pattern = f"{acc_level}/baseline_cross_entropy/cifar10/wide-resnet28-10/seed*/multi_composite_analysis.json"
    
    for json_file in base_path.glob(pattern):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Extract values
            results_by_acc[acc_level]['empirical_best_ece'].append(data['empirical_best_ece'])
            results_by_acc[acc_level]['empirical_best_layer'].append(data['empirical_best_layer'])
            
            # Check if baselines exists (it might be in a different structure if something failed)
            if 'baselines' in data and 'geometric_physical_space' in data['baselines']:
                geo_phys = data['baselines']['geometric_physical_space']
                results_by_acc[acc_level]['geometric_physical_ece'].append(geo_phys['ece'])
                results_by_acc[acc_level]['geometric_physical_acc'].append(geo_phys['acc'])
                results_by_acc[acc_level]['geometric_physical_brier'].append(geo_phys['brier'])
            else:
                 print(f"Warning: 'geometric_physical_space' not found in {json_file}")
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

# Print results
print("=" * 80)
print("Wide-ResNet-28-10 on CIFAR-10 - Aggregated Results")
print("=" * 80)

for acc_level in ['acc75', 'acc85', 'acc95']:
    data = results_by_acc[acc_level]
    n_seeds = len(data['empirical_best_ece'])
    
    if n_seeds == 0:
        print(f"\n{acc_level.upper()}: No data found")
        continue
    
    print(f"\n{acc_level.upper()} ({n_seeds} seeds)")
    print("-" * 40)
    
    # Empirical Best
    if len(data['empirical_best_ece']) > 0:
        print(f"Empirical Best Layer:")
        print(f"  ECE:   {np.mean(data['empirical_best_ece']):.4f} ± {np.std(data['empirical_best_ece']):.4f}")
        print(f"  Layer: {np.mean(data['empirical_best_layer']):.1f} ± {np.std(data['empirical_best_layer']):.1f}")
    
    # Geometric Physical Space
    if len(data['geometric_physical_ece']) > 0:
        print(f"\nGeometric Physical Space:")
        print(f"  ECE:      {np.mean(data['geometric_physical_ece']):.4f} ± {np.std(data['geometric_physical_ece']):.4f}")
        print(f"  Accuracy: {np.mean(data['geometric_physical_acc']):.4f} ± {np.std(data['geometric_physical_acc']):.4f}")
        print(f"  Brier:    {np.mean(data['geometric_physical_brier']):.4f} ± {np.std(data['geometric_physical_brier']):.4f}")

print("\n" + "=" * 80)

# Summary table for Slack message
print("\nSUMMARY FOR SLACK MESSAGE:")
print("-" * 40)
for acc_level in ['acc75', 'acc85', 'acc95']:
    data = results_by_acc[acc_level]
    if len(data['empirical_best_ece']) > 0:
        mean_ece = np.mean(data['empirical_best_ece'])
        # Avoid crashing if geometric_physical_acc is empty but empirical_best_ece is not (though they should align)
        if len(data['geometric_physical_acc']) > 0:
             mean_acc = np.mean(data['geometric_physical_acc'])
        else:
             mean_acc = 0.0 # Placeholder
        print(f"דיוק {acc_level[3:]}%: ECE = {mean_ece:.3f}")

# Add this at the very end for quick copy-paste
print("\nONE-LINE SUMMARY:")
for acc_level in ['acc75', 'acc85', 'acc95']:
    data = results_by_acc[acc_level]
    if len(data['empirical_best_ece']) > 0:
        print(f"• דיוק {acc_level[3:]}%: ECE = {np.mean(data['empirical_best_ece']):.3f}")

