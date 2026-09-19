import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from create_oracle_symlinks import ConfigurationSelector

# Mock ConfigurationSelector class to test just the reporting function
class MockSelector(ConfigurationSelector):
    def __init__(self, architecture_data):
        self.architecture_data = architecture_data
        self.ood_analysis = True

def run_verification():
    output_dir = Path("verification_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    # Mock Data
    # Config 1: Good OOD improvement (should be in positive list)
    # Config 2: Bad OOD improvement (should NOT be in positive list)
    
    arch_key = ('cifar10', 'resnet18', 'standard')
    
    # Architecture Data Structure
    # (L, d) -> {arch_key -> list of seed results}
    architecture_data = {
        (6, 256): {
            arch_key: [
                {
                    'baselines': {
                        'geometric_physical_space': {'ood_auroc': 0.8, 'ood_fpr95': 0.5},
                        'isotonic_toplabel': {'ood_auroc': 0.7, 'ood_fpr95': 0.6}
                    },
                    'result': {}
                }
            ]
        },
        (2, 128): {
            arch_key: [
                 {
                    'baselines': {
                        'geometric_physical_space': {'ood_auroc': 0.9, 'ood_fpr95': 0.2},
                        'isotonic_toplabel': {'ood_auroc': 0.85, 'ood_fpr95': 0.3}
                    },
                    'result': {}
                }
            ]
        }
    }

    # Metrics Structure
    metrics = {
        (6, 256): {
            'num_layers': 6,
            'target_dim': 256,
            'n_architectures': 1,
            'n_total_experiments': 1,
            'arch_stats': [
                {
                    'arch_key': arch_key,
                    'mean_ece': 0.05,
                    'std_ece': 0.01,
                    'mean_ood_auroc': 0.85, # Better than 0.8 and 0.7
                    'std_ood_auroc': 0.01,
                    'mean_ood_fpr95': 0.4,  # Better than 0.5 and 0.6
                    'std_ood_fpr95': 0.01,
                    'n_seeds': 1,
                    'mean_samples_per_sec': 100
                }
            ]
        },
        (2, 128): {
            'num_layers': 2,
            'target_dim': 128,
            'n_architectures': 1,
            'n_total_experiments': 1,
            'arch_stats': [
                {
                    'arch_key': arch_key,
                    'mean_ece': 0.1,
                    'std_ece': 0.01,
                    'mean_ood_auroc': 0.88, # Worse than GeoPhys (0.9), Better than Isotonic (0.85)
                    'std_ood_auroc': 0.01,
                    'mean_ood_fpr95': 0.25, # Worse than GeoPhys (0.2), Better than Isotonic (0.3)
                    'std_ood_fpr95': 0.01,
                    'n_seeds': 1,
                    'mean_samples_per_sec': 120
                }
            ]
        }
    }

    selector = MockSelector(architecture_data)
    ranked = [] # Not testing ranking output

    print("Running generate_detailed_report...")
    selector.generate_detailed_report(metrics, ranked, str(output_dir))

    # Check Output Files
    ood_stats_file = output_dir / "per_architecture_ood_statistics.csv"
    ood_pos_file = output_dir / "per_architecture_ood_positive_improvements.csv"

    print(f"\nChecking {ood_stats_file}...")
    if ood_stats_file.exists():
        df_stats = pd.read_csv(ood_stats_file)
        print("Columns found:", df_stats.columns.tolist())
        
        expected_cols = [
            'ood_auroc_improvement_vs_geophys_pct',
            'ood_auroc_improvement_vs_isotonic_pct',
            'ood_fpr95_improvement_vs_geophys_pct', 
            'ood_fpr95_improvement_vs_isotonic_pct'
        ]
        
        missing = [c for c in expected_cols if c not in df_stats.columns]
        if missing:
            print(f"❌ Missing columns: {missing}")
        else:
            print("✅ All expected columns present.")
            
        # Check calculations for (6, 256)
        # AUROC vs GeoPhys: (0.85 - 0.8) / 0.8 * 100 = 6.25%
        # AUROC vs Isotonic: (0.85 - 0.7) / 0.7 * 100 = 21.428...%
        row_6_256 = df_stats[df_stats['num_layers'] == 6].iloc[0]
        print(f"Stats for (6, 256):")
        print(f"  AUROC imp vs GeoPhys: {row_6_256['ood_auroc_improvement_vs_geophys_pct']:.4f}% (Expected ~6.25)")
        print(f"  AUROC imp vs Isotonic: {row_6_256['ood_auroc_improvement_vs_isotonic_pct']:.4f}% (Expected ~21.43)")
        
    else:
        print(f"❌ File not found: {ood_stats_file}")

    print(f"\nChecking {ood_pos_file}...")
    if ood_pos_file.exists():
        df_pos = pd.read_csv(ood_pos_file)
        print(f"Rows in positive list: {len(df_pos)}")
        
        if len(df_pos) == 1 and df_pos.iloc[0]['num_layers'] == 6:
            print("✅ Correctly filtered: Only (6, 256) is present.")
        else:
            print(f"❌ Incorrect filtering. Found rows: \n{df_pos[['num_layers', 'target_dim']]}")
    else:
        print(f"❌ File not found: {ood_pos_file}")

if __name__ == "__main__":
    run_verification()
