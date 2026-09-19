import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import gmean
import pandas as pd
from typing import Dict, List, Tuple

class ConfigurationSelector:
    """
    Select best configuration based on architecture-level comparisons
    """
    
    def __init__(self, results_dir: str, ood_analysis: bool = True):
        self.results_dir = Path(results_dir)
        self.results = []
        self.config_data = defaultdict(list)  # (L, d) -> list of experiments
        self.architecture_data = defaultdict(lambda: defaultdict(list))  # (L, d) -> {arch_key -> list of seed results}
        # Whether to compute and compare OOD metrics (AUROC/FPR95)
        # When False, behaviour matches original ECE-only analysis.
        self.ood_analysis = ood_analysis
        
    def load_results(self):
        """Load all available JSON result files"""
        print(f"Loading results from {self.results_dir}...")
        
        json_files = list(self.results_dir.glob("**/*.json"))
        print(f"Found {len(json_files)} JSON files")
        
        load_errors = []
        
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.results.append(data)
                    else:
                        load_errors.append(f"{json_file.name}: Not a dictionary")
            except json.JSONDecodeError as e:
                load_errors.append(f"{json_file.name}: Invalid JSON - {str(e)}")
            except Exception as e:
                load_errors.append(f"{json_file.name}: {str(e)}")
        
        print(f"Successfully loaded {len(self.results)} result files")
        
        if load_errors:
            print(f"⚠️  Failed to load {len(load_errors)} files:")
            for error in load_errors[:5]:
                print(f"  - {error}")
            if len(load_errors) > 5:
                print(f"  ... and {len(load_errors) - 5} more")
        
        print()
        return len(self.results)
    
    def extract_configuration_data(self):
        """
        Extract all (num_layers, target_dim) configurations
        Group by architecture: (dataset, model, training_method)
        """
        print("Extracting configuration data...")
        
        skipped_files = []
        
        for idx, result in enumerate(self.results):
            try:
                # Validate required keys exist
                if 'experiment_info' not in result:
                    skipped_files.append(f"File {idx}: Missing 'experiment_info'")
                    continue
                
                if 'baselines' not in result:
                    skipped_files.append(f"File {idx}: Missing 'baselines'")
                    continue
                
                if 'random_layer_ablation' not in result:
                    skipped_files.append(f"File {idx}: Missing 'random_layer_ablation'")
                    continue
                
                exp_info = result['experiment_info']
                baselines = result['baselines']
                
                # Validate experiment_info has required fields
                required_fields = ['dataset', 'model_name', 'training_method', 'seed']
                missing_fields = [f for f in required_fields if f not in exp_info]
                if missing_fields:
                    skipped_files.append(f"File {idx}: Missing fields: {missing_fields}")
                    continue
                
                # Architecture key: (dataset, model, training)
                arch_key = (
                    exp_info['dataset'],
                    exp_info['model_name'],
                    exp_info['training_method']
                )
                
                dataset_name = exp_info['dataset']

                # Optional dataset filtering: keep only CIFAR-10 / CIFAR-100 for now.
                # This cleanly drops SVHN (and any other datasets) from downstream analysis.
                if dataset_name not in {"cifar10", "cifar100"}:
                    continue
                
                metadata = {
                    'dataset': dataset_name,
                    'model': exp_info['model_name'],
                    'training': exp_info['training_method'],
                    'seed': exp_info['seed'],
                    'arch_key': arch_key
                }
                
                # Check if random_layer_ablation has results_by_target_dim
                if 'results_by_target_dim' not in result['random_layer_ablation']:
                    skipped_files.append(f"File {idx}: Missing 'results_by_target_dim'")
                    continue
                
                # Extract random layer ablation results
                for target_dim, dim_results in result['random_layer_ablation']['results_by_target_dim'].items():
                    if not isinstance(dim_results, dict):
                        continue
                    
                    for config in dim_results.get('global_random', []):
                        if 'num_layers' not in config or 'ece' not in config:
                            continue
                        
                        key = (config['num_layers'], int(target_dim))
                        
                        data_point = {
                            'ece': config['ece'],
                            'accuracy': config.get('accuracy', 0),
                            # Optional OOD metrics (may be missing for some experiments)
                            'ood_auroc': config.get('ood_auroc', None),
                            'ood_fpr95': config.get('ood_fpr95', None),
                            'samples_per_second': config.get('samples_per_second', 0),
                            'baselines': baselines,
                            'metadata': metadata,
                            'result': result
                        }
                        
                        # Add to both structures
                        self.config_data[key].append(data_point)
                        self.architecture_data[key][arch_key].append(data_point)
            
            except Exception as e:
                skipped_files.append(f"File {idx}: Error - {str(e)}")
                continue
        
        print(f"Found {len(self.config_data)} unique (num_layers, target_dim) configurations")
        print(f"Found {len(set(arch for config_archs in self.architecture_data.values() for arch in config_archs.keys()))} unique architectures")
        
        if skipped_files:
            i=1
            # print(f"\n⚠️  Skipped {len(skipped_files)} files with issues:")
            # for skip_msg in skipped_files[:10]:
            #     # print(f"  - {skip_msg}")
            #     continue
            # if len(skipped_files) > 10:
            #     continue
                # print(f"  ... and {len(skipped_files) - 10} more")
        
        # Print data coverage
        # print("\nData coverage per configuration:")
        # for config, data in sorted(self.config_data.items()):
        #     n_layers, dim = config
        #     n_archs = len(self.architecture_data[config])
        #     print(f"  L={n_layers}, d={dim}: {len(data)} experiments across {n_archs} architectures")
        
        return self.config_data
    
    def validate_data_integrity(self, expected_trials_per_config: int = 10):
        """
        Validate data integrity and detect duplicates or missing trials.
        
        Args:
            expected_trials_per_config: Expected number of trials per (num_layers, target_dim) config
        
        Returns:
            dict: Validation report with warnings and errors
        """
        print("\n" + "="*80)
        print("DATA INTEGRITY VALIDATION")
        print("="*80 + "\n")
        
        validation_report = {
            'total_files': len(self.results),
            'total_configs': len(self.config_data),
            'issues': [],
            'warnings': [],
            'duplicates_found': False,
            'missing_trials_found': False,
        }
        
        # Check for duplicate trial_seeds within each configuration
        print("Checking for duplicate trials...")
        
        for config_key, experiments in self.config_data.items():
            num_layers, target_dim = config_key
            
            # Group by architecture to check seed consistency
            by_arch = defaultdict(list)
            for exp in experiments:
                arch_key = exp['metadata']['arch_key']
                by_arch[arch_key].append(exp)
            
            for arch_key, arch_experiments in by_arch.items():
                dataset, model, training = arch_key
                n_trials = len(arch_experiments)
                
                # Check if trial count matches expected
                if n_trials > expected_trials_per_config:
                    issue = {
                        'type': 'DUPLICATE_TRIALS',
                        'severity': 'ERROR',
                        'config': f"L={num_layers}, d={target_dim}",
                        'architecture': f"{dataset}/{model}/{training}",
                        'expected': expected_trials_per_config,
                        'found': n_trials,
                        'excess': n_trials - expected_trials_per_config
                    }
                    validation_report['issues'].append(issue)
                    validation_report['duplicates_found'] = True
                    
                    # print(f"  ❌ DUPLICATE TRIALS: L={num_layers}, d={target_dim} "
                    #       f"on {dataset}/{model}/{training}")
                    # print(f"     Expected: {expected_trials_per_config} trials, "
                    #       f"Found: {n_trials} trials (+{n_trials - expected_trials_per_config} extra)")
                
                elif n_trials < expected_trials_per_config:
                    warning = {
                        'type': 'MISSING_TRIALS',
                        'severity': 'WARNING',
                        'config': f"L={num_layers}, d={target_dim}",
                        'architecture': f"{dataset}/{model}/{training}",
                        'expected': expected_trials_per_config,
                        'found': n_trials,
                        'missing': expected_trials_per_config - n_trials
                    }
                    validation_report['warnings'].append(warning)
                    validation_report['missing_trials_found'] = True
                    
                    # print(f"  ⚠️  INCOMPLETE: L={num_layers}, d={target_dim} "
                    #       f"on {dataset}/{model}/{training}")
                    # print(f"     Expected: {expected_trials_per_config} trials, "
                    #       f"Found: {n_trials} trials (-{expected_trials_per_config - n_trials} missing)")
                
                # Check for duplicate trial_seeds (indicates same trial run multiple times)
                if len(arch_experiments) > 1:
                    seeds = [exp['metadata']['seed'] for exp in arch_experiments]
                    if len(seeds) != len(set(seeds)):
                        duplicate_seeds = [s for s in set(seeds) if seeds.count(s) > 1]
                        issue = {
                            'type': 'DUPLICATE_SEEDS',
                            'severity': 'CRITICAL',
                            'config': f"L={num_layers}, d={target_dim}",
                            'architecture': f"{dataset}/{model}/{training}",
                            'duplicate_seeds': duplicate_seeds,
                            'message': 'Same seed run multiple times - exact duplicates detected'
                        }
                        validation_report['issues'].append(issue)
                        
                        # print(f"  🚨 CRITICAL: Duplicate seeds detected in L={num_layers}, d={target_dim}")
                        # print(f"     Architecture: {dataset}/{model}/{training}")
                        # print(f"     Duplicate seeds: {duplicate_seeds}")
        
        # Summary statistics
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        n_errors = len([i for i in validation_report['issues'] if i['severity'] in ['ERROR', 'CRITICAL']])
        n_warnings = len(validation_report['warnings'])
        
        # if n_errors == 0 and n_warnings == 0:
        #     print("✅ All checks passed! Data integrity is good.")
        # else:
        #     if n_errors > 0:
        #         print(f"❌ Found {n_errors} errors (duplicates or critical issues)")
        #     if n_warnings > 0:
        #         print(f"⚠️  Found {n_warnings} warnings (incomplete data)")
        
        # Configuration-level summary
        print("\nTrial counts per configuration:")
        config_trial_counts = defaultdict(lambda: defaultdict(int))
        for config_key, experiments in self.config_data.items():
            num_layers, target_dim = config_key
            by_arch = defaultdict(list)
            for exp in experiments:
                arch_key = exp['metadata']['arch_key']
                by_arch[arch_key].append(exp)
            
            for arch_key, arch_exps in by_arch.items():
                config_trial_counts[config_key][len(arch_exps)] += 1
        
        for config_key in sorted(config_trial_counts.keys()):
            num_layers, target_dim = config_key
            counts = config_trial_counts[config_key]
            
            # Format: "L=X, d=Y: N architectures with M trials"
            count_strs = [f"{count} archs with {n_trials} trials" 
                         for n_trials, count in sorted(counts.items())]
            
            status = "✅" if all(n == expected_trials_per_config for n in counts.keys()) else "⚠️"
            # print(f"  {status} L={num_layers}, d={target_dim}: {', '.join(count_strs)}")
        
        return validation_report
    
    def clean_duplicates(self, expected_trials_per_config: int = 10, strategy: str = 'keep_first'):
        """
        Remove duplicate trials from loaded data.
        
        Args:
            expected_trials_per_config: Target number of trials per configuration
            strategy: How to handle duplicates
                     - 'keep_first': Keep first N trials by seed
                     - 'keep_best': Keep N trials with best ECE
                     - 'keep_random': Keep N random trials
        
        Returns:
            dict: Cleaning report showing what was removed
        """
        print("\n" + "="*80)
        print(f"CLEANING DUPLICATES (strategy: {strategy})")
        print("="*80 + "\n")
        
        cleaning_report = {
            'strategy': strategy,
            'configs_cleaned': 0,
            'trials_removed': 0,
            'details': []
        }
        
        cleaned_config_data = defaultdict(list)
        cleaned_architecture_data = defaultdict(lambda: defaultdict(list))
        
        for config_key, experiments in self.config_data.items():
            num_layers, target_dim = config_key
            
            # Group by architecture
            by_arch = defaultdict(list)
            for exp in experiments:
                arch_key = exp['metadata']['arch_key']
                by_arch[arch_key].append(exp)
            
            config_had_duplicates = False
            
            for arch_key, arch_experiments in by_arch.items():
                n_trials = len(arch_experiments)
                
                if n_trials > expected_trials_per_config:
                    config_had_duplicates = True
                    removed = n_trials - expected_trials_per_config
                    
                    # Apply cleaning strategy
                    if strategy == 'keep_first':
                        # Sort by seed and keep first N
                        sorted_exps = sorted(arch_experiments, key=lambda x: x['metadata']['seed'])
                        kept_experiments = sorted_exps[:expected_trials_per_config]
                    
                    elif strategy == 'keep_best':
                        # Sort by ECE and keep best N
                        sorted_exps = sorted(arch_experiments, key=lambda x: x['ece'])
                        kept_experiments = sorted_exps[:expected_trials_per_config]
                    
                    elif strategy == 'keep_random':
                        # Random selection of N trials
                        import random
                        kept_experiments = random.sample(arch_experiments, expected_trials_per_config)
                    
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")
                    
                    dataset, model, training = arch_key
                    # print(f"  Cleaned L={num_layers}, d={target_dim} on {dataset}/{model}/{training}")
                    # print(f"    Removed {removed} duplicate trials, kept {len(kept_experiments)}")
                    
                    cleaning_report['trials_removed'] += removed
                    cleaning_report['details'].append({
                        'config': f"L={num_layers}, d={target_dim}",
                        'architecture': f"{dataset}/{model}/{training}",
                        'removed': removed,
                        'kept': len(kept_experiments)
                    })
                    
                    # Add cleaned experiments
                    for exp in kept_experiments:
                        cleaned_config_data[config_key].append(exp)
                        cleaned_architecture_data[config_key][arch_key].append(exp)
                
                else:
                    # No duplicates, keep all experiments
                    for exp in arch_experiments:
                        cleaned_config_data[config_key].append(exp)
                        cleaned_architecture_data[config_key][arch_key].append(exp)
            
            if config_had_duplicates:
                cleaning_report['configs_cleaned'] += 1
        
        # Replace data structures with cleaned versions
        self.config_data = cleaned_config_data
        self.architecture_data = cleaned_architecture_data
        
        print("\n" + "="*80)
        print("CLEANING SUMMARY")
        print("="*80)
        print(f"Configurations cleaned: {cleaning_report['configs_cleaned']}")
        print(f"Total trials removed: {cleaning_report['trials_removed']}")
        print("✅ Data cleaned successfully")
        
        return cleaning_report
    
    def filter_to_reliable_architectures(self, min_trials_per_arch: int = 6):
        """
        Filter to only include architectures with sufficient trials for reliable statistics.
        
        Args:
            min_trials_per_arch: Minimum number of trials required per architecture
        
        Returns:
            dict: Filtering report showing what was removed
        """
        print("\n" + "="*80)
        print(f"FILTERING TO RELIABLE ARCHITECTURES (min_trials={min_trials_per_arch})")
        print("="*80 + "\n")
        
        filtering_report = {
            'min_trials': min_trials_per_arch,
            'configs_affected': 0,
            'architectures_removed': 0,
            'architectures_kept': 0,
            'details': []
        }
        
        filtered_config_data = defaultdict(list)
        filtered_architecture_data = defaultdict(lambda: defaultdict(list))
        
        for config_key, arch_dict in self.architecture_data.items():
            num_layers, target_dim = config_key
            
            config_archs_before = len(arch_dict)
            config_archs_kept = 0
            config_archs_removed = 0
            
            for arch_key, seed_results in arch_dict.items():
                n_trials = len(seed_results)
                
                if n_trials >= min_trials_per_arch:
                    # Keep this architecture
                    for exp in seed_results:
                        filtered_config_data[config_key].append(exp)
                        filtered_architecture_data[config_key][arch_key].append(exp)
                    config_archs_kept += 1
                else:
                    # Remove this architecture (insufficient trials)
                    config_archs_removed += 1
                    dataset, model, training = arch_key
                    filtering_report['details'].append({
                        'config': f"L={num_layers}, d={target_dim}",
                        'architecture': f"{dataset}/{model}/{training}",
                        'n_trials': n_trials,
                        'reason': f'Only {n_trials} trials (need {min_trials_per_arch})'
                    })
            
            if config_archs_removed > 0:
                filtering_report['configs_affected'] += 1
                # print(f"  L={num_layers}, d={target_dim}: Kept {config_archs_kept}/{config_archs_before} architectures")
            
            filtering_report['architectures_removed'] += config_archs_removed
            filtering_report['architectures_kept'] += config_archs_kept
        
        # Replace data structures with filtered versions
        self.config_data = filtered_config_data
        self.architecture_data = filtered_architecture_data
        
        # print("\n" + "="*80)
        # print("FILTERING SUMMARY")
        # print("="*80)
        # print(f"Architectures kept: {filtering_report['architectures_kept']}")
        # print(f"Architectures removed: {filtering_report['architectures_removed']}")
        # print(f"Configurations affected: {filtering_report['configs_affected']}")
        # print("✅ Filtering complete")
        
        return filtering_report
    
    def compute_paper_statistics(self, config_key=(6, 256)):
        """
        Compute simple statistics for paper reporting.
        Returns mean ± std across ALL trials (not architecture-level aggregation).
        
        This is what should go in your paper table.
        
        Args:
            config_key: (num_layers, target_dim) tuple
        
        Returns:
            dict: Statistics suitable for paper reporting
        """
        num_layers, target_dim = config_key
        
        if config_key not in self.architecture_data:
            print(f"⚠️  Configuration L={num_layers}, d={target_dim} not found!")
            return None
        
        # Get ALL trials for this configuration (across all architectures and seeds)
        all_trials = []
        for arch_key, seed_results in self.architecture_data[config_key].items():
            all_trials.extend(seed_results)
        
        if not all_trials:
            return None
        
        # Extract metrics
        eces = [trial['ece'] for trial in all_trials]
        accuracies = [trial['accuracy'] for trial in all_trials]
        sps_values = [trial['samples_per_second'] for trial in all_trials if trial['samples_per_second'] > 0]
        
        # Overall statistics
        stats = {
            'config': f"L={num_layers}, d={target_dim}",
            'n_trials': len(eces),
            'n_architectures': len(self.architecture_data[config_key]),
            
            # ECE statistics (in percentage)
            'ece_mean_pct': np.mean(eces) * 100,
            'ece_std_pct': np.std(eces) * 100,
            'ece_median_pct': np.median(eces) * 100,
            'ece_min_pct': np.min(eces) * 100,
            'ece_max_pct': np.max(eces) * 100,
            
            # For LaTeX: properly formatted
            'ece_latex': f"{np.mean(eces)*100:.2f} \\pm {np.std(eces)*100:.2f}",
            
            # Accuracy statistics
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            
            # Throughput
            'sps_mean': np.mean(sps_values) if sps_values else 0,
            'sps_std': np.std(sps_values) if sps_values else 0,
        }
        
        # Per-dataset breakdown
        by_dataset = defaultdict(list)
        for trial in all_trials:
            dataset = trial['metadata']['dataset']
            by_dataset[dataset].append(trial['ece'])
        
        stats['by_dataset'] = {}
        for dataset, dataset_eces in by_dataset.items():
            stats['by_dataset'][dataset] = {
                'mean_pct': np.mean(dataset_eces) * 100,
                'std_pct': np.std(dataset_eces) * 100,
                'n': len(dataset_eces),
                'latex': f"{np.mean(dataset_eces)*100:.2f} \\pm {np.std(dataset_eces)*100:.2f}"
            }
        
        return stats
    
    def generate_paper_table(self, configs_to_include=None, output_file="paper_table.csv"):
        """
        Generate the actual table that goes in the paper.
        Compares multiple configurations with simple aggregated statistics.
        
        Args:
            configs_to_include: List of (num_layers, target_dim) tuples to include.
                               If None, includes top 5 from ranking.
            output_file: Where to save the CSV
        
        Returns:
            DataFrame with paper-ready statistics
        """
        print("\n" + "="*80)
        print("PAPER TABLE GENERATION")
        print("="*80 + "\n")
        
        if configs_to_include is None:
            # Use top 5 ranked configurations
            metrics = self.compute_architecture_level_metrics()
            ranked = self.rank_configurations(metrics, min_architectures=3)
            configs_to_include = [(c['num_layers'], c['target_dim']) for c in ranked[:5]]
        
        rows = []
        
        for config_key in configs_to_include:
            num_layers, target_dim = config_key
            stats = self.compute_paper_statistics(config_key)
            
            if stats is None:
                continue
            
            rows.append({
                'Configuration': f"L={num_layers}, d={target_dim}",
                'L': num_layers,
                'd': target_dim,
                'ECE (%)': f"{stats['ece_mean_pct']:.2f} ± {stats['ece_std_pct']:.2f}",
                'ECE_mean': stats['ece_mean_pct'],
                'ECE_std': stats['ece_std_pct'],
                'Accuracy (%)': f"{stats['accuracy_mean']:.2f}",
                'Throughput (sps)': f"{stats['sps_mean']:.1f}",
                'N_trials': stats['n_trials'],
                'N_architectures': stats['n_architectures'],
                'LaTeX': stats['ece_latex']
            })
        
        df = pd.DataFrame(rows)
        
        # Sort by ECE (best first)
        df = df.sort_values('ECE_mean')
        
        # Limit numeric precision in output table (5 decimal digits max)
        df = df.round(5)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"💾 Saved paper table to: {output_file}\n")
        
        # Display nicely
        display_cols = ['Configuration', 'ECE (%)', 'Accuracy (%)', 'Throughput (sps)', 'N_trials', 'N_architectures']
        print(df[display_cols].to_string(index=False))
        
        print("\n" + "="*80)
        print("LATEX TABLE (copy-paste ready)")
        print("="*80 + "\n")
        
        for _, row in df.iterrows():
            print(f"{row['Configuration']:<15} & {row['LaTeX']:<20} & {row['Accuracy (%)']:<8} & {row['Throughput (sps)']:<10} \\\\")
        
        return df
    
    def compare_analysis_methods(self, config_key=(6, 256)):
        """
        Compare architecture-level analysis (for selection) vs simple aggregation (for paper).
        Shows why the two approaches give different insights.
        """
        num_layers, target_dim = config_key
        
        print("\n" + "="*80)
        print(f"ANALYSIS COMPARISON FOR L={num_layers}, d={target_dim}")
        print("="*80 + "\n")
        
        # Method 1: Simple aggregation (for paper)
        paper_stats = self.compute_paper_statistics(config_key)
        
        # Method 2: Architecture-level (for selection)
        metrics = self.compute_architecture_level_metrics()
        arch_metrics = metrics.get(config_key)
        
        if not paper_stats or not arch_metrics:
            print("Configuration not found!")
            return
        
        print("METHOD 1: Simple Aggregation (What goes in PAPER)")
        print("-" * 80)
        print(f"  Mean ECE across ALL trials: {paper_stats['ece_mean_pct']:.2f}% ± {paper_stats['ece_std_pct']:.2f}%")
        print(f"  Total trials: {paper_stats['n_trials']}")
        print(f"  Interpretation: Average performance across all experiments")
        print()
        
        print("METHOD 2: Architecture-Level Analysis (For SELECTION)")
        print("-" * 80)
        print(f"  Win rate vs GeoPhys: {arch_metrics['win_rates'].get('geometric_physical_space', 0)*100:.1f}%")
        print(f"  Win rate vs Isotonic: {arch_metrics['win_rates'].get('isotonic_toplabel', 0)*100:.1f}%")
        print(f"  Number of architectures: {arch_metrics['n_architectures']}")
        print(f"  Interpretation: How often this config beats baselines across diverse settings")
        print()
        
        print("WHY BOTH MATTER:")
        print("-" * 80)
        print("  • Win rates (Method 2) → Choose which L and d to use")
        print("  • Mean ± std (Method 1) → Report final performance in paper")
        print("  • Architecture-level analysis accounts for statistical reliability")
        print("  • Simple aggregation gives the actual expected performance")
        print()
    
    def export_validation_report(self, validation_report: dict, output_dir: str = "."):
        """Export validation report to JSON and human-readable text"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        json_path = output_dir / "data_validation_report.json"
        with open(json_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        print(f"\n💾 Saved validation report (JSON): {json_path}")
        
        # Save human-readable report
        txt_path = output_dir / "data_validation_report.txt"
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATA VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total files analyzed: {validation_report['total_files']}\n")
            f.write(f"Total configurations: {validation_report['total_configs']}\n")
            f.write(f"Duplicates found: {validation_report['duplicates_found']}\n")
            f.write(f"Missing trials found: {validation_report['missing_trials_found']}\n\n")
            
            if validation_report['issues']:
                f.write("ERRORS/ISSUES:\n")
                f.write("-"*80 + "\n")
                for issue in validation_report['issues']:
                    f.write(f"\n[{issue['severity']}] {issue['type']}\n")
                    for key, value in issue.items():
                        if key not in ['type', 'severity']:
                            f.write(f"  {key}: {value}\n")
            
            if validation_report['warnings']:
                f.write("\nWARNINGS:\n")
                f.write("-"*80 + "\n")
                for warning in validation_report['warnings']:
                    f.write(f"\n[{warning['severity']}] {warning['type']}\n")
                    for key, value in warning.items():
                        if key not in ['type', 'severity']:
                            f.write(f"  {key}: {value}\n")
        
        print(f"💾 Saved validation report (text): {txt_path}")
    
    def compare_to_fixed_selections(self, config_key: Tuple[int, int], arch_key: Tuple, 
                                     mean_ece: float, seed_results: List) -> dict:
        """
        Compare a random configuration to fixed layer selections
        Uses mean ECE across seeds vs fixed selection ECE
        """
        num_layers, target_dim = config_key
        
        # Get a representative result (they should all have same fixed results)
        if not seed_results:
            return {}
        
        result = seed_results[0]['result']
        comparisons = {}
        
        # Compare to geo_comb (DAC fixed layers)
        geo_comb_results = result.get('geo_comb', {}).get('results', [])
        for fixed_config in geo_comb_results:
            if fixed_config['target_dim'] == target_dim:
                comparisons['geo_comb_fixed'] = {
                    'fixed_ece': fixed_config['ece'],
                    'random_wins': mean_ece < fixed_config['ece'],
                    'improvement': (fixed_config['ece'] - mean_ece) / fixed_config['ece'] * 100
                }
                break
        
        # Compare to tulip_comb (TULIP fixed layers)
        tulip_comb_results = result.get('tulip_comb', {}).get('results', [])
        for fixed_config in tulip_comb_results:
            if fixed_config['target_dim'] == target_dim:
                comparisons['tulip_comb_fixed'] = {
                    'fixed_ece': fixed_config['ece'],
                    'random_wins': mean_ece < fixed_config['ece'],
                    'improvement': (fixed_config['ece'] - mean_ece) / fixed_config['ece'] * 100
                }
                break
        
        return comparisons
    
    def compute_architecture_level_metrics(self):
        """
        Compute metrics at architecture level (dataset, model, training)
        Compare mean ECE across seeds
        """
        print("\nComputing architecture-level metrics...")
        
        results = {}
        
        use_ood = getattr(self, "ood_analysis", False)
        
        for config, arch_dict in self.architecture_data.items():
            num_layers, target_dim = config
            
            # Track wins vs each baseline (by architecture)
            wins_per_baseline = defaultdict(lambda: {'wins': 0, 'total': 0, 'improvements': []})
            fixed_comparisons = defaultdict(lambda: {'wins': 0, 'total': 0, 'improvements': []})
            # OOD win tracking (by architecture)
            ood_auroc_wins = defaultdict(lambda: {'wins': 0, 'total': 0, 'improvements': []})
            ood_fpr95_wins = defaultdict(lambda: {'wins': 0, 'total': 0, 'improvements': []})
            
            # Architecture-level statistics
            arch_stats = []
            all_eces = []
            all_samples_per_sec = []
            all_ood_aurocs = []
            all_ood_fpr95s = []
            
            for arch_key, seed_results in arch_dict.items():
                if not seed_results:
                    continue
                
                # Compute mean and std across seeds for this architecture
                eces = [r['ece'] for r in seed_results]
                samples_per_sec = [r['samples_per_second'] for r in seed_results if r['samples_per_second'] > 0]
                if use_ood:
                    ood_aurocs = [r['ood_auroc'] for r in seed_results if r.get('ood_auroc') is not None]
                    ood_fpr95s = [r['ood_fpr95'] for r in seed_results if r.get('ood_fpr95') is not None]
                else:
                    ood_aurocs = []
                    ood_fpr95s = []
                
                mean_ece = np.mean(eces)
                std_ece = np.std(eces)
                mean_sps = np.mean(samples_per_sec) if samples_per_sec else 0
                
                all_eces.extend(eces)
                if samples_per_sec:
                    all_samples_per_sec.extend(samples_per_sec)
                if ood_aurocs:
                    all_ood_aurocs.extend(ood_aurocs)
                if ood_fpr95s:
                    all_ood_fpr95s.extend(ood_fpr95s)
                
                arch_stats.append({
                    'arch_key': arch_key,
                    'mean_ece': mean_ece,
                    'std_ece': std_ece,
                    'mean_ood_auroc': np.mean(ood_aurocs) if ood_aurocs else None,
                    'std_ood_auroc': np.std(ood_aurocs) if ood_aurocs else None,
                    'mean_ood_fpr95': np.mean(ood_fpr95s) if ood_fpr95s else None,
                    'std_ood_fpr95': np.std(ood_fpr95s) if ood_fpr95s else None,
                    'n_seeds': len(seed_results),
                    'mean_samples_per_sec': mean_sps
                })
                
                # Compare mean ECE to baselines
                # Use first seed's baselines (they're the same for all seeds in same architecture)
                baselines = seed_results[0]['baselines']
                
                for baseline_name, baseline_data in baselines.items():
                    baseline_ece = baseline_data.get('ece')
                    if baseline_ece is None:
                        # Skip baselines without ECE
                        continue
                    
                    wins_per_baseline[baseline_name]['total'] += 1
                    if mean_ece < baseline_ece:
                        wins_per_baseline[baseline_name]['wins'] += 1
                    
                    rel_imp = (baseline_ece - mean_ece) / baseline_ece * 100 if baseline_ece else 0
                    wins_per_baseline[baseline_name]['improvements'].append(rel_imp)

                    if use_ood:
                        # OOD AUROC (higher is better)
                        baseline_ood_auroc = baseline_data.get('ood_auroc')
                        if baseline_ood_auroc is not None and ood_aurocs:
                            mean_ood_auroc = np.mean(ood_aurocs)
                            ood_auroc_wins[baseline_name]['total'] += 1
                            if mean_ood_auroc > baseline_ood_auroc:
                                ood_auroc_wins[baseline_name]['wins'] += 1
                            
                            rel_imp_auroc = (mean_ood_auroc - baseline_ood_auroc) / baseline_ood_auroc * 100
                            ood_auroc_wins[baseline_name]['improvements'].append(rel_imp_auroc)

                        # OOD FPR95 (lower is better)
                        baseline_ood_fpr95 = baseline_data.get('ood_fpr95')
                        if baseline_ood_fpr95 is not None and ood_fpr95s:
                            mean_ood_fpr95 = np.mean(ood_fpr95s)
                            ood_fpr95_wins[baseline_name]['total'] += 1
                            if mean_ood_fpr95 < baseline_ood_fpr95:
                                ood_fpr95_wins[baseline_name]['wins'] += 1
                            
                            rel_imp_fpr = (baseline_ood_fpr95 - mean_ood_fpr95) / baseline_ood_fpr95 * 100
                            ood_fpr95_wins[baseline_name]['improvements'].append(rel_imp_fpr)
                
                # Compare to fixed selections
                fixed_comp = self.compare_to_fixed_selections(config, arch_key, mean_ece, seed_results)
                for fixed_name, fixed_data in fixed_comp.items():
                    fixed_comparisons[fixed_name]['total'] += 1
                    if fixed_data['random_wins']:
                        fixed_comparisons[fixed_name]['wins'] += 1
                    fixed_comparisons[fixed_name]['improvements'].append(fixed_data['improvement'])
            
            # Compute win rates (by architecture, not by seed)
            win_rates = {
                name: data['wins'] / data['total'] if data['total'] > 0 else 0
                for name, data in wins_per_baseline.items()
            }

            if use_ood:
                ood_auroc_win_rates = {
                    name: data['wins'] / data['total'] if data['total'] > 0 else 0
                    for name, data in ood_auroc_wins.items()
                }
                ood_fpr95_win_rates = {
                    name: data['wins'] / data['total'] if data['total'] > 0 else 0
                    for name, data in ood_fpr95_wins.items()
                }
            else:
                ood_auroc_win_rates = {}
                ood_fpr95_win_rates = {}
            
            mean_improvements = {
                name: np.mean(data['improvements']) if data['improvements'] else 0
                for name, data in wins_per_baseline.items()
            }

            if use_ood:
                ood_auroc_mean_improvements = {
                    name: np.mean(data['improvements']) if data['improvements'] else 0
                    for name, data in ood_auroc_wins.items()
                }
                ood_fpr95_mean_improvements = {
                    name: np.mean(data['improvements']) if data['improvements'] else 0
                    for name, data in ood_fpr95_wins.items()
                }
            else:
                ood_auroc_mean_improvements = {}
                ood_fpr95_mean_improvements = {}
            
            fixed_win_rates = {
                name: data['wins'] / data['total'] if data['total'] > 0 else 0
                for name, data in fixed_comparisons.items()
            }
            
            fixed_mean_improvements = {
                name: np.mean(data['improvements']) if data['improvements'] else 0
                for name, data in fixed_comparisons.items()
            }
            
            results[config] = {
                'num_layers': num_layers,
                'target_dim': target_dim,
                'n_architectures': len(arch_dict),
                'n_total_experiments': len(all_eces),
                'win_rates': win_rates,
                'ood_auroc_win_rates': ood_auroc_win_rates,
                'ood_fpr95_win_rates': ood_fpr95_win_rates,
                'mean_improvements': mean_improvements,
                'ood_auroc_improvements': ood_auroc_mean_improvements,
                'ood_fpr95_improvements': ood_fpr95_mean_improvements,
                'fixed_win_rates': fixed_win_rates,
                'fixed_mean_improvements': fixed_mean_improvements,
                'global_mean_ece': np.mean(all_eces) if all_eces else float('inf'),
                'global_std_ece': np.std(all_eces) if all_eces else float('inf'),
                'global_median_ece': np.median(all_eces) if all_eces else float('inf'),
                'global_mean_ood_auroc': np.mean(all_ood_aurocs) if all_ood_aurocs else None,
                'global_mean_ood_fpr95': np.mean(all_ood_fpr95s) if all_ood_fpr95s else None,
                'mean_samples_per_sec': np.mean(all_samples_per_sec) if all_samples_per_sec else 0,
                'arch_stats': arch_stats
            }
        
        return results
    
    def rank_configurations(self, metrics: dict, min_architectures: int = 4):
        """
        Rank configurations based on architecture-level comparisons
        """
        print("\nRanking configurations...")
        print(f"Filtering for configurations with >= {min_architectures} architectures\n")
        
        ranked = []
        
        for config, data in metrics.items():
            num_layers, target_dim = config
            
            # Skip if insufficient data
            if data['n_architectures'] < 3:
                continue
            
            # PRIMARY: Win rate vs geometric_physical_space
            geo_phys_win_rate = data['win_rates'].get('geometric_physical_space', 0)
            
            # SECONDARY: Win rate vs isotonic_toplabel
            isotonic_win_rate = data['win_rates'].get('isotonic_toplabel', 0)
            
            # THIRD: Win rate vs fixed selections
            geo_comb_win_rate = data['fixed_win_rates'].get('geo_comb_fixed', 0)
            tulip_comb_win_rate = data['fixed_win_rates'].get('tulip_comb_fixed', 0)
            avg_fixed_win_rate = np.mean([geo_comb_win_rate, tulip_comb_win_rate]) if geo_comb_win_rate or tulip_comb_win_rate else 0
            
            # Composite score
            composite_score = (
                0.50 * geo_phys_win_rate +
                0.25 * isotonic_win_rate +
                0.15 * avg_fixed_win_rate +
                0.10 * (1 - min(data['global_std_ece'] / 0.05, 1))
            )
            
            # Efficiency metric (normalized, higher is better)
            max_sps = 200  # Approximate max samples/sec
            efficiency_score = min(data['mean_samples_per_sec'] / max_sps, 1.0)
            
            ranked.append({
                'num_layers': num_layers,
                'target_dim': target_dim,
                'composite_score': composite_score,
                'geo_phys_win_rate': geo_phys_win_rate,
                'isotonic_win_rate': isotonic_win_rate,
                'geo_comb_win_rate': geo_comb_win_rate,
                'tulip_comb_win_rate': tulip_comb_win_rate,
                'avg_fixed_win_rate': avg_fixed_win_rate,
                'mean_ece': data['global_mean_ece'],
                'std_ece': data['global_std_ece'],
                'median_ece': data['global_median_ece'],
                'n_architectures': data['n_architectures'],
                'n_experiments': data['n_total_experiments'],
                'samples_per_sec': data['mean_samples_per_sec'],
                'efficiency_score': efficiency_score,
                'geo_phys_improvement': data['mean_improvements'].get('geometric_physical_space', 0),
                'isotonic_improvement': data['mean_improvements'].get('isotonic_toplabel', 0)
            })
        
        # Sort by composite score
        ranked.sort(key=lambda x: (
            x['composite_score'],
            x['geo_phys_win_rate'],
            x['isotonic_win_rate']
        ), reverse=True)
        
        return ranked
    
    def print_rankings(self, ranked: List[dict], top_n: int = 15):
        """Print top configurations with efficiency metrics"""
        
        print(f"\n{'='*160}")
        print(f"TOP {top_n} CONFIGURATIONS (Architecture-Level Analysis)")
        print(f"{'='*160}\n")
        
        print(f"{'Rank':<5} {'L':<4} {'D':<6} {'Score':<7} "
              f"{'GeoPhys':<10} {'Isotonic':<10} {'GeoComb':<10} {'TULIP':<10} "
              f"{'AvgFixed':<10} {'MeanECE':<10} {'StdECE':<8} {'SPS':<8} {'Archs':<7} {'N':<5}")
        print("-" * 160)
        
        for i, config in enumerate(ranked[:top_n], 1):
            marker = "***" if (config['num_layers'] == 6 and config['target_dim'] == 256) else "   "
            
            print(f"{marker} {i:<2} {config['num_layers']:<4} {config['target_dim']:<6} {config['composite_score']:.4f}  "
                  f"{config['geo_phys_win_rate']*100:>6.1f}%    "
                  f"{config['isotonic_win_rate']*100:>6.1f}%    "
                  f"{config['geo_comb_win_rate']*100:>6.1f}%    "
                  f"{config['tulip_comb_win_rate']*100:>6.1f}%    "
                  f"{config['avg_fixed_win_rate']*100:>6.1f}%    "
                  f"{config['mean_ece']:.5f}  "
                  f"{config['std_ece']:.5f}  "
                  f"{config['samples_per_sec']:>6.1f}  "
                  f"{config['n_architectures']:<7} "
                  f"{config['n_experiments']:<5}")
        
        print("\nColumn descriptions:")
        print("  GeoPhys  = Win rate vs geometric_physical_space (by architecture) [PRIMARY]")
        print("  Isotonic = Win rate vs isotonic_toplabel (by architecture) [SECONDARY]")
        print("  GeoComb  = Win rate vs DAC fixed layers [THIRD]")
        print("  TULIP    = Win rate vs TULIP fixed layers [THIRD]")
        print("  AvgFixed = Average of GeoComb and TULIP")
        print("  MeanECE  = Mean ECE across all experiments")
        print("  StdECE   = Standard deviation of ECE across all experiments")
        print("  SPS      = Mean samples per second (throughput)")
        print("  Archs    = Number of unique architectures")
        print("  N        = Total number of experiments (seeds)")
        print("\n*** marks your target configuration (L=6, d=256)")
    
    def print_performance_efficiency_comparison(self, ranked: List[dict], top_n: int = 5):
        """
        Print focused comparison of top configurations showing performance-efficiency trade-off
        """
        print(f"\n{'='*120}")
        print("PERFORMANCE vs EFFICIENCY TRADE-OFF (Top 5)")
        print(f"{'='*120}\n")
        
        print(f"{'Rank':<6} {'Config':<12} {'GeoPhys':<12} {'Isotonic':<12} "
              f"{'MeanECE':<12} {'SPS':<12} {'Speedup':<12} {'Trade-off':<30}")
        print("-" * 120)
        
        # Use first config as baseline for speedup
        baseline_sps = ranked[0]['samples_per_sec'] if ranked else 1
        
        for i, config in enumerate(ranked[:top_n], 1):
            marker = "***" if (config['num_layers'] == 6 and config['target_dim'] == 256) else "   "
            config_str = f"L={config['num_layers']}, d={config['target_dim']}"
            speedup = config['samples_per_sec'] / baseline_sps if baseline_sps > 0 else 0
            
            # Trade-off assessment
            if i == 1:
                tradeoff = "Baseline (best composite)"
            else:
                geo_diff = (config['geo_phys_win_rate'] - ranked[0]['geo_phys_win_rate']) * 100
                if speedup > 1.2 and abs(geo_diff) < 2:
                    tradeoff = f"{speedup:.1f}x faster, ~same perf"
                elif speedup > 1.5:
                    tradeoff = f"{speedup:.1f}x faster ({geo_diff:+.1f}% geo)"
                elif geo_diff > 2:
                    tradeoff = f"{geo_diff:+.1f}% better geo"
                else:
                    tradeoff = f"Similar overall"
            
            print(f"{marker} #{i:<4} {config_str:<12} "
                  f"{config['geo_phys_win_rate']*100:>6.1f}%      "
                  f"{config['isotonic_win_rate']*100:>6.1f}%      "
                  f"{config['mean_ece']:.5f}   "
                  f"{config['samples_per_sec']:>6.1f} sps  "
                  f"{speedup:>5.2f}x       "
                  f"{tradeoff}")
        
        print("\nKey Insights:")
        
        # Find best GeoPhys
        best_geo = max(ranked[:top_n], key=lambda x: x['geo_phys_win_rate'])
        best_geo_idx = next(i for i, c in enumerate(ranked[:top_n], 1) 
                           if c['num_layers'] == best_geo['num_layers'] and c['target_dim'] == best_geo['target_dim'])
        
        # Find fastest
        fastest = max(ranked[:top_n], key=lambda x: x['samples_per_sec'])
        fastest_idx = next(i for i, c in enumerate(ranked[:top_n], 1) 
                          if c['num_layers'] == fastest['num_layers'] and c['target_dim'] == fastest['target_dim'])
        
        print(f"  • Best GeoPhys win rate: #{best_geo_idx} (L={best_geo['num_layers']}, d={best_geo['target_dim']}) at {best_geo['geo_phys_win_rate']*100:.1f}%")
        print(f"  • Fastest throughput: #{fastest_idx} (L={fastest['num_layers']}, d={fastest['target_dim']}) at {fastest['samples_per_sec']:.1f} sps")
        
        # Check if L=6, d=256 offers good trade-off
        target = next((c for c in ranked[:top_n] if c['num_layers'] == 6 and c['target_dim'] == 256), None)
        if target:
            target_idx = next(i for i, c in enumerate(ranked[:top_n], 1) 
                            if c['num_layers'] == 6 and c['target_dim'] == 256)
            speedup_vs_best = target['samples_per_sec'] / ranked[0]['samples_per_sec'] if ranked[0]['samples_per_sec'] > 0 else 0
            geo_diff_vs_best = (target['geo_phys_win_rate'] - ranked[0]['geo_phys_win_rate']) * 100
            
            if speedup_vs_best > 1.3 and abs(geo_diff_vs_best) < 3:
                print(f"  • L=6, d=256 offers strong trade-off: {speedup_vs_best:.1f}x faster with {geo_diff_vs_best:+.1f}% GeoPhys difference")
            elif target_idx == best_geo_idx:
                print(f"  • L=6, d=256 achieves best GeoPhys performance (PRIMARY criterion)")
    
    def check_target_configuration(self, ranked: List[dict], target_layers: int = 6, target_dim: int = 256):
        """Check where the target configuration ranks with efficiency context"""
        
        print(f"\n{'='*80}")
        print(f"TARGET CONFIGURATION CHECK: L={target_layers}, d={target_dim}")
        print(f"{'='*80}\n")
        
        target_config = None
        target_rank = None
        
        for i, config in enumerate(ranked, 1):
            if config['num_layers'] == target_layers and config['target_dim'] == target_dim:
                target_config = config
                target_rank = i
                break
        
        if target_config is None:
            print(f"❌ Configuration (L={target_layers}, d={target_dim}) not found in rankings!")
            return None
        
        print(f"📊 Rank: #{target_rank} out of {len(ranked)} configurations")
        print(f"\n📈 Performance Metrics (Architecture-Level):")
        print(f"   Composite Score:        {target_config['composite_score']:.4f}")
        print(f"   GeoPhys Win Rate:       {target_config['geo_phys_win_rate']*100:.1f}% [PRIMARY]")
        print(f"   Isotonic Win Rate:      {target_config['isotonic_win_rate']*100:.1f}% [SECONDARY]")
        print(f"   GeoComb Win Rate:       {target_config['geo_comb_win_rate']*100:.1f}% [THIRD]")
        print(f"   TULIP Win Rate:         {target_config['tulip_comb_win_rate']*100:.1f}% [THIRD]")
        print(f"   Avg Fixed Win Rate:     {target_config['avg_fixed_win_rate']*100:.1f}%")
        
        print(f"\n📉 ECE Statistics:")
        print(f"   Mean ECE:               {target_config['mean_ece']:.5f}")
        print(f"   Std ECE:                {target_config['std_ece']:.5f}")
        print(f"   Median ECE:             {target_config['median_ece']:.5f}")
        
        print(f"\n⚡ Computational Efficiency:")
        print(f"   Samples per second:     {target_config['samples_per_sec']:.1f}")
        print(f"   Efficiency score:       {target_config['efficiency_score']:.3f}")
        
        # Compare to #1
        if target_rank > 1 and ranked:
            best = ranked[0]
            speedup = target_config['samples_per_sec'] / best['samples_per_sec'] if best['samples_per_sec'] > 0 else 0
            geo_diff = (target_config['geo_phys_win_rate'] - best['geo_phys_win_rate']) * 100
            ece_diff = (target_config['mean_ece'] - best['mean_ece']) * 100 / best['mean_ece']
            
            print(f"\n📊 Comparison to #1 (L={best['num_layers']}, d={best['target_dim']}):")
            print(f"   GeoPhys difference:     {geo_diff:+.1f} percentage points")
            print(f"   ECE difference:         {ece_diff:+.1f}% relative")
            print(f"   Speed ratio:            {speedup:.2f}x")
            
            if speedup > 1.5 and abs(geo_diff) < 2:
                print(f"   → Strong efficiency trade-off: {speedup:.1f}x faster with ~equal performance")
            elif geo_diff > 0:
                print(f"   → Better PRIMARY criterion performance")
        
        print(f"\n💪 Relative Improvements:")
        print(f"   vs GeoPhys:             {target_config['geo_phys_improvement']:+.1f}%")
        print(f"   vs Isotonic:            {target_config['isotonic_improvement']:+.1f}%")
        
        print(f"\n📊 Data Coverage:")
        print(f"   Number of architectures: {target_config['n_architectures']}")
        print(f"   Total experiments:       {target_config['n_experiments']}")
        
        # Verdict
        print(f"\n{'='*80}")
        if target_rank == 1:
            print("🏆 VERDICT: Optimal configuration across all criteria!")
        elif target_rank <= 3:
            # Check if it's best on PRIMARY
            best_geo = max(ranked[:5], key=lambda x: x['geo_phys_win_rate'])
            if target_config['geo_phys_win_rate'] == best_geo['geo_phys_win_rate']:
                print("✅ VERDICT: Excellent choice! Best on PRIMARY criterion (GeoPhys).")
            else:
                print("✅ VERDICT: Excellent choice! Top 3 configuration.")
        elif target_rank <= 5:
            print("✅ VERDICT: Good choice! Top 5 configuration.")
        elif target_rank <= 10:
            print("⚠️  VERDICT: Acceptable choice, but consider top-ranked alternatives.")
        else:
            print("❌ VERDICT: Not optimal. Consider switching to a higher-ranked configuration.")
        print(f"{'='*80}\n")
        
        return target_config
    
    def generate_detailed_report(self, metrics: dict, ranked: List[dict], output_dir: str = "."):
        """Generate detailed CSV reports"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        include_ood = getattr(self, "ood_analysis", False)
        
        # Full ranking table
        df_ranked = pd.DataFrame(ranked)
        # Limit numeric precision to 5 decimal digits
        df_ranked = df_ranked.round(5)
        df_ranked.to_csv(output_dir / "configuration_ranking_architecture_level.csv", index=False)
        print(f"\n💾 Saved architecture-level ranking to: {output_dir / 'configuration_ranking_architecture_level.csv'}")
        
        # Per-architecture breakdown
        arch_breakdown = []
        for config, data in metrics.items():
            num_layers, target_dim = config
            arch_dict = self.architecture_data.get(config, {})
            
            for arch_stat in data['arch_stats']:
                arch_key = arch_stat['arch_key']
                dataset, model, training = arch_key
                mean_ece = arch_stat['mean_ece']
                
                # Default improvements (relative % vs various baselines)
                imp_vs_geophys = None
                imp_vs_isotonic = None
                imp_vs_geo_comb = None
                imp_vs_tulip = None
                
                seed_results = arch_dict.get(arch_key, [])
                if seed_results:
                    baselines = seed_results[0].get('baselines', {})
                    geo_phys = baselines.get('geometric_physical_space')
                    isotonic = baselines.get('isotonic_toplabel')
                    
                    # Relative improvement vs GeoPhys baseline: (baseline - ours) / baseline * 100
                    if geo_phys and 'ece' in geo_phys and geo_phys['ece'] > 0:
                        imp_vs_geophys = (geo_phys['ece'] - mean_ece) / geo_phys['ece'] * 100
                    
                    # Relative improvement vs Isotonic baseline
                    if isotonic and 'ece' in isotonic and isotonic['ece'] > 0:
                        imp_vs_isotonic = (isotonic['ece'] - mean_ece) / isotonic['ece'] * 100
                    
                    # Relative improvements vs fixed layer selections (GeoComb / TULIP)
                    fixed_comp = self.compare_to_fixed_selections(config, arch_key, mean_ece, seed_results)
                    if 'geo_comb_fixed' in fixed_comp:
                        imp_vs_geo_comb = fixed_comp['geo_comb_fixed'].get('improvement')
                    if 'tulip_comb_fixed' in fixed_comp:
                        imp_vs_tulip = fixed_comp['tulip_comb_fixed'].get('improvement')
                
                arch_breakdown.append({
                    'num_layers': num_layers,
                    'target_dim': target_dim,
                    'dataset': dataset,
                    'model': model,
                    'training': training,
                    'mean_ece': mean_ece,
                    'std_ece': arch_stat['std_ece'],
                    'n_seeds': arch_stat['n_seeds'],
                    'mean_samples_per_sec': arch_stat['mean_samples_per_sec'],
                    'improvement_vs_geophys_pct': imp_vs_geophys,
                    'improvement_vs_isotonic_pct': imp_vs_isotonic,
                    'improvement_vs_geo_comb_pct': imp_vs_geo_comb,
                    'improvement_vs_tulip_pct': imp_vs_tulip,
                })
        
        df_arch = pd.DataFrame(arch_breakdown)
        # Limit numeric precision to 5 decimal digits
        df_arch = df_arch.round(5)
        df_arch.to_csv(output_dir / "per_architecture_statistics.csv", index=False)
        print(f"💾 Saved per-architecture statistics to: {output_dir / 'per_architecture_statistics.csv'}")

        # OOD-focused tables (only if OOD analysis is enabled)
        if include_ood:
            # Configuration-level OOD metrics
            ood_rows = []
            for config, data in metrics.items():
                ood_rows.append({
                    'num_layers': data['num_layers'],
                    'target_dim': data['target_dim'],
                    'n_architectures': data['n_architectures'],
                    'n_experiments': data['n_total_experiments'],
                    'global_mean_ood_auroc': data.get('global_mean_ood_auroc'),
                    'global_mean_ood_fpr95': data.get('global_mean_ood_fpr95'),
                    'ood_auroc_win_vs_geophys': data.get('ood_auroc_win_rates', {}).get('geometric_physical_space'),
                    'ood_auroc_win_vs_isotonic': data.get('ood_auroc_win_rates', {}).get('isotonic_toplabel'),
                    'ood_fpr95_win_vs_geophys': data.get('ood_fpr95_win_rates', {}).get('geometric_physical_space'),
                    'ood_fpr95_win_vs_isotonic': data.get('ood_fpr95_win_rates', {}).get('isotonic_toplabel'),
                    'ood_auroc_improve_vs_geophys_pct': data.get('ood_auroc_improvements', {}).get('geometric_physical_space'),
                    'ood_auroc_improve_vs_isotonic_pct': data.get('ood_auroc_improvements', {}).get('isotonic_toplabel'),
                    'ood_fpr95_improve_vs_geophys_pct': data.get('ood_fpr95_improvements', {}).get('geometric_physical_space'),
                    'ood_fpr95_improve_vs_isotonic_pct': data.get('ood_fpr95_improvements', {}).get('isotonic_toplabel'),
                })

            df_ood = pd.DataFrame(ood_rows).round(5)
            df_ood.to_csv(output_dir / "configuration_ood_metrics.csv", index=False)
            print(f"💾 Saved OOD configuration metrics to: {output_dir / 'configuration_ood_metrics.csv'}")

            # Per-architecture OOD breakdown
            ood_arch_rows = []
            for config, data in metrics.items():
                arch_dict = self.architecture_data.get(config, {})
                for arch_stat in data['arch_stats']:
                    arch_key = arch_stat['arch_key']
                    dataset, model, training = arch_key

                    sum_ood_auroc_imp_geophys = None
                    sum_ood_auroc_imp_isotonic = None
                    sum_ood_fpr95_imp_geophys = None
                    sum_ood_fpr95_imp_isotonic = None

                    seed_results = arch_dict.get(arch_key, [])
                    if seed_results:
                         baselines = seed_results[0].get('baselines', {})
                         geo_phys = baselines.get('geometric_physical_space')
                         isotonic = baselines.get('isotonic_toplabel')

                         mean_ood_auroc = arch_stat.get('mean_ood_auroc')
                         mean_ood_fpr95 = arch_stat.get('mean_ood_fpr95')
                        
                         # Compare OOD AUROC to GeoPhys (higher is better)
                         if mean_ood_auroc is not None and geo_phys and 'ood_auroc' in geo_phys and geo_phys['ood_auroc'] is not None:
                             base_val = geo_phys['ood_auroc']
                             if base_val != 0:
                                 sum_ood_auroc_imp_geophys = (mean_ood_auroc - base_val) / abs(base_val) * 100

                         # Compare OOD AUROC to Isotonic
                         if mean_ood_auroc is not None and isotonic and 'ood_auroc' in isotonic and isotonic['ood_auroc'] is not None:
                             base_val = isotonic['ood_auroc']
                             if base_val != 0:
                                 sum_ood_auroc_imp_isotonic = (mean_ood_auroc - base_val) / abs(base_val) * 100

                         # Compare OOD FPR95 to GeoPhys (lower is better, so (base - ours)/base)
                         if mean_ood_fpr95 is not None and geo_phys and 'ood_fpr95' in geo_phys and geo_phys['ood_fpr95'] is not None:
                             base_val = geo_phys['ood_fpr95']
                             if base_val != 0:
                                 sum_ood_fpr95_imp_geophys = (base_val - mean_ood_fpr95) / abs(base_val) * 100

                         # Compare OOD FPR95 to Isotonic
                         if mean_ood_fpr95 is not None and isotonic and 'ood_fpr95' in isotonic and isotonic['ood_fpr95'] is not None:
                             base_val = isotonic['ood_fpr95']
                             if base_val != 0:
                                 sum_ood_fpr95_imp_isotonic = (base_val - mean_ood_fpr95) / abs(base_val) * 100

                    ood_arch_rows.append({
                        'num_layers': config[0],
                        'target_dim': config[1],
                        'dataset': dataset,
                        'model': model,
                        'training': training,
                        'mean_ood_auroc': arch_stat.get('mean_ood_auroc'),
                        'std_ood_auroc': arch_stat.get('std_ood_auroc'),
                        'mean_ood_fpr95': arch_stat.get('mean_ood_fpr95'),
                        'std_ood_fpr95': arch_stat.get('std_ood_fpr95'),
                        'n_seeds': arch_stat.get('n_seeds'),
                        'ood_auroc_improvement_vs_geophys_pct': sum_ood_auroc_imp_geophys,
                        'ood_auroc_improvement_vs_isotonic_pct': sum_ood_auroc_imp_isotonic,
                        'ood_fpr95_improvement_vs_geophys_pct': sum_ood_fpr95_imp_geophys,
                        'ood_fpr95_improvement_vs_isotonic_pct': sum_ood_fpr95_imp_isotonic,
                    })

            df_ood_arch = pd.DataFrame(ood_arch_rows).round(5)
            df_ood_arch.to_csv(output_dir / "per_architecture_ood_statistics.csv", index=False)
            print(f"💾 Saved per-architecture OOD statistics to: {output_dir / 'per_architecture_ood_statistics.csv'}")

            # Also save a filtered version that keeps only configurations with
            # positive improvements vs GeoPhys AND Isotonic baselines on OOD AUROC.
            if not df_ood_arch.empty and \
               'ood_auroc_improvement_vs_geophys_pct' in df_ood_arch.columns and \
               'ood_auroc_improvement_vs_isotonic_pct' in df_ood_arch.columns:
                df_ood_pos = df_ood_arch[
                    (df_ood_arch['ood_auroc_improvement_vs_geophys_pct'] > 0) &
                    (df_ood_arch['ood_auroc_improvement_vs_isotonic_pct'] > 0)
                ].copy()
                df_ood_pos = df_ood_pos.round(5)
                df_ood_pos.to_csv(output_dir / "per_architecture_ood_positive_improvements.csv", index=False)
                print(f"💾 Saved per-architecture OOD POSITIVE improvements table to: {output_dir / 'per_architecture_ood_positive_improvements.csv'}")

        # Also save a filtered version that keeps only configurations with
        # positive improvements vs GeoPhys AND Isotonic baselines (ECE).
        if not df_arch.empty and \
           'improvement_vs_geophys_pct' in df_arch.columns and \
           'improvement_vs_isotonic_pct' in df_arch.columns:
            df_pos = df_arch[
                (df_arch['improvement_vs_geophys_pct'] > 0) &
                (df_arch['improvement_vs_isotonic_pct'] > 0)
            ].copy()
            df_pos = df_pos.round(5)
            df_pos.to_csv(output_dir / "per_architecture_positive_improvements.csv", index=False)
            print(f"💾 Saved per-architecture POSITIVE improvements table to: {output_dir / 'per_architecture_positive_improvements.csv'}")

    def find_architecture_specific_optima(self) -> pd.DataFrame:
        """
        For each architecture (dataset, model, training), find the best (L, d)
        configuration based on mean ECE across seeds.
        """
        arch_optima: Dict[Tuple[str, str, str], List[dict]] = {}
        
        # Group all configs by architecture
        for config_key, arch_dict in self.architecture_data.items():
            num_layers, target_dim = config_key
            
            for arch_key, seed_results in arch_dict.items():
                if arch_key not in arch_optima:
                    arch_optima[arch_key] = []
                
                # Compute mean ECE for this arch+config
                mean_ece = np.mean([r['ece'] for r in seed_results])
                
                arch_optima[arch_key].append({
                    'num_layers': num_layers,
                    'target_dim': target_dim,
                    'mean_ece': mean_ece,
                    'n_seeds': len(seed_results)
                })
        
        # Find best config for each architecture
        rows = []
        for arch_key, configs in arch_optima.items():
            dataset, model, training = arch_key
            
            # Best = lowest mean ECE
            best = min(configs, key=lambda x: x['mean_ece'])
            
            rows.append({
                'dataset': dataset,
                'model': model,
                'training': training,
                'best_L': best['num_layers'],
                'best_d': best['target_dim'],
                'best_ece': best['mean_ece']
            })
        
        df = pd.DataFrame(rows)
        return df

    def analyze_optimal_config_patterns(self, arch_optima_df: pd.DataFrame) -> pd.DataFrame:
        """
        Check if model/dataset properties correlate with optimal L and d.
        """
        # Model depth mapping
        model_depths = {
            'resnet18': 18,
            'resnet50': 50,
            'resnet152': 152,
            'densenet121': 121,
            'wide_resnet28_10': 28
        }
        
        # Dataset complexity (number of classes)
        dataset_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'svhn': 10
        }
        
        arch_optima_df = arch_optima_df.copy()
        arch_optima_df['model_depth'] = arch_optima_df['model'].map(model_depths)
        arch_optima_df['num_classes'] = arch_optima_df['dataset'].map(dataset_classes)
        
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS: Model/Dataset Properties vs Optimal Config")
        print("="*80 + "\n")
        
        # Correlation: model depth vs optimal L
        if arch_optima_df[['model_depth', 'best_L']].dropna().shape[0] >= 2:
            corr_depth_L = arch_optima_df[['model_depth', 'best_L']].corr().iloc[0, 1]
            print(f"Correlation (Model Depth vs Optimal L): {corr_depth_L:.3f}")
        else:
            print("Correlation (Model Depth vs Optimal L): not enough data")
        
        # Correlation: num classes vs optimal d
        if arch_optima_df[['num_classes', 'best_d']].dropna().shape[0] >= 2:
            corr_classes_d = arch_optima_df[['num_classes', 'best_d']].corr().iloc[0, 1]
            print(f"Correlation (Dataset Complexity vs Optimal d): {corr_classes_d:.3f}")
        else:
            print("Correlation (Dataset Complexity vs Optimal d): not enough data")
        
        # Group analysis
        print("\nOptimal L by Model:")
        if not arch_optima_df.empty:
            print(arch_optima_df.groupby('model')['best_L'].agg(['mean', 'std', 'min', 'max']))
        
        print("\nOptimal d by Dataset:")
        if not arch_optima_df.empty:
            print(arch_optima_df.groupby('dataset')['best_d'].agg(['mean', 'std', 'min', 'max']))
        
        # Specific examples: ResNet18 vs ResNet152 on same dataset
        print("\nExample: ResNet18 vs ResNet152 on same dataset")
        for dataset in ['cifar10', 'cifar100']:
            r18 = arch_optima_df[(arch_optima_df['model'] == 'resnet18') &
                                 (arch_optima_df['dataset'] == dataset)]
            r152 = arch_optima_df[(arch_optima_df['model'] == 'resnet152') &
                                  (arch_optima_df['dataset'] == dataset)]
            
            if not r18.empty and not r152.empty:
                print(f"\n{dataset}:")
                print(f"  ResNet18:  L={r18.iloc[0]['best_L']}, d={r18.iloc[0]['best_d']}")
                print(f"  ResNet152: L={r152.iloc[0]['best_L']}, d={r152.iloc[0]['best_d']}")
        
        return arch_optima_df

    def compare_custom_vs_fixed_strategies(self, arch_optima_df: pd.DataFrame):
        """
        Compare strategies:
          1. Oracle: Each architecture uses its best config
          2. Fixed L=6, d=256: Everyone uses same config
          3. Fixed L=2, d=1024: Current best fixed config
          4. Fixed L=2, d=512: Alternative fixed config
        """
        strategies: Dict[str, List[float]] = {
            'oracle_custom': [],
            'fixed_6_256': [],
            'fixed_2_1024': [],
            'fixed_2_512': []
        }
        
        # Iterate over architectures
        for _, row in arch_optima_df.iterrows():
            dataset = row['dataset']
            model = row['model']
            training = row['training']
            oracle_L = row['best_L']
            oracle_d = row['best_d']
            arch_name = (dataset, model, training)
            
            # For this architecture, scan all configs to pick up ECE for each strategy
            for config_key, config_arch_dict in self.architecture_data.items():
                num_layers, target_dim = config_key
                
                if arch_name not in config_arch_dict:
                    continue
                
                mean_ece = np.mean([r['ece'] for r in config_arch_dict[arch_name]])
                
                # Oracle strategy
                if num_layers == oracle_L and target_dim == oracle_d:
                    strategies['oracle_custom'].append(mean_ece)
                
                # Fixed strategies
                if num_layers == 6 and target_dim == 256:
                    strategies['fixed_6_256'].append(mean_ece)
                if num_layers == 2 and target_dim == 1024:
                    strategies['fixed_2_1024'].append(mean_ece)
                if num_layers == 2 and target_dim == 512:
                    strategies['fixed_2_512'].append(mean_ece)
        
        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80 + "\n")
        
        for strategy_name, eces in strategies.items():
            if eces:
                print(f"{strategy_name:20} ECE: {np.mean(eces)*100:.2f}% ± {np.std(eces)*100:.2f}%  (n={len(eces)})")
        
        # Statistical test: Oracle vs best fixed (here L=2, d=1024)
        if strategies['oracle_custom'] and strategies['fixed_2_1024'] and \
           len(strategies['oracle_custom']) == len(strategies['fixed_2_1024']):
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(strategies['oracle_custom'], strategies['fixed_2_1024'])
            improvement = (np.mean(strategies['fixed_2_1024']) - np.mean(strategies['oracle_custom'])) / np.mean(strategies['fixed_2_1024']) * 100
            
            print(f"\nOracle vs Best Fixed (L=2, d=1024):")
            print(f"  Improvement: {improvement:.1f}%")
            print(f"  Statistical significance: p={p_value:.4f}")

    def generate_selection_guidelines(self, arch_optima_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate practical guidelines based on patterns in architecture-specific optima.
        """
        if arch_optima_df.empty:
            print("No architecture optima available to generate guidelines.")
            return arch_optima_df
        
        # Expect analyze_optimal_config_patterns to have already added these,
        # but recompute defensively in case this is called independently.
        model_depths = {
            'resnet18': 18,
            'resnet50': 50,
            'resnet152': 152,
            'densenet121': 121,
            'wide_resnet28_10': 28
        }
        dataset_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'svhn': 10
        }
        
        arch_optima_df = arch_optima_df.copy()
        if 'model_depth' not in arch_optima_df.columns:
            arch_optima_df['model_depth'] = arch_optima_df['model'].map(model_depths)
        if 'num_classes' not in arch_optima_df.columns:
            arch_optima_df['num_classes'] = arch_optima_df['dataset'].map(dataset_classes)
        
        print("\n" + "="*80)
        print("PRACTITIONER GUIDELINES")
        print("="*80 + "\n")
        
        print("Recommended Configuration Selection:")
        print("-" * 40)
        
        # Group by model depth bins
        arch_optima_df['depth_bin'] = pd.cut(
            arch_optima_df['model_depth'],
            bins=[0, 30, 60, 200],
            labels=['Shallow (<30)', 'Medium (30-60)', 'Deep (>60)']
        )
        
        arch_optima_df['complexity_bin'] = pd.cut(
            arch_optima_df['num_classes'],
            bins=[0, 20, 200],
            labels=['Simple (≤20 classes)', 'Complex (>20 classes)']
        )
        
        recommendations = arch_optima_df.groupby(['depth_bin', 'complexity_bin']).agg({
            'best_L': ['mean', 'std'],
            'best_d': ['mean', 'std'],
            'best_ece': 'mean'
        }).round(2)
        
        print(recommendations)
        
        print("\nSimplified Rules:")
        print("  • Shallow models (ResNet18):     Use L=2-4, d=128-512")
        print("  • Medium models (ResNet50):      Use L=4-6, d=256-1024")
        print("  • Deep models (ResNet152):       Use L=6-10, d=512-1024")
        print("  • Complex datasets (>20 classes): Increase d by ~2x")
        
        return arch_optima_df

    def analyze_configuration_universality(self, output_dir: str = "."):
        """
        Analyze which configurations work across how many architectures.
        "Works for" = beats both GeoPhys and Isotonic baselines on ECE.
        """
        print("\n" + "="*80)
        print("CONFIGURATION UNIVERSALITY ANALYSIS")
        print("="*80 + "\n")
        
        if not self.architecture_data:
            print("No architecture data available for universality analysis.")
            return pd.DataFrame()
        
        # Count how many architectures each (L, d) works for
        config_success_counts = defaultdict(lambda: {'total_archs': 0, 'success_archs': set()})
        
        for config_key, arch_dict in self.architecture_data.items():
            num_layers, target_dim = config_key
            
            for arch_key, seed_results in arch_dict.items():
                if not seed_results:
                    continue
                
                config_success_counts[config_key]['total_archs'] += 1
                
                # Check if this config beats both baselines for this architecture
                mean_ece = np.mean([r['ece'] for r in seed_results])
                baselines = seed_results[0].get('baselines', {})
                
                geo_phys = baselines.get('geometric_physical_space', {}).get('ece', float('inf'))
                isotonic = baselines.get('isotonic_toplabel', {}).get('ece', float('inf'))
                
                if mean_ece < geo_phys and mean_ece < isotonic:
                    config_success_counts[config_key]['success_archs'].add(arch_key)
        
        # Create results dataframe
        universality_data = []
        for config_key, counts in config_success_counts.items():
            num_layers, target_dim = config_key
            success_count = len(counts['success_archs'])
            total_count = counts['total_archs']
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0.0
            
            universality_data.append({
                'num_layers': num_layers,
                'target_dim': target_dim,
                'success_count': success_count,
                'total_count': total_count,
                'success_rate': success_rate
            })
        
        df = pd.DataFrame(universality_data)
        if df.empty:
            print("No successful configurations found for universality analysis.")
            return df
        
        df = df.sort_values('success_rate', ascending=False)
        
        print("Top 10 Most Universal Configurations:")
        print(df.head(10).to_string(index=False))
        
        # Save to CSV
        output_dir = Path(output_dir)
        df_out = df.round(5)
        df_out.to_csv(output_dir / "configuration_universality.csv", index=False)
        print(f"\n💾 Saved to: {output_dir / 'configuration_universality.csv'}")
        
        # Create pivot table for heatmap data
        pivot = df.pivot_table(
            index='num_layers',
            columns='target_dim',
            values='success_count',
            fill_value=0
        )
        pivot.round(5).to_csv(output_dir / "universality_heatmap_data.csv")
        print(f"💾 Saved heatmap data to: {output_dir / 'universality_heatmap_data.csv'}")
        
        return df

    def analyze_config_by_model_type(self, output_dir: str = "."):
        """
        For each model type, show which configurations work best.
        "Works" = configuration beats both GeoPhys and Isotonic baselines on ECE.
        """
        print("\n" + "="*80)
        print("CONFIGURATION PREFERENCES BY MODEL TYPE")
        print("="*80 + "\n")
        
        if not self.architecture_data:
            print("No architecture data available for model-type analysis.")
            return pd.DataFrame()
        
        model_preferences: Dict[str, List[dict]] = defaultdict(list)
        
        # For each architecture, find configs that beat both baselines
        for config_key, arch_dict in self.architecture_data.items():
            num_layers, target_dim = config_key
            
            for arch_key, seed_results in arch_dict.items():
                dataset, model, training = arch_key
                
                if not seed_results:
                    continue
                
                mean_ece = np.mean([r['ece'] for r in seed_results])
                baselines = seed_results[0].get('baselines', {})
                
                geo_phys = baselines.get('geometric_physical_space', {}).get('ece', float('inf'))
                isotonic = baselines.get('isotonic_toplabel', {}).get('ece', float('inf'))
                
                # If this config beats both baselines
                if mean_ece < geo_phys and mean_ece < isotonic:
                    model_preferences[model].append({
                        'num_layers': num_layers,
                        'target_dim': target_dim,
                        'mean_ece': mean_ece,
                        'dataset': dataset,
                        'training': training
                    })
        
        print("Configuration preferences by model:\n")
        
        summary_data = []
        for model, configs in sorted(model_preferences.items()):
            if not configs:
                continue
            
            df = pd.DataFrame(configs)
            
            # Most common L and d
            most_common_L = df['num_layers'].mode().values[0] if len(df) > 0 else None
            most_common_d = df['target_dim'].mode().values[0] if len(df) > 0 else None
            avg_L = df['num_layers'].mean()
            std_L = df['num_layers'].std()
            median_d = df['target_dim'].median()
            std_d = df['target_dim'].std()
            
            # Distribution of d values (%)
            d_counts = df['target_dim'].value_counts(normalize=True).sort_index()
            d_dist_str = ", ".join([f"d={d}:{pct*100:.0f}%" for d, pct in d_counts.items()])
            
            print(f"{model:20} - Successful configs: {len(configs):3d}  "
                  f"Most common: L={most_common_L}, d={most_common_d}  "
                  f"Avg L: {avg_L:.1f} (std {std_L:.1f}), Median d: {median_d:.0f} (std {std_d:.0f})")
            print(f"  d distribution: {d_dist_str}")
            
            summary_data.append({
                'model': model,
                'successful_configs': len(configs),
                'most_common_L': most_common_L,
                'most_common_d': most_common_d,
                'average_L': avg_L,
                'std_L': std_L,
                'median_d': median_d,
                'std_d': std_d,
                'd_64_pct': (df['target_dim'] == 64).mean() * 100,
                'd_128_pct': (df['target_dim'] == 128).mean() * 100,
                'd_256_pct': (df['target_dim'] == 256).mean() * 100,
                'd_512_pct': (df['target_dim'] == 512).mean() * 100,
                'd_1024_pct': (df['target_dim'] == 1024).mean() * 100,
            })
        
        output_dir = Path(output_dir)
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            summary_df = summary_df.round(5)
            summary_df.to_csv(output_dir / "config_preferences_by_model.csv", index=False)
            print(f"\n💾 Saved to: {output_dir / 'config_preferences_by_model.csv'}")
        else:
            print("No successful configurations found for any model type.")
        
        return summary_df

    def analyze_config_by_dataset(self, output_dir: str = "."):
        """
        Compare configuration preferences between datasets (e.g., CIFAR-10 vs CIFAR-100).
        "Preference" = configurations that beat both GeoPhys and Isotonic baselines.
        """
        print("\n" + "="*80)
        print("CONFIGURATION PREFERENCES BY DATASET")
        print("="*80 + "\n")
        
        if not self.architecture_data:
            print("No architecture data available for dataset-level analysis.")
            return pd.DataFrame()
        
        dataset_preferences: Dict[str, List[dict]] = defaultdict(list)
        
        for config_key, arch_dict in self.architecture_data.items():
            num_layers, target_dim = config_key
            
            for arch_key, seed_results in arch_dict.items():
                dataset, model, training = arch_key
                
                if not seed_results:
                    continue
                
                mean_ece = np.mean([r['ece'] for r in seed_results])
                baselines = seed_results[0].get('baselines', {})
                
                geo_phys = baselines.get('geometric_physical_space', {}).get('ece', float('inf'))
                isotonic = baselines.get('isotonic_toplabel', {}).get('ece', float('inf'))
                
                if mean_ece < geo_phys and mean_ece < isotonic:
                    dataset_preferences[dataset].append({
                        'num_layers': num_layers,
                        'target_dim': target_dim,
                        'mean_ece': mean_ece,
                        'model': model
                    })
        
        print("Configuration preferences by dataset:\n")
        
        summary_data = []
        for dataset, configs in sorted(dataset_preferences.items()):
            if not configs:
                continue
            
            df = pd.DataFrame(configs)
            
            most_common_L = df['num_layers'].mode().values[0] if len(df) > 0 else None
            most_common_d = df['target_dim'].mode().values[0] if len(df) > 0 else None
            avg_L = df['num_layers'].mean()
            std_L = df['num_layers'].std()
            median_d = df['target_dim'].median()
            std_d = df['target_dim'].std()
            
            # Distribution of d values (%)
            d_counts = df['target_dim'].value_counts(normalize=True).sort_index()
            d_dist_str = ", ".join([f"d={d}:{pct*100:.0f}%" for d, pct in d_counts.items()])
            
            print(f"{dataset:12} - Successful configs: {len(configs):3d}  "
                  f"Most common: L={most_common_L}, d={most_common_d}  "
                  f"Avg L: {avg_L:.1f} (std {std_L:.1f}), Median d: {median_d:.0f} (std {std_d:.0f})")
            print(f"  d distribution: {d_dist_str}")
            
            summary_data.append({
                'dataset': dataset,
                'successful_configs': len(configs),
                'most_common_L': most_common_L,
                'most_common_d': most_common_d,
                'average_L': avg_L,
                'std_L': std_L,
                'median_d': median_d,
                'std_d': std_d,
                'd_64_pct': (df['target_dim'] == 64).mean() * 100,
                'd_128_pct': (df['target_dim'] == 128).mean() * 100,
                'd_256_pct': (df['target_dim'] == 256).mean() * 100,
                'd_512_pct': (df['target_dim'] == 512).mean() * 100,
                'd_1024_pct': (df['target_dim'] == 1024).mean() * 100,
            })
        
        output_dir = Path(output_dir)
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            summary_df = summary_df.round(5)
            summary_df.to_csv(output_dir / "config_preferences_by_dataset.csv", index=False)
            print(f"\n💾 Saved to: {output_dir / 'config_preferences_by_dataset.csv'}")
        else:
            print("No successful configurations found for any dataset.")
        
        return summary_df

    def visualize_d_distribution(self, output_dir: str = "."):
        """
        Show distribution of successful target_dim (d) values by dataset and model.
        "Successful" = configuration beats both GeoPhys and Isotonic baselines.
        """
        print("\n" + "="*80)
        print("TARGET DIMENSION (d) DISTRIBUTION ANALYSIS")
        print("="*80 + "\n")
        
        dataset_d_dist: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        model_d_dist: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        for config_key, arch_dict in self.architecture_data.items():
            num_layers, target_dim = config_key
            
            for arch_key, seed_results in arch_dict.items():
                dataset, model, training = arch_key
                
                if not seed_results:
                    continue
                
                mean_ece = np.mean([r['ece'] for r in seed_results])
                baselines = seed_results[0].get('baselines', {})
                
                geo_phys = baselines.get('geometric_physical_space', {}).get('ece', float('inf'))
                isotonic = baselines.get('isotonic_toplabel', {}).get('ece', float('inf'))
                
                if mean_ece < geo_phys and mean_ece < isotonic:
                    dataset_d_dist[dataset][target_dim] += 1
                    model_d_dist[model][target_dim] += 1
        
        print("Distribution of successful d by DATASET:")
        print("-" * 80)
        for dataset in ['cifar10', 'cifar100']:
            if dataset in dataset_d_dist:
                total = sum(dataset_d_dist[dataset].values())
                # Compute mean/std of d for this dataset
                expanded_ds = [d for d, cnt in dataset_d_dist[dataset].items() for _ in range(cnt)]
                mean_d_ds = np.mean(expanded_ds) if expanded_ds else 0.0
                std_d_ds = np.std(expanded_ds) if expanded_ds else 0.0
                print(f"\n{dataset.upper()} (n={total}):")
                print(f"  Mean d: {mean_d_ds:.1f}, Std d: {std_d_ds:.1f}")
                for d in [64, 128, 256, 512, 1024]:
                    count = dataset_d_dist[dataset].get(d, 0)
                    pct = (count / total * 100) if total > 0 else 0
                    bar = "█" * int(pct / 2)
                    print(f"  d={d:4d}: {count:3d} ({pct:5.1f}%) {bar}")
        
        print("\n" + "="*80)
        print("Distribution of successful d by MODEL:")
        print("-" * 80)
        for model in sorted(model_d_dist.keys()):
            total = sum(model_d_dist[model].values())
            expanded_model = [d for d, cnt in model_d_dist[model].items() for _ in range(cnt)]
            mean_d_model = np.mean(expanded_model) if expanded_model else 0.0
            std_d_model = np.std(expanded_model) if expanded_model else 0.0
            print(f"\n{model.upper()} (n={total}):")
            print(f"  Mean d: {mean_d_model:.1f}, Std d: {std_d_model:.1f}")
            for d in [64, 128, 256, 512, 1024]:
                count = model_d_dist[model].get(d, 0)
                pct = (count / total * 100) if total > 0 else 0
                bar = "█" * int(pct / 2)
                print(f"  d={d:4d}: {count:3d} ({pct:5.1f}%) {bar}")
        
        # Save raw counts for external plotting if desired
        output_dir = Path(output_dir)
        ds_rows = []
        for dataset, counts in dataset_d_dist.items():
            total = sum(counts.values())
            for d, cnt in counts.items():
                ds_rows.append({
                    'dataset': dataset,
                    'd': d,
                    'count': cnt,
                    'pct': (cnt / total * 100) if total > 0 else 0
                })
        if ds_rows:
            pd.DataFrame(ds_rows).round(5).to_csv(output_dir / "d_distribution_by_dataset.csv", index=False)
        
        model_rows = []
        for model, counts in model_d_dist.items():
            total = sum(counts.values())
            for d, cnt in counts.items():
                model_rows.append({
                    'model': model,
                    'd': d,
                    'count': cnt,
                    'pct': (cnt / total * 100) if total > 0 else 0
                })
        if model_rows:
            pd.DataFrame(model_rows).round(5).to_csv(output_dir / "d_distribution_by_model.csv", index=False)
        
        return dataset_d_dist, model_d_dist

    def analyze_training_method_confounds(self, output_dir: str = "."):
        """
        Analyze if training method confounds dataset complexity findings.
        Critical question: Is L preference driven by dataset complexity or training method?
        """
        print("\n" + "="*80)
        print("TRAINING METHOD CONFOUND ANALYSIS")
        print("="*80 + "\n")
        
        if not self.architecture_data:
            print("No architecture data available for training-method analysis.")
            return {}
        
        dataset_training_counts = defaultdict(lambda: defaultdict(int))
        training_Ld_preferences = defaultdict(list)
        training_L_values = defaultdict(list)
        training_d_values = defaultdict(list)
        # map (dataset, training) -> list of (L, d)
        dataset_training_Ld = defaultdict(list)
        
        for config_key, arch_dict in self.architecture_data.items():
            num_layers, target_dim = config_key
            
            for arch_key, seed_results in arch_dict.items():
                dataset, model, training = arch_key
                
                if not seed_results:
                    continue
                
                mean_ece = np.mean([r['ece'] for r in seed_results])
                baselines = seed_results[0].get('baselines', {})
                
                geo_phys = baselines.get('geometric_physical_space', {}).get('ece', float('inf'))
                isotonic = baselines.get('isotonic_toplabel', {}).get('ece', float('inf'))
                
                if mean_ece < geo_phys and mean_ece < isotonic:
                    dataset_training_counts[dataset][training] += 1
                    training_Ld_preferences[training].append((num_layers, target_dim))
                    training_L_values[training].append(num_layers)
                    training_d_values[training].append(target_dim)
                    dataset_training_Ld[(dataset, training)].append((num_layers, target_dim))
        
        # 1. Training method distribution by dataset
        print("STEP 1: Training Method Distribution by Dataset")
        print("-" * 80)
        
        for dataset in ['cifar10', 'cifar100']:
            if dataset not in dataset_training_counts:
                continue
            
            total = sum(dataset_training_counts[dataset].values())
            print(f"\n{dataset.upper()} (n={total} successful configs):")
            
            for training, count in sorted(dataset_training_counts[dataset].items(), key=lambda x: x[1], reverse=True):
                pct = (count / total * 100) if total > 0 else 0
                bar = "█" * int(pct / 3)
                print(f"  {training:30s}: {count:3d} ({pct:5.1f}%) {bar}")
        
        cifar10_trainings = set(dataset_training_counts.get('cifar10', {}).keys())
        cifar100_trainings = set(dataset_training_counts.get('cifar100', {}).keys())
        
        if cifar10_trainings and cifar100_trainings:
            common = cifar10_trainings & cifar100_trainings
            print(f"\nCommon training methods: {len(common)}/{len(cifar10_trainings | cifar100_trainings)}")
            
            if len(common) < len(cifar10_trainings | cifar100_trainings):
                print("⚠️  WARNING: Different training methods used for different datasets!")
                print("    This could confound dataset complexity analysis.")
        
        # 2. Layer preferences by training method
        print("\n" + "="*80)
        print("STEP 2: Layer Preferences (L) by Training Method")
        print("-" * 80)
        
        training_L_stats = {}
        for training in sorted(training_L_values.keys()):
            L_vals = training_L_values[training]
            if not L_vals:
                continue
            
            mean_L = np.mean(L_vals)
            median_L = np.median(L_vals)
            mode_L_counts = defaultdict(int)
            for L in L_vals:
                mode_L_counts[L] += 1
            mode_L = max(mode_L_counts.items(), key=lambda x: x[1])[0]
            
            training_L_stats[training] = {
                'mean': mean_L,
                'median': median_L,
                'mode': mode_L,
                'n': len(L_vals)
            }
            
            print(f"\n{training:30s} (n={len(L_vals)}):")
            print(f"  Mean L: {mean_L:.2f}, Median L: {median_L:.0f}, Mode L: {mode_L}")
        
        if training_L_stats:
            highest_L = max(training_L_stats.items(), key=lambda x: x[1]['mean'])
            lowest_L = min(training_L_stats.items(), key=lambda x: x[1]['mean'])
            
            print(f"\n🔺 Highest L preference: {highest_L[0]} (mean={highest_L[1]['mean']:.2f})")
            print(f"🔻 Lowest L preference:  {lowest_L[0]} (mean={lowest_L[1]['mean']:.2f})")
            print(f"   Difference: {highest_L[1]['mean'] - lowest_L[1]['mean']:.2f} layers")
        
        # 3. Dataset comparison controlling for training method
        print("\n" + "="*80)
        print("STEP 3: Dataset Comparison CONTROLLING for Training Method")
        print("-" * 80)
        
        if cifar10_trainings and cifar100_trainings:
            common = cifar10_trainings & cifar100_trainings
            for training in sorted(common):
                cifar10_L = [L for (ds, tr), Ld_list in dataset_training_Ld.items()
                             if ds == 'cifar10' and tr == training
                             for L, d in Ld_list]
                cifar100_L = [L for (ds, tr), Ld_list in dataset_training_Ld.items()
                              if ds == 'cifar100' and tr == training
                              for L, d in Ld_list]
                
                if len(cifar10_L) < 5 or len(cifar100_L) < 5:
                    continue
                
                mean_diff = np.mean(cifar100_L) - np.mean(cifar10_L)
                median_diff = np.median(cifar100_L) - np.median(cifar10_L)
                
                print(f"\n{training:30s}:")
                print(f"  CIFAR-10:  Mean L={np.mean(cifar10_L):.2f}, Median L={np.median(cifar10_L):.0f} (n={len(cifar10_L)})")
                print(f"  CIFAR-100: Mean L={np.mean(cifar100_L):.2f}, Median L={np.median(cifar100_L):.0f} (n={len(cifar100_L)})")
                print(f"  Difference: {mean_diff:+.2f} (mean), {median_diff:+.0f} (median)")
                
                if len(cifar10_L) >= 10 and len(cifar100_L) >= 10:
                    from scipy import stats
                    statistic, p_value = stats.mannwhitneyu(cifar10_L, cifar100_L, alternative='two-sided')
                    sig_marker = "✓" if p_value < 0.05 else "✗"
                    print(f"  {sig_marker} p={p_value:.4f}")
        
        # 4. Diagnostic summary: does training method drive L?
        print("\n" + "="*80)
        print("DIAGNOSTIC SUMMARY")
        print("="*80 + "\n")
        
        all_L_values = []
        all_training_labels = []
        for training, L_vals in training_L_values.items():
            all_L_values.extend(L_vals)
            all_training_labels.extend([training] * len(L_vals))
        
        if len(set(all_training_labels)) > 1:
            from scipy import stats
            training_groups = [training_L_values[t] for t in training_L_values.keys() if training_L_values[t]]
            if len(training_groups) > 1:
                f_stat, p_value = stats.f_oneway(*training_groups)
                print(f"ANOVA: Does training method affect L preference?")
                print(f"  F-statistic: {f_stat:.2f}")
                print(f"  p-value: {p_value:.4f}")
                
                if p_value < 0.05:
                    print(f"  ✓ Training method SIGNIFICANTLY affects L preference")
                    print(f"    ⚠️  This could confound dataset complexity findings!")
                else:
                    print(f"  ✗ Training method does NOT significantly affect L preference")
                    print(f"    ✓ Dataset complexity finding is likely valid")
        
        # Save detailed results
        output_dir = Path(output_dir)
        training_stats_df = pd.DataFrame([
            {
                'training_method': training,
                'n_configs': stats_dict['n'],
                'mean_L': stats_dict['mean'],
                'median_L': stats_dict['median'],
                'mode_L': stats_dict['mode']
            }
            for training, stats_dict in training_L_stats.items()
        ])
        if not training_stats_df.empty:
            training_stats_df.round(5).to_csv(output_dir / "training_method_L_preferences.csv", index=False)
            print(f"\n💾 Saved training method stats to: {output_dir / 'training_method_L_preferences.csv'}")
        
        return {
            'dataset_training_counts': dict(dataset_training_counts),
            'training_L_stats': training_L_stats,
            'dataset_training_Ld': dict(dataset_training_Ld)
        }

    def run_analysis(self, min_architectures: int = 4, top_n: int = 15, output_dir: str = ".",
                     validate_data: bool = True, clean_duplicates: bool = False,
                     expected_trials: int = 8, filter_reliable: bool = True):
        """Run complete analysis pipeline with optional data validation"""
        
        print("\n" + "="*80)
        print("CONFIGURATION SELECTION ANALYSIS (Architecture-Level)")
        print("="*80 + "\n")
        
        # Load and extract
        n_results = self.load_results()
        if n_results == 0:
            print("❌ No results found! Check your results directory.")
            return None
        
        self.extract_configuration_data()
        
        # FILTERING STEP (optional but recommended)
        if filter_reliable and expected_trials > 0:
            min_trials = max(expected_trials - 2, int(expected_trials * 0.8))  # Allow 80% of expected
            print(f"\n📊 Filtering to architectures with at least {min_trials} trials...")
            filtering_report = self.filter_to_reliable_architectures(min_trials_per_arch=min_trials)
        
        # VALIDATION STEP
        validation_report = None
        if validate_data:
            validation_report = self.validate_data_integrity(expected_trials_per_config=expected_trials)
            self.export_validation_report(validation_report, output_dir=output_dir)
            
            # If duplicates found and cleaning requested
            if clean_duplicates and validation_report['duplicates_found']:
                print("\n🧹 Cleaning duplicates as requested...")
                cleaning_report = self.clean_duplicates(
                    expected_trials_per_config=expected_trials,
                    strategy='keep_first'  # Can make this configurable
                )
                
                # Re-validate after cleaning
                print("\n🔍 Re-validating after cleaning...")
                validation_report = self.validate_data_integrity(expected_trials_per_config=expected_trials)
        
        # Compute metrics (using cleaned data if cleaning was performed)
        metrics = self.compute_architecture_level_metrics()
        
        # Rank
        ranked = self.rank_configurations(metrics, min_architectures=min_architectures)
        
        if not ranked:
            print(f"❌ No configurations with >= {min_architectures} architectures found!")
            print("   Try reducing min_architectures or wait for more results.")
            return None
        
        # Print results
        self.print_rankings(ranked, top_n=top_n)
        
        # Print efficiency comparison
        self.print_performance_efficiency_comparison(ranked, top_n=5)
        
        # Check target
        target_config = self.check_target_configuration(ranked, target_layers=6, target_dim=256)
        
        # Generate reports
        self.generate_detailed_report(metrics, ranked, output_dir=output_dir)
        
        # ------------------------------------------------------------------
        # Architecture-specific configuration analysis
        # ------------------------------------------------------------------
        print("\n" + "="*80)
        print("ARCHITECTURE-SPECIFIC CONFIGURATION ANALYSIS")
        print("="*80)
        
        arch_optima_df = self.find_architecture_specific_optima()
        
        if not arch_optima_df.empty:
            # Analyze correlation patterns
            arch_optima_df = self.analyze_optimal_config_patterns(arch_optima_df)
            
            # Compare custom vs fixed strategies
            self.compare_custom_vs_fixed_strategies(arch_optima_df)
            
            # Generate practical guidelines
            arch_optima_df = self.generate_selection_guidelines(arch_optima_df)
            
            # Save architecture optima to CSV
            output_dir_path = Path(output_dir)
            arch_optima_out = arch_optima_df.round(5)
            arch_optima_out.to_csv(output_dir_path / "architecture_specific_optima.csv", index=False)
            print(f"\n💾 Saved architecture-specific optima to: {output_dir_path / 'architecture_specific_optima.csv'}")
            
            # Analyze configuration universality
            universality_df = self.analyze_configuration_universality(output_dir=output_dir_path)
            
            # Analyze preferences by model type
            model_prefs_df = self.analyze_config_by_model_type(output_dir=output_dir_path)

            # Check training method as potential confounder before dataset prefs
            training_confounds = self.analyze_training_method_confounds(output_dir=output_dir_path)
            
            # Analyze preferences by dataset
            dataset_prefs_df = self.analyze_config_by_dataset(output_dir=output_dir_path)

            # Visualize distribution of successful target_dim values
            dataset_d_dist, model_d_dist = self.visualize_d_distribution(output_dir=output_dir_path)
        else:
            print("Skipping architecture-specific analysis: no architecture optima found.")
        
        return {
            'metrics': metrics,
            'ranked': ranked,
            'target_config': target_config,
            'validation_report': validation_report
        }


# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":
    results_dir = "calibration_comparison_results/random_ablation/ood"
    
    selector = ConfigurationSelector(results_dir)
    
    # Run with validation and filtering enabled
    results = selector.run_analysis(
        min_architectures=5,
        top_n=15,
        output_dir="./configuration_analysis/ood",
        validate_data=True,
        clean_duplicates=True,        # Clean L=12 duplicates
        filter_reliable=False,        # Disable reliability filtering to see all results
        expected_trials=10
    )
    
    print("\n" + "="*80)
    print("GENERATING PAPER STATISTICS")
    print("="*80)
    
    # Generate paper table with top 5 configurations
    paper_df = selector.generate_paper_table(
        configs_to_include=None,  # Use top 5 from ranking
        output_file="configuration_analysis/paper_table.csv"
    )
    
    # Show comparison for L=6, d=256
    selector.compare_analysis_methods(config_key=(6, 256))
    
    # Also show statistics for top config
    if results and results['ranked']:
        top_config = (results['ranked'][0]['num_layers'], results['ranked'][0]['target_dim'])
        print(f"\n{'='*80}")
        print(f"TOP CONFIGURATION DETAILED STATS")
        print(f"{'='*80}")
        stats = selector.compute_paper_statistics(top_config)
        if stats:
            print(f"\nConfiguration: {stats['config']}")
            print(f"ECE: {stats['ece_mean_pct']:.2f}% ± {stats['ece_std_pct']:.2f}%")
            print(f"Accuracy: {stats['accuracy_mean']:.2f}%")
            print(f"Throughput: {stats['sps_mean']:.1f} sps")
            print(f"Based on {stats['n_trials']} trials across {stats['n_architectures']} architectures")
            
            print("\nPer-dataset breakdown:")
            for dataset, ds_stats in stats['by_dataset'].items():
                print(f"  {dataset}: {ds_stats['latex']}")

    print("\n" + "="*80)
    print("ARCHITECTURE-SPECIFIC ANALYSIS")
    print("="*80)
    
    if results:
        # Find optimal config for each architecture
        arch_optima_df = selector.find_architecture_specific_optima()
        
        if not arch_optima_df.empty:
            # Analyze patterns
            arch_optima_df = selector.analyze_optimal_config_patterns(arch_optima_df)
            
            # Compare strategies
            selector.compare_custom_vs_fixed_strategies(arch_optima_df)
            
            # Generate guidelines
            arch_optima_df = selector.generate_selection_guidelines(arch_optima_df)
            
            # Save results
            arch_optima_out = arch_optima_df.round(5)
            arch_optima_out.to_csv("configuration_analysis/architecture_specific_optima.csv", index=False)
            print(f"\n💾 Saved architecture optima to: configuration_analysis/architecture_specific_optima.csv")
            
            # Additional analyses
            universality_df = selector.analyze_configuration_universality(output_dir="./configuration_analysis")
            model_prefs_df = selector.analyze_config_by_model_type(output_dir="./configuration_analysis")
            training_confounds = selector.analyze_training_method_confounds(output_dir="./configuration_analysis")
            dataset_prefs_df = selector.analyze_config_by_dataset(output_dir="./configuration_analysis")
            dataset_d_dist, model_d_dist = selector.visualize_d_distribution(output_dir="./configuration_analysis")
        else:
            print("No architecture-specific optima found; skipping architecture-specific analysis.")
    
    print("\n✅ Analysis complete!")