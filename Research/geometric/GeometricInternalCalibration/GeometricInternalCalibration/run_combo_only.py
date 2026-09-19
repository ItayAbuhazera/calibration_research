#!/usr/bin/env python3
"""
Combo-only 20k trial runner: wide search over progressive combo structure only.
No side knobs (strict_band, refine, weights) - pure combo exploration.

Usage:
    python run_combo_only.py [--trials 20000] [--objective win_rate] [--force-tail]
"""

import os
import argparse
import optuna
from pathlib import Path
from enhanced_metric_analysis import EnhancedMetricAnalyzer
from optuna_combo_search import study_combo_only

def main():
    parser = argparse.ArgumentParser(description='Combo-only 20k trial search')
    parser.add_argument('--results-root', default='aaai_full_experiments/results/layer_selection_analysis_baselines_4',
                        help='Path to results directory')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: results_root/optuna)')
    parser.add_argument('--trials', type=int, default=20000,
                        help='Number of trials to run')
    parser.add_argument('--objective', choices=['win_rate', 'gap', 'robustness', 'within_epsilon'], 
                        default='win_rate', help='Objective to optimize')
    parser.add_argument('--force-tail', action='store_true', default=True,
                        help='Force tail metrics (margin_tail_cvar, impostor_gap_cvar) at the end')
    parser.add_argument('--no-force-tail', dest='force_tail', action='store_false',
                        help='Allow any tail metrics')
    parser.add_argument('--storage', default=None,
                        help='SQLite storage for resumable/parallel runs (e.g., sqlite:///combo_only.db)')
    parser.add_argument('--seed', type=int, default=20250927,
                        help='Random seed')
    parser.add_argument('--eps-choices', nargs='+', type=float, default=[0.001, 0.0015, 0.002],
                        help='Epsilon choices for search')
    
    args = parser.parse_args()
    
    # Setup paths
    results_root = args.results_root
    if args.output_dir is None:
        output_dir = Path(results_root) / "optuna"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Combo-only search: {args.trials} trials")
    print(f"   Results root: {results_root}")
    print(f"   Output dir: {output_dir}")
    print(f"   Objective: {args.objective}")
    print(f"   Force tail: {args.force_tail}")
    print(f"   Epsilon choices: {args.eps_choices}")
    if args.storage:
        print(f"   Storage: {args.storage}")
    
    # Initialize analyzer
    analyzer = EnhancedMetricAnalyzer(results_root, acc_threshold=0.70)
    
    # Load results
    tm_dirs = [p.name for p in Path(results_root).iterdir() if p.is_dir()]
    if not tm_dirs:
        raise RuntimeError(f"No training-method subdirs found under {results_root}")
    
    print(f"   Training methods: {tm_dirs}")
    analyzer.load_results(tm_dirs)
    
    # Create seed split
    split = analyzer.split_seeds(train_frac=0.8, random_state=42)
    print(f"   Train/test split: {analyzer.num_layers(split, 'train')} train, {analyzer.num_layers(split, 'test')} test layers")
    
    # Run combo-only study
    print(f"\n🚀 Starting {args.trials} trials...")
    study = study_combo_only(
        analyzer=analyzer,
        split=split,
        n_trials=args.trials,
        seed=args.seed,
        objective=args.objective,
        force_tail=args.force_tail,
        eps_choices=tuple(args.eps_choices),
        storage=args.storage,
        study_name=f'combo_only_{args.objective}',
        output_dir=str(output_dir)
    )
    
    # Evaluate best on train/test, report
    t = study.best_trial
    combo = t.user_attrs.get("combo", {})
    prog = list(combo.get("progressive", []))
    
    print(f"\n✅ Study completed!")
    print(f"   Best value: {study.best_value:.4f}")
    print(f"   Best trial: #{t.number}")
    
    # Evaluate on train/test
    tr = analyzer.evaluate_combo(split=split, part='train', progressive=prog,
                                eps=combo["eps"], strict_band=None, refine="none",
                                decisiveness_topm=None, kfold=None, weights=None)
    te = analyzer.evaluate_combo(split=split, part='test', progressive=prog,
                                eps=combo["eps"], strict_band=None, refine="none",
                                decisiveness_topm=None, kfold=None, weights=None)
    
    print(f"\n📊 Best combo details:")
    print(f"   ε: {combo.get('eps')}")
    print(f"   Metrics: {combo.get('metrics_sequence')}")
    print(f"   Keep Ks: {combo.get('keep_ks')}")
    print(f"   Length: {combo.get('n_metrics')}")
    
    # Show pruning effectiveness
    keep_ks = combo.get('keep_ks', [])
    if len(keep_ks) >= 2:
        shrink_ratio = keep_ks[-1] / max(1, keep_ks[0])
        print(f"   Pruning: {keep_ks[0]} → {keep_ks[-1]} (shrink: {shrink_ratio:.2f})")
    
    print(f"\n🎯 Performance metrics:")
    print(f"   Train: {tr}")
    print(f"   Test : {te}")
    
    # Save results
    import json
    results = {
        'best_value': study.best_value,
        'best_trial_number': t.number,
        'best_combo': combo,
        'train_perf': tr,
        'test_perf': te,
        'study_params': {
            'n_trials': args.trials,
            'objective': args.objective,
            'force_tail': args.force_tail,
            'eps_choices': args.eps_choices,
            'seed': args.seed
        }
    }
    
    results_file = output_dir / f'combo_only_{args.objective}_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Save top trials CSV
    try:
        from optuna_combo_search import save_top_trials
        paths = save_top_trials(study, str(output_dir), top_k=50, name=f'combo_only_{args.objective}')
        print(f"   Top trials CSV: {paths}")
    except Exception as e:
        print(f"   Warning: Could not save top trials CSV: {e}")
    
    # Robustness check: re-score top combos with different ε values
    print(f"\n🔍 Running robustness check...")
    try:
        from optuna_combo_search import rescore_top_combos_robustness
        robustness_df = rescore_top_combos_robustness(
            study, analyzer, split, top_k=50,
            eps_choices=(0.00075, 0.001, 0.0015, 0.002, 0.0025)
        )
        
        if not robustness_df.empty:
            robustness_file = output_dir / f'combo_only_{args.objective}_robustness.csv'
            robustness_df.to_csv(robustness_file, index=False)
            print(f"   Robustness check saved to: {robustness_file}")
            
            # Show best ε for top combos
            best_eps_per_combo = robustness_df.groupby('trial_rank').apply(
                lambda x: x.loc[x['beats_rate'].idxmax()]
            ).reset_index(drop=True)
            print(f"   Best ε for top-10 combos: {best_eps_per_combo.head(10)['eps'].tolist()}")
        else:
            print("   Warning: No robustness results generated")
    except Exception as e:
        print(f"   Warning: Robustness check failed: {e}")

if __name__ == "__main__":
    main()
