#!/usr/bin/env python3
"""
run_combo_search.py - Standalone runner for layer selection strategy search.
Uses existing optuna_combo_search + EnhancedMetricAnalyzer logic.
No ad-hoc scoring. No custom metrics. Pure orchestration.

ENVIRONMENT VARIABLES:
    ENABLE_OPTUNA_IMPORTANCES=1  - Enable param importance computation (disabled by default)
                                   Note: This can hang with many categorical params and large trials.
                                   Only enable if you need param importances and the study is small.

USAGE:
    # Run with default settings (param importances disabled)
    python run_combo_search.py aaai_full_experiments/results/layer_selection_analysis_baselines_4 \\
        --trials 500 --outdir runs/debug500
    
    # Run with param importances enabled (use with caution)
    ENABLE_OPTUNA_IMPORTANCES=1 \\
    python run_combo_search.py aaai_full_experiments/results/layer_selection_analysis_baselines_4 \\
        --trials 500 --outdir runs/debug500
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from optuna_combo_search import (
    study_combo_only,
    study_focused_pipeline,
    rescore_top_combos_robustness,
    save_top_trials,
)
from enhanced_metric_analysis import EnhancedMetricAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from utils.logging_config import get_logger
logger = get_logger(__name__)


def _rehydrate_progressive(progressive) -> List[Dict[str, Any]]:
    """
    Rehydrate progressive stages to standard list-of-dicts format.
    Trials can store progressive as tuple of dicts or tuple of (key,value) pairs.
    """
    if not progressive:
        return []
    out = []
    for stage in progressive:
        # stage may be a dict already, or a tuple of pairs
        d = stage if isinstance(stage, dict) else dict(stage)
        out.append({"metric": d.get("metric"), "keep_k": int(d.get("keep_k", 1))})
    return out


def print_summary(study, analyzer, split, objective_name: str):
    """Print human-readable summary of best trial"""
    best = study.best_trial
    combo = best.user_attrs.get("combo", {})
    
    logger.info("\n" + "="*80)
    logger.info(f"🏆 BEST {objective_name.upper()} COMBO")
    logger.info("="*80)
    
    v = best.value
    logger.info(f"Value: {v:.4f}" if isinstance(v, (int, float)) else f"Value: {v}")
    logger.info(f"Trial: #{best.number}")
    logger.info(f"\nCombo structure:")
    logger.info(f"  ε = {combo.get('eps', 'N/A')}")
    logger.info(f"  strict_band = {combo.get('strict_band', 'N/A')}")
    logger.info(f"  refine = {combo.get('refine', 'N/A')}")
    
    progressive = _rehydrate_progressive(combo.get('progressive', []))
    if progressive:
        logger.info(f"\n  Progressive stages ({len(progressive)}):")
        for i, stage in enumerate(progressive, 1):
            logger.info(f"    {i}. {stage.get('metric', '?'):40s} keep_k={stage.get('keep_k', '?')}")
    
    logger.info(f"\n  Final refinement:")
    logger.info(f"    decisiveness_topm = {combo.get('decisiveness_topm', 'N/A')}")
    logger.info(f"    kfold = {combo.get('kfold', 'N/A')}")
    logger.info(f"    weights = {combo.get('weights', 'N/A')}")


def evaluate_on_split(analyzer, split, combo: Dict[str, Any], part: str) -> Dict[str, float]:
    """Evaluate a combo on train or test using analyzer's exact logic"""
    prog = _rehydrate_progressive(combo.get('progressive'))
    return analyzer.evaluate_combo(
        split=split,
        part=part,
        eps=combo.get('eps'),
        progressive=prog,
        strict_band=combo.get('strict_band'),
        refine=combo.get('refine'),
        decisiveness_topm=combo.get('decisiveness_topm'),
        kfold=combo.get('kfold'),
        weights=combo.get('weights'),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Standalone layer selection strategy search using your existing infrastructure"
    )
    
    # Required
    parser.add_argument('results_dir', type=str, help="Results directory")
    
    # Search configuration
    parser.add_argument('--trials', type=int, default=10000,
                       help="Number of Optuna trials for main search (default: 10000)")
    parser.add_argument('--objective', choices=['win_rate', 'gap', 'robustness', 'within_epsilon'],
                       default='win_rate',
                       help="Optimization objective (default: win_rate)")
    parser.add_argument('--force-tail', action='store_true',
                       help="Force tail-safety metrics (margin/cvar) in first stage")
    
    # Focused pipeline
    parser.add_argument('--focused-trials', type=int, default=0,
                       help="Number of trials for focused pipeline search (default: 0 = disabled)")
    parser.add_argument('--focused-objective', 
                       choices=['win_rate', 'gap', 'robustness', 'within_epsilon'],
                       default='win_rate',
                       help="Objective for focused pipeline (default: win_rate)")
    
    # Data configuration
    parser.add_argument('--training-methods', type=str,
                       default='baseline_cross_entropy,baseline_brier,baseline_focal_adaptive',
                       help="Comma-separated training methods (default: baseline_cross_entropy,baseline_brier,baseline_focal_adaptive)")
    parser.add_argument('--acc-threshold', type=float, default=0.70,
                       help="Minimum accuracy threshold for seeds (default: 0.70)")
    
    # Output configuration
    parser.add_argument('--outdir', type=str, default='combo_search_results',
                       help="Output directory (default: combo_search_results)")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    # Robustness analysis
    parser.add_argument('--eps-rescore', type=str,
                       default='0.0005,0.00075,0.001,0.0015,0.002,0.0025,0.005',
                       help="Comma-separated ε values for robustness rescoring")
    parser.add_argument('--top-k-rescore', type=int, default=25,
                       help="Number of top trials to rescore (default: 25)")
    
    args = parser.parse_args()
    
    # ========================================================================
    # Setup
    # ========================================================================
    os.makedirs(args.outdir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("🔍 LAYER SELECTION COMBO SEARCH")
    logger.info("="*80)
    logger.info(f"Results dir: {args.results_dir}")
    logger.info(f"Output dir: {args.outdir}")
    logger.info(f"Main trials: {args.trials}")
    logger.info(f"Objective: {args.objective}")
    logger.info(f"Force tail metrics: {args.force_tail}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("")
    
    # ========================================================================
    # 1. Load Data
    # ========================================================================
    logger.info("📂 Loading results...")
    analyzer = EnhancedMetricAnalyzer(args.results_dir, acc_threshold=args.acc_threshold)
    
    training_methods = [m.strip() for m in args.training_methods.split(',') if m.strip()]
    analyzer.load_results(training_methods)
    
    # Create train/test split
    logger.info("✂️  Creating 80/20 train/test split...")
    split = analyzer.split_seeds(train_frac=0.8, random_state=args.seed)
    
    n_train = sum(len(parts.get('train', [])) for parts in split.values())
    n_test = sum(len(parts.get('test', [])) for parts in split.values())
    logger.info(f"   Train seeds: {n_train}")
    logger.info(f"   Test seeds: {n_test}")
    logger.info("")
    
    # ========================================================================
    # 2. Main Combo Search
    # ========================================================================
    logger.info("🎯 Running main combo search...")
    logger.info(f"   Strategy: {'tail-first' if args.force_tail else 'unconstrained'}")
    logger.info(f"   Trials: {args.trials}")
    logger.info("")
    
    logger.info(">>> Starting main combo search (study_combo_only)")
    study_combo = study_combo_only(
        analyzer=analyzer,
        split=split,
        n_trials=args.trials,
        objective=args.objective,
        force_tail=args.force_tail,
        seed=args.seed,
        storage=None,
        study_name=f"combo_only_{args.objective}_{args.seed}",
        output_dir=args.outdir,
    )
    logger.info("<<< Returned from study_combo_only")
    
    logger.info(">>> Starting print_summary")
    print_summary(study_combo, analyzer, split, args.objective)
    logger.info("<<< Finished print_summary")
    
    # ========================================================================
    # 3. Optional Focused Pipeline Search
    # ========================================================================
    study_focused = None
    if args.focused_trials > 0:
        logger.info("\n" + "="*80)
        logger.info("🎯 Running focused pipeline search...")
        logger.info(f"   Trials: {args.focused_trials}")
        logger.info(f"   Objective: {args.focused_objective}")
        logger.info("")
        
        study_focused = study_focused_pipeline(
            analyzer=analyzer,
            split=split,
            n_trials=args.focused_trials,
            objective=args.focused_objective,
            seed=args.seed,
        )
        
        print_summary(study_focused, analyzer, split, f"{args.focused_objective} (focused)")
    
    # ========================================================================
    # 4. Export Top Trials
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("💾 Exporting results...")
    logger.info("="*80)
    
    logger.info(">>> Starting save_top_trials (main)")
    paths_combo = save_top_trials(
        study_combo, 
        args.outdir, 
        top_k=args.top_k_rescore,
        name=f"combo_only_{args.objective}"
    )
    logger.info("<<< Finished save_top_trials (main)")
    
    if study_focused:
        paths_focused = save_top_trials(
            study_focused,
            args.outdir,
            top_k=args.top_k_rescore,
            name=f"focused_{args.focused_objective}"
        )
    else:
        paths_focused = {}
    
    logger.info(f"   Main search exports:")
    for key, path in paths_combo.items():
        logger.info(f"      {key}: {path}")
    
    if paths_focused:
        logger.info(f"   Focused search exports:")
        for key, path in paths_focused.items():
            logger.info(f"      {key}: {path}")
    
    # ========================================================================
    # 5. Evaluate Best Combo on Train/Test
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("📊 Evaluating best combo on train/test split...")
    logger.info("="*80)
    
    best_combo = study_combo.best_trial.user_attrs.get("combo", {})
    
    logger.info(">>> Starting evaluate_on_split (train)")
    train_perf = evaluate_on_split(analyzer, split, best_combo, part='train')
    logger.info("<<< Finished evaluate_on_split (train)")
    
    logger.info(">>> Starting evaluate_on_split (test)")
    test_perf = evaluate_on_split(analyzer, split, best_combo, part='test')
    logger.info("<<< Finished evaluate_on_split (test)")
    
    logger.info(f"\nTrain performance:")
    for k, v in train_perf.items():
        if isinstance(v, float):
            logger.info(f"   {k}: {v:.4f}")
        else:
            logger.info(f"   {k}: {v}")
    
    logger.info(f"\nTest performance:")
    for k, v in test_perf.items():
        if isinstance(v, float):
            logger.info(f"   {k}: {v:.4f}")
        else:
            logger.info(f"   {k}: {v}")
    
    # Check for overfitting
    if isinstance(train_perf.get('robustness'), (int, float)) and isinstance(test_perf.get('robustness'), (int, float)):
        gap = float(train_perf['robustness']) - float(test_perf['robustness'])
        status = ("⚠️  SEVERE OVERFITTING" if gap > 0.10
                else "⚠️  MODERATE OVERFITTING" if gap > 0.05
                else "✅ OK")
        logger.info("Train-test gap: %+.4f %s", gap, status)
    
    # ========================================================================
    # 6. Robustness Sweep Across ε
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("📈 Robustness analysis across ε values...")
    logger.info("="*80)
    
    eps_list = [float(x) for x in args.eps_rescore.split(',') if x.strip()]
    logger.info(f"   ε values: {eps_list}")
    logger.info(f"   Top-K trials: {args.top_k_rescore}")
    logger.info("")
    
    logger.info(">>> Starting rescore_top_combos_robustness")
    df_robustness = rescore_top_combos_robustness(
        study_combo,
        analyzer,
        split,
        top_k=args.top_k_rescore,
        eps_choices=eps_list
    )
    logger.info("<<< Finished rescore_top_combos_robustness")
    
    rb_csv = Path(args.outdir) / f"robustness_rescore_{args.objective}.csv"
    df_robustness.to_csv(rb_csv, index=False)
    logger.info(f"   Saved: {rb_csv}")
    
    # Print best ε for each metric
    if not df_robustness.empty:
        logger.info("\n   Best ε per trial (top 5 by robustness):")
        top5 = df_robustness.sort_values("robustness", ascending=False).head(5)
        for _, row in top5.iterrows():
            logger.info(
                f"      Trial #{int(row.get('trial_number', -1)):4d}: "
                f"ε={row.get('eps')} → robustness={row.get('robustness', 0):.4f}, "
                f"within_ε={row.get('within_epsilon', 0):.4f}"
            )
    
    # ========================================================================
    # 7. Save Portable Summary JSON
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("💾 Saving summary JSON...")
    logger.info("="*80)
    
    summary = {
        "config": {
            "results_dir": args.results_dir,
            "trials": args.trials,
            "objective": args.objective,
            "force_tail": args.force_tail,
            "focused_trials": args.focused_trials,
            "training_methods": training_methods,
            "seed": args.seed,
        },
        "best_combo": best_combo,
        "train_perf": train_perf,
        "test_perf": test_perf,
        "exports": {
            "main_search": paths_combo,
            "focused_search": paths_focused,
            "robustness_csv": str(rb_csv),
        },
        "optuna_best_value": study_combo.best_value,
        "optuna_best_trial": study_combo.best_trial.number,
    }
    
    # Add focused results if available
    if study_focused:
        summary["focused_best_combo"] = study_focused.best_trial.user_attrs.get("combo", {})
        summary["focused_best_value"] = study_focused.best_value
    
    summary_path = Path(args.outdir) / f"summary_{args.objective}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"   Saved: {summary_path}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("✅ SEARCH COMPLETE")
    logger.info("="*80)
    logger.info(f"📁 All results in: {args.outdir}")
    logger.info(f"📄 Summary JSON: {summary_path}")
    logger.info(f"\n🚀 Next steps:")
    logger.info(f"   1. Review {summary_path}")
    logger.info(f"   2. Check {rb_csv} for robustness across ε")
    logger.info(f"   3. Use best_combo in your main analysis:")
    logger.info(f"      python enhanced_metric_analysis.py {args.results_dir} \\")
    logger.info(f"        --use-combo {summary_path}")
    logger.info("")


if __name__ == '__main__':
    main()