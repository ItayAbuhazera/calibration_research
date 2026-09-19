"""
SLURM Wrapper for Alpha Ablation Study

This script parallelizes the alpha ablation experiment across SLURM job arrays.
It follows the pattern from run_hybrid_multi_metric_ensemble.py:
- Generates all experiment configurations
- Uses SLURM_ARRAY_TASK_ID to run individual experiments
- Skips already-completed experiments
- Handles errors gracefully

Usage:
    # Generate configs
    python run_alpha_ablation_slurm.py --generate-configs
    
    # Run single experiment by index
    python run_alpha_ablation_slurm.py --config-index 0
    
    # Run via SLURM (reads SLURM_ARRAY_TASK_ID)
    sbatch scripts/submit_alpha_ablation.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root on path for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.alpha_ablation_margin_cvar import (
    AlphaExperimentConfig,
    run_alpha_experiment,
    ALPHA_VALUES,
)

from utils.logging_config import get_logger
logger = get_logger(__name__)


# Default experimental parameters
DEFAULT_DATASETS = ["cifar10", "cifar100", "svhn"]
DEFAULT_MODELS = ["resnet18", "resnet50", "densenet121"]
DEFAULT_TRAINING_METHODS = ["augmix", "baseline_brier", "baseline_cross_entropy"]
DEFAULT_SEEDS = [15, 16, 17, 18, 19, 20]
DEFAULT_ALPHAS = ALPHA_VALUES  # [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def generate_all_configs(
    datasets: List[str],
    models: List[str],
    training_methods: List[str],
    seeds: List[int],
    alphas: List[float],
    output_base_dir: Path,
    normalization_methods: Optional[List[str]] = None,
    normalization_scales: Optional[List[float]] = None,
) -> List[Dict]:
    """
    Generate all experiment configurations and save to JSON.
    Each config represents one (dataset, model, training_method, seed, alpha) combo.
    All normalization variants will be tested within each experiment.
    
    Returns:
        List of config dictionaries suitable for JSON serialization
    """
    configs = []
    
    # Default normalization methods if not provided
    if normalization_methods is None:
        normalization_methods = ["tanh"]
    
    # Default normalization scales if not provided
    if normalization_scales is None:
        normalization_scales = [2.0]
    
    for dataset in datasets:
        for model in models:
            for training_method in training_methods:
                for seed in seeds:
                    for alpha in alphas:
                        # Format alpha for filename (e.g., 0.05 -> 005)
                        alpha_str = f"{int(alpha * 100):03d}"
                        
                        # Simple filename: seed{seed}_alpha{alpha_str}.json
                        # No method/scale in filename since all variants are in one file
                        filename = f"seed{seed}_alpha{alpha_str}.json"
                        
                        output_file = (
                            output_base_dir
                            / dataset
                            / model
                            / training_method
                            / filename
                        )
                        
                        config = {
                            'dataset': dataset,
                            'model': model,
                            'training_method': training_method,
                            'seed': seed,
                            'alpha': alpha,
                            'normalization_method': normalization_methods[0],  # Default/fallback
                            'normalization_scale': normalization_scales[0],  # Default/fallback
                            'normalization_methods': normalization_methods,  # Full list
                            'normalization_scales': normalization_scales,  # Full list
                            'output_file': str(output_file),
                        }
                        configs.append(config)
    
    logger.info(f"Generated {len(configs)} experiment configurations")
    logger.info(f"  Datasets: {len(datasets)} → {datasets}")
    logger.info(f"  Models: {len(models)} → {models}")
    logger.info(f"  Training methods: {len(training_methods)} → {training_methods}")
    logger.info(f"  Seeds: {len(seeds)} → {seeds}")
    logger.info(f"  Alphas: {len(alphas)} → {alphas}")
    logger.info(f"  Normalization methods: {normalization_methods}")
    logger.info(f"  Normalization scales: {normalization_scales}")
    logger.info(f"  Total: {len(datasets)} × {len(models)} × {len(training_methods)} × {len(seeds)} × {len(alphas)} = {len(configs)}")
    logger.info(f"  Each experiment tests {len(normalization_methods)} × {len(normalization_scales)} = {len(normalization_methods) * len(normalization_scales)} normalization variants")
    
    return configs


def save_configs(configs: List[Dict], config_file: Path) -> None:
    """Save experiment configurations to JSON file."""
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with config_file.open('w') as f:
        json.dump(configs, f, indent=2)
    logger.info(f"✓ Saved {len(configs)} configurations to: {config_file}")


def load_configs(config_file: Path) -> List[Dict]:
    """Load experiment configurations from JSON file."""
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with config_file.open('r') as f:
        configs = json.load(f)
    
    logger.info(f"Loaded {len(configs)} configurations from: {config_file}")
    return configs


def run_single_experiment(
    config: Dict,
    checkpoint_base_dir: Path,
    device: str = 'cuda',
    batch_size: int = 512,
    bins: int = 15,
    compression_ratio: Optional[float] = None,
    force: bool = False,
) -> bool:
    """
    Run a single alpha experiment from config dictionary.
    
    Returns:
        True if successful, False otherwise
    """
    output_file = Path(config['output_file'])
    
    # Check if already exists
    if output_file.exists() and not force:
        logger.info(f"⏭  Skipping existing experiment: {output_file.name}")
        logger.info(f"   Output file already exists: {output_file}")
        return True
    
    # Get normalization lists from config (new format)
    normalization_methods = config.get('normalization_methods', [config.get('normalization_method', 'tanh')])
    normalization_scales = config.get('normalization_scales', [config.get('normalization_scale', 2.0)])
    
    # Convert config dict to AlphaExperimentConfig
    exp_config = AlphaExperimentConfig(
        dataset=config['dataset'],
        model=config['model'],
        training_method=config['training_method'],
        seed=config['seed'],
        alpha=config['alpha'],
        normalization_method=normalization_methods[0],  # Default/fallback
        normalization_scale=normalization_scales[0],  # Default/fallback
        output_file=output_file,
    )
    
    # Log experiment details
    logger.info(f"\n{'='*80}")
    logger.info(f"Running Alpha Ablation Experiment (EFFICIENT MODE)")
    logger.info(f"{'='*80}")
    logger.info(f"  Dataset:         {exp_config.dataset}")
    logger.info(f"  Model:           {exp_config.model}")
    logger.info(f"  Training:        {exp_config.training_method}")
    logger.info(f"  Seed:            {exp_config.seed}")
    logger.info(f"  Alpha:           {exp_config.alpha}")
    logger.info(f"  Normalization Methods: {normalization_methods}")
    logger.info(f"  Normalization Scales:   {normalization_scales}")
    logger.info(f"  Output:          {exp_config.output_file}")
    logger.info(f"{'='*80}\n")
    
    try:
        result = run_alpha_experiment(
            config=exp_config,
            checkpoint_base_dir=checkpoint_base_dir,
            device=device,
            batch_size=batch_size,
            bins=bins,
            compression_ratio=compression_ratio,
            normalization_methods=normalization_methods,
            normalization_scales=normalization_scales,
        )
        
        if result is not None:
            logger.info(f"✓ Experiment completed successfully")
            logger.info(f"  Tested {len(result.get('selected_layer_per_config', {}))} selection strategies")
            # Log summary for each strategy
            for config_key, variant_result in result.get('selected_layer_per_config', {}).items():
                logger.info(f"    {config_key}: Layer {variant_result['layer']}, ECE {variant_result['test_ece']:.4f}")
            return True
        else:
            logger.error(f"✗ Experiment returned None (failed)")
            return False
            
    except Exception as e:
        logger.error(f"✗ Experiment failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SLURM wrapper for alpha ablation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Mode selection
    parser.add_argument(
        "--generate-configs",
        action='store_true',
        help="Generate all experiment configurations and save to JSON"
    )
    parser.add_argument(
        "--config-index",
        type=int,
        default=None,
        help="Run specific config by index (for testing or manual execution)"
    )
    
    # Configuration parameters
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default=DEFAULT_DATASETS,
        help="Datasets to test"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=DEFAULT_MODELS,
        help="Models to test"
    )
    parser.add_argument(
        "--training-methods",
        type=str,
        nargs='+',
        default=DEFAULT_TRAINING_METHODS,
        help="Training methods to test"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs='+',
        default=DEFAULT_SEEDS,
        help="Seeds to test"
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs='+',
        default=DEFAULT_ALPHAS,
        help="Alpha values to test"
    )
    parser.add_argument(
        "--normalization-methods",
        type=str,
        nargs='+',
        default=None,
        choices=["linear", "tanh"],
        help="Normalization methods to test: 'linear' (old method) or 'tanh' (new method). Default: ['tanh']"
    )
    parser.add_argument(
        "--normalization-scales",
        type=float,
        nargs='+',
        default=None,
        help="Normalization scale factors for tanh normalization (default: [2.0]). Only used with 'tanh' method. Larger values = less sensitive to CVaR changes."
    )
    
    # Paths
    parser.add_argument(
        "--checkpoint-base-dir",
        type=Path,
        default=Path("aaai_full_experiments/results/baseline"),
        help="Base directory containing trained model checkpoints"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/alpha_ablation_margin_cvar"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("configs/alpha_ablation_configs.json"),
        help="Path to save/load experiment configurations"
    )
    
    # Compute configuration
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for data processing"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=15,
        help="Number of bins for ECE calculation"
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=None,
        help="Compression ratio (None=auto: 32x for CIFAR-100, 16x for others)"
    )
    
    # Experiment control
    parser.add_argument(
        "--force",
        action='store_true',
        help="Force re-run even if output file exists"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    # Mode 1: Generate configurations
    if args.generate_configs:
        logger.info("="*80)
        logger.info("MODE: Generate Experiment Configurations")
        logger.info("="*80)
        
        configs = generate_all_configs(
            datasets=args.datasets,
            models=args.models,
            training_methods=args.training_methods,
            seeds=args.seeds,
            alphas=args.alphas,
            output_base_dir=args.output_dir,
            normalization_methods=args.normalization_methods,
            normalization_scales=args.normalization_scales,
        )
        
        save_configs(configs, args.config_file)
        
        logger.info("\n" + "="*80)
        logger.info("✓ Configuration generation complete")
        logger.info("="*80)
        logger.info(f"Next steps:")
        logger.info(f"  1. Test single job: python {__file__} --config-index 0")
        logger.info(f"  2. Submit SLURM array: sbatch scripts/submit_alpha_ablation.sh")
        logger.info(f"  3. Monitor progress: python scripts/check_alpha_ablation_progress.py")
        
        return 0
    
    # Mode 2: Run single experiment by index
    if args.config_index is not None:
        logger.info("="*80)
        logger.info(f"MODE: Run Single Experiment (Index: {args.config_index})")
        logger.info("="*80)
        
        configs = load_configs(args.config_file)
        
        if args.config_index < 0 or args.config_index >= len(configs):
            logger.error(f"Invalid config index: {args.config_index}")
            logger.error(f"Valid range: 0-{len(configs)-1}")
            return 1
        
        config = configs[args.config_index]
        logger.info(f"Running experiment {args.config_index + 1}/{len(configs)}")
        
        success = run_single_experiment(
            config=config,
            checkpoint_base_dir=args.checkpoint_base_dir,
            device=args.device,
            batch_size=args.batch_size,
            bins=args.bins,
            compression_ratio=args.compression_ratio,
            force=args.force,
        )
        
        return 0 if success else 1
    
    # Mode 3: SLURM array task (read SLURM_ARRAY_TASK_ID)
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        
        logger.info("="*80)
        logger.info(f"MODE: SLURM Array Task")
        logger.info("="*80)
        logger.info(f"  Job ID:       {os.environ.get('SLURM_JOB_ID', 'N/A')}")
        logger.info(f"  Array Task:   {task_id}")
        logger.info(f"  Array Job ID: {os.environ.get('SLURM_ARRAY_JOB_ID', 'N/A')}")
        logger.info(f"  Hostname:     {os.environ.get('HOSTNAME', 'N/A')}")
        logger.info("="*80 + "\n")
        
        configs = load_configs(args.config_file)
        
        if task_id < 0 or task_id >= len(configs):
            logger.error(f"Invalid SLURM_ARRAY_TASK_ID: {task_id}")
            logger.error(f"Valid range: 0-{len(configs)-1}")
            return 1
        
        config = configs[task_id]
        logger.info(f"Running experiment {task_id + 1}/{len(configs)}")
        
        success = run_single_experiment(
            config=config,
            checkpoint_base_dir=args.checkpoint_base_dir,
            device=args.device,
            batch_size=args.batch_size,
            bins=args.bins,
            compression_ratio=args.compression_ratio,
            force=args.force,
        )
        
        if success:
            logger.info("\n" + "="*80)
            logger.info(f"✓ SLURM Task {task_id} completed successfully")
            logger.info("="*80)
        else:
            logger.error("\n" + "="*80)
            logger.error(f"✗ SLURM Task {task_id} failed")
            logger.error("="*80)
        
        return 0 if success else 1
    
    # No mode specified
    parser.print_help()
    logger.error("\nError: Must specify one of:")
    logger.error("  --generate-configs")
    logger.error("  --config-index N")
    logger.error("  or run via SLURM (with SLURM_ARRAY_TASK_ID set)")
    return 1


if __name__ == "__main__":
    sys.exit(main())

