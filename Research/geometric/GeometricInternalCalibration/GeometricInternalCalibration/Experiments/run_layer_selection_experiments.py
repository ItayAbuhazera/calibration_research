#!/usr/bin/env python3
"""
run_layer_selection_experiments.py

Parallel runner for layer_selection.py experiments using SLURM.
Submits jobs for different (model, dataset, training_method, seed) combinations.

Usage:
    python run_layer_selection_experiments.py --config configs/layer_selection_sweep.yaml
    
Or with inline arguments:
    python run_layer_selection_experiments.py \
        --models resnet18 densenet121 \
        --datasets cifar10 cifar100 \
        --training-methods baseline_cross_entropy augmix \
        --seeds 0 1 2 \
        --results-base-dir experiments/trained_models \
        --output-base-dir experiments/layer_selection_results
"""

import argparse
import os
import subprocess
import sys
import shlex
import time
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from utils.logging_config import get_logger
logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_sbatch_command(
    job_name: str,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    training_method: str,
    seed: int,
    results_base_dir: str,
    output_base_dir: str,
    batch_size: int = 256,
    device: str = 'cuda',
    weights_csv: str = None,
    eval_baselines: bool = True,
    eval_corruptions: bool = False,
    cifar_c_dir: str = None,
    mce_only: bool = False,
    screen_layers: bool = False,
    target_acc: float = None,
    slurm_config: Dict[str, Any] = None
) -> str:
    """
    Generate an sbatch command string for a single layer_selection.py job.
    
    Returns:
        Complete sbatch command string ready to execute
    """
    
    # Default SLURM configuration
    if slurm_config is None:
        slurm_config = {}
    
    partition = slurm_config.get('partition', 'gpu')
    time_limit = slurm_config.get('time', '24:00:00')
    mem = slurm_config.get('mem', '32G')
    cpus = slurm_config.get('cpus', 4)
    gpus = slurm_config.get('gpus', 1)
    gpu_type = slurm_config.get('gpu_type', 'rtx_4090')
    conda_env = slurm_config.get('conda_env', 'tamar_n_env')
    
    # Get project root (parent of Experiments directory)
    project_root = Path(__file__).resolve().parent.parent
    
    # Create log directory
    log_dir = Path(output_dir) / "_layer_selection_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"
    
    # Build python command for the wrapped job
    python_cmd = [
        sys.executable,
        str(project_root / "Experiments" / "layer_selection.py"),
        "--model-name", model_name,
        "--dataset-name", dataset_name,
        "--training-method", training_method,
        "--seed", str(seed),
        "--results-base-dir", results_base_dir,
        "--output-base-dir", output_base_dir,
        "--batch-size", str(batch_size),
        "--device", device,
    ]
    
    # Optional arguments
    if weights_csv:
        python_cmd.extend(["--weights-csv", weights_csv])
    
    if not eval_baselines:
        python_cmd.append("--no-eval-baselines")
    
    if eval_corruptions:
        python_cmd.append("--eval-corruptions")
    
    if cifar_c_dir:
        python_cmd.extend(["--cifar-c-dir", cifar_c_dir])
    
    if mce_only:
        python_cmd.append("--mce-only")
    
    if screen_layers:
        python_cmd.append("--screen-layers")
    
    if target_acc is not None:
        python_cmd.extend(["--target-acc", str(target_acc)])
    
    safe_python_cmd = " ".join(shlex.quote(a) for a in python_cmd)
    
    # Create wrap script
    wrap_script = f"""
echo '========================================'
echo '🔬 SLURM JOB: Layer Selection Experiment'
echo '========================================'
echo 'Job ID       : $SLURM_JOB_ID'
echo 'Host         : $(hostname)'
echo 'Start Time   : $(date)'
echo '----------------------------------------'
echo 'Config:'
echo '  Model      : {model_name}'
echo '  Dataset    : {dataset_name}'
echo '  Training   : {training_method}'
echo '  Seed       : {seed}'
echo '  Batch Size : {batch_size}'
echo '  Device     : {device}'
echo '----------------------------------------'
module load anaconda || echo "Anaconda module not found, assuming env is active."
source activate {shlex.quote(conda_env)} || echo "Conda env '{conda_env}' activation failed."
export PYTHONPATH='{shlex.quote(str(project_root))}:$PYTHONPATH'
export CUDA_LAUNCH_BLOCKING=1
cd {shlex.quote(str(project_root))}
echo 'CMD: {safe_python_cmd}'
echo 'Environment Information'
echo '----------------------------------------'
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else \"N/A\")')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
nvidia-smi
echo '----------------------------------------'
echo 'CMD: {safe_python_cmd}'
{safe_python_cmd}
EXIT_CODE=$?
echo '----------------------------------------'
echo 'End Time     : $(date)'
echo 'Exit Code    : $EXIT_CODE'
echo '========================================'
"""
    clean_wrap_script = "\n".join(line.lstrip() for line in wrap_script.strip().split('\n'))
    
    # Build sbatch command
    sbatch_cmd = [
        "sbatch",
        f"--partition={shlex.quote(partition)}",
        f"--job-name={shlex.quote(job_name)}",
        f"--output={shlex.quote(str(log_file))}",
        f"--time={shlex.quote(time_limit)}",
        "--ntasks=1",
        f"--gpus={shlex.quote(str(gpus))}",
        f"--cpus-per-task={shlex.quote(str(cpus))}",
        f"--mem={shlex.quote(mem)}",
        f"--wrap={shlex.quote(clean_wrap_script)}",
    ]
    
    return " ".join(sbatch_cmd)


def submit_job(sbatch_cmd: str, dry_run: bool = False) -> str:
    """
    Submit SLURM job and return job ID.
    
    Args:
        sbatch_cmd: Complete sbatch command string
        dry_run: If True, only print the command without submitting
        
    Returns:
        Job ID (or "DRY_RUN" if dry_run=True)
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would submit: {sbatch_cmd}")
        return "DRY_RUN"
    
    try:
        result = subprocess.run(sbatch_cmd, shell=True, check=True, capture_output=True, text=True)
        # Parse job ID from output like "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        logger.info(f"✅ Submitted job {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to submit job: {e.stderr}")
        raise


def generate_experiment_grid(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all experiment configurations from config file.
    
    Returns:
        List of experiment configurations (dicts)
    """
    
    # Extract lists of parameters
    models = config.get('models', ['resnet18'])
    datasets = config.get('datasets', ['cifar10'])
    training_methods = config.get('training_methods', ['baseline_cross_entropy'])
    seeds = config.get('seeds', [0])
    
    # Generate Cartesian product
    experiments = []
    for model, dataset, training, seed in itertools.product(
        models, datasets, training_methods, seeds
    ):
        exp = {
            'model_name': model,
            'dataset_name': dataset,
            'training_method': training,
            'seed': seed,
        }
        experiments.append(exp)
    
    return experiments


def filter_existing_experiments(
    experiments: List[Dict[str, Any]],
    output_base_dir: str,
    skip_completed: bool = True
) -> List[Dict[str, Any]]:
    """
    Filter out experiments that have already been completed.
    
    Args:
        experiments: List of experiment configs
        output_base_dir: Base output directory
        skip_completed: If True, skip experiments with existing results
        
    Returns:
        Filtered list of experiments
    """
    if not skip_completed:
        return experiments
    
    filtered = []
    skipped_count = 0
    
    for exp in experiments:
        # Check if results file exists
        result_file = os.path.join(
            output_base_dir,
            exp['training_method'],
            exp['dataset_name'],
            exp['model_name'],
            f"seed{exp['seed']}",
            "multi_composite_analysis.json"
        )
        
        if os.path.exists(result_file):
            logger.info(f"⏭️  Skipping completed: {exp['model_name']}/{exp['dataset_name']}/{exp['training_method']}/seed{exp['seed']}")
            skipped_count += 1
        else:
            filtered.append(exp)
    
    logger.info(f"📊 Skipped {skipped_count} completed experiments, running {len(filtered)} new experiments")
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Parallel runner for layer_selection.py experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                       help="Path to YAML configuration file")
    
    # Experiment parameters (can override config file)
    parser.add_argument('--models', nargs='+', default=None,
                       help="Model architectures (e.g., resnet18 densenet121)")
    parser.add_argument('--datasets', nargs='+', default=None,
                       help="Datasets (e.g., cifar10 cifar100)")
    parser.add_argument('--training-methods', nargs='+', default=None,
                       help="Training methods (e.g., baseline_cross_entropy augmix)")
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                       help="Random seeds (e.g., 0 1 2)")
    
    # Paths
    parser.add_argument('--results-base-dir', type=str, default=None,
                       help="Base directory with pre-trained models")
    parser.add_argument('--output-base-dir', type=str, default=None,
                       help="Output directory for experiment results")
    parser.add_argument('--slurm-output-dir', type=str, default='slurm_jobs',
                       help="Directory for SLURM scripts and logs")
    
    # Technical settings
    parser.add_argument('--batch-size', type=int, default=256,
                       help="Batch size for data processing")
    parser.add_argument('--device', type=str, default='cuda',
                       help="Device for computations")
    parser.add_argument('--weights-csv', type=str, default=None,
                       help="Path to learned weights CSV for scoring layers")
    parser.add_argument('--target-acc', type=float, default=None,
                       help="Target accuracy for accuracy-specific paths")
    
    # Feature flags
    parser.add_argument('--eval-baselines', action='store_true', default=True,
                       help="Evaluate classical baselines")
    parser.add_argument('--no-eval-baselines', dest='eval_baselines', action='store_false')
    parser.add_argument('--eval-corruptions', action='store_true', default=False,
                       help="Evaluate on CIFAR-C and compute mCE")
    parser.add_argument('--cifar-c-dir', type=str, default=None,
                       help="Directory containing CIFAR-C data")
    parser.add_argument('--mce-only', action='store_true', default=False,
                       help="Skip clean ECE evaluation; compute mCE-only")
    parser.add_argument('--screen-layers', action='store_true', default=False,
                       help="Run meta-predictor screening")
    
    # SLURM settings
    parser.add_argument('--partition', type=str, default='gpu',
                       help="SLURM partition")
    parser.add_argument('--time', type=str, default='24:00:00',
                       help="Time limit (HH:MM:SS)")
    parser.add_argument('--mem', type=str, default='32G',
                       help="Memory per job")
    parser.add_argument('--cpus', type=int, default=4,
                       help="CPUs per task")
    parser.add_argument('--gpus', type=int, default=1,
                       help="Number of GPUs per job")
    parser.add_argument('--gpu-type', type=str, default='a100',
                       help="GPU type (e.g., a100, v100, rtx2080)")
    parser.add_argument('--conda-env', type=str, default='tamar_n_env',
                       help="Conda environment name")
    
    # Runner options
    parser.add_argument('--dry-run', action='store_true',
                       help="Generate scripts but don't submit jobs")
    parser.add_argument('--skip-completed', action='store_true', default=True,
                       help="Skip experiments with existing results")
    parser.add_argument('--no-skip-completed', dest='skip_completed', action='store_false',
                       help="Re-run all experiments even if results exist")
    parser.add_argument('--max-jobs', type=int, default=None,
                       help="Maximum number of jobs to submit (for testing)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        logger.info(f"📄 Loading configuration from {args.config}")
        config = load_config(args.config)
    
    # Override with command-line arguments
    if args.models:
        config['models'] = args.models
    if args.datasets:
        config['datasets'] = args.datasets
    if args.training_methods:
        config['training_methods'] = args.training_methods
    if args.seeds:
        config['seeds'] = args.seeds
    if args.results_base_dir:
        config['results_base_dir'] = args.results_base_dir
    if args.output_base_dir:
        config['output_base_dir'] = args.output_base_dir
    
    # Validate required parameters
    required = ['models', 'datasets', 'training_methods', 'seeds', 
                'results_base_dir', 'output_base_dir']
    missing = [k for k in required if k not in config]
    if missing:
        logger.error(f"❌ Missing required parameters: {missing}")
        logger.error("   Provide via --config file or command-line arguments")
        return 1
    
    # SLURM configuration
    slurm_config = {
        'partition': args.partition,
        'time': args.time,
        'mem': args.mem,
        'cpus': args.cpus,
        'gpus': args.gpus,
        'gpu_type': args.gpu_type,
        'conda_env': args.conda_env,
    }
    
    # Generate experiment grid
    logger.info("🔬 Generating experiment grid...")
    experiments = generate_experiment_grid(config)
    logger.info(f"   Generated {len(experiments)} total experiments")
    
    # Filter completed experiments
    experiments = filter_existing_experiments(
        experiments,
        config['output_base_dir'],
        skip_completed=args.skip_completed
    )
    
    if not experiments:
        logger.info("✅ All experiments already completed!")
        return 0
    
    # Limit number of jobs (for testing)
    if args.max_jobs and len(experiments) > args.max_jobs:
        logger.warning(f"⚠️  Limiting to {args.max_jobs} jobs (--max-jobs)")
        experiments = experiments[:args.max_jobs]
    
    # Create SLURM scripts and submit jobs
    logger.info(f"🚀 Submitting {len(experiments)} jobs...")
    
    submitted_jobs = []
    failed_jobs = []
    
    for i, exp in enumerate(experiments, 1):
        # Generate job name
        job_name = f"layer_sel_{exp['model_name']}_{exp['dataset_name']}_{exp['training_method']}_s{exp['seed']}"
        
        logger.info(f"[{i}/{len(experiments)}] Preparing {job_name}...")
        
        try:
            # Generate sbatch command
            sbatch_cmd = generate_sbatch_command(
                job_name=job_name,
                output_dir=args.slurm_output_dir,
                model_name=exp['model_name'],
                dataset_name=exp['dataset_name'],
                training_method=exp['training_method'],
                seed=exp['seed'],
                results_base_dir=config['results_base_dir'],
                output_base_dir=config['output_base_dir'],
                batch_size=args.batch_size,
                device=args.device,
                weights_csv=args.weights_csv,
                eval_baselines=args.eval_baselines,
                eval_corruptions=args.eval_corruptions,
                cifar_c_dir=args.cifar_c_dir,
                mce_only=args.mce_only,
                screen_layers=args.screen_layers,
                target_acc=args.target_acc,
                slurm_config=slurm_config,
            )
            
            # Submit job
            job_id = submit_job(sbatch_cmd, dry_run=args.dry_run)
            submitted_jobs.append((job_name, job_id, sbatch_cmd))
            if not args.dry_run:
                time.sleep(0.3)  # Small delay between submissions
            
        except Exception as e:
            logger.error(f"❌ Failed to submit {job_name}: {e}")
            failed_jobs.append((job_name, str(e)))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUBMISSION SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Successfully submitted: {len(submitted_jobs)}")
    logger.info(f"❌ Failed: {len(failed_jobs)}")
    
    if submitted_jobs:
        logger.info("\n📋 Submitted jobs:")
        for name, job_id, cmd in submitted_jobs[:10]:  # Show first 10
            logger.info(f"   {job_id}: {name}")
        if len(submitted_jobs) > 10:
            logger.info(f"   ... and {len(submitted_jobs) - 10} more")
    
    if failed_jobs:
        logger.info("\n❌ Failed jobs:")
        for name, error in failed_jobs:
            logger.info(f"   {name}: {error}")
    
    if args.dry_run:
        logger.info("\n🔍 DRY RUN MODE - No jobs were actually submitted")
        logger.info(f"   Commands would be executed directly with sbatch --wrap")
    else:
        logger.info(f"\n📊 Monitor jobs with: squeue -u $USER")
        logger.info(f"📁 SLURM logs in: {args.slurm_output_dir}/_layer_selection_logs/")
    
    logger.info("="*80)
    
    return 0 if not failed_jobs else 1


if __name__ == '__main__':
    exit(main())