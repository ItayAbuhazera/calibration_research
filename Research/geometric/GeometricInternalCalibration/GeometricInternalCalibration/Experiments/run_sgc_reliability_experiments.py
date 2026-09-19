#!/usr/bin/env python3
"""
run_sgc_reliability_experiments.py

Parallel runner for SGC reliability diagram generation using SLURM.
Submits jobs for different (model, dataset, seed, method) combinations.

Usage:
    python run_sgc_reliability_experiments.py --config configs/sgc_reliability_sweep.yaml

Or with inline arguments:
    python run_sgc_reliability_experiments.py \
        --models resnet50 resnet101 densenet121 \
        --datasets cifar10 cifar100 tiny_imagenet \
        --seeds 11 12 13 14 15 16 \
        --methods sgc sgc_lite sgc_trust_score temperature_scaling top_label_isotonic uncalibrated dac \
        --results-base-dir /path/to/checkpoints \
        --output-base-dir reliability_data
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
from typing import List, Dict, Any, Tuple
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

try:
    from Experiments.run_post_hoc_calibration import construct_model_path as construct_model_path_full
except ImportError:
    construct_model_path_full = None


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
    seed: int,
    method: str,
    results_base_dir: str,
    output_base_dir: str,
    training_method: str = 'baseline_cross_entropy',
    batch_size: int = 128,
    slurm_config: Dict[str, Any] = None,
    sgc_L: int = 6,
    sgc_d: int = 256,
    sgc_lite_K: int = 256,
    n_bins: int = 10,
    conda_env: str = "geo_cuda12",
) -> str:
    """
    Generate an sbatch command string for a single SGC reliability diagram job.

    Returns:
        Complete sbatch command string ready to execute
    """
    # Default SLURM configuration
    if slurm_config is None:
        slurm_config = {}

    partition = slurm_config.get('partition', 'gpu')
    time_limit = slurm_config.get('time', '0-06:00:00')
    mem = slurm_config.get('mem', '48G')
    cpus = slurm_config.get('cpus', 4)
    gpus = slurm_config.get('gpus', 1)
    gpu_type = slurm_config.get('gpu_type', 'rtx_3090')
    conda_env = slurm_config.get('conda_env', conda_env)

    # Get project root
    project_root = Path(__file__).resolve().parent.parent

    # Create log directory
    log_dir = Path(output_dir) / "_sgc_reliability_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"

    # Build python command
    python_cmd = [
        sys.executable,
        str(project_root / "Experiments" / "sgc_reliability_diagrams.py"),
        "--model", model_name,
        "--dataset", dataset_name,
        "--seed", str(seed),
        "--method", method,
        "--results-base-dir", results_base_dir,
        "--output-dir", output_base_dir,
        "--training-method", training_method,
        "--batch-size", str(batch_size),
        "--sgc-L", str(sgc_L),
        "--sgc-d", str(sgc_d),
        "--sgc-lite-K", str(sgc_lite_K),
        "--n-bins", str(n_bins),
    ]

    safe_python_cmd = " ".join(shlex.quote(a) for a in python_cmd)

    # Create wrap script
    wrap_script = f"""
echo '========================================'
echo 'SLURM JOB: SGC Reliability Diagram Generation'
echo '========================================'
echo 'Job ID       : $SLURM_JOB_ID'
echo 'Host         : $(hostname)'
echo 'Start Time   : $(date)'
echo '----------------------------------------'
echo 'Config:'
echo '  Model      : {model_name}'
echo '  Dataset    : {dataset_name}'
echo '  Method     : {method}'
echo '  Seed       : {seed}'
echo '  Training   : {training_method}'
echo '  Batch Size : {batch_size}'
echo '  SGC L      : {sgc_L}'
echo '  SGC d      : {sgc_d}'
echo '  SGC-Lite K : {sgc_lite_K}'
echo '  N Bins     : {n_bins}'
echo '----------------------------------------'
module load anaconda || echo "Anaconda module not found, assuming env is active."
source activate {shlex.quote(conda_env)} || echo "Conda env '{conda_env}' activation failed."
export PYTHONPATH='{shlex.quote(str(project_root))}:$PYTHONPATH'
export CUDA_LAUNCH_BLOCKING=1
cd {shlex.quote(str(project_root))}
echo 'Environment Information'
echo '----------------------------------------'
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else \"N/A\")')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
if [ "{slurm_config.get('device', 'cuda')}" = "cuda" ]; then
    nvidia-smi
fi
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
        f"--gpus={shlex.quote(f'{gpu_type}:{gpus}')}",
        f"--mem={shlex.quote(mem)}",
        f"--cpus-per-task={shlex.quote(str(cpus))}",
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
        logger.info(f"[DRY RUN] Would submit: {sbatch_cmd[:200]}...")
        return "DRY_RUN"

    try:
        result = subprocess.run(sbatch_cmd, shell=True, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        logger.info(f"Submitted job {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to submit job: {e.stderr}")
        raise


def generate_experiment_grid(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all experiment configurations from config file.

    Returns:
        List of experiment configurations (dicts)
    """
    models = config.get('models', ['resnet50'])
    datasets = config.get('datasets', ['cifar100'])
    methods = config.get('methods', ['sgc', 'sgc_lite', 'sgc_trust_score', 'temperature_scaling', 'top_label_isotonic', 'uncalibrated', 'dac'])
    training_methods = config.get('training_methods', ['baseline_cross_entropy'])
    seeds = config.get('seeds', [11, 12, 13, 14, 15, 16])

    experiments = []
    for model, dataset, method, training_method, seed in itertools.product(
        models, datasets, methods, training_methods, seeds
    ):
        experiments.append({
            'model_name': model,
            'dataset_name': dataset,
            'method': method,
            'training_method': training_method,
            'seed': seed,
        })

    return experiments


def construct_output_path(
    output_base_dir: str,
    model_name: str,
    dataset_name: str,
    method: str,
    seed: int
) -> str:
    """Construct the expected output file path for a given experiment."""
    return os.path.join(
        output_base_dir,
        f"{model_name}_{dataset_name}_seed{seed}_{method}_per_sample.npz"
    )


def validate_experiment_model(
    model_name: str,
    dataset_name: str,
    training_method: str,
    seed: int,
    results_base_dir: str,
) -> Tuple[bool, str]:
    """
    Check if model checkpoint/results exist.
    """
    import json

    if results_base_dir is None:
        return False, "results_base_dir not provided"

    # Construct path to results file
    results_path = os.path.join(
        results_base_dir,
        "baseline",
        training_method,
        dataset_name,
        model_name,
        f"seed{seed}",
        f"{training_method}_{dataset_name}_{model_name}_seed{seed}",
        "results.json"
    )

    if not os.path.exists(results_path):
        return False, f"Results file not found: {results_path}"

    try:
        with open(results_path, 'r') as f:
            results = json.load(f)

        if "evaluation_results" not in results:
            return False, f"'evaluation_results' not found in {results_path}"

        if "accuracy" not in results["evaluation_results"]:
            return False, "'accuracy' not found in evaluation_results"

        accuracy = results["evaluation_results"]["accuracy"]

        if accuracy < 50.0:
            return False, f"Low accuracy: {accuracy:.2f}% < 50%"

        return True, f"Valid model with {accuracy:.2f}% accuracy"

    except Exception as e:
        return False, f"Error reading results: {str(e)}"


def filter_existing_experiments(
    experiments: List[Dict[str, Any]],
    output_base_dir: str,
    results_base_dir: str,
    skip_completed: bool = True
) -> List[Dict[str, Any]]:
    """
    Filter out experiments that have already been completed or have invalid models.

    Args:
        experiments: List of experiment configs
        output_base_dir: Base output directory for reliability data
        results_base_dir: Base directory with pre-trained models
        skip_completed: If True, skip experiments with existing results

    Returns:
        Filtered list of experiments
    """
    filtered = []
    skipped_completed = 0
    skipped_invalid = 0

    for exp in experiments:
        # Check if output already exists
        output_path = construct_output_path(
            output_base_dir,
            exp['model_name'],
            exp['dataset_name'],
            exp['method'],
            exp['seed']
        )

        if skip_completed and os.path.exists(output_path):
            logger.debug(
                f"Skipping completed: {exp['model_name']}/{exp['dataset_name']}/"
                f"{exp['method']}/seed{exp['seed']}"
            )
            skipped_completed += 1
            continue

        # Validate model exists
        is_valid, reason = validate_experiment_model(
            exp['model_name'],
            exp['dataset_name'],
            exp['training_method'],
            exp['seed'],
            results_base_dir
        )

        if not is_valid:
            logger.debug(f"Skipping invalid: {exp['model_name']}/{exp['dataset_name']}/seed{exp['seed']} - {reason}")
            skipped_invalid += 1
            continue

        filtered.append(exp)

    logger.info(f"Filtered: {skipped_completed} completed, {skipped_invalid} invalid models")
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description='SLURM runner for SGC reliability diagram generation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Configuration
    parser.add_argument('--config', type=str, default=None,
                        help="Path to YAML configuration file")

    # Experiment parameters
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help="Models to evaluate (e.g., resnet50 resnet101 densenet121)")
    parser.add_argument('--datasets', nargs='+', type=str, default=None,
                        help="Datasets to evaluate (e.g., cifar10 cifar100 tiny_imagenet)")
    parser.add_argument('--methods', nargs='+', type=str, default=None,
                        help="Calibration methods (sgc, sgc_lite, sgc_trust_score, temperature_scaling, top_label_isotonic, uncalibrated, dac)")
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help="Random seeds for experiments")
    parser.add_argument('--training-methods', nargs='+', type=str,
                        default=['baseline_cross_entropy'],
                        dest='training_methods',
                        help="Training methods used for models")

    # SGC parameters
    parser.add_argument('--sgc-L', type=int, default=6,
                        dest='sgc_L',
                        help="Number of layers to sample for SGC")
    parser.add_argument('--sgc-d', type=int, default=256,
                        dest='sgc_d',
                        help="Target dimension for SGC projection")
    parser.add_argument('--sgc-lite-K', type=int, default=256,
                        dest='sgc_lite_K',
                        help="Number of coordinates for SGC-Lite")
    parser.add_argument('--n-bins', type=int, default=10,
                        dest='n_bins',
                        help="Number of bins for reliability diagrams")

    # Paths
    parser.add_argument('--results-base-dir', type=str, default=None,
                        help="Base directory with pre-trained model checkpoints")
    parser.add_argument('--output-base-dir', type=str, default='reliability_data',
                        help="Output directory for per-sample reliability data")
    parser.add_argument('--slurm-output-dir', type=str, default='slurm_jobs',
                        help="Directory for SLURM scripts and logs")

    # Technical settings
    parser.add_argument('--batch-size', type=int, default=128,
                        help="Batch size for data processing")

    # SLURM settings
    parser.add_argument('--partition', type=str, default='gpu',
                        help="SLURM partition")
    parser.add_argument('--time', type=str, default='0-06:00:00',
                        help="Time limit (d-hh:mm:ss format)")
    parser.add_argument('--mem', type=str, default='48G',
                        help="Memory per job")
    parser.add_argument('--cpus', type=int, default=4,
                        help="CPUs per task")
    parser.add_argument('--gpus', type=int, default=1,
                        help="Number of GPUs per job")
    parser.add_argument('--gpu-type', type=str, default='rtx_3090',
                        help="GPU type (e.g., a100, v100, rtx_4090, rtx_3090)")
    parser.add_argument('--conda-env', type=str, default='geo_cuda12',
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
    parser.add_argument('--submit-delay', type=float, default=0.3,
                        help="Delay in seconds between job submissions")

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

    # Override with command-line arguments
    if args.models:
        config['models'] = args.models
    if args.datasets:
        config['datasets'] = args.datasets
    if args.methods:
        config['methods'] = args.methods
    if args.seeds:
        config['seeds'] = args.seeds
    if args.training_methods:
        config['training_methods'] = args.training_methods
    if args.results_base_dir:
        config['results_base_dir'] = args.results_base_dir
    if args.output_base_dir:
        config['output_base_dir'] = args.output_base_dir

    # Set defaults if not provided
    config.setdefault('models', ['resnet50', 'resnet101', 'densenet121'])
    config.setdefault('datasets', ['cifar10', 'cifar100', 'tiny_imagenet'])
    config.setdefault('methods', ['sgc', 'sgc_lite', 'sgc_trust_score', 'temperature_scaling', 'top_label_isotonic', 'uncalibrated', 'dac'])
    config.setdefault('seeds', [11, 12, 13, 14, 15, 16])
    config.setdefault('training_methods', ['baseline_cross_entropy'])

    # Validate required parameters
    required = ['results_base_dir', 'output_base_dir']
    missing = [k for k in required if k not in config or config[k] is None]
    if missing:
        logger.error(f"Missing required parameters: {missing}")
        logger.error("   Provide via --config file or command-line arguments")
        return 1

    # Create output directory
    Path(config['output_base_dir']).mkdir(parents=True, exist_ok=True)

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
    logger.info("Generating experiment grid...")
    experiments = generate_experiment_grid(config)
    total_experiments = len(experiments)
    logger.info(f"   Generated {total_experiments} total experiments")
    logger.info(f"   Models: {config['models']}")
    logger.info(f"   Datasets: {config['datasets']}")
    logger.info(f"   Methods: {config['methods']}")
    logger.info(f"   Seeds: {config['seeds']}")

    # Filter completed / invalid experiments
    experiments = filter_existing_experiments(
        experiments,
        output_base_dir=config['output_base_dir'],
        results_base_dir=config['results_base_dir'],
        skip_completed=args.skip_completed
    )

    skipped_experiments = total_experiments - len(experiments)

    if not experiments:
        logger.info("All experiments already completed!")
        return 0

    # Limit number of jobs (for testing)
    if args.max_jobs and len(experiments) > args.max_jobs:
        logger.warning(f"Limiting to {args.max_jobs} jobs (--max-jobs)")
        experiments = experiments[:args.max_jobs]

    # Organize and submit jobs
    logger.info(f"Organizing {len(experiments)} experiments for submission...")

    # Group experiments by seed first -> method -> model -> dataset
    # This allows all experiments with the same seed to run in parallel
    experiments_by_seed = {}
    for exp in experiments:
        seed = exp['seed']
        experiments_by_seed.setdefault(seed, [])
        experiments_by_seed[seed].append(exp)

    total_jobs = len(experiments)
    logger.info(f"Submitting {total_jobs} jobs...")

    submitted_jobs = []
    failed_jobs = []
    job_counter = 0

    for seed in sorted(experiments_by_seed.keys()):
        logger.info(f"\nProcessing seed: {seed}...")
        seed_exps = experiments_by_seed[seed]

        # Sort by method, model, dataset
        sorted_exps = sorted(
            seed_exps,
            key=lambda x: (x['method'], x['model_name'], x['dataset_name'])
        )

        for exp in sorted_exps:
            job_name = f"sgc_{exp['method']}_{exp['model_name']}_{exp['dataset_name']}_s{exp['seed']}"

            try:
                sbatch_cmd = generate_sbatch_command(
                    job_name=job_name,
                    output_dir=args.slurm_output_dir,
                    model_name=exp['model_name'],
                    dataset_name=exp['dataset_name'],
                    seed=exp['seed'],
                    method=exp['method'],
                    results_base_dir=config['results_base_dir'],
                    output_base_dir=config['output_base_dir'],
                    training_method=exp['training_method'],
                    batch_size=args.batch_size,
                    slurm_config=slurm_config,
                    sgc_L=args.sgc_L,
                    sgc_d=args.sgc_d,
                    sgc_lite_K=args.sgc_lite_K,
                    n_bins=args.n_bins,
                    conda_env=args.conda_env,
                )

                job_counter += 1
                logger.info(f"   [{job_counter}/{total_jobs}] Submitting {job_name}...")

                job_id = submit_job(sbatch_cmd, dry_run=args.dry_run)
                submitted_jobs.append((job_name, job_id, sbatch_cmd))

                if not args.dry_run:
                    time.sleep(args.submit_delay)

            except Exception as e:
                logger.error(f"Failed to submit {job_name}: {e}")
                failed_jobs.append((job_name, str(e)))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUBMISSION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Successfully submitted: {len(submitted_jobs)}")
    logger.info(f"Skipped (completed/invalid): {skipped_experiments}")
    logger.info(f"Failed: {len(failed_jobs)}")

    if submitted_jobs:
        logger.info("\nSubmitted jobs:")
        for name, job_id, cmd in submitted_jobs[:10]:
            logger.info(f"   {job_id}: {name}")
        if len(submitted_jobs) > 10:
            logger.info(f"   ... and {len(submitted_jobs) - 10} more")

    if failed_jobs:
        logger.info("\nFailed jobs:")
        for name, error in failed_jobs:
            logger.info(f"   {name}: {error}")

    if args.dry_run:
        logger.info("\nDRY RUN MODE - No jobs were actually submitted")
    else:
        logger.info(f"\nMonitor jobs with: squeue -u $USER")
        logger.info(f"SLURM logs in: {args.slurm_output_dir}/_sgc_reliability_logs/")

    logger.info("=" * 80)

    return 0 if not failed_jobs else 1


if __name__ == '__main__':
    exit(main())