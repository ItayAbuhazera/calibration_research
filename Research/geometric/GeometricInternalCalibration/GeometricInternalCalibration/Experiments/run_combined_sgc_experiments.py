#!/usr/bin/env python3
"""
run_combined_sgc_experiments.py

Parallel runner for the combined SGC calibration experiment (Experiments/combined_sgc_calibration.py)
using SLURM. Submits jobs for different (model, dataset, seed) combinations.

Usage:
    python run_combined_sgc_experiments.py --config configs/combined_sgc_sweep.yaml

Or with inline arguments:
    python run_combined_sgc_experiments.py \
        --models resnet18 resnet50 \
        --datasets cifar10 cifar100 \
        --seeds 11 12 13 \
        --results-base-dir /path/to/models \
        --output-base-dir results/combined_sgc
"""

import argparse
import os
import subprocess
import sys
import shlex
import time
import yaml
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from utils.logging_config import get_logger

logger = get_logger(__name__)

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
    training_method: str,
    seed: int,
    results_base_dir: str,
    output_base_dir: str,
    batch_size: int = 128,
    n_bins: int = 15,
    num_layers: int = 6,
    projection_dim: int = 256,
    slurm_config: Dict[str, Any] = None,
    conda_env: str = "geo_cuda12",
) -> str:
    """
    Generate an sbatch command string for a single combined SGC calibration job.
    Uses proper GPU allocation format: gpu_type:count (e.g., rtx_4090:1)

    Returns:
        Complete sbatch command string ready to execute
    """

    # Default SLURM configuration
    if slurm_config is None:
        slurm_config = {}

    partition = slurm_config.get('partition', 'gpu')
    time_limit = slurm_config.get('time', '0-12:00:00')  # Format: d-hh:mm:ss
    mem = slurm_config.get('mem', '60G')
    cpus = slurm_config.get('cpus', 4)
    gpus = slurm_config.get('gpus', 1)
    gpu_type = slurm_config.get('gpu_type', 'rtx_3090')
    conda_env = slurm_config.get('conda_env', conda_env)

    # Get project root
    project_root = Path(__file__).resolve().parent.parent

    # Create log directory
    log_dir = Path(output_dir) / "_combined_sgc_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"

    # Build python command
    python_cmd = [
        sys.executable,
        str(project_root / "Experiments" / "combined_sgc_calibration.py"),
        "--dataset", dataset_name,
        "--model", model_name,
        "--seed", str(seed),
        "--training_method", training_method,
        "--results_dir", results_base_dir,
        "--output_dir", output_base_dir,
        "--batch_size", str(batch_size),
        "--n_bins", str(n_bins),
        "--num_layers", str(num_layers),
        "--projection_dim", str(projection_dim),
        "--use_gpu",
        "--device", "cuda",
    ]

    safe_python_cmd = " ".join(shlex.quote(a) for a in python_cmd)

    # Create wrap script
    wrap_script = f"""
echo '========================================'
echo '🔬 SLURM JOB: Combined SGC Calibration'
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
echo '  Num Layers : {num_layers}'
echo '  Proj Dim   : {projection_dim}'
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

    # Build sbatch command - FIXED GPU FORMAT
    sbatch_cmd = [
        "sbatch",
        f"--partition={shlex.quote(partition)}",
        f"--job-name={shlex.quote(job_name)}",
        f"--output={shlex.quote(str(log_file))}",
        f"--time={shlex.quote(time_limit)}",
        "--ntasks=1",
        f"--gpus={shlex.quote(f'{gpu_type}:{gpus}')}",  # Use gpu_type:count format
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
    seeds = config.get('seeds', [42])

    # Generate Cartesian product
    experiments = []
    for model, dataset, training_method, seed in itertools.product(
        models, datasets, training_methods, seeds
    ):
        experiments.append({
            'model_name': model,
            'dataset_name': dataset,
            'training_method': training_method,
            'seed': seed,
        })

    return experiments


def construct_model_path_simple(
    base_dir: str, 
    training_method: str,
    dataset_name: str, 
    model_name: str, 
    seed: int
) -> str:
    """
    Best-effort model path constructor for validation.
    Matches the structure used by run_post_hoc_calibration.py
    """
    if construct_model_path_full is not None:
        try:
            return construct_model_path_full(base_dir, training_method, dataset_name, model_name, seed)
        except Exception:
            pass

    # Heuristic fallback matching common structure:
    # base_dir/baseline/training_method/dataset/model/seedN/training_dataset_model_seedN/results.json
    results_path = Path(base_dir) / "baseline" / training_method / dataset_name / model_name / f"seed{seed}" / f"{training_method}_{dataset_name}_{model_name}_seed{seed}" / "results.json"
    
    if results_path.exists():
        return str(results_path)
    
    # Alternative: check for model.pth directly
    model_path = Path(base_dir) / "baseline" / training_method / dataset_name / model_name / f"seed{seed}" / "model.pth"
    return str(model_path)


def validate_experiment_model(
    model_name: str,
    dataset_name: str,
    training_method: str,
    seed: int,
    results_base_dir: str,
) -> Tuple[bool, str]:
    """
    Check if model results exist and has sufficient accuracy (>50%).
    """
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

    # Load and check accuracy
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


def check_results_complete(result_file: str) -> Tuple[bool, str]:
    """
    Check if a completed results file contains the expected structure from combined_sgc_calibration.py.

    Args:
        result_file: Path to the combined SGC results JSON file

    Returns:
        (is_complete, reason): True if complete, False with reason otherwise
    """
    if not os.path.exists(result_file):
        return False, "File does not exist"

    try:
        with open(result_file, 'r') as f:
            results = json.load(f)

        # Check for expected structure: should have 'methods' key with calibration results
        if isinstance(results, dict) and 'methods' in results:
            methods = results['methods']
            if isinstance(methods, dict) and len(methods) > 0:
                # Check if at least one method has expected metrics
                first_method = list(methods.values())[0]
                if isinstance(first_method, dict) and 'ece' in first_method:
                    return True, "Results complete"
                return False, "Methods missing expected metrics"
            return False, "Methods dict is empty"
        return False, "Missing 'methods' key"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


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
        output_base_dir: Base output directory for combined SGC results
        results_base_dir: Base directory with pre-trained models
        skip_completed: If True, skip experiments with existing results

    Returns:
        Filtered list of experiments
    """
    filtered = []
    skipped_completed = 0
    skipped_invalid = 0
    skipped_incomplete = 0

    for exp in experiments:
        result_file = os.path.join(
            output_base_dir,
            f"{exp['dataset_name']}_{exp['model_name']}_seed{exp['seed']}_combined_sgc.json"
        )

        if skip_completed and os.path.exists(result_file):
            is_complete, reason = check_results_complete(result_file)

            if is_complete:
                logger.info(
                    "⏭️  Skipping completed: "
                    f"{exp['model_name']}/{exp['dataset_name']}/seed{exp['seed']}"
                )
                skipped_completed += 1
                continue
            else:
                logger.warning(
                    "🔄 Re-running incomplete experiment: "
                    f"{exp['model_name']}/{exp['dataset_name']}/seed{exp['seed']} - {reason}"
                )
                skipped_incomplete += 1
                # Don't skip - allow it to be re-run

        is_valid, reason = validate_experiment_model(
            model_name=exp['model_name'],
            dataset_name=exp['dataset_name'],
            training_method=exp['training_method'],
            seed=exp['seed'],
            results_base_dir=results_base_dir
        )

        if not is_valid:
            logger.warning(
                "⚠️  Skipping invalid model: "
                f"{exp['model_name']}/{exp['dataset_name']}/seed{exp['seed']} - {reason}"
            )
            skipped_invalid += 1
            continue

        logger.debug(
            f"✅ Valid: {exp['model_name']}/{exp['dataset_name']}/seed{exp['seed']} - {reason}"
        )
        filtered.append(exp)

    logger.info(
        "📊 Skipped "
        f"{skipped_completed} completed experiments, "
        f"{skipped_incomplete} incomplete (will re-run), "
        f"{skipped_invalid} invalid models, running {len(filtered)} experiments"
    )
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Parallel runner for combined SGC calibration experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Configuration
    parser.add_argument('--config', type=str, default=None,
                        help="Path to YAML configuration file")

    # Experiment parameters (can override config file)
    parser.add_argument('--models', nargs='+', default=None,
                        help="Model architectures (e.g., resnet18 resnet50 resnet101 densenet121)")
    parser.add_argument('--datasets', nargs='+', default=None,
                        help="Datasets (e.g., cifar10 cifar100 tiny_imagenet)")
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help="Random seeds (e.g., 11 12 13)")
    parser.add_argument('--training-methods', nargs='+', default=None,
                        dest='training_methods',
                        help="Training methods (e.g., baseline_cross_entropy)")
    
    # Experimental parameters
    parser.add_argument('--num-layers', type=int, default=6,
                        dest='num_layers',
                        help="Number of layers to sample for SGC features")
    parser.add_argument('--projection-dim', type=int, default=256,
                        dest='projection_dim',
                        help="Target dimension for feature projection")
    parser.add_argument('--n-bins', type=int, default=15,
                        dest='n_bins',
                        help="Number of bins for ECE calculation")

    # Paths
    parser.add_argument('--results-base-dir', type=str, default=None,
                        help="Base directory with pre-trained models")
    parser.add_argument('--output-base-dir', type=str, default=None,
                        help="Output directory for combined SGC results")
    parser.add_argument('--slurm-output-dir', type=str, default='slurm_jobs',
                        help="Directory for SLURM scripts and logs")

    # Technical settings
    parser.add_argument('--batch-size', type=int, default=128,
                        help="Batch size for data processing")

    # SLURM settings
    parser.add_argument('--partition', type=str, default='gpu',
                        help="SLURM partition")
    parser.add_argument('--time', type=str, default='0-12:00:00',
                        help="Time limit (d-hh:mm:ss format)")
    parser.add_argument('--mem', type=str, default='64G',
                        help="Memory per job")
    parser.add_argument('--cpus', type=int, default=4,
                        help="CPUs per task")
    parser.add_argument('--gpus', type=int, default=1,
                        help="Number of GPUs per job")
    parser.add_argument('--gpu-type', type=str, default='rtx_3090',
                        help="GPU type (e.g., a100, v100, rtx_4090)")
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
    if args.seeds:
        config['seeds'] = args.seeds
    if args.training_methods:
        config['training_methods'] = args.training_methods
    if args.results_base_dir:
        config['results_base_dir'] = args.results_base_dir
    if args.output_base_dir:
        config['output_base_dir'] = args.output_base_dir

    # Validate required parameters
    required = ['models', 'datasets', 'seeds', 'results_base_dir', 'output_base_dir']
    # Note: training_methods has a default, so not strictly required
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
    total_experiments = len(experiments)
    logger.info(f"   Generated {total_experiments} total experiments")

    # Filter completed / invalid experiments
    experiments = filter_existing_experiments(
        experiments,
        output_base_dir=config['output_base_dir'],
        results_base_dir=config['results_base_dir'],
        skip_completed=args.skip_completed
    )

    skipped_experiments = total_experiments - len(experiments)

    if not experiments:
        logger.info("✅ All experiments already completed!")
        return 0

    # Limit number of jobs (for testing)
    if args.max_jobs and len(experiments) > args.max_jobs:
        logger.warning(f"⚠️  Limiting to {args.max_jobs} jobs (--max-jobs)")
        experiments = experiments[:args.max_jobs]

    # Organize and submit jobs
    logger.info(f"🚀 Organizing {len(experiments)} experiments for submission...")

    # Group experiments by seed -> model
    experiments_by_seed = {}
    for exp in experiments:
        seed = exp['seed']
        experiments_by_seed.setdefault(seed, {})
        model = exp['model_name']
        experiments_by_seed[seed].setdefault(model, [])
        experiments_by_seed[seed][model].append(exp)

    total_jobs = len(experiments)
    logger.info(f"🚀 Submitting {total_jobs} jobs in order: seed -> model -> dataset")

    submitted_jobs = []
    failed_jobs = []
    job_counter = 0

    for seed in sorted(experiments_by_seed.keys()):
        logger.info(f"\n🌱 Processing seed {seed}...")
        seed_jobs = []

        for model in sorted(experiments_by_seed[seed].keys()):
            sorted_exps = sorted(experiments_by_seed[seed][model],
                                 key=lambda x: x['dataset_name'])

            for exp in sorted_exps:
                job_name = f"csgc_{exp['model_name']}_{exp['dataset_name']}_{exp['training_method']}_s{exp['seed']}"

                try:
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
                        n_bins=args.n_bins,
                        num_layers=args.num_layers,
                        projection_dim=args.projection_dim,
                        slurm_config=slurm_config,
                        conda_env=args.conda_env,
                    )
                    seed_jobs.append((job_name, sbatch_cmd, exp))
                except Exception as e:
                    logger.error(f"❌ Failed to prepare {job_name}: {e}")
                    failed_jobs.append((job_name, str(e)))

        models_in_seed = sorted(set(j[2]['model_name'] for j in seed_jobs))
        logger.info(f"   Submitting {len(seed_jobs)} jobs for seed {seed} (models: {', '.join(models_in_seed)})")
        for job_name, sbatch_cmd, exp in seed_jobs:
            job_counter += 1
            logger.info(f"   [{job_counter}/{total_jobs}] Submitting {job_name}...")

            try:
                job_id = submit_job(sbatch_cmd, dry_run=args.dry_run)
                submitted_jobs.append((job_name, job_id, sbatch_cmd))
                if not args.dry_run:
                    time.sleep(0.3)
            except Exception as e:
                logger.error(f"❌ Failed to submit {job_name}: {e}")
                failed_jobs.append((job_name, str(e)))

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUBMISSION SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Successfully submitted: {len(submitted_jobs)}")
    logger.info(f"⏭️  Skipped (total): {skipped_experiments}")
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
        logger.info(f"📁 SLURM logs in: {args.slurm_output_dir}/_combined_sgc_logs/")

    logger.info("="*80)

    return 0 if not failed_jobs else 1


if __name__ == '__main__':
    exit(main())
