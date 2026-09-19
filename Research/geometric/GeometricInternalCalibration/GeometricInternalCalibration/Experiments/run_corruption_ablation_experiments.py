#!/usr/bin/env python3
"""
run_corruption_ablation_experiments.py

Parallel runner for corruption ablation experiments using SLURM.
Submits jobs for different (model, dataset, training_method, seed) combinations
to evaluate calibration methods on CIFAR-C corruptions.

Usage:
    python run_corruption_ablation_experiments.py --config configs/corruption_ablation_sweep.yaml
    
Or with inline arguments:
    python run_corruption_ablation_experiments.py \
        --models resnet18 densenet121 \
        --datasets cifar10 cifar100 \
        --training-methods baseline augmix \
        --seeds 11 12 13 \
        --results-base-dir aaai_full_experiments/results \
        --output-base-dir corruption_ablation_results \
        --cifar-c-dir data/cifar10-c
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
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from utils.logging_config import get_logger
from Experiments.run_post_hoc_calibration import construct_model_path
logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_model_path(
    model_name: str,
    dataset_name: str,
    training_method: str,
    seed: int,
    results_base_dir: str
) -> Tuple[str, bool]:
    """
    Find the model checkpoint path for a given experiment configuration using the
    same logic as the rest of the codebase (`construct_model_path`).
    """
    try:
        # Reuse the canonical path constructor so layout matches all other experiments.
        model_path = construct_model_path(
            results_base_dir,
            training_method,
            dataset_name,
            model_name,
            seed,
        )
    except Exception as e:
        logger.error(f"construct_model_path failed for {training_method}/{dataset_name}/{model_name}/seed{seed}: {e}")
        return "", False

    if os.path.exists(model_path):
        return model_path, True

    return "", False


def generate_sbatch_command(
    job_name: str,
    output_dir: str,
    model_path: str,
    model_name: str,
    dataset_name: str,
    cifar_c_dir: str,
    results_base_dir: str,
    output_base_dir: str,
    batch_size: int = 256,
    device: str = 'cuda',
    target_dim: int = 256,
    n_random_layers: int = 6,
    seed: int = 42,
    skip_baselines: bool = False,
    skip_geometric: bool = False,
    skip_coordinate: bool = False,
    run_normalization_ablation: bool = False,
    run_acc_cali: bool = False,
    output_name: str = None,
    slurm_config: Dict[str, Any] = None
) -> str:
    """
    Generate an sbatch command string for a single corruption ablation job.
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
    gpu_type = slurm_config.get('gpu_type', 'rtx_4090')
    conda_env = slurm_config.get('conda_env', 'geo_cuda12')
    
    # Get project root
    project_root = Path(__file__).resolve().parent.parent
    
    # Create log directory
    log_dir = Path(output_dir) / "_corruption_ablation_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"
    
    # Build python command
    python_cmd = [
        sys.executable,
        str(project_root / "Experiments" / "run_corruption_ablation.py"),
        "--model_path", model_path,
        "--model_name", model_name,
        "--dataset", dataset_name,
        "--cifar_c_dir", cifar_c_dir,
        "--batch_size", str(batch_size),
        "--target_dim", str(target_dim),
        "--n_random_layers", str(n_random_layers),
        "--seed", str(seed),
        "--output_dir", output_base_dir,
        "--device", device,
    ]
    
    if output_name:
        python_cmd.extend(["--output_name", output_name])
    if skip_baselines:
        python_cmd.append("--skip_baselines")
    if skip_geometric:
        python_cmd.append("--skip_geometric")
    if skip_coordinate:
        python_cmd.append("--skip_coordinate")
    if run_normalization_ablation:
        python_cmd.append("--run_normalization_ablation")
    if run_acc_cali:
        python_cmd.append("--run_acc_cali")
    
    safe_python_cmd = " ".join(shlex.quote(a) for a in python_cmd)
    
    # Create wrap script
    run_normalization_ablation_str = "True" if run_normalization_ablation else "False"
    run_acc_cali_str = "True" if run_acc_cali else "False"
    wrap_script = f"""
echo '========================================'
echo '🔬 SLURM JOB: Corruption Ablation Evaluation'
echo '========================================'
echo 'Job ID       : $SLURM_JOB_ID'
echo 'Host         : $(hostname)'
echo 'Start Time   : $(date)'
echo '----------------------------------------'
echo 'Config:'
echo '  Model Path : {shlex.quote(model_path)}'
echo '  Dataset    : {dataset_name}'
echo '  CIFAR-C Dir: {shlex.quote(cifar_c_dir)}'
echo '  Batch Size : {batch_size}'
echo '  Device     : {device}'
echo '  Target Dim : {target_dim}'
echo '  Num Layers : {n_random_layers}'
echo '  Seed       : {seed}'
if [ "{run_normalization_ablation_str}" = "True" ]; then
    echo '  Normalization Ablation : Enabled'
fi
if [ "{run_acc_cali_str}" = "True" ]; then
    echo '  Accuracy Calibration Baseline : Enabled'
fi
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
if [ "{device}" = "cuda" ]; then
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
    training_methods = config.get('training_methods', ['baseline'])
    seeds = config.get('seeds', [11])
    
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


def validate_experiment_model(
    model_name: str,
    dataset_name: str,
    training_method: str,
    seed: int,
    results_base_dir: str,
) -> Tuple[bool, str]:
    """
    Check if model checkpoint exists.
    
    Args:
        model_name, dataset_name, training_method, seed: Experiment config
        results_base_dir: Base directory with pre-trained models
        
    Returns:
        (is_valid, reason): True if model exists, False with reason string otherwise
    """
    model_path, found = find_model_path(
        model_name=model_name,
        dataset_name=dataset_name,
        training_method=training_method,
        seed=seed,
        results_base_dir=results_base_dir
    )
    
    if not found:
        return False, f"Model checkpoint not found"
    
    return True, f"Model found: {model_path}"


def check_results_complete(result_file: str) -> Tuple[bool, str]:
    """
    Check if a completed results file contains all required sections.
    
    Args:
        result_file: Path to the corruption ablation results JSON file
        
    Returns:
        (is_complete, reason): True if results are complete, False with reason otherwise
    """
    if not os.path.exists(result_file):
        return False, "File does not exist"
    
    try:
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Check for required sections
        required_sections = ['config']
        
        # Check if at least one evaluation section exists
        has_evaluation = False
        evaluation_sections = ['standard_baselines', 'geometric_methods', 'coordinate_methods', 'normalization_comparison', 'accuracy_calibration']
        
        for section in evaluation_sections:
            if section in results:
                section_data = results[section]
                if isinstance(section_data, dict) and len(section_data) > 0:
                    # Special handling for normalization_comparison
                    if section == 'normalization_comparison':
                        # Check if comparison section exists with valid results
                        if 'comparison' in section_data:
                            comp = section_data['comparison']
                            # Check for corruption mode (mCE) or clean test mode (clean ECE)
                            if isinstance(comp, dict):
                                if (comp.get('rank_mce') is not None and comp.get('percentile_mce') is not None) or \
                                   (comp.get('rank_clean_ece') is not None and comp.get('percentile_clean_ece') is not None):
                                    has_evaluation = True
                                    break
                        # Also check if rank_based or percentile sections have results
                        if 'rank_based' in section_data:
                            rank_data = section_data['rank_based']
                            if isinstance(rank_data, dict) and (rank_data.get('mce') is not None or rank_data.get('clean_ece') is not None):
                                has_evaluation = True
                                break
                        if 'percentile' in section_data:
                            perc_data = section_data['percentile']
                            if isinstance(perc_data, dict) and (perc_data.get('mce') is not None or perc_data.get('clean_ece') is not None):
                                has_evaluation = True
                                break
                    elif section == 'accuracy_calibration':
                        # Check if accuracy_calibration has valid results (mce or clean_ece)
                        if isinstance(section_data, dict):
                            if section_data.get('mce') is not None or section_data.get('clean_ece') is not None:
                                has_evaluation = True
                                break
                    else:
                        # Check if at least one method has results
                        for method, data in section_data.items():
                            if isinstance(data, dict) and 'mce' in data:
                                if data['mce'] is not None:
                                    has_evaluation = True
                                    break
                    if has_evaluation:
                        break
        
        if not has_evaluation:
            return False, "No evaluation results found"
        
        return True, "Results are complete"
        
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
        output_base_dir: Base output directory for results
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
        # Check if results file already exists
        # Output filename pattern: corruption_ablation_{model_name}.json
        # But we need to extract model_name from model_path
        model_path, found = find_model_path(
            model_name=exp['model_name'],
            dataset_name=exp['dataset_name'],
            training_method=exp['training_method'],
            seed=exp['seed'],
            results_base_dir=results_base_dir
        )
        
        if not found:
            logger.warning(
                f"⚠️  Skipping invalid model: "
                f"{exp['model_name']}/{exp['dataset_name']}/{exp['training_method']}/seed{exp['seed']} - Model not found"
            )
            skipped_invalid += 1
            continue
        
        # Extract model name from path for output filename
        model_name_from_path = Path(model_path).parent.name
        result_file = os.path.join(
            output_base_dir,
            f"corruption_ablation_{model_name_from_path}.json"
        )
        
        if skip_completed and os.path.exists(result_file):
            # Check if the results file is complete
            is_complete, reason = check_results_complete(result_file)
            
            if is_complete:
                logger.info(
                    f"⏭️  Skipping completed: "
                    f"{exp['model_name']}/{exp['dataset_name']}/{exp['training_method']}/seed{exp['seed']}"
                )
                skipped_completed += 1
                continue
            else:
                logger.warning(
                    f"🔄 Re-running incomplete experiment: "
                    f"{exp['model_name']}/{exp['dataset_name']}/{exp['training_method']}/seed{exp['seed']} - {reason}"
                )
                skipped_incomplete += 1
                # Don't skip - allow it to be re-run
        
        # Validate model exists
        is_valid, reason = validate_experiment_model(
            model_name=exp['model_name'],
            dataset_name=exp['dataset_name'],
            training_method=exp['training_method'],
            seed=exp['seed'],
            results_base_dir=results_base_dir
        )
        
        if not is_valid:
            logger.warning(
                f"⚠️  Skipping invalid model: "
                f"{exp['model_name']}/{exp['dataset_name']}/{exp['training_method']}/seed{exp['seed']} - {reason}"
            )
            skipped_invalid += 1
            continue
        
        logger.debug(
            f"✅ Valid: {exp['model_name']}/{exp['dataset_name']}/{exp['training_method']}/seed{exp['seed']} - {reason}"
        )
        filtered.append(exp)
    
    logger.info(
        f"📊 Skipped "
        f"{skipped_completed} completed experiments, "
        f"{skipped_incomplete} incomplete (will re-run), "
        f"{skipped_invalid} invalid models, running {len(filtered)} experiments"
    )
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Parallel runner for corruption ablation experiments",
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
                       help="Training methods (e.g., baseline augmix)")
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                       help="Random seeds (e.g., 11 12 13)")
    
    # Paths
    parser.add_argument('--results-base-dir', type=str, default=None,
                       help="Base directory with pre-trained models")
    parser.add_argument('--output-base-dir', type=str, default=None,
                       help="Output directory for results")
    parser.add_argument('--cifar-c-dir', type=str, default=None,
                       help="Path to CIFAR-C data directory")
    parser.add_argument('--slurm-output-dir', type=str, default='slurm_jobs',
                       help="Directory for SLURM scripts and logs")
    
    # Technical settings
    parser.add_argument('--batch-size', type=int, default=256,
                       help="Batch size for data processing")
    parser.add_argument('--device', type=str, default='cuda',
                       help="Device for computations")
    parser.add_argument('--target-dim', type=int, default=256,
                       help="Target dimension for features")
    parser.add_argument('--n-random-layers', type=int, default=6,
                       help="Number of random layers for SGC")
    
    # Skip options
    parser.add_argument('--skip-baselines', action='store_true',
                       help="Skip standard baseline methods")
    parser.add_argument('--skip-geometric', action='store_true',
                       help="Skip geometric methods")
    parser.add_argument('--skip-coordinate', action='store_true',
                       help="Skip coordinate methods")
    parser.add_argument('--run-normalization-ablation', action='store_true',
                       help="Run normalization method comparison (percentile vs rank-based)")
    parser.add_argument('--run-acc-cali', action='store_true',
                       help="Run accuracy calibration baseline (uses validation accuracy as constant confidence)")
    
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
    parser.add_argument('--gpu-type', type=str, default='rtx_4090',
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
    if args.training_methods:
        config['training_methods'] = args.training_methods
    if args.seeds:
        config['seeds'] = args.seeds
    if args.results_base_dir:
        config['results_base_dir'] = args.results_base_dir
    if args.output_base_dir:
        config['output_base_dir'] = args.output_base_dir
    if args.cifar_c_dir:
        config['cifar_c_dir'] = args.cifar_c_dir
    
    # Validate required parameters
    required = ['models', 'datasets', 'training_methods', 'seeds', 
                'results_base_dir', 'output_base_dir', 'cifar_c_dir']
    
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
    
    # Create SLURM scripts and submit jobs
    # Organize experiments: first by seed, then by model, then by training_method, then by dataset
    logger.info(f"🚀 Organizing {len(experiments)} experiments for submission...")
    
    # Group experiments by seed -> model -> training_method -> dataset
    experiments_by_seed = {}
    for exp in experiments:
        seed = exp['seed']
        if seed not in experiments_by_seed:
            experiments_by_seed[seed] = {}
        
        model = exp['model_name']
        if model not in experiments_by_seed[seed]:
            experiments_by_seed[seed][model] = {}
        
        training_method = exp['training_method']
        if training_method not in experiments_by_seed[seed][model]:
            experiments_by_seed[seed][model][training_method] = []
        
        experiments_by_seed[seed][model][training_method].append(exp)
    
    # Count total jobs for progress tracking
    total_jobs = len(experiments)
    
    logger.info(f"🚀 Submitting {total_jobs} jobs in order: seed -> model -> training_method -> dataset")
    
    submitted_jobs = []
    failed_jobs = []
    job_counter = 0
    
    # Submit jobs grouped by seed (all models for a seed submitted together for parallel execution)
    for seed in sorted(experiments_by_seed.keys()):
        logger.info(f"\n🌱 Processing seed {seed}...")
        seed_jobs = []
        
        # Collect all jobs for this seed
        for model in sorted(experiments_by_seed[seed].keys()):
            for training_method in sorted(experiments_by_seed[seed][model].keys()):
                # Sort experiments by dataset
                sorted_exps = sorted(experiments_by_seed[seed][model][training_method], 
                                    key=lambda x: x['dataset_name'])
                
                for exp in sorted_exps:
                    # Find model path
                    model_path, found = find_model_path(
                        model_name=exp['model_name'],
                        dataset_name=exp['dataset_name'],
                        training_method=exp['training_method'],
                        seed=exp['seed'],
                        results_base_dir=config['results_base_dir']
                    )
                    
                    if not found:
                        logger.error(f"❌ Model not found for {exp['model_name']}/{exp['dataset_name']}/{exp['training_method']}/seed{exp['seed']}")
                        failed_jobs.append((f"corruption_ablation_{exp['model_name']}_{exp['dataset_name']}_{exp['training_method']}_s{exp['seed']}", "Model not found"))
                        continue
                    
                    # Generate job name
                    job_name = f"corr_ablation_{exp['model_name']}_{exp['dataset_name']}_{exp['training_method']}_s{exp['seed']}"
                    
                    try:
                        # Generate sbatch command
                        sbatch_cmd = generate_sbatch_command(
                            job_name=job_name,
                            output_dir=args.slurm_output_dir,
                            model_path=model_path,
                            model_name=exp['model_name'],
                            dataset_name=exp['dataset_name'],
                            cifar_c_dir=config['cifar_c_dir'],
                            results_base_dir=config['results_base_dir'],
                            output_base_dir=config['output_base_dir'],
                            batch_size=args.batch_size,
                            device=args.device,
                            target_dim=args.target_dim,
                            n_random_layers=args.n_random_layers,
                            seed=exp['seed'],
                            skip_baselines=args.skip_baselines,
                            skip_geometric=args.skip_geometric,
                            skip_coordinate=args.skip_coordinate,
                            run_normalization_ablation=args.run_normalization_ablation,
                            run_acc_cali=args.run_acc_cali,
                            slurm_config=slurm_config,
                        )
                        seed_jobs.append((job_name, sbatch_cmd, exp))
                    except Exception as e:
                        logger.error(f"❌ Failed to prepare {job_name}: {e}")
                        failed_jobs.append((job_name, str(e)))
        
        # Submit all jobs for this seed
        models_in_seed = sorted(set(j[2]['model_name'] for j in seed_jobs))
        logger.info(f"   Submitting {len(seed_jobs)} jobs for seed {seed} (models: {', '.join(models_in_seed)})")
        for job_name, sbatch_cmd, exp in seed_jobs:
            job_counter += 1
            logger.info(f"   [{job_counter}/{total_jobs}] Submitting {job_name}...")
            
            try:
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
        logger.info(f"📁 SLURM logs in: {args.slurm_output_dir}/_corruption_ablation_logs/")
    
    logger.info("="*80)
    
    return 0 if not failed_jobs else 1


if __name__ == '__main__':
    exit(main())

