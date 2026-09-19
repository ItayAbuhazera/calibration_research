#!/usr/bin/env python3
"""
SLURM Job Submission for Global Coordinate Ablation Experiments

Submits jobs for run_coordinate_ablation.py across multiple models, datasets,
training methods, and seeds.
"""

import argparse
import glob
import itertools
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import model path construction from existing module
from Experiments.run_post_hoc_calibration import construct_model_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
from utils.logging_config import get_logger
logger = get_logger(__name__)


def check_model_exists(results_base_dir: str, dataset: str, model: str, method: str, seed: int) -> bool:
    """Check if the trained model exists."""
    # Try standard path structure first
    model_path = construct_model_path(results_base_dir, method, dataset, model, seed)
    if os.path.exists(model_path):
        return True
    
    # Special handling for DINOv2 models
    is_dinov2 = "dino" in model.lower()
    if is_dinov2:
        model_name_pattern = f"{method}_{dataset}_{model}_seed{seed}*"
        calibration_pattern = os.path.join(
            results_base_dir, "calibration_comparison", 
            model_name_pattern, "best_model.pth"
        )
        matching_files = glob.glob(calibration_pattern)
        if matching_files:
            return True
    
    # Fallback: check common model file patterns in various locations
    patterns = [
        os.path.join(results_base_dir, "baseline", method, dataset, model, f"seed{seed}", "best_model.pth"),
        os.path.join(results_base_dir, method, dataset, model, f"seed{seed}", "best_model.pth"),
        os.path.join(results_base_dir, dataset, model, method, f"seed{seed}", "best_model.pth"),
    ]
    for pattern in patterns:
        if os.path.exists(pattern):
            return True
    
    return False


def check_results_exist(output_base_dir: str, method: str, dataset: str, model: str, seed: int, nested: bool = False) -> bool:
    """Check if coordinate ablation results already exist and are complete."""
    strategy_suffix = "_nested" if nested else "_independent"
    output_filename = f"coordinate_ablation_{method}_{dataset}_{model}_seed{seed}{strategy_suffix}.json"
    output_path = os.path.join(output_base_dir, output_filename)
    
    if not os.path.exists(output_path):
        return False
    
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        # Check if we have results for coordinate counts
        results = data.get("results_by_num_coordinates", {})
        if not results:
            return False
        
        # Consider complete if we have at least some results
        # (incremental saving means partial results are valid)
        # Check if we have at least one complete trial
        for k_key, trials in results.items():
            if trials and len(trials) > 0:
                # Check if at least one trial has all required fields
                for trial in trials:
                    if 'ece' in trial and 'accuracy' in trial:
                        return True
        
        return False
        
    except Exception as e:
        logger.warning(f"Failed to read {output_path}: {e}")
        return False


def generate_sbatch_command(
    job_name: str,
    model_name: str,
    dataset_name: str,
    training_method: str,
    seed: int,
    results_base_dir: str,
    output_base_dir: str,
    batch_size: int = 128,
    device: str = 'cuda',
    num_random_seeds: int = 5,
    coordinate_counts: List[int] = None,
    slurm_config: Dict[str, Any] = None,
    nested_sampling: bool = False,
    dac_mode: str = "depth",
) -> str:
    """Generate sbatch command for coordinate ablation experiment."""
    
    if slurm_config is None:
        slurm_config = {}
    if coordinate_counts is None:
        coordinate_counts = [64, 128, 256, 512, 1024, 2048, 4096]
    
    partition = slurm_config.get('partition', 'gpu')
    time_limit = slurm_config.get('time', '2-00:00:00')  # 2 days - coordinate extraction is fast
    mem = slurm_config.get('mem', '32G')
    cpus = slurm_config.get('cpus', 4)
    gpus = slurm_config.get('gpus', 1)
    gpu_type = slurm_config.get('gpu_type', 'rtx_3090')
    conda_env = slurm_config.get('conda_env', 'tamar_n_env')
    
    project_root = Path(__file__).resolve().parent.parent
    log_dir = Path(output_base_dir) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"
    
    python_cmd = [
        sys.executable,
        str(project_root / "Experiments" / "run_coordinate_ablation.py"),
        "--model-name", model_name,
        "--dataset", dataset_name,
        "--training-method", training_method,
        "--seed", str(seed),
        "--results-base-dir", results_base_dir,
        "--output-dir", output_base_dir,
        "--batch-size", str(batch_size),
        "--device", device,
        "--num-random-seeds", str(num_random_seeds),
        "--coordinate-counts",
    ] + [str(k) for k in coordinate_counts] + [
        "--dac-mode", dac_mode,
    ]
    
    if nested_sampling:
        python_cmd.append("--nested-sampling")
    
    safe_python_cmd = " ".join(shlex.quote(a) for a in python_cmd)
    
    wrap_script = f"""
echo '========================================'
echo '🔬 SLURM JOB: Coordinate Ablation'
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
echo '  Coordinates: {coordinate_counts}'
echo '  DAC Mode   : {dac_mode}'
echo '----------------------------------------'
module load anaconda || echo "Anaconda module not found, assuming env is active."
source activate {shlex.quote(conda_env)} || echo "Conda env '{conda_env}' activation failed."
export PYTHONPATH='{shlex.quote(str(project_root))}:$PYTHONPATH'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
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
exit $EXIT_CODE
"""
    clean_wrap_script = "\n".join(line.lstrip() for line in wrap_script.strip().split('\n'))
    
    sbatch_cmd = [
        "sbatch",
        f"--partition={shlex.quote(partition)}",
        f"--job-name={shlex.quote(job_name)}",
        f"--output={shlex.quote(str(log_file))}",
        f"--time={shlex.quote(time_limit)}",
        "--ntasks=1",
        f"--gpus={shlex.quote(f'{gpu_type}:{gpus}')}",
        f"--cpus-per-task={shlex.quote(str(cpus))}",
        f"--mem={shlex.quote(mem)}",
        f"--wrap={shlex.quote(clean_wrap_script)}",
    ]
    
    return " ".join(sbatch_cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Submit Coordinate Ablation Jobs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--models', nargs='+', required=True,
                        help='Model names (e.g., resnet50 densenet121)')
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Dataset names (e.g., cifar10 cifar100)')
    parser.add_argument('--training-methods', nargs='+', required=True,
                        help='Training methods (e.g., baseline augmix)')
    parser.add_argument('--seeds', nargs='+', type=int, required=True,
                        help='Seeds to run')
    parser.add_argument('--results-base-dir', type=str, required=True,
                        help='Base directory containing trained models')
    parser.add_argument('--output-base-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for data processing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--num-random-seeds', type=int, default=5,
                        help='Number of random trials per K configuration')
    parser.add_argument('--coordinate-counts', nargs='+', type=int,
                        default=[64, 128, 256, 512, 1024, 2048, 4096],
                        help='Number of coordinates (K) to test')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without submitting')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if results exist')
    
    # SLURM configuration
    parser.add_argument('--partition', type=str, default='gpu',
                        help='SLURM partition')
    parser.add_argument('--gpu-type', type=str, default='rtx_3090',
                        help='GPU type (e.g., rtx_4090, rtx_6000)')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs per job')
    parser.add_argument('--cpus', type=int, default=4,
                        help='CPUs per task')
    parser.add_argument('--mem', type=str, default='32G',
                        help='Memory per job')
    parser.add_argument('--time', type=str, default='2-00:00:00',
                        help='Time limit (d-hh:mm:ss format)')
    parser.add_argument('--conda-env', type=str, default='tamar_n_env',
                        help='Conda environment name')
    parser.add_argument('--sampling-strategy', type=str,
                        choices=['independent', 'nested', 'both'],
                        default='both',
                        help='Sampling strategy to run: independent, nested, or both')
    parser.add_argument(
        "--dac-mode", type=str, default="depth",
        choices=["single", "interleaved", "depth", "all", "skip"],
        help="DAC feature structuring mode (skip to skip DAC calibration entirely)"
    )
    
    args = parser.parse_args()
    
    # Build SLURM config
    slurm_config = {
        'partition': args.partition,
        'gpu_type': args.gpu_type,
        'gpus': args.gpus,
        'cpus': args.cpus,
        'mem': args.mem,
        'time': args.time,
        'conda_env': args.conda_env,
    }
    
    # Generate experiment grid
    experiments = list(itertools.product(
        args.models, args.datasets, args.training_methods, args.seeds
    ))
    
    logger.info(f"Found {len(experiments)} experiment configurations")
    logger.info(f"Coordinate counts to test: {args.coordinate_counts}")
    logger.info(f"Random seeds per config: {args.num_random_seeds}")
    logger.info(f"Sampling strategy: {args.sampling_strategy}")
    
    strategies_to_run = ['independent', 'nested'] if args.sampling_strategy == 'both' else [args.sampling_strategy]
    
    submitted = 0
    skipped_no_model = 0
    skipped_exists = 0
    failed = 0
    
    for model, dataset, method, seed in experiments:
        # Check if model exists
        if not check_model_exists(args.results_base_dir, dataset, model, method, seed):
            logger.warning(f"Skipping {model}_{dataset}_{method}_s{seed}: model not found")
            skipped_no_model += 1
            continue
        
        # Submit jobs for each strategy
        for strategy in strategies_to_run:
            nested = (strategy == "nested")
            
            # Check if results already exist for this strategy
            if not args.force and check_results_exist(args.output_base_dir, method, dataset, model, seed, nested=nested):
                logger.info(f"Skipping {model}_{dataset}_{method}_s{seed} ({strategy}): results exist (use --force to re-run)")
                skipped_exists += 1
                continue
            
            job_name = f"coord_{strategy[:4]}_{model}_{dataset}_{method}_s{seed}"
            
            cmd = generate_sbatch_command(
                job_name=job_name,
                model_name=model,
                dataset_name=dataset,
                training_method=method,
                seed=seed,
                results_base_dir=args.results_base_dir,
                output_base_dir=args.output_base_dir,
                batch_size=args.batch_size,
                device=args.device,
                num_random_seeds=args.num_random_seeds,
                coordinate_counts=args.coordinate_counts,
                slurm_config=slurm_config,
                nested_sampling=nested,
                dac_mode=args.dac_mode,
            )
            
            if args.dry_run:
                logger.info(f"[DRY RUN] Would submit: {job_name}")
                logger.debug(f"  Command: {cmd[:200]}...")
            else:
                try:
                    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                    job_id = result.stdout.strip().split()[-1] if result.stdout.strip() else "UNKNOWN"
                    logger.info(f"✅ Submitted job {job_id}: {job_name}")
                    submitted += 1
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ Failed to submit {job_name}: {e.stderr if e.stderr else e}")
                    failed += 1
                
                time.sleep(0.3)  # Small delay between submissions
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUBMISSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total configurations: {len(experiments)}")
    logger.info(f"  ✅ Submitted: {submitted}")
    logger.info(f"  ⏭️  Skipped (no model): {skipped_no_model}")
    logger.info(f"  ⏭️  Skipped (results exist): {skipped_exists}")
    logger.info(f"  ❌ Failed: {failed}")
    logger.info("=" * 60)
    
    if not args.dry_run and submitted > 0:
        logger.info(f"\n📊 Monitor jobs with: squeue -u $USER")
        logger.info(f"📁 SLURM logs in: {args.output_base_dir}/_logs/")


if __name__ == '__main__':
    main()

