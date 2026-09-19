#!/usr/bin/env python3
"""
run_tulip_jobs.py

Parallel runner for TULIP experiments (original only).
Submits jobs for different (model, dataset, training_method, seed) combinations.

NOTE: TULIP has high memory requirements. Only ResNet18 and ResNet50 are practical.
DenseNet and larger models will be automatically filtered out due to OOM risk.

Usage:
    python Experiments/run_tulip_jobs.py \
        --models resnet18 resnet50 \
        --datasets cifar10 cifar100 \
        --training-methods baseline augmix \
        --seeds 11 \
        --results-base-dir aaai_full_experiments/results \
        --output-base-dir calibration_comparison_results/tulip
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import shlex
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from utils.logging_config import get_logger
logger = get_logger(__name__)


def check_tulip_results_exist(output_base_dir: str, training_method: str, dataset_name: str, 
                               model_name: str, seed: int) -> Dict[str, Any]:
    """
    Check if TULIP results already exist for this config.
    
    Returns:
        dict with 'complete', 'has_original', 'path'
    """
    output_filename = f"tulip_{training_method}_{dataset_name}_{model_name}_seed{seed}.json"
    output_path = os.path.join(output_base_dir, output_filename)
    
    status = {
        'complete': False,
        'has_original': False,
        'path': output_path
    }
    
    if not os.path.exists(output_path):
        return status
    
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        # Check if tulip_results section exists
        if "tulip_results" not in data:
            return status
        
        tulip_data = data["tulip_results"]
        
        # Check original TULIP
        if tulip_data.get("original_tulip") is not None:
            status['has_original'] = True
            status['complete'] = True
        
        return status
        
    except Exception as e:
        logger.warning(f"Failed to read {output_path}: {e}")
        return status


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
    slurm_config: Dict[str, Any] = None
) -> str:
    """Generate SBATCH command for TULIP experiment."""
    
    if slurm_config is None:
        slurm_config = {}
    
    partition = slurm_config.get('partition', 'gpu')
    time_limit = slurm_config.get('time', '6-00:00:00')  # TULIP may take longer
    mem = slurm_config.get('mem', '128G')  # TULIP needs more memory
    cpus = slurm_config.get('cpus', 4)
    gpus = slurm_config.get('gpus', 1)
    gpu_type = slurm_config.get('gpu_type', 'rtx_6000')
    conda_env = slurm_config.get('conda_env', 'tamar_n_env')
    
    project_root = Path(__file__).resolve().parent.parent
    log_dir = Path(output_base_dir) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"
    
    python_cmd = [
        sys.executable,
        str(project_root / "Experiments" / "run_tulip_comparison.py"),
        "--model-name", model_name,
        "--dataset", dataset_name,
        "--training-method", training_method,
        "--seed", str(seed),
        "--results-base-dir", results_base_dir,
        "--output-dir", output_base_dir,
        "--batch-size", str(batch_size),
        "--device", device,
    ]
    
    safe_python_cmd = " ".join(shlex.quote(a) for a in python_cmd)
    
    wrap_script = f"""
echo '========================================'
echo '🌷 SLURM JOB: TULIP Comparison'
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
    parser = argparse.ArgumentParser(description="Parallel runner for TULIP experiments")
    parser.add_argument('--models', nargs='+', required=True,
                        help='Model names (e.g., resnet18 densenet121)')
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Dataset names (e.g., cifar10 cifar100)')
    parser.add_argument('--training-methods', nargs='+', required=True,
                        help='Training methods (e.g., baseline augmix)')
    parser.add_argument('--seeds', nargs='+', type=int, required=True,
                        help='Random seeds')
    parser.add_argument('--results-base-dir', type=str, required=True,
                        help='Base directory containing trained models')
    parser.add_argument('--output-base-dir', type=str, required=True,
                        help='Output directory for TULIP results')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for experiments')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without submitting jobs')
    parser.add_argument('--force', action='store_true',
                        help='Rerun experiments even if results exist')
    
    # SLURM configuration
    parser.add_argument('--partition', type=str, default='gpu',
                        help='SLURM partition')
    parser.add_argument('--time', type=str, default='6-00:00:00',
                        help='Time limit (TULIP may take longer)')
    parser.add_argument('--mem', type=str, default='128G',
                        help='Memory per job (TULIP needs more memory)')
    parser.add_argument('--cpus', type=int, default=4,
                        help='CPUs per task')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs')
    parser.add_argument('--gpu-type', type=str, default='rtx_6000',
                        help='GPU type')
    parser.add_argument('--conda-env', type=str, default='tamar_n_env',
                        help='Conda environment name')
    
    args = parser.parse_args()
    
    # Filter models - only allow smaller models due to memory requirements
    supported_models = ['resnet18', 'resnet50']
    filtered_models = [m for m in args.models if any(sm in m.lower() for sm in supported_models)]
    
    if filtered_models != args.models:
        skipped_models = [m for m in args.models if m not in filtered_models]
        logger.warning(f"⚠️  Skipping models due to memory requirements: {skipped_models}")
        logger.warning(f"   TULIP is only practical for ResNet18/ResNet50. DenseNet and larger models will OOM.")
        if not filtered_models:
            logger.error("No supported models found. Exiting.")
            return
    
    # Generate experiment grid
    experiments = list(itertools.product(
        filtered_models, args.datasets, args.training_methods, args.seeds
    ))
    
    logger.info(f"Found {len(experiments)} experiments (filtered to supported models: {filtered_models})")
    
    # Collect SLURM config
    slurm_config = {
        'partition': args.partition,
        'time': args.time,
        'mem': args.mem,
        'cpus': args.cpus,
        'gpus': args.gpus,
        'gpu_type': args.gpu_type,
        'conda_env': args.conda_env,
    }
    
    submitted_jobs = []
    skipped_jobs = []
    partial_jobs = []
    
    for model, dataset, method, seed in experiments:
        # Check if model exists
        is_dinov2 = "dino" in model.lower()
        
        if is_dinov2:
            # DINOv2 models with wildcard suffix
            model_name_pattern = f"{method}_{dataset}_{model}_seed{seed}*"
            calibration_pattern = os.path.join(
                args.results_base_dir, "calibration_comparison", 
                model_name_pattern, "best_model.pth"
            )
            matching_files = glob.glob(calibration_pattern)
            
            if not matching_files:
                logger.warning(f"⚠️  Skipping {model}_{dataset}_{method}_s{seed}: DINOv2 model not found")
                continue
        else:
            # Standard path structure
            model_path = os.path.join(
                args.results_base_dir, "baseline", method, dataset, model, 
                f"seed{seed}", f"{method}_{dataset}_{model}_seed{seed}", "results.json"
            )
            
            if not os.path.exists(model_path):
                logger.warning(f"⚠️  Skipping {model}_{dataset}_{method}_s{seed}: model not found at {model_path}")
                continue
        
        # Check if TULIP results already exist
        status = check_tulip_results_exist(
            args.output_base_dir, method, dataset, model, seed
        )
        
        if status['complete'] and not args.force:
            logger.info(f"✓ Skipping {model}_{dataset}_{method}_s{seed}: TULIP results complete")
            skipped_jobs.append(f"{model}_{dataset}_{method}_s{seed}")
            continue
        
        # Log status if partial results exist
        if status['has_original']:
            logger.info(f"📝 Partial results for {model}_{dataset}_{method}_s{seed}: original_tulip exists")
            partial_jobs.append(f"{model}_{dataset}_{method}_s{seed}")
        else:
            logger.info(f"🆕 No existing results for {model}_{dataset}_{method}_s{seed}")
        
        # Generate job name
        job_name = f"tulip_{model}_{dataset}_{method}_s{seed}"
        
        # Generate SBATCH command
        cmd = generate_sbatch_command(
            job_name, model, dataset, method, seed,
            args.results_base_dir, args.output_base_dir,
            batch_size=args.batch_size,
            slurm_config=slurm_config
        )
        
        if args.dry_run:
            logger.info(f"[DRY RUN] Would submit: {job_name}")
            logger.debug(f"Command: {cmd}")
        else:
            try:
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                # Parse job ID from output like "Submitted batch job 12345"
                job_id = result.stdout.strip().split()[-1] if result.stdout.strip() else "UNKNOWN"
                logger.info(f"✅ Submitted job {job_id}: {job_name}")
                submitted_jobs.append(job_id)
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to submit {job_name}: {e.stderr if e.stderr else e}")
                continue
            
            time.sleep(0.3)  # Small delay between submissions
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUBMISSION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total experiments: {len(experiments)}")
    logger.info(f"Submitted jobs: {len(submitted_jobs)}")
    logger.info(f"Partial results (will extend): {len(partial_jobs)}")
    logger.info(f"Skipped (complete): {len(skipped_jobs)}")
    
    if args.dry_run:
        logger.info("\n[DRY RUN MODE - No jobs were actually submitted]")
    
    if submitted_jobs and not args.dry_run:
        logger.info(f"\nSubmitted job IDs: {', '.join(submitted_jobs)}")
        logger.info(f"\nMonitor jobs with: squeue -u $USER | grep tulip")
        logger.info(f"Cancel all jobs with: scancel {' '.join(submitted_jobs)}")


if __name__ == '__main__':
    main()

