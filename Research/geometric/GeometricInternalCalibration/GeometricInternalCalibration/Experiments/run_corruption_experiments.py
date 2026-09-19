#!/usr/bin/env python3
"""
Runner for corruption robustness experiments.

Tests calibration methods on CIFAR-10-C and CIFAR-100-C using the existing
compare_dac_geometric.py pipeline.
"""

import argparse
import itertools
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from utils.logging_config import get_logger
logger = get_logger(__name__)

# Corruption types from CIFAR-C benchmark
CORRUPTION_TYPES = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]
SEVERITY_LEVELS = [1, 2, 3, 4, 5]


def generate_corruption_experiments(
    models: List[str],
    datasets: List[str],
    training_methods: List[str],
    seeds: List[int],
    corruption_types: List[str],
    severity_levels: List[int]
) -> List[Dict[str, Any]]:
    """Generate experiment grid for corruption experiments."""
    experiments: List[Dict[str, Any]] = []
    for model, dataset, method, seed, corruption, severity in itertools.product(
        models, datasets, training_methods, seeds, corruption_types, severity_levels
    ):
        if dataset not in ['cifar10', 'cifar100']:
            continue
        experiments.append({
            'model_name': model,
            'dataset_name': dataset,
            'training_method': method,
            'seed': seed,
            'corruption_type': corruption,
            'corruption_severity': severity,
        })
    return experiments


def result_filename(base_dir: str, exp: Dict[str, Any]) -> str:
    """Construct output filename matching compare_dac_geometric.py."""
    return os.path.join(
        base_dir,
        f"ablation_{exp['training_method']}_{exp['dataset_name']}_{exp['model_name']}_seed{exp['seed']}_{exp['corruption_type']}_sev{exp['corruption_severity']}.json"
    )


def generate_sbatch_command_corruption(
    job_name: str,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    training_method: str,
    seed: int,
    corruption_type: str,
    corruption_severity: int,
    results_base_dir: str,
    output_base_dir: str,
    batch_size: int = 128,
    device: str = 'cuda',
    global_random_layers: int = 6,
    global_random_compression: float = 16.0,
    slurm_config: Dict[str, Any] = None
) -> str:
    """
    Generate sbatch command for corruption experiment.
    Mirrors generate_sbatch_command from run_dac_comparison_experiments.py
    with corruption parameters added.
    """
    if slurm_config is None:
        slurm_config = {}

    partition = slurm_config.get('partition', 'gpu')
    time_limit = slurm_config.get('time', '0-08:00:00')
    mem = slurm_config.get('mem', '32G')
    cpus = slurm_config.get('cpus', 4)
    gpus = slurm_config.get('gpus', 1)
    gpu_type = slurm_config.get('gpu_type', 'rtx_4090')
    conda_env = slurm_config.get('conda_env', 'tamar_n_env')

    project_root = Path(__file__).resolve().parent.parent
    log_dir = Path(output_dir) / "_corruption_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"

    python_cmd = [
        sys.executable,
        str(project_root / "Experiments" / "compare_dac_geometric.py"),
        "--model-name", model_name,
        "--dataset", dataset_name,
        "--training-method", training_method,
        "--seed", str(seed),
        "--results-base-dir", results_base_dir,
        "--output-dir", output_base_dir,
        "--batch-size", str(batch_size),
        "--device", device,
        "--corruption-type", corruption_type,
        "--corruption-severity", str(corruption_severity),
        "--global-random-layers", str(global_random_layers),
        "--global-random-compression", str(global_random_compression),
    ]

    safe_python_cmd = " ".join(shlex.quote(a) for a in python_cmd)

    wrap_script = f"""
echo '========================================'
echo '🔬 SLURM JOB: Corruption Calibration'
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
echo '  Corruption : {corruption_type}'
echo '  Severity   : {corruption_severity}'
echo '  Batch Size : {batch_size}'
echo '  Device     : {device}'
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
    parser = argparse.ArgumentParser(description='Run corruption robustness experiments')

    # Model/dataset config
    parser.add_argument('--models', nargs='+', default=['resnet18'],
                       help='Models to test')
    parser.add_argument('--datasets', nargs='+', default=['cifar10', 'cifar100'],
                       help='Datasets (corruption versions will be used)')
    parser.add_argument('--training-methods', nargs='+', default=['augmix', 'baseline_brier'],
                       help='Training methods')
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 12, 13],
                       help='Random seeds')

    # Corruption config
    parser.add_argument('--corruption-types', nargs='+', default=CORRUPTION_TYPES,
                       help='Corruption types to test')
    parser.add_argument('--severity-levels', nargs='+', type=int, default=SEVERITY_LEVELS,
                       help='Severity levels to test')

    # Paths
    parser.add_argument('--results-base-dir', type=str, required=True)
    parser.add_argument('--output-base-dir', type=str, required=True)

    # SLURM config
    parser.add_argument('--partition', type=str, default='gpu')
    parser.add_argument('--time', type=str, default='0-08:00:00')
    parser.add_argument('--mem', type=str, default='32G')
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--gpu-type', type=str, default='rtx_4090')
    parser.add_argument('--conda-env', type=str, default='tamar_n_env')

    # Options
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-completed', action='store_true', default=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    # Global Random config
    parser.add_argument('--global-random-layers', type=int, default=6,
                       help='Number of random layers for Global Random calibration')
    parser.add_argument('--global-random-compression', type=float, default=4.0,
                       help='Compression ratio for Global Random calibration')

    args = parser.parse_args()

    experiments = generate_corruption_experiments(
        models=args.models,
        datasets=args.datasets,
        training_methods=args.training_methods,
        seeds=args.seeds,
        corruption_types=args.corruption_types,
        severity_levels=args.severity_levels
    )

    logger.info(f"Generated {len(experiments)} corruption experiments")

    # Filter completed
    filtered_experiments: List[Dict[str, Any]] = []
    for exp in experiments:
        outfile = result_filename(args.output_base_dir, exp)
        if args.skip_completed and os.path.exists(outfile):
            try:
                with open(outfile, 'r') as f:
                    data = json.load(f)
                if data:
                    # Drop experiments with very low accuracy (random guessing)
                    acc = data.get('uncalibrated', {}).get('accuracy', None)
                    if acc is not None and acc <= 13.0:
                        logger.info(f"⏭️  Skipping low-accuracy result (<=13%): {outfile} (acc={acc})")
                        continue
                    logger.info(f"⏭️  Skipping completed: {outfile}")
                    continue
            except Exception:
                logger.warning(f"Could not parse existing file, will re-run: {outfile}")
        filtered_experiments.append(exp)

    if not filtered_experiments:
        logger.info("✅ All corruption experiments completed!")
        return 0

    slurm_config = {
        'partition': args.partition,
        'time': args.time,
        'mem': args.mem,
        'cpus': args.cpus,
        'gpus': args.gpus,
        'gpu_type': args.gpu_type,
        'conda_env': args.conda_env,
    }

    logger.info(f"🚀 Submitting {len(filtered_experiments)} jobs...")
    submitted, failed = [], []

    for i, exp in enumerate(filtered_experiments, 1):
        job_name = f"corr_{exp['model_name']}_{exp['dataset_name']}_{exp['training_method']}_s{exp['seed']}_{exp['corruption_type']}_sev{exp['corruption_severity']}"
        try:
            sbatch_cmd = generate_sbatch_command_corruption(
                job_name=job_name,
                output_dir=args.output_base_dir,
                model_name=exp['model_name'],
                dataset_name=exp['dataset_name'],
                training_method=exp['training_method'],
                seed=exp['seed'],
                corruption_type=exp['corruption_type'],
                corruption_severity=exp['corruption_severity'],
                results_base_dir=args.results_base_dir,
                output_base_dir=args.output_base_dir,
                batch_size=args.batch_size,
                device=args.device,
                global_random_layers=args.global_random_layers,
                global_random_compression=args.global_random_compression,
                slurm_config=slurm_config,
            )

            if args.dry_run:
                logger.info(f"[DRY RUN] {i}/{len(filtered_experiments)} would submit: {sbatch_cmd}")
                submitted.append((job_name, "DRY_RUN"))
            else:
                result = subprocess.run(sbatch_cmd, shell=True, check=True, capture_output=True, text=True)
                job_id = result.stdout.strip().split()[-1] if result.stdout.strip() else "UNKNOWN"
                logger.info(f"✅ Submitted job {job_id}: {job_name}")
                submitted.append((job_name, job_id))
                time.sleep(0.3)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to submit {job_name}: {e.stderr if e.stderr else e}")
            failed.append((job_name, str(e)))
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"❌ Failed to submit {job_name}: {e}")
            failed.append((job_name, str(e)))

    logger.info("\n" + "="*80)
    logger.info("SUBMISSION SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Submitted: {len(submitted)}")
    logger.info(f"❌ Failed: {len(failed)}")
    if submitted:
        logger.info("Sample submitted jobs:")
        for name, jid in submitted[:10]:
            logger.info(f"  {jid}: {name}")
    if failed:
        logger.info("Failed jobs:")
        for name, err in failed:
            logger.info(f"  {name}: {err}")
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - no jobs were submitted.")
    else:
        logger.info("📊 Monitor jobs with: squeue -u $USER")
        logger.info(f"📁 SLURM logs in: {args.output_base_dir}/_corruption_logs")

    return 0 if not failed else 1


if __name__ == '__main__':
    exit(main())

