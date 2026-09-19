#!/usr/bin/env python3
"""
run_random_ablation_jobs.py

Parallel runner for Random Layer Ablation experiments and Pixel-space baselines.
Submits jobs for different (model, dataset, training_method, seed) combinations.

NEW: Use --skip-single-layers to focus on multi-layer comparisons only.
This skips oracle experiments (geo_best, tulip_best) but runs all multi-layer
experiments (geo_comb, tulip_comb, random ablation).

Usage:
    python Experiments/run_random_ablation_jobs.py \
        --models resnet18 densenet121 \
        --datasets cifar10 cifar100 \
        --training-methods baseline augmix \
        --seeds 11 \
        --results-base-dir aaai_full_experiments/results \
        --output-base-dir calibration_comparison_results/random_ablation
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

# Standard CIFAR-10-C / CIFAR-100-C corruption types
CIFAR_C_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

def check_results_exist(output_base_dir: str, training_method: str, dataset_name: str, 
                        model_name: str, seed: int) -> bool:
    """Check if random_layer_ablation results already exist for this config."""
    output_filename = f"ablation_{training_method}_{dataset_name}_{model_name}_seed{seed}.json"
    output_path = os.path.join(output_base_dir, output_filename)
    
    if not os.path.exists(output_path):
        return False
    
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        # Check if random_layer_ablation section exists
        if "random_layer_ablation" not in data:
            return False
        
        # Check if first_k_layers and last_k_layers exist in all target dimensions
        rla_data = data["random_layer_ablation"]
        if "results_by_target_dim" not in rla_data:
            return False
        
        results_by_td = rla_data["results_by_target_dim"]
        
        # If no target dimensions exist, need to run
        if not results_by_td:
            logger.info("No target dimensions found in results, need to run")
            return False
        
        # Check if all target dimensions have both first_k_layers and last_k_layers
        for target_dim, target_data in results_by_td.items():
            if "first_k_layers" not in target_data or "last_k_layers" not in target_data:
                logger.info(f"Missing first_k_layers or last_k_layers for target_dim={target_dim}")
                return False
            
            # Check if the lists are non-empty
            if not target_data["first_k_layers"] or not target_data["last_k_layers"]:
                logger.info(f"Empty first_k_layers or last_k_layers for target_dim={target_dim}")
                return False
        

        # All checks passed - both baselines exist
        return True
    except Exception as e:
        logger.warning(f"Failed to read {output_path}: {e}")
        return False


def check_corruption_results_exist(
    output_base_dir: str,
    training_method: str,
    dataset_name: str,
    model_name: str,
    seed: int,
    corruption_type: str,
    corruption_severity: int,
) -> bool:
    """
    Check if random_layer_ablation corruption results already exist for this config.
    Mirrors the logic of check_results_exist but for CIFAR-C corruption files.
    """
    output_filename = (
        f"ablation_{training_method}_{dataset_name}_{model_name}_seed{seed}_"
        f"{corruption_type}_sev{corruption_severity}.json"
    )
    output_path = os.path.join(output_base_dir, output_filename)
    
    if not os.path.exists(output_path):
        return False
    
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        # Check if random_layer_ablation section exists
        if "random_layer_ablation" not in data:
            return False
        
        rla_data = data["random_layer_ablation"]
        if "results_by_target_dim" not in rla_data:
            return False
        
        results_by_td = rla_data["results_by_target_dim"]
        
        # If no target dimensions exist, need to run
        if not results_by_td:
            logger.info("No target dimensions found in corruption results, need to run")
            return False
        
        # Check if all target dimensions have both first_k_layers and last_k_layers
        for target_dim, target_data in results_by_td.items():
            if "first_k_layers" not in target_data or "last_k_layers" not in target_data:
                logger.info(
                    f"[corruption] Missing first_k_layers or last_k_layers for "
                    f"target_dim={target_dim}, corruption={corruption_type}, "
                    f"severity={corruption_severity}"
                )
                return False
            
            # Check if the lists are non-empty
            if not target_data["first_k_layers"] or not target_data["last_k_layers"]:
                logger.info(
                    f"[corruption] Empty first_k_layers or last_k_layers for "
                    f"target_dim={target_dim}, corruption={corruption_type}, "
                    f"severity={corruption_severity}"
                )
                return False
        
        # All checks passed - both baselines exist
        return True
    except Exception as e:
        logger.warning(f"Failed to read corruption results {output_path}: {e}")
        return False

def check_ood_results_exist(output_base_dir: str, training_method: str, dataset_name: str, 
                        model_name: str, seed: int) -> bool:
    """
    Check if OOD metrics (ood_auroc, ood_fpr95) already exist for this config.
    Returns True only if the file exists AND all relevant entries have OOD metrics.
    """
    # Filename structure is consistent with other checks
    # Note: if ood_output_dir was used, output_base_dir passed here should be that dir
    output_filename = f"ablation_{training_method}_{dataset_name}_{model_name}_seed{seed}.json"
    output_path = os.path.join(output_base_dir, output_filename)
    
    if not os.path.exists(output_path):
        return False
    
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
            
        # Recursive function to check for OOD metrics in all dictionaries
        def has_ood_metrics(item):
            # If it's a dict and looks like a result entry (has accuracy/ece), check for OOD
            if isinstance(item, dict):
                # Heuristic: if it has 'accuracy' or 'ece', it should also have 'ood_auroc'
                if 'accuracy' in item or 'ece' in item:
                    if 'ood_auroc' not in item or 'ood_fpr95' not in item:
                        return False
                # Recurse into children
                for v in item.values():
                    if not has_ood_metrics(v):
                        return False
            elif isinstance(item, list):
                for v in item:
                    if not has_ood_metrics(v):
                        return False
            return True

        # Check random_layer_ablation results if present
        if "random_layer_ablation" in data:
           if not has_ood_metrics(data["random_layer_ablation"]):
               return False
               
        # Check baselines if present
        if "baselines" in data:
           if not has_ood_metrics(data["baselines"]):
               return False
               
        # Check oracles if present
        for key in ["geo_best", "geo_comb", "tulip_best", "tulip_comb"]:
            if key in data:
                if not has_ood_metrics(data[key]):
                    return False
                    
        return True

    except Exception as e:
        logger.warning(f"Failed to check OOD results in {output_path}: {e}")
        return False

def generate_sbatch_command(
    job_name: str,
    model_name: str,
    dataset_name: str,
    training_method: str,
    seed: int,
    results_base_dir: str,
    output_base_dir: str,
    batch_size: int = 64,
    device: str = 'cuda',
    num_random_seeds: int = 5,
    layer_counts: List[int] = None,
    target_dims: List[int] = None,
    skip_block_random: bool = False,
    slurm_config: Dict[str, Any] = None,
    pixel_mode: bool = False,
    no_first_stage_compression: bool = False,
    skip_single_layers: bool = False,
    ood_only: bool = False,
    ood_output_dir: str = None,
    skip_ood: bool = False,
    corruption_type: str = None,
    corruption_severity: int = None,
) -> str:
    
    if slurm_config is None: slurm_config = {}
    if layer_counts is None: layer_counts = [2, 4, 5,  6, 8, 10, 12]
    if target_dims is None: target_dims = [64, 128, 256, 512, 1024, 2048, 3072, 4096]
    partition = slurm_config.get('partition', 'gpu')
    time_limit = slurm_config.get('time', '4-04:00:00') # 4 hours should be enough
    mem = slurm_config.get('mem', '64G')
    cpus = slurm_config.get('cpus', 4)
    gpus = slurm_config.get('gpus', 1)
    gpu_type = slurm_config.get('gpu_type', 'rtx_4090')
    conda_env = slurm_config.get('conda_env', 'tamar_n_env')
    
    project_root = Path(__file__).resolve().parent.parent
    log_dir = Path(output_base_dir) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"
    
    python_cmd = [
        sys.executable,
        str(project_root / "Experiments" / "run_random_layer_ablation.py"),
        "--model-name", model_name,
        "--dataset", dataset_name,
        "--training-method", training_method,
        "--seed", str(seed),
        "--results-base-dir", results_base_dir,
        "--output-dir", output_base_dir,
        "--batch-size", str(batch_size),
        "--device", device,
        "--num-random-seeds", str(num_random_seeds),
        "--target-dims"
    ] + [str(td) for td in target_dims]

    if pixel_mode:
        python_cmd.append("--pixel-mode")
        if no_first_stage_compression:
            python_cmd.append("--no-first-stage-compression")
    else:
        python_cmd += ["--layer-counts"] + [str(lc) for lc in layer_counts]
        if skip_block_random:
            python_cmd.append("--skip-block-random")
        if skip_single_layers:
            python_cmd.append("--skip-single-layers")

    if ood_only:
        python_cmd.append("--ood-only")
    if ood_output_dir:
        python_cmd += ["--ood-output-dir", ood_output_dir]
    if skip_ood:
        python_cmd.append("--skip-ood")
    if corruption_type is not None:
        python_cmd += [
            "--corruption-type",
            corruption_type,
            "--corruption-severity",
            str(corruption_severity),
        ]
    
    safe_python_cmd = " ".join(shlex.quote(a) for a in python_cmd)
    
    wrap_script = f"""
echo '========================================'
echo '🔬 SLURM JOB: Random Layer Ablation'
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
        f"--mem={shlex.quote(mem)}",
        f"--wrap={shlex.quote(clean_wrap_script)}",
    ]
    
    return " ".join(sbatch_cmd)

def main():
    parser = argparse.ArgumentParser(description="Runner for Random Ablation")
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--training-methods', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--results-base-dir', type=str, required=True)
    parser.add_argument('--output-base-dir', type=str, required=True)
    parser.add_argument('--num-random-seeds', type=int, default=1,
                        help='Number of random trials per configuration')
    parser.add_argument('--layer-counts', nargs='+', type=int, default=[2, 4, 5, 6, 8, 10, 12],
                        help='Number of layers to sample for global strategy')
    parser.add_argument('--target-dims', nargs='+', type=int, default=[64, 128, 256, 512, 1024, 2048, 3072, 4096],
                        help='Target dimensions to test (fixed output sizes)')
    parser.add_argument('--skip-block-random', action='store_true',
                        help='Skip block_random strategy (saves ~50%% computation time)')
    parser.add_argument('--run-pixel-experiments', action='store_true',
                        help='Submit pixel-space experiments (with and without Stage1 compression)')
    parser.add_argument('--run-layer-experiments', action='store_true',
                        help='Submit standard layer ablation experiments')
    parser.add_argument(
        '--skip-single-layers',
        action='store_true',
        default=False,
        help='Skip single-layer oracle experiments (geo_best, tulip_best)'
    )
    parser.add_argument(
        '--ood-only',
        action='store_true',
        help='Run only OOD detection evaluation on existing results (pass-through to ablation script)'
    )
    parser.add_argument(
        '--ood-output-dir',
        type=str,
        default=None,
        help='Optional separate directory for OOD results (pass-through to ablation script)'
    )
    parser.add_argument(
        '--skip-ood',
        action='store_true',
        help='Skip OOD (SVHN) evaluation in ablation script to save memory',
    )
    parser.add_argument(
        '--corruption-types',
        nargs='+',
        default=None,
        help=(
            "List of CIFAR-C corruption types to evaluate "
            "(e.g., 'gaussian_noise motion_blur fog')."
        ),
    )
    parser.add_argument(
        '--corruption-severities',
        nargs='+',
        type=int,
        default=[3],
        help='List of severity levels (1-5) to evaluate for CIFAR-C corruptions.',
    )
    parser.add_argument(
        '--run-all-corruptions',
        action='store_true',
        help='Run all 15 standard CIFAR-C corruptions (see CIFAR_C_CORRUPTIONS).',
    )
    parser.add_argument('--dry-run', action='store_true')
    
    args = parser.parse_args()
    
    # Generate grid
    experiments = list(itertools.product(
        args.models, args.datasets, args.training_methods, args.seeds
    ))
    
    logger.info(f"Found {len(experiments)} experiments to run.")
    
    # If no mode specified, default to running layer experiments to preserve legacy behaviour
    run_layers = args.run_layer_experiments or (not args.run_layer_experiments and not args.run_pixel_experiments)
    run_pixels = args.run_pixel_experiments

    for model, dataset, method, seed in experiments:
        # Check if model exists first (simple validation)
        # Special handling for DINOv2 models
        is_dinov2 = "dino" in model.lower()
        model_path = None
        
        if is_dinov2:
            # DINOv2 models are saved in calibration_comparison directory with wildcard suffix
            model_name_pattern = f"{method}_{dataset}_{model}_seed{seed}*"
            calibration_pattern = os.path.join(
                args.results_base_dir, "calibration_comparison", 
                model_name_pattern, "best_model.pth"
            )
            matching_files = glob.glob(calibration_pattern)
            
            if matching_files:
                # Use the first matching file (or most recent if multiple)
                model_path = matching_files[0]
                logger.info(f"Found DINOv2 model: {model_path}")
            else:
                logger.warning(f"DINOv2 model not found with pattern: {calibration_pattern}")
                # Fallback to standard path structure - check flat path first, then nested
                base_path = os.path.join(
                    args.results_base_dir, "baseline", method, dataset, model, f"seed{seed}"
                )
                flat_checkpoint = os.path.join(base_path, "best_model.pth")
                nested_checkpoint = os.path.join(
                    base_path, f"{method}_{dataset}_{model}_seed{seed}", "best_model.pth"
                )
                
                if os.path.exists(flat_checkpoint):
                    model_path = flat_checkpoint
                    logger.info(f"Found DINOv2 model at flat path: {model_path}")
                else:
                    model_path = nested_checkpoint
                    logger.info(f"Trying nested path: {model_path}")
        else:
            # Standard path structure for non-DINOv2 models
            model_path = os.path.join(
                args.results_base_dir, "baseline", method, dataset, model, 
                f"seed{seed}", f"{method}_{dataset}_{model}_seed{seed}", "results.json"
            )
        

        if not os.path.exists(model_path):
            logger.warning(f"Skipping missing model: {model_path}")
            continue

        # Check if we should skip this job for OOD-only mode
        if args.ood_only:
             # If ood_output_dir is specified, we check there. Otherwise we check the main output_base_dir.
            check_dir = args.ood_output_dir if args.ood_output_dir else args.output_base_dir
            if check_ood_results_exist(check_dir, method, dataset, model, seed):
                logger.info(f"⏭️  Skipping {model}_{dataset}_{method}_s{seed}: OOD results already complete")
                continue


        # ===== LAYER EXPERIMENTS =====
        if run_layers:
            if check_results_exist(args.output_base_dir, method, dataset, model, seed):
                logger.info(f"⏭️  Skipping {model}_{dataset}_{method}_s{seed}: layer results already complete")
            else:
                output_filename = f"ablation_{method}_{dataset}_{model}_seed{seed}.json"
                output_path = os.path.join(args.output_base_dir, output_filename)
                if os.path.exists(output_path):
                    logger.info(f"📝 Extending existing layer file for {model}_{dataset}_{method}_s{seed}")
                else:
                    logger.info(f"🆕 Creating new layer file for {model}_{dataset}_{method}_s{seed}")

                job_name = f"abl_layer_{model}_{dataset}_{method}_s{seed}"
                cmd = generate_sbatch_command(
                    job_name, model, dataset, method, seed,
                    args.results_base_dir, args.output_base_dir,
                    num_random_seeds=args.num_random_seeds,
                    layer_counts=args.layer_counts,
                    target_dims=args.target_dims,
                    skip_block_random=args.skip_block_random,
                    pixel_mode=False,
                    skip_single_layers=args.skip_single_layers,
                    ood_only=args.ood_only,
                    ood_output_dir=args.ood_output_dir,
                    skip_ood=args.skip_ood,
                )
                if args.dry_run:
                    logger.info(f"[DRY RUN] Would submit: {cmd}")
                else:
                    try:
                        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                        job_id = result.stdout.strip().split()[-1] if result.stdout.strip() else "UNKNOWN"
                        logger.info(f"✅ Submitted job {job_id}: {job_name}")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"❌ Failed to submit {job_name}: {e.stderr if e.stderr else e}")
                    time.sleep(0.3)

        # ===== PIXEL EXPERIMENTS =====
        if run_pixels:
            # A) Pixel with Stage 1 compression (mimic SPP+JL)
            job_name_p1 = f"abl_pix_comp_{model}_{dataset}_{method}_s{seed}"
            cmd_p1 = generate_sbatch_command(
                job_name_p1, model, dataset, method, seed,
                args.results_base_dir, args.output_base_dir,
                num_random_seeds=1,  # deterministic
                target_dims=args.target_dims,
                pixel_mode=True,
                no_first_stage_compression=False,
                ood_only=args.ood_only,
                ood_output_dir=args.ood_output_dir,
                skip_ood=args.skip_ood,
            )
            # B) Pixel without Stage 1 compression (raw -> projection)
            job_name_p2 = f"abl_pix_raw_{model}_{dataset}_{method}_s{seed}"
            cmd_p2 = generate_sbatch_command(
                job_name_p2, model, dataset, method, seed,
                args.results_base_dir, args.output_base_dir,
                num_random_seeds=1,
                target_dims=args.target_dims,
                pixel_mode=True,
                no_first_stage_compression=True,
                ood_only=args.ood_only,
                ood_output_dir=args.ood_output_dir,
                skip_ood=args.skip_ood,
            )

            for job_name, cmd in [(job_name_p1, cmd_p1), (job_name_p2, cmd_p2)]:
                if args.dry_run:
                    logger.info(f"[DRY RUN] Would submit: {cmd}")
                else:
                    try:
                        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                        job_id = result.stdout.strip().split()[-1] if result.stdout.strip() else "UNKNOWN"
                        logger.info(f"✅ Submitted job {job_id}: {job_name}")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"❌ Failed to submit {job_name}: {e.stderr if e.stderr else e}")
                    time.sleep(0.3)

    # ===== CORRUPTION EXPERIMENTS (CIFAR-10-C / CIFAR-100-C) =====
    # These experiments run on corrupted test data while keeping train/val clean.
    if args.corruption_types or args.run_all_corruptions:
        # Validate severities
        for sev in args.corruption_severities:
            if sev < 1 or sev > 5:
                raise ValueError(
                    f"Invalid corruption severity {sev}. "
                    "Valid severities are integers in [1, 2, 3, 4, 5]."
                )
        
        # Build list of corruption types to run
        corruption_types: List[str] = []
        if args.run_all_corruptions:
            corruption_types.extend(CIFAR_C_CORRUPTIONS)
        if args.corruption_types:
            corruption_types.extend(args.corruption_types)
        
        # Deduplicate while preserving order
        seen = set()
        unique_corruption_types: List[str] = []
        for c in corruption_types:
            if c not in seen:
                seen.add(c)
                unique_corruption_types.append(c)
        
        logger.info(
            f"Running corruption benchmarks for types={unique_corruption_types} "
            f"and severities={args.corruption_severities}"
        )
        
        for model, dataset, method, seed in experiments:
            # Restrict to CIFAR datasets
            if dataset not in ["cifar10", "cifar100"]:
                logger.info(
                    f"Skipping corruption experiments for dataset={dataset}; "
                    "only cifar10/cifar100 are supported."
                )
                continue
            
            # As above, only run if model checkpoint exists
            is_dinov2 = "dino" in model.lower()
            if is_dinov2:
                model_name_pattern = f"{method}_{dataset}_{model}_seed{seed}*"
                calibration_pattern = os.path.join(
                    args.results_base_dir, "calibration_comparison", 
                    model_name_pattern, "best_model.pth"
                )
                matching_files = glob.glob(calibration_pattern)
                if matching_files:
                    model_path = matching_files[0]
                    logger.info(f"[corruption] Found DINOv2 model: {model_path}")
                else:
                    logger.warning(
                        f"[corruption] DINOv2 model not found with pattern: "
                        f"{calibration_pattern}"
                    )
                    model_path = os.path.join(
                        args.results_base_dir, "baseline", method, dataset, model, 
                        f"seed{seed}", f"{method}_{dataset}_{model}_seed{seed}", "best_model.pth"
                    )
            else:
                model_path = os.path.join(
                    args.results_base_dir, "baseline", method, dataset, model, 
                    f"seed{seed}", f"{method}_{dataset}_{model}_seed{seed}", "results.json"
                )
            
            if not os.path.exists(model_path):
                logger.warning(f"[corruption] Skipping missing model: {model_path}")
                continue
            
            for corruption_type in unique_corruption_types:
                for severity in args.corruption_severities:
                    if check_corruption_results_exist(
                        args.output_base_dir,
                        method,
                        dataset,
                        model,
                        seed,
                        corruption_type,
                        severity,
                    ):
                        logger.info(
                            f"⏭️  Skipping corruption {model}_{dataset}_{method}_s{seed} "
                            f"{corruption_type}_sev{severity}: results already complete"
                        )
                        continue
                    
                    job_name = (
                        f"abl_corr_{model}_{dataset}_{corruption_type[:8]}_s{severity}"
                    )
                    cmd = generate_sbatch_command(
                        job_name,
                        model,
                        dataset,
                        method,
                        seed,
                        args.results_base_dir,
                        args.output_base_dir,
                        num_random_seeds=args.num_random_seeds,
                        layer_counts=args.layer_counts,
                        target_dims=args.target_dims,
                        skip_block_random=True,          # identical to global_random
                        pixel_mode=False,
                        skip_single_layers=True,         # focus on multi-layer comparisons
                        ood_only=args.ood_only,
                        ood_output_dir=args.ood_output_dir,
                        skip_ood=False,                  # underlying script handles OOD skipping
                        corruption_type=corruption_type,
                        corruption_severity=severity,
                    )
                    if args.dry_run:
                        logger.info(f"[DRY RUN][corruption] Would submit: {cmd}")
                    else:
                        try:
                            result = subprocess.run(
                                cmd,
                                shell=True,
                                check=True,
                                capture_output=True,
                                text=True,
                            )
                            job_id = (
                                result.stdout.strip().split()[-1]
                                if result.stdout.strip()
                                else "UNKNOWN"
                            )
                            logger.info(
                                f"✅ Submitted corruption job {job_id}: {job_name} "
                                f"({corruption_type}, sev={severity})"
                            )
                        except subprocess.CalledProcessError as e:
                            logger.error(
                                f"❌ Failed to submit corruption job {job_name}: "
                                f"{e.stderr if e.stderr else e}"
                            )
                        time.sleep(0.3)

if __name__ == '__main__':
    main()