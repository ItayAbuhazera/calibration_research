#!/usr/bin/env python3
"""
run_dac_comparison_experiments.py

Parallel runner for DAC vs Geometric calibration comparison experiments using SLURM.
Submits jobs for different (model, dataset, training_method, seed) combinations.

Usage:
    python run_dac_comparison_experiments.py --config configs/dac_comparison_sweep.yaml
    
Or with inline arguments:
    python run_dac_comparison_experiments.py \
        --models resnet18 densenet121 \
        --datasets cifar10 cifar100 \
        --training-methods baseline augmix \
        --seeds 11 12 13 \
        --results-base-dir results/results \
        --output-base-dir calibration_comparison_results
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def generate_sbatch_command(
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
    device: str = "cuda",
    num_layers: int = 6,
    target_dimension: int = 256,
    skip_single_layers: bool = False,
    force_rerun_geometric: bool = False,
    eval_corruptions: bool = False,
    cifar_c_dir: str = None,
    mce_only: bool = False,
    deep_ensemble_scan: bool = False,
    ensemble_size: int = 3,
    methods: List[str] = None,
    dac_knn_backend: str = "faiss",
    slurm_config: Dict[str, Any] = None,
) -> str:
    """
    Generate an sbatch command string for a single DAC comparison job.
    Uses proper GPU allocation format: gpu_type:count (e.g., rtx_4090:1)

    Returns:
        Complete sbatch command string ready to execute
    """

    # Default SLURM configuration
    if slurm_config is None:
        slurm_config = {}

    partition = slurm_config.get("partition", "gpu")
    time_limit = slurm_config.get("time", "0-12:00:00")  # Format: d-hh:mm:ss
    mem = slurm_config.get("mem", "70G")
    cpus = slurm_config.get("cpus", 4)
    gpus = slurm_config.get("gpus", 1)
    gpu_type = slurm_config.get("gpu_type", "rtx_6000")
    conda_env = slurm_config.get("conda_env", "geo_cuda12")

    # Get project root
    project_root = Path(__file__).resolve().parent.parent

    # Create log directory
    log_dir = Path(output_dir) / "_dac_comparison_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"

    # Build python command
    python_cmd = [
        sys.executable,
        str(project_root / "Experiments" / "compare_dac_geometric.py"),
        "--model-name",
        model_name,
        "--dataset",
        dataset_name,
        "--training-method",
        training_method,
        "--results-base-dir",
        results_base_dir,
        "--output-dir",
        output_base_dir,
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]

    # Deep ensemble scan mode doesn't use seed, num_layers, target_dimension, or corruption
    if deep_ensemble_scan:
        python_cmd.extend(
            [
                "--deep-ensemble-scan",
                "--ensemble-size",
                str(ensemble_size),
                "--seed",
                str(seed),  # Used as master_seed for random group sampling
            ]
        )
    else:
        # Normal mode: include all standard arguments
        python_cmd.extend(
            [
                "--seed",
                str(seed),
                "--num-layers",
                str(num_layers),
                "--target-dimension",
                str(target_dimension),
            ]
        )
        if corruption_type:
            python_cmd.extend(["--corruption-type", corruption_type])
        if corruption_severity is not None:
            python_cmd.extend(["--corruption-severity", str(corruption_severity)])
        if skip_single_layers:
            python_cmd.append("--skip-single-layers")
        if force_rerun_geometric:
            python_cmd.append("--force-rerun-geometric")
        if eval_corruptions:
            python_cmd.append("--eval-corruptions")
        if cifar_c_dir:
            python_cmd.extend(["--cifar-c-dir", cifar_c_dir])
        if mce_only:
            python_cmd.append("--mce-only")
    if corruption_type:
        python_cmd.extend(["--corruption-type", corruption_type])
    if corruption_severity is not None:
        python_cmd.extend(["--corruption-severity", str(corruption_severity)])
    if skip_single_layers:
        python_cmd.append("--skip-single-layers")
    if force_rerun_geometric:
        python_cmd.append("--force-rerun-geometric")
    if methods:
        for method in methods:
            python_cmd.extend(["--method", method])
    if dac_knn_backend:
        python_cmd.extend(["--dac-knn-backend", dac_knn_backend])

    safe_python_cmd = " ".join(shlex.quote(a) for a in python_cmd)

    # Verify flag is in command (for debugging)
    if force_rerun_geometric and "--force-rerun-geometric" not in safe_python_cmd:
        logger.warning(
            f"WARNING: --force-rerun-geometric flag was set but not found in command for {job_name}"
        )
        logger.warning(f"Command: {safe_python_cmd[:200]}...")

    # Create wrap script
    eval_corruptions_str = "True" if eval_corruptions else "False"
    mce_only_str = "True" if mce_only else "False"
    cifar_c_dir_str = cifar_c_dir or ""
    deep_ensemble_scan_str = "True" if deep_ensemble_scan else "False"

    wrap_script = f"""
echo '========================================'
echo ' SLURM JOB: DAC vs Geometric Comparison'
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
if [ "{deep_ensemble_scan_str}" = "True" ]; then
    echo '  Mode       : Deep Ensemble Scan'
    echo '  Ensemble Size : {ensemble_size}'
else
    echo '  Mode       : Standard Comparison'
    if [ "{eval_corruptions_str}" = "True" ]; then
        echo '  Eval Corruptions : Enabled'
        if [ -n "{cifar_c_dir_str}" ]; then
            echo '  CIFAR-C Dir     : {cifar_c_dir_str}'
        fi
        if [ "{mce_only_str}" = "True" ]; then
            echo '  MCE Only        : Enabled'
        fi
    fi
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
    clean_wrap_script = "\n".join(
        line.lstrip() for line in wrap_script.strip().split("\n")
    )

    # Build sbatch command - FIXED GPU FORMAT
    sbatch_cmd = [
        "sbatch",
        f"--partition={shlex.quote(partition)}",
        f"--job-name={shlex.quote(job_name)}",
        f"--output={shlex.quote(str(log_file))}",
        f"--time={shlex.quote(time_limit)}",
        "--ntasks=1",
        f"--gpus={shlex.quote(f'{gpu_type}:{gpus}')}",  # FIX: Use gpu_type:count format
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
        result = subprocess.run(
            sbatch_cmd, shell=True, check=True, capture_output=True, text=True
        )
        # Parse job ID from output like "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        logger.info(f" Submitted job {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        logger.error(f" Failed to submit job: {e.stderr}")
        raise


def generate_experiment_grid(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all experiment configurations from config file.

    Returns:
        List of experiment configurations (dicts)
    """

    # Extract lists of parameters
    models = config.get("models", ["resnet18"])
    datasets = config.get("datasets", ["cifar10"])
    training_methods = config.get("training_methods", ["baseline"])
    seeds = config.get("seeds", [11])
    corruption_types = config.get("corruption_types")
    severity_levels = config.get("corruption_severity_levels")
    deep_ensemble_scan = config.get("deep_ensemble_scan", False)

    # Deep ensemble scan mode: only iterate over model/dataset/training_method
    # Seeds are used as master_seed for random group sampling, not for model selection
    if deep_ensemble_scan:
        experiments = []
        # Use first seed as master_seed (or default to 42)
        master_seed = seeds[0] if seeds else 42
        for model, dataset, training in itertools.product(
            models, datasets, training_methods
        ):
            exp = {
                "model_name": model,
                "dataset_name": dataset,
                "training_method": training,
                "seed": master_seed,  # Used as master_seed for random sampling
                "corruption_type": None,  # Not used in ensemble scan mode
                "corruption_severity": None,  # Not used in ensemble scan mode
            }
            experiments.append(exp)
        return experiments

    # Normal mode: If corruption is not specified, run clean experiments only
    if not corruption_types:
        corruption_types = [None]
        severity_levels = [None]
    elif not severity_levels:
        severity_levels = [1, 2, 3, 4, 5]

    # Generate Cartesian product
    experiments = []
    for model, dataset, training, seed, corruption, severity in itertools.product(
        models, datasets, training_methods, seeds, corruption_types, severity_levels
    ):
        exp = {
            "model_name": model,
            "dataset_name": dataset,
            "training_method": training,
            "seed": seed,
            "corruption_type": corruption,
            "corruption_severity": severity,
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
    Check if model results exist and has sufficient accuracy (>50%).

    Args:
        model_name, dataset_name, training_method, seed: Experiment config
        results_base_dir: Base directory with pre-trained models

    Returns:
        (is_valid, reason): True if model exists and has >50% accuracy,
                           False with reason string otherwise
    """
    import json

    # Construct path to results file
    # Format: results_base_dir/baseline/training_method/dataset/model/seed/training_dataset_model_seed/results.json
    results_path = os.path.join(
        results_base_dir,
        "baseline",
        training_method,
        dataset_name,
        model_name,
        f"seed{seed}",
        f"{training_method}_{dataset_name}_{model_name}_seed{seed}",
        "results.json",
    )

    # Check if results file exists
    if not os.path.exists(results_path):
        return False, f"Results file not found: {results_path}"

    # Load and check accuracy
    try:
        with open(results_path, "r") as f:
            results = json.load(f)

        # Extract accuracy from evaluation_results
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


def check_metrics_in_results(result_file: str) -> Tuple[bool, str]:
    """
    Check if a completed results file contains all required sections with proper completeness.
    Uses the same required fields logic as compare_dac_geometric.py.

    Args:
        result_file: Path to the comparison results JSON file

    Returns:
        (is_complete, reason): True if all required sections are complete, False with reason otherwise
    """
    import json

    if not os.path.exists(result_file):
        return False, "File does not exist"

    # Required fields for different experiment types
    GEO_REQUIRED_FIELDS = ["ece", "calibrator_params"]
    DAC_REQUIRED_FIELDS = ["ece", "layer_weights"]

    def is_result_complete(entry, required_fields):
        """Check if a result entry has all required fields"""
        if not isinstance(entry, dict):
            return False
        if "error" in entry:
            return False
        for k in required_fields:
            if k not in entry or entry[k] is None:
                return False
        return True

    def is_section_complete(results, section_name, required_fields):
        """Check if a section exists and is complete"""
        # Handle nested sections (e.g., 'ablation.dac_with_sgc_aggregated_features')
        parts = section_name.split(".")
        section = results
        for part in parts:
            if not isinstance(section, dict) or part not in section:
                return False
            section = section[part]

        # Check if section is complete
        if isinstance(section, dict):
            # Single result dict
            return is_result_complete(section, required_fields)
        elif isinstance(section, list):
            # List of results - check if all are complete
            if len(section) == 0:
                return False
            return all(is_result_complete(item, required_fields) for item in section)
        else:
            return False

    try:
        with open(result_file, "r") as f:
            results = json.load(f)

        # Check standard baselines (must exist)
        if "standard_baselines" not in results:
            return False, "Missing 'standard_baselines' section"

        # Check key geometric sections (if not skipped)
        if "geometric_original" in results:
            geo_original = results["geometric_original"]
            # Check if it's marked as skipped
            if isinstance(geo_original, dict) and geo_original.get("skipped") is True:
                pass  # Skip this check
            elif isinstance(geo_original, dict):
                # geometric_original is a dict of layer results (layer1, layer2, etc.)
                # Check each layer result
                for layer_key, layer_result in geo_original.items():
                    if not is_result_complete(layer_result, GEO_REQUIRED_FIELDS):
                        return (
                            False,
                            f"Incomplete 'geometric_original.{layer_key}' section (missing calibrator_params)",
                        )
            else:
                return False, "Invalid 'geometric_original' section format"

        # Check DAC-weighted (uses DAC_REQUIRED_FIELDS)
        if "geometric_dac_weighted" in results:
            if not is_section_complete(
                results, "geometric_dac_weighted", DAC_REQUIRED_FIELDS
            ):
                return (
                    False,
                    "Incomplete 'geometric_dac_weighted' section (missing layer_weights)",
                )

        # Check ablation studies
        if "ablation" in results:
            ablation = results["ablation"]
            # DAC-based ablations need layer_weights
            if "dac_with_sgc_aggregated_features" in ablation:
                if not is_section_complete(
                    results,
                    "ablation.dac_with_sgc_aggregated_features",
                    DAC_REQUIRED_FIELDS,
                ):
                    return (
                        False,
                        "Incomplete 'ablation.dac_with_sgc_aggregated_features' (missing layer_weights)",
                    )
            if "dac_with_random_layer_selection" in ablation:
                if not is_section_complete(
                    results,
                    "ablation.dac_with_random_layer_selection",
                    DAC_REQUIRED_FIELDS,
                ):
                    return (
                        False,
                        "Incomplete 'ablation.dac_with_random_layer_selection' (missing layer_weights)",
                    )
            # Geometric-based ablations need calibrator_params
            if "sgc_with_dac_preprocessing" in ablation:
                if not is_section_complete(
                    results, "ablation.sgc_with_dac_preprocessing", GEO_REQUIRED_FIELDS
                ):
                    return (
                        False,
                        "Incomplete 'ablation.sgc_with_dac_preprocessing' (missing calibrator_params)",
                    )
            if "random_single_layer_dac_preprocessing" in ablation:
                if not is_section_complete(
                    results,
                    "ablation.random_single_layer_dac_preprocessing",
                    GEO_REQUIRED_FIELDS,
                ):
                    return (
                        False,
                        "Incomplete 'ablation.random_single_layer_dac_preprocessing' (missing calibrator_params)",
                    )

        # Check other geometric sections
        geometric_sections = [
            "global_random",
            "coordinate_sampling",
            "coordinate_spp",
            "coordinate_spp_dac",
            "geometric_concatenated",
            "sgc_faiss",
            "coordinate_sampling_faiss",
            "coordinate_spp_faiss",
        ]
        for section in geometric_sections:
            if section in results:
                if not is_section_complete(results, section, GEO_REQUIRED_FIELDS):
                    return (
                        False,
                        f"Incomplete '{section}' section (missing calibrator_params)",
                    )

        # Check DAC sections
        if "dac_with_coordinate_features" in results:
            if not is_section_complete(
                results, "dac_with_coordinate_features", DAC_REQUIRED_FIELDS
            ):
                return (
                    False,
                    "Incomplete 'dac_with_coordinate_features' section (missing layer_weights)",
                )

        # Check normalization ablation (geometric-based)
        if "normalization_ablation" in results:
            norm_ablation = results["normalization_ablation"]
            if isinstance(norm_ablation, dict) and "skipped" not in norm_ablation:
                # Check each experiment in normalization ablation
                for exp_name, exp_result in norm_ablation.items():
                    if not is_result_complete(exp_result, GEO_REQUIRED_FIELDS):
                        return (
                            False,
                            f"Incomplete 'normalization_ablation.{exp_name}' (missing calibrator_params)",
                        )

        # Check metric-guided calibration (if not skipped)
        if "metric_guided_calibration" in results:
            metric_data = results["metric_guided_calibration"]
            # Check if it's marked as skipped
            if isinstance(metric_data, dict) and metric_data.get("skipped") is True:
                pass  # Skip this check
            elif isinstance(metric_data, list):
                if len(metric_data) == 0:
                    return False, "'metric_guided_calibration' list is empty"
                for entry in metric_data:
                    if not is_result_complete(entry, GEO_REQUIRED_FIELDS):
                        return (
                            False,
                            "'metric_guided_calibration' has incomplete entries (missing calibrator_params)",
                        )
            elif isinstance(metric_data, dict):
                if len(metric_data) == 0:
                    return False, "'metric_guided_calibration' dict is empty"

        # Check analysis section (not required for deep ensemble scan)
        if "analysis" not in results or "best_overall_method" not in results.get(
            "analysis", {}
        ):
            # Check if this is a deep ensemble scan result
            if "deep_ensemble_scan" in results:
                # Deep ensemble scan mode - check if it has required fields
                ensemble_data = results["deep_ensemble_scan"]
                if isinstance(ensemble_data, dict):
                    if ensemble_data.get("skipped"):
                        return True, "Deep ensemble scan skipped (insufficient seeds)"
                    required_ensemble_fields = [
                        "ece_mean",
                        "ece_std",
                        "accuracy_mean",
                        "n_groups",
                    ]
                    if all(k in ensemble_data for k in required_ensemble_fields):
                        return True, "Deep ensemble scan complete"
                    else:
                        return (
                            False,
                            f"Deep ensemble scan incomplete (missing: {[k for k in required_ensemble_fields if k not in ensemble_data]})",
                        )
                return False, "Invalid 'deep_ensemble_scan' section format"
            return False, "Missing 'analysis.best_overall_method'"

        return True, "All required sections are complete"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def filter_existing_experiments(
    experiments: List[Dict[str, Any]],
    output_base_dir: str,
    results_base_dir: str,
    skip_completed: bool = True,
) -> List[Dict[str, Any]]:
    """
    Filter out experiments that have already been completed or have invalid models.

    Args:
        experiments: List of experiment configs
        output_base_dir: Base output directory for comparison results
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
        # Check if comparison results file already exists
        # Must match the filename pattern used in compare_dac_geometric.py
        if exp.get("corruption_type"):
            result_file = os.path.join(
                output_base_dir,
                f"ablation_{exp['training_method']}_{exp['dataset_name']}_{exp['model_name']}_seed{exp['seed']}_{exp['corruption_type']}_sev{exp.get('corruption_severity')}.json",
            )
        else:
            result_file = os.path.join(
                output_base_dir,
                f"ablation_{exp['training_method']}_{exp['dataset_name']}_{exp['model_name']}_seed{exp['seed']}.json",
            )

        if skip_completed and os.path.exists(result_file):
            # Check if the results file is complete (has metrics)
            is_complete, reason = check_metrics_in_results(result_file)

            if is_complete:
                logger.info(
                    "Skipping completed: "
                    f"{exp['model_name']}/{exp['dataset_name']}/{exp['training_method']}/seed{exp['seed']}"
                )
                skipped_completed += 1
                continue
            else:
                logger.warning(
                    " Re-running incomplete experiment: "
                    f"{exp['model_name']}/{exp['dataset_name']}/{exp['training_method']}/seed{exp['seed']} - {reason}"
                )
                skipped_incomplete += 1
                # Don't skip - allow it to be re-run

        # Check if model exists and has sufficient accuracy
        is_valid, reason = validate_experiment_model(
            model_name=exp["model_name"],
            dataset_name=exp["dataset_name"],
            training_method=exp["training_method"],
            seed=exp["seed"],
            results_base_dir=results_base_dir,
        )

        if not is_valid:
            logger.warning(
                "  Skipping invalid model: "
                f"{exp['model_name']}/{exp['dataset_name']}/{exp['training_method']}/seed{exp['seed']} - {reason}"
            )
            skipped_invalid += 1
            continue

        logger.debug(
            f" Valid: {exp['model_name']}/{exp['dataset_name']}/{exp['training_method']}/seed{exp['seed']} - {reason}"
        )
        filtered.append(exp)

    logger.info(
        " Skipped "
        f"{skipped_completed} completed experiments, "
        f"{skipped_incomplete} incomplete (will re-run), "
        f"{skipped_invalid} invalid models, running {len(filtered)} experiments"
    )
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Parallel runner for DAC vs Geometric calibration comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML configuration file"
    )

    # Experiment parameters (can override config file)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model architectures (e.g., resnet18 densenet121)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets (e.g., cifar10 cifar100 tiny_imagenet)",
    )
    parser.add_argument(
        "--training-methods",
        nargs="+",
        default=None,
        help="Training methods (e.g., baseline augmix)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds (e.g., 11 12 13)",
    )
    parser.add_argument(
        "--corruption-types",
        nargs="+",
        default=None,
        help="Optional corruption types (e.g., gaussian_noise). If omitted, runs on clean data.",
    )
    parser.add_argument(
        "--corruption-severity-levels",
        nargs="+",
        type=int,
        default=None,
        help="Optional corruption severity levels (1-5). If omitted, runs on clean data.",
    )

    # Paths
    parser.add_argument(
        "--results-base-dir",
        type=str,
        default=None,
        help="Base directory with pre-trained models",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default=None,
        help="Output directory for comparison results",
    )
    parser.add_argument(
        "--slurm-output-dir",
        type=str,
        default="slurm_jobs",
        help="Directory for SLURM scripts and logs",
    )

    # Technical settings
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for data processing"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for computations"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of random layers L for SGC (paper default: 6)",
    )
    parser.add_argument(
        "--target-dimension",
        type=int,
        default=256,
        help="Target dimension d for SGC (paper default: 256)",
    )

    # SLURM settings
    parser.add_argument("--partition", type=str, default="gpu", help="SLURM partition")
    parser.add_argument(
        "--time", type=str, default="0-12:00:00", help="Time limit (d-hh:mm:ss format)"
    )
    parser.add_argument("--mem", type=str, default="70G", help="Memory per job")
    parser.add_argument("--cpus", type=int, default=4, help="CPUs per task")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs per job")
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="rtx_6000",
        help="GPU type (e.g., a100, v100, rtx_4090)",
    )
    parser.add_argument(
        "--conda-env", type=str, default="geo_cuda12", help="Conda environment name"
    )

    # Runner options
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate scripts but don't submit jobs"
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="Skip experiments with existing results",
    )
    parser.add_argument(
        "--no-skip-completed",
        dest="skip_completed",
        action="store_false",
        help="Re-run all experiments even if results exist",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Maximum number of jobs to submit (for testing)",
    )
    parser.add_argument(
        "--skip-single-layers",
        action="store_true",
        default=True,
        help="Skip single-layer geometric calibration experiments",
    )
    parser.add_argument(
        "--force-rerun-geometric",
        action="store_true",
        help="Force re-run all geometric calibration experiments (uses rank-based normalization instead of buggy percentile normalization)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        action="append",
        default=None,
        help="Specific geometric methods to run (can be repeated). If provided, only these methods will run. Valid methods: sgc_with_dac_preprocessing_separation, sgc_with_dac_layers_separation, sgc_with_tulip_layers_separation, last_layer_only_baseline, global_random_separation, sgc_faiss, coordinate_sampling_separation, coordinate_spp_separation, coordinate_spp_faiss, coordinate_spp_dac, dac_with_coordinate_features",
    )
    parser.add_argument(
        "--dac-knn-backend",
        type=str,
        default="faiss",
        choices=["faiss", "torch", "both"],
        help='kNN backend for DAC calibration: "faiss" (default, FAISS-based), "torch" (PyTorch-based for fair timing with RGCL/RGCC), or "both" (run both and store separately)',
    )

    # CIFAR-C corruption evaluation options
    parser.add_argument(
        "--eval-corruptions",
        action="store_true",
        help="Enable mCE evaluation on CIFAR-C corruptions",
    )
    parser.add_argument(
        "--cifar-c-dir",
        type=str,
        default=None,
        help="Path to CIFAR-C data directory (default: data/cifar10-c or data/cifar100-c)",
    )
    parser.add_argument(
        "--mce-only",
        action="store_true",
        help="Skip clean ECE evaluation, only compute mCE on corruptions",
    )

    # Deep ensemble scan options
    parser.add_argument(
        "--deep-ensemble-scan",
        action="store_true",
        help="Run deep ensemble scan mode (find all seeds, create groups, evaluate each)",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=3,
        help="Number of models per ensemble group (default: 3, only used with --deep-ensemble-scan)",
    )

    args = parser.parse_args()

    # Debug: Log if force-rerun-geometric is set
    if args.force_rerun_geometric:
        logger.info("=" * 80)
        logger.info(" --force-rerun-geometric FLAG IS SET")
        logger.info("   All geometric calibration experiments will be force-rerun")
        logger.info("=" * 80)

    # Flatten methods list (since we used action='append')
    selected_methods = None
    if args.methods:
        selected_methods = [m for group in args.methods for m in group]

    # Load configuration
    config = {}
    if args.config:
        logger.info(f" Loading configuration from {args.config}")
        config = load_config(args.config)

    # Override with command-line arguments
    if args.models:
        config["models"] = args.models
    if args.datasets:
        config["datasets"] = args.datasets
    if args.training_methods:
        config["training_methods"] = args.training_methods
    if args.seeds:
        config["seeds"] = args.seeds
    if args.results_base_dir:
        config["results_base_dir"] = args.results_base_dir
    if args.output_base_dir:
        config["output_base_dir"] = args.output_base_dir
    if args.corruption_types:
        config["corruption_types"] = args.corruption_types
    if args.corruption_severity_levels:
        config["corruption_severity_levels"] = args.corruption_severity_levels
    if args.eval_corruptions:
        config["eval_corruptions"] = True
    if args.cifar_c_dir:
        config["cifar_c_dir"] = args.cifar_c_dir
    if args.mce_only:
        config["mce_only"] = True
    if args.deep_ensemble_scan:
        config["deep_ensemble_scan"] = True
    if args.ensemble_size:
        config["ensemble_size"] = args.ensemble_size

    # Validate required parameters
    # Deep ensemble scan mode doesn't require seeds (it discovers them automatically)
    if args.deep_ensemble_scan:
        required = [
            "models",
            "datasets",
            "training_methods",
            "results_base_dir",
            "output_base_dir",
        ]
        # Seeds are optional in ensemble scan mode (used as master_seed for random sampling)
        if "seeds" not in config:
            config["seeds"] = [42]  # Default master seed
            logger.info("Using default seed=42 as master_seed for ensemble scan mode")
    else:
        required = [
            "models",
            "datasets",
            "training_methods",
            "seeds",
            "results_base_dir",
            "output_base_dir",
        ]

    missing = [k for k in required if k not in config]
    if missing:
        logger.error(f" Missing required parameters: {missing}")
        logger.error("   Provide via --config file or command-line arguments")
        return 1

    # SLURM configuration
    slurm_config = {
        "partition": args.partition,
        "time": args.time,
        "mem": args.mem,
        "cpus": args.cpus,
        "gpus": args.gpus,
        "gpu_type": args.gpu_type,
        "conda_env": args.conda_env,
    }

    # Generate experiment grid
    logger.info(" Generating experiment grid...")
    if config.get("deep_ensemble_scan", False):
        logger.info("   Mode: Deep Ensemble Scan (will discover seeds automatically)")
        logger.info(f"   Ensemble size: {config.get('ensemble_size', 3)}")
    experiments = generate_experiment_grid(config)
    total_experiments = len(experiments)
    logger.info(f"   Generated {total_experiments} total experiments")

    # Filter completed / invalid experiments
    experiments = filter_existing_experiments(
        experiments,
        output_base_dir=config["output_base_dir"],
        results_base_dir=config["results_base_dir"],
        skip_completed=args.skip_completed,
    )

    skipped_experiments = total_experiments - len(experiments)

    if not experiments:
        logger.info(" All experiments already completed!")
        return 0

    # Limit number of jobs (for testing)
    if args.max_jobs and len(experiments) > args.max_jobs:
        logger.warning(f"  Limiting to {args.max_jobs} jobs (--max-jobs)")
        experiments = experiments[: args.max_jobs]

    # Create SLURM scripts and submit jobs
    # Reorganize experiments: first by seed, then by model, then by training_method, then by dataset
    # This allows multiple models to run in parallel for each seed
    logger.info(f" Organizing {len(experiments)} experiments for submission...")

    # Group experiments by seed -> model -> training_method -> dataset
    experiments_by_seed = {}
    for exp in experiments:
        seed = exp["seed"]
        if seed not in experiments_by_seed:
            experiments_by_seed[seed] = {}

        model = exp["model_name"]
        if model not in experiments_by_seed[seed]:
            experiments_by_seed[seed][model] = {}

        training_method = exp["training_method"]
        if training_method not in experiments_by_seed[seed][model]:
            experiments_by_seed[seed][model][training_method] = []

        experiments_by_seed[seed][model][training_method].append(exp)

    # Count total jobs for progress tracking
    total_jobs = len(experiments)

    logger.info(
        f" Submitting {total_jobs} jobs in order: seed -> model -> training_method -> dataset"
    )
    logger.info(f"   This allows multiple models to run in parallel for each seed")

    submitted_jobs = []
    failed_jobs = []
    job_counter = 0

    # Submit jobs grouped by seed (all models for a seed submitted together for parallel execution)
    for seed in sorted(experiments_by_seed.keys()):
        logger.info(f"\n Processing seed {seed}...")
        seed_jobs = []

        # Collect all jobs for this seed (all models, training methods, datasets)
        # Order: model -> training_method -> dataset (with corruption variants sorted)
        for model in sorted(experiments_by_seed[seed].keys()):
            for training_method in sorted(experiments_by_seed[seed][model].keys()):
                # Sort experiments by dataset, then corruption type, then severity
                sorted_exps = sorted(
                    experiments_by_seed[seed][model][training_method],
                    key=lambda x: (
                        x["dataset_name"],
                        x.get("corruption_type") or "",
                        x.get("corruption_severity") or 0,
                    ),
                )

                for exp in sorted_exps:
                    # Generate job name
                    if config.get("deep_ensemble_scan", False):
                        job_name = f"ensemble_scan_{exp['model_name']}_{exp['dataset_name']}_{exp['training_method']}"
                    else:
                        # Include corruption if present, or mCE eval if enabled
                        corruption_suffix = ""
                        if exp.get("corruption_type"):
                            corruption_suffix = f"_{exp['corruption_type']}_sev{exp.get('corruption_severity')}"
                        elif config.get("eval_corruptions", False):
                            corruption_suffix = "_mce"
                        job_name = f"dac_cmp_{exp['model_name']}_{exp['dataset_name']}_{exp['training_method']}_s{exp['seed']}{corruption_suffix}"

                    try:
                        # Generate sbatch command
                        sbatch_cmd = generate_sbatch_command(
                            job_name=job_name,
                            output_dir=args.slurm_output_dir,
                            model_name=exp["model_name"],
                            dataset_name=exp["dataset_name"],
                            training_method=exp["training_method"],
                            seed=exp["seed"],
                            corruption_type=exp.get("corruption_type"),
                            corruption_severity=exp.get("corruption_severity"),
                            results_base_dir=config["results_base_dir"],
                            output_base_dir=config["output_base_dir"],
                            batch_size=args.batch_size,
                            device=args.device,
                            num_layers=args.num_layers,
                            target_dimension=args.target_dimension,
                            skip_single_layers=args.skip_single_layers,
                            force_rerun_geometric=args.force_rerun_geometric,
                            eval_corruptions=config.get("eval_corruptions", False),
                            cifar_c_dir=config.get("cifar_c_dir"),
                            mce_only=config.get("mce_only", False),
                            deep_ensemble_scan=config.get("deep_ensemble_scan", False),
                            ensemble_size=config.get("ensemble_size", 3),
                            methods=selected_methods,
                            dac_knn_backend=args.dac_knn_backend,
                            slurm_config=slurm_config,
                        )
                        seed_jobs.append((job_name, sbatch_cmd, exp))
                    except Exception as e:
                        logger.error(f" Failed to prepare {job_name}: {e}")
                        failed_jobs.append((job_name, str(e)))

        # Submit all jobs for this seed (all models will run in parallel)
        models_in_seed = sorted(set(j[2]["model_name"] for j in seed_jobs))
        logger.info(
            f"   Submitting {len(seed_jobs)} jobs for seed {seed} (models: {', '.join(models_in_seed)})"
        )
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
                logger.error(f" Failed to submit {job_name}: {e}")
                failed_jobs.append((job_name, str(e)))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUBMISSION SUMMARY")
    logger.info("=" * 80)
    logger.info(f" Successfully submitted: {len(submitted_jobs)}")
    logger.info(f"Skipped (total): {skipped_experiments}")
    logger.info(f" Failed: {len(failed_jobs)}")

    if submitted_jobs:
        logger.info("\n Submitted jobs:")
        for name, job_id, cmd in submitted_jobs[:10]:  # Show first 10
            logger.info(f"   {job_id}: {name}")
        if len(submitted_jobs) > 10:
            logger.info(f"   ... and {len(submitted_jobs) - 10} more")

    if failed_jobs:
        logger.info("\n Failed jobs:")
        for name, error in failed_jobs:
            logger.info(f"   {name}: {error}")

    if args.dry_run:
        logger.info("\n DRY RUN MODE - No jobs were actually submitted")
        logger.info(f"   Commands would be executed directly with sbatch --wrap")
    else:
        logger.info(f"\n Monitor jobs with: squeue -u $USER")
        logger.info(f" SLURM logs in: {args.slurm_output_dir}/_dac_comparison_logs/")

    logger.info("=" * 80)

    return 0 if not failed_jobs else 1


if __name__ == "__main__":
    exit(main())
