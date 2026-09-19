"""
Hybrid Multi-Metric Ensemble Driver

Orchestrates the complete workflow:
1. Load existing layer selection metrics
2. Apply new multi-metric selection strategies
3. Run multi-layer ensemble with selected layers
4. Aggregate results
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
import shlex

# Ensure project root on path for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.metric_config import MetricPresets  # noqa: E402
from Experiments.layer_selection_strategies import (  # noqa: E402
    epsilon_ball_selection,
    metric_voting_selection,
    pareto_front_selection,
    single_metric_selection,
)


from utils.logging_config import get_logger
logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single ensemble experiment."""

    dataset: str
    model: str
    training_method: str
    seed: int
    selection_strategy_name: str
    weighting_strategy: str
    shared_cache_dir: Optional[Path]
    selected_layers: List[int]
    metrics_file: Path
    output_dir: Path
    disable_cache: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""

        d = asdict(self)
        d["metrics_file"] = str(self.metrics_file)
        d["output_dir"] = str(self.output_dir)
        d["shared_cache_dir"] = (
            str(self.shared_cache_dir) if self.shared_cache_dir is not None else None
        )
        return d


@dataclass
class StrategyDefinition:
    """Definition of a layer selection strategy."""

    name: str
    type: str  # 'pareto', 'voting', 'epsilon', 'single'
    metrics: List[str]
    params: Dict | None = None

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}


# Additional Pareto preset metric groups
PARETO_ALL_STARS = [
    'margin_tail_cvar',
    'local_intrinsic_dimensionality',
    'avg_class_separation_ratio',
    'impostor_gap_cvar',
]

PARETO_ACCURACY_FOCUSED = [
    'margin_tail_cvar',
    'impostor_gap_cvar',
    'margin_skewkurt_safety',
    'local_intrinsic_dimensionality',
]

PARETO_RELIABILITY_FOCUSED = [
    'margin_tail_cvar',
    'local_intrinsic_dimensionality',
    'avg_class_separation_ratio',
    'boundary_proximity_correlation',
    'calibration_decisiveness',
]


# Predefined strategies
STRATEGY_PRESETS: Dict[str, StrategyDefinition] = {
    "pareto_core": StrategyDefinition(
        "pareto_core", "pareto", MetricPresets.PARETO_CORE
    ),
    "pareto_extended": StrategyDefinition(
        "pareto_extended", "pareto", MetricPresets.PARETO_EXTENDED
    ),
    "pareto_calibration": StrategyDefinition(
        "pareto_calibration", "pareto", MetricPresets.CALIBRATION_FOCUSED
    ),
    "voting_top3_core": StrategyDefinition(
        "voting_top3_core", "voting", MetricPresets.PARETO_CORE, {"k_per_metric": 3}
    ),
    "voting_top5_core": StrategyDefinition(
        "voting_top5_core", "voting", MetricPresets.PARETO_CORE, {"k_per_metric": 5}
    ),
    "voting_top2_core": StrategyDefinition(
        "voting_top2_core", "voting", MetricPresets.PARETO_CORE, {"k_per_metric": 2}
    ),
    "epsilon_union": StrategyDefinition(
        "epsilon_union",
        "epsilon",
        MetricPresets.PARETO_CORE,
        {"epsilon": 0.01, "mode": "union"},
    ),
    "epsilon_intersection": StrategyDefinition(
        "epsilon_intersection",
        "epsilon",
        MetricPresets.PARETO_CORE,
        {"epsilon": 0.01, "mode": "intersection"},
    ),
    "single_ece_top5": StrategyDefinition("single_ece_top5", "single", ["ece_score"], {"k": 5}),
    "single_brier_top5": StrategyDefinition("single_brier_top5", "single", ["brier_score"], {"k": 5}),
    "single_margin_tail_cvar_top5": StrategyDefinition("single_margin_tail_cvar_top5", "single", ["margin_tail_cvar"], {"k": 5}),
    "single_multiscale_separation_top5": StrategyDefinition("single_multiscale_separation_top5", "single", ["multiscale_separation"], {"k": 5}),
    "single_calibration_decisiveness_top5": StrategyDefinition("single_calibration_decisiveness_top5", "single", ["calibration_decisiveness"], {"k": 5}),
    "single_confidence_distance_correlation_top5": StrategyDefinition("single_confidence_distance_correlation_top5", "single", ["confidence_distance_correlation"], {"k": 5}),
    "single_boundary_proximity_correlation_top5": StrategyDefinition("single_boundary_proximity_correlation_top5", "single", ["boundary_proximity_correlation"], {"k": 5}),
    "single_uncertainty_geometry_alignment_top5": StrategyDefinition("single_uncertainty_geometry_alignment_top5", "single", ["uncertainty_geometry_alignment"], {"k": 5}),
    "single_avg_class_separation_ratio_top5": StrategyDefinition("single_avg_class_separation_ratio_top5", "single", ["avg_class_separation_ratio"], {"k": 5}),
    "single_class_separation_uniformity_top5": StrategyDefinition("single_class_separation_uniformity_top5", "single", ["class_separation_uniformity"], {"k": 5}),
    "single_local_intrinsic_dimensionality_top5": StrategyDefinition("single_local_intrinsic_dimensionality_top5", "single", ["local_intrinsic_dimensionality"], {"k": 5}),
    "single_margin_tail_cvar_top1": StrategyDefinition("single_margin_tail_cvar_top1", "single", ["margin_tail_cvar"], {"k": 1}),
    "single_calibration_decisiveness_top1": StrategyDefinition("single_calibration_decisiveness_top1", "single", ["calibration_decisiveness"], {"k": 1}),
    "single_boundary_proximity_correlation_top1": StrategyDefinition("single_boundary_proximity_correlation_top1", "single", ["boundary_proximity_correlation"], {"k": 1}),
    "single_avg_class_separation_ratio_top1": StrategyDefinition("single_avg_class_separation_ratio_top1", "single", ["avg_class_separation_ratio"], {"k": 1}),
    "single_multiscale_separation_top1": StrategyDefinition("single_multiscale_separation_top1", "single", ["multiscale_separation"], {"k": 1}),
    "single_confidence_distance_correlation_top1": StrategyDefinition("single_confidence_distance_correlation_top1", "single", ["confidence_distance_correlation"], {"k": 1}),
    "single_uncertainty_geometry_alignment_top1": StrategyDefinition("single_uncertainty_geometry_alignment_top1", "single", ["uncertainty_geometry_alignment"], {"k": 1}),
    "single_local_intrinsic_dimensionality_top1": StrategyDefinition("single_local_intrinsic_dimensionality_top1", "single", ["local_intrinsic_dimensionality"], {"k": 1}),
    "single_ece_top1": StrategyDefinition("single_ece_top1", "single", ["ece_score"], {"k": 1}),
    "single_brier_top1": StrategyDefinition("single_brier_top1", "single", ["brier_score"], {"k": 1}),
    "single_class_separation_uniformity_top1": StrategyDefinition("single_class_separation_uniformity_top1", "single", ["class_separation_uniformity"], {"k": 1}),
    "single_margin_tail_cvar_top2": StrategyDefinition("single_margin_tail_cvar_top2", "single", ["margin_tail_cvar"], {"k": 2}),
    "single_multiscale_separation_top2": StrategyDefinition("single_multiscale_separation_top2", "single", ["multiscale_separation"], {"k": 2}),
    "single_calibration_decisiveness_top2": StrategyDefinition("single_calibration_decisiveness_top2", "single", ["calibration_decisiveness"], {"k": 2}),
    "single_boundary_proximity_correlation_top2": StrategyDefinition("single_boundary_proximity_correlation_top2", "single", ["boundary_proximity_correlation"], {"k": 2}),
    "single_avg_class_separation_ratio_top2": StrategyDefinition("single_avg_class_separation_ratio_top2", "single", ["avg_class_separation_ratio"], {"k": 2}),
    "single_local_intrinsic_dimensionality_top2": StrategyDefinition("single_local_intrinsic_dimensionality_top2", "single", ["local_intrinsic_dimensionality"], {"k": 2}),
    "single_ece_top3": StrategyDefinition("single_ece_top3", "single", ["ece_score"], {"k": 3}),
    "single_brier_top3": StrategyDefinition("single_brier_top3", "single", ["brier_score"], {"k": 3}),
    "single_margin_tail_cvar_top3": StrategyDefinition("single_margin_tail_cvar_top3", "single", ["margin_tail_cvar"], {"k": 3}),
    "single_multiscale_separation_top3": StrategyDefinition("single_multiscale_separation_top3", "single", ["multiscale_separation"], {"k": 3}),
    "single_calibration_decisiveness_top3": StrategyDefinition("single_calibration_decisiveness_top3", "single", ["calibration_decisiveness"], {"k": 3}),
    "single_confidence_distance_correlation_top3": StrategyDefinition("single_confidence_distance_correlation_top3", "single", ["confidence_distance_correlation"], {"k": 3}),
    "single_boundary_proximity_correlation_top3": StrategyDefinition("single_boundary_proximity_correlation_top3", "single", ["boundary_proximity_correlation"], {"k": 3}),
    "single_uncertainty_geometry_alignment_top3": StrategyDefinition("single_uncertainty_geometry_alignment_top3", "single", ["uncertainty_geometry_alignment"], {"k": 3}),
    "single_avg_class_separation_ratio_top3": StrategyDefinition("single_avg_class_separation_ratio_top3", "single", ["avg_class_separation_ratio"], {"k": 3}),
    "single_class_separation_uniformity_top3": StrategyDefinition("single_class_separation_uniformity_top3", "single", ["class_separation_uniformity"], {"k": 3}),
    "single_local_intrinsic_dimensionality_top3": StrategyDefinition("single_local_intrinsic_dimensionality_top3", "single", ["local_intrinsic_dimensionality"], {"k": 3}),
    "single_ece_top4": StrategyDefinition("single_ece_top4", "single", ["ece_score"], {"k": 4}),
    "single_brier_top4": StrategyDefinition("single_brier_top4", "single", ["brier_score"], {"k": 4}),
    "single_margin_tail_cvar_top4": StrategyDefinition("single_margin_tail_cvar_top4", "single", ["margin_tail_cvar"], {"k": 4}),
    "single_multiscale_separation_top4": StrategyDefinition("single_multiscale_separation_top4", "single", ["multiscale_separation"], {"k": 4}),
    "single_calibration_decisiveness_top4": StrategyDefinition("single_calibration_decisiveness_top4", "single", ["calibration_decisiveness"], {"k": 4}),
    "single_confidence_distance_correlation_top4": StrategyDefinition("single_confidence_distance_correlation_top4", "single", ["confidence_distance_correlation"], {"k": 4}),
    "single_boundary_proximity_correlation_top4": StrategyDefinition("single_boundary_proximity_correlation_top4", "single", ["boundary_proximity_correlation"], {"k": 4}),
    "single_uncertainty_geometry_alignment_top4": StrategyDefinition("single_uncertainty_geometry_alignment_top4", "single", ["uncertainty_geometry_alignment"], {"k": 4}),
    "single_avg_class_separation_ratio_top4": StrategyDefinition("single_avg_class_separation_ratio_top4", "single", ["avg_class_separation_ratio"], {"k": 4}),
    "single_class_separation_uniformity_top4": StrategyDefinition("single_class_separation_uniformity_top4", "single", ["class_separation_uniformity"], {"k": 4}),
    "single_local_intrinsic_dimensionality_top4": StrategyDefinition("single_local_intrinsic_dimensionality_top4", "single", ["local_intrinsic_dimensionality"], {"k": 4}),

    # New Pareto presets
    "pareto_all_stars": StrategyDefinition(
        "pareto_all_stars", "pareto", PARETO_ALL_STARS
    ),
    "pareto_accuracy": StrategyDefinition(
        "pareto_accuracy", "pareto", PARETO_ACCURACY_FOCUSED
    ),
    "pareto_reliability": StrategyDefinition(
        "pareto_reliability", "pareto", PARETO_RELIABILITY_FOCUSED
    ),
}


def _infer_strategy_topk(strategy: StrategyDefinition) -> Optional[int]:
    """Infer the k value for strategies that have a tiered (top-k) behavior."""
    if strategy.type == "single":
        return int(strategy.params.get("k", 1))
    if strategy.type == "voting":
        return int(strategy.params.get("k_per_metric", 1))
    import re
    match = re.search(r"_top(\d+)", strategy.name)
    if match:
        return int(match.group(1))
    return None


def find_metric_files(
    base_dir: Path, pattern: str = "**/all_metrics_per_layer.json"
) -> List[Tuple[Path, Dict]]:
    """
    Find all metric files and parse metadata from directory structure.

    Expected structure (flexible):
      .../<training_method>/<dataset>/<model>/seed<NUM>/all_metrics_per_layer.json

    Returns:
        List of (file_path, metadata_dict) tuples
    """

    logger.info(f"Searching for metric files in {base_dir}")

    files_with_metadata: List[Tuple[Path, Dict]] = []

    for file_path in base_dir.rglob("all_metrics_per_layer.json"):
        try:
            parts = file_path.parts
            # Locate 'seed<NUM>' segment
            seed_idx = None
            seed_val = None
            for idx in range(len(parts) - 1, -1, -1):
                m = re.fullmatch(r"seed(\d+)", parts[idx])
                if m:
                    seed_idx = idx
                    seed_val = int(m.group(1))
                    break
            if seed_idx is None:
                logger.debug(f"Skipping (no seed dir): {file_path}")
                continue

            # Infer model, dataset, training_method if possible
            try:
                model_dir = parts[seed_idx - 1]
                dataset_dir = parts[seed_idx - 2]
                training_method_dir = parts[seed_idx - 3]
            except IndexError:
                logger.debug(f"Insufficient path depth for {file_path}")
                continue

            metadata = {
                "training_method": training_method_dir,
                "dataset": dataset_dir,
                "model": model_dir,
                "seed": int(seed_val),
            }

            files_with_metadata.append((file_path, metadata))
        except Exception as e:
            logger.warning(f"Could not parse metadata from {file_path}: {e}")
            continue

    logger.info(f"Found {len(files_with_metadata)} valid metric files")
    return files_with_metadata


def load_and_format_metrics(file_path: Path) -> Optional[Dict[int, Dict[str, float]]]:
    """
    Load metrics file and convert to standard Dict[int, Dict[str, float]].

    Supports the following schemas:
      - {"metrics": [{"layer_idx": int, <metric>: float, ...}, ...]}
      - {"<layer_idx>": {<metric>: float, ...}, ...}
      - [{"layer_idx": int, "metrics": {<metric>: float, ...}}, ...]

    Returns None on failure.
    """

    try:
        with file_path.open("r") as f:
            data = json.load(f)

        layer_metrics: Dict[int, Dict[str, float]] = {}

        def coerce_metrics(raw: Dict) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for k, v in raw.items():
                if k == "layer_idx":
                    continue
                if isinstance(v, (int, float)):
                    out[k] = float(v)
            return out

        if isinstance(data, dict) and "metrics" in data and isinstance(data["metrics"], list):
            for item in data["metrics"]:
                if not isinstance(item, dict) or "layer_idx" not in item:
                    continue
                layer_idx = int(item["layer_idx"])
                layer_metrics[layer_idx] = coerce_metrics(item)
        elif isinstance(data, dict):
            # Possibly {"0": {...}, "1": {...}}
            for k, v in data.items():
                try:
                    layer_idx = int(k)
                except Exception:
                    continue
                if isinstance(v, dict):
                    layer_metrics[layer_idx] = coerce_metrics(v)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "layer_idx" in item:
                    layer_idx = int(item["layer_idx"])
                    if "metrics" in item and isinstance(item["metrics"], dict):
                        layer_metrics[layer_idx] = coerce_metrics(item["metrics"])
                    else:
                        layer_metrics[layer_idx] = coerce_metrics(item)

        if not layer_metrics:
            logger.warning(f"No valid metrics found in {file_path}")
            return None

        return layer_metrics
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def apply_selection_strategy(
    layer_metrics: Dict[int, Dict[str, float]], strategy: StrategyDefinition
) -> List[int]:
    """Apply a selection strategy to choose layers."""

    if strategy.type == "pareto":
        return pareto_front_selection(layer_metrics, strategy.metrics)
    if strategy.type == "voting":
        k = int(strategy.params.get("k_per_metric", 3))
        return metric_voting_selection(layer_metrics, strategy.metrics, k_per_metric=k)
    if strategy.type == "epsilon":
        eps = float(strategy.params.get("epsilon", 0.01))
        mode = str(strategy.params.get("mode", "union"))  # union|intersection
        return epsilon_ball_selection(layer_metrics, strategy.metrics, epsilon=eps, mode=mode)  # type: ignore[arg-type]
    if strategy.type == "single":
        k = int(strategy.params.get("k", 5))
        return single_metric_selection(layer_metrics, strategy.metrics[0], k=k)
    raise ValueError(f"Unknown strategy type: {strategy.type}")


def _experiment_completed(output_dir: Path) -> bool:
    """Heuristic to determine if an experiment already completed."""

    if not output_dir.exists():
        return False
    # Consider completed if sentinel exists or directory is non-empty
    sentinel = output_dir / "completed.ok"
    if sentinel.exists():
        return True
    # If any JSON artifacts exist, assume done
    if any(output_dir.glob("*.json")):
        return True
    # Non-empty directory may indicate completion
    try:
        return any(output_dir.iterdir())
    except Exception:
        return False


def generate_sbatch_command(
    config: ExperimentConfig,
    ensemble_script: Path,
    output_base_dir: Path,
    checkpoint_base_dir: Path,
    slurm_args: argparse.Namespace,
    conda_env: str,
) -> str:
    """
    Generate an sbatch command string for a single experiment.
    """

    layers_str = ",".join(map(str, config.selected_layers))
    project_root = ensemble_script.parent.parent

    # Shared cache (under seed) and strategy output dir (no weighting in path)
    shared_cache_dir = config.shared_cache_dir
    strategy_output_dir = config.output_dir

    # Build python command for the wrapped job
    python_cmd = [
        sys.executable,
        str(ensemble_script),
        "--dataset", config.dataset,
        "--model-name", config.model,
        "--seed", str(config.seed),
        "--training-method", config.training_method,
        "--results-base-dir", str(checkpoint_base_dir),
        # Specific output and shared cache paths
        "--output-dir", str(strategy_output_dir),
        "--selection-method", config.selection_strategy_name,
        "--layer_indices", layers_str,
        "--device", slurm_args.device,
        "--bins", str(slurm_args.bins),
        "--batch-size", str(slurm_args.batch_size),
        "--learned-max-iter", str(slurm_args.learned_max_iter),
        # Note: --compression-ratio is auto-selected in multi_layer_ensemble.py
        # (32x for CIFAR-100, 16x for others). Can be overridden here if needed.
    ]

    if shared_cache_dir is not None:
        python_cmd.extend(["--feat-cache", str(shared_cache_dir)])
    if config.disable_cache:
        python_cmd.append("--disable-cache")

    # Add corruption benchmark path if dataset is a -c variant
    corruption_path = None
    if config.dataset == "cifar10c":
        corruption_path = "data/cifar10-c"
    elif config.dataset == "cifar100c":
        corruption_path = "data/cifar100-c"

    if corruption_path:
        python_cmd.extend(["--corruption-dataset-path", corruption_path])

    # Expand weighting methods list
    python_cmd.append("--weighting-methods")
    all_weighting_methods = config.weighting_strategy.split(',')
    for wm in all_weighting_methods:
        python_cmd.append(wm)
    # Pass depth alphas
    if hasattr(slurm_args, 'depth_alphas') and slurm_args.depth_alphas:
        python_cmd.append("--depth-alphas")
        for alpha in slurm_args.depth_alphas:
            python_cmd.append(str(alpha))
    if slurm_args.learned_use_lbfgs:
        python_cmd.append("--learned-use-lbfgs")

    safe_python_cmd = " ".join(shlex.quote(a) for a in python_cmd)

    # Job metadata
    job_name = f"HybEns_{config.selection_strategy_name[:8]}_{config.model}_{config.dataset}_s{config.seed}"
    # Use a fixed, central log directory based on the main output directory
    log_dir = slurm_args.output_dir / "_hybrid_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"

    wrap_script = f"""
echo '========================================'
echo '🔬 SLURM JOB: Hybrid Multi-Layer Ensemble'
echo '========================================'
echo 'Job ID       : $SLURM_JOB_ID'
echo 'Host         : $(hostname)'
echo 'Start Time   : $(date)'
echo '----------------------------------------'
echo 'Config:'
echo '  Strategy   : {config.selection_strategy_name}'
echo '  Weighting  : {config.weighting_strategy}'
echo '  Method     : {config.training_method}'
echo '  Model      : {config.model}'
echo '  Dataset    : {config.dataset}'
echo '  Seed       : {config.seed}'
echo '  Layers     : {layers_str}'
echo '----------------------------------------'
module load anaconda || echo "Anaconda module not found, assuming env is active."
source activate {shlex.quote(conda_env)} || echo "Conda env '{conda_env}' activation failed."
export PYTHONPATH='{shlex.quote(str(project_root))}:$PYTHONPATH'
export CUDA_LAUNCH_BLOCKING=1
cd {shlex.quote(str(project_root))}
echo 'CMD: {safe_python_cmd}'
{safe_python_cmd}
echo '----------------------------------------'
echo 'End Time     : $(date)'
echo 'Exit Code    : $?'
echo '========================================'
"""
    clean_wrap_script = "\n".join(line.lstrip() for line in wrap_script.strip().split('\n'))

    sbatch_cmd = [
        "sbatch",
        f"--partition={shlex.quote(slurm_args.partition)}",
        f"--job-name={shlex.quote(job_name)}",
        f"--output={shlex.quote(str(log_file))}",
        f"--time={shlex.quote(slurm_args.time)}",
        "--ntasks=1",
        f"--gpus={shlex.quote(slurm_args.gpus)}",
        f"--cpus-per-task={shlex.quote(str(slurm_args.cpus))}",
        f"--mem={shlex.quote(slurm_args.mem)}",
        f"--wrap={shlex.quote(clean_wrap_script)}",
    ]

    return " ".join(sbatch_cmd)


def _build_experiments(
    metric_files: List[Tuple[Path, Dict]],
    strategies: List[StrategyDefinition],
    weighting_strategies: List[str],
    output_base: Path,
    run_on_corruptions: bool = False,
) -> List[ExperimentConfig]:
    experiments: List[ExperimentConfig] = []
    for file_path, metadata in metric_files:
        layer_metrics = load_and_format_metrics(file_path)
        if layer_metrics is None:
            continue

        target_dataset = metadata["dataset"]
        if run_on_corruptions:
            if target_dataset == "cifar10":
                target_dataset = "cifar10c"
            elif target_dataset == "cifar100":
                target_dataset = "cifar100c"

        disable_cache = run_on_corruptions
        if disable_cache:
            shared_cache_dir = None
        else:
            # Define the shared cache dir based on model/data/seed
            shared_cache_dir = (
                output_base
                / metadata["training_method"]
                / target_dataset
                / metadata["model"]
                / f"seed{metadata['seed']}"
                / "_shared_feat_cache"
            )

        for strategy in strategies:
            try:
                selected_layers = apply_selection_strategy(layer_metrics, strategy)
            except Exception as e:
                logger.error(
                    f"Failed to select layers for {strategy.name} on {metadata}: {e}"
                )
                continue

            if not selected_layers:
                logger.warning(
                    f"No layers selected for {strategy.name} on {metadata}"
                )
                continue

            # One experiment per strategy: aggregate all weightings in a single run/file
            out_dir = (
                output_base
                / metadata["training_method"]
                / target_dataset
                / metadata["model"]
                / f"seed{metadata['seed']}"
                / "experiments"
                / strategy.name
            )

            experiments.append(
                ExperimentConfig(
                    dataset=target_dataset,
                    model=metadata["model"],
                    training_method=metadata["training_method"],
                    seed=int(metadata["seed"]),
                    selection_strategy_name=strategy.name,
                    weighting_strategy=",".join(weighting_strategies),  # pass all weightings
                    shared_cache_dir=shared_cache_dir,
                    selected_layers=selected_layers,
                    metrics_file=file_path,
                    output_dir=out_dir,
                    disable_cache=disable_cache,
                )
            )

    return experiments


def _save_manifest(path: Path, manifest: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run hybrid multi-metric ensemble experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-results-dir",
        type=Path,
        default=Path("aaai_full_experiments/results/layer_selection_analysis_baselines_4"),
        help="Base directory containing metric files",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="pareto_core,voting_top3_core,single_ece",
        help="Comma-separated list of strategies to run",
    )
    parser.add_argument(
        "--weighting-strategies",
        type=str,
        default="uniform,learned,deep,shallow", # <-- Set the new default here
        help="Comma-separated list of weighting strategies (e.g., uniform,learned,shallow,deep)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hybrid_ensemble_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--ensemble-script",
        type=Path,
        default=Path("Experiments/multi_layer_ensemble.py"),
        help="Path to multi_layer_ensemble.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--checkpoint-base-dir",
        type=Path,
        default=Path("aaai_full_experiments/results/baseline"),
        help="Base directory containing original trained checkpoints (best_model.pth)",
    )
    # SLURM and pass-through args
    parser.add_argument('--results-base-dir', type=Path, required=True,
                        help='Base directory with pre-trained models (passed to ensemble script)')
    parser.add_argument('--conda-env', type=str, default="tamar_n_env", help='Conda environment name')
    parser.add_argument('--partition', type=str, default="gpu_partition", help='SLURM partition')
    parser.add_argument('--gpus', type=str, default="rtx_4090:1", help='SLURM GPU request')
    parser.add_argument('--cpus', type=int, default=4, help='SLURM CPUs per task')
    parser.add_argument('--mem', type=str, default="32G", help='SLURM memory request')
    parser.add_argument('--time', type=str, default="0-02:00:00", help='SLURM time limit (d-hh:mm:ss)')
    parser.add_argument('--device', type=str, default='cuda', help='Device for torch (cuda/cpu)')
    parser.add_argument('--bins', type=int, default=15, help='Number of ECE bins')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for ensemble script')
    parser.add_argument('--learned-use-lbfgs', action='store_true', default=True, help='Use LBFGS optimizer')
    parser.add_argument('--learned-max-iter', type=int, default=200, help='Max iterations for weight learning')
    parser.add_argument('--depth-alphas', type=float, nargs='+', default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                        help='List of alphas for deep/shallow (passed to ensemble script)')
    parser.add_argument(
        '--run-on-corruptions',
        action='store_true',
        default=False,
        help='Map clean datasets (cifar10/cifar100) to their corruption counterparts',
    )
    parser.add_argument(
        '--corruption-max-topk',
        type=int,
        default=1,
        help='When --run-on-corruptions is set, skip strategies whose top-k exceeds this limit',
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Parse strategies
    strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    strategies: List[StrategyDefinition] = []
    for name in strategy_names:
        if name not in STRATEGY_PRESETS:
            logger.error(f"Unknown strategy: {name}")
            logger.info(f"Available: {list(STRATEGY_PRESETS.keys())}")
            return 1
        strategy_def = STRATEGY_PRESETS[name]
        # if args.run_on_corruptions:
        #     topk = _infer_strategy_topk(strategy_def)
        #     if topk is not None and topk > args.corruption_max_topk:
        #         logger.info(
        #             f"Skipping strategy '{name}' for corruption run (k={topk} > limit {args.corruption_max_topk})"
        #         )
        #         continue
        strategies.append(strategy_def)

    if not strategies:
        logger.error("No valid strategies selected after applying corruption constraints.")
        return 1

    logger.info(f"Will run {len(strategies)} selection strategies: {strategy_names}")

    weighting_strategy_names = [s.strip() for s in args.weighting_strategies.split(",") if s.strip()]
    if not weighting_strategy_names:
        logger.error("No weighting strategies specified!")
        return 1
    logger.info(f"Will run {len(weighting_strategy_names)} weighting strategies: {weighting_strategy_names}")

    # Find metric files
    metric_files = find_metric_files(args.base_results_dir)
    if not metric_files:
        logger.error("No metric files found!")
        return 1

    # Build experiments
    experiments = _build_experiments(
        metric_files,
        strategies,
        weighting_strategy_names,
        args.output_dir,
        run_on_corruptions=args.run_on_corruptions,
    )
    logger.info(f"Prepared {len(experiments)} experiments")
    logger.info("Sorting experiments to group by model/dataset/seed for cache optimization...")

    def extract_top_k(strategy_name: str) -> int:
        """Extract k value from strategy name (e.g., 'single_X_top5' -> 5)."""
        import re
        match = re.search(r'_top(\d+)', strategy_name)
        if match:
            return int(match.group(1))
        # Default to 5 for strategies without explicit topk (backward compatibility)
        return 5

    # Sort by: (1) seed grouping, (2) descending k (larger k first for better cache reuse)
    experiments.sort(key=lambda e: (
        e.training_method,
        e.dataset, 
        e.model,
        e.seed,
        -extract_top_k(e.selection_strategy_name)  # Negative for descending (5,4,3,2,1)
    ))

    logger.info("Experiment order: seed-by-seed, top5→top1 within each seed (for cache efficiency)")

    if args.dry_run:
        logger.info("DRY RUN - showing first 5 experiments:")
        for exp in experiments[:5]:
            print(f"  {exp.selection_strategy_name}: {exp.dataset}/{exp.model}/seed{exp.seed}")
            print(f"    Layers: {exp.selected_layers}")
        return 0

    # Skip completed
    pending = [e for e in experiments if not _experiment_completed(e.output_dir)]
    skipped = len(experiments) - len(pending)
    if skipped:
        logger.info(f"Skipping {skipped} already-completed experiments")

    results: Dict[str, List[Dict]] = {"success": [], "failed": []}
    failures_log = args.output_dir / "failures.log"
    failures_log.parent.mkdir(parents=True, exist_ok=True)

    # Generate commands instead of running
    total = len(pending)
    if total == 0:
        logger.info("Nothing to run. Exiting.")
        return 0

    all_sbatch_commands: List[str] = []
    for exp in tqdm(pending, desc="Generating commands"):
        cmd = generate_sbatch_command(
            exp,
            args.ensemble_script,
            args.output_dir,
            args.checkpoint_base_dir,
            args,
            args.conda_env,
        )
        all_sbatch_commands.append(cmd)

    if args.dry_run:
        logger.info("DRY RUN - Printing sbatch commands:")
        for cmd in all_sbatch_commands:
            print(cmd)
            print("-" * 20)
        return 0
    else:
        logger.info(f"Submitting {len(all_sbatch_commands)} SLURM jobs...")
        submitted = 0
        failed = 0
        for cmd in tqdm(all_sbatch_commands, desc="Submitting jobs"):
            try:
                res = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                logger.debug(res.stdout.strip())
                submitted += 1
                time.sleep(0.3)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to submit job: {cmd}")
                logger.error(f"Error: {e.stderr}")
                failed += 1
            except Exception as e:
                logger.error(f"Unexpected submission error: {e}")
                failed += 1

        print("\n" + "=" * 80)
        print("SLURM SUBMISSION SUMMARY")
        print("=" * 80)
        print(f"Total commands generated : {len(all_sbatch_commands)}")
        print(f"✓ Submitted successfully : {submitted}")
        print(f"✗ Failed submissions     : {failed}")
        if failed == 0:
            print("\nUse 'squeue -u $USER' to monitor jobs.")
        else:
            print("\nCheck logs for submission errors.")
        return 0 if failed == 0 else 1

    # Save manifest
    manifest_path = args.output_dir / "experiment_manifest.json"
    _save_manifest(manifest_path, results)

    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {len(experiments)} (skipped: {skipped})")
    print(f"✓ Successful: {len(results['success'])}")
    print(f"✗ Failed: {len(results['failed'])}")
    print(f"\nManifest saved to: {manifest_path}")

    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())


