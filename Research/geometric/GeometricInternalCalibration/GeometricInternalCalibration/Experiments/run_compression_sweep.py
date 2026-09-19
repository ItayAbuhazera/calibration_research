MIN_TARGET_DIM_DEFAULT = 16

"""
Compression Sweep Runner
Orchestrates compression experiments across models, datasets, layers, and dimensions.

Modeled after 'run_hybrid_multi_metric_ensemble.py' to use 'sbatch --wrap'
and call a separate worker script ('compression_experiment.py').
"""

import argparse
import itertools
import json
import logging
import math
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from tqdm import tqdm
import shlex

# Ensure project root on path for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# (No internal imports needed for this runner)

from utils.logging_config import get_logger
logger = get_logger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for a single compression experiment."""
    model: str
    dataset: str
    training_loss: str
    seed: int
    layer_type: str
    compression_ratio: Optional[float] = None
    target_dim: Optional[int] = None
    # New fields
    compression_method: str = 'fixed_spp_jl'
    pyramid_levels: Optional[List[int]] = None
    mid_channels: Optional[int] = None

    def to_dict(self) -> Dict:
        return asdict(self)


COMPRESSION_CONFIGS = [
    {'method': 'fixed_spp_jl', 'pyramid_levels': [4, 2, 1]},
    {'method': 'fixed_spp_jl', 'pyramid_levels': [6, 3, 2, 1]},
    {'method': 'fixed_spp_jl', 'pyramid_levels': [8, 4, 2, 1]},
    {'method': 'channel_first_spp_jl', 'pyramid_levels': [4, 2, 1], 'mid_channels': 256},
    {'method': 'channel_first_spp_jl', 'pyramid_levels': [6, 3, 2, 1], 'mid_channels': 256},
    {'method': 'multiscale_spp_jl', 'pyramid_levels': [4, 2, 1]},
    {'method': 'multiscale_spp_jl', 'pyramid_levels': [6, 3, 2, 1]},
]

def _format_ratio_label(ratio: Optional[float]) -> str:
    if ratio is None:
        return "ratioNA"
    if math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
        return f"ratio{int(round(ratio))}"
    ratio_str = f"{ratio:.3f}".rstrip("0").rstrip(".")
    return f"ratio{ratio_str.replace('.', 'p')}"


def _load_uncalibrated_test_accuracy(
    results_root: Path,
    training_loss: str,
    dataset: str,
    model: str,
    seed: int,
) -> Optional[float]:
    """
    Try to read the uncalibrated test accuracy from baseline_results.json produced
    by the layer selection/baseline pipeline:
        {results_root}/{training_loss}/{dataset}/{model}/seed{seed}/baseline_results.json
    Returns float in [0,1] if available; otherwise None.
    """
    json_path = (
        Path(results_root)
        / training_loss
        / dataset
        / model
        / f"seed{seed}"
        / "baseline_results.json"
    )
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        # Prefer truly uncalibrated accuracy if present
        uncal = data.get("uncalibrated", {})
        acc = uncal.get("acc", None)
        if acc is not None:
            return float(acc)
        # Fallback: use any available numeric accuracy (e.g., a calibrated method)
        for v in data.values():
            if isinstance(v, dict) and v.get("acc") is not None:
                try:
                    return float(v["acc"])
                except Exception:
                    continue
    except Exception:
        return None
    return None


def load_layer_dimensions(
    layer_selection_results_path: Path,
    training_loss: str,
    dataset: str,
    model: str,
    seed: int,
) -> Dict[str, int]:
    """
    Load original dimensions for each layer from layer selection results.

    Returns:
        Dict mapping layer_name -> original_dim
    """
    json_path = (
        layer_selection_results_path
        / training_loss
        / dataset
        / model
        / f"seed{seed}"
        / "per_layer_ground_truth.json"
    )

    if not json_path.exists():
        return {}

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        layer_dims: Dict[str, int] = {}

        # Parse structure: look for original_dim in layer metrics
        for layer_id, metrics in data.items():
            if isinstance(metrics, dict):
                # Check for original_dim field
                orig_dim = metrics.get("original_dim")
                if orig_dim is not None:
                    layer_dims[str(layer_id)] = int(orig_dim)

                # Also check nested structures
                for split in ["train", "val", "test"]:
                    if split in metrics and isinstance(metrics[split], dict):
                        orig_dim = metrics[split].get("original_dim")
                        if orig_dim is not None:
                            layer_dims[str(layer_id)] = int(orig_dim)
                            break

        return layer_dims

    except Exception as e:
        logger.warning(f"Failed to load layer dimensions from {json_path}: {e}")
        return {}


def get_layer_original_dim(
    layer_selection_results_path: Path,
    training_loss: str,
    dataset: str,
    model: str,
    seed: int,
    layer_type: str,
) -> Optional[int]:
    """
    Get the original dimension for a specific layer type (best_layer / worst_layer).

    For data_layer: uses known image dims.
    For semantic layers: reads per_layer_ground_truth.json.

    Supports both:
      (A) NEW format:
          {
            "test": {
              "ece": { "5": ..., "10": ... },
              "original_dim": { "5": d5, "10": d10, ... }
            }
          }

      (B) OLD format:
          {
            "5": { "test": { "ece": ..., "original_dim": ... } },
            ...
          }
    """
    # ---- Physical layer: fixed by dataset ----
    if layer_type == "data_layer":
        ds = dataset.lower()
        if "cifar" in ds or "svhn" in ds:
            return 3 * 32 * 32
        elif "tiny_imagenet" in ds:
            return 3 * 64 * 64
        else:
            logger.warning(f"Unknown dataset for physical layer dim estimation: {dataset}")
            return None

    # ---- Semantic layers: read JSON ----
    json_path = (
        layer_selection_results_path
        / training_loss
        / dataset
        / model
        / f"seed{seed}"
        / "per_layer_ground_truth.json"
    )

    if not json_path.exists():
        logger.warning(f"Layer selection file not found: {json_path}")
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {json_path}: {e}")
        return None

    # ---------- NEW FORMAT: test.ece + test.original_dim ----------
    ece_map: Dict[str, float] = {}
    dim_map: Dict[str, int] = {}

    test_block = data.get("test", {})
    if isinstance(test_block, dict):
        ece_dict = test_block.get("ece")
        dim_dict = test_block.get("original_dim")
        if isinstance(ece_dict, dict) and isinstance(dim_dict, dict):
            for k, v in ece_dict.items():
                sk = str(k)
                if sk in dim_dict:
                    try:
                        ece_map[sk] = float(v)
                        dim_map[sk] = int(dim_dict[sk])
                    except (TypeError, ValueError):
                        continue

    # ---------- OLD FORMAT FALLBACK ----------
    if not ece_map or not dim_map:
        ece_map = {}
        dim_map = {}
        for layer_id, metrics in data.items():
            if not isinstance(metrics, dict):
                continue

            ece_val = None
            if "test" in metrics and isinstance(metrics["test"], dict):
                ece_val = metrics["test"].get("ece")
            elif "ece" in metrics:
                ece_val = metrics.get("ece")

            dim_val = None
            if "test" in metrics and isinstance(metrics["test"], dict):
                dim_val = metrics["test"].get("original_dim")
            elif "original_dim" in metrics:
                dim_val = metrics.get("original_dim")

            if ece_val is not None and dim_val is not None:
                try:
                    ece_map[str(layer_id)] = float(ece_val)
                    dim_map[str(layer_id)] = int(dim_val)
                except (TypeError, ValueError):
                    continue

    # ---------- Decide best / worst ----------
    if not ece_map or not dim_map:
        logger.warning(
            f"No overlapping ECE + original_dim entries in {json_path} "
            f"for layer_type={layer_type}"
        )
        return None

    candidates = [lid for lid in ece_map.keys() if lid in dim_map]
    if not candidates:
        logger.warning(
            f"No candidate layers with both ece and original_dim in {json_path} "
            f"for layer_type={layer_type}"
        )
        return None

    if layer_type == "best_layer":
        target_layer = min(candidates, key=lambda lid: ece_map[lid])
    elif layer_type == "worst_layer":
        target_layer = max(candidates, key=lambda lid: ece_map[lid])
    else:
        logger.warning(f"Unsupported layer_type: {layer_type}")
        return None

    return dim_map.get(target_layer)


def create_reference_baseline_configs(args) -> List[CompressionConfig]:
    """
    Generate REFERENCE baseline configurations for semantic layers.
    These use a fixed target dimension (e.g., 8192) instead of uncompressed.
    
    Physical layers still use uncompressed (ratio=1.0) baselines.
    """
    configs: List[CompressionConfig] = []
    
    models = args.models if args.models else ['resnet18', 'resnet50', 'densenet121']
    datasets = args.datasets if args.datasets else ['cifar10', 'cifar100', 'svhn']
    training_losses = args.training_losses if args.training_losses else [
        'baseline_cross_entropy', 'baseline_brier', 'baseline_focal_adaptive',
        'baseline_mmce_weighted', 'augmix'
    ]
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    
    # Physical layer: ratio=1.0 (true uncompressed)
    for model, dataset, training_loss, seed in itertools.product(
        models, datasets, training_losses, seeds
    ):
        dynamic_folder_name = f"{training_loss}_{dataset}_{model}_seed{seed}"
        model_path = (
            Path(args.checkpoint_base_dir)
            / training_loss
            / dataset
            / model
            / f"seed{seed}"
            / dynamic_folder_name
            / "best_model.pth"
        )
        if not model_path.exists():
            continue
        configs.append(
            CompressionConfig(
                model=model,
                dataset=dataset,
                training_loss=training_loss,
                seed=seed,
                layer_type='data_layer',
                compression_ratio=1.0,
                target_dim=None,
            )
        )
    
    # Semantic layers: use reference dimension rather than uncompressed
    for model, dataset, training_loss, seed in itertools.product(
        models, datasets, training_losses, seeds
    ):
        layer_results_path = (
            Path(args.layer_selection_results_path)
            / training_loss
            / dataset
            / model
            / f"seed{seed}"
            / "per_layer_ground_truth.json"
        )
        if not layer_results_path.exists():
            continue
        
        for layer_type in ['best_layer', 'worst_layer']:
            configs.append(
                CompressionConfig(
                    model=model,
                    dataset=dataset,
                    training_loss=training_loss,
                    seed=seed,
                    layer_type=layer_type,
                    compression_ratio=None,
                    target_dim=args.semantic_reference_dim,
                )
            )
    
    return configs


def create_experiment_configs(args) -> List[CompressionConfig]:
    """Generate all experiment configurations from CLI args."""
    configs = []
    # Cache model accuracies per (model,dataset,loss,seed)
    acc_cache: Dict[Tuple[str, str, str, int], Optional[float]] = {}
    layer_dim_cache: Dict[Tuple[str, str, str, int, str], Optional[int]] = {}
    ACC_THRESHOLD = 0.60
    
    models = args.models if args.models else ['resnet18', 'resnet50', 'densenet121']
    datasets = args.datasets if args.datasets else ['cifar10', 'cifar100', 'svhn']
    training_losses = args.training_losses if args.training_losses else [
        'baseline_cross_entropy', 'baseline_brier', 'baseline_focal_adaptive', 
        'baseline_mmce_weighted', 'augmix'
    ]
    layer_types = args.layer_types if args.layer_types else ['data_layer', 'best_layer', 'worst_layer']
    compression_ratios = [
        float(r) for r in (args.compression_ratios if args.compression_ratios else [1, 2, 4, 8, 16, 32])
    ]
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    
    # Generate all combinations
    if args.baselines_only:
        baseline_configs = create_reference_baseline_configs(args)
        # Apply accuracy gate, model availability, and dimension filters for generated configs
        filtered_configs: List[CompressionConfig] = []
        for config in baseline_configs:
            dynamic_folder_name = f"{config.training_loss}_{config.dataset}_{config.model}_seed{config.seed}"
            model_path = (
                Path(args.checkpoint_base_dir)
                / config.training_loss
                / config.dataset
                / config.model
                / f"seed{config.seed}"
                / dynamic_folder_name
                / "best_model.pth"
            )
            if not model_path.exists():
                logger.warning(f"Skipping config, model not found: {model_path}")
                continue
            if config.layer_type in ["best_layer", "worst_layer"]:
                layer_results_path = (
                    Path(args.layer_selection_results_path)
                    / config.training_loss
                    / config.dataset
                    / config.model
                    / f"seed{config.seed}"
                    / "per_layer_ground_truth.json"
                )
                if not layer_results_path.exists():
                    logger.warning(
                        f"Skipping {config.layer_type}, layer selection JSON not found: {layer_results_path}"
                    )
                    continue
            key = (config.model, config.dataset, config.training_loss, config.seed)
            if key not in acc_cache:
                acc_cache[key] = _load_uncalibrated_test_accuracy(
                    args.layer_selection_results_path,
                    config.training_loss,
                    config.dataset,
                    config.model,
                    config.seed,
                )
            acc_val = acc_cache[key]
            if acc_val is not None and acc_val < ACC_THRESHOLD:
                logger.info(
                    f"Skipping config due to low test accuracy ({acc_val:.4f} < {ACC_THRESHOLD:.2f}): "
                    f"{config.training_loss}/{config.dataset}/{config.model}/seed{config.seed}"
                )
                continue

            layer_key = (
                config.model,
                config.dataset,
                config.training_loss,
                config.seed,
                config.layer_type,
            )
            if layer_key not in layer_dim_cache:
                layer_dim_cache[layer_key] = get_layer_original_dim(
                    args.layer_selection_results_path,
                    config.training_loss,
                    config.dataset,
                    config.model,
                    config.seed,
                    config.layer_type,
                )
            original_dim = layer_dim_cache[layer_key]

            target_dim: Optional[int]
            if original_dim is not None:
                if config.target_dim is not None:
                    target_dim = int(config.target_dim)
                elif config.compression_ratio is not None and config.compression_ratio > 0:
                    target_dim = int(original_dim / config.compression_ratio)
                else:
                    target_dim = original_dim
            else:
                # Fall back to any explicit target_dim; otherwise skip filtering (let worker decide)
                target_dim = int(config.target_dim) if config.target_dim is not None else None
                logger.debug(
                    "Baseline target_dim unavailable; skipping max-dim filter: "
                    f"{config.training_loss}/{config.dataset}/{config.model}/seed{config.seed}/{config.layer_type}"
                )

            if target_dim is not None and target_dim > args.max_target_dim:
                logger.info(
                    f"Skipping baseline (too large): {config.layer_type} for "
                    f"{config.training_loss}/{config.dataset}/{config.model}/seed{config.seed} "
                    f"(target_dim={target_dim} > {args.max_target_dim})"
                )
                continue

            filtered_configs.append(config)

        if filtered_configs:
            logger.info(f"Generated {len(filtered_configs)} experiment configurations after filtering")
            by_layer = defaultdict(int)
            for cfg in filtered_configs:
                by_layer[cfg.layer_type] += 1
            logger.info("Configurations by layer type:")
            for layer_type, count in sorted(by_layer.items()):
                logger.info(f"  {layer_type}: {count}")
        else:
            logger.warning("No valid configurations generated after filtering!")

        return filtered_configs
    if args.baselines_only:
        baseline_configs = create_reference_baseline_configs(args)
        # Apply accuracy gate, model availability, and dimension filters for generated configs
        filtered_configs: List[CompressionConfig] = []
        for config in baseline_configs:
            # ... (existing filtering logic) ...
            # For baselines, we can leave compression settings as default or None
            # The loop below handles baselines, so I should just copy the filtering logic logic or assume
            # create_reference_baseline_configs sets them up correctly.
            # Currently create_reference_baseline_configs doesn't know about new fields, so they get defaults.
            # That is fine for baselines (no compression).
            
            # We duplicate the filtering logic here because I'm editing the block.
            # Actually, to avoid duplication and complexity, I will just modify the loop logic below for the main experiment.
            pass 

    # Main loop
    else:
        for config_tuple in itertools.product(
            models, datasets, training_losses, seeds, layer_types, compression_ratios, COMPRESSION_CONFIGS
        ):
            model, dataset, training_loss, seed, layer_type, ratio_value, comp_config = config_tuple
            
            try:
                compression_ratio = float(ratio_value)
            except (TypeError, ValueError):
                logger.warning(f"Skipping config due to invalid compression ratio: {ratio_value}")
                continue
            if compression_ratio <= 0:
                logger.warning(f"Skipping config due to non-positive compression ratio: {compression_ratio}")
                continue
        
            # Build the dynamic path to the model file
            dynamic_folder_name = f"{training_loss}_{dataset}_{model}_seed{seed}"
            model_path = (
                Path(args.checkpoint_base_dir)
                / training_loss
                / dataset
                / model
                / f"seed{seed}"
                / dynamic_folder_name
                / "best_model.pth"
            )

            if not model_path.exists():
                logger.warning(f"Skipping config, model not found: {model_path}")
                continue
            
            # Check if the layer selection results exist (needed for best/worst)
            if layer_type in ['best_layer', 'worst_layer']:
                layer_results_path = (
                    Path(args.layer_selection_results_path)
                    / training_loss / dataset / model / f"seed{seed}" / "per_layer_ground_truth.json"
                )
                if not layer_results_path.exists():
                    logger.warning(f"Skipping {layer_type}, layer selection JSON not found: {layer_results_path}")
                    continue
            
            # Accuracy gate (use cached baseline_results.json if available)
            key = (model, dataset, training_loss, seed)
            if key not in acc_cache:
                acc_cache[key] = _load_uncalibrated_test_accuracy(
                    args.layer_selection_results_path, training_loss, dataset, model, seed
                )
            acc_val = acc_cache[key]
            if acc_val is not None and acc_val < ACC_THRESHOLD:
                logger.info(
                    f"Skipping config due to low test accuracy ({acc_val:.4f} < {ACC_THRESHOLD:.2f}): "
                    f"{training_loss}/{dataset}/{model}/seed{seed}"
                )
                continue

            layer_key = (model, dataset, training_loss, seed, layer_type)
            if layer_key not in layer_dim_cache:
                layer_dim_cache[layer_key] = get_layer_original_dim(
                    args.layer_selection_results_path,
                    training_loss,
                    dataset,
                    model,
                    seed,
                    layer_type,
                )
            original_dim = layer_dim_cache[layer_key]
            if original_dim is None:
                logger.debug(
                    f"Skipping config (unknown original_dim): "
                    f"{training_loss}/{dataset}/{model}/seed{seed}/{layer_type}/ratio{compression_ratio}"
                )
                continue

            target_dim = int(original_dim / compression_ratio) if compression_ratio > 0 else original_dim

            if target_dim > args.max_target_dim:
                logger.info(
                    f"Skipping high-memory config: {training_loss}/{dataset}/{model}/seed{seed}/{layer_type} "
                    f"(ratio={compression_ratio:.1f}× → target_dim={target_dim} > {args.max_target_dim})"
                )
                continue

            configs.append(
                CompressionConfig(
                    model=model,
                    dataset=dataset,
                    training_loss=training_loss,
                    seed=seed,
                    layer_type=layer_type,
                    compression_ratio=compression_ratio,
                    compression_method=comp_config['method'],
                    pyramid_levels=comp_config['pyramid_levels'],
                    mid_channels=comp_config.get('mid_channels')
                )
            )
    
    if configs:
        logger.info(f"Generated {len(configs)} experiment configurations after filtering")
        by_layer = defaultdict(int)
        for cfg in configs:
            by_layer[cfg.layer_type] += 1
        logger.info("Configurations by layer type:")
        for layer_type, count in sorted(by_layer.items()):
            logger.info(f"  {layer_type}: {count}")
    else:
        logger.warning("No valid configurations generated after filtering!")

    return configs


def group_configs_into_batches(
    configs: List[CompressionConfig],
    batch_size: int
) -> List[List[CompressionConfig]]:
    """
    Group experiment configurations into batches for efficient execution.

    Groups configs by (model, dataset, training_loss) to maximize data reuse,
    then creates batches of the specified size.

    Args:
        configs: List of experiment configurations
        batch_size: Number of configs per batch

    Returns:
        List of config batches
    """
    if batch_size <= 1:
        return [[cfg] for cfg in configs]

    grouped = defaultdict(list)
    for cfg in configs:
        key = (cfg.model, cfg.dataset, cfg.training_loss)
        grouped[key].append(cfg)

    all_batches: List[List[CompressionConfig]] = []
    for _, group_configs in grouped.items():
        group_configs.sort(key=lambda c: (c.seed, c.layer_type, c.compression_ratio or 0.0))
        for i in range(0, len(group_configs), batch_size):
            batch = group_configs[i:i + batch_size]
            all_batches.append(batch)

    if all_batches:
        logger.info(
            f"Created {len(all_batches)} batches from {len(configs)} configs "
            f"(avg {len(configs) / len(all_batches):.1f} configs/batch)"
        )
    else:
        logger.warning("No batches created (no pending configs).")

    return all_batches


def _experiment_completed(config: CompressionConfig, output_base_dir: Path) -> bool:
    """Check if the final JSON file for this experiment already exists."""
    base_dir = (
        output_base_dir
        / config.training_loss
        / config.dataset
        / config.model
        / f"seed{config.seed}"
    )
    if not base_dir.exists():
        return False
        
    # Construct compression ID
    pyramid_levels = config.pyramid_levels if config.pyramid_levels else [4, 2, 1]
    method = config.compression_method
    mid_channels = config.mid_channels
    
    is_default = (method == 'fixed_spp_jl' and pyramid_levels == [4, 2, 1] and mid_channels is None)
    
    pyramid_str = 'x'.join(map(str, pyramid_levels))
    compression_id = f"{method}_pyramid_{pyramid_str}"
    if mid_channels is not None:
        compression_id += f"_mid{mid_channels}"

    if config.compression_ratio is not None:
        ratio_label = _format_ratio_label(config.compression_ratio)
        if is_default:
             pattern = f"{config.layer_type}_{ratio_label}_dim*_results.json"
        else:
             pattern = f"{config.layer_type}_{compression_id}_{ratio_label}_dim*_results.json"
        
        if any(base_dir.glob(pattern)):
            return True
    else:
        if config.target_dim is not None:
            if is_default:
                dim_pattern = f"{config.layer_type}_ratio*_dim{config.target_dim}_results.json"
            else:
                dim_pattern = f"{config.layer_type}_{compression_id}_ratio*_dim{config.target_dim}_results.json"
            if any(base_dir.glob(dim_pattern)):
                return True
    
    # Legacy fallback (only for default config)
    if is_default:
        legacy_pattern = f"{config.layer_type}_dim*_results.json"
        try:
            for p in base_dir.glob(legacy_pattern):
                with open(p, "r") as f:
                    data = json.load(f)
                if config.compression_ratio is not None:
                    cr = data.get("requested_compression_ratio")
                    if cr is None:
                        cr = data.get("compression_ratio")
                    if cr is None:
                        continue
                    try:
                        cr_value = float(cr)
                    except (TypeError, ValueError):
                        continue
                    if math.isclose(cr_value, config.compression_ratio, rel_tol=1e-3, abs_tol=1e-3):
                        return True
                elif config.target_dim is not None:
                    recorded_dim = data.get("target_dim")
                    if recorded_dim is None:
                        try:
                            recorded_dim = data.get("original_dim")
                        except Exception:
                            recorded_dim = None
                    if recorded_dim is not None and int(recorded_dim) == int(config.target_dim):
                        return True
        except Exception:
            return False
    return False


def generate_sbatch_command(
    config_or_batch: Union[CompressionConfig, List[CompressionConfig]],
    worker_script: Path,
    slurm_args: argparse.Namespace,
) -> str:
    """
    Generate an sbatch command string for experiment(s).

    Args:
        config_or_batch: Single config or list of configs (for batching)
        worker_script: Path to worker script (single or batch)
        slurm_args: SLURM arguments

    Returns:
        sbatch command string
    """

    is_batch = isinstance(config_or_batch, list)
    configs = config_or_batch if is_batch else [config_or_batch]
    config = configs[0]

    # --- 1. Build the Python worker command ---
    if is_batch:
        configs_json = json.dumps([cfg.to_dict() for cfg in configs])
        python_cmd = [
            sys.executable,
            str(worker_script),
            "--configs_json", configs_json,
            "--layer_selection_results_path", str(slurm_args.layer_selection_results_path),
            "--checkpoint_base_dir", str(slurm_args.checkpoint_base_dir),
            "--output_dir", str(slurm_args.output_dir),
            "--data_dir", str(slurm_args.data_dir),
            "--batch_size", str(slurm_args.batch_size),
            "--num_workers", str(slurm_args.cpus),
            "--gpu_id", "0",
            "--min_target_dim", str(slurm_args.min_target_dim),
        ]
        if getattr(slurm_args, "use_compression", False):
            python_cmd.append("--use_compression")
    else:
        python_cmd = [
            sys.executable,
            str(worker_script),
            "--model", config.model,
            "--dataset", config.dataset,
            "--training_loss", config.training_loss,
            "--seed", str(config.seed),
            "--layer_type", config.layer_type,
            "--layer_selection_results_path", str(slurm_args.layer_selection_results_path),
            "--checkpoint_base_dir", str(slurm_args.checkpoint_base_dir),
            "--output_dir", str(slurm_args.output_dir),
            "--data_dir", str(slurm_args.data_dir),
            "--batch_size", str(slurm_args.batch_size),
            "--num_workers", str(slurm_args.cpus),
            "--gpu_id", "0",
            "--min_target_dim", str(slurm_args.min_target_dim),
        ]
        if config.compression_ratio is not None:
            python_cmd.extend(["--compression_ratio", str(config.compression_ratio)])
        if config.target_dim is not None:
            python_cmd.extend(["--target_dim", str(config.target_dim)])
            
        # Add new compression args
        python_cmd.extend(["--compression_method", str(config.compression_method)])
        if config.pyramid_levels:
            python_cmd.append("--pyramid_levels")
            python_cmd.extend(map(str, config.pyramid_levels))
        if config.mid_channels is not None:
            python_cmd.extend(["--mid_channels", str(config.mid_channels)])

        should_use_compression = getattr(slurm_args, "use_compression", False)
        if config.compression_ratio is not None and config.compression_ratio > 1.0:
            should_use_compression = True
        elif config.target_dim is not None:
            should_use_compression = True
        if should_use_compression:
            python_cmd.append("--use_compression")

    safe_python_cmd = " ".join(shlex.quote(a) for a in python_cmd)

    # --- 2. Build job name and metadata ---
    if is_batch:
        batch_size = len(configs)
        ratio_label = f"batch{batch_size}"
        job_suffix = f"Batch{batch_size}"
        unique_models = len(set(c.model for c in configs))
        unique_datasets = len(set(c.dataset for c in configs))
        # Update batch config summary to indicate mixed compression methods?
        # Batch usually groups by model/dataset/loss, but might have different methods/levels if they weren't sorted out.
        # The grouping logic sorts by seed, layer, ratio. It doesn't separate by method.
        # So a batch might contain mixed methods.
        config_summary = f"{batch_size} configs ({unique_models} models, {unique_datasets} datasets)"
    else:
        if config.compression_ratio is not None:
            ratio_label = _format_ratio_label(config.compression_ratio)
        elif config.target_dim is not None:
            ratio_label = f"dim{config.target_dim}"
        else:
            ratio_label = "ratioNA"
            
        # Shorten method name for job suffix
        method_short = config.compression_method.replace('fixed_spp_jl', 'F').replace('channel_first_spp_jl', 'C').replace('multiscale_spp_jl', 'M')
        pyr_short = "".join(map(str, config.pyramid_levels)) if config.pyramid_levels else "421"
        
        job_suffix = f"{config.layer_type[:4]}_{method_short}{pyr_short}_{ratio_label}"
        config_summary = (
            f"Model: {config.model}, Dataset: {config.dataset}, Loss: {config.training_loss}, "
            f"Seed: {config.seed}, Layer: {config.layer_type}, Compression: "
            f"{'N/A' if config.compression_ratio is None else f'{config.compression_ratio:g}×'} "
            f"[{config.compression_method}, {config.pyramid_levels}]"
        )

    job_name = f"Comp_{config.model[:4]}_{config.dataset[:5]}_{config.training_loss[:4]}_s{config.seed}_{job_suffix}"

    log_dir = slurm_args.output_dir / "_slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"

    conda_env = slurm_args.conda_env

    wrap_script = f"""
echo '========================================'
echo '🔬 SLURM JOB: Compression Experiment{"s (BATCHED)" if is_batch else ""}'
echo '========================================'
echo 'Job ID       : $SLURM_JOB_ID'
echo 'Host         : $(hostname)'
echo 'Start Time   : $(date)'
echo '----------------------------------------'
echo 'Config{"s" if is_batch else ""}:'
echo '  {config_summary}'
echo '  Min Target  : {slurm_args.min_target_dim}'
echo '----------------------------------------'
module load anaconda || echo "Anaconda module not found, assuming env is active."
source activate {shlex.quote(conda_env)} || echo "Conda env '{conda_env}' activation failed."
export PYTHONPATH='{shlex.quote(str(PROJECT_ROOT))}:$PYTHONPATH'
export CUDA_LAUNCH_BLOCKING=1
cd {shlex.quote(str(PROJECT_ROOT))}
echo 'CMD: {safe_python_cmd}'
{safe_python_cmd}
echo '----------------------------------------'
echo 'End Time     : $(date)'
echo 'Exit Code    : $?'
echo '========================================'
"""
    clean_wrap_script = "\n".join(line.lstrip() for line in wrap_script.strip().split('\n'))

    time_limit = slurm_args.time
    if is_batch and len(configs) > 1:
        time_match = re.match(r'(?:(\d+)-)?(\d+):(\d+):(\d+)', time_limit)
        if time_match:
            days = int(time_match.group(1) or 0)
            hours = int(time_match.group(2))
            new_hours_total = hours * min(len(configs), 3)
            total_hours = days * 24 + new_hours_total
            new_days = total_hours // 24
            new_hours = total_hours % 24
            time_limit = f"{new_days}-{new_hours:02d}:{time_match.group(3)}:{time_match.group(4)}"

    sbatch_cmd = [
        "sbatch",
        f"--partition={shlex.quote(slurm_args.partition)}",
        f"--job-name={shlex.quote(job_name)}",
        f"--output={shlex.quote(str(log_file))}",
        f"--time={shlex.quote(time_limit)}",
        "--ntasks=1",
        f"--gpus={shlex.quote(slurm_args.gpus)}",
        f"--cpus-per-task={shlex.quote(str(slurm_args.cpus))}",
        f"--mem={shlex.quote(str(slurm_args.mem))}",
        f"--wrap={shlex.quote(clean_wrap_script)}",
    ]

    return " ".join(sbatch_cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Run compression experiments using sbatch --wrap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # --- Experiment Grid Config ---
    parser.add_argument('--models', nargs='+', default=None,
                        choices=['resnet18', 'resnet50', 'densenet121'],
                        help='Models to test (default: all)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        choices=['cifar10', 'cifar100', 'svhn', 'tiny_imagenet'],
                        help='Datasets to test (default: all)')
    parser.add_argument('--training_losses', nargs='+', default=None,
                        help='Training losses to test (default: all 5 baselines)')
    parser.add_argument('--layer_types', nargs='+', default=None,
                        choices=['data_layer', 'best_layer', 'worst_layer'],
                        help='Layer types to test (default: all)')
    parser.add_argument('--compression_ratios', nargs='+', type=float, default=None,
                        help='Compression ratios to evaluate (default: [1, 2, 4, 8, 16, 32])')
    parser.add_argument('--semantic_reference_dim', type=int, default=8192,
                        help='Reference dimension for semantic layer baselines')
    parser.add_argument('--seed_start', type=int, default=15, help='Starting seed value')
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of seeds to run')
    
    # --- Paths ---
    parser.add_argument('--layer_selection_results_path', type=Path,
                        default=Path('aaai_full_experiments/results/layer_selection_analysis_baselines_4'),
                        help='Path to *previous* layer selection results')
    parser.add_argument('--output_dir', type=Path,
                        default=Path('aaai_full_experiments/results/compression_experiments'),
                        help='Base output directory for results')
    parser.add_argument('--checkpoint_base_dir', type=Path,
                        default=Path("aaai_full_experiments/results/baseline"),
                        help="Base directory containing original trained checkpoints (checkpoint.pth.tar)")
    parser.add_argument('--worker_script', type=Path,
                        default=Path("Experiments/compression_experiment.py"),
                        help="Path to the worker script (compression_experiment.py)")
    parser.add_argument('--data_dir', type=Path, default=Path("./Data"), help="Directory for datasets")

    # --- SLURM Config (copied from your example) ---
    parser.add_argument('--conda_env', type=str, default="tamar_n_env", help='Conda environment name')
    parser.add_argument('--partition', type=str, default="gpu_partition", help='SLURM partition')
    parser.add_argument('--gpus', type=str, default="rtx_4090:1", help='SLURM GPU request')
    parser.add_argument('--cpus', type=int, default=4, help='SLURM CPUs per task')
    parser.add_argument('--mem', type=str, default="32G", help='SLURM memory request')
    parser.add_argument('--time', type=str, default="0-01:00:00", help='SLURM time limit (d-hh:mm:ss), 1 hour')

    # --- Worker Script Args (passed through) ---
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for feature extraction')
    parser.add_argument('--use_compression', action='store_true', default=False,
                        help='If set, the worker will compress (SPP+JL). Leave unset for baselines.')
    parser.add_argument('--baselines_only', action='store_true', default=False,
                        help='Generate and run only uncompressed baselines for all specified layer types.')
    parser.add_argument('--min_target_dim', type=int, default=MIN_TARGET_DIM_DEFAULT,
                        help='Minimum allowable target dimension when using compression ratios.')
    parser.add_argument('--max_target_dim', type=int, default=12000,
                        help='Skip experiments where target_dim would exceed this value (to avoid OOM)')
    parser.add_argument('--batch_configs', type=int, default=1,
                        help='Number of configurations to batch per SLURM job (default: 1 = no batching). '
                             'Recommended: 8-15 for faster completion with job overhead amortization.')
    
    # --- Runner Control ---
    parser.add_argument('--dry_run', action='store_true',
                        help='Generate and print commands but do not submit jobs')
    parser.add_argument('--overwrite', action='store_true',
                        help='Run experiments even if output file already exists')
    parser.add_argument('--log_level', default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # --- Setup ---
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Build Experiment Configs ---
    logger.info("Generating experiment configurations...")
    experiments = create_experiment_configs(args)
    logger.info(f"Generated {len(experiments)} total experiment configurations.")

    # --- 2. Filter Completed ---
    if args.overwrite:
        pending = experiments
        logger.info("Running all experiments (overwrite=True).")
    else:
        pending = [e for e in experiments if not _experiment_completed(e, args.output_dir)]
        skipped = len(experiments) - len(pending)
        if skipped:
            logger.info(f"Skipping {skipped} already-completed experiments.")
    
    if not pending:
        logger.info("No pending experiments to run. Exiting.")
        return 0

    # --- 2.5. Create Batches ---
    if args.batch_configs > 1:
        batches = group_configs_into_batches(pending, args.batch_configs)
        worker_script = PROJECT_ROOT / "Experiments" / "compression_experiment_batch.py"
        if not worker_script.exists():
            logger.error(f"Batch worker script not found: {worker_script}")
            logger.error("Please ensure compression_experiment_batch.py exists")
            return 1
        logger.info(f"Batching enabled: {len(pending)} configs → {len(batches)} batched jobs")
    else:
        batches = [[cfg] for cfg in pending]
        worker_script = args.worker_script
        logger.info(f"No batching: {len(pending)} individual jobs")

    # --- 3. Generate SLURM Commands ---
    logger.info(f"Generating {len(batches)} sbatch commands...")
    all_sbatch_commands = []
    for batch in tqdm(batches, desc="Generating commands"):
        cmd = generate_sbatch_command(
            batch if len(batch) > 1 else batch[0],
            worker_script,
            args
        )
        all_sbatch_commands.append(cmd)

    # --- 4. Execute or Dry Run ---
    if args.dry_run:
        logger.info(f"DRY RUN - Printing first 5 (of {len(all_sbatch_commands)}) sbatch commands:")
        for cmd in all_sbatch_commands[:5]:
            print(cmd)
            print("-" * 20)
        return 0
    
    logger.info(f"Submitting {len(all_sbatch_commands)} SLURM jobs...")
    submitted = 0
    failed = 0
    for cmd in tqdm(all_sbatch_commands, desc="Submitting jobs"):
        try:
            res = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            logger.debug(res.stdout.strip())
            submitted += 1
            time.sleep(0.1) # Be nice to the scheduler
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
    print(f"✗ Failed submissions     : {failed}")
    if failed == 0:
        print("\nUse 'squeue -u $USER' to monitor jobs.")
    else:
        print("\nCheck logs for submission errors.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())