"""
Batch Compression Experiment Worker
Runs multiple compression experiments sequentially in a single SLURM job.

This script is called by run_compression_sweep.py when --batch_configs > 1.
It receives a JSON list of configurations and runs them one by one using
the existing compression_experiment.py worker.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_config import get_logger
logger = get_logger(__name__)


def _format_ratio_label(ratio: Optional[float]) -> str:
    """
    Format a compression ratio into the naming-friendly label used by
    compression_experiment.py for its result filenames.
    """
    if ratio is None:
        return "ratioNA"
    if abs(ratio - round(ratio)) <= 1e-9:
        return f"ratio{int(round(ratio))}"
    ratio_str = f"{ratio:.3f}".rstrip("0").rstrip(".")
    return f"ratio{ratio_str.replace('.', 'p')}"


def _load_ece_for_config(config: Dict[str, Any], output_root: Path) -> Optional[float]:
    """
    Load the calibrated ECE for a given config from the JSON written by compression_experiment.py.

    This relies on the filename convention used in run_compression_experiment():
        {layer_type}_{ratio_label}_dim{target_dim}_results.json

    Strategy:
      - If compression_ratio is provided: use it to reconstruct ratio_label and glob.
      - Else if target_dim is provided: match on dim and wildcard ratio.
      - Return the first JSON that contains an 'ece' field.
    """
    base_dir = (
        Path(output_root)
        / config["training_loss"]
        / config["dataset"]
        / config["model"]
        / f"seed{config['seed']}"
    )

    if not base_dir.exists():
        return None

    layer_type = config["layer_type"]
    ratio = config.get("compression_ratio")
    target_dim = config.get("target_dim")
    
    # Construct compression ID
    method = config.get("compression_method", "fixed_spp_jl")
    pyramid_levels = config.get("pyramid_levels", [4, 2, 1])
    mid_channels = config.get("mid_channels")
    
    is_default = (method == 'fixed_spp_jl' and pyramid_levels == [4, 2, 1] and mid_channels is None)
    
    pyramid_str = 'x'.join(map(str, pyramid_levels))
    compression_id = f"{method}_pyramid_{pyramid_str}"
    if mid_channels is not None:
        compression_id += f"_mid{mid_channels}"

    if ratio is not None:
        ratio_label = _format_ratio_label(float(ratio))
        if is_default:
            pattern = f"{layer_type}_{ratio_label}_dim*_results.json"
        else:
            pattern = f"{layer_type}_{compression_id}_{ratio_label}_dim*_results.json"
        candidates = list(base_dir.glob(pattern))
    elif target_dim is not None:
        if is_default:
            pattern = f"{layer_type}_ratio*_dim{int(target_dim)}_results.json"
        else:
            pattern = f"{layer_type}_{compression_id}_ratio*_dim{int(target_dim)}_results.json"
        candidates = list(base_dir.glob(pattern))
    else:
        # Fallback for unknown ratio/dim? Probably won't happen if config is good.
        pattern = f"{layer_type}_ratio*_dim*_results.json"
        candidates = list(base_dir.glob(pattern))

    if not candidates and is_default:
        # Only check legacy patterns if we are using default config
        # (Legacy files won't have method info)
        pass
    elif not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for path in candidates:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            ece = data.get("ece", None)
            if ece is not None:
                return float(ece)
        except Exception:
            continue

    return None


def build_single_experiment_command(
    config: Dict[str, Any],
    worker_script: Path,
    global_args: argparse.Namespace,
) -> List[str]:
    """
    Build command-line arguments for a single experiment.
    
    Args:
        config: Configuration dictionary for one experiment
        worker_script: Path to compression_experiment.py
        global_args: Global arguments (paths, batch_size, etc.)
    
    Returns:
        List of command arguments
    """
    cmd = [
        sys.executable,
        str(worker_script),
        "--model", config["model"],
        "--dataset", config["dataset"],
        "--training_loss", config["training_loss"],
        "--seed", str(config["seed"]),
        "--layer_type", config["layer_type"],
        "--layer_selection_results_path", str(global_args.layer_selection_results_path),
        "--checkpoint_base_dir", str(global_args.checkpoint_base_dir),
        "--output_dir", str(global_args.output_dir),
        "--data_dir", str(global_args.data_dir),
        "--batch_size", str(global_args.batch_size),
        "--num_workers", str(global_args.num_workers),
        "--gpu_id", str(global_args.gpu_id),
        "--min_target_dim", str(global_args.min_target_dim),
    ]
    
    # Add compression-specific args
    if config.get("compression_ratio") is not None:
        cmd.extend(["--compression_ratio", str(config["compression_ratio"])])
    
    if config.get("target_dim") is not None:
        cmd.extend(["--target_dim", str(config["target_dim"])])

    # New compression args
    if config.get("compression_method"):
        cmd.extend(["--compression_method", str(config["compression_method"])])
    if config.get("pyramid_levels"):
        cmd.append("--pyramid_levels")
        cmd.extend(map(str, config["pyramid_levels"]))
    if config.get("mid_channels") is not None:
        cmd.extend(["--mid_channels", str(config["mid_channels"])])
    
    # Determine if compression should be used
    should_use_compression = global_args.use_compression
    if config.get("compression_ratio") is not None and config["compression_ratio"] > 1.0:
        should_use_compression = True
    elif config.get("target_dim") is not None:
        should_use_compression = True
    
    if should_use_compression:
        cmd.append("--use_compression")
    
    return cmd


def format_config_summary(config: Dict[str, Any]) -> str:
    """Create a one-line summary of a config for logging."""
    compression_str = "N/A"
    if config.get("compression_ratio") is not None:
        compression_str = f"{config['compression_ratio']:.1f}×"
    elif config.get("target_dim") is not None:
        compression_str = f"dim{config['target_dim']}"
    
    method = config.get("compression_method", "fixed")
    pyr = config.get("pyramid_levels", [4,2,1])
    
    return (
        f"{config['model']}/{config['dataset']}/{config['training_loss']}/"
        f"seed{config['seed']}/{config['layer_type']}/{compression_str} "
        f"[{method}, {pyr}]"
    )


def run_single_experiment(
    config: Dict[str, Any],
    config_idx: int,
    total_configs: int,
    worker_script: Path,
    global_args: argparse.Namespace,
) -> bool:
    """
    Run a single compression experiment.
    
    Args:
        config: Configuration dictionary
        config_idx: Index of this config (1-indexed for display)
        total_configs: Total number of configs in batch
        worker_script: Path to compression_experiment.py
        global_args: Global arguments
    
    Returns:
        True if successful, False if failed
    """
    config_summary = format_config_summary(config)
    
    logger.info("=" * 80)
    logger.info(f"BATCH PROGRESS: [{config_idx}/{total_configs}]")
    logger.info(f"CONFIG: {config_summary}")
    logger.info("=" * 80)
    
    cmd = build_single_experiment_command(config, worker_script, global_args)
    
    try:
        start_time = time.time()
        
        logger.info(f"Starting experiment at {time.strftime('%H:%M:%S')}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # Run the experiment
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per experiment
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            ece = _load_ece_for_config(config, global_args.output_dir)

            if ece is not None:
                logger.info(f"✓ SUCCESS - Completed in {elapsed:.1f}s | Calibrated ECE={ece:.6f}")
            else:
                logger.info(f"✓ SUCCESS - Completed in {elapsed:.1f}s | Calibrated ECE=N/A")

            logger.debug(f"Output:\n{result.stdout}")
            return True
        else:
            logger.error(f"✗ FAILED - Exit code {result.returncode} after {elapsed:.1f}s")
            logger.error(f"STDERR:\n{result.stderr}")
            if result.stdout:
                logger.error(f"STDOUT:\n{result.stdout}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ FAILED - Timeout after 1 hour")
        return False
    except Exception as e:
        logger.error(f"✗ FAILED - Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch worker for compression experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Batch-specific args
    parser.add_argument(
        '--configs_json',
        type=str,
        required=True,
        help='JSON string containing list of experiment configurations'
    )
    
    # Global args (passed through to single experiments)
    parser.add_argument('--layer_selection_results_path', type=Path, required=True)
    parser.add_argument('--checkpoint_base_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--min_target_dim', type=int, default=16)
    parser.add_argument('--use_compression', action='store_true', default=False)
    parser.add_argument('--log_level', default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [BATCH] %(levelname)s - %(message)s",
    )
    
    # Parse configurations
    try:
        configs = json.loads(args.configs_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse configs_json: {e}")
        return 1
    
    if not configs:
        logger.error("No configurations provided")
        return 1
    
    total_configs = len(configs)
    worker_script = PROJECT_ROOT / "Experiments" / "compression_experiment.py"
    
    if not worker_script.exists():
        logger.error(f"Worker script not found: {worker_script}")
        return 1
    
    # Print batch summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("BATCH COMPRESSION EXPERIMENTS")
    logger.info("=" * 80)
    logger.info(f"Total configurations: {total_configs}")
    logger.info(f"Worker script: {worker_script}")
    logger.info(f"GPU ID: {args.gpu_id}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    logger.info("")
    
    # Run experiments sequentially
    results = []
    successful = 0
    failed = 0
    
    for idx, config in enumerate(configs, start=1):
        success = run_single_experiment(
            config=config,
            config_idx=idx,
            total_configs=total_configs,
            worker_script=worker_script,
            global_args=args,
        )
        
        results.append({
            "config": config,
            "success": success,
        })
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Brief pause between experiments
        if idx < total_configs:
            time.sleep(2)
    
    # Print final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("BATCH COMPLETION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total configs: {total_configs}")
    logger.info(f"✓ Successful:  {successful}")
    logger.info(f"✗ Failed:      {failed}")
    logger.info(f"Success rate:  {successful/total_configs*100:.1f}%")
    logger.info("=" * 80)
    
    # Print failed configs if any
    if failed > 0:
        logger.warning("")
        logger.warning("FAILED CONFIGURATIONS:")
        for result in results:
            if not result["success"]:
                logger.warning(f"  - {format_config_summary(result['config'])}")
    
    # Return success if at least some experiments completed
    # (Don't fail the entire batch job if just one config fails)
    if successful > 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())