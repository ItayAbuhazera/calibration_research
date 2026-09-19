"""
DAC Comparison Experiment Runner

Launches hybrid ensemble jobs that compare DAC's predetermined layer schedule
against a curated set of metric-based selection strategies.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

# Ensure project root is on sys.path for relative imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.layer_selection_strategies import create_dac_comparison_strategies
from Experiments.run_hybrid_multi_metric_ensemble import (
    ExperimentConfig,
    generate_sbatch_command,
    load_and_format_metrics as load_layer_metrics,
    _experiment_completed,
)

from utils.logging_config import get_logger
logger = get_logger(__name__)

COMPARISON_STRATEGIES = [
    "optimal_kfold_ece",
    "optimal_margin_tail_cvar",
    "optimal_decisiveness",
    "optimal_nc1",
    "optimal_nc4",
    "top2_margin_tail_cvar",
    "top3_margin_tail_cvar",
    "top2_decisiveness",
    "top3_decisiveness",
    "psc_nc_filtered",
    "geometric_epsilon",
    "pareto_core",
    "voting_top3_core",
    "dac_predetermined",
]


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logging."""

    lvl = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _parse_list_arg(raw: Optional[str], default: List[str]) -> List[str]:
    if not raw:
        return list(default)
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items or list(default)


def _ensure_path(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dac_comparison(
    dataset: str,
    model: str,
    training_method: str,
    seed: int,
    metrics_file: Path,
    output_base_dir: Path,
    checkpoint_base_dir: Path,
    results_base_dir: Path,
    strategies_to_run: List[str],
    weighting_strategies: List[str],
    ensemble_script: Path,
    conda_env: str,
    slurm_args: argparse.Namespace,
    dry_run: bool = False,
    disable_cache: bool = False,
) -> List[ExperimentConfig]:
    """
    Build experiment configs and optionally submit SLURM jobs.
    """

    metrics_file = metrics_file.expanduser().resolve()
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    logger.info("Loading layer metrics from %s", metrics_file)
    layer_metrics = load_layer_metrics(metrics_file)
    if not layer_metrics:
        raise ValueError(f"No metrics available in {metrics_file}")

    strategies = create_dac_comparison_strategies(model_name=model)

    seed_root = (
        Path(output_base_dir).expanduser().resolve()
        / training_method
        / dataset
        / model
        / f"seed{seed}"
    )
    _ensure_path(seed_root)

    shared_cache_dir = None
    if not disable_cache:
        shared_cache_dir = seed_root / "_shared_feat_cache"
        _ensure_path(shared_cache_dir)

    weighting_csv = ",".join(weighting_strategies)
    experiments: List[ExperimentConfig] = []

    for strategy_name in strategies_to_run:
        selector = strategies.get(strategy_name)
        if selector is None:
            logger.warning("Strategy '%s' not recognized; skipping.", strategy_name)
            continue

        try:
            selected_layers = selector(layer_metrics)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Layer selection failed for %s: %s", strategy_name, exc)
            continue

        if not selected_layers:
            logger.warning("Strategy %s returned no layers; skipping.", strategy_name)
            continue

        unique_layers = list(dict.fromkeys(int(l) for l in selected_layers))
        logger.info(
            "%s → %d layers selected: %s",
            strategy_name,
            len(unique_layers),
            unique_layers,
        )

        experiments.append(
            ExperimentConfig(
                dataset=dataset,
                model=model,
                training_method=training_method,
                seed=seed,
                selection_strategy_name=strategy_name,
                weighting_strategy=weighting_csv,
                shared_cache_dir=shared_cache_dir,
                selected_layers=unique_layers,
                metrics_file=metrics_file,
                output_dir=seed_root / strategy_name,
                disable_cache=disable_cache,
            )
        )

    if not experiments:
        logger.warning("No experiments created; nothing to do.")
        return []

    pending = [exp for exp in experiments if not _experiment_completed(exp.output_dir)]
    skipped = len(experiments) - len(pending)
    if skipped:
        logger.info("Skipping %d completed experiments.", skipped)

    if not pending:
        logger.info("All requested experiments already completed.")
        return []

    all_commands: List[str] = []
    for exp in tqdm(pending, desc="Preparing commands"):
        cmd = generate_sbatch_command(
            exp,
            ensemble_script,
            slurm_args.output_dir,
            checkpoint_base_dir,
            slurm_args,
            conda_env,
        )
        all_commands.append(cmd)

    if dry_run:
        logger.info("DRY RUN: displaying %d sbatch commands.", len(all_commands))
        for cmd in all_commands:
            print(cmd)
            print("-" * 40)
        return pending

    logger.info("Submitting %d SLURM jobs...", len(all_commands))
    submitted = 0
    failures = 0
    for cmd in tqdm(all_commands, desc="Submitting jobs"):
        try:
            result = subprocess.run(
                cmd, shell=True, check=True, capture_output=True, text=True
            )
            logger.debug(result.stdout.strip())
            submitted += 1
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            logger.error("Failed to submit job:\n%s", exc.stderr)
            failures += 1
        except Exception as exc:  # pragma: no cover
            logger.error("Unexpected submission error: %s", exc)
            failures += 1

    logger.info("Submission summary: %d succeeded, %d failed.", submitted, failures)
    return pending


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DAC comparison layer selection experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core configuration
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--training-method", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--output-base-dir", type=Path, required=True)
    parser.add_argument("--results-base-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-base-dir", type=Path, default=None)

    # Strategy controls
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(COMPARISON_STRATEGIES),
        help="Comma-separated list of strategy names to run",
    )
    parser.add_argument(
        "--weighting-strategies",
        type=str,
        default="uniform,learned,deep,shallow",
        help="Comma-separated list of weighting methods passed to the ensemble script",
    )

    # SLURM / runner controls (mirroring hybrid script)
    parser.add_argument("--ensemble-script", type=Path, default=Path("Experiments/multi_layer_ensemble.py"))
    parser.add_argument("--conda-env", type=str, default="tamar_n_env")
    parser.add_argument("--partition", type=str, default="gpu_partition")
    parser.add_argument("--gpus", type=str, default="rtx_4090:1")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument("--time", type=str, default="0-02:00:00")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--depth-alphas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--learned-max-iter", type=int, default=200)
    parser.add_argument("--learned-use-lbfgs", action="store_true", default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("dac_comparison_runs"))

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()
    setup_logging(args.log_level)

    strategies_to_run = _parse_list_arg(args.strategies, COMPARISON_STRATEGIES)
    weighting_methods = _parse_list_arg(args.weighting_strategies, ["uniform", "learned", "deep", "shallow"])

    checkpoint_base_dir = args.checkpoint_base_dir or args.results_base_dir
    args.output_dir = _ensure_path(args.output_dir)

    try:
        run_dac_comparison(
            dataset=args.dataset,
            model=args.model,
            training_method=args.training_method,
            seed=args.seed,
            metrics_file=args.metrics_file,
            output_base_dir=args.output_base_dir,
            checkpoint_base_dir=checkpoint_base_dir,
            results_base_dir=args.results_base_dir,
            strategies_to_run=strategies_to_run,
            weighting_strategies=weighting_methods,
            ensemble_script=args.ensemble_script,
            conda_env=args.conda_env,
            slurm_args=args,
            dry_run=args.dry_run,
            disable_cache=args.disable_cache,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        logger.error("DAC comparison run failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())




