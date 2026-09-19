# Save as: Experiments/run_add_original_dim_sbatch.py

"""
SLURM Runner for add_original_dim_to_jsons.py

This script discovers all per_layer_ground_truth.json files under a given root,
groups them by (training_loss, dataset, model, seed), and submits one SLURM job
per group. Each job:

  - Requests a GPU node.
  - Activates the specified conda environment.
  - Runs add_original_dim_to_jsons.py restricted to that seed directory.

The worker script (add_original_dim_to_jsons.py) already:
  - Uses torch.cuda.is_available() to pick GPU when available.
  - Caches model + sample batch per (loss, dataset, model, seed).
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Set
import shlex

from tqdm import tqdm

# Project root: one level above Experiments/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_config import get_logger
logger = get_logger(__name__)


def find_seed_roots(layer_selection_results_path: Path) -> List[Path]:
    """
    Discover unique seed directories that contain per_layer_ground_truth.json.

    Example:
      aaai_full_experiments/results/layer_selection_analysis_baselines_4/
        baseline_brier/cifar10/densenet121/seed11/per_layer_ground_truth.json

    We return:
      [..., baseline_brier/cifar10/densenet121/seed11, ...]
    """
    seed_roots: Set[Path] = set()

    for json_path in layer_selection_results_path.rglob("per_layer_ground_truth.json"):
        seed_dir = json_path.parent
        seed_roots.add(seed_dir)

    return sorted(seed_roots)


def make_job_name(seed_root: Path) -> str:
    """
    Construct a readable SLURM job name from the seed_root path.

    Expected tail:
      .../<training_loss>/<dataset>/<model>/seedX
    """
    parts = seed_root.parts
    if len(parts) >= 4:
        training_loss = parts[-4]
        dataset = parts[-3]
        model = parts[-2]
        seed = parts[-1]
        return f"adddim_{model[:6]}_{dataset[:6]}_{training_loss[:6]}_{seed}"
    return f"adddim_{seed_root.name}"


def generate_sbatch_command(
    seed_root: Path,
    args: argparse.Namespace,
) -> str:
    """
    Build a full `sbatch --wrap "..."` command that:
      - Runs add_original_dim_to_jsons.py on this seed_root only
      - Uses GPU resources as requested in args
    """
    worker_script = PROJECT_ROOT / "Experiments" / "add_original_dim_to_jsons.py"

    # Python command that will run on the compute node
    python_cmd = [
        sys.executable,
        str(worker_script),
        "--layer_selection_results_path",
        str(seed_root),
        "--checkpoint_base_dir",
        str(args.checkpoint_base_dir),
        "--data_dir",
        str(args.data_dir),
        "--log_level",
        args.worker_log_level,
    ]
    safe_python_cmd = " ".join(shlex.quote(x) for x in python_cmd)

    job_name = make_job_name(seed_root)

    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"

    # Script executed *inside* sbatch --wrap
    wrap_script = f"""#!/bin/bash
echo "========================================"
echo "SLURM JOB: add_original_dim"
echo "Seed root   : {seed_root}"
echo "Job ID      : $SLURM_JOB_ID"
echo "Host        : $(hostname)"
echo "Start Time  : $(date)"
echo "----------------------------------------"
module load anaconda || echo "No 'module load anaconda'; assuming conda is available."
source activate {shlex.quote(args.conda_env)} || conda activate {shlex.quote(args.conda_env)} || echo "Conda env activation failed; using current env."
export PYTHONPATH={shlex.quote(str(PROJECT_ROOT))}:$PYTHONPATH
cd {shlex.quote(str(PROJECT_ROOT))}
echo "Running: {safe_python_cmd}"
{safe_python_cmd}
ret=$?
echo "----------------------------------------"
echo "End Time    : $(date)"
echo "Exit Code   : $ret"
echo "========================================"
exit $ret
"""
    # Strip leading spaces for nicer logs
    clean_wrap_script = "\n".join(
        line.lstrip() for line in wrap_script.strip().splitlines()
    )

    # Build sbatch command
    sbatch_cmd = [
        "sbatch",
        f"--partition={shlex.quote(args.partition)}",
        f"--job-name={shlex.quote(job_name)}",
        f"--output={shlex.quote(str(log_file))}",
        f"--time={shlex.quote(args.time)}",
        "--ntasks=1",
        f"--gpus={shlex.quote(args.gpus)}",
        f"--cpus-per-task={shlex.quote(str(args.cpus))}",
        f"--mem={shlex.quote(str(args.mem))}",
        f"--wrap={shlex.quote(clean_wrap_script)}",
    ]

    return " ".join(sbatch_cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs to add original_dim to per_layer_ground_truth.json files",
    )

    # Where to look / what to use
    parser.add_argument(
        "--layer_selection_results_path",
        type=Path,
        required=True,
        help="Root of layer selection results (contains per_layer_ground_truth.json files)",
    )
    parser.add_argument(
        "--checkpoint_base_dir",
        type=Path,
        required=True,
        help="Base dir containing trained model checkpoints",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./Data"),
        help="Dataset directory passed to the worker script",
    )

    # SLURM settings
    parser.add_argument("--conda_env", type=str, default="tamar_n_env")
    parser.add_argument("--partition", type=str, default="gpu_partition")
    parser.add_argument("--gpus", type=str, default="rtx_4090:1")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--mem", type=str, default="16G")
    parser.add_argument("--time", type=str, default="0-01:00:00")
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=PROJECT_ROOT / "aaai_full_experiments" / "results" / "_slurm_logs_adddim",
        help="Directory for SLURM stdout logs",
    )

    # Runner behavior
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print sbatch commands, do not submit",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level for this runner",
    )
    parser.add_argument(
        "--worker_log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level passed down to add_original_dim_to_jsons.py",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # 1. Discover seed roots
    if not args.layer_selection_results_path.exists():
        logger.error(f"layer_selection_results_path not found: {args.layer_selection_results_path}")
        sys.exit(1)

    seed_roots = find_seed_roots(args.layer_selection_results_path)
    if not seed_roots:
        logger.error(
            f"No per_layer_ground_truth.json files found under {args.layer_selection_results_path}"
        )
        sys.exit(1)

    logger.info(f"Discovered {len(seed_roots)} seed directories to process.")

    # 2. Build sbatch commands
    sbatch_cmds = []
    for seed_root in seed_roots:
        cmd = generate_sbatch_command(seed_root, args)
        sbatch_cmds.append(cmd)

    logger.info(f"Prepared {len(sbatch_cmds)} sbatch commands.")

    # 3. Execute or print
    if args.dry_run:
        logger.info("DRY RUN: Showing first few sbatch commands:")
        for cmd in sbatch_cmds[:10]:
            print(cmd)
            print("-" * 80)
        sys.exit(0)

    submitted = 0
    failed = 0
    for cmd in tqdm(sbatch_cmds, desc="Submitting jobs"):
        try:
            # Note: we capture output mainly to log job IDs if needed.
            res = subprocess.run(
                cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.debug(res.stdout.strip())
            submitted += 1
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit job:\nCMD: {cmd}\nERR: {e.stderr}")
            failed += 1

    logger.info(
        f"Submission summary: submitted={submitted}, failed={failed}, total={len(sbatch_cmds)}"
    )
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
