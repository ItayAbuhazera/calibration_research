#!/usr/bin/env python3
"""
Generate (and optionally submit) SLURM jobs to train Wide-ResNet-28-10 on CIFAR-10.

Reference script: Experiments/run_hybrid_multi_metric_ensemble.py

Each job wraps `Experiments/train_wide_resnet_cifar10.py` for a single seed so we
can launch seeds 10-20 in parallel with consistent resource settings.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TrainingJob:
    python_bin: str
    seed: int
    train_script: Path
    data_root: Path
    output_root: Path
    experiment_group: str
    method: str
    dataset: str
    model_name: str
    epochs: int
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    warmup_epochs: int
    download: bool
    target_acc: float
    depth: int
    width_factor: int
    num_workers: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create/submit SLURM jobs for WRN-28-10 CIFAR-10 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=10,
        help="Inclusive start of seed range.",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=20,
        help="Inclusive end of seed range.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=PROJECT_ROOT / "Experiments" / "train_wide_resnet_cifar10.py",
        help="Path to the training entry point.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "cifar-10-batches-py",
        help="Path to CIFAR-10 data directory (or its parent).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "aaai_full_experiments" / "results",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument("--experiment-group", type=str, default="baseline")
    parser.add_argument("--method", type=str, default="baseline_cross_entropy")
    parser.add_argument(
        "--python-bin",
        type=str,
        default="python",
        help="Python executable to invoke inside the job after env activation.",
    )
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model-name", type=str, default="wide-resnet28-10")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument(
        "--target-acc-list",
        type=str,
        default="75.0,85.0,95.0",
        help="Comma-separated list of target accuracies to test",
    )
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--width-factor", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if missing.")

    # SLURM settings
    parser.add_argument("--partition", type=str, default="gpu", help="Partition / queue.")
    parser.add_argument("--gpus", type=str, default="1", help="GPU request (e.g., 1 or rtx_4090:1).")
    parser.add_argument("--cpus", type=int, default=6, help="CPUs per task.")
    parser.add_argument("--mem", type=str, default="24G", help="Memory per job.")
    parser.add_argument("--time", type=str, default="1-00:00:00", help="Wall clock limit (d-hh:mm:ss).")
    parser.add_argument("--conda-env", type=str, default="tamar_n_env", help="Conda environment to activate.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "aaai_full_experiments" / "logs" / "wrn28_cifar10",
        help="Directory for SLURM stdout logs.",
    )
    parser.add_argument("--submit", action="store_true", help="Submit sbatch commands; otherwise print only.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs and exit early.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between submissions (seconds).")
    parser.add_argument(
        "--env-extra",
        type=str,
        default="",
        help="Extra commands to execute after activating conda (e.g., 'module load cuda/12').",
    )
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> List[TrainingJob]:
    if args.seed_end < args.seed_start:
        raise ValueError("seed_end must be >= seed_start")

    target_accs = [float(x.strip()) for x in args.target_acc_list.split(",")]

    jobs: List[TrainingJob] = []
    for target_acc in target_accs:
        acc_suffix = f"acc{int(target_acc)}"
        acc_group = f"{args.experiment_group}_{acc_suffix}"
        acc_output_root = args.output_root / acc_suffix
        for seed in range(args.seed_start, args.seed_end + 1):
            jobs.append(
                TrainingJob(
                    python_bin=args.python_bin,
                    seed=seed,
                    train_script=args.train_script,
                    data_root=args.data_root,
                    output_root=acc_output_root,
                    experiment_group=acc_group,
                    method=args.method,
                    dataset=args.dataset,
                    model_name=args.model_name,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    warmup_epochs=args.warmup_epochs,
                    download=args.download,
                    target_acc=target_acc,
                    depth=args.depth,
                    width_factor=args.width_factor,
                    num_workers=args.num_workers,
                )
            )
    return jobs


def build_python_command(job: TrainingJob) -> List[str]:
    cmd = [
        job.python_bin,
        str(job.train_script),
        "--seed",
        str(job.seed),
        "--data-root",
        str(job.data_root),
        "--output-root",
        str(job.output_root),
        "--experiment-group",
        job.experiment_group,
        "--method",
        job.method,
        "--dataset",
        job.dataset,
        "--model-name",
        job.model_name,
        "--epochs",
        str(job.epochs),
        "--batch-size",
        str(job.batch_size),
        "--num-workers",
        str(job.num_workers),
        "--lr",
        str(job.lr),
        "--momentum",
        str(job.momentum),
        "--weight-decay",
        str(job.weight_decay),
        "--warmup-epochs",
        str(job.warmup_epochs),
        "--target-acc",
        str(job.target_acc),
        "--depth",
        str(job.depth),
        "--width-factor",
        str(job.width_factor),
    ]
    if job.download:
        cmd.append("--download")
    return cmd


def generate_sbatch_command(
    job: TrainingJob,
    args: argparse.Namespace,
) -> str:
    python_cmd = build_python_command(job)
    safe_python_cmd = " ".join(shlex.quote(part) for part in python_cmd)

    job_name = f"wrn28_c10_s{job.seed}"
    log_dir = args.log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"

    wrap_lines = [
        "echo '========================================'",
        f"echo 'WRN28 CIFAR10 Seed {job.seed}'",
        "echo '========================================'",
        "echo \"Job ID       : $SLURM_JOB_ID\"",
        "echo \"Host         : $(hostname)\"",
        "echo \"Start Time   : $(date)\"",
        "echo '----------------------------------------'",
        f"module load anaconda || echo \"Anaconda module not found, assuming env is active.\"",
        f"source activate {shlex.quote(args.conda_env)} || echo \"Conda env '{args.conda_env}' activation failed.\"",
        f"export PYTHONPATH='{shlex.quote(str(PROJECT_ROOT))}:$PYTHONPATH'",
        f"export CUDA_LAUNCH_BLOCKING=1",
        f"cd {shlex.quote(str(PROJECT_ROOT))}",
    ]
    if args.env_extra:
        wrap_lines.append(args.env_extra)
    wrap_lines.extend(
        [
            f"echo 'CMD: {safe_python_cmd}'",
            safe_python_cmd,
            "exit_code=$?",
            "echo '----------------------------------------'",
            "echo \"End Time     : $(date)\"",
            "echo \"Exit Code    : ${exit_code}\"",
            "exit ${exit_code}",
        ]
    )
    wrap_script = "\n".join(wrap_lines)

    sbatch_cmd = [
        "sbatch",
        f"--partition={shlex.quote(args.partition)}",
        f"--job-name={shlex.quote(job_name)}",
        f"--output={shlex.quote(str(log_file))}",
        f"--time={shlex.quote(args.time)}",
        "--ntasks=1",
        f"--cpus-per-task={args.cpus}",
        f"--mem={shlex.quote(args.mem)}",
        f"--gpus={shlex.quote(args.gpus)}",
        f"--wrap={shlex.quote(wrap_script)}",
    ]
    return " ".join(sbatch_cmd)


def main() -> int:
    args = parse_args()
    jobs = build_jobs(args)

    print(f"Prepared {len(jobs)} jobs (seeds {args.seed_start}-{args.seed_end})")
    if args.dry_run:
        for job in jobs:
            print(f"Seed {job.seed}: {job.train_script}")
        return 0

    commands = [generate_sbatch_command(job, args) for job in jobs]

    if not args.submit:
        print("\nSBATCH COMMANDS (not submitted):")
        for cmd in commands:
            print(cmd)
            print("-" * 40)
        print("Use --submit to actually send the jobs.")
        return 0

    print(f"Submitting {len(commands)} jobs...")
    submitted = 0
    failed = 0
    for cmd in commands:
        try:
            subprocess.run(cmd, shell=True, check=True)
            submitted += 1
            time.sleep(args.sleep)
        except subprocess.CalledProcessError as exc:
            print(f"Failed to submit: {cmd}", file=sys.stderr)
            print(exc, file=sys.stderr)
            failed += 1

    print(f"Done. Submitted: {submitted}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

