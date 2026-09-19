#!/usr/bin/env python3
"""
Generate (and optionally submit) SLURM jobs to train Wide-ResNet-28-10 on CIFAR-10.

Reference script: Experiments/run_hybrid_multi_metric_ensemble.py

Each job wraps `Experiments/train_wide_resnet_cifar10.py` for a single seed so we
can launch seeds 10-20 in parallel with consistent resource settings.
"""

from __future__ import annotations

import argparse
import itertools
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
    num_classes: int


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
    parser.add_argument("--datasets", nargs="+", type=str, default=["cifar10"],
                       help="List of datasets (e.g., cifar10 cifar100)")
    parser.add_argument("--models", nargs="+", type=str, default=["wide-resnet28-10"],
                       help="List of model names (e.g., wide-resnet28-10 resnet18)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--target-acc", type=float, default=96.0)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--width-factor", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if missing.")

    # SLURM settings
    parser.add_argument("--partition", type=str, default="gpu", help="Partition / queue.")
    parser.add_argument("--gpu-type", type=str, default="rtx_4090", help="GPU type (e.g., rtx_6000, rtx_4090).")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to request.")
    parser.add_argument("--cpus", type=int, default=6, help="CPUs per task.")
    parser.add_argument("--mem", type=str, default="24G", help="Memory per job.")
    parser.add_argument("--time", type=str, default="1-00:00:00", help="Wall clock limit (d-hh:mm:ss).")
    parser.add_argument("--conda-env", type=str, default="tamar_n_env", help="Conda environment to activate.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for SLURM stdout logs. Defaults to aaai_full_experiments/logs/{model}_{dataset}.",
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

    # Validate model names for DINOv2
    valid_dinov2 = ["dinov2_small", "dinov2_base", "dinov2_large", "dinov2_giant"]
    for model_name in args.models:
        if model_name.startswith("dinov2") and model_name not in valid_dinov2:
            raise ValueError(f"Invalid DINOv2 model: {model_name}. Choose from {valid_dinov2}")

    jobs: List[TrainingJob] = []
    seeds = list(range(args.seed_start, args.seed_end + 1))

    # Generate all combinations of models, datasets, and seeds
    for model_name, dataset, seed in itertools.product(args.models, args.datasets, seeds):
        # Infer the correct number of classes from the dataset if the user relies
        # on the default value (10). This prevents accidentally training a
        # CIFAR-100 or Tiny ImageNet model with a 10-class head.
        dataset_lower = dataset.lower()
        if dataset_lower == "cifar100" and args.num_classes == 10:
            inferred_num_classes = 100
        elif dataset_lower == "tiny_imagenet" and args.num_classes == 10:
            inferred_num_classes = 200
        else:
            inferred_num_classes = args.num_classes
        
        jobs.append(
            TrainingJob(
                seed=seed,
                train_script=args.train_script,
                data_root=args.data_root,
                output_root=args.output_root,
                experiment_group=args.experiment_group,
                method=args.method,
                dataset=dataset,
                model_name=model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                download=args.download,
                target_acc=args.target_acc,
                depth=args.depth,
                width_factor=args.width_factor,
                num_workers=args.num_workers,
                num_classes=inferred_num_classes,
            )
        )
    return jobs


def build_python_command(job: TrainingJob) -> List[str]:
    cmd = [
        sys.executable,
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
        "--num-classes",
        str(job.num_classes),
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

    if job.model_name == "resnet18":
        short_model = "rn18"
    elif job.model_name == "resnet50":
        short_model = "rn50"
    elif job.model_name == "densenet121":
        short_model = "dn121"
    elif job.model_name == "resnet101":
        short_model = "rn101"
    elif job.model_name == "resnet152":
        short_model = "rn152"
    elif job.model_name == "wide-resnet28-10":
        short_model = "wrn28"
    elif job.model_name.startswith("dinov2"):
        short_model = job.model_name.replace("dinov2_", "dino")
    else:
        short_model = job.model_name

    job_name = f"{short_model}_{job.dataset}_s{job.seed}"

    if args.log_dir:
        log_dir = args.log_dir.expanduser().resolve()
    else:
        log_dir = (
            PROJECT_ROOT
            / "aaai_full_experiments"
            / "logs"
            / f"{job.model_name}_{job.dataset}"
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"

    wrap_lines = [
        "echo '========================================'",
        f"echo 'Training: {job.model_name} on {job.dataset} (seed {job.seed})'",
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
        f"--gpus={shlex.quote(f'{args.gpu_type}:{args.gpus}')}",
        f"--wrap={shlex.quote(wrap_script)}",
    ]
    return " ".join(sbatch_cmd)


def main() -> int:
    args = parse_args()
    jobs = build_jobs(args)

    num_models = len(args.models)
    num_datasets = len(args.datasets)
    num_seeds = args.seed_end - args.seed_start + 1
    print(f"Prepared {len(jobs)} jobs:")
    print(f"  Models: {num_models} ({', '.join(args.models)})")
    print(f"  Datasets: {num_datasets} ({', '.join(args.datasets)})")
    print(f"  Seeds: {num_seeds} ({args.seed_start}-{args.seed_end})")
    if args.dry_run:
        for job in jobs:
            print(f"  {job.model_name} / {job.dataset} / seed {job.seed}")
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

