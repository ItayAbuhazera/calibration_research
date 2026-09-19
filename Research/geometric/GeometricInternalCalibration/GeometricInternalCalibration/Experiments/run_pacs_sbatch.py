#!/usr/bin/env python3
"""
Generate (and optionally submit) SLURM jobs to train ResNet models on PACS with
domain adaptation (train on 3 domains, test on 1).

Each job wraps `Experiments/train_pacs_domain_adaptation.py` for a single
combination of (target_domain, model, method, seed).
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
    seed: int
    target_domain: str
    method: str
    model_name: str
    train_script: Path
    output_root: Path
    epochs: int
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    warmup_epochs: int
    num_workers: int
    num_classes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create/submit SLURM jobs for PACS domain adaptation training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="Inclusive start of seed range.",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=3,
        help="Inclusive end of seed range.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=PROJECT_ROOT / "Experiments" / "train_pacs_domain_adaptation.py",
        help="Path to the training entry point.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "aaai_full_experiments" / "results",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--target-domains",
        type=str,
        default="photo,art_painting,cartoon,sketch",
        help="Comma-separated list of target domains.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="resnet18,resnet50",
        help="Comma-separated list of model names.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="cross_entropy,brier,augmix",
        help="Comma-separated list of methods.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=7)

    # SLURM settings
    parser.add_argument("--partition", type=str, default="gpu", help="Partition / queue.")
    parser.add_argument(
        "--gpus",
        type=str,
        default="1",
        help="GPU request (e.g., 1 or rtx_4090:1).",
    )
    parser.add_argument("--cpus", type=int, default=4, help="CPUs per task.")
    parser.add_argument(
        "--mem",
        type=str,
        default="16G",
        help="Memory per job.",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="6:00:00",
        help="Wall clock limit (hh:mm:ss or d-hh:mm:ss).",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="tamar_n_env",
        help="Conda environment to activate.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "Directory for SLURM stdout logs. "
            "Defaults to aaai_full_experiments/logs/pacs_domain_adaptation."
        ),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit sbatch commands; otherwise print only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs and exit early.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Delay between submissions (seconds).",
    )
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

    seeds = list(range(args.seed_start, args.seed_end + 1))
    target_domains = [d.strip() for d in args.target_domains.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    jobs: List[TrainingJob] = []
    for target in target_domains:
        for model_name in models:
            for method in methods:
                for seed in seeds:
                    jobs.append(
                        TrainingJob(
                            seed=seed,
                            target_domain=target,
                            method=method,
                            model_name=model_name,
                            train_script=args.train_script,
                            output_root=args.output_root,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            warmup_epochs=args.warmup_epochs,
                            num_workers=args.num_workers,
                            num_classes=args.num_classes,
                        )
                    )
    return jobs


def build_python_command(job: TrainingJob) -> List[str]:
    cmd = [
        sys.executable,
        str(job.train_script),
        "--seed",
        str(job.seed),
        "--output-root",
        str(job.output_root),
        "--method",
        job.method,
        "--dataset",
        "pacs",
        "--target-domain",
        job.target_domain,
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
        "--num-classes",
        str(job.num_classes),
    ]
    return cmd


def generate_sbatch_command(
    job: TrainingJob,
    args: argparse.Namespace,
) -> str:
    python_cmd = build_python_command(job)
    safe_python_cmd = " ".join(shlex.quote(part) for part in python_cmd)

    # Short model/method identifiers for job names
    if job.model_name == "resnet18":
        short_model = "rn18"
    elif job.model_name == "resnet50":
        short_model = "rn50"
    else:
        short_model = job.model_name

    if job.method == "cross_entropy":
        short_method = "ce"
    else:
        short_method = job.method

    job_name = f"{short_model}_{short_method}_{job.target_domain}_s{job.seed}"

    if args.log_dir:
        log_dir = args.log_dir.expanduser().resolve()
    else:
        log_dir = (
            PROJECT_ROOT
            / "aaai_full_experiments"
            / "logs"
            / "pacs_domain_adaptation"
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"

    wrap_lines = [
        "echo '========================================'",
        f"echo 'PACS Domain Adaptation: {job.model_name}, method={job.method}, target={job.target_domain}, seed={job.seed}'",
        "echo '========================================'",
        "echo \"Job ID       : $SLURM_JOB_ID\"",
        "echo \"Host         : $(hostname)\"",
        "echo \"Start Time   : $(date)\"",
        "echo '----------------------------------------'",
        "module load anaconda || echo \"Anaconda module not found, assuming env is active.\"",
        f"source activate {shlex.quote(args.conda_env)} || echo \"Conda env '{args.conda_env}' activation failed.\"",
        f"export PYTHONPATH='{shlex.quote(str(PROJECT_ROOT))}:$PYTHONPATH'",
        "export CUDA_LAUNCH_BLOCKING=1",
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

    seeds = list(range(args.seed_start, args.seed_end + 1))
    target_domains = [d.strip() for d in args.target_domains.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(
        f"Prepared {len(jobs)} jobs "
        f"(domains={len(target_domains)}, models={len(models)}, "
        f"methods={len(methods)}, seeds={len(seeds)})"
    )

    # Breakdown by (model, method, domain)
    breakdown = {}
    for job in jobs:
        key = (job.model_name, job.method, job.target_domain)
        breakdown[key] = breakdown.get(key, 0) + 1

    print("Job breakdown (model, method, target_domain) -> count:")
    for (model_name, method, domain), count in sorted(breakdown.items()):
        print(f"  {model_name:10s} {method:12s} {domain:12s} : {count}")

    if args.dry_run and not args.submit:
        print("\nDry run enabled; not generating sbatch commands.")
        return 0

    commands = [generate_sbatch_command(job, args) for job in jobs]

    if not args.submit:
        print("\nSBATCH COMMANDS (not submitted):")
        for cmd in commands:
            print(cmd)
            print("-" * 40)
        print("Use --submit to actually send the jobs.")
        return 0

    print(f"\nSubmitting {len(commands)} jobs...")
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


