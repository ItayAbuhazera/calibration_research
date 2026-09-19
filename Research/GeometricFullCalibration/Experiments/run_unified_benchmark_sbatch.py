#!/usr/bin/env python3
"""
Generate (and optionally submit) SLURM jobs for unified calibration benchmark runs.

Each job wraps `Experiments/run_unified_benchmark.py` for one model/seed pair.
"""

from __future__ import annotations

import argparse
import os
import re
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
class BenchmarkJob:
    seed: int
    benchmark_script: Path
    dataset: str
    model: str
    method: str
    results_dir: Path
    output_dir: Path
    batch_size: int
    target_dimension: int
    num_layers: int
    num_coordinates: int
    device: str
    enable_post_fusion_temperature: bool
    enable_rgcl_tail_hybrids: bool
    rgcl_tail_sources: str
    debug_rgcl_tail_hybrid_smoke: bool
    debug_anchor_probs_diff: bool
    benchmark_extra_args: List[str]
    stab_metric: str = "l2"
    reuse_non_metric_from: Path | None = None
    enable_contrastive_beta_sweep: bool = True


def _normalize_benchmark_extra_args_argv(argv: List[str]) -> List[str]:
    """
    Normalize `--benchmark-extra-args VALUE` into `--benchmark-extra-args=VALUE`
    so VALUE can safely start with `--` without argparse mistaking it for an option.
    """
    normalized: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--benchmark-extra-args":
            if i + 1 >= len(argv):
                raise ValueError("--benchmark-extra-args requires a value")
            normalized.append(f"--benchmark-extra-args={argv[i + 1]}")
            i += 2
            continue
        normalized.append(token)
        i += 1
    return normalized


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create/submit SLURM jobs for unified benchmark seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=PROJECT_ROOT / "Experiments" / "run_unified_benchmark.py",
        help="Path to benchmark entry point.",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        type=str,
        help="One model architecture (backward-compatible single-model mode).",
    )
    model_group.add_argument(
        "--models",
        nargs="+",
        type=str,
        help="One or more model architectures; creates one job per model/seed pair.",
    )
    model_group.add_argument(
        "--all-models",
        action="store_true",
        help="Discover every model with checkpoints under --results-dir/--method/--dataset.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="baseline_cross_entropy",
        help="Training method used to resolve checkpoint path.",
    )
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Seeds to submit as one job each.",
    )
    seed_group.add_argument(
        "--all-available-seeds",
        action="store_true",
        help="For each model, discover and run only seeds that contain a checkpoint.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "models",
        help="Base directory used by run_unified_benchmark.py to locate trained models.",
    )
    parser.add_argument(
        "--output-template",
        type=str,
        default="results/unified_benchmark_c100_seed{seed}_rankgeom_mix",
        help=(
            "Output directory template. Must include '{seed}' and, when using more "
            "than one model, '{model}'. Also supports '{dataset}', '{method}', and "
            "'{stab_metric}'. When '{stab_metric}' is omitted, non-L2 runs are "
            "automatically suffixed with their metric so they cannot overwrite L2 runs."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-dimension", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-coordinates", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--allow-non-l2-full-rerun",
        action="store_true",
        help=(
            "Allow a non-L2 job when the matching completed L2 output directory "
            "is unavailable. By default the launcher refuses, preventing accidental "
            "recomputation of L2-only methods."
        ),
    )
    parser.add_argument(
        "--enable-post-fusion-temperature",
        action="store_true",
        help="Pass through to benchmark script.",
    )
    parser.set_defaults(enable_rgcl_tail_hybrids=True)
    rgcl_group = parser.add_mutually_exclusive_group()
    rgcl_group.add_argument(
        "--enable-rgcl-tail-hybrids",
        dest="enable_rgcl_tail_hybrids",
        action="store_true",
        help="Enable RGCL tail hybrid baselines in the benchmark script (default: enabled).",
    )
    rgcl_group.add_argument(
        "--disable-rgcl-tail-hybrids",
        dest="enable_rgcl_tail_hybrids",
        action="store_false",
        help="Disable RGCL tail hybrid baselines in the benchmark script.",
    )
    parser.add_argument(
        "--rgcl-tail-sources",
        type=str,
        default="base,temperature_scaling,vector_scaling,dirichlet",
        help="Comma-separated RGCL hybrid tail sources passed to benchmark script.",
    )
    parser.add_argument(
        "--debug-rgcl-tail-hybrid-smoke",
        action="store_true",
        help="Pass through to benchmark script.",
    )
    parser.add_argument(
        "--debug-anchor-probs-diff",
        action="store_true",
        help="Set DEBUG_ANCHOR_PROBS_DIFF=1 in the SLURM job environment.",
    )
    parser.set_defaults(enable_contrastive_beta_sweep=True)
    contrastive_group = parser.add_mutually_exclusive_group()
    contrastive_group.add_argument(
        "--enable_contrastive_beta_sweep",
        "--enable-contrastive-beta-sweep",
        dest="enable_contrastive_beta_sweep",
        action="store_true",
        help="Run the contrastive-beta experiment in submitted benchmark jobs (default: enabled).",
    )
    contrastive_group.add_argument(
        "--no_contrastive_beta_sweep",
        "--no-contrastive-beta-sweep",
        dest="enable_contrastive_beta_sweep",
        action="store_false",
        help="Disable the contrastive-beta experiment in submitted benchmark jobs.",
    )
    parser.add_argument(
        "--benchmark-extra-args",
        type=str,
        default="",
        help=(
            "Extra args appended to benchmark command (shell-split). "
            "Example: --benchmark-extra-args \"--enable_post_fusion_temperature\""
        ),
    )

    # SLURM settings
    parser.add_argument("--partition", type=str, default="gpu", help="SLURM partition.")
    parser.add_argument("--time", type=str, default="0-12:00:00", help="Time limit (d-hh:mm:ss).")
    parser.add_argument("--mem", type=str, default="60G", help="Memory per task.")
    parser.add_argument("--gpus", type=int, default=1, help="GPU count per task.")
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="rtx_4090",
        help="Optional GPU type constraint (e.g., rtx_4090). Leave empty for generic --gpus=<count>.",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="geo_cuda12",
        help="Conda environment name to activate in job.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "slurm_jobs" / "_unified_benchmark_logs",
        help="Directory for SLURM stdout logs.",
    )
    parser.add_argument(
        "--env-extra",
        type=str,
        default="",
        help="Extra shell command to run after conda activation.",
    )
    parser.add_argument(
        "--post-run-cmd",
        type=str,
        default="",
        help=(
            "Shell command to run after the benchmark completes successfully. "
            "Supports {seed}, {model}, {dataset}, {method}, and {stab_metric} "
            "placeholders, e.g. "
            "\"python Experiments/merge_kcal_into_run.py --src_dir results/.../seed{seed}/kcal_run --dst_dir results/.../seed{seed}/main_run\""
        ),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs. Without this flag only prints sbatch commands.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs and exit.")
    parser.add_argument("--sleep", type=float, default=0.3, help="Delay between submissions (seconds).")

    raw_argv = sys.argv[1:] if argv is None else argv
    normalized_argv = _normalize_benchmark_extra_args_argv(raw_argv)
    return parser.parse_args(normalized_argv)


def _parse_benchmark_extra_args(raw_value: str) -> List[str]:
    if not raw_value.strip():
        return []
    try:
        return shlex.split(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid --benchmark-extra-args: {exc}") from exc


def _stab_metric_from_extra_args(extra_args: List[str]) -> str:
    """Return argparse's effective --stab_metric value from pass-through args."""
    metric = "l2"
    i = 0
    while i < len(extra_args):
        token = extra_args[i]
        if token == "--stab_metric":
            if i + 1 >= len(extra_args):
                raise ValueError(
                    "--benchmark-extra-args contains --stab_metric without a value"
                )
            metric = extra_args[i + 1]
            i += 2
            continue
        if token.startswith("--stab_metric="):
            metric = token.split("=", 1)[1]
        i += 1

    supported = {"l2", "cosine", "whitened_cosine"}
    if metric not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(
            f"Invalid --stab_metric in --benchmark-extra-args: {metric!r}. "
            f"Expected one of: {choices}"
        )
    return metric


def _namespace_output_dir(
    output_dir: Path, output_template: str, stab_metric: str
) -> Path:
    """Keep legacy L2 paths while isolating non-L2 benchmark artifacts."""
    if stab_metric == "l2" or "{stab_metric}" in output_template:
        return output_dir
    return output_dir.with_name(f"{output_dir.name}_{stab_metric}")


def _experiment_type(method: str) -> str:
    if method.startswith("baseline_") or method in {
        "augmix",
        "augmix_constellation",
        "baseline",
    }:
        return "baseline"
    return "geometric"


def _candidate_dataset_roots(
    results_dir: Path, method: str, dataset: str
) -> List[Path]:
    """Return lightweight equivalents of the layouts understood by model_utils."""
    roots = [
        results_dir / _experiment_type(method) / method / dataset,
        results_dir / method / "baseline_cross_entropy" / dataset,
    ]
    return list(dict.fromkeys(roots))


def _checkpoint_candidates(
    results_dir: Path,
    method: str,
    dataset: str,
    model: str,
    seed: int,
) -> List[Path]:
    exp_name = f"{method}_{dataset}_{model}_seed{seed}"
    candidates: List[Path] = []
    for dataset_root in _candidate_dataset_roots(results_dir, method, dataset):
        seed_root = dataset_root / model / f"seed{seed}"
        candidates.extend(
            [
                seed_root / "best_model.pth",
                seed_root / exp_name / "best_model.pth",
            ]
        )
    return list(dict.fromkeys(candidates))


def _discover_models(results_dir: Path, method: str, dataset: str) -> List[str]:
    models = set()
    for dataset_root in _candidate_dataset_roots(results_dir, method, dataset):
        if not dataset_root.is_dir():
            continue
        for model_root in dataset_root.iterdir():
            if model_root.is_dir() and any(model_root.glob("seed*")):
                models.add(model_root.name)
    return sorted(models)


def _discover_seeds(
    results_dir: Path, method: str, dataset: str, model: str
) -> List[int]:
    seeds = set()
    for dataset_root in _candidate_dataset_roots(results_dir, method, dataset):
        model_root = dataset_root / model
        if not model_root.is_dir():
            continue
        for seed_root in model_root.glob("seed*"):
            match = re.fullmatch(r"seed(\d+)", seed_root.name)
            if not match or not seed_root.is_dir():
                continue
            seed = int(match.group(1))
            if any(
                path.is_file()
                for path in _checkpoint_candidates(
                    results_dir, method, dataset, model, seed
                )
            ):
                seeds.add(seed)
    return sorted(seeds)


def validate_args(args: argparse.Namespace) -> None:
    benchmark_script = args.benchmark_script.expanduser().resolve()
    if not benchmark_script.exists():
        raise ValueError(f"--benchmark-script does not exist: {benchmark_script}")
    if not benchmark_script.is_file():
        raise ValueError(f"--benchmark-script is not a file: {benchmark_script}")
    args.benchmark_script = benchmark_script

    if "{seed}" not in args.output_template:
        raise ValueError("--output-template must include '{seed}'")

    if not args.dataset or not str(args.dataset).strip():
        raise ValueError("--dataset is required for benchmark submission")

    results_dir = args.results_dir.expanduser()
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    args.results_dir = results_dir.resolve()

    if args.all_models:
        raw_models = _discover_models(args.results_dir, args.method, args.dataset)
        if not raw_models:
            searched = ", ".join(
                str(path)
                for path in _candidate_dataset_roots(
                    args.results_dir, args.method, args.dataset
                )
            )
            raise ValueError(
                "--all-models did not find any model directories with seed folders. "
                f"Searched: {searched}"
            )
    else:
        raw_models = [args.model] if args.model is not None else args.models
    models: List[str] = []
    for raw_model in raw_models:
        model = str(raw_model).strip()
        if not model:
            raise ValueError("Model names must not be empty")
        if model not in models:
            models.append(model)
    args.models = models

    if len(args.models) > 1 and "{model}" not in args.output_template:
        raise ValueError(
            "--output-template must include '{model}' when multiple models are requested "
            "so their outputs do not overwrite one another"
        )

    args.benchmark_extra_args_list = _parse_benchmark_extra_args(
        args.benchmark_extra_args
    )
    args.stab_metric = _stab_metric_from_extra_args(args.benchmark_extra_args_list)

    template_values = {
        "seed": args.seeds[0] if args.seeds else 0,
        "model": args.models[0],
        "dataset": args.dataset,
        "method": args.method,
        "stab_metric": args.stab_metric,
    }
    for option_name, template in (
        ("--output-template", args.output_template),
        ("--post-run-cmd", args.post_run_cmd),
    ):
        if not template:
            continue
        try:
            template.format(**template_values)
        except KeyError as exc:
            supported = ", ".join(f"{{{name}}}" for name in template_values)
            raise ValueError(
                f"{option_name} contains unsupported placeholder {exc}. "
                f"Supported placeholders: {supported}"
            ) from exc

def build_jobs(args: argparse.Namespace) -> List[BenchmarkJob]:
    jobs: List[BenchmarkJob] = []
    for model in args.models:
        model_seeds = (
            _discover_seeds(args.results_dir, args.method, args.dataset, model)
            if args.all_available_seeds
            else args.seeds
        )
        if not model_seeds:
            raise ValueError(
                "No checkpoints found for "
                f"method={args.method}, dataset={args.dataset}, model={model} "
                f"under {args.results_dir}"
            )
        for seed in model_seeds:
            template_values = {
                "seed": seed,
                "model": model,
                "dataset": args.dataset,
                "method": args.method,
                "stab_metric": args.stab_metric,
            }
            unsuffixed_output_dir = Path(
                args.output_template.format(**template_values)
            )
            output_dir = unsuffixed_output_dir
            output_dir = _namespace_output_dir(
                output_dir, args.output_template, args.stab_metric
            )
            if not output_dir.is_absolute():
                output_dir = PROJECT_ROOT / output_dir

            reuse_non_metric_from = None
            if args.stab_metric != "l2":
                l2_template_values = {**template_values, "stab_metric": "l2"}
                reuse_non_metric_from = Path(
                    args.output_template.format(**l2_template_values)
                )
                if not reuse_non_metric_from.is_absolute():
                    reuse_non_metric_from = PROJECT_ROOT / reuse_non_metric_from
                if not reuse_non_metric_from.is_dir():
                    if args.allow_non_l2_full_rerun:
                        reuse_non_metric_from = None
                    else:
                        raise ValueError(
                            "Non-L2 jobs require the matching completed L2 output "
                            "directory so invariant/L2-only methods can be reused. "
                            f"Missing: {reuse_non_metric_from}. Pass "
                            "--allow-non-l2-full-rerun to explicitly recompute them."
                        )
            jobs.append(
                BenchmarkJob(
                    seed=seed,
                    benchmark_script=args.benchmark_script,
                    dataset=args.dataset,
                    model=model,
                    method=args.method,
                    results_dir=args.results_dir,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    target_dimension=args.target_dimension,
                    num_layers=args.num_layers,
                    num_coordinates=args.num_coordinates,
                    device=args.device,
                    enable_post_fusion_temperature=args.enable_post_fusion_temperature,
                    enable_rgcl_tail_hybrids=args.enable_rgcl_tail_hybrids,
                    rgcl_tail_sources=args.rgcl_tail_sources,
                    debug_rgcl_tail_hybrid_smoke=args.debug_rgcl_tail_hybrid_smoke,
                    debug_anchor_probs_diff=args.debug_anchor_probs_diff,
                    benchmark_extra_args=args.benchmark_extra_args_list,
                    stab_metric=args.stab_metric,
                    reuse_non_metric_from=reuse_non_metric_from,
                    enable_contrastive_beta_sweep=args.enable_contrastive_beta_sweep,
                )
            )
    return jobs


def build_python_command(job: BenchmarkJob, conda_env: str = "geo_cuda12") -> List[str]:
    python_bin = str(Path(f"/home/{os.environ.get('USER','itayab')}/.conda/envs/{conda_env}/bin/python"))
    cmd = [
        python_bin,
        str(job.benchmark_script),
        "--dataset",
        job.dataset,
        "--model",
        job.model,
        "--method",
        job.method,
        "--seed",
        str(job.seed),
        "--results_dir",
        str(job.results_dir),
        "--output_dir",
        str(job.output_dir),
        "--batch_size",
        str(job.batch_size),
        "--target_dimension",
        str(job.target_dimension),
        "--num_layers",
        str(job.num_layers),
        "--num_coordinates",
        str(job.num_coordinates),
        "--device",
        job.device,
    ]
    if job.enable_post_fusion_temperature:
        cmd.append("--enable_post_fusion_temperature")
    if job.enable_rgcl_tail_hybrids:
        cmd.append("--enable_rgcl_tail_hybrids")
    if job.rgcl_tail_sources:
        cmd.extend(["--rgcl_tail_sources", job.rgcl_tail_sources])
    if job.reuse_non_metric_from is not None:
        cmd.extend(
            ["--reuse_non_metric_from", str(job.reuse_non_metric_from)]
        )
    if job.debug_rgcl_tail_hybrid_smoke:
        cmd.append("--debug_rgcl_tail_hybrid_smoke")
    if job.enable_contrastive_beta_sweep:
        cmd.append("--enable_contrastive_beta_sweep")
    else:
        cmd.append("--disable_contrastive_beta_sweep")
    if job.benchmark_extra_args:
        cmd.extend(job.benchmark_extra_args)
    return cmd


def generate_sbatch_command(job: BenchmarkJob, args: argparse.Namespace) -> str:
    python_cmd = build_python_command(job, conda_env=args.conda_env)
    safe_python_cmd = " ".join(shlex.quote(part) for part in python_cmd)

    short_model = job.model.replace("-", "")
    job_name = f"ubm_{short_model}_{job.dataset}_s{job.seed}"
    metric_tag = {"cosine": "cos", "whitened_cosine": "wcos"}.get(
        job.stab_metric
    )
    if metric_tag:
        job_name = f"{job_name}_{metric_tag}"

    log_dir = args.log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{job_name}_%j.out"

    conda_python = str(Path(f"/home/{os.environ.get('USER','itayab')}/.conda/envs/{args.conda_env}/bin/python"))
    conda_run = f"conda run -n {shlex.quote(args.conda_env)}"

    wrap_lines = [
        "echo '========================================'",
        "echo 'SLURM JOB: Unified Benchmark'",
        "echo '========================================'",
        'echo "Job ID       : $SLURM_JOB_ID"',
        'echo "Host         : $(hostname)"',
        'echo "Start Time   : $(date)"',
        "echo '----------------------------------------'",
        f"echo 'Dataset      : {job.dataset}'",
        f"echo 'Model        : {job.model}'",
        f"echo 'Seed         : {job.seed}'",
        f"echo 'Stab Metric  : {job.stab_metric}'",
        f"echo 'Reuse L2 From: {job.reuse_non_metric_from or 'none'}'",
        f"echo 'Output Dir   : {shlex.quote(str(job.output_dir))}'",
        "echo '----------------------------------------'",
        "module load anaconda || true",
        f"export PYTHONPATH='{shlex.quote(str(PROJECT_ROOT))}:${{PYTHONPATH:-}}'",
        "export CUDA_LAUNCH_BLOCKING=1",
        f"cd {shlex.quote(str(PROJECT_ROOT))}",
        f'echo "Python       : {conda_python}"',
        f'echo "PyTorch      : $({conda_run} python -c \'import torch; print(torch.__version__)\')"',
        f'echo "CUDA         : $({conda_run} python -c \'import torch; print(torch.cuda.is_available())\')"',
    ]
    if args.env_extra:
        wrap_lines.append(args.env_extra)
    if job.debug_anchor_probs_diff:
        wrap_lines.extend(
            [
                "export DEBUG_ANCHOR_PROBS_DIFF=1",
                "echo 'DEBUG_ANCHOR_PROBS_DIFF=1'",
            ]
        )
    raw_post = (
        args.post_run_cmd.strip().format(
            seed=job.seed,
            model=job.model,
            dataset=job.dataset,
            method=job.method,
            stab_metric=job.stab_metric,
        )
        if args.post_run_cmd.strip()
        else ""
    )
    # Replace bare `python` with the full conda env path (same as the main command)
    if raw_post.startswith("python "):
        post_run_cmd = raw_post.replace("python ", f"{conda_python} ", 1)
    else:
        post_run_cmd = raw_post
    post_run_lines = []
    if post_run_cmd:
        post_run_lines = [
            "echo '--- post-run step ---'",
            f"echo 'POST CMD: {post_run_cmd}'",
            f"if [ ${{exit_code}} -eq 0 ]; then {post_run_cmd}; else echo 'Skipping post-run: benchmark failed.'; fi",
        ]

    wrap_lines.extend(
        [
            "echo '----------------------------------------'",
            f"echo 'CMD: {safe_python_cmd}'",
            safe_python_cmd,
            "exit_code=$?",
            *post_run_lines,
            "echo '----------------------------------------'",
            'echo "End Time     : $(date)"',
            'echo "Exit Code    : ${exit_code}"',
            "exit ${exit_code}",
        ]
    )
    wrap_script = "\n".join(wrap_lines)

    gpu_type = args.gpu_type.strip()
    gpu_request = f"{gpu_type}:{args.gpus}" if gpu_type else str(args.gpus)

    sbatch_cmd = [
        "sbatch",
        f"--partition={shlex.quote(args.partition)}",
        f"--job-name={shlex.quote(job_name)}",
        f"--output={shlex.quote(str(log_file))}",
        f"--time={shlex.quote(args.time)}",
        "--ntasks=1",
        f"--mem={shlex.quote(args.mem)}",
        f"--gpus={shlex.quote(gpu_request)}",
        f"--wrap={shlex.quote(wrap_script)}",
    ]
    return " ".join(sbatch_cmd)


def main() -> int:
    args = parse_args()
    validate_args(args)
    jobs = build_jobs(args)

    print(f"Prepared {len(jobs)} unified benchmark jobs:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Models: {' '.join(args.models)}")
    print(f"  Stability metric: {args.stab_metric}")
    if args.all_available_seeds:
        counts_by_model = {
            model: sum(job.model == model for job in jobs) for model in args.models
        }
        print(
            "  Seeds: all available ("
            + ", ".join(
                f"{model}={count}" for model, count in counts_by_model.items()
            )
            + ")"
        )
    else:
        print(f"  Seeds: {' '.join(str(seed) for seed in args.seeds)}")
    print(f"  Output template: {args.output_template}")

    if args.dry_run:
        for job in jobs:
            python_cmd = " ".join(shlex.quote(part) for part in build_python_command(job, conda_env=args.conda_env))
            print(f"  seed {job.seed} -> {job.output_dir}")
            print(f"    python: {python_cmd}")
        return 0

    commands = [generate_sbatch_command(job, args) for job in jobs]
    if not args.submit:
        print("\nSBATCH COMMANDS (not submitted):")
        for job, cmd in zip(jobs, commands):
            python_cmd = " ".join(shlex.quote(part) for part in build_python_command(job, conda_env=args.conda_env))
            print(f"seed {job.seed} python command:")
            print(python_cmd)
            print("sbatch command:")
            print(cmd)
            print("-" * 40)
        print("Use --submit to actually send the jobs.")
        return 0

    print(f"Submitting {len(commands)} jobs...")
    submitted = 0
    failed = 0
    for cmd in commands:
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(result.stdout.strip())
            submitted += 1
            time.sleep(args.sleep)
        except subprocess.CalledProcessError as exc:
            print(f"Failed to submit: {cmd}", file=sys.stderr)
            if exc.stdout:
                print(exc.stdout, file=sys.stderr)
            if exc.stderr:
                print(exc.stderr, file=sys.stderr)
            failed += 1

    print(f"Done. Submitted: {submitted}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
