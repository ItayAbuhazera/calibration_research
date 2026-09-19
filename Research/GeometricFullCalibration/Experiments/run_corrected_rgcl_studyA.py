"""
Corrected internal-only RGCL, Study A (BENCHMARK_IMPLEMENTATION_PLAN.md §4/§8b).

For one checkpoint, computes:
  - the three primary corrected draws (D1=9101, D2=9102, D3=9103, L=6 --
    matching Published RGCL's default, per §4's "apples-to-apples Study-A
    comparison" requirement; this is a *different* L from Study B's L=5
    count-matched-to-DAC draws, even though the seeds are shared).
  - the "legacy/auxiliary" own-checkpoint-seed draw (L=6): for checkpoint
    seed=1 specifically (the only historically fc-contaminated seed, per
    artifacts/recoverability/seed1/preflight_complete.json), this is a FRESH
    corrected draw (select_random_rgc_layers_corrected); for any other
    checkpoint it reuses the ORIGINAL, uncorrected select_random_rgc_layers
    with that checkpoint's own seed, since those draws are already confirmed
    fc-free and are retained as legacy/auxiliary evidence rather than
    recomputed (recomputing them via the corrected function would silently
    change which layers get drawn, since removing fc from the candidate pool
    shifts every subsequent np.random.choice index -- see plan §16).

Each draw's projected 256-d RGCL vector (SPP + fixed Gaussian projection,
reusing the exact same combined-hook extraction pipeline built for Study B --
Experiments/extract_unified_studyAB_representations.py, including its
layer_names=[] workaround) is fit+calibrated via GeometricCalibrator, exactly
mirroring Experiments/run_rgc_experiments.py::run_global_random_calibration's
own fit/calibrate/metrics pattern (same class, same isotonic mapper) -- reused
via GeometricCalibrator directly rather than reimplemented.

Scope note: this script computes the above for ONE checkpoint only (per the
frozen plan's own Phase 1 budget -- "smoke-tested on one checkpoint only").
The primary D1/D2/D3 x 5-checkpoint crossed design (Study A's headline
evidence, plus the checkpoint x draw variance decomposition) is Phase 2 scope
per §2 and is NOT computed by running this script once per checkpoint --
running it for the remaining 4 checkpoints is Phase 2 work, not implied by
running this script on checkpoint 1.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from Calibrators.geometric_calibrator import GeometricCalibrator
from Experiments.extract_unified_studyAB_representations import extract_combined_representations
from Experiments.run_rgc_experiments import (
    calculate_accuracy,
    calculate_adaptive_ece,
    calculate_brier_score,
    calculate_ece,
    select_random_rgc_layers,
    select_random_rgc_layers_corrected,
)
from Experiments.studyAB_artifact_schema import build_artifact_row, config_hash
from utils.calibration_utils import get_all_data_as_numpy
from utils.model_utils import PyTorchModelAdapter, get_data_loaders, load_trained_model

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_DRAW_SEEDS = (9101, 9102, 9103)
HISTORICALLY_CONTAMINATED_SEEDS = {1}  # confirmed via preflight_complete.json


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def _git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip())


def resolve_studyA_draws(model, model_name, dataset_name, device, checkpoint_seed, num_layers=6):
    draw_layers: Dict[int, list] = {}
    for seed in PRIMARY_DRAW_SEEDS:
        layers = select_random_rgc_layers_corrected(
            model=model, model_name=model_name, dataset_name=dataset_name,
            device=device, num_layers=num_layers, seed=seed,
        )
        assert not any(n == "fc" or n.startswith("fc.") for n in layers), (
            f"fc leaked into primary draw seed={seed} -- must never happen"
        )
        draw_layers[seed] = layers

    if checkpoint_seed in HISTORICALLY_CONTAMINATED_SEEDS:
        legacy_layers = select_random_rgc_layers_corrected(
            model=model, model_name=model_name, dataset_name=dataset_name,
            device=device, num_layers=num_layers, seed=checkpoint_seed,
        )
        legacy_source = "corrected_fresh_substitute"
    else:
        legacy_layers = select_random_rgc_layers(
            model=model, model_name=model_name, dataset_name=dataset_name,
            device=device, num_layers=num_layers, seed=checkpoint_seed,
        )
        legacy_source = "historical_reused_confirmed_fc_free"
    assert not any(n == "fc" or n.startswith("fc.") for n in legacy_layers), (
        f"fc leaked into legacy draw for checkpoint_seed={checkpoint_seed} -- must never happen"
    )
    draw_layers[checkpoint_seed] = legacy_layers

    return draw_layers, legacy_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--model", default="resnet101")
    parser.add_argument("--checkpoint", default="seed1")
    parser.add_argument("--checkpoint_seed", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--target_dimension", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument(
        "--max_samples_per_split", type=int, default=None,
        help="Truncate each split after materializing (dry-run/testing only; "
        "omit for the real scientific run, which uses the full splits).",
    )
    args = parser.parse_args()

    t_start = time.time()
    commit = _git_commit()
    dirty = _git_dirty()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    num_classes = 100 if "cifar100" in args.dataset.lower() else 10
    checkpoint_path = (
        REPO_ROOT.parent / "geometric" / "GeometricInternalCalibration" / "GeometricInternalCalibration"
        / "aaai_full_experiments" / "results" / "baseline" / "baseline_cross_entropy"
        / args.dataset / args.model / args.checkpoint / "best_model.pth"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_trained_model(str(checkpoint_path), args.model, num_classes, device, dataset=args.dataset)
    model.eval()
    model_adapter = PyTorchModelAdapter(model, device, args.dataset)

    train_loader, val_loader, test_loader, loader_num_classes = get_data_loaders(
        args.dataset, args.batch_size, seed=args.data_seed
    )
    assert loader_num_classes == num_classes

    t_data0 = time.time()
    train_raw, train_labels = get_all_data_as_numpy(train_loader)
    val_raw, val_labels = get_all_data_as_numpy(val_loader)
    test_raw, test_labels = get_all_data_as_numpy(test_loader)
    if args.max_samples_per_split is not None:
        n = args.max_samples_per_split
        train_raw, train_labels = train_raw[:n], train_labels[:n]
        val_raw, val_labels = val_raw[:n], val_labels[:n]
        test_raw, test_labels = test_raw[:n], test_labels[:n]
    t_data1 = time.time()

    draw_layers, legacy_source = resolve_studyA_draws(
        model=model, model_name=args.model, dataset_name=args.dataset,
        device=device, checkpoint_seed=args.checkpoint_seed, num_layers=args.num_layers,
    )
    combined_layers = list(dict.fromkeys(n for names in draw_layers.values() for n in names))

    def make_loader(raw, labels):
        ds = torch.utils.data.TensorDataset(torch.from_numpy(raw), torch.from_numpy(labels))
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    t_extract0 = time.time()
    train_result = extract_combined_representations(
        model=model, loader=make_loader(train_raw, train_labels), split_name="train",
        combined_layers=combined_layers, draw_layers=draw_layers, device=device,
        target_dim=args.target_dimension, pooling_mode="max",
    )
    val_result = extract_combined_representations(
        model=model, loader=make_loader(val_raw, val_labels), split_name="val",
        combined_layers=combined_layers, draw_layers=draw_layers, device=device,
        target_dim=args.target_dimension, pooling_mode="max",
    )
    test_result = extract_combined_representations(
        model=model, loader=make_loader(test_raw, test_labels), split_name="test",
        combined_layers=combined_layers, draw_layers=draw_layers, device=device,
        target_dim=args.target_dimension, pooling_mode="max",
    )
    t_extract1 = time.time()
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None

    base_config = {
        "dataset": args.dataset, "model": args.model, "checkpoint": args.checkpoint,
        "checkpoint_seed": args.checkpoint_seed, "primary_draw_seeds": list(PRIMARY_DRAW_SEEDS),
        "num_layers": args.num_layers, "target_dimension": args.target_dimension,
        "data_seed": args.data_seed,
    }
    cfg_hash = config_hash(base_config)
    provenance = {
        "git_commit": commit, "git_dirty": dirty, "config_hash": cfg_hash,
        "plan_version": "BENCHMARK_IMPLEMENTATION_PLAN.md",
        "extraction_script": "Experiments/extract_unified_studyAB_representations.py",
        "script": "Experiments/run_corrected_rgcl_studyA.py",
    }

    t_fit0 = time.time()
    rows = []
    draw_metadata = {}
    for seed, layers in draw_layers.items():
        is_legacy = seed == args.checkpoint_seed
        train_proj = train_result["draw_projected_features"][seed]
        val_proj = val_result["draw_projected_features"][seed]
        test_proj = test_result["draw_projected_features"][seed]

        geo_cal = GeometricCalibrator(
            model=model_adapter, X_train_embed=train_proj, y_train=train_labels,
            library="fast_separation", auto_select_layer=False, device=str(device),
            scoring_method="separation",
        )
        geo_cal.fit(X_val_embed=val_proj, y_val=val_labels, X_val_original=val_raw, fit_batch_size=args.batch_size)
        model_probs_test = model_adapter.predict_proba(test_raw, batch_size=args.batch_size)
        calibrated_probs = geo_cal.calibrate_batched_precomputed(
            X_test_embed=test_proj, X_test_original=test_raw,
            model_probs=np.asarray(model_probs_test), batch_size=args.batch_size,
        )

        method_name = (
            f"corrected_rgcl_studyA_legacy_seed{args.checkpoint_seed}" if is_legacy
            else f"corrected_rgcl_studyA_d{seed}"
        )
        metrics = {
            "top_label_ece": float(calculate_ece(calibrated_probs, test_labels)),
            "adaptive_ece": float(calculate_adaptive_ece(calibrated_probs, test_labels)),
            "accuracy": float(calculate_accuracy(calibrated_probs, test_labels)),
        }
        rows.append(
            build_artifact_row(
                method="corrected_rgcl_studyA", checkpoint_seed=args.checkpoint_seed,
                rgc_draw_seed=None if is_legacy else seed,
                representation_strategy=(
                    f"legacy_own_seed_{legacy_source}" if is_legacy else "corrected_internal_L6"
                ),
                statistic="gc_separation", mapper="geometric_isotonic_fit",
                dataset=args.dataset, corruption=None, severity=None, split="clean_test",
                sample_ids_ref=f"{args.output_dir}/test_sample_ids.npy",
                sample_count=int(len(test_labels)), metrics=metrics, provenance=provenance,
            )
        )
        draw_metadata[method_name] = {"layers": layers, "draw_seed": None if is_legacy else int(seed)}
        del geo_cal, calibrated_probs, model_probs_test
    t_fit1 = time.time()

    os.makedirs(args.output_dir, exist_ok=True)
    for row, method_name in zip(rows, draw_metadata.keys()):
        with open(Path(args.output_dir) / f"{method_name}.json", "w") as f:
            json.dump(row, f, indent=2)
    np.save(Path(args.output_dir) / "test_sample_ids.npy", test_result["sample_ids"])
    np.save(Path(args.output_dir) / "test_labels.npy", test_labels)

    t_end = time.time()
    report = {
        "provenance": provenance,
        "config": base_config,
        "checkpoint_path": str(checkpoint_path),
        "draw_layers": {str(k): v for k, v in draw_layers.items()},
        "legacy_source": legacy_source,
        "published_rgcl_note": "Published RGCL (uncorrected, own-seed, run_unified_benchmark.py's 'rgcl' method) is a separate row/provenance, not produced by this script.",
        "split_sizes": {"train": int(len(train_raw)), "val": int(len(val_raw)), "test": int(len(test_raw))},
        "timing_s": {
            "data_materialize": t_data1 - t_data0,
            "combined_extraction_3_splits": t_extract1 - t_extract0,
            "fit_calibrate_4_draws": t_fit1 - t_fit0,
            "total": t_end - t_start,
        },
        "resource_usage": {
            "device": str(device),
            "peak_gpu_memory_mb": peak_gpu_mb,
        },
    }
    with open(Path(args.output_dir) / "corrected_rgcl_studyA_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({"draw_layers": report["draw_layers"], "legacy_source": legacy_source}, indent=2))
    print(f"Wrote corrected_rgcl_studyA report to {args.output_dir}/corrected_rgcl_studyA_report.json")


if __name__ == "__main__":
    main()
