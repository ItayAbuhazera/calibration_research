"""
Phase 1 smoke test for BENCHMARK_IMPLEMENTATION_PLAN.md's Study A/B benchmark
(§8, §12 Phase 1 step (c), §17 step 9).

STRICTLY NON-SCIENTIFIC. Checkpoint seed1 only, small predefined subsets only
(a class-stratified reference bank drawn from the training set, plus small
disjoint fit/test slices of the validation set). This script exists solely to
prove the combined-hook extraction and Study-B factorial pipeline work
end-to-end, on the real checkpoint and real architecture, before any Phase
0/1/2 scientific execution. Its numbers must never be cited as calibration-
quality evidence and must never feed a ResearchBrain conclusion.

Produces:
  <output_dir>/smoke_report.json  -- full report incl. provenance, timing,
                                      resource usage, and the raw values the
                                      caller-side verification checks (run
                                      separately, off-cluster) will re-derive.
  <output_dir>/artifacts/*.json   -- one §9-schema row per method/cell.
  <output_dir>/representations/   -- the combined-hook extraction cache
                                      (train_ref/fit/test splits), so the
                                      verification pass can re-check §10.1-4/
                                      §10.6-9 directly against the artifacts
                                      instead of trusting this script's own
                                      in-process assertions alone.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from Calibrators.density_aware_calibration import DensityAwareCalibrator
from Calibrators.studyB_common_pipeline import CommonFactorialCalibrator
from Experiments.extract_unified_studyAB_representations import (
    DEFAULT_DRAW_SEEDS,
    STUDY_B_NUM_LAYERS,
    extract_combined_representations,
    resolve_combined_layer_set,
)
from Experiments.run_rgc_experiments import (
    calculate_accuracy,
    calculate_adaptive_ece,
    calculate_brier_score,
    calculate_ece,
    select_random_rgc_layers,
)
from Experiments.studyAB_artifact_schema import build_artifact_row, config_hash
from utils.calibration_utils import get_all_data_as_numpy
from utils.model_utils import get_data_loaders, load_trained_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def _git_dirty() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode()
    return bool(out.strip())


def _stratified_subset(raw, labels, per_class, num_classes, rng):
    chosen = []
    for c in range(num_classes):
        idxs = np.flatnonzero(labels == c)
        if len(idxs) == 0:
            continue
        take = idxs if len(idxs) <= per_class else rng.choice(idxs, size=per_class, replace=False)
        chosen.append(take)
    chosen = np.concatenate(chosen)
    rng.shuffle(chosen)
    return raw[chosen], labels[chosen]


def _make_loader(raw, labels, batch_size):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(raw), torch.from_numpy(labels))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def _model_outputs(model, raw, device, batch_size):
    logits_chunks = []
    with torch.no_grad():
        for i in range(0, len(raw), batch_size):
            batch = torch.from_numpy(raw[i : i + batch_size]).to(device)
            logits_chunks.append(model(batch).cpu().numpy())
    logits = np.concatenate(logits_chunks, axis=0).astype(np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return logits, probs


def _layer_dict(result: Dict, layers) -> Dict[str, np.ndarray]:
    return {l: result["dac_style_features"][l] for l in layers}


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--model", default="resnet101")
    parser.add_argument("--checkpoint", default="seed1")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ref_per_class", type=int, default=20)
    parser.add_argument("--fit_n", type=int, default=300)
    parser.add_argument("--test_n", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_dimension", type=int, default=256)
    args = parser.parse_args()

    t_start = time.time()
    commit = _git_commit()
    dirty = _git_dirty()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    num_classes = 100 if "cifar100" in args.dataset.lower() else 10
    checkpoint_path = (
        REPO_ROOT.parent
        / "geometric"
        / "GeometricInternalCalibration"
        / "GeometricInternalCalibration"
        / "aaai_full_experiments"
        / "results"
        / "baseline"
        / "baseline_cross_entropy"
        / args.dataset
        / args.model
        / args.checkpoint
        / "best_model.pth"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_trained_model(str(checkpoint_path), args.model, num_classes, device, dataset=args.dataset)
    model.eval()

    train_loader, val_loader, test_loader, loader_num_classes = get_data_loaders(
        args.dataset, args.batch_size, seed=args.seed
    )
    assert loader_num_classes == num_classes

    t_data0 = time.time()
    train_raw, train_labels = get_all_data_as_numpy(train_loader)
    val_raw, val_labels = get_all_data_as_numpy(val_loader)
    t_data1 = time.time()

    rng = np.random.default_rng(args.seed)
    ref_raw, ref_labels = _stratified_subset(train_raw, train_labels, args.ref_per_class, num_classes, rng)
    del train_raw, train_labels

    if args.fit_n + args.test_n > len(val_raw):
        raise ValueError(
            f"fit_n+test_n ({args.fit_n + args.test_n}) exceeds validation split size ({len(val_raw)})"
        )
    perm = rng.permutation(len(val_raw))
    fit_idx = perm[: args.fit_n]
    test_idx = perm[args.fit_n : args.fit_n + args.test_n]
    fit_raw, fit_labels = val_raw[fit_idx], val_labels[fit_idx]
    test_raw, test_labels = val_raw[test_idx], val_labels[test_idx]
    del val_raw, val_labels

    # --- layer resolution, called twice for the §10.14 determinism check ---
    resolved_1 = resolve_combined_layer_set(
        model=model, model_name=args.model, dataset_name=args.dataset,
        device=device, draw_seeds=DEFAULT_DRAW_SEEDS, num_layers=STUDY_B_NUM_LAYERS,
    )
    resolved_2 = resolve_combined_layer_set(
        model=model, model_name=args.model, dataset_name=args.dataset,
        device=device, draw_seeds=DEFAULT_DRAW_SEEDS, num_layers=STUDY_B_NUM_LAYERS,
    )
    determinism_ok = bool(
        resolved_1["draw_layers"] == resolved_2["draw_layers"]
        and resolved_1["dac_layers"] == resolved_2["dac_layers"]
    )

    dac_layers = resolved_1["dac_layers"]
    draw_layers = resolved_1["draw_layers"]
    combined_layers = resolved_1["combined_layers"]

    # §10.11: Published RGCL (uncorrected, historical seed==checkpoint id
    # convention, L=6) stays a separate code path from corrected RGCL.
    published_layers = select_random_rgc_layers(
        model=model, model_name=args.model, dataset_name=args.dataset,
        device=device, num_layers=6, seed=1,
    )

    # --- combined-hook extraction: one forward pass per split ---
    t_extract0 = time.time()
    ref_result = extract_combined_representations(
        model=model, loader=_make_loader(ref_raw, ref_labels, args.batch_size), split_name="train_ref",
        combined_layers=combined_layers, draw_layers=draw_layers, device=device,
        target_dim=args.target_dimension, pooling_mode="max",
    )
    fit_result = extract_combined_representations(
        model=model, loader=_make_loader(fit_raw, fit_labels, args.batch_size), split_name="fit",
        combined_layers=combined_layers, draw_layers=draw_layers, device=device,
        target_dim=args.target_dimension, pooling_mode="max",
    )
    test_result = extract_combined_representations(
        model=model, loader=_make_loader(test_raw, test_labels, args.batch_size), split_name="test",
        combined_layers=combined_layers, draw_layers=draw_layers, device=device,
        target_dim=args.target_dimension, pooling_mode="max",
    )
    t_extract1 = time.time()
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None

    fit_logits, fit_probs = _model_outputs(model, fit_raw, device, args.batch_size)
    test_logits, test_probs = _model_outputs(model, test_raw, device, args.batch_size)

    seed_a = DEFAULT_DRAW_SEEDS[0]
    random_layers = draw_layers[seed_a]

    ref_random = _layer_dict(ref_result, random_layers)
    fit_random = _layer_dict(fit_result, random_layers)
    test_random = _layer_dict(test_result, random_layers)
    ref_dac = _layer_dict(ref_result, dac_layers)
    fit_dac = _layer_dict(fit_result, dac_layers)
    test_dac = _layer_dict(test_result, dac_layers)

    snap_random = {k: v.copy() for k, v in ref_random.items()}
    snap_dac = {k: v.copy() for k, v in ref_dac.items()}

    dac_k = 200  # get_dac_k_value("cifar100")

    t_fit0 = time.time()
    cell_A = CommonFactorialCalibrator(random_layers, "gc_separation")
    cell_A.fit(ref_random, ref_labels, fit_random, fit_probs, fit_labels)
    probs_A = cell_A.calibrate(test_random, test_probs)

    cell_B = CommonFactorialCalibrator(random_layers, "dac_density", k=dac_k)
    cell_B.fit(ref_random, ref_labels, fit_random, fit_probs, fit_labels)
    probs_B = cell_B.calibrate(test_random, test_probs)

    cell_C = CommonFactorialCalibrator(dac_layers, "gc_separation")
    cell_C.fit(ref_dac, ref_labels, fit_dac, fit_probs, fit_labels)
    probs_C = cell_C.calibrate(test_dac, test_probs)

    cell_D = CommonFactorialCalibrator(dac_layers, "dac_density", k=dac_k)
    cell_D.fit(ref_dac, ref_labels, fit_dac, fit_probs, fit_labels)
    probs_D = cell_D.calibrate(test_dac, test_probs)
    t_fit1 = time.time()

    mutation_ok = all(np.array_equal(v, snap_random[k]) for k, v in ref_random.items()) and all(
        np.array_equal(v, snap_dac[k]) for k, v in ref_dac.items()
    )

    # native DAC: a genuinely separate code path (DensityAwareCalibrator),
    # reusing the exact same cached DAC-layer features as cell D.
    native = DensityAwareCalibrator(k=dac_k, use_gpu=torch.cuda.is_available())
    native.fit(
        [ref_dac[l] for l in dac_layers],
        [fit_dac[l] for l in dac_layers],
        fit_logits,
        fit_labels,
    )
    native_probs = native.calibrate([test_dac[l] for l in dac_layers], test_logits)

    t_end = time.time()

    # --- §9 artifact rows ---
    base_config = {
        "dataset": args.dataset, "model": args.model, "checkpoint": args.checkpoint,
        "draw_seeds": list(DEFAULT_DRAW_SEEDS), "num_layers": STUDY_B_NUM_LAYERS,
        "target_dimension": args.target_dimension, "dac_k": dac_k,
        "ref_per_class": args.ref_per_class, "fit_n": args.fit_n, "test_n": args.test_n,
        "seed": args.seed,
    }
    cfg_hash = config_hash(base_config)
    provenance = {
        "git_commit": commit,
        "git_dirty": dirty,
        "config_hash": cfg_hash,
        "plan_version": "BENCHMARK_IMPLEMENTATION_PLAN.md",
        "extraction_script": "Experiments/extract_unified_studyAB_representations.py",
        "smoke_script": "Experiments/smoke_studyAB_phase1.py",
        "hostname": socket.gethostname(),
    }

    def metrics_for(probs, labels, scalar_only):
        m = {
            "top_label_ece": float(calculate_ece(probs, labels)),
            "adaptive_ece": float(calculate_adaptive_ece(probs, labels)),
            "accuracy": float(calculate_accuracy(probs, labels)),
        }
        if not scalar_only:
            m["nll"] = float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None))))
            m["brier"] = float(calculate_brier_score(probs, labels))
        return m

    rows = []
    cell_defs = [
        ("studyB_cell_A", probs_A, "corrected_random_internal_L5", "gc_separation", seed_a),
        ("studyB_cell_B", probs_B, "corrected_random_internal_L5", "dac_density", seed_a),
        ("studyB_cell_C", probs_C, "dac_prescribed_L5", "gc_separation", None),
        ("studyB_cell_D", probs_D, "dac_prescribed_L5", "dac_density", None),
    ]
    for method, probs, repr_strategy, statistic, draw_seed in cell_defs:
        rows.append(
            build_artifact_row(
                method=method, checkpoint_seed=1, rgc_draw_seed=draw_seed,
                representation_strategy=repr_strategy, statistic=statistic,
                mapper="common_rank_isotonic", dataset=args.dataset, corruption=None,
                severity=None, split="smoke_test",
                sample_ids_ref=f"smoke:{args.output_dir}/representations/test/sample_ids.npy",
                sample_count=int(len(test_labels)),
                metrics=metrics_for(probs, test_labels, scalar_only=True),
                provenance=provenance,
            )
        )
    rows.append(
        build_artifact_row(
            method="native_dac", checkpoint_seed=1, rgc_draw_seed=None,
            representation_strategy="dac_prescribed_L5", statistic="dac_native_weighted_sum",
            mapper="native_dac", dataset=args.dataset, corruption=None, severity=None,
            split="smoke_test", sample_ids_ref=f"smoke:{args.output_dir}/representations/test/sample_ids.npy",
            sample_count=int(len(test_labels)),
            metrics=metrics_for(native_probs, test_labels, scalar_only=False),
            provenance=provenance,
        )
    )

    os.makedirs(args.output_dir, exist_ok=True)
    artifacts_dir = Path(args.output_dir) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        with open(artifacts_dir / f"{row['method']}.json", "w") as f:
            json.dump(row, f, indent=2)

    repr_dir = Path(args.output_dir) / "representations"
    for split_name, result, raw, labels_arr in (
        ("train_ref", ref_result, ref_raw, ref_labels),
        ("fit", fit_result, fit_raw, fit_labels),
        ("test", test_result, test_raw, test_labels),
    ):
        split_dir = repr_dir / split_name
        (split_dir / "dac_style_layers").mkdir(parents=True, exist_ok=True)
        for name, arr in result["dac_style_features"].items():
            safe = name.replace("/", "_").replace("#", "_")
            np.save(split_dir / "dac_style_layers" / f"{safe}.npy", arr)
        (split_dir / "draw_projected").mkdir(parents=True, exist_ok=True)
        for seed, arr in result["draw_projected_features"].items():
            np.save(split_dir / "draw_projected" / f"seed{seed}.npy", arr)
        np.save(split_dir / "labels.npy", labels_arr)
        np.save(split_dir / "sample_ids.npy", result["sample_ids"])

    storage_bytes = _dir_size_bytes(Path(args.output_dir))

    report = {
        "provenance": provenance,
        "config": base_config,
        "checkpoint_path": str(checkpoint_path),
        "layer_resolution": {
            "dac_layers": dac_layers,
            "draw_layers": draw_layers,
            "combined_layers": combined_layers,
            "published_rgcl_layers_seed1_L6": published_layers,
            "determinism_ok": determinism_ok,
        },
        "split_sizes": {
            "train_ref": int(len(ref_raw)),
            "fit": int(args.fit_n),
            "test": int(args.test_n),
        },
        "in_process_checks": {
            "fc_absent_from_all_draws": all(
                "fc" not in names and not any(n == "fc" or n.startswith("fc.") for n in names)
                for names in draw_layers.values()
            ),
            "dac_layers_exact": dac_layers == ["conv1", "layer1", "layer2", "layer3", "layer4"],
            "draw_seeds_exact": list(draw_layers.keys()) == list(DEFAULT_DRAW_SEEDS),
            "determinism_ok": determinism_ok,
            "no_mutation_between_cells": mutation_ok,
            "labels_aligned_across_cells": bool(
                np.array_equal(test_labels, test_result["labels"])
            ),
        },
        "timing_s": {
            "data_materialize": t_data1 - t_data0,
            "combined_extraction_3_splits": t_extract1 - t_extract0,
            "study_b_fit_calibrate": t_fit1 - t_fit0,
            "total": t_end - t_start,
        },
        "resource_usage": {
            "device": str(device),
            "gpu_name": gpu_name,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "output_dir_bytes": storage_bytes,
        },
    }

    with open(Path(args.output_dir) / "smoke_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report["in_process_checks"], indent=2))
    print(f"Wrote smoke report to {args.output_dir}/smoke_report.json")


if __name__ == "__main__":
    main()
