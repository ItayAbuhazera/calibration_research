"""
Extract evaluation-only features/logits for CIFAR-100-C cells that the Phase
0/1 sweep never ran.

WHY A SEPARATE SCRIPT
---------------------
The reliability shift study needs only three arrays per corruption cell:
`test_features` (the research-selected layer, global-average-pooled),
`test_logits` and `test_labels`. Re-running the full unified benchmark for a
new corruption cell would additionally load every calibrator's fitted state
and write method checkpoints -- far more work, and more chances to disturb
existing artifacts. This script does the extraction and nothing else.

IDENTICAL-BY-CONSTRUCTION
-------------------------
It imports `FeatureExtractor`, `load_trained_model` and
`_extract_split_features_to_disk` from `Experiments/run_unified_benchmark.py`
and calls them with the same arguments the runner uses for its test split, so
the arrays are produced by the same code path rather than a reimplementation.
`--verify_against` re-extracts an EXISTING cell and asserts the result matches
that cell's stored arrays, which is the check that this script is a faithful
stand-in.

WRITES ONLY NEW PATHS
---------------------
Output goes to `<out_root>/<corruption>_s<severity>/intermediates/features/`.
The script refuses to write into a directory that already contains a
`test_features.npy` unless `--force` is passed, so an existing Phase 0/1 cell
can never be overwritten by accident.

Evaluation only: train and validation remain clean and are not re-extracted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Experiments.run_unified_benchmark import (  # noqa: E402
    FeatureExtractor,
    _extract_split_features_to_disk,
    load_trained_model,
    resolve_device,
)
from utils.calibration_utils import load_cifar_c_loader  # noqa: E402
from utils import preprocessing_protocol as pp  # noqa: E402

# The 15 standard CIFAR-100-C corruptions. Discovered from disk at runtime;
# this list is only the expected set, used to report anything missing.
STANDARD_CIFAR_C = (
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)


def discover_corruptions(cifar_c_dir: str) -> List[str]:
    found = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(cifar_c_dir)
        if f.endswith(".npy") and os.path.splitext(f)[0] != "labels"
    )
    return found


def _test_transform(dataset: str, protocol: str = pp.PROTOCOL_CORRECTED):
    """Evaluation transform from the ONE authoritative specification.

    AMENDED 2026-09-21: this used a third, unrelated set of constants (CIFAR-100
    dataset statistics 0.5071/0.4865/0.4409), which matched neither the
    checkpoints' training normalization nor the benchmark's legacy test
    transform, despite the comment claiming it matched the clean test pipeline.
    See docs/normalization_audit.md. Cells extracted before this amendment (none
    are referenced by any stored study output at the time of writing) must not
    be mixed with cells extracted now; the protocol is recorded in
    extraction_metadata.json and stamped next to the features.
    """
    return pp.eval_transform(dataset, protocol, corruption=True)


def extract_cell(
    *,
    model,
    model_name: str,
    dataset: str,
    device: torch.device,
    cifar_c_dir: str,
    corruption: str,
    severity: int,
    out_cell_dir: str,
    batch_size: int,
    force: bool = False,
    protocol: str = pp.PROTOCOL_CORRECTED,
) -> Tuple[str, float]:
    feature_dir = os.path.join(out_cell_dir, "intermediates", "features")
    target = os.path.join(feature_dir, "test_features.npy")
    if os.path.exists(target) and not force:
        raise FileExistsError(
            f"{target} already exists; refusing to overwrite an existing cell "
            "(pass --force only for a directory you created)"
        )
    os.makedirs(feature_dir, exist_ok=True)
    if protocol != pp.PROTOCOL_LEGACY:
        pp.write_stamp(os.path.join(out_cell_dir, "intermediates"), dataset, protocol)

    loader = load_cifar_c_loader(
        dataset, corruption, severity, _test_transform(dataset, protocol), batch_size,
        cifar_c_dir=cifar_c_dir,
    )
    extractor = FeatureExtractor(model, model_name)
    start = time.perf_counter()
    try:
        _, features_path, logits_path, labels_path, _ = _extract_split_features_to_disk(
            feature_extractor=extractor,
            data_loader=loader,
            device=device,
            out_prefix=os.path.join(feature_dir, "test"),
            include_raw=False,
            penultimate_capture=None,
        )
    finally:
        extractor.cleanup()
    elapsed = time.perf_counter() - start

    os.replace(features_path, target)
    os.replace(logits_path, os.path.join(feature_dir, "test_logits.npy"))
    os.replace(labels_path, os.path.join(feature_dir, "test_labels.npy"))

    with open(os.path.join(feature_dir, "extraction_metadata.json"), "w") as f:
        json.dump({
            "corruption": corruption,
            "severity": int(severity),
            "selected_layer": extractor.selected_layer_name,
            "cifar_c_dir": cifar_c_dir,
            "preprocessing_protocol": protocol,
            "evaluation_only": True,
            "train_val_reextracted": False,
            "extraction_time_s": elapsed,
        }, f, indent=2)
    return target, elapsed


def verify_against(existing_cell: str, produced_dir: str) -> None:
    """Assert freshly-extracted arrays match an already-stored Phase 0/1 cell."""
    src = os.path.join(existing_cell, "intermediates", "features")
    for name in ("test_features", "test_logits", "test_labels"):
        a = np.load(os.path.join(src, f"{name}.npy"))
        b = np.load(os.path.join(produced_dir, f"{name}.npy"))
        if a.shape != b.shape:
            raise AssertionError(f"{name}: shape {a.shape} vs {b.shape}")
        if name == "test_labels":
            if not np.array_equal(a, b):
                raise AssertionError(f"{name}: labels differ")
        else:
            max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            rel = max_abs / max(float(np.max(np.abs(a))), 1e-12)
            print(f"  {name}: max_abs_diff={max_abs:.3e} relative={rel:.3e}")
            if rel > 1e-4:
                raise AssertionError(f"{name}: extraction does not reproduce the stored cell")
    print("  VERIFIED: extraction reproduces the stored Phase 0/1 cell")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="resnet101")
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--num_classes", type=int, default=100)
    p.add_argument("--cifar_c_dir", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--corruptions", default=None,
                   help="comma-separated; default = every corruption found on disk")
    p.add_argument("--severities", default="1,3,5")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--preprocessing_protocol", choices=list(pp.PROTOCOLS), default=pp.PROTOCOL_CORRECTED,
                   help="evaluation normalization; --verify_against uses the protocol of the cell it verifies")
    p.add_argument("--verify_against", default=None,
                   help="an existing cell dir; re-extract it and compare instead of sweeping")
    args = p.parse_args()

    device = resolve_device(args.device)
    model = load_trained_model(
        args.checkpoint, args.model, args.num_classes, device, dataset=args.dataset
    )
    model.eval()

    available = discover_corruptions(args.cifar_c_dir)
    missing = [c for c in STANDARD_CIFAR_C if c not in available]
    print(f"Corruptions on disk ({len(available)}): {', '.join(available)}")
    if missing:
        print(f"NOT available from the standard 15: {', '.join(missing)}")

    if args.verify_against:
        name = os.path.basename(os.path.normpath(args.verify_against))
        corruption, sev = name.rsplit("_s", 1)
        tmp = os.path.join(args.out_root, f"__verify__{name}")
        # an existing cell is compared under the protocol IT was produced with
        vproto = pp.read_protocol(os.path.join(args.verify_against, "intermediates"))
        print(f"Re-extracting {corruption} s{sev} to verify against {args.verify_against}")
        extract_cell(
            model=model, model_name=args.model, dataset=args.dataset, device=device,
            cifar_c_dir=args.cifar_c_dir, corruption=corruption, severity=int(sev),
            out_cell_dir=tmp, batch_size=args.batch_size, force=True, protocol=vproto,
        )
        verify_against(args.verify_against, os.path.join(tmp, "intermediates", "features"))
        return

    corruptions = args.corruptions.split(",") if args.corruptions else available
    severities = [int(s) for s in args.severities.split(",")]
    total = 0
    for corruption in corruptions:
        for severity in severities:
            cell = os.path.join(args.out_root, f"{corruption}_s{severity}")
            target = os.path.join(cell, "intermediates", "features", "test_features.npy")
            if os.path.exists(target) and args.skip_existing:
                print(f"  skip (exists): {corruption} s{severity}")
                continue
            _, elapsed = extract_cell(
                model=model, model_name=args.model, dataset=args.dataset, device=device,
                cifar_c_dir=args.cifar_c_dir, corruption=corruption, severity=severity,
                out_cell_dir=cell, batch_size=args.batch_size, force=args.force,
                protocol=args.preprocessing_protocol,
            )
            total += 1
            print(f"  {corruption} s{severity}: {elapsed:.1f}s -> {cell}", flush=True)
    print(f"Extracted {total} cells")


if __name__ == "__main__":
    main()
