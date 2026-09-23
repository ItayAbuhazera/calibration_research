"""Cached-artifact loader and reconciliation for Stage 0 (docs/stage0_execution_spec.md).

Reads only existing cached artifacts (atlas base logits, layer-pilot probe logits, fixed-gate
deep-candidate/labels) -- no new forward passes, no new features. See spec Section 1 for the
verified path/shape/dtype table this module implements.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict

import numpy as np

from . import spec

LAYER3_22_INDEX = 8  # candidates[8] == {"module": "layer3.22", ...}; verified against frozen_state.json
ATLAS_ROOT = "results/atlas"
LAYER_PILOT_ROOT = "results/layer_pilot"
FIXED_GATE_ROOT = "results/fixed_gate"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def frozen_state(seed: int) -> Dict:
    return json.load(open(f"{LAYER_PILOT_ROOT}/checkpoint_seed{seed}/frozen_state.json"))


def evidence_index(evidence):
    """None -> layer3.22 (index 8); 'L<k>' -> probe layer index k (0-11); 'xckpt' -> other checkpoint's logits."""
    if evidence is None:
        return LAYER3_22_INDEX
    if evidence == "xckpt":
        return None
    assert evidence.startswith("L") and 0 <= int(evidence[1:]) <= 11, evidence
    return int(evidence[1:])


def load_cell(seed: int, cell: str, evidence=None) -> Dict[str, np.ndarray]:
    """One condition's canonical arrays for one checkpoint seed.

    evidence (Stage 0 evidence ablation, docs/stage0_evidence_ablation_spec.md): which 100-column
    second predictor P is returned as "p". Default None keeps the Stage 0 behaviour (layer3.22).

    Returns z (N,100) float64, p_layer322 (N,100) float64 (pre-temperature, cast up from the
    cached float16 -- the quantization floor is NOT recovered by the cast, see spec Section 1),
    cand_j_deep (N,) int64, labels (N,) int64, base_pred (N,) int64.
    """
    atlas = np.load(f"{ATLAS_ROOT}/seed{seed}/u0/{cell}.npz")
    pilot = np.load(f"{LAYER_PILOT_ROOT}/checkpoint_seed{seed}/{cell}/per_sample.npz")
    fg = np.load(f"{FIXED_GATE_ROOT}/seed{seed}/per_sample_{cell}.npz")

    z = atlas["logits"].astype(np.float64)
    idx = evidence_index(evidence)
    if idx is None:
        other = 6 - seed
        assert other in (2, 4) and other != seed
        p = np.load(f"{ATLAS_ROOT}/seed{other}/u0/{cell}.npz")["logits"].astype(np.float64)
        fg_other = np.load(f"{FIXED_GATE_ROOT}/seed{other}/per_sample_{cell}.npz")
        assert np.array_equal(fg_other["labels"].astype(np.int64), fg["labels"].astype(np.int64)), f"other-checkpoint labels differ ({cell})"
    else:
        p = pilot["raw__probe_logits"][:, idx, :].astype(np.float64)
    labels = fg["labels"].astype(np.int64)
    cand_j = fg["cand_j__deep"].astype(np.int64)
    base_pred_fg = fg["base_pred"].astype(np.int64)

    n = z.shape[0]
    assert p.shape == (n, spec.NUM_CLASSES)
    assert labels.shape == (n,) and cand_j.shape == (n,)
    assert np.array_equal(pilot["labels"].astype(np.int64), labels), f"seed{seed}/{cell}: layer_pilot vs fixed_gate label mismatch"

    atlas_pred = z.argmax(1)
    assert (atlas_pred == base_pred_fg).all(), f"seed{seed}/{cell}: atlas argmax vs fixed_gate base_pred disagree ({(atlas_pred != base_pred_fg).sum()} rows)"
    pilot_pred = pilot["pred__base_model"].astype(np.int64)
    assert (atlas_pred == pilot_pred).all(), f"seed{seed}/{cell}: atlas argmax vs layer_pilot pred__base_model disagree"

    return {
        "z": z,
        "p": p,
        "cand_j_deep": cand_j,
        "labels": labels,
        "base_pred": base_pred_fg,
        "n": n,
    }


def load_all_cells(seed: int, evidence=None) -> Dict[str, Dict[str, np.ndarray]]:
    return {cell: load_cell(seed, cell, evidence) for cell in spec.CONDITIONS}


def probe_temperature_layer322(seed: int) -> float:
    fs = frozen_state(seed)
    return float(fs["probes"]["layers"]["layer3.22"]["temperature"])


def probe_temperatures_all12(seed: int) -> np.ndarray:
    fs = frozen_state(seed)
    return np.asarray(fs["probe_temperatures"], dtype=np.float64)


def candidate_layer_names(seed: int) -> list:
    fs = frozen_state(seed)
    return [c["module"] for c in fs["candidates"]]


def test_duplicate_groups() -> Dict[str, list]:
    """Pixel-identical groups within the 10 000-image clean test set (computed once, cached).

    Property of the CIFAR-100 test images, identical for both checkpoint seeds. See spec
    Section 1: 2 groups, both label-conflicting, found by SHA-256 of the raw clean block of
    results/atlas/shared/test_sets_full.npy.
    """
    cache = "results/stage0/shared/test_duplicate_groups.json"
    if os.path.exists(cache):
        return json.load(open(cache))["groups"]
    ts = np.load(f"{ATLAS_ROOT}/shared/test_sets_full.npy", mmap_mode="r")
    clean = np.asarray(ts[0:10000])
    labels = np.load(f"{ATLAS_ROOT}/shared/test_labels.npy")
    hashes: Dict[str, list] = {}
    for i in range(10000):
        h = _sha256_bytes(clean[i].tobytes())
        hashes.setdefault(h, []).append(i)
    groups = {h: idxs for h, idxs in hashes.items() if len(idxs) > 1}
    out = {h: {"indices": idxs, "labels": [int(labels[i]) for i in idxs]} for h, idxs in groups.items()}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump({"groups": out, "n_groups": len(out), "note": "SHA-256 of raw clean-block pixels, computed 2026-09-22"}, open(cache, "w"), indent=1)
    return out


def bank_test_duplicate_indices(seed: int) -> Dict[str, int]:
    """Counts only (indices are not individually recoverable from the atlas manifest, see spec Section 1)."""
    d = json.load(open(f"{ATLAS_ROOT}/shared/p0_manifest.json"))
    return d[f"seed{seed}"]["duplicate_audit"]


def test_vs_train_duplicate_test_indices() -> list:
    """Original test-set indices pixel-identical to a CIFAR-100 TRAIN image (materialized-pixel
    hash of data/cifar-100-python's raw train/test batches, computed 2026-09-23; permitted under
    the controlling prompt's "reading materialized pixels to verify IDs/duplicate hashes").

    Computed against the full 50,000-image train set, a superset of either seed's 45,000-row
    fitting bank -- so this is the conservative (superset-safe) exclusion list for both seeds.
    The atlas manifest's per-seed counts (10 for seed 2, 9 for seed 4) are each <= len(this list)
    because each seed's actual bank is a proper subset of the 50,000 train images; one of these
    10 may not be memorized by seed 4's specific bank, making its exclusion slightly conservative
    there (drops one extra, unproblematic row), never anti-conservative.
    """
    cache = "results/stage0/shared/test_vs_train_duplicate_indices.json"
    d = json.load(open(cache))
    return sorted(int(k) for k in d.keys())
