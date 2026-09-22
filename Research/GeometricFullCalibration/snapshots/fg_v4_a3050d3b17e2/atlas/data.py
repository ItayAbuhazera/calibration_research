"""Data roles, query sets and shared caches.

Roles (per checkpoint seed; row indices into that seed's benchmark-materialized 5 000-row validation array):
  reference   train split (45 000): bank, class prior, covariance, PCA/clustering fits
  fit         inherited inner-FIT (2 500): intervention models, fitted baselines
  selection   1 250 rows: layer/metric choices, gate hyper-parameters and thresholds
  calibration 1 250 rows: final scalar temperature only, after the whole decision policy is frozen
  test / development corruption cells: evaluation and labelled diagnostics only
The 1 250/1 250 division of the old 2 500-row selection role is prospective (class-stratified, fixed seed) and does
not rewrite the earlier studies' split history; these rows have historical exposure.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np

from . import spec

ROOT = "results/atlas"
BENCH = "results/studyAB/phase0_corrected_v2"
SHARED = f"{ROOT}/shared"
CIFAR_C = ("/home/itayab/PyCharmProjects/Research/geometric/GeometricInternalCalibration/"
           "GeometricInternalCalibration/data/cifar100-c")
CKPT_ROOT = ("/home/itayab/PyCharmProjects/Research/geometric/GeometricInternalCalibration/GeometricInternalCalibration/"
             "aaai_full_experiments/results")


def seed_dir(seed: int) -> str:
    return f"{ROOT}/seed{seed}"


def load_seed_arrays(seed: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    sp = f"{BENCH}/evaluation/checkpoint_seed{seed}/clean/intermediates/splits"
    L = lambda k: np.load(f"{sp}/{k}", mmap_mode="r") if k.endswith("_raw.npy") else np.load(f"{sp}/{k}")  # noqa: E731
    return {"train": (L("train_raw.npy"), L("train_labels.npy")), "val": (L("val_raw.npy"), L("val_labels.npy")),
            "test": (L("test_raw.npy"), L("test_labels.npy"))}


def make_roles(val_y: np.ndarray) -> Dict[str, np.ndarray]:
    from sklearn.model_selection import StratifiedShuffleSplit
    from utils.decision_audit import make_inner_validation_split
    fit, pool = make_inner_validation_split(val_y, select_fraction=0.5, seed=spec.INNER_SEED)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=spec.ROLE_SPLIT_SEED)
    a, b = next(sss.split(pool, val_y[pool]))
    sel, cal = np.sort(pool[a]), np.sort(pool[b])
    assert len(set(fit) & set(sel)) == 0 and len(set(fit) & set(cal)) == 0 and len(set(sel) & set(cal)) == 0
    return {"fit": np.sort(fit), "selection": sel, "calibration": cal}


def make_subset(test_y: np.ndarray) -> np.ndarray:
    """Class-stratified 2 000-image test subset (20 per class); outcome-independent; identical for every seed."""
    rng = np.random.default_rng(spec.SUBSET_SEED)
    ids = []
    for c in range(spec.NUM_CLASSES):
        pool = np.flatnonzero(test_y == c)
        ids.append(np.sort(rng.choice(pool, spec.SUBSET_PER_CLASS, replace=False)))
    return np.sort(np.concatenate(ids))


def full_sets_path() -> str:
    return f"{SHARED}/test_sets_full.npy"


def load_full_sets() -> np.ndarray:
    """[13*10000, 3, 32, 32] float32, condition-major (CONDITIONS order), corrected protocol, natural test order."""
    return np.load(full_sets_path(), mmap_mode="r")


def cond_slice(cond: str) -> slice:
    i = spec.CONDITIONS.index(cond)
    return slice(i * 10000, (i + 1) * 10000)


def atlas_query_meta(seed: int) -> Dict[str, np.ndarray]:
    """Order of the atlas query array: val (5 000 rows) then, for each condition, the 2 000 subset images."""
    sub = np.load(f"{SHARED}/subset_ids.npy")
    n_val = 5000
    set_id = np.concatenate([np.zeros(n_val, np.int16)] + [np.full(len(sub), i + 1, np.int16) for i in range(len(spec.CONDITIONS))])
    img = np.concatenate([np.arange(n_val)] + [sub for _ in spec.CONDITIONS]).astype(np.int32)
    return {"set_id": set_id, "img_id": img}


def atlas_queries(seed: int, arrays=None) -> np.ndarray:
    """Materialize [31 000, 3, 32, 32] float32 (381 MB): val rows + test-subset images of every condition."""
    arrays = arrays or load_seed_arrays(seed)
    sub = np.load(f"{SHARED}/subset_ids.npy")
    full = load_full_sets()
    parts = [np.asarray(arrays["val"][0])]
    for c in spec.CONDITIONS:
        parts.append(np.asarray(full[cond_slice(c)][sub]))
    return np.concatenate(parts)


def atlas_labels(seed: int, arrays=None) -> np.ndarray:
    arrays = arrays or load_seed_arrays(seed)
    sub = np.load(f"{SHARED}/subset_ids.npy")
    ty = np.load(f"{SHARED}/test_labels.npy")
    return np.concatenate([arrays["val"][1]] + [ty[sub] for _ in spec.CONDITIONS]).astype(np.int64)
