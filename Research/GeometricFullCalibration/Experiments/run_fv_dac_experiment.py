#!/usr/bin/env python
"""
Full-Vector DAC (FV-DAC) standalone experiment runner.

Modelled on Experiments/run_unified_benchmark.py but deliberately NOT a
second benchmark: it reuses that benchmark's splits, checkpoints, feature
extraction, metrics, decision audit, method-semantics registry and
fit-once/evaluate-many discipline, and adds exactly one new method family.
`run_unified_benchmark.py` is not modified by this study.

--------------------------------------------------------------------------
STAGES
--------------------------------------------------------------------------

  --stage fit       clean only. Rebuilds a LABELLED reference bank, verifies
                    it reproduces the frozen native DAC, fits beta on the
                    clean inner-FIT half of validation, selects K_c on the
                    disjoint inner-SELECT half, writes the frozen state, and
                    evaluates clean test.

  --stage evaluate  corruption cells. Loads the frozen state or FAILS. There
                    is no code path from this stage to any fitting function
                    -- see `_FitGuard`, which is installed before the
                    corrupted loader is even constructed.

--------------------------------------------------------------------------
WHY THE FROZEN BANK IS *LABELLED* RATHER THAN REBUILT
--------------------------------------------------------------------------

The frozen DAC state stores the bank FEATURES but not the bank LABELS, and
the benchmark's train loader uses SubsetRandomSampler, so the pickled bank's
row order does not correspond to intermediates/splits/train_labels.npy.

Substituting a freshly extracted bank is NOT an option: convolutions run in
TF32 by default on Ampere+, so the same image's feature differs by ~1e-3
between processes (measured: 9.1e-4 at `conv1` on seed 4). A bank perturbed
at that scale moves the k-th-neighbour distance, hence S(x), hence the
probabilities -- and `beta = 0` would then reproduce native DAC only
approximately, destroying the experiment's matched operating point.

So the frozen bank IS the bank. A re-extraction of the materialized train
split is used ONLY to recover which label belongs to which frozen row, via an
sparse minimum-cost bijection in the equal-weight concatenation of every DAC
layer, then cross-checked layer by layer. See `_label_frozen_bank`.

"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.storage
import torchvision.transforms as transforms
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Calibrators.density_aware_calibration import (  # noqa: E402
    DensityAwareCalibrator,
    extract_dac_features,
    get_dac_k_value,
    get_dac_target_layers,
)
from Calibrators.full_vector_dac import (  # noqa: E402
    ARM_BRIER,
    ARM_DENSITY_ONLY,
    ARM_LOGIT_SPACE,
    ARM_LOGNORM,
    ARM_PERMUTED,
    ARM_PRIMARY,
    ARM_SHARED_LAYER,
    BETA_SEARCH_MAX,
    FITTED_ARMS,
    PERMUTED_LABEL_SEED_BASE,
    PREDECLARED_KC,
    ClassConditionalKNN,
    FVDACFrozenState,
    aggregate_R,
    fv_dac_probs,
    apply_arm,
    array_hash,
    bank_set_hash,
    centered_logit_representation,
    class_distances_multi,
    dac_preprocess,
    dac_temperature,
    density_only_prediction,
    fit_beta,
    fit_shared_layer_weights,
    nll_of,
    lognorm_statistics,
    normalized_euclidean,
    lognorm_transform,
    normalized_layer_weights,
    np_softmax,
    permuted_bank_labels,
)
from utils.calibration_utils import get_all_data_as_numpy, load_cifar_c_loader  # noqa: E402
from utils.decision_audit import decision_audit, make_inner_validation_split  # noqa: E402
from utils.logging_config import get_logger  # noqa: E402
from utils.method_metadata import (  # noqa: E402
    SEMANTIC_SCHEMA_VERSION,
    require_method_semantics,
    structural_axes_for_method,
)
from utils.model_utils import construct_model_path, load_trained_model  # noqa: E402
from utils.unified_metrics import evaluate_all, validate_probability_matrix  # noqa: E402

logger = get_logger(__name__)

#: Methods this runner emits, in report order. The last one is a diagnostic
#: with no probability vector, handled separately.
FV_DAC_METHODS = ("base_model", "native_dac") + FITTED_ARMS

#: Baselines that are NOT recomputed here -- they are read out of the frozen
#: Phase 0/1 artifacts for the matching cell so FV-DAC never re-runs (or
#: silently re-tunes) a method the canonical benchmark already owns.
REUSED_PHASE0_BASELINES = (
    "temperature_scaling",
    "vector_scaling",
    "odir_dirichlet",
    "kcal",
)


# ==========================================================================
# Fit guard -- structural, not conventional
# ==========================================================================


class _FitGuard:
    """Makes fitting unreachable during a corruption evaluation.

    Installed BEFORE any corrupted array is loaded. Every FV-DAC fitting
    entry point consults it, so an evaluation-only run that somehow reached
    a fit path raises instead of silently producing a refit number.
    """

    _blocked = False
    _reason = ""

    @classmethod
    def block(cls, reason: str) -> None:
        cls._blocked = True
        cls._reason = reason

    @classmethod
    def check(cls, what: str) -> None:
        if cls._blocked:
            raise RuntimeError(
                f"FV-DAC refuses to run {what}: {cls._reason}. Corruption "
                "evaluation is evaluation-only; the frozen state must already "
                "exist (fit-once/evaluate-many)."
            )

    @classmethod
    def is_blocked(cls) -> bool:
        return cls._blocked


def guarded_fit_beta(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    _FitGuard.check("beta fitting")
    return fit_beta(*args, **kwargs)


def guarded_fit_shared_layer(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    _FitGuard.check("shared-layer weight fitting")
    return fit_shared_layer_weights(*args, **kwargs)


# ==========================================================================
# Small helpers
# ==========================================================================


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True
        ).strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def _git_dirty(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(repo_dir), text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _atomic_write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_ready)
    os.replace(tmp, path)


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


def _load_cuda_pickle_on_any_device(path: str) -> Any:
    """Unpickle a state file that may hold CUDA tensors, onto whatever we have.

    The frozen Phase 0/1 `native_dac.pkl` was written on a GPU node, so a
    plain pickle.load fails on CPU. Nothing about the fitted values changes.
    """
    original = torch.storage._load_from_bytes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.storage._load_from_bytes = lambda b: torch.load(  # type: ignore[assignment]
        io.BytesIO(b), map_location=device
    )
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    finally:
        torch.storage._load_from_bytes = original  # type: ignore[assignment]


def _resolve_device(requested: str) -> torch.device:
    """Fail fast rather than silently degrading a GPU run to CPU."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested but CUDA is unavailable. Refusing to fall "
            "back to CPU silently: a CPU run would take hours and would not be the "
            "job that was scheduled."
        )
    return torch.device(requested)


def _retarget_dac(native_dac: DensityAwareCalibrator, device: torch.device) -> None:
    """Move a depickled DAC calibrator wholly onto one device.

    Each LayerKNNScorer pickles its own `device` alongside its bank tensor. If
    the two disagree after unpickling (bank mapped to CPU, device field still
    saying cuda), scorer.score() would mix devices and raise. Normalising both
    here keeps the frozen calibrator usable for the reproduction check.
    """
    native_dac.use_gpu = device.type == "cuda"
    for scorer in native_dac.layer_scorers:
        scorer.device = device
        if scorer.train_features is not None:
            scorer.train_features = scorer.train_features.to(device)


def _loader_from_arrays(raw: np.ndarray, labels: np.ndarray, batch_size: int) -> DataLoader:
    """Deterministic, order-preserving loader over a materialized split.

    shuffle=False is essential: these arrays are the benchmark's own
    materialized splits and their row order IS the split's identity.
    """
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(raw)).float(),
        torch.from_numpy(np.ascontiguousarray(labels)).long(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _corruption_transform() -> transforms.Compose:
    """LEGACY (legacy_v1_mixed_norm) -- the transform run_unified_benchmark.py used for CIFAR-*-C before 2026-09-21.

    Note it applies ImageNet normalization statistics while the clean path
    uses CIFAR statistics. That asymmetry is a pre-existing property of the
    frozen Phase 0/1 protocol; FV-DAC reproduces it verbatim so its rows stay
    comparable with the frozen native_dac rows rather than quietly measuring
    a different thing. It is flagged in docs/full_vector_dac_experiment.md.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _phase0_clean_dir(phase0_root: str, seed: int) -> str:
    return os.path.join(phase0_root, "evaluation", f"checkpoint_seed{seed}", "clean")


def _phase0_cell_dir(phase0_root: str, seed: int, cell: str) -> str:
    return os.path.join(phase0_root, "evaluation", f"checkpoint_seed{seed}", cell)


def _cell_name(corruption: Optional[str], severity: Optional[int]) -> str:
    return "clean" if corruption is None else f"{corruption}_s{severity}"


# ==========================================================================
# Feature / distance computation
# ==========================================================================


def _extract(
    model: torch.nn.Module,
    raw: np.ndarray,
    labels: np.ndarray,
    layers: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray]:
    loader = _loader_from_arrays(raw, labels, batch_size)
    feats, logits, lbls = extract_dac_features(model, loader, list(layers), device)
    got = lbls.numpy()
    if not np.array_equal(got, np.asarray(labels).astype(got.dtype)):
        raise RuntimeError(
            "Label ordering changed during extraction -- the loader must be "
            "order-preserving for the split identity to hold."
        )
    return feats, logits.numpy().astype(np.float64), got


def _compute_distances(
    bank_layers: Sequence[np.ndarray],
    bank_labels: np.ndarray,
    bank_labels_permuted: np.ndarray,
    query_features: Sequence[torch.Tensor],
    query_logits: np.ndarray,
    bank_logit_repr: np.ndarray,
    num_classes: int,
    kc_values: Sequence[int],
    dac_k: int,
    device: torch.device,
    knn_batch_size: int,
) -> Dict[str, np.ndarray]:
    """All class-conditioned distances plus native DAC's own global scores.

    Returns
    -------
    r_real      [N, L, C, n_kc]
    r_permuted  [N, L, C, n_kc]
    r_logit     [N, 1, C, n_kc]
    s_global    [N, L]  (class-AGNOSTIC k-th NN distance = native DAC's s_l)
    """
    n = query_logits.shape[0]
    n_layers = len(bank_layers)
    n_kc = len(kc_values)
    r_real = np.empty((n, n_layers, num_classes, n_kc), dtype=np.float32)
    r_perm = np.empty((n, n_layers, num_classes, n_kc), dtype=np.float32)
    s_global = np.empty((n, n_layers), dtype=np.float32)

    for li in range(n_layers):
        bank = torch.from_numpy(np.ascontiguousarray(bank_layers[li])).to(device).float()
        op_real = ClassConditionalKNN(bank, bank_labels, num_classes, device)
        op_perm = ClassConditionalKNN(bank, bank_labels_permuted, num_classes, device)
        outs, glob = class_distances_multi(
            [op_real, op_perm],
            dac_preprocess(query_features[li], device),
            kc_values,
            batch_size=knn_batch_size,
            also_global_k=dac_k,
        )
        r_real[:, li] = outs[0]
        r_perm[:, li] = outs[1]
        s_global[:, li] = glob
        del bank, op_real, op_perm, outs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Output-space mechanism control: one representation source of dim C.
    bank_z = torch.from_numpy(np.ascontiguousarray(bank_logit_repr)).to(device).float()
    op_z = ClassConditionalKNN(bank_z, bank_labels, num_classes, device)
    query_z = torch.from_numpy(centered_logit_representation(query_logits)).to(device)
    outs_z, _ = class_distances_multi(
        [op_z], query_z, kc_values, batch_size=knn_batch_size, also_global_k=None
    )
    r_logit = outs_z[0][:, None, :, :]
    del bank_z, op_z, query_z
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "r_real": r_real,
        "r_permuted": r_perm,
        "r_logit": r_logit,
        "s_global": s_global,
    }


# ==========================================================================
# Flip / intervention analysis (section 15 of the experiment spec)
# ==========================================================================


def _rank_of_truth(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """1-indexed rank of the ground-truth class (1 = predicted)."""
    truth_score = probs[np.arange(len(labels)), labels][:, None]
    return (probs > truth_score).sum(axis=1) + 1


def _binned_flip_rate(
    signal: np.ndarray, changed: np.ndarray, n_bins: int = 5
) -> Dict[str, Any]:
    """Flip rate by equal-mass bins of a per-sample signal."""
    if len(signal) == 0:
        return {}
    edges = np.quantile(signal, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    idx = np.clip(np.searchsorted(edges, signal, side="right") - 1, 0, n_bins - 1)
    out = {}
    for b in range(n_bins):
        m = idx == b
        out[f"bin{b}"] = {
            "n": int(m.sum()),
            "signal_mean": float(signal[m].mean()) if m.any() else None,
            "flip_rate": float(changed[m].mean()) if m.any() else None,
        }
    return out


def flip_analysis(
    base_probs: np.ndarray,
    method_probs: np.ndarray,
    labels: np.ndarray,
    s_dac: np.ndarray,
) -> Dict[str, Any]:
    """W / H / U decomposition plus the targeting diagnostics.

    The point of these is to separate useful targeted correction from
    indiscriminate flipping: a method that flips a lot and nets +0 is a very
    different object from one that flips rarely and nets +0.
    """
    base_pred = np.argmax(base_probs, axis=1)
    new_pred = np.argmax(method_probs, axis=1)
    n = len(labels)
    changed = base_pred != new_pred
    base_ok = base_pred == labels
    new_ok = new_pred == labels

    w = int(np.sum(changed & ~base_ok & new_ok))       # wrong -> correct
    h = int(np.sum(changed & base_ok & ~new_ok))       # correct -> wrong
    u = int(np.sum(changed & ~base_ok & ~new_ok))      # wrong -> different wrong
    f = w + h + u

    sorted_base = np.sort(base_probs, axis=1)
    top2_margin = sorted_base[:, -1] - sorted_base[:, -2]

    dest = new_pred[changed]
    if len(dest) > 0:
        counts = np.bincount(dest, minlength=base_probs.shape[1]).astype(np.float64)
        p = counts / counts.sum()
        nz = p[p > 0]
        entropy = float(-(nz * np.log(nz)).sum())
        top1_share = float(counts.max() / counts.sum())
    else:
        entropy, top1_share = None, None

    return {
        "total_flips": int(f),
        "wrong_to_correct": w,
        "correct_to_wrong": h,
        "wrong_to_different_wrong": u,
        "net_useful_flips": int(w - h),
        "argmax_change_rate": float(changed.mean()),
        "accuracy_delta": float(new_ok.mean() - base_ok.mean()),
        "accuracy_delta_from_flips": float((w - h) / n),
        "flip_identity_holds": bool(
            abs((new_ok.mean() - base_ok.mean()) - (w - h) / n) < 1e-12
        ),
        "intervention_precision": (float(w / f) if f > 0 else None),
        "decisive_precision": (float(w / (w + h)) if (w + h) > 0 else None),
        "fraction_base_errors_repaired": (
            float(w / int((~base_ok).sum())) if (~base_ok).any() else None
        ),
        "flip_rate_by_base_top2_margin": _binned_flip_rate(top2_margin, changed),
        "flip_rate_by_dac_S": _binned_flip_rate(np.asarray(s_dac, dtype=np.float64), changed),
        "gt_rank_before_mean": float(_rank_of_truth(base_probs, labels).mean()),
        "gt_rank_after_mean": float(_rank_of_truth(method_probs, labels).mean()),
        "destination_class_entropy": entropy,
        "destination_class_top1_share": top1_share,
        "n": int(n),
    }


# ==========================================================================
# Method entry construction (mirrors the benchmark's semantics discipline)
# ==========================================================================


def _method_entry(
    name: str,
    probs: np.ndarray,
    labels: np.ndarray,
    base_probs: np.ndarray,
    s_dac: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sem = require_method_semantics(name)
    validate_probability_matrix(probs, labels)
    entry: Dict[str, Any] = {
        "method_name": name,
        "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
        "output_semantics": sem["output_semantics"],
        "effective_prediction_source": sem["effective_prediction_source"],
        "metric_bucket": sem["metric_bucket"],
        "can_change_argmax": sem["can_change_argmax"],
        "structural_axes": structural_axes_for_method(name),
        "information_budget": sem.get("information_budget"),
        "metrics": evaluate_all(probs, labels),
        "decision_audit": decision_audit(base_probs, probs, labels),
        "flip_analysis": flip_analysis(base_probs, probs, labels, s_dac),
        "source": "fv_dac_runner",
    }
    if extra:
        entry.update(extra)
    return entry


def _reused_phase0_rows(phase0_cell_dir: str) -> List[Dict[str, Any]]:
    """Read the canonical benchmark's own rows for this cell, unmodified.

    FV-DAC never recomputes Temperature Scaling / Vector Scaling / ODIR /
    KCal: their frozen Phase 0/1 numbers are the comparison, and recomputing
    them here would risk producing a second, subtly different value for a
    method the benchmark already owns.
    """
    path = os.path.join(phase0_cell_dir, "summary_metrics.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception:  # pragma: no cover
        return []
    rows = []
    for m in payload.get("methods", []):
        if m.get("method_name") in REUSED_PHASE0_BASELINES:
            rows.append(
                {
                    "method_name": m["method_name"],
                    "metrics": m.get("metrics"),
                    "decision_audit": m.get("decision_audit"),
                    "source": "phase0_frozen",
                    "source_path": path,
                }
            )
    return rows


# ==========================================================================
# FIT STAGE
# ==========================================================================


def run_fit(args: argparse.Namespace) -> None:
    device = _resolve_device(args.device)

    clean_dir = _phase0_clean_dir(args.phase0_root, args.seed)
    splits = os.path.join(clean_dir, "intermediates", "splits")
    for key in ("train_raw", "train_labels", "val_raw", "val_labels", "test_raw", "test_labels"):
        p = os.path.join(splits, f"{key}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing materialized split {p}. FV-DAC reuses the frozen Phase 0/1 "
                "splits verbatim and never re-derives them."
            )

    dac_state_path = os.path.join(
        args.phase0_root, "fitted_method", f"checkpoint_seed{args.seed}", "native_dac.pkl"
    )
    if not os.path.exists(dac_state_path):
        raise FileNotFoundError(
            f"Frozen native DAC state not found at {dac_state_path}. FV-DAC extends the "
            "frozen benchmark DAC; it does not refit it."
        )

    t0 = time.perf_counter()
    logger.info("Loading frozen native DAC state: %s", dac_state_path)
    native_dac: DensityAwareCalibrator = _load_cuda_pickle_on_any_device(dac_state_path)
    _retarget_dac(native_dac, device)
    dac_layer_weights = np.asarray(native_dac.weights[:-1], dtype=np.float64)
    dac_intercept = float(native_dac.weights[-1])
    dac_k = int(native_dac.k)
    alpha, alpha_degenerate = normalized_layer_weights(dac_layer_weights)
    logger.info(
        "Frozen DAC: k=%d, layers=%d, w=%s, w0=%.6f, alpha=%s%s",
        dac_k, native_dac.num_layers, np.array2string(dac_layer_weights, precision=6),
        dac_intercept, np.array2string(alpha, precision=6),
        " (DEGENERATE -> uniform)" if alpha_degenerate else "",
    )

    train_raw = np.load(os.path.join(splits, "train_raw.npy"), mmap_mode="r")
    train_labels = np.load(os.path.join(splits, "train_labels.npy"))
    val_raw = np.load(os.path.join(splits, "val_raw.npy"), mmap_mode="r")
    val_labels = np.load(os.path.join(splits, "val_labels.npy"))
    test_raw = np.load(os.path.join(splits, "test_raw.npy"), mmap_mode="r")
    test_labels = np.load(os.path.join(splits, "test_labels.npy"))
    num_classes = int(train_labels.max()) + 1

    model_path = construct_model_path(
        args.results_dir, args.method, args.dataset, args.model, args.seed
    )
    model = load_trained_model(model_path, args.model, num_classes, device, dataset=args.dataset)
    dac_layers = get_dac_target_layers(args.model, model)
    if len(dac_layers) != native_dac.num_layers:
        raise RuntimeError(
            f"Layer-count mismatch: frozen DAC has {native_dac.num_layers} layers but "
            f"get_dac_target_layers returned {len(dac_layers)} ({dac_layers}). The "
            "frozen benchmark DAC is the canonical baseline; refusing to guess."
        )
    if get_dac_k_value(args.dataset) != dac_k:
        raise RuntimeError(
            f"k mismatch: frozen DAC k={dac_k}, dataset default "
            f"{get_dac_k_value(args.dataset)}."
        )

    # ---- extract ------------------------------------------------------
    logger.info("Extracting DAC features: train bank (%d rows)", len(train_labels))
    train_feats, train_logits, _ = _extract(
        model, np.asarray(train_raw), train_labels, dac_layers, device, args.batch_size
    )
    logger.info("Extracting DAC features: validation (%d rows)", len(val_labels))
    val_feats, val_logits, _ = _extract(
        model, np.asarray(val_raw), val_labels, dac_layers, device, args.batch_size
    )
    logger.info("Extracting DAC features: clean test (%d rows)", len(test_labels))
    test_feats, test_logits, _ = _extract(
        model, np.asarray(test_raw), test_labels, dac_layers, device, args.batch_size
    )

    # ---- label the FROZEN bank (rather than substituting our own) ------
    # The re-extracted features are used ONLY to recover the frozen bank's
    # labels; the frozen bank itself is what FV-DAC then uses, so S_DAC stays
    # exactly native DAC's. See _label_frozen_bank for why.
    mine = [
        dac_preprocess(f, torch.device("cpu")).numpy().astype(np.float32)
        for f in train_feats
    ]
    verification, bank_labels, bank_perm = _label_frozen_bank(
        native_dac, mine, train_labels, device,
        tolerance=args.bank_match_tolerance,
        mine_raw=np.asarray(train_raw),
    )
    logger.info("Frozen-bank labelling: %s", json.dumps(verification, default=_json_ready))

    bank_layers = [
        sc.train_features.detach().cpu().numpy().astype(np.float32)
        for sc in native_dac.layer_scorers
    ]
    del mine, train_feats
    # Reordered into FROZEN-bank order so row j of every bank -- including the
    # output-space control's -- is the same example as bank_labels[j].
    bank_logit_repr = centered_logit_representation(train_logits)[bank_perm]

    # ---- persist the labelled bank ------------------------------------
    state_dir = os.path.join(args.fv_dac_state_dir, f"checkpoint_seed{args.seed}")
    bank_dir = os.path.join(state_dir, "bank")
    os.makedirs(bank_dir, exist_ok=True)
    bank_feature_hashes, bank_set_hashes, bank_dims = [], [], []
    for li, b in enumerate(bank_layers):
        np.save(os.path.join(bank_dir, f"layer{li}.npy"), b)
        bank_feature_hashes.append(array_hash(b))
        bank_set_hashes.append(bank_set_hash(b, decimals=args.bank_hash_decimals))
        bank_dims.append(int(b.shape[1]))
    np.save(os.path.join(bank_dir, "bank_labels.npy"), bank_labels.astype(np.int64))
    np.save(os.path.join(bank_dir, "bank_logit_repr.npy"), bank_logit_repr)

    permuted_seed = PERMUTED_LABEL_SEED_BASE + int(args.seed)
    bank_labels_permuted = permuted_bank_labels(bank_labels, permuted_seed)
    np.save(os.path.join(bank_dir, "bank_labels_permuted.npy"), bank_labels_permuted)

    # ---- distances on clean val / clean test --------------------------
    logger.info("Computing class-conditioned distances (validation)")
    d_val = _compute_distances(
        bank_layers, bank_labels, bank_labels_permuted, val_feats, val_logits,
        bank_logit_repr, num_classes, PREDECLARED_KC, dac_k, device, args.knn_batch_size,
    )
    logger.info("Computing class-conditioned distances (clean test)")
    d_test = _compute_distances(
        bank_layers, bank_labels, bank_labels_permuted, test_feats, test_logits,
        bank_logit_repr, num_classes, PREDECLARED_KC, dac_k, device, args.knn_batch_size,
    )

    s_val = dac_temperature(d_val["s_global"], dac_layer_weights, dac_intercept)
    s_test = dac_temperature(d_test["s_global"], dac_layer_weights, dac_intercept)

    # ---- native DAC reproduction check --------------------------------
    native_probs_test = np_softmax(test_logits / s_test[:, None])
    repro = _verify_native_dac_reproduction(
        native_dac, test_feats, test_logits, native_probs_test, args.native_dac_tolerance
    )
    logger.info("Native DAC reproduction: %s", json.dumps(repro))

    del native_dac
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- split roles: fit beta on inner-FIT, select K_c on inner-SELECT
    fit_idx, select_idx = make_inner_validation_split(
        val_labels, select_fraction=args.inner_val_fraction, seed=args.inner_val_seed
    )
    if len(np.intersect1d(fit_idx, select_idx)) != 0:
        raise RuntimeError("inner fit/select split is not disjoint")
    logger.info(
        "Split roles -- bank: %d | beta-fit: %d | K_c-select: %d | test: %d",
        len(train_labels), len(fit_idx), len(select_idx), len(test_labels),
    )

    kc_table, selected_kc = _select_kc(
        val_logits, d_val, s_val, val_labels, alpha, fit_idx, select_idx, args
    )
    logger.info("Selected K_c = %d (clean inner-select NLL, primary arm)", selected_kc)

    # ---- fit every arm at the selected K_c ----------------------------
    kc_pos = list(PREDECLARED_KC).index(selected_kc)
    r_fit = d_val["r_real"][fit_idx, :, :, kc_pos]
    r_fit_perm = d_val["r_permuted"][fit_idx, :, :, kc_pos]
    r_fit_logit = d_val["r_logit"][fit_idx, :, :, kc_pos]
    z_fit = val_logits[fit_idx]
    s_fit = s_val[fit_idx]
    y_fit = val_labels[fit_idx]

    mu, sigma = lognorm_statistics(r_fit)
    v_fit = lognorm_transform(r_fit, mu, sigma)

    fitted: Dict[str, Any] = {}
    fitted[ARM_PRIMARY] = guarded_fit_beta(
        z_fit, aggregate_R(r_fit, alpha), s_fit, y_fit, objective="nll", sign=-1.0,
        beta_max=args.beta_max,
    )
    fitted[ARM_BRIER] = guarded_fit_beta(
        z_fit, aggregate_R(r_fit, alpha), s_fit, y_fit, objective="brier", sign=-1.0,
        beta_max=args.beta_max,
    )
    fitted[ARM_LOGNORM] = guarded_fit_beta(
        z_fit, aggregate_R(v_fit, alpha), s_fit, y_fit, objective="nll", sign=+1.0,
        beta_max=args.beta_max,
    )
    fitted[ARM_SHARED_LAYER] = guarded_fit_shared_layer(
        z_fit, r_fit, s_fit, y_fit, objective="nll", b_max=args.beta_max,
    )
    fitted[ARM_PERMUTED] = guarded_fit_beta(
        z_fit, aggregate_R(r_fit_perm, alpha), s_fit, y_fit, objective="nll", sign=-1.0,
        beta_max=args.beta_max,
    )
    fitted[ARM_LOGIT_SPACE] = guarded_fit_beta(
        z_fit, aggregate_R(r_fit_logit, np.ones(1)), s_fit, y_fit, objective="nll",
        sign=-1.0, beta_max=args.beta_max,
    )
    for arm, params in fitted.items():
        logger.info("fitted %-22s %s", arm, json.dumps(params, default=_json_ready))

    # ---- freeze --------------------------------------------------------
    state = FVDACFrozenState(
        checkpoint_seed=int(args.seed),
        dataset=args.dataset,
        model=args.model,
        checkpoint_path=model_path,
        dac_layers=list(dac_layers),
        dac_k=dac_k,
        dac_layer_weights=dac_layer_weights.tolist(),
        dac_intercept=dac_intercept,
        dac_state_path=dac_state_path,
        dac_state_sha256=_sha256_file(dac_state_path) if args.hash_dac_state else "skipped",
        alpha=alpha.tolist(),
        alpha_degenerate=bool(alpha_degenerate),
        selected_kc=int(selected_kc),
        predeclared_kc=list(PREDECLARED_KC),
        kc_selection_table=kc_table,
        fitted=fitted,
        fitting_objective_primary="nll",
        lognorm_mu=mu.tolist(),
        lognorm_sigma=sigma.tolist(),
        bank_dir=bank_dir,
        bank_size=int(len(bank_labels)),
        bank_layer_dims=bank_dims,
        bank_feature_hashes=bank_feature_hashes,
        bank_labels_hash=array_hash(bank_labels.astype(np.int64)),
        bank_set_hashes=bank_set_hashes,
        permuted_label_seed=int(permuted_seed),
        permuted_labels_hash=array_hash(bank_labels_permuted),
        preprocessing=(
            "clean train/val/test: the frozen Phase 0/1 materialized splits "
            "(CIFAR normalization); CIFAR-100-C: ToTensor + ImageNet normalization, "
            "verbatim from run_unified_benchmark.py"
        ),
        split_roles={
            "reference_bank": "clean train (45k) -- features and labels",
            "beta_fit": f"clean validation inner-FIT ({len(fit_idx)} rows)",
            "kc_select": f"clean validation inner-SELECT ({len(select_idx)} rows)",
            "evaluation": "clean test + CIFAR-100-C (evaluation only)",
            "inner_val_fraction": float(args.inner_val_fraction),
            "inner_val_seed": int(args.inner_val_seed),
            "fit_select_disjoint": True,
            "declared_deviation": (
                "native DAC's own weights were fitted (in the frozen Phase 0/1 run) on "
                "the WHOLE validation split, including the inner-select half. Those "
                "weights are frozen and identical across all K_c candidates, so they "
                "cannot bias the K_c comparison, but the inner-select NLL is not "
                "perfectly independent of them."
            ),
        },
        git_commit=_git_commit(PROJECT_ROOT),
        created_at=_iso_now(),
        notes="FV-DAC frozen state; beta fitted on clean inner-fit only, never refit.",
    )

    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, "fv_dac_state.json")
    _atomic_write_json(state_path, json.loads(state.to_json()))
    _atomic_write_json(
        os.path.join(state_dir, "fv_dac_state_hash.json"),
        {
            "state_hash": state.state_hash(),
            "state_path": state_path,
            "created_at": _iso_now(),
            "git_commit": state.git_commit,
            "git_dirty": _git_dirty(PROJECT_ROOT),
            "bank_verification": verification,
            "native_dac_reproduction": repro,
        },
    )
    logger.info("Frozen FV-DAC state hash: %s", state.state_hash())

    # ---- clean test evaluation -----------------------------------------
    _evaluate_cell(
        args=args,
        state=state,
        cell=_cell_name(None, None),
        logits=test_logits,
        labels=test_labels,
        distances=d_test,
        kc_pos=kc_pos,
        s_dac=s_test,
        extra_metadata={
            "bank_verification": verification,
            "native_dac_reproduction": repro,
            "kc_selection_table": kc_table,
            "fit_seconds": float(time.perf_counter() - t0),
        },
    )
    logger.info("FIT stage complete in %.1fs", time.perf_counter() - t0)


def _label_frozen_bank(
    native_dac: DensityAwareCalibrator,
    mine: Sequence[np.ndarray],
    mine_labels: np.ndarray,
    device: torch.device,
    *,
    tolerance: float,
    mine_raw: Optional[np.ndarray] = None,
    max_ambiguous_fraction: float = 0.005,
    duplicate_pixel_fraction: float = 0.05,
    batch_size: int = 512,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Recover labels for the FROZEN DAC bank, and use that bank unchanged.

    Why not simply use a freshly extracted bank? Because then `S_DAC` would be
    *our* temperature, not native DAC's. Convolutions on Ampere+ run in TF32 by
    default (`torch.backends.cudnn.allow_tf32 = True`), whose 10-bit mantissa
    plus batch-size-dependent algorithm selection makes the same image's
    feature differ by ~1e-3 between processes. The first seed-4 run measured
    exactly that: a max nearest-match distance of 9.1e-4 at `conv1`. A bank
    perturbed at that scale shifts the k-th-neighbour distance, hence S(x),
    hence the probabilities -- so `beta = 0` would reproduce native DAC only
    approximately. The experiment's matched operating point requires it
    exactly.

    So the frozen bank IS the bank. The re-extracted features are used only to
    recover which label belongs to which frozen row, by a sparse global
    minimum-cost bijection in an equal-weight concatenation of ALL normalized
    DAC layers and then verifying the SAME permutation layer by layer. A
    greedy nearest map is not enough: the first seed-4 attempt produced 11
    duplicate assignments, despite equal bank cardinality. Requirements, all
    fatal:

      * the minimum-cost map is a full bijection over each row's eight nearest
        candidates (with equal set sizes this is a set-equality statement);
      * its max distance is <= tolerance;
      * the recovered LABELS are unique: every candidate within `tolerance`
        of a frozen row carries the same label as the assigned one. (A
        geometric margin criterion was tried first and is unusable --
        near-duplicate CIFAR-100 train images sit closer to each other than
        the TF32 perturbation separating a row from its own counterpart. A
        swap between two near-duplicates is inert unless it moves a label.)
      * the same permutation matches within tolerance at every layer, which
        is what rules out a coincidental re-pairing.

    Returns (report, labels_in_frozen_bank_order, permutation), where
    `permutation[j]` is the re-extracted row matching frozen row `j` -- needed
    to put the logit-space control's own bank into frozen-bank order too.
    """
    dims = [b.shape[1] for b in mine]
    report: Dict[str, Any] = {
        "method": "sparse minimum-cost bijection on equal-weight concatenated normalized DAC layers",
        "match_layers": list(range(len(mine))),
        "match_layer_dims": dims,
        "candidates_per_frozen_row": 8,
        "tolerance": float(tolerance),
        "rationale": (
            "TF32 convolutions make cross-process feature extraction reproducible "
            "only to ~1e-3, so the frozen bank is used as-is and only its labels "
            "are recovered."
        ),
    }

    for li, b in enumerate(mine):
        frozen_shape = tuple(native_dac.layer_scorers[li].train_features.shape)
        if frozen_shape != b.shape:
            raise RuntimeError(
                f"layer {li}: frozen bank shape {frozen_shape} != re-extracted "
                f"{b.shape}; the reference-bank membership is not the same."
            )

    # Each scorer already stores L2-normalized rows.  Concatenating layers
    # after a 1/sqrt(L) factor preserves unit norm and makes every DAC source
    # contribute equally, regardless of its feature dimensionality.
    n_layers = len(mine)
    scale = float(n_layers) ** -0.5
    frozen_parts = [
        scorer.train_features.to(device).float() * scale
        for scorer in native_dac.layer_scorers
    ]
    mine_parts = [
        torch.from_numpy(np.ascontiguousarray(b)).to(device).float() * scale
        for b in mine
    ]
    frozen = torch.cat(frozen_parts, dim=1)
    mine_t = torch.cat(mine_parts, dim=1)
    n_rows = int(frozen.shape[0])

    n_candidates = int(report["candidates_per_frozen_row"])
    best, second, candidate_dist, candidate_idx = [], [], [], []
    for start in range(0, n_rows, batch_size):
        d = normalized_euclidean(frozen[start : start + batch_size], mine_t)
        nearest = torch.topk(d, k=n_candidates, dim=1, largest=False)
        best.append(nearest.values[:, 0])
        second.append(nearest.values[:, 1])
        candidate_dist.append(nearest.values)
        candidate_idx.append(nearest.indices)
        del d, nearest
    best_t, second_t = torch.cat(best), torch.cat(second)
    candidate_dist_t = torch.cat(candidate_dist).cpu().numpy()
    candidate_idx_t = torch.cat(candidate_idx).cpu().numpy()
    greedy_idx = candidate_idx_t[:, 0]
    greedy_unique = int(np.unique(greedy_idx).size)

    # The true correspondence is a permutation. Greedy row-wise nearest
    # matching can assign two frozen rows to one materialized row under small
    # TF32 perturbations, so solve the explicitly constrained global problem.
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), n_candidates)
    cols = candidate_idx_t.reshape(-1).astype(np.int64, copy=False)
    costs = candidate_dist_t.reshape(-1).astype(np.float64, copy=False)
    # scipy's sparse matcher treats an exact zero as absent; distances are
    # non-negative, so this harmless offset preserves every ordering.
    graph = csr_matrix((costs + 1e-12, (rows, cols)), shape=(n_rows, n_rows))
    try:
        row_ind, col_ind = min_weight_full_bipartite_matching(graph)
    except ValueError as exc:
        raise RuntimeError(
            "frozen-bank labelling failed: no full bijection exists within the "
            f"top-{n_candidates} multi-layer candidates per row. Refusing to guess labels."
        ) from exc
    if not np.array_equal(row_ind, np.arange(n_rows)):
        raise RuntimeError("sparse assignment did not return every frozen-bank row")
    perm = col_ind.astype(np.int64, copy=False)
    assigned = candidate_dist_t[np.arange(n_rows),
                                (candidate_idx_t == perm[:, None]).argmax(axis=1)]
    max_match = float(assigned.max())
    margin = float(second_t.min())
    del frozen, mine_t, frozen_parts, mine_parts, best, second, candidate_dist, candidate_idx, best_t, second_t, graph
    if device.type == "cuda":
        torch.cuda.empty_cache()

    report.update(
        {
            "n_rows": n_rows,
            "max_match_distance": max_match,
            "min_second_nearest_distance": margin,
            "greedy_n_unique_matches": greedy_unique,
            "bijection": bool(np.unique(perm).size == n_rows),
            "separation_ratio": (max_match / margin) if margin > 0 else None,
        }
    )
    logger.info("frozen-bank match: %s", json.dumps(report, default=_json_ready))

    if np.unique(perm).size != n_rows:
        raise RuntimeError(
            "frozen-bank labelling failed: the global assignment is not a bijection. "
            "Refusing to guess labels."
        )
    if max_match > tolerance:
        raise RuntimeError(
            f"frozen-bank labelling failed: max match distance {max_match:.3e} > "
            f"{tolerance:.3e} in the multi-layer representation. The materialized train split "
            "does not appear to be the same example set as the frozen DAC bank."
        )
    # Decisiveness is a LABEL question, not a geometric one.
    #
    # An earlier version required max_match << min_second_nearest_distance.
    # That criterion is unsatisfiable here and, more importantly, wrong: the
    # seed-4 run measured min_second_nearest_distance = 1.381e-3 against a
    # worst true-match distance of 3.435e-3, i.e. CIFAR-100's train split
    # contains genuine NEAR-DUPLICATE images that sit closer to each other
    # than the TF32 perturbation separating a row from its own counterpart.
    # No geometric margin test can ever pass in the presence of duplicates.
    #
    # But a swap between two near-duplicate rows is scientifically inert
    # unless it changes a LABEL: the two feature vectors differ by ~1e-3, far
    # below any scale the class-conditioned distances resolve, so B_{l,k} is
    # bit-for-bit the same set either way. What must be established is that
    # every plausible match for a frozen row carries the SAME label -- then
    # the recovered labelling is unique regardless of how ties break.
    mine_labels_arr = np.asarray(mine_labels).astype(np.int64)
    assigned_labels = mine_labels_arr[perm]
    plausible = candidate_dist_t <= tolerance
    candidate_labels = mine_labels_arr[candidate_idx_t]
    label_conflict = plausible & (candidate_labels != assigned_labels[:, None])
    n_ambiguous = int((plausible.sum(axis=1) > 1).sum())
    conflict_rows = np.flatnonzero(label_conflict.any(axis=1))

    report.update(
        {
            "decisiveness_criterion": (
                "every candidate within `tolerance` of a frozen row carries the same "
                "label as the assigned one (label-level determinacy). A geometric "
                "margin criterion is not usable: near-duplicate CIFAR-100 train "
                "images sit closer together than the TF32 perturbation."
            ),
            "n_geometrically_ambiguous_rows": n_ambiguous,
            "n_label_conflicting_rows": int(conflict_rows.size),
            "labels_uniquely_determined": bool(conflict_rows.size == 0),
        }
    )

    if conflict_rows.size:
        # A label-ambiguous row means two candidate source rows sit within
        # `tolerance` of the same frozen row but disagree on the class. Before
        # accepting or rejecting, establish what those rows ARE, in pixels:
        # CIFAR-100's train split is known to contain duplicate images, and a
        # duplicate pair is something NO feature-based matcher could ever
        # separate. This is a fact about the dataset, not about our matching.
        if mine_raw is None:
            raise RuntimeError(
                "frozen-bank labelling is not decisive and no raw images were "
                f"supplied to diagnose it: {conflict_rows.size} of {n_rows} rows "
                "have a plausible match carrying a different label."
            )
        raw = np.asarray(mine_raw).reshape(len(mine_labels_arr), -1)
        conflict_partner = np.where(
            label_conflict[conflict_rows], np.arange(label_conflict.shape[1]), 10**9
        ).min(axis=1)
        partner_idx = candidate_idx_t[conflict_rows, conflict_partner]
        assigned_idx = perm[conflict_rows]
        pair_pixel = np.linalg.norm(
            raw[assigned_idx].astype(np.float64) - raw[partner_idx].astype(np.float64),
            axis=1,
        )
        rng = np.random.default_rng(0)
        probe = rng.integers(0, raw.shape[0], size=(2000, 2))
        baseline = float(
            np.median(
                np.linalg.norm(
                    raw[probe[:, 0]].astype(np.float64) - raw[probe[:, 1]].astype(np.float64),
                    axis=1,
                )
            )
        )
        duplicate_cut = duplicate_pixel_fraction * baseline
        all_duplicates = bool(np.all(pair_pixel <= duplicate_cut))
        fraction = float(conflict_rows.size) / float(n_rows)

        report.update(
            {
                "ambiguity_diagnosis": {
                    "n_label_conflicting_rows": int(conflict_rows.size),
                    "fraction_of_bank": fraction,
                    "max_conflict_pair_pixel_distance": float(pair_pixel.max()),
                    "median_conflict_pair_pixel_distance": float(np.median(pair_pixel)),
                    "median_random_pair_pixel_distance": baseline,
                    "duplicate_cut": duplicate_cut,
                    "all_conflict_pairs_are_duplicate_images": all_duplicates,
                    "example_rows": conflict_rows[:10].tolist(),
                },
                "worst_case_bank_perturbation": 2.0 * float(tolerance),
                "ambiguity_impact_bound": (
                    "Both candidates lie within `tolerance` of the same frozen row, so "
                    "by the triangle inequality they differ from each other by at most "
                    "2*tolerance. A label swap between them therefore perturbs the "
                    "affected class-conditional bank B_{l,k} by at most 2*tolerance in "
                    "ONE member out of ~450 -- the same order as the TF32 perturbation "
                    "already accepted throughout, and far below any scale the "
                    "class-conditioned K_c-th neighbour distance resolves."
                ),
            }
        )
        logger.warning(
            "frozen-bank labelling: %d/%d rows (%.4f%%) are label-ambiguous; "
            "conflict pairs are duplicate images: %s (max pixel distance %.4g vs "
            "random-pair median %.4g)",
            conflict_rows.size, n_rows, 100.0 * fraction, all_duplicates,
            float(pair_pixel.max()), baseline,
        )

        if fraction > max_ambiguous_fraction:
            raise RuntimeError(
                f"frozen-bank labelling is not decisive: {conflict_rows.size} of "
                f"{n_rows} rows ({100.0 * fraction:.3f}%) are label-ambiguous, above "
                f"the {100.0 * max_ambiguous_fraction:.3f}% ceiling. Refusing to "
                "proceed on a labelling this uncertain."
            )
        if not all_duplicates:
            raise RuntimeError(
                "frozen-bank labelling is not decisive: some label-conflicting "
                f"candidate pairs are NOT duplicate images (max pixel distance "
                f"{float(pair_pixel.max()):.4g} > {duplicate_cut:.4g}). That points at "
                "a genuine matching failure rather than dataset duplicates; refusing "
                "to guess labels."
            )

    del candidate_dist_t, candidate_idx_t, plausible, candidate_labels, label_conflict

    # Cross-check: the SAME permutation must also match at every individual layer.
    cross = []
    for li, b in enumerate(mine):
        fz = native_dac.layer_scorers[li].train_features.to(device).float()
        mt = torch.from_numpy(np.ascontiguousarray(b[perm])).to(device).float()
        worst = 0.0
        for start in range(0, n_rows, 4096):
            stop = min(start + 4096, n_rows)
            d = torch.linalg.vector_norm(fz[start:stop] - mt[start:stop], dim=1)
            worst = max(worst, float(d.max()))
            del d
        cross.append({"layer": li, "max_paired_distance": worst})
        del fz, mt
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if worst > tolerance:
            raise RuntimeError(
                f"frozen-bank labelling cross-check failed at layer {li}: the "
                "permutation found in the multi-layer representation pairs rows that differ "
                f"by up to {worst:.3e} > {tolerance:.3e} there."
            )
    report["cross_layer_checks"] = cross
    report["permutation_hash"] = array_hash(perm.astype(np.int64))
    report["all_checks_passed"] = True

    return report, np.asarray(mine_labels)[perm].astype(np.int64), perm


def _verify_native_dac_reproduction(
    native_dac: DensityAwareCalibrator,
    test_feats: Sequence[torch.Tensor],
    test_logits: np.ndarray,
    reproduced_probs: np.ndarray,
    tolerance: float,
) -> Dict[str, Any]:
    """softmax(z / S) from the re-extracted bank must equal frozen native DAC."""
    frozen_probs = native_dac.calibrate([f.numpy() for f in test_feats], test_logits)
    max_abs = float(np.max(np.abs(frozen_probs - reproduced_probs)))
    argmax_agreement = float(
        np.mean(np.argmax(frozen_probs, axis=1) == np.argmax(reproduced_probs, axis=1))
    )
    ok = max_abs <= tolerance
    result = {
        "max_abs_prob_difference": max_abs,
        "argmax_agreement": argmax_agreement,
        "tolerance": float(tolerance),
        "passed": bool(ok),
    }
    if not ok:
        raise RuntimeError(
            "Re-extracted bank does not reproduce the frozen native DAC "
            f"(max |dp| = {max_abs:.3e} > {tolerance:.3e}). beta=0 would then NOT "
            "reproduce native DAC, which is the experiment's matched operating point."
        )
    return result


def _select_kc(
    val_logits: np.ndarray,
    d_val: Dict[str, np.ndarray],
    s_val: np.ndarray,
    val_labels: np.ndarray,
    alpha: np.ndarray,
    fit_idx: np.ndarray,
    select_idx: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], int]:
    """Fit beta per K_c on inner-FIT; compare on the disjoint inner-SELECT.

    The primary arm alone decides K_c. Sensitivity arms and controls inherit
    it -- they are predeclared scientific arms, never candidates to be
    promoted after the fact.
    """
    table: Dict[str, Any] = {"rule": "argmin clean inner-select NLL of fv_dac_nll", "candidates": {}}
    best_kc, best_nll = None, np.inf
    for pos, kc in enumerate(PREDECLARED_KC):
        r_fit = d_val["r_real"][fit_idx, :, :, pos]
        r_sel = d_val["r_real"][select_idx, :, :, pos]
        params = guarded_fit_beta(
            val_logits[fit_idx], aggregate_R(r_fit, alpha), s_val[fit_idx],
            val_labels[fit_idx], objective="nll", sign=-1.0, beta_max=args.beta_max,
        )
        probs_sel = fv_dac_probs(
            val_logits[select_idx], aggregate_R(r_sel, alpha), s_val[select_idx],
            float(params["beta"]), sign=-1.0,
        )
        sel_nll = nll_of(probs_sel, val_labels[select_idx])
        sel_acc = float(
            np.mean(np.argmax(probs_sel, axis=1) == val_labels[select_idx])
        )
        table["candidates"][str(kc)] = {
            "beta": params["beta"],
            "fit_nll": params["objective_value"],
            "fit_nll_at_beta_zero": params["objective_value_at_zero"],
            "select_nll": sel_nll,
            "select_accuracy": sel_acc,
            "at_upper_boundary": params["at_upper_boundary"],
        }
        logger.info(
            "K_c=%-4d beta=%.6g  inner-fit NLL=%.6f  inner-select NLL=%.6f  sel acc=%.4f",
            kc, params["beta"], params["objective_value"], sel_nll, sel_acc,
        )
        if sel_nll < best_nll:
            best_nll, best_kc = sel_nll, kc
    table["selected_kc"] = int(best_kc)
    table["selected_select_nll"] = float(best_nll)
    return table, int(best_kc)


# ==========================================================================
# EVALUATE STAGE (corruption cells -- evaluation only)
# ==========================================================================


def run_evaluate(args: argparse.Namespace) -> None:
    if args.corruption_type is None or args.corruption_severity is None:
        raise ValueError("--stage evaluate requires --corruption_type and --corruption_severity")

    state_dir = os.path.join(args.fv_dac_state_dir, f"checkpoint_seed{args.seed}")
    state_path = os.path.join(state_dir, "fv_dac_state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"FAIL: frozen FV-DAC state missing at {state_path}. Corruption evaluation "
            "never fits -- run --stage fit for this checkpoint first."
        )

    # Installed BEFORE any corrupted byte is read.
    _FitGuard.block(
        f"evaluation-only corruption cell "
        f"{args.corruption_type} severity={args.corruption_severity}"
    )

    with open(state_path) as f:
        payload = json.load(f)
    state = FVDACFrozenState.from_dict(payload)
    recorded_hash_path = os.path.join(state_dir, "fv_dac_state_hash.json")
    recorded_hash = None
    if os.path.exists(recorded_hash_path):
        with open(recorded_hash_path) as f:
            recorded_hash = json.load(f).get("state_hash")
    if recorded_hash and recorded_hash != state.state_hash():
        raise RuntimeError(
            "Frozen FV-DAC state hash mismatch: the state file changed after it was "
            f"frozen ({recorded_hash} != {state.state_hash()}). Refusing to evaluate."
        )

    device = _resolve_device(args.device)

    bank_dir = state.bank_dir
    bank_layers = [
        np.load(os.path.join(bank_dir, f"layer{li}.npy"))
        for li in range(len(state.dac_layers))
    ]
    bank_labels = np.load(os.path.join(bank_dir, "bank_labels.npy"))
    bank_labels_permuted = np.load(os.path.join(bank_dir, "bank_labels_permuted.npy"))
    bank_logit_repr = np.load(os.path.join(bank_dir, "bank_logit_repr.npy"))
    if array_hash(bank_labels.astype(np.int64)) != state.bank_labels_hash:
        raise RuntimeError("reference-bank label hash mismatch against the frozen state")
    if array_hash(bank_labels_permuted) != state.permuted_labels_hash:
        raise RuntimeError("permuted-label hash mismatch against the frozen state")
    num_classes = int(bank_labels.max()) + 1

    loader = load_cifar_c_loader(
        dataset_name=args.dataset,
        corruption=args.corruption_type,
        severity=args.corruption_severity,
        test_transform=_corruption_transform(),
        batch_size=args.batch_size,
        cifar_c_dir=args.cifar_c_dir,
    )
    raw, labels = get_all_data_as_numpy(loader)
    logger.info(
        "Corruption cell %s severity=%d: %d rows",
        args.corruption_type, args.corruption_severity, len(labels),
    )

    # Optional integrity check against the benchmark's own materialized cell.
    cell = _cell_name(args.corruption_type, args.corruption_severity)
    bench_raw = os.path.join(
        _phase0_cell_dir(args.phase0_root, args.seed, cell),
        "intermediates", "splits", "test_raw.npy",
    )
    cell_matches_benchmark = None
    if os.path.exists(bench_raw):
        ref = np.load(bench_raw, mmap_mode="r")
        cell_matches_benchmark = bool(
            ref.shape == raw.shape and np.allclose(np.asarray(ref), raw, atol=1e-6)
        )
        logger.info("Cell bytes match the frozen benchmark cell: %s", cell_matches_benchmark)
        if not cell_matches_benchmark:
            raise RuntimeError(
                f"The corrupted cell built here differs from the benchmark's own "
                f"materialized {bench_raw}. FV-DAC must evaluate the identical data."
            )

    model = load_trained_model(
        state.checkpoint_path, args.model, num_classes, device, dataset=args.dataset
    )
    feats, logits, _ = _extract(
        model, raw, labels, state.dac_layers, device, args.batch_size
    )

    distances = _compute_distances(
        bank_layers, bank_labels, bank_labels_permuted, feats, logits, bank_logit_repr,
        num_classes, [state.selected_kc], int(state.dac_k), device, args.knn_batch_size,
    )
    s_dac = dac_temperature(
        distances["s_global"], state.dac_layer_weights, state.dac_intercept
    )

    _evaluate_cell(
        args=args,
        state=state,
        cell=cell,
        logits=logits,
        labels=labels,
        distances=distances,
        kc_pos=0,
        s_dac=s_dac,
        extra_metadata={
            "cell_matches_benchmark_materialized_split": cell_matches_benchmark,
            "fit_guard_blocked": _FitGuard.is_blocked(),
        },
    )


# ==========================================================================
# Shared evaluation / artifact writing
# ==========================================================================


def _evaluate_cell(
    *,
    args: argparse.Namespace,
    state: FVDACFrozenState,
    cell: str,
    logits: np.ndarray,
    labels: np.ndarray,
    distances: Dict[str, np.ndarray],
    kc_pos: int,
    s_dac: np.ndarray,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    out_dir = os.path.join(args.output_dir, f"checkpoint_seed{args.seed}", cell)
    os.makedirs(out_dir, exist_ok=True)

    r_real = distances["r_real"][:, :, :, kc_pos]
    r_perm = distances["r_permuted"][:, :, :, kc_pos]
    r_logit = distances["r_logit"][:, :, :, kc_pos]
    alpha = np.asarray(state.alpha, dtype=np.float64)

    R_real = aggregate_R(r_real, alpha)
    base_probs = np_softmax(logits)
    native_probs = np_softmax(logits / s_dac[:, None])

    probs_by_method: Dict[str, np.ndarray] = {
        "base_model": base_probs,
        "native_dac": native_probs,
    }
    for arm in FITTED_ARMS:
        probs_by_method[arm] = apply_arm(
            arm, state,
            logits=logits, temperature=s_dac,
            r_layers_real=r_real, r_layers_permuted=r_perm, r_logit_space=r_logit,
        )

    # beta = 0 MUST reproduce native DAC, numerically. Checked every cell,
    # not once at implementation time.
    zero_beta_probs = fv_dac_probs(logits, R_real, s_dac, 0.0, sign=-1.0)
    beta_zero_max_diff = float(np.max(np.abs(zero_beta_probs - native_probs)))
    if beta_zero_max_diff > 1e-12:
        raise RuntimeError(
            f"beta=0 does not reproduce native DAC (max |dp| = {beta_zero_max_diff:.3e})."
        )

    methods: List[Dict[str, Any]] = []
    for name, probs in probs_by_method.items():
        extra: Dict[str, Any] = {}
        if name in state.fitted:
            extra["fitted_parameters"] = state.fitted[name]
            extra["n_fitted_parameters"] = (
                len(state.fitted[name]["b"]) if name == ARM_SHARED_LAYER else 1
            )
            extra["selected_kc"] = int(state.selected_kc)
        if name == "native_dac":
            extra["dac_layers"] = list(state.dac_layers)
            extra["dac_k"] = int(state.dac_k)
        methods.append(
            _method_entry(name, probs, labels, base_probs, s_dac, extra=extra)
        )

    # ---- density-only diagnostic (never a fitted competitor) -----------
    dens_pred = density_only_prediction(r_real, alpha)
    base_pred = np.argmax(base_probs, axis=1)
    fv_pred = np.argmax(probs_by_method[ARM_PRIMARY], axis=1)
    base_wrong = base_pred != labels
    density_only = {
        "method_name": ARM_DENSITY_ONLY,
        "accuracy": float(np.mean(dens_pred == labels)),
        "agreement_with_base": float(np.mean(dens_pred == base_pred)),
        "agreement_with_fv_dac_primary": float(np.mean(dens_pred == fv_pred)),
        "recoverability_among_base_errors": (
            float(np.mean(dens_pred[base_wrong] == labels[base_wrong]))
            if base_wrong.any() else None
        ),
        "note": "diagnostic only: argmin_k R_k(x). Detects whether FV-DAC is merely "
                "interpolating toward a nearest-density classifier.",
    }

    representation_capacity = {
        "hidden": {
            "n_sources": len(state.dac_layers),
            "source_names": list(state.dac_layers),
            "dims": list(state.bank_layer_dims),
            "bank_size": int(state.bank_size),
            "kc": int(state.selected_kc),
            "n_fitted_params": 1,
        },
        "logit_space": {
            "n_sources": 1,
            "source_names": ["centered_logits"],
            "dims": [int(logits.shape[1])],
            "bank_size": int(state.bank_size),
            "kc": int(state.selected_kc),
            "n_fitted_params": 1,
        },
        "caveat": (
            "The logit-space arm is a MECHANISM control, not a strictly "
            "capacity-matched control: it uses 1 output-space source while the "
            "hidden arm aggregates several internal layers. A positive "
            "hidden-vs-logit result must be reported with this asymmetry stated."
        ),
    }

    summary = {
        "metadata": {
            "experiment": "fv_dac",
            "cell": cell,
            "dataset": args.dataset,
            "model": args.model,
            "checkpoint_seed": int(args.seed),
            "checkpoint_path": state.checkpoint_path,
            "corruption_type": args.corruption_type,
            "corruption_severity": args.corruption_severity,
            "n_samples": int(len(labels)),
            "n_classes": int(logits.shape[1]),
            "fv_dac_state_hash": state.state_hash(),
            "selected_kc": int(state.selected_kc),
            "predeclared_kc": list(state.predeclared_kc),
            "alpha": state.alpha,
            "alpha_degenerate": bool(state.alpha_degenerate),
            "dac_layer_weights": state.dac_layer_weights,
            "dac_intercept": state.dac_intercept,
            "split_roles": state.split_roles,
            "representation_capacity": representation_capacity,
            "beta_zero_reproduces_native_dac_max_abs_diff": beta_zero_max_diff,
            "git_commit": _git_commit(PROJECT_ROOT),
            "git_dirty": _git_dirty(PROJECT_ROOT),
            "created_at": _iso_now(),
            "evaluation_only": bool(_FitGuard.is_blocked()),
            **(extra_metadata or {}),
        },
        "methods": methods,
        "reused_phase0_baselines": _reused_phase0_rows(
            _phase0_cell_dir(args.phase0_root, args.seed, cell)
        ),
        "density_only_diagnostic": density_only,
    }
    _atomic_write_json(os.path.join(out_dir, "fv_dac_metrics.json"), summary)

    # ---- per-sample artifact -------------------------------------------
    sorted_base = np.sort(base_probs, axis=1)
    per_sample: Dict[str, np.ndarray] = {
        "labels": labels.astype(np.int32),
        "base_logits": logits.astype(np.float32),
        "dac_S": s_dac.astype(np.float32),
        "base_pred": base_pred.astype(np.int32),
        "base_top2_margin": (sorted_base[:, -1] - sorted_base[:, -2]).astype(np.float32),
        "gt_rank_base": _rank_of_truth(base_probs, labels).astype(np.int32),
        "density_only_pred": dens_pred.astype(np.int32),
        "R_min": np.min(R_real, axis=1).astype(np.float32),
        "R_argmin": np.argmin(R_real, axis=1).astype(np.int32),
        "R_at_base_pred": R_real[np.arange(len(labels)), base_pred].astype(np.float32),
        "R_at_label": R_real[np.arange(len(labels)), labels].astype(np.float32),
    }
    for name, probs in probs_by_method.items():
        per_sample[f"pred__{name}"] = np.argmax(probs, axis=1).astype(np.int32)
        per_sample[f"gt_rank__{name}"] = _rank_of_truth(probs, labels).astype(np.int32)
        if name != "base_model":
            per_sample[f"probs__{name}"] = probs.astype(np.float16)
    np.savez_compressed(os.path.join(out_dir, "fv_dac_per_sample.npz"), **per_sample)

    _log_cell_table(cell, methods, density_only)


def _log_cell_table(
    cell: str, methods: List[Dict[str, Any]], density_only: Dict[str, Any]
) -> None:
    base_acc = next(m["metrics"]["accuracy"] for m in methods if m["method_name"] == "base_model")
    logger.info("==== %s ====", cell)
    logger.info(
        "%-22s %8s %8s %8s %8s %8s %6s %6s %6s",
        "method", "acc", "dAcc", "nll", "brier", "ece", "W", "H", "U",
    )
    for m in methods:
        met, fa = m["metrics"], m["flip_analysis"]
        logger.info(
            "%-22s %8.4f %8.4f %8.4f %8.4f %8.4f %6d %6d %6d",
            m["method_name"], met["accuracy"], met["accuracy"] - base_acc,
            met["nll"], met["brier"], met["top_label_ece"],
            fa["wrong_to_correct"], fa["correct_to_wrong"],
            fa["wrong_to_different_wrong"],
        )
    logger.info(
        "%-22s %8.4f  (agree base %.4f / FV-DAC %.4f)",
        "density_only[diag]", density_only["accuracy"],
        density_only["agreement_with_base"],
        density_only["agreement_with_fv_dac_primary"],
    )


# ==========================================================================
# CLI
# ==========================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full-Vector DAC standalone experiment")
    p.add_argument("--stage", choices=["fit", "evaluate"], required=True)
    p.add_argument("--dataset", type=str, default="cifar100")
    p.add_argument("--model", type=str, default="resnet101")
    p.add_argument("--method", type=str, default="baseline_cross_entropy")
    p.add_argument("--seed", type=int, required=True, help="checkpoint seed (1..5)")
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument(
        "--phase0_root", type=str, required=True,
        help="Phase 0/1 root holding fitted_method/ and evaluation/ (frozen splits, "
             "frozen native DAC state and the baseline rows FV-DAC reuses).",
    )
    p.add_argument("--fv_dac_state_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--corruption_type", type=str, default=None)
    p.add_argument("--corruption_severity", type=int, default=None, choices=[1, 2, 3, 4, 5])
    p.add_argument("--cifar_c_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--knn_batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--inner_val_fraction", type=float, default=0.5,
        help="Fraction of clean validation reserved for K_c SELECTION. The remainder "
             "fits beta. Repository default (utils/decision_audit.py); do not change "
             "mid-experiment.",
    )
    p.add_argument("--inner_val_seed", type=int, default=123)
    p.add_argument("--beta_max", type=float, default=BETA_SEARCH_MAX)
    p.add_argument("--bank_hash_decimals", type=int, default=5)
    p.add_argument(
        "--bank_match_tolerance", type=float, default=5e-3,
        help="Max allowed distance when matching each frozen bank row to its "
             "materialized-split counterpart in the multi-layer representation. The 5e-3 default is calibrated to the seed-4 "
             "full-vector TF32 perturbation (3.435e-3); it remains tiny relative to distinct-row separation, which is checked. "
             "(~1e-5) and tiny relative to the distance between DISTINCT bank rows, "
             "which the job reports as separation margin so the choice is checkable.",
    )
    p.add_argument(
        "--native_dac_tolerance", type=float, default=1e-5,
        help="Max allowed |dp| between frozen native DAC probabilities and probabilities "
             "recomputed from its unchanged frozen bank.",
    )
    p.add_argument(
        "--hash_dac_state", action="store_true",
        help="SHA-256 the 700MB frozen native_dac.pkl into the state record. Off by "
             "default because it costs a full read of the file.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    logger.info("FV-DAC %s stage | seed=%d | %s", args.stage, args.seed, _iso_now())
    if args.stage == "fit":
        if args.corruption_type is not None:
            raise ValueError(
                "--stage fit is CLEAN ONLY. Corrupted data must never reach a fit path."
            )
        run_fit(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
