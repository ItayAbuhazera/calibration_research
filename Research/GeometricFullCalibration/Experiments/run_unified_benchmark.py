"""
Unified calibration benchmark.

Runs baseline and calibration methods on one shared checkpoint/split protocol and
recomputes all core metrics with a single evaluator for apples-to-apples comparison.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Calibrators.geometric_calibrator import (
    FullVectorDistanceFusionCalibrator,
    FullVectorGeometricFusionCalibrator,
    SoftmaxKNNBlendCalibrator,
    extract_toplabel_anchor,
)
from Calibrators.kcal_lite import KCalLiteCalibrator
from Calibrators.kcal import KCalCalibrator
from Calibrators.kcal_factorial import (
    VALID_FUSIONS as KCAL_FACTORIAL_VALID_FUSIONS,
    build_posterior_grid as build_kcal_factorial_posterior_grid,
    effective_kernel_parameter as kcal_factorial_effective_kernel_parameter,
    select_factorial_outputs as select_kcal_factorial_outputs,
)
from Calibrators.beta_calibration import BetaCalibration
from Calibrators.odir_dirichlet_calibration import ODIRDirichletCalibration
from Calibrators.ovr_isotonic_calibration import OneVsRestIsotonicCalibration
from Calibrators.isotonic_regression import TopLabelIsotonicCalibrator
from Calibrators.temperature_scaling import TemperatureScaling
from Calibrators.vector_scaling import VectorScaling
from Experiments.run_post_hoc_calibration import FeatureExtractor
from Experiments.run_rgc_experiments import (
    extract_and_aggregate_sgc_features,
    filter_non_feature_layers,
    normalize_discovered_layers,
    run_coordinate_calibration,
    run_global_random_calibration,
    run_sgc_with_dac_layers,
    run_sgc_with_tulip_layers,
)
from utils.calibration_utils import get_all_data_as_numpy, load_cifar_c_loader
from utils.decision_audit import decision_audit as _decision_audit, make_inner_validation_split
from utils.method_metadata import (
    PRED_BASE,
    SCALAR_CONFIDENCE,
    SCALAR_ONLY_BUCKET,
    SEMANTIC_SCHEMA_VERSION,
    method_semantics,
    require_method_semantics,
    structural_axes_for_method,
)
from utils.stability_space import StabilitySpace
from utils.logging_config import get_logger
from utils.model_utils import (
    PyTorchModelAdapter,
    construct_model_path,
    get_data_loaders,
    load_trained_model,
)
from utils.unified_metrics import (
    confidence_ece,
    evaluate_all,
    evaluate_scalar_confidence,
    scalar_confidence_from_surrogate,
    top_label_ece,
    validate_probability_matrix,
)
from utils.utils import discover_model_layers

logger = get_logger(__name__)

BIN_EDGES = [0.0, 0.1, 0.5, 0.9, 1.0 + 1e-9]
BIN_LABELS = ["ct_lt_0.1", "ct_0.1_to_0.5", "ct_0.5_to_0.9", "ct_geq_0.9"]

# Every method family is enabled in a plain unified-benchmark invocation. Debug
# switches and architectural sub-options (for example GLAD-PI's temperature
# branch) intentionally remain opt-in because they do not add distinct table
# rows. Matching --disable_* switches are registered by
# _configure_default_method_flags so focused jobs remain possible.
DEFAULT_ENABLED_METHOD_FLAGS = (
    "enable_post_fusion_temperature",
    "enable_rgcl_tail_hybrids",
    "enable_full_vector_geometric_fusion",
    "enable_knn_blend_baseline",
    "enable_kcal_lite_baseline",
    "enable_rgcl_neighbor_correction",
    "enable_kcal_baseline",
    "enable_kcal_factorial",
    "enable_pts_baseline",
    "enable_trust_score_baseline",
    "enable_aar_lightweight",
    "enable_glad_pi",
    "enable_glad_pi_zero_geometry",
    "enable_mahalanobis_confidence",
    "enable_contrastive_beta_sweep",
)

CONTRASTIVE_BETA_METHOD_NAMES = (
    "contrastive_beta_vs",
    "contrastive_beta_vs_post_temperature",
    "contrastive_beta_ts",
    "contrastive_beta_ts_post_temperature",
)
CONTRASTIVE_SELECTION_CRITERION = "select_net_flips_nll_brier_ece_gated"
DEFAULT_CONTRASTIVE_NLL_TOLERANCE = 0.02
DEFAULT_CONTRASTIVE_BRIER_TOLERANCE = 0.05
DEFAULT_CONTRASTIVE_ECE_ABS_TOLERANCE = 0.005


def _configure_default_method_flags(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        **{flag_name: True for flag_name in DEFAULT_ENABLED_METHOD_FLAGS}
    )
    for flag_name in DEFAULT_ENABLED_METHOD_FLAGS:
        disable_name = "--disable_" + flag_name.removeprefix("enable_")
        parser.add_argument(
            disable_name,
            dest=flag_name,
            action="store_false",
            help=(
                f"Disable {flag_name.removeprefix('enable_').replace('_', ' ')} "
                "(enabled by default)."
            ),
        )
    parser.add_argument(
        "--disable_all_optional_methods",
        action="store_true",
        help=(
            "Disable every default-on optional method family. Individual --enable_* "
            "flags are retained for compatibility, but this global switch takes precedence."
        ),
    )


def _apply_disable_all_optional_methods(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "disable_all_optional_methods", False)):
        return
    for flag_name in DEFAULT_ENABLED_METHOD_FLAGS:
        setattr(args, flag_name, False)


FITTED_STATE_SUFFIXES = (".pkl", ".pt")


def _fitted_state_dir_hash(fitted_state_dir: str | None) -> str | None:
    """Aggregate content hash of every fitted-state artifact (FITTED_STATE_SUFFIXES:
    .pkl calibrators and .pt such as kcal_calibrator.pt) currently in
    fitted_state_dir, sorted by filename -- the "fitted artifact ID/hash"
    item 3 of the corruption-cell protocol requires every evaluation
    artifact to record. Two runs pointing at the same fitted_state_dir with
    unchanged contents get the identical hash; this is the direct,
    file-content-level demonstration that no refitting occurred."""
    if not fitted_state_dir or not os.path.isdir(fitted_state_dir):
        return None
    h = hashlib.sha256()
    for name in sorted(os.listdir(fitted_state_dir)):
        if not name.endswith(FITTED_STATE_SUFFIXES):
            continue
        h.update(name.encode("utf-8"))
        with open(os.path.join(fitted_state_dir, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def _git_provenance(repo_dir: str) -> Dict[str, Any]:
    """HEAD commit plus dirty-tree information. When tracked files differ from
    HEAD, git_diff_hash is the sha256 of `git diff HEAD` (tracked files only,
    so untracked benchmark artifacts never affect it) under repo_dir only;
    None when clean."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--binary", "--no-ext-diff", "--", "."],
            cwd=repo_dir, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {"git_commit": None, "git_dirty": None, "git_diff_hash": None}
    dirty = len(diff) > 0
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_diff_hash": hashlib.sha256(diff).hexdigest() if dirty else None,
    }


def _write_fit_once_provenance(args: argparse.Namespace) -> None:
    """
    Fit-once/evaluate-many protocol provenance (BENCHMARK_IMPLEMENTATION_PLAN.md
    Phase 0/1 corruption-cell requirement, item 3): every run -- clean or
    corruption -- writes fit_once_provenance.json into its own --output_dir
    recording the fitted-state artifact hash, the clean fitting split
    identity, the resolved config hash, and this run's evaluation
    corruption/severity. For a --corruption_type run this is written BEFORE
    any method executes, so the recorded fitted_state_hash is exactly what
    every method's _fit_or_load_* call will load (and, since those calls
    raise rather than fit fresh when state is missing, exactly what every
    method's evaluation in this run is guaranteed to have used).
    """
    resolved_config = {
        "dataset": args.dataset, "model": args.model, "method": args.method,
        "seed": args.seed, "num_layers": args.num_layers,
        "target_dimension": args.target_dimension, "batch_size": args.batch_size,
    }
    config_hash = hashlib.sha256(
        json.dumps(resolved_config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    provenance = {
        **_git_provenance(PROJECT_ROOT),
        "config_hash": config_hash,
        "resolved_config": resolved_config,
        "fitted_state_dir": args.fitted_state_dir,
        "fitted_state_hash": _fitted_state_dir_hash(args.fitted_state_dir),
        "clean_fitting_split": {
            "checkpoint_seed": args.seed, "dataset": args.dataset, "model": args.model,
            "note": "train/val split from utils.model_utils.get_data_loaders; always clean "
                    "regardless of --corruption_type (only the test split is ever corrupted)",
        },
        "evaluation": {
            "corruption_type": args.corruption_type,
            "corruption_severity": args.corruption_severity,
            "split": "clean_test" if args.corruption_type is None else "corrupted_test",
        },
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "fit_once_provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)


def _fit_or_load_state(
    args: argparse.Namespace,
    method_name: str,
    obj: Any,
    fit_callable: Callable[[], Any],
) -> Any:
    """
    Fit-once/evaluate-many protocol guard (BENCHMARK_IMPLEMENTATION_PLAN.md
    Phase 0/1 corruption-cell requirement): on a --corruption_type run,
    fit_callable must NEVER execute -- corrupted test data must only ever
    reach a frozen object's calibrate/predict path, never any .fit() call.
    This is enforced structurally, not just by convention: with
    args.corruption_type set, this function either loads previously pickled
    state into `obj` in place (restoring exactly what the clean run fit) or
    raises -- it never falls through to fit_callable().

    On a clean run (args.corruption_type is None), fits as usual via
    fit_callable() and persists obj's state to args.fitted_state_dir (when
    given) for later corruption-cell reuse.

    Returns fit_callable()'s return value, or None when state was loaded
    from disk instead of freshly fit.
    """
    is_corruption_run = getattr(args, "corruption_type", None) is not None
    fitted_state_dir = getattr(args, "fitted_state_dir", None)

    if is_corruption_run and not fitted_state_dir:
        raise RuntimeError(
            f"{method_name}: --corruption_type is set but --fitted_state_dir is not -- "
            "refusing to fit on corrupted data. Provide --fitted_state_dir pointing "
            "at a completed clean run's fitted state."
        )

    path = os.path.join(fitted_state_dir, f"{method_name}.pkl") if fitted_state_dir else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        obj.__dict__.clear()
        obj.__dict__.update(loaded.__dict__)
        return None

    if is_corruption_run:
        raise RuntimeError(
            f"{method_name}: --corruption_type is set but no fitted state found at "
            f"{path!r} -- refusing to fit fresh on corrupted data. Run the clean "
            "protocol with --fitted_state_dir first so this method's state exists."
        )

    result = fit_callable()
    if path:
        os.makedirs(fitted_state_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    return result


def _run_geo_cal_function_fit_once(
    args: argparse.Namespace,
    method_name: str,
    fn: Callable[..., Dict[str, Any]],
    fn_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fit-once/evaluate-many wrapper for the monolithic fit+calibrate functions
    (run_global_random_calibration, run_sgc_with_dac_layers) that accept
    precomputed_geo_cal/return_fitted_calibrator. Same structural guarantee
    as _fit_or_load_state: on a --corruption_type run, fn is called ONLY with
    a previously-fitted geo_cal supplied (train/val features are never
    touched, no GeometricCalibrator construction/.fit() call happens) -- or
    this raises. See _fit_or_load_state's docstring for the full rationale.
    """
    is_corruption_run = getattr(args, "corruption_type", None) is not None
    fitted_state_dir = getattr(args, "fitted_state_dir", None)

    if is_corruption_run and not fitted_state_dir:
        raise RuntimeError(
            f"{method_name}: --corruption_type is set but --fitted_state_dir is not -- "
            "refusing to fit on corrupted data."
        )

    path = os.path.join(fitted_state_dir, f"{method_name}.pkl") if fitted_state_dir else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            precomputed_geo_cal = pickle.load(f)
        return fn(**fn_kwargs, precomputed_geo_cal=precomputed_geo_cal)

    if is_corruption_run:
        raise RuntimeError(
            f"{method_name}: --corruption_type is set but no fitted state found at "
            f"{path!r} -- refusing to fit fresh on corrupted data."
        )

    result = fn(**fn_kwargs, return_fitted_calibrator=bool(path))
    if path:
        os.makedirs(fitted_state_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(result["fitted_geo_cal"], f)
    return result


def _fit_or_load_value(
    args: argparse.Namespace,
    method_name: str,
    fit_callable: Callable[[], Any],
) -> Any:
    """
    Like _fit_or_load_state but for a plain picklable VALUE (e.g. a dict of
    selected hyperparameters) rather than an object to mutate in place --
    same fit-once/evaluate-many guarantee: on a --corruption_type run,
    fit_callable is never called.
    """
    is_corruption_run = getattr(args, "corruption_type", None) is not None
    fitted_state_dir = getattr(args, "fitted_state_dir", None)

    if is_corruption_run and not fitted_state_dir:
        raise RuntimeError(
            f"{method_name}: --corruption_type is set but --fitted_state_dir is not -- "
            "refusing to fit on corrupted data."
        )

    path = os.path.join(fitted_state_dir, f"{method_name}.pkl") if fitted_state_dir else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    if is_corruption_run:
        raise RuntimeError(
            f"{method_name}: --corruption_type is set but no fitted state found at "
            f"{path!r} -- refusing to fit fresh on corrupted data."
        )

    value = fit_callable()
    if path:
        os.makedirs(fitted_state_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(value, f)
    return value


def _fit_or_load_state_with_result(
    args: argparse.Namespace,
    method_name: str,
    obj: Any,
    fit_callable: Callable[[], Any],
) -> Any:
    """
    Combines _fit_or_load_state's object-restore guarantee with persisting
    fit_callable's return value too, for callers that need direct-indexed
    access to fit diagnostics (e.g. GLADPICalibrator.fit's selection-history
    dict) beyond what obj.get_params() exposes. Returns fit_callable()'s
    value on a fresh fit, or the pickled value from the matching clean run
    on a --corruption_type run (which never calls fit_callable).
    """
    is_corruption_run = getattr(args, "corruption_type", None) is not None
    fitted_state_dir = getattr(args, "fitted_state_dir", None)
    if is_corruption_run and not fitted_state_dir:
        raise RuntimeError(
            f"{method_name}: --corruption_type is set but --fitted_state_dir is not -- "
            "refusing to fit on corrupted data."
        )

    state_path = os.path.join(fitted_state_dir, f"{method_name}.pkl") if fitted_state_dir else None
    result_path = os.path.join(fitted_state_dir, f"{method_name}__fit_result.pkl") if fitted_state_dir else None

    if state_path and os.path.exists(state_path) and os.path.exists(result_path):
        with open(state_path, "rb") as f:
            loaded = pickle.load(f)
        obj.__dict__.clear()
        obj.__dict__.update(loaded.__dict__)
        with open(result_path, "rb") as f:
            return pickle.load(f)

    if is_corruption_run:
        raise RuntimeError(
            f"{method_name}: --corruption_type is set but no fitted state found at "
            f"{state_path!r} -- refusing to fit fresh on corrupted data."
        )

    result = fit_callable()
    if state_path:
        os.makedirs(fitted_state_dir, exist_ok=True)
        with open(state_path, "wb") as f:
            pickle.dump(obj, f)
        with open(result_path, "wb") as f:
            pickle.dump(result, f)
    return result


def _select_rgcl_layers(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    device: torch.device,
    num_layers: int,
    seed: int,
) -> List[str]:
    """Reproduce RGCL's deterministic layer selection without fitting it again."""
    is_dinov2 = model_name is not None and "dinov2" in model_name.lower()
    if is_dinov2:
        input_shape = (1, 3, 224, 224)
    elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
        input_shape = (1, 3, 64, 64)
    else:
        input_shape = (1, 3, 32, 32)
    discovered = normalize_discovered_layers(
        model,
        discover_model_layers(model, device=device, input_shape=input_shape),
    )
    filtered, _ = filter_non_feature_layers(discovered, model)
    layer_names = [item["name"] for item in filtered]
    if num_layers >= len(layer_names):
        return layer_names
    rng = np.random.RandomState(seed)
    return list(rng.choice(layer_names, size=num_layers, replace=False))


def _temperature_to_float(temp_value: Any) -> float:
    """Handle temperature stored as tensor or python float."""
    if hasattr(temp_value, "detach"):
        return float(temp_value.detach().cpu().item())
    return float(temp_value)


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _parse_float_grid(raw: str, *, flag_name: str) -> np.ndarray:
    try:
        values = np.asarray(
            [float(value.strip()) for value in raw.split(",") if value.strip()],
            dtype=np.float64,
        )
    except ValueError as exc:
        raise ValueError(f"{flag_name} must contain comma-separated numbers") from exc
    if len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError(f"{flag_name} must contain at least one finite value")
    return values


def _kcal_factorial_method_name(
    representation: str,
    fusion: str,
    estimator: str,
    reference_bank: str,
    kernel: str,
) -> str:
    bank_slug = "cal" if reference_bank == "validation" else "train"
    return (
        f"kcal_factorial_{representation}_{fusion}_{estimator}_"
        f"{bank_slug}_{kernel}"
    )


def _kcal_factorial_feature_paths(output_dir: str) -> Dict[str, str]:
    feature_dir = _stage_dir(output_dir, "kcal_factorial_features")
    return {
        "rgcl_train": os.path.join(feature_dir, "rgcl_train.npy"),
        "rgcl_validation": os.path.join(feature_dir, "rgcl_validation.npy"),
        "rgcl_test": os.path.join(feature_dir, "rgcl_test.npy"),
        "rgcl_metadata": os.path.join(feature_dir, "rgcl_metadata.json"),
        "pi_train": os.path.join(feature_dir, "pi_train.npy"),
        "pi_validation": os.path.join(feature_dir, "pi_validation.npy"),
        "pi_test": os.path.join(feature_dir, "pi_test.npy"),
        "pi_metadata": os.path.join(feature_dir, "pi_metadata.json"),
    }


def _make_method_entry(
    method_name: str,
    probs: np.ndarray,
    y_test: np.ndarray,
    method_family: str,
    can_change_argmax: bool,
    tuning_procedure: str,
    tuning_split: str,
    tuning_objective: str,
    extra: Dict[str, Any] | None = None,
    anchor_ct_for_stratification: np.ndarray | None = None,
    base_probs_for_anchor_bins: np.ndarray | None = None,
    base_probs: np.ndarray | None = None,
) -> Dict[str, Any]:
    # The canonical registry — not the call site — decides how this method is
    # evaluated. `can_change_argmax` is still passed so a drifting call site
    # fails loudly here instead of silently overriding the registry.
    semantics = require_method_semantics(method_name)
    if bool(can_change_argmax) != bool(semantics["can_change_argmax"]):
        raise ValueError(
            f"{method_name}: call site passed can_change_argmax="
            f"{bool(can_change_argmax)} but the canonical registry "
            f"(utils/method_metadata.py) says {bool(semantics['can_change_argmax'])}. "
            "Fix the call site or the registry -- these must never disagree."
        )
    # Two INDEPENDENT dispatches, exactly as the frozen §6 table defines them:
    #   metric_bucket              -> WHICH metrics may be emitted
    #                                 (scalar_only forbids NLL/Brier/classwise)
    #   effective_prediction_source -> WHOSE prediction and confidence are scored
    # Trust Score switch is the case that forces them apart: it is scalar_only
    # ("+ argmax-change metrics for switch") yet its own switched decision is
    # the scientific object, so it is scored on its own argmax.
    is_scalar_bucket = semantics["metric_bucket"] == SCALAR_ONLY_BUCKET
    uses_base_prediction = semantics["effective_prediction_source"] == PRED_BASE
    is_scalar = semantics["output_semantics"] == SCALAR_CONFIDENCE

    entry: Dict[str, Any] = {
        "method_name": method_name,
        "method_family": method_family,
        "can_change_argmax": bool(semantics["can_change_argmax"]),
        "tuning_procedure": tuning_procedure,
        "tuning_split": tuning_split,
        "tuning_objective": tuning_objective,
        "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
        "output_semantics": semantics["output_semantics"],
        "effective_prediction_source": semantics["effective_prediction_source"],
        "metric_bucket": semantics["metric_bucket"],
    }

    # ---- 1. effective prediction + confidence (never the surrogate's argmax
    #         for a base-prediction method) ---------------------------------
    if uses_base_prediction:
        if base_probs is None:
            raise ValueError(
                f"{method_name} is a scalar-confidence method whose effective "
                "prediction is the base prediction, so base_probs is required to "
                "recover its confidence. Pass base_probs at the call site."
            )
        scalar = scalar_confidence_from_surrogate(probs, base_probs)
        effective_pred = scalar["effective_pred"]
        effective_confidence = scalar["confidence"]
        entry["surrogate_only"] = True
        entry["surrogate_reconstruction"] = semantics["surrogate_reconstruction"]
        entry["surrogate_argmax_change_rate"] = float(
            np.mean(scalar["surrogate_pred"] != effective_pred)
        )
        # Cannot move, by construction; recorded so the claim is visible in the
        # artifact rather than merely implied.
        entry["effective_argmax_change_rate"] = 0.0
    else:
        effective_pred = np.argmax(probs, axis=1)
        effective_confidence = np.max(probs, axis=1)
        entry["surrogate_only"] = False
        entry["surrogate_argmax_change_rate"] = None
        if base_probs is not None:
            entry["effective_argmax_change_rate"] = float(
                np.mean(effective_pred != np.argmax(base_probs, axis=1))
            )
    effective_correct = (effective_pred == y_test).astype(np.float64)

    # ---- 2. metric bucket -------------------------------------------------
    if is_scalar_bucket:
        entry["metrics"] = evaluate_scalar_confidence(
            effective_confidence, effective_correct
        )
    else:
        entry["metrics"] = evaluate_all(probs, y_test)

    # Structural axes from the same registry entry.
    entry["structural_axes"] = structural_axes_for_method(method_name)
    # Unified decision audit when base probabilities are available. For a
    # scalar method this describes the SURROGATE matrix only and is kept for
    # backward compatibility / diagnostics; `effective_argmax_change_rate`
    # above is the scientific quantity.
    if base_probs is not None:
        entry["decision_audit"] = _decision_audit(base_probs, probs, y_test)
        if uses_base_prediction:
            entry["decision_audit"]["describes"] = "surrogate_matrix_only"
        elif not is_scalar_bucket:
            entry["nll_subsets"] = _nll_by_base_prediction_subset(base_probs, probs, y_test)
    if anchor_ct_for_stratification is not None and base_probs_for_anchor_bins is not None:
        if is_scalar_bucket:
            entry["metrics_by_gc_dac_anchor_ct_bin"] = _scalar_metrics_by_anchor_ct_bin(
                confidence=effective_confidence,
                correct=effective_correct,
                anchor_ct=anchor_ct_for_stratification,
                argmax_change_rate=entry.get("effective_argmax_change_rate"),
            )
        else:
            entry["metrics_by_gc_dac_anchor_ct_bin"] = _metrics_by_anchor_ct_bin(
                probs=probs,
                labels=y_test,
                anchor_ct=anchor_ct_for_stratification,
                base_probs=base_probs_for_anchor_bins,
            )
        entry["gc_dac_anchor_ct_bin_edges"] = list(BIN_EDGES)
        entry["gc_dac_anchor_ct_bin_labels"] = list(BIN_LABELS)
    if extra:
        entry.update(extra)
    return entry


def _scalar_metrics_by_anchor_ct_bin(
    confidence: np.ndarray,
    correct: np.ndarray,
    anchor_ct: np.ndarray,
    argmax_change_rate: float | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Anchor-c_t stratification for a scalar-confidence method.

    Mirrors `_metrics_by_anchor_ct_bin` but is computed from the scalar
    semantics, so it never fabricates an NLL from a surrogate matrix (those
    fields are a real None, per the frozen §6 rule).
    """
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    anchor_ct = np.asarray(anchor_ct, dtype=np.float64).reshape(-1)
    if not (confidence.shape == correct.shape == anchor_ct.shape):
        raise ValueError("confidence, correct and anchor_ct must align")

    bin_idx = np.digitize(anchor_ct, BIN_EDGES, right=False) - 1
    out: Dict[str, Dict[str, Any]] = {}
    for b, label in enumerate(BIN_LABELS):
        mask = bin_idx == b
        count = int(np.sum(mask))
        if count == 0:
            out[label] = {
                "count": 0,
                "mean_anchor_ct": None,
                "accuracy": None,
                "nll": None,
                "brier_top_label": None,
                "delta_nll_vs_base": None,
                "argmax_change_rate_vs_base": None,
                "mean_confidence": None,
                "top_label_ece": None,
            }
            continue
        out[label] = {
            "count": count,
            "mean_anchor_ct": float(np.mean(anchor_ct[mask])),
            "accuracy": float(np.mean(correct[mask])),
            # Forbidden for a scalar method (§6): never reconstructed.
            "nll": None,
            "brier_top_label": float(np.mean((confidence[mask] - correct[mask]) ** 2)),
            "delta_nll_vs_base": None,
            "argmax_change_rate_vs_base": argmax_change_rate,
            "mean_confidence": float(np.mean(confidence[mask])),
            "top_label_ece": confidence_ece(confidence[mask], correct[mask]),
        }
    return out


def _metrics_by_anchor_ct_bin(
    probs: np.ndarray,
    labels: np.ndarray,
    anchor_ct: np.ndarray,
    base_probs: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, Dict[str, Any]]:
    if probs.ndim != 2 or base_probs.ndim != 2:
        raise ValueError("probs and base_probs must be 2D arrays")
    if probs.shape != base_probs.shape:
        raise ValueError("probs and base_probs must have identical shape")
    n_samples, _ = probs.shape
    if labels.ndim != 1 or anchor_ct.ndim != 1:
        raise ValueError("labels and anchor_ct must be 1D arrays")
    if len(labels) != n_samples or len(anchor_ct) != n_samples:
        raise ValueError("labels and anchor_ct must align to probs rows")

    idx = np.arange(n_samples)
    method_true_probs = np.clip(probs[idx, labels], eps, 1.0)
    base_true_probs = np.clip(base_probs[idx, labels], eps, 1.0)
    method_nll = -np.log(method_true_probs)
    base_nll = -np.log(base_true_probs)

    method_preds = np.argmax(probs, axis=1)
    base_preds = np.argmax(base_probs, axis=1)
    is_correct = (method_preds == labels).astype(np.float64)
    argmax_changed = method_preds != base_preds
    top_conf = np.max(probs, axis=1)
    top_correct = (method_preds == labels).astype(np.float64)
    brier_top = (top_conf - top_correct) ** 2

    bin_idx = np.digitize(anchor_ct, BIN_EDGES, right=False) - 1
    out: Dict[str, Dict[str, Any]] = {}
    for b, label in enumerate(BIN_LABELS):
        mask = bin_idx == b
        count = int(np.sum(mask))
        if count == 0:
            out[label] = {
                "count": 0,
                "mean_anchor_ct": None,
                "accuracy": None,
                "nll": None,
                "brier_top_label": None,
                "delta_nll_vs_base": None,
                "argmax_change_rate_vs_base": None,
            }
            continue
        out[label] = {
            "count": count,
            "mean_anchor_ct": float(np.mean(anchor_ct[mask])),
            "accuracy": float(np.mean(is_correct[mask])),
            "nll": float(np.mean(method_nll[mask])),
            "brier_top_label": float(np.mean(brier_top[mask])),
            "delta_nll_vs_base": float(np.mean(method_nll[mask]) - np.mean(base_nll[mask])),
            "argmax_change_rate_vs_base": float(np.mean(argmax_changed[mask])),
        }
    return out


def _anchored_model_tail_probs(
    model_probs: np.ndarray, anchor_top_conf: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """
    Build q by anchoring top class t=argmax(p) to c_t and renormalizing the tail:
      q_t = c_t
      q_j = (1 - c_t) * p_j / (1 - p_t), j != t
    If (1 - p_t) is too small, distribute (1 - c_t) uniformly on non-top classes.
    """
    if model_probs.ndim != 2:
        raise ValueError("model_probs must be a 2D array")
    if anchor_top_conf.ndim != 1 or len(anchor_top_conf) != model_probs.shape[0]:
        raise ValueError("anchor_top_conf must be a 1D array aligned to model_probs rows")

    n_samples, n_classes = model_probs.shape
    if n_classes < 2:
        raise ValueError("anchored_model_tail requires at least 2 classes")

    q = np.zeros_like(model_probs, dtype=np.float64)
    top_idx = np.argmax(model_probs, axis=1)
    c_t = np.clip(anchor_top_conf.astype(np.float64), 0.0, 1.0)
    p_non_top = model_probs.astype(np.float64).copy()
    p_non_top[np.arange(n_samples), top_idx] = 0.0
    non_top_sum = np.sum(p_non_top, axis=1)
    tail_mass = 1.0 - c_t

    safe = non_top_sum > eps
    if np.any(safe):
        safe_rows = np.where(safe)[0]
        q[safe_rows] = tail_mass[safe_rows, None] * (
            p_non_top[safe_rows] / non_top_sum[safe_rows, None]
        )
        q[safe_rows, top_idx[safe_rows]] = c_t[safe_rows]

    if np.any(~safe):
        unsafe_rows = np.where(~safe)[0]
        uniform_tail = tail_mass[unsafe_rows] / float(n_classes - 1)
        q[unsafe_rows] = uniform_tail[:, None]
        q[unsafe_rows, top_idx[unsafe_rows]] = c_t[unsafe_rows]

    q = np.clip(q, 0.0, 1.0)
    return q


def _fit_post_fusion_topiso_probs(
    fused_val_probs: np.ndarray,
    val_labels: np.ndarray,
    fused_test_probs: np.ndarray,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Fit top-coordinate isotonic repair on fused validation outputs."""
    if fused_val_probs.ndim != 2 or fused_test_probs.ndim != 2:
        raise ValueError("post_fusion_topiso requires 2D probability matrices")
    if fused_val_probs.shape[1] != fused_test_probs.shape[1]:
        raise ValueError("Validation and test fused probabilities must share class count")
    if len(val_labels) != fused_val_probs.shape[0]:
        raise ValueError("Validation labels must align to fused validation probabilities")

    val_top_idx = np.argmax(fused_val_probs, axis=1)
    val_top_prob = fused_val_probs[np.arange(len(val_top_idx)), val_top_idx]
    val_top_correct = (val_top_idx == val_labels).astype(np.float64)

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(val_top_prob, val_top_correct)

    test_top_idx = np.argmax(fused_test_probs, axis=1)
    test_top_prob = fused_test_probs[np.arange(len(test_top_idx)), test_top_idx]
    repaired_top_prob = np.clip(iso.transform(test_top_prob), 0.0, 1.0)
    repaired_probs = _anchored_model_tail_probs(
        fused_test_probs,
        repaired_top_prob,
    )

    return repaired_probs, {
        "mean_top_prob_before": float(np.mean(test_top_prob)),
        "mean_top_prob_after": float(np.mean(repaired_top_prob)),
    }


def _compose_anchored_tail_probs(
    anchor_top_idx: np.ndarray,
    anchor_top_conf: np.ndarray,
    tail_probs: np.ndarray,
    eps: float = 1e-12,
    renorm_tol: float = 1e-10,
) -> np.ndarray:
    """
    Generic anchored full-vector reconstruction:
      q[t] = c_t
      q[j != t] = (1 - c_t) * tail_j / sum_{k != t} tail_k
    If the conditional tail has no mass, distribute its explicit tail budget
    uniformly over non-top classes while preserving the anchor.
    """
    if tail_probs.ndim != 2:
        raise ValueError("tail_probs must be a 2D array")
    n_samples, n_classes = tail_probs.shape
    if n_classes < 2:
        raise ValueError("anchored composition requires at least 2 classes")
    if anchor_top_idx.ndim != 1 or anchor_top_idx.shape[0] != n_samples:
        raise ValueError("anchor_top_idx must be a 1D array aligned to tail_probs")
    if anchor_top_conf.ndim != 1 or anchor_top_conf.shape[0] != n_samples:
        raise ValueError("anchor_top_conf must be a 1D array aligned to tail_probs")
    if np.any(anchor_top_idx < 0) or np.any(anchor_top_idx >= n_classes):
        raise ValueError("anchor_top_idx contains out-of-range class indices")

    q = np.zeros((n_samples, n_classes), dtype=np.float64)
    c_t = np.clip(anchor_top_conf.astype(np.float64), 0.0, 1.0)
    tail = np.clip(tail_probs.astype(np.float64), 0.0, np.inf)
    tail[np.arange(n_samples), anchor_top_idx] = 0.0
    non_top_sum = np.sum(tail, axis=1)
    tail_mass = 1.0 - c_t

    safe = non_top_sum > eps
    if np.any(safe):
        safe_rows = np.where(safe)[0]
        q[safe_rows] = tail_mass[safe_rows, None] * (
            tail[safe_rows] / non_top_sum[safe_rows, None]
        )
        q[safe_rows, anchor_top_idx[safe_rows]] = c_t[safe_rows]

    if np.any(~safe):
        unsafe_rows = np.where(~safe)[0]
        uniform_tail = tail_mass[unsafe_rows] / float(n_classes - 1)
        q[unsafe_rows] = uniform_tail[:, None]
        q[unsafe_rows, anchor_top_idx[unsafe_rows]] = c_t[unsafe_rows]

    row_sums = np.sum(q, axis=1, keepdims=True)
    tiny_rows = row_sums.squeeze(-1) <= eps
    if np.any(tiny_rows):
        raise ValueError("anchored composition produced a zero-mass probability row")

    renorm_rows = np.abs(row_sums.squeeze(-1) - 1.0) > renorm_tol
    if np.any(renorm_rows):
        q[renorm_rows] = q[renorm_rows] / row_sums[renorm_rows]
    return np.clip(q, 0.0, 1.0)


def _validate_anchor_toplock(
    probs: np.ndarray,
    anchor_top_idx: np.ndarray,
    anchor_top_conf: np.ndarray,
    tol: float = 1e-8,
) -> tuple[bool, float]:
    anchored_values = probs[np.arange(probs.shape[0]), anchor_top_idx]
    max_abs_diff = (
        float(np.max(np.abs(anchored_values - anchor_top_conf)))
        if len(anchor_top_conf) > 0
        else 0.0
    )
    return max_abs_diff <= tol, max_abs_diff


def _rank_normalized_non_top(distances: np.ndarray, top_idx: np.ndarray) -> np.ndarray:
    """
    Rank-normalize non-top distances to [0, 1] per-sample.
    Smaller distance -> smaller rank value.
    Top class is set to 0 in returned matrix and should be masked out by callers.
    """
    n_samples, n_classes = distances.shape
    out = np.zeros_like(distances, dtype=np.float64)
    for i in range(n_samples):
        t = int(top_idx[i])
        non_top = np.ones(n_classes, dtype=bool)
        non_top[t] = False
        d = distances[i, non_top].astype(np.float64)
        k = d.shape[0]
        if k <= 1:
            out[i, non_top] = 0.0
            continue
        order = np.argsort(d, kind="stable")
        ranks = np.empty(k, dtype=np.int64)
        ranks[order] = np.arange(k)
        out[i, non_top] = ranks.astype(np.float64) / float(k - 1)
    return out


def _anchored_rankgeom_tail_mixture_probs(
    model_probs: np.ndarray,
    anchor_top_conf: np.ndarray,
    distance_matrix: np.ndarray,
    mix_lambda: float,
    alpha: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Anchored tail redistribution with model/geometric non-top mixture:
      q_t = c_t
      q_j = (1-c_t) * ((1-lambda) * m_j + lambda * g_j), j != t
    where m_j is model non-top conditional and g_j uses rank-normalized
    non-top distances with stable exp normalization.
    A zero-mass conditional tail is represented explicitly by a uniform
    distribution over non-top classes; the anchored top probability is unchanged.
    """
    if model_probs.ndim != 2 or distance_matrix.ndim != 2:
        raise ValueError("model_probs and distance_matrix must be 2D arrays")
    if model_probs.shape != distance_matrix.shape:
        raise ValueError("model_probs and distance_matrix must have identical shape")
    if anchor_top_conf.ndim != 1 or anchor_top_conf.shape[0] != model_probs.shape[0]:
        raise ValueError("anchor_top_conf must be 1D and aligned to model_probs rows")

    n_samples, n_classes = model_probs.shape
    if n_classes < 2:
        raise ValueError("anchored_rankgeom_tail_mixture requires at least 2 classes")

    lam = float(np.clip(mix_lambda, 0.0, 1.0))
    alpha = float(alpha)

    top_idx = np.argmax(model_probs, axis=1)
    c_t = np.clip(anchor_top_conf.astype(np.float64), 0.0, 1.0)
    tail_mass = 1.0 - c_t

    q = np.zeros_like(model_probs, dtype=np.float64)
    q[np.arange(n_samples), top_idx] = c_t

    rank_norm = _rank_normalized_non_top(distance_matrix, top_idx)
    for i in range(n_samples):
        t = int(top_idx[i])
        non_top = np.ones(n_classes, dtype=bool)
        non_top[t] = False

        # Conditional model-tail on non-top classes.
        p_non_top = model_probs[i, non_top].astype(np.float64)
        p_sum = float(np.sum(p_non_top))
        if p_sum > eps:
            m = p_non_top / p_sum
        else:
            m = np.full_like(p_non_top, 1.0 / float(n_classes - 1), dtype=np.float64)

        # Geometric non-top weights from rank-normalized distances.
        r = rank_norm[i, non_top]
        geom_logits = -alpha * r
        geom_logits = geom_logits - np.max(geom_logits)  # stable exp normalization
        g_unnorm = np.exp(geom_logits)
        g_sum = float(np.sum(g_unnorm))
        if g_sum > eps:
            g = g_unnorm / g_sum
        else:
            g = np.full_like(g_unnorm, 1.0 / float(n_classes - 1), dtype=np.float64)

        u = (1.0 - lam) * m + lam * g
        q[i, non_top] = tail_mass[i] * u
        q[i, t] = c_t[i]

    q = np.clip(q, 0.0, 1.0)
    row_sums = np.sum(q, axis=1, keepdims=True)
    q = q / np.where(row_sums <= eps, 1.0, row_sums)
    return q


def _multiclass_nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    idx = np.arange(len(labels))
    return float(-np.mean(np.log(np.clip(probs[idx, labels], eps, 1.0))))


def _nll_by_final_prediction_subset(
    probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12
) -> Dict[str, float | None]:
    preds = np.argmax(probs, axis=1)
    true_probs = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    nll = -np.log(true_probs)
    is_correct = preds == labels
    return {
        "nll_correct": float(np.mean(nll[is_correct])) if np.any(is_correct) else None,
        "nll_incorrect": float(np.mean(nll[~is_correct])) if np.any(~is_correct) else None,
    }


def _nll_by_base_prediction_subset(
    base_probs: np.ndarray, method_probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12
) -> Dict[str, float | None]:
    base_preds = np.argmax(base_probs, axis=1)
    base_true_probs = np.clip(base_probs[np.arange(len(labels)), labels], eps, 1.0)
    base_nll = -np.log(base_true_probs)
    true_probs = np.clip(method_probs[np.arange(len(labels)), labels], eps, 1.0)
    nll = -np.log(true_probs)
    base_is_correct = base_preds == labels
    n_base_correct = int(np.sum(base_is_correct))
    n_base_incorrect = int(np.sum(~base_is_correct))
    nll_on_base_correct = float(np.mean(nll[base_is_correct])) if np.any(base_is_correct) else None
    nll_on_base_incorrect = (
        float(np.mean(nll[~base_is_correct])) if np.any(~base_is_correct) else None
    )
    base_nll_on_base_correct = (
        float(np.mean(base_nll[base_is_correct])) if np.any(base_is_correct) else None
    )
    base_nll_on_base_incorrect = (
        float(np.mean(base_nll[~base_is_correct])) if np.any(~base_is_correct) else None
    )
    return {
        "nll_on_base_correct": nll_on_base_correct,
        "nll_on_base_incorrect": nll_on_base_incorrect,
        "count_base_correct": n_base_correct,
        "count_base_incorrect": n_base_incorrect,
        "delta_nll_on_base_correct": (
            float(nll_on_base_correct - base_nll_on_base_correct)
            if nll_on_base_correct is not None and base_nll_on_base_correct is not None
            else None
        ),
        "delta_nll_on_base_incorrect": (
            float(nll_on_base_incorrect - base_nll_on_base_incorrect)
            if nll_on_base_incorrect is not None and base_nll_on_base_incorrect is not None
            else None
        ),
        # Legacy keys kept for backward compatibility with existing consumers.
        "nll_when_base_correct": (
            float(np.mean(nll[base_is_correct])) if np.any(base_is_correct) else None
        ),
        "nll_when_base_incorrect": (
            float(np.mean(nll[~base_is_correct])) if np.any(~base_is_correct) else None
        ),
    }


def _argmax_flip_metadata(
    base_probs: np.ndarray, method_probs: np.ndarray, labels: np.ndarray
) -> Dict[str, float | int]:
    base_preds = np.argmax(base_probs, axis=1)
    method_preds = np.argmax(method_probs, axis=1)
    changed = method_preds != base_preds
    flip_to_correct = int(
        np.sum(changed & (base_preds != labels) & (method_preds == labels))
    )
    flip_to_wrong = int(np.sum(changed & (base_preds == labels) & (method_preds != labels)))
    n = len(labels)
    return {
        "argmax_change_rate": float(np.mean(changed)),
        "flip_to_correct_count": flip_to_correct,
        "flip_to_wrong_count": flip_to_wrong,
        "flip_to_correct_rate": float(flip_to_correct / n),
        "flip_to_wrong_rate": float(flip_to_wrong / n),
    }


def _contrastive_distance_signal(
    dist_matrix: np.ndarray,
    base_probs: np.ndarray,
) -> np.ndarray:
    """Return s(x)[k] = d(x, predicted_class) - d(x, k)."""
    distances = np.asarray(dist_matrix)
    probabilities = np.asarray(base_probs)
    if distances.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("dist_matrix and base_probs must both have shape [N, C]")
    if distances.shape != probabilities.shape:
        raise ValueError(
            "dist_matrix and base_probs must have the same shape; "
            f"got {distances.shape} and {probabilities.shape}"
        )
    if not np.all(np.isfinite(distances)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("dist_matrix and base_probs must contain only finite values")

    predicted_class = np.argmax(probabilities, axis=1)
    d_predicted = distances[np.arange(len(predicted_class)), predicted_class]
    return d_predicted[:, None] - distances


def _apply_contrastive_correction(
    logits: np.ndarray,
    a: np.ndarray | float,
    b: np.ndarray | float,
    beta: float,
    s: np.ndarray,
) -> np.ndarray:
    """Apply softmax(a * logits + b + beta * s) with stable NumPy softmax."""
    corrected = _contrastive_corrected_logits(logits, a, b, beta, s)
    corrected -= np.max(corrected, axis=1, keepdims=True)
    exp_corrected = np.exp(corrected)
    return exp_corrected / np.sum(exp_corrected, axis=1, keepdims=True)


def _contrastive_corrected_logits(
    logits: np.ndarray,
    a: np.ndarray | float,
    b: np.ndarray | float,
    beta: float,
    s: np.ndarray,
) -> np.ndarray:
    """Return a * logits + b + beta * s before softmax or post-temperature scaling."""
    logits_array = np.asarray(logits, dtype=np.float64)
    signal = np.asarray(s, dtype=np.float64)
    if logits_array.ndim != 2 or signal.shape != logits_array.shape:
        raise ValueError("logits and s must have the same shape [N, C]")

    scale = np.asarray(a, dtype=np.float64)
    bias = np.asarray(b, dtype=np.float64)
    if scale.ndim > 1 or (scale.ndim == 1 and scale.shape[0] != logits_array.shape[1]):
        raise ValueError("a must be a scalar or have shape [C]")
    if bias.ndim > 1 or (bias.ndim == 1 and bias.shape[0] != logits_array.shape[1]):
        raise ValueError("b must be a scalar or have shape [C]")

    return logits_array * scale + bias + float(beta) * signal


def _contrastive_gate_signature(
    nll_tolerance: float,
    brier_tolerance: float,
    ece_abs_tolerance: float,
    *,
    post_temperature: bool,
) -> str:
    return (
        f"{CONTRASTIVE_SELECTION_CRITERION}"
        f"|nll_tolerance={float(nll_tolerance):.12g}"
        f"|brier_tolerance={float(brier_tolerance):.12g}"
        f"|ece_abs_tolerance={float(ece_abs_tolerance):.12g}"
        f"|post_temperature={str(bool(post_temperature)).lower()}"
    )


def _fit_contrastive_post_temperature(
    corrected_logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Fit the existing 0.1..10.0 temperature grid on top-label ECE."""
    best_temperature = 1.0
    best_ece = float("inf")
    zero_signal = np.zeros_like(corrected_logits, dtype=np.float64)
    for step in range(1, 101):
        temperature = step / 10.0
        probs = _apply_contrastive_correction(
            corrected_logits,
            1.0 / temperature,
            0.0,
            0.0,
            zero_signal,
        )
        candidate_ece = top_label_ece(probs, labels)
        if candidate_ece < best_ece:
            best_ece = candidate_ece
            best_temperature = temperature
    return float(best_temperature)


class ContrastiveBetaSelector:
    def __init__(
        self,
        nll_tolerance: float = DEFAULT_CONTRASTIVE_NLL_TOLERANCE,
        brier_tolerance: float = DEFAULT_CONTRASTIVE_BRIER_TOLERANCE,
        ece_abs_tolerance: float = DEFAULT_CONTRASTIVE_ECE_ABS_TOLERANCE,
        *,
        post_temperature: bool = False,
    ) -> None:
        for name, value in (
            ("nll_tolerance", nll_tolerance),
            ("brier_tolerance", brier_tolerance),
            ("ece_abs_tolerance", ece_abs_tolerance),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.nll_tolerance = float(nll_tolerance)
        self.brier_tolerance = float(brier_tolerance)
        self.ece_abs_tolerance = float(ece_abs_tolerance)
        self.post_temperature = bool(post_temperature)

    @property
    def gate_signature(self) -> str:
        return _contrastive_gate_signature(
            self.nll_tolerance,
            self.brier_tolerance,
            self.ece_abs_tolerance,
            post_temperature=self.post_temperature,
        )

    def select(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        base_probs: np.ndarray,
        signal: np.ndarray,
        a: np.ndarray | float,
        b: np.ndarray | float,
        beta_grid: List[float],
    ) -> Dict[str, Any]:
        """Evaluate and select beta on one held-out selection split."""
        if not beta_grid:
            raise ValueError("beta_grid must contain at least one value")
        normalized_grid = [float(beta) for beta in beta_grid]
        if 0.0 not in normalized_grid:
            normalized_grid.insert(0, 0.0)

        sweep_curve: List[Dict[str, Any]] = []
        for beta in normalized_grid:
            corrected_logits = _contrastive_corrected_logits(logits, a, b, beta, signal)
            post_temperature = None
            if self.post_temperature:
                post_temperature = _fit_contrastive_post_temperature(
                    corrected_logits, labels
                )
                probs = _apply_contrastive_correction(
                    corrected_logits,
                    1.0 / post_temperature,
                    0.0,
                    0.0,
                    np.zeros_like(corrected_logits),
                )
            else:
                probs = _apply_contrastive_correction(logits, a, b, beta, signal)
            metrics = evaluate_all(probs, labels)
            flips = _argmax_flip_metadata(base_probs, probs, labels)
            flip_to_correct = int(flips["flip_to_correct_count"])
            flip_to_wrong = int(flips["flip_to_wrong_count"])
            net_flips = flip_to_correct - flip_to_wrong
            sweep_curve.append(
                {
                    "beta": float(beta),
                    "nll": float(metrics["nll"]),
                    "brier": float(metrics["brier"]),
                    "accuracy": float(metrics["accuracy"]),
                    "ece": float(metrics["top_label_ece"]),
                    "net_flips": net_flips,
                    "flip_to_correct": flip_to_correct,
                    "flip_to_wrong": flip_to_wrong,
                    "select_nll": float(metrics["nll"]),
                    "select_brier": float(metrics["brier"]),
                    "select_top_label_ece": float(metrics["top_label_ece"]),
                    "select_accuracy": float(metrics["accuracy"]),
                    "select_net_flips": net_flips,
                    "post_temperature": post_temperature,
                }
            )

        beta_zero = next(row for row in sweep_curve if row["beta"] == 0.0)
        ref_nll = float(beta_zero["select_nll"])
        ref_brier = float(beta_zero["select_brier"])
        ref_ece = float(beta_zero["select_top_label_ece"])
        nll_budget = ref_nll * (1.0 + self.nll_tolerance)
        brier_budget = ref_brier * (1.0 + self.brier_tolerance)
        ece_budget = ref_ece + self.ece_abs_tolerance

        for row in sweep_curve:
            row["nll_gate_pass"] = bool(row["select_nll"] <= nll_budget)
            row["brier_gate_pass"] = bool(row["select_brier"] <= brier_budget)
            row["ece_gate_pass"] = bool(
                row["select_top_label_ece"] <= ece_budget
            )
            row["net_flips_gate_pass"] = bool(row["select_net_flips"] >= 0)
            row["gate_pass"] = bool(
                row["nll_gate_pass"]
                and row["brier_gate_pass"]
                and row["ece_gate_pass"]
                and row["net_flips_gate_pass"]
            )

        candidates = [row for row in sweep_curve if row["gate_pass"]]
        if candidates:
            selected = max(
                candidates,
                key=lambda row: (
                    int(row["select_net_flips"]),
                    float(row["select_accuracy"]),
                    -float(row["select_nll"]),
                    -abs(float(row["beta"])),
                ),
            )
            selected_fallback = False
        else:
            selected = beta_zero
            selected_fallback = True

        nll_optimal = min(
            sweep_curve,
            key=lambda row: (float(row["select_nll"]), abs(float(row["beta"]))),
        )
        return {
            "selection_criterion": CONTRASTIVE_SELECTION_CRITERION,
            "gate_signature": self.gate_signature,
            "nll_tolerance": self.nll_tolerance,
            "brier_tolerance": self.brier_tolerance,
            "ece_abs_tolerance": self.ece_abs_tolerance,
            "selected_beta": float(selected["beta"]),
            "selected_fallback": selected_fallback,
            "selected_post_temperature": selected["post_temperature"],
            "reference_nll": ref_nll,
            "ref_nll": ref_nll,
            "ref_brier": ref_brier,
            "ref_ece": ref_ece,
            "nll_budget": float(nll_budget),
            "brier_budget": float(brier_budget),
            "ece_budget": float(ece_budget),
            "best_select_net_flips": int(selected["select_net_flips"]),
            "best_select_nll": float(selected["select_nll"]),
            "best_select_brier": float(selected["select_brier"]),
            "best_select_top_label_ece": float(
                selected["select_top_label_ece"]
            ),
            "nll_optimal_beta": float(nll_optimal["beta"]),
            "nll_optimal_beta_select_nll": float(nll_optimal["select_nll"]),
            "sweep_curve": sweep_curve,
        }


def _select_contrastive_beta(
    logits: np.ndarray,
    labels: np.ndarray,
    base_probs: np.ndarray,
    signal: np.ndarray,
    a: np.ndarray | float,
    b: np.ndarray | float,
    beta_grid: List[float],
    nll_tolerance: float = DEFAULT_CONTRASTIVE_NLL_TOLERANCE,
    brier_tolerance: float = DEFAULT_CONTRASTIVE_BRIER_TOLERANCE,
    ece_abs_tolerance: float = DEFAULT_CONTRASTIVE_ECE_ABS_TOLERANCE,
    *,
    post_temperature: bool = False,
) -> Dict[str, Any]:
    selector = ContrastiveBetaSelector(
        nll_tolerance=nll_tolerance,
        brier_tolerance=brier_tolerance,
        ece_abs_tolerance=ece_abs_tolerance,
        post_temperature=post_temperature,
    )
    return selector.select(logits, labels, base_probs, signal, a, b, beta_grid)


def _build_contrastive_beta_outputs(
    *,
    val_logits: np.ndarray,
    test_logits: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    val_base_probs: np.ndarray,
    test_base_probs: np.ndarray,
    val_distances: np.ndarray,
    test_distances: np.ndarray,
    beta_grid: List[float],
    nll_tolerance: float,
    brier_tolerance: float,
    ece_abs_tolerance: float,
    inner_val_fraction: float,
    inner_val_seed: int,
    feature_source: str,
    method_names: set[str] | None = None,
    anchor_ct: np.ndarray | None = None,
    anchor_base_probs: np.ndarray | None = None,
) -> Dict[str, tuple[Dict[str, Any], np.ndarray]]:
    """Build requested contrastive rows from aligned cached benchmark arrays."""
    requested = set(CONTRASTIVE_BETA_METHOD_NAMES if method_names is None else method_names)
    unknown = requested - set(CONTRASTIVE_BETA_METHOD_NAMES)
    if unknown:
        raise ValueError(f"Unknown contrastive methods requested: {sorted(unknown)}")

    grid = [float(beta) for beta in beta_grid]
    if not grid:
        raise ValueError("contrastive beta grid must contain at least one value")
    if not all(np.isfinite(beta) and beta >= 0.0 for beta in grid):
        raise ValueError("contrastive beta grid values must be finite and non-negative")
    if 0.0 not in grid:
        grid.insert(0, 0.0)

    fit_idx, select_idx = make_inner_validation_split(
        val_labels,
        select_fraction=inner_val_fraction,
        seed=inner_val_seed,
    )
    signal_val_raw = _contrastive_distance_signal(val_distances, val_base_probs).astype(
        np.float64, copy=False
    )
    signal_test_raw = _contrastive_distance_signal(
        test_distances, test_base_probs
    ).astype(np.float64, copy=False)
    signal_mean = float(np.mean(signal_val_raw[fit_idx]))
    signal_std = max(float(np.std(signal_val_raw[fit_idx])), 1e-8)
    signal_val = (signal_val_raw - signal_mean) / signal_std
    signal_test = (signal_test_raw - signal_mean) / signal_std

    selectors = {
        False: ContrastiveBetaSelector(
            nll_tolerance=nll_tolerance,
            brier_tolerance=brier_tolerance,
            ece_abs_tolerance=ece_abs_tolerance,
        ),
        True: ContrastiveBetaSelector(
            nll_tolerance=nll_tolerance,
            brier_tolerance=brier_tolerance,
            ece_abs_tolerance=ece_abs_tolerance,
            post_temperature=True,
        ),
    }
    num_classes = int(np.shape(val_logits)[1])
    variants: List[tuple[str, np.ndarray, np.ndarray, str, Dict[str, Any]]] = []

    if requested & {"contrastive_beta_vs", "contrastive_beta_vs_post_temperature"}:
        vector_scaling = VectorScaling()
        vector_scaling.fit(val_logits[fit_idx], val_labels[fit_idx])
        if vector_scaling.a is None or vector_scaling.b is None:
            raise RuntimeError("Vector scaling did not produce fitted a and b vectors")
        variants.append(
            (
                "vs",
                np.asarray(vector_scaling.a, dtype=np.float64),
                np.asarray(vector_scaling.b, dtype=np.float64),
                "vector_scaling",
                {"vector_scaling_params": vector_scaling.get_params()},
            )
        )

    if requested & {"contrastive_beta_ts", "contrastive_beta_ts_post_temperature"}:
        temperature_scaling = TemperatureScaling(cross_validate="nll")
        temperature_scaling.fit(val_logits[fit_idx], val_labels[fit_idx])
        temperature = _temperature_to_float(temperature_scaling.temperature)
        variants.append(
            (
                "ts",
                np.full(num_classes, 1.0 / temperature, dtype=np.float64),
                np.zeros(num_classes, dtype=np.float64),
                "temperature_scaling",
                {
                    "temperature": temperature,
                    "temperature_scaling_params": temperature_scaling.get_params(),
                },
            )
        )

    outputs: Dict[str, tuple[Dict[str, Any], np.ndarray]] = {}
    for suffix, a, b, a_source, fit_meta in variants:
        parent_name = f"contrastive_beta_{suffix}"
        for post_temperature in (False, True):
            method_name = parent_name + ("_post_temperature" if post_temperature else "")
            if method_name not in requested:
                continue
            selection = selectors[post_temperature].select(
                val_logits[select_idx],
                val_labels[select_idx],
                val_base_probs[select_idx],
                signal_val[select_idx],
                a,
                b,
                grid,
            )
            corrected_test = _contrastive_corrected_logits(
                test_logits, a, b, selection["selected_beta"], signal_test
            )
            pre_temperature_probs = _apply_contrastive_correction(
                test_logits, a, b, selection["selected_beta"], signal_test
            )
            selected_temperature = selection["selected_post_temperature"]
            if post_temperature:
                if selected_temperature is None:
                    raise RuntimeError(f"{method_name}: selected temperature is missing")
                probs = _apply_contrastive_correction(
                    corrected_test,
                    1.0 / selected_temperature,
                    0.0,
                    0.0,
                    np.zeros_like(corrected_test),
                )
                if not np.array_equal(
                    np.argmax(pre_temperature_probs, axis=1), np.argmax(probs, axis=1)
                ):
                    raise RuntimeError(
                        f"{method_name}: positive scalar temperature changed an argmax"
                    )
            else:
                probs = pre_temperature_probs

            extra: Dict[str, Any] = {
                **selection,
                **fit_meta,
                "beta_grid": [float(row["beta"]) for row in selection["sweep_curve"]],
                "a_source": a_source,
                "inner_val_fraction": inner_val_fraction,
                "inner_val_seed": inner_val_seed,
                "inner_val_fit_n": int(len(fit_idx)),
                "inner_val_select_n": int(len(select_idx)),
                "feature_source": feature_source,
                "signal_standardization": "global_mean_std_from_inner_val_fit",
                "signal_fit_mean": signal_mean,
                "signal_fit_std": signal_std,
                "selection_criterion": CONTRASTIVE_SELECTION_CRITERION,
            }
            if post_temperature:
                extra.update(
                    {
                        "parent_method": parent_name,
                        "post_temperature": selected_temperature,
                        "post_temperature_objective": "inner_val_select_ece",
                        "post_temperature_fit_split": "inner_val_select",
                        "post_temperature_reuses_beta_select_split": True,
                        "post_temperature_params": {
                            "temperature": selected_temperature,
                            "cross_validate": "ece",
                            "temperature_grid": "0.1:0.1:10.0",
                        },
                        "argmax_change_rate_vs_parent": 0.0,
                        "pre_temperature_test_metrics": evaluate_all(
                            pre_temperature_probs, test_labels
                        ),
                    }
                )
            flip_meta = _argmax_flip_metadata(test_base_probs, probs, test_labels)
            extra.update(flip_meta)
            extra["net_flips"] = int(
                flip_meta["flip_to_correct_count"] - flip_meta["flip_to_wrong_count"]
            )
            extra.update(_nll_by_base_prediction_subset(test_base_probs, probs, test_labels))
            extra.update(_nll_by_final_prediction_subset(probs, test_labels))
            entry = _make_method_entry(
                method_name,
                probs,
                test_labels,
                "decision_improving_contrastive",
                True,
                (
                    "contrastive_beta_grid_then_scalar_temperature"
                    if post_temperature
                    else "contrastive_beta_grid"
                ),
                "inner_val_select",
                CONTRASTIVE_SELECTION_CRITERION,
                extra,
                anchor_ct_for_stratification=anchor_ct,
                base_probs_for_anchor_bins=anchor_base_probs,
                base_probs=test_base_probs,
            )
            outputs[method_name] = (entry, probs)
    return outputs


def _neighbor_correction_geometry(
    distance_matrix: np.ndarray,
    base_pred: np.ndarray,
) -> tuple:
    """Return (alt_pred, trust, separation) for each sample.

    trust = d_nearest_other / (d_predicted + eps)  -- < 1 means another class is geometrically closer
    separation = (d_nearest_other - d_predicted) / 2  -- < 0 means same condition
    """
    rows = np.arange(len(base_pred))
    d_pred = distance_matrix[rows, base_pred]
    alt_dm = distance_matrix.copy()
    alt_dm[rows, base_pred] = np.inf
    alt_pred = np.argmin(alt_dm, axis=1)
    d_other = alt_dm[rows, alt_pred]
    trust = d_other / (d_pred + 1e-8)
    separation = (d_other - d_pred) / 2.0
    return alt_pred, trust, separation


def _select_neighbor_threshold(
    score: np.ndarray,
    upper_boundary: float,
    base_pred: np.ndarray,
    alt_pred: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Sweep score values below upper_boundary; return the threshold that maximises accuracy.

    Tie-break: conservatively choose fewest prediction changes.
    Returns the threshold along with validation-split flip counts and the switched mask.
    """
    eligible_idx = np.flatnonzero(score < upper_boundary)
    order = eligible_idx[np.argsort(score[eligible_idx], kind="stable")]
    ordered_scores = score[order]
    ordered_gain = (
        (alt_pred[order] == labels[order]).astype(np.int64)
        - (base_pred[order] == labels[order]).astype(np.int64)
    )
    cumulative_gain = np.r_[0, np.cumsum(ordered_gain)]
    best_gain = int(np.max(cumulative_gain))
    switch_count = int(np.flatnonzero(cumulative_gain == best_gain)[0])  # fewest switches

    if switch_count == 0 or len(ordered_scores) == 0:
        threshold = float(ordered_scores[0]) if len(ordered_scores) else upper_boundary
    elif switch_count >= len(ordered_scores):
        threshold = float(upper_boundary)
    else:
        threshold = float(
            (ordered_scores[switch_count - 1] + ordered_scores[switch_count]) / 2.0
        )

    switched = score < threshold
    fixes = int(np.sum(switched & (base_pred != labels) & (alt_pred == labels)))
    breaks = int(np.sum(switched & (base_pred == labels) & (alt_pred != labels)))
    wrong_wrong = int(np.sum(switched & (base_pred != labels) & (alt_pred != labels)))
    return {
        "threshold": threshold,
        "upper_boundary": float(upper_boundary),
        "eligible_count_at_default_boundary": int(len(eligible_idx)),
        "val_switch_count": int(np.sum(switched)),
        "val_flip_correct": fixes,
        "val_flip_wrong": breaks,
        "val_wrong_to_wrong": wrong_wrong,
        "val_net_gain": best_gain,
        "val_base_accuracy": float(np.mean(base_pred == labels)),
        "val_thresholded_accuracy": float(
            np.mean(np.where(switched, alt_pred, base_pred) == labels)
        ),
    }


def _neighbor_correction_probs(
    base_probs: np.ndarray,
    switched_mask: np.ndarray,
    base_pred: np.ndarray,
    alt_pred: np.ndarray,
) -> np.ndarray:
    """Swap probabilities of predicted and alternative class for switched samples."""
    result = base_probs.copy()
    rows = np.flatnonzero(switched_mask)
    if len(rows):
        p_pred = result[rows, base_pred[rows]].copy()
        p_alt = result[rows, alt_pred[rows]].copy()
        result[rows, base_pred[rows]] = p_alt
        result[rows, alt_pred[rows]] = p_pred
    return result


def _validate_probability_matrix(
    probs: np.ndarray,
    n_rows: int,
    n_classes: int,
    method_name: str,
    row_sum_tol: float = 1e-6,
) -> None:
    if probs.ndim != 2:
        raise ValueError(f"{method_name}: calibrated output must be 2D NxK probabilities")
    if probs.shape != (n_rows, n_classes):
        raise ValueError(
            f"{method_name}: expected output shape {(n_rows, n_classes)}, got {probs.shape}"
        )
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"{method_name}: calibrated output contains non-finite values")
    if np.any(probs < 0.0):
        raise ValueError(f"{method_name}: calibrated output contains negative probabilities")
    row_sums = np.sum(probs, axis=1)
    if not np.allclose(row_sums, 1.0, atol=row_sum_tol, rtol=0.0):
        raise ValueError(f"{method_name}: calibrated output rows must sum to 1")


def _fit_calibrate_append_simple_posthoc(
    methods: List[Dict[str, Any]],
    method_name: str,
    method_family: str,
    can_change_argmax: bool,
    tuning_procedure: str,
    tuning_split: str,
    tuning_objective: str,
    fit_inputs: np.ndarray,
    fit_labels: np.ndarray,
    test_inputs: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    fit_fn: Callable[[np.ndarray, np.ndarray], Dict[str, Any] | None],
    calibrate_fn: Callable[[np.ndarray], np.ndarray],
    base_probs_for_diagnostics: np.ndarray | None = None,
    include_decision_diagnostics: bool = False,
    extra: Dict[str, Any] | None = None,
    anchor_ct_for_stratification: np.ndarray | None = None,
    base_probs_for_anchor_bins: np.ndarray | None = None,
    base_probs: np.ndarray | None = None,
) -> np.ndarray:
    fit_info = fit_fn(fit_inputs, fit_labels)
    calibrated_probs = calibrate_fn(test_inputs)
    _validate_probability_matrix(
        calibrated_probs,
        n_rows=len(y_test),
        n_classes=n_classes,
        method_name=method_name,
    )

    method_extra: Dict[str, Any] = {}
    if extra:
        method_extra.update(extra)
    if fit_info:
        method_extra.update(fit_info)
    if include_decision_diagnostics:
        if base_probs_for_diagnostics is None:
            raise ValueError(
                f"{method_name}: base_probs_for_diagnostics is required for decision diagnostics"
            )
        method_extra.update(
            _argmax_flip_metadata(base_probs_for_diagnostics, calibrated_probs, y_test)
        )
        method_extra.update(
            _nll_by_base_prediction_subset(base_probs_for_diagnostics, calibrated_probs, y_test)
        )

    # Resolve base_probs for unified decision_audit: prefer explicit base_probs,
    # fall back to base_probs_for_diagnostics when decision diagnostics are active.
    _resolved_base_probs = base_probs
    if _resolved_base_probs is None and include_decision_diagnostics:
        _resolved_base_probs = base_probs_for_diagnostics

    methods.append(
        _make_method_entry(
            method_name,
            calibrated_probs,
            y_test,
            method_family,
            can_change_argmax,
            tuning_procedure,
            tuning_split,
            tuning_objective,
            method_extra if method_extra else None,
            anchor_ct_for_stratification=anchor_ct_for_stratification,
            base_probs_for_anchor_bins=base_probs_for_anchor_bins,
            base_probs=_resolved_base_probs,
        )
    )
    return calibrated_probs


def _non_top_distance_summary(
    base_probs: np.ndarray, distance_matrix: np.ndarray
) -> Dict[str, float | None]:
    if distance_matrix.ndim != 2 or base_probs.ndim != 2:
        return {
            "non_top_distance_mean": None,
            "non_top_distance_std": None,
            "non_top_distance_min": None,
            "non_top_distance_max": None,
        }
    if distance_matrix.shape != base_probs.shape:
        return {
            "non_top_distance_mean": None,
            "non_top_distance_std": None,
            "non_top_distance_min": None,
            "non_top_distance_max": None,
        }
    n_samples, n_classes = base_probs.shape
    if n_classes < 2:
        return {
            "non_top_distance_mean": None,
            "non_top_distance_std": None,
            "non_top_distance_min": None,
            "non_top_distance_max": None,
        }
    top_idx = np.argmax(base_probs, axis=1)
    mask = np.ones((n_samples, n_classes), dtype=bool)
    mask[np.arange(n_samples), top_idx] = False
    non_top = distance_matrix[mask]
    if non_top.size == 0:
        return {
            "non_top_distance_mean": None,
            "non_top_distance_std": None,
            "non_top_distance_min": None,
            "non_top_distance_max": None,
        }
    return {
        "non_top_distance_mean": float(np.mean(non_top)),
        "non_top_distance_std": float(np.std(non_top)),
        "non_top_distance_min": float(np.min(non_top)),
        "non_top_distance_max": float(np.max(non_top)),
    }


def _summarize_anchor_stats(anchor: np.ndarray) -> Dict[str, float | None]:
    if anchor.size == 0:
        return {"min": None, "mean": None, "median": None, "max": None, "std": None}
    anchor = anchor.astype(np.float64)
    return {
        "min": float(np.min(anchor)),
        "mean": float(np.mean(anchor)),
        "median": float(np.median(anchor)),
        "max": float(np.max(anchor)),
        "std": float(np.std(anchor)),
    }


def _crossfit_gc_dac_validation_anchor(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_y: np.ndarray,
    val_base_probs: np.ndarray,
    device: torch.device,
    batch_size: int,
    target_dim: int,
    pooling_mode: str,
    seed: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, Dict[str, Any]]:
    # This helper is intentionally keyed to the tuning-order basis
    # (val_raw/val_y/val_base_probs) so anchor indices match grid-search inputs.
    if len(val_raw) != len(val_y) or val_base_probs.shape[0] != len(val_y):
        raise RuntimeError(
            "Validation ordering mismatch: val_raw, val_y, and val_base_probs must align."
        )
    if val_base_probs.ndim != 2:
        raise ValueError("val_base_probs must be a 2D array")
    if len(np.unique(val_y)) < 2:
        raise ValueError("Cross-fit GC-DAC requires at least two validation classes")

    val_anchor_c_t = np.empty(len(val_y), dtype=np.float64)
    fold_sizes: List[int] = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_indices = np.zeros(len(val_y), dtype=np.int64)

    for fit_idx, holdout_idx in splitter.split(all_indices, val_y):
        fold_sizes.append(int(len(holdout_idx)))
        fold_out = run_sgc_with_dac_layers(
            model=model,
            model_name=model_name,
            dataset_name=dataset_name,
            model_adapter=model_adapter,
            train_raw=train_raw,
            train_labels=train_labels,
            val_raw=val_raw[fit_idx],
            val_labels=val_y[fit_idx],
            test_raw=val_raw[holdout_idx],
            test_labels=val_y[holdout_idx],
            device=device,
            target_dim=target_dim,
            batch_size=batch_size,
            seed=seed,
            pooling_mode=pooling_mode,
            precomputed_test_probs=val_base_probs[holdout_idx],
            train_loader=None,
            val_loader=None,
            test_loader=None,
            return_probs=True,
        )
        fold_probs = fold_out["calibrated_probs"]
        top_idx = np.argmax(val_base_probs[holdout_idx], axis=1)
        val_anchor_c_t[holdout_idx] = fold_probs[np.arange(len(holdout_idx)), top_idx]

    return val_anchor_c_t, {"folds": int(n_splits), "per_fold_sizes": fold_sizes}


def _run_full_vector_fusion(
    model_adapter: PyTorchModelAdapter,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_base_probs: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_base_probs: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
    stab_metric: str = "l2",
    whitening_components: int = 128,
    whitening_eps: float = 1e-6,
) -> Dict[str, Any]:
    fusion = FullVectorDistanceFusionCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=train_labels,
        metric=stab_metric,
        library="fast_separation",
        whitening_components=whitening_components,
        whitening_eps=whitening_eps,
    )

    rng = np.random.default_rng(seed)
    bs = min(128, len(val_labels))
    idx = rng.choice(len(val_labels), size=bs, replace=False)
    val_batch_embed = val_features[idx]
    val_batch_probs = val_base_probs[idx]
    pred = np.argmax(val_batch_probs, axis=1)
    dist_matrix_batch = fusion.stab_space.calc_per_class_1nn_distances(val_batch_embed)
    vector_pred_dist = dist_matrix_batch[np.arange(bs), pred]
    scalar_same_dist = fusion.stab_space.calc_predicted_class_same_distances(
        val_batch_embed, val_batch_probs
    )
    consistency_max_abs_diff = float(np.max(np.abs(vector_pred_dist - scalar_same_dist)))

    fusion.fit(X_val_embed=val_features, y_val=val_labels, model_probs=val_base_probs)
    fused_test_probs, test_details = fusion.calibrate(
        X_test_embed=test_features, model_probs=test_base_probs, return_details=True
    )
    fused_val_probs = fusion.calibrate(X_test_embed=val_features, model_probs=val_base_probs)

    base_preds = np.argmax(test_base_probs, axis=1)
    fused_preds = np.argmax(fused_test_probs, axis=1)
    changed = fused_preds != base_preds
    changed_to_correct = int(
        np.sum(changed & (fused_preds == test_labels) & (base_preds != test_labels))
    )
    changed_to_wrong = int(
        np.sum(changed & (fused_preds != test_labels) & (base_preds == test_labels))
    )
    net_flips = int(changed_to_correct - changed_to_wrong)

    return {
        "fused_test_probs": fused_test_probs,
        "fused_val_probs": fused_val_probs,
        "stab_space": fusion.stab_space,
        "selected_beta": float(fusion.best_beta),
        "beta_selection": fusion.beta_selection,
        "validation_gates": {
            "beta0_max_abs_diff": float(fusion.beta0_max_abs_diff),
            "predicted_class_distance_consistency_max_abs_diff": consistency_max_abs_diff,
        },
        "argmax_change_rate": float(np.mean(changed)),
        "changed_to_correct": changed_to_correct,
        "changed_to_wrong": changed_to_wrong,
        "net_flips": net_flips,
        "distance_matrix": test_details["distance_matrix"],
    }


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    flat_rows: List[Dict[str, Any]] = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k not in ("metrics", "decision_audit", "nll_subsets")}
        for mk, mv in row["metrics"].items():
            flat[mk] = mv
        # Flatten decision_audit sub-dict with da_ prefix so all decision fields land in CSV.
        if "decision_audit" in row and isinstance(row["decision_audit"], dict):
            for dk, dv in row["decision_audit"].items():
                flat[f"da_{dk}"] = dv
        # Flatten nll_subsets sub-dict with nll_ prefix.
        if "nll_subsets" in row and isinstance(row["nll_subsets"], dict):
            for nk, nv in row["nll_subsets"].items():
                if nk not in flat:  # don't clobber existing top-level keys
                    flat[nk] = nv
        # keep concise in csv
        if isinstance(flat.get("beta_selection"), dict):
            flat["beta_selection"] = json.dumps(flat["beta_selection"])
        if isinstance(flat.get("validation_gates"), dict):
            flat["validation_gates"] = json.dumps(flat["validation_gates"])
        if isinstance(flat.get("lambda_selection"), dict):
            flat["lambda_selection"] = json.dumps(flat["lambda_selection"])
        if isinstance(flat.get("structural_axes"), dict):
            flat["structural_axes"] = json.dumps(flat["structural_axes"])
        flat_rows.append(flat)

    fieldnames: List[str] = []
    for r in flat_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in flat_rows:
            writer.writerow(r)


def _sanitize_npz_key(name: str) -> str:
    """Replace characters invalid in NPZ keys (anything not alphanumeric or underscore)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _write_csv_atomic(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    _write_csv(tmp, rows)
    os.replace(tmp, path)


def _assert_argmax_map_matches_registry(
    method_can_change_argmax: Dict[str, bool]
) -> None:
    """Fail loudly if any method's runtime flag contradicts the canonical registry.

    `method_can_change_argmax_map` is written at many call sites; this is the
    single gate that stops those writes drifting away from
    utils/method_metadata.py. Methods absent from the registry (externally
    imported probability sets) are reported but not fatal -- they carry their
    own provenance.
    """
    mismatches = []
    unregistered = []
    for name, flag in sorted(method_can_change_argmax.items()):
        sem = method_semantics(name)
        if sem is None:
            unregistered.append(name)
            continue
        if bool(flag) != bool(sem["can_change_argmax"]):
            mismatches.append(
                f"{name}: runtime={bool(flag)} registry={bool(sem['can_change_argmax'])}"
            )
    if unregistered:
        logger.warning(
            "Methods with no canonical semantics registry entry (external imports?): %s",
            ", ".join(unregistered),
        )
    if mismatches:
        raise RuntimeError(
            "can_change_argmax disagrees with utils/method_metadata.py for: "
            + "; ".join(mismatches)
        )


def _write_per_sample_npz(
    run_dir: str,
    labels_test: np.ndarray,
    base_probs_test: np.ndarray,
    gc_dac_anchor_ct_test: np.ndarray,
    gc_dac_top_pred_test: np.ndarray,
    method_probs_by_name: Dict[str, np.ndarray],
    method_can_change_argmax: Dict[str, bool],
    artifact_ctx: Dict[str, Any] | None = None,
    labels_val: np.ndarray | None = None,
    base_probs_val: np.ndarray | None = None,
    full_vector_distance_fusion_probs_val: np.ndarray | None = None,
) -> str:
    per_sample_dir = os.path.join(run_dir, "per_sample")
    os.makedirs(per_sample_dir, exist_ok=True)
    npz_path = os.path.join(per_sample_dir, "per_sample_arrays.npz")
    manifest_path = os.path.join(per_sample_dir, "per_sample_manifest.json")

    if artifact_ctx is None:
        artifact_ctx = {}

    distance_arrays: Dict[str, np.ndarray] = artifact_ctx.get("distance_arrays", {})
    method_artifact_metadata: Dict[str, Dict[str, Any]] = artifact_ctx.get(
        "method_artifact_metadata", {}
    )
    selected_params: Dict[str, Dict[str, Any]] = artifact_ctx.get("selected_params", {})
    missing_fields: List[Dict[str, str]] = list(artifact_ctx.get("missing_fields", []))
    run_meta: Dict[str, Any] = artifact_ctx.get("run_meta", {})

    _assert_argmax_map_matches_registry(method_can_change_argmax)
    method_index = sorted(method_probs_by_name.keys())
    n_samples = int(labels_test.shape[0])
    base_pred = np.argmax(base_probs_test, axis=1).astype(np.int64)

    # Legacy keys — preserved exactly for backward compatibility.
    payload: Dict[str, Any] = {
        "labels_test": labels_test.astype(np.int64),
        "base_probs_test": base_probs_test.astype(np.float64),
        "gc_dac_anchor_ct_test": gc_dac_anchor_ct_test.astype(np.float64),
        "gc_dac_top_pred_test": gc_dac_top_pred_test.astype(np.int64),
        "method_index": np.array(method_index, dtype=np.str_),
        "method_can_change_argmax": np.array(
            [bool(method_can_change_argmax[m]) for m in method_index], dtype=np.bool_
        ),
    }
    validation_arrays = (
        labels_val,
        base_probs_val,
        full_vector_distance_fusion_probs_val,
    )
    if any(arr is not None for arr in validation_arrays):
        if any(arr is None for arr in validation_arrays):
            raise ValueError("Validation artifact arrays must be provided together")
        assert labels_val is not None
        assert base_probs_val is not None
        assert full_vector_distance_fusion_probs_val is not None
        if len(labels_val) != len(base_probs_val) or len(labels_val) != len(
            full_vector_distance_fusion_probs_val
        ):
            raise ValueError(
                "Validation labels, base probabilities, and fused probabilities must align"
            )
        # Persist the extraction-aligned validation order used by fusion tuning.
        payload.update(
            {
                "labels_val": labels_val.astype(np.int64),
                "base_probs_val": base_probs_val.astype(np.float64),
                "method_probs__base_model_val": base_probs_val.astype(np.float64),
                "method_probs__full_vector_distance_fusion_val": (
                    full_vector_distance_fusion_probs_val.astype(np.float64)
                ),
            }
        )
    for method_name in method_index:
        payload[f"method_probs__{method_name}"] = method_probs_by_name[method_name].astype(
            np.float64
        )

    # Stable row index and base prediction.
    payload["sample_idx"] = np.arange(n_samples, dtype=np.int64)
    payload["base_pred"] = base_pred

    # Per-method argmax predictions and flip diagnostics (decision-changing methods only).
    for method_name in method_index:
        skey = _sanitize_npz_key(method_name)
        method_pred = np.argmax(method_probs_by_name[method_name], axis=1).astype(np.int64)
        payload[f"method_pred__{skey}"] = method_pred
        if method_can_change_argmax.get(method_name, False):
            changed = method_pred != base_pred
            payload[f"argmax_changed__{skey}"] = changed
            payload[f"flip_to_correct__{skey}"] = (
                changed & (base_pred != labels_test) & (method_pred == labels_test)
            )
            payload[f"flip_to_wrong__{skey}"] = (
                changed & (base_pred == labels_test) & (method_pred != labels_test)
            )

    # Selected parameters as float32 scalars keyed per method.
    for method_name, params in selected_params.items():
        skey = _sanitize_npz_key(method_name)
        for param_name, param_val in params.items():
            pkey = _sanitize_npz_key(param_name)
            try:
                payload[f"selected_{pkey}__{skey}"] = np.array(
                    float(param_val), dtype=np.float32
                )
            except (TypeError, ValueError):
                missing_fields.append(
                    {
                        "field": f"selected_param:{method_name}.{param_name}",
                        "reason": "non-scalar value skipped",
                    }
                )

    # Distance arrays as float32 — [N,C] matrices or [N,C,k] top-k tensors.
    stored_distance_keys: Dict[str, str] = {}
    for dist_name, dist_arr in distance_arrays.items():
        if dist_arr is None:
            missing_fields.append(
                {"field": f"distance_array:{dist_name}", "reason": "array is None"}
            )
            continue
        dist_arr = np.asarray(dist_arr)
        if dist_arr.ndim not in (2, 3):
            missing_fields.append(
                {
                    "field": f"distance_array:{dist_name}",
                    "reason": f"unexpected ndim={dist_arr.ndim}",
                }
            )
            continue
        npz_dkey = _sanitize_npz_key(dist_name)
        payload[npz_dkey] = dist_arr.astype(np.float32)
        stored_distance_keys[dist_name] = npz_dkey

    np.savez_compressed(npz_path, **payload)

    # Manifest — written alongside the NPZ regardless of geometry availability.
    array_descriptors: List[Dict[str, Any]] = [
        {"name": k, "shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)}
        for k, v in payload.items()
    ]
    manifest: Dict[str, Any] = {
        "artifact_schema_version": 2,
        "arrays": array_descriptors,
        "method_names": method_index,
        "method_can_change_argmax": {
            m: bool(method_can_change_argmax.get(m, False)) for m in method_index
        },
        "distance_array_keys": stored_distance_keys,
        "score_reconstruction": _json_ready(method_artifact_metadata),
        "selected_params": _json_ready(selected_params),
        "missing_fields": missing_fields,
    }
    manifest.update(_json_ready(run_meta))

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Unified per-sample manifest: %s", manifest_path)
    return npz_path


def _iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _intermediates_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "intermediates")


def _stage_dir(output_dir: str, name: str) -> str:
    path = os.path.join(_intermediates_dir(output_dir), name)
    os.makedirs(path, exist_ok=True)
    return path


def _progress_path(output_dir: str) -> str:
    return os.path.join(_intermediates_dir(output_dir), "progress.json")


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2)
    os.replace(tmp, path)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_npy(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def _save_npy_atomic(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.npy"
    np.save(tmp_path, arr)
    os.replace(tmp_path, path)


def _load_progress(output_dir: str, args_fingerprint: str) -> Dict[str, Any]:
    p = _progress_path(output_dir)
    if os.path.exists(p):
        progress = _load_json(p)
        existing_fp = str(progress.get("args_fingerprint", ""))
        if existing_fp and existing_fp != args_fingerprint:
            raise RuntimeError(
                "Existing intermediates were created with different args. "
                f"Expected fingerprint {args_fingerprint}, found {existing_fp}."
            )
        return progress
    return {
        "version": 1,
        "status": "running",
        "args_fingerprint": args_fingerprint,
        "completed_stages": [],
        "stages": {},
    }


def _write_progress(output_dir: str, progress: Dict[str, Any]) -> None:
    _atomic_write_json(_progress_path(output_dir), progress)


def _mark_stage_started(output_dir: str, progress: Dict[str, Any], stage: str) -> None:
    stages = progress.setdefault("stages", {})
    stage_obj = stages.setdefault(stage, {})
    stage_obj["status"] = "running"
    stage_obj["started_at"] = _iso_now()
    _write_progress(output_dir, progress)


def _mark_stage_complete(
    output_dir: str,
    progress: Dict[str, Any],
    stage: str,
    files: Dict[str, str] | None = None,
) -> None:
    stages = progress.setdefault("stages", {})
    stage_obj = stages.setdefault(stage, {})
    stage_obj["status"] = "complete"
    stage_obj["completed_at"] = _iso_now()
    if files:
        stage_obj["files"] = files
    completed = set(progress.setdefault("completed_stages", []))
    completed.add(stage)
    progress["completed_stages"] = sorted(completed)
    _write_progress(output_dir, progress)


def resolve_device(requested: str) -> torch.device:
    """Resolve the requested device without ever silently downgrading CUDA to CPU."""
    device = torch.device(requested)
    if device.type == "cpu":
        return device
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {requested!r} was requested but torch.cuda.is_available() "
            "is False. Refusing to silently fall back to CPU; fix the CUDA "
            "environment/allocation or pass --device cpu explicitly."
        )
    return device


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stage_complete(progress: Dict[str, Any], stage: str) -> bool:
    if stage not in set(progress.get("completed_stages", [])):
        return False
    stage_files = progress.get("stages", {}).get(stage, {}).get("files", {})
    return all(os.path.exists(p) for p in stage_files.values())


def _stage2_required_methods(args: argparse.Namespace) -> List[str]:
    """Methods whose checkpoints must be complete for stage2_early to count as done."""
    required_methods = [
        "base_model",
        "temperature_scaling",
        "vector_scaling",
        "beta_calibration",
        "ovr_isotonic",
        "odir_dirichlet",
        "rgcl",
        "gc_dac",
        "anchored_model_tail",
    ]
    if not getattr(args, "disable_rgcc", False):
        required_methods.append("rgcc")
    if not getattr(args, "disable_gc_tulip", False):
        required_methods.append("gc_tulip")
    if args.enable_rgcl_tail_hybrids:
        requested_tail_sources = {
            s.strip().lower() for s in args.rgcl_tail_sources.split(",") if s.strip()
        }
        tail_method_map = {
            "base": "rgcl_tail_base",
            "temperature_scaling": "rgcl_tail_temperature_scaling",
            "vector_scaling": "rgcl_tail_vector_scaling",
            "dirichlet": "rgcl_tail_dirichlet",
        }
        for k, v in tail_method_map.items():
            if k in requested_tail_sources:
                required_methods.append(v)
    return required_methods


def _args_fingerprint(args: argparse.Namespace) -> str:
    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "method": args.method,
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "target_dimension": int(args.target_dimension),
        "num_layers": int(args.num_layers),
        "num_coordinates": int(args.num_coordinates),
        "enable_post_fusion_temperature": bool(args.enable_post_fusion_temperature),
        "enable_rgcl_tail_hybrids": bool(args.enable_rgcl_tail_hybrids),
        "rgcl_tail_sources": str(args.rgcl_tail_sources),
    }
    # Preserve the historical fingerprint for L2 runs while ensuring that a
    # non-L2 invocation cannot reuse incompatible L2 intermediates.
    stab_metric = str(getattr(args, "stab_metric", "l2"))
    if stab_metric != "l2":
        payload["stab_metric"] = stab_metric
        if stab_metric == "whitened_cosine":
            payload.update(
                {
                    "whitening_components": int(args.whitening_components),
                    "whitening_eps": float(args.whitening_eps),
                }
            )

    # Full-KCal invocations likewise cannot reuse incompatible completed stages.
    if bool(getattr(args, "enable_kcal_baseline", False)) or bool(
        getattr(args, "enable_kcal_factorial", False)
    ):
        payload.update(
            {
                "enable_kcal_baseline": bool(
                    getattr(args, "enable_kcal_baseline", False)
                ),
                "kcal_projection": str(args.kcal_projection),
                "kcal_projection_dim": args.kcal_projection_dim,
                "kcal_projection_epochs": int(args.kcal_projection_epochs),
                "kcal_references_per_class": int(args.kcal_references_per_class),
                "kcal_bandwidth_folds": int(args.kcal_bandwidth_folds),
            }
        )
    if bool(getattr(args, "enable_kcal_factorial", False)):
        payload.update(
            {
                "enable_kcal_factorial": True,
                "kcal_factorial_scope": str(args.kcal_factorial_scope),
                "kcal_factorial_fusions": str(args.kcal_factorial_fusions),
                "kcal_factorial_k_per_class": int(
                    args.kcal_factorial_k_per_class
                ),
                "kcal_factorial_cv_folds": int(args.kcal_factorial_cv_folds),
                "kcal_factorial_strength_grid": str(
                    args.kcal_factorial_strength_grid
                ),
                "kcal_factorial_alpha_grid": str(args.kcal_factorial_alpha_grid),
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _method_store_dir(output_dir: str) -> str:
    root = os.path.join(_intermediates_dir(output_dir), "method_outputs")
    os.makedirs(root, exist_ok=True)
    return root


def _method_checkpoint_paths(output_dir: str, method_name: str) -> Dict[str, str]:
    method_dir = os.path.join(_method_store_dir(output_dir), method_name)
    os.makedirs(method_dir, exist_ok=True)
    return {
        "dir": method_dir,
        "entry": os.path.join(method_dir, "entry.json"),
        "probs": os.path.join(method_dir, "probs.npy"),
        "meta": os.path.join(method_dir, "meta.json"),
    }


def _method_registry_path(output_dir: str) -> str:
    return os.path.join(_method_store_dir(output_dir), "method_registry.json")


def _load_method_registry(output_dir: str) -> Dict[str, Any]:
    path = _method_registry_path(output_dir)
    if not os.path.exists(path):
        return {
            "ordered_methods": [],
            "method_paths": {},
            "method_can_change_argmax": {},
            "method_stage": {},
        }
    reg = _load_json(path)
    reg.setdefault("ordered_methods", [])
    reg.setdefault("method_paths", {})
    reg.setdefault("method_can_change_argmax", {})
    reg.setdefault("method_stage", {})
    return reg


def _save_method_registry(output_dir: str, registry: Dict[str, Any]) -> None:
    _atomic_write_json(_method_registry_path(output_dir), registry)


def _is_method_checkpoint_complete(
    output_dir: str,
    method_name: str,
    expected_rows: int | None = None,
    expected_classes: int | None = None,
    expected_gate_signature: str | None = None,
) -> bool:
    paths = _method_checkpoint_paths(output_dir, method_name)
    if not (os.path.exists(paths["entry"]) and os.path.exists(paths["probs"]) and os.path.exists(paths["meta"])):
        return False
    try:
        meta = _load_json(paths["meta"])
        if not bool(meta.get("completed", False)):
            return False
        if method_name in CONTRASTIVE_BETA_METHOD_NAMES:
            if expected_gate_signature is None:
                expected_gate_signature = _contrastive_gate_signature(
                    DEFAULT_CONTRASTIVE_NLL_TOLERANCE,
                    DEFAULT_CONTRASTIVE_BRIER_TOLERANCE,
                    DEFAULT_CONTRASTIVE_ECE_ABS_TOLERANCE,
                    post_temperature=method_name.endswith("_post_temperature"),
                )
            if meta.get("gate_signature") != expected_gate_signature:
                return False
        probs = np.load(paths["probs"], mmap_mode="r")
        if expected_rows is not None and expected_classes is not None:
            if probs.shape != (expected_rows, expected_classes):
                return False
        meta_shape = tuple(meta.get("probability_shape", []))
        if meta_shape and tuple(probs.shape) != meta_shape:
            return False
        return True
    except Exception:
        return False


def _save_method_checkpoint(
    output_dir: str,
    method_name: str,
    entry: Dict[str, Any],
    probs: np.ndarray,
    can_change_argmax: bool,
    stage_name: str,
) -> None:
    paths = _method_checkpoint_paths(output_dir, method_name)
    now = _iso_now()
    meta_payload = {
        "method_name": method_name,
        "stage_name": stage_name,
        "can_change_argmax": bool(can_change_argmax),
        "probability_shape": [int(x) for x in probs.shape],
        "dtype": str(probs.dtype),
        "completed": True,
        "updated_at": now,
    }
    if method_name in CONTRASTIVE_BETA_METHOD_NAMES:
        meta_payload["gate_signature"] = entry.get("gate_signature")
    if os.path.exists(paths["meta"]):
        try:
            prev = _load_json(paths["meta"])
            meta_payload["created_at"] = str(prev.get("created_at", now))
        except Exception:
            meta_payload["created_at"] = now
    else:
        meta_payload["created_at"] = now

    _atomic_write_json(paths["entry"], entry)
    _save_npy_atomic(paths["probs"], probs)
    _atomic_write_json(paths["meta"], meta_payload)

    registry = _load_method_registry(output_dir)
    ordered = list(registry.get("ordered_methods", []))
    if method_name not in ordered:
        ordered.append(method_name)
    method_paths = dict(registry.get("method_paths", {}))
    method_paths[method_name] = {
        "entry": paths["entry"],
        "probs": paths["probs"],
        "meta": paths["meta"],
    }
    change_map = dict(registry.get("method_can_change_argmax", {}))
    change_map[method_name] = bool(can_change_argmax)
    stage_map = dict(registry.get("method_stage", {}))
    stage_map[method_name] = stage_name
    registry["ordered_methods"] = ordered
    registry["method_paths"] = method_paths
    registry["method_can_change_argmax"] = change_map
    registry["method_stage"] = stage_map
    _save_method_registry(output_dir, registry)
    try:
        _write_incremental_summary(output_dir)
    except Exception as e:
        logger.error("Failed to save incremental summary: %s", e)


def _load_method_entry(output_dir: str, method_name: str) -> Dict[str, Any]:
    return _load_json(_method_checkpoint_paths(output_dir, method_name)["entry"])


def _load_method_probs(
    output_dir: str, method_name: str, mmap_mode: str | None = None
) -> np.ndarray:
    return np.load(_method_checkpoint_paths(output_dir, method_name)["probs"], mmap_mode=mmap_mode)


def _load_completed_rankgeom_resume_state(
    output_dir: str, rankgeom_selection: Dict[str, Any] | None
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Resume path for a completed anchored_rankgeom_tail_mixture checkpoint.

    Needs only on-disk artifacts (no model, adapter or train_raw). The selected
    lambda/alpha come from rankgeom_selection.json when present, else from the
    method checkpoint entry; if neither has them this raises rather than
    silently substituting (0, 0).
    """
    method = "anchored_rankgeom_tail_mixture"
    probs = _load_method_probs(output_dir, method, mmap_mode="r")
    if rankgeom_selection is not None and {
        "selected_lambda",
        "selected_alpha",
    } <= set(rankgeom_selection):
        return probs, rankgeom_selection
    entry = _load_method_entry(output_dir, method)
    if "selected_lambda" not in entry or "selected_alpha" not in entry:
        raise RuntimeError(
            f"Completed {method} checkpoint in {output_dir} has no recoverable "
            "selected_lambda/selected_alpha (rankgeom_selection.json missing and "
            "not present in the method entry); refusing to guess."
        )
    return probs, {
        "selected_lambda": float(entry["selected_lambda"]),
        "selected_alpha": float(entry["selected_alpha"]),
    }


def _load_all_method_entries_in_order(output_dir: str) -> List[Dict[str, Any]]:
    registry = _load_method_registry(output_dir)
    entries = [_load_method_entry(output_dir, m) for m in registry.get("ordered_methods", [])]
    _assert_uniform_semantic_schema(entries, output_dir)
    return entries


def _assert_uniform_semantic_schema(
    entries: List[Dict[str, Any]], output_dir: str
) -> None:
    """Refuse to assemble a summary that mixes metric semantics.

    A method checkpoint written before SEMANTIC_SCHEMA_VERSION carries
    surrogate-derived scalar metrics (and a fabricated NLL/Brier for
    scalar-only methods). Resuming such a run would silently combine those
    rows with correctly-scored new rows in one summary. Rather than guess, we
    stop and point at the two ways forward: re-score the stale run with
    Experiments/rescore_scalar_semantics.py, or start a fresh output_dir.
    """
    stale = sorted(
        str(e.get("method_name"))
        for e in entries
        if e.get("semantic_schema_version") != SEMANTIC_SCHEMA_VERSION
    )
    if not stale:
        return
    raise RuntimeError(
        f"{output_dir}: {len(stale)} method checkpoint(s) were written under an "
        f"older metric semantics than {SEMANTIC_SCHEMA_VERSION!r} and cannot be "
        f"mixed into one summary: {', '.join(stale[:12])}"
        + (" ..." if len(stale) > 12 else "")
        + ". Either re-score the existing run in place with "
        "`python -m Experiments.rescore_scalar_semantics --run_dir <dir>` (no "
        "refitting needed; it writes a new derived/ artifact and leaves the run "
        "untouched), or re-run into a fresh --output_dir. Existing results are "
        "never rewritten automatically."
    )


def _load_method_can_change_argmax_map(output_dir: str) -> Dict[str, bool]:
    registry = _load_method_registry(output_dir)
    return {
        str(k): bool(v) for k, v in registry.get("method_can_change_argmax", {}).items()
    }


def _summary_json_path(output_dir: str) -> str:
    return os.path.join(output_dir, "summary_metrics.json")


def _summary_csv_path(output_dir: str) -> str:
    return os.path.join(output_dir, "summary_metrics.csv")


def _build_summary_metadata(
    args: argparse.Namespace,
    model_path: str,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
) -> Dict[str, Any]:
    metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "method": args.method,
        "seed": int(args.seed),
        "checkpoint_path": model_path,
        "split_sizes": {
            "train": int(len(train_labels)),
            "validation": int(len(val_labels)),
            "test": int(len(test_labels)),
        },
        "shared_metric_module": "utils.unified_metrics",
        "stability_space": {
            "metric": args.stab_metric,
            "library": "fast_separation",
            "whitening_components": int(args.whitening_components),
            "whitening_eps": float(args.whitening_eps),
        },
    }
    if getattr(args, "reuse_non_metric_from", ""):
        metadata["reused_non_metric_from"] = str(args.reuse_non_metric_from)
    if args.enable_full_vector_geometric_fusion:
        metadata["generalized_fusion_feature_source"] = args.fusion_feature_source
    if bool(getattr(args, "enable_kcal_factorial", False)):
        metadata["kcal_factorial"] = {
            "scope": args.kcal_factorial_scope,
            "fusions": args.kcal_factorial_fusions,
            "k_per_class": int(args.kcal_factorial_k_per_class),
            "cv_folds": int(args.kcal_factorial_cv_folds),
            "strength_grid": args.kcal_factorial_strength_grid,
            "alpha_grid": args.kcal_factorial_alpha_grid,
            "distance_scaling": "median_reference_bank_distance",
        }
    return metadata


def _load_summary_metadata(output_dir: str) -> Dict[str, Any]:
    progress_path = _progress_path(output_dir)
    if not os.path.exists(progress_path):
        return {}
    try:
        progress = _load_json(progress_path)
    except Exception:
        return {}
    metadata = progress.get("summary_metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _write_incremental_summary(
    output_dir: str,
    metadata: Dict[str, Any] | None = None,
    write_csv: bool = True,
) -> tuple[str, str, List[Dict[str, Any]]]:
    os.makedirs(output_dir, exist_ok=True)
    if metadata is None:
        metadata = _load_summary_metadata(output_dir)
    methods = _load_all_method_entries_in_order(output_dir)
    summary = {
        "metadata": metadata,
        "methods": methods,
    }
    json_path = _summary_json_path(output_dir)
    csv_path = _summary_csv_path(output_dir)
    _atomic_write_json(json_path, summary)
    if write_csv and methods:
        _write_csv_atomic(csv_path, methods)
    logger.info("Results saved incrementally to: %s", json_path)
    return json_path, csv_path, methods


def _patch_contrastive_per_sample_npz(
    npz_path: str,
    outputs: Dict[str, tuple[Dict[str, Any], np.ndarray]],
) -> None:
    """Replace only contrastive arrays in an otherwise completed per-sample artifact."""
    with np.load(npz_path, allow_pickle=False) as stored:
        payload = {key: stored[key] for key in stored.files}

    previous_methods = [str(name) for name in payload.get("method_index", [])]
    previous_flags = [bool(value) for value in payload.get("method_can_change_argmax", [])]
    can_change = dict(zip(previous_methods, previous_flags))
    method_names = sorted(set(previous_methods) | set(outputs))
    labels = np.asarray(payload["labels_test"])
    base_pred = np.asarray(payload.get("base_pred", np.argmax(payload["base_probs_test"], axis=1)))
    for method_name, (_, probs) in outputs.items():
        skey = _sanitize_npz_key(method_name)
        pred = np.argmax(probs, axis=1).astype(np.int64)
        changed = pred != base_pred
        payload[f"method_probs__{method_name}"] = np.asarray(probs, dtype=np.float64)
        payload[f"method_pred__{skey}"] = pred
        payload[f"argmax_changed__{skey}"] = changed
        payload[f"flip_to_correct__{skey}"] = changed & (base_pred != labels) & (pred == labels)
        payload[f"flip_to_wrong__{skey}"] = changed & (base_pred == labels) & (pred != labels)
        can_change[method_name] = True
    payload["method_index"] = np.asarray(method_names, dtype=np.str_)
    payload["method_can_change_argmax"] = np.asarray(
        [can_change.get(name, False) for name in method_names], dtype=np.bool_
    )

    tmp_path = f"{npz_path}.tmp.npz"
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, npz_path)

    manifest_path = os.path.join(os.path.dirname(npz_path), "per_sample_manifest.json")
    if os.path.exists(manifest_path):
        manifest = _load_json(manifest_path)
        manifest["method_names"] = method_names
        manifest["method_can_change_argmax"] = {
            name: bool(can_change.get(name, False)) for name in method_names
        }
        manifest["arrays"] = [
            {
                "name": key,
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
            }
            for key, value in payload.items()
        ]
        _atomic_write_json(manifest_path, manifest)


def _refresh_completed_contrastive_methods(
    args: argparse.Namespace,
    expected_signatures: Dict[str, str],
) -> bool:
    """Refresh stale contrastive checkpoints without traversing unrelated methods."""
    if args.fusion_feature_source != "single_layer":
        logger.warning(
            "Cannot use contrastive-only completed-run refresh with RGCL distances; "
            "falling back to the normal benchmark path."
        )
        return False

    feature_dir = os.path.join(_intermediates_dir(args.output_dir), "features")
    fusion_dir = os.path.join(_intermediates_dir(args.output_dir), "fusion")
    early_dir = os.path.join(_intermediates_dir(args.output_dir), "early")
    per_sample_npz = os.path.join(args.output_dir, "per_sample", "per_sample_arrays.npz")

    def _first_existing(*candidates: str) -> str | None:
        return next((p for p in candidates if os.path.exists(p)), None)

    _val_base_probs_path = _first_existing(os.path.join(fusion_dir, "val_base_probs.npy"))
    _test_base_probs_path = _first_existing(os.path.join(fusion_dir, "test_base_probs.npy"))
    paths = {
        # Prefer feature-dir logits (RGCL-aligned). When features/ was cleaned up, fall
        # back to log-probabilities derived from the fusion base-prob arrays, which are
        # already in RGCL-aligned order and consistent with val_labels / test_labels below.
        # NOTE: early/logits_val.npy is in the *original* DataLoader order, NOT the
        # RGCL-aligned order, so it must NOT be used as a fallback here.
        "val_logits": _first_existing(os.path.join(feature_dir, "val_logits.npy"))
        or (f"__logprob__:{_val_base_probs_path}" if _val_base_probs_path else None),
        "test_logits": _first_existing(os.path.join(feature_dir, "test_logits.npy"))
        or (f"__logprob__:{_test_base_probs_path}" if _test_base_probs_path else None),
        "val_labels": _first_existing(os.path.join(feature_dir, "val_labels.npy")),
        "test_labels": _first_existing(os.path.join(feature_dir, "test_labels.npy")),
        "val_base_probs": _val_base_probs_path,
        "test_base_probs": _test_base_probs_path,
        "val_distances": _first_existing(os.path.join(fusion_dir, "val_distance_matrix.npy")),
        "test_distances": _first_existing(os.path.join(fusion_dir, "test_distance_matrix.npy")),
        "anchor_ct": _first_existing(os.path.join(early_dir, "gc_dac_anchor_ct_test.npy")),
        "anchor_base_probs": _first_existing(os.path.join(early_dir, "base_probs_test.npy")),
    }

    # Fall back to per_sample_arrays.npz for labels when feature dir was cleaned up.
    # per_sample/labels_val and per_sample/labels_test are stored in RGCL-aligned order
    # (same order as fusion/val_base_probs and fusion/test_base_probs).
    _per_sample_loaded: Dict[str, np.ndarray] | None = None
    for _label_key, _npz_key in (("val_labels", "labels_val"), ("test_labels", "labels_test")):
        if paths[_label_key] is None and os.path.exists(per_sample_npz):
            if _per_sample_loaded is None:
                _per_sample_loaded = dict(np.load(per_sample_npz))
            if _npz_key in _per_sample_loaded:
                paths[_label_key] = f"__npz__:{_npz_key}"

    missing = [k for k, v in paths.items() if v is None]
    if missing:
        logger.warning(
            "Cannot use contrastive-only completed-run refresh because cached arrays "
            "are missing (%s); falling back to the normal benchmark path.",
            ", ".join(missing),
        )
        return False

    def _load_path(p: str) -> np.ndarray:
        if p.startswith("__npz__:"):
            key = p[len("__npz__:"):]
            return np.load(per_sample_npz)[key]
        if p.startswith("__logprob__:"):
            probs = np.load(p[len("__logprob__:"):]).astype(np.float64)
            return np.log(np.clip(probs, 1e-40, 1.0))
        return np.load(p)

    test_labels = _load_path(paths["test_labels"])
    num_classes = int(_load_path(paths["test_logits"]).shape[1])
    incomplete = {
        name
        for name in CONTRASTIVE_BETA_METHOD_NAMES
        if not _is_method_checkpoint_complete(
            args.output_dir,
            name,
            len(test_labels),
            num_classes,
            expected_gate_signature=expected_signatures[name],
        )
    }
    if not incomplete:
        return True

    try:
        beta_grid = [
            float(value) for value in args.contrastive_beta_grid.split(",") if value.strip()
        ]
    except ValueError as error:
        raise ValueError(
            f"--contrastive_beta_grid must contain numeric values: {error}"
        ) from error
    logger.info(
        "Completed-run refresh: recomputing only contrastive methods: %s",
        ", ".join(sorted(incomplete)),
    )
    outputs = _build_contrastive_beta_outputs(
        val_logits=np.asarray(_load_path(paths["val_logits"])),
        test_logits=np.asarray(_load_path(paths["test_logits"])),
        val_labels=_load_path(paths["val_labels"]),
        test_labels=test_labels,
        val_base_probs=_load_path(paths["val_base_probs"]),
        test_base_probs=_load_path(paths["test_base_probs"]),
        val_distances=_load_path(paths["val_distances"]),
        test_distances=_load_path(paths["test_distances"]),
        beta_grid=beta_grid,
        nll_tolerance=args.contrastive_nll_tolerance,
        brier_tolerance=args.contrastive_brier_tolerance,
        ece_abs_tolerance=args.contrastive_ece_abs_tolerance,
        inner_val_fraction=args.inner_val_fraction,
        inner_val_seed=args.inner_val_seed,
        feature_source="single_layer",
        method_names=incomplete,
        anchor_ct=_load_path(paths["anchor_ct"]),
        anchor_base_probs=_load_path(paths["anchor_base_probs"]),
    )
    for method_name, (entry, probs) in outputs.items():
        _save_method_checkpoint(
            args.output_dir,
            method_name,
            entry,
            probs,
            True,
            stage_name="stage4_late_outputs",
        )
    _patch_contrastive_per_sample_npz(
        os.path.join(args.output_dir, "per_sample", "per_sample_arrays.npz"), outputs
    )
    _write_incremental_summary(args.output_dir)
    return True


def _late_anchor_intermediate_paths(output_dir: str) -> Dict[str, str]:
    """Return file paths for expensive late-stage anchor intermediates."""
    late_dir = _stage_dir(output_dir, "late")
    return {
        "val_anchor_c_t": os.path.join(late_dir, "val_anchor_c_t.npy"),
        "val_anchor_diag": os.path.join(late_dir, "val_anchor_diag.json"),
        "rankgeom_selection": os.path.join(late_dir, "rankgeom_selection.json"),
        "anchored_rankgeom_probs": os.path.join(late_dir, "anchored_rankgeom_tail_mixture_probs.npy"),
    }


_STAB_METRIC_DEPENDENT_METHODS = {
    "anchored_rankgeom_tail_mixture",
    "full_vector_distance_fusion",
    "post_fusion_topiso",
    "full_vector_distance_fusion_post_temperature",
    "trust_score_original_diagnostic",
    "trust_score_original_switch",
    "aar_lightweight",
    "glad_pi",
}
_STAB_METRIC_DEPENDENT_PREFIXES = (
    "full_vector_geometric_fusion_",
    "softmax_knn_blend",
    "kcal_lite",
    "rgcl_neighbor_correction_",
    "contrastive_beta_",
)


def _method_depends_on_stab_metric(method_name: str) -> bool:
    return method_name in _STAB_METRIC_DEPENDENT_METHODS or method_name.startswith(
        _STAB_METRIC_DEPENDENT_PREFIXES
    )


def _link_or_copy_reused_file(source: str, target: str) -> None:
    """Reuse large immutable artifacts with a hard link when possible."""
    if os.path.exists(target):
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _reuse_non_metric_outputs(
    source_output_dir: str,
    target_output_dir: str,
    stage2_files: Dict[str, str],
    progress: Dict[str, Any],
    expected_rows: int,
    expected_classes: int,
) -> List[str]:
    """Import completed metric-invariant/L2 baselines without executing them."""
    source_output_dir = os.path.abspath(source_output_dir)
    target_output_dir = os.path.abspath(target_output_dir)
    if source_output_dir == target_output_dir:
        raise ValueError("--reuse_non_metric_from must differ from --output_dir")
    if not os.path.isdir(source_output_dir):
        raise FileNotFoundError(
            f"L2 reuse directory does not exist: {source_output_dir}"
        )

    source_early = os.path.join(_intermediates_dir(source_output_dir), "early")
    reused_array_sources = {
        "base_probs_test": os.path.join(source_early, "base_probs_test.npy"),
        "gc_dac_probs": os.path.join(source_early, "gc_dac_probs.npy"),
        "gc_dac_top_idx_test": os.path.join(source_early, "gc_dac_top_idx_test.npy"),
        "gc_dac_anchor_ct_test": os.path.join(
            source_early, "gc_dac_anchor_ct_test.npy"
        ),
        "logits_val": os.path.join(source_early, "logits_val.npy"),
        "logits_test": os.path.join(source_early, "logits_test.npy"),
    }
    missing_arrays = [
        path for path in reused_array_sources.values() if not os.path.isfile(path)
    ]
    if missing_arrays:
        raise FileNotFoundError(
            "L2 reuse directory is missing required early artifacts: "
            + ", ".join(missing_arrays)
        )
    for key, source in reused_array_sources.items():
        _link_or_copy_reused_file(source, stage2_files[key])

    source_registry = _load_method_registry(source_output_dir)
    target_registry = _load_method_registry(target_output_dir)
    target_order = list(target_registry.get("ordered_methods", []))
    target_paths = dict(target_registry.get("method_paths", {}))
    target_change_map = dict(target_registry.get("method_can_change_argmax", {}))
    target_stage_map = dict(target_registry.get("method_stage", {}))
    reused_methods: List[str] = []

    for method_name in source_registry.get("ordered_methods", []):
        if _method_depends_on_stab_metric(method_name):
            continue
        if not _is_method_checkpoint_complete(
            source_output_dir,
            method_name,
            expected_rows,
            expected_classes,
        ):
            continue
        source_paths = _method_checkpoint_paths(source_output_dir, method_name)
        target_method_paths = _method_checkpoint_paths(target_output_dir, method_name)
        for artifact in ("entry", "probs", "meta"):
            _link_or_copy_reused_file(
                source_paths[artifact], target_method_paths[artifact]
            )
        if method_name not in target_order:
            target_order.append(method_name)
        target_paths[method_name] = {
            key: target_method_paths[key] for key in ("entry", "probs", "meta")
        }
        target_change_map[method_name] = bool(
            source_registry.get("method_can_change_argmax", {}).get(
                method_name, False
            )
        )
        target_stage_map[method_name] = str(
            source_registry.get("method_stage", {}).get(
                method_name, "reused_non_metric"
            )
        )
        reused_methods.append(method_name)

    target_registry.update(
        {
            "ordered_methods": target_order,
            "method_paths": target_paths,
            "method_can_change_argmax": target_change_map,
            "method_stage": target_stage_map,
        }
    )
    _save_method_registry(target_output_dir, target_registry)

    methods_so_far = _load_all_method_entries_in_order(target_output_dir)
    _atomic_write_json(stage2_files["methods_so_far"], {"methods": methods_so_far})
    np.savez_compressed(stage2_files["method_probs_by_name"])
    _atomic_write_json(
        stage2_files["method_can_change_argmax"], target_change_map
    )

    source_anchor_paths = _late_anchor_intermediate_paths(source_output_dir)
    target_anchor_paths = _late_anchor_intermediate_paths(target_output_dir)
    for key in ("val_anchor_c_t", "val_anchor_diag"):
        if os.path.isfile(source_anchor_paths[key]):
            _link_or_copy_reused_file(
                source_anchor_paths[key], target_anchor_paths[key]
            )

    _mark_stage_complete(
        target_output_dir, progress, "stage2_early", files=stage2_files
    )
    return reused_methods


class _PenultimateFeatureCapture:
    """Capture the input to the model's final linear classifier."""

    def __init__(self, model: torch.nn.Module):
        candidates = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        ]
        if not candidates:
            raise ValueError("Full KCal requires a model with a linear classifier")
        self.layer_name, classifier = candidates[-1]
        self.features: torch.Tensor | None = None
        self._handle = classifier.register_forward_pre_hook(self._capture)

    def _capture(self, _module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("Classifier pre-hook did not receive a tensor input")
        features = inputs[0]
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        self.features = features

    def reset(self) -> None:
        self.features = None

    def cleanup(self) -> None:
        self._handle.remove()


def _extract_split_features_to_disk(
    feature_extractor: FeatureExtractor,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_prefix: str,
    include_raw: bool,
    penultimate_capture: _PenultimateFeatureCapture | None = None,
) -> tuple[str | None, str, str, str, str | None]:
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    sampler = getattr(data_loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "__len__"):
        n = len(sampler)
    else:
        n = len(getattr(data_loader, "dataset"))
    raw_mm = None
    feat_mm = None
    logits_mm = None
    labels_mm = None
    penultimate_mm = None
    idx = 0
    wrote_any = False

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(data_loader):
            data = data.to(device)
            targets = targets.to(device)

            if hasattr(feature_extractor, "features"):
                feature_extractor.features = None
            if hasattr(feature_extractor, "layer_features"):
                feature_extractor.layer_features.clear()
            if penultimate_capture is not None:
                penultimate_capture.reset()

            logits = feature_extractor.model(data)
            features = feature_extractor.features
            if features is None:
                raise RuntimeError("Feature extractor hook produced no features.")
            if len(features.shape) == 4:
                features = features.mean(dim=[2, 3])

            bsz = int(data.shape[0])
            data_cpu = data.detach().cpu().numpy()
            feat_cpu = features.detach().cpu().numpy()
            logits_cpu = logits.detach().cpu().numpy()
            labels_cpu = targets.detach().cpu().numpy()
            penultimate_cpu = None
            if penultimate_capture is not None:
                if penultimate_capture.features is None:
                    raise RuntimeError("Penultimate classifier hook produced no features")
                penultimate_cpu = penultimate_capture.features.detach().cpu().numpy()

            if feat_mm is None:
                feat_mm = np.lib.format.open_memmap(
                    f"{out_prefix}_features.npy",
                    mode="w+",
                    dtype=feat_cpu.dtype,
                    shape=(n, feat_cpu.shape[1]),
                )
                logits_mm = np.lib.format.open_memmap(
                    f"{out_prefix}_logits.npy",
                    mode="w+",
                    dtype=logits_cpu.dtype,
                    shape=(n, logits_cpu.shape[1]),
                )
                labels_mm = np.lib.format.open_memmap(
                    f"{out_prefix}_labels.npy",
                    mode="w+",
                    dtype=labels_cpu.dtype,
                    shape=(n,),
                )
                if include_raw:
                    raw_mm = np.lib.format.open_memmap(
                        f"{out_prefix}_raw.npy",
                        mode="w+",
                        dtype=data_cpu.dtype,
                        shape=(n, *data_cpu.shape[1:]),
                    )
                if penultimate_cpu is not None:
                    penultimate_mm = np.lib.format.open_memmap(
                        f"{out_prefix}_penultimate.npy",
                        mode="w+",
                        dtype=penultimate_cpu.dtype,
                        shape=(n, penultimate_cpu.shape[1]),
                    )

            end = idx + bsz
            if end > n:
                raise RuntimeError("Feature extraction wrote past expected split length.")
            if raw_mm is not None:
                raw_mm[idx:end] = data_cpu
            feat_mm[idx:end] = feat_cpu
            logits_mm[idx:end] = logits_cpu
            labels_mm[idx:end] = labels_cpu
            if penultimate_mm is not None and penultimate_cpu is not None:
                penultimate_mm[idx:end] = penultimate_cpu
            idx = end
            wrote_any = True

            del data, targets, logits, features, data_cpu, feat_cpu, logits_cpu, labels_cpu
            if penultimate_cpu is not None:
                del penultimate_cpu
            if torch.cuda.is_available() and (batch_idx % 10 == 0):
                torch.cuda.empty_cache()

    if not wrote_any or idx != n:
        raise RuntimeError(f"Feature extraction incomplete for {out_prefix}: wrote {idx} of {n}.")

    del raw_mm, feat_mm, logits_mm, labels_mm, penultimate_mm
    _cleanup_memory()
    raw_path = f"{out_prefix}_raw.npy" if include_raw else None
    penultimate_path = (
        f"{out_prefix}_penultimate.npy" if penultimate_capture is not None else None
    )
    return (
        raw_path,
        f"{out_prefix}_features.npy",
        f"{out_prefix}_logits.npy",
        f"{out_prefix}_labels.npy",
        penultimate_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified calibration benchmark runner")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, default="baseline_cross_entropy")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--results_dir", type=str, default="results/models")
    parser.add_argument("--output_dir", type=str, default="results/unified_benchmark")
    parser.add_argument(
        "--reuse_non_metric_from",
        type=str,
        default="",
        help=(
            "Completed L2 output directory whose metric-invariant and L2-only "
            "checkpoints should be reused instead of recomputed. Intended for "
            "cosine/whitened-cosine follow-up runs."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--corruption_type",
        type=str,
        default=None,
        help=(
            "CIFAR-*-C corruption name (e.g. gaussian_noise, defocus_blur, fog, "
            "jpeg_compression). Train/val stay clean (for calibration fitting); "
            "only the test split is replaced with the corrupted cell, via "
            "utils.calibration_utils.load_cifar_c_loader. Requires "
            "--corruption_severity. Leave unset for the clean protocol."
        ),
    )
    parser.add_argument(
        "--corruption_severity",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5],
        help="CIFAR-*-C severity level (1-5). Requires --corruption_type.",
    )
    parser.add_argument(
        "--cifar_c_dir",
        type=str,
        default=None,
        help=(
            "Root directory containing <corruption>.npy + labels.npy in the "
            "standard Hendrycks CIFAR-*-C layout. Defaults to "
            "utils.calibration_utils._cifar_c_root's own default "
            "(./data/<dataset>-c) if not given."
        ),
    )
    parser.add_argument(
        "--fitted_state_dir",
        type=str,
        default=None,
        help=(
            "Fit-once/evaluate-many protocol (required whenever --corruption_type "
            "is set): directory holding one pickled fitted-calibrator-state file "
            "per method. On a clean run (no --corruption_type), each method is fit "
            "on clean train/val as usual and its fitted state is persisted here. On "
            "a --corruption_type run, every method's fit step is skipped entirely -- "
            "its state is loaded from here instead -- so corrupted data can only "
            "ever reach the evaluation/calibrate path, never any .fit() call. "
            "Passing this on a clean run before the corresponding pickles exist "
            "creates them; passing it when they already exist reuses them (the "
            "clean run itself then also becomes reuse-safe/idempotent)."
        ),
    )
    parser.add_argument("--target_dimension", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_coordinates", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--enable_post_fusion_temperature", action="store_true")
    parser.add_argument("--enable_rgcl_tail_hybrids", action="store_true")
    parser.add_argument(
        "--rgcl_tail_sources",
        type=str,
        default="base,temperature_scaling,vector_scaling,dirichlet",
        help="Comma-separated tail sources for RGCL hybrids.",
    )
    parser.add_argument("--debug_rgcl_tail_hybrid_smoke", action="store_true")
    parser.add_argument("--enable_full_vector_geometric_fusion", action="store_true")
    parser.add_argument(
        "--enable_knn_blend_baseline",
        action="store_true",
        help=(
            "Run the independent softmax-kNN blend baseline "
            "(q = alpha*p_model + (1-alpha)*softmax(-gamma*D)), "
            "tuned on validation NLL. Does not require "
            "--enable_full_vector_geometric_fusion."
        ),
    )
    parser.add_argument(
        "--enable_kcal_lite_baseline",
        action="store_true",
        help=(
            "Run the KCal-lite top-k KDE-lite baseline. "
            "Unlike softmax-kNN blend, KCal-lite averages k kernel affinities per class. "
            "At k=1 it reduces to the softmax-kNN posterior. "
            "Tuned on validation NLL. Does not require "
            "--enable_full_vector_geometric_fusion."
        ),
    )
    parser.add_argument(
        "--kcal_k_per_class",
        type=int,
        default=1,
        help="Number of nearest neighbors per class for KCal-lite (default: 1).",
    )
    parser.add_argument(
        "--enable_rgcl_neighbor_correction",
        action="store_true",
        help=(
            "Run the RGCL nearest-neighbor argmax correction method. "
            "Uses the calibration split to select the trust-score or separation threshold "
            "that maximises accuracy when switching to the geometrically-nearest other class. "
            "Requires --fusion_feature_source rgcl together with "
            "--enable_full_vector_geometric_fusion, --enable_knn_blend_baseline, "
            "or --enable_kcal_lite_baseline so RGCL features are available, "
            "or it extracts them independently."
        ),
    )
    parser.add_argument(
        "--enable_kcal_baseline",
        action="store_true",
        help=(
            "Run full KCal with a learned penultimate-feature projection, "
            "cross-validated RBF bandwidth, and the complete validation KDE reference set."
        ),
    )
    parser.add_argument(
        "--kcal_projection",
        choices=["skip_elu", "bn_linear"],
        default="skip_elu",
        help="Projection architecture for full KCal.",
    )
    parser.add_argument(
        "--kcal_projection_dim",
        type=int,
        default=None,
        help="Full KCal projection width; defaults to min(penultimate dimension, num_classes).",
    )
    parser.add_argument(
        "--kcal_projection_epochs",
        type=int,
        default=50,
        help="Epochs used to learn the full KCal projection on the training split.",
    )
    parser.add_argument(
        "--kcal_references_per_class",
        type=int,
        default=32,
        help="Per-class reference sample count in each full KCal projection-training step.",
    )
    parser.add_argument(
        "--kcal_bandwidth_folds",
        type=int,
        default=20,
        help="Requested stratified folds for full KCal bandwidth selection.",
    )
    parser.add_argument(
        "--enable_kcal_factorial",
        action="store_true",
        help=(
            "Run controlled KCal/KCal-lite-RGCL factorial variants and include "
            "them in the unified table. Reuses fitted KCal-Pi and cached RGCL "
            "embeddings. Also runs the canonical full-KCal row."
        ),
    )
    parser.add_argument(
        "--kcal_factorial_scope",
        choices=["core", "full"],
        default="full",
        help=(
            "'core' runs representation x fusion with full validation-bank RBF KDE; "
            "'full' additionally crosses full/top-k estimator, validation/train "
            "reference bank, and RBF/exponential-L2 kernel (default: full)."
        ),
    )
    parser.add_argument(
        "--kcal_factorial_fusions",
        type=str,
        default="replace,blend_frozen,blend_joint",
        help=(
            "Comma-separated factorial fusion rows. blend_frozen uses the exact "
            "replacement-selected geometry posterior; blend_joint jointly tunes "
            "kernel strength and alpha."
        ),
    )
    parser.add_argument(
        "--kcal_factorial_k_per_class",
        type=int,
        default=5,
        help="Per-class neighbor count for factorial top-k cells (default: 5).",
    )
    parser.add_argument(
        "--kcal_factorial_cv_folds",
        type=int,
        default=5,
        help="Cross-fitting folds for validation-reference factorial cells (default: 5).",
    )
    parser.add_argument(
        "--kcal_factorial_strength_grid",
        type=str,
        default="0.03,0.1,0.3,1,3,10,30",
        help=(
            "Dimensionless kernel-strength grid after median-distance normalization."
        ),
    )
    parser.add_argument(
        "--kcal_factorial_alpha_grid",
        type=str,
        default="0,0.1,0.25,0.5,0.75,0.9,1",
        help="Convex-blend alpha grid for factorial blend rows.",
    )
    parser.add_argument(
        "--kcal_factorial_reference_batch_size",
        type=int,
        default=2048,
        help="Reference chunk size for factorial full-KDE evaluation (default: 2048).",
    )
    parser.add_argument(
        "--fvgf_score_modes",
        type=str,
        default="neg_distance,margin,log_trust_ratio,rank_log_trust",
        help="Comma-separated score modes for full-vector geometric fusion.",
    )
    parser.add_argument(
        "--fvgf_lambda_grid",
        type=str,
        default="",
        help="Optional comma-separated lambda grid for full-vector geometric fusion; empty uses calibrator default.",
    )
    parser.add_argument(
        "--stab_metric",
        choices=["l2", "cosine", "whitened_cosine"],
        default="l2",
        help="Geometry metric shared by full-vector distance methods (default: l2).",
    )
    parser.add_argument(
        "--whitening_components",
        type=int,
        default=128,
        help="Maximum PCA components for --stab_metric whitened_cosine (default: 128).",
    )
    parser.add_argument(
        "--whitening_eps",
        type=float,
        default=1e-6,
        help="Variance floor for --stab_metric whitened_cosine (default: 1e-6).",
    )
    parser.add_argument(
        "--fusion_feature_source",
        type=str,
        choices=["single_layer", "rgcl"],
        default="single_layer",
        help=(
            "Feature source for full-vector geometric fusion, softmax-kNN blend, "
            "and KCal-lite. "
            "'single_layer' uses the FeatureExtractor layer and preserves existing "
            "method names (default). 'rgcl' uses RGCL multi-layer projection from "
            "the top-label run and emits full_vector_geometric_fusion_rgcl_{score_mode} "
            "methods. Ignored unless at least one corresponding opt-in method is enabled."
        ),
    )
    # --- Stage A new flags ---
    parser.add_argument(
        "--enable_pts_baseline",
        action="store_true",
        help=(
            "Run the Parameterized Temperature Scaling (PTS) baseline. "
            "Predicts a per-sample positive scalar temperature from logit-derived features. "
            "Argmax-invariant by the scalar-temperature structural lemma."
        ),
    )
    parser.add_argument(
        "--inner_val_fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of the validation split reserved for hyperparameter selection "
            "(the 'select' set). The remaining fraction is used for fitting learned "
            "calibrators. Default 0.5 — chosen because the net-flip selection metric "
            "is discrete and noisy. Only used when a method requires held-out selection "
            "(e.g. GLAD-PI)."
        ),
    )
    parser.add_argument(
        "--inner_val_seed",
        type=int,
        default=123,
        help="Random seed for the inner validation split. Default 123.",
    )
    parser.add_argument(
        "--external_method_json",
        type=str,
        default=None,
        help=(
            "Path to a JSON file describing externally calibrated probability matrices "
            "to evaluate alongside native methods. Schema: "
            '{"methods": {"name": {"test_probs": "path.npy", "metadata": {...}}}}'
        ),
    )
    # --- Stage B new flags ---
    parser.add_argument(
        "--enable_trust_score_baseline",
        action="store_true",
        help=(
            "Run Trust Score baselines (diagnostic + switch, alpha=0). "
            "Requires --enable_rgcl_neighbor_correction or --enable_full_vector_geometric_fusion "
            "so that distance matrices and training features are available."
        ),
    )
    parser.add_argument(
        "--trust_score_filter_k",
        type=int,
        default=10,
        help="Number of nearest neighbors for Trust Score density filter (Phase 1, alpha=0 only). Default 10.",
    )
    parser.add_argument(
        "--enable_aar_lightweight",
        action="store_true",
        help=(
            "Run the AAR-Lightweight calibrator (NOT official AAR). "
            "Uses per-sample mean 1-NN distance as atypicality proxy. "
            "Requires per-class distance matrices (--enable_full_vector_geometric_fusion or similar)."
        ),
    )
    parser.add_argument(
        "--enable_glad_pi",
        action="store_true",
        help=(
            "Run GLAD-PI (Geometric Logit Additive Decision-calibrator with Permutation Invariance). "
            "Requires distance matrices from --enable_full_vector_geometric_fusion or similar."
        ),
    )
    parser.add_argument(
        "--enable_glad_pi_zero_geometry",
        action="store_true",
        help=(
            "Run GLAD-PI's zero-geometry twin: identical architecture, hyperparameters, "
            "and split as --enable_glad_pi, with distance_k masked to 0 before the MLP "
            "(capacity-matched, geometry-ablated control). Requires the same distance "
            "matrices as --enable_glad_pi. Enabled by default."
        ),
    )
    parser.add_argument(
        "--enable_mahalanobis_confidence",
        action="store_true",
        help=(
            "Run Mahalanobis confidence (variant A): per-class mean + Ledoit-Wolf-shrunk "
            "shared covariance on penultimate features, scalar min-class-distance score "
            "mapped via isotonic + uniform-spread (scalar-only, argmax-preserving). "
            "Reuses the same penultimate-feature capture as full KCal instead of a "
            "dedicated forward pass. Enabled by default."
        ),
    )
    parser.add_argument(
        "--enable_contrastive_beta_sweep",
        action="store_true",
        help=(
            "Run contrastive-beta sweeps on top of vector and temperature scaling. "
            "Requires --enable_full_vector_geometric_fusion. Enabled by default."
        ),
    )
    parser.add_argument(
        "--contrastive_beta_grid",
        type=str,
        default="0,0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0,100.0",
        help="Comma-separated beta values for the contrastive distance correction sweep.",
    )
    parser.add_argument(
        "--contrastive_nll_tolerance",
        type=float,
        default=DEFAULT_CONTRASTIVE_NLL_TOLERANCE,
        help="Select-set NLL tolerance fraction for contrastive-beta selection (default: 0.02).",
    )
    parser.add_argument(
        "--contrastive_brier_tolerance",
        type=float,
        default=DEFAULT_CONTRASTIVE_BRIER_TOLERANCE,
        help="Select-set Brier tolerance fraction for contrastive-beta selection (default: 0.05).",
    )
    parser.add_argument(
        "--contrastive_ece_abs_tolerance",
        type=float,
        default=DEFAULT_CONTRASTIVE_ECE_ABS_TOLERANCE,
        help="Absolute select-set top-label-ECE tolerance for contrastive-beta selection (default: 0.005).",
    )
    parser.add_argument(
        "--glad_pi_beta_grid",
        type=str,
        default="0,0.01,0.03,0.1,0.3,1,3,10",
        help="Comma-separated beta values for GLAD-PI margin-loss weight grid. Default: 0,0.01,0.03,0.1,0.3,1,3,10",
    )
    parser.add_argument(
        "--glad_pi_nll_tolerance",
        type=float,
        default=0.05,
        help="NLL tolerance fraction for GLAD-PI beta selection. Default 0.05 (5%%).",
    )
    parser.add_argument(
        "--glad_pi_epochs",
        type=int,
        default=100,
        help="Training epochs per beta for GLAD-PI. Default 100.",
    )
    parser.add_argument(
        "--glad_pi_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension of the GLAD-PI per-class MLP. Default 64.",
    )
    parser.add_argument(
        "--glad_pi_lr",
        type=float,
        default=1e-3,
        help="Learning rate for GLAD-PI. Default 1e-3.",
    )
    parser.add_argument(
        "--glad_pi_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for GLAD-PI. Default 1e-4.",
    )
    parser.add_argument(
        "--glad_pi_use_temperature",
        action="store_true",
        help="Enable optional temperature scaling branch in GLAD-PI. Disabled by default.",
    )
    parser.add_argument(
        "--cleanup-intermediates",
        action="store_true",
        help=(
            "Delete intermediates/splits/ and intermediates/features/ after a successful run. "
            "Saves ~50-65 GB per experiment while keeping per_sample, summary, and .pt files."
        ),
    )
    parser.add_argument(
        "--disable_rgcc",
        action="store_true",
        help=(
            "Skip the legacy RGCC (coordinate calibration) method. Enabled by default "
            "for backward compatibility; not part of the frozen Phase 0/1 method set."
        ),
    )
    parser.add_argument(
        "--disable_gc_tulip",
        action="store_true",
        help=(
            "Skip the legacy GC-TULIP method (~22 min/checkpoint). Enabled by default "
            "for backward compatibility; not part of the frozen Phase 0/1 method set."
        ),
    )
    _configure_default_method_flags(parser)
    args = parser.parse_args()
    _apply_disable_all_optional_methods(args)
    _write_fit_once_provenance(args)

    if args.reuse_non_metric_from:
        args.reuse_non_metric_from = os.path.abspath(args.reuse_non_metric_from)
        if args.stab_metric == "l2":
            parser.error("--reuse_non_metric_from requires a non-L2 --stab_metric")
        if os.path.abspath(args.output_dir) == args.reuse_non_metric_from:
            parser.error("--reuse_non_metric_from must differ from --output_dir")

    if args.kcal_projection_dim is not None and args.kcal_projection_dim < 1:
        parser.error("--kcal_projection_dim must be positive")
    if args.kcal_projection_epochs < 1:
        parser.error("--kcal_projection_epochs must be positive")
    if args.kcal_references_per_class < 1:
        parser.error("--kcal_references_per_class must be positive")
    if args.kcal_bandwidth_folds < 2:
        parser.error("--kcal_bandwidth_folds must be at least 2")
    for _contrastive_tolerance_name in (
        "contrastive_nll_tolerance",
        "contrastive_brier_tolerance",
        "contrastive_ece_abs_tolerance",
    ):
        _contrastive_tolerance_value = getattr(args, _contrastive_tolerance_name)
        if (
            not np.isfinite(_contrastive_tolerance_value)
            or _contrastive_tolerance_value < 0.0
        ):
            parser.error(
                f"--{_contrastive_tolerance_name} must be finite and non-negative"
            )
    if args.kcal_factorial_k_per_class < 1:
        parser.error("--kcal_factorial_k_per_class must be positive")
    if args.kcal_factorial_cv_folds < 2:
        parser.error("--kcal_factorial_cv_folds must be at least 2")
    if args.kcal_factorial_reference_batch_size < 1:
        parser.error("--kcal_factorial_reference_batch_size must be positive")
    try:
        _kcal_factorial_strength_grid = _parse_float_grid(
            args.kcal_factorial_strength_grid,
            flag_name="--kcal_factorial_strength_grid",
        )
        _kcal_factorial_alpha_grid = _parse_float_grid(
            args.kcal_factorial_alpha_grid,
            flag_name="--kcal_factorial_alpha_grid",
        )
    except ValueError as exc:
        parser.error(str(exc))
    if np.any(_kcal_factorial_strength_grid <= 0.0):
        parser.error("--kcal_factorial_strength_grid values must be positive")
    if np.any(
        (_kcal_factorial_alpha_grid < 0.0)
        | (_kcal_factorial_alpha_grid > 1.0)
    ):
        parser.error("--kcal_factorial_alpha_grid values must lie in [0, 1]")
    _kcal_factorial_fusions = [
        value.strip()
        for value in args.kcal_factorial_fusions.split(",")
        if value.strip()
    ]
    if not _kcal_factorial_fusions:
        parser.error("--kcal_factorial_fusions must contain at least one fusion")
    invalid_factorial_fusions = sorted(
        set(_kcal_factorial_fusions) - KCAL_FACTORIAL_VALID_FUSIONS
    )
    if invalid_factorial_fusions:
        parser.error(
            "Invalid --kcal_factorial_fusions values: "
            + ",".join(invalid_factorial_fusions)
        )
    if len(_kcal_factorial_fusions) != len(set(_kcal_factorial_fusions)):
        parser.error("--kcal_factorial_fusions contains duplicates")

    if (
        args.fusion_feature_source == "rgcl"
        and not args.enable_full_vector_geometric_fusion
        and not args.enable_knn_blend_baseline
        and not args.enable_kcal_lite_baseline
        and not args.enable_kcal_factorial
    ):
        logger.warning(
            "--fusion_feature_source rgcl has no effect without "
            "--enable_full_vector_geometric_fusion, --enable_knn_blend_baseline, "
            "--enable_kcal_lite_baseline, or --enable_kcal_factorial; ignoring."
        )

    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(_intermediates_dir(args.output_dir), exist_ok=True)
    fp = _args_fingerprint(args)
    progress = _load_progress(args.output_dir, fp)
    _contrastive_gate_signatures = {
        method_name: _contrastive_gate_signature(
            args.contrastive_nll_tolerance,
            args.contrastive_brier_tolerance,
            args.contrastive_ece_abs_tolerance,
            post_temperature=method_name.endswith("_post_temperature"),
        )
        for method_name in CONTRASTIVE_BETA_METHOD_NAMES
    }

    _early_json = os.path.join(args.output_dir, "summary_metrics.json")
    _early_csv = os.path.join(args.output_dir, "summary_metrics.csv")
    _early_npz = os.path.join(args.output_dir, "per_sample", "per_sample_arrays.npz")
    _required_contrastive_outputs_complete = (
        not (
            args.enable_contrastive_beta_sweep
            and args.enable_full_vector_geometric_fusion
        )
        or all(
            _is_method_checkpoint_complete(
                args.output_dir,
                method_name,
                expected_gate_signature=_contrastive_gate_signatures[method_name],
            )
            for method_name in CONTRASTIVE_BETA_METHOD_NAMES
        )
    )
    _completed_run_outputs_exist = (
        _stage_complete(progress, "stage4_late_outputs")
        and os.path.exists(_early_json)
        and os.path.exists(_early_csv)
        and os.path.exists(_early_npz)
    )
    if _completed_run_outputs_exist:
        if _required_contrastive_outputs_complete:
            logger.info("Run already complete. Final outputs exist, skipping all stages.")
            return
        if _refresh_completed_contrastive_methods(args, _contrastive_gate_signatures):
            logger.info("Completed contrastive-only refresh; all other methods were skipped.")
            return

    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset, args.batch_size, seed=args.seed
    )

    if args.corruption_type is not None or args.corruption_severity is not None:
        if args.corruption_type is None or args.corruption_severity is None:
            raise ValueError("--corruption_type and --corruption_severity must be given together")
        # Train/val stay clean (calibration fitting is unaffected); only the
        # test split is replaced with the corrupted cell, matching
        # utils/model_utils.py::get_data_loaders' own corruption-experiment
        # convention ("Train/Val loaders remain CLEAN ... Test loader uses
        # corrupted data") -- get_data_loaders itself cannot be used for this
        # directly because its CIFAR-100-C path calls
        # data/cifar100_c.py::get_cifar100c_loader, which is an unimplemented
        # stub in this repo. load_cifar_c_loader is the real, working
        # implementation already used elsewhere (utils/calibration_utils.py,
        # Experiments/layer_selection.py).
        _corruption_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        _corruption_transform = transforms.Compose(
            [transforms.ToTensor(), _corruption_normalize]
        )
        test_loader = load_cifar_c_loader(
            dataset_name=args.dataset,
            corruption=args.corruption_type,
            severity=args.corruption_severity,
            test_transform=_corruption_transform,
            batch_size=args.batch_size,
            cifar_c_dir=args.cifar_c_dir,
        )
        logger.info(
            "Corruption cell active: %s severity=%d (cifar_c_dir=%s); train/val remain clean",
            args.corruption_type, args.corruption_severity, args.cifar_c_dir,
        )

    split_dir = _stage_dir(args.output_dir, "splits")
    stage1_files = {
        "train_raw": os.path.join(split_dir, "train_raw.npy"),
        "train_labels": os.path.join(split_dir, "train_labels.npy"),
        "val_raw": os.path.join(split_dir, "val_raw.npy"),
        "val_labels": os.path.join(split_dir, "val_labels.npy"),
        "test_raw": os.path.join(split_dir, "test_raw.npy"),
        "test_labels": os.path.join(split_dir, "test_labels.npy"),
    }

    if not _stage_complete(progress, "stage1_splits"):
        _mark_stage_started(args.output_dir, progress, "stage1_splits")
        for split_name, loader, raw_key, label_key in [
            ("train", train_loader, "train_raw", "train_labels"),
            ("validation", val_loader, "val_raw", "val_labels"),
            ("test", test_loader, "test_raw", "test_labels"),
        ]:
            logger.info("Materializing %s split to disk.", split_name)
            raw, labels = get_all_data_as_numpy(loader)
            _save_npy(stage1_files[raw_key], raw)
            _save_npy(stage1_files[label_key], labels)
            del raw, labels
            _cleanup_memory()
        _mark_stage_complete(args.output_dir, progress, "stage1_splits", files=stage1_files)

    train_labels = np.load(stage1_files["train_labels"])
    val_labels = np.load(stage1_files["val_labels"])
    test_labels = np.load(stage1_files["test_labels"])

    early_dir = _stage_dir(args.output_dir, "early")
    late_dir = _stage_dir(args.output_dir, "late")
    stage2_files = {
        "base_probs_test": os.path.join(early_dir, "base_probs_test.npy"),
        "gc_dac_probs": os.path.join(early_dir, "gc_dac_probs.npy"),
        "gc_dac_top_idx_test": os.path.join(early_dir, "gc_dac_top_idx_test.npy"),
        "gc_dac_anchor_ct_test": os.path.join(early_dir, "gc_dac_anchor_ct_test.npy"),
        "logits_val": os.path.join(early_dir, "logits_val.npy"),
        "logits_test": os.path.join(early_dir, "logits_test.npy"),
        "methods_so_far": os.path.join(late_dir, "methods_so_far.json"),
        "method_probs_by_name": os.path.join(late_dir, "method_probs_by_name.npz"),
        "method_can_change_argmax": os.path.join(late_dir, "method_can_change_argmax.json"),
    }

    def _stage2_ready() -> bool:
        if not _stage_complete(progress, "stage2_early"):
            return False
        if not all(os.path.exists(p) for p in stage2_files.values()):
            return False
        try:
            registry = _load_method_registry(args.output_dir)
            required_methods = _stage2_required_methods(args)
            for m in required_methods:
                if not _is_method_checkpoint_complete(args.output_dir, m, len(test_labels), num_classes):
                    return False
                if m not in registry.get("ordered_methods", []):
                    return False
            return True
        except Exception:
            return False

    results_dir = args.results_dir
    if not os.path.exists(results_dir):
        fallback_results_dir = os.path.join("results", "models")
        if os.path.exists(fallback_results_dir):
            logger.warning(
                "Results dir not found: %s. Falling back to %s.",
                results_dir,
                fallback_results_dir,
            )
            results_dir = fallback_results_dir
    model_path = construct_model_path(
        results_dir, args.method, args.dataset, args.model, args.seed
    )
    summary_metadata = _build_summary_metadata(
        args,
        model_path,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
    )
    if progress.get("summary_metadata") != summary_metadata:
        progress["summary_metadata"] = summary_metadata
        _write_progress(args.output_dir, progress)
    _write_incremental_summary(args.output_dir, metadata=summary_metadata)

    if args.reuse_non_metric_from and not _stage2_ready():
        reused_methods = _reuse_non_metric_outputs(
            source_output_dir=args.reuse_non_metric_from,
            target_output_dir=args.output_dir,
            stage2_files=stage2_files,
            progress=progress,
            expected_rows=len(test_labels),
            expected_classes=num_classes,
        )
        logger.info(
            "Reused %d non-metric/L2 method checkpoints from %s; "
            "no L2 calibrators were executed in this run.",
            len(reused_methods),
            args.reuse_non_metric_from,
        )
        if not _stage2_ready():
            raise RuntimeError(
                "The requested L2 reuse directory is incomplete; refusing to "
                "fall back to executing L2 methods in a non-L2 run. Source: "
                f"{args.reuse_non_metric_from}"
            )

    if not _stage2_ready():
        _mark_stage_started(args.output_dir, progress, "stage2_early")
        train_raw = np.load(stage1_files["train_raw"], mmap_mode="r")
        val_raw = np.load(stage1_files["val_raw"], mmap_mode="r")
        test_raw = np.load(stage1_files["test_raw"], mmap_mode="r")
        model = load_trained_model(model_path, args.model, num_classes, device, dataset=args.dataset)
        model_adapter = PyTorchModelAdapter(model, device, args.dataset)
        def _save_or_skip_method(
            method_name: str,
            entry: Dict[str, Any],
            probs: np.ndarray,
            can_change_argmax: bool,
            stage_name: str,
        ) -> None:
            if _is_method_checkpoint_complete(
                args.output_dir, method_name, len(test_labels), num_classes
            ):
                logger.info("Skipping completed method %s", method_name)
                return
            logger.info("Starting method %s", method_name)
            _save_method_checkpoint(
                args.output_dir,
                method_name,
                entry,
                probs,
                can_change_argmax=can_change_argmax,
                stage_name=stage_name,
            )
            logger.info("Saved method checkpoint: %s", method_name)

        if _is_method_checkpoint_complete(args.output_dir, "base_model", len(test_labels), num_classes) and os.path.exists(stage2_files["base_probs_test"]):
            logger.info("Skipping completed method %s", "base_model")
            base_probs_test = np.load(stage2_files["base_probs_test"], mmap_mode="r")
        else:
            logger.info("Starting method %s", "base_model")
            base_probs_test = model_adapter.predict_proba(test_raw, batch_size=args.batch_size)

        if _is_method_checkpoint_complete(args.output_dir, "gc_dac", len(test_labels), num_classes) and all(
            os.path.exists(stage2_files[k]) for k in ["gc_dac_probs", "gc_dac_top_idx_test", "gc_dac_anchor_ct_test"]
        ):
            logger.info("Skipping completed method %s", "gc_dac")
            gc_dac_probs = np.load(stage2_files["gc_dac_probs"], mmap_mode="r")
            gc_dac_test_top_idx = np.load(stage2_files["gc_dac_top_idx_test"], mmap_mode="r")
            gc_dac_test_anchor_ct = np.load(stage2_files["gc_dac_anchor_ct_test"], mmap_mode="r")
        else:
            logger.info("Starting method %s", "gc_dac")
            gc_dac = _run_geo_cal_function_fit_once(
                args, "gc_dac", run_sgc_with_dac_layers,
                dict(
                    model=model,
                    model_name=args.model,
                    dataset_name=args.dataset,
                    model_adapter=model_adapter,
                    train_raw=train_raw,
                    train_labels=train_labels,
                    val_raw=val_raw,
                    val_labels=val_labels,
                    test_raw=test_raw,
                    test_labels=test_labels,
                    device=device,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    precomputed_test_probs=np.asarray(base_probs_test),
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    target_dim=args.target_dimension,
                    pooling_mode="max",
                    return_probs=True,
                ),
            )
            gc_dac_probs = gc_dac["calibrated_probs"]
            gc_dac_test_top_idx = np.argmax(np.asarray(base_probs_test), axis=1)
            gc_dac_test_anchor_ct = gc_dac_probs[np.arange(len(gc_dac_test_top_idx)), gc_dac_test_top_idx]
            gc_dac_entry = _make_method_entry(
                "gc_dac",
                gc_dac_probs,
                test_labels,
                "top_label_calibration",
                False,
                "geometric_isotonic_fit",
                "validation",
                "validation_accuracy_vs_stability",
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=np.asarray(base_probs_test),
                base_probs=np.asarray(base_probs_test),
            )
            _save_method_checkpoint(
                args.output_dir,
                "gc_dac",
                gc_dac_entry,
                gc_dac_probs,
                False,
                stage_name="stage2_early",
            )
            logger.info("Saved method checkpoint: %s", "gc_dac")
            _save_npy_atomic(stage2_files["gc_dac_probs"], gc_dac_probs)
            _save_npy_atomic(
                stage2_files["gc_dac_top_idx_test"],
                gc_dac_test_top_idx.astype(np.int64),
            )
            _save_npy_atomic(
                stage2_files["gc_dac_anchor_ct_test"],
                gc_dac_test_anchor_ct.astype(np.float64),
            )
            del gc_dac, gc_dac_entry
            _cleanup_memory()
            logger.info("[cleanup] released arrays after method: gc_dac")

        # ------------------------------------------------------------------ #
        # Native DAC (Phase 1, §8f): DensityAwareCalibrator's own            #
        # softmax(z / S(x,w)) statistic on its own prescribed layers --      #
        # a fully separate row from gc_dac (which reuses DAC's layers but    #
        # the GC separation statistic). Integration only, no new math:       #
        # reuses the existing extract_dac_features/DensityAwareCalibrator    #
        # infrastructure already used elsewhere in this repo (e.g.          #
        # Experiments/compare_dac_geometric.py).                             #
        # ------------------------------------------------------------------ #
        if _is_method_checkpoint_complete(
            args.output_dir, "native_dac", len(test_labels), num_classes
        ):
            logger.info("Skipping completed method %s", "native_dac")
        else:
            from Calibrators.density_aware_calibration import (
                DensityAwareCalibrator,
                get_dac_target_layers,
                extract_dac_features,
                get_dac_k_value,
            )

            native_dac_layers = get_dac_target_layers(args.model, model)
            native_dac_k = get_dac_k_value(args.dataset)
            native_dac = DensityAwareCalibrator(
                k=native_dac_k, use_gpu=(device.type == "cuda")
            )

            def _fit_native_dac() -> None:
                # Train/val extraction deferred in here so a --corruption_type
                # run (which always loads pickled state instead) never pays
                # for it -- train/val are clean regardless, but re-extracting
                # them on every corruption cell would be pure waste.
                train_feats, _, _ = extract_dac_features(model, train_loader, native_dac_layers, device)
                val_feats, val_logits, val_labels_ = extract_dac_features(
                    model, val_loader, native_dac_layers, device
                )
                native_dac.fit(
                    train_features_list=[f.numpy() for f in train_feats],
                    val_features_list=[f.numpy() for f in val_feats],
                    val_logits=val_logits.numpy(),
                    val_labels=val_labels_.numpy(),
                )

            _fit_or_load_state(args, "native_dac", native_dac, _fit_native_dac)

            native_dac_test_feats, native_dac_test_logits, _ = extract_dac_features(
                model, test_loader, native_dac_layers, device
            )
            native_dac_probs_test = native_dac.calibrate(
                [f.numpy() for f in native_dac_test_feats],
                native_dac_test_logits.numpy(),
            )
            _validate_probability_matrix(
                native_dac_probs_test,
                n_rows=len(test_labels),
                n_classes=num_classes,
                method_name="native_dac",
            )
            native_dac_entry = _make_method_entry(
                "native_dac",
                native_dac_probs_test,
                test_labels,
                "full_vector_density_calibration",
                False,
                "dac_squared_error_weight_fit",
                "validation",
                "dac_paper_objective",
                {
                    "dac_layers": list(native_dac_layers),
                    "dac_k": int(native_dac_k),
                },
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=np.asarray(base_probs_test),
                base_probs=np.asarray(base_probs_test),
            )
            _save_or_skip_method(
                "native_dac",
                native_dac_entry,
                native_dac_probs_test,
                False,
                stage_name="stage2_early",
            )
            del (
                native_dac,
                native_dac_entry,
                native_dac_test_feats,
                native_dac_probs_test,
            )
        _cleanup_memory()
        logger.info("[cleanup] released arrays after method: native_dac")

        base_entry = _make_method_entry(
            "base_model",
            base_probs_test,
            test_labels,
            "top_label_calibration",
            False,
            "none",
            "none",
            "none",
            anchor_ct_for_stratification=gc_dac_test_anchor_ct,
            base_probs_for_anchor_bins=np.asarray(base_probs_test),
        )
        _save_or_skip_method(
            "base_model",
            base_entry,
            base_probs_test,
            can_change_argmax=False,
            stage_name="stage2_early",
        )
        _cleanup_memory()
        logger.info("[cleanup] released arrays after method: base_model")

        need_posthoc_logits = any(
            not _is_method_checkpoint_complete(args.output_dir, m, len(test_labels), num_classes)
            for m in ["temperature_scaling", "vector_scaling", "beta_calibration", "ovr_isotonic", "odir_dirichlet"]
        )
        logits_files_exist = os.path.exists(stage2_files["logits_val"]) and os.path.exists(
            stage2_files["logits_test"]
        )
        need_posthoc_logits = need_posthoc_logits or (not logits_files_exist)
        if need_posthoc_logits:
            logger.info("Starting shared logits extraction for post-hoc methods")
            logits_val = model_adapter.predict_logits(val_raw, batch_size=args.batch_size)
            logits_test = model_adapter.predict_logits(test_raw, batch_size=args.batch_size)
            _save_npy(stage2_files["logits_val"], logits_val)
            _save_npy(stage2_files["logits_test"], logits_test)
        else:
            logits_val = np.load(stage2_files["logits_val"], mmap_mode="r")
            logits_test = np.load(stage2_files["logits_test"], mmap_mode="r")
            logger.info("Loaded post-hoc logits from disk checkpoints")

        stage2_methods_tmp: List[Dict[str, Any]] = []
        if _is_method_checkpoint_complete(
            args.output_dir, "temperature_scaling", len(test_labels), num_classes
        ):
            logger.info("Skipping completed method %s", "temperature_scaling")
        else:
            ts = TemperatureScaling()
            ts_probs_test = _fit_calibrate_append_simple_posthoc(
                methods=stage2_methods_tmp,
                method_name="temperature_scaling",
                method_family="top_label_calibration",
                can_change_argmax=False,
                tuning_procedure="scalar_temperature_fit",
                tuning_split="validation",
                tuning_objective="validation_nll",
                fit_inputs=logits_val,
                fit_labels=val_labels,
                test_inputs=logits_test,
                y_test=test_labels,
                n_classes=num_classes,
                fit_fn=lambda x, y: (
                    _fit_or_load_state(args, "temperature_scaling", ts, lambda: ts.fit(x, y)),
                    {"temperature": _temperature_to_float(ts.temperature)},
                )[1],
                calibrate_fn=ts.calibrate,
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=base_probs_test,
                base_probs=base_probs_test,
            )
            _save_or_skip_method(
                "temperature_scaling",
                stage2_methods_tmp.pop(),
                ts_probs_test,
                False,
                stage_name="stage2_early",
            )
            del ts_probs_test, ts
        _cleanup_memory()
        logger.info("[cleanup] released arrays after method: temperature_scaling")

        if _is_method_checkpoint_complete(
            args.output_dir, "vector_scaling", len(test_labels), num_classes
        ):
            logger.info("Skipping completed method %s", "vector_scaling")
        else:
            vector_scaling = VectorScaling()
            vector_scaling_probs_test = _fit_calibrate_append_simple_posthoc(
                methods=stage2_methods_tmp,
                method_name="vector_scaling",
                method_family="full_vector_post_hoc",
                can_change_argmax=True,
                tuning_procedure="validation_multiclass_fit",
                tuning_split="validation",
                tuning_objective="validation_nll",
                fit_inputs=logits_val,
                fit_labels=val_labels,
                test_inputs=logits_test,
                y_test=test_labels,
                n_classes=num_classes,
                fit_fn=lambda x, y: (
                    _fit_or_load_state(args, "vector_scaling", vector_scaling, lambda: vector_scaling.fit(x, y)),
                    vector_scaling.get_params(),
                )[1],
                calibrate_fn=vector_scaling.calibrate,
                base_probs_for_diagnostics=base_probs_test,
                include_decision_diagnostics=True,
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=base_probs_test,
            )
            _save_or_skip_method(
                "vector_scaling",
                stage2_methods_tmp.pop(),
                vector_scaling_probs_test,
                True,
                stage_name="stage2_early",
            )
            del vector_scaling_probs_test, vector_scaling
        _cleanup_memory()
        logger.info("[cleanup] released arrays after method: vector_scaling")

        if _is_method_checkpoint_complete(
            args.output_dir, "beta_calibration", len(test_labels), num_classes
        ):
            logger.info("Skipping completed method %s", "beta_calibration")
        else:
            beta_calibration = BetaCalibration()
            beta_probs_test = _fit_calibrate_append_simple_posthoc(
                methods=stage2_methods_tmp,
                method_name="beta_calibration",
                method_family="full_vector_post_hoc",
                can_change_argmax=True,
                tuning_procedure="validation_one_vs_rest_fit",
                tuning_split="validation",
                tuning_objective="validation_binary_beta_fit_per_class",
                fit_inputs=logits_val,
                fit_labels=val_labels,
                test_inputs=logits_test,
                y_test=test_labels,
                n_classes=num_classes,
                fit_fn=lambda x, y: (
                    _fit_or_load_state(args, "beta_calibration", beta_calibration, lambda: beta_calibration.fit(x, y)),
                    beta_calibration.get_params(),
                )[1],
                calibrate_fn=beta_calibration.calibrate,
                base_probs_for_diagnostics=base_probs_test,
                include_decision_diagnostics=True,
                extra={"implementation": "one_vs_rest_multiclass"},
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=base_probs_test,
            )
            _save_or_skip_method(
                "beta_calibration",
                stage2_methods_tmp.pop(),
                beta_probs_test,
                True,
                stage_name="stage2_early",
            )
            del beta_probs_test, beta_calibration
        _cleanup_memory()
        logger.info("[cleanup] released arrays after method: beta_calibration")

        if _is_method_checkpoint_complete(
            args.output_dir, "ovr_isotonic", len(test_labels), num_classes
        ):
            logger.info("Skipping completed method %s", "ovr_isotonic")
        else:
            ovr_isotonic = OneVsRestIsotonicCalibration()
            ovr_isotonic_probs_test = _fit_calibrate_append_simple_posthoc(
                methods=stage2_methods_tmp,
                method_name="ovr_isotonic",
                method_family="full_vector_post_hoc",
                can_change_argmax=True,
                tuning_procedure="validation_one_vs_rest_isotonic",
                tuning_split="validation",
                tuning_objective="per_class_binary_isotonic_fit",
                fit_inputs=logits_val,
                fit_labels=val_labels,
                test_inputs=logits_test,
                y_test=test_labels,
                n_classes=num_classes,
                fit_fn=lambda x, y: (
                    _fit_or_load_state(args, "ovr_isotonic", ovr_isotonic, lambda: ovr_isotonic.fit(x, y)),
                    ovr_isotonic.get_params(),
                )[1],
                calibrate_fn=ovr_isotonic.calibrate,
                base_probs_for_diagnostics=base_probs_test,
                include_decision_diagnostics=True,
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=base_probs_test,
            )
            _save_or_skip_method(
                "ovr_isotonic",
                stage2_methods_tmp.pop(),
                ovr_isotonic_probs_test,
                True,
                stage_name="stage2_early",
            )
            del ovr_isotonic_probs_test, ovr_isotonic
        _cleanup_memory()
        logger.info("[cleanup] released arrays after method: ovr_isotonic")

        if _is_method_checkpoint_complete(
            args.output_dir, "odir_dirichlet", len(test_labels), num_classes
        ):
            logger.info("Skipping completed method %s", "odir_dirichlet")
        else:
            odir = ODIRDirichletCalibration()
            odir_probs_test = _fit_calibrate_append_simple_posthoc(
                methods=stage2_methods_tmp,
                method_name="odir_dirichlet",
                method_family="full_vector_post_hoc",
                can_change_argmax=True,
                tuning_procedure="validation_multiclass_fit",
                tuning_split="validation",
                tuning_objective="validation_nll",
                fit_inputs=logits_val,
                fit_labels=val_labels,
                test_inputs=logits_test,
                y_test=test_labels,
                n_classes=num_classes,
                fit_fn=lambda x, y: (
                    _fit_or_load_state(args, "odir_dirichlet", odir, lambda: odir.fit(x, y)),
                    odir.get_params(),
                )[1],
                calibrate_fn=odir.calibrate,
                base_probs_for_diagnostics=base_probs_test,
                include_decision_diagnostics=True,
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=base_probs_test,
            )
            _save_or_skip_method(
                "odir_dirichlet",
                stage2_methods_tmp.pop(),
                odir_probs_test,
                True,
                stage_name="stage2_early",
            )
            del odir_probs_test, odir
        _cleanup_memory()
        logger.info("[cleanup] released arrays after method: odir_dirichlet")

        if _is_method_checkpoint_complete(
            args.output_dir, "top_label_isotonic", len(test_labels), num_classes
        ):
            logger.info("Skipping completed method %s", "top_label_isotonic")
        else:
            top_label_isotonic = TopLabelIsotonicCalibrator()
            top_label_isotonic_probs_test = _fit_calibrate_append_simple_posthoc(
                methods=stage2_methods_tmp,
                method_name="top_label_isotonic",
                method_family="top_label_calibration",
                can_change_argmax=False,
                tuning_procedure="isotonic_pav_fit",
                tuning_split="validation",
                tuning_objective="binary_correctness",
                fit_inputs=logits_val,
                fit_labels=val_labels,
                test_inputs=logits_test,
                y_test=test_labels,
                n_classes=num_classes,
                fit_fn=lambda x, y: (
                    _fit_or_load_state(
                        args, "top_label_isotonic", top_label_isotonic, lambda: top_label_isotonic.fit(x, y)
                    ),
                    top_label_isotonic.get_params(),
                )[1],
                calibrate_fn=top_label_isotonic.calibrate,
                base_probs_for_diagnostics=base_probs_test,
                include_decision_diagnostics=True,
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=base_probs_test,
            )
            _save_or_skip_method(
                "top_label_isotonic",
                stage2_methods_tmp.pop(),
                top_label_isotonic_probs_test,
                False,
                stage_name="stage2_early",
            )
            del top_label_isotonic_probs_test, top_label_isotonic
        _cleanup_memory()
        logger.info("[cleanup] released arrays after method: top_label_isotonic")

        common_kwargs = dict(
            model=model,
            model_name=args.model,
            dataset_name=args.dataset,
            model_adapter=model_adapter,
            train_raw=train_raw,
            train_labels=train_labels,
            val_raw=val_raw,
            val_labels=val_labels,
            test_raw=test_raw,
            test_labels=test_labels,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            precomputed_test_probs=base_probs_test,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )
        if _is_method_checkpoint_complete(args.output_dir, "rgcl", len(test_labels), num_classes):
            logger.info("Skipping completed method %s", "rgcl")
            rgcl_probs = _load_method_probs(args.output_dir, "rgcl", mmap_mode="r")
        else:
            logger.info("[resume] method incomplete, recomputing: rgcl")
            # Published RGCL: historical, unmodified reproduction. Its own
            # per-checkpoint seed (args.seed) draw is never corrected --
            # includes fc for checkpoint seed=1, by design (plan §3/§16);
            # fit-once/evaluate-many still applies on corruption runs (fit
            # is on clean train/val exactly as always, only re-evaluated on
            # whichever test split is active).
            rgcl = _run_geo_cal_function_fit_once(
                args, "rgcl", run_global_random_calibration,
                dict(
                    **common_kwargs,
                    num_layers=args.num_layers,
                    target_dim=args.target_dimension,
                    pooling_mode="max",
                    return_probs=True,
                ),
            )
            rgcl_probs = rgcl["calibrated_probs"]
            rgcl_entry = _make_method_entry(
                "rgcl",
                rgcl_probs,
                test_labels,
                "top_label_calibration",
                False,
                "geometric_isotonic_fit",
                "validation",
                "validation_accuracy_vs_stability",
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=base_probs_test,
                base_probs=base_probs_test,
            )
            _save_method_checkpoint(
                args.output_dir,
                "rgcl",
                rgcl_entry,
                rgcl_probs,
                False,
                stage_name="stage2_early",
            )
            logger.info("Saved method checkpoint: %s", "rgcl")
            del rgcl
        _cleanup_memory()
        logger.info("[cleanup] released arrays after method: rgcl")

        if args.enable_rgcl_tail_hybrids:
            requested_tail_sources = {
                s.strip().lower() for s in args.rgcl_tail_sources.split(",") if s.strip()
            }
            allowed_tail_sources = {"base", "temperature_scaling", "vector_scaling", "dirichlet"}
            invalid_tail_sources = sorted(requested_tail_sources - allowed_tail_sources)
            if invalid_tail_sources:
                raise ValueError(
                    "Invalid --rgcl_tail_sources values: "
                    + ",".join(invalid_tail_sources)
                    + ". Allowed: base,temperature_scaling,vector_scaling,dirichlet"
                )
            rgcl_top_idx, rgcl_anchor_c_t = extract_toplabel_anchor(
                model_probs=base_probs_test, calibrated_probs=rgcl_probs
            )
            tail_source_map = {
                "base": "base_model",
                "temperature_scaling": "temperature_scaling",
                "vector_scaling": "vector_scaling",
                "dirichlet": "odir_dirichlet",
            }
            hybrid_method_map = {
                "base": "rgcl_tail_base",
                "temperature_scaling": "rgcl_tail_temperature_scaling",
                "vector_scaling": "rgcl_tail_vector_scaling",
                "dirichlet": "rgcl_tail_dirichlet",
            }
            for tail_key in ["base", "temperature_scaling", "vector_scaling", "dirichlet"]:
                if tail_key not in requested_tail_sources:
                    continue
                tail_source_method = tail_source_map[tail_key]
                tail_probs = _load_method_probs(args.output_dir, tail_source_method)
                hybrid_method_name = hybrid_method_map[tail_key]
                if _is_method_checkpoint_complete(
                    args.output_dir, hybrid_method_name, len(test_labels), num_classes
                ):
                    logger.info("Skipping completed method %s", hybrid_method_name)
                    del tail_probs
                    _cleanup_memory()
                    continue
                hybrid_probs = _compose_anchored_tail_probs(
                    anchor_top_idx=rgcl_top_idx,
                    anchor_top_conf=rgcl_anchor_c_t,
                    tail_probs=tail_probs,
                )
                _validate_probability_matrix(
                    hybrid_probs,
                    n_rows=len(test_labels),
                    n_classes=num_classes,
                    method_name=hybrid_method_name,
                )
                is_anchor_locked, max_anchor_abs_diff = _validate_anchor_toplock(
                    probs=hybrid_probs,
                    anchor_top_idx=rgcl_top_idx,
                    anchor_top_conf=rgcl_anchor_c_t,
                )
                if not is_anchor_locked:
                    logger.warning(
                        "%s: anchor lock drift detected (max_abs_diff=%.3e)",
                        hybrid_method_name,
                        max_anchor_abs_diff,
                    )
                hybrid_extra: Dict[str, Any] = {
                    "anchor_source": "rgcl",
                    "tail_source": tail_source_method,
                    "mean_anchor_confidence": float(np.mean(rgcl_anchor_c_t)),
                    "max_anchor_abs_diff": max_anchor_abs_diff,
                }
                hybrid_extra.update(_argmax_flip_metadata(base_probs_test, hybrid_probs, test_labels))
                hybrid_extra.update(
                    _nll_by_base_prediction_subset(base_probs_test, hybrid_probs, test_labels)
                )
                hybrid_extra.update(_nll_by_final_prediction_subset(hybrid_probs, test_labels))
                hybrid_entry = _make_method_entry(
                    hybrid_method_name,
                    hybrid_probs,
                    test_labels,
                    "decision_changing_full_vector",
                    True,
                    "anchored_to_rgcl_top_label",
                    "validation",
                    "locked_rgcl_anchor_plus_tail_renorm",
                    hybrid_extra,
                    anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                    base_probs_for_anchor_bins=base_probs_test,
                    base_probs=base_probs_test,
                )
                _save_or_skip_method(
                    hybrid_method_name,
                    hybrid_entry,
                    hybrid_probs,
                    True,
                    stage_name="stage2_early",
                )
                del tail_probs, hybrid_probs
                _cleanup_memory()
                logger.info("[cleanup] released arrays after method: %s", hybrid_method_name)

        if args.disable_rgcc:
            logger.info("Skipping method %s (--disable_rgcc)", "rgcc")
        elif _is_method_checkpoint_complete(args.output_dir, "rgcc", len(test_labels), num_classes):
            logger.info("Skipping completed method %s", "rgcc")
        else:
            logger.info("[resume] method incomplete, recomputing: rgcc")
            rgcc = run_coordinate_calibration(
                **common_kwargs,
                num_coordinates=args.num_coordinates,
                return_probs=True,
            )
            rgcc_entry = _make_method_entry(
                "rgcc",
                rgcc["calibrated_probs"],
                test_labels,
                "top_label_calibration",
                False,
                "geometric_isotonic_fit",
                "validation",
                "validation_accuracy_vs_stability",
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=base_probs_test,
                base_probs=base_probs_test,
            )
            _save_method_checkpoint(
                args.output_dir,
                "rgcc",
                rgcc_entry,
                rgcc["calibrated_probs"],
                False,
                stage_name="stage2_early",
            )
            logger.info("Saved method checkpoint: %s", "rgcc")
            del rgcc
        _cleanup_memory()
        logger.info("[cleanup] released arrays after method: rgcc")

        anchored_model_tail_probs = _anchored_model_tail_probs(base_probs_test, gc_dac_test_anchor_ct)
        anchored_extra: Dict[str, Any] = {
            "anchor_source_method": "gc_dac",
            "non_top_distance_mean": None,
            "non_top_distance_std": None,
            "non_top_distance_min": None,
            "non_top_distance_max": None,
        }
        anchored_extra.update(_nll_by_final_prediction_subset(anchored_model_tail_probs, test_labels))
        anchored_extra.update(
            _nll_by_base_prediction_subset(base_probs_test, anchored_model_tail_probs, test_labels)
        )
        anchored_extra.update(
            _argmax_flip_metadata(base_probs_test, anchored_model_tail_probs, test_labels)
        )
        anchored_entry = _make_method_entry(
            "anchored_model_tail",
            anchored_model_tail_probs,
            test_labels,
            "decision_changing_full_vector",
            True,
            "anchored_to_scalar_top_label",
            "validation",
            "locked_anchor_from_gc_dac",
            anchored_extra,
            anchor_ct_for_stratification=gc_dac_test_anchor_ct,
            base_probs_for_anchor_bins=base_probs_test,
            base_probs=base_probs_test,
        )
        _save_or_skip_method(
            "anchored_model_tail",
            anchored_entry,
            anchored_model_tail_probs,
            True,
            stage_name="stage2_early",
        )
        del anchored_model_tail_probs
        _cleanup_memory()
        logger.info("[cleanup] released arrays after method: anchored_model_tail")

        if args.disable_gc_tulip:
            logger.info("Skipping method %s (--disable_gc_tulip)", "gc_tulip")
        elif _is_method_checkpoint_complete(args.output_dir, "gc_tulip", len(test_labels), num_classes):
            logger.info("Skipping completed method %s", "gc_tulip")
        else:
            logger.info("[resume] method incomplete, recomputing: gc_tulip")
            # Not a Phase 0/1 method (out of scope for this plan's reported
            # tables) but unconditional/core with no --disable flag; still
            # guarded so a corruption run doesn't pay its ~22min/checkpoint
            # extraction cost 12 times over for nothing.
            gc_tulip = _run_geo_cal_function_fit_once(
                args, "gc_tulip", run_sgc_with_tulip_layers,
                dict(
                    **common_kwargs,
                    target_dim=args.target_dimension,
                    pooling_mode="max",
                    return_probs=True,
                ),
            )
            gc_tulip_entry = _make_method_entry(
                "gc_tulip",
                gc_tulip["calibrated_probs"],
                test_labels,
                "top_label_calibration",
                False,
                "geometric_isotonic_fit",
                "validation",
                "validation_accuracy_vs_stability",
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=base_probs_test,
                base_probs=base_probs_test,
            )
            _save_method_checkpoint(
                args.output_dir,
                "gc_tulip",
                gc_tulip_entry,
                gc_tulip["calibrated_probs"],
                False,
                stage_name="stage2_early",
            )
            logger.info("Saved method checkpoint: %s", "gc_tulip")
            del gc_tulip
        _cleanup_memory()
        logger.info("[cleanup] released arrays after method: gc_tulip")

        _save_npy_atomic(stage2_files["base_probs_test"], base_probs_test)
        _save_npy_atomic(stage2_files["gc_dac_probs"], np.asarray(gc_dac_probs))
        _save_npy_atomic(
            stage2_files["gc_dac_top_idx_test"],
            np.asarray(gc_dac_test_top_idx).astype(np.int64),
        )
        _save_npy_atomic(
            stage2_files["gc_dac_anchor_ct_test"],
            np.asarray(gc_dac_test_anchor_ct).astype(np.float64),
        )
        registry = _load_method_registry(args.output_dir)
        methods_so_far = _load_all_method_entries_in_order(args.output_dir)
        _atomic_write_json(stage2_files["methods_so_far"], {"methods": methods_so_far})
        # Compatibility artifact only; per-method checkpoints are the authoritative source.
        np.savez_compressed(stage2_files["method_probs_by_name"])
        _atomic_write_json(stage2_files["method_can_change_argmax"], registry["method_can_change_argmax"])

        del methods_so_far, common_kwargs, base_probs_test, gc_dac_probs, rgcl_probs
        del train_raw, val_raw, test_raw, model, model_adapter, logits_val, logits_test
        _cleanup_memory()
        _mark_stage_complete(args.output_dir, progress, "stage2_early", files=stage2_files)

    if not _stage2_ready():
        raise RuntimeError("Stage 2 is incomplete: required early checkpoint files are missing.")

    base_probs_test = np.load(stage2_files["base_probs_test"])
    gc_dac_probs = np.load(stage2_files["gc_dac_probs"])
    gc_dac_test_top_idx = np.load(stage2_files["gc_dac_top_idx_test"])
    gc_dac_test_anchor_ct = np.load(stage2_files["gc_dac_anchor_ct_test"])
    method_can_change_argmax_map = _load_method_can_change_argmax_map(args.output_dir)

    # ------------------------------------------------------------------ #
    # PTS baseline (parameterized temperature scaling)                     #
    # Needs logits only — runs immediately after stage 2, before features. #
    # ------------------------------------------------------------------ #
    if args.enable_pts_baseline:
        from Calibrators.parameterized_temperature_scaling import ParameterizedTemperatureScaling

        if _is_method_checkpoint_complete(
            args.output_dir, "parameterized_temperature_scaling", len(test_labels), num_classes
        ):
            logger.info("Skipping completed method parameterized_temperature_scaling")
        else:
            logger.info("Running PTS (parameterized temperature scaling) baseline.")
            _pts_logits_val = np.load(stage2_files["logits_val"], mmap_mode="r")
            _pts_logits_test = np.load(stage2_files["logits_test"], mmap_mode="r")

            # Inner split: select_fraction of val → selection set (unused for PTS hyperparam
            # selection since PTS uses early stopping only, but split is computed for GLAD-PI).
            _pts_fit_idx, _pts_select_idx = make_inner_validation_split(
                val_labels, select_fraction=args.inner_val_fraction, seed=args.inner_val_seed
            )
            # PTS trains on the fit split only.
            _pts_logits_fit = np.asarray(_pts_logits_val)[_pts_fit_idx]
            _pts_labels_fit = val_labels[_pts_fit_idx]

            _pts = ParameterizedTemperatureScaling(
                hidden_dim=32, lr=1e-3, epochs=50, patience=5
            )
            _pts_fit_info = _fit_or_load_state(
                args, "parameterized_temperature_scaling", _pts,
                lambda: _pts.fit(_pts_logits_fit, _pts_labels_fit),
            ) or {}
            _pts_probs_test = _pts.calibrate(np.asarray(_pts_logits_test))

            _pts_entry = _make_method_entry(
                "parameterized_temperature_scaling",
                _pts_probs_test,
                test_labels,
                method_family="top_label_calibration",
                can_change_argmax=False,
                tuning_procedure="mlp_temperature_nll_fit",
                tuning_split="inner_val_fit",
                tuning_objective="validation_nll",
                base_probs=base_probs_test,
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=base_probs_test,
                extra={
                    "best_val_nll": _pts_fit_info.get("best_val_nll"),
                    "epochs_trained": _pts_fit_info.get("epochs_trained"),
                    "pts_params": _pts.get_params(),
                    "inner_val_fraction": args.inner_val_fraction,
                    "inner_val_seed": args.inner_val_seed,
                    "inner_val_fit_n": int(len(_pts_fit_idx)),
                    "inner_val_select_n": int(len(_pts_select_idx)),
                },
            )
            _save_method_checkpoint(
                args.output_dir,
                "parameterized_temperature_scaling",
                _pts_entry,
                _pts_probs_test,
                can_change_argmax=False,
                stage_name="stage2_pts",
            )
            method_can_change_argmax_map["parameterized_temperature_scaling"] = False
            logger.info("PTS baseline complete. best_val_nll=%.4f", _pts_fit_info.get("best_val_nll", float("nan")))
            del _pts, _pts_probs_test, _pts_logits_val, _pts_logits_test
            del _pts_fit_idx, _pts_select_idx, _pts_logits_fit, _pts_labels_fit
            _cleanup_memory()
        _write_incremental_summary(args.output_dir)

    feature_dir = _stage_dir(args.output_dir, "features")
    fusion_dir = _stage_dir(args.output_dir, "fusion")
    _kcal_features_required = bool(
        args.enable_kcal_baseline or args.enable_kcal_factorial
    )
    # Mahalanobis confidence (§8d) reuses the same penultimate-feature
    # capture KCal already performs, rather than a dedicated fresh forward
    # pass -- but must not force full KCal itself to run when only
    # Mahalanobis is requested, so this is a separate, broader gate used
    # only for the stage3 extraction below (KCal's own execution at its
    # call site further down stays keyed on _kcal_features_required alone).
    _penultimate_features_required = bool(
        _kcal_features_required or args.enable_mahalanobis_confidence
    )
    stage3_files = {
        "train_features": os.path.join(feature_dir, "train_features.npy"),
        "train_y": os.path.join(feature_dir, "train_y.npy"),
        "val_raw_aligned": os.path.join(feature_dir, "val_raw_aligned.npy"),
        "val_features": os.path.join(feature_dir, "val_features.npy"),
        "val_logits": os.path.join(feature_dir, "val_logits.npy"),
        "val_y": os.path.join(feature_dir, "val_labels.npy"),
        "test_features": os.path.join(feature_dir, "test_features.npy"),
        "test_logits": os.path.join(feature_dir, "test_logits.npy"),
        "test_y": os.path.join(feature_dir, "test_labels.npy"),
        "feature_metadata": os.path.join(feature_dir, "feature_metadata.json"),
        "val_base_probs": os.path.join(fusion_dir, "val_base_probs.npy"),
        "test_base_probs": os.path.join(fusion_dir, "test_base_probs.npy"),
        "fused_val_probs": os.path.join(fusion_dir, "fused_val_probs.npy"),
        "fused_test_probs": os.path.join(fusion_dir, "fused_test_probs.npy"),
        "val_distance_matrix": os.path.join(fusion_dir, "val_distance_matrix.npy"),
        "test_distance_matrix": os.path.join(fusion_dir, "test_distance_matrix.npy"),
        "fusion_metadata": os.path.join(fusion_dir, "fusion_metadata.json"),
    }
    if _penultimate_features_required:
        stage3_files.update(
            {
                "kcal_train_penultimate": os.path.join(
                    feature_dir, "kcal_train_penultimate.npy"
                ),
                "kcal_val_penultimate": os.path.join(
                    feature_dir, "kcal_val_penultimate.npy"
                ),
                "kcal_test_penultimate": os.path.join(
                    feature_dir, "kcal_test_penultimate.npy"
                ),
            }
        )
    if not (_stage_complete(progress, "stage3_features_fusion") and all(os.path.exists(p) for p in stage3_files.values())):
        _mark_stage_started(args.output_dir, progress, "stage3_features_fusion")
        model = load_trained_model(model_path, args.model, num_classes, device, dataset=args.dataset)
        model_adapter = PyTorchModelAdapter(model, device, args.dataset)
        feature_extractor = FeatureExtractor(model, args.model)
        penultimate_capture = (
            _PenultimateFeatureCapture(model) if _penultimate_features_required else None
        )
        single_layer_extraction_start = time.perf_counter()
        single_layer_name = feature_extractor.selected_layer_name
        try:
            (
                _,
                train_features_path,
                _,
                train_y_path,
                train_penultimate_path,
            ) = _extract_split_features_to_disk(
                feature_extractor=feature_extractor,
                data_loader=train_loader,
                device=device,
                out_prefix=os.path.join(feature_dir, "train"),
                include_raw=False,
                penultimate_capture=penultimate_capture,
            )
            (
                val_raw_path,
                val_features_path,
                val_logits_path,
                val_y_path,
                val_penultimate_path,
            ) = _extract_split_features_to_disk(
                feature_extractor=feature_extractor,
                data_loader=val_loader,
                device=device,
                out_prefix=os.path.join(feature_dir, "val"),
                include_raw=True,
                penultimate_capture=penultimate_capture,
            )
            (
                _,
                test_features_path,
                test_logits_path,
                test_y_path,
                test_penultimate_path,
            ) = _extract_split_features_to_disk(
                feature_extractor=feature_extractor,
                data_loader=test_loader,
                device=device,
                out_prefix=os.path.join(feature_dir, "test"),
                include_raw=False,
                penultimate_capture=penultimate_capture,
            )
        finally:
            feature_extractor.cleanup()
            if penultimate_capture is not None:
                penultimate_capture.cleanup()
        single_layer_extraction_time = time.perf_counter() - single_layer_extraction_start

        os.replace(train_features_path, stage3_files["train_features"])
        os.replace(train_y_path, stage3_files["train_y"])
        if val_raw_path is None:
            raise RuntimeError("Validation aligned raw array was not written.")
        os.replace(val_raw_path, stage3_files["val_raw_aligned"])
        os.replace(val_features_path, stage3_files["val_features"])
        os.replace(val_logits_path, stage3_files["val_logits"])
        os.replace(val_y_path, stage3_files["val_y"])
        os.replace(test_features_path, stage3_files["test_features"])
        os.replace(test_logits_path, stage3_files["test_logits"])
        os.replace(test_y_path, stage3_files["test_y"])
        if _penultimate_features_required:
            if any(
                path is None
                for path in [
                    train_penultimate_path,
                    val_penultimate_path,
                    test_penultimate_path,
                ]
            ):
                raise RuntimeError("Full KCal penultimate feature extraction was incomplete")
            assert train_penultimate_path is not None
            assert val_penultimate_path is not None
            assert test_penultimate_path is not None
            os.replace(
                train_penultimate_path, stage3_files["kcal_train_penultimate"]
            )
            os.replace(val_penultimate_path, stage3_files["kcal_val_penultimate"])
            os.replace(test_penultimate_path, stage3_files["kcal_test_penultimate"])

        train_features = np.load(stage3_files["train_features"])
        train_y = np.load(stage3_files["train_y"])
        val_features = np.load(stage3_files["val_features"])
        val_logits2 = np.load(stage3_files["val_logits"])
        val_y = np.load(stage3_files["val_y"])
        test_features = np.load(stage3_files["test_features"])
        test_logits2 = np.load(stage3_files["test_logits"])
        test_y = np.load(stage3_files["test_y"])
        feature_metadata: Dict[str, Any] = {
            "feature_source": "single_layer",
            "selected_layers": [single_layer_name],
            "actual_num_layers": 1,
            "target_dimension": int(train_features.shape[1]),
            "pooling_mode": "global_average_pooling",
            "extraction_time_s": float(single_layer_extraction_time),
        }
        if _penultimate_features_required and penultimate_capture is not None:
            penultimate_shape = np.load(
                stage3_files["kcal_train_penultimate"], mmap_mode="r"
            ).shape
            feature_metadata["kcal_penultimate"] = {
                "layer": penultimate_capture.layer_name,
                "dimension": int(penultimate_shape[1]),
                "capture": "final_linear_classifier_input",
            }
        _atomic_write_json(stage3_files["feature_metadata"], feature_metadata)

        val_base_probs = F.softmax(torch.tensor(val_logits2), dim=1).numpy()
        test_base_probs = F.softmax(torch.tensor(test_logits2), dim=1).numpy()
        fusion_out = _run_full_vector_fusion(
            model_adapter=model_adapter,
            train_features=train_features,
            train_labels=train_y,
            val_features=val_features,
            val_base_probs=val_base_probs,
            val_labels=val_y,
            test_features=test_features,
            test_base_probs=test_base_probs,
            test_labels=test_y,
            seed=args.seed,
            stab_metric=args.stab_metric,
            whitening_components=args.whitening_components,
            whitening_eps=args.whitening_eps,
        )
        val_distance_matrix = fusion_out["stab_space"].calc_per_class_1nn_distances(val_features)
        _save_npy(stage3_files["val_base_probs"], val_base_probs)
        _save_npy(stage3_files["test_base_probs"], test_base_probs)
        _save_npy(stage3_files["fused_val_probs"], fusion_out["fused_val_probs"])
        _save_npy(stage3_files["fused_test_probs"], fusion_out["fused_test_probs"])
        _save_npy(stage3_files["val_distance_matrix"], val_distance_matrix)
        _save_npy(stage3_files["test_distance_matrix"], fusion_out["distance_matrix"])
        _atomic_write_json(
            stage3_files["fusion_metadata"],
            {
                "selected_beta": fusion_out["selected_beta"],
                "beta_selection": fusion_out["beta_selection"],
                "stability_space": fusion_out["stab_space"].get_params(),
                "validation_gates": fusion_out["validation_gates"],
                "argmax_change_rate": fusion_out["argmax_change_rate"],
                "changed_to_correct": fusion_out["changed_to_correct"],
                "changed_to_wrong": fusion_out["changed_to_wrong"],
                "net_flips": fusion_out["net_flips"],
            },
        )
        del model, model_adapter, train_features, val_features, test_features, val_logits2, test_logits2, fusion_out
        _cleanup_memory()
        _mark_stage_complete(args.output_dir, progress, "stage3_features_fusion", files=stage3_files)

    val_raw_aligned = np.load(stage3_files["val_raw_aligned"])
    val_base_probs = np.load(stage3_files["val_base_probs"])
    test_base_probs = np.load(stage3_files["test_base_probs"])
    fused_val_probs = np.load(stage3_files["fused_val_probs"])
    fused_test_probs = np.load(stage3_files["fused_test_probs"])
    val_distance_matrix = np.load(stage3_files["val_distance_matrix"])
    test_distance_matrix = np.load(stage3_files["test_distance_matrix"])
    fusion_meta = _load_json(stage3_files["fusion_metadata"])
    val_y = np.load(stage3_files["val_y"])
    test_y = np.load(stage3_files["test_y"])

    final_json = os.path.join(args.output_dir, "summary_metrics.json")
    final_csv = os.path.join(args.output_dir, "summary_metrics.csv")
    final_npz = os.path.join(args.output_dir, "per_sample", "per_sample_arrays.npz")
    if (
        _stage_complete(progress, "stage4_late_outputs")
        and os.path.exists(final_json)
        and os.path.exists(final_csv)
        and os.path.exists(final_npz)
        and _required_contrastive_outputs_complete
    ):
        logger.info("All stages already complete. Final outputs exist, skipping computation.")
        return

    _mark_stage_started(args.output_dir, progress, "stage4_late_outputs")
    late_anchor_paths = _late_anchor_intermediate_paths(args.output_dir)
    need_rankgeom_method = not _is_method_checkpoint_complete(
        args.output_dir, "anchored_rankgeom_tail_mixture", len(test_labels), num_classes
    )

    # Train loaders can be shuffled, so exact train label order is not required here.
    if len(val_y) != len(val_labels) or len(test_y) != len(test_labels):
        raise ValueError("Feature extraction split sizes do not match shared split sizes")
    if not np.array_equal(val_y, val_labels):
        logger.warning(
            "Validation label order differs between extraction and shared arrays; "
            "using extraction-aligned labels for fusion branch."
        )
    if not np.array_equal(test_y, test_labels):
        logger.warning(
            "Test label order differs between extraction and shared arrays; "
            "using extraction-aligned labels for fusion branch."
        )

    base_top_idx = gc_dac_test_top_idx
    anchor_c_t = gc_dac_test_anchor_ct
    anchored_model_tail_probs = _load_method_probs(args.output_dir, "anchored_model_tail")

    # Stage 3 already persisted fusion outputs and distance matrices.

    # Alignment policy (Option A): ensure the anchor-path arrays share one sample order
    # by construction. We use the extraction-aligned raw data (`val_raw_aligned`) so it
    # pairs row-for-row with `val_y`, `val_base_probs`, and `val_distance_matrix`.
    # This avoids the DataLoader two-pass ordering drift that the pre-existing warning
    # already flagged; the length assertion below guards against extraction size drift.
    if len(val_raw_aligned) != len(val_y):
        raise RuntimeError(
            "Validation ordering mismatch: val_raw_aligned and val_y lengths differ."
        )
    if val_base_probs.shape[0] != len(val_y) or val_distance_matrix.shape[0] != len(val_y):
        raise RuntimeError(
            "Validation ordering mismatch: val_base_probs/val_distance_matrix rows must match val_y."
        )

    validation_anchor_mode = (
        "reused_l2_gc_dac_crossfit"
        if args.reuse_non_metric_from
        else "gc_dac_crossfit"
    )
    model: torch.nn.Module | None = None
    model_adapter: PyTorchModelAdapter | None = None
    train_raw: np.ndarray | None = None
    val_anchor_c_t: np.ndarray | None = None
    val_anchor_diag: Dict[str, Any] | None = None
    rankgeom_selection: Dict[str, Any] | None = None
    if os.path.exists(late_anchor_paths["val_anchor_c_t"]) and os.path.exists(late_anchor_paths["val_anchor_diag"]):
        val_anchor_c_t = np.load(late_anchor_paths["val_anchor_c_t"], mmap_mode="r")
        val_anchor_diag = _load_json(late_anchor_paths["val_anchor_diag"])
        logger.info("Loaded late-stage anchor intermediates from disk")
    if os.path.exists(late_anchor_paths["rankgeom_selection"]):
        rankgeom_selection = _load_json(late_anchor_paths["rankgeom_selection"])

    if (
        args.reuse_non_metric_from
        and need_rankgeom_method
        and (val_anchor_c_t is None or val_anchor_diag is None)
    ):
        raise RuntimeError(
            "Focused non-L2 run cannot reuse the validation anchor cache from "
            f"{args.reuse_non_metric_from}; refusing to execute an L2 fallback."
        )
    if need_rankgeom_method and (val_anchor_c_t is None or val_anchor_diag is None):
        logger.info("Starting late-stage anchor intermediate computation")
        model = load_trained_model(model_path, args.model, num_classes, device, dataset=args.dataset)
        model_adapter = PyTorchModelAdapter(model, device, args.dataset)
        train_raw = np.load(stage1_files["train_raw"], mmap_mode="r")
        val_anchor_c_t, val_anchor_diag = _crossfit_gc_dac_validation_anchor(
            model=model,
            model_name=args.model,
            dataset_name=args.dataset,
            model_adapter=model_adapter,
            train_raw=train_raw,
            train_labels=train_labels,
            val_raw=val_raw_aligned,
            val_y=val_y,
            val_base_probs=val_base_probs,
            device=device,
            batch_size=args.batch_size,
            target_dim=args.target_dimension,
            pooling_mode="max",
            seed=args.seed,
            n_splits=5,
        )
        _save_npy(late_anchor_paths["val_anchor_c_t"], np.asarray(val_anchor_c_t).astype(np.float64))
        _atomic_write_json(late_anchor_paths["val_anchor_diag"], val_anchor_diag)
    if (
        need_rankgeom_method
        and not args.reuse_non_metric_from
        and (model is None or model_adapter is None or train_raw is None)
    ):
        model = load_trained_model(model_path, args.model, num_classes, device, dataset=args.dataset)
        model_adapter = PyTorchModelAdapter(model, device, args.dataset)
        train_raw = np.load(stage1_files["train_raw"], mmap_mode="r")

    if need_rankgeom_method:
        val_pred = np.argmax(val_base_probs, axis=1)
        val_is_correct = val_pred == val_y
        validation_anchor_stats_all = _summarize_anchor_stats(val_anchor_c_t)
        validation_anchor_stats_correct = _summarize_anchor_stats(val_anchor_c_t[val_is_correct])
        validation_anchor_stats_incorrect = _summarize_anchor_stats(val_anchor_c_t[~val_is_correct])

        if (
            os.environ.get("DEBUG_LEAKAGE_CHECK", "0") == "1"
            and not args.reuse_non_metric_from
        ):
            # Debug-only comparison to show the expected leakage shift.
            # Uses extraction-aligned raw data so probs/features/labels stay row-aligned.
            debug_leaky = run_sgc_with_dac_layers(
                model=model,
                model_name=args.model,
                dataset_name=args.dataset,
                model_adapter=model_adapter,
                train_raw=train_raw,
                train_labels=train_labels,
                val_raw=val_raw_aligned,
                val_labels=val_y,
                test_raw=val_raw_aligned,
                test_labels=val_y,
                device=device,
                target_dim=args.target_dimension,
                batch_size=args.batch_size,
                seed=args.seed,
                pooling_mode="max",
                precomputed_test_probs=val_base_probs,
                train_loader=None,
                val_loader=None,
                test_loader=None,
                return_probs=True,
            )
            debug_leaky_probs = debug_leaky["calibrated_probs"]
            debug_top_idx = np.argmax(val_base_probs, axis=1)
            debug_leaky_anchor = debug_leaky_probs[np.arange(len(debug_top_idx)), debug_top_idx]
            logger.info(
                "DEBUG_LEAKAGE_CHECK validation anchor stats | leaky_all=%s | crossfit_all=%s",
                _summarize_anchor_stats(debug_leaky_anchor),
                validation_anchor_stats_all,
            )
            logger.info(
                "DEBUG_LEAKAGE_CHECK validation anchor stats | leaky_correct=%s | crossfit_correct=%s",
                _summarize_anchor_stats(debug_leaky_anchor[val_is_correct]),
                validation_anchor_stats_correct,
            )
            logger.info(
                "DEBUG_LEAKAGE_CHECK validation anchor stats | leaky_incorrect=%s | crossfit_incorrect=%s",
                _summarize_anchor_stats(debug_leaky_anchor[~val_is_correct]),
                validation_anchor_stats_incorrect,
            )

        lambda_grid = [0.0, 0.5, 1.0]
        alpha_grid = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
        nll_by_lambda_alpha: Dict[str, float] = {}
        best_pair: tuple[float, float] | None = None
        best_val_nll = float("inf")
        lambda0_cached: np.ndarray | None = None
        for mix_lambda in lambda_grid:
            if mix_lambda == 0.0:
                if lambda0_cached is None:
                    lambda0_cached = _anchored_rankgeom_tail_mixture_probs(
                        model_probs=val_base_probs,
                        anchor_top_conf=val_anchor_c_t,
                        distance_matrix=val_distance_matrix,
                        mix_lambda=0.0,
                        alpha=0.0,
                    )
                nll0 = _multiclass_nll(lambda0_cached, val_y)
                for alpha in alpha_grid:
                    key = f"lambda={mix_lambda:.1f}|alpha={alpha:.1f}"
                    nll_by_lambda_alpha[key] = nll0
                if nll0 < best_val_nll:
                    best_val_nll = nll0
                    best_pair = (0.0, 0.0)
                continue

            for alpha in alpha_grid:
                val_probs_mixed = _anchored_rankgeom_tail_mixture_probs(
                    model_probs=val_base_probs,
                    anchor_top_conf=val_anchor_c_t,
                    distance_matrix=val_distance_matrix,
                    mix_lambda=mix_lambda,
                    alpha=alpha,
                )
                nll = _multiclass_nll(val_probs_mixed, val_y)
                key = f"lambda={mix_lambda:.1f}|alpha={alpha:.1f}"
                nll_by_lambda_alpha[key] = nll
                if nll < best_val_nll:
                    best_val_nll = nll
                    best_pair = (mix_lambda, alpha)

        if best_pair is None:
            raise RuntimeError("Failed to select lambda/alpha for anchored_rankgeom_tail_mixture")
        selected_lambda, selected_alpha = best_pair
        selected_key = f"lambda={selected_lambda:.1f}|alpha={selected_alpha:.1f}"
        val_nll_at_selected_crossfit_anchor = float(nll_by_lambda_alpha[selected_key])

    val_nll_at_selected_full_anchor: float | None = None
    if not need_rankgeom_method:
        # Completed-method resume: diagnostics were already saved with the
        # checkpoint; model/train_raw are intentionally not loaded.
        pass
    elif args.reuse_non_metric_from:
        logger.info(
            "Skipping full-fit L2 anchor diagnostic in focused %s run",
            args.stab_metric,
        )
    else:
        full_anchor_val_gc_dac = run_sgc_with_dac_layers(
            model=model,
            model_name=args.model,
            dataset_name=args.dataset,
            model_adapter=model_adapter,
            train_raw=train_raw,
            train_labels=train_labels,
            val_raw=val_raw_aligned,
            val_labels=val_y,
            test_raw=val_raw_aligned,
            test_labels=val_y,
            device=device,
            target_dim=args.target_dimension,
            batch_size=args.batch_size,
            seed=args.seed,
            pooling_mode="max",
            precomputed_test_probs=val_base_probs,
            train_loader=None,
            val_loader=None,
            test_loader=None,
            return_probs=True,
        )
        full_anchor_val_probs = full_anchor_val_gc_dac["calibrated_probs"]
        full_anchor_val_top_idx = np.argmax(val_base_probs, axis=1)
        val_full_anchor_c_t = full_anchor_val_probs[
            np.arange(len(full_anchor_val_top_idx)), full_anchor_val_top_idx
        ]
        val_probs_selected_full_anchor = _anchored_rankgeom_tail_mixture_probs(
            model_probs=val_base_probs,
            anchor_top_conf=val_full_anchor_c_t,
            distance_matrix=val_distance_matrix,
            mix_lambda=selected_lambda,
            alpha=selected_alpha,
        )
        val_nll_at_selected_full_anchor = _multiclass_nll(
            val_probs_selected_full_anchor, val_y
        )

    test_anchor_top_idx = np.argmax(test_base_probs, axis=1)
    test_anchor_c_t = gc_dac_probs[np.arange(len(test_anchor_top_idx)), test_anchor_top_idx]
    if need_rankgeom_method:
        anchored_rankgeom_mixture_probs = _anchored_rankgeom_tail_mixture_probs(
            model_probs=test_base_probs,
            anchor_top_conf=test_anchor_c_t,
            distance_matrix=test_distance_matrix,
            mix_lambda=selected_lambda,
            alpha=selected_alpha,
        )
        test_nll_at_selected = _multiclass_nll(anchored_rankgeom_mixture_probs, test_y)
    if os.environ.get("DEBUG_ANCHOR_PROBS_DIFF", "0") == "1":
        n_samples = len(base_top_idx)
        if n_samples == 0:
            logger.info("DEBUG_ANCHOR_PROBS_DIFF | empty test set; skipping diagnostics")
        else:
            logger.info("DEBUG_ANCHOR_PROBS_DIFF | ===== begin test-set diagnostics =====")

            top_idx = base_top_idx
            anchor_top_conf = anchor_c_t
            c_t = gc_dac_probs[np.arange(n_samples), top_idx]
            p_t = base_probs_test[np.arange(n_samples), top_idx]
            max_abs_diff_c_t = float(np.max(np.abs(c_t - anchor_top_conf)))
            logger.info(
                "DEBUG_ANCHOR_PROBS_DIFF | max_abs_diff_c_t=%.16e (expected 0 if anchor wiring is correct)",
                max_abs_diff_c_t,
            )

            confidences_gc_dac = np.max(gc_dac_probs, axis=1)
            confidences_anchored = np.max(anchored_model_tail_probs, axis=1)
            confidence_diff = confidences_gc_dac - confidences_anchored
            confidence_diff_abs = np.abs(confidence_diff)
            logger.info(
                "DEBUG_ANCHOR_PROBS_DIFF | confidence diff stats (gc_dac - anchored): max_abs=%.16e mean=%.16e std=%.16e count_abs_gt_1e-10=%d",
                float(np.max(confidence_diff_abs)),
                float(np.mean(confidence_diff)),
                float(np.std(confidence_diff)),
                int(np.sum(confidence_diff_abs > 1e-10)),
            )

            argmax_gc_dac = np.argmax(gc_dac_probs, axis=1)
            argmax_anchored = np.argmax(anchored_model_tail_probs, axis=1)
            argmax_mismatch_count = int(np.sum(argmax_gc_dac != argmax_anchored))
            logger.info(
                "DEBUG_ANCHOR_PROBS_DIFF | argmax mismatches gc_dac vs anchored=%d/%d",
                argmax_mismatch_count,
                n_samples,
            )

            top_k = min(5, n_samples)
            worst_idx = np.argsort(confidence_diff_abs)[-top_k:][::-1]

            def _top3_entries(vec: np.ndarray) -> str:
                idx3 = np.argsort(-vec)[:3]
                return "[" + ", ".join(f"{int(j)}:{float(vec[j]):.6f}" for j in idx3) + "]"

            def _topk_entries(vec: np.ndarray, k: int) -> str:
                idxk = np.argsort(-vec)[:k]
                return "[" + ", ".join(f"{int(j)}:{float(vec[j]):.6f}" for j in idxk) + "]"

            logger.info(
                "DEBUG_ANCHOR_PROBS_DIFF | worst-%d confidence-diff samples | idx | gc_max | an_max | gc_arg | an_arg | c_t | p_t | top_idx | gc_top3 | an_top3 | y_true",
                top_k,
            )
            for i in worst_idx:
                logger.info(
                    "DEBUG_ANCHOR_PROBS_DIFF | sample | %d | %.9f | %.9f | %d | %d | %.9f | %.9f | %d | %s | %s | %d",
                    int(i),
                    float(confidences_gc_dac[i]),
                    float(confidences_anchored[i]),
                    int(argmax_gc_dac[i]),
                    int(argmax_anchored[i]),
                    float(c_t[i]),
                    float(p_t[i]),
                    int(top_idx[i]),
                    _top3_entries(gc_dac_probs[i]),
                    _top3_entries(anchored_model_tail_probs[i]),
                    int(test_labels[i]),
                )

            metrics_gc_dac = evaluate_all(gc_dac_probs, test_labels)
            metrics_anchored = evaluate_all(anchored_model_tail_probs, test_labels)
            logger.info(
                "DEBUG_ANCHOR_PROBS_DIFF | evaluate_all gc_dac=%s | anchored_model_tail=%s",
                metrics_gc_dac,
                metrics_anchored,
            )

            def _manual_adaptive_ece_equal_mass_15(probs: np.ndarray, labels: np.ndarray) -> float:
                conf = np.max(probs, axis=1).astype(np.float64)
                pred = np.argmax(probs, axis=1)
                correct = (pred == labels).astype(np.float64)
                order = np.argsort(conf)
                ece = 0.0
                for bin_idx in np.array_split(order, 15):
                    if bin_idx.size == 0:
                        continue
                    bin_acc = float(np.mean(correct[bin_idx]))
                    bin_conf = float(np.mean(conf[bin_idx]))
                    ece += abs(bin_acc - bin_conf) * (float(bin_idx.size) / float(len(labels)))
                return float(ece)

            manual_adaptive_ece_gc_dac = _manual_adaptive_ece_equal_mass_15(
                gc_dac_probs, test_labels
            )
            manual_adaptive_ece_anchored = _manual_adaptive_ece_equal_mass_15(
                anchored_model_tail_probs, test_labels
            )
            logger.info(
                "DEBUG_ANCHOR_PROBS_DIFF | manual adaptive-ECE (15 equal-mass bins): gc_dac=%.16e anchored_model_tail=%.16e",
                manual_adaptive_ece_gc_dac,
                manual_adaptive_ece_anchored,
            )

            logger.info(
                "DEBUG_ANCHOR_PROBS_DIFF | ANCHORED_MODEL_TAIL vs ANCHORED_RANKGEOM_TAIL_MIXTURE@lambda=0,alpha=0"
            )
            mixture_at_zero_probs = _anchored_rankgeom_tail_mixture_probs(
                model_probs=test_base_probs,
                anchor_top_conf=test_anchor_c_t,
                distance_matrix=test_distance_matrix,
                mix_lambda=0.0,
                alpha=0.0,
            )
            row_max_abs_diff = np.max(
                np.abs(anchored_model_tail_probs - mixture_at_zero_probs), axis=1
            )
            logger.info(
                "DEBUG_ANCHOR_PROBS_DIFF | row-wise max-abs-diff anchored_model_tail vs mixture@0,0: max=%.16e mean=%.16e std=%.16e count_gt_1e-10=%d count_gt_1e-6=%d",
                float(np.max(row_max_abs_diff)),
                float(np.mean(row_max_abs_diff)),
                float(np.std(row_max_abs_diff)),
                int(np.sum(row_max_abs_diff > 1e-10)),
                int(np.sum(row_max_abs_diff > 1e-6)),
            )

            matrix_abs_diff = np.abs(anchored_model_tail_probs - mixture_at_zero_probs)
            logger.info(
                "DEBUG_ANCHOR_PROBS_DIFF | matrix abs-diff anchored_model_tail vs mixture@0,0: max=%.16e mean=%.16e count_gt_1e-10=%d",
                float(np.max(matrix_abs_diff)),
                float(np.mean(matrix_abs_diff)),
                int(np.sum(matrix_abs_diff > 1e-10)),
            )

            argmax_mismatch_zero = int(
                np.sum(
                    np.argmax(anchored_model_tail_probs, axis=1)
                    != np.argmax(mixture_at_zero_probs, axis=1)
                )
            )
            logger.info(
                "DEBUG_ANCHOR_PROBS_DIFF | argmax mismatches anchored_model_tail vs mixture@0,0=%d/%d",
                argmax_mismatch_zero,
                n_samples,
            )

            metrics_anchored_model_tail = evaluate_all(anchored_model_tail_probs, test_y)
            metrics_mixture_at_zero = evaluate_all(mixture_at_zero_probs, test_y)
            logger.info(
                "DEBUG_ANCHOR_PROBS_DIFF | evaluate_all anchored_model_tail=%s | anchored_rankgeom_tail_mixture@0,0=%s",
                metrics_anchored_model_tail,
                metrics_mixture_at_zero,
            )

            manual_adaptive_ece_anchored_model_tail = _manual_adaptive_ece_equal_mass_15(
                anchored_model_tail_probs, test_y
            )
            manual_adaptive_ece_mixture_at_zero = _manual_adaptive_ece_equal_mass_15(
                mixture_at_zero_probs, test_y
            )
            logger.info(
                "DEBUG_ANCHOR_PROBS_DIFF | manual adaptive-ECE (15 equal-mass bins): anchored_model_tail=%.16e anchored_rankgeom_tail_mixture@0,0=%.16e",
                manual_adaptive_ece_anchored_model_tail,
                manual_adaptive_ece_mixture_at_zero,
            )

            diff_rows = np.where(row_max_abs_diff > 1e-10)[0]
            if diff_rows.size == 0:
                logger.info(
                    "DEBUG_ANCHOR_PROBS_DIFF | no rows with max-abs-diff > 1e-10 between anchored_model_tail and mixture@0,0"
                )
            else:
                top_rows = diff_rows[np.argsort(row_max_abs_diff[diff_rows])[-3:][::-1]]
                mixture_top_idx = np.argmax(test_base_probs, axis=1)
                mixture_p_t = test_base_probs[np.arange(len(mixture_top_idx)), mixture_top_idx]
                logger.info(
                    "DEBUG_ANCHOR_PROBS_DIFF | top-3 row-wise max-abs-diff samples anchored_model_tail vs mixture@0,0 | idx | row_max_abs_diff | anchored_top5 | mixture_top5 | base_top3 | c_t | p_t | y_true"
                )
                for i in top_rows:
                    logger.info(
                        "DEBUG_ANCHOR_PROBS_DIFF | sample | %d | %.16e | %s | %s | %s | %.9f | %.9f | %d",
                        int(i),
                        float(row_max_abs_diff[i]),
                        _topk_entries(anchored_model_tail_probs[i], 5),
                        _topk_entries(mixture_at_zero_probs[i], 5),
                        _top3_entries(test_base_probs[i]),
                        float(test_anchor_c_t[i]),
                        float(mixture_p_t[i]),
                        int(test_y[i]),
                    )
            logger.info("DEBUG_ANCHOR_PROBS_DIFF | ===== end test-set diagnostics =====")
    anchored_top_label_ece = float(
        evaluate_all(anchored_model_tail_probs, test_y).get("top_label_ece")
    )
    if need_rankgeom_method:
        selected_top_label_ece = float(
            evaluate_all(anchored_rankgeom_mixture_probs, test_y).get("top_label_ece")
        )
        anchored_rankgeom_extra: Dict[str, Any] = {
            "selected_lambda": float(selected_lambda),
            "selected_alpha": float(selected_alpha),
            "anchor_source_method": "gc_dac",
            "validation_anchor_mode": validation_anchor_mode,
            "validation_anchor_folds": int(val_anchor_diag.get("folds", 0)),
            "validation_anchor_fold_sizes": val_anchor_diag.get("per_fold_sizes"),
            "validation_anchor_stats_all": validation_anchor_stats_all,
            "validation_anchor_stats_correct": validation_anchor_stats_correct,
            "validation_anchor_stats_incorrect": validation_anchor_stats_incorrect,
            "nll_by_lambda_alpha": nll_by_lambda_alpha,
            "anchored_model_tail_top_label_ece": anchored_top_label_ece,
            "selected_top_label_ece": selected_top_label_ece,
            "ece_exceeds_anchor_baseline": bool(
                selected_top_label_ece > anchored_top_label_ece + 0.005
            ),
            "anchor_ct_below_0_5_count": int(np.sum(test_anchor_c_t < 0.5)),
            "anchor_ct_below_0_1_count": int(np.sum(test_anchor_c_t < 0.1)),
            "anchor_consistency_audit": {
                "val_nll_at_selected_crossfit_anchor": float(val_nll_at_selected_crossfit_anchor),
                "val_nll_at_selected_full_anchor": (
                    None
                    if val_nll_at_selected_full_anchor is None
                    else float(val_nll_at_selected_full_anchor)
                ),
                "test_nll_at_selected": float(test_nll_at_selected),
                "anchor_consistency_gap_crossfit_minus_full": (
                    None
                    if val_nll_at_selected_full_anchor is None
                    else float(
                        val_nll_at_selected_crossfit_anchor
                        - val_nll_at_selected_full_anchor
                    )
                ),
                "val_test_gap_full_anchor": (
                    None
                    if val_nll_at_selected_full_anchor is None
                    else float(
                        val_nll_at_selected_full_anchor - test_nll_at_selected
                    )
                ),
            },
        }
        anchored_rankgeom_extra.update(
            _argmax_flip_metadata(test_base_probs, anchored_rankgeom_mixture_probs, test_y)
        )
        anchored_rankgeom_extra.update(
            _nll_by_base_prediction_subset(test_base_probs, anchored_rankgeom_mixture_probs, test_y)
        )
        anchored_rankgeom_extra["changed_to_correct"] = anchored_rankgeom_extra.pop(
            "flip_to_correct_count"
        )
        anchored_rankgeom_extra["changed_to_wrong"] = anchored_rankgeom_extra.pop(
            "flip_to_wrong_count"
        )
        anchored_rankgeom_extra["net_flips"] = int(
            anchored_rankgeom_extra["changed_to_correct"]
            - anchored_rankgeom_extra["changed_to_wrong"]
        )
        anchored_rankgeom_extra.update(
            _nll_by_final_prediction_subset(anchored_rankgeom_mixture_probs, test_y)
        )
        anchored_rankgeom_extra.update(
            _non_top_distance_summary(test_base_probs, test_distance_matrix)
        )
        logger.info("Starting method %s", "anchored_rankgeom_tail_mixture")
        anchored_rankgeom_entry = _make_method_entry(
            "anchored_rankgeom_tail_mixture",
            anchored_rankgeom_mixture_probs,
            test_y,
            "decision_changing_full_vector",
            True,
            "grid_search_lambda_alpha",
            "validation",
            "validation_nll",
            anchored_rankgeom_extra,
            anchor_ct_for_stratification=gc_dac_test_anchor_ct,
            base_probs_for_anchor_bins=base_probs_test,
            base_probs=test_base_probs,
        )
        _save_method_checkpoint(
            args.output_dir,
            "anchored_rankgeom_tail_mixture",
            anchored_rankgeom_entry,
            anchored_rankgeom_mixture_probs,
            True,
            stage_name="stage4_late_outputs",
        )
        logger.info("Saved method checkpoint: %s", "anchored_rankgeom_tail_mixture")
        rankgeom_selection = {
            "selected_lambda": float(selected_lambda),
            "selected_alpha": float(selected_alpha),
            "val_nll_at_selected_crossfit_anchor": float(val_nll_at_selected_crossfit_anchor),
            "val_nll_at_selected_full_anchor": (
                None
                if val_nll_at_selected_full_anchor is None
                else float(val_nll_at_selected_full_anchor)
            ),
            "test_nll_at_selected": float(test_nll_at_selected),
        }
        _atomic_write_json(late_anchor_paths["rankgeom_selection"], rankgeom_selection)
    else:
        logger.info("Skipping completed method %s", "anchored_rankgeom_tail_mixture")
        anchored_rankgeom_mixture_probs, rankgeom_selection = (
            _load_completed_rankgeom_resume_state(args.output_dir, rankgeom_selection)
        )
    method_can_change_argmax_map["anchored_rankgeom_tail_mixture"] = True
    del anchored_rankgeom_mixture_probs
    if need_rankgeom_method and model is not None and model_adapter is not None and train_raw is not None:
        del train_raw, model_adapter, model
    _cleanup_memory()
    logger.info("[cleanup] released arrays after method: anchored_rankgeom_tail_mixture")

    if _is_method_checkpoint_complete(
        args.output_dir, "full_vector_distance_fusion", len(test_labels), num_classes
    ):
        logger.info("Skipping completed method %s", "full_vector_distance_fusion")
    else:
        fusion_entry = _make_method_entry(
            "full_vector_distance_fusion",
            fused_test_probs,
            test_y,
            "decision_changing_full_vector",
            True,
            "beta_grid_search",
            "validation",
            "validation_nll_over_beta_grid",
            {
                "selected_beta": float(fusion_meta["selected_beta"]),
                "beta_selection": fusion_meta["beta_selection"],
                "stability_space": fusion_meta.get("stability_space", {}),
                "validation_gates": fusion_meta["validation_gates"],
                "argmax_change_rate": float(fusion_meta["argmax_change_rate"]),
                "changed_to_correct": int(fusion_meta["changed_to_correct"]),
                "changed_to_wrong": int(fusion_meta["changed_to_wrong"]),
                "net_flips": int(fusion_meta["net_flips"]),
                **_nll_by_base_prediction_subset(test_base_probs, fused_test_probs, test_y),
                **_non_top_distance_summary(test_base_probs, test_distance_matrix),
            },
            anchor_ct_for_stratification=gc_dac_test_anchor_ct,
            base_probs_for_anchor_bins=base_probs_test,
            base_probs=test_base_probs,
        )
        _save_method_checkpoint(
            args.output_dir,
            "full_vector_distance_fusion",
            fusion_entry,
            fused_test_probs,
            True,
            stage_name="stage4_late_outputs",
        )
        logger.info("Saved method checkpoint: %s", "full_vector_distance_fusion")
    method_can_change_argmax_map["full_vector_distance_fusion"] = True
    _cleanup_memory()
    logger.info("[cleanup] released arrays after method: full_vector_distance_fusion")

    if _is_method_checkpoint_complete(
        args.output_dir, "post_fusion_topiso", len(test_labels), num_classes
    ):
        logger.info("Skipping completed method %s", "post_fusion_topiso")
    else:
        logger.info("Starting method %s", "post_fusion_topiso")
        post_topiso_probs, post_topiso_extra = _fit_post_fusion_topiso_probs(
            fused_val_probs=fused_val_probs,
            val_labels=val_y,
            fused_test_probs=fused_test_probs,
        )
        _validate_probability_matrix(
            post_topiso_probs,
            n_rows=len(test_y),
            n_classes=num_classes,
            method_name="post_fusion_topiso",
        )
        fusion_preds = np.argmax(fused_test_probs, axis=1)
        post_topiso_preds = np.argmax(post_topiso_probs, axis=1)
        argmax_change_rate_vs_fusion = float(np.mean(post_topiso_preds != fusion_preds))
        # argmax_change_rate_vs_fusion is informational; the method is structurally decision-changing per addendum 2026-04-26 in the card.

        # Pilot framing: primary deltas are vs fusion; gc_dac is the top-label ECE specialist.
        fusion_metrics = evaluate_all(fused_test_probs, test_y)
        post_topiso_metrics = evaluate_all(post_topiso_probs, test_y)
        gc_dac_metrics = evaluate_all(gc_dac_probs, test_y)
        post_topiso_extra.update(
            {
                "parent_method": "full_vector_distance_fusion",
                "argmax_change_rate_vs_fusion": argmax_change_rate_vs_fusion,
                "top_label_ece_delta_vs_fusion": float(
                    post_topiso_metrics["top_label_ece"] - fusion_metrics["top_label_ece"]
                ),
                "nll_delta_vs_fusion": float(post_topiso_metrics["nll"] - fusion_metrics["nll"]),
                "brier_delta_vs_fusion": float(
                    post_topiso_metrics["brier"] - fusion_metrics["brier"]
                ),
                "accuracy_delta_vs_fusion": float(
                    post_topiso_metrics["accuracy"] - fusion_metrics["accuracy"]
                ),
                "adaptive_ece_delta_vs_fusion": float(
                    post_topiso_metrics["adaptive_ece"] - fusion_metrics["adaptive_ece"]
                ),
                "top_label_ece_delta_vs_gc_dac": float(
                    post_topiso_metrics["top_label_ece"] - gc_dac_metrics["top_label_ece"]
                ),
                **_nll_by_base_prediction_subset(test_base_probs, post_topiso_probs, test_y),
            }
        )
        post_topiso_entry = _make_method_entry(
            "post_fusion_topiso",
            post_topiso_probs,
            test_y,
            "decision_changing_full_vector",
            True,
            "fusion_then_top_isotonic",
            "validation",
            "top_label_correctness_after_fusion",
            post_topiso_extra,
            anchor_ct_for_stratification=gc_dac_test_anchor_ct,
            base_probs_for_anchor_bins=base_probs_test,
            base_probs=test_base_probs,
        )
        _save_method_checkpoint(
            args.output_dir,
            "post_fusion_topiso",
            post_topiso_entry,
            post_topiso_probs,
            True,
            stage_name="stage4_late_outputs",
        )
        logger.info("Saved method checkpoint: %s", "post_fusion_topiso")
        del post_topiso_probs
    method_can_change_argmax_map["post_fusion_topiso"] = True
    _cleanup_memory()
    logger.info("[cleanup] released arrays after method: post_fusion_topiso")

    if args.enable_post_fusion_temperature:
        fusion_preds = np.argmax(fused_test_probs, axis=1)
        if _is_method_checkpoint_complete(
            args.output_dir,
            "full_vector_distance_fusion_post_temperature",
            len(test_labels),
            num_classes,
        ):
            logger.info("Skipping completed method %s", "full_vector_distance_fusion_post_temperature")
        else:
            fused_val_logits = np.log(np.clip(fused_val_probs, 1e-12, 1.0))
            fused_test_logits = np.log(np.clip(fused_test_probs, 1e-12, 1.0))
            post_ts = TemperatureScaling()
            # Fit on fusion-branch aligned validation labels to preserve split consistency.
            post_ts.fit(fused_val_logits, val_y)
            post_ts_probs = post_ts.calibrate(fused_test_logits)
            post_preds = np.argmax(post_ts_probs, axis=1)
            argmax_change_rate_post_vs_fusion = float(np.mean(post_preds != fusion_preds))
            post_ts_entry = _make_method_entry(
                "full_vector_distance_fusion_post_temperature",
                post_ts_probs,
                test_y,
                "decision_changing_full_vector",
                True,
                "scalar_temperature_fit_after_fusion",
                "validation",
                "validation_nll_after_fusion",
                {
                    "post_fusion_temperature": _temperature_to_float(post_ts.temperature),
                    "post_fusion_calibration_type": "temperature",
                    "argmax_change_rate_post_vs_fusion": argmax_change_rate_post_vs_fusion,
                    # Keep legacy key for backward compatibility with historical outputs.
                    "post_temperature": _temperature_to_float(post_ts.temperature),
                    **_nll_by_base_prediction_subset(test_base_probs, post_ts_probs, test_y),
                },
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=base_probs_test,
                base_probs=test_base_probs,
            )
            _save_method_checkpoint(
                args.output_dir,
                "full_vector_distance_fusion_post_temperature",
                post_ts_entry,
                post_ts_probs,
                True,
                stage_name="stage4_late_outputs",
            )
            logger.info("Saved method checkpoint: %s", "full_vector_distance_fusion_post_temperature")
            del post_ts, post_ts_probs, post_preds, fused_val_logits, fused_test_logits
        method_can_change_argmax_map["full_vector_distance_fusion_post_temperature"] = True
        _cleanup_memory()
        logger.info("[cleanup] released arrays after method: full_vector_distance_fusion_post_temperature")

    optional_full_vector_methods = any(
        [
            args.enable_full_vector_geometric_fusion,
            args.enable_knn_blend_baseline,
            args.enable_kcal_lite_baseline,
            args.enable_rgcl_neighbor_correction,
            args.enable_kcal_factorial,
        ]
    )
    methods: List[Dict[str, Any]] = []
    method_probs_by_name: Dict[str, np.ndarray] = {}
    _single_layer_feature_meta = _load_json(stage3_files["feature_metadata"])
    _single_layer_name = _single_layer_feature_meta["selected_layers"][0]

    if optional_full_vector_methods:
        model = load_trained_model(
            model_path, args.model, num_classes, device, dataset=args.dataset
        )
        model_adapter = PyTorchModelAdapter(model, device, args.dataset)
        train_features = np.load(stage3_files["train_features"], mmap_mode="r")
        train_y = np.load(stage3_files["train_y"])
        val_features = np.load(stage3_files["val_features"], mmap_mode="r")
        test_features = np.load(stage3_files["test_features"], mmap_mode="r")
        fusion_out = {
            "stab_space": StabilitySpace(
                train_features,
                train_y,
                library="fast_separation",
                metric=args.stab_metric,
                whitening_components=args.whitening_components,
                whitening_eps=args.whitening_eps,
            ),
            "selected_beta": float(fusion_meta["selected_beta"]),
        }
        if args.fusion_feature_source == "rgcl" or args.enable_kcal_factorial:
            train_raw = np.load(stage1_files["train_raw"], mmap_mode="r")
            test_raw = np.load(stage1_files["test_raw"], mmap_mode="r")
            rgcl = {
                "selected_layers": _select_rgcl_layers(
                    model=model,
                    model_name=args.model,
                    dataset_name=args.dataset,
                    device=device,
                    num_layers=args.num_layers,
                    seed=args.seed,
                )
            }

    # Cache rgcl features/stability-space so the knn-blend block can reuse them
    # when --enable_full_vector_geometric_fusion + --fusion_feature_source rgcl is also set.
    _knn_blend_rgcl_stab_space = None
    _knn_blend_rgcl_train_features = None
    _knn_blend_rgcl_val_features = None
    _knn_blend_rgcl_test_features = None
    _kcal_factorial_paths = _kcal_factorial_feature_paths(args.output_dir)
    _kcal_factorial_selected_params: Dict[str, Dict[str, Any]] = {}
    _kcal_factorial_method_artifacts: Dict[str, Dict[str, Any]] = {}

    if args.enable_kcal_factorial:
        rgcl_cache_files = [
            _kcal_factorial_paths["rgcl_train"],
            _kcal_factorial_paths["rgcl_validation"],
            _kcal_factorial_paths["rgcl_test"],
            _kcal_factorial_paths["rgcl_metadata"],
        ]
        if all(os.path.exists(path) for path in rgcl_cache_files):
            logger.info("Reusing cached RGCL embeddings for KCal factorial study")
            _factorial_rgcl_train = np.load(
                _kcal_factorial_paths["rgcl_train"], mmap_mode="r"
            )
            _factorial_rgcl_validation = np.load(
                _kcal_factorial_paths["rgcl_validation"], mmap_mode="r"
            )
            _factorial_rgcl_test = np.load(
                _kcal_factorial_paths["rgcl_test"], mmap_mode="r"
            )
            _factorial_rgcl_meta = _load_json(
                _kcal_factorial_paths["rgcl_metadata"]
            )
        else:
            logger.info("Extracting and caching RGCL embeddings for KCal factorial study")
            _factorial_rgcl_train_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(train_raw), torch.from_numpy(train_labels)
                ),
                batch_size=args.batch_size,
                shuffle=False,
            )
            _factorial_rgcl_validation_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(val_raw_aligned), torch.from_numpy(val_y)
                ),
                batch_size=args.batch_size,
                shuffle=False,
            )
            _factorial_rgcl_test_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(test_raw),
                    torch.zeros(len(test_raw), dtype=torch.long),
                ),
                batch_size=args.batch_size,
                shuffle=False,
            )
            (
                _factorial_rgcl_train,
                _factorial_rgcl_validation,
                _factorial_rgcl_test,
                _factorial_rgcl_extraction_info,
            ) = extract_and_aggregate_sgc_features(
                model=model,
                layer_names=rgcl["selected_layers"],
                train_loader=_factorial_rgcl_train_loader,
                val_loader=_factorial_rgcl_validation_loader,
                test_loader=_factorial_rgcl_test_loader,
                device=device,
                target_dim=args.target_dimension,
                seed=args.seed,
                pooling_mode="max",
            )
            _save_npy_atomic(
                _kcal_factorial_paths["rgcl_train"], _factorial_rgcl_train
            )
            _save_npy_atomic(
                _kcal_factorial_paths["rgcl_validation"],
                _factorial_rgcl_validation,
            )
            _save_npy_atomic(
                _kcal_factorial_paths["rgcl_test"], _factorial_rgcl_test
            )
            _factorial_rgcl_meta = {
                "feature_source": "rgcl",
                "selected_layers": list(rgcl["selected_layers"]),
                "actual_num_layers": len(rgcl["selected_layers"]),
                "target_dimension": int(args.target_dimension),
                "output_dimension": int(_factorial_rgcl_train.shape[1]),
                "pooling_mode": "max",
                "seed": int(args.seed),
                "extraction_time_s": float(
                    _factorial_rgcl_extraction_info.get("extraction_time_s", 0.0)
                ),
            }
            _atomic_write_json(
                _kcal_factorial_paths["rgcl_metadata"], _factorial_rgcl_meta
            )
            _factorial_rgcl_train = np.load(
                _kcal_factorial_paths["rgcl_train"], mmap_mode="r"
            )
            _factorial_rgcl_validation = np.load(
                _kcal_factorial_paths["rgcl_validation"], mmap_mode="r"
            )
            _factorial_rgcl_test = np.load(
                _kcal_factorial_paths["rgcl_test"], mmap_mode="r"
            )

        if len(_factorial_rgcl_train) != len(train_y):
            raise RuntimeError("Cached RGCL train features are misaligned")
        if len(_factorial_rgcl_validation) != len(val_y):
            raise RuntimeError("Cached RGCL validation features are misaligned")
        if len(_factorial_rgcl_test) != len(test_y):
            raise RuntimeError("Cached RGCL test features are misaligned")
        _knn_blend_rgcl_train_features = _factorial_rgcl_train
        _knn_blend_rgcl_val_features = _factorial_rgcl_validation
        _knn_blend_rgcl_test_features = _factorial_rgcl_test
        _knn_blend_rgcl_stab_space = StabilitySpace(
            _factorial_rgcl_train,
            train_y,
            library="fast_separation",
            metric=args.stab_metric,
            whitening_components=args.whitening_components,
            whitening_eps=args.whitening_eps,
        )

        # Always include the observed KCal-lite-RGCL endpoint in the same final
        # table. If the normal KCal-lite RGCL block is explicitly enabled below,
        # let that path own the row and its distance artifact instead.
        _factorial_owns_kcal_lite_rgcl = not (
            args.enable_kcal_lite_baseline
            and args.fusion_feature_source == "rgcl"
        )
        if _factorial_owns_kcal_lite_rgcl:
            _factorial_endpoint_name = "kcal_lite_rgcl"
            if _is_method_checkpoint_complete(
                args.output_dir,
                _factorial_endpoint_name,
                len(test_y),
                num_classes,
            ):
                _factorial_endpoint_entry = _load_method_entry(
                    args.output_dir, _factorial_endpoint_name
                )
                _kcal_factorial_selected_params[_factorial_endpoint_name] = {
                    "alpha": float(_factorial_endpoint_entry["selected_alpha"]),
                    "gamma": float(_factorial_endpoint_entry["selected_gamma"]),
                    "k_per_class": int(
                        _factorial_endpoint_entry["k_per_class"]
                    ),
                }
                _kcal_factorial_method_artifacts[_factorial_endpoint_name] = {
                    "distance_array_key": None,
                    "formula": "KCalLiteCalibrator",
                    "feature_source": "rgcl",
                    "feature_cache": {
                        "train": _kcal_factorial_paths["rgcl_train"],
                        "validation": _kcal_factorial_paths["rgcl_validation"],
                        "test": _kcal_factorial_paths["rgcl_test"],
                    },
                }
                method_can_change_argmax_map[_factorial_endpoint_name] = True
            else:
                logger.info(
                    "Evaluating exact KCal-lite-RGCL endpoint from cached embeddings"
                )
                _factorial_endpoint_kcal = KCalLiteCalibrator(
                    model=model_adapter,
                    y_train=train_y,
                    k_per_class=args.kcal_factorial_k_per_class,
                    stability_space=_knn_blend_rgcl_stab_space,
                )
                _factorial_endpoint_val_distances = (
                    _knn_blend_rgcl_stab_space.calc_per_class_knn_distances(
                        _factorial_rgcl_validation,
                        args.kcal_factorial_k_per_class,
                    )
                )
                _factorial_endpoint_test_distances = (
                    _knn_blend_rgcl_stab_space.calc_per_class_knn_distances(
                        _factorial_rgcl_test,
                        args.kcal_factorial_k_per_class,
                    )
                )
                _factorial_endpoint_kcal.fit(
                    X_val_embed=_factorial_rgcl_validation,
                    y_val=val_y,
                    model_probs=val_base_probs,
                    val_knn_distances=_factorial_endpoint_val_distances,
                )
                _factorial_endpoint_probs = _factorial_endpoint_kcal.calibrate(
                    X_test_embed=_factorial_rgcl_test,
                    model_probs=test_base_probs,
                    test_knn_distances=_factorial_endpoint_test_distances,
                )
                _validate_probability_matrix(
                    _factorial_endpoint_probs,
                    n_rows=len(test_y),
                    n_classes=num_classes,
                    method_name=_factorial_endpoint_name,
                )
                _factorial_endpoint_flip_meta = _argmax_flip_metadata(
                    test_base_probs, _factorial_endpoint_probs, test_y
                )
                _factorial_endpoint_extra: Dict[str, Any] = {
                    "selected_alpha": float(_factorial_endpoint_kcal.best_alpha),
                    "selected_gamma": float(_factorial_endpoint_kcal.best_gamma),
                    "alpha_grid": _factorial_endpoint_kcal.alpha_grid.tolist(),
                    "gamma_grid": _factorial_endpoint_kcal.gamma_grid.tolist(),
                    "k_per_class": int(args.kcal_factorial_k_per_class),
                    "validation_nll_grid": (
                        _factorial_endpoint_kcal.validation_nll_grid.tolist()
                    ),
                    "feature_source": "rgcl",
                    "distance_source": "reused_cached_factorial_rgcl_features",
                    "selected_layers": list(rgcl["selected_layers"]),
                    "feature_cache": {
                        "train": _kcal_factorial_paths["rgcl_train"],
                        "validation": _kcal_factorial_paths["rgcl_validation"],
                        "test": _kcal_factorial_paths["rgcl_test"],
                    },
                    "included_as_factorial_endpoint": True,
                    "net_flips": int(
                        _factorial_endpoint_flip_meta["flip_to_correct_count"]
                        - _factorial_endpoint_flip_meta["flip_to_wrong_count"]
                    ),
                }
                _factorial_endpoint_extra.update(_factorial_endpoint_flip_meta)
                _factorial_endpoint_extra.update(
                    _nll_by_base_prediction_subset(
                        test_base_probs, _factorial_endpoint_probs, test_y
                    )
                )
                _factorial_endpoint_extra.update(
                    _nll_by_final_prediction_subset(
                        _factorial_endpoint_probs, test_y
                    )
                )
                _factorial_endpoint_entry = _make_method_entry(
                    _factorial_endpoint_name,
                    _factorial_endpoint_probs,
                    test_y,
                    "decision_changing_full_vector",
                    True,
                    "alpha_gamma_grid_search",
                    "validation",
                    "validation_nll",
                    _factorial_endpoint_extra,
                    anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                    base_probs_for_anchor_bins=base_probs_test,
                    base_probs=test_base_probs,
                )
                _save_method_checkpoint(
                    args.output_dir,
                    _factorial_endpoint_name,
                    _factorial_endpoint_entry,
                    _factorial_endpoint_probs,
                    True,
                    stage_name="stage4_kcal_factorial_endpoint",
                )
                method_can_change_argmax_map[_factorial_endpoint_name] = True
                _kcal_factorial_selected_params[_factorial_endpoint_name] = {
                    "alpha": float(_factorial_endpoint_kcal.best_alpha),
                    "gamma": float(_factorial_endpoint_kcal.best_gamma),
                    "k_per_class": int(args.kcal_factorial_k_per_class),
                }
                _kcal_factorial_method_artifacts[_factorial_endpoint_name] = {
                    "distance_array_key": None,
                    "formula": "KCalLiteCalibrator",
                    "feature_source": "rgcl",
                    "feature_cache": _factorial_endpoint_extra["feature_cache"],
                }
                del (
                    _factorial_endpoint_kcal,
                    _factorial_endpoint_val_distances,
                    _factorial_endpoint_test_distances,
                    _factorial_endpoint_probs,
                    _factorial_endpoint_entry,
                )
                _cleanup_memory()

    # Per-method FVGF selected lambda and score mode collected inside the fvgf loops below.
    _fvgf_selected_lambdas: Dict[str, float] = {}
    _fvgf_score_modes_by_name: Dict[str, str] = {}

    if args.enable_full_vector_geometric_fusion:
        _all_valid_modes = set(FullVectorGeometricFusionCalibrator._VALID_SCORE_MODES)
        requested_score_modes = [s.strip() for s in args.fvgf_score_modes.split(",") if s.strip()]
        if not requested_score_modes:
            raise ValueError(
                "--fvgf_score_modes must specify at least one mode when --enable_full_vector_geometric_fusion is set"
            )
        invalid_modes = sorted(m for m in requested_score_modes if m not in _all_valid_modes)
        if invalid_modes:
            raise ValueError(
                "Invalid --fvgf_score_modes values: "
                + ",".join(invalid_modes)
                + ". Allowed: "
                + ",".join(sorted(_all_valid_modes))
            )
        if len(requested_score_modes) != len(set(requested_score_modes)):
            raise ValueError("--fvgf_score_modes contains duplicate score modes")

        fvgf_lambda_grid = None
        if args.fvgf_lambda_grid.strip():
            try:
                fvgf_lambda_grid = np.array(
                    [float(x) for x in args.fvgf_lambda_grid.split(",") if x.strip()],
                    dtype=np.float64,
                )
            except ValueError as exc:
                raise ValueError(
                    f"--fvgf_lambda_grid must contain numeric values: {exc}"
                ) from exc
            if len(fvgf_lambda_grid) == 0:
                fvgf_lambda_grid = None

        if args.fusion_feature_source == "single_layer":
            for score_mode in requested_score_modes:
                method_name = f"full_vector_geometric_fusion_{score_mode}"
                fvgf = FullVectorGeometricFusionCalibrator(
                    model=model_adapter,
                    X_train_embed=train_features,
                    y_train=train_y,
                    metric=args.stab_metric,
                    library="fast_separation",
                    whitening_components=args.whitening_components,
                    whitening_eps=args.whitening_eps,
                    lambda_grid=fvgf_lambda_grid,
                    score_mode=score_mode,
                    stability_space=fusion_out["stab_space"],
                )
                fvgf.fit(X_val_embed=val_features, y_val=val_y, model_probs=val_base_probs)
                fvgf_test_probs = fvgf.calibrate(
                    X_test_embed=test_features, model_probs=test_base_probs
                )
                _validate_probability_matrix(
                    fvgf_test_probs,
                    n_rows=len(test_y),
                    n_classes=num_classes,
                    method_name=method_name,
                )

                flip_meta = _argmax_flip_metadata(test_base_probs, fvgf_test_probs, test_y)
                net_flips = int(
                    flip_meta["flip_to_correct_count"] - flip_meta["flip_to_wrong_count"]
                )

                fvgf_extra: Dict[str, Any] = {
                    "score_mode": score_mode,
                    "selected_lambda": float(fvgf.best_lambda),
                    "lambda_selection": fvgf.lambda_selection,
                    "lambda_zero_max_abs_diff": fvgf.lambda0_max_abs_diff,
                    "net_flips": net_flips,
                }
                fvgf_extra.update(flip_meta)
                fvgf_extra.update(
                    _nll_by_base_prediction_subset(test_base_probs, fvgf_test_probs, test_y)
                )
                fvgf_extra.update(_nll_by_final_prediction_subset(fvgf_test_probs, test_y))
                fvgf_extra.update(_single_layer_feature_meta)

                methods.append(
                    _make_method_entry(
                        method_name,
                        fvgf_test_probs,
                        test_y,
                        "decision_changing_full_vector",
                        True,
                        "lambda_grid_search",
                        "validation",
                        "validation_nll",
                        fvgf_extra,
                        anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                        base_probs_for_anchor_bins=base_probs_test,
                        base_probs=test_base_probs,
                    )
                )
                method_probs_by_name[method_name] = fvgf_test_probs
                method_can_change_argmax_map[method_name] = True
                _fvgf_selected_lambdas[method_name] = float(fvgf.best_lambda)
                _fvgf_score_modes_by_name[method_name] = score_mode

        else:  # fusion_feature_source == "rgcl"
            rgcl_selected_layers = rgcl["selected_layers"]
            if _knn_blend_rgcl_train_features is not None:
                logger.info("Reusing cached RGCL embeddings for full-vector fusion")
                rgcl_fvgf_train_features = _knn_blend_rgcl_train_features
                rgcl_fvgf_val_features = _knn_blend_rgcl_val_features
                rgcl_fvgf_test_features = _knn_blend_rgcl_test_features
                rgcl_fvgf_extraction_info = {
                    "extraction_time_s": float(
                        _factorial_rgcl_meta.get("extraction_time_s", 0.0)
                    )
                }
            else:
                _rgcl_fvgf_train_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(train_raw),
                        torch.from_numpy(train_labels),
                    ),
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                _rgcl_fvgf_val_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(val_raw_aligned),
                        torch.from_numpy(val_y),
                    ),
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                _rgcl_fvgf_test_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(test_raw),
                        torch.zeros(len(test_raw), dtype=torch.long),
                    ),
                    batch_size=args.batch_size,
                    shuffle=False,
                )

                (
                    rgcl_fvgf_train_features,
                    rgcl_fvgf_val_features,
                    rgcl_fvgf_test_features,
                    rgcl_fvgf_extraction_info,
                ) = extract_and_aggregate_sgc_features(
                    model=model,
                    layer_names=rgcl_selected_layers,
                    train_loader=_rgcl_fvgf_train_loader,
                    val_loader=_rgcl_fvgf_val_loader,
                    test_loader=_rgcl_fvgf_test_loader,
                    device=device,
                    target_dim=args.target_dimension,
                    seed=args.seed,
                    pooling_mode="max",
                )

            assert len(rgcl_fvgf_train_features) == len(train_y), (
                f"RGCL fvgf train features row count {len(rgcl_fvgf_train_features)} "
                f"!= train_y length {len(train_y)}"
            )
            assert len(rgcl_fvgf_val_features) == len(val_y), (
                f"RGCL fvgf val features row count {len(rgcl_fvgf_val_features)} "
                f"!= val_y length {len(val_y)}"
            )
            assert len(rgcl_fvgf_test_features) == len(test_y), (
                f"RGCL fvgf test features row count {len(rgcl_fvgf_test_features)} "
                f"!= test_y length {len(test_y)}"
            )

            rgcl_fvgf_stab_space = StabilitySpace(
                rgcl_fvgf_train_features,
                train_labels,
                library="fast_separation",
                metric=args.stab_metric,
                whitening_components=args.whitening_components,
                whitening_eps=args.whitening_eps,
            )

            _rgcl_fvgf_feature_meta: Dict[str, Any] = {
                "feature_source": "rgcl",
                "selected_layers": list(rgcl_selected_layers),
                "actual_num_layers": len(rgcl_selected_layers),
                "target_dimension": args.target_dimension,
                "pooling_mode": "max",
                "extraction_time_s": float(
                    rgcl_fvgf_extraction_info.get("extraction_time_s", 0.0)
                ),
            }

            for score_mode in requested_score_modes:
                method_name = f"full_vector_geometric_fusion_rgcl_{score_mode}"
                fvgf = FullVectorGeometricFusionCalibrator(
                    model=model_adapter,
                    X_train_embed=rgcl_fvgf_train_features,
                    y_train=train_labels,
                    metric=args.stab_metric,
                    library="fast_separation",
                    whitening_components=args.whitening_components,
                    whitening_eps=args.whitening_eps,
                    lambda_grid=fvgf_lambda_grid,
                    score_mode=score_mode,
                    stability_space=rgcl_fvgf_stab_space,
                )
                fvgf.fit(
                    X_val_embed=rgcl_fvgf_val_features,
                    y_val=val_y,
                    model_probs=val_base_probs,
                )
                fvgf_test_probs = fvgf.calibrate(
                    X_test_embed=rgcl_fvgf_test_features,
                    model_probs=test_base_probs,
                )
                _validate_probability_matrix(
                    fvgf_test_probs,
                    n_rows=len(test_y),
                    n_classes=num_classes,
                    method_name=method_name,
                )

                flip_meta = _argmax_flip_metadata(test_base_probs, fvgf_test_probs, test_y)
                net_flips = int(
                    flip_meta["flip_to_correct_count"] - flip_meta["flip_to_wrong_count"]
                )

                fvgf_extra: Dict[str, Any] = {
                    "score_mode": score_mode,
                    "selected_lambda": float(fvgf.best_lambda),
                    "lambda_selection": fvgf.lambda_selection,
                    "lambda_zero_max_abs_diff": fvgf.lambda0_max_abs_diff,
                    "net_flips": net_flips,
                }
                fvgf_extra.update(flip_meta)
                fvgf_extra.update(
                    _nll_by_base_prediction_subset(test_base_probs, fvgf_test_probs, test_y)
                )
                fvgf_extra.update(_nll_by_final_prediction_subset(fvgf_test_probs, test_y))
                fvgf_extra.update(_rgcl_fvgf_feature_meta)

                methods.append(
                    _make_method_entry(
                        method_name,
                        fvgf_test_probs,
                        test_y,
                        "decision_changing_full_vector",
                        True,
                        "lambda_grid_search",
                        "validation",
                        "validation_nll",
                        fvgf_extra,
                        anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                        base_probs_for_anchor_bins=base_probs_test,
                        base_probs=test_base_probs,
                    )
                )
                method_probs_by_name[method_name] = fvgf_test_probs
                method_can_change_argmax_map[method_name] = True
                _fvgf_selected_lambdas[method_name] = float(fvgf.best_lambda)
                _fvgf_score_modes_by_name[method_name] = score_mode

            # Cache so knn_blend block can reuse without re-extracting.
            _knn_blend_rgcl_stab_space = rgcl_fvgf_stab_space
            _knn_blend_rgcl_train_features = rgcl_fvgf_train_features
            _knn_blend_rgcl_val_features = rgcl_fvgf_val_features
            _knn_blend_rgcl_test_features = rgcl_fvgf_test_features

    if args.enable_knn_blend_baseline:
        _use_rgcl_blend = args.fusion_feature_source == "rgcl"

        if _use_rgcl_blend:
            if _knn_blend_rgcl_stab_space is not None:
                # Reuse already-computed rgcl features from fvgf block.
                _blend_stab_space = _knn_blend_rgcl_stab_space
                _blend_val_features = _knn_blend_rgcl_val_features
                _blend_test_features = _knn_blend_rgcl_test_features
                _blend_val_dm = _blend_stab_space.calc_per_class_1nn_distances(
                    _blend_val_features
                )
                _blend_test_dm = _blend_stab_space.calc_per_class_1nn_distances(
                    _blend_test_features
                )
                _blend_distance_source = "reused_full_vector_geometric_fusion_rgcl"
            else:
                # Compute rgcl features independently for knn blend.
                _blend_train_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(train_raw),
                        torch.from_numpy(train_labels),
                    ),
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                _blend_val_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(val_raw_aligned),
                        torch.from_numpy(val_y),
                    ),
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                _blend_test_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(test_raw),
                        torch.zeros(len(test_raw), dtype=torch.long),
                    ),
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                (
                    _blend_train_feats,
                    _blend_val_features,
                    _blend_test_features,
                    _,
                ) = extract_and_aggregate_sgc_features(
                    model=model,
                    layer_names=rgcl["selected_layers"],
                    train_loader=_blend_train_loader,
                    val_loader=_blend_val_loader,
                    test_loader=_blend_test_loader,
                    device=device,
                    target_dim=args.target_dimension,
                    seed=args.seed,
                    pooling_mode="max",
                )
                _blend_stab_space = StabilitySpace(
                    _blend_train_feats,
                    train_labels,
                    library="fast_separation",
                    metric=args.stab_metric,
                    whitening_components=args.whitening_components,
                    whitening_eps=args.whitening_eps,
                )
                _blend_val_dm = _blend_stab_space.calc_per_class_1nn_distances(
                    _blend_val_features
                )
                _blend_test_dm = _blend_stab_space.calc_per_class_1nn_distances(
                    _blend_test_features
                )
                _blend_distance_source = "computed_knn_blend_rgcl_independent"
                # Cache so kcal_lite_rgcl can reuse without re-extracting.
                _knn_blend_rgcl_stab_space = _blend_stab_space
                _knn_blend_rgcl_train_features = _blend_train_feats
                _knn_blend_rgcl_val_features = _blend_val_features
                _knn_blend_rgcl_test_features = _blend_test_features

            _blend_method_name = "softmax_knn_blend_rgcl"
            _blend_feature_source = "rgcl"
        else:
            # single_layer — reuse always-available fusion matrices.
            _blend_stab_space = fusion_out["stab_space"]
            _blend_val_features = val_features
            _blend_test_features = test_features
            _blend_val_dm = val_distance_matrix
            _blend_test_dm = test_distance_matrix
            _blend_distance_source = "reused_single_layer_full_vector_fusion"
            _blend_method_name = "softmax_knn_blend"
            _blend_feature_source = "single_layer"

        knn_blend = SoftmaxKNNBlendCalibrator(
            model=model_adapter,
            y_train=train_y,
            stability_space=_blend_stab_space,
        )
        knn_blend.fit(
            X_val_embed=_blend_val_features,
            y_val=val_y,
            model_probs=val_base_probs,
            val_distance_matrix=_blend_val_dm,
        )
        blend_test_probs = knn_blend.calibrate(
            X_test_embed=_blend_test_features,
            model_probs=test_base_probs,
            test_distance_matrix=_blend_test_dm,
        )
        _validate_probability_matrix(
            blend_test_probs,
            n_rows=len(test_y),
            n_classes=num_classes,
            method_name=_blend_method_name,
        )

        blend_flip_meta = _argmax_flip_metadata(test_base_probs, blend_test_probs, test_y)
        blend_extra: Dict[str, Any] = {
            "selected_alpha": knn_blend.best_alpha,
            "selected_gamma": knn_blend.best_gamma,
            "alpha_grid": knn_blend.alpha_grid.tolist(),
            "gamma_grid": knn_blend.gamma_grid.tolist(),
            "validation_nll_grid": knn_blend.validation_nll_grid.tolist(),
            "feature_source": _blend_feature_source,
            "distance_source": _blend_distance_source,
            "net_flips": int(
                blend_flip_meta["flip_to_correct_count"]
                - blend_flip_meta["flip_to_wrong_count"]
            ),
        }
        blend_extra.update(blend_flip_meta)
        blend_extra.update(
            _nll_by_base_prediction_subset(test_base_probs, blend_test_probs, test_y)
        )
        blend_extra.update(_nll_by_final_prediction_subset(blend_test_probs, test_y))

        methods.append(
            _make_method_entry(
                _blend_method_name,
                blend_test_probs,
                test_y,
                "decision_changing_full_vector",
                True,
                "alpha_gamma_grid_search",
                "validation",
                "validation_nll",
                blend_extra,
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=base_probs_test,
                base_probs=test_base_probs,
            )
        )
        method_probs_by_name[_blend_method_name] = blend_test_probs
        method_can_change_argmax_map[_blend_method_name] = True

    if args.enable_kcal_lite_baseline:
        _kcal_use_rgcl = args.fusion_feature_source == "rgcl"

        if _kcal_use_rgcl:
            if _knn_blend_rgcl_stab_space is not None:
                # Reuse RGCL features already computed by fvgf or knn_blend blocks.
                _kcal_stab_space = _knn_blend_rgcl_stab_space
                _kcal_val_features = _knn_blend_rgcl_val_features
                _kcal_test_features = _knn_blend_rgcl_test_features
                _kcal_distance_source = "reused_rgcl_features"
                _kcal_selected_layers = list(rgcl["selected_layers"])
            else:
                # Extract RGCL features independently for KCal-lite.
                _kcal_train_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(train_raw),
                        torch.from_numpy(train_labels),
                    ),
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                _kcal_val_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(val_raw_aligned),
                        torch.from_numpy(val_y),
                    ),
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                _kcal_test_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(test_raw),
                        torch.zeros(len(test_raw), dtype=torch.long),
                    ),
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                (
                    _kcal_train_feats,
                    _kcal_val_features,
                    _kcal_test_features,
                    _,
                ) = extract_and_aggregate_sgc_features(
                    model=model,
                    layer_names=rgcl["selected_layers"],
                    train_loader=_kcal_train_loader,
                    val_loader=_kcal_val_loader,
                    test_loader=_kcal_test_loader,
                    device=device,
                    target_dim=args.target_dimension,
                    seed=args.seed,
                    pooling_mode="max",
                )
                _kcal_stab_space = StabilitySpace(
                    _kcal_train_feats,
                    train_labels,
                    library="fast_separation",
                    metric=args.stab_metric,
                    whitening_components=args.whitening_components,
                    whitening_eps=args.whitening_eps,
                )
                _kcal_distance_source = "computed_kcal_lite_rgcl_independent"
                _kcal_selected_layers = list(rgcl["selected_layers"])
                # Cache for any subsequent consumers.
                _knn_blend_rgcl_stab_space = _kcal_stab_space
                _knn_blend_rgcl_train_features = _kcal_train_feats
                _knn_blend_rgcl_val_features = _kcal_val_features
                _knn_blend_rgcl_test_features = _kcal_test_features

            _kcal_method_name = "kcal_lite_rgcl"
            _kcal_feature_source = "rgcl"
        else:
            # single_layer: reuse always-available single-layer fusion space.
            _kcal_stab_space = fusion_out["stab_space"]
            _kcal_val_features = val_features
            _kcal_test_features = test_features
            _kcal_distance_source = "reused_single_layer_full_vector_fusion"
            _kcal_method_name = "kcal_lite"
            _kcal_feature_source = "single_layer"
            _kcal_selected_layers = [_single_layer_name]

        kcal = KCalLiteCalibrator(
            model=model_adapter,
            y_train=train_y,
            k_per_class=args.kcal_k_per_class,
            stability_space=_kcal_stab_space,
        )

        _kcal_val_knn_dm = _kcal_stab_space.calc_per_class_knn_distances(
            _kcal_val_features, args.kcal_k_per_class
        )
        _kcal_test_knn_dm = _kcal_stab_space.calc_per_class_knn_distances(
            _kcal_test_features, args.kcal_k_per_class
        )

        kcal.fit(
            X_val_embed=_kcal_val_features,
            y_val=val_y,
            model_probs=val_base_probs,
            val_knn_distances=_kcal_val_knn_dm,
        )
        kcal_test_probs = kcal.calibrate(
            X_test_embed=_kcal_test_features,
            model_probs=test_base_probs,
            test_knn_distances=_kcal_test_knn_dm,
        )
        _validate_probability_matrix(
            kcal_test_probs,
            n_rows=len(test_y),
            n_classes=num_classes,
            method_name=_kcal_method_name,
        )

        kcal_flip_meta = _argmax_flip_metadata(test_base_probs, kcal_test_probs, test_y)
        kcal_extra: Dict[str, Any] = {
            "selected_alpha": kcal.best_alpha,
            "selected_gamma": kcal.best_gamma,
            "alpha_grid": kcal.alpha_grid.tolist(),
            "gamma_grid": kcal.gamma_grid.tolist(),
            "k_per_class": args.kcal_k_per_class,
            "validation_nll_grid": kcal.validation_nll_grid.tolist(),
            "feature_source": _kcal_feature_source,
            "distance_source": _kcal_distance_source,
            "selected_layers": _kcal_selected_layers,
            "net_flips": int(
                kcal_flip_meta["flip_to_correct_count"]
                - kcal_flip_meta["flip_to_wrong_count"]
            ),
        }
        kcal_extra.update(kcal_flip_meta)
        kcal_extra.update(
            _nll_by_base_prediction_subset(test_base_probs, kcal_test_probs, test_y)
        )
        kcal_extra.update(_nll_by_final_prediction_subset(kcal_test_probs, test_y))

        methods.append(
            _make_method_entry(
                _kcal_method_name,
                kcal_test_probs,
                test_y,
                "decision_changing_full_vector",
                True,
                "alpha_gamma_grid_search",
                "validation",
                "validation_nll",
                kcal_extra,
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=base_probs_test,
                base_probs=test_base_probs,
            )
        )
        method_probs_by_name[_kcal_method_name] = kcal_test_probs
        method_can_change_argmax_map[_kcal_method_name] = True

    if args.enable_rgcl_neighbor_correction:
        _nc_use_rgcl = args.fusion_feature_source == "rgcl"

        if _nc_use_rgcl and _knn_blend_rgcl_stab_space is not None:
            _nc_stab_space = _knn_blend_rgcl_stab_space
            _nc_val_features = _knn_blend_rgcl_val_features
            _nc_test_features = _knn_blend_rgcl_test_features
            _nc_distance_source = "reused_rgcl_features"
            _nc_selected_layers = list(rgcl["selected_layers"])
        elif _nc_use_rgcl:
            # RGCL features not yet cached; extract independently.
            _nc_train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=args.batch_size,
                shuffle=False,
            )
            _nc_val_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_raw_aligned), torch.from_numpy(val_y)),
                batch_size=args.batch_size,
                shuffle=False,
            )
            _nc_test_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(test_raw),
                    torch.zeros(len(test_raw), dtype=torch.long),
                ),
                batch_size=args.batch_size,
                shuffle=False,
            )
            (
                _nc_train_feats,
                _nc_val_features,
                _nc_test_features,
                _,
            ) = extract_and_aggregate_sgc_features(
                model=model,
                layer_names=rgcl["selected_layers"],
                train_loader=_nc_train_loader,
                val_loader=_nc_val_loader,
                test_loader=_nc_test_loader,
                device=device,
                target_dim=args.target_dimension,
                seed=args.seed,
                pooling_mode="max",
            )
            _nc_stab_space = StabilitySpace(
                _nc_train_feats,
                train_labels,
                library="fast_separation",
                metric=args.stab_metric,
                whitening_components=args.whitening_components,
                whitening_eps=args.whitening_eps,
            )
            _nc_distance_source = "computed_rgcl_neighbor_correction_independent"
            _nc_selected_layers = list(rgcl["selected_layers"])
            _knn_blend_rgcl_stab_space = _nc_stab_space
            _knn_blend_rgcl_train_features = _nc_train_feats
            _knn_blend_rgcl_val_features = _nc_val_features
            _knn_blend_rgcl_test_features = _nc_test_features
        else:
            _nc_stab_space = fusion_out["stab_space"]
            _nc_val_features = val_features
            _nc_test_features = test_features
            _nc_distance_source = "reused_single_layer_full_vector_fusion"
            _nc_selected_layers = [_single_layer_name]

        logger.info("Computing per-class 1-NN distances for neighbor correction...")
        _nc_val_dm = _nc_stab_space.calc_per_class_1nn_distances(_nc_val_features)
        _nc_test_dm = _nc_stab_space.calc_per_class_1nn_distances(_nc_test_features)

        val_base_pred_nc = np.argmax(val_base_probs, axis=1)
        test_base_pred_nc = np.argmax(test_base_probs, axis=1)

        val_alt_pred, val_trust, val_separation = _neighbor_correction_geometry(
            _nc_val_dm, val_base_pred_nc
        )
        test_alt_pred, test_trust, test_separation = _neighbor_correction_geometry(
            _nc_test_dm, test_base_pred_nc
        )

        _nc_common_extra = {
            "feature_source": "rgcl" if _nc_use_rgcl else "single_layer",
            "distance_source": _nc_distance_source,
            "selected_layers": _nc_selected_layers,
        }

        for _nc_score_name, _nc_val_score, _nc_test_score, _nc_upper in [
            ("trust", val_trust, test_trust, 1.0),
            ("separation", val_separation, test_separation, 0.0),
        ]:
            _nc_method_name = f"rgcl_neighbor_correction_{_nc_score_name}"
            _nc_val_result = _select_neighbor_threshold(
                _nc_val_score, _nc_upper, val_base_pred_nc, val_alt_pred, val_y
            )
            _nc_threshold = _nc_val_result["threshold"]
            _nc_switched_test = _nc_test_score < _nc_threshold
            _nc_test_probs = _neighbor_correction_probs(
                test_base_probs, _nc_switched_test, test_base_pred_nc, test_alt_pred
            )
            _validate_probability_matrix(
                _nc_test_probs, n_rows=len(test_y), n_classes=num_classes,
                method_name=_nc_method_name,
            )

            _nc_test_fixes = int(np.sum(
                _nc_switched_test & (test_base_pred_nc != test_y) & (test_alt_pred == test_y)
            ))
            _nc_test_breaks = int(np.sum(
                _nc_switched_test & (test_base_pred_nc == test_y) & (test_alt_pred != test_y)
            ))
            _nc_test_wrong_wrong = int(np.sum(
                _nc_switched_test & (test_base_pred_nc != test_y) & (test_alt_pred != test_y)
            ))

            _nc_extra: Dict[str, Any] = {
                "score_type": _nc_score_name,
                "val_selected_threshold": _nc_val_result["threshold"],
                "val_upper_boundary": _nc_val_result["upper_boundary"],
                "val_eligible_count": _nc_val_result["eligible_count_at_default_boundary"],
                "val_switch_count": _nc_val_result["val_switch_count"],
                "val_flip_correct": _nc_val_result["val_flip_correct"],
                "val_flip_wrong": _nc_val_result["val_flip_wrong"],
                "val_wrong_to_wrong": _nc_val_result["val_wrong_to_wrong"],
                "val_net_gain": _nc_val_result["val_net_gain"],
                "val_base_accuracy": _nc_val_result["val_base_accuracy"],
                "val_thresholded_accuracy": _nc_val_result["val_thresholded_accuracy"],
                "test_switch_count": int(np.sum(_nc_switched_test)),
                "test_flip_correct": _nc_test_fixes,
                "test_flip_wrong": _nc_test_breaks,
                "test_wrong_to_wrong": _nc_test_wrong_wrong,
                "test_net_gain": _nc_test_fixes - _nc_test_breaks,
            }
            _nc_extra.update(_nc_common_extra)
            _nc_extra.update(_argmax_flip_metadata(test_base_probs, _nc_test_probs, test_y))
            _nc_extra.update(
                _nll_by_base_prediction_subset(test_base_probs, _nc_test_probs, test_y)
            )
            _nc_extra.update(_nll_by_final_prediction_subset(_nc_test_probs, test_y))

            methods.append(
                _make_method_entry(
                    _nc_method_name,
                    _nc_test_probs,
                    test_y,
                    "decision_changing_full_vector",
                    True,
                    "lower_tail_threshold_sweep",
                    "validation",
                    "validation_accuracy",
                    _nc_extra,
                    anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                    base_probs_for_anchor_bins=base_probs_test,
                    base_probs=test_base_probs,
                )
            )
            method_probs_by_name[_nc_method_name] = _nc_test_probs
            method_can_change_argmax_map[_nc_method_name] = True
            logger.info(
                "neighbor_correction_%s: val_threshold=%.4f val_gain=%d "
                "test_switches=%d test_fixes=%d test_breaks=%d",
                _nc_score_name,
                _nc_threshold,
                _nc_val_result["val_net_gain"],
                int(np.sum(_nc_switched_test)),
                _nc_test_fixes,
                _nc_test_breaks,
            )

    if optional_full_vector_methods:
        del model, model_adapter, train_features, train_y, val_features, test_features, fusion_out
        _cleanup_memory()

    _kcal_full_params: Dict[str, Any] | None = None
    if _kcal_features_required:
        kcal_method_name = "kcal"
        # Fit-once/evaluate-many: when --fitted_state_dir is given, KCal's
        # own existing load-checkpoint-if-present mechanism is redirected to
        # read/write there instead of this run's own (per-corruption-cell)
        # output_dir, so every corruption cell for a checkpoint reuses the
        # exact same fitted projection+bandwidth+KDE bank. On a
        # --corruption_type run with no checkpoint found there, refuse to
        # fit fresh rather than silently doing so (even though KCal's fit
        # inputs -- train/val -- are always clean, a fresh fit could still
        # differ in fitted parameters across cells due to SGD/GPU
        # non-determinism, violating "freeze once, reuse exactly").
        if args.fitted_state_dir:
            kcal_checkpoint_path = os.path.join(args.fitted_state_dir, "kcal_calibrator.pt")
            if args.corruption_type is not None and not os.path.exists(kcal_checkpoint_path):
                raise RuntimeError(
                    f"kcal: --corruption_type is set but no fitted checkpoint found at "
                    f"{kcal_checkpoint_path!r} -- refusing to fit fresh on a corruption run."
                )
        else:
            kcal_checkpoint_path = os.path.join(
                _stage_dir(args.output_dir, "kcal"), "calibrator.pt"
            )
        if _is_method_checkpoint_complete(
            args.output_dir, kcal_method_name, len(test_y), num_classes
        ):
            logger.info("Skipping completed method %s", kcal_method_name)
            kcal_entry = _load_method_entry(args.output_dir, kcal_method_name)
            _kcal_full_params = {
                "bandwidth": float(kcal_entry["selected_bandwidth"]),
                "projection_dim": int(kcal_entry["projection_dim"]),
                "projection_type": kcal_entry["projection_type"],
                "bandwidth_folds_used": int(kcal_entry["bandwidth_folds_used"]),
            }
        else:
            logger.info("Starting full KCal learned projection and KDE calibration")
            kcal_train_embed = np.load(
                stage3_files["kcal_train_penultimate"], mmap_mode="r"
            )
            kcal_train_y = np.load(stage3_files["train_y"])
            kcal_val_embed = np.load(
                stage3_files["kcal_val_penultimate"], mmap_mode="r"
            )
            kcal_test_embed = np.load(
                stage3_files["kcal_test_penultimate"], mmap_mode="r"
            )
            if os.path.exists(kcal_checkpoint_path):
                logger.info("Loading fitted full KCal checkpoint: %s", kcal_checkpoint_path)
                kcal_full = KCalCalibrator.load_checkpoint(
                    kcal_checkpoint_path, device=device
                )
            else:
                kcal_full = KCalCalibrator(
                    projection_type=args.kcal_projection,
                    projection_dim=args.kcal_projection_dim,
                    projection_epochs=args.kcal_projection_epochs,
                    query_batch_size=args.batch_size,
                    references_per_class=args.kcal_references_per_class,
                    bandwidth_folds=args.kcal_bandwidth_folds,
                    prediction_batch_size=args.batch_size,
                    reference_batch_size=max(1024, args.batch_size * 8),
                    seed=args.seed,
                    device=device,
                )
                kcal_full.fit(
                    X_train_embed=kcal_train_embed,
                    y_train=kcal_train_y,
                    X_cal_embed=kcal_val_embed,
                    y_cal=val_y,
                )
                kcal_full.save_checkpoint(kcal_checkpoint_path)
                logger.info("Saved fitted full KCal checkpoint: %s", kcal_checkpoint_path)

            kcal_full_probs = kcal_full.calibrate(kcal_test_embed)
            _validate_probability_matrix(
                kcal_full_probs,
                n_rows=len(test_y),
                n_classes=num_classes,
                method_name=kcal_method_name,
            )
            kcal_full_params = kcal_full.get_params()
            _kcal_full_params = {
                "bandwidth": float(kcal_full_params["bandwidth"]),
                "projection_dim": int(kcal_full_params["output_dim"]),
                "projection_type": kcal_full_params["projection_type"],
                "bandwidth_folds_used": int(
                    kcal_full_params["bandwidth_folds_used"]
                ),
            }
            kcal_full_extra: Dict[str, Any] = {
                "selected_bandwidth": _kcal_full_params["bandwidth"],
                "projection_type": _kcal_full_params["projection_type"],
                "projection_dim": _kcal_full_params["projection_dim"],
                "projection_input_dim": int(kcal_full_params["input_dim"]),
                "projection_epochs": int(kcal_full_params["projection_epochs"]),
                "references_per_class": int(
                    kcal_full_params["references_per_class"]
                ),
                "bandwidth_folds_requested": int(
                    kcal_full_params["bandwidth_folds"]
                ),
                "bandwidth_folds_used": _kcal_full_params["bandwidth_folds_used"],
                "bandwidth_search_history": kcal_full_params[
                    "bandwidth_search_history"
                ],
                "projection_loss_history": kcal_full_params[
                    "projection_loss_history"
                ],
                "calibration_reference_count": int(
                    kcal_full_params["calibration_reference_count"]
                ),
                "projection_training_split": "train",
                "kde_reference_split": "validation",
                "feature_source": "penultimate_classifier_input",
                "uses_all_calibration_references": True,
                "blends_model_probabilities": False,
            }
            kcal_full_extra.update(
                _argmax_flip_metadata(test_base_probs, kcal_full_probs, test_y)
            )
            kcal_full_extra["net_flips"] = int(
                kcal_full_extra["flip_to_correct_count"]
                - kcal_full_extra["flip_to_wrong_count"]
            )
            kcal_full_extra.update(
                _nll_by_base_prediction_subset(
                    test_base_probs, kcal_full_probs, test_y
                )
            )
            kcal_full_extra.update(
                _nll_by_final_prediction_subset(kcal_full_probs, test_y)
            )
            methods.append(
                _make_method_entry(
                    kcal_method_name,
                    kcal_full_probs,
                    test_y,
                    "kernel_density_full_vector",
                    True,
                    "learned_projection_then_bandwidth_cv",
                    "train_and_validation",
                    "projection_loo_nll_and_validation_cv_nll",
                    kcal_full_extra,
                    anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                    base_probs_for_anchor_bins=base_probs_test,
                    base_probs=test_base_probs,
                )
            )
            method_probs_by_name[kcal_method_name] = kcal_full_probs
            del kcal_train_embed, kcal_train_y, kcal_val_embed, kcal_test_embed
            del kcal_full, kcal_full_params
            _cleanup_memory()
        method_can_change_argmax_map[kcal_method_name] = True

    if args.enable_kcal_factorial:
        logger.info(
            "Starting controlled KCal factorial study (scope=%s)",
            args.kcal_factorial_scope,
        )
        pi_cache_files = [
            _kcal_factorial_paths["pi_train"],
            _kcal_factorial_paths["pi_validation"],
            _kcal_factorial_paths["pi_test"],
            _kcal_factorial_paths["pi_metadata"],
        ]
        if all(os.path.exists(path) for path in pi_cache_files):
            logger.info("Reusing cached KCal-Pi embeddings for factorial study")
        else:
            if not os.path.exists(kcal_checkpoint_path):
                raise RuntimeError(
                    "KCal factorial study requires the fitted full-KCal checkpoint"
                )
            logger.info("Projecting and caching train/validation/test embeddings with KCal-Pi")
            _factorial_kcal = KCalCalibrator.load_checkpoint(
                kcal_checkpoint_path, device=device
            )
            _factorial_penultimate_train = np.load(
                stage3_files["kcal_train_penultimate"], mmap_mode="r"
            )
            _factorial_penultimate_validation = np.load(
                stage3_files["kcal_val_penultimate"], mmap_mode="r"
            )
            _factorial_penultimate_test = np.load(
                stage3_files["kcal_test_penultimate"], mmap_mode="r"
            )
            _factorial_pi_train = _factorial_kcal.transform(
                _factorial_penultimate_train
            )
            # This is exactly the fitted calibration representation and avoids
            # a redundant projection pass.
            _factorial_pi_validation = np.asarray(
                _factorial_kcal.calibration_projected, dtype=np.float32
            )
            _factorial_pi_test = _factorial_kcal.transform(
                _factorial_penultimate_test
            )
            _save_npy_atomic(
                _kcal_factorial_paths["pi_train"], _factorial_pi_train
            )
            _save_npy_atomic(
                _kcal_factorial_paths["pi_validation"],
                _factorial_pi_validation,
            )
            _save_npy_atomic(
                _kcal_factorial_paths["pi_test"], _factorial_pi_test
            )
            _atomic_write_json(
                _kcal_factorial_paths["pi_metadata"],
                {
                    "feature_source": "learned_kcal_pi",
                    "projection_type": _factorial_kcal.config.projection_type,
                    "input_dimension": int(_factorial_kcal.input_dim),
                    "output_dimension": int(_factorial_kcal.output_dim),
                    "projection_checkpoint": kcal_checkpoint_path,
                    "seed": int(args.seed),
                },
            )
            del (
                _factorial_kcal,
                _factorial_penultimate_train,
                _factorial_penultimate_validation,
                _factorial_penultimate_test,
                _factorial_pi_train,
                _factorial_pi_validation,
                _factorial_pi_test,
            )
            _cleanup_memory()

        _factorial_feature_sets = {
            "pi": {
                "train": np.load(
                    _kcal_factorial_paths["pi_train"], mmap_mode="r"
                ),
                "validation": np.load(
                    _kcal_factorial_paths["pi_validation"], mmap_mode="r"
                ),
                "test": np.load(
                    _kcal_factorial_paths["pi_test"], mmap_mode="r"
                ),
                "metadata": _load_json(_kcal_factorial_paths["pi_metadata"]),
            },
            "rgcl": {
                "train": np.load(
                    _kcal_factorial_paths["rgcl_train"], mmap_mode="r"
                ),
                "validation": np.load(
                    _kcal_factorial_paths["rgcl_validation"], mmap_mode="r"
                ),
                "test": np.load(
                    _kcal_factorial_paths["rgcl_test"], mmap_mode="r"
                ),
                "metadata": _load_json(
                    _kcal_factorial_paths["rgcl_metadata"]
                ),
            },
        }
        _factorial_train_labels = np.load(stage3_files["train_y"])

        if args.kcal_factorial_scope == "core":
            _factorial_geometry_axes = [("full", "validation", "rbf_sq")]
        else:
            _factorial_geometry_axes = list(
                product(
                    ["full", "topk"],
                    ["validation", "train"],
                    ["rbf_sq", "exp_l2"],
                )
            )

        for _factorial_representation in ["pi", "rgcl"]:
            _factorial_features = _factorial_feature_sets[
                _factorial_representation
            ]
            for (
                _factorial_estimator,
                _factorial_bank,
                _factorial_kernel,
            ) in _factorial_geometry_axes:
                _factorial_names = {
                    fusion_name: _kcal_factorial_method_name(
                        _factorial_representation,
                        fusion_name,
                        _factorial_estimator,
                        _factorial_bank,
                        _factorial_kernel,
                    )
                    for fusion_name in _kcal_factorial_fusions
                }
                if all(
                    _is_method_checkpoint_complete(
                        args.output_dir,
                        method_name,
                        len(test_y),
                        num_classes,
                    )
                    for method_name in _factorial_names.values()
                ):
                    logger.info(
                        "Skipping completed factorial geometry cell: %s/%s/%s/%s",
                        _factorial_representation,
                        _factorial_estimator,
                        _factorial_bank,
                        _factorial_kernel,
                    )
                    for method_name in _factorial_names.values():
                        method_can_change_argmax_map[method_name] = True
                        completed_entry = _load_method_entry(
                            args.output_dir, method_name
                        )
                        _kcal_factorial_selected_params[method_name] = {
                            "strength": float(completed_entry["selected_strength"]),
                            "alpha": float(completed_entry["selected_alpha"]),
                        }
                        _kcal_factorial_method_artifacts[method_name] = {
                            "distance_array_key": None,
                            "formula": "controlled_kcal_factorial",
                            "factorial_axes": completed_entry["factorial_axes"],
                            "feature_cache": completed_entry["feature_cache"],
                        }
                    continue

                logger.info(
                    "Computing factorial geometry cell: repr=%s estimator=%s bank=%s kernel=%s",
                    _factorial_representation,
                    _factorial_estimator,
                    _factorial_bank,
                    _factorial_kernel,
                )
                _factorial_grid = build_kcal_factorial_posterior_grid(
                    train_features=_factorial_features["train"],
                    train_labels=_factorial_train_labels,
                    validation_features=_factorial_features["validation"],
                    validation_labels=val_y,
                    test_features=_factorial_features["test"],
                    estimator=_factorial_estimator,
                    reference_bank=_factorial_bank,
                    kernel=_factorial_kernel,
                    strengths=_kcal_factorial_strength_grid,
                    k_per_class=args.kcal_factorial_k_per_class,
                    cv_folds=args.kcal_factorial_cv_folds,
                    seed=args.seed,
                    device=device,
                    query_batch_size=args.batch_size,
                    reference_batch_size=args.kcal_factorial_reference_batch_size,
                )
                _factorial_outputs = select_kcal_factorial_outputs(
                    _factorial_grid,
                    validation_labels=val_y,
                    validation_base_probs=val_base_probs,
                    test_base_probs=test_base_probs,
                    alpha_grid=_kcal_factorial_alpha_grid,
                )

                for _factorial_fusion in _kcal_factorial_fusions:
                    _factorial_method_name = _factorial_names[_factorial_fusion]
                    if _is_method_checkpoint_complete(
                        args.output_dir,
                        _factorial_method_name,
                        len(test_y),
                        num_classes,
                    ):
                        logger.info(
                            "Skipping completed method %s", _factorial_method_name
                        )
                        completed_entry = _load_method_entry(
                            args.output_dir, _factorial_method_name
                        )
                        method_can_change_argmax_map[_factorial_method_name] = True
                        _kcal_factorial_selected_params[
                            _factorial_method_name
                        ] = {
                            "strength": float(
                                completed_entry["selected_strength"]
                            ),
                            "alpha": float(completed_entry["selected_alpha"]),
                        }
                        _kcal_factorial_method_artifacts[
                            _factorial_method_name
                        ] = {
                            "distance_array_key": None,
                            "formula": "controlled_kcal_factorial",
                            "factorial_axes": completed_entry["factorial_axes"],
                            "feature_cache": completed_entry["feature_cache"],
                            "geometry_posterior_sha256": completed_entry[
                                "geometry_posterior_sha256"
                            ],
                        }
                        continue
                    _factorial_selected = _factorial_outputs[_factorial_fusion]
                    _factorial_probs = np.asarray(
                        _factorial_selected.test_probs, dtype=np.float64
                    )
                    _validate_probability_matrix(
                        _factorial_probs,
                        n_rows=len(test_y),
                        n_classes=num_classes,
                        method_name=_factorial_method_name,
                    )
                    _factorial_geometry_probs = np.asarray(
                        _factorial_grid.test_probs[
                            _factorial_selected.selected_strength_index
                        ],
                        dtype=np.float32,
                    )
                    _factorial_geometry_hash = hashlib.sha256(
                        _factorial_geometry_probs.tobytes()
                    ).hexdigest()
                    _factorial_axes = {
                        "representation": _factorial_representation,
                        "fusion": _factorial_fusion,
                        "estimator": _factorial_estimator,
                        "reference_bank": _factorial_bank,
                        "kernel": _factorial_kernel,
                    }
                    _factorial_feature_cache = {
                        "train": _kcal_factorial_paths[
                            f"{_factorial_representation}_train"
                        ],
                        "validation": _kcal_factorial_paths[
                            f"{_factorial_representation}_validation"
                        ],
                        "test": _kcal_factorial_paths[
                            f"{_factorial_representation}_test"
                        ],
                    }
                    _factorial_flip_meta = _argmax_flip_metadata(
                        test_base_probs, _factorial_probs, test_y
                    )
                    _factorial_extra: Dict[str, Any] = {
                        "controlled_factorial": True,
                        "factorial_scope": args.kcal_factorial_scope,
                        "factorial_axes": _factorial_axes,
                        "feature_source": _factorial_representation,
                        "feature_metadata": _factorial_features["metadata"],
                        "feature_cache": _factorial_feature_cache,
                        "selected_strength": float(
                            _factorial_selected.selected_strength
                        ),
                        "selected_strength_at_grid_edge": bool(
                            _factorial_selected.selected_strength_index
                            in {0, len(_factorial_grid.strengths) - 1}
                        ),
                        "strength_grid": _factorial_grid.strengths.tolist(),
                        "distance_normalizer": float(
                            _factorial_grid.distance_normalizer
                        ),
                        "distance_normalization": "median_reference_bank_distance",
                        "selected_alpha": float(
                            _factorial_selected.selected_alpha
                        ),
                        "selected_alpha_at_grid_edge": bool(
                            _factorial_selected.selected_alpha
                            in {
                                float(_kcal_factorial_alpha_grid.min()),
                                float(_kcal_factorial_alpha_grid.max()),
                            }
                        ),
                        "alpha_grid": _kcal_factorial_alpha_grid.tolist(),
                        "validation_nll": float(
                            _factorial_selected.validation_nll
                        ),
                        "validation_nll_grid": (
                            _factorial_selected.validation_nll_grid.tolist()
                        ),
                        "geometry_strength_source": (
                            _factorial_selected.geometry_strength_source
                        ),
                        "geometry_posterior_sha256": _factorial_geometry_hash,
                        "same_geometry_as_replacement": bool(
                            _factorial_fusion in {"replace", "blend_frozen"}
                        ),
                        "reference_count": int(_factorial_grid.reference_count),
                        "crossfit_folds_used": int(_factorial_grid.folds_used),
                        "k_per_class": _factorial_grid.k_per_class,
                        "class_prior_handling": (
                            "empirical_reference_counts"
                            if _factorial_estimator == "full"
                            else "equal_per_class_topk"
                        ),
                        "blends_model_probabilities": bool(
                            _factorial_fusion != "replace"
                        ),
                        "net_flips": int(
                            _factorial_flip_meta["flip_to_correct_count"]
                            - _factorial_flip_meta["flip_to_wrong_count"]
                        ),
                    }
                    _factorial_extra.update(
                        kcal_factorial_effective_kernel_parameter(
                            kernel=_factorial_kernel,
                            selected_strength=_factorial_selected.selected_strength,
                            distance_normalizer=_factorial_grid.distance_normalizer,
                        )
                    )
                    _factorial_extra.update(_factorial_flip_meta)
                    _factorial_extra.update(
                        _nll_by_base_prediction_subset(
                            test_base_probs, _factorial_probs, test_y
                        )
                    )
                    _factorial_extra.update(
                        _nll_by_final_prediction_subset(
                            _factorial_probs, test_y
                        )
                    )
                    _factorial_entry = _make_method_entry(
                        _factorial_method_name,
                        _factorial_probs,
                        test_y,
                        "controlled_kernel_density_factorial",
                        True,
                        (
                            "replacement_strength_then_alpha_grid"
                            if _factorial_fusion == "blend_frozen"
                            else "joint_strength_alpha_grid"
                            if _factorial_fusion == "blend_joint"
                            else "kernel_strength_grid"
                        ),
                        "validation_crossfit"
                        if _factorial_bank == "validation"
                        else "validation",
                        "validation_nll",
                        _factorial_extra,
                        anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                        base_probs_for_anchor_bins=base_probs_test,
                        base_probs=test_base_probs,
                    )
                    _save_method_checkpoint(
                        args.output_dir,
                        _factorial_method_name,
                        _factorial_entry,
                        _factorial_probs,
                        True,
                        stage_name="stage4_kcal_factorial",
                    )
                    method_can_change_argmax_map[_factorial_method_name] = True
                    _kcal_factorial_selected_params[_factorial_method_name] = {
                        "strength": float(
                            _factorial_selected.selected_strength
                        ),
                        "alpha": float(_factorial_selected.selected_alpha),
                    }
                    _kcal_factorial_method_artifacts[_factorial_method_name] = {
                        "distance_array_key": None,
                        "formula": "controlled_kcal_factorial",
                        "factorial_axes": _factorial_axes,
                        "feature_cache": _factorial_feature_cache,
                        "geometry_posterior_sha256": _factorial_geometry_hash,
                    }
                    logger.info("Saved factorial method %s", _factorial_method_name)

                del _factorial_grid, _factorial_outputs
                _cleanup_memory()

        del _factorial_feature_sets, _factorial_train_labels
        _cleanup_memory()

    # ------------------------------------------------------------------ #
    # Stage B distance matrix resolver                                   #
    # Trust Score, AAR-lightweight, and GLAD-PI all use per-class 1-NN  #
    # distances. Resolve the correct matrices based on feature_source.   #
    # ------------------------------------------------------------------ #
    _sb_val_dm: np.ndarray = val_distance_matrix    # single-layer baseline
    _sb_test_dm: np.ndarray = test_distance_matrix  # single-layer baseline
    _sb_feature_source: str = "single_layer"

    _stage_b_enabled = (
        args.enable_trust_score_baseline
        or args.enable_aar_lightweight
        or args.enable_glad_pi
        or args.enable_contrastive_beta_sweep
    )
    if _stage_b_enabled and args.fusion_feature_source == "rgcl":
        if _knn_blend_rgcl_stab_space is not None:
            logger.info("Computing RGCL per-class 1-NN distances for Stage B methods")
            _sb_val_dm = _knn_blend_rgcl_stab_space.calc_per_class_1nn_distances(
                np.asarray(_knn_blend_rgcl_val_features, dtype=np.float32)
            )
            _sb_test_dm = _knn_blend_rgcl_stab_space.calc_per_class_1nn_distances(
                np.asarray(_knn_blend_rgcl_test_features, dtype=np.float32)
            )
            _sb_feature_source = "rgcl"
        else:
            logger.warning(
                "--fusion_feature_source rgcl requested for Stage B methods but RGCL stab_space is "
                "not available. Enable at least one of --enable_full_vector_geometric_fusion, "
                "--enable_knn_blend_baseline, --enable_kcal_lite_baseline, or "
                "--enable_rgcl_neighbor_correction to pre-compute RGCL features. "
                "Falling back to single-layer distances for Stage B methods."
            )
            _sb_feature_source = "single_layer_fallback_rgcl_unavailable"

    # ------------------------------------------------------------------ #
    # Trust Score baseline (diagnostic + switch, alpha=0)               #
    # Alpha=0 means no density filtering (Phase 1 only).                #
    # trust ratio = d_other / d_pred; threshold swept on validation.    #
    # NOTE: When using the same feature space as --enable_rgcl_neighbor_ #
    # correction, trust_score_original_switch is equivalent to          #
    # rgcl_neighbor_correction_trust (same computation, same threshold). #
    # ------------------------------------------------------------------ #
    if args.enable_trust_score_baseline:
        _ts_val_base_pred = np.argmax(val_base_probs, axis=1).astype(np.int64)
        _ts_test_base_pred = np.argmax(test_base_probs, axis=1).astype(np.int64)

        _ts_val_alt_pred, _ts_val_trust, _ = _neighbor_correction_geometry(
            _sb_val_dm, _ts_val_base_pred
        )
        _ts_test_alt_pred, _ts_test_trust, _ = _neighbor_correction_geometry(
            _sb_test_dm, _ts_test_base_pred
        )

        _ts_equiv_note = (
            "Alpha=0 Trust Score ratio (d_other/d_pred). "
            f"Feature source: {_sb_feature_source}. "
            "When using the same feature space as rgcl_neighbor_correction_trust, "
            "predictions are identical by construction."
        )

        # Diagnostic variant: no prob change
        _ts_diag_name = "trust_score_original_diagnostic"
        if not _is_method_checkpoint_complete(args.output_dir, _ts_diag_name, len(test_y), num_classes):
            _ts_diag_extra: Dict[str, Any] = {
                "mean_trust_score": float(np.mean(_ts_test_trust)),
                "median_trust_score": float(np.median(_ts_test_trust)),
                "fraction_below_1": float(np.mean(_ts_test_trust < 1.0)),
                "feature_source": _sb_feature_source,
                "alpha": 0.0,
                "equivalence_note": _ts_equiv_note,
            }
            _ts_diag_extra.update(_argmax_flip_metadata(test_base_probs, test_base_probs, test_y))
            _ts_diag_entry = _make_method_entry(
                _ts_diag_name,
                test_base_probs,
                test_y,
                "trust_score",
                False,
                "trust_score_alpha0",
                "validation",
                "validation_accuracy",
                _ts_diag_extra,
                anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                base_probs_for_anchor_bins=base_probs_test,
                base_probs=test_base_probs,
            )
            methods.append(_ts_diag_entry)
            method_probs_by_name[_ts_diag_name] = test_base_probs
            method_can_change_argmax_map[_ts_diag_name] = False
            logger.info(
                "trust_score_original_diagnostic: mean_trust=%.4f fraction_below_1=%.4f",
                float(np.mean(_ts_test_trust)),
                float(np.mean(_ts_test_trust < 1.0)),
            )
        else:
            logger.info("Skipping completed method %s", _ts_diag_name)

        # Switch variant: threshold sweep on val
        _ts_switch_name = "trust_score_original_switch"
        if not _is_method_checkpoint_complete(args.output_dir, _ts_switch_name, len(test_y), num_classes):
            _ts_val_result = _fit_or_load_value(
                args, "trust_score_original_switch",
                lambda: _select_neighbor_threshold(
                    _ts_val_trust, 1.0, _ts_val_base_pred, _ts_val_alt_pred, val_y
                ),
            )
            _ts_threshold = _ts_val_result["threshold"]
            _ts_switched_test = _ts_test_trust < _ts_threshold
            _ts_switch_probs = _neighbor_correction_probs(
                test_base_probs, _ts_switched_test, _ts_test_base_pred, _ts_test_alt_pred
            )
            _validate_probability_matrix(
                _ts_switch_probs, n_rows=len(test_y), n_classes=num_classes,
                method_name=_ts_switch_name,
            )
            _ts_test_fixes = int(np.sum(
                _ts_switched_test & (_ts_test_base_pred != test_y) & (_ts_test_alt_pred == test_y)
            ))
            _ts_test_breaks = int(np.sum(
                _ts_switched_test & (_ts_test_base_pred == test_y) & (_ts_test_alt_pred != test_y)
            ))
            _ts_switch_extra: Dict[str, Any] = {
                "selected_threshold": _ts_threshold,
                "val_net_gain": _ts_val_result["val_net_gain"],
                "val_base_accuracy": _ts_val_result["val_base_accuracy"],
                "val_thresholded_accuracy": _ts_val_result["val_thresholded_accuracy"],
                "test_switch_count": int(np.sum(_ts_switched_test)),
                "test_flip_correct": _ts_test_fixes,
                "test_flip_wrong": _ts_test_breaks,
                "test_net_gain": _ts_test_fixes - _ts_test_breaks,
                "feature_source": _sb_feature_source,
                "alpha": 0.0,
                "equivalence_note": _ts_equiv_note,
            }
            _ts_switch_extra.update(_argmax_flip_metadata(test_base_probs, _ts_switch_probs, test_y))
            _ts_switch_extra.update(_nll_by_base_prediction_subset(test_base_probs, _ts_switch_probs, test_y))
            _ts_switch_extra.update(_nll_by_final_prediction_subset(_ts_switch_probs, test_y))
            methods.append(
                _make_method_entry(
                    _ts_switch_name,
                    _ts_switch_probs,
                    test_y,
                    "trust_score",
                    True,
                    "trust_score_threshold_sweep",
                    "validation",
                    "validation_accuracy",
                    _ts_switch_extra,
                    anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                    base_probs_for_anchor_bins=base_probs_test,
                    base_probs=test_base_probs,
                )
            )
            method_probs_by_name[_ts_switch_name] = _ts_switch_probs
            method_can_change_argmax_map[_ts_switch_name] = True
            logger.info(
                "trust_score_original_switch: threshold=%.4f val_net_gain=%d "
                "test_switches=%d test_fixes=%d test_breaks=%d",
                _ts_threshold,
                _ts_val_result["val_net_gain"],
                int(np.sum(_ts_switched_test)),
                _ts_test_fixes,
                _ts_test_breaks,
            )
            del _ts_switch_probs
        else:
            logger.info("Skipping completed method %s", _ts_switch_name)

    # ------------------------------------------------------------------ #
    # AAR-Lightweight baseline                                            #
    # Uses per-class 1-NN distances (_sb_val_dm/_sb_test_dm) and         #
    # stage 2 logits.                                                     #
    # ------------------------------------------------------------------ #
    if args.enable_aar_lightweight:
        _aar_name = "aar_lightweight"
        if _is_method_checkpoint_complete(args.output_dir, _aar_name, len(test_y), num_classes):
            logger.info("Skipping completed method %s", _aar_name)
        else:
            from Calibrators.aar_calibration import AARLightweightCalibrator, METADATA as _AAR_METADATA

            _aar_logits_val = np.load(stage2_files["logits_val"], mmap_mode="r")
            _aar_logits_test = np.load(stage2_files["logits_test"], mmap_mode="r")

            # Use inner fit split for fitting
            _aar_fit_idx, _aar_select_idx = make_inner_validation_split(
                val_y, select_fraction=args.inner_val_fraction, seed=args.inner_val_seed
            )
            _aar = AARLightweightCalibrator()
            _aar_fit_result = _aar.fit(
                np.asarray(_aar_logits_val)[_aar_fit_idx],
                val_y[_aar_fit_idx],
                _sb_val_dm[_aar_fit_idx],
            )
            _aar_probs_test = _aar.calibrate(np.asarray(_aar_logits_test), _sb_test_dm)
            _validate_probability_matrix(
                _aar_probs_test, n_rows=len(test_y), n_classes=num_classes, method_name=_aar_name,
            )
            _aar_extra: Dict[str, Any] = {
                "fit_nll": _aar_fit_result.get("fit_nll"),
                "w0": _aar_fit_result.get("w0"),
                "w1": _aar_fit_result.get("w1"),
                "feature_source": _sb_feature_source,
                "inner_val_fraction": args.inner_val_fraction,
                "inner_val_seed": args.inner_val_seed,
                "inner_val_fit_n": int(len(_aar_fit_idx)),
                **_AAR_METADATA,
            }
            _aar_extra.update(_argmax_flip_metadata(test_base_probs, _aar_probs_test, test_y))
            _aar_extra.update(_nll_by_base_prediction_subset(test_base_probs, _aar_probs_test, test_y))
            _aar_extra.update(_nll_by_final_prediction_subset(_aar_probs_test, test_y))
            methods.append(
                _make_method_entry(
                    _aar_name,
                    _aar_probs_test,
                    test_y,
                    "atypicality_reweighted",
                    True,
                    "lbfgsb_nll",
                    "inner_val_fit",
                    "validation_nll",
                    _aar_extra,
                    anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                    base_probs_for_anchor_bins=base_probs_test,
                    base_probs=test_base_probs,
                )
            )
            method_probs_by_name[_aar_name] = _aar_probs_test
            method_can_change_argmax_map[_aar_name] = True
            logger.info(
                "aar_lightweight: fit_nll=%.4f w0=%.4f w1=%.4f",
                _aar_fit_result.get("fit_nll", float("nan")),
                _aar_fit_result.get("w0", float("nan")),
                _aar_fit_result.get("w1", float("nan")),
            )
            del _aar, _aar_probs_test, _aar_logits_val, _aar_logits_test
            _cleanup_memory()
        _write_incremental_summary(args.output_dir)

    # ------------------------------------------------------------------ #
    # GLAD-PI baseline                                                    #
    # ------------------------------------------------------------------ #
    if args.enable_glad_pi:
        _glad_name = "glad_pi"
        if _is_method_checkpoint_complete(args.output_dir, _glad_name, len(test_y), num_classes):
            logger.info("Skipping completed method %s", _glad_name)
        else:
            from Calibrators.glad_pi import GLADPICalibrator

            # Use stage3 logits (extraction-aligned) so rows match val_y and val_distance_matrix.
            # stage2_files["logits_val"] is in stage-1 DataLoader order, which differs from the
            # stage-3 extraction order due to multi-worker pass-ordering drift (4954/5000 rows
            # differ on CIFAR-100 ResNet18 seed21). Using stage-2 logits with stage-3 indices
            # causes a total row misalignment that collapses select accuracy to ~chance.
            _glad_logits_val = np.asarray(np.load(stage3_files["val_logits"], mmap_mode="r"))
            _glad_logits_test = np.asarray(np.load(stage3_files["test_logits"], mmap_mode="r"))

            # Parse beta grid
            try:
                _glad_beta_grid = [
                    float(x) for x in args.glad_pi_beta_grid.split(",") if x.strip()
                ]
            except ValueError as _ge:
                raise ValueError(f"--glad_pi_beta_grid must contain numeric values: {_ge}") from _ge
            if not _glad_beta_grid:
                raise ValueError("--glad_pi_beta_grid must contain at least one value")

            _glad_fit_idx, _glad_select_idx = make_inner_validation_split(
                val_y, select_fraction=args.inner_val_fraction, seed=args.inner_val_seed
            )

            # Diagnostic: confirm distance matrix scale and logit/label alignment
            _diag_dm_fit = _sb_val_dm[_glad_fit_idx]
            _diag_dm_sel = _sb_val_dm[_glad_select_idx]
            _diag_logits_acc = float(np.mean(np.argmax(_glad_logits_val, axis=1) == val_y))
            logger.info(
                "GLAD-PI diagnostics -- "
                "fit dm: shape=%s mean=%.4f std=%.4f first5=%s ; "
                "select dm: shape=%s mean=%.4f std=%.4f first5=%s ; "
                "test dm: shape=%s mean=%.4f std=%.4f ; "
                "logit/label overall acc=%.4f (expect ~base model acc; low value means misalignment)",
                _diag_dm_fit.shape, float(np.mean(_diag_dm_fit)), float(np.std(_diag_dm_fit)),
                str(np.round(_diag_dm_fit[0, :5], 3)),
                _diag_dm_sel.shape, float(np.mean(_diag_dm_sel)), float(np.std(_diag_dm_sel)),
                str(np.round(_diag_dm_sel[0, :5], 3)),
                _sb_test_dm.shape, float(np.mean(_sb_test_dm)), float(np.std(_sb_test_dm)),
                _diag_logits_acc,
            )
            del _diag_dm_fit, _diag_dm_sel, _diag_logits_acc

            _glad = GLADPICalibrator(
                hidden_dim=args.glad_pi_hidden_dim,
                lr=args.glad_pi_lr,
                weight_decay=args.glad_pi_weight_decay,
                epochs=args.glad_pi_epochs,
                beta_grid=_glad_beta_grid,
                nll_tolerance=args.glad_pi_nll_tolerance,
                use_temperature=args.glad_pi_use_temperature,
            )
            _glad_fit_result = _fit_or_load_state_with_result(
                args, "glad_pi", _glad,
                lambda: _glad.fit(
                    _glad_logits_val[_glad_fit_idx],
                    val_y[_glad_fit_idx],
                    _sb_val_dm[_glad_fit_idx],
                    _glad_logits_val[_glad_select_idx],
                    val_y[_glad_select_idx],
                    _sb_val_dm[_glad_select_idx],
                ),
            )
            _glad_probs_test = _glad.calibrate(_glad_logits_test, _sb_test_dm)
            _validate_probability_matrix(
                _glad_probs_test, n_rows=len(test_y), n_classes=num_classes, method_name=_glad_name,
            )
            _glad_extra: Dict[str, Any] = {
                "selected_beta": _glad_fit_result["selected_beta"],
                "selected_fallback": _glad_fit_result["selected_fallback"],
                "reference_nll": _glad_fit_result["reference_nll"],
                "nll_budget": _glad_fit_result["nll_budget"],
                "best_select_net_flips": _glad_fit_result["best_select_net_flips"],
                "best_select_nll": _glad_fit_result["best_select_nll"],
                "selection_results": _glad_fit_result["selection_results"],
                "beta_grid": _glad_beta_grid,
                "hidden_dim": args.glad_pi_hidden_dim,
                "use_temperature": args.glad_pi_use_temperature,
                "inner_val_fraction": args.inner_val_fraction,
                "inner_val_seed": args.inner_val_seed,
                "inner_val_fit_n": int(len(_glad_fit_idx)),
                "inner_val_select_n": int(len(_glad_select_idx)),
                "feature_source": _sb_feature_source,
            }
            _glad_extra.update(_argmax_flip_metadata(test_base_probs, _glad_probs_test, test_y))
            _glad_extra.update(_nll_by_base_prediction_subset(test_base_probs, _glad_probs_test, test_y))
            _glad_extra.update(_nll_by_final_prediction_subset(_glad_probs_test, test_y))
            methods.append(
                _make_method_entry(
                    _glad_name,
                    _glad_probs_test,
                    test_y,
                    "decision_improving_full_vector",
                    True,
                    "shared_per_class_mlp_beta_grid",
                    "inner_val_select",
                    "select_net_flips_nll_gated",
                    _glad_extra,
                    anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                    base_probs_for_anchor_bins=base_probs_test,
                    base_probs=test_base_probs,
                )
            )
            method_probs_by_name[_glad_name] = _glad_probs_test
            method_can_change_argmax_map[_glad_name] = True
            logger.info(
                "glad_pi: selected_beta=%.4f fallback=%s select_net_flips=%d select_nll=%.4f",
                _glad_fit_result["selected_beta"],
                _glad_fit_result["selected_fallback"],
                _glad_fit_result["best_select_net_flips"],
                _glad_fit_result["best_select_nll"],
            )
            del _glad, _glad_probs_test, _glad_logits_val, _glad_logits_test
            _cleanup_memory()

    # ------------------------------------------------------------------ #
    # GLAD-PI zero-geometry twin (capacity-matched, geometry-ablated       #
    # control per §8e of the benchmark plan): identical architecture,     #
    # hyperparameters, and split as glad_pi above, with distance_k masked #
    # to 0 before the MLP. Not a separate subclass -- same GLADPICalibrator #
    # class, zero_geometry=True -- so parameter count matches exactly.    #
    # ------------------------------------------------------------------ #
    if args.enable_glad_pi_zero_geometry:
        _gladz_name = "glad_pi_zero_geometry"
        if _is_method_checkpoint_complete(args.output_dir, _gladz_name, len(test_y), num_classes):
            logger.info("Skipping completed method %s", _gladz_name)
        else:
            from Calibrators.glad_pi import GLADPICalibrator

            _gladz_logits_val = np.asarray(np.load(stage3_files["val_logits"], mmap_mode="r"))
            _gladz_logits_test = np.asarray(np.load(stage3_files["test_logits"], mmap_mode="r"))

            try:
                _gladz_beta_grid = [
                    float(x) for x in args.glad_pi_beta_grid.split(",") if x.strip()
                ]
            except ValueError as _ge:
                raise ValueError(f"--glad_pi_beta_grid must contain numeric values: {_ge}") from _ge
            if not _gladz_beta_grid:
                raise ValueError("--glad_pi_beta_grid must contain at least one value")

            # Identical (deterministic) inner-validation split as glad_pi above.
            _gladz_fit_idx, _gladz_select_idx = make_inner_validation_split(
                val_y, select_fraction=args.inner_val_fraction, seed=args.inner_val_seed
            )

            _gladz = GLADPICalibrator(
                hidden_dim=args.glad_pi_hidden_dim,
                lr=args.glad_pi_lr,
                weight_decay=args.glad_pi_weight_decay,
                epochs=args.glad_pi_epochs,
                beta_grid=_gladz_beta_grid,
                nll_tolerance=args.glad_pi_nll_tolerance,
                use_temperature=args.glad_pi_use_temperature,
                zero_geometry=True,
            )
            _gladz_fit_result = _fit_or_load_state_with_result(
                args, "glad_pi_zero_geometry", _gladz,
                lambda: _gladz.fit(
                    _gladz_logits_val[_gladz_fit_idx],
                    val_y[_gladz_fit_idx],
                    _sb_val_dm[_gladz_fit_idx],
                    _gladz_logits_val[_gladz_select_idx],
                    val_y[_gladz_select_idx],
                    _sb_val_dm[_gladz_select_idx],
                ),
            )
            _gladz_probs_test = _gladz.calibrate(_gladz_logits_test, _sb_test_dm)
            _validate_probability_matrix(
                _gladz_probs_test, n_rows=len(test_y), n_classes=num_classes, method_name=_gladz_name,
            )
            _gladz_extra: Dict[str, Any] = {
                "selected_beta": _gladz_fit_result["selected_beta"],
                "selected_fallback": _gladz_fit_result["selected_fallback"],
                "reference_nll": _gladz_fit_result["reference_nll"],
                "nll_budget": _gladz_fit_result["nll_budget"],
                "best_select_net_flips": _gladz_fit_result["best_select_net_flips"],
                "best_select_nll": _gladz_fit_result["best_select_nll"],
                "selection_results": _gladz_fit_result["selection_results"],
                "beta_grid": _gladz_beta_grid,
                "hidden_dim": args.glad_pi_hidden_dim,
                "use_temperature": args.glad_pi_use_temperature,
                "zero_geometry": True,
                "inner_val_fraction": args.inner_val_fraction,
                "inner_val_seed": args.inner_val_seed,
                "inner_val_fit_n": int(len(_gladz_fit_idx)),
                "inner_val_select_n": int(len(_gladz_select_idx)),
                "feature_source": _sb_feature_source,
            }
            _gladz_extra.update(_argmax_flip_metadata(test_base_probs, _gladz_probs_test, test_y))
            _gladz_extra.update(_nll_by_base_prediction_subset(test_base_probs, _gladz_probs_test, test_y))
            _gladz_extra.update(_nll_by_final_prediction_subset(_gladz_probs_test, test_y))
            methods.append(
                _make_method_entry(
                    _gladz_name,
                    _gladz_probs_test,
                    test_y,
                    "decision_improving_full_vector",
                    True,
                    "shared_per_class_mlp_beta_grid_zero_geometry",
                    "inner_val_select",
                    "select_net_flips_nll_gated",
                    _gladz_extra,
                    anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                    base_probs_for_anchor_bins=base_probs_test,
                    base_probs=test_base_probs,
                )
            )
            method_probs_by_name[_gladz_name] = _gladz_probs_test
            method_can_change_argmax_map[_gladz_name] = True
            logger.info(
                "glad_pi_zero_geometry: selected_beta=%.4f fallback=%s select_net_flips=%d select_nll=%.4f",
                _gladz_fit_result["selected_beta"],
                _gladz_fit_result["selected_fallback"],
                _gladz_fit_result["best_select_net_flips"],
                _gladz_fit_result["best_select_nll"],
            )
            del _gladz, _gladz_probs_test, _gladz_logits_val, _gladz_logits_test
            _cleanup_memory()

    # ------------------------------------------------------------------ #
    # Mahalanobis confidence (Phase 1, §8d): reuses the penultimate       #
    # features already captured for full KCal (via                       #
    # _penultimate_features_required, guaranteed present whenever this    #
    # method is enabled) instead of a dedicated forward pass. All splits  #
    # here (penultimate features, val_base_probs/test_base_probs,        #
    # val_y/test_y) come from the same stage3 extraction pass, so they   #
    # share one consistent row ordering.                                 #
    # ------------------------------------------------------------------ #
    if args.enable_mahalanobis_confidence:
        _mahal_name = "mahalanobis_confidence"
        if _is_method_checkpoint_complete(args.output_dir, _mahal_name, len(test_y), num_classes):
            logger.info("Skipping completed method %s", _mahal_name)
        else:
            from Calibrators.mahalanobis_confidence import MahalanobisConfidenceCalibrator

            # All three splits must come from the same (classifier-input
            # penultimate) representation; the research-selected train layer is
            # a different space with a different dimension.
            _mahal_train_features = np.load(stage3_files["kcal_train_penultimate"])
            _mahal_train_labels = np.load(stage3_files["train_y"])
            _mahal_val_penultimate = np.load(stage3_files["kcal_val_penultimate"])
            _mahal_test_penultimate = np.load(stage3_files["kcal_test_penultimate"])
            assert (
                _mahal_train_features.shape[1]
                == _mahal_val_penultimate.shape[1]
                == _mahal_test_penultimate.shape[1]
            ), (
                "Mahalanobis feature dimension mismatch: "
                f"train={_mahal_train_features.shape[1]}, "
                f"val={_mahal_val_penultimate.shape[1]}, "
                f"test={_mahal_test_penultimate.shape[1]}"
            )

            _mahal = MahalanobisConfidenceCalibrator()
            _fit_or_load_state(
                args, "mahalanobis_confidence", _mahal,
                lambda: _mahal.fit(
                    train_features=_mahal_train_features,
                    train_labels=_mahal_train_labels,
                    fit_features=_mahal_val_penultimate,
                    fit_base_probs=val_base_probs,
                    fit_labels=val_y,
                ),
            )
            _mahal_fit_info = _mahal.get_params()
            _mahal_probs_test = _mahal.calibrate(_mahal_test_penultimate, test_base_probs)
            _validate_probability_matrix(
                _mahal_probs_test, n_rows=len(test_y), n_classes=num_classes, method_name=_mahal_name,
            )
            _mahal_extra: Dict[str, Any] = {
                "shrinkage_used": _mahal_fit_info["shrinkage_used"],
                "feature_dim": _mahal_fit_info["feature_dim"],
                "penultimate_layer": "final_linear_classifier_input",
            }
            _mahal_extra.update(_argmax_flip_metadata(test_base_probs, _mahal_probs_test, test_y))
            _mahal_extra.update(_nll_by_base_prediction_subset(test_base_probs, _mahal_probs_test, test_y))
            methods.append(
                _make_method_entry(
                    _mahal_name,
                    _mahal_probs_test,
                    test_y,
                    "top_label_calibration",
                    False,
                    "ledoit_wolf_mahalanobis_isotonic_fit",
                    "validation",
                    "binary_correctness",
                    _mahal_extra,
                    anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                    base_probs_for_anchor_bins=base_probs_test,
                    base_probs=test_base_probs,
                )
            )
            method_probs_by_name[_mahal_name] = _mahal_probs_test
            method_can_change_argmax_map[_mahal_name] = False
            logger.info(
                "mahalanobis_confidence: shrinkage_used=%.4f feature_dim=%d",
                _mahal_fit_info["shrinkage_used"],
                _mahal_fit_info["feature_dim"],
            )
            del (
                _mahal,
                _mahal_probs_test,
                _mahal_train_features,
                _mahal_train_labels,
                _mahal_val_penultimate,
                _mahal_test_penultimate,
            )
        _cleanup_memory()

    # ------------------------------------------------------------------ #
    # Contrastive-beta sweep                                              #
    # softmax(a * z + b + beta * (d_predicted - d_class))                #
    # ------------------------------------------------------------------ #
    if args.enable_contrastive_beta_sweep:
        _contrastive_names = CONTRASTIVE_BETA_METHOD_NAMES
        if not args.enable_full_vector_geometric_fusion:
            logger.warning(
                "Skipping contrastive-beta sweep: --enable_full_vector_geometric_fusion "
                "is required to provide per-class distance matrices."
            )
        elif (
            _sb_val_dm is None
            or _sb_test_dm is None
            or np.shape(_sb_val_dm) != np.shape(val_base_probs)
            or np.shape(_sb_test_dm) != np.shape(test_base_probs)
            or not np.all(np.isfinite(_sb_val_dm))
            or not np.all(np.isfinite(_sb_test_dm))
        ):
            logger.warning(
                "Skipping contrastive-beta sweep: aligned finite per-class distance "
                "matrices are not available."
            )
        elif all(
            _is_method_checkpoint_complete(
                args.output_dir,
                name,
                len(test_y),
                num_classes,
                expected_gate_signature=_contrastive_gate_signatures[name],
            )
            for name in _contrastive_names
        ):
            for _contrastive_name in _contrastive_names:
                logger.info("Skipping completed method %s", _contrastive_name)
                method_can_change_argmax_map[_contrastive_name] = True
        else:
            try:
                _contrastive_beta_grid = [
                    float(value)
                    for value in args.contrastive_beta_grid.split(",")
                    if value.strip()
                ]
            except ValueError as _contrastive_grid_error:
                raise ValueError(
                    "--contrastive_beta_grid must contain numeric values: "
                    f"{_contrastive_grid_error}"
                ) from _contrastive_grid_error
            if not _contrastive_beta_grid:
                raise ValueError("--contrastive_beta_grid must contain at least one value")
            if not all(
                np.isfinite(beta) and beta >= 0.0
                for beta in _contrastive_beta_grid
            ):
                raise ValueError(
                    "--contrastive_beta_grid values must be finite and non-negative"
                )
            if 0.0 not in _contrastive_beta_grid:
                _contrastive_beta_grid.insert(0, 0.0)

            _contrastive_incomplete = {
                name
                for name in _contrastive_names
                if not _is_method_checkpoint_complete(
                    args.output_dir,
                    name,
                    len(test_y),
                    num_classes,
                    expected_gate_signature=_contrastive_gate_signatures[name],
                )
            }
            _contrastive_outputs = _build_contrastive_beta_outputs(
                val_logits=np.asarray(np.load(stage3_files["val_logits"], mmap_mode="r")),
                test_logits=np.asarray(np.load(stage3_files["test_logits"], mmap_mode="r")),
                val_labels=val_y,
                test_labels=test_y,
                val_base_probs=val_base_probs,
                test_base_probs=test_base_probs,
                val_distances=_sb_val_dm,
                test_distances=_sb_test_dm,
                beta_grid=_contrastive_beta_grid,
                nll_tolerance=args.contrastive_nll_tolerance,
                brier_tolerance=args.contrastive_brier_tolerance,
                ece_abs_tolerance=args.contrastive_ece_abs_tolerance,
                inner_val_fraction=args.inner_val_fraction,
                inner_val_seed=args.inner_val_seed,
                feature_source=_sb_feature_source,
                method_names=_contrastive_incomplete,
                anchor_ct=gc_dac_test_anchor_ct,
                anchor_base_probs=base_probs_test,
            )
            for _contrastive_name in _contrastive_names:
                if _contrastive_name not in _contrastive_outputs:
                    logger.info("Skipping completed method %s", _contrastive_name)
                    method_can_change_argmax_map[_contrastive_name] = True
                    continue
                _contrastive_entry, _contrastive_probs = _contrastive_outputs[
                    _contrastive_name
                ]
                methods.append(_contrastive_entry)
                method_probs_by_name[_contrastive_name] = _contrastive_probs
                method_can_change_argmax_map[_contrastive_name] = True
                logger.info(
                    "%s: selected_beta=%.4f fallback=%s select_net_flips=%d",
                    _contrastive_name,
                    _contrastive_entry["selected_beta"],
                    _contrastive_entry["selected_fallback"],
                    _contrastive_entry["best_select_net_flips"],
                )
            del _contrastive_outputs
            _cleanup_memory()

    if args.enable_glad_pi or args.enable_contrastive_beta_sweep:
        _write_incremental_summary(args.output_dir)

    # --- Assemble artifact context for extended per-sample NPZ ---
    _artifact_distance_arrays: Dict[str, np.ndarray] = {
        "distance_matrix": test_distance_matrix,
    }
    _artifact_method_meta: Dict[str, Dict[str, Any]] = {
        "full_vector_distance_fusion": {
            "distance_array_key": "distance_matrix",
            "formula": "FullVectorDistanceFusionCalibrator",
            "feature_source": "single_layer",
        },
        "anchored_rankgeom_tail_mixture": {
            "score_mode": "rank_normalized_non_top_distances",
            "distance_array_key": "distance_matrix",
            "epsilon": None,
            "formula": "_anchored_rankgeom_tail_mixture_probs",
        },
    }
    temperature_entry = _load_method_entry(args.output_dir, "temperature_scaling")
    rankgeom_params = rankgeom_selection or {"selected_lambda": 0.0, "selected_alpha": 0.0}
    _artifact_selected_params: Dict[str, Dict[str, Any]] = {
        "temperature_scaling": {"temperature": float(temperature_entry["temperature"])},
        "full_vector_distance_fusion": {"beta": float(fusion_meta["selected_beta"])},
        "anchored_rankgeom_tail_mixture": {
            "lambda": float(rankgeom_params["selected_lambda"]),
            "alpha": float(rankgeom_params["selected_alpha"]),
        },
    }
    _artifact_missing: List[Dict[str, str]] = []

    for _fvgf_mn, _fvgf_lam in _fvgf_selected_lambdas.items():
        _fvgf_dist_key: str | None = (
            "distance_matrix" if args.fusion_feature_source == "single_layer" else None
        )
        if _fvgf_dist_key is None:
            _artifact_missing.append(
                {
                    "field": f"distance_array:{_fvgf_mn}",
                    "reason": "rgcl fvgf internal distance matrix not extracted to artifact",
                }
            )
        _artifact_selected_params[_fvgf_mn] = {"lambda": _fvgf_lam}
        _artifact_method_meta[_fvgf_mn] = {
            "score_mode": _fvgf_score_modes_by_name.get(_fvgf_mn, "unknown"),
            "distance_array_key": _fvgf_dist_key,
            "formula": "FullVectorGeometricFusionCalibrator",
            "feature_source": args.fusion_feature_source,
        }

    if args.enable_knn_blend_baseline:
        _blend_art_dist_key = (
            "distance_matrix"
            if not (args.fusion_feature_source == "rgcl")
            else f"distance_matrix__{_blend_feature_source}"
        )
        if args.fusion_feature_source == "rgcl":
            _artifact_distance_arrays[_blend_art_dist_key] = _blend_test_dm
        _artifact_selected_params[_blend_method_name] = {
            "alpha": float(knn_blend.best_alpha),
            "gamma": float(knn_blend.best_gamma),
        }
        _artifact_method_meta[_blend_method_name] = {
            "distance_array_key": _blend_art_dist_key,
            "formula": "SoftmaxKNNBlendCalibrator",
            "feature_source": _blend_feature_source,
        }

    if args.enable_kcal_lite_baseline:
        _kcal_art_dist_key = f"knn_distances__{_kcal_feature_source}"
        _artifact_distance_arrays[_kcal_art_dist_key] = _kcal_test_knn_dm
        _artifact_selected_params[_kcal_method_name] = {
            "alpha": float(kcal.best_alpha),
            "gamma": float(kcal.best_gamma),
            "k_per_class": int(args.kcal_k_per_class),
        }
        _artifact_method_meta[_kcal_method_name] = {
            "distance_array_key": _kcal_art_dist_key,
            "formula": "KCalLiteCalibrator",
            "k_per_class": int(args.kcal_k_per_class),
            "feature_source": _kcal_feature_source,
        }

    if _kcal_features_required:
        if _kcal_full_params is None:
            raise RuntimeError("Full KCal metadata is unavailable after calibration")
        _artifact_selected_params["kcal"] = {
            "bandwidth": float(_kcal_full_params["bandwidth"]),
            "projection_dim": int(_kcal_full_params["projection_dim"]),
        }
        _artifact_method_meta["kcal"] = {
            "distance_array_key": None,
            "formula": "full_rbf_kde_over_projected_validation_references",
            "feature_source": "penultimate_classifier_input",
            "projection_type": _kcal_full_params["projection_type"],
            "bandwidth_folds_used": int(
                _kcal_full_params["bandwidth_folds_used"]
            ),
            "uses_all_calibration_references": True,
            "blends_model_probabilities": False,
        }
        _artifact_missing.append(
            {
                "field": "distance_array:kcal",
                "reason": (
                    "full pairwise KCal kernel distances are intentionally not persisted; "
                    "the fitted projection and calibration references are in the KCal checkpoint"
                ),
            }
        )

    for _factorial_method_name, _factorial_params in (
        _kcal_factorial_selected_params.items()
    ):
        _artifact_selected_params[_factorial_method_name] = _factorial_params
        _artifact_method_meta[_factorial_method_name] = (
            _kcal_factorial_method_artifacts[_factorial_method_name]
        )
        _artifact_missing.append(
            {
                "field": f"distance_array:{_factorial_method_name}",
                "reason": (
                    "factorial full pairwise/top-k tuning distances are not persisted; "
                    "precomputed representation paths and posterior hash are recorded"
                ),
            }
        )

    artifact_ctx: Dict[str, Any] = {
        "distance_arrays": _artifact_distance_arrays,
        "method_artifact_metadata": _artifact_method_meta,
        "selected_params": _artifact_selected_params,
        "missing_fields": _artifact_missing,
        "run_meta": {
            "dataset": args.dataset,
            "model": args.model,
            "seed": args.seed,
            "checkpoint_path": model_path,
            "split_sizes": {
                "train": int(len(train_labels)),
                "validation": int(len(val_labels)),
                "test": int(len(test_labels)),
            },
        },
    }

    for entry in methods:
        method_name = entry["method_name"]
        _save_method_checkpoint(
            args.output_dir,
            method_name,
            entry,
            method_probs_by_name[method_name],
            bool(entry["can_change_argmax"]),
            stage_name="stage4_late_outputs",
        )
        logger.info("Saved method checkpoint: %s", method_name)

    if val_anchor_c_t is not None and val_anchor_diag is not None:
        _save_npy(late_anchor_paths["val_anchor_c_t"], np.asarray(val_anchor_c_t).astype(np.float64))
        _atomic_write_json(late_anchor_paths["val_anchor_diag"], val_anchor_diag)
    if rankgeom_selection is not None:
        _atomic_write_json(late_anchor_paths["rankgeom_selection"], rankgeom_selection)
    _save_npy(
        late_anchor_paths["anchored_rankgeom_probs"],
        np.asarray(
            _load_method_probs(
                args.output_dir, "anchored_rankgeom_tail_mixture", mmap_mode="r"
            )
        ).astype(np.float64),
    )

    # ------------------------------------------------------------------ #
    # External method probability import                                   #
    # Evaluates externally calibrated probability matrices (e.g. official  #
    # AAR outputs) using the same metrics as native methods.               #
    # ------------------------------------------------------------------ #
    if args.external_method_json is not None:
        _ext_json_path = args.external_method_json
        if not os.path.exists(_ext_json_path):
            logger.warning("--external_method_json path does not exist: %s", _ext_json_path)
        else:
            with open(_ext_json_path, "r") as _ef:
                _ext_spec = json.load(_ef)
            _ext_methods = _ext_spec.get("methods", {})
            for _ext_name, _ext_cfg in _ext_methods.items():
                if _is_method_checkpoint_complete(
                    args.output_dir, _ext_name, len(test_labels), num_classes
                ):
                    logger.info("Skipping completed external method %s", _ext_name)
                    continue
                _ext_probs_path = _ext_cfg.get("test_probs")
                if _ext_probs_path is None or not os.path.exists(_ext_probs_path):
                    logger.warning(
                        "External method %s: test_probs path missing or not found (%s). Skipping.",
                        _ext_name,
                        _ext_probs_path,
                    )
                    continue
                logger.info("Loading external method probabilities: %s from %s", _ext_name, _ext_probs_path)
                _ext_probs = np.load(_ext_probs_path)
                if _ext_probs.dtype != np.float64:
                    _ext_probs = _ext_probs.astype(np.float64)
                try:
                    validate_probability_matrix(_ext_probs, test_labels)
                except ValueError as _e:
                    logger.error(
                        "External method %s failed probability validation: %s. Skipping.",
                        _ext_name,
                        _e,
                    )
                    continue

                _ext_meta = _ext_cfg.get("metadata", {})
                _ext_can_change = bool(_ext_meta.get("can_change_argmax", True))
                _ext_family = str(_ext_meta.get("method_family", "external_full_vector_posthoc"))
                _ext_tuning = str(_ext_meta.get("tuning_procedure", "external"))
                _ext_split = str(_ext_meta.get("tuning_split", "external_or_validation"))
                _ext_obj = str(_ext_meta.get("tuning_objective", "external"))

                _ext_entry = _make_method_entry(
                    _ext_name,
                    _ext_probs,
                    test_labels,
                    method_family=_ext_family,
                    can_change_argmax=_ext_can_change,
                    tuning_procedure=_ext_tuning,
                    tuning_split=_ext_split,
                    tuning_objective=_ext_obj,
                    base_probs=base_probs_test,
                    anchor_ct_for_stratification=gc_dac_test_anchor_ct,
                    base_probs_for_anchor_bins=base_probs_test,
                    extra={
                        "external_source": _ext_probs_path,
                        "external_spec": {k: v for k, v in _ext_cfg.items() if k != "test_probs"},
                    },
                )
                _save_method_checkpoint(
                    args.output_dir,
                    _ext_name,
                    _ext_entry,
                    _ext_probs,
                    _ext_can_change,
                    stage_name="stage4_external",
                )
                method_can_change_argmax_map[_ext_name] = _ext_can_change
                logger.info("External method %s saved.", _ext_name)
                del _ext_probs, _ext_entry
            _write_incremental_summary(args.output_dir)

    json_path, csv_path, methods = _write_incremental_summary(
        args.output_dir,
        metadata=summary_metadata,
    )
    method_registry = _load_method_registry(args.output_dir)
    method_probs_by_name = {
        m: np.asarray(_load_method_probs(args.output_dir, m, mmap_mode="r"))
        for m in method_registry.get("ordered_methods", [])
        if _is_method_checkpoint_complete(args.output_dir, m, len(test_labels), num_classes)
    }
    method_can_change_argmax_for_npz = dict(method_registry.get("method_can_change_argmax", {}))
    method_can_change_argmax_for_npz.update(method_can_change_argmax_map)
    per_sample_npz_path = _write_per_sample_npz(
        run_dir=args.output_dir,
        labels_val=val_y,
        base_probs_val=val_base_probs,
        full_vector_distance_fusion_probs_val=fused_val_probs,
        labels_test=test_labels,
        base_probs_test=base_probs_test,
        gc_dac_anchor_ct_test=gc_dac_test_anchor_ct,
        gc_dac_top_pred_test=gc_dac_test_top_idx,
        method_probs_by_name=method_probs_by_name,
        method_can_change_argmax=method_can_change_argmax_for_npz,
        artifact_ctx=artifact_ctx,
    )

    logger.info(f"Unified summary JSON: {json_path}")
    logger.info(f"Unified summary CSV: {csv_path}")
    logger.info(f"Unified per-sample NPZ: {per_sample_npz_path}")

    # Diagnostic: warn about methods missing structural_axes or decision_audit so
    # unregistered names are caught before real experiments.
    _missing_axes: list[str] = []
    _missing_audit: list[str] = []
    for _m in methods:
        _mn = _m.get("method_name", "?")
        if _mn == "base_model":
            continue  # base_model intentionally skipped for decision_audit
        if "structural_axes" not in _m:
            _missing_axes.append(_mn)
        if "decision_audit" not in _m:
            _missing_audit.append(_mn)
    if _missing_axes:
        logger.warning(
            "Methods missing structural_axes (add to method_metadata.py registry): %s",
            ", ".join(_missing_axes),
        )
    if _missing_audit:
        logger.warning(
            "Methods missing decision_audit (pass base_probs to _make_method_entry): %s",
            ", ".join(_missing_audit),
        )
    _mark_stage_complete(
        args.output_dir,
        progress,
        "stage4_late_outputs",
        files={
            "summary_json": json_path,
            "summary_csv": csv_path,
            "per_sample_npz": per_sample_npz_path,
        },
    )
    # The initial write records the frozen state before evaluation. Rewriting
    # after successful completion makes a clean run record every persisted
    # fitted calibrator artifact as well.
    _write_fit_once_provenance(args)
    del method_probs_by_name, method_can_change_argmax_for_npz
    _cleanup_memory()

    if getattr(args, "cleanup_intermediates", False):
        import shutil
        for _subdir in ("splits", "features"):
            _target = os.path.join(args.output_dir, "intermediates", _subdir)
            if os.path.isdir(_target):
                try:
                    shutil.rmtree(_target)
                    logger.info("Cleaned up intermediates/%s (--cleanup-intermediates)", _subdir)
                except OSError as _e:
                    logger.warning("Could not clean up intermediates/%s: %s", _subdir, _e)


if __name__ == "__main__":
    main()
