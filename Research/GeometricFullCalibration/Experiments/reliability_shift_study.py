"""
Study A (shift) -- does hidden-representation geometry become incrementally
informative as corruption degrades the classifier?

HYPOTHESIS
----------
Hidden geometry may be redundant in-distribution because the output head
already exposes most reliability information, yet become useful under
corruption shift as the head's mapping degrades. The clean seed-4 result
(R + G_H ~ R + zero, while R + G_Z > R + zero) is explicitly NOT treated as a
stop criterion.

FREEZE-ON-CLEAN PROTOCOL (mandatory, enforced in code)
------------------------------------------------------
Everything that is fitted is fitted on CLEAN data exactly once:

  * class-centroid reference banks (hidden, logit, projected hidden) --
    from the clean TRAIN split;
  * the fixed random projection matrix -- seeded, data-independent;
  * every reliability learner, including its feature standardization
    statistics -- from the clean calibration/validation split, using the
    clean inner fit/select indices and the clean hyperparameter selection.

The frozen state is persisted once, hashed, and then LOADED for every
corruption cell. A corruption cell only ever supplies evaluation inputs
(test features, test logits, test labels). There is no corruption-specific
refitting, model selection, threshold tuning or normalization fitting, and
corruption labels are never seen during fitting. `evaluate_cell` takes a
frozen state and cannot fit anything -- the learners it receives are already
fitted and it calls only `predict_confidence`.

Every corruption artifact records the frozen state's sha256 so reuse is
provable, not asserted.

TARGET (unchanged from the clean study)
---------------------------------------
    correct_c = 1[ argmax(base_classifier(x_c)) == y ]
i.e. the corrupted base model's own correctness. No arm changes a prediction,
and decision intervention remains a separate study.

METRIC ORIENTATION
------------------
All reported "gains" are oriented so POSITIVE ALWAYS MEANS BETTER:
    gain_auroc = AUROC(arm) - AUROC(R+zero)
    gain_aurc  = AURC(R+zero) - AURC(arm)      <-- sign flipped, lower is better
    gain_excess_aurc = excess(R+zero) - excess(arm)
    gain_ece / gain_adaptive_ece = ECE(R+zero) - ECE(arm)
    gain_coverage = coverage(arm) - coverage(R+zero) at matched risk
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Calibrators.reliability_learner import ReliabilityLearner  # noqa: E402
from Experiments.reliability_experiment import (  # noqa: E402
    INNER_FIT_FRACTION,
    INNER_SPLIT_SEED,
    LEARNER_SEED,
    PROJECTION_SEED,
    _class_centroids,
    _class_distance_matrix,
    _git_provenance,
    _softmax,
    geometry_features,
    inner_split,
    load_run_arrays,
    recoverability_diagnostic,
    regular_features,
)
from utils.selective_metrics import (  # noqa: E402
    correctness_auroc,
    coverage_at_matched_risk,
    risk_coverage_curve,
)
from utils.unified_metrics import evaluate_scalar_confidence  # noqa: E402

STUDY_VERSION = "reliability_shift_v1"

# The arms carried into the shift sweep. The five shuffle seeds are
# deliberately NOT repeated per cell: the clean study already showed the
# association control behaves correctly, and repeating it here would multiply
# compute before there is anything to confirm.
SHIFT_ARMS = ("R", "R_plus_zero", "R_plus_G_Z", "R_plus_G_Hproj", "R_plus_G_H")
PRIMARY_DENOMINATOR = "R_plus_zero"

# Standard CIFAR-100-C corruption families. Used only for a SECONDARY summary;
# the per-corruption table is primary.
CORRUPTION_FAMILIES = {
    "gaussian_noise": "noise", "shot_noise": "noise", "impulse_noise": "noise",
    "defocus_blur": "blur", "glass_blur": "blur", "motion_blur": "blur", "zoom_blur": "blur",
    "snow": "weather", "frost": "weather", "fog": "weather", "brightness": "weather",
    "contrast": "digital", "elastic_transform": "digital", "pixelate": "digital",
    "jpeg_compression": "digital",
}


# ====================================================================== state
def fit_clean_state(clean_run_dir: str) -> Dict[str, Any]:
    """Fit EVERYTHING on clean data, once. This is the only fitting step."""
    data = load_run_arrays(clean_run_dir)
    num_classes = int(data["base_probs_test"].shape[1])
    hidden_dim = int(data["train_features"].shape[1])

    rng_proj = np.random.default_rng(PROJECTION_SEED)
    projection = rng_proj.normal(size=(hidden_dim, num_classes)) / np.sqrt(hidden_dim)

    centroids = {
        "G_H": _class_centroids(data["train_features"], data["train_y"], num_classes),
        "G_Z": _class_centroids(data["train_logits"], data["train_y"], num_classes),
        "G_Hproj": _class_centroids(
            np.asarray(data["train_features"], dtype=np.float64) @ projection,
            data["train_y"], num_classes,
        ),
    }

    val_pred = np.argmax(data["base_probs_val"], axis=1)
    correct_val = (val_pred == data["val_labels"]).astype(np.float64)
    fit_idx, sel_idx = inner_split(len(data["val_labels"]), INNER_SPLIT_SEED, INNER_FIT_FRACTION)

    reg_val, reg_names = regular_features(data["val_logits"])
    geo_blocks = {
        key: geometry_features(
            _class_distance_matrix(
                np.asarray(data["val_features"], dtype=np.float64) @ projection
                if key == "G_Hproj"
                else (data["val_logits"] if key == "G_Z" else data["val_features"]),
                centroids[key],
            ),
            val_pred, "geo",
        )[0]
        for key in ("G_H", "G_Z", "G_Hproj")
    }
    geo_width = geo_blocks["G_H"].shape[1]

    arm_features = {
        "R": reg_val,
        "R_plus_zero": np.concatenate([reg_val, np.zeros((reg_val.shape[0], geo_width))], axis=1),
        "R_plus_G_H": np.concatenate([reg_val, geo_blocks["G_H"]], axis=1),
        "R_plus_G_Z": np.concatenate([reg_val, geo_blocks["G_Z"]], axis=1),
        "R_plus_G_Hproj": np.concatenate([reg_val, geo_blocks["G_Hproj"]], axis=1),
    }

    learners: Dict[str, ReliabilityLearner] = {}
    fit_info: Dict[str, Any] = {}
    for arm in SHIFT_ARMS:
        learner = ReliabilityLearner(seed=LEARNER_SEED)
        info = learner.fit(
            arm_features[arm][fit_idx], correct_val[fit_idx],
            arm_features[arm][sel_idx], correct_val[sel_idx],
        )
        learners[arm] = learner
        fit_info[arm] = {
            **{k: v for k, v in info.items() if k != "selection_results"},
            "selection_results": info["selection_results"],
            "initial_parameter_fingerprint": learner.initial_parameter_fingerprint(
                arm_features[arm].shape[1]
            ),
        }

    return {
        "study_version": STUDY_VERSION,
        "clean_run_dir": os.path.abspath(clean_run_dir),
        "num_classes": num_classes,
        "hidden_dim": hidden_dim,
        "projection": projection,
        "centroids": centroids,
        "learners": learners,
        "fit_info": fit_info,
        "regular_feature_names": reg_names,
        "geometry_width": int(geo_width),
        "clean_base_accuracy_test": float(
            np.mean(np.argmax(data["base_probs_test"], axis=1) == data["test_labels"])
        ),
        "clean_base_accuracy_val": float(correct_val.mean()),
        "seeds": {
            "learner_seed": LEARNER_SEED,
            "inner_split_seed": INNER_SPLIT_SEED,
            "projection_seed": PROJECTION_SEED,
            "inner_fit_fraction": INNER_FIT_FRACTION,
        },
        "split_sizes": {
            "reference_bank_n": int(data["train_y"].shape[0]),
            "fit_n": int(fit_idx.size),
            "select_n": int(sel_idx.size),
        },
    }


def save_state(state: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=4)
    return state_hash(path)


def state_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(path: str) -> Tuple[Dict[str, Any], str]:
    with open(path, "rb") as f:
        return pickle.load(f), state_hash(path)


# ================================================================= evaluation
def _cell_inputs(cell_dir: str) -> Dict[str, np.ndarray]:
    """Evaluation inputs only. A cell supplies no fitting data of any kind."""
    feat = os.path.join(cell_dir, "intermediates", "features")
    out = {
        "test_features": np.load(os.path.join(feat, "test_features.npy")),
        "test_logits": np.load(os.path.join(feat, "test_logits.npy")),
        "test_labels": np.load(os.path.join(feat, "test_labels.npy")).reshape(-1),
    }
    if out["test_features"].shape[0] != out["test_labels"].shape[0]:
        raise ValueError(f"{cell_dir}: feature/label row mismatch")
    return out


def evaluate_cell(
    state: Dict[str, Any],
    cell_dir: str,
    *,
    corruption: str,
    severity: Optional[int],
    n_boot: int = 1000,
    boot_seed: int = 4242,
) -> Dict[str, Any]:
    """Evaluate the FROZEN clean-fitted arms on one (corruption, severity) cell.

    Nothing here fits: `state["learners"]` arrive already fitted and only
    `predict_confidence` is called, and the centroids/projection/normalization
    all come from `state`.
    """
    inputs = _cell_inputs(cell_dir)
    logits = np.asarray(inputs["test_logits"], dtype=np.float64)
    labels = inputs["test_labels"]
    base_probs = _softmax(logits)
    base_pred = np.argmax(base_probs, axis=1)
    correct = (base_pred == labels).astype(np.float64)

    projection = state["projection"]
    centroids = state["centroids"]
    dist = {
        "G_H": _class_distance_matrix(inputs["test_features"], centroids["G_H"]),
        "G_Z": _class_distance_matrix(logits, centroids["G_Z"]),
        "G_Hproj": _class_distance_matrix(
            np.asarray(inputs["test_features"], dtype=np.float64) @ projection,
            centroids["G_Hproj"],
        ),
    }

    reg, _ = regular_features(logits)
    geo_width = state["geometry_width"]
    arm_features = {
        "R": reg,
        "R_plus_zero": np.concatenate([reg, np.zeros((reg.shape[0], geo_width))], axis=1),
        "R_plus_G_H": np.concatenate([reg, geometry_features(dist["G_H"], base_pred, "geo")[0]], axis=1),
        "R_plus_G_Z": np.concatenate([reg, geometry_features(dist["G_Z"], base_pred, "geo")[0]], axis=1),
        "R_plus_G_Hproj": np.concatenate([reg, geometry_features(dist["G_Hproj"], base_pred, "geo")[0]], axis=1),
    }

    confidences = {
        arm: state["learners"][arm].predict_confidence(arm_features[arm]) for arm in SHIFT_ARMS
    }
    metrics = {arm: evaluate_scalar_confidence(confidences[arm], correct) for arm in SHIFT_ARMS}

    base_acc = float(correct.mean())
    cell: Dict[str, Any] = {
        "corruption": corruption,
        "severity": severity,
        "family": CORRUPTION_FAMILIES.get(corruption, "clean" if corruption == "clean" else "unknown"),
        "n": int(labels.size),
        "base_accuracy": base_acc,
        "base_degradation": float(state["clean_base_accuracy_test"] - base_acc),
        "arms": {arm: {"metrics": metrics[arm]} for arm in SHIFT_ARMS},
        "frozen_state_reused": True,
    }

    # ---- oriented gains: POSITIVE ALWAYS MEANS BETTER ---------------------
    den = PRIMARY_DENOMINATOR
    gains: Dict[str, Any] = {}
    for arm in SHIFT_ARMS:
        if arm == den:
            continue
        gains[arm] = _oriented_gains(metrics[arm], metrics[den])
        gains[arm]["coverage_at_matched_risk"] = coverage_at_matched_risk(
            confidences[den], correct, confidences[arm], correct, reference_coverage=0.8
        )
        gains[arm]["gain_coverage"] = gains[arm]["coverage_at_matched_risk"]["delta_coverage"]
        gains[arm]["paired_bootstrap"] = _paired_boot(
            confidences[den], confidences[arm], correct, n_boot=n_boot, seed=boot_seed
        )
    # The head-to-head the hypothesis turns on.
    gains["R_plus_G_H__vs__R_plus_G_Z"] = _oriented_gains(metrics["R_plus_G_H"], metrics["R_plus_G_Z"])
    gains["R_plus_G_H__vs__R_plus_G_Z"]["paired_bootstrap"] = _paired_boot(
        confidences["R_plus_G_Z"], confidences["R_plus_G_H"], correct, n_boot=n_boot, seed=boot_seed
    )
    cell["gains_vs_" + den] = gains

    # ---- Stage 9: recoverability under shift ------------------------------
    cell["recoverability"] = {
        space: recoverability_diagnostic(dist[space], base_pred, labels)
        for space in ("G_H", "G_Z", "G_Hproj")
    }
    return cell


def _oriented_gains(arm: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    """Every value positive-is-better; lower-is-better metrics are sign-flipped."""
    def hi(key):  # higher is better
        a, b = arm.get(key), ref.get(key)
        return None if a is None or b is None else float(a) - float(b)

    def lo(key):  # lower is better -> flip
        a, b = arm.get(key), ref.get(key)
        return None if a is None or b is None else float(b) - float(a)

    return {
        "gain_auroc": hi("correctness_auroc"),
        "gain_aurc": lo("aurc"),
        "gain_excess_aurc": lo("excess_aurc"),
        "gain_ece": lo("top_label_ece"),
        "gain_adaptive_ece": lo("adaptive_ece"),
        "orientation": "positive = better for every field above",
    }


def _paired_boot(
    ref_conf: np.ndarray, arm_conf: np.ndarray, correct: np.ndarray,
    *, n_boot: int, seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = correct.size
    d_auroc, d_aurc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        c = correct[idx]
        if c.min() == c.max():
            continue
        a, b = correctness_auroc(ref_conf[idx], c), correctness_auroc(arm_conf[idx], c)
        if a is None or b is None:
            continue
        d_auroc.append(b - a)
        # oriented: positive = arm has LOWER aurc = better
        d_aurc.append(
            float(np.mean(risk_coverage_curve(ref_conf[idx], c)["risk"]))
            - float(np.mean(risk_coverage_curve(arm_conf[idx], c)["risk"]))
        )

    def ci(v):
        if not v:
            return {"mean": None, "ci_lo": None, "ci_hi": None, "p_gt_0": None}
        a = np.asarray(v)
        return {"mean": float(a.mean()), "ci_lo": float(np.percentile(a, 2.5)),
                "ci_hi": float(np.percentile(a, 97.5)), "p_gt_0": float(np.mean(a > 0))}

    return {"n_boot": n_boot, "seed": seed,
            "gain_auroc": ci(d_auroc), "gain_aurc": ci(d_aurc)}


# ==================================================================== driver
def discover_cells(evaluation_dir: str) -> List[Tuple[str, str, Optional[int]]]:
    """(cell_dir, corruption, severity) for every cell with extracted features."""
    out = []
    for name in sorted(os.listdir(evaluation_dir)):
        cell_dir = os.path.join(evaluation_dir, name)
        if not os.path.isdir(cell_dir):
            continue
        if not os.path.exists(os.path.join(cell_dir, "intermediates", "features", "test_features.npy")):
            continue
        if name == "clean":
            out.append((cell_dir, "clean", None))
            continue
        if "_s" in name:
            corruption, sev = name.rsplit("_s", 1)
            if sev.isdigit():
                out.append((cell_dir, corruption, int(sev)))
    return out


def run_sweep(
    evaluation_dir: str, clean_run_dir: str, out_dir: str,
    *, n_boot: int = 1000, force: bool = False,
) -> Dict[str, Any]:
    if os.path.exists(out_dir) and not force:
        raise FileExistsError(f"{out_dir} already exists; pass --force")
    os.makedirs(out_dir, exist_ok=True)

    state = fit_clean_state(clean_run_dir)
    state_path = os.path.join(out_dir, "frozen_clean_state.pkl")
    frozen_hash = save_state(state, state_path)

    # Reload from disk so every cell provably uses the persisted frozen state
    # rather than the in-memory object that produced it.
    state, reloaded_hash = load_state(state_path)
    if reloaded_hash != frozen_hash:
        raise RuntimeError("Frozen state hash changed between write and read")

    cells = []
    for cell_dir, corruption, severity in discover_cells(evaluation_dir):
        cell = evaluate_cell(state, cell_dir, corruption=corruption, severity=severity, n_boot=n_boot)
        cell["cell_dir"] = cell_dir
        cell["frozen_state_sha256"] = reloaded_hash
        cells.append(cell)
        sev = "clean" if severity is None else f"s{severity}"
        gh = cell[f"gains_vs_{PRIMARY_DENOMINATOR}"]["R_plus_G_H"]["gain_auroc"]
        print(f"  {corruption:20s} {sev:6s} base_acc={cell['base_accuracy']:.4f} "
              f"gain_auroc(G_H)={gh:+.5f}", flush=True)

    payload = {
        "study_version": STUDY_VERSION,
        "evaluation_dir": os.path.abspath(evaluation_dir),
        "clean_run_dir": os.path.abspath(clean_run_dir),
        "frozen_state_path": os.path.abspath(state_path),
        "frozen_state_sha256": reloaded_hash,
        "primary_denominator": PRIMARY_DENOMINATOR,
        "arms": list(SHIFT_ARMS),
        "target": "correct_c = (argmax(base_classifier(x_c)) == y)",
        "clean_base_accuracy_test": state["clean_base_accuracy_test"],
        "fit_info": state["fit_info"],
        "seeds": state["seeds"],
        "split_sizes": state["split_sizes"],
        "cells": cells,
        "protocol": {
            "corruption_specific_refitting": False,
            "corruption_specific_model_selection": False,
            "corruption_specific_threshold_tuning": False,
            "corruption_specific_normalization": False,
            "corruption_labels_seen_during_fitting": False,
            "note": "every learner, centroid bank, projection and normalization "
                    "state was fitted on clean data and loaded from "
                    "frozen_clean_state.pkl for each cell",
        },
        "provenance": _git_provenance(str(Path(__file__).resolve().parents[1])),
    }
    with open(os.path.join(out_dir, "shift_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation_dir", required=True)
    parser.add_argument("--clean_run_dir", default=None, help="default: <evaluation_dir>/clean")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    clean = args.clean_run_dir or os.path.join(args.evaluation_dir, "clean")
    payload = run_sweep(args.evaluation_dir, clean, args.out_dir,
                        n_boot=args.n_boot, force=args.force)
    print(f"Wrote {args.out_dir} ({len(payload['cells'])} cells)")


if __name__ == "__main__":
    main()
