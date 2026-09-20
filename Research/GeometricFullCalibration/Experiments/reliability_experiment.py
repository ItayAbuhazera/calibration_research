"""
Study A -- incremental-geometry reliability experiment (clean split only).

THE QUESTION
------------
Does representation-space geometry carry information about BASE-MODEL
correctness that strong ordinary output-space signals do not already carry?

The target is fixed for every arm:

    C_base(x) = 1[ argmax(base_probs(x)) == y ]

No arm changes a prediction. This is deliberately separated from the
decision-intervention study (Study B); accuracy and net flips are NOT evidence
here, because the label being predicted is the base model's own correctness.

ARMS (identical learner, identical seed, identical splits, identical budget)
---------------------------------------------------------------------------
  R                  regular output-only signals
  R + zero           same input width as the geometry arms, geometry block
                     zeroed  -- the ARCHITECTURAL control
  R + shuffled(G_H)  hidden geometry with the sample association destroyed by
                     a deterministic permutation -- the ASSOCIATION control
                     (several permutation seeds)
  R + G_Z            class-reference geometry built in LOGIT space -- the
                     "is it internal representations, or just class-conditional
                     reference geometry?" control
  R + G_H            class-reference geometry built in HIDDEN space
  R + G_Hproj        hidden geometry after a fixed, non-learned random
                     projection to the logit dimensionality -- the
                     DIMENSIONALITY control

Arms 2-6 share an input width and a trainable parameter count, and all arms
share the initial-parameter fingerprint for their width. Only the contents of
the geometry block differ.

DATA PROTOCOL (no refitting of the classifier, no test tuning)
--------------------------------------------------------------
  reference bank for class centroids : TRAIN split (labels used; never test)
  learner fit / select               : the frozen calibration/validation split,
                                       split deterministically into inner
                                       fit/select
  evaluation                         : clean TEST split
Everything is read from a completed run's stored intermediates. Nothing is
refit, and nothing in the source run is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Calibrators.reliability_learner import ReliabilityLearner  # noqa: E402
from utils.selective_metrics import (  # noqa: E402
    correctness_auroc,
    coverage_at_matched_risk,
    coverage_at_risk,
    risk_coverage_curve,
)
from utils.unified_metrics import evaluate_scalar_confidence  # noqa: E402

EXPERIMENT_VERSION = "reliability_v1"

# Deterministic seeds, recorded in the artifact.
LEARNER_SEED = 20260920
INNER_SPLIT_SEED = 7
SHUFFLE_SEEDS = (101, 102, 103, 104, 105)
PROJECTION_SEED = 991
INNER_FIT_FRACTION = 0.7

TOP_K = 10  # how many sorted logits / distances enter the summaries


# ---------------------------------------------------------------- utilities
def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def _class_centroids(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Per-class mean of L2-normalized reference features (the labeled bank)."""
    feats = _l2_normalize(features)
    centroids = np.zeros((num_classes, feats.shape[1]), dtype=np.float64)
    for c in range(num_classes):
        mask = labels == c
        if not np.any(mask):
            raise ValueError(f"Reference bank has no examples of class {c}")
        centroids[c] = feats[mask].mean(axis=0)
    return centroids


def _class_distance_matrix(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Euclidean distance from each L2-normalized sample to each class centroid.

    IDENTICAL definition in hidden space and logit space -- same
    normalization, same metric, same reference bank construction -- so the
    G_H / G_Z contrast isolates the space, not the statistic.
    """
    feats = _l2_normalize(features)
    a2 = np.sum(feats * feats, axis=1, keepdims=True)
    b2 = np.sum(centroids * centroids, axis=1)
    sq = a2 - 2.0 * (feats @ centroids.T) + b2[None, :]
    return np.sqrt(np.clip(sq, 0.0, None))


# --------------------------------------------------------- feature blocks
def regular_features(logits: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Ordinary output-space signals only. NO internal representation.

    Includes the full sorted logit and sorted probability profiles (top-K) so
    geometry is not competing against a handful of handcrafted scalars, plus
    the shift-sensitive statistics that a softmax-only context cannot express
    (energy / log-sum-exp and logit norm).
    """
    z = np.asarray(logits, dtype=np.float64)
    p = _softmax(z)
    z_sorted = np.sort(z, axis=1)[:, ::-1]
    p_sorted = np.sort(p, axis=1)[:, ::-1]

    cols: List[np.ndarray] = []
    names: List[str] = []

    # Clamp to the class count: a problem with fewer classes than TOP_K must
    # not produce more feature NAMES than feature COLUMNS.
    k = min(TOP_K, z.shape[1])
    cols.append(z_sorted[:, :k]); names += [f"logit_sorted_{i}" for i in range(k)]
    cols.append(p_sorted[:, :k]); names += [f"prob_sorted_{i}" for i in range(k)]

    msp = p_sorted[:, 0]
    entropy = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)
    logit_margin = z_sorted[:, 0] - z_sorted[:, 1]
    prob_margin = p_sorted[:, 0] - p_sorted[:, 1]
    energy = np.log(np.exp(z - z.max(axis=1, keepdims=True)).sum(axis=1)) + z.max(axis=1)
    logit_norm = np.linalg.norm(z, axis=1)
    topk_mass = p_sorted[:, :5].sum(axis=1)
    logit_mean = z.mean(axis=1)
    logit_std = z.std(axis=1)
    neg_entropy_ratio = msp / np.maximum(entropy, 1e-12)

    for arr, nm in [
        (msp, "msp"), (entropy, "entropy"), (logit_margin, "logit_margin"),
        (prob_margin, "prob_margin"), (energy, "energy"), (logit_norm, "logit_norm"),
        (topk_mass, "top5_prob_mass"), (logit_mean, "logit_mean"),
        (logit_std, "logit_std"), (neg_entropy_ratio, "msp_over_entropy"),
    ]:
        cols.append(arr[:, None]); names.append(nm)

    return np.concatenate(cols, axis=1), names


def geometry_features(
    distance_matrix: np.ndarray, base_pred: np.ndarray, prefix: str
) -> Tuple[np.ndarray, List[str]]:
    """Summarize a per-class distance vector.

    The SAME summary is applied to hidden-space, logit-space, projected and
    shuffled distance matrices, so every geometry arm has an identical feature
    interface and identical width regardless of the underlying representation
    dimensionality.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    rows = np.arange(n)
    d_sorted = np.sort(d, axis=1)
    d_pred = d[rows, base_pred]
    order = np.argsort(d, axis=1)
    pred_rank = np.argmax(order == base_pred[:, None], axis=1).astype(np.float64)

    k = min(TOP_K, d.shape[1])
    cols = [d_sorted[:, :k]]
    names = [f"{prefix}_dist_sorted_{i}" for i in range(k)]
    extras = [
        (d_pred, "dist_to_pred"),
        (d_sorted[:, 0], "dist_min"),
        (d_pred - d_sorted[:, 0], "pred_minus_min"),
        (d_sorted[:, 1] - d_sorted[:, 0], "gap_min_second"),
        (pred_rank, "pred_class_rank"),
        (d.mean(axis=1), "dist_mean"),
        (d.std(axis=1), "dist_std"),
        (d_pred / np.maximum(d.mean(axis=1), 1e-12), "dist_pred_over_mean"),
    ]
    for arr, nm in extras:
        cols.append(arr[:, None]); names.append(f"{prefix}_{nm}")
    return np.concatenate(cols, axis=1), names


# ------------------------------------------------------------------ loading
def load_run_arrays(run_dir: str) -> Dict[str, Any]:
    inter = os.path.join(run_dir, "intermediates")
    feat = os.path.join(inter, "features")
    per_sample = os.path.join(run_dir, "per_sample", "per_sample_arrays.npz")
    d = np.load(per_sample, allow_pickle=True)

    out = {
        "train_features": np.load(os.path.join(feat, "train_features.npy")),
        "train_logits": np.load(os.path.join(feat, "train_logits.npy")),
        "train_y": np.load(os.path.join(feat, "train_y.npy")),
        "val_logits": np.load(os.path.join(feat, "val_logits.npy")),
        "val_features": np.load(os.path.join(feat, "val_features.npy")),
        "test_logits": np.load(os.path.join(feat, "test_logits.npy")),
        "test_features": np.load(os.path.join(feat, "test_features.npy")),
        "val_labels": np.asarray(d["labels_val"]).reshape(-1),
        "test_labels": np.asarray(d["labels_test"]).reshape(-1),
        "base_probs_val": np.asarray(d["base_probs_val"], dtype=np.float64),
        "base_probs_test": np.asarray(d["base_probs_test"], dtype=np.float64),
        "per_sample_path": per_sample,
    }
    # Alignment guard: the stored logits must reproduce the stored base probs.
    for split in ("val", "test"):
        recomputed = _softmax(out[f"{split}_logits"])
        stored = out[f"base_probs_{split}"]
        if recomputed.shape != stored.shape:
            raise ValueError(f"{split}: logits/base_probs shape mismatch")
        agree = float(np.mean(np.argmax(recomputed, axis=1) == np.argmax(stored, axis=1)))
        if agree < 0.999:
            raise ValueError(
                f"{split}: softmax(stored logits) disagrees with stored base_probs on "
                f"{100 * (1 - agree):.2f}% of rows -- arrays are not aligned"
            )
    return out


def inner_split(n: int, seed: int, fit_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic fit/select split of the calibration set, shared by all arms."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(fit_fraction * n))
    return np.sort(perm[:cut]), np.sort(perm[cut:])


# ------------------------------------------------------------------- arms
def build_arms(data: Dict[str, Any]) -> Dict[str, Any]:
    """Construct every arm's val/test feature matrices. No labels from test."""
    num_classes = int(data["base_probs_test"].shape[1])
    val_pred = np.argmax(data["base_probs_val"], axis=1)
    test_pred = np.argmax(data["base_probs_test"], axis=1)

    reg_val, reg_names = regular_features(data["val_logits"])
    reg_test, _ = regular_features(data["test_logits"])

    # --- reference banks: TRAIN split only, in each space -------------------
    cent_h = _class_centroids(data["train_features"], data["train_y"], num_classes)
    cent_z = _class_centroids(data["train_logits"], data["train_y"], num_classes)

    # Dimensionality control: fixed, non-learned Gaussian projection of the
    # hidden space down to the logit dimensionality. The projection matrix is
    # seeded and built without reference to any data, so it cannot leak.
    rng_proj = np.random.default_rng(PROJECTION_SEED)
    hidden_dim = int(data["train_features"].shape[1])
    proj = rng_proj.normal(size=(hidden_dim, num_classes)) / np.sqrt(hidden_dim)
    cent_hp = _class_centroids(np.asarray(data["train_features"], dtype=np.float64) @ proj,
                               data["train_y"], num_classes)

    dm = {
        "G_H": (
            _class_distance_matrix(data["val_features"], cent_h),
            _class_distance_matrix(data["test_features"], cent_h),
        ),
        "G_Z": (
            _class_distance_matrix(data["val_logits"], cent_z),
            _class_distance_matrix(data["test_logits"], cent_z),
        ),
        "G_Hproj": (
            _class_distance_matrix(np.asarray(data["val_features"], dtype=np.float64) @ proj, cent_hp),
            _class_distance_matrix(np.asarray(data["test_features"], dtype=np.float64) @ proj, cent_hp),
        ),
    }

    geo_val, geo_names = geometry_features(dm["G_H"][0], val_pred, "geo")
    geo_width = geo_val.shape[1]

    arms: Dict[str, Any] = {}

    # 1. R -- regular only (narrower input; that is why arm 2 exists).
    arms["R"] = {
        "val": reg_val, "test": reg_test,
        "names": list(reg_names),
        "geometry_source": None,
        "description": "regular output-only signals",
    }

    # 2. R + zero -- architectural control, full width, geometry block zeroed.
    arms["R_plus_zero"] = {
        "val": np.concatenate([reg_val, np.zeros((reg_val.shape[0], geo_width))], axis=1),
        "test": np.concatenate([reg_test, np.zeros((reg_test.shape[0], geo_width))], axis=1),
        "names": list(reg_names) + geo_names,
        "geometry_source": "zeros",
        "description": "architectural control: geometry block present but identically zero",
    }

    # 3. Geometry arms (true hidden, logit, projected hidden).
    for key, label in (("G_H", "R_plus_G_H"), ("G_Z", "R_plus_G_Z"), ("G_Hproj", "R_plus_G_Hproj")):
        gv, names_v = geometry_features(dm[key][0], val_pred, "geo")
        gt, _ = geometry_features(dm[key][1], test_pred, "geo")
        arms[label] = {
            "val": np.concatenate([reg_val, gv], axis=1),
            "test": np.concatenate([reg_test, gt], axis=1),
            "names": list(reg_names) + names_v,
            "geometry_source": key,
            "description": f"regular signals + {key} class-reference geometry",
        }

    # 4. Shuffled hidden geometry -- association control, several seeds.
    #    The permutation is drawn from a seeded RNG over row indices only; no
    #    labels and no correctness information are used to construct it.
    for shuffle_seed in SHUFFLE_SEEDS:
        rng = np.random.default_rng(shuffle_seed)
        gv_true, names_v = geometry_features(dm["G_H"][0], val_pred, "geo")
        gt_true, _ = geometry_features(dm["G_H"][1], test_pred, "geo")
        pv = rng.permutation(gv_true.shape[0])
        pt = rng.permutation(gt_true.shape[0])
        arms[f"R_plus_shuffledG_H_s{shuffle_seed}"] = {
            "val": np.concatenate([reg_val, gv_true[pv]], axis=1),
            "test": np.concatenate([reg_test, gt_true[pt]], axis=1),
            "names": list(reg_names) + names_v,
            "geometry_source": f"G_H_shuffled_seed{shuffle_seed}",
            "description": "association control: true G_H marginals, sample association destroyed",
        }

    return {
        "arms": arms,
        "distance_matrices": dm,
        "regular_names": reg_names,
        "geometry_names": geo_names,
        "num_classes": num_classes,
        "hidden_dim": hidden_dim,
    }


# -------------------------------------------------------------- evaluation
def paired_bootstrap_delta(
    conf_a: np.ndarray, conf_b: np.ndarray, correct: np.ndarray,
    *, n_boot: int = 2000, seed: int = 4242,
) -> Dict[str, Any]:
    """Paired bootstrap CI for (metric_b - metric_a) on the SAME samples.

    Resampling rows jointly preserves the pairing, which is what makes tiny
    deltas interpretable: the two arms are scored on identical resamples.
    """
    rng = np.random.default_rng(seed)
    n = correct.size
    d_auroc, d_aurc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        c = correct[idx]
        if c.min() == c.max():
            continue
        a_auroc = correctness_auroc(conf_a[idx], c)
        b_auroc = correctness_auroc(conf_b[idx], c)
        if a_auroc is None or b_auroc is None:
            continue
        d_auroc.append(b_auroc - a_auroc)
        d_aurc.append(
            float(np.mean(risk_coverage_curve(conf_b[idx], c)["risk"]))
            - float(np.mean(risk_coverage_curve(conf_a[idx], c)["risk"]))
        )
    def _ci(v):
        if not v:
            return {"mean": None, "ci_lo": None, "ci_hi": None, "p_gt_0": None}
        arr = np.asarray(v)
        return {
            "mean": float(arr.mean()),
            "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
            "p_gt_0": float(np.mean(arr > 0)),
        }
    return {"n_boot": int(n_boot), "seed": int(seed),
            "delta_auroc": _ci(d_auroc), "delta_aurc": _ci(d_aurc)}


def recoverability_diagnostic(
    distance_matrix_test: np.ndarray, base_pred: np.ndarray, labels: np.ndarray,
) -> Dict[str, Any]:
    """Can geometry alone rank the ground-truth class for BASE-MODEL ERRORS?

    DIAGNOSTIC ONLY -- never used for fitting or model selection. Conservative
    terminology: a base error whose ground truth geometry ranks first is
    called "geometry-recoverable", which is a statement about this statistic
    on this split, not a claim about where the model failed.
    """
    d = np.asarray(distance_matrix_test, dtype=np.float64)
    wrong = base_pred != labels
    n_wrong = int(wrong.sum())
    if n_wrong == 0:
        return {"n_base_errors": 0}
    order = np.argsort(d[wrong], axis=1)
    gt = labels[wrong]
    gt_rank = np.argmax(order == gt[:, None], axis=1)  # 0-based
    right = ~wrong
    order_r = np.argsort(d[right], axis=1)
    gt_r = labels[right]
    gt_rank_right = np.argmax(order_r == gt_r[:, None], axis=1)
    return {
        "n_base_errors": n_wrong,
        "n_base_correct": int(right.sum()),
        "geometry_recoverable_top1_rate": float(np.mean(gt_rank == 0)),
        "geometry_recoverable_top5_rate": float(np.mean(gt_rank < 5)),
        "median_gt_rank_on_errors": float(np.median(gt_rank)),
        "mean_gt_rank_on_errors": float(np.mean(gt_rank)),
        "geometry_top1_rate_on_base_correct": float(np.mean(gt_rank_right == 0)),
        "note": "diagnostic only; not used for fitting or model selection",
    }


def run_experiment(run_dir: str, *, n_boot: int = 2000) -> Dict[str, Any]:
    data = load_run_arrays(run_dir)
    built = build_arms(data)
    arms = built["arms"]

    val_labels = data["val_labels"]
    test_labels = data["test_labels"]
    val_pred = np.argmax(data["base_probs_val"], axis=1)
    test_pred = np.argmax(data["base_probs_test"], axis=1)
    # THE fixed target: base-model correctness. Never a method's own prediction.
    correct_val = (val_pred == val_labels).astype(np.float64)
    correct_test = (test_pred == test_labels).astype(np.float64)

    fit_idx, sel_idx = inner_split(len(val_labels), INNER_SPLIT_SEED, INNER_FIT_FRACTION)

    results: Dict[str, Any] = {}
    confidences: Dict[str, np.ndarray] = {}
    for name, arm in arms.items():
        learner = ReliabilityLearner(seed=LEARNER_SEED)
        fit_info = learner.fit(
            arm["val"][fit_idx], correct_val[fit_idx],
            arm["val"][sel_idx], correct_val[sel_idx],
        )
        conf = learner.predict_confidence(arm["test"])
        confidences[name] = conf
        metrics = evaluate_scalar_confidence(conf, correct_test)
        results[name] = {
            "description": arm["description"],
            "geometry_source": arm["geometry_source"],
            "input_dim": int(arm["val"].shape[1]),
            "n_features": len(arm["names"]),
            "metrics": metrics,
            "fit_info": {k: v for k, v in fit_info.items() if k != "selection_results"},
            "selection_results": fit_info["selection_results"],
            "initial_parameter_fingerprint": learner.initial_parameter_fingerprint(
                int(arm["val"].shape[1])
            ),
            # Effective prediction is the base prediction, by construction.
            "effective_argmax_change_rate": 0.0,
            "effective_prediction_source": "base",
            "metric_bucket": "scalar_only",
        }

    # ---- paired deltas vs R and vs the architectural control --------------
    comparisons: Dict[str, Any] = {}
    for reference in ("R", "R_plus_zero"):
        ref_conf = confidences[reference]
        block: Dict[str, Any] = {}
        for name, conf in confidences.items():
            if name == reference:
                continue
            block[name] = {
                "delta_auroc": float(results[name]["metrics"]["correctness_auroc"]
                                     - results[reference]["metrics"]["correctness_auroc"]),
                "delta_aurc": float(results[name]["metrics"]["aurc"]
                                    - results[reference]["metrics"]["aurc"]),
                "delta_excess_aurc": float(results[name]["metrics"]["excess_aurc"]
                                           - results[reference]["metrics"]["excess_aurc"]),
                "coverage_at_matched_risk": coverage_at_matched_risk(
                    ref_conf, correct_test, conf, correct_test, reference_coverage=0.8
                ),
                "paired_bootstrap": paired_bootstrap_delta(
                    ref_conf, conf, correct_test, n_boot=n_boot
                ),
            }
        comparisons[reference] = block

    # ---- Stage 14: recoverability diagnostic ------------------------------
    recoverability = {
        space: recoverability_diagnostic(built["distance_matrices"][space][1], test_pred, test_labels)
        for space in ("G_H", "G_Z", "G_Hproj")
    }

    payload: Dict[str, Any] = {
        "experiment_version": EXPERIMENT_VERSION,
        "run_dir": os.path.abspath(run_dir),
        "target": "C_base = (argmax(base_probs) == y); no arm changes a prediction",
        "base_accuracy_test": float(correct_test.mean()),
        "base_accuracy_val": float(correct_val.mean()),
        "arms": results,
        "comparisons": comparisons,
        "recoverability_diagnostic": recoverability,
        "information_budget": {
            "reference_bank": "train split class centroids (labels used; test never used)",
            "reference_bank_n": int(data["train_y"].shape[0]),
            "learner_fit_n": int(fit_idx.size),
            "learner_select_n": int(sel_idx.size),
            "evaluation_n": int(test_labels.size),
            "inner_split_seed": INNER_SPLIT_SEED,
            "inner_fit_fraction": INNER_FIT_FRACTION,
            "learner_seed": LEARNER_SEED,
            "shuffle_seeds": list(SHUFFLE_SEEDS),
            "projection_seed": PROJECTION_SEED,
            "hidden_dim": built["hidden_dim"],
            "logit_dim": built["num_classes"],
            "regular_feature_names": built["regular_names"],
            "geometry_feature_names": built["geometry_names"],
            "hyperparameter_trials_per_arm": len(ReliabilityLearner().weight_decay_grid),
            "corruption_specific_fitting": False,
            "test_labels_used_for_fitting": False,
        },
        "provenance": {
            "refit_of_classifier": False,
            "source_per_sample": data["per_sample_path"],
            "source_per_sample_sha256": _sha256(data["per_sample_path"]),
            **_git_provenance(str(Path(__file__).resolve().parents[1])),
        },
    }
    payload["_confidences"] = confidences
    payload["_correct_test"] = correct_test
    return payload


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_provenance(repo_dir: str) -> Dict[str, Any]:
    import subprocess
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
    return {"git_commit": commit, "git_dirty": dirty,
            "git_diff_hash": hashlib.sha256(diff).hexdigest() if dirty else None}


def write_results(payload: Dict[str, Any], out_dir: str, *, force: bool = False) -> str:
    if os.path.exists(out_dir) and not force:
        raise FileExistsError(f"{out_dir} already exists; pass --force")
    os.makedirs(out_dir, exist_ok=True)
    confidences = payload.pop("_confidences")
    correct = payload.pop("_correct_test")
    np.savez_compressed(
        os.path.join(out_dir, "per_sample_reliability.npz"),
        correct_base_test=correct.astype(np.int8),
        **{f"confidence__{k}": v for k, v in confidences.items()},
    )
    with open(os.path.join(out_dir, "reliability_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = run_experiment(args.run_dir, n_boot=args.n_boot)
    out_dir = args.out_dir or os.path.join(args.run_dir, "derived", EXPERIMENT_VERSION)
    print(f"Wrote {write_results(payload, out_dir, force=args.force)}")


if __name__ == "__main__":
    main()
