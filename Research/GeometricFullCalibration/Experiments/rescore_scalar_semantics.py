"""
Re-score an existing Phase 0/1 run under corrected scalar-confidence semantics.

WHY THIS EXISTS
---------------
Runs produced before SEMANTIC_SCHEMA_VERSION="scalar_semantics_v2" evaluated
scalar-confidence methods through their reconstructed NxC surrogate matrix:
top-label ECE, adaptive ECE and accuracy were taken from
`argmax(surrogate)` / `max(surrogate)` instead of the base prediction and the
calibrated scalar assigned to it. For the proportional-tail reconstructions
(top_label_isotonic, mahalanobis_confidence, Study-B cells) the surrogate
argmax can differ from the base prediction, so those metrics were wrong. They
also carried a multiclass NLL/Brier/classwise ECE that the frozen
BENCHMARK_IMPLEMENTATION_PLAN.md §6 table forbids for a scalar method.

NO REFITTING IS NEEDED to correct this. Every input required
(`base_probs_test`, `labels_test` and each method's stored probability
matrix) is already in the run's `per_sample/per_sample_arrays.npz`, so this
script is a pure re-scoring of stored outputs.

GUARANTEES
----------
  * READ-ONLY with respect to the source run: nothing under the source
    directory is written, moved or deleted.
  * Output goes to a NEW versioned directory (default:
    <run_dir>/derived/scalar_semantics_v2/) and the script refuses to
    overwrite an existing one unless --force is given.
  * Provenance records the source artifact path, its sha256, the git commit
    and dirty-diff hash of the re-scoring code, the semantic schema version,
    and an explicit `refit_occurred: false`.

Usage:
    python -m Experiments.rescore_scalar_semantics \\
        --run_dir results/studyAB/phase0/evaluation/checkpoint_seed4/clean
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.method_metadata import (  # noqa: E402
    PRED_BASE,
    SCALAR_CONFIDENCE,
    SCALAR_ONLY_BUCKET,
    SEMANTIC_SCHEMA_VERSION,
    method_semantics,
)
from utils.selective_metrics import (  # noqa: E402
    coverage_at_matched_risk,
    paired_complementarity,
    risk_coverage_curve,
)
from utils.unified_metrics import (  # noqa: E402
    evaluate_all,
    evaluate_scalar_confidence,
    scalar_confidence_from_surrogate,
)

DERIVED_SUBDIR = os.path.join("derived", SEMANTIC_SCHEMA_VERSION)

# The strongest NON-GEOMETRIC scalar confidence available in a Phase 0/1 run.
# Every geometric method is compared against this, not against raw MSP, so a
# geometric win has to clear a calibrated output-space baseline.
DEFAULT_REFERENCE_METHOD = "top_label_isotonic"


def _sha256_file(path: str) -> str:
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
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_diff_hash": hashlib.sha256(diff).hexdigest() if dirty else None,
    }


def rescore_run(
    run_dir: str,
    *,
    reference_method: str = DEFAULT_REFERENCE_METHOD,
    curve_points: int = 101,
) -> Dict[str, Any]:
    """Re-score one run directory. Returns the derived payload (does not write)."""
    npz_path = os.path.join(run_dir, "per_sample", "per_sample_arrays.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No per-sample artifact at {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    labels = np.asarray(data["labels_test"]).reshape(-1)
    base_probs = np.asarray(data["base_probs_test"], dtype=np.float64)
    base_pred = np.argmax(base_probs, axis=1)
    base_correct = (base_pred == labels).astype(np.float64)

    method_names = [str(m) for m in data["method_index"]]
    rows: Dict[str, Any] = {}
    unregistered: List[str] = []
    per_sample: Dict[str, np.ndarray] = {
        "labels_test": labels.astype(np.int64),
        "base_pred": base_pred.astype(np.int64),
        "base_correct": base_correct.astype(np.int8),
    }

    for name in method_names:
        key = f"method_probs__{name}"
        if key not in data:
            continue
        sem = method_semantics(name)
        if sem is None:
            unregistered.append(name)
            continue
        probs = np.asarray(data[key], dtype=np.float64)
        # Same two independent dispatches as the runner: the bucket decides
        # WHICH metrics may exist, the prediction source decides WHOSE
        # prediction/confidence are scored.
        is_scalar_bucket = sem["metric_bucket"] == SCALAR_ONLY_BUCKET
        uses_base = sem["effective_prediction_source"] == PRED_BASE

        row: Dict[str, Any] = {
            "method_name": name,
            "output_semantics": sem["output_semantics"],
            "effective_prediction_source": sem["effective_prediction_source"],
            "metric_bucket": sem["metric_bucket"],
            "can_change_argmax": sem["can_change_argmax"],
            "representation_geometry_dependent": sem["representation_geometry_dependent"],
            "sample_dependent": sem["sample_dependent"],
        }

        if uses_base:
            scalar = scalar_confidence_from_surrogate(probs, base_probs)
            effective_pred = base_pred
            confidence = scalar["confidence"]
            correct = base_correct
            row["surrogate_only"] = True
            row["surrogate_reconstruction"] = sem["surrogate_reconstruction"]
            row["surrogate_argmax_change_rate"] = float(
                np.mean(scalar["surrogate_pred"] != base_pred)
            )
            row["effective_argmax_change_rate"] = 0.0
        else:
            effective_pred = np.argmax(probs, axis=1)
            confidence = np.max(probs, axis=1)
            correct = (effective_pred == labels).astype(np.float64)
            row["surrogate_only"] = False
            row["surrogate_argmax_change_rate"] = None
            row["effective_argmax_change_rate"] = float(
                np.mean(effective_pred != base_pred)
            )
        row["metrics"] = (
            evaluate_scalar_confidence(confidence, correct)
            if is_scalar_bucket
            else evaluate_all(probs, labels)
        )

        curve = risk_coverage_curve(confidence, correct)
        n = curve["coverage"].size
        idx = np.unique(np.linspace(0, n - 1, min(curve_points, n)).astype(int))
        row["risk_coverage_curve"] = {
            "coverage": [float(v) for v in curve["coverage"][idx]],
            "risk": [float(v) for v in curve["risk"][idx]],
        }
        rows[name] = row
        per_sample[f"confidence__{name}"] = confidence.astype(np.float64)
        per_sample[f"correct__{name}"] = correct.astype(np.int8)
        per_sample[f"effective_pred__{name}"] = effective_pred.astype(np.int64)

    # ---- head-to-head vs the strongest non-geometric scalar baseline -------
    comparisons: Dict[str, Any] = {}
    if reference_method in rows:
        ref_conf = per_sample[f"confidence__{reference_method}"]
        ref_corr = per_sample[f"correct__{reference_method}"].astype(np.float64)
        ref_metrics = rows[reference_method]["metrics"]
        for name, row in rows.items():
            if name == reference_method:
                continue
            cand_conf = per_sample[f"confidence__{name}"]
            cand_corr = per_sample[f"correct__{name}"].astype(np.float64)
            cmp_row: Dict[str, Any] = {
                "reference_method": reference_method,
                "geometry_arm": bool(row["representation_geometry_dependent"]),
                "delta_top_label_ece": _delta(row["metrics"], ref_metrics, "top_label_ece"),
                "delta_adaptive_ece": _delta(row["metrics"], ref_metrics, "adaptive_ece"),
                "delta_correctness_auroc": _delta(
                    row["metrics"], ref_metrics, "correctness_auroc"
                ),
                "delta_aurc": _delta(row["metrics"], ref_metrics, "aurc"),
                "delta_excess_aurc": _delta(row["metrics"], ref_metrics, "excess_aurc"),
            }
            cmp_row["coverage_at_matched_risk"] = coverage_at_matched_risk(
                ref_conf, ref_corr, cand_conf, cand_corr, reference_coverage=0.8
            )
            # Oracle headroom / complementarity (§I): diagnostic only.
            if row["effective_argmax_change_rate"] == 0.0 and np.array_equal(
                cand_corr, ref_corr
            ):
                cmp_row["paired_complementarity"] = paired_complementarity(
                    ref_conf, cand_conf, ref_corr
                )
            else:
                cmp_row["paired_complementarity"] = {
                    "skipped": "methods disagree on the effective prediction, so a "
                               "paired per-sample confidence loss is not comparable"
                }
            comparisons[name] = cmp_row

    payload = {
        "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
        "run_dir": os.path.abspath(run_dir),
        "n_samples": int(labels.size),
        "base_accuracy": float(np.mean(base_correct)),
        "reference_method": reference_method,
        "methods": rows,
        "comparisons_vs_reference": comparisons,
        "unregistered_methods_skipped": unregistered,
        "provenance": {
            "refit_occurred": False,
            "note": "Pure re-scoring of stored per-sample outputs. No model was "
                    "loaded, no calibrator was fitted, and no fitted state was read.",
            "source_artifact": os.path.abspath(npz_path),
            "source_artifact_sha256": _sha256_file(npz_path),
            "source_summary_metrics": _optional_hash(
                os.path.join(run_dir, "summary_metrics.json")
            ),
            "source_fit_once_provenance": _optional_hash(
                os.path.join(run_dir, "fit_once_provenance.json")
            ),
            **_git_provenance(str(Path(__file__).resolve().parents[1])),
        },
    }
    payload["_per_sample"] = per_sample
    return payload


def _delta(candidate: Dict[str, Any], reference: Dict[str, Any], key: str) -> Optional[float]:
    a, b = candidate.get(key), reference.get(key)
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _optional_hash(path: str) -> Optional[Dict[str, str]]:
    if not os.path.exists(path):
        return None
    return {"path": os.path.abspath(path), "sha256": _sha256_file(path)}


def write_derived(payload: Dict[str, Any], out_dir: str, *, force: bool = False) -> str:
    if os.path.exists(out_dir) and not force:
        raise FileExistsError(
            f"{out_dir} already exists; pass --force to regenerate it. "
            "(This script never writes into the source run directory.)"
        )
    os.makedirs(out_dir, exist_ok=True)
    per_sample = payload.pop("_per_sample")
    np.savez_compressed(os.path.join(out_dir, "per_sample_scalar_semantics.npz"), **per_sample)
    with open(os.path.join(out_dir, "summary_metrics_rescored.json"), "w") as f:
        json.dump(payload, f, indent=2)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="A run directory containing per_sample/")
    parser.add_argument("--out_dir", default=None, help=f"Default: <run_dir>/{DERIVED_SUBDIR}")
    parser.add_argument("--reference_method", default=DEFAULT_REFERENCE_METHOD)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = rescore_run(args.run_dir, reference_method=args.reference_method)
    out_dir = args.out_dir or os.path.join(args.run_dir, DERIVED_SUBDIR)
    written = write_derived(payload, out_dir, force=args.force)
    print(f"Wrote {written}")


if __name__ == "__main__":
    main()
