#!/usr/bin/env python
"""
Aggregate FV-DAC per-cell artifacts into versioned summaries.

Reads only `results/fv_dac/evaluation/checkpoint_seed*/<cell>/fv_dac_metrics.json`
(and the matching per-sample NPZ for paired bootstraps). It never recomputes a
method, never fills a missing cell, and never writes under `results/studyAB/`.

Outputs (under --output_dir, default results/fv_dac/aggregate):

  per_cell.csv                 one row per checkpoint x cell x method
  by_severity.csv              mean DeltaAcc by severity, per method
  by_corruption.csv            mean DeltaAcc by corruption, per method
  by_seed.csv                  per-checkpoint corruption means, per method
  summary.json                 the same, plus coverage and missing cells
  continuation_rule.json       mechanical application of the FROZEN rule
                               (--apply_continuation_rule only)

The continuation rule is NOT re-derived here: it is transcribed from
docs/full_vector_dac_experiment.md §11 / the vault preregistration, which were
both written before any CIFAR-C FV-DAC result existed. This script only
evaluates it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Calibrators.full_vector_dac import ARM_PRIMARY, ARM_SHARED_LAYER, FITTED_ARMS  # noqa: E402
from utils.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

#: The 12-cell POC grid, frozen before any FV-DAC corruption result.
POC_CORRUPTIONS = ("gaussian_noise", "defocus_blur", "fog", "jpeg_compression")
PRIMARY_SEVERITIES = (1, 3, 5)

METRIC_COLUMNS = (
    "accuracy", "nll", "brier", "top_label_ece", "adaptive_ece", "classwise_ece",
    "auroc", "aurc",
)
FLIP_COLUMNS = (
    "argmax_change_rate", "wrong_to_correct", "correct_to_wrong",
    "wrong_to_different_wrong", "net_useful_flips", "intervention_precision",
    "decisive_precision", "fraction_base_errors_repaired",
)


def _parse_cell(cell: str) -> Tuple[Optional[str], Optional[int]]:
    if cell == "clean":
        return None, None
    corruption, _, sev = cell.rpartition("_s")
    return corruption, int(sev)


def _metric(entry: Dict[str, Any], key: str) -> Optional[float]:
    """Pull a metric, tolerating the benchmark's several AUROC/AURC spellings."""
    metrics = entry.get("metrics") or {}
    if key in metrics:
        return metrics[key]
    for alt in (f"{key}_correctness", f"correctness_{key}", key.upper()):
        if alt in metrics:
            return metrics[alt]
    return None


def collect(evaluation_root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = Path(evaluation_root)
    if not root.exists():
        raise FileNotFoundError(f"no FV-DAC evaluation root at {root}")

    for ckpt_dir in sorted(root.glob("checkpoint_seed*")):
        seed = int(ckpt_dir.name.replace("checkpoint_seed", ""))
        for cell_dir in sorted(ckpt_dir.iterdir()):
            path = cell_dir / "fv_dac_metrics.json"
            if not path.exists():
                continue
            with open(path) as f:
                payload = json.load(f)
            meta = payload["metadata"]
            corruption, severity = _parse_cell(cell_dir.name)
            by_name = {m["method_name"]: m for m in payload["methods"]}
            base_acc = by_name["base_model"]["metrics"]["accuracy"]

            for entry in payload["methods"]:
                row: Dict[str, Any] = {
                    "checkpoint_seed": seed,
                    "cell": cell_dir.name,
                    "corruption": corruption or "clean",
                    "severity": severity,
                    "method": entry["method_name"],
                    "source": entry.get("source", "fv_dac_runner"),
                    "n": meta.get("n_samples"),
                    "selected_kc": meta.get("selected_kc"),
                    "state_hash": meta.get("fv_dac_state_hash"),
                }
                for key in METRIC_COLUMNS:
                    row[key] = _metric(entry, key)
                row["delta_accuracy_vs_base"] = (
                    None if row["accuracy"] is None else row["accuracy"] - base_acc
                )
                fa = entry.get("flip_analysis") or {}
                for key in FLIP_COLUMNS:
                    row[key] = fa.get(key)
                row["flip_identity_holds"] = fa.get("flip_identity_holds")
                params = entry.get("fitted_parameters") or {}
                row["beta"] = params.get("beta")
                row["b"] = json.dumps(params["b"]) if "b" in params else None
                row["n_fitted_parameters"] = entry.get("n_fitted_parameters")
                rows.append(row)

            # Frozen Phase 0/1 baseline rows, carried through unmodified.
            for entry in payload.get("reused_phase0_baselines", []):
                metrics = entry.get("metrics") or {}
                rows.append(
                    {
                        "checkpoint_seed": seed,
                        "cell": cell_dir.name,
                        "corruption": corruption or "clean",
                        "severity": severity,
                        "method": entry["method_name"],
                        "source": "phase0_frozen",
                        "n": meta.get("n_samples"),
                        **{k: metrics.get(k) for k in METRIC_COLUMNS},
                        "delta_accuracy_vs_base": (
                            None if metrics.get("accuracy") is None
                            else metrics["accuracy"] - base_acc
                        ),
                    }
                )

            diag = payload.get("density_only_diagnostic")
            if diag:
                rows.append(
                    {
                        "checkpoint_seed": seed,
                        "cell": cell_dir.name,
                        "corruption": corruption or "clean",
                        "severity": severity,
                        "method": diag["method_name"],
                        "source": "diagnostic",
                        "n": meta.get("n_samples"),
                        "accuracy": diag["accuracy"],
                        "delta_accuracy_vs_base": diag["accuracy"] - base_acc,
                        "agreement_with_base": diag.get("agreement_with_base"),
                        "agreement_with_fv_dac_primary": diag.get(
                            "agreement_with_fv_dac_primary"
                        ),
                        "recoverability_among_base_errors": diag.get(
                            "recoverability_among_base_errors"
                        ),
                    }
                )
    return rows


def paired_bootstrap_delta(
    npz_path: str, method: str, n_boot: int = 2000, seed: int = 20260920
) -> Optional[Dict[str, float]]:
    """Paired image bootstrap of DeltaAcc WITHIN one cell.

    Pairing matters: the same images are scored by base and by the method, so
    resampling images (not methods) is the only honest interval here. This is
    an image-level interval for one checkpoint -- it says nothing about
    between-checkpoint variability, which is reported separately and never
    pooled as if corrupted images were training-seed replicates.
    """
    if not os.path.exists(npz_path):
        return None
    with np.load(npz_path) as data:
        if f"pred__{method}" not in data:
            return None
        labels = data["labels"]
        base_ok = (data["base_pred"] == labels).astype(np.float64)
        method_ok = (data[f"pred__{method}"] == labels).astype(np.float64)
    diff = method_ok - base_ok
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    return {
        "delta_accuracy": float(diff.mean()),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "n_boot": int(n_boot),
    }


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        logger.warning("no rows for %s", path)
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def _corruption_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r["corruption"] != "clean" and r.get("delta_accuracy_vs_base") is not None]


def _group_mean(
    rows: List[Dict[str, Any]], keys: Tuple[str, ...], value: str
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[Any, ...], List[float]] = defaultdict(list)
    for r in rows:
        v = r.get(value)
        if v is None:
            continue
        buckets[tuple(r[k] for k in keys)].append(float(v))
    out = []
    for key, vals in sorted(buckets.items(), key=lambda kv: [str(x) for x in kv[0]]):
        arr = np.asarray(vals)
        out.append(
            {
                **dict(zip(keys, key)),
                f"mean_{value}": float(arr.mean()),
                f"std_{value}": float(arr.std(ddof=1)) if len(arr) > 1 else None,
                "n_cells": int(len(arr)),
                "fraction_positive": float((arr > 0).mean()),
            }
        )
    return out


# ==========================================================================
# Frozen continuation rule (transcribed, not re-derived)
# ==========================================================================


def apply_continuation_rule(
    rows: List[Dict[str, Any]], seed: int
) -> Dict[str, Any]:
    """Mechanically apply the FROZEN 12-cell continuation rule for one seed.

    Transcribed verbatim from docs/full_vector_dac_experiment.md §11, which was
    written before any FV-DAC CIFAR-C number existed. This function only
    computes C1-C5; it does not decide anything the rule did not already say.
    """
    poc = [
        r for r in rows
        if r["checkpoint_seed"] == seed
        and r["corruption"] in POC_CORRUPTIONS
        and r["severity"] in PRIMARY_SEVERITIES
    ]

    def arm_report(arm: str) -> Dict[str, Any]:
        cells = [r for r in poc if r["method"] == arm]
        deltas = np.array(
            [r["delta_accuracy_vs_base"] for r in cells if r["delta_accuracy_vs_base"] is not None]
        )
        w = int(sum(r.get("wrong_to_correct") or 0 for r in cells))
        h = int(sum(r.get("correct_to_wrong") or 0 for r in cells))
        betas = [r.get("beta") for r in cells if r.get("beta") is not None]
        per_family = {}
        for fam in POC_CORRUPTIONS:
            fam_d = [
                r["delta_accuracy_vs_base"] for r in cells
                if r["corruption"] == fam and r["delta_accuracy_vs_base"] is not None
            ]
            per_family[fam] = float(np.mean(fam_d)) if fam_d else None
        leave_one_out = {}
        for fam in POC_CORRUPTIONS:
            rest = [
                r["delta_accuracy_vs_base"] for r in cells
                if r["corruption"] != fam and r["delta_accuracy_vs_base"] is not None
            ]
            leave_one_out[f"without_{fam}"] = float(np.mean(rest)) if rest else None
        return {
            "arm": arm,
            "n_cells": int(len(deltas)),
            "mean_delta_accuracy": float(deltas.mean()) if len(deltas) else None,
            "cells_ge_zero": int((deltas >= 0).sum()),
            "cells_gt_zero": int((deltas > 0).sum()),
            "W": w, "H": h, "net": w - h,
            "beta": betas[0] if betas else None,
            "per_corruption_mean_delta": per_family,
            "leave_one_corruption_out_mean_delta": leave_one_out,
        }

    def nll_ece_health(arm: str) -> Dict[str, Any]:
        arm_cells = {r["cell"]: r for r in poc if r["method"] == arm}
        dac_cells = {r["cell"]: r for r in poc if r["method"] == "native_dac"}
        shared = sorted(set(arm_cells) & set(dac_cells))
        if not shared:
            return {"n_cells": 0}
        d_nll = np.array([arm_cells[c]["nll"] - dac_cells[c]["nll"] for c in shared])
        d_ece = np.array(
            [arm_cells[c]["top_label_ece"] - dac_cells[c]["top_label_ece"] for c in shared]
        )
        return {
            "n_cells": len(shared),
            "mean_nll_minus_native_dac": float(d_nll.mean()),
            "mean_top_label_ece_minus_native_dac": float(d_ece.mean()),
        }

    primary = arm_report(ARM_PRIMARY)
    permuted = arm_report("fv_dac_permuted")
    shared_layer = arm_report(ARM_SHARED_LAYER)
    health = nll_ece_health(ARM_PRIMARY)
    health_shared = nll_ece_health(ARM_SHARED_LAYER)

    def criteria(rep: Dict[str, Any], hlth: Dict[str, Any]) -> Dict[str, Any]:
        beta = rep["beta"]
        mean_d = rep["mean_delta_accuracy"]
        loo = [v for v in rep["leave_one_corruption_out_mean_delta"].values() if v is not None]
        c1 = beta is not None and beta > 0.0
        c2 = mean_d is not None and mean_d > 0 and rep["net"] > 0
        c3 = (
            rep["cells_ge_zero"] >= 7
            and rep["cells_gt_zero"] >= 5
            and bool(loo) and all(v > 0 for v in loo)
        )
        p_mean, a_mean = permuted["mean_delta_accuracy"], mean_d
        c4 = (
            p_mean is not None and a_mean is not None
            and (p_mean <= 0 or (a_mean > 0 and p_mean <= 0.5 * a_mean))
        )
        c5 = (
            hlth.get("mean_nll_minus_native_dac") is not None
            and hlth["mean_nll_minus_native_dac"] <= 0.05
            and hlth["mean_top_label_ece_minus_native_dac"] <= 0.02
        )
        return {
            "C1_beta_alive": bool(c1),
            "C2_direction": bool(c2),
            "C3_consistency": bool(c3),
            "C4_control_not_explanation": bool(c4),
            "C5_probabilistic_health": bool(c5),
            "all_satisfied": bool(c1 and c2 and c3 and c4 and c5),
        }

    primary_criteria = criteria(primary, health)
    shared_criteria = criteria(shared_layer, health_shared)

    if primary_criteria["all_satisfied"]:
        decision = "CONTINUE — full corruption extraction (primary arm)"
    elif shared_criteria["all_satisfied"]:
        decision = (
            "CONTINUE — shared-layer branch only (Outcome B); the shared-layer arm "
            "is reported as such and is NOT promoted to primary"
        )
    else:
        decision = "STOP — classify as currently negative / inconclusive"

    return {
        "rule_source": (
            "docs/full_vector_dac_experiment.md §11 and ResearchBrain/05_Experiments/"
            "2026-09-20 Full-Vector DAC POC.md, both written before any FV-DAC "
            "CIFAR-C result existed"
        ),
        "checkpoint_seed": seed,
        "poc_grid": {"corruptions": list(POC_CORRUPTIONS), "severities": list(PRIMARY_SEVERITIES)},
        "primary": primary,
        "primary_health_vs_native_dac": health,
        "primary_criteria": primary_criteria,
        "shared_layer": shared_layer,
        "shared_layer_health_vs_native_dac": health_shared,
        "shared_layer_criteria": shared_criteria,
        "permuted_control": permuted,
        "decision": decision,
    }


# ==========================================================================


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate FV-DAC results")
    p.add_argument(
        "--evaluation_root", type=str,
        default=str(PROJECT_ROOT / "results" / "fv_dac" / "evaluation"),
    )
    p.add_argument(
        "--output_dir", type=str,
        default=str(PROJECT_ROOT / "results" / "fv_dac" / "aggregate"),
    )
    p.add_argument("--bootstrap", action="store_true", help="paired image bootstrap per cell")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument(
        "--apply_continuation_rule", type=int, default=None, metavar="SEED",
        help="Evaluate the FROZEN 12-cell continuation rule for this checkpoint seed.",
    )
    args = p.parse_args()

    rows = collect(args.evaluation_root)
    if not rows:
        raise SystemExit(f"no FV-DAC cells found under {args.evaluation_root}")
    os.makedirs(args.output_dir, exist_ok=True)
    _write_csv(os.path.join(args.output_dir, "per_cell.csv"), rows)

    corr = _corruption_rows(rows)
    by_sev = _group_mean(corr, ("method", "severity"), "delta_accuracy_vs_base")
    by_corr = _group_mean(corr, ("method", "corruption"), "delta_accuracy_vs_base")
    by_seed = _group_mean(corr, ("method", "checkpoint_seed"), "delta_accuracy_vs_base")
    overall = _group_mean(corr, ("method",), "delta_accuracy_vs_base")
    _write_csv(os.path.join(args.output_dir, "by_severity.csv"), by_sev)
    _write_csv(os.path.join(args.output_dir, "by_corruption.csv"), by_corr)
    _write_csv(os.path.join(args.output_dir, "by_seed.csv"), by_seed)

    seeds = sorted({r["checkpoint_seed"] for r in rows})
    cells = sorted({(r["checkpoint_seed"], r["cell"]) for r in rows})
    summary: Dict[str, Any] = {
        "evaluation_root": args.evaluation_root,
        "checkpoint_seeds_present": seeds,
        "n_cells": len(cells),
        "cells_present": [f"seed{s}/{c}" for s, c in cells],
        "mean_delta_accuracy_over_corruption_cells": overall,
        "by_severity": by_sev,
        "by_corruption": by_corr,
        "by_seed": by_seed,
        "note": (
            "Corrupted images are NOT treated as independent training-seed "
            "replicates: checkpoint-level and corruption-level aggregation are "
            "reported separately, and every checkpoint seed is shown."
        ),
    }

    if args.bootstrap:
        boots = []
        for seed, cell in cells:
            npz = os.path.join(
                args.evaluation_root, f"checkpoint_seed{seed}", cell, "fv_dac_per_sample.npz"
            )
            for arm in FITTED_ARMS:
                res = paired_bootstrap_delta(npz, arm, n_boot=args.n_boot)
                if res:
                    boots.append({"checkpoint_seed": seed, "cell": cell, "method": arm, **res})
        _write_csv(os.path.join(args.output_dir, "paired_bootstrap.csv"), boots)
        summary["paired_bootstrap_rows"] = len(boots)

    _write_csv_summary(args.output_dir, summary)

    if args.apply_continuation_rule is not None:
        report = apply_continuation_rule(rows, args.apply_continuation_rule)
        path = os.path.join(args.output_dir, "continuation_rule.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Continuation rule -> %s", report["decision"])
        print(json.dumps(report, indent=2))

    logger.info("Wrote FV-DAC aggregate to %s (%d cells)", args.output_dir, len(cells))


def _write_csv_summary(output_dir: str, summary: Dict[str, Any]) -> None:
    path = os.path.join(output_dir, "summary.json")
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
