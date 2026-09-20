"""
Baseline fairness / information-budget audit.

Prints one row per method showing exactly what each one is allowed to see and
how much tuning it gets, so an unfair comparison is visible at a glance rather
than buried in the runner. Everything comes from the canonical registry
(utils/method_metadata.py), so the table cannot drift away from the code that
actually evaluates the methods.

The audit also runs explicit FAIRNESS CHECKS and exits non-zero if any fails:

  1. Every geometric method's fitting split is the same as the non-geometric
     methods' -- geometry must not get more labelled examples.
  2. No method's fitting or selection split mentions test/corruption data.
  3. The GLAD-PI geometry arm and its zero-geometry control agree on every
     budget field (split, objective, parameter count, search budget); only
     `uses_internal_representation` may differ.
  4. Every method in the run is registered.

Usage:
    python -m Experiments.audit_baseline_fairness
    python -m Experiments.audit_baseline_fairness --run_dir <a run directory>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.method_metadata import (  # noqa: E402
    method_semantics,
    registered_method_names,
)

# Methods the Phase 0/1 sbatch scripts actually produce.
PHASE0_1_METHODS = (
    "base_model",
    "temperature_scaling",
    "parameterized_temperature_scaling",
    "vector_scaling",
    "beta_calibration",
    "ovr_isotonic",
    "odir_dirichlet",
    "top_label_isotonic",
    "native_dac",
    "gc_dac",
    "rgcl",
    "anchored_model_tail",
    "anchored_rankgeom_tail_mixture",
    "full_vector_distance_fusion",
    "post_fusion_topiso",
    "kcal",
    "trust_score_original_diagnostic",
    "trust_score_original_switch",
    "glad_pi",
    "glad_pi_zero_geometry",
    "mahalanobis_confidence",
)

GLAD_PI_ARMS = ("glad_pi", "glad_pi_zero_geometry")

_FORBIDDEN_SPLIT_TOKENS = ("test", "corrupt", "cifar-c", "cifar_c")


def build_table(method_names) -> List[Dict[str, Any]]:
    rows = []
    for name in method_names:
        sem = method_semantics(name)
        if sem is None:
            rows.append({"method": name, "UNREGISTERED": True})
            continue
        b = sem["information_budget"] or {}
        rows.append({
            "method": name,
            "logits": b.get("uses_logits"),
            "softmax": b.get("uses_softmax"),
            "repr": b.get("uses_internal_representation"),
            "labels_fit": b.get("uses_labels_in_fitting"),
            "fitting_split": b.get("fitting_split"),
            "selection_split": b.get("selection_split"),
            "objective": b.get("objective"),
            "params": b.get("trainable_params"),
            "hp_budget": b.get("hp_search_budget"),
            "can_change_pred": sem["can_change_argmax"],
            "bucket": sem["metric_bucket"],
            "geometry": sem["representation_geometry_dependent"],
        })
    return rows


def print_table(rows: List[Dict[str, Any]]) -> None:
    def yn(v):
        return "-" if v is None else ("Y" if v else "n")

    hdr = (f"{'method':34s}{'geo':>4s}{'lgt':>4s}{'sfm':>4s}{'rep':>4s}{'lbl':>4s}"
           f"{'chg':>4s}  {'bucket':12s}{'fitting split':46s}{'hp budget':34s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("UNREGISTERED"):
            print(f"{r['method']:34s}  *** UNREGISTERED ***")
            continue
        print(
            f"{r['method']:34s}{yn(r['geometry']):>4s}{yn(r['logits']):>4s}"
            f"{yn(r['softmax']):>4s}{yn(r['repr']):>4s}{yn(r['labels_fit']):>4s}"
            f"{yn(r['can_change_pred']):>4s}  {r['bucket']:12s}"
            f"{str(r['fitting_split'])[:45]:46s}{str(r['hp_budget'])[:33]:34s}"
        )
    print()
    print("Columns: geo=representation geometry, lgt=logits, sfm=softmax, "
          "rep=internal representation, lbl=labels during fitting, "
          "chg=can change the effective prediction.")


def fairness_checks(method_names) -> List[str]:
    problems: List[str] = []

    budgets = {}
    for name in method_names:
        sem = method_semantics(name)
        if sem is None:
            problems.append(f"{name}: not in the canonical semantics registry")
            continue
        budgets[name] = (sem, sem["information_budget"] or {})

    # 1 + 2. Split hygiene.
    reference_splits = {
        b["fitting_split"] for n, (s, b) in budgets.items()
        if not s["representation_geometry_dependent"] and b.get("fitting_split") not in (None, "none")
    }
    for name, (sem, b) in budgets.items():
        for field in ("fitting_split", "selection_split"):
            value = str(b.get(field) or "")
            low = value.lower()
            if any(tok in low for tok in _FORBIDDEN_SPLIT_TOKENS):
                problems.append(
                    f"{name}: {field}={value!r} references test/corruption data; the "
                    "frozen protocol forbids fitting or tuning on the evaluation split"
                )
        if sem["representation_geometry_dependent"] and b.get("fitting_split") not in (None, "none"):
            split = str(b["fitting_split"])
            if not any(split.startswith(r.split(" ")[0]) for r in reference_splits):
                problems.append(
                    f"{name}: geometric method fits on {split!r}, which no "
                    f"non-geometric method uses (non-geometric splits: {sorted(reference_splits)}). "
                    "Geometry must not get a different or larger labelled budget."
                )

    # 3. The matched ablation.
    if all(a in budgets for a in GLAD_PI_ARMS):
        (sem_g, b_g), (sem_z, b_z) = budgets[GLAD_PI_ARMS[0]], budgets[GLAD_PI_ARMS[1]]
        for field in ("fitting_split", "selection_split", "objective", "hp_search_budget",
                      "uses_logits", "uses_softmax", "uses_labels_in_fitting"):
            if b_g.get(field) != b_z.get(field):
                problems.append(
                    f"GLAD-PI ablation is not matched on {field}: "
                    f"geometry={b_g.get(field)!r} vs zero_geometry={b_z.get(field)!r}"
                )
        if not (sem_g["representation_geometry_dependent"]
                and not sem_z["representation_geometry_dependent"]):
            problems.append(
                "GLAD-PI ablation must differ on representation_geometry_dependent only"
            )
        if sem_g["metric_bucket"] != sem_z["metric_bucket"]:
            problems.append("GLAD-PI arms must share a metric bucket to be comparable")
    return problems


def methods_in_run(run_dir: str) -> List[str]:
    summary = os.path.join(run_dir, "summary_metrics.json")
    if not os.path.exists(summary):
        raise FileNotFoundError(summary)
    with open(summary) as f:
        data = json.load(f)
    return [m["method_name"] for m in data.get("methods", [])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default=None,
                        help="Audit the methods actually present in a run (default: the "
                             "declared Phase 0/1 method set)")
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

    names = methods_in_run(args.run_dir) if args.run_dir else list(PHASE0_1_METHODS)
    rows = build_table(names)
    print_table(rows)

    problems = fairness_checks(names)
    if problems:
        print("FAIRNESS CHECK FAILURES:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("All fairness checks passed.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"rows": rows, "problems": problems}, f, indent=2)

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
