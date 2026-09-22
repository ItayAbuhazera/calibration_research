"""
Aggregate the layer-selection pilot and apply the frozen §9 decision rule.

Reads  results/layer_pilot/checkpoint_seed{S}/{cell}/cell_metrics.json
Writes results/layer_pilot/aggregate/{per_cell.csv, per_seed_arm.csv,
       by_corruption.csv, by_severity.csv, criteria.json, verdict.json}

Nothing here is tuned: the criteria are exactly docs/layer_selection_pilot_spec.md §9,
copied into CRITERIA below. Two seeds support no robustness claim, and no
significance test is computed (the 24 seed x cell pairs are not independent:
one reference bank and the same 10 000 images per seed).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

# Frozen thresholds (spec §9). Engineering decisions, not statistical thresholds.
CRITERIA = {
    "P1_mean_delta_acc_pp_seed_avg": 1.0,
    "P1_mean_delta_acc_pp_each_seed": 0.5,
    "P2_margin_over_controls_pp": 0.25,
    "P3_min_cells_with_W_gt_H": 20,
    "P3_total_cells": 24,
    "P4_ece_worse_max": 0.02,
    "P4_nll_worse_max": 0.05,
    "P5_permuted_max_fraction_of_arm": 0.5,
    "none_seed_avg_pp": 0.5,
    "none_each_seed_pp": 0.25,
}
CANDIDATE_ARMS = [f"{f}_greedy_L{L}" for f in ("A", "B") for L in (4, 6, 8)]
CELL_ORDER = ["clean"] + [f"{c}_s{s}" for c in ("gaussian_noise", "defocus_blur", "fog", "jpeg_compression") for s in (1, 3, 5)]


def load(root: str, seeds: List[int]) -> Dict[int, Dict[str, Any]]:
    data: Dict[int, Dict[str, Any]] = {}
    for s in seeds:
        data[s] = {}
        for cell in CELL_ORDER:
            p = os.path.join(root, f"checkpoint_seed{s}", cell, "cell_metrics.json")
            if os.path.exists(p):
                with open(p) as f:
                    data[s][cell] = json.load(f)
    return data


def cells_only(seed_data: Dict[str, Any]) -> List[str]:
    return [c for c in CELL_ORDER if c != "clean" and c in seed_data]


def arm_mean(seed_data, arm, key, cells):
    vals = []
    for c in cells:
        a = seed_data[c]["arms"][arm]
        v = a["delta_accuracy"] if key == "delta_accuracy" else a["metrics"][key]
        vals.append(v)
    return float(np.mean(vals))


def summarize(data: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    seeds = sorted(data)
    arms = list(next(iter(next(iter(data.values())).values()))["arms"])
    out: Dict[str, Any] = {"seeds": seeds, "arms": {}}
    for arm in arms:
        rec: Dict[str, Any] = {"per_seed": {}}
        for s in seeds:
            cs = cells_only(data[s])
            if not cs:
                continue
            W = sum(data[s][c]["arms"][arm].get("flips", {}).get("W", 0) for c in cs)
            H = sum(data[s][c]["arms"][arm].get("flips", {}).get("H", 0) for c in cs)
            U = sum(data[s][c]["arms"][arm].get("flips", {}).get("U", 0) for c in cs)
            n = sum(data[s][c]["n"] for c in cs)
            rec["per_seed"][s] = {
                "n_cells": len(cs),
                "delta_acc_pp": 100 * arm_mean(data[s], arm, "delta_accuracy", cs),
                "acc": arm_mean(data[s], arm, "accuracy", cs),
                "nll": arm_mean(data[s], arm, "nll", cs),
                "brier": arm_mean(data[s], arm, "brier", cs),
                "ece": arm_mean(data[s], arm, "top_label_ece", cs),
                "adaptive_ece": arm_mean(data[s], arm, "adaptive_ece", cs),
                "W": W, "H": H, "U": U, "net": W - H,
                "decisive_precision": (W / (W + H)) if (W + H) else None,
                "intervention_precision": (W / (W + H + U)) if (W + H + U) else None,
                "cells_W_gt_H": sum(
                    1 for c in cs
                    if data[s][c]["arms"][arm].get("flips", {}).get("W", 0) > data[s][c]["arms"][arm].get("flips", {}).get("H", 0)
                ),
                "clean_delta_acc_pp": (100 * data[s]["clean"]["arms"][arm]["delta_accuracy"]) if "clean" in data[s] else None,
                "n_samples": n,
            }
        ps = rec["per_seed"]
        if ps:
            rec["delta_acc_pp_seed_avg"] = float(np.mean([v["delta_acc_pp"] for v in ps.values()]))
            for k in ("nll", "ece", "brier", "adaptive_ece", "acc"):
                rec[f"{k}_seed_avg"] = float(np.mean([v[k] for v in ps.values()]))
            rec["cells_W_gt_H_total"] = int(sum(v["cells_W_gt_H"] for v in ps.values()))
            rec["cells_total"] = int(sum(v["n_cells"] for v in ps.values()))
        out["arms"][arm] = rec
    return out




def apply_rule(summ: Dict[str, Any]) -> Dict[str, Any]:
    A = summ["arms"]
    seeds = summ["seeds"]
    nat = A["native_dac"]
    vs = A["vector_scaling"]
    logit_probe = A["B_logit_probe"]
    results = {}
    for arm in CANDIDATE_ARMS:
        r = A[arm]
        fam = arm[0]
        sa = r["delta_acc_pp_seed_avg"]
        each = [r["per_seed"][s]["delta_acc_pp"] for s in seeds if s in r["per_seed"]]
        crit = {
            "P1": bool(sa >= CRITERIA["P1_mean_delta_acc_pp_seed_avg"] and min(each) >= CRITERIA["P1_mean_delta_acc_pp_each_seed"]),
            "P2": bool(
                sa - vs["delta_acc_pp_seed_avg"] >= CRITERIA["P2_margin_over_controls_pp"]
                and sa - logit_probe["delta_acc_pp_seed_avg"] >= CRITERIA["P2_margin_over_controls_pp"]
            ),
            "P3": bool(r["cells_W_gt_H_total"] >= CRITERIA["P3_min_cells_with_W_gt_H"] * len(seeds) / 2),
            "P4": bool(
                r["ece_seed_avg"] - nat["ece_seed_avg"] <= CRITERIA["P4_ece_worse_max"]
                and r["nll_seed_avg"] - nat["nll_seed_avg"] <= CRITERIA["P4_nll_worse_max"]
            ),
        }
        if fam == "A":
            perm = A[arm.replace("A_greedy", "A_permuted_greedy")]["delta_acc_pp_seed_avg"]
            crit["P5"] = bool(perm <= CRITERIA["P5_permuted_max_fraction_of_arm"] * sa or perm <= 0)
        else:
            crit["P5"] = None  # not applicable to Family B
        crit["promising"] = bool(all(v for v in crit.values() if v is not None))
        crit["reaches_none_floor"] = bool(
            sa >= CRITERIA["none_seed_avg_pp"] and min(each) >= CRITERIA["none_each_seed_pp"]
        )
        crit["delta_acc_pp_seed_avg"] = sa
        crit["delta_acc_pp_each_seed"] = each
        results[arm] = crit

    if any(c["promising"] for c in results.values()):
        verdict = "promising_diagnostic_requires_replication"
    else:
        floor = [a for a, c in results.items() if c["reaches_none_floor"]]
        healthy = [a for a in floor if results[a]["P4"] and (results[a]["P5"] in (True, None))]
        verdict = "no_material_evidence_to_continue_tested_family" if not healthy else "inconclusive"
    return {"criteria": CRITERIA, "candidate_arms": results, "verdict": verdict}


def write_tables(data, summ, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    seeds = sorted(data)
    arms = list(summ["arms"])
    with open(os.path.join(out_dir, "per_cell.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "cell", "arm", "acc", "delta_acc", "nll", "brier", "ece", "adaptive_ece", "W", "H", "U", "net",
                    "decisive_precision", "intervention_precision", "frac_base_errors_repaired", "gt_rank_base", "gt_rank_method",
                    "flip_enrichment_vs_margin_matched"])
        for s in seeds:
            for cell, d in data[s].items():
                for arm in arms:
                    a = d["arms"][arm]
                    fl = a.get("flips", {})
                    m = a["metrics"]
                    w.writerow([s, cell, arm, m["accuracy"], a["delta_accuracy"], m["nll"], m["brier"], m["top_label_ece"], m["adaptive_ece"],
                                fl.get("W"), fl.get("H"), fl.get("U"), fl.get("net"), fl.get("decisive_precision"),
                                fl.get("intervention_precision"), fl.get("frac_base_errors_repaired"),
                                fl.get("gt_rank_mean_base"), fl.get("gt_rank_mean_method"),
                                a.get("margin_matched_enrichment", {}).get("enrichment_ratio")])
    with open(os.path.join(out_dir, "per_seed_arm.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "n_cells", "delta_acc_pp", "acc", "nll", "brier", "ece", "adaptive_ece", "W", "H", "U", "net",
                    "decisive_precision", "intervention_precision", "cells_W_gt_H", "clean_delta_acc_pp"])
        for arm in arms:
            for s, v in summ["arms"][arm]["per_seed"].items():
                w.writerow([arm, s] + [v[k] for k in ("n_cells", "delta_acc_pp", "acc", "nll", "brier", "ece", "adaptive_ece", "W", "H", "U", "net",
                                                        "decisive_precision", "intervention_precision", "cells_W_gt_H", "clean_delta_acc_pp")])
    for name, keyfn in (("by_corruption", lambda c: c.rsplit("_s", 1)[0]), ("by_severity", lambda c: "s" + c.rsplit("_s", 1)[1])):
        agg = defaultdict(list)
        for s in seeds:
            for cell, d in data[s].items():
                if cell == "clean":
                    continue
                for arm in arms:
                    agg[(arm, keyfn(cell))].append(d["arms"][arm]["delta_accuracy"])
        with open(os.path.join(out_dir, f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arm", "group", "n_seed_cells", "mean_delta_acc_pp"])
            for (arm, g), v in sorted(agg.items()):
                w.writerow([arm, g, len(v), 100 * float(np.mean(v))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/layer_pilot")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 4])
    args = ap.parse_args()
    data = load(args.root, args.seeds)
    for s in args.seeds:
        missing = [c for c in CELL_ORDER if c not in data[s]]
        if missing:
            raise SystemExit(f"seed {s} is missing cells {missing}; refusing to aggregate an incomplete pilot")
    summ = summarize(data)
    verdict = apply_rule(summ)
    out = os.path.join(args.root, "aggregate")
    write_tables(data, summ, out)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1, default=float)
    print(json.dumps(verdict, indent=1, default=float))


if __name__ == "__main__":
    main()
