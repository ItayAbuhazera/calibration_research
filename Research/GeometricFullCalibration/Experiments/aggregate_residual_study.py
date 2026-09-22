"""
Aggregate the residual-evidence study and apply the frozen mechanical gates
(docs/residual_evidence_study_spec.md §7 and §8).

  python Experiments/aggregate_residual_study.py --root results/residual_study/dev --seeds 2 4 --mode dev
  python Experiments/aggregate_residual_study.py --root results/residual_study/confirm --seeds 1 3 5 --mode confirm

Nothing is tuned here. The best corruption row of any table is descriptive only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

DEV_FAMILIES = ("gaussian_noise", "defocus_blur", "fog", "jpeg_compression")
PRIMARY, CONTROL = "proc::hidden_decision", "proc::output_decision"
POLICIES = ("hidden_nll", "hidden_decision", "output_nll", "output_decision")

# frozen thresholds (spec §7 / §8) -- research-budget decisions, not theoretical constants
GATE = {"g1_macro_gain_over_base_pp": 0.50, "g2_margin_over_each_pp": 0.25, "g4_nll_worse_max": 0.02,
        "g4_ece_worse_max": 0.02, "g5_clean_acc_drop_max_pp": 0.20, "g5_clean_nll_rise_max": 0.01,
        "g3_min_families_improving": 3, "bootstrap_reps": 2000, "bootstrap_alpha": 0.05}


def load(root: str, seeds: List[int]):
    data: Dict[int, Dict[str, Any]] = {}
    for s in seeds:
        sd = os.path.join(root, f"checkpoint_seed{s}")
        st = json.load(open(os.path.join(sd, "frozen_state.json")))
        cells = {}
        for c in sorted(os.listdir(sd)):
            p = os.path.join(sd, c, "cell_metrics.json")
            if os.path.exists(p):
                cells[c] = json.load(open(p))
        data[s] = {"state": st, "cells": cells}
    return data


def fam(cell: str) -> str:
    return cell.rsplit("_s", 1)[0]


def macro(cells: Dict[str, Any], arm: str, key: str, corrupt_only=True, fams=None) -> float:
    vals = []
    for c, d in cells.items():
        if corrupt_only and c == "clean":
            continue
        if fams is not None and fam(c) not in fams:
            continue
        vals.append(d["metrics"][arm][key])
    return float(np.mean(vals))


def acc_gain_pp(cells, arm, ref="base_model", fams=None) -> float:
    return 100 * (macro(cells, arm, "accuracy", fams=fams) - macro(cells, ref, "accuracy", fams=fams))


def avg_over_seeds(fn, data):
    return float(np.mean([fn(data[s]) for s in data]))


def arm_names(data) -> List[str]:
    s0 = next(iter(data.values()))
    return list(next(iter(s0["cells"].values()))["metrics"])


def arm_table(data, fams=None) -> List[Dict[str, Any]]:
    rows = []
    for a in arm_names(data):
        r = {"arm": a}
        for s, d in data.items():
            r[f"acc_gain_pp_s{s}"] = acc_gain_pp(d["cells"], a, fams=fams)
        r["acc_gain_pp_avg"] = float(np.mean([r[f"acc_gain_pp_s{s}"] for s in data]))
        for k in ("nll", "brier", "top_label_ece", "adaptive_ece", "classwise_ece"):
            r[f"{k}_avg"] = avg_over_seeds(lambda d, k=k, a=a: macro(d["cells"], a, k, fams=fams), data)
        if a != "base_model":
            for k in ("W", "H", "U"):
                r[k] = int(sum(c["flips_vs_base"][a][k] for d in data.values() for cn, c in d["cells"].items()
                               if cn != "clean" and (fams is None or fam(cn) in fams)))
            r["clean_acc_gain_pp_avg"] = float(np.mean([100 * (d["cells"]["clean"]["metrics"][a]["accuracy"] -
                                                               d["cells"]["clean"]["metrics"]["base_model"]["accuracy"]) for d in data.values()]))
        rows.append(r)
    return rows


def gate(data, mode: str, stage0_ok: bool = True) -> Dict[str, Any]:
    seeds = sorted(data)
    P, C = PRIMARY, CONTROL
    out: Dict[str, Any] = {"mode": mode, "seeds": seeds, "criteria": {}}
    anchor = "anchor"
    gain = lambda a, ref="base_model": avg_over_seeds(lambda d: acc_gain_pp(d["cells"], a, ref), data)  # noqa: E731
    g1 = gain(P)
    out["criteria"]["1_macro_gain_over_base_pp"] = {"value": g1, "pass": bool(g1 >= GATE["g1_macro_gain_over_base_pp"])}
    over = {ref: gain(P, ref) for ref in ("vector_scaling", anchor, C)}
    out["criteria"]["2_margin_over_VS_anchor_outputDecision_pp"] = {
        "values": over, "pass": bool(all(v >= GATE["g2_margin_over_each_pp"] for v in over.values()))}
    per_seed = {s: acc_gain_pp(data[s]["cells"], P, C) for s in seeds}
    if mode == "dev":
        fam_imp = {}
        for f in DEV_FAMILIES:
            fam_imp[f] = float(np.mean([acc_gain_pp(data[s]["cells"], P, C, fams=(f,)) for s in seeds]))
        c3 = all(v > 0 for v in per_seed.values()) and sum(v > 0 for v in fam_imp.values()) >= GATE["g3_min_families_improving"]
        out["criteria"]["3_advantage_over_outputDecision_each_checkpoint_and_families"] = {
            "per_checkpoint_pp": per_seed, "per_family_pp": fam_imp, "pass": bool(c3)}
    else:
        c3 = all(v > 0 for v in per_seed.values())
        dev_f = {f: float(np.mean([acc_gain_pp(data[s]["cells"], P, C, fams=(f,)) for s in seeds])) for f in DEV_FAMILIES}
        held = sorted({fam(c) for s in seeds for c in data[s]["cells"] if c != "clean"} - set(DEV_FAMILIES))
        held_pp = float(np.mean([acc_gain_pp(data[s]["cells"], P, C, fams=tuple(held)) for s in seeds])) if held else None
        out["criteria"]["3_positive_over_outputDecision_in_every_checkpoint"] = {
            "per_checkpoint_pp": per_seed, "dev_families_pp": dev_f, "held_out_families_pp": held_pp,
            "n_held_out_families": len(held), "pass": bool(c3)}
    nat = lambda k: avg_over_seeds(lambda d: macro(d["cells"], "native_dac", k), data)  # noqa: E731
    prm = lambda k: avg_over_seeds(lambda d: macro(d["cells"], P, k), data)  # noqa: E731
    dn, de, db = prm("nll") - nat("nll"), prm("top_label_ece") - nat("top_label_ece"), prm("brier") - nat("brier")
    per_cond = {}
    for s in seeds:
        for c, d in data[s]["cells"].items():
            if c == "clean":
                continue
            per_cond[f"{s}:{c}"] = {k: d["metrics"][P][k] - d["metrics"]["native_dac"][k] for k in ("nll", "brier", "top_label_ece")}
    out["criteria"]["4_calibration_vs_native_dac"] = {
        "nll_worse": dn, "ece_worse": de, "brier_worse_reported": db,
        "n_conditions_nll_worse_by_gt_0.02": int(sum(v["nll"] > 0.02 for v in per_cond.values())),
        "n_conditions_brier_worse": int(sum(v["brier"] > 0 for v in per_cond.values())), "n_conditions": len(per_cond),
        "pass": bool(dn <= GATE["g4_nll_worse_max"] and de <= GATE["g4_ece_worse_max"])}
    cl = lambda a, k: avg_over_seeds(lambda d: d["cells"]["clean"]["metrics"][a][k], data)  # noqa: E731
    dacc, dnll = 100 * (cl(P, "accuracy") - cl(anchor, "accuracy")), cl(P, "nll") - cl(anchor, "nll")
    out["criteria"]["5_clean_vs_anchor"] = {"acc_change_pp": dacc, "nll_change": dnll,
                                            "pass": bool(dacc >= -GATE["g5_clean_acc_drop_max_pp"] and dnll <= GATE["g5_clean_nll_rise_max"])}
    conv = all(data[s]["state"]["all_converged"] for s in seeds)
    complete = all(len(data[s]["cells"]) >= (13 if mode == "dev" else 76) for s in seeds)
    ev_only = all(c["evaluation_only"] and c["preprocessing_protocol"] == "corrected_v2_train_norm"
                  for s in seeds for c in data[s]["cells"].values())
    controls = all(a in arm_names(data) for a in ("base_model", "temperature_scaling", "vector_scaling", "native_dac", "anchor", C))
    out["criteria"]["6_provenance_and_numerics"] = {
        "all_fits_converged": conv, "all_cells_present": complete, "evaluation_only_and_corrected_protocol": ev_only,
        "controls_present": controls, "stage0_reconciliation_ok": stage0_ok,
        "pass": bool(conv and complete and ev_only and controls and stage0_ok)}
    out["all_pass"] = bool(all(v["pass"] for v in out["criteria"].values()))
    out["failing"] = [k for k, v in out["criteria"].items() if not v["pass"]]
    return out


def bootstrap_primary_vs_control(root: str, seeds: List[int], reps: int, alpha: float, seed_rng: int = 20260925) -> Dict[str, Any]:
    """Paired image-level bootstrap. Per image: mean over (checkpoint x corruption cell) of
    1{primary correct} - 1{control correct}; resample the 10 000 original image IDs (shared by all
    checkpoints and cells; CIFAR-C cells are copies of the same test images)."""
    diffs = []
    cell_names = None
    for s in seeds:
        sd = os.path.join(root, f"checkpoint_seed{s}")
        cs = sorted(c for c in os.listdir(sd) if os.path.exists(os.path.join(sd, c, "per_sample.npz")) and c != "clean")
        cell_names = cs if cell_names is None else cell_names
        acc = np.zeros(10000)
        for c in cs:
            d = np.load(os.path.join(sd, c, "per_sample.npz"))
            y = d["labels"].astype(int)
            acc += (d[f"pred__{PRIMARY}"].astype(int) == y).astype(float) - (d[f"pred__{CONTROL}"].astype(int) == y)
        diffs.append(acc / len(cs))
    m = np.mean(diffs, axis=0)  # per image, averaged over checkpoints and cells (equal macro weights)
    rng = np.random.default_rng(seed_rng)
    n = len(m)
    boots = np.array([m[rng.integers(0, n, n)].mean() for _ in range(reps)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"contrast": f"{PRIMARY} - {CONTROL}", "estimate_pp": 100 * float(m.mean()), "ci95_pp": [100 * float(lo), 100 * float(hi)],
            "reps": reps, "n_images": int(n), "cells_per_checkpoint": len(cell_names),
            "note": "conditional on these checkpoints; image-level bootstrap only; not seed-population uncertainty; "
                    "not selection-adjusted for the research search"}


def matched_comparison(data) -> Dict[str, Any]:
    """Hidden vs output Decision procedures on identical images: paired 2x2 and true-class-rank change."""
    res = {}
    for s, d in data.items():
        sd = os.path.join(d["_root"], f"checkpoint_seed{s}")
        both = only_h = only_o = neither = 0
        drank, improved, worsened, n = 0.0, 0, 0, 0
        for c in d["cells"]:
            if c == "clean":
                continue
            z = np.load(os.path.join(sd, c, "per_sample.npz"))
            y = z["labels"].astype(int)
            h, o = z[f"pred__{PRIMARY}"].astype(int) == y, z[f"pred__{CONTROL}"].astype(int) == y
            both += int((h & o).sum()); only_h += int((h & ~o).sum()); only_o += int((~h & o).sum()); neither += int((~h & ~o).sum())
            rh, ro = z[f"gtrank__{PRIMARY}"].astype(int), z[f"gtrank__{CONTROL}"].astype(int)
            drank += float((rh - ro).sum()); improved += int((rh < ro).sum()); worsened += int((rh > ro).sum()); n += len(y)
        res[s] = {"both_correct": both, "only_hidden": only_h, "only_output": only_o, "neither": neither,
                  "mean_gt_rank_hidden_minus_output": drank / n, "frac_rank_better_hidden": improved / n,
                  "frac_rank_worse_hidden": worsened / n}
    return res


def family_repair_overlap(data) -> Dict[str, Any]:
    """Overlap of repaired examples between the family-level Decision-selected arms (G vs S, DG vs DS, etc.)."""
    out = {}
    for s, d in data.items():
        sd = os.path.join(d["_root"], f"checkpoint_seed{s}")
        sel = {f: v["decision"] for f, v in d["state"]["family_level_selection"].items()}
        acc = defaultdict(lambda: defaultdict(int))
        for c in d["cells"]:
            if c == "clean":
                continue
            z = np.load(os.path.join(sd, c, "per_sample.npz"))
            y = z["labels"].astype(int); base = z["pred__base_model"].astype(int)
            Wm = {f: (z[f"pred__{a}"].astype(int) != base) & (base != y) & (z[f"pred__{a}"].astype(int) == y) for f, a in sel.items()}
            for a, b in (("G", "S"), ("DG", "DS"), ("G", "O"), ("S", "O"), ("DG", "DL"), ("DS", "DL"), ("S", "DS")):
                acc[f"{a}|{b}"]["inter"] += int((Wm[a] & Wm[b]).sum()); acc[f"{a}|{b}"]["union"] += int((Wm[a] | Wm[b]).sum())
                acc[f"{a}_only_vs_{b}"]["n"] += int((Wm[a] & ~Wm[b]).sum()); acc[f"{b}_only_vs_{a}"]["n"] += int((Wm[b] & ~Wm[a]).sum())
            for f in Wm:
                acc[f"W_total::{f}"]["n"] += int(Wm[f].sum())
        out[s] = {"selected_family_arms": sel, "counts": {k: dict(v) for k, v in acc.items()}}
    return out


def policy_comparison(data) -> Dict[str, Any]:
    out = {}
    for s, d in data.items():
        cells = d["cells"]
        row = {}
        for pol in POLICIES:
            a = f"proc::{pol}"
            row[pol] = {"selected": d["state"]["selection"][pol],
                        "acc_gain_pp": acc_gain_pp(cells, a), "nll": macro(cells, a, "nll"), "ece": macro(cells, a, "top_label_ece"),
                        "brier": macro(cells, a, "brier"),
                        "net_flips_vs_anchor": int(sum(c["flips_vs_anchor"][a]["W"] - c["flips_vs_anchor"][a]["H"]
                                                       for cn, c in cells.items() if cn != "clean")),
                        "net_flips_vs_base": int(sum(c["flips_vs_base"][a]["W"] - c["flips_vs_base"][a]["H"]
                                                     for cn, c in cells.items() if cn != "clean"))}
        out[s] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--mode", choices=["dev", "confirm"], required=True)
    ap.add_argument("--stage0", default="results/residual_study/stage0_stage1/stage0_reconciliation.json")
    args = ap.parse_args()
    data = load(args.root, args.seeds)
    for s in data:
        data[s]["_root"] = args.root
    stage0_ok = True
    if args.mode == "dev" and os.path.exists(args.stage0):
        s0 = json.load(open(args.stage0))["seeds"]
        for s in args.seeds:
            r = s0[str(s)]["reconciliation"]
            stage0_ok &= bool(r["test_labels_identical_pilot_vs_benchmark"] and r["inner_fit_hash_matches_pilot"]
                              and r["native_dac_source_set_matches"] and r["benchmark_intermediates_protocol"] == "corrected_v2_train_norm")
    out_dir = os.path.join(args.root, "aggregate")
    os.makedirs(out_dir, exist_ok=True)
    g = gate(data, args.mode, stage0_ok)
    if args.mode == "confirm":
        b = bootstrap_primary_vs_control(args.root, args.seeds, GATE["bootstrap_reps"], GATE["bootstrap_alpha"])
        g["bootstrap"] = b
        g["criteria"]["7_paired_bootstrap_lower_bound_positive"] = {"ci95_pp": b["ci95_pp"], "pass": bool(b["ci95_pp"][0] > 0)}
        g["all_pass"] = bool(all(v["pass"] for v in g["criteria"].values()))
        g["failing"] = [k for k, v in g["criteria"].items() if not v["pass"]]
    json.dump(g, open(os.path.join(out_dir, "gate.json"), "w"), indent=1, default=float)
    tab = arm_table(data)
    with open(os.path.join(out_dir, "arm_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in tab for k in r}, key=lambda k: (k != "arm", k)))
        w.writeheader(); [w.writerow(r) for r in tab]
    extra = {"policy_comparison": policy_comparison(data), "matched_hidden_vs_output": matched_comparison(data),
             "family_repair_overlap": family_repair_overlap(data),
             "selections": {s: {"anchor": data[s]["state"]["anchor"]["chosen"], "procedures": data[s]["state"]["selection"],
                                "family_level": data[s]["state"]["family_level_selection"],
                                "state_hash": data[s]["state"]["state_hash"]} for s in data},
             "held_out_family_arm_table": arm_table(data, fams=tuple(sorted({fam(c) for s in data for c in data[s]["cells"] if c != "clean"} - set(DEV_FAMILIES)))) if args.mode == "confirm" else None,
             "dev_family_arm_table": arm_table(data, fams=DEV_FAMILIES)}
    json.dump(extra, open(os.path.join(out_dir, "details.json"), "w"), indent=1, default=float)
    print(json.dumps(g, indent=1, default=float))


if __name__ == "__main__":
    main()
