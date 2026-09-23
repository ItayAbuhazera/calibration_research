"""Aggregation for the Stage 0 evidence ablation (docs/stage0_evidence_ablation_spec.md).

Pools the ablation out-of-fold predictions, computes Delta_T / Delta_S / gap / clean increment per
evidence source, the primary contrast D_A, the frozen rules R1-R3, standalone accuracy and
disagreement with Z, a q_Z determinism check against the Stage 0 outputs, and a convergence audit.

    python -m atlas.stage0_ablation_aggregate
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
from scipy.stats import spearmanr

from . import spec, stage0_data
from .stage0_aggregate import BOOT_SEED, group_id_for_bootstrap, grouped_bootstrap, grouped_bootstrap_diff_of_diffs
from .stage0_fit import LAMBDA_GRID

ABL = "results/stage0_ablation"
OUT = f"{ABL}/report"
LAYERS = [f"L{k}" for k in range(12)]
REF = "L8"  # layer3.22; Stage 0 outputs, reused
SOURCES = ["L11", "xckpt"] + [f"L{k}" for k in (0, 1, 2, 3, 4, 5, 6, 7, 9, 10)]
FULL_REGIMES = ("T-8k1", "S-8k1", "T-2.5k1", "S-2.5k1")
LAYER_REGIMES = ("T-8k1", "S-8k1")
EDGES = (LAMBDA_GRID[0], LAMBDA_GRID[-1])


def path(evidence, seed, regime, fold):
    if evidence == REF:
        return f"results/stage0/seed{seed}/{regime}/fold{fold}.npz"
    return f"{ABL}/{evidence}/seed{seed}/{regime}/fold{fold}.npz"


def regimes_for(evidence):
    return FULL_REGIMES if evidence in ("L11", "xckpt", REF) else LAYER_REGIMES


def pool(evidence, seed, regime):
    files = [path(evidence, seed, regime, f) for f in range(5)]
    if not all(os.path.exists(p) for p in files):
        return None
    d = [np.load(p, allow_pickle=True) for p in files]
    out = {}
    cov = np.zeros(10000, bool)
    for cond in spec.CONDITIONS:
        y = np.full(10000, -1, np.int64)
        pz = np.full((10000, spec.NUM_CLASSES), np.nan)
        pzp = np.full((10000, spec.NUM_CLASSES), np.nan)
        for x in d:
            i = x["test_idx"]
            y[i] = x[f"labels__{cond}"]; pz[i] = x[f"probs__q_Z__{cond}"]; pzp[i] = x[f"probs__q_ZP__{cond}"]; cov[i] = True
        out[cond] = (y, pz, pzp)
    assert cov.all()
    return out


def correct(probs, y):
    return (probs.argmax(1) == y).astype(np.float64)


def nll_brier(probs, y):
    idx = np.arange(len(y))
    nll = float(-np.mean(np.log(np.clip(probs[idx, y], 1e-12, 1))))
    br = float(np.mean(np.sum((probs - np.eye(probs.shape[1])[y]) ** 2, axis=1)))
    return nll, br


def diff_arrays(pooled):
    macro = np.mean([correct(pooled[c][2], pooled[c][0]) - correct(pooled[c][1], pooled[c][0]) for c in spec.CELLS], axis=0)
    clean = correct(pooled["clean"][2], pooled["clean"][0]) - correct(pooled["clean"][1], pooled["clean"][0])
    return macro, clean


def standalone(evidence, seed):
    cells = stage0_data.load_all_cells(seed, None if evidence == REF else evidence)
    res = {}
    for c in spec.CONDITIONS:
        d = cells[c]
        res[c] = {"acc_pct": 100 * float((d["p"].argmax(1) == d["labels"]).mean()),
                  "disagree_with_Z": float((d["p"].argmax(1) != d["z"].argmax(1)).mean())}
    res["macro12"] = {k: float(np.mean([res[c][k] for c in spec.CELLS])) for k in ("acc_pct", "disagree_with_Z")}
    return res


def determinism_and_convergence():
    rows, maxdev, n_bad, n_files = [], 0.0, 0, 0
    for ev in SOURCES:
        for seed in spec.DEV_SEEDS:
            for rg in regimes_for(ev):
                for f in range(5):
                    p = path(ev, seed, rg, f)
                    if not os.path.exists(p):
                        continue
                    a = np.load(p, allow_pickle=True); b = np.load(path(REF, seed, rg, f), allow_pickle=True)
                    dev = max(float(np.abs(a[f"probs__q_Z__{c}"].astype(np.float64) - b[f"probs__q_Z__{c}"].astype(np.float64)).max()) for c in spec.CONDITIONS)
                    maxdev = max(maxdev, dev); n_files += 1; n_bad += dev > 1e-6
                    for arm in ("q_Z", "q_ZP"):
                        lam = float(a[f"meta__{arm}__selected_lambda"])
                        rows.append({"evidence": ev, "seed": seed, "regime": rg, "fold": f, "arm": arm, "lambda": lam,
                                     "edge": lam in EDGES, "converged": bool(a[f"meta__{arm}__converged"]), "retried": bool(a[f"meta__{arm}__retried"])})
    n = len(rows)
    return {"q_Z_vs_stage0_max_abs_prob_dev": maxdev, "n_files_compared": n_files, "n_files_dev_gt_1e-6": int(n_bad),
            "n_arm_fits": n, "n_unconverged": sum(not r["converged"] for r in rows), "n_retried": sum(r["retried"] for r in rows),
            "n_lambda_at_grid_edge": sum(r["edge"] for r in rows),
            "lambda_histogram": {str(k): sum(r["lambda"] == k for r in rows) for k in LAMBDA_GRID}}


def main():
    os.makedirs(OUT, exist_ok=True)
    gid = group_id_for_bootstrap()
    out = {"per_source": {}, "checks": determinism_and_convergence()}
    diffs = {}
    for ev in [REF] + SOURCES:
        for seed in spec.DEV_SEEDS:
            entry = {"regimes": {}}
            dd = {}
            for rg in regimes_for(ev):
                pooled = pool(ev, seed, rg)
                if pooled is None:
                    entry["regimes"][rg] = {"status": "incomplete"}
                    continue
                macro, clean = diff_arrays(pooled)
                dd[rg] = (macro, clean)
                b = grouped_bootstrap(macro, gid, seed=BOOT_SEED + 5000 + seed)
                nz = np.mean([nll_brier(pooled[c][1], pooled[c][0]) for c in spec.CELLS], axis=0)
                nzp = np.mean([nll_brier(pooled[c][2], pooled[c][0]) for c in spec.CELLS], axis=0)
                entry["regimes"][rg] = {"delta_acc_pp": 100 * float(macro.mean()), "ci95_pp": [100 * x for x in b["ci95"]],
                                        "clean_increment_pp": 100 * float(clean.mean()),
                                        "q_Z_nll_brier_macro12": nz.tolist(), "q_ZP_nll_brier_macro12": nzp.tolist()}
            for tag, t, s_ in (("8k1", "T-8k1", "S-8k1"), ("2.5k1", "T-2.5k1", "S-2.5k1")):
                if t in dd and s_ in dd:
                    g = grouped_bootstrap_diff_of_diffs(dd[t][0], dd[s_][0], gid, seed=BOOT_SEED + 6000 + seed)
                    ci = grouped_bootstrap(dd[s_][1], gid, seed=BOOT_SEED + 7000 + seed)
                    entry[f"gap_{tag}"] = {"estimate_pp": 100 * g["estimate"], "ci95_pp": [100 * x for x in g["ci95"]]}
                    entry[f"S_clean_increment_{tag}"] = {"estimate_pp": 100 * ci["estimate"], "ci95_pp": [100 * x for x in ci["ci95"]]}
            entry["standalone"] = standalone(ev, seed)
            diffs[(ev, seed)] = dd
            out["per_source"].setdefault(ev, {})[str(seed)] = entry

    # primary contrast D_A and rules
    rules = {"D_A_T8k1": {}}
    r1, r2 = [], []
    for seed in spec.DEV_SEEDS:
        s = str(seed)
        ref = out["per_source"][REF][s]["regimes"].get("T-8k1", {})
        a = out["per_source"]["L11"][s]["regimes"].get("T-8k1", {})
        if "delta_acc_pp" not in ref or "delta_acc_pp" not in a:
            rules["status"] = "incomplete"; break
        b = grouped_bootstrap_diff_of_diffs(diffs[(REF, seed)]["T-8k1"][0], diffs[("L11", seed)]["T-8k1"][0], gid, seed=BOOT_SEED + 8000 + seed)
        rules["D_A_T8k1"][s] = {"estimate_pp": 100 * b["estimate"], "ci95_pp": [100 * x for x in b["ci95"]],
                                "delta_T_ref": ref["delta_acc_pp"], "delta_T_A": a["delta_acc_pp"], "ratio_A_over_ref": a["delta_acc_pp"] / ref["delta_acc_pp"]}
        r1.append(a["delta_acc_pp"] >= 0.5 * ref["delta_acc_pp"])
        pb = out["per_source"]["xckpt"][s]
        t = pb["regimes"]["T-8k1"]; sr = pb["regimes"]["S-8k1"]
        r2.append(t["delta_acc_pp"] >= 1.0 and t["ci95_pp"][0] > 0 and sr["delta_acc_pp"] <= 0.5 and abs(sr["clean_increment_pp"]) <= 0.5)
        rules.setdefault("P_B_8k1", {})[s] = {"delta_T": t["delta_acc_pp"], "delta_T_ci": t["ci95_pp"], "delta_S": sr["delta_acc_pp"], "clean_increment_S": sr["clean_increment_pp"]}
    if "status" not in rules:
        r3 = []
        for seed in spec.DEV_SEEDS:
            s = str(seed)
            xs = np.array([out["per_source"][ev][s]["standalone"]["macro12"]["disagree_with_Z"] for ev in LAYERS])
            ys = np.array([out["per_source"][ev][s]["regimes"]["T-8k1"]["delta_acc_pp"] for ev in LAYERS])
            rho = float(spearmanr(xs, ys).correlation)
            rho_depth = float(spearmanr(np.arange(12), ys).correlation)
            m, c0 = np.polyfit(xs, ys, 1)
            res = ys - (m * xs + c0); rmse = float(np.sqrt(np.mean(res ** 2)))
            xb = out["per_source"]["xckpt"][s]["standalone"]["macro12"]["disagree_with_Z"]
            yb = out["per_source"]["xckpt"][s]["regimes"]["T-8k1"]["delta_acc_pp"]
            rb = float(yb - (m * xb + c0))
            r3.append(rho >= 0.8 and abs(rb) <= 2 * rmse)
            rules.setdefault("R3_detail", {})[s] = {"spearman_disagreement_vs_deltaT": rho, "spearman_depth_vs_deltaT_transparency": rho_depth,
                                                    "line_slope": float(m), "line_intercept": float(c0), "residual_rmse": rmse,
                                                    "P_B_disagreement": xb, "P_B_deltaT": yb, "P_B_residual": rb,
                                                    "layer_points": {ev: [float(x), float(y)] for ev, x, y in zip(LAYERS, xs, ys)}}
        met = {"R1": bool(all(r1)) and len(r1) == 2, "R2": bool(all(r2)) and len(r2) == 2, "R3": bool(all(r3)) and len(r3) == 2}
        first = next((k for k in ("R1", "R2", "R3") if met[k]), None)
        rules["met"] = met
        rules["per_checkpoint"] = {"R1": r1, "R2": r2, "R3": r3}
        rules["verdict"] = {"R1": "depth-specific story stops", "R2": "generic fusion under shift", "R3": "generic diversity",
                            None: "depth-specific pattern survives (propose confirmation package; do not start it)"}[first]
    out["primary_and_rules"] = rules
    json.dump(out, open(f"{OUT}/ablation_aggregate.json", "w"), indent=1)
    print(json.dumps({"checks": out["checks"], "primary_and_rules": {k: v for k, v in rules.items() if k != "R3_detail"}}, indent=1))


if __name__ == "__main__":
    main()
