"""Aggregation and frozen decision rules for the regime-map pilot (docs/regime_map_pilot_spec_v2.md).

    python -m atlas.regime_aggregate
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import spec
from .stage0_aggregate import BOOT_SEED, group_id_for_bootstrap, grouped_bootstrap, grouped_bootstrap_diff_of_diffs
from .regime_map import STATES

ROOT = "results/regime_map"
OUT = f"{ROOT}/report"
REGIMES = ("T-8k1", "S-8k1", "T-2.5k1", "S-2.5k1")
MANIP_MIN_PP, FINAL_MAX_CLEAN_PP, FINAL_MIN_GAP_PP, KILL_RATIO, SUPPORT_RATIO, DOSE_TOL_PP = 0.3, 0.3, 1.0, 0.8, 0.5, 0.3
DOSES = ("a", "b1", "b3", "b10")


def evaluate_rules(st: dict, diff_b10_minus_a_ci: list) -> dict:
    """One fine-tuning seed. st[dose] = {gap8, gap8_ci, clean_inc, clean_inc_ci} in pp for dose in a, b1, b3, b10
    (12-cell macro gap at 8k x 1; S-8k x 1 clean-view increment). Rows are checked top to bottom; the first that applies is
    this seed's verdict (frozen in the spec v2, section 5)."""
    a, b = st["a"], st["b10"]
    dose_ok, dose_violations = True, []
    for i in DOSES:
        for j in DOSES:
            if st[i]["clean_inc"] - st[j]["clean_inc"] >= DOSE_TOL_PP and st[j]["gap8"] < st[i]["gap8"] - DOSE_TOL_PP:
                dose_ok = False; dose_violations.append((i, j))
    r = {"dose_condition_ok": dose_ok, "dose_violations": dose_violations}
    if a["clean_inc"] <= MANIP_MIN_PP or a["clean_inc_ci"][0] <= 0:
        r.update(row=1, verdict="manipulation check 1 failed (a is not clean-complementary): inconclusive; report only")
    elif b["clean_inc"] > FINAL_MAX_CLEAN_PP:
        r.update(row=2, verdict="manipulation check 2 failed (fine-tuning did not reach a clean-redundant state): rows 3 and 5 cannot be read; inconclusive; report the gaps and the dose condition descriptively")
    elif b["gap8_ci"][0] <= 0 or b["gap8"] < FINAL_MIN_GAP_PP:
        r.update(row=3, verdict="Stage 0 pattern does not transfer to a fine-tuned pretrained model: restrict the claim to from-scratch ResNet-101")
    elif a["gap8"] >= KILL_RATIO * b["gap8"]:
        r.update(row=4, verdict="kill 'clean redundancy controls recoverability' (P clean-complementary yet not source-recoverable)")
    elif a["gap8"] <= SUPPORT_RATIO * b["gap8"] and diff_b10_minus_a_ci[0] > 0 and dose_ok:
        r.update(row=5, verdict="supports clean redundancy as the controlling variable: propose a confirmation and baselines package (do not start it)")
    else:
        r.update(row=6, verdict="inconclusive (report)")
    return r


def combine_seeds(per_seed: dict) -> dict:
    rows = {k: v["row"] for k, v in per_seed.items()}
    if len(set(rows.values())) == 1:
        row = next(iter(rows.values()))
        return {"row": row, "verdict": next(iter(per_seed.values()))["verdict"], "seeds_agree": True, "rows_by_seed": rows}
    return {"row": None, "verdict": "inconclusive: the two fine-tuning seeds fire different rows", "seeds_agree": False, "rows_by_seed": rows}


def path(state, regime, fold):
    return f"{ROOT}/fits/{state}/{regime}/fold{fold}.npz"


def pool(state, regime):
    files = [path(state, regime, f) for f in range(5)]
    if not all(os.path.exists(p) for p in files):
        return None
    d = [np.load(p, allow_pickle=True) for p in files]
    out, cov = {}, np.zeros(10000, bool)
    for cond in spec.CONDITIONS:
        y = np.full(10000, -1, np.int64); pz = np.full((10000, spec.NUM_CLASSES), np.nan); pzp = pz.copy()
        for x in d:
            i = x["test_idx"]; y[i] = x[f"labels__{cond}"]; pz[i] = x[f"probs__q_Z__{cond}"]; pzp[i] = x[f"probs__q_ZP__{cond}"]; cov[i] = True
        out[cond] = (y, pz, pzp)
    assert cov.all()
    return out


def cor(p, y):
    return (p.argmax(1) == y).astype(np.float64)


def arrays(pooled):
    macro = np.mean([cor(pooled[c][2], pooled[c][0]) - cor(pooled[c][1], pooled[c][0]) for c in spec.CELLS], axis=0)
    clean = cor(pooled["clean"][2], pooled["clean"][0]) - cor(pooled["clean"][1], pooled["clean"][0])
    return macro, clean


def nll_brier(p, y):
    i = np.arange(len(y))
    return float(-np.mean(np.log(np.clip(p[i, y], 1e-12, 1)))), float(np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1)))


def audit():
    rows = []
    for s in STATES:
        for rg in REGIMES:
            for f in range(5):
                p = path(s, rg, f)
                if os.path.exists(p):
                    d = np.load(p, allow_pickle=True)
                    for arm in ("q_Z", "q_ZP"):
                        lam = float(d[f"meta__{arm}__selected_lambda"])
                        rows.append({"state": s, "regime": rg, "fold": f, "arm": arm, "lambda": lam, "edge": lam in (1e-1, 1e-5),
                                     "converged": bool(d[f"meta__{arm}__converged"]), "retried": bool(d[f"meta__{arm}__retried"])})
    return {"n_arm_fits": len(rows), "n_unconverged": sum(not r["converged"] for r in rows), "n_retried": sum(r["retried"] for r in rows),
            "n_lambda_at_grid_edge": sum(r["edge"] for r in rows)}


def main():
    os.makedirs(OUT, exist_ok=True)
    gid = group_id_for_bootstrap()
    res = {"label": "regime-map pilot v2; state (a) one fit, states b* two fine-tuning seeds; intervals are image-bootstrap only (no training-seed variance)", "states": {}}
    dd = {}
    for s in STATES:
        e = {"regimes": {}}
        summ = f"{ROOT}/{s}/summary.json"
        if os.path.exists(summ):
            sm = json.load(open(summ))
            cs = sm["conditions"]
            e["standalone"] = {"acc_Z_clean": 100 * cs["clean"]["acc_Z"], "acc_P_clean": 100 * cs["clean"]["acc_P"],
                               "acc_Z_macro12": 100 * float(np.mean([cs[c]["acc_Z"] for c in spec.CELLS])), "acc_P_macro12": 100 * float(np.mean([cs[c]["acc_P"] for c in spec.CELLS])),
                               "disagree_macro12": float(np.mean([cs[c]["disagree_P_vs_Z"] for c in spec.CELLS])), "both_wrong_macro12": float(np.mean([cs[c]["both_wrong"] for c in spec.CELLS])),
                               "per_condition": cs}
            e["hyperparameters"] = {"probe_layer3_gap": {k: sm["probe_layer3_gap"][k] for k in ("selected_lambda", "initial_grid", "extensions_up", "extensions_down", "at_grid_edge_after_extension", "inner_fit_nll_by_lambda")},
                                    "head_linear_layer4_gap": ({k: sm["head_linear_layer4_gap"][k] for k in ("selected_lambda", "initial_grid", "extensions_up", "extensions_down", "at_grid_edge_after_extension", "inner_fit_nll_by_lambda")} if "head_linear_layer4_gap" in sm else None),
                                    "train_acc_Z": sm["train_acc_Z"], "val_acc_Z": sm["val_acc_Z"], "train_acc_P": sm["train_acc_P"], "val_acc_P": sm["val_acc_P"]}
        A = {}
        for rg in REGIMES:
            p = pool(s, rg)
            if p is None:
                e["regimes"][rg] = {"status": "incomplete"}; continue
            A[rg] = arrays(p)
            b = grouped_bootstrap(A[rg][0], gid, seed=BOOT_SEED + 14000 + STATES.index(s) * 10 + REGIMES.index(rg))
            e["regimes"][rg] = {"delta_acc_pp": 100 * float(A[rg][0].mean()), "delta_ci95_pp": [100 * x for x in b["ci95"]],
                                "q_Z_macro12_acc": 100 * float(np.mean([cor(p[c][1], p[c][0]).mean() for c in spec.CELLS])),
                                "q_ZP_macro12_acc": 100 * float(np.mean([cor(p[c][2], p[c][0]).mean() for c in spec.CELLS])),
                                "q_Z_clean_acc": 100 * float(cor(p["clean"][1], p["clean"][0]).mean()), "q_ZP_clean_acc": 100 * float(cor(p["clean"][2], p["clean"][0]).mean()),
                                "q_Z_nll_brier_macro12": np.mean([nll_brier(p[c][1], p[c][0]) for c in spec.CELLS], axis=0).tolist(),
                                "q_ZP_nll_brier_macro12": np.mean([nll_brier(p[c][2], p[c][0]) for c in spec.CELLS], axis=0).tolist()}
        for tag, t, s_ in (("8k1", "T-8k1", "S-8k1"), ("2.5k1", "T-2.5k1", "S-2.5k1")):
            if t in A and s_ in A:
                g = grouped_bootstrap_diff_of_diffs(A[t][0], A[s_][0], gid, seed=BOOT_SEED + 15000 + STATES.index(s) * 10 + len(tag))
                c = grouped_bootstrap(A[s_][1], gid, seed=BOOT_SEED + 16000 + STATES.index(s) * 10 + len(tag))
                e[f"gap_{tag}"] = {"estimate_pp": 100 * g["estimate"], "ci95_pp": [100 * x for x in g["ci95"]]}
                e[f"S_clean_increment_{tag}"] = {"estimate_pp": 100 * c["estimate"], "ci95_pp": [100 * x for x in c["ci95"]]}
        res["states"][s] = e; dd[s] = A
    res["audit"] = audit()
    per_seed = {}
    if all("gap_8k1" in res["states"][s] for s in STATES):
        for k in (1, 2):
            names = {"a": "a", "b1": f"b1_s{k}", "b3": f"b3_s{k}", "b10": f"b10_s{k}"}
            st = {d: {"gap8": res["states"][n]["gap_8k1"]["estimate_pp"], "gap8_ci": res["states"][n]["gap_8k1"]["ci95_pp"],
                      "clean_inc": res["states"][n]["S_clean_increment_8k1"]["estimate_pp"], "clean_inc_ci": res["states"][n]["S_clean_increment_8k1"]["ci95_pp"]} for d, n in names.items()}
            gb = dd[names["b10"]]["T-8k1"][0] - dd[names["b10"]]["S-8k1"][0]; ga = dd["a"]["T-8k1"][0] - dd["a"]["S-8k1"][0]
            d_ = grouped_bootstrap(gb - ga, gid, seed=BOOT_SEED + 17000 + k)
            diff = {"estimate_pp": 100 * d_["estimate"], "ci95_pp": [100 * x for x in d_["ci95"]]}
            res[f"gap_b10_minus_gap_a_seed{k}"] = diff
            per_seed[f"seed{k}"] = evaluate_rules(st, diff["ci95_pp"])
        res["rules_by_seed"] = per_seed
        res["rules"] = combine_seeds(per_seed)
    json.dump(res, open(f"{OUT}/regime_aggregate.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ("audit", "rules", "rules_by_seed")}, indent=1))


if __name__ == "__main__":
    main()
