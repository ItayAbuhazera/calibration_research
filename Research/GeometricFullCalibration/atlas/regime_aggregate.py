"""Aggregation and frozen decision rules for the regime-map pilot (docs/regime_map_pilot_spec.md).

    python -m atlas.regime_aggregate
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import spec
from .stage0_aggregate import BOOT_SEED, group_id_for_bootstrap, grouped_bootstrap, grouped_bootstrap_diff_of_diffs

ROOT = "results/regime_map"
OUT = f"{ROOT}/report"
STATES = ("a", "b1", "b3", "b10")
REGIMES = ("T-8k1", "S-8k1", "T-2.5k1", "S-2.5k1")
MANIP_MIN_PP, FINAL_MIN_GAP_PP, KILL_RATIO, SUPPORT_RATIO, DOSE_TOL_PP = 0.3, 1.0, 0.8, 0.5, 0.3


def evaluate_rules(st: dict, diff_b10_minus_a_ci: list) -> dict:
    """st[state] = {gap8, gap8_ci, clean_inc, clean_inc_ci} in pp (12-cell macro gap at 8k x 1; S-8k x 1 clean-view increment).
    Rows are checked top to bottom; the first that applies is the verdict (frozen in the spec, section 5)."""
    a, b = st["a"], st["b10"]
    dose_ok, dose_violations = True, []
    order = [s for s in STATES if s in st]
    for i in order:
        for j in order:
            if st[i]["clean_inc"] - st[j]["clean_inc"] >= DOSE_TOL_PP and st[j]["gap8"] < st[i]["gap8"] - DOSE_TOL_PP:
                dose_ok = False; dose_violations.append((i, j))
    r = {"dose_condition_ok": dose_ok, "dose_violations": dose_violations}
    if a["clean_inc"] <= MANIP_MIN_PP or a["clean_inc_ci"][0] <= 0:
        r.update(row=1, verdict="manipulation check failed: inconclusive (report only)")
    elif b["gap8_ci"][0] <= 0 or b["gap8"] < FINAL_MIN_GAP_PP:
        r.update(row=2, verdict="Stage 0 pattern does not transfer to a fine-tuned pretrained model: restrict the claim to from-scratch ResNet-101")
    elif a["gap8"] >= KILL_RATIO * b["gap8"]:
        r.update(row=3, verdict="kill 'clean redundancy controls recoverability' (P clean-complementary yet not source-recoverable)")
    elif a["gap8"] <= SUPPORT_RATIO * b["gap8"] and diff_b10_minus_a_ci[0] > 0 and dose_ok:
        r.update(row=4, verdict="supports clean redundancy as the controlling variable: propose a confirmation and baselines package (do not start it)")
    else:
        r.update(row=5, verdict="inconclusive (report)")
    return r


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


def main():
    os.makedirs(OUT, exist_ok=True)
    gid = group_id_for_bootstrap()
    res, dd = {"label": "regime-map pilot; single training seed per state; intervals are image-bootstrap only"}, {}
    for s in STATES:
        e = {"regimes": {}}
        A = {}
        for rg in REGIMES:
            p = pool(s, rg)
            if p is None:
                e["regimes"][rg] = {"status": "incomplete"}; continue
            A[rg] = arrays(p)
            e["regimes"][rg] = {
                "delta_acc_pp": 100 * float(A[rg][0].mean()),
                "q_Z_macro12_acc": 100 * float(np.mean([cor(p[c][1], p[c][0]).mean() for c in spec.CELLS])),
                "q_ZP_macro12_acc": 100 * float(np.mean([cor(p[c][2], p[c][0]).mean() for c in spec.CELLS])),
                "q_Z_clean_acc": 100 * float(cor(p["clean"][1], p["clean"][0]).mean()), "q_ZP_clean_acc": 100 * float(cor(p["clean"][2], p["clean"][0]).mean())}
        for tag, t, s_ in (("8k1", "T-8k1", "S-8k1"), ("2.5k1", "T-2.5k1", "S-2.5k1")):
            if t in A and s_ in A:
                g = grouped_bootstrap_diff_of_diffs(A[t][0], A[s_][0], gid, seed=BOOT_SEED + 11000 + len(tag))
                c = grouped_bootstrap(A[s_][1], gid, seed=BOOT_SEED + 12000 + len(tag))
                e[f"gap_{tag}"] = {"estimate_pp": 100 * g["estimate"], "ci95_pp": [100 * x for x in g["ci95"]]}
                e[f"S_clean_increment_{tag}"] = {"estimate_pp": 100 * c["estimate"], "ci95_pp": [100 * x for x in c["ci95"]]}
        res[s] = e; dd[s] = A
    if all("gap_8k1" in res[s] for s in STATES):
        st = {s: {"gap8": res[s]["gap_8k1"]["estimate_pp"], "gap8_ci": res[s]["gap_8k1"]["ci95_pp"],
                  "clean_inc": res[s]["S_clean_increment_8k1"]["estimate_pp"], "clean_inc_ci": res[s]["S_clean_increment_8k1"]["ci95_pp"]} for s in STATES}
        gb = dd["b10"]["T-8k1"][0] - dd["b10"]["S-8k1"][0]; ga = dd["a"]["T-8k1"][0] - dd["a"]["S-8k1"][0]
        d = grouped_bootstrap(gb - ga, gid, seed=BOOT_SEED + 13000)
        res["gap_b10_minus_gap_a"] = {"estimate_pp": 100 * d["estimate"], "ci95_pp": [100 * x for x in d["ci95"]]}
        res["rules"] = evaluate_rules(st, res["gap_b10_minus_gap_a"]["ci95_pp"])
    json.dump(res, open(f"{OUT}/regime_aggregate.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ("rules", "gap_b10_minus_gap_a")}, indent=1))


if __name__ == "__main__":
    main()
