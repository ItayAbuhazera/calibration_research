"""Stage 1 aggregation and Gate 1 (docs/regime_map_followup_amendment_1.md, sections 1, 4).
Anchored, target-fitted (ORACLE DIAGNOSTIC), 12-cell macro; paired bootstrap over evaluation images with ONE shared
resample-index array per state used by every arm.

    python -m atlas.followup_stage1_aggregate
"""
import json
import os

import numpy as np

from . import spec
from .regime_map import STATES
from .stage0_aggregate import BOOT_SEED, group_id_for_bootstrap

ROOT = "results/regime_map_followup/stage1"
OUT = "results/regime_map_followup/report"
ARMS = ("zonly", "zp", "c1b")
B = 2000
D_MIN_PP, CAP_FRAC = 0.5, 0.5


def gate1(D: dict, c1b_frac: dict) -> dict:
    """D[seed], c1b_frac[seed] for the two b10 fine-tuning seeds (point estimates; D in pp, fraction unitless). STOP if either fires in EITHER seed."""
    small = [s for s in D if D[s] < D_MIN_PP]
    cap = [s for s in c1b_frac if c1b_frac[s] >= CAP_FRAC]
    if small:
        return {"stop": True, "reason": "D < 0.5 pp in b10 seed(s) " + ",".join(sorted(small)) + " (effect too small to matter)"}
    if cap:
        return {"stop": True, "reason": "C1b explains >= 50% of D in b10 seed(s) " + ",".join(sorted(cap)) + " (capacity)"}
    return {"stop": False, "reason": "Gate 1 passed"}


def pool(state):
    files = [f"{ROOT}/{state}/fold{f}.npz" for f in range(5)]
    if not all(os.path.exists(p) for p in files):
        return None
    d = [np.load(p, allow_pickle=True) for p in files]
    out = {a: np.zeros((10000, len(spec.CONDITIONS))) for a in ARMS + ("base",)}
    cov = np.zeros(10000, bool)
    for x in d:
        i = x["test_idx"]
        for a in ARMS + ("base",):
            out[a][i] = x[f"correct__{a}"].T
        cov[i] = True
    assert cov.all()
    return out, d


def main():
    os.makedirs(OUT, exist_ok=True)
    gid = group_id_for_bootstrap()
    _, inv = np.unique(gid, return_inverse=True); ng = inv.max() + 1; cnt = np.bincount(inv)
    gmean = lambda x: np.bincount(inv, weights=x) / cnt
    res = {"label": "ORACLE DIAGNOSTIC (target labels); anchored residual readouts; T-8k x 1; image-bootstrap intervals only", "states": {}}
    q = {}
    for si, s in enumerate(STATES):
        p = pool(s)
        if p is None:
            res["states"][s] = {"status": "incomplete"}; continue
        out, files = p
        macro = {a: out[a][:, 1:].mean(1) for a in out}; clean = {a: out[a][:, 0] for a in out}
        rng = np.random.default_rng(BOOT_SEED + 30000 + si); idx = rng.integers(0, ng, (B, ng))
        def ci(g_num, g_den=None):
            est = g_num[idx].mean(1) if g_den is None else g_num[idx].mean(1) / g_den[idx].mean(1)
            return [float(np.quantile(est, .025)), float(np.quantile(est, .975))]
        e = {"arms": {}}
        Gm = {a: gmean(macro[a]) for a in out}; Gc = {a: gmean(clean[a]) for a in out}
        for a in ARMS:
            vb_m, vb_c = macro[a] - macro["base"], clean[a] - clean["base"]
            dz_m, dz_c = macro[a] - macro["zonly"], clean[a] - clean["zonly"]
            e["arms"][a] = {
                "macro12_acc": 100 * float(macro[a].mean()), "clean_acc": 100 * float(clean[a].mean()),
                "minus_base_macro12_pp": 100 * float(vb_m.mean()), "minus_base_macro12_ci": [100 * x for x in ci(Gm[a] - Gm["base"])],
                "minus_base_clean_pp": 100 * float(vb_c.mean()), "minus_base_clean_ci": [100 * x for x in ci(Gc[a] - Gc["base"])],
                "delta_T_vs_anchored_zonly_pp": 100 * float(dz_m.mean()), "delta_T_ci": [100 * x for x in ci(Gm[a] - Gm["zonly"])],
                "delta_clean_vs_anchored_zonly_pp": 100 * float(dz_c.mean()), "delta_clean_ci": [100 * x for x in ci(Gc[a] - Gc["zonly"])]}
            e["arms"][a]["selected_lambda_by_fold"] = [float(f[f"meta__{a}__lambda"]) for f in files]
            e["arms"][a]["n_unconverged"] = int(sum(not bool(f[f"meta__{a}__converged"]) for f in files)); e["arms"][a]["n_retried"] = int(sum(bool(f[f"meta__{a}__retried"]) for f in files))
        e["base_macro12_acc"], e["base_clean_acc"] = 100 * float(macro["base"].mean()), 100 * float(clean["base"].mean())
        D = float((macro["zp"] - macro["zonly"]).mean()); Dg = Gm["zp"] - Gm["zonly"]
        for a in ("c1b",):
            frac = float((macro[a] - macro["zonly"]).mean()) / D if D != 0 else float("nan")
            e[f"{a}_explains_fraction_of_D"] = {"estimate": frac, "ci95": ci(Gm[a] - Gm["zonly"], Dg) if D != 0 else None}
        e["D_pp"] = 100 * D; e["D_ci95_pp"] = e["arms"]["zp"]["delta_T_ci"]
        res["states"][s] = e; q[s] = e
    A = json.load(open("results/regime_map/report/regime_aggregate.json"))
    for s in STATES:
        if s in q:
            R = A["states"][s]
            res["states"][s]["secondary_unanchored_phase2"] = {"clean_increment_8k1_pp": R["S_clean_increment_8k1"]["estimate_pp"], "delta_S_8k1_pp": R["regimes"]["S-8k1"]["delta_acc_pp"],
                                                                "delta_T_8k1_pp": R["regimes"]["T-8k1"]["delta_acc_pp"], "gap_8k1_pp": R["gap_8k1"]["estimate_pp"]}
    if all(s in q for s in ("b10_s1", "b10_s2")):
        res["gate1"] = gate1({s: q[s]["D_pp"] for s in ("b10_s1", "b10_s2")}, {s: q[s]["c1b_explains_fraction_of_D"]["estimate"] for s in ("b10_s1", "b10_s2")})
    json.dump(res, open(f"{OUT}/stage1_anchored.json", "w"), indent=1)
    f = lambda v, c: "%+.2f [%+.2f, %+.2f]" % (v, *c)
    lines = ["| state | anchored Z-only − base (macro / clean) | anchored (Z,P) − base (macro / clean) | D = Δ_T(Z,P) | C1b − Z-only (Δ_T) | C1b fraction of D |", "|---|---|---|---|---|---|"]
    for s in STATES:
        if s not in q:
            continue
        a = q[s]["arms"]; fr = q[s]["c1b_explains_fraction_of_D"]
        lines.append(f"| {s} | {f(a['zonly']['minus_base_macro12_pp'], a['zonly']['minus_base_macro12_ci'])} / {f(a['zonly']['minus_base_clean_pp'], a['zonly']['minus_base_clean_ci'])} | "
                     f"{f(a['zp']['minus_base_macro12_pp'], a['zp']['minus_base_macro12_ci'])} / {f(a['zp']['minus_base_clean_pp'], a['zp']['minus_base_clean_ci'])} | {f(q[s]['D_pp'], q[s]['D_ci95_pp'])} | "
                     f"{f(a['c1b']['delta_T_vs_anchored_zonly_pp'], a['c1b']['delta_T_ci'])} | {fr['estimate']:.3f} [{fr['ci95'][0]:.3f}, {fr['ci95'][1]:.3f}] |")
    open(f"{OUT}/stage1_anchored_table.md", "w").write("\n".join(lines))
    print("\n".join(lines)); print(json.dumps(res.get("gate1"), indent=1))


if __name__ == "__main__":
    main()
