"""Stage 2 aggregation, Gate 2 and the decision table (docs/regime_map_followup_amendment_1.md sec. 4). Anchored, target-fitted
(ORACLE DIAGNOSTIC), 12-cell macro; decision states b10_s1 and b10_s2 only; one shared resample-index array per state (same as Stage 1).

    python -m atlas.followup_stage2_aggregate
"""
import json
import os

import numpy as np

from . import spec
from .regime_map import STATES
from .stage0_aggregate import BOOT_SEED, group_id_for_bootstrap
from .followup_stage1_aggregate import gate1, pool as pool1, ARMS as ARMS1, B

S2 = "results/regime_map_followup/stage2"
S1 = "results/regime_map_followup/stage1"
OUT = "results/regime_map_followup/report"
ARMS2 = ("zonly", "c1e", "c1c", "c1d", "kernel", "zk", "prow")
GATE2_FRAC, HEAD_FRAC = 0.5, 0.75
DEC = ("b10_s1", "b10_s2")


def decide(st: dict) -> dict:
    """st[seed] = {D, c1e_frac, d_c1d, d_zk, D_minus_c1d_ci}. Gate 2, then HEAD-DISCARD, INTERMEDIATE-SPECIFIC, else INCONCLUSIVE (both seeds must agree)."""
    if all(st[s]["c1e_frac"] >= GATE2_FRAC for s in st):
        return {"outcome": "STOP-GATE-2", "reason": "Delta_T(Z, P_L) explains >= 50% of D in both b10 seeds (second-classifier effect, not layer3 information)"}
    if all(st[s]["d_c1d"] >= st[s]["D"] and st[s]["d_c1d"] > 0 and st[s]["d_zk"] >= HEAD_FRAC * st[s]["d_c1d"] for s in st):
        return {"outcome": "HEAD-DISCARD", "reason": "Delta_T(C1d) >= D and Delta_T(Z + P_ker h_L) >= 75% of Delta_T(C1d) in both b10 seeds"}
    if all(st[s]["D"] - st[s]["d_c1d"] > 0 and st[s]["D_minus_c1d_ci"][0] > 0 for s in st):
        return {"outcome": "INTERMEDIATE-SPECIFIC", "reason": "D - Delta_T(C1d) > 0 with the 95% interval excluding 0 in both b10 seeds"}
    return {"outcome": "INCONCLUSIVE", "reason": "none of the pre-declared rows holds in both b10 seeds (or the seeds disagree); stop"}


def pool2(state):
    files = [f"{S2}/{state}/fold{f}.npz" for f in range(5)]
    if not all(os.path.exists(p) for p in files):
        return None
    d = [np.load(p, allow_pickle=True) for p in files]
    out = {a: np.zeros((10000, len(spec.CONDITIONS))) for a in ARMS2}
    pdiff = []
    for x in d:
        i = x["test_idx"]
        for a in ARMS2:
            out[a][i] = x[f"correct__{a}"].T
        pdiff.append(float(np.abs(x["probs__prow"] - x["probs__zonly"])[:, :, :].mean()))
    return out, d, pdiff


def main():
    os.makedirs(OUT, exist_ok=True)
    gid = group_id_for_bootstrap(); _, inv = np.unique(gid, return_inverse=True); ng = inv.max() + 1; cnt = np.bincount(inv)
    gmean = lambda x: np.bincount(inv, weights=x) / cnt
    res = {"label": "ORACLE DIAGNOSTIC (target labels); anchored residual readouts; T-8k x 1; image-bootstrap intervals only", "states": {}}
    stats = {}
    for si, s in enumerate(STATES):
        p1, p2 = pool1(s), pool2(s)
        if p1 is None or p2 is None:
            res["states"][s] = {"status": "incomplete"}; continue
        o1, _ = p1; o2, files2, pdiff = p2
        macro = {a: o1[a][:, 1:].mean(1) for a in o1}; clean = {a: o1[a][:, 0] for a in o1}
        for a in ARMS2:
            if a != "zonly":
                macro[a], clean[a] = o2[a][:, 1:].mean(1), o2[a][:, 0]
        macro["zonly2"], clean["zonly2"] = o2["zonly"][:, 1:].mean(1), o2["zonly"][:, 0]
        rng = np.random.default_rng(BOOT_SEED + 30000 + si); idx = rng.integers(0, ng, (B, ng))
        G = {a: gmean(macro[a]) for a in macro}; Gc = {a: gmean(clean[a]) for a in clean}
        ci = lambda num, den=None: [float(np.quantile((num[idx].mean(1) if den is None else num[idx].mean(1) / den[idx].mean(1)), q)) for q in (.025, .975)]
        Dm = float((macro["zp"] - macro["zonly"]).mean()); Dg = G["zp"] - G["zonly"]
        e = {"arms": {}, "D_pp": 100 * Dm, "D_ci95_pp": [100 * x for x in ci(Dg)]}
        for a in ("zp", "c1b", "c1e", "c1c", "c1d", "kernel", "zk", "prow"):
            e["arms"][a] = {"macro12_acc": 100 * float(macro[a].mean()), "clean_acc": 100 * float(clean[a].mean()),
                            "minus_base_macro12_pp": 100 * float((macro[a] - macro["base"]).mean()), "minus_base_macro12_ci": [100 * x for x in ci(G[a] - G["base"])],
                            "minus_base_clean_pp": 100 * float((clean[a] - clean["base"]).mean()), "minus_base_clean_ci": [100 * x for x in ci(Gc[a] - Gc["base"])],
                            "delta_T_pp": 100 * float((macro[a] - macro["zonly"]).mean()), "delta_T_ci": [100 * x for x in ci(G[a] - G["zonly"])],
                            "explains_fraction_of_D": float((macro[a] - macro["zonly"]).mean()) / Dm if Dm != 0 else None,
                            "explains_fraction_ci": ci(G[a] - G["zonly"], Dg) if Dm != 0 else None}
            if a in files2[0].files and False:
                pass
        for a in ("c1e", "c1c", "c1d", "kernel", "zk", "prow"):
            e["arms"][a]["selected_lambda_by_fold"] = [float(f[f"meta__{a}__lambda"]) for f in files2]
            e["arms"][a]["n_unconverged"] = int(sum(not bool(f[f"meta__{a}__converged"]) for f in files2)); e["arms"][a]["n_retried"] = int(sum(bool(f[f"meta__{a}__retried"]) for f in files2))
        e["D_minus_c1d_pp"] = 100 * float((macro["zp"] - macro["c1d"]).mean()); e["D_minus_c1d_ci"] = [100 * x for x in ci(G["zp"] - G["c1d"])]
        e["prow_sanity"] = {"macro12_acc_minus_anchored_zonly_pp": 100 * float((macro["prow"] - macro["zonly"]).mean()), "mean_abs_prob_diff_vs_anchored_zonly": float(np.mean(pdiff))}
        e["zonly_reproduction_stage2_vs_stage1"] = {"n_predictions_differing": int(sum((f["correct__zonly"] != np.load(f"{S1}/{s}/fold{f['fold']}.npz")["correct__zonly"]).sum() for f in files2))}
        res["states"][s] = e; stats[s] = e
    if all(s in stats for s in DEC):
        st = {s: {"D": stats[s]["D_pp"], "c1e_frac": stats[s]["arms"]["c1e"]["explains_fraction_of_D"], "d_c1d": stats[s]["arms"]["c1d"]["delta_T_pp"],
                  "d_zk": stats[s]["arms"]["zk"]["delta_T_pp"], "D_minus_c1d_ci": stats[s]["D_minus_c1d_ci"]} for s in DEC}
        res["gate1_recheck"] = gate1({s: st[s]["D"] for s in DEC}, {s: json.load(open(f"{OUT}/stage1_anchored.json"))["states"][s]["c1b_explains_fraction_of_D"]["estimate"] for s in DEC})
        res["decision_inputs"] = st; res["decision"] = decide(st)
    json.dump(res, open(f"{OUT}/stage2_anchored.json", "w"), indent=1, default=float)
    f = lambda v, c: "%+.2f [%+.2f, %+.2f]" % (v, *c)
    L = ["| state | D = Δ_T(Z,P) | C1b frac | C1e (Z,P_L) Δ_T / frac of D | C1c (h_L) Δ_T | C1d (Z,h_L) Δ_T | K Δ_T | Z+K Δ_T | D − Δ_T(C1d) |", "|---|---|---|---|---|---|---|---|---|"]
    S1J = json.load(open(f"{OUT}/stage1_anchored.json"))["states"]
    for s in STATES:
        if s not in stats:
            continue
        a = stats[s]["arms"]; c1b = S1J[s]["c1b_explains_fraction_of_D"]["estimate"]
        L.append(f"| {s} | {f(stats[s]['D_pp'], stats[s]['D_ci95_pp'])} | {c1b:.3f} | {f(a['c1e']['delta_T_pp'], a['c1e']['delta_T_ci'])} / {a['c1e']['explains_fraction_of_D']:.3f} [{a['c1e']['explains_fraction_ci'][0]:.3f}, {a['c1e']['explains_fraction_ci'][1]:.3f}] | "
                 f"{f(a['c1c']['delta_T_pp'], a['c1c']['delta_T_ci'])} | {f(a['c1d']['delta_T_pp'], a['c1d']['delta_T_ci'])} | {f(a['kernel']['delta_T_pp'], a['kernel']['delta_T_ci'])} | {f(a['zk']['delta_T_pp'], a['zk']['delta_T_ci'])} | {f(stats[s]['D_minus_c1d_pp'], stats[s]['D_minus_c1d_ci'])} |")
    open(f"{OUT}/stage2_anchored_table.md", "w").write("\n".join(L)); print("\n".join(L)); print(json.dumps(res.get("decision"), indent=1))


if __name__ == "__main__":
    main()
