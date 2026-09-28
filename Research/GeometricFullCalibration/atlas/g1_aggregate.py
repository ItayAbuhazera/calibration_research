"""G1 aggregation, bootstrap and the frozen decision (docs/g1_conditional_access_spec.md sec. 5-6).
ORACLE DIAGNOSTIC (target labels); anchored readouts; 12-cell macro accuracy; image-group bootstrap, one shared index array per checkpoint.

    python -m atlas.g1_aggregate
"""
import json
import os

import numpy as np

from . import spec
from .g1_fit import OUT as FITS, HROOT, REGIME_ARMS
from .g1_rules import decide
from .stage0_aggregate import BOOT_SEED, group_id_for_bootstrap

REPORT = "results/g1/report"
B = 2000
SEEDS = (2, 4)


def pool(seed, regime, arm):
    files = [f"{FITS}/seed{seed}/{regime}/{arm}/fold{f}.npz" for f in range(5)]
    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing fold files: {missing}")
    cor = np.full((10000, len(spec.CONDITIONS)), np.nan); meta = []
    for p in files:
        x = np.load(p, allow_pickle=True)
        cor[x["test_idx"]] = x["correct"].T
        meta.append({"fold": int(x["fold"]), "lambda": float(x["selected_lambda"]), "edge": str(x["edge_status"]), "converged": bool(x["converged"]),
                     "retried": bool(x["retried"]), "wall_min": float(x["wall_time_s"]) / 60})
    assert not np.isnan(cor).any()
    return cor, meta


def main():
    os.makedirs(REPORT, exist_ok=True)
    gid = group_id_for_bootstrap(); _, inv = np.unique(gid, return_inverse=True); ng = inv.max() + 1; cnt = np.bincount(inv)
    gmean = lambda v: np.bincount(inv, weights=v, minlength=ng) / cnt
    res = {"label": "G1 ORACLE DIAGNOSTIC (target labels of the 12 exposed development cells); anchored readouts; 12-cell macro accuracy (pp); "
                    "95% image-group bootstrap intervals conditional on the fitted CV predictions (no training-seed variance)", "checkpoints": {}}
    per, validity = {}, {}
    for seed in SEEDS:
        rng = np.random.default_rng(BOOT_SEED + 40000 + seed); idx = rng.integers(0, ng, (B, ng))
        macro, clean, metas = {}, {}, {}
        for regime, arms in REGIME_ARMS.items():
            for arm in arms:
                if arm == "Dshuf":
                    continue
                c, m = pool(seed, regime, arm); key = f"{arm}@{regime}"
                macro[key], clean[key], metas[key] = c[:, 1:].mean(1), c[:, 0], m
        G = {k: gmean(v) for k, v in macro.items()}
        def est(a, b):
            return 100 * float((macro[a] - macro[b]).mean()), [100 * float(np.quantile((G[a] - G[b])[idx].mean(1), q)) for q in (.025, .975)]
        def est2(a, b, c, d):   # (a - b) - (c - d)
            diff = (G[a] - G[b]) - (G[c] - G[d])
            return 100 * float(((macro[a] - macro[b]) - (macro[c] - macro[d])).mean()), [100 * float(np.quantile(diff[idx].mean(1), q)) for q in (.025, .975)]
        e8, e25 = "@T-8k1", "@T-2.5k1"
        q = {}
        for name, (a, b) in {"dcond8": ("D" + e8, "C" + e8), "dcond25": ("D" + e25, "C" + e25), "dE": ("D" + e8, "E" + e8), "dF": ("F" + e8, "C" + e8),
                             "dDF": ("D" + e8, "F" + e8), "GminusC": ("G" + e8, "C" + e8), "nest": ("D" + e8, "B" + e8), "BminusA8": ("B" + e8, "A" + e8),
                             "BminusA25": ("B" + e25, "A" + e25), "CminusB8": ("C" + e8, "B" + e8), "CminusB25": ("C" + e25, "B" + e25),
                             "CminusA8": ("C" + e8, "A" + e8), "DminusA8": ("D" + e8, "A" + e8)}.items():
            q[name], q[name + "_ci"] = est(a, b)
        q["budget_drop"], q["budget_drop_ci"] = est2("D" + e25, "C" + e25, "D" + e8, "C" + e8)
        q["catchup"], q["catchup_ci"] = est2("C" + e8, "B" + e8, "C" + e25, "B" + e25)
        q["s"] = abs(q["GminusC"])
        for k in ("B", "C", "G"):
            q[f"acc{k}"] = 100 * float(macro[k + e8].mean())
        # shuffled control, fold 0 only
        xs = np.load(f"{FITS}/seed{seed}/T-8k1/Dshuf/fold0.npz", allow_pickle=True); t0 = xs["test_idx"]
        q["shuf"] = 100 * float(xs["correct"][1:].mean(0).mean() - macro["C" + e8][t0].mean())
        cons = json.load(open(f"{HROOT}/seed{seed}/consistency.json"))
        dec_meta = {k: v for k, v in metas.items() if k.split("@")[0] in ("A", "B", "C", "D", "E", "F", "G")}
        unres = {k: sum(m["edge"] == "unresolved_edge" for m in v) for k, v in dec_meta.items()}
        validity[seed] = {"v_extract": bool(cons["consistency_passed"]), "v_converged": all(m["converged"] for v in dec_meta.values() for m in v),
                          "v_edge": all(unres[k] < 2 for k in unres if k.split("@")[0] in ("C", "D")), "shuf": q["shuf"]}
        per[seed] = q
        res["checkpoints"][str(seed)] = {"contrasts": q, "validity": validity[seed], "unresolved_edges_by_arm": unres,
                                         "shuffle_control": {"fold0_Dshuf_minus_C_pp": q["shuf"], "lambda": float(xs["selected_lambda"]), "edge": str(xs["edge_status"]), "converged": bool(xs["converged"])},
                                         "arms": {k: {"macro12_acc": 100 * float(macro[k].mean()), "clean_acc_descriptive": 100 * float(clean[k].mean()),
                                                      "folds": metas[k]} for k in macro},
                                         "extraction": {k: cons[k] for k in ("max_abs_dz", "max_abs_dz_model_forward_vs_atlas", "argmax_agreement", "temp", "consistency_passed")}}
    res["decision"] = decide(per, validity)
    json.dump(res, open(f"{REPORT}/g1_aggregate.json", "w"), indent=1, default=float)
    f = lambda v, c: "%+.2f [%+.2f, %+.2f]" % (v, *c)
    L = ["| quantity (pp) | checkpoint 2 | checkpoint 4 |", "|---|---|---|"]
    for k, lab in (("dcond8", "**Δ_cond = D − C (8k, primary)**"), ("dcond25", "Δ_cond (2.5k)"), ("budget_drop", "Δ_cond(2.5k) − Δ_cond(8k)"),
                   ("dE", "D − E (vs P_4.2 redundant summary)"), ("dF", "F − C (Z_other conditional)"), ("dDF", "D − F"), ("GminusC", "G − C (equivalent span)"),
                   ("nest", "D − B (nesting)"), ("BminusA8", "B − A (Stage-0 continuity, 8k)"), ("CminusA8", "C − A (8k)"), ("CminusB8", "C − B (8k)"),
                   ("CminusB25", "C − B (2.5k)"), ("catchup", "(C−B)8k − (C−B)2.5k")):
        L.append(f"| {lab} | {f(per[2][k], per[2][k + '_ci'])} | {f(per[4][k], per[4][k + '_ci'])} |")
    L.append(f"| D-shuf − C (fold 0, point) | {per[2]['shuf']:+.2f} | {per[4]['shuf']:+.2f} |")
    L.append(""); L.append("Decision: " + json.dumps(res["decision"]))
    open(f"{REPORT}/g1_table.md", "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
