"""N1a aggregation (docs/n1a_action_ambiguity_spec.md sec. 4-6): Phase 1 oracle action structure (from cached logits), Phase 2 output-only
ambiguity (from the fits), image-group bootstrap, frozen decision. Deterministic.

    python -m atlas.n1a_aggregate
"""
import json
import os

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import n1a, spec, stage0_folds
from .n1a_fit import OUT as FITS
from .n1a_rules import decide
from .stage0_aggregate import BOOT_SEED, group_id_for_bootstrap

REPORT = "results/n1a/report"
B = 2000
K_NN = 20


def phase1(base, fold_of):
    """Counts per (family, severity, fold) for route and ensemble."""
    data = n1a.load(base); tab = []
    for c in spec.CELLS:
        d = data[c]; oc = n1a.outcomes(d["zb"], d["zo"], d["y"])
        fam, sev = c.rsplit("_s", 1)
        for k in range(5):
            m = fold_of == k
            for act in ("route", "ens"):
                u = oc[act][m]; kp = oc["keep"][m]
                tab.append({"family": fam, "severity": int(sev), "fold": k, "action": act, "n": int(m.sum()),
                            "A_both_correct": int(((kp == 1) & (u == 1)).sum()), "B_both_wrong": int(((kp == 0) & (u == 0)).sum()),
                            "C_repair": int(((kp == 0) & (u == 1)).sum()), "D_harm": int(((kp == 1) & (u == 0)).sum())})
    return tab


def pool(base, family, arm):
    xs = [np.load(f"{FITS}/base{base}/{family}/{arm}/fold{k}.npz", allow_pickle=True) for k in range(5)]
    cat = lambda key: np.concatenate([x[key] for x in xs])
    meta = [{"fold": int(x["fold"]), "lambda": float(x["selected_lambda"]), "converged": bool(x["converged"]), "retried": bool(x["retried"]),
             "train_mean_delta": float(x["train_mean_delta"]), "wall_min": float(x["wall_time_s"]) / 60} for x in xs]
    fold = np.concatenate([np.full(len(x["d_route"]), int(x["fold"])) for x in xs])
    return {"img": cat("image_id"), "cell": cat("cell_index"), "d": cat("d_route").astype(np.int64), "e": cat("e"), "z": cat("zstd_eval"), "fold": fold,
            "tmd": np.concatenate([np.full(len(x["d_route"]), float(x["train_mean_delta"])) for x in xs])}, meta


def knn_ambiguity(P):
    """Per row: min(p+, p-) over the k nearest evaluation rows (same family & fold; different image) plus itself."""
    m = np.zeros(len(P["d"]))
    for k in np.unique(P["fold"]):
        idx = np.where(P["fold"] == k)[0]; z = P["z"][idx].astype(np.float64); img = P["img"][idx]; d = P["d"][idx]
        nn = NearestNeighbors(n_neighbors=K_NN + 8).fit(z); _, nb = nn.kneighbors(z)
        for r in range(len(idx)):
            cand = [j for j in nb[r] if img[j] != img[r]][:K_NN]
            dd = np.concatenate([[d[r]], d[cand]])
            m[idx[r]] = min((dd == 1).mean(), (dd == -1).mean())
    return m


def main():
    os.makedirs(REPORT, exist_ok=True)
    plan = stage0_folds.load_plan(); fold_of = np.array(plan["fold_id"])
    gid = group_id_for_bootstrap(); _, ginv = np.unique(gid, return_inverse=True); ng = ginv.max() + 1
    res = {"label": "N1a development audit on exposed cells (checkpoints 2/4 routed symmetrically); no internal representations; "
                    "family + image-identity holdout; 95% image-group bootstrap", "bases": {}}
    per_base = {}
    for base in n1a.BASES:
        rng = np.random.default_rng(BOOT_SEED + 50000 + base); idx = rng.integers(0, ng, (B, ng))
        R = np.zeros((B, ng), np.float32)
        for bi in range(B):
            R[bi] = np.bincount(idx[bi], minlength=ng)
        fam_stats, conv, metas = {}, True, {}
        boot = {k: np.zeros((B, len(n1a.FAMILIES))) for k in ("h", "MA", "GZ", "GZZo", "Q", "MB")}
        for fi, fam in enumerate(n1a.FAMILIES):
            PZ, mZ = pool(base, fam, "Z"); PR, mR = pool(base, fam, "ZZo")
            order = np.lexsort((PZ["cell"], PZ["img"])); orderR = np.lexsort((PR["cell"], PR["img"]))
            for k in ("img", "cell", "d", "e", "z", "fold", "tmd"):
                PZ[k] = PZ[k][order]; PR[k] = PR[k][orderR]
            assert np.array_equal(PZ["img"], PR["img"]) and np.array_equal(PZ["d"], PR["d"])
            conv &= all(m["converged"] for m in mZ + mR); metas[fam] = {"Z": mZ, "ZZo": mR}
            d, g = PZ["d"], ginv[PZ["img"]]
            rZ, rR = PZ["e"] > 0, PR["e"] > 0
            edges = np.quantile(PZ["e"], np.linspace(0, 1, 11)[1:-1]); bins = np.digitize(PZ["e"], edges)
            mb = knn_ambiguity(PZ)
            # per-group sufficient statistics
            cols = {"n": np.ones_like(d, float), "pos": (d == 1).astype(float), "neg": (d == -1).astype(float), "gz": d * rZ, "gr": d * rR, "mb": mb}
            S = {k: np.bincount(g, weights=v, minlength=ng) for k, v in cols.items()}
            Sb = {s: np.stack([np.bincount(g[bins == j], weights=(d[bins == j] == (1 if s == "pos" else -1)).astype(float), minlength=ng) for j in range(10)], 1) for s in ("pos", "neg")}
            N = R @ S["n"]
            pos, neg = R @ S["pos"] / N, R @ S["neg"] / N
            boot["h"][:, fi] = np.minimum(pos, neg); boot["GZ"][:, fi] = R @ S["gz"] / N; boot["GZZo"][:, fi] = R @ S["gr"] / N
            boot["Q"][:, fi] = boot["GZZo"][:, fi] - boot["GZ"][:, fi]; boot["MB"][:, fi] = R @ S["mb"] / N
            boot["MA"][:, fi] = np.minimum(R @ Sb["pos"], R @ Sb["neg"]).sum(1) / N
            n = len(d); P1, M1 = (d == 1).mean(), (d == -1).mean()
            ma = sum(min((d[bins == j] == 1).sum(), (d[bins == j] == -1).sum()) for j in range(10)) / n
            const_train = float(np.mean(np.where(PZ["tmd"] > 0, d, 0)))
            topk = {}
            for q in (5, 10, 20):
                for name, e in (("Z", PZ["e"]), ("ZZo", PR["e"])):
                    thr = np.quantile(e, 1 - q / 100); topk[f"{name}_top{q}pct_gain_pp"] = 100 * float(np.mean(np.where(e >= thr, d, 0)))
            fam_stats[fam] = {"n_eval_rows": n, "support_repair": int((d == 1).sum()), "support_harm": int((d == -1).sum()),
                              "P_repair": 100 * P1, "P_harm": 100 * M1, "P_unchanged": 100 * (1 - P1 - M1), "G_oracle": 100 * P1,
                              "G_const_eval": 100 * max(0.0, float(d.mean())), "G_const_train": 100 * const_train, "h": 100 * min(P1, M1),
                              "G_Z": 100 * float(np.mean(d * rZ)), "G_ZZo_reference": 100 * float(np.mean(d * rR)), "Q": 100 * float(np.mean(d * rR) - np.mean(d * rZ)),
                              "route_rate_Z": 100 * float(rZ.mean()), "route_rate_ZZo": 100 * float(rR.mean()),
                              "R_Z_regret_vs_oracle": 100 * (P1 - float(np.mean(d * rZ))), "MA": 100 * ma, "MB": 100 * float(mb.mean()),
                              "captured_fraction_of_headroom_Z": (float(np.mean(d * rZ)) - max(0.0, float(d.mean()))) / min(P1, M1) if min(P1, M1) > 0 else None,
                              "budget_curves": topk,
                              "by_severity": {int(s): {"P_repair": 100 * float((d[PZ['cell'] == spec.CELLS.index(f'{fam}_s{s}')] == 1).mean()),
                                                       "P_harm": 100 * float((d[PZ['cell'] == spec.CELLS.index(f'{fam}_s{s}')] == -1).mean())} for s in spec.DEV_SEVERITIES}}
        ci = lambda a: [100 * float(np.quantile(a.mean(1), q)) for q in (.025, .975)]
        pt = lambda k: float(np.mean([fam_stats[f][k] for f in n1a.FAMILIES]))
        q = {"h": pt("h"), "h_ci": ci(boot["h"]), "MA": pt("MA"), "MA_ci": ci(boot["MA"]), "MA_by_family": {f: fam_stats[f]["MA"] for f in n1a.FAMILIES},
             "MB": pt("MB"), "MB_ci": ci(boot["MB"]), "Q": pt("Q"), "Q_ci": ci(boot["Q"]), "G_Z": pt("G_Z"), "G_Z_ci": ci(boot["GZ"]),
             "G_oracle": pt("G_oracle"), "G_const_eval": pt("G_const_eval"), "G_const_train": pt("G_const_train"),
             "support": {f: (fam_stats[f]["support_repair"], fam_stats[f]["support_harm"]) for f in n1a.FAMILIES}, "converged": bool(conv)}
        per_base[base] = q
        res["bases"][str(base)] = {"family_macro": q, "families": fam_stats, "phase1": phase1(base, fold_of), "fit_meta": metas}
    res["decision"] = decide(per_base)
    json.dump(res, open(f"{REPORT}/n1a_aggregate.json", "w"), indent=1, default=float)
    print(json.dumps({b: {k: v for k, v in q.items() if k != "support"} for b, q in per_base.items()}, indent=1, default=float)); print(res["decision"])


if __name__ == "__main__":
    main()
