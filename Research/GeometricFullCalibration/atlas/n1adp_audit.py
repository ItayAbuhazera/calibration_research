"""N1a-DP pre-freeze stochasticity / sensitivity audit (docs/n1a_dp_stochasticity_audit_plan.md). ENGINEERING; FROZEN BEFORE EXECUTION.

Access boundary (hard): for audit unit (base b, held-out family f, fold k) only the unit's TRAINING rows are ever built — the three training
families x fold-k outer-train images. No row of family f and no fold-k outer-test image is constructed (asserted). Real Delta_route
targets are used only on those rows.

Procedures (stochastic families lgbm, mlp):
  A  real-unit HPO on the unit's training rows (production shift-transfer context, 3 inner leave-one-training-family-out splits) ->
     I2 quantity = inner selected utility - inner best-constant utility (pp). 5 full-HPO replicates (audit master seeds) + 5 fixed-config
     re-evaluations of replicate 0's selection with new training seeds ("final-fit-only").
  B  pseudo-unit inside the training rows: pseudo-held-out family g (preregistered, cyclic successor of f); pseudo-train = the two other
     training families x inner-fit images; pseudo-eval = g x inner-val images (image-disjoint); pseudo HPO = shift-transfer with 2 inner
     splits over a fixed duplicate-group-aware 75/25 sub-split of the inner-fit images -> phi, U, M_A (the threshold-driving quantities) on
     pseudo-eval. 5 full-HPO replicates + 5 final-fit-only refits of replicate 0's selection.
  Output: ONLY across-replicate variability (SD, max-min, deviations from the replicate mean). No mean, no per-replicate level.
Null (all six families): Delta permuted within each training environment; production procedure (panel seed) on the real unit's training
  rows -> I2 margin and I1 route rate (on pseudo-eval rows' features; label-free). Values are under permuted labels only.

    python -m atlas.n1adp_audit A    --family lgbm --unit 0
    python -m atlas.n1adp_audit B    --family mlp  --unit 3
    python -m atlas.n1adp_audit null --family knn  --unit 1
    python -m atlas.n1adp_audit summary
"""
import argparse
import json
import os
import time

import numpy as np

from . import common, decoder_panel as dp, n1a, n1adp, stage0_data
from .stage0_aggregate import group_id_for_bootstrap

OUT = "results/n1adp_audit"
UNITS = ((2, "gaussian_noise", 0), (4, "defocus_blur", 1), (2, "fog", 2), (4, "jpeg_compression", 3))
AUDIT_SEEDS = tuple(910001 + r for r in range(5))
SUBSPLIT_SEED = 910100
NULL_SEED = 910200
STOCHASTIC = ("lgbm", "mlp")
NBIN = 10


def pseudo_family(heldout):
    F = n1a.FAMILIES
    return F[(F.index(heldout) + 1) % len(F)]


def training_rows(base, heldout, fold):
    """The unit's training rows only (asserted): 3 training families x outer-train images."""
    sp = n1a.split(fold, heldout); data = {}
    for c in sp["train_cells"]:                          # load ONLY the training cells (held-out family cells never read)
        db, do = stage0_data.load_cell(base, c), stage0_data.load_cell(n1a.other(base), c)
        assert np.array_equal(db["labels"], do["labels"])
        data[c] = {"zb": db["z"], "zo": do["z"], "y": db["labels"]}
    X, D, env, img, cell = [], [], [], [], []
    for c in sp["train_cells"]:
        Xc, Dc, mc = n1a.rows(data, sp["train_ids"], (c,), "Z")
        X.append(Xc); D.append(Dc); env += [n1adp.family_of(c)] * len(Dc); img.append(mc[:, 0]); cell.append(mc[:, 1])
    X, D, env, img = np.concatenate(X), np.concatenate(D), np.array(env), np.concatenate(img)
    assert heldout not in set(env.tolist()), "outer held-out family must never be built"
    assert not set(img.tolist()) & set(sp["eval_ids"].tolist()), "outer-test images must never be built"
    return {"X": X, "y": D + 1, "env": env, "img": img, "fit_ids": sp["fit_ids"], "val_ids": sp["val_ids"]}


def real_unit_splits(R):
    fit = np.isin(R["img"], R["fit_ids"])
    return [(np.flatnonzero((R["env"] != g) & fit), np.flatnonzero((R["env"] == g) & ~fit)) for g in sorted(set(R["env"]))]


def pseudo_unit(R, heldout):
    """Pseudo-train / pseudo-eval rows and pseudo inner splits, all within the unit's training rows."""
    g = pseudo_family(heldout); gid = group_id_for_bootstrap()
    F = np.array(sorted(R["fit_ids"])); groups = np.unique(gid[F])
    perm = np.random.default_rng(SUBSPLIT_SEED).permutation(groups); g1 = set(perm[: int(round(0.75 * len(groups)))].tolist())
    F1 = set(F[np.isin(gid[F], list(g1))].tolist())
    fit = np.isin(R["img"], R["fit_ids"]); val = np.isin(R["img"], R["val_ids"])
    tr = np.flatnonzero((R["env"] != g) & fit); ev = np.flatnonzero((R["env"] == g) & val)
    assert not set(R["img"][tr].tolist()) & set(R["img"][ev].tolist())
    env_tr = R["env"][tr]; in1 = np.fromiter((i in F1 for i in R["img"][tr]), bool, len(tr))
    splits = [(np.flatnonzero((env_tr != e) & in1), np.flatnonzero((env_tr == e) & ~in1)) for e in sorted(set(env_tr.tolist()))]
    return tr, ev, splits, g


def ctx_for(ids, splits, env, forbidden):
    return dp.HPOContext("N1a-DP-audit", ids, "shift_transfer", splits, n1adp.neg_policy_utility, forbidden_env=forbidden, env_train=env)


def policy_metrics(lg, y):
    """Threshold-driving quantities on evaluation rows (fractions): phi, U, M_A (+ G)."""
    P = np.exp(lg - lg.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); e = P[:, 2] - P[:, 0]; d = np.asarray(y) - 1
    G = float(np.mean(d * (e > 0))); Gor = float((d == 1).mean()); Gc = max(0.0, float(d.mean()))
    edges = np.quantile(e, np.linspace(0, 1, NBIN + 1)[1:-1]); b = np.digitize(e, edges)
    MA = sum(min((d[b == j] == 1).sum(), (d[b == j] == -1).sum()) for j in range(NBIN)) / len(d)
    return {"G": 100 * G, "phi": (G - Gc) / (Gor - Gc), "U": 100 * (Gor - G), "MA": 100 * MA, "route_rate": float((e > 0).mean())}


def spread(vals):
    """Only variability: SD (ddof=1), max-min, deviations from the replicate mean. The mean itself is NOT emitted."""
    v = np.asarray(vals, float)
    return {"sd": float(v.std(ddof=1)), "range": float(v.max() - v.min()), "centered": [float(x) for x in v - v.mean()], "n": int(len(v))}


def eval_fixed(family, params, best_iters, X, y, splits, seed):
    """Inner utility (macro over splits) of a FIXED configuration with new training seeds (final-fit-only replicate for procedure A)."""
    utils = []
    for si, (fi, vi) in enumerate(splits):
        tseed = dp.derive_seed(seed, si); it = int(best_iters[si])
        if family == "lgbm":
            bst = dp._lgbm_train(params, X[fi], y[fi], None, "classification", 3, tseed, it)
            lg = dp._lgbm_raw(bst, X[vi], "classification", 3)
        else:
            s = dp.Std().fit(X[fi]); net, _ = dp._mlp_train(params, s(X[fi]), y[fi], None, "classification", 3, tseed, it)
            torch = dp._torch()
            with torch.no_grad():
                lg = net(torch.as_tensor(s(X[vi]), dtype=torch.float32)).double().numpy()
        utils.append(-n1adp.neg_policy_utility(lg, y[vi]))
    return float(np.mean(utils))


def _best_iters(rec):
    return next(t["best_iters"] for t in rec["trials"] if t.get("params") == rec["selected"])


def run_A(family, u):
    b, f, k = UNITS[u]; R = training_rows(b, f, k); splits = real_unit_splits(R); const = n1adp.inner_const_utility(R["y"], splits)
    margins, sel0 = [], None
    for r, ms in enumerate(AUDIT_SEEDS):
        dp.MASTER_SEED = ms
        _, rec = dp.fit_decoder(family, R["X"], R["y"], "classification", ctx_for({"unit": u, "proc": "A"}, splits, R["env"], f), K=3)
        margins.append(100 * (-rec["selected_objective"] - const))
        if r == 0:
            sel0 = (rec["selected"], _best_iters(rec))
    dp.MASTER_SEED = AUDIT_SEEDS[0]
    fixed = [100 * (eval_fixed(family, sel0[0], sel0[1], R["X"], R["y"], splits, dp.derive_seed("fitonly", j)) - const) for j in range(5)]
    return {"inner_margin_pp": {"full_hpo": spread(margins), "final_fit_only": spread(fixed)}}


def run_B(family, u):
    b, f, k = UNITS[u]; R = training_rows(b, f, k); tr, ev, splits, g = pseudo_unit(R, f)
    Xt, yt, Xe, ye = R["X"][tr], R["y"][tr], R["X"][ev], R["y"][ev]
    full, sel0 = {q: [] for q in ("phi", "U", "MA", "G")}, None
    for r, ms in enumerate(AUDIT_SEEDS):
        dp.MASTER_SEED = ms
        m, rec = dp.fit_decoder(family, Xt, yt, "classification", ctx_for({"unit": u, "proc": "B"}, splits, R["env"][tr], g), K=3)
        met = policy_metrics(m.predict(Xe), ye)
        for q in full:
            full[q].append(met[q])
        if r == 0:
            sel0 = {"params": rec["selected"], "best_iters": _best_iters(rec)}
    dp.MASTER_SEED = AUDIT_SEEDS[0]
    fo = {q: [] for q in full}
    for j in range(5):
        m = dp._refit(family, Xt, yt, None, "classification", 3, sel0, ctx_for({"unit": u, "proc": "B", "fitonly": j}, splits, R["env"][tr], g))
        met = policy_metrics(m.predict(Xe), ye)
        for q in fo:
            fo[q].append(met[q])
    return {"pseudo_family": g, **{q: {"full_hpo": spread(full[q]), "final_fit_only": spread(fo[q])} for q in full}}


def run_null(family, u):
    b, f, k = UNITS[u]; R = training_rows(b, f, k); splits = real_unit_splits(R); _, ev, _, g = pseudo_unit(R, f)
    y = R["y"].copy(); rng = np.random.default_rng(NULL_SEED + u)
    for e in sorted(set(R["env"].tolist())):
        idx = np.flatnonzero(R["env"] == e); y[idx] = y[rng.permutation(idx)]
    dp.MASTER_SEED = 20260928                                              # production panel seed
    m, rec = dp.fit_decoder(family, R["X"], y, "classification", ctx_for({"unit": u, "proc": "null"}, splits, R["env"], f), K=3)
    lg = m.predict(R["X"][ev]); P = np.exp(lg - lg.max(1, keepdims=True)); rr = float(((P[:, 2] - P[:, 0]) > 0).mean())
    return {"null": True, "inner_margin_pp": 100 * (-rec["selected_objective"] - n1adp.inner_const_utility(y, splits)), "route_rate": rr,
            "selected": rec["selected"]}


def summary():
    """s_family(q) = max over audit units of the full-HPO replicate SD of q; tau_family = max(0.10 pp, 2 s_inner); null STOP check."""
    res = {"note": "variability only; no mean performance", "stochastic": {}, "null": {}}
    for fam in STOCHASTIC:
        A = [json.load(open(f"{OUT}/A/{fam}/u{u}.json")) for u in range(len(UNITS))]
        B = [json.load(open(f"{OUT}/B/{fam}/u{u}.json")) for u in range(len(UNITS))]
        s = {"inner_margin_pp": max(a["inner_margin_pp"]["full_hpo"]["sd"] for a in A)}
        for q in ("phi", "U", "MA", "G"):
            s[q] = max(x[q]["full_hpo"]["sd"] for x in B)
        fo = {"inner_margin_pp": max(a["inner_margin_pp"]["final_fit_only"]["sd"] for a in A), **{q: max(x[q]["final_fit_only"]["sd"] for x in B) for q in ("phi", "U", "MA", "G")}}
        res["stochastic"][fam] = {"s": s, "final_fit_only_sd_max": fo, "tau_pp": max(0.10, 2 * s["inner_margin_pp"]),
                                  "per_unit": {"A": A, "B": B}}
    stop = []
    for fam in dp.FAMILIES:
        tau = res["stochastic"].get(fam, {}).get("tau_pp", 0.10)
        rows = [json.load(open(f"{OUT}/null/{fam}/u{u}.json")) for u in range(len(UNITS))]
        passes = [(0.01 <= r["route_rate"] <= 0.99) and (r["inner_margin_pp"] >= tau) for r in rows]
        res["null"][fam] = {"tau_pp": tau, "units": rows, "I1_and_I2_pass": passes, "n_pass": int(sum(passes))}
        if sum(passes) >= 2:
            stop.append(fam)
    res["STOP"] = bool(stop); res["stop_families"] = stop
    res["stop_rule"] = "STOP if any family passes I1 and I2 in >= 2 of the 4 null units"
    common.atomic_json(f"{OUT}/summary.json", res); print(json.dumps({k: v for k, v in res.items() if k != "stochastic"}, indent=1, default=str)[:3000])
    print(json.dumps({f: {"s": v["s"], "tau_pp": v["tau_pp"], "final_fit_only_sd_max": v["final_fit_only_sd_max"]} for f, v in res["stochastic"].items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("proc", choices=("A", "B", "null", "summary"))
    ap.add_argument("--family", choices=dp.FAMILIES); ap.add_argument("--unit", type=int, choices=range(len(UNITS)))
    a = ap.parse_args()
    if a.proc == "summary":
        summary()
    else:
        assert a.proc == "null" or a.family in STOCHASTIC
        t0 = time.time(); out = {"A": run_A, "B": run_B, "null": run_null}[a.proc](a.family, a.unit)
        out.update(unit=dict(zip(("base", "heldout", "fold"), UNITS[a.unit])), family=a.family, proc=a.proc, seconds=time.time() - t0,
                   host=os.uname().nodename)
        common.atomic_json(f"{OUT}/{a.proc}/{a.family}/u{a.unit}.json", out); print("wrote", a.proc, a.family, a.unit, f"{out['seconds']:.0f}s")
