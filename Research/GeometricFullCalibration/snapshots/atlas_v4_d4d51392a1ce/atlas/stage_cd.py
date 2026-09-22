"""Stage C+D (CPU, per checkpoint): gates, probability construction, calibration and full evaluation.

Reads the frozen clean selections (shortlist.json, metric_selection.json), the full-size neighbour files (knn_F, u0),
base logits and the role manifests. Fits gates on FIT rows, selects on SELECTION rows, fits the final temperature on the
CALIBRATION rows, then evaluates on all 10 000 clean test images and the 12 development cells (validation rows are never
evaluation data). Target labels enter ONLY the evaluation block at the end of ``evaluate_pipelines``.
  python -m atlas.stage_cd --seed 2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np

from . import common, data, gate, spec, stats

SETS = ("val",) + spec.CONDITIONS


def _load_neighbors(root, cand, name):
    if cand["kind"] == "hidden":
        z = np.load(f"{root}/knn_F/{cand['site']}__{cand['pool']}__{cand['metric']}__{name}.npz")
        return z[f"{cand['metric']}_idx"], z[f"{cand['metric']}_dist"], z["qnorm"]
    z = np.load(f"{root}/u0/{name}.npz")
    return z[f"{cand['metric']}_idx"], z[f"{cand['metric']}_dist"], None


def candidates_from(root):
    sl = json.load(open(f"{root}/shortlist.json"))["chosen"]
    ms = json.load(open(f"{root}/metric_selection.json"))
    c = [{"name": p, "kind": "hidden", "pool": p, "site": sl[p]["site"], "metric": ms["hidden"][p]["metric"]} for p in spec.POOLS]
    c.append({"name": "out", "kind": "output", "pool": "logits", "site": "logits", "metric": ms["output"]["metric"]})
    return c


def run(seed: int, out_root: str = None):
    root = out_root or data.seed_dir(seed)
    marker = f"{root}/cd/done.json"
    if common.is_done(marker):
        print("skip"); return
    t0 = time.time()
    arrays = data.load_seed_arrays(seed)
    val_y = arrays["val"][1].astype(np.int64); bank_y = arrays["train"][1].astype(np.int64)
    ty = np.load(f"{data.SHARED}/test_labels.npy").astype(np.int64)
    prior = np.bincount(bank_y, minlength=spec.NUM_CLASSES) / len(bank_y)
    roles = {k: np.asarray(v) for k, v in json.load(open(f"{data.seed_dir(seed)}/roles.json"))["roles"].items()}
    u0 = json.load(open(f"{root}/u0/done.json"))
    w, w0 = np.asarray(u0["native_weights"]), u0["native_intercept"]
    U = {s: np.load(f"{root}/u0/{s}.npz") for s in SETS}
    Z = {s: U[s]["logits"].astype(np.float64) for s in SETS}
    SG = {s: U[s]["s_glob"].astype(np.float64) for s in SETS}
    Y = {s: (val_y if s == "val" else ty) for s in SETS}
    Q0 = {s: stats.softmax(Z[s] / np.maximum(SG[s] @ w + w0, 1e-12)[:, None]) for s in SETS}
    fit, sel, cal = roles["fit"], roles["selection"], roles["calibration"]
    rows = {"fit": fit, "selection": sel, "calibration": cal}

    # ---------------- controls (fitted on FIT rows; hyper-parameters on SELECTION rows) ----------------
    from Calibrators.temperature_scaling import TemperatureScaling
    from Calibrators.vector_scaling import VectorScaling
    from Calibrators import residual_readout as RR
    ts = TemperatureScaling(); ts.fit(Z["val"][fit], val_y[fit])
    vs = VectorScaling(); vs.fit(Z["val"][fit], val_y[fit])
    msf = {lam: RR.fit_matrix_scaling(Z["val"][fit], val_y[fit], lam) for lam in RR.LAMBDAS}
    ms_lam = min(msf, key=lambda l: RR.nll_np(RR.apply_matrix_scaling(Z["val"][sel], msf[l]["A"], msf[l]["b"]), val_y[sel]))
    probs: Dict[str, Dict[str, np.ndarray]] = {"base": {}, "ts": {}, "vs": {}, "ms": {}, "native_dac": {}}
    for s in SETS:
        probs["base"][s] = stats.softmax(Z[s]); probs["ts"][s] = np.asarray(ts.calibrate(Z[s]))
        probs["vs"][s] = np.asarray(vs.calibrate(Z[s]))
        probs["ms"][s] = stats.softmax(RR.apply_matrix_scaling(Z[s], msf[ms_lam]["A"], msf[ms_lam]["b"]))
        probs["native_dac"][s] = Q0[s]
    meta = {"controls": {"ts_temperature": float(np.asarray(ts.temperature).ravel()[0]), "ms_lambda": ms_lam,
                         "ms_converged": {str(l): bool(v["converged"]) for l, v in msf.items()},
                         "fit_rows": int(len(fit)), "note": "all controls refit on FIT rows only (compatible roles); the full-5000-row benchmark "
                         "baselines are historical context, not used here"}, "gates": {}}

    # ---------------- candidates and gates ----------------
    cands = candidates_from(root)
    base_pred = {s: Z[s].argmax(1) for s in SETS}
    challenger_stats = {}
    for cnd in cands:
        nb = {s: _load_neighbors(root, cnd, s) for s in SETS}
        pg, cnt, kth, qn = {}, {}, {}, {}
        for s in SETS:
            idx, dist, qnorm = nb[s]
            pg[s], cnt[s] = stats.p_geo(idx, bank_y, prior)
            kth[s] = dist[:, -1].astype(np.float64)
            qn[s] = qnorm.astype(np.float64) if qnorm is not None else np.linalg.norm(Z[s] - Z[s].mean(1, keepdims=True), axis=1)
        jc = {s: pg[s].argmax(1) for s in SETS}
        for fs in gate.FEATURE_SETS:
            def feats(s, r=None):
                sl = slice(None) if r is None else r
                z_, i_, j_ = Z[s][sl], base_pred[s][sl], jc[s][sl]
                return gate.f0_features(z_, i_, j_) if fs == "F0" else gate.f1_features(z_, i_, j_, pg[s][sl], kth[s][sl], qn[s][sl])
            i_f, j_f, y_f = base_pred["val"][fit], jc["val"][fit], val_y[fit]
            fitted = gate.fit_gates(feats("val", fit), i_f, j_f, y_f)
            best, table = (gate.select_gate(fitted, feats("val", sel), base_pred["val"][sel], jc["val"][sel], val_y[sel]) if fitted["models"]
                           else ({"kind": "never", "lambda": float("inf"), "theta": None, "net": 0.0, "interventions": 0}, []))
            name = f"{cnd['name']}_{fs}"
            def gvec(s):
                if best["kind"] == "never":
                    return np.zeros(len(Z[s]), bool)
                return gate.gate_from(fitted["models"][best["lambda"]], feats(s), base_pred[s], jc[s], best["theta"])
            probs[name] = {s: gate.build_q(Q0[s], pg[s], gvec(s)) for s in SETS}
            meta["gates"][name] = {"candidate": cnd, "features": fs, "fit_counts": {k: v for k, v in fitted.items() if k != "models"},
                                   "selected": best, "search_table": table[:40], "n_search_configs": len(table)}
        probs[f"{cnd['name']}_alone"] = {s: pg[s] for s in SETS}
        challenger_stats[cnd["name"]] = {"site": cnd["site"], "pool": cnd["pool"], "metric": cnd["metric"]}
    meta["candidates"] = challenger_stats

    # ---------------- final temperatures (calibration rows only) ----------------
    T = {}
    for name, d in probs.items():
        T[name] = {"nll": gate.fit_temperature(d["val"][cal], val_y[cal], "nll"), "brier": gate.fit_temperature(d["val"][cal], val_y[cal], "brier")}
    meta["temperatures"] = T
    common.atomic_json(f"{root}/cd/meta.json", meta)

    # ---------------- evaluation (the only place test/corruption labels are used) ----------------
    from utils.unified_metrics import evaluate_all
    results: Dict = {}
    per_sample: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in spec.CONDITIONS}
    for name, d in probs.items():
        results[name] = {}
        for var, fn in (("raw", lambda q, n=name: q), ("T_nll", lambda q, n=name: gate.apply_temperature(q, T[n]["nll"])),
                        ("T_brier", lambda q, n=name: gate.apply_temperature(q, T[n]["brier"]))):
            results[name][var] = {}
            for c in spec.CONDITIONS:
                q = fn(d[c]); y = Y[c]
                m = evaluate_all(q, y)
                rec = {k: (None if v is None else float(v)) for k, v in m.items()}
                rec["nll"], rec["brier_sum"] = stats.nll(q, y), stats.brier(q, y)
                rec.update({"wh_vs_base": stats.wh_stats(base_pred[c], q.argmax(1), y),
                            "gt_rank_mean_base": float(stats.gt_rank(probs["base"][c], y).mean()), "gt_rank_mean_after": float(stats.gt_rank(q, y).mean())})
                results[name][var][c] = rec
                if var == "T_nll":
                    per_sample[c][f"pred__{name}"] = q.argmax(1).astype(np.int16)
                    per_sample[c][f"rank__{name}"] = stats.gt_rank(q, y).astype(np.int16)
    for c in spec.CONDITIONS:
        per_sample[c]["labels"] = Y[c].astype(np.int16); per_sample[c]["base_pred"] = base_pred[c].astype(np.int16)
        common.atomic_npz(f"{root}/cd/per_sample_{c}.npz", **per_sample[c])
    common.atomic_json(f"{root}/cd/results.json", results)
    common.mark_done(marker, {"seed": seed, "seconds": time.time() - t0, "pipelines": sorted(probs)})
    print("cd done", time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out_root", default=None)
    a = ap.parse_args(); run(a.seed, a.out_root)
