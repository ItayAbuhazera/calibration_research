"""Fixed-gate study (CPU): stage `freeze` fits and selects every gate on clean rows and writes an immutable `gates_frozen.json`;
stage `eval` refuses to run without it, evaluates clean generalization on the calibration rows, fits final temperatures, then evaluates
the 10 000-image clean set and the 12 development cells. Spec: docs/fixed_gate_study_spec.md.
  python -m atlas.fg_run --seed 2 --stage freeze|eval
Outcome labels: freeze reads only validation-row labels of the fit/selection roles; the only place test/corruption labels are read is the
`evaluate` block of the eval stage (after temperatures are fixed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from typing import Dict, List

import numpy as np

from . import common, data, fg_data, fg_gate as FG, gate, spec, stats

SETS = fg_data.SETS


def gate_specs():
    """[(candidate, family, order_label, n)] — deep: 4 families x (3 orders x {625,1250} + full 2500 once); controls: full only."""
    out = []
    for fam in FG.FAMILIES:
        for o in range(len(FG.ORDER_SEEDS)):
            for n in (625, 1250):
                out.append(("deep", fam, o, n))
        out.append(("deep", fam, "full", 2500))
    for c in ("gap4", "out"):
        for fam in FG.FAMILIES:
            out.append((c, fam, "full", 2500))
    return out


def gname(c, fam, o, n):
    return f"{c}|{fam}|o{o}|n{n}"


def _feat(cand, ctx, fam, s, rows):
    z, i = ctx["Z"][s][rows], ctx["base_pred"][s][rows]
    c = cand[s]
    return FG.features(fam, z, i, c["j"][rows], c["pg"][rows], c["kth"][rows], c["qnorm"][rows])


def freeze(seed: int, root_out: str = None):
    root = root_out or f"{fg_data.FG}/seed{seed}"
    marker = f"{root}/frozen.done.json"
    if common.is_done(marker):
        print("skip"); return
    ctx = fg_data.load_ctx(seed)
    fit, sel = ctx["roles"]["fit"], ctx["roles"]["selection"]
    vy = ctx["val_y"]
    cands = {c: fg_data.load_candidate(ctx, c, sets=("val",)) for c in fg_data.CANDS}
    orders = [fit[FG.stratified_order(vy[fit], s)] for s in FG.ORDER_SEEDS]       # absolute row indices, nested prefixes
    frozen: Dict = {}
    for (c, fam, o, n) in gate_specs():
        rows = fit if o == "full" else orders[o][:n]
        assert len(rows) == n
        cand = cands[c]
        Xf = _feat(cand, ctx, fam, "val", rows); i_f, j_f = ctx["base_pred"]["val"][rows], cand["val"]["j"][rows]
        fitted = FG.fit_family(Xf, i_f, j_f, vy[rows])
        if fitted["degenerate"]:
            best, table = {"kind": "never", "lambda": float("inf"), "theta": None, "net": 0.0, "interventions": 0, "order": -1}, []
        else:
            Xs = _feat(cand, ctx, fam, "val", sel)
            best, table = FG.select_config(fitted, Xs, ctx["base_pred"]["val"][sel], cand["val"]["j"][sel], vy[sel])
        frozen[gname(c, fam, o, n)] = {
            "candidate": c, "family": fam, "order": o, "n": n, "fit_rows_sha256": hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest()[:16],
            "counts": {k: fitted[k] for k in ("n", "m", "W", "H", "U", "degenerate")},
            "design": {k: fitted.get(k) for k in ("n_features", "constant_features", "design_rank_with_intercept")},
            "selected": {k: (None if (isinstance(v, float) and np.isinf(v)) else v) for k, v in best.items()},
            "selected_lambda_is_never": best["kind"] == "never",
            "search_table": [{**t, "lambda": (None if np.isinf(t["lambda"]) else t["lambda"])} for t in table],
            "models": {str(l): FG.ser(m) for l, m in fitted["models"].items()}}
    payload = {"seed": seed, "spec_sha256": open("docs/fixed_gate_study_spec.frozen.sha256").read().strip(), "gates": frozen,
               "candidates": {c: {k: v for k, v in ctx["defs"][c].items() if k != "dir"} for c in fg_data.CANDS},
               "order_seeds": list(FG.ORDER_SEEDS), "role_sizes": {k: int(len(v)) for k, v in ctx["roles"].items()}}
    path = f"{root}/gates_frozen.json"
    common.atomic_json(path, payload)
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    common.mark_done(marker, {"seed": seed, "sha256": common.file_sha(path), "n_gates": len(frozen)})
    print("frozen", len(frozen), common.file_sha(path))


# --------------------------------------------------------------------------------------------- evaluation
def ece15(q, y):
    conf = q.max(1); ok = (q.argmax(1) == y).astype(float)
    b = np.minimum((conf * 15).astype(int), 14); e = 0.0
    for k in range(15):
        m = b == k
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def metrics(q, y, base_pred):
    p = q.argmax(1); rk = stats.gt_rank(q, y)
    lq = np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1.0))
    rec = {"accuracy": float((p == y).mean()), "nll": float(-lq.mean()), "brier_sum": stats.brier(q, y), "ece15": ece15(q, y), "gt_rank_mean": float(rk.mean())}
    rec["vs_base"] = stats.wh_stats(base_pred, p, y)
    return rec, p, rk, lq


def evaluate(seed: int, root_out: str = None):
    root = root_out or f"{fg_data.FG}/seed{seed}"
    marker = f"{root}/eval.done.json"
    if common.is_done(marker):
        print("skip"); return
    fpath = f"{root}/gates_frozen.json"
    fm = json.load(open(f"{root}/frozen.done.json"))
    if not os.path.exists(fpath) or common.file_sha(fpath) != fm["sha256"]:
        raise SystemExit("frozen gate file missing or hash mismatch: refusing to evaluate")
    fz = json.load(open(fpath))
    t0 = time.time()
    ctx = fg_data.load_ctx(seed)
    roles = ctx["roles"]; fit, sel, cal = roles["fit"], roles["selection"], roles["calibration"]
    vy = ctx["val_y"]; Z, Y, BP = ctx["Z"], ctx["Y"], ctx["base_pred"]
    w, w0 = np.asarray(ctx["u0"]["native_weights"]), ctx["u0"]["native_intercept"]
    SG = {s: ctx["U"][s]["s_glob"].astype(np.float64) for s in SETS}
    Q0 = {s: stats.softmax(Z[s] / np.maximum(SG[s] @ w + w0, 1e-12)[:, None]) for s in SETS}
    cands = {c: fg_data.load_candidate(ctx, c) for c in fg_data.CANDS}
    from Calibrators.temperature_scaling import TemperatureScaling
    from Calibrators.vector_scaling import VectorScaling
    from Calibrators import residual_readout as RR
    ts = TemperatureScaling(); ts.fit(Z["val"][fit], vy[fit]); vs = VectorScaling(); vs.fit(Z["val"][fit], vy[fit])
    msf = {lam: RR.fit_matrix_scaling(Z["val"][fit], vy[fit], lam) for lam in RR.LAMBDAS}
    ms_lam = min(msf, key=lambda l: RR.nll_np(RR.apply_matrix_scaling(Z["val"][sel], msf[l]["A"], msf[l]["b"]), vy[sel]))

    # ---- pipeline definitions: name -> (function set -> q_raw, optional gate info)
    pipes: Dict[str, dict] = {}
    pipes["base"] = {"q": lambda s: stats.softmax(Z[s])}
    pipes["native_dac"] = {"q": lambda s: Q0[s]}                                   # == never-intervene
    pipes["ts"] = {"q": lambda s: np.asarray(ts.calibrate(Z[s]))}
    pipes["vs"] = {"q": lambda s: np.asarray(vs.calibrate(Z[s]))}
    pipes["ms"] = {"q": lambda s: stats.softmax(RR.apply_matrix_scaling(Z[s], msf[ms_lam]["A"], msf[ms_lam]["b"]))}
    gate_cache: Dict = {}
    gen: Dict = {}

    def gate_out(name, s):
        """(g, score) of frozen gate `name` on set s; cached for the most recent 8 (name, set) pairs."""
        key = (name, s)
        if key not in gate_cache:
            g = fz["gates"][name]; cand = cands[g["candidate"]]; sc = g["selected"]
            if len(gate_cache) > 8:
                gate_cache.clear()
            if sc["kind"] == "never":
                gate_cache[key] = (np.zeros(len(BP[s]), bool), np.full(len(BP[s]), np.nan))
            else:
                cfg = {"kind": "gate", "lambda": sc["lambda"], "theta": sc["theta"]}
                model = FG.deser(g["models"][str(sc["lambda"])])
                X = FG.features(g["family"], Z[s], BP[s], cand[s]["j"], cand[s]["pg"], cand[s]["kth"], cand[s]["qnorm"])
                gate_cache[key] = FG.apply_config(cfg, model, X, BP[s], cand[s]["j"])
        return gate_cache[key]
    for name, g in fz["gates"].items():
        pipes[name] = {"q": (lambda s, name=name: gate.build_q(Q0[s], cands[fz["gates"][name]["candidate"]][s]["pg"], gate_out(name, s)[0])), "gate": name}
    for c in fg_data.CANDS:
        pipes[f"{c}_alone"] = {"q": (lambda s, c=c: cands[c][s]["pg"])}
        pipes[f"{c}_always"] = {"q": (lambda s, c=c: gate.build_q(Q0[s], cands[c][s]["pg"], cands[c][s]["j"] != BP[s]))}

    # ---- (1) clean generalization of frozen gates on fit / selection / calibration rows and clean test, BEFORE temperature or any corruption label
    gen = {}
    for name, g in fz["gates"].items():
        cand = cands[g["candidate"]]; sc = g["selected"]
        cfgl = float("inf") if sc["lambda"] is None else sc["lambda"]
        res = {}
        for label, s, rows in (("fit", "val", fit), ("selection", "val", sel), ("calibration", "val", cal), ("clean_test", "clean", np.arange(len(Y["clean"])))):
            gg, sc_ = gate_out(name, s)
            gg, sc_ = gg[rows], sc_[rows]
            i_, j_, y_ = BP[s][rows], cand[s]["j"][rows], Y[s][rows]
            D = gate.utility_target(i_, j_, y_); dis = j_ != i_
            r = {"n": int(len(y_)), "m": int(dis.sum()), "interventions": int(gg.sum()), "coverage": float(gg.mean()),
                 "W": int((gg & (D > 0)).sum()), "H": int((gg & (D < 0)).sum()), "U": int((gg & (D == 0)).sum()), "net_gain": float(D[gg].sum() / len(y_)),
                 "always_net_gain": float(D[dis].sum() / len(y_)), "always_W": int((dis & (D > 0)).sum()), "always_H": int((dis & (D < 0)).sum()), "always_U": int((dis & (D == 0)).sum())}
            if sc["kind"] == "gate" and not np.isnan(sc_).all():
                r["mse_all_disagreements"] = float(np.mean((D[dis] - sc_[dis]) ** 2)) if dis.any() else None
                mdl = FG.deser(g["models"][str(cfgl)])
                r["mse_const_fit_mean"] = float(np.mean((D[dis] - mdl["b"]) ** 2)) if dis.any() else None
            res[label] = r
        gen[name] = res
    common.atomic_json(f"{root}/gate_generalization.json", gen)

    # ---- (2) temperatures on the calibration rows only
    T = {}
    for name, p in pipes.items():
        T[name] = gate.fit_temperature(p["q"]("val")[cal], vy[cal], "nll")
    common.atomic_json(f"{root}/temperatures.json", T)

    # ---- (3) evaluation (labels of test/corruption sets used here only)
    results: Dict = {}; per: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in spec.CONDITIONS}
    argmax_bad: Dict = {}
    for name, p in pipes.items():
        results[name] = {"raw": {}, "T": {}, "roles": {}}
        for s in SETS:
            q = p["q"](s); y = Y[s]
            if "gate" in p:
                gg = gate_out(p["gate"], s)[0]; cj = cands[fz["gates"][p["gate"]]["candidate"]][s]["j"]
                argmax_bad[f"{name}|{s}"] = int((q.argmax(1) != np.where(gg, cj, BP[s])).sum())
            qT = gate.apply_temperature(q, T[name])
            if s == "val":
                for role, rows in (("fit", fit), ("selection", sel), ("calibration", cal)):
                    results[name]["roles"][role] = {"raw": metrics(q[rows], y[rows], BP[s][rows])[0], "T": metrics(qT[rows], y[rows], BP[s][rows])[0]}
                continue
            rr, pr, rkr, lqr = metrics(q, y, BP[s]); rT, pT, rkT, lqT = metrics(qT, y, BP[s])
            assert (pT == pr).all(), "temperature changed the argmax"
            results[name]["raw"][s] = rr; results[name]["T"][s] = rT
            per[s][f"pred__{name}"] = pr.astype(np.int16); per[s][f"lqy_raw__{name}"] = lqr.astype(np.float32); per[s][f"lqy_T__{name}"] = lqT.astype(np.float32)
            per[s][f"rank_raw__{name}"] = rkr.astype(np.int16); per[s][f"rank_T__{name}"] = rkT.astype(np.int16)
            if "gate" in p:
                gg, sc_ = gate_out(p["gate"], s); per[s][f"g__{name}"] = gg.astype(np.uint8)
                if name.endswith("|o" + "full|n2500"):
                    per[s][f"score__{name}"] = sc_.astype(np.float32)
        print(name, round(time.time() - t0), flush=True)
    for s in spec.CONDITIONS:
        per[s]["labels"] = Y[s].astype(np.int16); per[s]["base_pred"] = BP[s].astype(np.int16)
        for c in fg_data.CANDS:
            per[s][f"cand_j__{c}"] = cands[c][s]["j"].astype(np.int16)
        common.atomic_npz(f"{root}/per_sample_{s}.npz", **per[s])
    meta = {"seed": seed, "ms_lambda": ms_lam, "ts_temperature": float(np.asarray(ts.temperature).ravel()[0]),
            "argmax_mismatch_counts_nonzero": {k: v for k, v in argmax_bad.items() if v}, "argmax_checked": len(argmax_bad), "seconds": time.time() - t0,
            "temperature_argmax_preserved": True}
    common.atomic_json(f"{root}/results.json", results); common.atomic_json(f"{root}/eval_meta.json", meta)
    common.mark_done(marker, {"seed": seed, "pipelines": len(pipes)})
    print("eval done", time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--stage", required=True, choices=["freeze", "eval"])
    ap.add_argument("--out_root", default=None); a = ap.parse_args()
    (freeze if a.stage == "freeze" else evaluate)(a.seed, a.out_root)
