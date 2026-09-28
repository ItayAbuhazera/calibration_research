"""Decoder Panel v1 ENGINEERING benchmarks (docs/decoder_panel_v1_resource_plan.md). NO SCIENTIFIC LABELS ARE READ.

Targets are synthetic: a fixed random teacher applied to label-free inputs (atlas logits Z, cached H_L). The purpose is runtime,
memory, Optuna-budget convergence (best-so-far after 5/10/20/30/40/50 trials) and CPU-vs-GPU MLP timing on representative
G1-DP and N1a-DP task units. Nothing here measures or exposes a scientific outcome.

    python -m atlas.dp_engineering --mode g1 --family lgbm [--device cuda] [--trials 50]
    python -m atlas.dp_engineering --mode n1a --family knn
"""
import argparse
import json
import os
import resource
import shutil
import time

import numpy as np

from . import decoder_panel as dp, n1a, spec, stage0_folds

OUT = "results/decoder_panel/engineering"
TEACHER_SEED = 777001


def _teacher(Xs, k_out, seed):
    rng = np.random.default_rng(seed); d = Xs.shape[1]
    R1 = rng.standard_normal((d, 64)) / np.sqrt(d); R2 = rng.standard_normal((64, k_out))
    return np.tanh(Xs @ R1 * 2) @ R2


def g1_unit(seed=2, fold=0, scratch=None):
    """Seed-2 fold-0 rows (T-8k x 1 layout) with label-free inputs; synthetic 100-class target from a teacher on (z, H_L)."""
    plan = stage0_folds.load_plan(); o = plan["outer"][fold]
    train = np.array(o["train_idx"]); inner = np.array(o["inner_fit_mask"], bool); ca = np.array(o["cell_assignment_local"])
    t0 = time.time()
    src = "results/g1/hL/seed%d" % seed
    if scratch:
        os.makedirs(scratch, exist_ok=True)
        for c in spec.CELLS:
            shutil.copy(f"{src}/h_{c}.npy", scratch)
        src = scratch
    copy_s = time.time() - t0
    z = np.empty((len(train), 100)); h = np.empty((len(train), 2048), np.float32)
    for ci, c in enumerate(spec.CELLS):
        m = ca == ci
        z[m] = np.load(f"results/atlas/seed{seed}/u0/{c}.npz")["logits"][train[m]]
        h[m] = np.load(f"{src}/h_{c}.npy", mmap_mode="r")[train[m]]
    big = np.concatenate([z, h, h[:, :1024]], 1)                      # 3172-d stand-in for the widest arm (z + H_L + H_3.22)
    s = dp.Std().fit(np.concatenate([z, h], 1)); y = (z + 3 * _teacher(s(np.concatenate([z, h], 1)), 100, TEACHER_SEED)
                                                        + np.random.default_rng(TEACHER_SEED + 1).gumbel(size=(len(train), 100))).argmax(1)
    fi, vi = np.flatnonzero(inner), np.flatnonzero(~inner)
    return {"A": z, "C": np.concatenate([z, h], 1), "I": big}, z, y, (fi, vi), copy_s


def n1a_unit(base=2, family="fog", fold=0):
    """Base-2, fog held out, fold 0 (N1a layout: 3 training families x 3 severities x training images); synthetic Delta target."""
    sp = n1a.split(fold, family); zb = {c: np.load(f"results/atlas/seed{base}/u0/{c}.npz")["logits"] for c in sp["train_cells"]}
    ids = sp["train_ids"]; X, env, img = [], [], []
    for c in sp["train_cells"]:
        X.append(n1a.output_features(zb[c][ids])); env += [c.rsplit("_s", 1)[0]] * len(ids); img.append(ids)
    X = np.concatenate(X); env = np.array(env); img = np.concatenate(img)
    s = _teacher(dp.Std().fit(X)(X), 1, TEACHER_SEED)[:, 0]; s = s / s.std()
    u = np.random.default_rng(TEACHER_SEED + 2).random(len(X)); pp = 0.09 * (1 + np.tanh(s)); pm = 0.09 * (1 - np.tanh(s))
    d = np.where(u < pp, 1, np.where(u < pp + pm, -1, 0)); y = d + 1
    fit_img = set(sp["fit_ids"].tolist())
    in_fit = np.array([i in fit_img for i in img])
    fams = sorted(set(env)); splits = []
    for f in fams:
        splits.append((np.flatnonzero((env != f) & in_fit), np.flatnonzero((env == f) & ~in_fit)))
    return X, y, env, splits


def neg_utility(pred, y, _rows=None):
    p = np.exp(pred - pred.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    route = (p[:, 2] - p[:, 0]) > 0
    return -float(np.mean((y - 1) * route))


def best_so_far(trials):
    b, out = np.inf, []
    for t in trials:
        b = min(b, t["objective"]); out.append(b)
    return {str(k): out[k - 1] for k in (5, 10, 20, 30, 40, 50) if k <= len(out)}


def main(mode, family, device, trials, arms):
    os.makedirs(OUT, exist_ok=True); tag = f"{mode}_{family}_{device}_{trials}"
    res = {"mode": mode, "family": family, "device": device, "trials": trials, "host": os.uname().nodename,
           "cpus": dp.n_threads(), "note": "ENGINEERING ONLY - synthetic teacher targets, no scientific labels"}
    t0 = time.time()
    if mode == "g1":
        scratch = f"/tmp/dp_eng_{os.environ.get('SLURM_JOB_ID', 'x')}"
        Xs, z, y, (fi, vi), copy_s = g1_unit(scratch=scratch); res["scratch_copy_seconds_H_L_12cells"] = copy_s
        shutil.rmtree(scratch, ignore_errors=True)
        ctx = dp.HPOContext("eng_g1", {"seed": 2, "fold": 0}, "target_pooled", [(fi, vi)], dp.nll_objective)
        res["load_seconds"] = time.time() - t0; res["arms"] = {}
        for arm in arms:
            t1 = time.time()
            m, rec = dp.fit_decoder(family, Xs[arm], y, "classification", ctx, offset=z, n_trials=trials, device=device)
            tp = time.time(); _ = m.predict(np.tile(Xs[arm][:2000], (13, 1)), np.tile(z[:2000], (13, 1))); pred_s = time.time() - tp
            res["arms"][arm] = {"dim": int(Xs[arm].shape[1]), "seconds_total": time.time() - t1, "hpo_seconds": rec["hpo_seconds"],
                                "refit_seconds": rec["refit_seconds"], "predict_26000_rows_seconds": pred_s, "selected": rec["selected"],
                                "best_so_far": best_so_far(rec["trials"]) if family in dp.OPTUNA_FAMILIES else None,
                                "trial_seconds": [t.get("seconds") for t in rec["trials"]] if family in dp.OPTUNA_FAMILIES else None,
                                "refit_meta": {k: v for k, v in rec["refit_meta"].items() if k != "jl_audit"},
                                "jl_audit": rec["refit_meta"].get("jl_audit"), "grid_edge": rec["grid_edge"],
                                "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
            print(arm, json.dumps(res["arms"][arm], default=str)[:600], flush=True)
    else:
        X, y, env, splits = n1a_unit()
        ctx = dp.HPOContext("eng_n1a", {"base": 2, "heldout": "fog", "fold": 0}, "shift_transfer", splits, neg_utility,
                            forbidden_env="fog", env_train=env)
        res["load_seconds"] = time.time() - t0; t1 = time.time()
        m, rec = dp.fit_decoder(family, X, y, "classification", ctx, K=3, n_trials=trials, device=device)
        res["arms"] = {"Z": {"dim": int(X.shape[1]), "n_train": int(len(y)), "seconds_total": time.time() - t1, "hpo_seconds": rec["hpo_seconds"],
                             "refit_seconds": rec["refit_seconds"], "selected": rec["selected"], "grid_edge": rec["grid_edge"],
                             "best_so_far": best_so_far(rec["trials"]) if family in dp.OPTUNA_FAMILIES else None,
                             "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}}
        print(json.dumps(res["arms"], default=str)[:800], flush=True)
    res["wall_seconds"] = time.time() - t0
    dp.save_record(f"{OUT}/{tag}.json", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("g1", "n1a"), required=True); ap.add_argument("--family", choices=dp.FAMILIES, required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--trials", type=int, default=dp.N_TRIALS)
    ap.add_argument("--arms", default="A,C,I")
    a = ap.parse_args(); main(a.mode, a.family, a.device, a.trials, a.arms.split(","))
