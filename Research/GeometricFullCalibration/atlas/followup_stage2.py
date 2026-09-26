"""Stage 2 fits (amendment 1 sec. 2, 6): anchored C1e, C1c, C1d, K, Z+K, Prow (+ a Z-only re-fit for probabilities and a reproduction check),
T-8k x 1 target fits (ORACLE DIAGNOSTIC), one (state, fold). CPU only. Inputs: results/regime_map/<state>/<cond>.npz (z, p) and
results/regime_map_followup/hL/<state>/ (h_L, p_L, head).

    python -m atlas.followup_stage2 --state b10_s1 --fold 0
"""
import argparse
import json
import os
import time

import numpy as np

from . import regime_data, spec, stage0_folds
from .followup_anchored import INF, predict_logits, select_and_fit

HROOT = "results/regime_map_followup/hL"
OUT = "results/regime_map_followup/stage2"
ARMS = ("c1e", "c1c", "c1d", "kernel", "zk", "prow")
ALL = ("zonly",) + ARMS


def projectors(state):
    hd = np.load(f"{HROOT}/{state}/head.npz")
    Wh = hd["W_h"]; Wp = np.linalg.pinv(Wh)
    kind = str(hd["kind"]) if hd["kind"].shape == () else "b"
    mu, sg = (hd["mu"], hd["sigma"]) if "mu" in hd.files else (None, None)
    def u_of(h):
        h = np.asarray(h, dtype=np.float64)
        if mu is None:
            return h
        return (h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12) - mu) / sg
    def split(h):
        u = u_of(h); r = (u @ Wh.T) @ Wp.T
        return u - r, r                                        # P_ker u, P_row u
    return split


def gather(cells, H, PL, split, ids, cell_assign):
    """Rows of the T-8k x 1 rule: each image at its preassigned corrupted cell."""
    n = len(ids); z = np.empty((n, 100)); p = np.empty((n, 100)); pl = np.empty((n, 100)); h = np.empty((n, 2048)); y = np.empty(n, np.int64)
    for ci, cell in enumerate(spec.CELLS):
        m = cell_assign == ci
        if not m.any():
            continue
        z[m], p[m], y[m] = cells[cell]["z"][ids[m]], cells[cell]["p"][ids[m]], cells[cell]["labels"][ids[m]]
        pl[m], h[m] = PL[cell][ids[m]], H[cell][ids[m]]
    k, r = split(h)
    return {"z": z, "p": p, "pl": pl, "h": h, "k": k, "r": r}, y


def run_one(state, fold, force=False):
    out_dir = f"{OUT}/{state}"; os.makedirs(out_dir, exist_ok=True); path = f"{out_dir}/fold{fold}.npz"
    if os.path.exists(path) and not force:
        return path
    cons = json.load(open(f"{HROOT}/{state}/consistency.json")); assert cons["consistency_passed"], "extraction consistency failed"
    t0 = time.time()
    plan = stage0_folds.load_plan(); o = plan["outer"][fold]
    train, test = np.array(o["train_idx"]), np.array(o["test_idx"]); inner = np.array(o["inner_fit_mask"], bool); ca = np.array(o["cell_assignment_local"], np.int64)
    cells = regime_data.load_all_cells(state)
    H = {c: np.load(f"{HROOT}/{state}/h_{c}.npy", mmap_mode="r") for c in spec.CONDITIONS}
    PL = {c: np.load(f"{HROOT}/{state}/pl_{c}.npy", mmap_mode="r").astype(np.float64) for c in spec.CONDITIONS}
    split = projectors(state)
    full, y_f = gather(cells, H, PL, split, train, ca)
    fit, y_i = gather(cells, H, PL, split, train[inner], ca[inner]); val, y_v = gather(cells, H, PL, split, train[~inner], ca[~inner])
    kinds = {"zonly": "zonly", "c1e": "c1e", "c1c": "c1c", "c1d": "c1d", "kernel": "kernel", "zk": "zk", "prow": "prow"}
    fits = {a: select_and_fit(kinds[a], fit, val, full, y_i, y_v, y_f) for a in ALL}
    n = len(test); C = len(spec.CONDITIONS)
    cor = {a: np.zeros((C, n), np.uint8) for a in ALL}; nll = {a: np.zeros((C, n), np.float32) for a in ALL}
    probs = {a: np.zeros((C, n, 100), np.float32) for a in ("zonly", "prow")}
    for ci, c in enumerate(spec.CONDITIONS):
        d = cells[c]; y = d["labels"][test]; h = np.asarray(H[c][test]); k, r = split(h)
        inp = {"z": d["z"][test], "p": d["p"][test], "pl": PL[c][test], "h": h.astype(np.float64), "k": k, "r": r}
        for a in ALL:
            lg = predict_logits(fits[a], inp); lg = lg - lg.max(1, keepdims=True); lse = np.log(np.exp(lg).sum(1))
            cor[a][ci] = (lg.argmax(1) == y); nll[a][ci] = lse - lg[np.arange(n), y]
            if a in probs:
                probs[a][ci] = np.exp(lg - lse[:, None])
    rec = {"test_idx": test, "conditions": np.array(spec.CONDITIONS), "state": state, "fold": fold, "arms": np.array(ALL)}
    for a in ALL:
        rec[f"correct__{a}"], rec[f"nll__{a}"] = cor[a], nll[a]
        rec[f"meta__{a}__lambda"] = fits[a]["selected_lambda"]; rec[f"meta__{a}__converged"] = fits[a]["converged"]; rec[f"meta__{a}__retried"] = fits[a]["retried"]
        rec[f"meta__{a}__table"] = json.dumps(fits[a]["table"])
    for a, v in probs.items():
        rec[f"probs__{a}"] = v
    rec["wall_time_s"] = time.time() - t0
    np.savez(f"{out_dir}/fold{fold}.tmp.npz", **rec); os.replace(f"{out_dir}/fold{fold}.tmp.npz", path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--state", required=True); ap.add_argument("--fold", type=int, required=True, choices=range(5)); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); print("wrote", run_one(a.state, a.fold, a.force))
