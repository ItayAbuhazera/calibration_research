"""G1 fits (docs/g1_conditional_access_spec.md sec. 3-4): one (checkpoint, regime, arm, fold) per call, anchored target-fitted readouts
(ORACLE DIAGNOSTIC: labels of the 12 exposed development cells), CPU only.

    python -m atlas.g1_fit --seed 2 --regime T-8k1 --arm D --fold 0
    python -m atlas.g1_fit --seed 2 --regime T-8k1 --arm Dshuf --fold 0      # fold 0 only

Reuses atlas.followup_anchored.fit_anchored / nll_offset / pick unchanged; adds the frozen grid-edge rule.
"""
import argparse
import json
import os
import time

import numpy as np

from . import spec, stage0_data, stage0_folds, stage0_fit
from .followup_anchored import INF, fit_anchored, nll_offset, pick

HROOT = "results/g1/hL"
OUT = "results/g1/fits"
GRID = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
EXT_STRONG = (1e0, 1e1)      # stronger ridge (larger lambda) beyond the grid's strong edge
EXT_WEAK = (1e-6, 1e-7)      # weaker ridge beyond the grid's weak edge
PERM_SEED = 20260923         # = stage0_shuffle.PERM_SEED; partition offsets +10/+20/+30/+40 as there
ARMS = {"A": ("z",), "B": ("z", "p"), "C": ("z", "h"), "D": ("z", "h", "p"), "E": ("z", "h", "p42"), "F": ("z", "h", "zo"),
        "G": ("z", "k"), "Dshuf": ("z", "h", "p")}
REGIME_ARMS = {"T-8k1": ("A", "B", "C", "D", "E", "F", "G", "Dshuf"), "T-2.5k1": ("A", "B", "C", "D")}


def kernel_projector(W: np.ndarray) -> np.ndarray:
    """P_ker = I - W^+ W for a head W of shape [C, d]."""
    return np.eye(W.shape[1]) - np.linalg.pinv(W) @ W


class Feat:
    """Concatenate the arm's raw inputs and standardize per coordinate with statistics of the rows it is fitted on."""

    def __init__(self, keys):
        self.keys = keys

    def raw(self, inp):
        return np.concatenate([np.asarray(inp[k], dtype=np.float64) for k in self.keys], 1)

    def fit(self, inp):
        self.m, self.s = stage0_fit.standardize_fit(self.raw(inp)); return self

    def __call__(self, inp):
        return stage0_fit.standardize_apply(self.raw(inp), self.m, self.s)


def lambda_path(Xf, off_f, y_f, Xv, off_v, y_v):
    """Frozen path: f = 0 plus GRID, then the grid-edge extensions. Returns (table, selected, edge_status)."""
    def evaluate(lam):
        fo = fit_anchored(Xf, off_f, y_f, lam)
        return {"lambda": lam, "val_nll": nll_offset(off_v + Xv @ fo["W"] + fo["b"], y_v), "converged": fo["converged"], "retried": fo["retried"]}
    return path_with_edges(evaluate, nll_offset(off_v, y_v))


def path_with_edges(evaluate, f0_val_nll):
    """evaluate(lam) -> table row. Applies the frozen grid-edge rule (spec sec. 3)."""
    table = [{"lambda": INF, "val_nll": f0_val_nll, "converged": True, "retried": False}]

    def add(lam):
        table.append(evaluate(lam))

    for lam in GRID:
        add(lam)
    sel = pick(table)["lambda"]
    for ext, edge in ((EXT_STRONG, GRID[0]), (EXT_WEAK, GRID[-1])):
        if sel != edge:
            continue
        for lam in ext:
            add(lam); new = pick(table)["lambda"]
            if new != lam:
                break
            sel = new
        sel = pick(table)["lambda"]
    outer = {EXT_STRONG[-1], EXT_WEAK[-1]}
    edge_status = "unresolved_edge" if sel in outer else ("extended_resolved" if len(table) > 1 + len(GRID) else "interior_or_f0")
    return table, sel, edge_status


def load_inputs(seed: int):
    cells = stage0_data.load_all_cells(seed)                                   # z, p (= P_3.22), labels (+ all Stage-0 assertions)
    other = stage0_data.load_all_cells(seed, "xckpt")                          # p = Z_other (asserts equal labels)
    head = np.load(f"{HROOT}/seed{seed}/head.npz")
    cons = json.load(open(f"{HROOT}/seed{seed}/consistency.json"))
    assert cons["consistency_passed"], "G1 extraction consistency gate failed; no fit may run"
    Pk = kernel_projector(head["W_eff"])
    out = {}
    for c in spec.CONDITIONS:
        pil = np.load(f"{stage0_data.LAYER_PILOT_ROOT}/checkpoint_seed{seed}/{c}/per_sample.npz")["raw__probe_logits"]
        assert np.array_equal(other[c]["labels"], cells[c]["labels"])
        out[c] = {"z": cells[c]["z"], "p": cells[c]["p"], "p42": pil[:, 11, :].astype(np.float64), "zo": other[c]["p"],
                  "labels": cells[c]["labels"], "h": np.load(f"{HROOT}/seed{seed}/h_{c}.npy", mmap_mode="r")}
    return out, Pk


def gather(data, ids, cell_assign, keys, Pk, p_ids=None):
    """T-x1 rows: each image at its preassigned corrupted cell. p_ids (same order) replaces the image identity used for 'p' only."""
    p_ids = ids if p_ids is None else p_ids
    n = len(ids); inp = {k: None for k in set(keys) | {"z"}}; y = np.empty(n, np.int64)
    dims = {"z": 100, "p": 100, "p42": 100, "zo": 100, "h": 2048, "k": 2048}
    for k in inp:
        inp[k] = np.empty((n, dims[k]))
    for ci, cell in enumerate(spec.CELLS):
        m = cell_assign == ci
        if not m.any():
            continue
        d = data[cell]; y[m] = d["labels"][ids[m]]
        for k in inp:
            if k == "k":
                continue
            src = p_ids[m] if k == "p" else ids[m]
            inp[k][m] = np.asarray(d[k][src], dtype=np.float64)
    if "k" in inp:
        h = np.empty((n, 2048))
        for ci, cell in enumerate(spec.CELLS):
            m = cell_assign == ci
            if m.any():
                h[m] = np.asarray(data[cell]["h"][ids[m]], dtype=np.float64)
        inp["k"] = h @ Pk.T
    return inp, y


def run_one(seed, regime, arm, fold, force=False):
    assert arm in REGIME_ARMS[regime], (regime, arm)
    assert arm != "Dshuf" or fold == 0, "D-shuf is fold 0 only"
    out_dir = f"{OUT}/seed{seed}/{regime}/{arm}"; os.makedirs(out_dir, exist_ok=True); path = f"{out_dir}/fold{fold}.npz"
    if os.path.exists(path) and not force:
        return path
    t0 = time.time()
    plan = stage0_folds.load_plan(); o = plan["outer"][fold]
    train, test = np.array(o["train_idx"]), np.array(o["test_idx"]); inner = np.array(o["inner_fit_mask"], bool); ca = np.array(o["cell_assignment_local"], np.int64)
    pos = np.array(o["nested_2500_local_positions"]) if regime == "T-2.5k1" else np.arange(len(train))
    train, inner, ca = train[pos], inner[pos], ca[pos]
    data, Pk = load_inputs(seed)
    keys = ARMS[arm]
    fit_ids, val_ids = train[inner], train[~inner]
    if arm == "Dshuf":
        rng = lambda off: np.random.default_rng(PERM_SEED + off)
        p_fit, p_val, p_full, p_test = (ids[rng(off).permutation(len(ids))] for ids, off in ((fit_ids, 10), (val_ids, 20), (train, 30), (test, 40)))
    else:
        p_fit, p_val, p_full, p_test = fit_ids, val_ids, train, test
    fin, y_i = gather(data, fit_ids, ca[inner], keys, Pk, p_fit)
    vin, y_v = gather(data, val_ids, ca[~inner], keys, Pk, p_val)
    ful, y_f = gather(data, train, ca, keys, Pk, p_full)
    feat = Feat(keys).fit(fin)
    table, sel, edge = lambda_path(feat(fin), fin["z"], y_i, feat(vin), vin["z"], y_v)
    if sel == INF:
        ffull, W, b, conv, retr, g = None, None, None, True, False, 0.0
    else:
        ffull = Feat(keys).fit(ful); fo = fit_anchored(ffull(ful), ful["z"], y_f, sel)
        W, b, conv, retr, g = fo["W"], fo["b"], fo["converged"], fo["retried"], fo["grad_inf"]
    n, C = len(test), len(spec.CONDITIONS)
    cor = np.zeros((C, n), np.uint8); nll = np.zeros((C, n), np.float32)
    for ci, c in enumerate(spec.CONDITIONS):
        d = data[c]; y = d["labels"][test]
        inp = {k: (np.asarray(d[k][p_test if k == "p" else test], dtype=np.float64) if k != "k" else None) for k in set(keys) | {"z"}}
        if "k" in keys:
            inp["k"] = np.asarray(d["h"][test], dtype=np.float64) @ Pk.T
        lg = inp["z"] if ffull is None else inp["z"] + ffull(inp) @ W + b
        lg = lg - lg.max(1, keepdims=True); lse = np.log(np.exp(lg).sum(1))
        cor[ci] = lg.argmax(1) == y; nll[ci] = lse - lg[np.arange(n), y]
    rec = {"test_idx": test, "conditions": np.array(spec.CONDITIONS), "seed": seed, "regime": regime, "arm": arm, "fold": fold,
           "keys": np.array(keys), "correct": cor, "nll": nll, "selected_lambda": sel, "edge_status": edge, "converged": conv,
           "retried": retr, "final_grad_inf": g, "table": json.dumps(table), "n_train_images": len(train), "wall_time_s": time.time() - t0}
    np.savez(f"{out_dir}/fold{fold}.tmp.npz", **rec); os.replace(f"{out_dir}/fold{fold}.tmp.npz", path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(2, 4)); ap.add_argument("--regime", required=True, choices=tuple(REGIME_ARMS))
    ap.add_argument("--arm", required=True, choices=tuple(ARMS)); ap.add_argument("--fold", type=int, required=True, choices=range(5)); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); print("wrote", run_one(a.seed, a.regime, a.arm, a.fold, a.force))
