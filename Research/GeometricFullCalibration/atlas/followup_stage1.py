"""Stage 1 of the regime-map follow-up (amendment 1 sec. 7): anchored Z-only, (Z, P) and C1b on the EXISTING z/p arrays,
T-8k x 1 target fits (ORACLE DIAGNOSTIC), one (state, fold) per call. CPU only.

    python -m atlas.followup_stage1 --state a --fold 0
"""
import argparse
import json
import os
import time

import numpy as np

from . import regime_data, spec, stage0_folds
from .stage0c_run import build_rows
from .followup_anchored import INF, predict_logits, select_and_fit

ARMS = ("zonly", "zp", "c1b")
OUT = "results/regime_map_followup/stage1"


def run_one(state: str, fold: int, force: bool = False) -> str:
    out_dir = f"{OUT}/{state}"; os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/fold{fold}.npz"
    if os.path.exists(path) and not force:
        return path
    t0 = time.time()
    plan = stage0_folds.load_plan(); o = plan["outer"][fold]
    train, test = np.array(o["train_idx"]), np.array(o["test_idx"])
    inner = np.array(o["inner_fit_mask"], bool); ca = np.array(o["cell_assignment_local"], np.int64)
    cells = regime_data.load_all_cells(state)
    Xz_f, Xp_f, y_f, _ = build_rows(cells, train, "one", ca)
    Xz_i, Xp_i, y_i, _ = build_rows(cells, train[inner], "one", ca[inner])
    Xz_v, Xp_v, y_v, _ = build_rows(cells, train[~inner], "one", ca[~inner])
    full, fit, val = {"z": Xz_f, "p": Xp_f}, {"z": Xz_i, "p": Xp_i}, {"z": Xz_v, "p": Xp_v}
    rec = {"test_idx": test, "conditions": np.array(spec.CONDITIONS), "state": state, "fold": fold, "arms": np.array(ARMS)}
    fits = {a: select_and_fit(a, fit, val, full, y_i, y_v, y_f) for a in ARMS}
    n = len(test)
    cor = {a: np.zeros((len(spec.CONDITIONS), n), np.uint8) for a in ARMS + ("base",)}
    nll = {a: np.zeros((len(spec.CONDITIONS), n), np.float32) for a in ARMS + ("base",)}
    for ci, c in enumerate(spec.CONDITIONS):
        d = cells[c]; y = d["labels"][test]; inp = {"z": d["z"][test], "p": d["p"][test]}
        for a in ARMS + ("base",):
            lg = inp["z"] if a == "base" else predict_logits(fits[a], inp)
            lg = lg - lg.max(1, keepdims=True); lse = np.log(np.exp(lg).sum(1))
            cor[a][ci] = (lg.argmax(1) == y); nll[a][ci] = lse - lg[np.arange(n), y]
    for a in ARMS + ("base",):
        rec[f"correct__{a}"], rec[f"nll__{a}"] = cor[a], nll[a]
    for a in ARMS:
        rec[f"meta__{a}__lambda"] = fits[a]["selected_lambda"]; rec[f"meta__{a}__converged"] = fits[a]["converged"]; rec[f"meta__{a}__retried"] = fits[a]["retried"]
        rec[f"meta__{a}__table"] = json.dumps(fits[a]["table"])
    assert (rec["correct__zonly"] == rec["correct__base"]).all() or fits["zonly"]["selected_lambda"] != INF   # f = 0 must reproduce the base head exactly
    rec["wall_time_s"] = time.time() - t0
    np.savez(f"{out_dir}/fold{fold}.tmp.npz", **rec); os.replace(f"{out_dir}/fold{fold}.tmp.npz", path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--state", required=True); ap.add_argument("--fold", type=int, required=True, choices=range(5)); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); print("wrote", run_one(a.state, a.fold, a.force))
