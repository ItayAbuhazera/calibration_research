"""N1a selector fit, one (base, held-out family, fold, arm) per call (docs/n1a_action_ambiguity_spec.md sec. 2-3). CPU only.

    python -m atlas.n1a_fit --base 2 --family fog --fold 0 --arm Z
"""
import argparse
import json
import os
import time

import numpy as np

from . import n1a, stage0_fit

OUT = "results/n1a/fits"


def run_one(base, family, fold, arm, force=False):
    out_dir = f"{OUT}/base{base}/{family}/{arm}"; os.makedirs(out_dir, exist_ok=True); path = f"{out_dir}/fold{fold}.npz"
    if os.path.exists(path) and not force:
        return path
    t0 = time.time()
    data = n1a.load(base); sp = n1a.split(fold, family)
    Xi, Di, _ = n1a.rows(data, sp["fit_ids"], sp["train_cells"], arm)
    Xv, Dv, _ = n1a.rows(data, sp["val_ids"], sp["train_cells"], arm)
    Xf, Df, _ = n1a.rows(data, sp["train_ids"], sp["train_cells"], arm)
    Xe, De, me = n1a.rows(data, sp["eval_ids"], sp["eval_cells"], arm)
    assert not set(me[:, 0].tolist()) & set(sp["train_ids"].tolist())
    fo = stage0_fit.fit_arm(Xf, Df + 1, Xi, Di + 1, Xv, Dv + 1, 3)
    P = stage0_fit.predict_probs(Xe, fo["scaler_mean"], fo["scaler_std"], fo["W"], fo["b"])
    zb_tr = np.concatenate([data[c]["zb"][sp["train_ids"]] for c in sp["train_cells"]])
    mu, sd = stage0_fit.standardize_fit(zb_tr)
    zb_ev = np.concatenate([data[c]["zb"][sp["eval_ids"]] for c in sp["eval_cells"]])
    rec = {"base": base, "family": family, "fold": fold, "arm": arm, "image_id": me[:, 0], "cell_index": me[:, 1], "d_route": De.astype(np.int8),
           "probs": P.astype(np.float32), "e": (P[:, 2] - P[:, 0]).astype(np.float64), "zstd_eval": stage0_fit.standardize_apply(zb_ev, mu, sd).astype(np.float32),
           "train_mean_delta": float(Df.mean()), "selected_lambda": fo["selected_lambda"], "converged": fo["converged"], "retried": fo["retried"],
           "table": json.dumps(fo["selection"]["table"]), "n_train_rows": len(Df), "n_eval_rows": len(De), "wall_time_s": time.time() - t0}
    np.savez(f"{out_dir}/fold{fold}.tmp.npz", **rec); os.replace(f"{out_dir}/fold{fold}.tmp.npz", path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, required=True, choices=n1a.BASES); ap.add_argument("--family", required=True, choices=n1a.FAMILIES)
    ap.add_argument("--fold", type=int, required=True, choices=range(5)); ap.add_argument("--arm", required=True, choices=("Z", "ZZo")); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); print("wrote", run_one(a.base, a.family, a.fold, a.arm, a.force))
