"""N1a-DP — Output Decoder Panel action-ambiguity audit (docs/n1a_dp_spec.md). SHIFT-TRANSFER HPO. Output evidence only.

Preserves the N1a action problem (atlas/n1a.py): Delta_route = 1[argmax Z_o = y] - 1[argmax Z_b = y], bases 2 and 4 symmetric, 3-way target
y = Delta + 1, pre-action evidence F_Z(b) (206 deterministic functions of the base logits; Z_o never enters), outer holdout = one entire
corruption family (3 severities) x the Stage-0 outer image fold (image-identity disjoint). Model selection: inner leave-one-TRAINING-family-
out splits (fit = the other two training families x inner-fit images; val = the held-out training family x inner-val images), objective =
macro (over the three inner environments) negative realized policy utility of `route iff P(+1) - P(-1) > 0`. The outer held-out family never
enters selection, early stopping, preprocessing or refit. No internal representation is used.

    python -m atlas.n1adp fit --family lgbm --base 2 --heldout fog --fold 0
"""
import argparse
import json
import os
import time

import numpy as np

from . import common, decoder_panel as dp, n1a, stage0_fit

EXPERIMENT = "N1a-DP"
FITS = "results/n1adp/fits"


def neg_policy_utility(pred, y, _rows=None):
    """Loss to minimize = -(mean over rows of Delta * 1[route]), route iff P(+1) - P(-1) > 0 (pred = 3-class logits; y = Delta + 1)."""
    p = np.exp(pred - pred.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return -float(np.mean((np.asarray(y) - 1) * ((p[:, 2] - p[:, 0]) > 0)))


def family_of(cell):
    return cell.rsplit("_s", 1)[0]


def unit_rows(data, base, heldout, fold):
    """Training rows (3 training families x 3 severities x outer-train images) with environments and inner splits; evaluation rows."""
    sp = n1a.split(fold, heldout)
    assert heldout not in {family_of(c) for c in sp["train_cells"]}
    X, D, env, img, cell = [], [], [], [], []
    for c in sp["train_cells"]:
        Xc, Dc, mc = n1a.rows(data, sp["train_ids"], (c,), "Z")
        X.append(Xc); D.append(Dc); env += [family_of(c)] * len(Dc); img.append(mc[:, 0]); cell.append(mc[:, 1])
    X, D, env, img, cell = np.concatenate(X), np.concatenate(D), np.array(env), np.concatenate(img), np.concatenate(cell)
    fit_set = set(sp["fit_ids"].tolist()); in_fit = np.fromiter((i in fit_set for i in img), bool, len(img))
    splits = []
    for g in sorted(set(env)):
        splits.append((np.flatnonzero((env != g) & in_fit), np.flatnonzero((env == g) & ~in_fit)))
    Xe, De, me = n1a.rows(data, sp["eval_ids"], sp["eval_cells"], "Z")
    assert not set(me[:, 0].tolist()) & set(img.tolist()), "image identity leaks between selector training and evaluation"
    assert {family_of(spec_c) for spec_c in np.array(n1a.spec.CELLS)[me[:, 1]]} == {heldout}
    return {"X": X, "y": D + 1, "env": env, "img": img, "cell": cell, "splits": splits, "Xe": Xe, "De": De, "me": me, "sp": sp}


def context(base, heldout, fold, U):
    return dp.HPOContext(EXPERIMENT, {"base": base, "heldout": heldout, "fold": fold, "arm": "Z"}, "shift_transfer", U["splits"],
                         neg_policy_utility, forbidden_env=heldout, env_train=U["env"])


def out_paths(family, base, heldout, fold):
    d = f"{FITS}/{family}/base{base}/{heldout}"
    return {"dir": d, "npz": f"{d}/fold{fold}.npz", "study": f"{d}/fold{fold}.study.json", "done": f"{d}/fold{fold}.done.json"}


def is_complete(family, base, heldout, fold):
    p = out_paths(family, base, heldout, fold)
    if not os.path.exists(p["done"]):
        return False
    try:
        d = json.load(open(p["done"]))
        return all(os.path.exists(p[k]) and common.file_sha(p[k]) == d[f"{k}_sha256"] for k in ("npz", "study"))
    except Exception:
        return False


def run(family, base, heldout, fold, device="cpu", force=False):
    if not force and is_complete(family, base, heldout, fold):
        print("complete; skipping", family, base, heldout, fold); return
    t0 = time.time()
    data = n1a.load(base); U = unit_rows(data, base, heldout, fold)
    model, rec = dp.fit_decoder(family, U["X"], U["y"], "classification", context(base, heldout, fold, U), K=3, device=device)
    lg = model.predict(U["Xe"]); P = np.exp(lg - lg.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    sp = U["sp"]
    zb_tr = np.concatenate([data[c]["zb"][sp["train_ids"]] for c in sp["train_cells"]]); mu, sd = stage0_fit.standardize_fit(zb_tr)
    zb_ev = np.concatenate([data[c]["zb"][sp["eval_ids"]] for c in sp["eval_cells"]])
    out = {"base": base, "heldout": heldout, "fold": fold, "family": family, "image_id": U["me"][:, 0], "cell_index": U["me"][:, 1],
           "d_route": U["De"].astype(np.int8), "probs": P.astype(np.float32), "e": (P[:, 2] - P[:, 0]).astype(np.float64),
           "zstd_eval": stage0_fit.standardize_apply(zb_ev, mu, sd).astype(np.float32), "train_mean_delta": float((U["y"] - 1).mean()),
           "n_train_rows": len(U["y"]), "n_eval_rows": len(U["De"]), "wall_time_s": time.time() - t0}
    rec["host"] = os.uname().nodename; rec["cpus"] = dp.n_threads()
    p = out_paths(family, base, heldout, fold); os.makedirs(p["dir"], exist_ok=True)
    np.savez(p["npz"] + ".tmp.npz", **out); os.replace(p["npz"] + ".tmp.npz", p["npz"])
    dp.save_record(p["study"], rec)
    common.atomic_json(p["done"], {"npz_sha256": common.file_sha(p["npz"]), "study_sha256": common.file_sha(p["study"]),
                                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "wall_time_s": out["wall_time_s"]})
    print(f"{family} base{base} {heldout} fold{fold}: selected {rec['selected']} ({out['wall_time_s']:.0f}s)", flush=True)


def units():
    """Deterministic array mapping: (base, held-out family, fold) in lexicographic order (40 units)."""
    return [(b, f, k) for b in n1a.BASES for f in n1a.FAMILIES for k in range(5)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=("fit", "fit_index"))
    ap.add_argument("--family", choices=dp.FAMILIES, required=True); ap.add_argument("--base", type=int, choices=n1a.BASES)
    ap.add_argument("--heldout", choices=n1a.FAMILIES); ap.add_argument("--fold", type=int, choices=range(5)); ap.add_argument("--index", type=int)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.stage == "fit":
        run(a.family, a.base, a.heldout, a.fold, a.device, a.force)
    else:
        b, f, k = units()[a.index]; run(a.family, b, f, k, a.device, a.force)
