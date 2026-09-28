"""N1a action-ambiguity audit (docs/n1a_action_ambiguity_spec.md): paired action outcomes, pre-action output features, and the
family + image-identity holdout. No internal representation is used anywhere in N1a.
"""
import numpy as np

from . import spec, stage0_data, stage0_folds

FAMILIES = tuple(spec.DEV_CORRUPTIONS)
BASES = (2, 4)


def other(base: int) -> int:
    assert base in BASES
    return 6 - base


def cells_of(family: str):
    return tuple(c for c in spec.CELLS if c.startswith(family + "_s"))


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def outcomes(zb, zo, y):
    """Paired action outcomes for one condition: keep, route, ensemble correctness and Delta_route / Delta_ens."""
    keep = (zb.argmax(1) == y).astype(np.int8); route = (zo.argmax(1) == y).astype(np.int8)
    ens = ((softmax(zb) + softmax(zo)).argmax(1) == y).astype(np.int8)
    return {"keep": keep, "route": route, "ens": ens, "d_route": (route - keep).astype(np.int8), "d_ens": (ens - keep).astype(np.int8)}


def output_features(z):
    """Pre-action output evidence F_Z computed from ONE model's logits only (206 columns; standardization happens in the fitter)."""
    z = np.asarray(z, dtype=np.float64); p = softmax(z); ps = -np.sort(-p, 1)
    ent = -(p * np.log(np.clip(p, 1e-300, None))).sum(1)
    onehot = np.eye(z.shape[1])[z.argmax(1)]
    extra = np.stack([ent, ps[:, 0] - ps[:, 1], ps[:, 0] - ps[:, 2], ps[:, 0] - ps[:, 4], ps[:, 0], z.max(1)], 1)
    return np.concatenate([z, extra, onehot], 1)


def load(base: int):
    """Per cell: z_b, z_o, labels (asserted equal across checkpoints)."""
    o = other(base); out = {}
    for c in spec.CELLS:
        db, do = stage0_data.load_cell(base, c), stage0_data.load_cell(o, c)
        assert np.array_equal(db["labels"], do["labels"])
        out[c] = {"zb": db["z"], "zo": do["z"], "y": db["labels"]}
    return out


def split(fold: int, family: str):
    """Image ids: training (fit/val via the Stage-0 inner mask) and evaluation; asserts image-identity disjointness."""
    o = stage0_folds.load_plan()["outer"][fold]
    train, test = np.array(o["train_idx"]), np.array(o["test_idx"]); inner = np.array(o["inner_fit_mask"], bool)
    assert not set(train.tolist()) & set(test.tolist()), "image identity leaks between training and evaluation"
    train_cells = tuple(c for f in FAMILIES if f != family for c in cells_of(f))
    return {"fit_ids": train[inner], "val_ids": train[~inner], "train_ids": train, "eval_ids": test,
            "train_cells": train_cells, "eval_cells": cells_of(family)}


def rows(data, ids, cells, arm: str):
    """Stack (cells x ids) rows. arm 'Z' uses F_Z(base) only; 'ZZo' (post-action reference) appends F_Z(other)."""
    X, D, meta = [], [], []
    for c in cells:
        d = data[c]; fz = output_features(d["zb"][ids])
        if arm == "ZZo":
            fz = np.concatenate([fz, output_features(d["zo"][ids])], 1)
        else:
            assert arm == "Z" and fz.shape[1] == 206
        oc = outcomes(d["zb"][ids], d["zo"][ids], d["y"][ids])
        X.append(fz); D.append(oc["d_route"]); meta.append(np.c_[ids, np.full(len(ids), spec.CELLS.index(c))])
    return np.concatenate(X), np.concatenate(D).astype(np.int64), np.concatenate(meta)
