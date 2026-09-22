"""Context loader for the fixed-gate study: base logits, roles, labels, and the three frozen candidates' neighbour artifacts.

Candidates (spec sec. 2): deep = layer3.22 / 2x2 / unit_l2 (recomputed under results/fixed_gate), gap4 = the atlas-selected GAP layer4
candidate (site and metric from the atlas manifests), out = the atlas-selected output-space candidate (metric from metric_selection.json).
All base logits come from the atlas u0 artifact (FP32, TF32 off).
"""
from __future__ import annotations

import json
from typing import Dict

import numpy as np

from . import data, spec, stats

SETS = ("val",) + spec.CONDITIONS
import os
FG = os.environ.get("FG_ROOT", "results/fixed_gate")
CANDS = ("deep", "gap4", "out")


def cand_defs(seed: int) -> Dict[str, Dict]:
    root = data.seed_dir(seed)
    sl = json.load(open(f"{root}/shortlist.json"))["chosen"]["gap"]["site"]
    ms = json.load(open(f"{root}/metric_selection.json"))
    return {"deep": {"kind": "hidden", "site": "layer3.22", "pool": "grid2", "metric": "unit_l2", "dir": f"{FG}/seed{seed}/knn_F"},
            "gap4": {"kind": "hidden", "site": sl, "pool": "gap", "metric": ms["hidden"]["gap"]["metric"], "dir": f"{root}/knn_F"},
            "out": {"kind": "output", "site": "logits", "pool": "logits", "metric": ms["output"]["metric"], "dir": f"{root}/u0"}}


def load_ctx(seed: int) -> Dict:
    root = data.seed_dir(seed)
    arrays = data.load_seed_arrays(seed)
    val_y = arrays["val"][1].astype(np.int64); bank_y = arrays["train"][1].astype(np.int64)
    ty = np.load(f"{data.SHARED}/test_labels.npy").astype(np.int64)
    roles = {k: np.asarray(v) for k, v in json.load(open(f"{root}/roles.json"))["roles"].items()}
    U = {s: np.load(f"{root}/u0/{s}.npz") for s in SETS}
    Z = {s: U[s]["logits"].astype(np.float64) for s in SETS}
    Y = {s: (val_y if s == "val" else ty) for s in SETS}
    prior = np.bincount(bank_y, minlength=spec.NUM_CLASSES) / len(bank_y)
    return {"seed": seed, "root": root, "roles": roles, "Z": Z, "Y": Y, "val_y": val_y, "bank_y": bank_y, "prior": prior, "U": U,
            "base_pred": {s: Z[s].argmax(1) for s in SETS}, "defs": cand_defs(seed), "u0": json.load(open(f"{root}/u0/done.json"))}


def load_candidate(ctx: Dict, name: str, sets=SETS) -> Dict[str, Dict[str, np.ndarray]]:
    d = ctx["defs"][name]; out = {}
    for s in sets:
        if d["kind"] == "hidden":
            z = np.load(f"{d['dir']}/{d['site']}__{d['pool']}__{d['metric']}__{s}.npz")
            idx, dist, qn = z[f"{d['metric']}_idx"], z[f"{d['metric']}_dist"], z["qnorm"].astype(np.float64)
        else:
            z = np.load(f"{d['dir']}/{s}.npz")
            idx, dist = z[f"{d['metric']}_idx"], z[f"{d['metric']}_dist"]
            zz = ctx["Z"][s]; qn = np.linalg.norm(zz - zz.mean(1, keepdims=True), axis=1)
        pg, cnt = stats.p_geo(idx, ctx["bank_y"], ctx["prior"])
        out[s] = {"pg": pg, "counts": cnt, "kth": dist[:, -1].astype(np.float64), "qnorm": qn, "j": pg.argmax(1), "idx": idx}
    return out
