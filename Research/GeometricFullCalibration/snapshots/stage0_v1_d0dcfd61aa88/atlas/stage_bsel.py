"""Stage BSEL (CPU): choose one metric per pooling family (and one output-space metric) on clean-SELECTION rows.

Same rule as the layer shortlist: maximum net gain (W-H)/n; ties: lower smoothed NLL, then lower cost (unit_l2 < raw_l2 <
mahalanobis). Reads validation rows only; writes an immutable ``metric_selection.json``.
"""
from __future__ import annotations

import json
import os
import stat

import numpy as np

from . import common, data, spec, stage_stats as ss, stats

COST_ORDER = {"unit_l2": 0, "raw_l2": 1, "mahalanobis": 2}


def rows_for(idx, dist, bank_y, prior, bp, val_y, roles):
    return {r: ss.row_for(idx, dist, bank_y, prior, bp, val_y, rv) for r, rv in roles.items()}


def run(seed: int, out_root: str = None):
    root, roles, val_y, bank_y, prior, sub, ty = ss.load_common(seed)
    root = out_root or root
    sl = json.load(open(f"{root}/shortlist.json"))["chosen"]
    bp = ss.base_logits(root, sub, "val").argmax(1)
    out = {"seed": seed, "rule": "clean-selection net gain; ties: geo_nll, cost order unit_l2<raw_l2<mahalanobis", "hidden": {}, "output": {},
           "provenance": {"target_labels_read": False}}
    for p in spec.POOLS:
        s = sl[p]["site"]
        tab = {}
        i0, d0, _ = ss.unit_arrays(root, s, p)
        tab["unit_l2"] = rows_for(i0, d0, bank_y, prior, bp, val_y, roles)
        zb = np.load(f"{root}/knn_B/{s}__{p}__atlas.npz")
        info = json.load(open(f"{root}/knn_B/{s}__{p}.done.json"))["info"]
        for m in ("raw_l2", "mahalanobis"):
            tab[m] = rows_for(zb[f"{m}_idx"][:5000], zb[f"{m}_dist"][:5000], bank_y, prior, bp, val_y, roles)
        best = min(tab, key=lambda m: (-round(tab[m]["selection"]["net_utility"], 12), tab[m]["selection"]["geo_nll"], COST_ORDER[m]))
        out["hidden"][p] = {"site": s, "metric": best, "selection_net_utility": {m: tab[m]["selection"]["net_utility"] for m in tab},
                            "selection_geo_nll": {m: tab[m]["selection"]["geo_nll"] for m in tab},
                            "fit_net_utility": {m: tab[m]["fit"]["net_utility"] for m in tab}, "bank_info": info}
    zv = np.load(f"{root}/u0/val.npz")
    tab = {m: rows_for(zv[f"{m}_idx"], zv[f"{m}_dist"], bank_y, prior, bp, val_y, roles) for m in spec.METRICS}
    best = min(tab, key=lambda m: (-round(tab[m]["selection"]["net_utility"], 12), tab[m]["selection"]["geo_nll"], COST_ORDER[m]))
    out["output"] = {"metric": best, "selection_net_utility": {m: tab[m]["selection"]["net_utility"] for m in tab},
                     "selection_geo_nll": {m: tab[m]["selection"]["geo_nll"] for m in tab},
                     "mahalanobis_info": json.load(open(f"{root}/u0/done.json"))["logit_mahalanobis"]}
    path = f"{root}/metric_selection.json"
    if os.path.exists(path):
        raise SystemExit("metric_selection.json is immutable and already exists")
    common.atomic_json(path, out); os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); a = ap.parse_args(); run(a.seed)
