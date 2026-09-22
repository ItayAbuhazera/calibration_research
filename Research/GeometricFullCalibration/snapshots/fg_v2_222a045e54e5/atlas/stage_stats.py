"""Stage S (CPU): atlas statistics, the clean-only shortlist, and target diagnostics.

Outcome-blind ordering is enforced in code:
  clean_stats + select_shortlist read ONLY validation rows (roles fit / selection / calibration) and their labels; the
  result is written once as a read-only ``shortlist.json`` with provenance. ``target_stats`` refuses to run unless that
  immutable artifact exists, and its outputs (per-corruption best layers) are labelled TARGET-LABELLED ORACLE ranking:
  they never feed candidate choice for Experiments B/C.
  python -m atlas.stage_stats --seed 2 --what clean|select|target
"""
from __future__ import annotations

import argparse
import json
import os
import stat
from typing import Dict, List

import numpy as np

from . import common, data, spec, stats

QBLOCK = 2000


def load_common(seed: int):
    root = data.seed_dir(seed)
    roles = json.load(open(f"{root}/roles.json"))["roles"]
    roles = {k: np.asarray(v) for k, v in roles.items()}
    arrays = data.load_seed_arrays(seed)
    val_y = arrays["val"][1].astype(np.int64)
    bank_y = arrays["train"][1].astype(np.int64)
    prior = np.bincount(bank_y, minlength=spec.NUM_CLASSES) / len(bank_y)
    sub = np.load(f"{data.SHARED}/subset_ids.npy")
    ty = np.load(f"{data.SHARED}/test_labels.npy")
    return root, roles, val_y, bank_y, prior, sub, ty


def base_logits(root: str, sub: np.ndarray, what: str = "all"):
    v = np.load(f"{root}/u0/val.npz")["logits"]
    if what == "val":
        return v
    conds = [np.load(f"{root}/u0/{c}.npz")["logits"][sub] for c in spec.CONDITIONS]
    return np.concatenate([v] + conds)


def unit_arrays(root: str, site: str, pool: str, folder: str = "knn_unit"):
    z = np.load(f"{root}/stage_a/../{folder}/{site}__{pool}.npz") if False else np.load(f"{root}/{folder}/{site}__{pool}.npz")
    return z["idx"], z["dist"], z["qnorm"]


def row_for(idx, dist, bank_y, prior, base_pred, y, rows):
    pg, counts = stats.p_geo(idx[rows], bank_y, prior)
    return stats.candidate_row(pg, counts, dist[rows, -1], base_pred[rows], y[rows])


def clean_stats(seed: int, folder: str = "knn_unit", suffix: str = "unit_l2"):
    root, roles, val_y, bank_y, prior, sub, ty = load_common(seed)
    bp = base_logits(root, sub, "val").argmax(1)
    out = {}
    for s in spec.SITE_NAMES:
        for p in spec.POOLS:
            idx, dist, qn = unit_arrays(root, s, p, folder)
            out.setdefault(s, {})[p] = {r: row_for(idx, dist, bank_y, prior, bp, val_y, rv) for r, rv in roles.items()}
    common.atomic_json(f"{root}/stats_clean_{suffix}.json", out)
    return out


def rank_key(row, cost, order):
    return (-row["net_utility"], row["geo_nll"], cost, order)


def select_shortlist(seed: int):
    """One (site, pool) per pooling family, by clean-selection net gain (W-H)/n; ties: lower smoothed NLL, lower measured
    query cost, fixed depth order. The best is kept even when every gain is negative (flagged)."""
    root, roles, val_y, bank_y, prior, sub, ty = load_common(seed)
    st = json.load(open(f"{root}/stats_clean_unit_l2.json"))
    cost = {}
    for f in sorted(os.listdir(f"{root}/stage_a")):
        if f.endswith(".done.json"):
            m = json.load(open(f"{root}/stage_a/{f}"))
            for k, v in m["files"].items():
                cost[k] = v["knn_ms_per_query_batch_amortized"]
    order = {s: i for i, s in enumerate(spec.SITE_NAMES)}
    ranking, chosen = {}, {}
    bp = base_logits(root, sub, "val").argmax(1)
    sel_rows = roles["selection"]
    for p in spec.POOLS:
        tab = sorted(spec.SITE_NAMES, key=lambda s: rank_key(st[s][p]["selection"], cost[f"{s}__{p}"], order[s]))
        ranking[p] = [{"site": s, "net_utility_selection": st[s][p]["selection"]["net_utility"], "geo_nll_selection": st[s][p]["selection"]["geo_nll"],
                       "net_utility_fit": st[s][p]["fit"]["net_utility"], "query_ms": cost[f"{s}__{p}"]} for s in tab]
        chosen[p] = {"site": tab[0], "net_utility_selection": st[tab[0]][p]["selection"]["net_utility"],
                     "all_gains_negative": bool(all(st[s][p]["selection"]["net_utility"] < 0 for s in spec.SITE_NAMES)),
                     "dim": spec.pool_dim(tab[0], p)}
    # descriptive stability: paired resampling of the selection rows (does not alter the rule)
    rng = np.random.default_rng(spec.ROLE_SPLIT_SEED)
    D = {}
    for p in spec.POOLS:
        for s in spec.SITE_NAMES:
            idx, dist, _ = unit_arrays(root, s, p)
            pg, _c = stats.p_geo(idx[sel_rows], bank_y, prior)
            D[(s, p)] = (pg.argmax(1) == val_y[sel_rows]).astype(float) - (bp[sel_rows] == val_y[sel_rows]).astype(float)
    stab = {}
    for p in spec.POOLS:
        mat = np.stack([D[(s, p)] for s in spec.SITE_NAMES])           # [sites, 1250]
        wins = np.zeros(len(spec.SITE_NAMES))
        for _ in range(200):
            b = rng.integers(0, mat.shape[1], mat.shape[1])
            wins[np.argmax(mat[:, b].mean(1))] += 1                    # argmax: first (shallowest) wins ties
        stab[p] = {s: float(w / 200) for s, w in zip(spec.SITE_NAMES, wins) if w > 0}
    art = {"seed": seed, "rule": "clean-selection net gain (W-H)/n; ties: geo_nll, query cost, depth order",
           "n_configurations_searched": len(spec.SITE_NAMES) * len(spec.POOLS), "chosen": chosen, "ranking": ranking,
           "selection_bootstrap_win_frequency_descriptive": stab, "selection_rows": int(len(sel_rows)),
           "provenance": {"target_labels_read": False, "target_predictions_read": False, "files_read": ["roles.json", "u0/val.npz(logits)",
                          "stats_clean_unit_l2.json", "knn_unit/*.npz (validation rows only are used)", "stage_a markers (timing)"],
                          "note": "selected validation gain is optimistic (search over 102 configurations); not an effect estimate"}}
    path = f"{root}/shortlist.json"
    if os.path.exists(path):
        raise SystemExit("shortlist.json already exists and is immutable")
    common.atomic_json(path, art)
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return art


def target_stats(seed: int, folder: str = "knn_unit", suffix: str = "unit_l2"):
    root, roles, val_y, bank_y, prior, sub, ty = load_common(seed)
    sl = f"{root}/shortlist.json"
    if not os.path.exists(sl) or os.access(sl, os.W_OK):
        raise SystemExit("clean shortlist must exist and be read-only before any target statistic is computed")
    bp = base_logits(root, sub).argmax(1)
    yy = np.concatenate([val_y] + [ty[sub] for _ in spec.CONDITIONS])
    out = {}
    for s in spec.SITE_NAMES:
        for p in spec.POOLS:
            idx, dist, qn = unit_arrays(root, s, p, folder)
            for ci, c in enumerate(spec.CONDITIONS):
                rows = np.arange(5000 + ci * QBLOCK, 5000 + (ci + 1) * QBLOCK)
                out.setdefault(s, {}).setdefault(p, {})[c] = row_for(idx, dist, bank_y, prior, bp, yy, rows)
    common.atomic_json(f"{root}/stats_target_{suffix}.json", out)
    oracle = {p: {c: max(spec.SITE_NAMES, key=lambda s: out[s][p][c]["net_utility"]) for c in spec.CONDITIONS} for p in spec.POOLS}
    common.atomic_json(f"{root}/target_oracle_ranking_{suffix}.json", {"label": "TARGET-LABELLED ORACLE ranking; diagnostic only; never used for selection",
                                                                       "best_site_by_condition": oracle})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--what", choices=["clean", "select", "target"], required=True)
    a = ap.parse_args()
    {"clean": clean_stats, "select": select_shortlist, "target": target_stats}[a.what](a.seed)
