"""Stage A (GPU): exact k=50 neighbours for every (site, pooling) of one site group, unit-L2 metric.

One forward pass over the reference bank fills the group's three pooled, unit-normalized banks on the GPU;
one pass over the 31 000 atlas queries (5 000 validation rows + the 2 000-image test subset under clean and the 12
development cells) computes exact neighbours immediately. Only (indices, distances, pre-L2 norms) are written; no
activation tensors or pooled features are cached. Resumable: a group with a valid completion marker is skipped.
  python -m atlas.stage_a --seed 2 --group 3 [--smoke 300]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from . import common, data, features, knn, pooling, spec


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run(seed: int, gi: int, smoke: int = 0, out_root: str = None, bs: int = 250):
    dev = common.setup_torch()
    common.check_protocol(seed)
    sites = spec.plan_groups()[gi]
    root = out_root or data.seed_dir(seed)
    marker = f"{root}/stage_a/group{gi:02d}.done.json"
    if common.is_done(marker):
        print("skip", marker); return
    arrays = data.load_seed_arrays(seed)
    train_x, train_y = arrays["train"]
    Q = data.atlas_queries(seed, arrays)
    if smoke:
        train_x, train_y = train_x[:min(len(train_y), 8000)], train_y[:8000]
        idx = np.concatenate([np.arange(smoke), 5000 + np.arange(smoke)])
        Q = Q[idx]
    nb, nq = len(train_y), Q.shape[0]
    model, ckpt = common.load_model(seed, dev)
    tap = features.SiteTap(model, sites)
    units = [(s, p) for s in sites for p in spec.POOLS]
    banks = {(s, p): torch.empty(nb, spec.pool_dim(s, p), device=dev) for s, p in units}
    bnorm = {u: np.empty(nb, np.float32) for u in units}
    t0 = time.time()
    for st, x in features.batches(train_x, bs, dev):
        _, maps = features.forward_maps(model, tap, x)
        e = st + x.shape[0]
        for s, p in units:
            v, n = pooling.unit(pooling.pool(maps[s], p))
            banks[(s, p)][st:e] = v; bnorm[(s, p)][st:e] = n.cpu().numpy()
    t_bank = time.time() - t0
    idxs = {u: np.empty((nq, spec.K_NN), np.int32) for u in units}
    dist = {u: np.empty((nq, spec.K_NN), np.float32) for u in units}
    qn = {u: np.empty(nq, np.float32) for u in units}
    lvl = {s: np.empty((nq, len(spec.valid_levels(spec.SITE_SIZE[s]))), np.float32) for s in sites}
    tsec = {u: 0.0 for u in units}
    for st, x in features.batches(Q, bs, dev):
        _, maps = features.forward_maps(model, tap, x)
        e = st + x.shape[0]
        for s in sites:
            lvl[s][st:e] = pooling.spp_level_sq_norms(maps[s]).cpu().numpy()
        for s, p in units:
            a = pooling.pool(maps[s], p)
            v, n = pooling.unit(a)
            _sync(); t1 = time.perf_counter()
            d, i = knn.knn_unit(v, banks[(s, p)])
            _sync(); tsec[(s, p)] += time.perf_counter() - t1
            idxs[(s, p)][st:e] = i.cpu().numpy().astype(np.int32); dist[(s, p)][st:e] = d.cpu().numpy(); qn[(s, p)][st:e] = n.cpu().numpy()
    files = {}
    for s, p in units:
        f = f"{root}/knn_unit/{s}__{p}.npz"
        common.atomic_npz(f, idx=idxs[(s, p)], dist=dist[(s, p)], qnorm=qn[(s, p)])
        files[f"{s}__{p}"] = {"sha256": common.file_sha(f), "dim": spec.pool_dim(s, p), "knn_seconds_total": tsec[(s, p)],
                              "knn_ms_per_query_batch_amortized": 1000 * tsec[(s, p)] / nq,
                              "bank_norm_mean": float(bnorm[(s, p)].mean()), "bank_norm_min": float(bnorm[(s, p)].min()),
                              "bank_zero_norm": int((bnorm[(s, p)] < 1e-6).sum()), "query_zero_norm": int((qn[(s, p)] < 1e-6).sum())}
    for s in sites:
        common.atomic_npz(f"{root}/spp_level_sq_norms/{s}.npz", lvl=lvl[s])
    common.mark_done(marker, {"seed": seed, "group": gi, "sites": sites, "n_bank": nb, "n_query": nq, "checkpoint": ckpt,
                              "bank_pass_seconds": t_bank, "total_seconds": time.time() - t0, "files": files, "smoke": smoke,
                              "peak_gpu_gb": (torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0) / 1e9})
    print(f"group {gi} {sites} done in {time.time()-t0:.0f}s peak {(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)/1e9:.1f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--out_root", default=None)
    a = ap.parse_args()
    run(a.seed, a.group, a.smoke, a.out_root)
