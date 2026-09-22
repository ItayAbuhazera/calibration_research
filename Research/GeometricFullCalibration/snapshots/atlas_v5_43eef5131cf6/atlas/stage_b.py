"""Stage B (GPU): distance-geometry comparison on the clean-selected candidates.

For one (site, pooling) candidate, on the atlas queries, with the SAME full labelled bank, k=50, smoothing and query IDs:
  raw_l2       norm-preserving Euclidean on the pre-L2 pooled vector a (no per-coordinate standardization);
  mahalanobis  regularized low-rank + isotropic-residual Mahalanobis on a (source-only covariance; see atlas/knn.py).
The unit_l2 neighbours already exist from Stage A. Raw and Mahalanobis distances are translation invariant, so the bank is
centered IN PLACE by the source mean (one bank copy in memory; SPP of layer4 is 7.7 GB).  ``mode=F`` recomputes ONE metric
for ALL 135 000 query images (full 10 000 test images x clean + 12 cells + validation) for Experiment C/D.
  python -m atlas.stage_b --seed 2 --site layer3.22 --pool spp [--mode atlas|F] [--metric mahalanobis]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from . import common, data, features, knn, pooling, spec


def run(seed: int, site: str, pool: str, mode: str = "atlas", metric: str = "unit_l2", smoke: int = 0, out_root: str = None, bs: int = 250):
    dev = common.setup_torch(); common.check_protocol(seed)
    root = out_root or data.seed_dir(seed)
    tag = f"{site}__{pool}" + ("" if mode == "atlas" else f"__{metric}")
    folder = "knn_B" if mode == "atlas" else "knn_F"
    marker = f"{root}/{folder}/{tag}.done.json"
    if common.is_done(marker):
        print("skip", marker); return
    arrays = data.load_seed_arrays(seed)
    train_x, train_y = arrays["train"]
    if mode == "atlas":
        Q = data.atlas_queries(seed, arrays)
        sets = {"atlas": Q}
    else:
        full = data.load_full_sets()
        sets = {"val": np.asarray(arrays["val"][0])}
        for c in spec.CONDITIONS:
            sets[c] = full[data.cond_slice(c)]
    if smoke:
        train_x = train_x[:8000]; sets = {k: v[:smoke] for k, v in sets.items()}
    model, ckpt = common.load_model(seed, dev)
    tap = features.SiteTap(model, [site])
    nb = train_x.shape[0]
    bank = torch.empty(nb, spec.pool_dim(site, pool), device=dev)
    t0 = time.time()
    with torch.no_grad():
        for st, x in features.batches(train_x, bs, dev):
            _, maps = features.forward_maps(model, tap, x)
            bank[st:st + x.shape[0]] = pooling.pool(maps[site], pool)
    bnorm = bank.norm(dim=1)
    mu = bank.mean(0)
    info = {"bank_norm_mean": float(bnorm.mean()), "bank_zero_norm": int((bnorm < 1e-6).sum())}
    bank.sub_(mu)                                   # centered in place (raw_l2 and mahalanobis are translation invariant)
    metrics = spec.METRICS[1:] if mode == "atlas" else (metric,)
    if mode == "F" and metric == "unit_l2":
        bank.add_(mu); bank = bank / bank.norm(dim=1, keepdim=True).clamp_min(1e-12)
    res = {}
    mah = None
    for m in metrics:
        if m == "mahalanobis":
            mah = knn.LowRankMahalanobis.fit(bank, seed=0)
            info["mahalanobis"] = mah.info
            pb = mah.prepare_bank(bank, inplace=True)
        elif m == "raw_l2":
            bsq = bank.pow(2).sum(1)
    if mode == "F" and metric == "unit_l2":
        pass
    for name, arr in sets.items():
        n = arr.shape[0]
        out = {m: (np.empty((n, spec.K_NN), np.int32), np.empty((n, spec.K_NN), np.float32)) for m in metrics}
        qn = np.empty(n, np.float32)
        with torch.no_grad():
            for st, x in features.batches(arr, bs, dev):
                _, maps = features.forward_maps(model, tap, x)
                a = pooling.pool(maps[site], pool); e = st + x.shape[0]
                qn[st:e] = a.norm(dim=1).cpu().numpy()
                for m in metrics:
                    if m == "raw_l2":
                        d, i = knn.knn_raw(a - mu, bank, bsq)
                    elif m == "mahalanobis":
                        d, i = mah.knn(a - mu, pb)
                    else:
                        v, _ = pooling.unit(a); d, i = knn.knn_unit(v, bank)
                    out[m][0][st:e] = i.cpu().numpy().astype(np.int32); out[m][1][st:e] = d.cpu().numpy()
        payload = {"qnorm": qn}
        for m in metrics:
            payload[f"{m}_idx"], payload[f"{m}_dist"] = out[m]
        common.atomic_npz(f"{root}/{folder}/{tag}__{name}.npz", **payload)
    common.mark_done(marker, {"seed": seed, "site": site, "pool": pool, "mode": mode, "metrics": list(metrics), "checkpoint": ckpt,
                              "sets": {k: int(v.shape[0]) for k, v in sets.items()}, "info": info, "seconds": time.time() - t0,
                              "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0), "smoke": smoke})
    print("stage_b done", tag, time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--site", required=True); ap.add_argument("--pool", required=True)
    ap.add_argument("--mode", default="atlas", choices=["atlas", "F"]); ap.add_argument("--metric", default="unit_l2")
    ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--out_root", default=None)
    a = ap.parse_args(); run(a.seed, a.site, a.pool, a.mode, a.metric, a.smoke, a.out_root)
