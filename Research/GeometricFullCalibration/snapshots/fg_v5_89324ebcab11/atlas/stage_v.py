"""Stage V (GPU, per checkpoint): clustering and PCA-map diagnostics for the source-selected representations.

For each clean-selected (site, pooling) candidate and for the full-logit control: K-means (K=100, fixed seed, n_init=3,
50 Lloyd iterations, k-means++ init) on SOURCE unit-L2 reference features -- one declared common clustering space, not a
per-method metric -- then assignment of every atlas query to the frozen nearest centre; and a 2-D PCA fitted on the same clean
reference features (visualization only, never used for classification) into which clean and corrupted queries of the same
image IDs are projected. Assignments and coordinates are written; the ARI/NMI/composition analysis is CPU (atlas.viz).
  python -m atlas.stage_v --seed 2
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import common, data, features, pooling, spec


def kmeans(X: torch.Tensor, K: int, seed: int, n_init: int, iters: int, chunk: int = 8192):
    n = X.shape[0]
    best = None
    for r in range(n_init):
        g = torch.Generator(device=X.device).manual_seed(seed + r)
        C = X[torch.randint(n, (1,), generator=g, device=X.device)].clone()
        d2 = (X - C[0]).pow(2).sum(1)
        for _ in range(K - 1):
            j = torch.multinomial(d2.clamp_min(0) / d2.clamp_min(0).sum(), 1, generator=g)
            C = torch.cat([C, X[j]])
            d2 = torch.minimum(d2, (X - X[j]).pow(2).sum(1))
        for _ in range(iters):
            assign = torch.cat([(X[s:s + chunk] @ C.T - 0.5 * C.pow(2).sum(1)).argmax(1) for s in range(0, n, chunk)])
            newC = torch.zeros_like(C).index_add_(0, assign, X)
            cnt = torch.bincount(assign, minlength=K).float()
            keep = cnt > 0
            newC[keep] /= cnt[keep, None]
            newC[~keep] = C[~keep]
            if torch.equal(newC, C):
                break
            C = newC
        assign = torch.cat([(X[s:s + chunk] @ C.T - 0.5 * C.pow(2).sum(1)).argmax(1) for s in range(0, n, chunk)])
        inertia = float(sum((X[s:s + chunk] - C[assign[s:s + chunk]]).pow(2).sum() for s in range(0, n, chunk)))
        if best is None or inertia < best[0]:
            best = (inertia, C.clone(), assign)
    return best


def analyse(bank_unit: torch.Tensor, query_unit_fn, nq: int, tag: str, root: str, dev):
    t0 = time.time()
    inertia, C, a_bank = kmeans(bank_unit, spec.KMEANS_K, spec.KMEANS_SEED, spec.KMEANS_NINIT, spec.KMEANS_ITERS)
    mu = bank_unit.mean(0)
    Ac = bank_unit - mu
    torch.manual_seed(0)
    _, S, V = torch.svd_lowrank(Ac, q=10, niter=6)
    tot = float(Ac.pow(2).sum())
    ev = (S ** 2)[:2].cpu().numpy() / tot
    comps = V[:, :2]
    q_assign = np.empty(nq, np.int16); q_xy = np.empty((nq, 2), np.float32)
    for st, e, v in query_unit_fn():
        q_assign[st:e] = (v @ C.T - 0.5 * C.pow(2).sum(1)).argmax(1).cpu().numpy()
        q_xy[st:e] = ((v - mu) @ comps).cpu().numpy()
    b_xy = (Ac @ comps).cpu().numpy().astype(np.float32)
    common.atomic_npz(f"{root}/viz/{tag}.npz", bank_assign=a_bank.cpu().numpy().astype(np.int16), query_assign=q_assign, query_xy=q_xy,
                      bank_xy=b_xy[::10], explained_var_ratio=ev, inertia=np.array([inertia]))
    return {"seconds": time.time() - t0, "inertia": inertia, "pca_explained_variance_ratio_2d": ev.tolist(),
            "note": "PCA is a visualization map only; explained variance of 2 components is small for high-dimensional features"}


def run(seed: int, smoke: int = 0, out_root: str = None, bs: int = 250):
    dev = common.setup_torch(); common.check_protocol(seed)
    root = out_root or data.seed_dir(seed)
    marker = f"{root}/viz/done.json"
    if common.is_done(marker):
        print("skip"); return
    sl = json.load(open(f"{root}/shortlist.json"))["chosen"]
    arrays = data.load_seed_arrays(seed)
    train_x, train_y = arrays["train"]
    Q = data.atlas_queries(seed, arrays)
    if smoke:
        train_x = train_x[:8000]; Q = Q[:smoke]
    model, ckpt = common.load_model(seed, dev)
    info = {}
    for p in spec.POOLS:
        site = sl[p]["site"]
        tap = features.SiteTap(model, [site])
        bank = torch.empty(train_x.shape[0], spec.pool_dim(site, p), device=dev)
        with torch.no_grad():
            for st, x in features.batches(train_x, bs, dev):
                _, maps = features.forward_maps(model, tap, x)
                bank[st:st + x.shape[0]] = pooling.unit(pooling.pool(maps[site], p))[0]

        def qfn():
            with torch.no_grad():
                for st, x in features.batches(Q, bs, dev):
                    _, maps = features.forward_maps(model, tap, x)
                    yield st, st + x.shape[0], pooling.unit(pooling.pool(maps[site], p))[0]
        info[p] = {"site": site, **analyse(bank, qfn, Q.shape[0], f"{site}__{p}", root, dev)}
        tap.close(); del bank
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # full-logit control from the saved logits (centered, unit)
    ztr = torch.from_numpy(np.load(f"{root}/u0/train.npz")["logits"]).to(dev)
    zq = [np.load(f"{root}/u0/val.npz")["logits"]] + [np.load(f"{root}/u0/{c}.npz")["logits"][np.load(f"{data.SHARED}/subset_ids.npy")] for c in spec.CONDITIONS]
    zq = torch.from_numpy(np.concatenate(zq)).to(dev)
    if smoke:
        zq = zq[:smoke]
    u = lambda z: F.normalize(z - z.mean(1, keepdim=True), dim=1)  # noqa: E731
    def qfn2():
        for st in range(0, zq.shape[0], 5000):
            yield st, min(st + 5000, zq.shape[0]), u(zq[st:st + 5000])
    info["logits"] = analyse(u(ztr), qfn2, zq.shape[0], "logits", root, dev)
    common.mark_done(marker, {"seed": seed, "checkpoint": ckpt, "info": info, "kmeans": {"K": spec.KMEANS_K, "seed": spec.KMEANS_SEED, "n_init": spec.KMEANS_NINIT, "iters": spec.KMEANS_ITERS}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--out_root", default=None)
    a = ap.parse_args(); run(a.seed, a.smoke, a.out_root)
