"""Latency audit of the FROZEN fixed deep configuration (per checkpoint, GPU).

All variants are measured in ONE process on the same hooked model, interleaved round-robin (so drift and clock/thermal state affect every variant alike),
with data and banks resident on the GPU, CUDA synchronization before and after every timed call, warm-up, and many repetitions; median/p95/p99 reported.
Variants (same input batch, same batch size, same weights):
  backbone          model(x)                                   -- no hooks
  backbone_hooked   model(x) with the layer3.22 + native-DAC hooks attached (hook cost only)
  search_only       pooled layer3.22 features -> unit-L2 -> exact kNN over the resident 45 000 x 4096 bank (features precomputed; isolates search)
  full_gpu          hooked forward + 2x2 pool + normalize + exact search + native-DAC k=200 radii (5 banks)
  full_pipeline     full_gpu + neighbour indices to CPU + p_geo + Z1 features + frozen ridge + gate + q construction + temperature (numpy) => final probabilities
Not included: image decoding/host->device copy of pixels (the batch is resident), one-off bank construction. No optimization is applied (exact search, fp32, TF32 off).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from . import common, data, fg_data, fg_gate as FG, features, gate, knn, pooling, spec, stage_u0, stats

SITE, POOL = "layer3.22", "grid2"


def timed(fn):
    torch.cuda.synchronize(); t = time.perf_counter(); out = fn(); torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t), out


def main(seed: int, reps1: int = 300, reps256: int = 60):
    dev = common.setup_torch()
    root = f"{fg_data.FG}/seed{seed}"
    fz = json.load(open(f"{root}/gates_frozen.json"))
    g = fz["gates"]["deep|Z1|ofull|n2500"]
    model, ckpt = common.load_model(seed, dev)
    arrays = data.load_seed_arrays(seed); train_x = arrays["train"][0]; bank_y = torch.from_numpy(arrays["train"][1].astype(np.int64))
    prior = np.bincount(bank_y.numpy(), minlength=100) / 45000
    tap = features.SiteTap(model, [SITE]); nt = stage_u0.NativeTap(model)
    bank = torch.empty(45000, spec.pool_dim(SITE, POOL), device=dev); nat_bank = {n: [] for n in stage_u0.NATIVE}
    with torch.no_grad():
        for st, x in features.batches(train_x, 250, dev):
            _, maps = features.forward_maps(model, tap, x); bank[st:st + x.shape[0]] = pooling.pool(maps[SITE], POOL)
            for n in stage_u0.NATIVE:
                nat_bank[n].append(nt.f[n])
    bank = bank / bank.norm(dim=1, keepdim=True).clamp_min(1e-12); nat_bank = {k: torch.cat(v) for k, v in nat_bank.items()}
    Q = torch.from_numpy(np.ascontiguousarray(data.atlas_queries(seed, arrays)[5000:5000 + 256])).to(dev).float()
    w, w0 = np.asarray(json.load(open(f"{data.seed_dir(seed)}/u0/done.json"))["native_weights"]), json.load(open(f"{data.seed_dir(seed)}/u0/done.json"))["native_intercept"]
    sel = g["selected"]; lam = sel["lambda"] if sel["kind"] == "gate" else 1.0
    model_r = FG.deser(g["models"][str(lam)]); theta = sel["theta"] if sel["kind"] == "gate" else 0.0
    T = 1.0
    def search(a):
        return knn.knn_unit(pooling.unit(a)[0], bank)

    def radii(x):
        return [torch.kthvalue(torch.sqrt(torch.clamp(2 - 2 * (nt.f[n] @ nat_bank[n].T), min=0)), 200, dim=1).values for n in stage_u0.NATIVE]

    def full_gpu(x):
        tap.maps.clear(); z = model(x).float(); a = pooling.pool(tap.maps[SITE], POOL)
        d, i = search(a); r = torch.stack(radii(x), 1)
        return z, d, i, r, a.norm(dim=1)

    def full_pipeline(x):
        z, d, i, r, an = full_gpu(x)
        z = z.cpu().numpy().astype(np.float64); idx = i.cpu().numpy(); dist = d.cpu().numpy().astype(np.float64); s_glob = r.cpu().numpy().astype(np.float64)
        pg, _ = stats.p_geo(idx, bank_y.numpy(), prior)
        base = z.argmax(1); j = pg.argmax(1)
        qn = an.cpu().numpy().astype(np.float64)
        X = FG.features("Z1", z, base, j, pg, dist[:, -1], qn)
        gg = ((j != base) & (FG.predict(model_r, X) > theta))
        q0 = stats.softmax(z / np.maximum(s_glob @ w + w0, 1e-12)[:, None])
        q = gate.build_q(q0, pg, gg)
        return gate.apply_temperature(q, T)

    res = {"hardware": torch.cuda.get_device_name(0), "torch": torch.__version__, "tf32": False, "checkpoint": ckpt, "config": {"site": SITE, "pool": POOL, "metric": "unit_l2", "k": 50, "gate": "deep|Z1|ofull|n2500", "gate_kind": sel["kind"], "bank": [45000, int(bank.shape[1])],
           "bank_resident_bytes": int(bank.numel() * 4), "native_banks_resident_bytes": int(sum(v.numel() * 4 for v in nat_bank.values())), "exact_search": True, "fp32": True}}
    with torch.no_grad():
        for bs, reps in ((1, reps1), (256, reps256)):
            x = Q[:bs]
            # precomputed pooled features for the search-only variant
            tap.maps.clear(); model(x); a_pre = pooling.pool(tap.maps[SITE], POOL);
            variants = {"backbone_hooked": lambda: model(x), "search_only": lambda: search(a_pre), "full_gpu": lambda: full_gpu(x), "full_pipeline": lambda: full_pipeline(x)}
            for _ in range(30 if bs == 1 else 8):
                for f in variants.values():
                    f()
            times = {k: [] for k in variants}
            for _ in range(reps):
                for k, f in variants.items():
                    times[k].append(timed(f)[0])
            # hook-free backbone: temporarily remove the hooks
            saved = [h for h in tap.handles + nt.h]
            for h in saved:
                h.remove()
            bb = []
            for _ in range(30 if bs == 1 else 8):
                model(x)
            for _ in range(reps):
                bb.append(timed(lambda: model(x))[0])
            times["backbone"] = bb
            # re-attach hooks for the next batch size
            tap = features.SiteTap(model, [SITE]); nt = stage_u0.NativeTap(model)
            summ = {k: {"median_ms": float(np.median(v)), "p95_ms": float(np.quantile(v, .95)), "p99_ms": float(np.quantile(v, .99)), "mean_ms": float(np.mean(v)), "min_ms": float(np.min(v)), "n": len(v)} for k, v in times.items()}
            summ["note"] = "variants interleaved round-robin within one process; hook-free 'backbone' measured in a separate block on the same batch with hooks removed"
            res[f"bs{bs}"] = summ
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    common.atomic_json(f"{root}/latency_fixed_config.json", res)
    print(json.dumps({k: {kk: round(vv["median_ms"], 2) for kk, vv in res[k].items() if isinstance(vv, dict)} for k in ("bs1", "bs256")}, indent=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); main(ap.parse_args().seed)
