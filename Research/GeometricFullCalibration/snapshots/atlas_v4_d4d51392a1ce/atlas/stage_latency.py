"""Stage L (GPU, per checkpoint): real-time cost of the selected pipelines.

End-to-end per query: backbone forward (to the deepest tapped site), pooling, exact bank search, native-DAC radii (five
banks, k=200), gate features + ridge (CPU numpy) and the scalar temperature. Reported at batch size 1 and 256 with warm-up,
CUDA synchronization, hardware and memory. The full high-dimensional bank search is exact (no ANN/compression).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import common, data, features, knn, pooling, spec, gate, stage_u0


def timeit(fn, n_warm, n_meas):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n_meas):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / n_meas


def run(seed: int, out_root: str = None):
    dev = common.setup_torch()
    root = out_root or data.seed_dir(seed)
    sl = json.load(open(f"{root}/shortlist.json"))["chosen"]; ms = json.load(open(f"{root}/metric_selection.json"))
    arrays = data.load_seed_arrays(seed)
    train_x = arrays["train"][0]
    model, ckpt = common.load_model(seed, dev)
    nt = stage_u0.NativeTap(model)
    Q = torch.from_numpy(np.ascontiguousarray(data.atlas_queries(seed, arrays)[5000:5000 + 512])).to(dev).float()
    nat_bank = {}
    zs = []
    with torch.no_grad():
        for st in range(0, 45000, 250):
            x = torch.from_numpy(np.ascontiguousarray(train_x[st:st + 250])).to(dev).float(); z = model(x)
            for n in stage_u0.NATIVE:
                nat_bank.setdefault(n, []).append(nt.f[n])
    nat_bank = {k: torch.cat(v) for k, v in nat_bank.items()}
    res = {"hardware": torch.cuda.get_device_name(0), "torch": torch.__version__, "tf32": False, "warmup": 20}
    with torch.no_grad():
        for bs, nm in ((1, 200), (256, 10)):
            xb = Q[:bs]
            res[f"bs{bs}"] = {"backbone_only_ms_per_batch": timeit(lambda: model(xb), 20, nm)}
            res[f"bs{bs}"]["backbone_plus_native_dac_radii_ms"] = timeit(lambda: (model(xb), [torch.kthvalue(torch.sqrt(torch.clamp(2 - 2 * (nt.f[n] @ nat_bank[n].T), min=0)), 200, dim=1) for n in stage_u0.NATIVE]), 20, nm)
        for p in spec.POOLS:
            site = sl[p]["site"]; metric = ms["hidden"][p]["metric"]
            tap = features.SiteTap(model, [site])
            bank = torch.empty(45000, spec.pool_dim(site, p), device=dev)
            for st in range(0, 45000, 250):
                x = torch.from_numpy(np.ascontiguousarray(train_x[st:st + 250])).to(dev).float(); _, maps = features.forward_maps(model, tap, x)
                bank[st:st + x.shape[0]] = pooling.pool(maps[site], p)
            mu = bank.mean(0); bank.sub_(mu)
            if metric == "unit_l2":
                bank.add_(mu); bank = bank / bank.norm(dim=1, keepdim=True).clamp_min(1e-12); bsq = pb = mah = None
            elif metric == "raw_l2":
                bsq = bank.pow(2).sum(1); mah = pb = None
            else:
                mah = knn.LowRankMahalanobis.fit(bank); pb = mah.prepare_bank(bank, inplace=True); bsq = None
            def search(xb):
                _, maps = features.forward_maps(model, tap, xb); a = pooling.pool(maps[site], p)
                if metric == "unit_l2":
                    return knn.knn_unit(pooling.unit(a)[0], bank)
                if metric == "raw_l2":
                    return knn.knn_raw(a - mu, bank, bsq)
                return mah.knn(a - mu, pb)
            for bs, nm in ((1, 100), (256, 5)):
                xb = Q[:bs]
                res[f"bs{bs}"][f"{p}:{site}:{metric}:backbone_pool_search_ms"] = timeit(lambda: search(xb), 20 if bs == 1 else 3, nm)
            res.setdefault("memory", {})[f"{p}:{site}"] = {"bank_bytes": int(bank.numel() * 4), "dim": int(bank.shape[1])}
            # gate + temperature cost on CPU (numpy), per query
            z = np.random.default_rng(0).normal(size=(256, 100)); i = z.argmax(1); j = np.roll(i, 1)
            pg = np.abs(np.random.default_rng(1).normal(size=(256, 100))); pg /= pg.sum(1, keepdims=True)
            t = time.perf_counter()
            for _ in range(20):
                X = gate.f1_features(z, i, j, pg, np.ones(256), np.ones(256))
                _ = gate.apply_temperature(pg, 1.1)
            res.setdefault("cpu_gate_features_and_temperature_ms_per_256", {})[p] = 1000 * (time.perf_counter() - t) / 20
            tap.close(); del bank, pb
            torch.cuda.empty_cache()
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    common.atomic_json(f"{root}/latency.json", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); a = ap.parse_args(); run(a.seed)
