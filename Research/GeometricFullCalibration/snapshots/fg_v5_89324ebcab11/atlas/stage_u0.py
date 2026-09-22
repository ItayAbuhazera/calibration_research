"""Stage U0 (GPU, per checkpoint): base logits, native-DAC statistics and the logit-space kNN control.

Forward passes over the reference bank (45 000) and over ALL query sets: validation (5 000), clean test (10 000) and the
12 development cells (10 000 each). Writes base logits, native-DAC class-agnostic k=200 radii for the five benchmark
sources (conv1, layer1..layer4; logits are NOT a source), the native DAC weights refit on the FIT rows only (same
objective/optimizer as ``DensityAwareCalibrator.fit``), and exact k=50 neighbours in centered-logit space under the three
metrics (unit_l2, raw_l2, mahalanobis). Output-space features are centered logits z - mean(z) (softmax-gauge invariant).
  python -m atlas.stage_u0 --seed 2 [--smoke 300]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import optimize

from . import common, data, features, knn, spec

NATIVE = ("conv1", "layer1", "layer2", "layer3", "layer4")
K_DAC = 200


class NativeTap:
    def __init__(self, model):
        mods = dict(model.named_modules())
        self.f = {}
        self.h = [mods[n].register_forward_hook(self._mk(n)) for n in NATIVE]

    def _mk(self, n):
        def hook(_m, _i, out):
            self.f[n] = F.normalize(out.mean(dim=(2, 3)).float(), p=2, dim=-1)
        return hook


def fit_native_dac(s_fit: np.ndarray, z_fit: np.ndarray, y_fit: np.ndarray):
    """Squared-error DAC fit (paper Eq. 9), identical to DensityAwareCalibrator.fit's optimizer settings."""
    onehot = np.eye(z_fit.shape[1])[y_fit]

    def loss(w):
        T = np.clip(s_fit @ w[:-1] + w[-1], 1e-12, None)
        p = z_fit / T[:, None]
        p = np.exp(p - p.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        return float(((onehot - p) ** 2).sum())

    x0 = np.ones(s_fit.shape[1] + 1); x0[-1] = 0.0
    r = optimize.minimize(loss, x0, method="L-BFGS-B", bounds=[(0, None)] * s_fit.shape[1] + [(None, None)], tol=1e-12)
    return r.x, float(r.fun)


def logit_feats(z: torch.Tensor):
    c = z - z.mean(1, keepdim=True)
    return c, F.normalize(c, dim=1)


def run(seed: int, smoke: int = 0, out_root: str = None, bs: int = 250):
    dev = common.setup_torch(); common.check_protocol(seed)
    root = (out_root or data.seed_dir(seed)) + "/u0"
    marker = f"{root}/done.json"
    if common.is_done(marker):
        print("skip"); return
    arrays = data.load_seed_arrays(seed)
    train_x, train_y = arrays["train"]; val_x, val_y = arrays["val"]
    full = data.load_full_sets()
    if smoke:
        train_x, train_y = train_x[:8000], train_y[:8000]; val_x, val_y = val_x[:smoke], val_y[:smoke]
    model, ckpt = common.load_model(seed, dev)
    nt = NativeTap(model)
    t0 = time.time()
    nb = len(train_y)
    zs, nat = [], {n: [] for n in NATIVE}
    with torch.no_grad():
        for st, x in features.batches(train_x, bs, dev):
            z = model(x).float(); zs.append(z)
            for n in NATIVE: nat[n].append(nt.f[n])
    ztr = torch.cat(zs); nat_bank = {n: torch.cat(v).contiguous() for n, v in nat.items()}
    c_tr, u_tr = logit_feats(ztr)
    mah = knn.LowRankMahalanobis.fit(c_tr)
    pb = mah.prepare_bank(c_tr.clone())
    bank_sq = c_tr.pow(2).sum(1)
    minfo = mah.info

    def process(arr):
        n = arr.shape[0]
        lg = np.empty((n, 100), np.float32); s_glob = np.empty((n, 5), np.float32)
        out = {m: (np.empty((n, spec.K_NN), np.int32), np.empty((n, spec.K_NN), np.float32)) for m in spec.METRICS}
        with torch.no_grad():
            for st, x in features.batches(arr, bs, dev):
                e = st + x.shape[0]
                z = model(x).float(); lg[st:e] = z.cpu().numpy()
                for j, nme in enumerate(NATIVE):
                    d = torch.sqrt(torch.clamp(2 - 2 * (nt.f[nme] @ nat_bank[nme].T), min=0))
                    s_glob[st:e, j] = torch.kthvalue(d, K_DAC, dim=1).values.cpu().numpy()
                c, u = logit_feats(z)
                for m, (d, i) in (("unit_l2", knn.knn_unit(u, u_tr)), ("raw_l2", knn.knn_raw(c, c_tr, bank_sq)), ("mahalanobis", mah.knn(c, pb))):
                    out[m][0][st:e] = i.cpu().numpy().astype(np.int32); out[m][1][st:e] = d.cpu().numpy()
        return lg, s_glob, out

    sets = {"val": val_x}
    for cnd in spec.CONDITIONS:
        sets[cnd] = full[data.cond_slice(cnd)] if not smoke else full[data.cond_slice(cnd)][:smoke]
    res, sizes = {}, {}
    for name, arr in sets.items():
        res[name] = process(arr); sizes[name] = int(arr.shape[0])
    # native DAC on FIT rows only
    roles = data.make_roles(arrays["val"][1]) if not smoke else {"fit": np.arange(min(smoke, 200))}
    fi = roles["fit"][roles["fit"] < len(val_y)]
    w, f = fit_native_dac(res["val"][1][fi].astype(np.float64), res["val"][0][fi].astype(np.float64), val_y[fi])
    ltr = ztr.cpu().numpy()
    common.atomic_npz(f"{root}/train.npz", logits=ltr)
    for name, (lg, sg, out) in res.items():
        payload = {"logits": lg, "s_glob": sg}
        for m in spec.METRICS:
            payload[f"{m}_idx"], payload[f"{m}_dist"] = out[m]
        common.atomic_npz(f"{root}/{name}.npz", **payload)
    common.mark_done(marker, {"seed": seed, "checkpoint": ckpt, "sizes": sizes, "native_layers": list(NATIVE), "native_k": K_DAC,
                              "native_weights": w[:-1].tolist(), "native_intercept": float(w[-1]), "native_fit_rows": int(len(fi)),
                              "native_fit_objective": f, "logit_mahalanobis": {k: v for k, v in minfo.items()},
                              "seconds": time.time() - t0, "smoke": smoke, "peak_gpu_gb": (torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0) / 1e9})
    print("u0 done", time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--out_root", default=None)
    a = ap.parse_args(); run(a.seed, a.smoke, a.out_root)
