"""G1 extraction (docs/g1_conditional_access_spec.md sec. 2), GPU: H_L (layer4 output, GAP, 2048-d) of the CIFAR ResNet-101 for the
13 exposed conditions of one checkpoint, the effective head (W_eff = fc.weight / temp, b_eff = fc.bias / temp) and the frozen
consistency gate max |z_atlas - (H_L W_eff^T + b_eff)| <= 1e-2. A failure exits non-zero and nothing further may be fitted.

    python -m atlas.g1_extract --seed 2
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from . import common, data, features, spec

OUT = "results/g1/hL"
TOL = 1e-2
BS = 250


def head_arrays(model):
    temp = float(model.temp.item() if torch.is_tensor(model.temp) else model.temp)
    W = model.fc.weight.detach().double().cpu().numpy() / temp
    b = model.fc.bias.detach().double().cpu().numpy() / temp
    return W, b, temp


def reconstruct(h: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(h, dtype=np.float64) @ W.T + b


def main(seed: int):
    dev = common.setup_torch(); common.check_protocol(seed)
    out = f"{OUT}/seed{seed}"; os.makedirs(out, exist_ok=True)
    model, ckpt = common.load_model(seed, dev)
    W, b, temp = head_arrays(model)
    np.savez(f"{out}/head.npz", W_eff=W, b_eff=b, temp=temp)
    tap = features.SiteTap(model, ["layer4"])
    full = data.load_full_sets()
    t0 = time.time(); per = {}
    maxdev = maxdev_model = 0.0; agree = n_all = 0
    for cond in spec.CONDITIONS:
        arr = full[data.cond_slice(cond)]
        H = np.empty((arr.shape[0], 2048), np.float32); Zm = np.empty((arr.shape[0], 100), np.float32)
        for st, x in features.batches(arr, BS, dev):
            z, maps = features.forward_maps(model, tap, x)
            e = st + x.shape[0]
            H[st:e] = maps["layer4"].mean(dim=(2, 3)).float().cpu().numpy(); Zm[st:e] = z.cpu().numpy()
        np.save(f"{out}/h_{cond}.npy", H)
        za = np.load(f"{data.seed_dir(seed)}/u0/{cond}.npz")["logits"].astype(np.float64)
        zr = reconstruct(H, W, b)
        d = float(np.abs(zr - za).max()); dm = float(np.abs(Zm.astype(np.float64) - za).max())
        per[cond] = {"max_abs_dz_reconstruct_vs_atlas": d, "max_abs_dz_model_vs_atlas": dm,
                     "argmax_agreement": float((zr.argmax(1) == za.argmax(1)).mean())}
        maxdev, maxdev_model = max(maxdev, d), max(maxdev_model, dm)
        agree += int((zr.argmax(1) == za.argmax(1)).sum()); n_all += len(za)
    tap.close()
    ok = maxdev <= TOL
    res = {"seed": seed, "checkpoint": ckpt, "temp": temp, "max_abs_dz": maxdev, "max_abs_dz_model_forward_vs_atlas": maxdev_model,
           "argmax_agreement": agree / n_all, "threshold": f"max|z_atlas - (H_L W_eff^T + b_eff)| <= {TOL}", "consistency_passed": bool(ok),
           "per_condition": per, "seconds": time.time() - t0, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    json.dump(res, open(f"{out}/consistency.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "per_condition"}, indent=1, default=float))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True, choices=(2, 4)); main(ap.parse_args().seed)
