"""GPU numerics audit (per checkpoint): which precision/extraction path reproduces the corrected benchmark's base predictions?

For every condition whose benchmark base-model probabilities exist, the SAME checkpoint and pixels are run over the full 10 000 images in:
  bench_adapter_default  PyTorchModelAdapter(model, torch.device('cuda')).predict_proba, batch 256, library defaults (cudnn TF32 conv ON, matmul TF32 OFF;
                         note `use_amp = (self.device == "cuda")` is False for a torch.device, so autocast is NOT active in the benchmark path -- verified)
  bench_adapter_amp      same adapter with device string "cuda" (autocast fp16 active) -- a counterfactual, not the benchmark path
  fp32_strict_b250       atlas path (TF32 off, batch 250)   [must equal u0 logits]
  fp32_strict_b256       TF32 off, batch 256
  tf32_all_b256          TF32 on for conv and matmul, batch 256
Compared with the benchmark argmax/probabilities on the atlas-vs-benchmark mismatch image IDs plus a seeded random control of 1 000 agreeing images.
Only argmax, margins and the analysed subset's logits are stored. Idempotent per (condition).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import common, data, fg_data, spec, stats

MODES = ("bench_adapter_default", "bench_adapter_amp", "fp32_strict_b250", "fp32_strict_b256", "tf32_all_b256")


def run_mode(model, arr, mode):
    from utils.model_utils import PyTorchModelAdapter
    if mode.startswith("bench_adapter"):
        torch.backends.cudnn.allow_tf32 = True; torch.backends.cuda.matmul.allow_tf32 = False
        dev = torch.device("cuda") if mode.endswith("default") else "cuda"
        ad = PyTorchModelAdapter(model, dev, "cifar100")
        return ad.predict_proba(np.asarray(arr), batch_size=256), None
    tf = mode.startswith("tf32")
    torch.backends.cudnn.allow_tf32 = tf; torch.backends.cuda.matmul.allow_tf32 = tf
    bs = 250 if mode.endswith("b250") else 256
    zs = []
    with torch.no_grad():
        for s in range(0, len(arr), bs):
            x = torch.from_numpy(np.ascontiguousarray(arr[s:s + bs])).cuda().float()
            zs.append(model(x).float().cpu())
    z = torch.cat(zs).numpy().astype(np.float64)
    return stats.softmax(z), z


def main(seed: int):
    dev = torch.device("cuda")
    model, ckpt = common.load_model(seed, dev)
    full = data.load_full_sets()
    root = f"{fg_data.FG}/seed{seed}/numerics"; os.makedirs(root, exist_ok=True)
    bench = data.BENCH + "/evaluation"
    ty = np.load(f"{data.SHARED}/test_labels.npy")
    rng = np.random.default_rng(20261020)
    for c in spec.CONDITIONS:
        out = f"{root}/{c}.json"
        pbp = f"{bench}/checkpoint_seed{seed}/{c}/intermediates/method_outputs/base_model/probs.npy"
        if os.path.exists(out) or not os.path.exists(pbp) or not os.path.exists(f"{bench}/checkpoint_seed{seed}/{c}/summary_metrics.json"):
            continue
        t0 = time.time()
        pb = np.load(pbp).astype(np.float64)
        arr = np.asarray(full[data.cond_slice(c)])
        z_atlas = np.load(f"{data.seed_dir(seed)}/u0/{c}.npz")["logits"].astype(np.float64)
        pa = stats.softmax(z_atlas)
        mism = np.flatnonzero(pa.argmax(1) != pb.argmax(1))
        agree = np.flatnonzero(pa.argmax(1) == pb.argmax(1)); ctrl = np.sort(rng.choice(agree, 1000, replace=False))
        rec = {"seed": seed, "cond": c, "mismatch_ids": mism.tolist(), "n_mismatch_atlas_vs_bench": int(len(mism)), "modes": {}}
        for m in MODES:
            p, z = run_mode(model, arr, m)
            pm = p.argmax(1)
            info = {"argmax_mismatch_vs_bench_full": int((pm != pb.argmax(1)).sum()), "argmax_mismatch_vs_atlas_u0_full": int((pm != pa.argmax(1)).sum()),
                    "max_abs_dprob_vs_bench_full": float(np.abs(p - pb).max()), "median_abs_dprob_vs_bench_full": float(np.median(np.abs(p - pb).max(1))),
                    "mismatch_ids_reproduced_bench_argmax": int((pm[mism] == pb.argmax(1)[mism]).sum()) if len(mism) else 0,
                    "control_argmax_mismatch_vs_bench": int((pm[ctrl] != pb.argmax(1)[ctrl]).sum()), "acc": float((pm == ty).mean()), "acc_bench": float((pb.argmax(1) == ty).mean()),
                    "acc_atlas": float((pa.argmax(1) == ty).mean())}
            if z is not None:
                info["max_abs_dlogit_vs_atlas_u0_full"] = float(np.abs(z - z_atlas).max())
            rec["modes"][m] = info
        top2 = np.sort(z_atlas[mism], 1)[:, -2:] if len(mism) else np.zeros((0, 2))
        rec["mismatch_margins_atlas_logit"] = (top2[:, 1] - top2[:, 0]).tolist()
        rec["mismatch_true_label"] = ty[mism].tolist()
        rec["mismatch_atlas_pred"] = pa.argmax(1)[mism].tolist(); rec["mismatch_bench_pred"] = pb.argmax(1)[mism].tolist()
        rec["seconds"] = time.time() - t0
        rec["environment"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "checkpoint": ckpt}
        common.atomic_json(out, rec); print(c, rec["n_mismatch_atlas_vs_bench"], {m: rec["modes"][m]["argmax_mismatch_vs_bench_full"] for m in MODES}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); main(ap.parse_args().seed)
