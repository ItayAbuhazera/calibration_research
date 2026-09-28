"""G1-DP raw-representation extraction (docs/g1_dp_spec.md sec. 2), GPU, label-free.

H_3.22_raw = the exact input of the Stage-0 P_3.22 probe: global-average-pooled `layer3.22` output of the CIFAR ResNet-101, L2-normalized
per row (1024-d), strict FP32 (`atlas.common.setup_torch`), corrected_v2 inputs (atlas `test_sets_full.npy`), for the 13 exposed
conditions of one checkpoint. Stored float32.

Frozen consistency gate (no labels): reconstruct P_3.22 with the saved layer-pilot probe (index 8 of probe_weights.pt,
((x - mean)/std) @ W + b) and compare with the cached float16 `raw__probe_logits[:, 8, :]`:
    max |P_rec - P_cache| <= 0.05  and  TIE-AWARE top-class agreement >= 0.999 in every condition (docs/g1_dp_spec_amendment_1.md:
    a row is consistent iff argmax P_rec lies in T(x) = the set of classes attaining the max of the stored float16 P_cache),
and the model forward logits must match the atlas logits (max |dz| <= 1e-2, the G1 threshold). Failure exits non-zero.

    python -m atlas.g1dp_extract --seed 2
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import common, data, features, spec, stage0_data

OUT = "results/g1dp/h322"
SITE = "layer3.22"
PROBE_INDEX = 8
TOL_P, TOL_ARGMAX, TOL_Z = 0.05, 0.999, 1e-2
BS = 250


def probe_logits(x: np.ndarray, probe) -> np.ndarray:
    W, b, mean, std = (t.double().numpy() for t in probe[:4])
    return ((np.asarray(x, np.float64) - mean) / std) @ W + b


def tie_aware_agreement(pr: np.ndarray, pc_f16: np.ndarray) -> float:
    """Fraction of rows whose reconstructed argmax is among the maximizing classes of the stored float16 vector (Amendment 1)."""
    T = pc_f16 == pc_f16.max(1, keepdims=True)
    return float(T[np.arange(len(pr)), pr.argmax(1)].mean())


def main(seed: int):
    dev = common.setup_torch(); common.check_protocol(seed)
    out = f"{OUT}/seed{seed}"; os.makedirs(out, exist_ok=True)
    fs = stage0_data.frozen_state(seed); assert fs["candidates"][PROBE_INDEX]["module"] == SITE
    probe = torch.load(f"{stage0_data.LAYER_PILOT_ROOT}/checkpoint_seed{seed}/probe_weights.pt", map_location="cpu")["layer_probes"][PROBE_INDEX]
    model, ckpt = common.load_model(seed, dev)
    tap = features.SiteTap(model, [SITE]); full = data.load_full_sets()
    per, t0 = {}, time.time()
    for cond in spec.CONDITIONS:
        arr = full[data.cond_slice(cond)]
        H = np.empty((arr.shape[0], 1024), np.float32); Zm = np.empty((arr.shape[0], 100), np.float32)
        for st, x in features.batches(arr, BS, dev):
            z, maps = features.forward_maps(model, tap, x); e = st + x.shape[0]
            H[st:e] = F.normalize(maps[SITE].mean(dim=(2, 3)).float(), p=2, dim=-1).cpu().numpy(); Zm[st:e] = z.cpu().numpy()
        np.save(f"{out}/h_{cond}.tmp.npy", H); os.replace(f"{out}/h_{cond}.tmp.npy", f"{out}/h_{cond}.npy")
        pc16 = np.load(f"{stage0_data.LAYER_PILOT_ROOT}/checkpoint_seed{seed}/{cond}/per_sample.npz")["raw__probe_logits"][:, PROBE_INDEX, :]
        pc = pc16.astype(np.float64)
        pr = probe_logits(H, probe)
        za = np.load(f"{data.seed_dir(seed)}/u0/{cond}.npz")["logits"].astype(np.float64)
        per[cond] = {"max_abs_dP": float(np.abs(pr - pc).max()), "argmax_agreement_P": float((pr.argmax(1) == pc.argmax(1)).mean()), "tie_aware_agreement_P": tie_aware_agreement(pr, pc16),
                     "max_abs_dz_model_vs_atlas": float(np.abs(Zm.astype(np.float64) - za).max()), "sha256": common.file_sha(f"{out}/h_{cond}.npy")}
        print(cond, per[cond], flush=True)
    tap.close()
    mP = max(v["max_abs_dP"] for v in per.values()); mA = min(v["tie_aware_agreement_P"] for v in per.values()); mA_plain = min(v["argmax_agreement_P"] for v in per.values())
    mZ = max(v["max_abs_dz_model_vs_atlas"] for v in per.values())
    ok = mP <= TOL_P and mA >= TOL_ARGMAX and mZ <= TOL_Z
    res = {"seed": seed, "checkpoint": ckpt, "site": SITE, "definition": "L2-normalized GAP(layer3.22), 1024-d, strict FP32",
           "max_abs_dP": mP, "min_tie_aware_agreement_P": mA, "min_plain_argmax_agreement_P": mA_plain, "amendment": "g1_dp_spec_amendment_1", "max_abs_dz": mZ,
           "gate": f"max|P_rec-P_cache|<={TOL_P}, tie-aware top-class agreement>={TOL_ARGMAX}, max|dz|<={TOL_Z}", "consistency_passed": bool(ok),
           "per_condition": per, "seconds": time.time() - t0, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    common.atomic_json(f"{out}/consistency.json", res)
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True, choices=(2, 4)); main(ap.parse_args().seed)
