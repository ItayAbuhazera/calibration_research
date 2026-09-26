"""Stage 2 extraction (docs/regime_map_followup_amendment_1.md sec. 6-7), GPU: h_L for the test conditions of one state, the head
(W, b [, mu, sigma]) used for the Check 2 projectors, the C1e probe logits p_L (P's exact protocol on h_L), and the consistency checks.
Consistency thresholds (frozen): b*: max|z_stored - (W h_L + b)| <= 1e-2; (a): the head refit at the stored lambda reproduces the stored argmax
on >= 99.9% of rows. A failure exits non-zero (nothing further is fitted).

    python -m atlas.followup_extract --state b10_s1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import regime_map as RM, spec

OUT = "results/regime_map_followup/hL"


def main(state):
    from Calibrators import layer_readouts as LR
    from utils.decision_audit import make_inner_validation_split
    RM.strict_fp32()
    out = f"{OUT}/{state}"; os.makedirs(out, exist_ok=True)
    kind, ep, sk = RM.split_state(state)
    sd = None if kind == "a" else torch.load(f"{RM.OUT}/ckpt/s{sk}/b{ep}.pt", map_location="cpu")
    m = RM.build_model(sd).to("cuda")
    t0 = time.time()
    Xtr, ytr = RM.load_split("train"); Xva, yva = RM.load_split("val")
    g4t = RM.extract(m, Xtr)[1]; g4v = RM.extract(m, Xva)[1]
    fit_idx, _ = make_inner_validation_split(yva, select_fraction=0.5, seed=123)
    PL, info_pl = RM.fit_linear(g4t, ytr, g4v[fit_idx], yva[fit_idx], RM.PROBE_GRID, "cuda")      # P's exact protocol, applied to h_L
    summ = json.load(open(f"{RM.OUT}/{state}/summary.json"))
    if kind == "a":
        lam = summ["head_linear_layer4_gap"]["selected_lambda"]
        H = LR.fit_probe(F.normalize(g4t, dim=1).to("cuda"), torch.from_numpy(ytr).to("cuda"), RM.NC, lam)
        head = {"kind": "a", "W_h": H.weight.T.cpu().numpy().astype(np.float64), "b": H.bias.cpu().numpy().astype(np.float64),
                "mu": H.mean.cpu().numpy().astype(np.float64), "sigma": H.std.cpu().numpy().astype(np.float64), "lambda": lam}
        zfun = lambda g4: RM.apply_linear(H, g4).numpy().astype(np.float64)
    else:
        W, b = sd["fc.weight"].numpy().astype(np.float64), sd["fc.bias"].numpy().astype(np.float64)
        head = {"kind": "b", "W_h": W, "b": b}
        zfun = lambda g4: g4.numpy().astype(np.float64) @ W.T + b
    np.savez(f"{out}/head.npz", **head)
    maxdev, agree_num, n_all, mabs = 0.0, 0, 0, 0.0
    for cond in spec.CONDITIONS:
        arr, y = RM.test_condition(cond)
        g3, g4, lg = RM.extract(m, arr)
        np.save(f"{out}/h_{cond}.npy", g4.numpy().astype(np.float32))
        np.save(f"{out}/pl_{cond}.npy", RM.apply_linear(PL, g4).numpy().astype(np.float32))
        st = np.load(f"{RM.OUT}/{state}/{cond}.npz"); assert np.array_equal(st["labels"].astype(np.int64), y)
        zr, zs = zfun(g4), st["z"].astype(np.float64)
        maxdev = max(maxdev, float(np.abs(zr - zs).max())); mabs += float(np.abs(zr - zs).mean()) / len(spec.CONDITIONS)
        agree_num += int((zr.argmax(1) == zs.argmax(1)).sum()); n_all += len(y)
    agree = agree_num / n_all
    ok = (maxdev <= 1e-2) if kind == "b" else (agree >= 0.999)
    res = {"state": state, "kind": kind, "max_abs_dz": maxdev, "mean_abs_dz": mabs, "argmax_agreement": agree, "consistency_passed": bool(ok),
           "threshold": "max|dz| <= 1e-2" if kind == "b" else "argmax agreement >= 0.999", "C1e_probe": info_pl,
           "seconds": time.time() - t0, "extract_img_per_s_train_val_test": (len(ytr) + len(yva) + n_all) / (time.time() - t0)}
    json.dump(res, open(f"{out}/consistency.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--state", required=True, choices=RM.STATES); main(ap.parse_args().state)
