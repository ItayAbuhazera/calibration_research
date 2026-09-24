"""CLI for the regime-map pilot (docs/regime_map_pilot_spec.md).

  python -m atlas.regime_run finetune
  python -m atlas.regime_run state --state a|b1|b3|b10
  python -m atlas.regime_run smoke            # few-hundred-image smoke test + throughput timing (GPU)
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from . import regime_map as RM


def smoke():
    RM.strict_fp32()
    out = {"weights_sha256": RM.weights_ok(), "device": torch.cuda.get_device_name(0)}
    # 1) fine-tune throughput (fp16 autocast), 40 steps after a 5-step warmup
    RM.finetune(save_dir=f"{RM.OUT}/smoke/ckpt", max_steps=5, log=print)
    torch.cuda.synchronize()
    r = RM.finetune(save_dir=f"{RM.OUT}/smoke/ckpt", max_steps=40, log=print)
    out["finetune_throughput"] = r
    # 2) strict-fp32 extraction throughput on 2000 test-condition images
    m = RM.build_model().to("cuda")
    arr, _ = RM.test_condition("gaussian_noise_s3", 2000)
    RM.extract(m, arr[:250]); torch.cuda.synchronize()
    t = time.time(); RM.extract(m, arr); torch.cuda.synchronize()
    out["extract_fp32_img_per_s"] = 2000 / (time.time() - t)
    out["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    # 3) tiny end-to-end pipeline: state a (frozen) on 600 train / 500 val / 300 test images, 3 conditions
    s = RM.build_state("a", out_dir=f"{RM.OUT}/smoke/a", n_train=600, n_val=500, n_test=300, conds=["clean", "gaussian_noise_s3", "fog_s5"])
    out["state_a_smoke"] = {k: s[k] for k in ("train_acc_Z", "val_acc_Z", "train_acc_P", "val_acc_P", "probe_layer3_gap", "head_linear_layer4_gap", "conditions")}
    # 4) artifact schema check: loadable by the Stage 0 loader
    from . import regime_data
    d = regime_data.load_cell("a", "clean", root=f"{RM.OUT}/smoke")
    out["schema"] = {k: list(v.shape) if hasattr(v, "shape") else v for k, v in d.items()}
    json.dump(out, open(f"{RM.OUT}/smoke/smoke.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["finetune", "state", "smoke"])
    ap.add_argument("--state", choices=RM.STATES)
    a = ap.parse_args()
    if a.cmd == "finetune":
        RM.finetune()
    elif a.cmd == "state":
        RM.build_state(a.state)
    else:
        os.makedirs(f"{RM.OUT}/smoke", exist_ok=True)
        smoke()
