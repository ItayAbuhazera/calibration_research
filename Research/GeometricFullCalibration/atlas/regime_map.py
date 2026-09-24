"""Regime-map pilot (docs/regime_map_pilot_spec.md): pretrained ResNet-50 on CIFAR-100, four model
states, layer3-GAP probe evidence P, artifacts in the Stage 0 loader format.

States: a   = frozen ImageNet backbone + linear head trained on CIFAR-100 train
        b1  = same backbone fine-tuned end to end, saved after epoch 1 of 10
        b3  = ... after epoch 3
        b10 = ... final (epoch 10)
Resolution policy (all states): CIFAR images are corrupted at native 32x32 (the stored CIFAR-100-C
arrays), then bilinear-upsampled to 224x224 on the GPU and normalized with ImageNet statistics.
Extraction runs in strict fp32 (TF32 off); fine-tuning uses fp16 autocast.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from . import spec

OUT = "results/regime_map"
BENCH_SPLITS = "results/studyAB/phase0_corrected_v2/evaluation/checkpoint_seed2/clean/intermediates/splits"
FULL_SETS = "results/atlas/shared/test_sets_full.npy"
TEST_LABELS = "results/atlas/shared/test_labels.npy"
WEIGHTS_FILE = f"{OUT}/weights/resnet50-0676ba61.pth"
WEIGHTS_NAME = "torchvision ResNet50_Weights.IMAGENET1K_V1"
WEIGHTS_SHA256 = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
IN_MEAN, IN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
RES = 224
STATES = ("a", "b1", "b3", "b10")
FT_EPOCHS, FT_SAVE = 10, (1, 3, 10)
FT_SEED = 20260924
PROBE_GRID = (1e-4, 1e-3, 1e-2)            # layer-pilot recipe, unchanged across states
HEAD_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)  # regime (a) linear head only (deeper, so a wider grid)
NC = spec.NUM_CLASSES


def strict_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def weights_ok():
    h = hashlib.sha256(open(WEIGHTS_FILE, "rb").read()).hexdigest()
    assert h == WEIGHTS_SHA256, f"weights hash mismatch {h}"
    return h


def load_split(name):
    arr = np.load(f"{BENCH_SPLITS}/{name}_raw.npy", mmap_mode="r")
    return arr, np.load(f"{BENCH_SPLITS}/{name}_labels.npy").astype(np.int64)


def test_condition(cond, n=None):
    i = spec.CONDITIONS.index(cond)
    arr = np.load(FULL_SETS, mmap_mode="r")[i * 10000:(i + 1) * 10000]
    y = np.load(TEST_LABELS).astype(np.int64)
    return (arr[:n], y[:n]) if n else (arr, y)


def to_input(xn: torch.Tensor) -> torch.Tensor:
    """xn: [B,3,32,32] CIFAR-normalized float (as stored) on the target device -> ImageNet-normalized [B,3,224,224]."""
    dev = xn.device
    m, s = torch.tensor(CIFAR_MEAN, device=dev).view(1, 3, 1, 1), torch.tensor(CIFAR_STD, device=dev).view(1, 3, 1, 1)
    raw = torch.round((xn * s + m).clamp(0, 1) * 255) / 255          # exact 8-bit pixels
    up = F.interpolate(raw, size=(RES, RES), mode="bilinear", align_corners=False)
    im, isd = torch.tensor(IN_MEAN, device=dev).view(1, 3, 1, 1), torch.tensor(IN_STD, device=dev).view(1, 3, 1, 1)
    return ((up - im) / isd).contiguous(memory_format=torch.channels_last)


def build_model(state_dict=None, seed=FT_SEED):
    weights_ok()
    m = torchvision.models.resnet50()
    m.load_state_dict(torch.load(WEIGHTS_FILE, map_location="cpu"))
    g = torch.Generator().manual_seed(seed)
    fc = nn.Linear(2048, NC)
    with torch.no_grad():
        fc.weight.copy_(torch.randn(NC, 2048, generator=g) * 0.01); fc.bias.zero_()
    m.fc = fc
    if state_dict is not None:
        m.load_state_dict(state_dict)
    return m.to(memory_format=torch.channels_last)


def forward_feats(m, x):
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.layer2(m.layer1(x))
    f3 = m.layer3(x)
    f4 = m.layer4(f3)
    g3, g4 = f3.mean((2, 3)), f4.mean((2, 3))
    return g3, g4, m.fc(g4)


@torch.no_grad()
def extract(m, arr, bs=250, device="cuda"):
    """arr: array of CIFAR-normalized 32x32 images -> (g3 [N,1024], g4 [N,2048], logits [N,100]) float32 on CPU."""
    m.eval()
    g3, g4, lg = [], [], []
    for i in range(0, len(arr), bs):
        x = to_input(torch.from_numpy(np.ascontiguousarray(arr[i:i + bs])).to(device))
        a, b, c = forward_feats(m, x)
        g3.append(a.float().cpu()); g4.append(b.float().cpu()); lg.append(c.float().cpu())
    return torch.cat(g3), torch.cat(g4), torch.cat(lg)


def finetune(save_dir=f"{OUT}/ckpt", epochs=FT_EPOCHS, save_epochs=FT_SAVE, max_steps=None, bs=128, device="cuda", log=print):
    torch.manual_seed(FT_SEED); np.random.seed(FT_SEED)
    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")
    X = torch.from_numpy(np.ascontiguousarray(Xtr)).to(device); Y = torch.from_numpy(ytr).to(device)
    m = build_model().to(device)
    dec, nodec = [], []
    for n_, p in m.named_parameters():
        (nodec if p.ndim == 1 else dec).append(p)
    head = {id(p) for p in m.fc.parameters()}
    groups = [{"params": [p for p in dec if id(p) not in head], "lr": 0.01, "weight_decay": 5e-4},
              {"params": [p for p in nodec if id(p) not in head], "lr": 0.01, "weight_decay": 0.0},
              {"params": list(m.fc.parameters()), "lr": 0.1, "weight_decay": 5e-4}]
    opt = torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    base = [g["lr"] for g in opt.param_groups]
    spe = len(X) // bs
    total = epochs * spe
    scaler = torch.cuda.amp.GradScaler()
    os.makedirs(save_dir, exist_ok=True)
    step, t0, imgs = 0, time.time(), 0
    hist = []
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(len(X), device=device)
        tl = tc = 0.0
        for k in range(spe):
            idx = perm[k * bs:(k + 1) * bs]
            xb, yb = X[idx], Y[idx]
            flip = torch.rand(len(xb), device=device) < 0.5
            xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(3), xb)
            pad = F.pad(xb, (4, 4, 4, 4))
            offs = np.random.randint(0, 9, (len(xb) // 8 + 1, 2))          # random crop, one offset per 8 samples
            xb = torch.cat([pad[i:i + 8, :, offs[i // 8][1]:offs[i // 8][1] + 32, offs[i // 8][0]:offs[i // 8][0] + 32] for i in range(0, len(xb), 8)])
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * 0.5 * (1 + np.cos(np.pi * step / total))
            with torch.autocast("cuda", dtype=torch.float16):
                _, _, out = forward_feats(m, to_input(xb))
                loss = F.cross_entropy(out.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += float(loss) ; tc += float((out.argmax(1) == yb).float().mean())
            step += 1; imgs += len(xb)
            if max_steps and step >= max_steps:
                torch.cuda.synchronize()
                return {"steps": step, "images": imgs, "seconds": time.time() - t0, "img_per_s": imgs / (time.time() - t0)}
        _, _, lv = extract(m, Xva[:2000], device=device)
        rec = {"epoch": ep, "train_loss": tl / spe, "train_acc": tc / spe, "val2000_acc": float((lv.argmax(1).numpy() == yva[:2000]).mean()), "elapsed_s": time.time() - t0}
        hist.append(rec); log(json.dumps(rec))
        if ep in save_epochs:
            torch.save(m.state_dict(), f"{save_dir}/b{ep}.pt")
    json.dump({"history": hist, "weights": WEIGHTS_NAME, "weights_sha256": WEIGHTS_SHA256, "seed": FT_SEED, "epochs": epochs}, open(f"{save_dir}/finetune_log.json", "w"), indent=1)
    return hist


def fit_linear(x_tr, y_tr, x_val_fit, y_val_fit, grid, device="cuda"):
    """Layer-pilot recipe: L2-normalized GAP -> z-score -> multinomial LR (Calibrators.layer_readouts.fit_probe);
    lambda by inner-FIT NLL; scalar temperature by 1-D NLL on inner-FIT (stored, not applied)."""
    from Calibrators import layer_readouts as LR
    xt = F.normalize(x_tr, dim=1).to(device); yt = torch.from_numpy(y_tr).to(device)
    xv = F.normalize(x_val_fit, dim=1).to(device); yv = torch.from_numpy(y_val_fit).to(device)
    table, best = {}, None
    for lam in grid:
        pr = LR.fit_probe(xt, yt, NC, lam)
        nll = float(F.cross_entropy(pr.logits(xv), yv))
        table[str(lam)] = nll
        if best is None or nll < best[1] - 1e-12:
            best = (pr, nll, lam)
    pr = best[0]
    pr.temperature = LR.fit_probe_temperature(pr.logits(xv).cpu().numpy(), y_val_fit)
    return pr, {"selected_lambda": best[2], "inner_fit_nll_by_lambda": table, "temperature": pr.temperature,
                "lambda_at_grid_edge": best[2] in (grid[0], grid[-1])}


def apply_linear(pr, g):
    with torch.no_grad():
        return pr.logits(F.normalize(g, dim=1).to(pr.weight.device)).float().cpu()


def build_state(state, out_dir=None, n_train=None, n_val=None, n_test=None, conds=None, device="cuda", log=print):
    """Extract features, fit Z (state a only) and the layer3-GAP probe P on this state's own train features, write per-condition
    artifacts {z, p, labels} float32 plus a summary JSON."""
    from utils.decision_audit import make_inner_validation_split
    strict_fp32()
    out_dir = out_dir or f"{OUT}/{state}"
    os.makedirs(out_dir, exist_ok=True)
    m = build_model(None if state == "a" else torch.load(f"{OUT}/ckpt/{state}.pt", map_location="cpu")).to(device)
    Xtr, ytr = load_split("train"); Xva, yva = load_split("val")
    def strat(X, y, n):  # smoke-test subset: first n/100 rows of every class (keeps all classes present)
        k = max(1, n // NC)
        idx = np.sort(np.concatenate([np.flatnonzero(y == c)[:k] for c in range(NC)]))
        return X[idx], y[idx]
    if n_train: Xtr, ytr = strat(Xtr, ytr, n_train)
    if n_val: Xva, yva = strat(Xva, yva, n_val)
    t0 = time.time()
    g3t, g4t, lgt = extract(m, Xtr, device=device); g3v, g4v, lgv = extract(m, Xva, device=device)
    t_train_extract = time.time() - t0
    fit_idx, _ = make_inner_validation_split(yva, select_fraction=0.5, seed=123)
    summary = {"state": state, "weights": WEIGHTS_NAME, "weights_sha256": WEIGHTS_SHA256, "resolution": RES, "n_train": len(ytr), "n_val": len(yva)}
    P, info_p = fit_linear(g3t, ytr, g3v[fit_idx], yva[fit_idx], PROBE_GRID, device)
    summary["probe_layer3_gap"] = info_p
    if state == "a":
        H, info_h = fit_linear(g4t, ytr, g4v[fit_idx], yva[fit_idx], HEAD_GRID, device)
        summary["head_linear_layer4_gap"] = info_h
        zfun = lambda g4, lg: apply_linear(H, g4)
    else:
        zfun = lambda g4, lg: lg
    summary["train_acc_Z"] = float((zfun(g4t, lgt).argmax(1).numpy() == ytr).mean()); summary["train_acc_P"] = float((apply_linear(P, g3t).argmax(1).numpy() == ytr).mean())
    summary["val_acc_Z"] = float((zfun(g4v, lgv).argmax(1).numpy() == yva).mean()); summary["val_acc_P"] = float((apply_linear(P, g3v).argmax(1).numpy() == yva).mean())
    summary["conditions"] = {}
    t1, n_img = time.time(), 0
    for cond in (conds or spec.CONDITIONS):
        arr, y = test_condition(cond, n_test)
        g3, g4, lg = extract(m, arr, device=device); n_img += len(y)
        z, p = zfun(g4, lg).numpy(), apply_linear(P, g3).numpy()
        np.savez(f"{out_dir}/{cond}.npz", z=z.astype(np.float32), p=p.astype(np.float32), labels=y.astype(np.int16))
        summary["conditions"][cond] = {"acc_Z": float((z.argmax(1) == y).mean()), "acc_P": float((p.argmax(1) == y).mean()), "disagree_P_vs_Z": float((p.argmax(1) != z.argmax(1)).mean()),
                                        "both_wrong": float(((p.argmax(1) != y) & (z.argmax(1) != y)).mean())}
        log(f"{state} {cond} accZ {summary['conditions'][cond]['acc_Z']:.4f} accP {summary['conditions'][cond]['acc_P']:.4f}")
    summary["timing"] = {"train_val_extract_s": t_train_extract, "n_train_val_images": len(ytr) + len(yva), "test_extract_s": time.time() - t1, "n_test_images": n_img}
    json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=1)
    return summary
