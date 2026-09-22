"""
Bounded paired clean-input diagnostic: legacy vs corrected preprocessing.

The SAME images and the SAME checkpoint are pushed through both the legacy
(ImageNet statistics on the CIFAR-100 clean test split) and the corrected
(training statistics) evaluation transforms. Nothing here selects a protocol:
the corrected protocol is fixed by checkpoint provenance (see
utils/preprocessing_protocol.py). This script only QUANTIFIES what the
mismatch did.

Three blocks
------------
A. Provenance reproduction. The checkpoint's own training record
   (results.json -> evaluation_results) was produced on the validation split
   under the training (CIFAR) statistics. We rebuild that split from the seed
   and re-evaluate under BOTH transforms. Reproducing the recorded numbers
   under CIFAR statistics (and not under ImageNet ones) is direct evidence of
   which normalization the checkpoint saw, independent of test data.
B. Paired base-model effect on the clean test split: accuracy, NLL, top-1
   agreement, logit shift.
C. Representation-distance / neighbour diagnostic against the clean reference
   bank (train split, CIFAR statistics -- exactly how every DAC bank was
   built): for the five native-DAC layers, the k-th-neighbour distance s_l(x)
   (native DAC's own statistic, k=200) and top-10 neighbour-set overlap for
   the same test image under the two transforms; and the validation-vs-test
   s_l(x) contrast under each protocol. Validation and test are both clean,
   unseen images, so under a consistent protocol their s_l should be
   exchangeable; a gap that appears only under the legacy protocol is a
   preprocessing artifact.

Output: <out_dir>/preprocessing_diagnostic_seed{S}.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import preprocessing_protocol as pp  # noqa: E402
from utils.model_utils import construct_model_path, load_trained_model  # noqa: E402

DAC_LAYERS = ("conv1", "layer1", "layer2", "layer3", "layer4")
K_DAC = 200  # get_dac_k_value("cifar100")


def val_indices(seed: int, n: int = 50000, valid_size: float = 0.1):
    """Replicates data/cifar100.py::get_train_valid_loader's split."""
    idx = list(range(n))
    split = int(np.floor(valid_size * n))
    np.random.seed(seed)
    np.random.shuffle(idx)
    return idx[split:], idx[:split]  # train, valid


def make_loader(train: bool, transform, indices, batch_size: int, root: str):
    ds = datasets.CIFAR100(root=root, train=train, download=False, transform=transform)
    if indices is not None:
        ds = Subset(ds, list(indices))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)


@torch.no_grad()
def run(model, loader, device):
    """Returns logits [N,C], labels [N], pooled+normalized DAC-layer features."""
    feats = {k: [] for k in DAC_LAYERS}
    hooks = []
    mods = dict(model.named_modules())

    def mk(name):
        def hook(_m, _i, out):
            f = out.mean(dim=(2, 3)) if out.ndim == 4 else out
            feats[name].append(F.normalize(f.float(), p=2, dim=-1).cpu())

        return hook

    for name in DAC_LAYERS:
        hooks.append(mods[name].register_forward_hook(mk(name)))
    logits, labels = [], []
    model.eval()
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    for h in hooks:
        h.remove()
    return (
        torch.cat(logits),
        torch.cat(labels),
        {k: torch.cat(v) for k, v in feats.items()},
    )


def nll_acc_ece(logits: torch.Tensor, labels: torch.Tensor, bins: int = 15):
    p = F.softmax(logits, dim=1)
    nll = F.cross_entropy(logits, labels).item()
    conf, pred = p.max(1)
    correct = (pred == labels).float()
    ece = 0.0
    edges = torch.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(correct[m].mean().item() - conf[m].mean().item())
    return {"acc": correct.mean().item(), "nll": nll, "ece15": ece}


def knn_stats(queries: torch.Tensor, bank: torch.Tensor, device, k: int, top: int = 10):
    """k-th NN distance (true Euclidean on unit vectors) and top-`top` neighbour ids."""
    bank_d = bank.to(device)
    kth, nbr = [], []
    for s in range(0, queries.shape[0], 512):
        q = queries[s : s + 512].to(device)
        d = torch.sqrt(torch.clamp(2.0 - 2.0 * (q @ bank_d.t()), min=0.0))
        kth.append(torch.kthvalue(d, k, dim=1).values.cpu())
        nbr.append(torch.topk(d, top, dim=1, largest=False).indices.cpu())
    return torch.cat(kth).numpy(), torch.cat(nbr).numpy()


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([len(set(x) & set(y)) / a.shape[1] for x, y in zip(a, b)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--results_dir", required=True, help="checkpoint root (as run_unified_benchmark --results_dir)")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_query", type=int, default=10000, help="test images used for block C (bounded)")
    ap.add_argument("--batch_size", type=int, default=250)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable (refusing to fall back to CPU)")
    device = torch.device(args.device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()

    ckpt = construct_model_path(args.results_dir, "baseline_cross_entropy", "cifar100", "resnet101", args.seed)
    model = load_trained_model(ckpt, "resnet101", 100, device, dataset="cifar100")

    legacy_tf = pp.eval_transform("cifar100", pp.PROTOCOL_LEGACY)
    corr_tf = pp.eval_transform("cifar100", pp.PROTOCOL_CORRECTED)
    train_idx, valid_idx = val_indices(args.seed)

    out = {
        "seed": args.seed,
        "checkpoint": ckpt,
        "protocols": {"legacy": pp.PROTOCOL_LEGACY, "corrected": pp.PROTOCOL_CORRECTED},
        "constants": {
            "legacy_test": pp.eval_normalization("cifar100", pp.PROTOCOL_LEGACY),
            "corrected_test": pp.eval_normalization("cifar100", pp.PROTOCOL_CORRECTED),
        },
        "tf32_disabled": True,
    }

    # ---- A. provenance reproduction on the validation split -----------------
    rec_path = os.path.join(
        os.path.dirname(ckpt), f"baseline_cross_entropy_cifar100_resnet101_seed{args.seed}", "results.json"
    )
    recorded = json.load(open(rec_path))["evaluation_results"] if os.path.exists(rec_path) else None
    blk = {"training_record_evaluation_results(val, CIFAR stats)": recorded}
    val_out = {}
    for name, tf in (("corrected_cifar_stats", corr_tf), ("legacy_imagenet_stats", legacy_tf)):
        lg, lb, ft = run(model, make_loader(True, tf, valid_idx, args.batch_size, args.data_root), device)
        m = nll_acc_ece(lg, lb)
        blk[name] = m
        val_out[name] = (lg, lb, ft)
    out["A_val_reproduction"] = blk

    # ---- reference bank (train split, CIFAR stats == how every DAC bank was built)
    _, tlab, bank = run(model, make_loader(True, corr_tf, train_idx, args.batch_size, args.data_root), device)

    # ---- B/C. clean test under both transforms ------------------------------
    n = min(args.n_query, 10000)
    test_idx = list(range(n))
    res = {}
    for name, tf in (("legacy", legacy_tf), ("corrected", corr_tf)):
        res[name] = run(model, make_loader(False, tf, test_idx, args.batch_size, args.data_root), device)
    (lgL, lab, ftL), (lgC, lab2, ftC) = res["legacy"], res["corrected"]
    assert torch.equal(lab, lab2), "test labels differ between transforms"

    predL, predC = lgL.argmax(1), lgC.argmax(1)
    wrong_to_right = int(((predL != lab) & (predC == lab)).sum())
    right_to_wrong = int(((predL == lab) & (predC != lab)).sum())
    out["B_clean_test_base_model"] = {
        "n": n,
        "legacy": nll_acc_ece(lgL, lab),
        "corrected": nll_acc_ece(lgC, lab),
        "top1_agreement": float((predL == predC).float().mean()),
        "corrected_fixes_legacy_errors(W)": wrong_to_right,
        "corrected_breaks_legacy_correct(H)": right_to_wrong,
        "mean_abs_logit_shift": float((lgL - lgC).abs().mean()),
        "mean_max_prob": {
            "legacy": float(F.softmax(lgL, 1).max(1).values.mean()),
            "corrected": float(F.softmax(lgC, 1).max(1).values.mean()),
        },
    }

    C = {}
    lv = val_out["corrected_cifar_stats"][2]
    for layer in DAC_LAYERS:
        b = bank[layer]
        k = min(K_DAC, b.shape[0])
        sL, nbL = knn_stats(ftL[layer], b, device, k)
        sC, nbC = knn_stats(ftC[layer], b, device, k)
        sV, _ = knn_stats(lv[layer], b, device, k)
        C[layer] = {
            "dim": int(b.shape[1]),
            "s_l_mean": {"legacy_test": float(sL.mean()), "corrected_test": float(sC.mean()), "val_cifar": float(sV.mean())},
            "s_l_paired_mean_abs_diff_legacy_vs_corrected": float(np.abs(sL - sC).mean()),
            "s_l_paired_mean_rel_diff": float(np.mean((sL - sC) / np.maximum(sC, 1e-12))),
            "test_vs_val_mean_ratio": {
                "legacy": float(sL.mean() / sV.mean()),
                "corrected": float(sC.mean() / sV.mean()),
            },
            "top10_neighbour_overlap_legacy_vs_corrected": overlap(nbL, nbC),
        }
    out["C_dac_layer_distance_diagnostic"] = C
    out["wall_seconds"] = time.time() - t0

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"preprocessing_diagnostic_seed{args.seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print("wrote", path)


if __name__ == "__main__":
    main()
