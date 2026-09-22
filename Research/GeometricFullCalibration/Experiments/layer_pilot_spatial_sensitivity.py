"""
Bounded spatial-pooling sensitivity for the layer-selection pilot
(docs/layer_selection_pilot_spec.md §4.3). Family A only.

Pre-declared and NOT searched: replace global average pooling by ONE fixed
alternative, a 2x2 adaptive average pool (dim -> 4*dim), applied to the layers of
the GAP-selected nested set A_8 of the main pilot. The selection is NOT re-run;
beta is refitted per nested set A_1 ⊂ A_4 ⊂ A_6 ⊂ A_8 on inner-FIT with the same
equal-weight aggregation, K_c and S_DAC as the main pilot. S_DAC still uses native
DAC's GAP features. No other pooling configuration is tried.

A null here bounds only THIS alternative: it does not show the absence of
information in the unpooled tensor.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Calibrators.density_aware_calibration import DensityAwareCalibrator  # noqa: E402,F401
from Calibrators import layer_readouts as LR  # noqa: E402
from Calibrators.full_vector_dac import np_softmax, normalized_euclidean  # noqa: E402
from Experiments import layer_selection_pilot as P  # noqa: E402
from utils import preprocessing_protocol as pp  # noqa: E402
from utils.decision_audit import make_inner_validation_split  # noqa: E402
from utils.model_utils import construct_model_path, load_trained_model  # noqa: E402
from utils.unified_metrics import evaluate_all  # noqa: E402

GRID = 2


class DualTap:
    """GAP+L2 for the native-DAC layers, grid2+L2 for the selected layers."""

    def __init__(self, model, gap_names: List[str], grid_names: List[str]):
        mods = dict(model.named_modules())
        self.gap, self.grid, self.h = {}, {}, []
        for n in gap_names:
            self.h.append(mods[n].register_forward_hook(self._mk(n, False)))
        for n in grid_names:
            self.h.append(mods[n].register_forward_hook(self._mk(n, True)))

    def _mk(self, n, grid):
        def hook(_m, _i, out):
            if grid:
                f = F.adaptive_avg_pool2d(out, (GRID, GRID)).flatten(1)
                self.grid[n] = F.normalize(f.float(), p=2, dim=-1)
            else:
                f = out.mean(dim=(2, 3)) if out.ndim == 4 else out
                self.gap[n] = F.normalize(f.float(), p=2, dim=-1)

        return hook


@torch.no_grad()
def run_split(model, tap, images, batch_size, device, banks=None, gap_banks=None, sel_names=None, native_names=None,
              collect=False):
    n = images.shape[0]
    z = np.empty((n, P.NUM_CLASSES), np.float32)
    feats_gap = {k: [] for k in native_names}
    feats_grid = {k: [] for k in sel_names}
    r = np.empty((n, len(sel_names), P.NUM_CLASSES), np.float32) if banks else None
    s = np.empty((n, len(native_names)), np.float32) if gap_banks else None
    for st, x in P.iter_batches(images, batch_size, device):
        e = st + x.shape[0]
        tap.gap.clear(); tap.grid.clear()
        z[st:e] = model(x).float().cpu().numpy()
        if collect:
            for k in native_names: feats_gap[k].append(tap.gap[k])
            for k in sel_names: feats_grid[k].append(tap.grid[k])
        if banks:
            for j, name in enumerate(sel_names):
                r_t, _, _ = banks[name].query(tap.grid[name])
                r[st:e, j] = r_t
            for j, name in enumerate(native_names):
                d = normalized_euclidean(tap.gap[name], gap_banks[name])
                s[st:e, j] = torch.kthvalue(d, LR.K_DAC, dim=1).values.cpu().numpy()
    out = {"z": z, "r": r, "s_glob": s}
    if collect:
        out["gap"] = {k: torch.cat(v) for k, v in feats_gap.items()}
        out["grid"] = {k: torch.cat(v) for k, v in feats_grid.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--benchmark_root", required=True)
    ap.add_argument("--pilot_root", required=True, help="results/layer_pilot (main GAP pilot, must be complete)")
    ap.add_argument("--cifar_c_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA unavailable")
    device = torch.device(args.device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    seed_dir = os.path.join(args.pilot_root, f"checkpoint_seed{args.seed}")
    state = json.load(open(os.path.join(seed_dir, "frozen_state.json")))
    order = state["selection"]["family_a"]["order"]
    sets = LR.nested_sets(order)
    sel_idx_list = sets[8]
    sel_names = [LR.CANDIDATE_NAMES[i] for i in sel_idx_list]  # nested: sets[L] = prefix of this list
    native_names = list(LR.NATIVE_DAC_LAYERS)

    clean_dir = os.path.join(args.benchmark_root, "evaluation", f"checkpoint_seed{args.seed}", "clean")
    fitted_dir = os.path.join(args.benchmark_root, "fitted_method", f"checkpoint_seed{args.seed}")
    for d in (os.path.join(clean_dir, "intermediates"), fitted_dir):
        pp.require_compatible(d, P.DATASET, pp.PROTOCOL_CORRECTED, d)
        if pp.read_protocol(d) != pp.PROTOCOL_CORRECTED:
            raise SystemExit(f"{d} is not corrected-protocol")
    sp = lambda k: os.path.join(clean_dir, "intermediates", "splits", k)  # noqa: E731
    train_raw, train_y = np.load(sp("train_raw.npy"), mmap_mode="r"), np.load(sp("train_labels.npy"))
    val_raw, val_y = np.load(sp("val_raw.npy"), mmap_mode="r"), np.load(sp("val_labels.npy"))
    test_raw, test_y = np.load(sp("test_raw.npy"), mmap_mode="r"), np.load(sp("test_labels.npy"))
    fit_idx, select_idx = make_inner_validation_split(val_y, select_fraction=0.5, seed=P.INNER_SEED)
    assert LR.state_hash({}, [fit_idx]) == state["inner_split"]["fit_idx_hash"], "inner split differs from the main pilot"

    ckpt = construct_model_path(args.results_dir, P.METHOD, P.DATASET, P.MODEL, args.seed)
    model = load_trained_model(ckpt, P.MODEL, P.NUM_CLASSES, device, dataset=P.DATASET).eval()
    # native DAC's GAP layers use the same module names as native_dac (layer1..4 are Sequential outputs)
    tap = DualTap(model, native_names, sel_names)

    perm = LR.permuted_labels(train_y, LR.PERMUTED_LABEL_SEED + args.seed)  # unused, LayerBank requires it
    print("[fit] banks", flush=True)
    tr = run_split(model, tap, train_raw, args.batch_size, device, sel_names=sel_names, native_names=native_names, collect=True)
    banks = {n: P.LayerBank(tr["grid"][n], train_y, perm, device) for n in sel_names}
    gap_banks = {n: tr["gap"][n].to(device).contiguous() for n in native_names}
    del tr
    torch.cuda.empty_cache()

    native = pickle.load(open(os.path.join(fitted_dir, "native_dac.pkl"), "rb"))
    w, w0 = np.asarray(native.weights[:-1], np.float64), float(native.weights[-1])
    assert np.allclose(w, state["native_dac"]["weights"]) and abs(w0 - state["native_dac"]["intercept"]) < 1e-9

    val = run_split(model, tap, val_raw, args.batch_size, device, banks=banks, gap_banks=gap_banks,
                    sel_names=sel_names, native_names=native_names)
    s_val = LR.s_dac(val["s_glob"], w, w0)
    arms: Dict[str, Any] = {}
    for L in LR.LAYER_COUNTS:
        cols = list(range(L))  # r columns are ordered as sel_names, a prefix = the nested set
        fit = LR.fit_family_a(val["z"][fit_idx], val["r"][fit_idx], cols, s_val[fit_idx], val_y[fit_idx])
        q = LR.family_a_probs(val["z"][select_idx], val["r"][select_idx], cols, fit["beta"], s_val[select_idx])
        arms[f"A_grid{GRID}_greedy_L{L}"] = {
            "layers": sel_idx_list[:L], "cols": cols, "beta": fit["beta"], "inner_select_nll": LR.nll(q, val_y[select_idx]),
            "inner_fit_nll_at_beta0": fit["objective_value_at_zero"], "dims_grid": [banks[n].dim for n in sel_names[:L]],
        }
    out_root = os.path.join(args.out_dir, f"checkpoint_seed{args.seed}")
    os.makedirs(out_root, exist_ok=True)
    json.dump({"pooling": f"grid{GRID}", "selection_reused_from": os.path.join(seed_dir, "frozen_state.json"),
               "main_state_hash": state["state_hash"], "arms": arms}, open(os.path.join(out_root, "frozen_state.json"), "w"), indent=1)

    cells = [("clean", None, None)] + [(f"{c}_s{s}", c, s) for c in P.CORRUPTIONS for s in P.SEVERITIES]
    for cell, corr, sev in cells:
        cdir = os.path.join(out_root, cell)
        os.makedirs(cdir, exist_ok=True)
        if os.path.exists(os.path.join(cdir, "cell_metrics.json")):
            continue
        if corr is None:
            images, labels = test_raw, test_y
        else:
            images, labels = P.load_corrupted(args.cifar_c_dir, corr, sev, args.batch_size)
        arr = run_split(model, tap, images, args.batch_size, device, banks=banks, gap_banks=gap_banks,
                        sel_names=sel_names, native_names=native_names)
        s_x = LR.s_dac(arr["s_glob"], w, w0)
        base = np_softmax(arr["z"].astype(np.float64))
        probs = {"base_model": base, "native_dac": np_softmax(arr["z"].astype(np.float64) / s_x[:, None])}
        for name, a in arms.items():
            probs[name] = LR.family_a_probs(arr["z"], arr["r"], a["cols"], a["beta"], s_x)
        rec: Dict[str, Any] = {"cell": cell, "seed": args.seed, "n": int(len(labels)), "pooling": f"grid{GRID}", "arms": {}}
        for name, p in probs.items():
            m = evaluate_all(p, labels)
            e = {"metrics": {k: (None if v is None else float(v)) for k, v in m.items()},
                 "delta_accuracy": float(m["accuracy"] - (base.argmax(1) == labels).mean())}
            if name != "base_model":
                e["flips"] = LR.flip_decomposition(base, p, labels)
            rec["arms"][name] = e
        json.dump(rec, open(os.path.join(cdir, "cell_metrics.json"), "w"), indent=1, default=float)
        print(f"[eval] {cell}", flush=True)


if __name__ == "__main__":
    main()
