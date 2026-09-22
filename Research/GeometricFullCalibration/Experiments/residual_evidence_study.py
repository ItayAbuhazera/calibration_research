"""
Residual-evidence study runner (docs/residual_evidence_study_spec.md; frozen before any
Stage-2 result). One invocation = one checkpoint seed, two strictly ordered phases:

  FIT       clean data only: reference banks (train), anchor, feature maps, residual fits,
            both clean-selection policies -> hashed frozen state written to disk.
  EVALUATE  clean test + the requested corruption cells. `evaluate_cell` receives the frozen
            state and evaluation arrays only; it has no path to any fitting function.

Evidence sources at ONE layer, `layer3.22` (post-activation block output):
  G  GAP -> fixed Gaussian map to 100-d          S  adaptive 2x2 pool, flattened -> fixed map
  DG class-wise K_c-th-neighbour radii, GAP bank DS same from the 2x2 bank
  O  fixed random ReLU features of standardized logits
  DL class-wise K_c-th-neighbour radii in (centered, L2-normalized) logit space, same labeled bank
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Calibrators.density_aware_calibration import DensityAwareCalibrator  # noqa: E402
from Calibrators.full_vector_dac import (  # noqa: E402
    ClassConditionalKNN, centered_logit_representation, normalized_euclidean)
from Calibrators import layer_readouts as LR  # noqa: E402
from Calibrators import residual_readout as RR  # noqa: E402
from Calibrators.temperature_scaling import TemperatureScaling  # noqa: E402
from Calibrators.vector_scaling import VectorScaling  # noqa: E402
from utils import preprocessing_protocol as pp  # noqa: E402
from utils.calibration_utils import get_all_data_as_numpy, load_cifar_c_loader  # noqa: E402
from utils.decision_audit import make_inner_validation_split  # noqa: E402
from utils.model_utils import construct_model_path, load_trained_model  # noqa: E402
from utils.unified_metrics import evaluate_all  # noqa: E402

VERSION = "residual_study_v1"
LAYER = "layer3.22"
K_C = 5          # inherited from the closed FV-DAC pilot (verified in the frozen spec)
K_DAC = 200
NATIVE_LAYERS = ("conv1", "layer1", "layer2", "layer3", "layer4")
NUM_CLASSES = 100
INNER_SEED = 123
DEV_CORRUPTIONS = ("gaussian_noise", "defocus_blur", "fog", "jpeg_compression")
ALL_CORRUPTIONS = ("brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_noise",
                   "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "shot_noise", "snow",
                   "zoom_blur")
SEVERITIES_DEV = (1, 3, 5)
SEVERITIES_ALL = (1, 2, 3, 4, 5)
CKPT_METHOD, DATASET, MODEL = "baseline_cross_entropy", "cifar100", "resnet101"


def lam_tag(lam: float) -> str:
    return "inf" if not np.isfinite(lam) else f"{lam:g}"


def arm_name(family: str, lam: float) -> str:
    return "anchor" if family == "zero" else f"{family}_l{lam_tag(lam)}"


# ==========================================================================
# data
# ==========================================================================


def orig_split(seed: int, n: int = 50000, valid_size: float = 0.1):
    idx = list(range(n))
    split = int(np.floor(valid_size * n))
    np.random.seed(seed)
    np.random.shuffle(idx)
    return np.asarray(idx[split:]), np.asarray(idx[:split])


def materialize(seed: int, data_root: str = "./data"):
    """Deterministic corrected-protocol arrays for a seed: train/val in the ORDER of the
    original-index partition (data/cifar100.py's shuffle), test in natural order."""
    from torchvision import datasets
    from torch.utils.data import DataLoader, Subset

    tf = pp.eval_transform(DATASET, pp.PROTOCOL_CORRECTED)
    tr_ds = datasets.CIFAR100(root=data_root, train=True, download=False, transform=tf)
    te_ds = datasets.CIFAR100(root=data_root, train=False, download=False, transform=tf)
    train_idx, val_idx = orig_split(seed)

    def load(ds, idxs):
        dl = DataLoader(Subset(ds, [int(i) for i in idxs]), batch_size=500, shuffle=False, num_workers=4)
        xs, ys = [], []
        for x, y in dl:
            xs.append(x.numpy()); ys.append(y.numpy())
        return np.concatenate(xs), np.concatenate(ys)

    tr = load(tr_ds, train_idx)
    va = load(tr_ds, val_idx)
    te = load(te_ds, np.arange(10000))
    return {"train": tr, "val": va, "test": te, "train_idx": train_idx, "val_idx": val_idx}


def load_benchmark_arrays(root: str, seed: int):
    sp = os.path.join(root, "evaluation", f"checkpoint_seed{seed}", "clean", "intermediates", "splits")
    ld = lambda k: np.load(os.path.join(sp, k), mmap_mode="r") if k.endswith("raw.npy") else np.load(os.path.join(sp, k))  # noqa: E731
    return {"train": (ld("train_raw.npy"), ld("train_labels.npy")), "val": (ld("val_raw.npy"), ld("val_labels.npy")),
            "test": (ld("test_raw.npy"), ld("test_labels.npy"))}


# ==========================================================================
# model taps and neighbour search
# ==========================================================================


class Taps:
    def __init__(self, model):
        mods = dict(model.named_modules())
        self.gap: Dict[str, torch.Tensor] = {}
        self.grid: Optional[torch.Tensor] = None
        self.h = []
        for n in NATIVE_LAYERS:
            self.h.append(mods[n].register_forward_hook(self._gap(n)))
        self.h.append(mods[LAYER].register_forward_hook(self._layer()))

    def _gap(self, n):
        def hook(_m, _i, out):
            f = out.mean(dim=(2, 3)) if out.ndim == 4 else out
            self.gap[n] = F.normalize(f.float(), p=2, dim=-1)
        return hook

    def _layer(self):
        def hook(_m, _i, out):
            self.gap[LAYER] = F.normalize(out.mean(dim=(2, 3)).float(), p=2, dim=-1)
            self.grid = F.normalize(F.adaptive_avg_pool2d(out, (2, 2)).flatten(1).float(), p=2, dim=-1)
        return hook


class ClassRadii:
    """Class-wise K_c-th-neighbour Euclidean distance (audited operator) against one labeled bank."""

    def __init__(self, bank: torch.Tensor, labels: np.ndarray, device):
        self.bank = bank.to(device).float().contiguous()
        self.op = ClassConditionalKNN(self.bank, labels, NUM_CLASSES, device)
        self.op.validate_kc([K_C])
        self.pos = torch.tensor([K_C - 1], device=device)

    def __call__(self, q: torch.Tensor) -> np.ndarray:
        dist = normalized_euclidean(q, self.bank)
        return self.op._gather(dist, K_C, self.pos)[:, :, 0]


class Engine:
    def __init__(self, model, device, batch_size):
        self.model, self.device, self.bs = model, device, batch_size
        self.taps = Taps(model)
        self.r_gap = self.r_grid = self.r_logit = None
        self.native_banks: Dict[str, torch.Tensor] = {}
        self.timers = {"DG": 0.0, "DS": 0.0, "DL": 0.0, "native": 0.0, "n": 0}

    @torch.no_grad()
    def forward(self, x):
        self.taps.gap.clear()
        z = self.model(x).float()
        return z, self.taps.gap, self.taps.grid

    @torch.no_grad()
    def train_pass(self, images: np.ndarray):
        zs, gap, grid = [], [], []
        nat = {n: [] for n in NATIVE_LAYERS}
        for s in range(0, images.shape[0], self.bs):
            x = torch.from_numpy(np.ascontiguousarray(images[s:s + self.bs])).to(self.device).float()
            z, g, gr = self.forward(x)
            zs.append(z.cpu().numpy())
            gap.append(g[LAYER]); grid.append(gr)
            for n in NATIVE_LAYERS:
                nat[n].append(g[n])
        return np.concatenate(zs), torch.cat(gap), torch.cat(grid), {n: torch.cat(v) for n, v in nat.items()}

    @torch.no_grad()
    def extract(self, images: np.ndarray, *, time_it: bool = False) -> Dict[str, np.ndarray]:
        n = images.shape[0]
        out = {"z": np.empty((n, NUM_CLASSES), np.float32),
               "gap": np.empty((n, 1024), np.float32), "grid": np.empty((n, 4096), np.float32),
               "DG": np.empty((n, NUM_CLASSES), np.float32), "DS": np.empty_like(np.empty((n, NUM_CLASSES), np.float32)),
               "DL": np.empty((n, NUM_CLASSES), np.float32), "s_glob": np.empty((n, 5), np.float32)}
        def sync():
            if time_it:
                torch.cuda.synchronize()
            return time.perf_counter()
        for s in range(0, n, self.bs):
            x = torch.from_numpy(np.ascontiguousarray(images[s:s + self.bs])).to(self.device).float()
            e = s + x.shape[0]
            z, g, gr = self.forward(x)
            out["z"][s:e] = z.cpu().numpy()
            out["gap"][s:e] = g[LAYER].cpu().numpy()
            out["grid"][s:e] = gr.cpu().numpy()
            t0 = sync(); out["DG"][s:e] = self.r_gap(g[LAYER]); t1 = sync()
            out["DS"][s:e] = self.r_grid(gr); t2 = sync()
            zc = torch.from_numpy(centered_logit_representation(z.cpu().numpy())).to(self.device)
            out["DL"][s:e] = self.r_logit(zc); t3 = sync()
            for j, nme in enumerate(NATIVE_LAYERS):
                d = normalized_euclidean(g[nme], self.native_banks[nme])
                out["s_glob"][s:e, j] = torch.kthvalue(d, K_DAC, dim=1).values.cpu().numpy()
            if time_it:
                self.timers["DG"] += t1 - t0; self.timers["DS"] += t2 - t1; self.timers["DL"] += t3 - t2
                self.timers["n"] += e - s
        return out


# ==========================================================================
# FIT
# ==========================================================================


def sdac(s_glob, w, w0):
    return LR.s_dac(s_glob, w, w0)


def evidence_arrays(ev: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Raw evidence per family (before phi)."""
    return {"G": ev["gap"], "S": ev["grid"], "DG": ev["DG"], "DS": ev["DS"], "O": ev["z"], "DL": ev["DL"]}


def fit_phase(pilot: Engine, data, seed: int, native_source: Dict[str, Any], fitted_native_dir: Optional[str]):
    dev = pilot.device
    train_x, train_y = data["train"]
    val_x, val_y = data["val"]
    fit_idx, sel_idx = make_inner_validation_split(val_y, select_fraction=0.5, seed=INNER_SEED)
    assert set(fit_idx).isdisjoint(sel_idx)
    t0 = time.time()
    tz, gap_bank, grid_bank, nat = pilot.train_pass(train_x)
    pilot.r_gap = ClassRadii(gap_bank, train_y, dev)
    pilot.r_grid = ClassRadii(grid_bank, train_y, dev)
    pilot.r_logit = ClassRadii(torch.from_numpy(centered_logit_representation(tz)), train_y, dev)
    pilot.native_banks = {k: v.contiguous() for k, v in nat.items()}

    # native DAC (five-source benchmark variant): frozen benchmark pickle for the development seeds,
    # in-pipeline refit (same class, full validation split, as the benchmark did) for new checkpoints
    val_ev = pilot.extract(val_x)
    if fitted_native_dir:
        native = pickle.load(open(os.path.join(fitted_native_dir, "native_dac.pkl"), "rb"))
        native_src = "benchmark_frozen_pickle"
    else:
        native = DensityAwareCalibrator(k=K_DAC, use_gpu=True)
        feats_tr = [nat[n].cpu().numpy() for n in NATIVE_LAYERS]
        # validation features for the refit: recompute in a dedicated pass (GAP native layers)
        vf = {n: [] for n in NATIVE_LAYERS}
        for s in range(0, val_x.shape[0], pilot.bs):
            x = torch.from_numpy(np.ascontiguousarray(val_x[s:s + pilot.bs])).to(dev).float()
            _, g, _ = pilot.forward(x)
            for n in NATIVE_LAYERS: vf[n].append(g[n].cpu().numpy())
        native.fit(feats_tr, [np.concatenate(vf[n]) for n in NATIVE_LAYERS], val_ev["z"], val_y)
        native_src = "in_pipeline_refit_full_validation"
    assert native.num_layers == 5 and native.k == K_DAC
    nw, nb = np.asarray(native.weights[:-1], np.float64), float(native.weights[-1])

    z_fit, z_sel = val_ev["z"][fit_idx].astype(np.float64), val_ev["z"][sel_idx].astype(np.float64)
    y_fit, y_sel = val_y[fit_idx], val_y[sel_idx]

    # ---- output-only anchor ------------------------------------------------
    ts = TemperatureScaling(); ts.fit(z_fit, y_fit)
    vs = VectorScaling(); vs.fit(z_fit, y_fit)
    def vs_logits(z):
        return np.log(np.clip(np.asarray(vs.calibrate(np.asarray(z, dtype=np.float64))), 1e-300, 1.0))
    cands: List[Dict[str, Any]] = [{"kind": "identity", "lambda": float("inf")}, {"kind": "vector_scaling", "lambda": float("nan")}]
    anchor_fits = {}
    for lam in RR.LAMBDAS:
        anchor_fits[lam] = RR.fit_matrix_scaling(z_fit, y_fit, lam)
        cands.append({"kind": "matrix_scaling", "lambda": lam})
    def apply_anchor(c, z):
        if c["kind"] == "identity": return np.asarray(z, dtype=np.float64)
        if c["kind"] == "vector_scaling": return vs_logits(z)
        f = anchor_fits[c["lambda"]]
        return RR.apply_matrix_scaling(z, f["A"], f["b"])
    for c in cands:
        lg = apply_anchor(c, z_sel)
        c["select_nll"] = RR.nll_np(lg, y_sel); c["select_acc"] = float(np.mean(lg.argmax(1) == y_sel))
        c["fit_nll"] = RR.nll_np(apply_anchor(c, z_fit), y_fit)
        if c["kind"] == "matrix_scaling":
            f = anchor_fits[c["lambda"]]; c.update({"converged": f["converged"], "n_iter": f["n_iter"], "grad_inf": f["grad_inf"], "continued": f["continued"]})
    ai = RR.select_anchor(cands)
    anchor = cands[ai]
    anchor_fit_logits, anchor_sel_logits = apply_anchor(anchor, z_fit), apply_anchor(anchor, z_sel)

    # ---- evidence maps, residual fits -----------------------------------------
    ev_val = evidence_arrays(val_ev)
    maps, arms, W_store = {}, [], {}
    distort, ranks = {}, {}
    for fam in RR.ALL_FAMILIES:
        raw_fit = ev_val[fam][fit_idx].astype(np.float64)
        fm = RR.make_feature_map(fam, raw_fit)
        maps[fam] = fm
        phi_fit, phi_sel = fm(raw_fit), fm(ev_val[fam][sel_idx].astype(np.float64))
        if fam in ("G", "S"):
            distort[fam] = RR.projection_distortion(raw_fit, fm.proj)
        ranks[fam] = {"raw_dim": int(raw_fit.shape[1]), "phi_fit": RR.effective_rank(phi_fit)}
        for lam in RR.LAMBDAS:
            f = RR.fit_residual(anchor_fit_logits, phi_fit, y_fit, lam)
            lg_fit = anchor_fit_logits + RR.correction_logits(phi_fit, f["W"])
            lg_sel = anchor_sel_logits + RR.correction_logits(phi_sel, f["W"])
            rec = {"family": fam, "lambda": lam, "name": arm_name(fam, lam), "fit_nll": RR.nll_np(lg_fit, y_fit),
                   "fit_acc": float(np.mean(lg_fit.argmax(1) == y_fit)),
                   "select_nll": RR.nll_np(lg_sel, y_sel), "select_acc": float(np.mean(lg_sel.argmax(1) == y_sel)),
                   "converged": f["converged"], "continued": f["continued"], "n_iter": f["n_iter"], "grad_inf": f["grad_inf"],
                   "W_fro": float(np.linalg.norm(f["W"])), "n_params": int(f["W"].size)}
            arms.append(rec); W_store[rec["name"]] = f["W"]
    anchor_nll, anchor_acc = anchor["select_nll"], anchor["select_acc"]
    zero = RR.zero_candidate(anchor_nll, anchor_acc)
    def pool_of(fams, extra=True):
        return [a for a in arms if a["family"] in fams] + ([zero] if extra else [])
    sel = {}
    for pname, fams in (("hidden", RR.HIDDEN_FAMILIES), ("output", RR.OUTPUT_FAMILIES)):
        pool = pool_of(fams)
        i_n, i_d = RR.nll_policy(pool), RR.decision_policy(pool, anchor_nll)
        sel[f"{pname}_nll"] = arm_name(pool[i_n]["family"], pool[i_n]["lambda"])
        sel[f"{pname}_decision"] = arm_name(pool[i_d]["family"], pool[i_d]["lambda"])
    fam_sel = {}
    for fam in RR.ALL_FAMILIES:
        pool = pool_of((fam,))
        fam_sel[fam] = {"nll": arm_name(pool[RR.nll_policy(pool)]["family"], pool[RR.nll_policy(pool)]["lambda"]),
                        "decision": arm_name(pool[RR.decision_policy(pool, anchor_nll)]["family"],
                                             pool[RR.decision_policy(pool, anchor_nll)]["lambda"])}
    # fixed-edge bins (clean-selection data): base logit top-2 margin; correction strength of the primary procedure
    def margin(z):
        t = np.sort(np.asarray(z, dtype=np.float64), axis=1)
        return t[:, -1] - t[:, -2]
    m_edges = RR.quantile_edges(margin(z_sel))
    prim = sel["hidden_decision"]
    fam_p = prim.split("_l")[0]
    if prim != "anchor":
        cs_sel = np.abs(RR.correction_logits(maps[fam_p](ev_val[fam_p][sel_idx].astype(np.float64)), W_store[prim])).max(1)
        s_edges = RR.quantile_edges(cs_sel)
    else:
        s_edges = None

    state = {
        "version": VERSION, "seed": seed, "layer": LAYER, "k_c": K_C, "k_dac": K_DAC,
        "preprocessing": pp.preprocessing_stamp(DATASET, pp.PROTOCOL_CORRECTED),
        "native_dac": {"layers": list(NATIVE_LAYERS), "weights": nw.tolist(), "intercept": nb, "source": native_src,
                       "in_pipeline_or_pickle_note": "five-source benchmark variant; logits NOT a source"},
        "inner_split": {"seed": INNER_SEED, "n_fit": int(len(fit_idx)), "n_select": int(len(sel_idx)),
                        "fit_rows_hash": LR.state_hash({}, [fit_idx]), "select_rows_hash": LR.state_hash({}, [sel_idx])},
        "bank": {"size": int(len(train_y)), "labels_hash": LR.state_hash({}, [train_y]),
                 "min_class_count": int(np.bincount(train_y).min())},
        "anchor": {"chosen": anchor["kind"] + ("" if anchor["kind"] != "matrix_scaling" else f"_l{lam_tag(anchor['lambda'])}"),
                   "candidates": [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in c.items()} for c in cands]},
        "arms": arms, "selection": sel, "family_level_selection": fam_sel,
        "anchor_select_nll": anchor_nll, "anchor_select_acc": anchor_acc,
        "primary_procedure": "hidden_decision", "lambdas": list(RR.LAMBDAS), "nll_tolerance": RR.NLL_TOLERANCE,
        "map_seeds": {"G": RR.SEED_MAP_G, "S": RR.SEED_MAP_S, "O": RR.SEED_MAP_O},
        "projection_distortion": distort, "feature_rank_summary": ranks,
        "jl_sufficient_dim_n45000": {str(e): RR.jl_required_dim(45000, e) for e in (0.1, 0.2, 0.3, 0.5)},
        "margin_bin_edges": m_edges.tolist(), "strength_bin_edges": None if s_edges is None else s_edges.tolist(),
        "all_converged": bool(all(a["converged"] for a in arms) and all(c.get("converged", True) for c in cands)),
        "fit_count": {"residual": len(arms), "anchor_matrix_scaling": len(RR.LAMBDAS), "ts": 1, "vs": 1,
                      "native_dac": 0 if fitted_native_dir else 1},
    }
    state["state_hash"] = LR.state_hash({k: v for k, v in state.items() if k != "state_hash"},
                                        [W_store[k] for k in sorted(W_store)])
    frozen = {"maps": maps, "W": W_store, "anchor_cand": anchor, "apply_anchor": apply_anchor, "ts": ts, "vs": vs,
              "vs_logits": vs_logits, "nw": nw, "nb": nb, "m_edges": m_edges, "s_edges": s_edges, "sel": sel}
    print(f"[fit] seed {seed}: {time.time() - t0:.0f}s anchor={state['anchor']['chosen']} sel={sel}", flush=True)
    return state, frozen, {"fit_idx": fit_idx, "sel_idx": sel_idx}


# ==========================================================================
# EVALUATE: frozen state + evaluation arrays only. No fitting function is referenced below.
# ==========================================================================


def eval_logits(name: str, ev: Dict[str, np.ndarray], frozen: Dict[str, Any], anchor_logits: np.ndarray) -> np.ndarray:
    if name == "anchor":
        return anchor_logits
    fam = name.split("_l")[0]
    raw = evidence_arrays(ev)[fam].astype(np.float64)
    phi = frozen["maps"][fam](raw)
    return anchor_logits + RR.correction_logits(phi, frozen["W"][name])


def evaluate_cell(ev: Dict[str, np.ndarray], labels: np.ndarray, state: Dict[str, Any], frozen: Dict[str, Any]):
    z = ev["z"].astype(np.float64)
    base = RR.softmax_np(z)
    anchor_logits = frozen["apply_anchor"](frozen["anchor_cand"], z)
    s_x = LR.s_dac(ev["s_glob"], frozen["nw"], frozen["nb"])
    probs = {"base_model": base, "temperature_scaling": frozen["ts"].calibrate(z),
             "vector_scaling": np.asarray(frozen["vs"].calibrate(z)),
             "native_dac": RR.softmax_np(z / s_x[:, None])}
    logit_of = {"anchor": anchor_logits}
    probs["anchor"] = RR.softmax_np(anchor_logits)
    for a in state["arms"]:
        lg = eval_logits(a["name"], ev, frozen, anchor_logits)
        logit_of[a["name"]] = lg
        probs[a["name"]] = RR.softmax_np(lg)
    for pol, target in state["selection"].items():
        probs[f"proc::{pol}"] = probs[target]
    assert np.abs(probs["anchor"] - RR.softmax_np(anchor_logits)).max() == 0.0
    metrics, flips, flips_vs_anchor = {}, {}, {}
    base_pred = base.argmax(1)
    for nme, p in probs.items():
        m = evaluate_all(p, labels)
        metrics[nme] = {k: (None if v is None else float(v)) for k, v in m.items()}
        if nme != "base_model":
            flips[nme] = LR.flip_decomposition(base, p, labels)
        if nme not in ("base_model", "anchor"):
            flips_vs_anchor[nme] = LR.flip_decomposition(probs["anchor"], p, labels)
    # binned signed utility for the procedures (edges fixed on clean-selection data)
    bins = {}
    zt = np.sort(z, axis=1)
    mvals = zt[:, -1] - zt[:, -2]
    for pol in state["selection"]:
        p = probs[f"proc::{pol}"]
        bins[pol] = {"by_base_margin": RR.bin_utility(mvals, np.asarray(state["margin_bin_edges"]), base_pred, p.argmax(1), labels)}
    prim = state["selection"][state["primary_procedure"]]
    if prim != "anchor" and state["strength_bin_edges"] is not None:
        corr = logit_of[prim] - anchor_logits
        strength = np.abs(corr).max(1)
        bins[state["primary_procedure"]]["by_correction_strength"] = RR.bin_utility(
            strength, np.asarray(state["strength_bin_edges"]), base_pred, probs[f"proc::{state['primary_procedure']}"].argmax(1), labels)
    return probs, metrics, flips, flips_vs_anchor, bins, logit_of


def per_sample(probs, labels, ev, logit_of, state):
    rec = {"labels": labels.astype(np.int16), "z": ev["z"].astype(np.float32)}
    for nme, p in probs.items():
        rec[f"pred__{nme}"] = p.argmax(1).astype(np.int16)
        rec[f"gtrank__{nme}"] = LR.gt_rank(p, labels).astype(np.int16)  # from float64 probabilities
        if nme.startswith("proc::") or nme in ("anchor", "native_dac"):
            t = np.sort(p, axis=1)
            rec[f"conf__{nme}"] = p.max(1).astype(np.float32)
            rec[f"top2margin__{nme}"] = (t[:, -1] - t[:, -2]).astype(np.float32)
    prim = state["selection"][state["primary_procedure"]]
    if prim != "anchor":
        rec["strength__primary"] = np.abs(logit_of[prim] - logit_of["anchor"]).max(1).astype(np.float32)
    return rec


def load_corrupted(cifar_c_dir, corruption, severity, batch_size):
    tf = pp.eval_transform(DATASET, pp.PROTOCOL_CORRECTED, corruption=True)
    return get_all_data_as_numpy(load_cifar_c_loader(DATASET, corruption, severity, tf, batch_size, cifar_c_dir=cifar_c_dir))


def cells_for(mode: str):
    if mode == "dev12":
        cs = [(f"{c}_s{s}", c, s) for c in DEV_CORRUPTIONS for s in SEVERITIES_DEV]
    elif mode == "full75":
        cs = [(f"{c}_s{s}", c, s) for c in ALL_CORRUPTIONS for s in SEVERITIES_ALL]
    else:
        raise ValueError(mode)
    return [("clean", None, None)] + cs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--role", choices=["dev", "confirm"], required=True)
    ap.add_argument("--conditions", choices=["dev12", "full75"], required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--benchmark_root", required=True)
    ap.add_argument("--cifar_c_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_cells", type=int, default=None, help="smoke only")
    ap.add_argument("--smoke_subset", type=int, default=None, help="smoke only: truncate val/test")
    ap.add_argument("--spec_hash", default=None, help="sha256 of the frozen spec; recorded, and checked when given")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA unavailable")
    device = torch.device(args.device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.spec_hash:
        spec = Path(__file__).resolve().parents[1] / "docs" / "residual_evidence_study_spec.md"
        cur = hashlib.sha256(spec.read_bytes()).hexdigest()
        # the spec's results section is appended after freezing; the frozen part ends at the marker line
        frozen_part = spec.read_text().split("<!-- END FROZEN -->")[0].encode()
        if hashlib.sha256(frozen_part).hexdigest() != args.spec_hash:
            raise SystemExit("frozen spec section does not match --spec_hash; refusing to run")

    out_root = os.path.join(args.out_dir, f"checkpoint_seed{args.seed}")
    os.makedirs(out_root, exist_ok=True)
    if args.role == "dev":
        b = load_benchmark_arrays(args.benchmark_root, args.seed)
        for d in (os.path.join(args.benchmark_root, "evaluation", f"checkpoint_seed{args.seed}", "clean", "intermediates"),
                  os.path.join(args.benchmark_root, "fitted_method", f"checkpoint_seed{args.seed}")):
            if pp.read_protocol(d) != pp.PROTOCOL_CORRECTED:
                raise SystemExit(f"{d} is not corrected-protocol")
        sub = args.smoke_subset
        data = {"train": b["train"], "val": tuple(a[:sub] if sub else a for a in b["val"]),
                "test": tuple(a[:sub] if sub else a for a in b["test"])}
        fitted_dir = os.path.join(args.benchmark_root, "fitted_method", f"checkpoint_seed{args.seed}")
        data_source = "benchmark_materialized_arrays"
    else:
        mat = materialize(args.seed, args.data_root)
        data = {"train": mat["train"], "val": mat["val"], "test": mat["test"]}
        fitted_dir = None
        data_source = "self_materialized_deterministic_order"

    ckpt = construct_model_path(args.results_dir, CKPT_METHOD, DATASET, MODEL, args.seed)
    model = load_trained_model(ckpt, MODEL, NUM_CLASSES, device, dataset=DATASET).eval()
    eng = Engine(model, device, args.batch_size)
    t0 = time.time()
    state, frozen, idx = fit_phase(eng, data, args.seed, {}, fitted_dir)
    state["data_source"] = data_source
    state["checkpoint"] = ckpt
    state["spec_hash_argument"] = args.spec_hash
    state["role"] = args.role
    with open(os.path.join(out_root, "frozen_state.json"), "w") as f:
        json.dump(state, f, indent=1, default=float)
    np.savez(os.path.join(out_root, "residual_weights.npz"), **{k: v for k, v in frozen["W"].items()})
    fit_seconds = time.time() - t0
    print(f"[fit] done {fit_seconds:.0f}s hash {state['state_hash']} converged={state['all_converged']}", flush=True)

    for cell, corr, sev in cells_for(args.conditions)[: (args.max_cells or None)]:
        cdir = os.path.join(out_root, cell)
        os.makedirs(cdir, exist_ok=True)
        if os.path.exists(os.path.join(cdir, "cell_metrics.json")):
            continue
        t1 = time.time()
        if corr is None:
            images, labels = data["test"]
        else:
            images, labels = load_corrupted(args.cifar_c_dir, corr, sev, args.batch_size)
            if args.smoke_subset:
                images, labels = images[:args.smoke_subset], labels[:args.smoke_subset]
        for k in ("DG", "DS", "DL"):
            eng.timers[k] = 0.0
        eng.timers["n"] = 0
        ev = eng.extract(images, time_it=(corr is None))
        probs, metrics, flips, flips_va, bins, logit_of = evaluate_cell(ev, np.asarray(labels), state, frozen)
        rec = {"cell": cell, "corruption": corr, "severity": sev, "seed": args.seed, "n": int(len(labels)),
               "state_hash": state["state_hash"], "evaluation_only": True, "preprocessing_protocol": pp.PROTOCOL_CORRECTED,
               "metrics": metrics, "flips_vs_base": flips, "flips_vs_anchor": flips_va, "bins": bins,
               "wall_seconds": time.time() - t1}
        if corr is None:
            rec["bank_query_seconds_per_10k_queries"] = {k: eng.timers[k] * 10000 / max(eng.timers["n"], 1) for k in ("DG", "DS", "DL")}
            rec["storage_bytes_banks"] = {"DG": 45000 * 1024 * 4, "DS": 45000 * 4096 * 4, "DL": 45000 * 100 * 4}
        np.savez(os.path.join(cdir, "per_sample.npz"), **per_sample(probs, np.asarray(labels), ev, logit_of, state))
        json.dump(rec, open(os.path.join(cdir, "cell_metrics.json"), "w"), default=float)
        print(f"[eval] {cell}: {time.time() - t1:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
