"""
Layer-selection pilot (L in {1, 4, 6, 8}) -- runner.

Specification: docs/layer_selection_pilot_spec.md (frozen before any corrected
corruption result). Readout / selection math: Calibrators/layer_readouts.py.

One invocation = one checkpoint seed, in two strictly ordered phases inside one
process:

  FIT       clean data only. Builds the reference banks from the benchmark's
            materialized train split, fits Family-A betas, Family-B probes,
            runs both greedy selections on the inner-SELECT role, and writes a
            hashed frozen state. Corruption data is never loaded here.
  EVALUATE  clean test + the 12 CIFAR-100-C cells. Reads ONLY the frozen state
            (`evaluate_cell` has no path to any fitting function; a test asserts
            this from the source).

All inputs are produced under the corrected preprocessing protocol
(`utils.preprocessing_protocol`); a legacy-protocol benchmark directory is
rejected.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Calibrators.density_aware_calibration import DensityAwareCalibrator  # noqa: E402,F401  (pickle)
from Calibrators.full_vector_dac import (  # noqa: E402
    ClassConditionalKNN,
    centered_logit_representation,
    np_softmax,
    normalized_euclidean,
)
from Calibrators import layer_readouts as LR  # noqa: E402
from utils import preprocessing_protocol as pp  # noqa: E402
from utils.calibration_utils import get_all_data_as_numpy, load_cifar_c_loader  # noqa: E402
from utils.decision_audit import make_inner_validation_split  # noqa: E402
from utils.model_utils import construct_model_path, load_trained_model  # noqa: E402
from utils.unified_metrics import evaluate_all  # noqa: E402

PILOT_VERSION = "layer_pilot_v1"
SPLIT_PLAN = "layer_pilot_split_v1"
INNER_SEED = 123
CORRUPTIONS = ("gaussian_noise", "defocus_blur", "fog", "jpeg_compression")
SEVERITIES = (1, 3, 5)
NUM_CLASSES = 100
DATASET = "cifar100"
MODEL = "resnet101"
METHOD = "baseline_cross_entropy"


# ==========================================================================
# Model feature access
# ==========================================================================


class FeatureTap:
    """Forward hooks on the 12 candidate blocks + native-DAC `conv1`.

    Every tap is global-average-pooled over (H, W) and L2-normalized, the exact
    operation of native DAC's hooks / LayerKNNScorer._preprocess."""

    def __init__(self, model: torch.nn.Module) -> None:
        mods = dict(model.named_modules())
        self.names = list(LR.CANDIDATE_NAMES) + ["conv1"]
        missing = [n for n in self.names if n not in mods]
        if missing:
            raise KeyError(f"modules not found in model: {missing}")
        self.store: Dict[str, torch.Tensor] = {}
        self.handles = []
        for n in self.names:
            self.handles.append(mods[n].register_forward_hook(self._mk(n)))

    def _mk(self, name: str):
        def hook(_m, _i, out):
            f = out.mean(dim=(2, 3)) if out.ndim == 4 else out
            self.store[name] = F.normalize(f.float(), p=2, dim=-1)

        return hook

    def close(self) -> None:
        for h in self.handles:
            h.remove()


@torch.no_grad()
def forward_pass(model, tap: FeatureTap, x: torch.Tensor):
    tap.store.clear()
    z = model(x).float()
    return z, {k: v for k, v in tap.store.items()}


def iter_batches(arr: np.ndarray, batch_size: int, device: torch.device):
    for s in range(0, arr.shape[0], batch_size):
        yield s, torch.from_numpy(np.ascontiguousarray(arr[s : s + batch_size])).to(device).float()


# ==========================================================================
# Reference banks and neighbour search
# ==========================================================================


class LayerBank:
    """One source's reference bank (unit vectors, GPU) with true-label and
    permuted-label class-conditional operators sharing ONE tensor."""

    def __init__(self, feats: torch.Tensor, labels: np.ndarray, perm_labels: np.ndarray, device):
        self.bank = feats.to(device).float().contiguous()
        self.op_true = ClassConditionalKNN(self.bank, labels, NUM_CLASSES, device)
        self.op_perm = ClassConditionalKNN(self.bank, perm_labels, NUM_CLASSES, device)
        assert self.op_true.bank.data_ptr() == self.op_perm.bank.data_ptr() == self.bank.data_ptr()
        self.device = device
        self.positions = torch.tensor([LR.K_C - 1], device=device)
        self.op_true.validate_kc([LR.K_C])

    @property
    def dim(self) -> int:
        return int(self.bank.shape[1])

    def query(self, q: torch.Tensor, global_k: Optional[int] = None):
        dist = normalized_euclidean(q, self.bank)
        r_t = self.op_true._gather(dist, LR.K_C, self.positions)[:, :, 0]
        r_p = self.op_perm._gather(dist, LR.K_C, self.positions)[:, :, 0]
        g = None
        if global_k is not None:
            g = torch.kthvalue(dist, min(global_k, dist.shape[1]), dim=1).values.cpu().numpy()
        return r_t, r_p, g


class Pilot:
    """Everything the FIT phase builds and the EVALUATE phase reads."""

    def __init__(self, model, device: torch.device, batch_size: int):
        self.model, self.device, self.batch_size = model, device, batch_size
        self.tap = FeatureTap(model)
        self.banks: Dict[int, LayerBank] = {}
        self.conv1_bank: Optional[torch.Tensor] = None
        self.logit_bank: Optional[LayerBank] = None
        self.native_w: Optional[np.ndarray] = None
        self.native_b: Optional[float] = None
        self.probes: List[LR.Probe] = []
        self.logit_probe: Optional[LR.Probe] = None
        self.timers: Dict[str, float] = {}
        self.timing_on = False

    # ------------------------------------------------------------ extraction
    def extract(
        self,
        images: np.ndarray,
        *,
        with_distances: bool,
        with_probes: bool,
        keep_feats: bool = False,
    ) -> Dict[str, Any]:
        n = images.shape[0]
        z_all = np.empty((n, NUM_CLASSES), dtype=np.float32)
        out: Dict[str, Any] = {"z": z_all}
        if with_distances:
            out["r"] = np.empty((n, LR.N_CANDIDATES, NUM_CLASSES), dtype=np.float32)
            out["r_perm"] = np.empty_like(out["r"])
            out["r_logit"] = np.empty((n, NUM_CLASSES), dtype=np.float32)
            out["s_glob"] = np.empty((n, len(LR.NATIVE_DAC_LAYERS)), dtype=np.float32)
        if with_probes:
            out["probe_logits"] = np.empty((n, LR.N_CANDIDATES, NUM_CLASSES), dtype=np.float32)
            out["logit_probe_logits"] = np.empty((n, NUM_CLASSES), dtype=np.float32)
        feats_keep: Dict[str, List[torch.Tensor]] = {k: [] for k in self.tap.names} if keep_feats else {}

        native_col = {name: i for i, name in enumerate(LR.NATIVE_DAC_LAYERS)}
        alias_of = {c: nm for nm, c in LR.NATIVE_ALIAS_TO_CANDIDATE.items()}

        for s, x in iter_batches(images, self.batch_size, self.device):
            z, feats = forward_pass(self.model, self.tap, x)
            e = s + x.shape[0]
            z_all[s:e] = z.cpu().numpy()
            if keep_feats:
                for k, v in feats.items():
                    feats_keep[k].append(v)
            if with_distances:
                for ci, name in enumerate(LR.CANDIDATE_NAMES):
                    self._tick()
                    gk = LR.K_DAC if ci in alias_of else None
                    r_t, r_p, g = self.banks[ci].query(feats[name], global_k=gk)
                    out["r"][s:e, ci] = r_t
                    out["r_perm"][s:e, ci] = r_p
                    if g is not None:
                        out["s_glob"][s:e, native_col[alias_of[ci]]] = g
                    self._tock(name)
                # native stem (S_DAC only)
                d = normalized_euclidean(feats["conv1"], self.conv1_bank)
                out["s_glob"][s:e, native_col["conv1"]] = (
                    torch.kthvalue(d, LR.K_DAC, dim=1).values.cpu().numpy()
                )
                zc = torch.from_numpy(centered_logit_representation(z.cpu().numpy())).to(self.device)
                r_t, _, _ = self.logit_bank.query(zc)
                out["r_logit"][s:e] = r_t
            if with_probes:
                for ci, name in enumerate(LR.CANDIDATE_NAMES):
                    out["probe_logits"][s:e, ci] = self.probes[ci].logits(feats[name]).cpu().numpy()
                zl = z
                out["logit_probe_logits"][s:e] = self.logit_probe.logits(zl).cpu().numpy()
        if keep_feats:
            out["feats"] = {k: torch.cat(v) for k, v in feats_keep.items()}
        return out

    def _tick(self) -> None:
        if self.timing_on:
            torch.cuda.synchronize()
            self._t0 = time.perf_counter()

    def _tock(self, name: str) -> None:
        if self.timing_on:
            torch.cuda.synchronize()
            self.timers[name] = self.timers.get(name, 0.0) + (time.perf_counter() - self._t0)


# ==========================================================================
# FIT phase (clean data only)
# ==========================================================================


def build_banks(pilot: Pilot, train_raw: np.ndarray, train_labels: np.ndarray, perm_labels: np.ndarray):
    dev = pilot.device
    feats: Dict[str, List[torch.Tensor]] = {k: [] for k in pilot.tap.names}
    logits: List[np.ndarray] = []
    for s, x in iter_batches(train_raw, pilot.batch_size, dev):
        z, f = forward_pass(pilot.model, pilot.tap, x)
        logits.append(z.cpu().numpy())
        for k, v in f.items():
            feats[k].append(v)
    bank_feats = {k: torch.cat(v) for k, v in feats.items()}
    train_logits = np.concatenate(logits)
    for ci, name in enumerate(LR.CANDIDATE_NAMES):
        pilot.banks[ci] = LayerBank(bank_feats[name], train_labels, perm_labels, dev)
    pilot.conv1_bank = bank_feats["conv1"].contiguous()
    pilot.logit_bank = LayerBank(
        torch.from_numpy(centered_logit_representation(train_logits)), train_labels, perm_labels, dev
    )
    return bank_feats, train_logits


def fit_probes(
    pilot: Pilot,
    bank_feats: Dict[str, torch.Tensor],
    train_labels: np.ndarray,
    val: Dict[str, Any],
    fit_idx: np.ndarray,
    val_labels: np.ndarray,
) -> Dict[str, Any]:
    dev = pilot.device
    ytr = torch.from_numpy(train_labels).long().to(dev)
    yfit = val_labels[fit_idx]
    record: Dict[str, Any] = {"layers": {}, "logit_probe": {}}
    probes: List[LR.Probe] = []
    for ci, name in enumerate(LR.CANDIDATE_NAMES):
        xtr = bank_feats[name]
        mean, std = xtr.mean(0), xtr.std(0) + 1e-6
        xv = val["feats"][name]
        best = None
        tried = {}
        for lam in LR.PROBE_LAMBDAS:
            p = LR.fit_probe(xtr, ytr, NUM_CLASSES, lam, mean=mean, std=std)
            lg = p.logits(xv[fit_idx]).cpu().numpy()
            v = LR.nll(np_softmax(lg), yfit)
            tried[str(lam)] = v
            if best is None or v < best[0]:
                best = (v, lam, p, lg)
        _, lam, p, lg = best
        p.temperature = LR.fit_probe_temperature(lg, yfit)
        probes.append(p)
        record["layers"][name] = {
            "dim": int(xtr.shape[1]), "lambda": lam, "lambda_grid_inner_fit_nll": tried,
            "temperature": p.temperature, "n_params": p.n_params,
            "fit_data": "train(45000)", "hyperparams_role": "validation inner-FIT",
        }
    pilot.probes = probes

    # full-logit probe: fitted on validation inner-FIT (train logits are degenerate)
    zv = torch.from_numpy(val["z"]).to(dev)
    zfit = zv[fit_idx]
    yfit_t = torch.from_numpy(yfit).long().to(dev)
    # In-sample fit and in-sample hyperparameter selection would be optimistic
    # for a 10 101-parameter probe on 2 500 rows, so lambda is chosen by 2-fold
    # cross-fit inside inner-FIT.
    half = len(fit_idx) // 2
    perm = np.random.default_rng(0).permutation(len(fit_idx))
    a, b = perm[:half], perm[half:]
    best = None
    tried = {}
    for lam in LR.PROBE_LAMBDAS:
        v = 0.0
        for tr, te in ((a, b), (b, a)):
            p = LR.fit_probe(zfit[tr], yfit_t[tr], NUM_CLASSES, lam)
            v += LR.nll(np_softmax(p.logits(zfit[te]).cpu().numpy()), yfit[te]) / 2
        tried[str(lam)] = v
        if best is None or v < best[0]:
            best = (v, lam)
    lam = best[1]
    lp = LR.fit_probe(zfit, yfit_t, NUM_CLASSES, lam)
    lp.temperature = LR.fit_probe_temperature(lp.logits(zfit).cpu().numpy(), yfit)
    pilot.logit_probe = lp
    record["logit_probe"] = {
        "dim": NUM_CLASSES, "lambda": lam, "lambda_grid_inner_fit_2fold_nll": tried,
        "temperature": lp.temperature, "n_params": lp.n_params,
        "fit_data": "validation inner-FIT (2 500)", "hyperparams_role": "2-fold cross-fit inside inner-FIT",
    }
    return record


def val_probe_logits(pilot: Pilot, val: Dict[str, Any]) -> None:
    n = val["z"].shape[0]
    pl = np.empty((n, LR.N_CANDIDATES, NUM_CLASSES), dtype=np.float32)
    for ci, name in enumerate(LR.CANDIDATE_NAMES):
        pl[:, ci] = pilot.probes[ci].logits(val["feats"][name]).cpu().numpy()
    val["probe_logits"] = pl
    val["logit_probe_logits"] = pilot.logit_probe.logits(torch.from_numpy(val["z"]).to(pilot.device)).cpu().numpy()


def run_selection(val: Dict[str, Any], s_val: np.ndarray, fit_idx, sel_idx, y_val, probe_temps) -> Dict[str, Any]:
    z, r = val["z"], val["r"]
    yf, ys = y_val[fit_idx], y_val[sel_idx]

    def score_a(S: Tuple[int, ...]) -> float:
        fit = LR.fit_family_a(z[fit_idx], r[fit_idx], S, s_val[fit_idx], yf)
        return LR.nll(LR.family_a_probs(z[sel_idx], r[sel_idx], S, fit["beta"], s_val[sel_idx]), ys)

    def score_b(S: Tuple[int, ...]) -> float:
        return LR.nll(LR.family_b_probs(val["probe_logits"][sel_idx], probe_temps, S), ys)

    sel_a = LR.greedy_forward(score_a, LR.N_CANDIDATES, max(LR.LAYER_COUNTS))
    sel_b = LR.greedy_forward(score_b, LR.N_CANDIDATES, max(LR.LAYER_COUNTS))
    return {"family_a": sel_a, "family_b": sel_b}


def build_arms(state_sel: Dict[str, Any], val: Dict[str, Any], s_val, fit_idx, sel_idx, y_val, probe_temps,
               probe_params: Sequence[int], logit_probe_params: int):
    """Fit every reported arm's scalar(s) on inner-FIT and score inner-SELECT."""
    z, r, rp = val["z"], val["r"], val["r_perm"]
    yf, ys = y_val[fit_idx], y_val[sel_idx]
    arms: Dict[str, Dict[str, Any]] = {}
    sets_a = LR.nested_sets(state_sel["family_a"]["order"])
    sets_b = LR.nested_sets(state_sel["family_b"]["order"])

    def fa(name, S, rr, tag):
        fit = LR.fit_family_a(z[fit_idx], rr[fit_idx], S, s_val[fit_idx], yf)
        q = LR.family_a_probs(z[sel_idx], rr[sel_idx], S, fit["beta"], s_val[sel_idx])
        arms[name] = {
            "family": "A", "kind": tag, "layers": list(S), "beta": fit["beta"],
            "beta_at_boundary": fit["at_lower_boundary"] or fit["at_upper_boundary"],
            "inner_fit_nll": fit["objective_value"], "inner_select_nll": LR.nll(q, ys),
            "inner_fit_nll_at_beta0": fit["objective_value_at_zero"], "n_fitted_params": 1,
        }

    def fb(name, S, tag):
        q = LR.family_b_probs(val["probe_logits"][sel_idx], probe_temps, S)
        arms[name] = {
            "family": "B", "kind": tag, "layers": list(S), "inner_select_nll": LR.nll(q, ys),
            "n_fitted_params": int(sum(probe_params[i] for i in S)),
        }

    for L in LR.LAYER_COUNTS:
        fa(f"A_greedy_L{L}", sets_a[L], r, "greedy")
        fa(f"A_depth_L{L}", LR.depth_spaced_indices(L), r, "depth_spaced")
        fa(f"A_permuted_greedy_L{L}", sets_a[L], rp, "permuted_label_control")
        fb(f"B_greedy_L{L}", sets_b[L], "greedy")
        fb(f"B_depth_L{L}", LR.depth_spaced_indices(L), "depth_spaced")
    # single-source logit-space control: R_k from centered-logit bank
    fit = LR.fit_family_a(z[fit_idx], val["r_logit"][fit_idx], [0], s_val[fit_idx], yf)
    arms["A_logit_space"] = {
        "family": "A", "kind": "logit_space_control", "layers": ["logits"], "beta": fit["beta"],
        "beta_at_boundary": fit["at_lower_boundary"] or fit["at_upper_boundary"],
        "inner_fit_nll": fit["objective_value"], "inner_select_nll": LR.nll(
            LR.family_a_probs(z[sel_idx], val["r_logit"][sel_idx], [0], fit["beta"], s_val[sel_idx]), ys),
        "inner_fit_nll_at_beta0": fit["objective_value_at_zero"], "n_fitted_params": 1,
    }
    arms["B_final_repr_probe"] = {
        "family": "B", "kind": "final_representation_probe", "layers": [LR.N_CANDIDATES - 1],
        "n_fitted_params": int(probe_params[LR.N_CANDIDATES - 1]),
        "inner_select_nll": LR.nll(LR.family_b_probs(val["probe_logits"][sel_idx], probe_temps, [LR.N_CANDIDATES - 1]), ys),
    }
    arms["B_logit_probe"] = {
        "family": "B", "kind": "full_logit_probe", "layers": ["logits"], "n_fitted_params": int(logit_probe_params),
    }
    arms["native_dac"] = {"family": "A", "kind": "beta0_native_dac", "layers": [], "beta": 0.0, "n_fitted_params": 0}
    return arms


# ==========================================================================
# EVALUATE phase: reads the frozen state only. No fitting function is
# referenced anywhere below this line's `evaluate_cell` (asserted by a test).
# ==========================================================================


def arm_probs(name: str, spec: Dict[str, Any], arr: Dict[str, Any], state: Dict[str, Any], s_dac_x: np.ndarray):
    fam, kind = spec["family"], spec["kind"]
    if fam == "A":
        if kind == "logit_space_control":
            return LR.family_a_probs(arr["z"], arr["r_logit"], [0], spec["beta"], s_dac_x)
        rr = arr["r_perm"] if kind == "permuted_label_control" else arr["r"]
        if kind == "beta0_native_dac":
            return LR.family_a_probs(arr["z"], arr["r"], [0], 0.0, s_dac_x)
        return LR.family_a_probs(arr["z"], rr, spec["layers"], spec["beta"], s_dac_x)
    if kind == "full_logit_probe":
        return LR.probe_probs(arr["logit_probe_logits"], state["logit_probe"]["temperature"])
    return LR.family_b_probs(arr["probe_logits"], state["probe_temperatures"], spec["layers"])


def evaluate_cell(cell: str, arr: Dict[str, Any], labels: np.ndarray, state: Dict[str, Any], baselines: Dict[str, Any]):
    """Score every arm on one cell. Frozen state in, metrics out; no fitting."""
    z = arr["z"].astype(np.float64)
    s_x = LR.s_dac(arr["s_glob"], np.asarray(state["native_dac"]["weights"]), state["native_dac"]["intercept"])
    base = np_softmax(z)
    probs: Dict[str, np.ndarray] = {"base_model": base}
    for name, fn in baselines.items():
        probs[name] = fn(arr["z"])
    for name, spec in state["arms"].items():
        probs[name] = arm_probs(name, spec, arr, state, s_x)
    # structural check: beta = 0 IS native DAC (softmax(z / S_DAC)), every cell
    ref = np_softmax(z / s_x[:, None])
    beta0_dev = float(np.abs(LR.family_a_probs(arr["z"], arr["r"], [0], 0.0, s_x) - ref).max())
    assert beta0_dev < 1e-12, f"beta=0 does not reproduce native DAC (max dev {beta0_dev})"
    probs["native_dac"] = ref

    per_arm: Dict[str, Any] = {}
    for name, p in probs.items():
        m = evaluate_all(p, labels)
        entry = {"metrics": {k: (None if v is None else float(v)) for k, v in m.items()}}
        entry["delta_accuracy"] = float(m["accuracy"] - (probs["base_model"].argmax(1) == labels).mean())
        if name != "base_model":
            entry["flips"] = LR.flip_decomposition(base, p, labels)
            entry["margin_matched_enrichment"] = LR.margin_matched_enrichment(base, p, labels)
        per_arm[name] = entry
    return per_arm, probs, {"beta0_native_dac_max_dev": beta0_dev}


def per_sample_record(probs: Dict[str, np.ndarray], labels: np.ndarray, arr: Dict[str, Any]) -> Dict[str, np.ndarray]:
    rec: Dict[str, np.ndarray] = {"labels": labels.astype(np.int16), "z": arr["z"].astype(np.float16)}
    for name, p in probs.items():
        rec[f"pred__{name}"] = p.argmax(1).astype(np.int16)
        rec[f"conf__{name}"] = p.max(1).astype(np.float16)
        rec[f"gtrank__{name}"] = LR.gt_rank(p, labels).astype(np.int16)
    for k in ("r", "r_perm", "r_logit", "s_glob", "probe_logits", "logit_probe_logits"):
        if k in arr:
            rec[f"raw__{k}"] = arr[k].astype(np.float16)
    return rec


# ==========================================================================
# Orchestration
# ==========================================================================


def load_frozen_baselines(fitted_dir: str) -> Dict[str, Any]:
    fns = {}
    for name in ("temperature_scaling", "vector_scaling"):
        with open(os.path.join(fitted_dir, f"{name}.pkl"), "rb") as f:
            obj = pickle.load(f)
        fns[name] = (lambda o: (lambda logits: np.asarray(o.calibrate(np.asarray(logits, dtype=np.float64)))))(obj)
    return fns


def load_corrupted(cifar_c_dir: str, corruption: str, severity: int, batch_size: int):
    tf = pp.eval_transform(DATASET, pp.PROTOCOL_CORRECTED, corruption=True)
    loader = load_cifar_c_loader(DATASET, corruption, severity, tf, batch_size, cifar_c_dir=cifar_c_dir)
    return get_all_data_as_numpy(loader)


def benchmark_crosscheck(root: str, seed: int, cell: str, images, labels, probs) -> Dict[str, Any]:
    """Record (never assert on) agreement with the benchmark's own corrected-protocol
    cell: identical pixels/labels, and how far the pilot's recomputed native DAC is
    from the benchmark's stored native_dac probabilities (TF32 vs fp32 forward)."""
    inter = os.path.join(root, "evaluation", f"checkpoint_seed{seed}", cell, "intermediates")
    res: Dict[str, Any] = {"benchmark_cell_present": os.path.isdir(inter)}
    if not res["benchmark_cell_present"]:
        return res
    try:
        b_raw = np.load(os.path.join(inter, "splits", "test_raw.npy"), mmap_mode="r")
        b_lab = np.load(os.path.join(inter, "splits", "test_labels.npy"))
        n = len(labels)
        res["identical_pixels"] = bool(b_raw.shape[0] >= n and np.array_equal(np.asarray(b_raw[:n]), np.asarray(images)))
        res["identical_labels"] = bool(np.array_equal(b_lab[:n], labels))
        b_nat = os.path.join(inter, "method_outputs", "native_dac", "probs.npy")
        if os.path.exists(b_nat):
            bp = np.load(b_nat)[:n]
            res["native_dac_max_abs_prob_diff_vs_benchmark"] = float(np.abs(bp - probs["native_dac"]).max())
            res["native_dac_argmax_agreement_vs_benchmark"] = float(np.mean(bp.argmax(1) == probs["native_dac"].argmax(1)))
        b_base = os.path.join(inter, "method_outputs", "base_model", "probs.npy")
        if os.path.exists(b_base):
            bb = np.load(b_base)[:n]
            res["base_argmax_agreement_vs_benchmark"] = float(np.mean(bb.argmax(1) == probs["base_model"].argmax(1)))
    except Exception as exc:  # cross-check is informational
        res["error"] = repr(exc)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--benchmark_root", required=True, help="results/studyAB/phase0_corrected_v2")
    ap.add_argument("--cifar_c_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_cells", type=int, default=None, help="smoke tests only")
    ap.add_argument("--smoke_subset", type=int, default=None, help="smoke tests only: truncate every split")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable (refusing to fall back to CPU)")
    device = torch.device(args.device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    clean_dir = os.path.join(args.benchmark_root, "evaluation", f"checkpoint_seed{args.seed}", "clean")
    fitted_dir = os.path.join(args.benchmark_root, "fitted_method", f"checkpoint_seed{args.seed}")
    inter = os.path.join(clean_dir, "intermediates")
    for d, what in ((inter, "benchmark intermediates"), (fitted_dir, "benchmark fitted state")):
        if pp.read_protocol(d) != pp.PROTOCOL_CORRECTED:
            raise SystemExit(f"{what} at {d} is not stamped {pp.PROTOCOL_CORRECTED}; refusing to use it")
        pp.require_compatible(d, DATASET, pp.PROTOCOL_CORRECTED, what)

    out_root = os.path.join(args.out_dir, f"checkpoint_seed{args.seed}")
    os.makedirs(out_root, exist_ok=True)
    sp = lambda k: os.path.join(inter, "splits", k)  # noqa: E731
    sub = args.smoke_subset
    cut = (lambda a: a[:sub]) if sub else (lambda a: a)
    train_raw, train_y = np.load(sp("train_raw.npy"), mmap_mode="r"), np.load(sp("train_labels.npy"))
    val_raw, val_y = cut(np.load(sp("val_raw.npy"), mmap_mode="r")), cut(np.load(sp("val_labels.npy")))
    test_raw, test_y = cut(np.load(sp("test_raw.npy"), mmap_mode="r")), cut(np.load(sp("test_labels.npy")))
    assert train_raw.shape[0] == train_y.shape[0] and val_raw.shape[0] == val_y.shape[0]
    counts = np.bincount(train_y, minlength=NUM_CLASSES)
    assert counts.min() >= LR.K_C

    ckpt = construct_model_path(args.results_dir, METHOD, DATASET, MODEL, args.seed)
    model = load_trained_model(ckpt, MODEL, NUM_CLASSES, device, dataset=DATASET)
    model.eval()
    pilot = Pilot(model, device, args.batch_size)
    t_start = time.time()

    # ------------------------------------------------------------- FIT
    perm_labels = LR.permuted_labels(train_y, LR.PERMUTED_LABEL_SEED + args.seed)
    print("[fit] building banks", flush=True)
    bank_feats, _ = build_banks(pilot, train_raw, train_y, perm_labels)
    with open(os.path.join(fitted_dir, "native_dac.pkl"), "rb") as f:
        native = pickle.load(f)
    assert native.num_layers == len(LR.NATIVE_DAC_LAYERS) and native.k == LR.K_DAC, "unexpected native DAC layout"
    pilot.native_w, pilot.native_b = np.asarray(native.weights[:-1], dtype=np.float64), float(native.weights[-1])

    print("[fit] validation pass", flush=True)
    val = pilot.extract(val_raw, with_distances=True, with_probes=False, keep_feats=True)
    fit_idx, sel_idx = make_inner_validation_split(val_y, select_fraction=0.5, seed=INNER_SEED)
    assert set(fit_idx).isdisjoint(sel_idx)
    probe_record = fit_probes(pilot, bank_feats, train_y, val, fit_idx, val_y)
    val_probe_logits(pilot, val)
    del bank_feats
    torch.cuda.empty_cache()
    probe_params = [p.n_params for p in pilot.probes]
    probe_temps = [p.temperature for p in pilot.probes]
    s_val = LR.s_dac(val["s_glob"], pilot.native_w, pilot.native_b)

    print("[fit] greedy selection", flush=True)
    selection = run_selection(val, s_val, fit_idx, sel_idx, val_y, probe_temps)
    arms = build_arms(selection, val, s_val, fit_idx, sel_idx, val_y, probe_temps,
                      probe_params, pilot.logit_probe.n_params)
    val_z_hash = LR.state_hash({}, [val["z"]])
    state = {
        "version": PILOT_VERSION, "split_plan": SPLIT_PLAN, "seed": args.seed,
        "preprocessing": pp.preprocessing_stamp(DATASET, pp.PROTOCOL_CORRECTED),
        "checkpoint": ckpt, "k_c": LR.K_C, "k_dac": LR.K_DAC,
        "candidates": [{"index": i, "module": n, "dim": d} for i, (n, d) in enumerate(LR.CANDIDATE_LAYERS)],
        "native_dac": {
            "layers": list(LR.NATIVE_DAC_LAYERS), "weights": pilot.native_w.tolist(), "intercept": pilot.native_b,
            "source": os.path.join(fitted_dir, "native_dac.pkl"),
        },
        "inner_split": {"seed": INNER_SEED, "n_fit": int(len(fit_idx)), "n_select": int(len(sel_idx)),
                        "fit_idx_hash": LR.state_hash({}, [fit_idx]), "select_idx_hash": LR.state_hash({}, [sel_idx])},
        "bank": {"size": int(len(train_y)), "min_class_count": int(counts.min()), "labels_hash": LR.state_hash({}, [train_y]),
                 "permuted_label_seed": LR.PERMUTED_LABEL_SEED + args.seed,
                 "permuted_labels_hash": LR.state_hash({}, [perm_labels])},
        "val_logits_hash": val_z_hash,
        "probes": probe_record, "probe_temperatures": probe_temps, "logit_probe": probe_record["logit_probe"],
        "selection": selection, "arms": arms,
        "depth_spaced": {str(L): LR.depth_spaced_indices(L) for L in LR.LAYER_COUNTS},
        "declared_deviations": [
            "native DAC S_DAC weights were fitted by the benchmark on the whole validation split",
            "Family-B layer probes fitted on train; full-logit probe fitted on validation inner-FIT",
        ],
    }
    weights_hash = LR.state_hash({}, [p.weight.cpu().numpy() for p in pilot.probes] + [pilot.logit_probe.weight.cpu().numpy()])
    state["probe_weights_hash"] = weights_hash
    state["state_hash"] = LR.state_hash({k: v for k, v in state.items() if k != "state_hash"})
    with open(os.path.join(out_root, "frozen_state.json"), "w") as f:
        json.dump(state, f, indent=1, default=float)
    with open(os.path.join(out_root, "selection_trace.json"), "w") as f:
        json.dump(selection, f, indent=1, default=float)
    torch.save({"layer_probes": [(p.weight.cpu(), p.bias.cpu(), p.mean.cpu(), p.std.cpu(), p.lam, p.temperature) for p in pilot.probes],
                "logit_probe": (pilot.logit_probe.weight.cpu(), pilot.logit_probe.bias.cpu(), pilot.logit_probe.mean.cpu(),
                                pilot.logit_probe.std.cpu(), pilot.logit_probe.lam, pilot.logit_probe.temperature)},
               os.path.join(out_root, "probe_weights.pt"))
    fit_seconds = time.time() - t_start
    print(f"[fit] done in {fit_seconds:.0f}s; state hash {state['state_hash']}", flush=True)
    del val
    baselines = load_frozen_baselines(fitted_dir)

    # ------------------------------------------------------------- EVALUATE
    cells: List[Tuple[str, Optional[str], Optional[int]]] = [("clean", None, None)]
    cells += [(f"{c}_s{s}", c, s) for c in CORRUPTIONS for s in SEVERITIES]
    if args.max_cells:
        cells = cells[: args.max_cells]
    for cell, corr, sev in cells:
        cdir = os.path.join(out_root, cell)
        os.makedirs(cdir, exist_ok=True)
        if os.path.exists(os.path.join(cdir, "cell_metrics.json")):
            print(f"[eval] skip completed {cell}", flush=True)
            continue
        t0 = time.time()
        if corr is None:
            images, labels = test_raw, test_y
        else:
            images, labels = load_corrupted(args.cifar_c_dir, corr, sev, args.batch_size)
            if sub:
                images, labels = images[:sub], labels[:sub]
        pilot.timing_on = corr is None  # per-layer reference-search cost, measured once on clean test
        pilot.timers.clear()
        arr = pilot.extract(images, with_distances=True, with_probes=True)
        per_arm, probs, chk = evaluate_cell(cell, arr, labels, state, baselines)
        rec = {
            "cell": cell, "corruption": corr, "severity": sev, "seed": args.seed, "n": int(len(labels)),
            "state_hash": state["state_hash"], "evaluation_only": True,
            "preprocessing_protocol": pp.PROTOCOL_CORRECTED, "checks": chk,
            "arms": per_arm, "wall_seconds": time.time() - t0,
        }
        if corr is None:
            rec["reference_search_seconds_per_layer_on_clean_test"] = dict(pilot.timers)
            rec["dims"] = {n: d for n, d in LR.CANDIDATE_LAYERS}
        rec["benchmark_crosscheck"] = benchmark_crosscheck(args.benchmark_root, args.seed, cell, images, labels, probs)
        np.savez(os.path.join(cdir, "per_sample.npz"), **per_sample_record(probs, labels, arr))
        with open(os.path.join(cdir, "cell_metrics.json"), "w") as f:
            json.dump(rec, f, indent=1, default=float)
        print(f"[eval] {cell}: {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
