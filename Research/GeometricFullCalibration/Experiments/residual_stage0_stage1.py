"""
Residual-evidence study, Stage 0 (provenance / reconciliation / index manifests) and
Stage 1 (cheap diagnostics from EXISTING corrected per-example predictions).

Stage 0 reads only artifacts. It rebuilds the CIFAR-100 train/validation ORIGINAL-index
partition for a checkpoint seed exactly as data/cifar100.py::get_train_valid_loader does
(np.random.seed(seed); shuffle; first 10 % = validation), maps every benchmark-materialized
row to its original CIFAR index by row hash under the corrected transform, and writes an
index manifest. It also reconciles the layer pilot with the corrected benchmark.

Stage 1 uses the pilot's stored per-example predictions of FROZEN challengers. It fits
nothing and trains no gate. "Oracle union" = an evaluation-only label oracle choosing among
already-fixed predictions: not a deployment method, not a Bayes bound, not headroom over
all possible readouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Calibrators import layer_readouts as LR  # noqa: E402
from utils import preprocessing_protocol as pp  # noqa: E402
from utils.decision_audit import make_inner_validation_split  # noqa: E402

CKPT_ROOT = ("/home/itayab/PyCharmProjects/Research/geometric/GeometricInternalCalibration/"
             "GeometricInternalCalibration/aaai_full_experiments/results/baseline/baseline_cross_entropy/cifar100/resnet101")
BENCH = "results/studyAB/phase0_corrected_v2"
PILOT = "results/layer_pilot"
CELLS = ["clean"] + [f"{c}_s{s}" for c in ("gaussian_noise", "defocus_blur", "fog", "jpeg_compression") for s in (1, 3, 5)]

# Stage-1 challengers (frozen list; per-example predictions exist for the 12 dev cells + clean)
CHALLENGERS = ["vector_scaling", "A_greedy_L4", "A_depth_L4", "B_greedy_L4", "B_logit_probe",
               "B_final_repr_probe", "A_logit_space"]
FULL_LOGIT_CONTROLS = ["vector_scaling", "B_logit_probe"]
GEOMETRIC_CHALLENGERS = ["A_greedy_L4", "A_depth_L4", "A_logit_space"]


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def orig_split(seed: int, n: int = 50000, valid_size: float = 0.1):
    """data/cifar100.py::get_train_valid_loader's partition of CIFAR-100 train, in shuffled order."""
    idx = list(range(n))
    split = int(np.floor(valid_size * n))
    np.random.seed(seed)
    np.random.shuffle(idx)
    return np.asarray(idx[split:]), np.asarray(idx[:split])


def row_hashes(a: np.ndarray) -> List[bytes]:
    return [hashlib.blake2b(np.ascontiguousarray(a[i]).tobytes(), digest_size=16).digest() for i in range(a.shape[0])]


def build_manifest(seed: int, out_dir: str, data_root: str = "./data") -> Dict[str, Any]:
    from torchvision import datasets

    clean = os.path.join(BENCH, "evaluation", f"checkpoint_seed{seed}", "clean", "intermediates", "splits")
    tf = pp.eval_transform("cifar100", pp.PROTOCOL_CORRECTED)
    ds = datasets.CIFAR100(root=data_root, train=True, download=False, transform=tf)
    train_idx, val_idx = orig_split(seed)
    res: Dict[str, Any] = {"seed": seed, "protocol": pp.PROTOCOL_CORRECTED}
    import torch

    def rebuilt(idxs):
        return torch.stack([ds[int(i)][0] for i in idxs]).numpy()

    for name, idxs in (("val", val_idx), ("train", train_idx)):
        raw = np.load(os.path.join(clean, f"{name}_raw.npy"), mmap_mode="r")
        lab = np.load(os.path.join(clean, f"{name}_labels.npy"))
        mine = rebuilt(idxs)
        h_bench = row_hashes(raw)
        h_mine = {h: int(i) for i, h in zip(idxs, row_hashes(mine))}
        # CIFAR-100 contains bit-identical duplicate images: map by hash, resolve ties by label
        missing = [k for k, h in enumerate(h_bench) if h not in h_mine]
        res[f"{name}_set_identical"] = bool(len(missing) == 0 and len(set(h_bench)) >= len(h_mine) - 20)
        res[f"{name}_rows_not_found_in_rebuilt"] = len(missing)
        res[f"{name}_n"] = int(raw.shape[0])
        res[f"{name}_labels_hash"] = sha_arr(lab)
        mapping = [h_mine.get(h, -1) for h in h_bench]
        y_all = np.asarray(ds.targets)
        res[f"{name}_label_agrees_with_original"] = bool(np.all(y_all[np.asarray(mapping)[np.asarray(mapping) >= 0]] ==
                                                                lab[np.asarray(mapping) >= 0]))
        res[f"{name}_original_index_in_materialized_order"] = mapping
    val_lab = np.load(os.path.join(clean, "val_labels.npy"))
    fit, sel = make_inner_validation_split(val_lab, select_fraction=0.5, seed=123)
    vmap = np.asarray(res["val_original_index_in_materialized_order"])
    res["inner_fit_original_index"] = vmap[fit].tolist()
    res["inner_select_original_index"] = vmap[sel].tolist()
    res["inner_fit_rows_hash"] = sha_arr(fit)
    res["inner_select_rows_hash"] = sha_arr(sel)
    res["test_n"] = int(np.load(os.path.join(clean, "test_labels.npy")).shape[0])
    res["test_labels_hash"] = sha_arr(np.load(os.path.join(clean, "test_labels.npy")))
    res["array_file_hashes"] = {k: sha256_file(os.path.join(clean, k)) for k in
                                ("train_raw.npy", "val_raw.npy", "test_raw.npy", "train_labels.npy", "val_labels.npy", "test_labels.npy")}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"manifest_seed{seed}.json"), "w") as f:
        json.dump(res, f)
    return {k: v for k, v in res.items() if not isinstance(v, list)}


def reconcile(seed: int) -> Dict[str, Any]:
    r: Dict[str, Any] = {"seed": seed}
    inter = os.path.join(BENCH, "evaluation", f"checkpoint_seed{seed}", "clean", "intermediates")
    fitted = os.path.join(BENCH, "fitted_method", f"checkpoint_seed{seed}")
    r["benchmark_intermediates_protocol"] = pp.read_protocol(inter)
    r["benchmark_fitted_state_protocol"] = pp.read_protocol(fitted)
    st = json.load(open(os.path.join(PILOT, f"checkpoint_seed{seed}", "frozen_state.json")))
    r["pilot_preprocessing"] = st["preprocessing"]["protocol"]
    r["pilot_state_hash"] = st["state_hash"]
    ent = json.load(open(os.path.join(inter, "method_outputs", "native_dac", "entry.json")))
    r["native_dac_layers_benchmark"] = ent["dac_layers"]
    r["native_dac_k_benchmark"] = ent["dac_k"]
    r["native_dac_layers_pilot"] = st["native_dac"]["layers"]
    r["native_dac_source_set_matches"] = list(ent["dac_layers"]) == list(st["native_dac"]["layers"])
    r["native_dac_is_five_source_no_logits"] = len(ent["dac_layers"]) == 5
    lab_b = np.load(os.path.join(inter, "splits", "test_labels.npy"))
    d = np.load(os.path.join(PILOT, f"checkpoint_seed{seed}", "clean", "per_sample.npz"))
    r["test_labels_identical_pilot_vs_benchmark"] = bool(np.array_equal(lab_b, d["labels"].astype(lab_b.dtype)))
    for m in ("native_dac", "temperature_scaling", "vector_scaling", "base_model"):
        p = os.path.join(inter, "method_outputs", m, "probs.npy")
        if os.path.exists(p):
            pb = np.load(p)
            r[f"argmax_agreement_{m}"] = float(np.mean(pb.argmax(1) == d[f"pred__{m}"]))
    fi, se = make_inner_validation_split(np.load(os.path.join(inter, "splits", "val_labels.npy")), 0.5, 123)
    r["inner_fit_hash_matches_pilot"] = LR.state_hash({}, [fi]) == st["inner_split"]["fit_idx_hash"]
    r["inner_select_hash_matches_pilot"] = LR.state_hash({}, [se]) == st["inner_split"]["select_idx_hash"]
    cc = os.path.join(PILOT, "aggregate", "crosscheck_vs_benchmark.json")
    if os.path.exists(cc):
        rows = json.load(open(cc))
        r["metric_crosscheck_worst_abs_diff"] = rows["worst_abs_diff"]
        r["metric_crosscheck_cells_missing"] = sum(1 for x in rows["rows"] if x.get("status"))
    r["benchmark_corruption_cells_present"] = sorted(
        c for c in CELLS[1:]
        if os.path.exists(os.path.join(BENCH, "evaluation", f"checkpoint_seed{seed}", c, "summary_metrics.json")))
    r["checkpoint_sha256"] = sha256_file(os.path.join(CKPT_ROOT, f"seed{seed}", "best_model.pth"))
    return r


def job_status() -> Dict[str, Any]:
    try:
        q = subprocess.check_output(["squeue", "-u", os.environ.get("USER", "itayab"), "-h", "-o", "%i|%j|%T|%M"], text=True)
    except Exception as e:  # pragma: no cover
        return {"error": repr(e)}
    rows = [x.split("|") for x in q.strip().splitlines()]
    return {"squeue": rows}


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


def stage1(seeds=(2, 4)) -> Dict[str, Any]:
    out: Dict[str, Any] = {"challengers": CHALLENGERS, "per_seed": {}, "pooled": {}}
    acc: Dict[str, Dict[str, np.ndarray]] = {}
    for seed in seeds:
        per_cell = {}
        for cell in CELLS:
            d = np.load(os.path.join(PILOT, f"checkpoint_seed{seed}", cell, "per_sample.npz"))
            y = d["labels"].astype(int)
            base = d["pred__base_model"].astype(int)
            base_ok = base == y
            rank_base = d["gtrank__base_model"].astype(int)  # from full-precision probabilities (int16)
            rec: Dict[str, Any] = {"n": int(len(y)), "base_acc": float(base_ok.mean()), "base_errors": int((~base_ok).sum()),
                                   "runner_up_correct_among_base_errors": float(np.mean(rank_base[~base_ok] == 2)),
                                   "arms": {}}
            preds = {}
            for a in CHALLENGERS:
                p = d[f"pred__{a}"].astype(int)
                preds[a] = p
                ch = p != base
                W = int((ch & ~base_ok & (p == y)).sum()); H = int((ch & base_ok & (p != y)).sum())
                U = int(ch.sum()) - W - H
                Fl = W + H + U
                rec["arms"][a] = {
                    "intervention_rate": Fl / len(y), "W": W, "H": H, "U": U,
                    "W_over_total": W / Fl if Fl else None, "W_over_W_plus_H": W / (W + H) if W + H else None,
                    "net_over_N": (W - H) / len(y), "W_over_base_errors": W / max((~base_ok).sum(), 1),
                    # candidate correctness (given a flip on a base error) vs detection (flip rate on errors vs correct)
                    "candidate_correct_given_flip_on_error": (W / (W + U)) if (W + U) else None,
                    "flip_rate_on_base_errors": float(ch[~base_ok].mean()), "flip_rate_on_base_correct": float(ch[base_ok].mean()),
                    "acc": float((p == y).mean()),
                }
            corr = {a: preds[a] == y for a in CHALLENGERS}
            def union(names):
                m = base_ok.copy()
                for n in names:
                    m |= corr[n]
                return float(m.mean())
            rec["oracle_union_acc"] = {"base+each": {a: union([a]) for a in CHALLENGERS},
                                       "base+all_challengers": union(CHALLENGERS),
                                       "base+full_logit_controls": union(FULL_LOGIT_CONTROLS),
                                       "base+geometric_challengers": union(GEOMETRIC_CHALLENGERS),
                                       "base+full_logit+geometric": union(FULL_LOGIT_CONTROLS + GEOMETRIC_CHALLENGERS)}
            rec["oracle_headroom_over_base_pp"] = {k: 100 * (v - rec["base_acc"]) if not isinstance(v, dict) else
                                                   {kk: 100 * (vv - rec["base_acc"]) for kk, vv in v.items()}
                                                   for k, v in rec["oracle_union_acc"].items()}
            rec["oracle_headroom_geometric_beyond_full_logit_pp"] = 100 * (
                rec["oracle_union_acc"]["base+full_logit+geometric"] - rec["oracle_union_acc"]["base+full_logit_controls"])
            # repaired-set overlap
            Wsets = {a: (preds[a] != base) & ~base_ok & (preds[a] == y) for a in CHALLENGERS}
            ov = {}
            for a, b in combinations(CHALLENGERS, 2):
                inter, un = int((Wsets[a] & Wsets[b]).sum()), int((Wsets[a] | Wsets[b]).sum())
                ov[f"{a}|{b}"] = {"jaccard": inter / un if un else None, "inter": inter, "union": un}
            rec["repaired_overlap"] = ov
            rec["repaired_not_by_any_full_logit_control"] = {
                a: int((Wsets[a] & ~(Wsets[FULL_LOGIT_CONTROLS[0]] | Wsets[FULL_LOGIT_CONTROLS[1]])).sum())
                for a in CHALLENGERS}
            rec["repaired_total"] = {a: int(Wsets[a].sum()) for a in CHALLENGERS}
            per_cell[cell] = rec
            if cell != "clean":
                for key in ("base_ok",):
                    pass
                acc.setdefault("W", {});
        out["per_seed"][seed] = per_cell
    # pooled over 12 corruption cells x seeds
    def pool(fn):
        v = [fn(out["per_seed"][s][c]) for s in seeds for c in CELLS[1:]]
        return float(np.mean(v))
    pooled: Dict[str, Any] = {"n_seed_cells": len(seeds) * 12}
    for a in CHALLENGERS:
        pooled[a] = {k: pool(lambda r, a=a, k=k: r["arms"][a][k]) for k in
                     ("intervention_rate", "W_over_base_errors", "net_over_N", "acc")}
        Ws = sum(out["per_seed"][s][c]["arms"][a]["W"] for s in seeds for c in CELLS[1:])
        Hs = sum(out["per_seed"][s][c]["arms"][a]["H"] for s in seeds for c in CELLS[1:])
        Us = sum(out["per_seed"][s][c]["arms"][a]["U"] for s in seeds for c in CELLS[1:])
        pooled[a].update({"W": Ws, "H": Hs, "U": Us, "W_over_total": Ws / max(Ws + Hs + Us, 1),
                          "W_over_W_plus_H": Ws / max(Ws + Hs, 1)})
    pooled["base_acc"] = pool(lambda r: r["base_acc"])
    pooled["runner_up_correct_among_base_errors"] = pool(lambda r: r["runner_up_correct_among_base_errors"])
    for k in ("base+all_challengers", "base+full_logit_controls", "base+geometric_challengers", "base+full_logit+geometric"):
        pooled[f"oracle_headroom_pp::{k}"] = pool(lambda r, k=k: r["oracle_headroom_over_base_pp"][k])
    pooled["oracle_headroom_geometric_beyond_full_logit_pp"] = pool(lambda r: r["oracle_headroom_geometric_beyond_full_logit_pp"])
    pooled["oracle_headroom_pp::base+each"] = {
        a: pool(lambda r, a=a: r["oracle_headroom_over_base_pp"]["base+each"][a]) for a in CHALLENGERS}
    out["pooled"] = pooled
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/residual_study/stage0_stage1")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--skip_manifest", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    s0: Dict[str, Any] = {"jobs": job_status(), "seeds": {}}
    for s in (2, 4):
        s0["seeds"][s] = {"reconciliation": reconcile(s)}
        if not args.skip_manifest:
            s0["seeds"][s]["manifest_summary"] = build_manifest(s, args.out_dir, args.data_root)
    s0["checkpoint_sha256_all_seeds"] = {
        str(s): sha256_file(os.path.join(CKPT_ROOT, f"seed{s}", "best_model.pth")) for s in (1, 2, 3, 4, 5)
        if os.path.exists(os.path.join(CKPT_ROOT, f"seed{s}", "best_model.pth"))}
    json.dump(s0, open(os.path.join(args.out_dir, "stage0_reconciliation.json"), "w"), indent=1, default=float)
    s1 = stage1()
    json.dump(s1, open(os.path.join(args.out_dir, "stage1_candidate_headroom.json"), "w"), indent=1, default=float)
    print(json.dumps({"stage0": {s: v["reconciliation"] for s, v in s0["seeds"].items()}}, indent=1, default=float)[:6000])
    print(json.dumps(s1["pooled"], indent=1, default=float))


if __name__ == "__main__":
    main()
