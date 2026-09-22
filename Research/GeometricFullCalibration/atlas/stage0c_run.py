"""Stage 0c runner: fits q_Z and q_ZP for one (seed, regime, outer fold) and scores all 13
conditions on the held-out fold (docs/stage0_execution_spec.md Sections 3-4).

    python -m atlas.stage0c_run --seed 2 --regime T-8k12 --fold 0
    python -m atlas.stage0c_run --seed 2 --regime T-8k12 --fold 0 --fast   # 1-lambda smoke test

Idempotent: refuses to overwrite an existing output unless --force is given.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import spec, stage0_data, stage0_fit, stage0_folds

REGIMES = ("T-8k12", "T-8k1", "T-2.5k12", "T-2.5k1", "S-8k1", "S-2.5k1")
OUT_ROOT = "results/stage0"


def regime_view_rule(regime: str) -> str:
    return "all12" if regime.endswith("12") else ("clean" if regime.startswith("S-") else "one")


def regime_uses_nested(regime: str) -> bool:
    return regime.startswith(("T-2.5k", "S-2.5k"))


def build_rows(cells_data, image_ids: np.ndarray, view_rule: str, cell_assignment: np.ndarray = None,
               p_image_ids: np.ndarray = None):
    """image_ids: original test-index positions. Returns Xz, Xp, y, cond_tag (per row).

    p_image_ids (same length/order as image_ids, defaults to image_ids): the image id used to
    fetch P only -- Z/labels always come from image_ids. Used by stage0_shuffle.py to break the
    (P, Z/label) association per image while preserving per-condition/per-view structure.
    """
    if p_image_ids is None:
        p_image_ids = image_ids
    if view_rule == "all12":
        Xz, Xp, Y, cond = [], [], [], []
        for ci, cell in enumerate(spec.CELLS):
            d = cells_data[cell]
            Xz.append(d["z"][image_ids]); Xp.append(d["p"][p_image_ids]); Y.append(d["labels"][image_ids])
            cond.append(np.full(len(image_ids), ci, dtype=np.int64))
        return np.concatenate(Xz), np.concatenate(Xp), np.concatenate(Y), np.concatenate(cond)
    if view_rule == "clean":
        d = cells_data["clean"]
        return d["z"][image_ids], d["p"][p_image_ids], d["labels"][image_ids], np.full(len(image_ids), -1, dtype=np.int64)
    if view_rule == "one":
        assert cell_assignment is not None
        Xz = np.empty((len(image_ids), spec.NUM_CLASSES)); Xp = np.empty((len(image_ids), spec.NUM_CLASSES))
        Y = np.empty(len(image_ids), dtype=np.int64); cond = cell_assignment.copy()
        for ci, cell in enumerate(spec.CELLS):
            m = cell_assignment == ci
            if not m.any():
                continue
            d = cells_data[cell]
            Xz[m] = d["z"][image_ids[m]]; Xp[m] = d["p"][p_image_ids[m]]; Y[m] = d["labels"][image_ids[m]]
        return Xz, Xp, Y, cond
    raise ValueError(view_rule)


def run_one(seed: int, regime: str, fold: int, fast: bool = False, force: bool = False) -> str:
    out_dir = f"{OUT_ROOT}/seed{seed}/{regime}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/fold{fold}.npz"
    if os.path.exists(out_path) and not force:
        return out_path

    t_start = time.time()
    plan = stage0_folds.load_plan()
    o = plan["outer"][fold]
    train_idx = np.array(o["train_idx"])
    test_idx = np.array(o["test_idx"])
    inner_fit_mask = np.array(o["inner_fit_mask"], dtype=bool)
    cell_assignment_local = np.array(o["cell_assignment_local"], dtype=np.int64)

    if regime_uses_nested(regime):
        local_pos = np.array(o["nested_2500_local_positions"])
    else:
        local_pos = np.arange(len(train_idx))

    train_images = train_idx[local_pos]
    inner_mask_train = inner_fit_mask[local_pos]
    cell_assign_train = cell_assignment_local[local_pos]

    view_rule = regime_view_rule(regime)
    cells_data = stage0_data.load_all_cells(seed)

    ca_full = cell_assign_train if view_rule == "one" else None
    ca_fit = cell_assign_train[inner_mask_train] if view_rule == "one" else None
    ca_val = cell_assign_train[~inner_mask_train] if view_rule == "one" else None

    Xz_full, Xp_full, y_full, _ = build_rows(cells_data, train_images, view_rule, ca_full)
    Xz_if, Xp_if, y_if, _ = build_rows(cells_data, train_images[inner_mask_train], view_rule, ca_fit)
    Xz_iv, Xp_iv, y_iv, _ = build_rows(cells_data, train_images[~inner_mask_train], view_rule, ca_val)

    lambda_grid = (1e-1,) if fast else stage0_fit.LAMBDA_GRID

    arms = {}
    for arm_name, cols in (("q_Z", "z"), ("q_ZP", "zp")):
        if cols == "z":
            Xfull, Xif, Xiv = Xz_full, Xz_if, Xz_iv
        else:
            Xfull = np.concatenate([Xz_full, Xp_full], axis=1)
            Xif = np.concatenate([Xz_if, Xp_if], axis=1)
            Xiv = np.concatenate([Xz_iv, Xp_iv], axis=1)
        fit_out = stage0_fit.fit_arm(Xfull, y_full, Xif, y_if, Xiv, y_iv, spec.NUM_CLASSES, lambda_grid)
        arms[arm_name] = fit_out

    # deterministic transform check: B=0, (A,b) copied from q_Z ⇒ q_ZP-with-B0 reproduces q_Z exactly
    A_qz, b_qz = arms["q_Z"]["W"], arms["q_Z"]["b"]
    mean_qz, std_qz = arms["q_Z"]["scaler_mean"], arms["q_Z"]["scaler_std"]
    W_check = np.concatenate([A_qz, np.zeros_like(A_qz)], axis=0)
    mean_check = np.concatenate([mean_qz, np.zeros(spec.NUM_CLASSES)])
    std_check = np.concatenate([std_qz, np.ones(spec.NUM_CLASSES)])
    probe_x = np.concatenate([Xz_full[:5], Xp_full[:5]], axis=1)
    p_ref = stage0_fit.predict_probs(Xz_full[:5], mean_qz, std_qz, A_qz, b_qz)
    p_check = stage0_fit.predict_probs(probe_x, mean_check, std_check, W_check, b_qz)
    transform_check_max_dev = float(np.abs(p_ref - p_check).max())
    assert transform_check_max_dev < 1e-9, f"B=0 transform check failed: {transform_check_max_dev}"

    # score all 13 conditions on the held-out fold
    record = {"test_idx": test_idx, "regime": regime, "seed": seed, "fold": fold,
              "transform_check_max_dev": transform_check_max_dev, "n_train_images": len(train_images),
              "wall_time_s": None}
    for cond in spec.CONDITIONS:
        d = cells_data[cond]
        labels_test = d["labels"][test_idx]
        record[f"labels__{cond}"] = labels_test.astype(np.int16)
        for arm_name in ("q_Z", "q_ZP"):
            fit_out = arms[arm_name]
            if arm_name == "q_Z":
                Xtest = d["z"][test_idx]
            else:
                Xtest = np.concatenate([d["z"][test_idx], d["p"][test_idx]], axis=1)
            probs = stage0_fit.predict_probs(Xtest, fit_out["scaler_mean"], fit_out["scaler_std"], fit_out["W"], fit_out["b"])
            record[f"probs__{arm_name}__{cond}"] = probs.astype(np.float32)

    for arm_name in ("q_Z", "q_ZP"):
        fo = arms[arm_name]
        record[f"meta__{arm_name}__selected_lambda"] = fo["selected_lambda"]
        record[f"meta__{arm_name}__converged"] = fo["converged"]
        record[f"meta__{arm_name}__grad_inf_norm"] = fo["grad_inf_norm"]
        record[f"meta__{arm_name}__retried"] = fo["retried"]
        record[f"meta__{arm_name}__inner_val_nll_table"] = json.dumps(fo["selection"]["table"])

    record["wall_time_s"] = time.time() - t_start
    tmp_path = f"{out_dir}/fold{fold}.tmp.npz"
    np.savez(tmp_path, **record)
    os.replace(tmp_path, out_path)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(2, 4))
    ap.add_argument("--regime", required=True, choices=REGIMES)
    ap.add_argument("--fold", type=int, required=True, choices=range(5))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    p = run_one(args.seed, args.regime, args.fold, fast=args.fast, force=args.force)
    print("wrote", p)
