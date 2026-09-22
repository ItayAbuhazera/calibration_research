"""Shuffled-P control (docs/stage0_execution_spec.md Section 5): fold 0 only, q_ZP on
{T-8k12, T-2.5k12, T-8k1} x {seed 2, seed 4} = 6 fits, P's image identity independently permuted
within each of {inner-fit, inner-val, outer-refit-train, outer-eval} -- never across those
partitions, never using labels. Compared against the REAL (unshuffled) q_Z/q_ZP already written
by atlas.stage0c_run for the same regime/fold/seed.

    python -m atlas.stage0_shuffle --seed 2 --regime T-8k12
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import spec, stage0_data, stage0_fit, stage0_folds
from .stage0c_run import build_rows, regime_view_rule, regime_uses_nested

PERM_SEED = 20260922 + 1
FOLD = 0
REGIMES = ("T-8k12", "T-2.5k12", "T-8k1")
OUT_ROOT = "results/stage0/shuffle_control"


def _perm_within(image_ids: np.ndarray, seed: int) -> np.ndarray:
    """Random permutation of image_ids among themselves (image-identity shuffle within one partition)."""
    rng = np.random.default_rng(seed)
    return image_ids[rng.permutation(len(image_ids))]


def run_one(seed: int, regime: str, force: bool = False) -> str:
    out_dir = OUT_ROOT
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/seed{seed}_{regime}.npz"
    if os.path.exists(out_path) and not force:
        return out_path

    t0 = time.time()
    plan = stage0_folds.load_plan()
    o = plan["outer"][FOLD]
    train_idx = np.array(o["train_idx"])
    test_idx = np.array(o["test_idx"])
    inner_fit_mask = np.array(o["inner_fit_mask"], dtype=bool)
    cell_assignment_local = np.array(o["cell_assignment_local"], dtype=np.int64)
    local_pos = np.array(o["nested_2500_local_positions"]) if regime_uses_nested(regime) else np.arange(len(train_idx))

    train_images = train_idx[local_pos]
    inner_mask_train = inner_fit_mask[local_pos]
    cell_assign_train = cell_assignment_local[local_pos]
    view_rule = regime_view_rule(regime)
    cells_data = stage0_data.load_all_cells(seed)

    fit_images = train_images[inner_mask_train]
    val_images = train_images[~inner_mask_train]
    ca_fit = cell_assign_train[inner_mask_train] if view_rule == "one" else None
    ca_val = cell_assign_train[~inner_mask_train] if view_rule == "one" else None
    ca_full = cell_assign_train if view_rule == "one" else None

    p_fit = _perm_within(fit_images, PERM_SEED + 10)
    p_val = _perm_within(val_images, PERM_SEED + 20)
    p_full = _perm_within(train_images, PERM_SEED + 30)
    p_test = _perm_within(test_idx, PERM_SEED + 40)

    Xz_if, Xp_if, y_if, _ = build_rows(cells_data, fit_images, view_rule, ca_fit, p_image_ids=p_fit)
    Xz_iv, Xp_iv, y_iv, _ = build_rows(cells_data, val_images, view_rule, ca_val, p_image_ids=p_val)
    Xz_full, Xp_full, y_full, _ = build_rows(cells_data, train_images, view_rule, ca_full, p_image_ids=p_full)

    Xif = np.concatenate([Xz_if, Xp_if], axis=1)
    Xiv = np.concatenate([Xz_iv, Xp_iv], axis=1)
    Xfull = np.concatenate([Xz_full, Xp_full], axis=1)

    fit_out = stage0_fit.fit_arm(Xfull, y_full, Xif, y_if, Xiv, y_iv, spec.NUM_CLASSES, stage0_fit.LAMBDA_GRID)

    record = {"seed": seed, "regime": regime, "fold": FOLD, "test_idx": test_idx,
              "selected_lambda": fit_out["selected_lambda"], "converged": fit_out["converged"],
              "grad_inf_norm": fit_out["grad_inf_norm"], "retried": fit_out["retried"],
              "perm_seed": PERM_SEED}
    for cond in spec.CONDITIONS:
        d = cells_data[cond]
        labels_test = d["labels"][test_idx]
        Xtest = np.concatenate([d["z"][test_idx], d["p"][p_test]], axis=1)
        probs = stage0_fit.predict_probs(Xtest, fit_out["scaler_mean"], fit_out["scaler_std"], fit_out["W"], fit_out["b"])
        record[f"labels__{cond}"] = labels_test.astype(np.int16)
        record[f"probs__q_ZP_shuffled__{cond}"] = probs.astype(np.float32)
    record["wall_time_s"] = time.time() - t0

    tmp_path = f"{out_dir}/seed{seed}_{regime}.tmp.npz"
    np.savez(tmp_path, **record)
    os.replace(tmp_path, out_path)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=(2, 4))
    ap.add_argument("--regime", required=True, choices=REGIMES)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    print("wrote", run_one(args.seed, args.regime, force=args.force))
