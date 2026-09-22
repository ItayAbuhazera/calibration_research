"""Shared fold/split plan for Stage 0c (docs/stage0_execution_spec.md Section 2).

The 10 000 CIFAR-100 test images and their duplicate structure are checkpoint-independent
(both seed-2 and seed-4 fits reuse the identical index-level plan built here), so the plan is
computed once and cached to results/stage0/shared/fold_plan.json.

    python -m atlas.stage0_folds            # build (idempotent) and print a summary
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from . import spec, stage0_data

FOLD_SEED = 20260922
N_OUTER = 5
NESTED_N = 2500
PLAN_PATH = "results/stage0/shared/fold_plan.json"


def duplicate_group_ids(n: int = 10000) -> np.ndarray:
    """group_id[i] = smallest index in i's pixel-identical group; singleton otherwise."""
    gid = np.arange(n)
    for entry in stage0_data.test_duplicate_groups().values():
        idxs = sorted(entry["indices"])
        gid[idxs] = idxs[0]
    return gid


def _group_reps(gid: np.ndarray) -> np.ndarray:
    return np.unique(gid)


def make_outer_folds(labels: np.ndarray, gid: np.ndarray, n_splits: int = N_OUTER, seed: int = FOLD_SEED) -> np.ndarray:
    """fold_id[i] in [0, n_splits); duplicate groups never split across folds."""
    n = len(labels)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_id = np.full(n, -1, dtype=np.int64)
    for k, (_, test_idx) in enumerate(sgkf.split(np.zeros(n), labels, groups=gid)):
        fold_id[test_idx] = k
    assert (fold_id >= 0).all()
    for entry in stage0_data.test_duplicate_groups().values():
        idxs = entry["indices"]
        assert len({fold_id[i] for i in idxs}) == 1, f"duplicate group {idxs} split across folds"
    return fold_id


def _grouped_75_25(train_idx: np.ndarray, gid: np.ndarray, seed: int) -> np.ndarray:
    """Returns a boolean array (len == len(train_idx)) True = inner-fit (~75%), False = inner-val (~25%), grouped."""
    groups_here = gid[train_idx]
    reps = np.unique(groups_here)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(reps))
    reps_shuffled = reps[perm]
    n_val_groups = max(1, round(0.25 * len(reps_shuffled)))
    val_reps = set(reps_shuffled[:n_val_groups].tolist())
    is_val = np.array([g in val_reps for g in groups_here])
    return ~is_val  # True = inner-fit


def make_nested_subset(train_idx: np.ndarray, gid: np.ndarray, inner_fit_mask: np.ndarray, target_n: int, seed: int) -> np.ndarray:
    """Subset of train_idx (group-preserving) closest to target_n distinct images, inheriting inner_fit_mask."""
    groups_here = gid[train_idx]
    reps = np.unique(groups_here)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(reps))
    chosen_groups: List[int] = []
    count = 0
    rep_to_members = {}
    for i, g in enumerate(groups_here):
        rep_to_members.setdefault(g, []).append(i)
    for gi in order:
        g = reps[gi]
        members = rep_to_members[g]
        if count > 0 and abs((count + len(members)) - target_n) > abs(count - target_n) and count >= target_n * 0.98:
            continue
        chosen_groups.append(g)
        count += len(members)
        if count >= target_n:
            break
    sel_local = np.concatenate([rep_to_members[g] for g in chosen_groups])
    sel_local.sort()
    return sel_local  # positions into train_idx / inner_fit_mask


def make_cell_assignment(train_idx: np.ndarray, gid: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """One of spec.CELLS per index in train_idx; balanced over cells and, where feasible, classes.

    Assigned per duplicate-group (never per raw index) so a group's members share one cell.
    """
    n_cells = len(spec.CELLS)
    groups_here = gid[train_idx]
    reps = np.unique(groups_here)
    rep_label = {}
    rep_to_members = {}
    for i, g in enumerate(groups_here):
        rep_to_members.setdefault(g, []).append(i)
    for g, members in rep_to_members.items():
        rep_label[g] = int(labels[train_idx[members[0]]])

    by_class: Dict[int, List[int]] = {}
    for g in reps:
        by_class.setdefault(rep_label[g], []).append(g)
    rng = np.random.default_rng(seed)
    class_order = list(by_class.keys())
    rng.shuffle(class_order)
    cell_of_rep: Dict[int, int] = {}
    global_counter = 0  # NOT reset per class: avoids every class favoring low cell indices
    for c in class_order:
        gs = list(by_class[c])
        rng.shuffle(gs)
        for g in gs:
            cell_of_rep[g] = global_counter % n_cells
            global_counter += 1

    out = np.zeros(len(train_idx), dtype=np.int64)
    for i, g in enumerate(groups_here):
        out[i] = cell_of_rep[g]
    return out  # index into spec.CELLS, positions align with train_idx


def build_plan(force: bool = False) -> Dict:
    if os.path.exists(PLAN_PATH) and not force:
        return json.load(open(PLAN_PATH))

    labels = np.load("results/atlas/shared/test_labels.npy").astype(np.int64)
    n = len(labels)
    gid = duplicate_group_ids(n)
    fold_id = make_outer_folds(labels, gid, N_OUTER, FOLD_SEED)

    outer = []
    for f in range(N_OUTER):
        train_idx = np.where(fold_id != f)[0]
        test_idx = np.where(fold_id == f)[0]
        inner_fit_mask = _grouped_75_25(train_idx, gid, seed=FOLD_SEED + 1000 + f)
        nested_local = make_nested_subset(train_idx, gid, inner_fit_mask, NESTED_N, seed=FOLD_SEED + 2000 + f)
        cell_assign = make_cell_assignment(train_idx, gid, labels, seed=FOLD_SEED + 3000 + f)
        n_nested_images = len(nested_local)
        outer.append({
            "fold": f,
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "inner_fit_mask": inner_fit_mask.tolist(),
            "nested_2500_local_positions": nested_local.tolist(),
            "n_nested_images_actual": int(n_nested_images),
            "cell_assignment_local": cell_assign.tolist(),
        })

    plan = {
        "fold_seed": FOLD_SEED,
        "n_outer": N_OUTER,
        "nested_target_n": NESTED_N,
        "n_duplicate_groups": len(stage0_data.test_duplicate_groups()),
        "fold_id": fold_id.tolist(),
        "duplicate_group_id": gid.tolist(),
        "outer": outer,
    }
    os.makedirs(os.path.dirname(PLAN_PATH), exist_ok=True)
    json.dump(plan, open(PLAN_PATH, "w"))
    return plan


def load_plan() -> Dict:
    return build_plan(force=False)


if __name__ == "__main__":
    plan = build_plan(force=False)
    for o in plan["outer"]:
        print(f"fold {o['fold']}: train={len(o['train_idx'])} test={len(o['test_idx'])} "
              f"inner_fit={sum(o['inner_fit_mask'])} inner_val={len(o['inner_fit_mask']) - sum(o['inner_fit_mask'])} "
              f"nested_n_actual={o['n_nested_images_actual']}")
