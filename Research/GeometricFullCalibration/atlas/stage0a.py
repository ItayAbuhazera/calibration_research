"""Stage 0a -- descriptive rank-of-repairs analysis (docs/stage0_execution_spec.md, controlling
prompt Section 8/0a). Purely descriptive: no fitting, no continuation gate. Uses only cached
artifacts (atlas base logits, fixed-gate deep candidate j, labels).

    python -m atlas.stage0a
"""
from __future__ import annotations

import json
import os

import numpy as np

from Calibrators import layer_readouts as LR
from . import spec, stage0_data

OUT = "results/stage0/report"


def _onehot(idx: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(idx), k), dtype=np.float64)
    out[np.arange(len(idx)), idx] = 1.0
    return out


def rank_bucket(rank: np.ndarray) -> dict:
    return {
        "rank1_unexpected_count": int((rank == 1).sum()),
        "rank2": int((rank == 2).sum()),
        "ranks3_5": int(((rank >= 3) & (rank <= 5)).sum()),
        "rank_gt5": int((rank > 5).sum()),
        "n": int(len(rank)),
    }


def analyze_cell(seed: int, cell: str) -> dict:
    d = stage0_data.load_cell(seed, cell)
    z, labels, cand_j, base_pred = d["z"], d["labels"], d["cand_j_deep"], d["base_pred"]
    n = d["n"]

    onehot_j = _onehot(cand_j, spec.NUM_CLASSES)
    flips = LR.flip_decomposition(z, onehot_j, labels)
    disagreement_rate = float((cand_j != base_pred).mean())

    changed = cand_j != base_pred
    rank_j_under_base_all = LR.gt_rank(z, cand_j)
    rank_j_changed = rank_bucket(rank_j_under_base_all[changed])

    base_wrong = base_pred != labels
    n_base_errors = int(base_wrong.sum())
    true_rank_all_base_errors = LR.gt_rank(z, labels)[base_wrong]
    true_rank_bucket = rank_bucket(true_rank_all_base_errors)

    return {
        "seed": seed, "cell": cell, "n": n,
        "n_base_errors": n_base_errors, "base_error_rate": n_base_errors / n,
        "disagreement_rate": disagreement_rate,
        "W": flips["W"], "H": flips["H"], "U": flips["U"], "flips": flips["flips"],
        "net_W_minus_H": flips["net"], "delta_acc_from_flips": flips["delta_acc_from_flips"],
        "decisive_precision_W_over_WplusH": flips["decisive_precision"],
        "intervention_precision_W_over_flips": flips["intervention_precision"],
        "rank_of_j_under_base_logits__among_disagreements": rank_j_changed,
        "true_class_rank_among_all_base_errors": true_rank_bucket,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for seed in spec.DEV_SEEDS:
        for cell in spec.CONDITIONS:
            rows.append(analyze_cell(seed, cell))

    macro = {}
    for seed in spec.DEV_SEEDS:
        cells_only = [r for r in rows if r["seed"] == seed and r["cell"] in spec.CELLS]
        W = sum(r["W"] for r in cells_only); H = sum(r["H"] for r in cells_only); U = sum(r["U"] for r in cells_only)
        n = sum(r["n"] for r in cells_only)
        n_err = sum(r["n_base_errors"] for r in cells_only)
        rank2 = sum(r["rank_of_j_under_base_logits__among_disagreements"]["rank2"] for r in cells_only)
        r35 = sum(r["rank_of_j_under_base_logits__among_disagreements"]["ranks3_5"] for r in cells_only)
        rgt5 = sum(r["rank_of_j_under_base_logits__among_disagreements"]["rank_gt5"] for r in cells_only)
        r1 = sum(r["rank_of_j_under_base_logits__among_disagreements"]["rank1_unexpected_count"] for r in cells_only)
        macro[str(seed)] = {
            "pooled_12_cells": {"W": W, "H": H, "U": U, "N": n, "n_base_errors": n_err,
                                 "delta_acc_from_flips": (W - H) / n,
                                 "rank_of_j_disagreements": {"rank1_unexpected": r1, "rank2": rank2, "ranks3_5": r35, "rank_gt5": rgt5,
                                                              "denominator": r1 + rank2 + r35 + rgt5},
                                 "runner_up_frequency": rank2 / (r1 + rank2 + r35 + rgt5) if (r1 + rank2 + r35 + rgt5) else None},
        }

    fit_row_limitation = ("The fixed-gate study's 'fit' role (2500/1250/1250 rows) is a clean-only "
                           "validation split disjoint from the 10000-image test set; no corrupted-condition "
                           "per-sample array with cand_j exists for it. This 0a analysis therefore covers "
                           "clean test + the 12 corruption cells only; no separate historical-fit-row table "
                           "is reported. See docs/stage0_execution_spec.md Section 1.")

    out = {"per_cell": rows, "macro_pooled_12_cells_by_seed": macro,
           "fit_row_data_limitation": fit_row_limitation,
           "note": "Descriptive only. Runner-up frequency neither proves nor disproves class-agnostic "
                   "information; no accuracy gate controls continuation from this analysis."}
    json.dump(out, open(f"{OUT}/stage0a_rank_tables.json", "w"), indent=1)
    print(json.dumps(macro, indent=1))
    print("wrote", f"{OUT}/stage0a_rank_tables.json")


if __name__ == "__main__":
    main()
