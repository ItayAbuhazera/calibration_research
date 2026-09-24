"""Pre-reading noise bound (card note 2026-09-24): argmax differences between each refit q_Z and the
Stage 0 q_Z, per checkpoint x regime x evidence source, and the worst-case effect on each declared
contrast in pp (differences / N; N = 10,000 images per condition). Reads only q_Z predictions.

    python -m atlas.stage0_ablation_noise_bound
"""
import json
import numpy as np
from . import spec

SRC = ["L11", "xckpt", "L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L9", "L10"]
FULL = ("T-8k1", "S-8k1", "T-2.5k1", "S-2.5k1")


def main():
    rows = {}
    for ev in SRC:
        for seed in (2, 4):
            for rg in (FULL if ev in ("L11", "xckpt") else FULL[:2]):
                per_cond = {c: 0 for c in spec.CONDITIONS}
                for f in range(5):
                    a = np.load(f"results/stage0_ablation/{ev}/seed{seed}/{rg}/fold{f}.npz", allow_pickle=True)
                    b = np.load(f"results/stage0/seed{seed}/{rg}/fold{f}.npz", allow_pickle=True)
                    for c in spec.CONDITIONS:
                        per_cond[c] += int((a[f"probs__q_Z__{c}"].argmax(1) != b[f"probs__q_Z__{c}"].argmax(1)).sum())
                n12 = sum(per_cond[c] for c in spec.CELLS)
                rows[f"{ev}|seed{seed}|{rg}"] = {"argmax_diffs_12cells": n12, "argmax_diffs_clean": per_cond["clean"],
                    "argmax_diffs_max_single_cell": max(per_cond[c] for c in spec.CELLS),
                    "worst_case_macro12_pp": 100 * n12 / (12 * 10000), "worst_case_clean_pp": 100 * per_cond["clean"] / 10000,
                    "worst_case_single_cell_pp": 100 * max(per_cond[c] for c in spec.CELLS) / 10000}
    worst = max(max(r["worst_case_macro12_pp"], r["worst_case_clean_pp"], r["worst_case_single_cell_pp"]) for r in rows.values())
    # a gap contrast uses one T and one S regime: bound = sum of the two macro12 bounds; D_A uses two q_Z arms of which one is refit
    gap = 0.0
    for ev in SRC:
        for seed in (2, 4):
            for tag in (("T-8k1", "S-8k1"), ("T-2.5k1", "S-2.5k1")):
                if f"{ev}|seed{seed}|{tag[0]}" in rows:
                    gap = max(gap, rows[f"{ev}|seed{seed}|{tag[0]}"]["worst_case_macro12_pp"] + rows[f"{ev}|seed{seed}|{tag[1]}"]["worst_case_macro12_pp"])
    out = {"rows": rows, "max_worst_case_pp_any_entry": worst, "max_worst_case_gap_pp": gap, "stop_threshold_pp": 0.1,
           "stop": bool(worst >= 0.1 or gap >= 0.1)}
    json.dump(out, open("results/stage0_ablation/report/noise_bound.json", "w"), indent=1)
    print("max worst-case entry pp:", worst, "max worst-case gap pp:", gap, "STOP:", out["stop"], "total diffs", sum(r["argmax_diffs_12cells"] + r["argmax_diffs_clean"] for r in rows.values()))
    hdr = "| evidence | " + " | ".join(f"s{s} {r}" for s in (2, 4) for r in FULL) + " |\n|---|" + "---|" * 8
    lines = [hdr]
    for ev in SRC:
        cells = []
        for s in (2, 4):
            for r in FULL:
                k = f"{ev}|seed{s}|{r}"
                cells.append(f"{rows[k]['argmax_diffs_12cells']}/{rows[k]['argmax_diffs_clean']}" if k in rows else "–")
        lines.append(f"| {ev} | " + " | ".join(cells) + " |")
    open("results/stage0_ablation/report/noise_bound_table.md", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
