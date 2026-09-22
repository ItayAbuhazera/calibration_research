"""Effect of the atlas-FP32 vs benchmark-path (TF32-conv, batch 128) base-prediction differences on this study's paired gains.
For each checkpoint/condition with a numerics record: (i) # mismatch images, (ii) worst-case change of ANY method's paired accuracy gain vs base = mismatches / N
(if every mismatch image sat on an intervention and resolved adversarially), (iii) # mismatch images that are also intervention rows of the deep Z1/Z0/C0/C1 gates
(the only rows where a base flip could change their paired gain), (iv) accuracy of the atlas base vs the benchmark base on that condition."""
import glob, json
import numpy as np
from . import common, fg_data, spec


def main():
    out = {}
    for seed in (2, 4):
        ps = {c: np.load(f"{fg_data.FG}/seed{seed}/per_sample_{c}.npz") for c in spec.CONDITIONS}
        rows = []
        for f in sorted(glob.glob(f"{fg_data.FG}/seed{seed}/numerics/*.json")):
            r = json.load(open(f)); c = r["cond"]; ids = np.asarray(r["mismatch_ids"], int); N = 10000
            rec = {"cond": c, "n_mismatch": len(ids), "worst_case_gain_change_pp": 100 * len(ids) / N,
                   "bench_path_reproduces_bench_argmax_mismatches_b256": r["modes"]["bench_adapter_default"]["argmax_mismatch_vs_bench_full"],
                   "atlas_base_acc": r["modes"]["fp32_strict_b250"]["acc_atlas"], "bench_base_acc": r["modes"]["bench_adapter_default"]["acc_bench"]}
            for fam in ("Z1", "Z0", "C0", "C1"):
                g = ps[c][f"g__deep|{fam}|ofull|n2500"].astype(bool)
                rec[f"mismatch_on_intervention_deep_{fam}"] = int(g[ids].sum()) if len(ids) else 0
            rows.append(rec)
        out[seed] = {"cells": rows, "n_cells": len(rows), "mean_mismatch_per_cell": float(np.mean([r["n_mismatch"] for r in rows])), "max_mismatch": int(max(r["n_mismatch"] for r in rows)),
                     "macro_worst_case_gain_change_pp": float(np.mean([r["worst_case_gain_change_pp"] for r in rows])),
                     "total_mismatch_on_deep_Z1_interventions": int(sum(r["mismatch_on_intervention_deep_Z1"] for r in rows)),
                     "max_abs_base_acc_diff_pp": float(100 * max(abs(r["atlas_base_acc"] - r["bench_base_acc"]) for r in rows))}
    common.atomic_json(f"{fg_data.FG}/report/numerics_effect.json", out)
    print(json.dumps({s: {k: v for k, v in d.items() if k != "cells"} for s, d in out.items()}, indent=1))


if __name__ == "__main__":
    main()
