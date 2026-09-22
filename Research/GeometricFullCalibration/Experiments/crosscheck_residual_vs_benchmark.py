"""
Reconcile the residual-evidence study's in-pipeline baseline rows (base / TS / VS / native DAC) and the
layer pilot's with the corrected-protocol unified benchmark, cell by cell, plus per-example agreement.
Informational; nothing is fitted. Cells the benchmark has not produced yet are listed as missing.
"""
import json, os, sys
import numpy as np

CELLS = ["clean"] + [f"{c}_s{s}" for c in ("gaussian_noise", "defocus_blur", "fog", "jpeg_compression") for s in (1, 3, 5)]
METHODS = ("base_model", "native_dac", "temperature_scaling", "vector_scaling")
BENCH = "results/studyAB/phase0_corrected_v2/evaluation"


def main(resid="results/residual_study/dev", pilot="results/layer_pilot", out="results/residual_study/stage0_stage1/corruption_reconciliation.json"):
    rows, missing = [], []
    for seed in (2, 4):
        for cell in CELLS:
            bp = os.path.join(BENCH, f"checkpoint_seed{seed}", cell, "summary_metrics.json")
            if not os.path.exists(bp):
                missing.append(f"{seed}:{cell}"); continue
            b = {r["method_name"]: r["metrics"] for r in json.load(open(bp))["methods"]}
            r = json.load(open(os.path.join(resid, f"checkpoint_seed{seed}", cell, "cell_metrics.json")))["metrics"]
            p = json.load(open(os.path.join(pilot, f"checkpoint_seed{seed}", cell, "cell_metrics.json")))["arms"]
            inter = os.path.join(BENCH, f"checkpoint_seed{seed}", cell, "intermediates")
            ps = np.load(os.path.join(resid, f"checkpoint_seed{seed}", cell, "per_sample.npz"))
            for m in METHODS:
                if m not in b:
                    continue
                rec = {"seed": seed, "cell": cell, "method": m,
                       "d_acc_resid": r[m]["accuracy"] - b[m]["accuracy"], "d_nll_resid": r[m]["nll"] - b[m]["nll"],
                       "d_ece_resid": r[m]["top_label_ece"] - b[m]["top_label_ece"],
                       "d_acc_pilot": p[m]["metrics"]["accuracy"] - b[m]["accuracy"], "d_nll_pilot": p[m]["metrics"]["nll"] - b[m]["nll"]}
                pr = os.path.join(inter, "method_outputs", m, "probs.npy")
                if os.path.exists(pr):
                    rec["argmax_agreement_resid_vs_benchmark"] = float(np.mean(np.load(pr).argmax(1) == ps[f"pred__{m}"]))
                rows.append(rec)
    worst = {k: float(max((abs(r[k]) for r in rows), default=0.0)) for k in ("d_acc_resid", "d_nll_resid", "d_ece_resid", "d_acc_pilot", "d_nll_pilot")}
    agree = [r["argmax_agreement_resid_vs_benchmark"] for r in rows if "argmax_agreement_resid_vs_benchmark" in r]
    res = {"cells_compared": len({(r["seed"], r["cell"]) for r in rows}), "cells_missing_in_benchmark": missing, "worst_abs_diff": worst,
           "min_argmax_agreement": float(min(agree)) if agree else None, "rows": rows}
    json.dump(res, open(out, "w"), indent=1)
    print({k: v for k, v in res.items() if k != "rows"})


if __name__ == "__main__":
    main(*sys.argv[1:])
