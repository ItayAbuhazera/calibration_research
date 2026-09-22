"""
Cross-check the layer pilot's baseline rows against the corrected-protocol unified
benchmark run for the same checkpoint/cell (informational; nothing is fitted).

For every (seed, cell) present in both trees, compare base_model, native_dac,
temperature_scaling and vector_scaling accuracy / NLL / ECE as computed by the pilot
(strict-fp32 forward, frozen benchmark calibrators applied to the pilot's logits) with
the benchmark's own `summary_metrics.json`. Differences are expected to be tiny
(TF32 vs fp32 forward passes only).
"""
import json, os, sys

CELLS = ["clean"] + [f"{c}_s{s}" for c in ("gaussian_noise", "defocus_blur", "fog", "jpeg_compression") for s in (1, 3, 5)]
METHODS = ("base_model", "native_dac", "temperature_scaling", "vector_scaling")


def bench_metrics(path):
    d = json.load(open(path))
    rows = d["methods"]
    return {r["method_name"]: r["metrics"] for r in rows}


def main(pilot="results/layer_pilot", bench="results/studyAB/phase0_corrected_v2/evaluation"):
    out, worst = [], {"acc": 0.0, "nll": 0.0, "ece": 0.0}
    for seed in (2, 4):
        for cell in CELLS:
            bp = os.path.join(bench, f"checkpoint_seed{seed}", cell, "summary_metrics.json")
            pp = os.path.join(pilot, f"checkpoint_seed{seed}", cell, "cell_metrics.json")
            if not (os.path.exists(bp) and os.path.exists(pp)):
                out.append({"seed": seed, "cell": cell, "status": "benchmark cell missing"}); continue
            b, p = bench_metrics(bp), json.load(open(pp))["arms"]
            for m in METHODS:
                if m not in b:
                    continue
                bm = b[m]
                acc_b = bm.get("accuracy"); nll_b = bm.get("nll"); ece_b = bm.get("top_label_ece", bm.get("ece"))
                pm = p[m]["metrics"]
                rec = {"seed": seed, "cell": cell, "method": m,
                       "d_acc": pm["accuracy"] - acc_b, "d_nll": pm["nll"] - nll_b,
                       "d_ece": pm["top_label_ece"] - ece_b if ece_b is not None else None}
                out.append(rec)
                worst["acc"] = max(worst["acc"], abs(rec["d_acc"])); worst["nll"] = max(worst["nll"], abs(rec["d_nll"]))
                if rec["d_ece"] is not None:
                    worst["ece"] = max(worst["ece"], abs(rec["d_ece"]))
    json.dump({"worst_abs_diff": worst, "rows": out}, open(os.path.join(pilot, "aggregate", "crosscheck_vs_benchmark.json"), "w"), indent=1)
    print("worst abs diff:", worst, "rows:", len(out), "missing:", sum(1 for r in out if r.get("status")))


if __name__ == "__main__":
    main(*sys.argv[1:])
