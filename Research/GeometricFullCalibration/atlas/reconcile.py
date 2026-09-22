"""CPU: reconcile the atlas program's inputs and base-side controls with the corrected-v2 unified benchmark, cell by cell.

Checked per (checkpoint, condition) that exists in the benchmark tree: identical labels; pixel-identical query arrays (atlas shared
corrupted sets vs the benchmark's materialized ``test_raw``); base logits vs the benchmark's base probabilities (max |dp|, argmax
agreement, and whether every disagreement is a near tie of the ATLAS logits); metric agreement of base / native DAC (the atlas refits
native DAC on FIT rows only; the benchmark on the whole validation split) and of TS / VS (the benchmark fits TS/VS on the whole
validation split, the atlas on FIT rows only) -- differences there are DESIGNED role differences, not numerical error.
Conditions the benchmark has not produced are listed as missing; nothing here is fitted.
"""
import json, os
import numpy as np

from . import data, spec, stats

BENCH = data.BENCH + "/evaluation"


def main(out_path="results/atlas/report/reconciliation.json"):
    out = {"per_cell": [], "missing": []}
    ty = np.load(f"{data.SHARED}/test_labels.npy")
    full = data.load_full_sets()
    for seed in spec.DEV_SEEDS:
        root = data.seed_dir(seed)
        cd = json.load(open(f"{root}/cd/results.json"))
        for c in spec.CONDITIONS:
            inter = f"{BENCH}/checkpoint_seed{seed}/{c}/intermediates"
            sm = f"{BENCH}/checkpoint_seed{seed}/{c}/summary_metrics.json"
            if not os.path.exists(sm):
                out["missing"].append(f"{seed}:{c}"); continue
            rec = {"seed": seed, "cond": c}
            b_lab = np.load(f"{inter}/splits/test_labels.npy"); rec["labels_identical"] = bool(np.array_equal(b_lab, ty))
            b_raw = np.load(f"{inter}/splits/test_raw.npy", mmap_mode="r")
            rec["pixels_identical"] = bool(np.array_equal(np.asarray(b_raw), np.asarray(full[data.cond_slice(c)])))
            z = np.load(f"{root}/u0/{c}.npz")["logits"].astype(np.float64)
            pb = np.load(f"{inter}/method_outputs/base_model/probs.npy")
            pa = stats.softmax(z)
            rec["base_max_abs_dprob"] = float(np.abs(pa - pb).max())
            dis = pa.argmax(1) != pb.argmax(1)
            rec["base_argmax_agreement"] = float(1 - dis.mean()); rec["base_n_disagree"] = int(dis.sum())
            top2 = np.sort(z, 1)[:, -2:]; marg = top2[:, 1] - top2[:, 0]
            rec["disagree_atlas_logit_margin_max"] = float(marg[dis].max()) if dis.any() else None
            rec["all_disagreements_are_near_ties(<0.02 logit)"] = bool((marg[dis] < 0.02).all()) if dis.any() else True
            bm = {r["method_name"]: r["metrics"] for r in json.load(open(sm))["methods"]}
            for m, a in (("base_model", "base"), ("native_dac", "native_dac"), ("temperature_scaling", "ts"), ("vector_scaling", "vs")):
                if m in bm:
                    rec[f"{m}_d_acc"] = cd[a]["raw"][c]["accuracy"] - bm[m]["accuracy"]; rec[f"{m}_d_nll"] = cd[a]["raw"][c]["nll"] - bm[m]["nll"]
            out["per_cell"].append(rec)
    def agg(k):
        v = [abs(r[k]) for r in out["per_cell"] if k in r and r[k] is not None]
        return float(max(v)) if v else None
    out["summary"] = {"cells_compared": len(out["per_cell"]), "labels_identical_all": all(r["labels_identical"] for r in out["per_cell"]),
                      "pixels_identical_all": all(r["pixels_identical"] for r in out["per_cell"]),
                      "min_base_argmax_agreement": min((r["base_argmax_agreement"] for r in out["per_cell"]), default=None),
                      "all_base_disagreements_near_ties": all(r["all_disagreements_are_near_ties(<0.02 logit)"] for r in out["per_cell"]),
                      "max_abs_base_d_nll": agg("base_model_d_nll"), "max_abs_base_d_acc": agg("base_model_d_acc"),
                      "max_abs_ts_d_nll_designed_role_difference": agg("temperature_scaling_d_nll"),
                      "max_abs_vs_d_nll_designed_role_difference": agg("vector_scaling_d_nll"),
                      "max_abs_native_d_nll_designed_role_difference": agg("native_dac_d_nll")}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps(out["summary"], indent=1), "missing:", len(out["missing"]))


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:2]))
