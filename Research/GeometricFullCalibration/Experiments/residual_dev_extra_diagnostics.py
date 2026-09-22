"""Extra descriptive diagnostics for the residual-evidence development stage (no fitting, no selection).

Per checkpoint: fit/select gaps of every fitted arm, selected lambda / ||W||, feature rank summaries,
projection distortion; per corruption family x severity accuracy gains; mean true-class rank and top-two
margin (base vs corrected); binned signed utility (edges fixed on clean-selection data); cost.
"""
import json, os, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/residual_study/dev"
SEEDS = [int(s) for s in (sys.argv[2:] or ["2", "4"])]
PROCS = ["proc::hidden_nll", "proc::hidden_decision", "proc::output_nll", "proc::output_decision"]
CELLS = None


def main():
    out = {"per_seed": {}}
    for s in SEEDS:
        sd = f"{ROOT}/checkpoint_seed{s}"
        st = json.load(open(f"{sd}/frozen_state.json"))
        cells = sorted(c for c in os.listdir(sd) if os.path.isdir(f"{sd}/{c}"))
        rec = {"anchor": st["anchor"]["chosen"], "selection": st["selection"],
               "projection_distortion": st["projection_distortion"], "feature_rank_summary": st["feature_rank_summary"],
               "jl_sufficient_dim": st["jl_sufficient_dim_n45000"], "fit_count": st["fit_count"],
               "arms": [{k: a[k] for k in ("name", "fit_nll", "select_nll", "fit_acc", "select_acc", "W_fro", "n_iter", "converged")} for a in st["arms"]],
               "anchor_candidates": st["anchor"]["candidates"]}
        m = {c: json.load(open(f"{sd}/{c}/cell_metrics.json")) for c in cells}
        table = {}
        for arm in ["vector_scaling", "anchor", "native_dac"] + PROCS:
            table[arm] = {c: 100 * (m[c]["metrics"][arm]["accuracy"] - m[c]["metrics"]["base_model"]["accuracy"]) for c in cells}
        rec["acc_gain_pp_by_cell"] = table
        rk, mg, bins = {}, {}, {"margin": {p: None for p in PROCS}, "strength": None}
        acc_r = {p: [] for p in PROCS + ["base_model", "anchor", "native_dac"]}
        for c in cells:
            if c == "clean":
                continue
            z = np.load(f"{sd}/{c}/per_sample.npz")
            for a in acc_r:
                key = a
                acc_r[a].append((z[f"gtrank__{key}"].astype(float).mean(),
                                 z[f"top2margin__{key}"].astype(float).mean() if f"top2margin__{key}" in z.files else np.nan))
        rec["mean_gt_rank_and_top2margin_over_12_cells"] = {a: {"gt_rank": float(np.mean([r[0] for r in v])),
                                                                "top2_prob_margin": float(np.nanmean([r[1] for r in v])) if not np.all(np.isnan([r[1] for r in v])) else None}
                                                            for a, v in acc_r.items()}
        for p in PROCS:
            tot = [{"n": 0, "W": 0, "H": 0, "U": 0} for _ in range(5)]
            for c in cells:
                if c == "clean":
                    continue
                for i, x in enumerate(m[c]["bins"][p.replace("proc::", "")]["by_base_margin"]):
                    for k in ("n", "W", "H", "U"):
                        tot[i][k] += x[k]
            for i, t in enumerate(tot):
                t["intervention_rate"] = (t["W"] + t["H"] + t["U"]) / max(t["n"], 1)
                t["signed_utility"] = (t["W"] - t["H"]) / max(t["n"], 1)
            bins["margin"][p.replace("proc::", "")] = tot
        prim = "hidden_decision"
        tot = [{"n": 0, "W": 0, "H": 0, "U": 0} for _ in range(5)]
        has = False
        for c in cells:
            if c == "clean":
                continue
            b = m[c]["bins"][prim].get("by_correction_strength")
            if b:
                has = True
                for i, x in enumerate(b):
                    for k in ("n", "W", "H", "U"):
                        tot[i][k] += x[k]
        if has:
            for t in tot:
                t["intervention_rate"] = (t["W"] + t["H"] + t["U"]) / max(t["n"], 1)
                t["signed_utility"] = (t["W"] - t["H"]) / max(t["n"], 1)
            bins["strength"] = tot
        rec["binned_utility_corruption_cells_pooled"] = bins
        clean = m["clean"]
        rec["cost"] = {"bank_query_seconds_per_10k": clean["bank_query_seconds_per_10k_queries"], "bank_bytes": clean["storage_bytes_banks"],
                       "fit_params_per_readout": 10000}
        out["per_seed"][s] = rec
    os.makedirs(f"{ROOT}/aggregate", exist_ok=True)
    json.dump(out, open(f"{ROOT}/aggregate/extra_diagnostics.json", "w"), indent=1, default=float)
    for s, r in out["per_seed"].items():
        print("seed", s, "anchor", r["anchor"], r["selection"])
        print(" rank/margin:", {k: (round(v["gt_rank"], 3), None if v["top2_prob_margin"] is None else round(v["top2_prob_margin"], 3)) for k, v in r["mean_gt_rank_and_top2margin_over_12_cells"].items()})
        print(" cost:", r["cost"]["bank_query_seconds_per_10k"])
        for p in ("proc::hidden_decision",):
            print(" margin-bin utility", p, [(b["n"], round(b["intervention_rate"], 3), round(1e4 * b["signed_utility"], 1)) for b in r["binned_utility_corruption_cells_pooled"]["margin"][p.replace("proc::", "")]])
        if r["binned_utility_corruption_cells_pooled"]["strength"]:
            print(" strength-bin utility", [(b["n"], round(b["intervention_rate"], 3), round(1e4 * b["signed_utility"], 1)) for b in r["binned_utility_corruption_cells_pooled"]["strength"]])
        print(" ranks:", {k: r["feature_rank_summary"][k]["phi_fit"]["participation_ratio"].__round__(1) for k in r["feature_rank_summary"]})
        print(" distortion:", {k: (round(v["ratio_mean"], 3), round(v["ratio_std"], 3), round(v["rel_sq_dist_error_p95"], 3)) for k, v in r["projection_distortion"].items()})


if __name__ == "__main__":
    main()
