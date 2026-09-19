#!/usr/bin/env python3
import os, json, argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- IO helpers ----------
def load_metric_doc(md_path: str) -> pd.DataFrame:
    with open(md_path, "r") as f:
        j = json.load(f)
    rows = []
    cand_layers = j.get("candidate_layers", [])
    holdout_map = j.get("holdout_ece_per_layer", {})
    best_ece = float(j.get("best_holdout_ece", np.inf))
    metrics = j.get("metrics_topk", {})
    for mname, md in metrics.items():
        rows.append({
            "metric": mname,
            "direction": md.get("direction"),
            "top1_layer": int(md.get("top1_layer", -1)),
            "topk_layers": md.get("topk_layers", []),
            "chosen_metric_value": float(md.get("chosen_metric_value", np.nan)),
            "holdout_ece_top1": float(md.get("holdout_ece_top1", np.nan)),
            "gap_from_best": float(md.get("holdout_ece_gap_from_best", np.nan)),
            "num_candidates": len(cand_layers),
            "best_ece_valB": float(best_ece),
        })
    # per-layer holdout ECE (for correlation plots)
    per_layer = []
    for L, e in holdout_map.items():
        per_layer.append({"layer_idx": int(L), "holdout_ece_valB": float(e)})
    per_layer_df = pd.DataFrame(per_layer)
    return pd.DataFrame(rows), per_layer_df

def load_ensemble_results(er_path: str) -> pd.DataFrame:
    with open(er_path, "r") as f:
        j = json.load(f)
    rows = []
    for key, val in j.items():
        if key == "baselines":
            # flatten baselines into rows
            for bname, b in val.items():
                if isinstance(b, dict) and "ece" in b:
                    rows.append({
                        "kind": "baseline",
                        "method": bname,
                        "k": None,
                        "test_ece": float(b.get("ece", np.nan)),
                        "test_acc": float(b.get("acc", np.nan)),
                        "test_brier": float(b.get("brier", np.nan)),
                    })
            continue
        if not isinstance(val, dict): 
            continue
        # key looks like: top3_uniform, top5_learned, all_deep, etc.
        m = re.match(r"(top(\d+)|all)_(\w+)", key)
        if not m: 
            continue
        k = None if m.group(1) == "all" else int(m.group(2))
        method = m.group(3)
        rows.append({
            "kind": "ensemble",
            "method": method,
            "k": k,
            "test_ece": float(val.get("test_ece", np.nan)),
            "test_acc": float(val.get("test_accuracy", np.nan)),
            "test_brier": float(val.get("test_brier", np.nan)),
        })
    return pd.DataFrame(rows)

def find_experiments(root: str):
    # metric docs live in .../metric_docs/metric_docs_and_topk.json
    md_paths = glob.glob(os.path.join(root, "**", "metric_docs", "metric_docs_and_topk.json"), recursive=True)
    exps = []
    for md in md_paths:
        exp_dir = str(Path(md).parents[2])  # go up to .../multi_layer_ensemble
        er = os.path.join(exp_dir, "ensemble_results", "ensemble_results.json")
        # some setups save ensemble_results.json directly in exp_dir
        if not os.path.exists(er):
            er = os.path.join(exp_dir, "ensemble_results.json")
        exps.append({"exp_dir": exp_dir, "metric_doc": md, "ensemble_res": er if os.path.exists(er) else None})
    return exps

# ---------- Plots ----------
def bar_win_counts(df_metrics, gap_thresh, out_path):
    # count how often metric’s gap_from_best <= thresh
    win_df = (df_metrics.assign(hit = df_metrics["gap_from_best"] <= gap_thresh)
                        .groupby("metric")["hit"].mean().sort_values(ascending=False))
    plt.figure(figsize=(10, 5))
    win_df.plot(kind="bar")
    plt.ylabel(f"Hit rate (gap ≤ {gap_thresh})")
    plt.title("Metric Top-1 ≈ Oracle (Val-B) – Hit Rate")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return win_df.reset_index().rename(columns={"hit":"hit_rate"})

def scatter_metric_vs_ece(df_metrics, metric_name, out_path):
    sub = df_metrics[df_metrics["metric"] == metric_name]
    if sub.empty:
        return False
    x = sub["chosen_metric_value"].values
    y = sub["holdout_ece_top1"].values
    if len(x) < 2:
        return False
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y, nan_policy="omit")
    plt.figure()
    plt.scatter(x, y, s=20)
    plt.xlabel(f"{metric_name} (chosen value)")
    plt.ylabel("Holdout ECE (Val-B)")
    plt.title(f"{metric_name} vs Val-B ECE (Spearman ρ={rho:.3f})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True

def line_ece_vs_k(df_ens, out_path):
    sub = df_ens[df_ens["kind"] == "ensemble"].copy()
    if sub.empty: 
        return False
    # for plotting, replace None (all layers) with max(k)+ for ordering
    ks = sorted([k for k in sub["k"].dropna().unique()])
    has_all = sub["k"].isna().any()
    order = ks + (["all"] if has_all else [])
    plt.figure()
    for method, g in sub.groupby("method"):
        # compute mean per k in case multiple runs per exp
        series = []
        for k in order:
            if k == "all":
                v = g[g["k"].isna()]["test_ece"].mean()
            else:
                v = g[g["k"] == k]["test_ece"].mean()
            series.append(v)
        xs = list(range(len(order)))
        plt.plot(xs, series, marker="o", label=method)
    plt.xticks(range(len(order)), order)
    plt.xlabel("K (top-k layers)")
    plt.ylabel("Test ECE")
    plt.title("Ensemble Test ECE vs K by Weighting Method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs", help="Root that contains experiments")
    ap.add_argument("--gap", type=float, default=0.002, help="ΔECE threshold for a hit (Val-B)")
    ap.add_argument("--out", type=str, default="analysis_out", help="Where to save plots/tables")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    exps = find_experiments(args.root)
    all_metric_rows, all_perlayer_rows, all_ens_rows = [], [], []
    index_rows = []

    for e in exps:
        exp_id = Path(e["exp_dir"]).as_posix().replace(args.root.rstrip("/"), "").strip("/")
        md_path = e["metric_doc"]
        dfm, dflayers = load_metric_doc(md_path)
        dfm["exp_id"] = exp_id
        dflayers["exp_id"] = exp_id
        all_metric_rows.append(dfm)
        all_perlayer_rows.append(dflayers)
        if e["ensemble_res"] and os.path.exists(e["ensemble_res"]):
            dfe = load_ensemble_results(e["ensemble_res"])
            dfe["exp_id"] = exp_id
            all_ens_rows.append(dfe)
        index_rows.append({"exp_id": exp_id, "metric_doc": md_path, "ensemble_res": e["ensemble_res"]})

    if not all_metric_rows:
        print("No metric_docs found.")
        return

    df_metrics = pd.concat(all_metric_rows, ignore_index=True)
    df_layers = pd.concat(all_perlayer_rows, ignore_index=True) if all_perlayer_rows else pd.DataFrame()
    df_ens = pd.concat(all_ens_rows, ignore_index=True) if all_ens_rows else pd.DataFrame()
    df_index = pd.DataFrame(index_rows)

    # Save raw tables
    df_index.to_csv(os.path.join(args.out, "experiments_index.csv"), index=False)
    df_metrics.to_csv(os.path.join(args.out, "metrics_top1_summary.csv"), index=False)
    if not df_layers.empty:
        df_layers.to_csv(os.path.join(args.out, "per_layer_valB_ece.csv"), index=False)
    if not df_ens.empty:
        df_ens.to_csv(os.path.join(args.out, "ensemble_results_flat.csv"), index=False)

    # Global win counts plot
    wins = bar_win_counts(df_metrics, args.gap, os.path.join(args.out, "metric_win_rates.png"))
    wins.to_csv(os.path.join(args.out, "metric_win_rates.csv"), index=False)

    # Example correlation plots for a few key metrics (make if present)
    for metric_name in ["kfold_ece_utility", "spearman_stability_accuracy", "confidence_distance_correlation"]:
        ok = scatter_metric_vs_ece(df_metrics, metric_name, os.path.join(args.out, f"{metric_name}_vs_valB_ece.png"))

    # ECE vs K lines (test) if ensembles exist
    if not df_ens.empty:
        _ = line_ece_vs_k(df_ens, os.path.join(args.out, "test_ece_vs_k_by_method.png"))

        # Leaderboard: which K/method wins per experiment
        def pick_winner(g):
            best = g.loc[g["test_ece"].idxmin()]
            return pd.Series({
                "winner_method": best["method"],
                "winner_k": best["k"],
                "winner_ece": best["test_ece"]
            })
        winners = (df_ens[df_ens["kind"]=="ensemble"]
                   .groupby("exp_id").apply(pick_winner).reset_index())
        winners.to_csv(os.path.join(args.out, "ensemble_winners_by_experiment.csv"), index=False)

        # Aggregated win rates by method and by K
        agg_method = winners.groupby("winner_method")["exp_id"].count().sort_values(ascending=False)
        agg_method.to_csv(os.path.join(args.out, "ensemble_method_win_counts.csv"))
        agg_k = winners["winner_k"].value_counts(dropna=False).sort_index()
        agg_k.to_csv(os.path.join(args.out, "ensemble_k_win_counts.csv"))

        # Simple bars
        plt.figure()
        agg_method.plot(kind="bar")
        plt.ylabel("#Experiments won")
        plt.title("Ensemble winners by weighting method (Test)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "ensemble_method_win_counts.png"))
        plt.close()

        plt.figure()
        agg_k.plot(kind="bar")
        plt.ylabel("#Experiments won")
        plt.title("Which K wins most often (Test)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "ensemble_k_win_counts.png"))
        plt.close()

    print(f"Saved tables/plots to: {args.out}")

if __name__ == "__main__":
    main()
