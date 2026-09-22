"""Stage REPORT (CPU): complete numerical tables and the mechanical evaluation of the frozen hypotheses (spec §8).

Writes results/atlas/report/{summary.json, tables/*.csv, REPORT.md}. Nothing here selects anything: the "best hidden candidate"
is the clean-selected pooling family with the largest clean-selection net gain (source-only); the "strongest output control" is the
comparator with the highest macro development-corruption accuracy among the declared output controls (a target-informed comparator
choice that can only make the comparison harder for the hidden pipeline; labelled as such).
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from . import data, spec
from . import stage_stats as ss

CELLS = spec.CELLS
OUTPUT_CONTROLS = ("vs", "ms", "out_F0", "out_F1")


def macro(res, n, v, k, cells=CELLS):
    return float(np.mean([res[n][v][c][k] for c in cells]))


def wh_macro(res, n, v, k, cells=CELLS):
    return float(np.mean([res[n][v][c]["wh_vs_base"][k] for c in cells]))


def load_seed(seed):
    root = data.seed_dir(seed)
    d = {"root": root, "sl": json.load(open(f"{root}/shortlist.json")), "ms": json.load(open(f"{root}/metric_selection.json")),
         "cd": json.load(open(f"{root}/cd/results.json")), "meta": json.load(open(f"{root}/cd/meta.json")),
         "target": json.load(open(f"{root}/stats_target_unit_l2.json")), "clean": json.load(open(f"{root}/stats_clean_unit_l2.json"))}
    for f, k in (("latency.json", "lat"), ("viz/done.json", "viz")):
        if os.path.exists(f"{root}/{f}"):
            d[k] = json.load(open(f"{root}/{f}"))
    return d


def best_hidden_pool(S):
    return max(spec.POOLS, key=lambda p: S["sl"]["chosen"][p]["net_utility_selection"])


def unique_counts(seed, h, o):
    root = data.seed_dir(seed)
    W = {"h": [], "o": []}; H = {"h": [], "o": []}
    for c in CELLS:
        z = np.load(f"{root}/cd/per_sample_{c}.npz"); y = z["labels"].astype(int); b = z["base_pred"].astype(int)
        for k, n in (("h", h), ("o", o)):
            p = z[f"pred__{n}"].astype(int)
            W[k].append((p != b) & (b != y) & (p == y)); H[k].append((p != b) & (b == y) & (p != y))
    Wh, Wo, Hh, Ho = (np.concatenate(v) for v in (W["h"], W["o"], H["h"], H["o"]))
    return {"unique_W_h": int((Wh & ~Wo).sum()), "unique_H_h": int((Hh & ~Ho).sum()), "W_h": int(Wh.sum()), "H_h": int(Hh.sum()),
            "W_o": int(Wo.sum()), "H_o": int(Ho.sum()), "n_cells": len(CELLS)}


def hypotheses(D):
    out = {}
    seeds = sorted(D)
    # H-A: spatial information (candidate net utility on atlas subset, macro over 12 cells)
    ha = {}
    for s in seeds:
        S = D[s]
        util = {}
        for p in spec.POOLS:
            site = S["sl"]["chosen"][p]["site"]
            util[p] = float(np.mean([S["target"][site][p][c]["net_utility"] for c in CELLS]))
        ha[s] = {"macro_net_utility": util, "spatial_minus_gap_pp": {p: 100 * (util[p] - util["gap"]) for p in ("grid2", "spp")}}
    out["H-A"] = {"per_seed": ha, "supported": bool(all(max(ha[s]["spatial_minus_gap_pp"].values()) >= 0.5 for s in seeds)),
                  "rule": "clean-selected grid2 or spp beats clean-selected gap by >= 0.5 pp macro corruption net utility in both checkpoints"}
    hb = {}
    for s in seeds:
        S = D[s]; p = best_hidden_pool(S)
        u = unique_counts(s, f"{p}_alone", "out_alone")
        hb[s] = {"best_hidden_pool": p, **u, "unique_net": u["unique_W_h"] - u["unique_H_h"]}
    out["H-B"] = {"per_seed": hb, "supported": bool(all(hb[s]["unique_net"] > 0 for s in seeds)),
                  "rule": "best hidden candidate: unique repairs minus unique harms vs the output-space candidate > 0 in both checkpoints"}
    hc = {}
    for s in seeds:
        S = D[s]; gains = {}
        for p in spec.POOLS:
            t = S["ms"]["hidden"][p]["selection_net_utility"]
            gains[p] = {m: 100 * (t[m] - t["unit_l2"]) for m in ("raw_l2", "mahalanobis")}
        hc[s] = gains
    out["H-C"] = {"per_seed": hc, "supported": bool(all(any(max(g.values()) >= 0.5 for g in hc[s].values()) for s in seeds)),
                  "rule": "raw_l2 or mahalanobis beats unit_l2 on clean-selection net gain by >= 0.5 pp for >= 1 pooling, both checkpoints"}
    hd = {}
    for s in seeds:
        S = D[s]; p = best_hidden_pool(S); cd = S["cd"]
        a = {f: 100 * (macro(cd, f"{p}_{f}", "raw", "accuracy")) for f in ("F0", "F1")}
        o1 = 100 * macro(cd, "out_F1", "raw", "accuracy")
        hd[s] = {"pool": p, "acc_F0": a["F0"], "acc_F1": a["F1"], "F1_minus_F0_pp": a["F1"] - a["F0"], "out_F1": o1, "hidden_F1_minus_out_F1_pp": a["F1"] - o1}
    out["H-D"] = {"per_seed": hd, "supported": bool(all(hd[s]["F1_minus_F0_pp"] >= 0.10 and hd[s]["hidden_F1_minus_out_F1_pp"] > 0 for s in seeds)),
                  "rule": "same hidden candidate: F1 - F0 >= 0.10 pp in both checkpoints and hidden-F1 > output-F1"}
    he = {}
    for s in seeds:
        S = D[s]; p = best_hidden_pool(S); cd = S["cd"]
        strongest = max(OUTPUT_CONTROLS, key=lambda n: macro(cd, n, "T_nll", "accuracy"))
        h = f"{p}_F1"
        he[s] = {"pipeline": h, "strongest_output_control": strongest,
                 "gain_over_base_pp": 100 * (macro(cd, h, "T_nll", "accuracy") - macro(cd, "base", "raw", "accuracy")),
                 "gain_over_strongest_output_pp": 100 * (macro(cd, h, "T_nll", "accuracy") - macro(cd, strongest, "T_nll", "accuracy")),
                 "nll_minus_control": macro(cd, h, "T_nll", "nll") - macro(cd, strongest, "T_nll", "nll"),
                 "brier_minus_control": macro(cd, h, "T_nll", "brier_sum") - macro(cd, strongest, "T_nll", "brier_sum")}
    out["H-E"] = {"per_seed": he, "supported": bool(all(he[s]["gain_over_base_pp"] >= 0.5 and he[s]["gain_over_strongest_output_pp"] >= 0.25 and
                                                        he[s]["nll_minus_control"] <= 0 and he[s]["brier_minus_control"] <= 0 for s in seeds)),
                  "rule": "hidden-F1 post-T >= +0.5 pp over base and >= +0.25 pp over the strongest output control with non-worsening NLL and Brier, both checkpoints"}
    return out


def tables(D):
    os.makedirs("results/atlas/report/tables", exist_ok=True)
    w = lambda n, rows: (lambda f: (csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k not in ("seed", "pipeline", "variant"), k))).writeheader(),  # noqa: E731
                                    csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: (k not in ("seed", "pipeline", "variant"), k))).writerows(rows)))(open(f"results/atlas/report/tables/{n}.csv", "w", newline=""))
    rows, prob, shortl, cond_rows = [], [], [], []
    for s, S in D.items():
        cd = S["cd"]
        for n in cd:
            for v in cd[n]:
                r = {"seed": s, "pipeline": n, "variant": v}
                for k in ("accuracy", "nll", "brier_sum", "top_label_ece", "adaptive_ece", "classwise_ece", "aurc"):
                    r[f"macro12_{k}"] = macro(cd, n, v, k) if cd[n][v]["clean"].get(k) is not None else None
                    r[f"clean_{k}"] = cd[n][v]["clean"].get(k)
                for k in ("W", "H", "U", "total_flips", "net", "intervention_precision", "decisive_precision", "rescued_fraction_of_base_errors"):
                    vals = [cd[n][v][c]["wh_vs_base"][k] for c in CELLS]
                    r[f"sum12_{k}" if k in ("W", "H", "U", "total_flips", "net") else f"mean12_{k}"] = (sum(vals) if k in ("W", "H", "U", "total_flips", "net") else
                                                                                                          (None if any(x is None for x in vals) else float(np.mean(vals))))
                r["gt_rank_base_macro"] = macro(cd, n, v, "gt_rank_mean_base"); r["gt_rank_after_macro"] = macro(cd, n, v, "gt_rank_mean_after")
                prob.append(r)
                for c in spec.CONDITIONS:
                    q = cd[n][v][c]
                    cond_rows.append({"seed": s, "pipeline": n, "variant": v, "condition": c, "accuracy": q["accuracy"], "nll": q["nll"], "brier_sum": q["brier_sum"],
                                      "ece": q["top_label_ece"], **{f"wh_{k}": q["wh_vs_base"][k] for k in ("W", "H", "U", "total_flips", "net")}})
        for p in spec.POOLS:
            c = S["sl"]["chosen"][p]
            shortl.append({"seed": s, "pool": p, "site": c["site"], "dim": c["dim"], "selection_net_utility": c["net_utility_selection"], "all_gains_negative": c["all_gains_negative"],
                           "metric": S["ms"]["hidden"][p]["metric"], **{f"sel_net_{m}": v for m, v in S["ms"]["hidden"][p]["selection_net_utility"].items()}})
    w("pipelines_macro", prob); w("pipelines_by_condition", cond_rows); w("shortlist_and_metrics", shortl)
    return prob


def atlas_tables(D):
    """Layer x pooling atlas summary: where do correct alternatives on base errors come from, at what harm (target-labelled DIAGNOSTIC
    for the corruption columns; the clean-selected candidate is source-only)."""
    out = {}
    for s, S in D.items():
        T, sl = S["target"], S["sl"]["chosen"]
        oracle = json.load(open(f"{S['root']}/target_oracle_ranking_unit_l2.json"))["best_site_by_condition"]
        out[s] = {}
        for p in spec.POOLS:
            rows = []
            for site in spec.SITE_NAMES:
                g = lambda k: float(np.mean([T[site][p][c][k] for c in CELLS]))  # noqa: E731
                rows.append({"site": site, "cand_acc_among_base_errors": g("candidate_acc_among_base_errors"), "harm_rate_among_base_correct": g("harm_rate_among_base_correct"),
                             "net_utility": g("net_utility"), "cand_acc": g("cand_acc"), "geo_nll": g("geo_nll"), "geo_brier": g("geo_brier"),
                             "local_true_label_proportion": g("local_true_label_proportion"), "W": int(sum(T[site][p][c]["W"] for c in CELLS)),
                             "H": int(sum(T[site][p][c]["H"] for c in CELLS)), "U": int(sum(T[site][p][c]["U"] for c in CELLS))})
            best_err = max(rows, key=lambda r: r["cand_acc_among_base_errors"]); best_net = max(rows, key=lambda r: r["net_utility"])
            sel = next(r for r in rows if r["site"] == sl[p]["site"])
            out[s][p] = {"rows": rows, "clean_selected": sel, "target_oracle_best_by_cand_acc_on_errors": best_err, "target_oracle_best_by_net_utility": best_net,
                         "oracle_best_site_by_condition": oracle[p]}
    return out


def output_control(D):
    """Full-logit kNN control (u0), same k/smoothing/query IDs, all three metrics, on the 2000-image subset."""
    from . import stats
    out = {}
    for s, S in D.items():
        root, roles, val_y, bank_y, prior, sub, ty = ss.load_common(s)
        res = {}
        for m in spec.METRICS:
            rows = {}
            for c in spec.CELLS:
                z = np.load(f"{root}/u0/{c}.npz")
                bp = z["logits"][sub].argmax(1)
                pg, cnt = stats.p_geo(z[f"{m}_idx"][sub], bank_y, prior)
                rows[c] = stats.candidate_row(pg, cnt, z[f"{m}_dist"][sub][:, -1], bp, ty[sub])
            res[m] = {k: float(np.mean([rows[c][k] for c in CELLS if rows[c][k] is not None])) for k in
                      ("candidate_acc_among_base_errors", "harm_rate_among_base_correct", "net_utility", "cand_acc", "geo_nll", "geo_brier")}
        out[s] = res
    return out


def metric_target(D):
    """Experiment B on the corruption cells (target-labelled diagnostic; the metric was chosen on clean-selection rows)."""
    from . import stats
    out = {}
    for s, S in D.items():
        root, roles, val_y, bank_y, prior, sub, ty = ss.load_common(s)
        bp = ss.base_logits(root, sub).argmax(1)
        yy = np.concatenate([val_y] + [ty[sub] for _ in spec.CONDITIONS])
        out[s] = {}
        for p in spec.POOLS:
            site = S["sl"]["chosen"][p]["site"]
            zb = np.load(f"{root}/knn_B/{site}__{p}__atlas.npz")
            i0, d0, _ = ss.unit_arrays(root, site, p)
            arr = {"unit_l2": (i0, d0), "raw_l2": (zb["raw_l2_idx"], zb["raw_l2_dist"]), "mahalanobis": (zb["mahalanobis_idx"], zb["mahalanobis_dist"])}
            out[s][p] = {"site": site}
            for m, (idx, dist) in arr.items():
                rr = []
                for ci, c in enumerate(CELLS):
                    rows = np.arange(5000 + (spec.CONDITIONS.index(c)) * 2000, 5000 + (spec.CONDITIONS.index(c) + 1) * 2000)
                    rr.append(ss.row_for(idx, dist, bank_y, prior, bp, yy, rows))
                out[s][p][m] = {k: float(np.mean([r[k] for r in rr if r[k] is not None])) for k in
                                ("candidate_acc_among_base_errors", "harm_rate_among_base_correct", "net_utility", "cand_acc", "geo_nll", "kth_radius_mean")}
    return out


def main():
    D = {s: load_seed(s) for s in spec.DEV_SEEDS}
    H = hypotheses(D)
    prob = tables(D)
    AT, OC, MT = atlas_tables(D), output_control(D), metric_target(D)
    summary = {"hypotheses": H, "atlas_tables": AT, "output_space_knn_control": OC, "metric_comparison_on_corruptions_diagnostic": MT, "shortlist": {s: D[s]["sl"]["chosen"] for s in D},
               "metric_selection": {s: {"hidden": {p: D[s]["ms"]["hidden"][p]["metric"] for p in spec.POOLS}, "output": D[s]["ms"]["output"]["metric"]} for s in D},
               "gate_selection": {s: {k: {"selected": v["selected"], "n_disagree_fit": v["fit_counts"]["n_disagree"], "n_useful_fit": v["fit_counts"]["n_useful"],
                                          "n_harmful_fit": v["fit_counts"]["n_harmful"]} for k, v in D[s]["meta"]["gates"].items()} for s in D},
               "temperatures": {s: D[s]["meta"]["temperatures"] for s in D}, "latency": {s: D[s].get("lat") for s in D}}
    os.makedirs("results/atlas/report", exist_ok=True)
    json.dump(summary, open("results/atlas/report/summary.json", "w"), indent=1, default=float)
    lines = ["# Atlas program: mechanical hypothesis evaluation (development, checkpoints 2 and 4)", ""]
    for k, v in H.items():
        lines.append(f"* **{k}**: {'SUPPORTED' if v['supported'] else 'refuted'} — {v['rule']}")
        for s, x in v["per_seed"].items():
            lines.append(f"    * seed {s}: `{json.dumps(x, default=float)}`")
    open("results/atlas/report/REPORT.md", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
