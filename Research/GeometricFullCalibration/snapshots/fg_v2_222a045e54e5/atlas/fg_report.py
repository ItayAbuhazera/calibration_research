"""CPU report for the fixed-gate study: tables, paired image-level bootstrap, learning curves, utility bins, figures.
  python -m atlas.fg_report
Reads results/fixed_gate/seed{2,4}/{gates_frozen,gate_generalization,results,temperatures}.json + per_sample_*.npz. Writes results/fixed_gate/report/.
Descriptive only: per-checkpoint paired bootstrap over the 10 000 original image IDs (all conditions and methods of an image move together);
not selection-adjusted, cells are not IID, two checkpoints do not establish robustness.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import common, fg_data, fg_gate as FG, spec

OUT = f"{fg_data.FG}/report"
SEEDS = (2, 4)
B = 2000
CELLS = spec.CELLS
FAMS = FG.FAMILIES


def pn(c, fam, o="full", n=2500):
    return f"{c}|{fam}|o{o}|n{n}"


def load(seed):
    root = f"{fg_data.FG}/seed{seed}"
    d = {"root": root, "res": json.load(open(f"{root}/results.json")), "fz": json.load(open(f"{root}/gates_frozen.json"))["gates"],
         "gen": json.load(open(f"{root}/gate_generalization.json")), "T": json.load(open(f"{root}/temperatures.json")), "meta": json.load(open(f"{root}/eval_meta.json"))}
    d["ps"] = {c: np.load(f"{root}/per_sample_{c}.npz") for c in spec.CONDITIONS}
    return d


def macro(res, name, var, key, cells=CELLS):
    return float(np.mean([res[name][var][c][key] for c in cells]))


def macro_vs(res, name, key, cells=CELLS):
    return float(np.mean([res[name]["raw"][c]["vs_base"][key] for c in cells]))


def pooled_wh(res, name, cells=CELLS):
    W = sum(res[name]["raw"][c]["vs_base"]["W"] for c in cells); H = sum(res[name]["raw"][c]["vs_base"]["H"] for c in cells)
    U = sum(res[name]["raw"][c]["vs_base"]["U"] for c in cells)
    nerr = sum(res[name]["raw"][c]["vs_base"]["n_base_errors"] for c in cells); N = sum(res[name]["raw"][c]["vs_base"]["n"] for c in cells)
    return {"W": W, "H": H, "U": U, "coverage": (W + H + U) / N, "W/(W+H+U)": W / (W + H + U) if W + H + U else None, "W/(W+H)": W / (W + H) if W + H else None,
            "rescued_of_base_errors": W / nerr, "harm_prob": H / N, "net_utility": (W - H) / N}


def per_image(d, name, kind, cells):
    """per-image (mean over `cells`) of correctness or -log q_true, index = original test-image ID."""
    if kind == "acc":
        return np.mean([d["ps"][c][f"pred__{name}"].astype(int) == d["ps"][c]["labels"].astype(int) for c in cells], axis=0)
    return np.mean([-d["ps"][c][f"lqy_{kind}__{name}"].astype(float) for c in cells], axis=0)


def boot(diff, seed):
    rng = np.random.default_rng(20261010 + seed)
    n = len(diff); idx = rng.integers(0, n, (B, n))
    b = diff[idx].mean(1)
    return {"estimate": float(diff.mean()), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))]}


CONTRASTS = [("Z1-Z0 (primary, deep n2500)", pn("deep", "Z1"), pn("deep", "Z0")), ("C1-C0 (deep)", pn("deep", "C1"), pn("deep", "C0")), ("Z0-C0 (deep)", pn("deep", "Z0"), pn("deep", "C0")),
             ("deep Z1 - never(native_dac)", pn("deep", "Z1"), "native_dac"), ("deep Z1 - base", pn("deep", "Z1"), "base"), ("deep Z1 - VS", pn("deep", "Z1"), "vs"), ("deep Z1 - MS", pn("deep", "Z1"), "ms"),
             ("deep Z0 - never", pn("deep", "Z0"), "native_dac"), ("deep C1 - never", pn("deep", "C1"), "native_dac"), ("deep C0 - never", pn("deep", "C0"), "native_dac"),
             ("deep Z1 - out Z1 (matched output-candidate gate)", pn("deep", "Z1"), pn("out", "Z1")), ("deep Z1 - layer4 Z1", pn("deep", "Z1"), pn("gap4", "Z1")),
             ("out Z1 - never", pn("out", "Z1"), "native_dac"), ("layer4 Z1 - never", pn("gap4", "Z1"), "native_dac"), ("out Z1 - Z0", pn("out", "Z1"), pn("out", "Z0")),
             ("deep always - base", "deep_always", "base"), ("deep alone - base", "deep_alone", "base"), ("VS - base", "vs", "base"), ("MS - base", "ms", "base"), ("never(native_dac) - base", "native_dac", "base")]


def bootstrap_all(D):
    out = {}
    for seed, d in D.items():
        r = {}
        for lab, a, b in CONTRASTS:
            r[lab] = {}
            for kind, k2 in (("acc", "acc_pp"), ("raw", "nll_raw"), ("T", "nll_T")):
                for scope, cells in (("macro12", CELLS), ("clean", ["clean"])):
                    diff = per_image(d, a, kind, cells) - per_image(d, b, kind, cells)
                    bb = boot(diff, seed); scale = 100 if kind == "acc" else 1
                    r[lab][f"{k2}:{scope}"] = {"estimate": bb["estimate"] * scale, "ci95": [x * scale for x in bb["ci95"]]}
        out[seed] = r
    return out


def training_table(D):
    rows = []
    for seed, d in D.items():
        for c in fg_data.CANDS:
            for fam in FAMS:
                g = d["fz"][pn(c, fam)]; sel = g["selected"]; lam = sel["lambda"]
                mdl = g["models"].get(str(lam)) if lam is not None else None
                rows.append({"seed": seed, "candidate": c, "family": fam, **g["counts"], "decisive_W+H": g["counts"]["W"] + g["counts"]["H"], "features": g["design"]["n_features"],
                             "fitted_params_incl_intercept": (g["design"]["n_features"] + 1) if g["design"]["n_features"] else None, "design_rank": g["design"]["design_rank_with_intercept"],
                             "selected": sel, "selected_df": (mdl or {}).get("df"), "constant_features": g["design"]["constant_features"]})
    return rows


def learning_curve(D):
    rows = []
    for seed, d in D.items():
        for fam in FAMS:
            for o in [0, 1, 2, "full"]:
                for n in ((625, 1250) if o != "full" else (2500,)):
                    nm = pn("deep", fam, o, n); g = d["fz"][nm]; sel = g["selected"]; lam = sel["lambda"]
                    mdl = g["models"].get(str(lam)) if lam is not None else None
                    gen = d["gen"][nm]
                    rows.append({"seed": seed, "family": fam, "order": o, "n": n, **g["counts"], "selected": sel, "design_rank": g["design"]["design_rank_with_intercept"],
                                 "train_mse": (mdl or {}).get("train_mse"), "train_mse_const": (mdl or {}).get("train_mse_const"), "df": (mdl or {}).get("df"),
                                 "calibration": gen["calibration"], "clean_test": gen["clean_test"], "selection": gen["selection"], "fit": gen["fit"],
                                 "macro12_net_utility": macro_vs(d["res"], nm, "net_utility"), "macro12_coverage": macro_vs(d["res"], nm, "intervention_rate"), **{f"macro12_{k}": v for k, v in pooled_wh(d["res"], nm).items() if k in ("W", "H", "U")}})
    return rows


def pipeline_table(D):
    rows = []
    for seed, d in D.items():
        res = d["res"]
        names = ["base", "native_dac", "ts", "vs", "ms"] + [f"{c}_alone" for c in fg_data.CANDS] + [f"{c}_always" for c in fg_data.CANDS] + [pn(c, f) for c in fg_data.CANDS for f in FAMS]
        for nm in names:
            r = {"seed": seed, "pipeline": nm, "T": d["T"][nm]}
            for var in ("raw", "T"):
                r[f"clean_{var}"] = {k: res[nm][var]["clean"][k] for k in ("accuracy", "nll", "brier_sum", "ece15", "gt_rank_mean")}
                r[f"macro12_{var}"] = {k: macro(res, nm, var, k) for k in ("accuracy", "nll", "brier_sum", "ece15", "gt_rank_mean")}
            r["clean_vs_base"] = {k: res[nm]["raw"]["clean"]["vs_base"][k] for k in ("W", "H", "U", "net_utility", "intervention_rate", "intervention_precision", "decisive_precision", "rescued_fraction_of_base_errors")}
            r["macro12_vs_base"] = pooled_wh(res, nm)
            r["macro12_dacc_pp"] = 100 * (macro(res, nm, "raw", "accuracy") - macro(res, "base", "raw", "accuracy"))
            r["clean_dacc_pp"] = 100 * (res[nm]["raw"]["clean"]["accuracy"] - res["base"]["raw"]["clean"]["accuracy"])
            r["per_cell_dacc_pp"] = {c: 100 * (res[nm]["raw"][c]["accuracy"] - res["base"]["raw"][c]["accuracy"]) for c in spec.CONDITIONS}
            if nm in d["gen"]:
                r["roles"] = {k: {kk: d["gen"][nm][k][kk] for kk in ("m", "interventions", "coverage", "W", "H", "U", "net_gain")} for k in ("fit", "selection", "calibration")}
            rows.append(r)
    return rows


def utility_bins(D):
    """Quintile bins of the primary gate's predicted utility (edges from clean-SELECTION disagreements), signed utility on clean test / corruption families."""
    from . import fg_run
    out = {}
    for seed, d in D.items():
        ctx = fg_data.load_ctx(seed)
        out[seed] = {}
        for c in fg_data.CANDS:
            nm = pn(c, "Z1"); g = d["fz"][nm]
            if g["selected"]["kind"] == "never":
                out[seed][c] = {"note": "selected config is never-intervene: no scores"}; continue
            cand = fg_data.load_candidate(ctx, c, sets=("val",))
            sel = ctx["roles"]["selection"]
            X = FG.features("Z1", ctx["Z"]["val"][sel], ctx["base_pred"]["val"][sel], cand["val"]["j"][sel], cand["val"]["pg"][sel], cand["val"]["kth"][sel], cand["val"]["qnorm"][sel])
            mdl = FG.deser(g["models"][str(g["selected"]["lambda"])]); u = FG.predict(mdl, X)
            dis = cand["val"]["j"][sel] != ctx["base_pred"]["val"][sel]
            edges = np.quantile(u[dis], [0, .2, .4, .6, .8, 1.0]); edges[0], edges[-1] = -np.inf, np.inf
            def binstats(conds):
                acc = {b: {"n": 0, "W": 0, "H": 0, "U": 0} for b in range(5)}
                for cnd in conds:
                    z = d["ps"][cnd]; sc = z[f"score__{nm}"].astype(float); j = z[f"cand_j__{c}"].astype(int); i = z["base_pred"].astype(int); y = z["labels"].astype(int)
                    m = j != i; D_ = (j == y).astype(int) - (i == y).astype(int)
                    b = np.clip(np.searchsorted(edges, sc, side="right") - 1, 0, 4)
                    for k in range(5):
                        mk = m & (b == k)
                        acc[k]["n"] += int(mk.sum()); acc[k]["W"] += int((mk & (D_ > 0)).sum()); acc[k]["H"] += int((mk & (D_ < 0)).sum()); acc[k]["U"] += int((mk & (D_ == 0)).sum())
                return {k: {**v, "mean_D": ((v["W"] - v["H"]) / v["n"]) if v["n"] else None} for k, v in acc.items()}
            out[seed][c] = {"edges": [None if np.isinf(e) else float(e) for e in edges], "clean_test": binstats(["clean"]),
                            **{fam: binstats([x for x in CELLS if x.startswith(fam)]) for fam in spec.DEV_CORRUPTIONS}}
            # selection-row own signed utility for the same bins
            Dsel = (cand["val"]["j"][sel] == ctx["val_y"][sel]).astype(int) - (ctx["base_pred"]["val"][sel] == ctx["val_y"][sel]).astype(int)
            bsel = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, 4)
            out[seed][c]["selection_rows"] = {k: {"n": int((dis & (bsel == k)).sum()), "mean_D": float(Dsel[dis & (bsel == k)].mean()) if (dis & (bsel == k)).any() else None} for k in range(5)}
    return out


def figures(D, LC, PT, UB):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    col = {"C0": "#1f77b4", "C1": "#2ca02c", "Z0": "#ff7f0e", "Z1": "#d62728"}
    for xk in ("n", "m"):
        fig, ax = plt.subplots(2, 6, figsize=(26, 8))
        for r, seed in enumerate(SEEDS):
            panels = [("held-out MSE / const-MSE (calibration rows)", lambda x: (x["calibration"].get("mse_all_disagreements") or np.nan) / (x["calibration"].get("mse_const_fit_mean") or np.nan)),
                      ("train MSE (fit disagreements)", lambda x: x["train_mse"] if x["train_mse"] is not None else np.nan),
                      ("coverage (calibration rows)", lambda x: x["calibration"]["coverage"]),
                      ("net gain (calibration rows), pp", lambda x: 100 * x["calibration"]["net_gain"]),
                      ("net gain (clean test), pp", lambda x: 100 * x["clean_test"]["net_gain"]),
                      ("harms H / rescues W (clean test)", lambda x: x["clean_test"]["H"] - x["clean_test"]["W"])]
            for k, (t, f) in enumerate(panels):
                a = ax[r, k]
                for fam in FAMS:
                    for o in (0, 1, 2):
                        pts = [x for x in LC if x["seed"] == seed and x["family"] == fam and x["order"] in (o, "full")]
                        pts = sorted(pts, key=lambda x: x[xk])
                        a.plot([p[xk] for p in pts], [f(p) for p in pts], "-o", color=col[fam], alpha=.55, ms=4, label=fam if o == 0 else None)
                a.set_title(f"seed {seed}: {t}", fontsize=8); a.set_xlabel("n (fit rows)" if xk == "n" else "m (fit disagreements)"); a.axhline(0, color="k", lw=.5)
                if xk == "n":
                    a.set_xscale("log")
        ax[0, 0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{OUT}/figures/learning_curve_vs_{xk}.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    for a, seed in zip(ax, SEEDS):
        rows = [r for r in PT if r["seed"] == seed]
        names = [r["pipeline"] for r in rows if r["pipeline"] not in ("base",)]
        vals = [r["macro12_dacc_pp"] for r in rows if r["pipeline"] != "base"]
        a.barh(range(len(names)), vals, color=["#d62728" if n.startswith("deep|") else "#1f77b4" if n.startswith("out|") else "#2ca02c" if n.startswith("gap4|") else "#888" for n in names])
        a.set_yticks(range(len(names))); a.set_yticklabels(names, fontsize=6); a.axvline(0, color="k", lw=.5); a.set_title(f"seed {seed}: 12-cell macro accuracy change vs base (pp), raw decision")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/macro_accuracy_change.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for r, seed in enumerate(SEEDS):
        for k, c in enumerate(fg_data.CANDS):
            u = UB[seed][c]
            if "edges" not in u:
                ax[r, k].set_title(f"seed {seed} {c}: never"); continue
            for lab, sty in (("clean_test", "k-o"), ("gaussian_noise", "r--s"), ("defocus_blur", "b--s"), ("fog", "g--s"), ("jpeg_compression", "m--s"), ("selection_rows", "k:x")):
                v = [u[lab][str(b) if str(b) in u[lab] else b]["mean_D"] for b in range(5)] if lab in u else [np.nan] * 5
                ax[r, k].plot(range(5), [np.nan if x is None else x for x in v], sty, label=lab, ms=4)
            ax[r, k].axhline(0, color="k", lw=.5); ax[r, k].set_title(f"seed {seed} {c} Z1: mean utility per selection-derived score bin (low->high)", fontsize=8)
    ax[0, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{OUT}/figures/utility_bins.png", dpi=110); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    D = {s: load(s) for s in SEEDS}
    TT, LC, PT = training_table(D), learning_curve(D), pipeline_table(D)
    BS = bootstrap_all(D); UB = utility_bins(D)
    for name, obj in (("training_events", TT), ("learning_curve", LC), ("pipelines", PT), ("bootstrap", BS), ("utility_bins", UB)):
        common.atomic_json(f"{OUT}/{name}.json", obj)
    figures(D, LC, PT, UB)
    print("report written", OUT)


if __name__ == "__main__":
    main()
