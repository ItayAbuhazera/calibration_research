"""POST-HOC, secondary and descriptive analyses declared in the card's pre-results addendum
(2026-09-24; no decision attached; run after the frozen R1-R3 verdict is reported).

  1. per source: standalone accuracy, disagreement with argmax Z, both-wrong rate (clean and per cell)
  2. G(layer3.22) - G(other-checkpoint logits), G = Delta_T - Delta_S (12-cell macro), 8k x 1 and 2.5k x 1,
     per checkpoint, paired image-group bootstrap (one shared resample of the combined per-image array)
  3. across the 13 sources: S-8k x 1 clean increment vs Delta_S / Delta_T at 8k x 1, scatter + Spearman

    python -m atlas.stage0_ablation_secondary
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr

from . import spec, stage0_data
from .stage0_aggregate import BOOT_SEED, group_id_for_bootstrap, grouped_bootstrap
from .stage0_ablation_aggregate import LAYERS, OUT, REF, diff_arrays, pool

SOURCES13 = LAYERS + ["xckpt"]


def overlap(evidence, seed):
    cells = stage0_data.load_all_cells(seed, None if evidence == REF else evidence)
    res = {}
    for c in spec.CONDITIONS:
        d = cells[c]
        pw, zw = d["p"].argmax(1) != d["labels"], d["z"].argmax(1) != d["labels"]
        res[c] = {"acc_pct": 100 * float((~pw).mean()), "disagree_with_Z": float((d["p"].argmax(1) != d["z"].argmax(1)).mean()),
                  "both_wrong": float((pw & zw).mean())}
    res["macro12"] = {k: float(np.mean([res[c][k] for c in spec.CELLS])) for k in ("acc_pct", "disagree_with_Z", "both_wrong")}
    return res


def main():
    gid = group_id_for_bootstrap()
    out = {"label": "post-hoc secondary descriptive; no decision attached"}
    out["item1_overlap"] = {ev: {str(s): overlap(ev, s) for s in spec.DEV_SEEDS} for ev in SOURCES13}

    out["item2_gap_comparison"] = {}
    tables = {}
    for seed in spec.DEV_SEEDS:
        for ev in SOURCES13:
            pooled = {rg: pool(ev, seed, rg) for rg in ("T-8k1", "S-8k1", "T-2.5k1", "S-2.5k1") if ev in ("L8", "xckpt", "L11") or rg in ("T-8k1", "S-8k1")}
            tables[(ev, seed)] = {rg: (diff_arrays(p) if p is not None else None) for rg, p in pooled.items()}
    for seed in spec.DEV_SEEDS:
        res = {}
        for tag, t, s_ in (("8k1", "T-8k1", "S-8k1"), ("2.5k1", "T-2.5k1", "S-2.5k1")):
            g = {ev: tables[(ev, seed)][t][0] - tables[(ev, seed)][s_][0] for ev in (REF, "xckpt")}
            b = grouped_bootstrap(g[REF] - g["xckpt"], gid, seed=BOOT_SEED + 9000 + seed)
            res[tag] = {"G_ref_pp": 100 * float(g[REF].mean()), "G_xckpt_pp": 100 * float(g["xckpt"].mean()),
                        "G_ref_minus_G_xckpt_pp": 100 * b["estimate"], "ci95_pp": [100 * x for x in b["ci95"]]}
        out["item2_gap_comparison"][str(seed)] = res

    out["item3_clean_increment_vs_share"] = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, seed in zip(axes, spec.DEV_SEEDS):
        pts = {}
        for ev in SOURCES13:
            T, S = tables[(ev, seed)]["T-8k1"], tables[(ev, seed)]["S-8k1"]
            dT, dS, clean = 100 * float(T[0].mean()), 100 * float(S[0].mean()), 100 * float(S[1].mean())
            pts[ev] = {"delta_T_pp": dT, "delta_S_pp": dS, "S_clean_increment_pp": clean, "share_dS_over_dT": dS / dT if dT else None,
                       "delta_T_below_0.5pp_ratio_unstable": dT < 0.5}
        xs = np.array([p["S_clean_increment_pp"] for p in pts.values()]); ys = np.array([p["share_dS_over_dT"] for p in pts.values()], float)
        r = spearmanr(xs, ys)
        out["item3_clean_increment_vs_share"][str(seed)] = {"points": pts, "spearman": float(r.correlation), "spearman_p_descriptive": float(r.pvalue), "n": len(pts)}
        for ev, p in pts.items():
            unstable = p["delta_T_below_0.5pp_ratio_unstable"]
            ax.scatter(p["S_clean_increment_pp"], p["share_dS_over_dT"], s=28, marker="o" if not unstable else "x",
                       color="#D55E00" if ev == "xckpt" else ("#0072B2" if ev == REF else "#555555"))
            ax.annotate({"L8": "layer3.22", "xckpt": "other ckpt"}.get(ev, ev), (p["S_clean_increment_pp"], p["share_dS_over_dT"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="#999999", lw=0.8); ax.axvline(0, color="#999999", lw=0.8)
        ax.set_title(f"checkpoint {seed}: Spearman {r.correlation:+.2f} (n=13, descriptive)", fontsize=9)
        ax.set_xlabel("S-8k×1 clean-view increment (pp)")
    axes[0].set_ylabel("Δ_S / Δ_T at 8k×1 (x = Δ_T < 0.5 pp, unstable)")
    fig.tight_layout(); fig.savefig(f"{OUT}/ablation_clean_increment_vs_share.png", dpi=150)
    json.dump(out, open(f"{OUT}/ablation_secondary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "item1_overlap"}, indent=1))


if __name__ == "__main__":
    main()
