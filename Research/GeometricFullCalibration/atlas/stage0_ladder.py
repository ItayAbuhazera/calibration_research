"""POST-HOC DESCRIPTIVE (not in the frozen protocol; no new fits): absolute accuracy/NLL/Brier
ladder per cell and checkpoint from stored out-of-fold predictions and the base logits, plus the
0a/0b figures from existing report JSON.

    base | T-8k12 q_Z | T-8k12 q_ZP | S-8k1 q_Z | S-8k1 q_ZP
"T-fitted q_Z vs base" is target recalibration alone; "q_ZP vs q_Z" within a regime is the added
layer3.22 probe evidence; S rows are clean-fitted. Read with docs/stage0_execution_spec.md.

    python -m atlas.stage0_ladder
"""
from __future__ import annotations

import json
import os

import numpy as np

from utils.unified_metrics import evaluate_all
from . import spec, stage0_data
from .stage0_aggregate import OUT, pool_oof

ROWS = [("base", None, None), ("T-8k12 q_Z", "T-8k12", "q_Z"), ("T-8k12 q_ZP", "T-8k12", "q_ZP"),
        ("S-8k1 q_Z", "S-8k1", "q_Z"), ("S-8k1 q_ZP", "S-8k1", "q_ZP")]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ladder():
    out = {"label": "post-hoc descriptive; from stored OOF predictions and base logits; no new fits"}
    md = ["# Stage 0 post-hoc absolute ladder (accuracy % / NLL / Brier)", "",
          "Post-hoc descriptive. Base = softmax of the canonical FP32 atlas logits. Pooled out-of-fold, 10,000 images per condition.", ""]
    for seed in spec.DEV_SEEDS:
        pooled = {r: pool_oof(seed, r) for r in ("T-8k12", "S-8k1")}
        cells = stage0_data.load_all_cells(seed)
        res = {}
        for cond in spec.CONDITIONS:
            y = cells[cond]["labels"]
            res[cond] = {}
            for name, reg, arm in ROWS:
                probs = softmax(cells[cond]["z"]) if reg is None else pooled[reg][f"probs__{arm}__{cond}"]
                lab = y if reg is None else pooled[reg][f"labels__{cond}"]
                m = evaluate_all(probs, lab, include_selective=False)
                res[cond][name] = {"acc_pct": 100 * m["accuracy"], "nll": m["nll"], "brier": m["brier"]}
        res["macro12"] = {name: {k: float(np.mean([res[c][name][k] for c in spec.CELLS])) for k in ("acc_pct", "nll", "brier")} for name, _, _ in ROWS}
        out[str(seed)] = res
        md += [f"## Checkpoint seed {seed}", "", "| condition | " + " | ".join(n for n, _, _ in ROWS) + " |", "|---|" + "---|" * len(ROWS)]
        for cond in list(spec.CONDITIONS) + ["macro12"]:
            md.append(f"| {cond} | " + " | ".join(f"{res[cond][n]['acc_pct']:.2f} / {res[cond][n]['nll']:.3f} / {res[cond][n]['brier']:.3f}" for n, _, _ in ROWS) + " |")
        md.append("")
    json.dump(out, open(f"{OUT}/stage0_posthoc_accuracy_ladder.json", "w"), indent=1)
    open(f"{OUT}/stage0_posthoc_accuracy_ladder.md", "w").write("\n".join(md))
    return out


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    figdir = f"{OUT}/figures"
    os.makedirs(figdir, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    a0 = json.load(open(f"{OUT}/stage0a_rank_tables.json"))["per_cell"]
    b = json.load(open(f"{OUT}/stage0b_layer_probe_eval.json"))
    colors = {"W": "#0072B2", "H": "#D55E00", "U": "#999999"}  # Okabe-Ito blue / vermillion / gray, fixed order

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, seed in zip(axes, spec.DEV_SEEDS):
        rows = [r for r in a0 if r["seed"] == seed]
        x = np.arange(len(rows)); bottom = np.zeros(len(rows))
        for k in ("W", "H", "U"):
            v = np.array([100 * r[k] / r["n"] for r in rows])
            ax.bar(x, v, bottom=bottom, color=colors[k], width=0.7, label=k, edgecolor="white", linewidth=0.8)
            bottom += v
        ax.set_xticks(x); ax.set_xticklabels([r["cell"].replace("_", " ") for r in rows], rotation=60, ha="right", fontsize=7)
        ax.set_title(f"0a: deep candidate j vs base, checkpoint {seed}")
    axes[0].set_ylabel("% of images"); fig.suptitle("W wrong→correct, H correct→wrong, U wrong→different wrong", fontsize=9, y=1.0)
    axes[0].legend(frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(f"{figdir}/0a_flip_decomposition.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    bucket = ["rank2", "ranks3_5", "rank_gt5"]; lab = ["rank 2", "ranks 3–5", "rank > 5"]
    for ax, seed in zip(axes, spec.DEV_SEEDS):
        rows = [r for r in a0 if r["seed"] == seed and r["cell"] in spec.CELLS]
        j = np.array([[sum(r["rank_of_j_under_base_logits__among_disagreements"][k] for r in rows) for k in bucket]], float)[0]
        t = np.array([[sum(r["true_class_rank_among_all_base_errors"][k] for r in rows) for k in bucket]], float)[0]
        xx = np.arange(3)
        ax.bar(xx - 0.19, 100 * j / j.sum(), 0.36, color="#0072B2", label="candidate j among disagreements", edgecolor="white")
        ax.bar(xx + 0.19, 100 * t / t.sum(), 0.36, color="#999999", label="true class among all base errors", edgecolor="white")
        ax.set_xticks(xx); ax.set_xticklabels(lab); ax.set_title(f"0a: base-logit rank, 12 cells pooled, checkpoint {seed}")
    axes[0].set_ylabel("% of denominator"); axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{figdir}/0a_rank_distributions.png", dpi=150); plt.close(fig)

    layers = []
    for r in b:
        if r["layer_name"] not in layers:
            layers.append(r["layer_name"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, seed in zip(axes, spec.DEV_SEEDS):
        M = np.array([[next(r["delta_accuracy_pp_vs_base"] for r in b if r["seed"] == seed and r["layer_name"] == L and r["cell"] == c)
                       for c in spec.CONDITIONS] for L in layers])
        im = ax.imshow(M, cmap="RdBu", norm=TwoSlopeNorm(vcenter=0, vmin=-10, vmax=10), aspect="auto")
        ax.set_xticks(range(len(spec.CONDITIONS))); ax.set_xticklabels([c.replace("_", " ") for c in spec.CONDITIONS], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers, fontsize=7)
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                ax.text(k, i, f"{M[i, k]:.0f}", ha="center", va="center", fontsize=5.5, color="black")
        ax.set_title(f"0b: probe accuracy minus base (pp), checkpoint {seed}")
    fig.colorbar(im, ax=axes, fraction=0.02, label="pp vs base (diverging, 0 = base)")
    fig.savefig(f"{figdir}/0b_layer_probe_heatmap.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for seed, c, ls in zip(spec.DEV_SEEDS, ("#0072B2", "#D55E00"), ("-", "--")):
        d = [np.mean([r["delta_accuracy_pp_vs_base"] for r in b if r["seed"] == seed and r["layer_name"] == L and r["cell"] in spec.CELLS]) for L in layers]
        n = [np.mean([r["nll"] - r["base_nll"] for r in b if r["seed"] == seed and r["layer_name"] == L and r["cell"] in spec.CELLS]) for L in layers]
        axes[0].plot(range(len(layers)), d, ls, color=c, marker="o", ms=4, lw=2, label=f"checkpoint {seed}")
        axes[1].plot(range(len(layers)), n, ls, color=c, marker="o", ms=4, lw=2, label=f"checkpoint {seed}")
    for ax, t in zip(axes, ("accuracy − base (pp)", "NLL − base")):
        ax.axhline(0, color="#888888", lw=1); ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"0b: 12-cell macro, {t}")
    axes[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{figdir}/0b_layer_depth_curves.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ladder()
    figures()
    print("done")
