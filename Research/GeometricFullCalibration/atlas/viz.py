"""Stage VIZ (CPU): scientific figures and diagnostics from atlas artifacts. No synthetic images: every panel is drawn
from stored statistics, neighbour files, K-means/PCA outputs or actual dataset images.

Figures per checkpoint (results/atlas/figures/seed{S}/):
  heatmap_<metric>.{png,pdf}         layer x pooling x condition, selected cells boxed, sample counts in titles
  depth_curves.{png,pdf}             candidate net utility / class rank / neighbourhood purity vs depth; clean->corruption paired change
  pca_trajectories_<rep>.{png,pdf}   clean->corrupted trajectories of the same image IDs in a fixed clean-reference PCA map
  gallery_<pool>.{png,pdf}           neighbour galleries stratified over W/H/U x corruption family (failures included)
  overlap.{png,pdf}                  hidden vs output vs VS repairs and harms in absolute counts
  pareto.{png,pdf}                   accuracy gain vs NLL / Brier change, pre- and post-temperature
  clustering.{png,pdf}               K-means ARI/NMI, association with correctness / condition
Target-coloured or outcome-conditioned panels are labelled DIAGNOSTIC in their titles.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Dict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import data, spec, stats  # noqa: E402

MEAN = np.array([0.4914, 0.4822, 0.4465]); STD = np.array([0.2023, 0.1994, 0.2010])
COLS = ["val-sel", "val-fit", "clean"] + list(spec.CELLS)


def save(fig, base):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{base}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_stats(seed):
    root = data.seed_dir(seed)
    return (json.load(open(f"{root}/stats_clean_unit_l2.json")), json.load(open(f"{root}/stats_target_unit_l2.json")),
            json.load(open(f"{root}/shortlist.json")))


def grid(clean, target, pool, key):
    M = np.full((len(spec.SITE_NAMES), len(COLS)), np.nan)
    for i, s in enumerate(spec.SITE_NAMES):
        v = [clean[s][pool]["selection"], clean[s][pool]["fit"], target[s][pool]["clean"]] + [target[s][pool][c] for c in spec.CELLS]
        M[i] = [np.nan if r[key] is None else r[key] for r in v]
    return M


def heatmaps(seed):
    clean, target, sl = load_stats(seed)
    out = f"results/atlas/figures/seed{seed}"
    keys = {"candidate_acc_among_base_errors": ("Candidate accuracy on base errors", "viridis"), "harm_rate_among_base_correct": ("Harm rate on base-correct", "magma_r"),
            "net_utility": ("Net utility (W-H)/n", "RdBu"), "geo_nll": ("Smoothed geo NLL", "viridis_r"), "geo_brier": ("Geo Brier (sum)", "viridis_r")}
    for key, (title, cmap) in keys.items():
        mats = {p: grid(clean, target, p, key) for p in spec.POOLS}
        allv = np.concatenate([m.ravel() for m in mats.values()]); allv = allv[~np.isnan(allv)]
        lo, hi = np.percentile(allv, [1, 99])
        if key == "net_utility":
            a = max(abs(lo), abs(hi)); lo, hi = -a, a
        fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
        for ax, p in zip(axes, spec.POOLS):
            im = ax.imshow(mats[p], aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f"{p} (dim {spec.pool_dim('layer3.22', p)} @ layer3.22)", fontsize=9)
            ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COLS, rotation=90, fontsize=6)
            ax.set_yticks(range(len(spec.SITE_NAMES))); ax.set_yticklabels(spec.SITE_NAMES, fontsize=6)
            r = spec.SITE_NAMES.index(sl["chosen"][p]["site"])
            ax.add_patch(Rectangle((-0.5, r - 0.5), len(COLS), 1, fill=False, ec="k", lw=1.5))
        fig.colorbar(im, ax=axes, shrink=0.6)
        fig.suptitle(f"seed {seed}: {title}  (common scale over pooling families; boxed = clean-selected; n: selection 1250, fit 2500, test-subset 2000/cell)", fontsize=10)
        save(fig, f"{out}/heatmap_{key}")


def depth_curves(seed):
    clean, target, sl = load_stats(seed)
    out = f"results/atlas/figures/seed{seed}"
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    x = np.arange(len(spec.SITE_NAMES))
    fams = spec.DEV_CORRUPTIONS
    for j, p in enumerate(spec.POOLS):
        nu_c = grid(clean, target, p, "net_utility"); rk = grid(clean, target, p, "geo_gt_rank_mean"); lp = grid(clean, target, p, "local_true_label_proportion")
        for ax, M, lab in ((axes[0, j], nu_c, "net utility"), (axes[1, j], lp, "local true-label proportion"), (axes[2, j], rk, "GT rank (geo)")):
            ax.plot(x, M[:, 0], "k-", label="val selection"); ax.plot(x, M[:, 2], "k--", label="clean test")
            for f in fams:
                cols = [3 + i for i, c in enumerate(spec.CELLS) if c.startswith(f)]
                ax.plot(x, np.nanmean(M[:, cols], axis=1), label=f, alpha=.8)
            ax.set_title(f"{p}: {lab}", fontsize=9); ax.grid(alpha=.3)
        axes[0, j].axvline(spec.SITE_NAMES.index(sl["chosen"][p]["site"]), color="r", lw=1)
    axes[2, 0].set_xticks(x[::3]); axes[2, 0].set_xticklabels([spec.SITE_NAMES[i] for i in x[::3]], rotation=90, fontsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"seed {seed}: depth curves (red line = clean-selected layer). Corruption curves are TARGET-LABELLED DIAGNOSTICS.", fontsize=10)
    save(fig, f"{out}/depth_curves")
    # paired clean -> corruption change in utility
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for j, p in enumerate(spec.POOLS):
        M = grid(clean, target, p, "net_utility")
        for f in fams:
            cols = [3 + i for i, c in enumerate(spec.CELLS) if c.startswith(f)]
            axes[j].plot(x, np.nanmean(M[:, cols], axis=1) - M[:, 2], label=f)
        axes[j].axhline(0, color="k", lw=.5); axes[j].set_title(f"{p}: corrupted minus clean net utility", fontsize=9); axes[j].grid(alpha=.3)
    axes[0].legend(fontsize=7)
    save(fig, f"{out}/clean_to_corruption_change")


def class_names():
    p = "data/cifar-100-python/meta"
    return pickle.load(open(p, "rb"), encoding="latin1")["fine_label_names"] if os.path.exists(p) else [str(i) for i in range(100)]


def denorm(x):
    return np.clip(np.transpose(x, (1, 2, 0)) * STD + MEAN, 0, 1)


def galleries(seed):
    from . import stage_stats as ss
    root, roles, val_y, bank_y, prior, sub, ty = ss.load_common(seed)
    sl = json.load(open(f"{root}/shortlist.json"))["chosen"]
    arrays = data.load_seed_arrays(seed); train_x = arrays["train"][0]
    meta = data.atlas_query_meta(seed); full = data.load_full_sets(); val_x = arrays["val"][0]

    class _Q:   # lazy row access: only the ~12 picked query images are read
        def __getitem__(self, row):
            sid, img = int(meta["set_id"][row]), int(meta["img_id"][row])
            return np.asarray(val_x[img]) if sid == 0 else np.asarray(full[data.cond_slice(spec.CONDITIONS[sid - 1])][img])
    Q = _Q()
    bp = ss.base_logits(root, sub).argmax(1); yy = data.atlas_labels(seed, arrays)
    names = class_names()
    cond_of = np.concatenate([np.full(5000, -1)] + [np.full(len(sub), i) for i in range(len(spec.CONDITIONS))])
    for p in spec.POOLS:
        s = sl[p]["site"]
        idx, dist, _ = ss.unit_arrays(root, s, p)
        pg, _c = stats.p_geo(idx, bank_y, prior)
        j = pg.argmax(1)
        cat = np.where((bp != yy) & (j == yy), "W", np.where((bp == yy) & (j != yy), "H", np.where((bp != yy) & (j != yy) & (j != bp), "U", "-")))
        picks = []
        for fam in spec.DEV_CORRUPTIONS:
            fam_cells = [i + 1 for i, c in enumerate(spec.CONDITIONS[1:]) if c.startswith(fam)]
            rows = np.flatnonzero(np.isin(cond_of, fam_cells))
            for k in ("W", "H", "U"):
                cand = rows[cat[rows] == k]
                if len(cand):
                    h = sorted(cand, key=lambda r: hashlib.md5(f"{seed}-{p}-{fam}-{k}-{r}".encode()).hexdigest())[0]
                    picks.append((fam, k, int(h)))
                else:
                    picks.append((fam, k, None))
        fig, axes = plt.subplots(len(picks), 6, figsize=(9, 1.65 * len(picks)))
        for r, (fam, k, row) in enumerate(picks):
            for c in range(6):
                axes[r, c].axis("off")
            if row is None:
                axes[r, 0].text(0, .5, f"{fam} {k}: none", fontsize=6); continue
            axes[r, 0].imshow(denorm(Q[row])); cond = spec.CONDITIONS[cond_of[row]]
            axes[r, 0].set_title(f"{k} {cond}\ntrue {names[yy[row]]}\nbase {names[bp[row]]}\ngeo {names[j[row]]}", fontsize=5, loc="left")
            for c, b in enumerate(idx[row, :5]):
                axes[r, c + 1].imshow(denorm(np.asarray(train_x[int(b)]))); axes[r, c + 1].set_title(names[bank_y[b]], fontsize=5)
        fig.suptitle(f"seed {seed} {p} @ {s}: neighbour gallery, stratified W/H/U x corruption family (DIAGNOSTIC; failures included; hash-ordered picks)", fontsize=7)
        save(fig, f"results/atlas/figures/seed{seed}/gallery_{p}")
        json.dump([{"family": f, "category": k, "query_row": r} for f, k, r in picks], open(f"results/atlas/figures/seed{seed}/gallery_{p}_picks.json", "w"))


def pca_trajectories(seed):
    root = data.seed_dir(seed)
    sl = json.load(open(f"{root}/shortlist.json"))["chosen"]
    sub = np.load(f"{data.SHARED}/subset_ids.npy")
    reps = [(f"{sl[p]['site']}__{p}", f"{p}@{sl[p]['site']}") for p in spec.POOLS] + [("logits", "logits (centered, unit)")]
    ids = np.arange(0, len(sub), 66)[:30]
    fams = {"gaussian_noise": "tab:red", "defocus_blur": "tab:blue", "fog": "tab:green", "jpeg_compression": "tab:orange"}
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (tag, title) in zip(axes, reps):
        z = np.load(f"{root}/viz/{tag}.npz")
        ax.scatter(z["bank_xy"][:, 0], z["bank_xy"][:, 1], s=1, c="0.85")
        xy = z["query_xy"]
        base = 5000 + 0 * 2000
        for i in ids:
            c0 = xy[base + i]
            for k, cell in enumerate(spec.CELLS):
                if not cell.endswith("_s3"):
                    continue
                ci = spec.CONDITIONS.index(cell)
                c1 = xy[5000 + ci * 2000 + i]
                ax.annotate("", xy=c1, xytext=c0, arrowprops=dict(arrowstyle="->", color=fams[cell.rsplit("_s", 1)[0]], lw=.5, alpha=.7))
            ax.plot(*c0, "k.", ms=3)
        ev = z["explained_var_ratio"]
        ax.set_title(f"{title}\nPCA fitted on clean reference; explained var {ev[0]:.3f}, {ev[1]:.3f}", fontsize=8)
    fig.suptitle(f"seed {seed}: clean (black) -> severity-3 corrupted trajectories of 30 fixed image IDs (visualization only; not a classifier metric; DIAGNOSTIC)", fontsize=9)
    save(fig, f"results/atlas/figures/seed{seed}/pca_trajectories")


def clustering(seed):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from . import stage_stats as ss
    root, roles, val_y, bank_y, prior, sub, ty = ss.load_common(seed)
    sl = json.load(open(f"{root}/shortlist.json"))["chosen"]
    arrays = data.load_seed_arrays(seed)
    yy = data.atlas_labels(seed, arrays); bp = ss.base_logits(root, sub).argmax(1)
    cond_of = np.concatenate([np.full(5000, -1)] + [np.full(len(sub), i) for i in range(13)])
    rows, res = [], {}
    for tag in [f"{sl[p]['site']}__{p}" for p in spec.POOLS] + ["logits"]:
        z = np.load(f"{root}/viz/{tag}.npz")
        ba, qa = z["bank_assign"].astype(int), z["query_assign"].astype(int)
        rec = {"tag": tag, "ARI_bank": adjusted_rand_score(bank_y, ba), "NMI_bank": normalized_mutual_info_score(bank_y, ba),
               "ARI_val": adjusted_rand_score(yy[:5000], qa[:5000]), "NMI_val": normalized_mutual_info_score(yy[:5000], qa[:5000]),
               "cluster_size_min_max_bank": [int(np.bincount(ba, minlength=100).min()), int(np.bincount(ba, minlength=100).max())]}
        for ci, c in enumerate(spec.CONDITIONS):
            m = cond_of == ci
            rec[f"ARI_{c}"] = adjusted_rand_score(yy[m], qa[m]); rec[f"NMI_{c}"] = normalized_mutual_info_score(yy[m], qa[m])
        qm = cond_of >= 0
        rec["NMI_cluster_vs_condition"] = normalized_mutual_info_score(cond_of[qm], qa[qm])
        rec["NMI_cluster_vs_base_correct"] = normalized_mutual_info_score((bp[qm] == yy[qm]).astype(int), qa[qm])
        # class purity of the frozen clusters on the bank
        pur = [np.bincount(bank_y[ba == k], minlength=100).max() / max((ba == k).sum(), 1) for k in range(100)]
        rec["mean_cluster_purity_bank"] = float(np.mean(pur))
        rows.append(rec)
    json.dump(rows, open(f"results/atlas/figures/seed{seed}/clustering.json", "w"), indent=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    tags = [r["tag"] for r in rows]
    axes[0].bar(range(len(rows)), [r["ARI_bank"] for r in rows], width=.4, label="ARI"); axes[0].bar(np.arange(len(rows)) + .4, [r["NMI_bank"] for r in rows], width=.4, label="NMI")
    axes[0].set_title("K-means (K=100) vs true labels, reference bank"); axes[0].set_xticks(range(len(rows))); axes[0].set_xticklabels(tags, rotation=60, fontsize=6); axes[0].legend()
    for r in rows:
        axes[1].plot([r[f"NMI_{c}"] for c in spec.CONDITIONS], label=r["tag"])
    axes[1].set_xticks(range(13)); axes[1].set_xticklabels(spec.CONDITIONS, rotation=90, fontsize=6); axes[1].set_title("held-out NMI vs true label by condition (DIAGNOSTIC)"); axes[1].legend(fontsize=5)
    axes[2].bar(range(len(rows)), [r["NMI_cluster_vs_condition"] for r in rows], width=.4, label="NMI(cluster,condition)")
    axes[2].bar(np.arange(len(rows)) + .4, [r["NMI_cluster_vs_base_correct"] for r in rows], width=.4, label="NMI(cluster,base-correct)")
    axes[2].set_xticks(range(len(rows))); axes[2].set_xticklabels(tags, rotation=60, fontsize=6); axes[2].legend(fontsize=6); axes[2].set_title("clusters reflect corruption/correctness?")
    save(fig, f"results/atlas/figures/seed{seed}/clustering")


def overlap_and_pareto(seed):
    root = data.seed_dir(seed)
    sl = json.load(open(f"{root}/shortlist.json"))["chosen"]
    res = json.load(open(f"{root}/cd/results.json"))
    ps = {c: np.load(f"{root}/cd/per_sample_{c}.npz") for c in spec.CONDITIONS}
    names = ["gap_alone", "grid2_alone", "spp_alone", "out_alone", "vs", "ms"]
    tot = {n: {"W": 0, "H": 0} for n in names}; uniq = {}
    sets = {n: {"W": [], "H": []} for n in names}
    for c in spec.CELLS:
        z = ps[c]; y = z["labels"].astype(int); b = z["base_pred"].astype(int)
        for n in names:
            p = z[f"pred__{n}"].astype(int)
            sets[n]["W"].append((p != b) & (b != y) & (p == y)); sets[n]["H"].append((p != b) & (b == y) & (p != y))
    S = {n: {k: np.concatenate(v) for k, v in d.items()} for n, d in sets.items()}
    table = {}
    for h in ("gap_alone", "grid2_alone", "spp_alone"):
        for o in ("out_alone", "vs", "ms"):
            table[f"{h} vs {o}"] = {"W_h": int(S[h]["W"].sum()), "H_h": int(S[h]["H"].sum()), "W_o": int(S[o]["W"].sum()), "H_o": int(S[o]["H"].sum()),
                                    "unique_W_h": int((S[h]["W"] & ~S[o]["W"]).sum()), "unique_H_h": int((S[h]["H"] & ~S[o]["H"]).sum()),
                                    "unique_W_o": int((S[o]["W"] & ~S[h]["W"]).sum()), "unique_H_o": int((S[o]["H"] & ~S[h]["H"]).sum()),
                                    "jaccard_W": float((S[h]["W"] & S[o]["W"]).sum() / max((S[h]["W"] | S[o]["W"]).sum(), 1))}
    json.dump(table, open(f"results/atlas/figures/seed{seed}/overlap.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(11, 4)); keys = list(table)
    x = np.arange(len(keys))
    ax.bar(x - .2, [table[k]["unique_W_h"] for k in keys], .2, label="unique repairs (hidden)"); ax.bar(x, [table[k]["unique_H_h"] for k in keys], .2, label="unique harms (hidden)")
    ax.bar(x + .2, [table[k]["unique_W_o"] for k in keys], .2, label="unique repairs (control)")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=7); ax.legend(fontsize=7)
    ax.set_title(f"seed {seed}: absolute unique repairs/harms of the frozen candidate challengers (12 dev cells x 10 000; candidate applied everywhere; oracle-free set differences)")
    save(fig, f"results/atlas/figures/seed{seed}/overlap")
    # Pareto: macro corruption dAcc vs dNLL / dBrier (relative to base raw)
    def macro(n, v, k):
        return float(np.mean([res[n][v][c][k] for c in spec.CELLS]))
    base_acc, base_nll, base_br = macro("base", "raw", "accuracy"), macro("base", "raw", "nll"), macro("base", "raw", "brier_sum")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for n in res:
        for v, mk in (("raw", "o"), ("T_nll", "^")):
            da = 100 * (macro(n, v, "accuracy") - base_acc)
            axes[0].scatter(macro(n, v, "nll") - base_nll, da, marker=mk, s=25); axes[1].scatter(macro(n, v, "brier_sum") - base_br, da, marker=mk, s=25)
            if v == "T_nll":
                axes[0].annotate(n, (macro(n, v, "nll") - base_nll, da), fontsize=5); axes[1].annotate(n, (macro(n, v, "brier_sum") - base_br, da), fontsize=5)
    for ax, l in zip(axes, ("dNLL vs base (raw softmax)", "dBrier vs base")):
        ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5); ax.set_xlabel(l); ax.set_ylabel("d accuracy (pp)"); ax.grid(alpha=.3)
    fig.suptitle(f"seed {seed}: Pareto, macro over 12 dev cells (o = pre-temperature, ^ = post NLL-temperature)")
    save(fig, f"results/atlas/figures/seed{seed}/pareto")


def index_html():
    figs = []
    for r, _, fs in os.walk("results/atlas/figures"):
        figs += [os.path.join(r, f) for f in sorted(fs) if f.endswith(".png")]
    html = ["<html><body><h1>Representation atlas figures</h1><p>Outcome-conditioned panels are diagnostics; see docs/atlas_program_spec.md.</p>"]
    for f in sorted(figs):
        html.append(f"<h3>{os.path.relpath(f, 'results/atlas/figures')}</h3><img src='{os.path.relpath(f, 'results/atlas/figures')}' width='1200'>")
    open("results/atlas/figures/index.html", "w").write("\n".join(html) + "</body></html>")


def main(seed=None):
    seeds = [int(seed)] if seed else list(spec.DEV_SEEDS)
    for s in seeds:
        for fn in (heatmaps, depth_curves, pca_trajectories, clustering, galleries, overlap_and_pareto):
            try:
                fn(s); print("ok", fn.__name__, s)
            except Exception as e:  # keep going; failures are recorded, not hidden
                import traceback; traceback.print_exc(); print("FAILED", fn.__name__, s, repr(e))
    index_html()


if __name__ == "__main__":
    import sys; main(*sys.argv[1:])
