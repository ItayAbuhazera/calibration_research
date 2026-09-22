"""Check the freshly extracted fixed-candidate neighbours against the atlas Stage-A neighbours (layer3.22/grid2/unit_l2) on the 31 000 atlas queries.
Reports the fraction of identical neighbour sets, identical top-1 neighbour, identical j (argmax p_geo) and max |distance| difference. Descriptive."""
import json, sys
import numpy as np
from . import common, data, fg_data, spec, stats


def main(seed):
    ctx = fg_data.load_ctx(seed)
    A = np.load(f"{ctx['root']}/knn_unit/layer3.22__grid2.npz"); ai, ad = A["idx"], A["dist"]
    sub = np.load(f"{data.SHARED}/subset_ids.npy"); rec = {"seed": seed, "per_set": {}}
    offs = {"val": (0, np.arange(5000))}
    for k, c in enumerate(spec.CONDITIONS):
        offs[c] = (5000 + k * len(sub), sub)
    tot = {"same_set": 0, "same_top1": 0, "same_j": 0, "n": 0, "maxd": 0.0}
    for s, (o, ids) in offs.items():
        z = np.load(f"{fg_data.FG}/seed{seed}/knn_F/layer3.22__grid2__unit_l2__{s}.npz")
        ni, nd = z["unit_l2_idx"][ids], z["unit_l2_dist"][ids]; oi, od = ai[o:o + len(ids)], ad[o:o + len(ids)]
        ss = np.mean([set(a) == set(b) for a, b in zip(ni, oi)]); t1 = np.mean(ni[:, 0] == oi[:, 0])
        pn_, _ = stats.p_geo(ni, ctx["bank_y"], ctx["prior"]); po_, _ = stats.p_geo(oi, ctx["bank_y"], ctx["prior"])
        sj = np.mean(pn_.argmax(1) == po_.argmax(1)); md = float(np.abs(nd - od).max())
        rec["per_set"][s] = {"same_neighbour_set": float(ss), "same_top1": float(t1), "same_j": float(sj), "max_abs_dist_diff": md}
        n = len(ids); tot["same_set"] += ss * n; tot["same_top1"] += t1 * n; tot["same_j"] += sj * n; tot["n"] += n; tot["maxd"] = max(tot["maxd"], md)
    rec["overall"] = {k: (v / tot["n"] if k not in ("n", "maxd") else v) for k, v in tot.items()}
    common.atomic_json(f"{fg_data.FG}/seed{seed}/verify_extraction.json", rec); print(rec["overall"])


if __name__ == "__main__":
    main(int(sys.argv[1]))
