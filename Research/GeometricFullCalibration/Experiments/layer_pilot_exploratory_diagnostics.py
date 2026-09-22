"""
EXPLORATORY, POST-HOC descriptive diagnostics on the layer-pilot evaluation cells.

Not part of the frozen §9 decision rule and not used for any selection. It uses
target labels of the 12 corruption cells purely to DESCRIBE the statistics that the
theory plan (T3) reasons about; it fits no gate and no readout.

For every evaluation sample take the base top-2 pair (i = top-1, j = runner-up) and
D = 1{j=y} - 1{i=y}. Keep D != 0 (W vs H). delta_l = r_{l,i} - r_{l,j} (> 0 favours j).
Reports per layer d' and AUC for separating W from H, the within-class correlation
between layers (rho), the equal-weight d' of the greedy nested sets, and the same AUC
for the base logit margin z_i - z_j (a logit-only comparator).
"""
import json, os, sys
import numpy as np
from scipy.stats import rankdata, norm

CELLS = [f"{c}_s{s}" for c in ("gaussian_noise", "defocus_blur", "fog", "jpeg_compression") for s in (1, 3, 5)]
NAMES = ["layer1.0", "layer1.2", "layer2.1", "layer2.3", "layer3.2", "layer3.7", "layer3.12", "layer3.17",
         "layer3.22", "layer4.0", "layer4.1", "layer4.2"]


def auc(s, lab):
    r = rankdata(s); n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(root="results/layer_pilot"):
    out = {}
    for seed in (2, 4):
        st = json.load(open(f"{root}/checkpoint_seed{seed}/frozen_state.json"))
        order = st["selection"]["family_a"]["order"]
        Ds, Xs, Ms = [], [], []
        for c in CELLS:
            d = np.load(f"{root}/checkpoint_seed{seed}/{c}/per_sample.npz")
            y = d["labels"].astype(int); z = d["z"].astype(np.float32); r = d["raw__r"].astype(np.float32)
            top2 = np.argsort(-z, 1)[:, :2]; i, j = top2[:, 0], top2[:, 1]; n = np.arange(len(y))
            D = (j == y).astype(int) - (i == y).astype(int); m = D != 0
            Ds.append(D[m]); Xs.append((r[n, :, i] - r[n, :, j])[m]); Ms.append(-(z[n, i] - z[n, j])[m])
        D = np.concatenate(Ds); X = np.concatenate(Xs); M = np.concatenate(Ms); lab = (D > 0).astype(int)
        pos, neg = X[D > 0], X[D < 0]
        mu = (pos.mean(0) - neg.mean(0)) / 2
        Xc = np.concatenate([pos - pos.mean(0), neg - neg.mean(0)])
        C = np.cov(Xc.T); sd = np.sqrt(np.diag(C)); R = C / np.outer(sd, sd)
        off = R[np.triu_indices(12, 1)]
        rec = {"n_W": int(len(pos)), "n_H": int(len(neg)),
               "d_prime": dict(zip(NAMES, (mu / sd).round(4).tolist())),
               "auc_W_vs_H": dict(zip(NAMES, [round(auc(X[:, k], lab), 4) for k in range(12)])),
               "auc_base_logit_margin": round(auc(M, lab), 4),
               "auc_mean_of_12_layers": round(auc(X.mean(1), lab), 4),
               "rho_mean": float(off.mean()), "rho_min": float(off.min()), "rho_max": float(off.max()),
               "greedy_equal_weight_d_prime": {}}
        for L in (1, 2, 3, 4, 6, 8):
            S = order[:L]; m_ = mu[S].mean(); v = np.ones(L) @ C[np.ix_(S, S)] @ np.ones(L) / L ** 2
            rec["greedy_equal_weight_d_prime"][L] = round(float(m_ / np.sqrt(v)), 4)
        out[seed] = rec
    os.makedirs(f"{root}/aggregate", exist_ok=True)
    json.dump(out, open(f"{root}/aggregate/exploratory_diagnostics.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
