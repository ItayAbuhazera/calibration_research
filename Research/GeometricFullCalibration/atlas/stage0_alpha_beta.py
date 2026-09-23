"""POST-HOC diagnostic (not in the frozen protocol): scalar mixing model
    q(x) = softmax(alpha * z(x) + beta * p(x) + b),   alpha, beta scalars, b in R^100,
fitted per outer fold, on raw (unstandardized, pre-temperature) z and layer3.22 probe logits p,
mean cross-entropy, no penalty. Two label sources:
    target: T-8k12 training rows (outer-train images x 12 corrupted cells)
    clean : S-8k1 training rows (outer-train images, clean view)
Comparator with beta fixed at 0 (alpha, b only) is fitted on the same rows.
Evaluation: pooled out-of-fold, 12-cell macro accuracy. "Share of Delta_T recovered" =
(macro12 acc[alpha,beta] - macro12 acc[alpha only]) / Delta_T(T-8k12, q_ZP - q_Z) from the frozen study.

    python -m atlas.stage0_alpha_beta
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from . import spec, stage0_data, stage0_folds
from .stage0c_run import build_rows
from .stage0_aggregate import OUT

torch.set_default_dtype(torch.float64)
K = spec.NUM_CLASSES


def fit(Z, P, y, use_beta, max_iter=500):
    Zt, Pt, yt = map(torch.from_numpy, (np.ascontiguousarray(Z), np.ascontiguousarray(P), np.ascontiguousarray(y, dtype=np.int64)))
    a = torch.ones(1, requires_grad=True)
    be = torch.zeros(1, requires_grad=True)
    b = torch.zeros(K, requires_grad=True)
    params = [a, b] + ([be] if use_beta else [])
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        lg = a * Zt + (be * Pt if use_beta else 0) + b
        loss = torch.nn.functional.cross_entropy(lg, yt)
        loss.backward()
        return loss

    opt.step(closure)
    loss = closure()
    g = max(p.grad.abs().max().item() for p in params)
    st = opt.state[opt.param_groups[0]["params"][0]]
    return {"alpha": a.item(), "beta": be.item() if use_beta else 0.0, "b": b.detach().numpy(), "nll": loss.item(),
            "grad_inf": g, "n_iter": int(st.get("n_iter", -1)), "hit_cap": st.get("n_iter", 0) >= max_iter}


def acc(fitres, Z, P, y):
    lg = fitres["alpha"] * Z + fitres["beta"] * P + fitres["b"]
    return float((lg.argmax(1) == y).mean())


def main():
    plan = stage0_folds.load_plan()
    agg = json.load(open(f"{OUT}/stage0c_aggregate.json"))
    out = {"label": "POST-HOC diagnostic, not part of the frozen protocol; live tree, uncommitted code"}
    for seed in spec.DEV_SEEDS:
        cells = stage0_data.load_all_cells(seed)
        rec = {"target": {"alpha": [], "beta": [], "fits": []}, "clean": {"alpha": [], "beta": [], "fits": []}}
        correct = {(src, m): {c: np.zeros(10000, bool) for c in spec.CELLS} for src in ("target", "clean") for m in ("ab", "a")}
        for f in range(5):
            o = plan["outer"][f]
            train, test = np.array(o["train_idx"]), np.array(o["test_idx"])
            t0 = time.time()
            for src, rule in (("target", "all12"), ("clean", "clean")):
                Zr, Pr, yr, _ = build_rows(cells, train, rule)
                fits = {"ab": fit(Zr, Pr, yr, True), "a": fit(Zr, Pr, yr, False)}
                rec[src]["alpha"].append(fits["ab"]["alpha"]); rec[src]["beta"].append(fits["ab"]["beta"])
                rec[src]["fits"].append({m: {k: v for k, v in fits[m].items() if k != "b"} for m in fits})
                for m in fits:
                    for c in spec.CELLS:
                        d = cells[c]
                        correct[(src, m)][c][test] = (fits[m]["alpha"] * d["z"][test] + fits[m]["beta"] * d["p"][test] + fits[m]["b"]).argmax(1) == d["labels"][test]
            print(f"seed{seed} fold{f} done {time.time()-t0:.0f}s", flush=True)
        dT = agg["primary_secondary_by_seed"][str(seed)]["macro12"]["T-8k12"]["delta_acc_pp"]
        res = {}
        for src in ("target", "clean"):
            macro = {m: 100 * float(np.mean([correct[(src, m)][c].mean() for c in spec.CELLS])) for m in ("ab", "a")}
            gain = macro["ab"] - macro["a"]
            res[src] = {"alpha_by_fold": rec[src]["alpha"], "beta_by_fold": rec[src]["beta"],
                        "macro12_acc_alpha_beta": macro["ab"], "macro12_acc_alpha_only": macro["a"],
                        "gain_pp": gain, "share_of_DeltaT_recovered": gain / dT, "DeltaT_pp": dT,
                        "n_unconverged_or_capped": sum(x[m]["hit_cap"] for x in rec[src]["fits"] for m in x),
                        "max_grad_inf": max(x[m]["grad_inf"] for x in rec[src]["fits"] for m in x)}
        out[str(seed)] = res
        print(json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in res.items()}, indent=1), flush=True)
    json.dump(out, open(f"{OUT}/stage0_posthoc_alpha_beta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
