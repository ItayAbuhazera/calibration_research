"""Gate families C0/C1/Z0/Z1, nested class-stratified fit orders and frozen-gate (de)serialization for the fixed-gate study.

Spec: docs/fixed_gate_study_spec.md (frozen). Conventions (all float64):
  z      [N,100] base logits (atlas u0 artifact), i = argmax z (base class), p = softmax(z)
  j      [N]     candidate class = argmax p_geo,   pg [N,100] smoothed kNN distribution
  D      = 1{j=y} - 1{i=y} in {-1,0,+1};  gate g = 1{j != i and clip(b + w.s(F),-1,1) > theta}
  ridge  min_{b,w} mean_r (D_r - b - w.F_r)^2 + lam ||w||^2 over the m disagreement rows r of the fit subset; b unpenalized;
         closed form (X'X + lam m I) w = X'(D - b) on standardized X (standardization uses the same m rows) => b = mean D.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import gate as G
from . import stats

FAMILIES = ("C0", "C1", "Z0", "Z1")
LAMBDAS = (0.01, 1.0, 100.0)
THETAS = (0.0, 0.02, 0.05, 0.10, 0.20)
PREFIXES = (625, 1250, 2500)
ORDER_SEEDS = (20261001, 20261002, 20261003)
CONST_TOL = 1e-8


def base_terms(z: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """C0 (5): z_i - z_j, top-two logit margin, p_i, p_j, entropy(p)."""
    n = len(z); ar = np.arange(n)
    p = stats.softmax(z); top = np.sort(z, axis=1)
    ent = -(p * np.log(np.clip(p, 1e-300, 1.0))).sum(1)
    return np.stack([z[ar, i] - z[ar, j], top[:, -1] - top[:, -2], p[ar, i], p[ar, j], ent], 1)


def geo_terms(pg: np.ndarray, i: np.ndarray, j: np.ndarray, kth: np.ndarray, qnorm: np.ndarray) -> np.ndarray:
    """4 geometric features: p_geo(j)-p_geo(i), entropy(p_geo), k-th neighbour radius, log(max(pre-L2 norm, 1e-12))."""
    ar = np.arange(len(i))
    return np.stack([pg[ar, j] - pg[ar, i], stats.neighborhood_entropy(pg), kth, np.log(np.maximum(qnorm, 1e-12))], 1)


def features(family: str, z, i, j, pg, kth, qnorm) -> np.ndarray:
    c0 = base_terms(z, i, j)
    if family == "C0":
        return c0
    if family == "C1":
        return np.concatenate([c0, geo_terms(pg, i, j, kth, qnorm)], 1)
    zc = z - z.mean(1, keepdims=True)
    if family == "Z0":
        return np.concatenate([c0, zc], 1)
    if family == "Z1":
        return np.concatenate([c0, zc, geo_terms(pg, i, j, kth, qnorm)], 1)
    raise ValueError(family)


def stratified_order(y_fit: np.ndarray, seed: int) -> np.ndarray:
    """Positions 0..n-1 (into the fit rows) in a class-stratified round-robin order; every prefix is near class-balanced and nested."""
    rng = np.random.default_rng(seed)
    per = {c: rng.permutation(np.flatnonzero(y_fit == c)) for c in np.unique(y_fit)}
    out: List[int] = []
    for r in range(max(len(v) for v in per.values())):
        for c in rng.permutation(np.array(sorted(per))):
            if r < len(per[c]):
                out.append(int(per[c][r]))
    assert len(out) == len(y_fit) and len(set(out)) == len(out)
    return np.asarray(out, dtype=np.int64)


def fit_family(X: np.ndarray, i, j, y, lambdas: Sequence[float] = LAMBDAS) -> Dict:
    """Fit on the given rows' disagreements. Returns counts, design diagnostics and one model per lambda (None if degenerate)."""
    dis = j != i
    D = G.utility_target(i, j, y)
    out = {"n": int(len(y)), "m": int(dis.sum()), "W": int(((D > 0) & dis).sum()), "H": int(((D < 0) & dis).sum()),
           "U": int((dis & (D == 0) & (j != y)).sum()), "degenerate": bool(dis.sum() < 5), "models": {}}
    if out["degenerate"]:
        return out
    Xd = X[dis]; mean = Xd.mean(0); sd = np.sqrt(Xd.var(0))
    const = sd < CONST_TOL
    st = G.Standardizer(mean, sd + 1e-6); Xs = st(Xd); d = D[dis]
    m = len(d)
    out["n_features"] = int(X.shape[1]); out["constant_features"] = np.flatnonzero(const).tolist()
    out["design_rank_with_intercept"] = int(np.linalg.matrix_rank(np.concatenate([np.ones((m, 1)), Xs], 1)))
    for lam in lambdas:
        w, b = G.ridge_fit(Xs, d, lam)
        H = Xs @ np.linalg.solve(Xs.T @ Xs + lam * m * np.eye(Xs.shape[1]), Xs.T)
        pred = np.clip(Xs @ w + b, -1, 1)
        out["models"][lam] = {"mean": mean, "std": sd + 1e-6, "w": w, "b": b, "df": float(np.trace(H)), "train_mse": float(np.mean((d - pred) ** 2)),
                              "train_mse_const": float(np.mean((d - d.mean()) ** 2))}
    return out


def predict(model: Dict, X: np.ndarray) -> np.ndarray:
    return np.clip(((X - model["mean"]) / model["std"]) @ model["w"] + model["b"], -1.0, 1.0)


def select_config(fit: Dict, X_sel, i_s, j_s, y_s) -> Tuple[Dict, List[Dict]]:
    """Max Sum(gD)/n_sel over {never} U lambda x theta; ties: fewer interventions, larger lambda (never = inf), then grid order."""
    n = len(y_s); D = G.utility_target(i_s, j_s, y_s)
    table = [{"kind": "never", "lambda": float("inf"), "theta": None, "net": 0.0, "interventions": 0, "order": -1}]
    order = 0
    for lam in sorted(fit["models"]):
        u = predict(fit["models"][lam], X_sel)
        for th in THETAS:
            g = (j_s != i_s) & (u > th)
            table.append({"kind": "gate", "lambda": lam, "theta": th, "net": float(D[g].sum()) / n, "interventions": int(g.sum()), "order": order})
            order += 1
    best = min(table, key=lambda c: (-round(c["net"], 12), c["interventions"], -c["lambda"], c["order"]))
    return best, table


def apply_config(cfg: Dict, model: Optional[Dict], X, i, j):
    """(g bool [N], score float [N] or NaN). never => g = 0; disagreement mask forces g = 0 where i = j."""
    if cfg["kind"] == "never" or model is None:
        return np.zeros(len(i), bool), np.full(len(i), np.nan)
    u = predict(model, X)
    return ((j != i) & (u > cfg["theta"])), u


def ser(model: Optional[Dict]):
    if model is None:
        return None
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items()}


def deser(d: Optional[Dict]):
    if d is None:
        return None
    return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in d.items()}
