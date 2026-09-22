"""Small clean-trained intervention gate and probability construction (Experiments C and D).

Utility target D = 1{j=Y} - 1{i=Y} in {-1,0,+1}, with base class i = argmax of the base logits and the candidate class
j = argmax p_geo. If j = i no intervention is possible. A ridge regression of D on standardized features is fitted ONLY
on clean-fit disagreements (j != i); predictions are clipped to [-1,1]; the gate is g = 1{j != i and D_hat > theta}.
Two feature sets for the SAME candidate:
  F0  centered logits, one-hot base and candidate class identities, base probabilities of i and j, base top-two logit
      margin, candidate logit gap z_j - z_i, predictive entropy
  F1  F0 + geometric candidate-minus-base probability p_geo(j)-p_geo(i), neighbourhood entropy of p_geo,
      k-th neighbour radius, pre-L2 feature norm
Configuration (ridge strength, theta, plus never-intervene) is chosen on the clean-SELECTION rows by maximum net accuracy
gain (W-H)/n; ties: fewer interventions, then stronger regularization (never-intervene = infinite strength).
Probabilities: q_raw = (1-g) q0 + g p_geo, q0 = clean-fitted native DAC (argmax = i), p_geo smoothed kNN (argmax = j).
Final scalar temperature q = softmax(log q_raw / T), T > 0, fitted on the calibration split: preserves argmax.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import optimize

from . import stats

RIDGES = (0.01, 1.0, 100.0)
THETAS = (0.0, 0.02, 0.05, 0.10, 0.20)
FEATURE_SETS = ("F0", "F1")


def f0_features(z: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    n, c = z.shape
    zc = z - z.mean(1, keepdims=True)
    p = stats.softmax(z)
    top = np.sort(z, axis=1)
    oh_i = np.zeros((n, c)); oh_i[np.arange(n), i] = 1
    oh_j = np.zeros((n, c)); oh_j[np.arange(n), j] = 1
    ent = -(p * np.log(np.clip(p, 1e-300, 1))).sum(1)
    extra = np.stack([p[np.arange(n), i], p[np.arange(n), j], top[:, -1] - top[:, -2], z[np.arange(n), j] - z[np.arange(n), i], ent], 1)
    return np.concatenate([zc, oh_i, oh_j, extra], 1)


def f1_features(z, i, j, pg: np.ndarray, kth_radius: np.ndarray, qnorm: np.ndarray) -> np.ndarray:
    n = len(i)
    geo = np.stack([pg[np.arange(n), j] - pg[np.arange(n), i], stats.neighborhood_entropy(pg), kth_radius, qnorm], 1)
    return np.concatenate([f0_features(z, i, j), geo], 1)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(x):
        return Standardizer(x.mean(0), np.sqrt(x.var(0)) + 1e-6)

    def __call__(self, x):
        return (x - self.mean) / self.std


def ridge_fit(x: np.ndarray, d: np.ndarray, lam: float):
    """min_w mean((d - b - x w)^2) + lam ||w||^2 (unpenalized intercept b = mean d) on standardized x; closed form."""
    b = d.mean()
    w = np.linalg.solve(x.T @ x + lam * len(x) * np.eye(x.shape[1]), x.T @ (d - b))
    return w, float(b)


def utility_target(i, j, y):
    return (j == y).astype(float) - (i == y).astype(float)


def fit_gates(X_fit: np.ndarray, i, j, y, ridges: Sequence[float] = RIDGES):
    """One ridge model per strength, trained on clean-fit disagreements only; returns dict lam -> model or None."""
    dis = j != i
    out = {"n_fit": int(len(y)), "n_disagree": int(dis.sum()), "n_useful": int(((utility_target(i, j, y) > 0) & dis).sum()),
           "n_harmful": int(((utility_target(i, j, y) < 0) & dis).sum()), "models": {}}
    if dis.sum() < 5:
        return out
    st = Standardizer.fit(X_fit[dis])
    Xs = st(X_fit[dis]); d = utility_target(i, j, y)[dis]
    for lam in ridges:
        w, b = ridge_fit(Xs, d, lam)
        out["models"][lam] = (st, w, b)
    return out


def predict_utility(model, X):
    st, w, b = model
    return np.clip(st(X) @ w + b, -1.0, 1.0)


def gate_from(model, X, i, j, theta: float) -> np.ndarray:
    return ((j != i) & (predict_utility(model, X) > theta)).astype(bool)


def select_gate(fitted: Dict, X_sel, i_s, j_s, y_s):
    """Maximum net gain on the selection rows over {never} U ridges x thetas. Returns config dict."""
    n = len(y_s)
    cands = [{"kind": "never", "lambda": float("inf"), "theta": None, "net": 0.0, "interventions": 0}]
    for lam, model in fitted["models"].items():
        u = predict_utility(model, X_sel)
        for th in THETAS:
            g = (j_s != i_s) & (u > th)
            net = float(((j_s == y_s).astype(float) - (i_s == y_s).astype(float))[g].sum()) / n
            cands.append({"kind": "gate", "lambda": lam, "theta": th, "net": net, "interventions": int(g.sum())})
    best = min(cands, key=lambda c: (-round(c["net"], 12), c["interventions"], -c["lambda"]))
    return best, cands


def build_q(q0: np.ndarray, pg: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.where(g[:, None], pg, q0)


def fit_temperature(q: np.ndarray, y: np.ndarray, objective: str = "nll") -> float:
    """Positive scalar T minimizing NLL (or Brier) of softmax(log q / T) on the calibration rows; 1-D bounded search."""
    lg = np.log(np.clip(q, 1e-12, 1.0))
    fn = stats.nll if objective == "nll" else stats.brier

    def f(t):
        return fn(stats.softmax(lg / np.exp(t)), y)
    r = optimize.minimize_scalar(f, bounds=(np.log(0.05), np.log(20.0)), method="bounded", options={"xatol": 1e-6})
    return float(np.exp(r.x))


def apply_temperature(q: np.ndarray, T: float) -> np.ndarray:
    return stats.softmax(np.log(np.clip(q, 1e-12, 1.0)) / T)
