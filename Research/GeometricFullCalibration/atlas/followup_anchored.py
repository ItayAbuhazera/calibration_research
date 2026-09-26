"""Anchored (residual) readouts for the regime-map follow-up (docs/regime_map_followup_amendment_1.md, A1).

    q(x) = softmax(z + f(x)),  f(x) = X_tilde(x) W + b,  ORACLE DIAGNOSTIC when fitted with target labels.

Stage 0 conventions: per-coordinate standardization on the fit rows only, mean cross-entropy + lambda ||W||_F^2 (no 1/2, no 1/K),
bias unpenalized, zero init, float64 L-BFGS with the Stage 0 tolerances/retry policy. The lambda path is
{1e-1,...,1e-5} plus the candidate f = 0 (exactly the base head, lambda = inf); selection = minimum inner-validation NLL,
ties (|dNLL| < 1e-9) -> f = 0, then the larger lambda.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import stage0_fit

torch.set_default_dtype(torch.float64)
GRID = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
INF = float("inf")
R_SEED = 20260927
GRAD_SANITY = 1e-4


def c1b_matrix(k: int = 100) -> np.ndarray:
    return np.random.default_rng(R_SEED).standard_normal((k, k)) / 10.0


class Features:
    """Fit-time feature transform: statistics come from the fit rows only. kinds:
    zonly: z | zp: [z, p] | c1b: [z, ReLU(R z~)] | c1e: [z, p_L] | c1c: h | c1d: [z, h]  (inputs are raw arrays in a dict)."""

    def __init__(self, kind: str):
        self.kind = kind

    def _raw(self, inp: Dict[str, np.ndarray]) -> np.ndarray:
        k = self.kind
        if k == "zonly":
            return inp["z"]
        if k == "zp":
            return np.concatenate([inp["z"], inp["p"]], 1)
        if k == "c1e":
            return np.concatenate([inp["z"], inp["pl"]], 1)
        if k == "c1c":
            return inp["h"]
        if k == "c1d":
            return np.concatenate([inp["z"], inp["h"]], 1)
        if k == "kernel":
            return inp["k"]
        if k == "zk":
            return np.concatenate([inp["z"], inp["k"]], 1)
        if k == "prow":
            return inp["r"]
        raise ValueError(k)

    def fit(self, inp: Dict[str, np.ndarray]) -> "Features":
        if self.kind == "c1b":
            self.m1, self.s1 = stage0_fit.standardize_fit(inp["z"])
            self.R = c1b_matrix(inp["z"].shape[1])
            phi = np.maximum(stage0_fit.standardize_apply(inp["z"], self.m1, self.s1) @ self.R.T, 0.0)
            self.m2, self.s2 = stage0_fit.standardize_fit(phi)
        else:
            self.m, self.s = stage0_fit.standardize_fit(self._raw(inp))
        return self

    def __call__(self, inp: Dict[str, np.ndarray]) -> np.ndarray:
        if self.kind == "c1b":
            zt = stage0_fit.standardize_apply(inp["z"], self.m1, self.s1)
            phi = np.maximum(zt @ self.R.T, 0.0)
            return np.concatenate([zt, stage0_fit.standardize_apply(phi, self.m2, self.s2)], 1)
        return stage0_fit.standardize_apply(self._raw(inp), self.m, self.s)


def fit_anchored(X: np.ndarray, off: np.ndarray, y: np.ndarray, lam: float, max_iter: int = stage0_fit.MAX_ITER,
                 retry_max_iter: int = stage0_fit.RETRY_MAX_ITER) -> Dict:
    n, d = X.shape
    k = off.shape[1]
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float64)); Ot = torch.from_numpy(np.ascontiguousarray(off, dtype=np.float64))
    yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
    W = torch.zeros((d, k), requires_grad=True); b = torch.zeros((k,), requires_grad=True)

    def make(mi):
        return torch.optim.LBFGS([W, b], lr=1.0, max_iter=mi, history_size=20, line_search_fn="strong_wolfe",
                                 tolerance_grad=stage0_fit.TOL_GRAD, tolerance_change=stage0_fit.TOL_CHANGE)

    def loss_fn():
        return F.cross_entropy(Ot + Xt @ W + b, yt) + lam * (W * W).sum()

    def closure():
        opt.zero_grad(); loss = loss_fn(); loss.backward(); return loss

    def state(opt_, mi):
        opt.zero_grad(); l = loss_fn(); l.backward()
        g = max(W.grad.abs().max().item(), b.grad.abs().max().item())
        cap = opt_.state[opt_.param_groups[0]["params"][0]].get("n_iter", mi) >= mi
        return g, cap, float(l)

    opt = make(max_iter); opt.step(closure)
    g, cap, loss = state(opt, max_iter)
    converged, retried = (not cap) and g <= GRAD_SANITY, False
    if not converged:
        opt = make(retry_max_iter); opt.step(closure)
        g, cap, loss = state(opt, retry_max_iter)
        converged, retried = (not cap) and g <= GRAD_SANITY, True
    return {"W": W.detach().numpy(), "b": b.detach().numpy(), "converged": bool(converged), "retried": retried, "grad_inf": g, "loss": loss}


def nll_offset(off: np.ndarray, y: np.ndarray) -> float:
    z = off - off.max(1, keepdims=True)
    lse = np.log(np.exp(z).sum(1))
    return float(np.mean(lse - z[np.arange(len(y)), y]))


def pick(table: List[Dict], tie: float = 1e-9) -> Dict:
    best = min(t["val_nll"] for t in table)
    tied = [t for t in table if abs(t["val_nll"] - best) < tie]
    return sorted(tied, key=lambda t: -t["lambda"])[0]      # inf sorts first, then larger lambda


def select_and_fit(kind: str, fit_in: Dict[str, np.ndarray], val_in: Dict[str, np.ndarray], full_in: Dict[str, np.ndarray],
                   y_fit: np.ndarray, y_val: np.ndarray, y_full: np.ndarray, grid: Tuple[float, ...] = GRID, k: int = 100) -> Dict:
    """Lambda path on inner-fit/inner-val rows (features from inner-fit statistics), then refit at the selection on all outer-train rows."""
    off_fit, off_val, off_full = fit_in["z"], val_in["z"], full_in["z"]
    feat = Features(kind).fit(fit_in)
    Xf, Xv = feat(fit_in), feat(val_in)
    table = [{"lambda": INF, "val_nll": nll_offset(off_val, y_val), "converged": True}]
    for lam in grid:
        fo = fit_anchored(Xf, off_fit, y_fit, lam)
        logits = off_val + Xv @ fo["W"] + fo["b"]
        table.append({"lambda": lam, "val_nll": nll_offset(logits, y_val), "converged": fo["converged"], "retried": fo["retried"]})
    sel = pick(table)
    out = {"table": table, "selected_lambda": sel["lambda"]}
    if sel["lambda"] == INF:
        out.update(kind=kind, feat=None, W=None, b=None, converged=True, retried=False, final_grad_inf=0.0)
    else:
        feat_full = Features(kind).fit(full_in)
        fo = fit_anchored(feat_full(full_in), off_full, y_full, sel["lambda"])
        out.update(kind=kind, feat=feat_full, W=fo["W"], b=fo["b"], converged=fo["converged"], retried=fo["retried"], final_grad_inf=fo["grad_inf"])
    return out


def predict_logits(fit: Dict, inp: Dict[str, np.ndarray]) -> np.ndarray:
    if fit["feat"] is None:
        return inp["z"].astype(np.float64)
    return inp["z"] + fit["feat"](inp) @ fit["W"] + fit["b"]
