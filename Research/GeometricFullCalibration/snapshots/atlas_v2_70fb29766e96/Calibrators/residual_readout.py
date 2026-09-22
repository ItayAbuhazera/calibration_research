"""
Math for the residual-evidence study (docs/residual_evidence_study_spec.md).

A strong output-only ANCHOR B(z) is fitted first and frozen; then, for each evidence
source F, a residual readout

    t_F(x) = B(z(x)) + W_F phi_F(x),      q_F = softmax(t_F)

is fitted on the SAME clean-fit rows by

    mean_NLL(q_F, y) + lambda * ||W_F||_F^2 / (2K)          (W class-centered).

Everything here is deterministic (float64, CPU, zero init, fixed L-BFGS settings)
and reads only clean-role arrays. Selection policies take precomputed clean-selection
scores; nothing in this module can see an evaluation label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

K = 100  # classes == readout dimension d
LAMBDAS: Tuple[float, ...] = (1e-4, 1e-2, 1.0, 100.0)
VAR_FLOOR = 1e-6
NLL_TOLERANCE = 0.01  # Decision-policy allowance vs the anchor (empirical, not a risk certificate)
LBFGS_MAX_ITER = 500
LBFGS_MAX_ITER_CONTINUED = 2000  # one deterministic continuation if not converged
GRAD_TOL = 1e-4  # converged iff ||grad||_inf <= GRAD_TOL at the returned point

HIDDEN_FAMILIES: Tuple[str, ...] = ("G", "S", "DG", "DS")
OUTPUT_FAMILIES: Tuple[str, ...] = ("O", "DL")
ALL_FAMILIES: Tuple[str, ...] = HIDDEN_FAMILIES + OUTPUT_FAMILIES
FAMILY_ORDER: Dict[str, int] = {f: i for i, f in enumerate(ALL_FAMILIES)}

# data-independent seeds for the frozen random maps (spec §5)
SEED_MAP_G = 20260922
SEED_MAP_S = 20260923
SEED_MAP_O = 20260924


# ---------------------------------------------------------------------------
# frozen random maps and standardization
# ---------------------------------------------------------------------------


def gaussian_map(d_in: int, d_out: int, seed: int) -> np.ndarray:
    """Gaussian random map [d_out, d_in], entries N(0, 1/d_out): E||Px||^2 = ||x||^2."""
    g = np.random.default_rng(seed)
    return g.normal(0.0, 1.0 / np.sqrt(d_out), size=(d_out, d_in))


def logit_nonlinear_map(seed: int, k: int = K) -> Tuple[np.ndarray, np.ndarray]:
    """B_ij ~ N(0, 1/K), c_i ~ N(0, 1) for phi_O = ReLU(B * standardized_logits + c)."""
    g = np.random.default_rng(seed)
    return g.normal(0.0, 1.0 / np.sqrt(k), size=(k, k)), g.normal(0.0, 1.0, size=k)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=np.float64)
        return Standardizer(x.mean(0), np.sqrt(x.var(0) + VAR_FLOOR))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std


@dataclass
class FeatureMap:
    """phi_F: raw evidence -> standardized 100-d readout features (fit-only statistics)."""

    family: str
    proj: Optional[np.ndarray]  # G/S: [100, d_in]
    nl_B: Optional[np.ndarray] = None  # O
    nl_c: Optional[np.ndarray] = None
    pre: Optional[Standardizer] = None  # O: standardizer of the raw logits
    post: Optional[Standardizer] = None

    def raw(self, ev: np.ndarray) -> np.ndarray:
        ev = np.asarray(ev, dtype=np.float64)
        if self.family in ("G", "S"):
            return ev @ self.proj.T  # no post-projection L2 renormalization
        if self.family == "O":
            return np.maximum(self.pre(ev) @ self.nl_B.T + self.nl_c, 0.0)
        return ev  # DG / DS / DL: the full K-dimensional radius vector, unprojected

    def __call__(self, ev: np.ndarray) -> np.ndarray:
        return self.post(self.raw(ev))


def make_feature_map(family: str, ev_fit: np.ndarray) -> FeatureMap:
    """Build phi_F from FIT rows only (the maps themselves are data-independent)."""
    if family == "G":
        fm = FeatureMap(family, gaussian_map(ev_fit.shape[1], K, SEED_MAP_G))
    elif family == "S":
        fm = FeatureMap(family, gaussian_map(ev_fit.shape[1], K, SEED_MAP_S))
    elif family == "O":
        Bm, c = logit_nonlinear_map(SEED_MAP_O)
        fm = FeatureMap(family, None, Bm, c, pre=Standardizer.fit(ev_fit))
    elif family in ("DG", "DS", "DL"):
        fm = FeatureMap(family, None)
    else:
        raise ValueError(family)
    fm.post = Standardizer.fit(fm.raw(ev_fit))
    return fm


def projection_distortion(x: np.ndarray, proj: np.ndarray, n_pairs: int = 2000, seed: int = 0) -> Dict[str, float]:
    """||P(u-v)|| / ||u-v|| on fixed clean fit pairs (disclosure only; never used to pick a map)."""
    g = np.random.default_rng(seed)
    n = x.shape[0]
    i, j = g.integers(0, n, n_pairs), g.integers(0, n, n_pairs)
    keep = i != j
    d = x[i[keep]] - x[j[keep]]
    r = np.linalg.norm(d @ proj.T, axis=1) / np.maximum(np.linalg.norm(d, axis=1), 1e-12)
    return {"n_pairs": int(keep.sum()), "ratio_mean": float(r.mean()), "ratio_std": float(r.std()),
            "ratio_min": float(r.min()), "ratio_max": float(r.max()),
            "rel_sq_dist_error_p95": float(np.quantile(np.abs(r ** 2 - 1.0), 0.95))}


def jl_required_dim(n: int, eps: float) -> int:
    """Dasgupta-Gupta sufficient k = 4 ln n / (eps^2/2 - eps^3/3) (a sufficient, worst-case bound)."""
    return int(np.ceil(4.0 * np.log(n) / (eps ** 2 / 2.0 - eps ** 3 / 3.0)))


def effective_rank(x: np.ndarray) -> Dict[str, float]:
    """Spectrum summary of centered features: participation ratio and entropy-based rank."""
    xc = np.asarray(x, dtype=np.float64) - np.mean(x, axis=0)
    s = np.linalg.svd(xc, compute_uv=False) ** 2
    s = s / s.sum()
    return {"participation_ratio": float(1.0 / np.sum(s ** 2)),
            "entropy_rank": float(np.exp(-np.sum(s[s > 0] * np.log(s[s > 0])))),
            "top1_var_frac": float(s[0]), "n_components_90pct": int(np.searchsorted(np.cumsum(s), 0.9) + 1)}


# ---------------------------------------------------------------------------
# losses / fitting
# ---------------------------------------------------------------------------


def nll_np(logits: np.ndarray, y: np.ndarray) -> float:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    lse = np.log(np.exp(z).sum(1))
    return float(np.mean(lse - z[np.arange(len(y)), y]))


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def center_classes(w: torch.Tensor) -> torch.Tensor:
    """W is [K classes, d]: remove the common-softmax-offset component W_k -> W_k - mean_k W_k."""
    return w - w.mean(dim=0, keepdim=True)


def _lbfgs(params: List[torch.Tensor], closure_loss, max_iter: int) -> Dict[str, Any]:
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        loss = closure_loss()
        loss.backward()
        return loss

    opt.step(closure)
    n_iter = int(opt.state[opt._params[0]].get("n_iter", -1))
    opt.zero_grad()
    loss = closure_loss()
    loss.backward()
    gmax = float(max(p.grad.abs().max() for p in params))
    return {"n_iter": n_iter, "grad_inf": gmax, "loss": float(loss)}


def _fit_with_continuation(params, closure_loss) -> Dict[str, Any]:
    info = _lbfgs(params, closure_loss, LBFGS_MAX_ITER)
    info["continued"] = False
    if info["grad_inf"] > GRAD_TOL:  # one deterministic continuation
        info = _lbfgs(params, closure_loss, LBFGS_MAX_ITER_CONTINUED)
        info["continued"] = True
    info["converged"] = bool(info["grad_inf"] <= GRAD_TOL)
    return info


def fit_matrix_scaling(z: np.ndarray, y: np.ndarray, lam: float) -> Dict[str, Any]:
    """B(z)=A z + b minimizing mean NLL + lam (||A-I||_F^2 + ||b||^2) / (2K); start at the identity."""
    zt = torch.from_numpy(np.asarray(z, dtype=np.float64))
    yt = torch.from_numpy(np.asarray(y)).long()
    A = torch.eye(K, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    eye = torch.eye(K, dtype=torch.float64)

    def loss():
        return F.cross_entropy(zt @ A.T + b, yt) + lam * (((A - eye) ** 2).sum() + (b ** 2).sum()) / (2 * K)

    info = _fit_with_continuation([A, b], loss)
    info.update({"A": A.detach().numpy().copy(), "b": b.detach().numpy().copy(), "lambda": float(lam)})
    return info


def apply_matrix_scaling(z: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(z, dtype=np.float64) @ A.T + b


def fit_residual(anchor_logits: np.ndarray, phi: np.ndarray, y: np.ndarray, lam: float) -> Dict[str, Any]:
    """W in R^{K x d}: minimize mean NLL(softmax(B(z) + W phi), y) + lam ||W_c||_F^2 / (2K); start at W=0."""
    bt = torch.from_numpy(np.asarray(anchor_logits, dtype=np.float64))
    pt = torch.from_numpy(np.asarray(phi, dtype=np.float64))
    yt = torch.from_numpy(np.asarray(y)).long()
    W = torch.zeros(K, pt.shape[1], dtype=torch.float64, requires_grad=True)

    def loss_c():
        Wc = center_classes(W)
        return F.cross_entropy(bt + pt @ Wc.T, yt) + lam * (Wc ** 2).sum() / (2 * K)

    info = _fit_with_continuation([W], loss_c)
    Wc = center_classes(W).detach().numpy().copy()
    info.update({"W": Wc, "lambda": float(lam)})
    return info


def correction_logits(phi: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.asarray(phi, dtype=np.float64) @ W.T


# ---------------------------------------------------------------------------
# anchor + selection policies (clean-selection data only)
# ---------------------------------------------------------------------------


def select_anchor(cands: Sequence[Dict[str, Any]]) -> int:
    """cands: dicts with 'select_nll'. Lowest NLL; deterministic tie -> earlier candidate."""
    best = 0
    for i, c in enumerate(cands):
        if c["select_nll"] < cands[best]["select_nll"] - 1e-12:
            best = i
    return best


def nll_policy(pool: Sequence[Dict[str, Any]]) -> int:
    """Index of the minimum clean-selection NLL; ties -> larger lambda, then family order."""
    def key(i):
        c = pool[i]
        return (round(c["select_nll"], 12), -c["lambda"], FAMILY_ORDER.get(c["family"], -1))
    return min(range(len(pool)), key=key)


def decision_policy(pool: Sequence[Dict[str, Any]], anchor_nll: float, tol: float = NLL_TOLERANCE) -> int:
    """Maximize clean-selection accuracy s.t. NLL <= anchor NLL + tol. Ties: lower NLL, larger lambda,
    fixed family order. The pool MUST contain the anchor (family 'zero', lambda=inf), which is always feasible."""
    feas = [i for i, c in enumerate(pool) if c["select_nll"] <= anchor_nll + tol + 1e-12]
    def key(i):
        c = pool[i]
        return (-round(c["select_acc"], 12), round(c["select_nll"], 12), -c["lambda"], FAMILY_ORDER.get(c["family"], -1))
    return min(feas, key=key)


def zero_candidate(anchor_nll: float, anchor_acc: float) -> Dict[str, Any]:
    return {"family": "zero", "lambda": float("inf"), "select_nll": anchor_nll, "select_acc": anchor_acc, "W": None}


# ---------------------------------------------------------------------------
# fixed-edge binned diagnostics
# ---------------------------------------------------------------------------


def quantile_edges(x: np.ndarray, n_bins: int = 5) -> np.ndarray:
    e = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.asarray(e, dtype=np.float64)


def bin_utility(bin_values: np.ndarray, edges: np.ndarray, base_pred: np.ndarray, pred: np.ndarray,
                y: np.ndarray) -> List[Dict[str, Any]]:
    """Per-bin intervention rate and signed utility (W-H)/n_bin, using edges fixed on clean-selection data."""
    b = np.searchsorted(edges, bin_values, side="right")
    out = []
    for k in range(len(edges) + 1):
        m = b == k
        n = int(m.sum())
        if n == 0:
            out.append({"bin": k, "n": 0, "intervention_rate": None, "signed_utility": None, "W": 0, "H": 0, "U": 0})
            continue
        ch = (pred[m] != base_pred[m])
        W = int(np.sum(ch & (pred[m] == y[m])))
        H = int(np.sum(ch & (base_pred[m] == y[m])))
        U = int(np.sum(ch)) - W - H
        out.append({"bin": k, "n": n, "intervention_rate": float(ch.mean()), "signed_utility": (W - H) / n,
                    "W": W, "H": H, "U": U})
    return out
