"""Exact batched nearest-neighbour search with deterministic tie handling, and metrics.

Metrics (all on the SAME full labelled reference bank, k=50, same query IDs):
  unit_l2     Euclidean distance between unit-normalized vectors (cosine / squared-Euclidean on unit vectors give
              the same ranking and are not counted as separate evidence);
  raw_l2      Euclidean distance on the pre-L2 pooled vectors a (norm preserved, no per-coordinate standardization);
  mahalanobis regularized Mahalanobis on a with a source-only covariance: top-r eigenspace (randomized SVD, fixed
              niter) + the mean residual variance on ALL omitted directions, shrunk by gamma toward mean marginal
              variance * I. Sigma = (1-g) [U_r L_r U_r^T + s2 (I - U_r U_r^T)] + g m I. Inverse via the decomposition, so
              the orthogonal residual still contributes.
Ties: candidates (TOPK_CAND smallest) are ordered by (distance, bank index) using a stable sort, so equal
distances resolve to the smaller bank index; the first k are returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from . import spec


def order_topk(d2: torch.Tensor, k: int = spec.K_NN, cand: int = spec.TOPK_CAND) -> Tuple[torch.Tensor, torch.Tensor]:
    """d2: [b, M] squared distances. Returns (dist [b,k] sqrt, idx [b,k] int64), sorted by (distance, index)."""
    vals, idx = torch.topk(d2, cand, dim=1, largest=False)
    o1 = torch.argsort(idx, dim=1)
    idx, vals = torch.gather(idx, 1, o1), torch.gather(vals, 1, o1)
    o2 = torch.argsort(vals, dim=1, stable=True)
    idx, vals = torch.gather(idx, 1, o2)[:, :k], torch.gather(vals, 1, o2)[:, :k]
    return vals.clamp_min(0).sqrt(), idx


def knn_unit(q: torch.Tensor, bank: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """q [b,d], bank [M,d], both unit-normalized: d^2 = 2 - 2 q.b."""
    return order_topk(2.0 - 2.0 * (q @ bank.T))


def knn_raw(q: torch.Tensor, bank: torch.Tensor, bank_sq: torch.Tensor):
    """Norm-preserving Euclidean: d^2 = |q|^2 + |b|^2 - 2 q.b (float32; bank_sq = |b|^2 precomputed)."""
    d2 = q.pow(2).sum(1, keepdim=True) + bank_sq.unsqueeze(0) - 2.0 * (q @ bank.T)
    return order_topk(d2)


@dataclass
class LowRankMahalanobis:
    mu: torch.Tensor          # [d]
    U: torch.Tensor           # [d, r]
    eig_r: torch.Tensor       # [r] covariance eigenvalues in the top-r space (unshrunk)
    e_r: torch.Tensor         # [r] shrunk eigenvalues
    e_res: float              # shrunk residual eigenvalue
    info: Dict

    @staticmethod
    def fit(bank: torch.Tensor, seed: int = 0, rank: int = spec.MAH_RANK, gamma: float = spec.MAH_SHRINK,
            niter: int = spec.MAH_NITER) -> "LowRankMahalanobis":
        n, d = bank.shape
        r = int(min(rank, d, n - 1))
        mu = bank.mean(0)
        A = bank - mu                                   # centered copy
        total_var = float((A.pow(2).sum() / (n - 1)).item())   # trace of the sample covariance
        torch.manual_seed(seed)
        if r >= d:                                       # small d: exact eigendecomposition
            cov = (A.T @ A) / (n - 1)
            w, V = torch.linalg.eigh(cov.double())
            order = torch.argsort(w, descending=True)[:r]
            lam, U = w[order].float(), V[:, order].float()
        else:
            _, S, Vh = torch.svd_lowrank(A, q=min(r + 20, d, n), niter=niter)
            lam, U = (S[:r] ** 2 / (n - 1)), Vh[:, :r]
        m = total_var / d
        res = max((total_var - float(lam.sum())) / max(d - r, 1), spec.MAH_FLOOR)
        e_r = (1 - gamma) * lam + gamma * m
        e_res = (1 - gamma) * res + gamma * m
        info = {"rank": r, "dim": d, "shrinkage": gamma, "mean_marginal_variance": m, "residual_variance": res,
                "top_eigenvalues": lam[:5].tolist(), "eig_r_min": float(lam[-1]), "explained_frac": float(lam.sum()) / total_var,
                "condition_number_shrunk": float(max(e_r.max().item(), e_res) / min(e_r.min().item(), e_res)),
                "svd_niter": niter if r < d else None, "exact": bool(r >= d)}
        del A
        return LowRankMahalanobis(mu, U, lam, e_r, float(max(e_res, spec.MAH_FLOOR)), info)

    def prepare_bank(self, bank: torch.Tensor, inplace: bool = False) -> Dict[str, torch.Tensor]:
        """Whitened top-r coordinates w, raw top-r coordinates z, residual squared norm and the centered bank.
        ``inplace=True`` centers ``bank`` in place (no second copy of a large bank)."""
        c = bank.sub_(self.mu) if inplace else bank - self.mu
        z = c @ self.U
        w = z / self.e_r.sqrt()
        rsq = c.pow(2).sum(1) - z.pow(2).sum(1)
        return {"c": c, "z": z, "w": w, "rsq": rsq}

    def knn(self, q: torch.Tensor, pb: Dict[str, torch.Tensor]):
        qc = q - self.mu
        zq = qc @ self.U
        wq = zq / self.e_r.sqrt()
        rq = qc.pow(2).sum(1) - zq.pow(2).sum(1)
        top = (wq.pow(2).sum(1, keepdim=True) + pb["w"].pow(2).sum(1)[None] - 2 * wq @ pb["w"].T)
        G = qc @ pb["c"].T - zq @ pb["z"].T           # residual-subspace inner products
        res = (rq[:, None] + pb["rsq"][None] - 2 * G).clamp_min(0) / self.e_res
        return order_topk(top.clamp_min(0) + res)
