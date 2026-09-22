"""Decision statistics for frozen challengers (kNN diagnostic classifier and gates).

Conventions
-----------
* base prediction i = argmax of the base logits; challenger prediction j.  D = 1{j=Y} - 1{i=Y} in {-1,0,+1}.
* W: base wrong -> challenger correct;  H: base correct -> challenger wrong;  U: base wrong -> different wrong.
  ``total_flips = W+H+U``; ``intervention_precision = W/total_flips``; ``decisive_precision = W/(W+H)``
  (never call the latter intervention precision). Undefined ratios are ``None``, never 0.
* p_geo(c|x) = (n_c(x) + alpha*pi_c)/(k+alpha)  with n_c the class count among the k neighbours and pi_c
  the SOURCE-bank class prior; a smoothed count distribution, not a calibrated posterior. j = argmax p_geo with
  the lowest class index winning ties (np.argmax semantics).
* GT rank: 1 + number of classes ordered ahead of the true class by (-probability, class index) -- deterministic.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import spec


def _r(a, b) -> Optional[float]:
    return None if b == 0 else float(a) / float(b)


def p_geo(neigh_idx: np.ndarray, bank_labels: np.ndarray, prior: np.ndarray, alpha: float = spec.ALPHA):
    """neigh_idx [N,k] indices into the bank -> (p [N,C] float64, counts [N,C])."""
    n, k = neigh_idx.shape
    c = len(prior)
    lab = bank_labels[neigh_idx]
    counts = np.zeros((n, c), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(n), k), lab.ravel()), 1.0)
    return (counts + alpha * prior[None]) / (k + alpha), counts


def neighborhood_entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-300, 1.0)
    return -(p * np.log(q)).sum(1)


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Sum-convention multiclass Brier: mean_n sum_c (p_nc - 1{c=y_n})^2."""
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return float(np.mean(((p - oh) ** 2).sum(1)))


def gt_rank(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    pt = p[np.arange(len(y)), y][:, None]
    cls = np.arange(p.shape[1])[None]
    ahead = (p > pt) | ((p == pt) & (cls < y[:, None]))
    return ahead.sum(1) + 1


def wh_stats(base: np.ndarray, cand: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    """Counts and ratios behind every reported number (W/H/U, precisions, rescued fraction, harm rate, net utility)."""
    n = len(y)
    b_ok, c_ok = base == y, cand == y
    ch = base != cand
    W = int((ch & ~b_ok & c_ok).sum()); H = int((ch & b_ok & ~c_ok).sum()); U = int((ch & ~b_ok & ~c_ok).sum())
    nerr, nok = int((~b_ok).sum()), int(b_ok.sum())
    F = W + H + U
    out = {"n": n, "W": W, "H": H, "U": U, "total_flips": F, "net": W - H, "net_utility": (W - H) / n,
           "n_base_errors": nerr, "n_base_correct": nok,
           "candidate_acc_among_base_errors": _r(int((c_ok & ~b_ok).sum()), nerr),
           "harm_rate_among_base_correct": _r(int((b_ok & ~c_ok).sum()), nok),
           "intervention_precision": _r(W, F), "decisive_precision": _r(W, W + H),
           "rescued_fraction_of_base_errors": _r(W, nerr), "intervention_rate": F / n,
           "base_acc": float(b_ok.mean()), "cand_acc": float(c_ok.mean())}
    assert abs(out["net_utility"] - (out["cand_acc"] - out["base_acc"])) < 1e-12   # DeltaAcc = (W-H)/N
    return out


def candidate_row(pg: np.ndarray, counts: np.ndarray, kth_radius: np.ndarray, base_pred: np.ndarray, y: np.ndarray) -> Dict:
    """One (site, pool, query-set) record for the frozen kNN challenger."""
    j = pg.argmax(1)
    r = wh_stats(base_pred, j, y)
    r.update({"local_true_label_proportion": float(counts[np.arange(len(y)), y].mean() / counts.sum(1).mean()),
              "geo_nll": nll(pg, y), "geo_brier": brier(pg, y), "geo_gt_rank_mean": float(gt_rank(pg, y).mean()),
              "kth_radius_mean": float(np.mean(kth_radius))})
    return r
