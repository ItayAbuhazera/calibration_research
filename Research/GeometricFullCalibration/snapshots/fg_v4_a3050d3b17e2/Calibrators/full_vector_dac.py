"""
Full-Vector Density-Aware Calibration (FV-DAC).

A minimal, low-capacity, class-symmetric extension of native Density-Aware
Calibration (Tomani et al., ICML 2023) whose only structural change is that
the reference-bank search becomes *class-conditioned*.

--------------------------------------------------------------------------
THE ONE EQUATION THAT MATTERS
--------------------------------------------------------------------------

Native DAC produces a sample-dependent positive scalar temperature

    S(x, w) = max( w_0 + sum_l w_l * s_l(x), 1e-12 )
    q_DAC(x) = softmax( z(x) / S(x, w) )

and, because S > 0, is order-preserving:  argmax_k q_DAC = argmax_k z.
Native DAC therefore cannot change the decision, by construction.

FV-DAC keeps every piece of DAC's machinery -- the same selected layers,
the same spatial pooling, the same L2 normalization, the same Euclidean
distance convention, the same reference bank, the same fitted layer weights
and the same frozen S(x) -- and changes ONLY the domain of the kNN search:

    B_{l,k}    = { h_l(x_i) : y_i = k }                (class-conditioned bank)
    r_{l,k}(x) = K_c-th nearest-neighbour distance( h_l(x), B_{l,k} )
    alpha_l    = w_l_hat / sum_j w_j_hat               (frozen, normalized)
    R_k(x)     = sum_l alpha_l * r_{l,k}(x)

    q_beta(x)  = softmax( ( z(x) - beta * R(x) ) / S_DAC(x) ),   beta >= 0

Sign convention: a SMALLER class-conditioned distance means LESS penalty and
therefore MORE support for that class.

Critical properties, all covered by tests in tests/test_full_vector_dac.py:
  * beta = 0 reproduces native DAC *numerically*, not approximately;
  * exactly one new free scalar in the primary arm;
  * class-symmetric -- no per-class bias, no per-class scaling, no MLP, no
    gating network;
  * gauge invariant -- adding a common constant to every logit leaves the
    output unchanged (this is why per-class temperature q_k ∝ exp(z_k/T_k(x))
    was rejected as the primary form; see docs/full_vector_dac_experiment.md);
  * it CAN change the argmax.

--------------------------------------------------------------------------
ARMS
--------------------------------------------------------------------------

fv_dac_nll           primary; beta fitted by NLL on clean inner-fit data
fv_dac_brier         predeclared objective sensitivity (native DAC is
                     Brier-fitted, so this asks whether the objective matters)
fv_dac_lognorm       Sensitivity A -- conditioning:
                     v_{l,k} = -(log r_{l,k} - mu_{l,k}) / sigma_l
                     q = softmax( (z + beta * V) / S_DAC )
fv_dac_shared_layer  Sensitivity B -- decision-specific layer weights:
                     q = softmax( (z - sum_l b_l * r_{l,·}) / S_DAC ),
                     b_l >= 0, SHARED across classes (O(L) params, never b_{l,k})
fv_dac_permuted      negative control -- seeded permutation of bank labels
fv_dac_logit_space   mechanism control -- same operator on centered,
                     L2-normalized logits; S_DAC (hidden-space) unchanged
fv_dac_density_only  diagnostic -- argmin_k R_k(x), never fitted

Nothing in this module reads corrupted data, and nothing in it is called
from Experiments/run_unified_benchmark.py; FV-DAC is a standalone study
until a formulation is frozen and validated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import optimize

# --------------------------------------------------------------------------
# Frozen protocol constants. These are design points, not a tuning grid --
# see docs/full_vector_dac_experiment.md and the preregistration note
# ResearchBrain/05_Experiments/2026-09-20 Full-Vector DAC POC.md.
# --------------------------------------------------------------------------

#: Predeclared class-conditioned neighbour orders.
#:   5   = strongly local
#:   20  = intermediate
#:   200 = literal same-K extension (native CIFAR-100 DAC uses K = 200)
#: NOT to be expanded if results are poor.
PREDECLARED_KC = (5, 20, 200)

#: Upper bound of the 1-D beta search. Generous by design; the frozen state
#: records whether the optimum landed on the boundary so a pinned beta is
#: visible rather than silent.
BETA_SEARCH_MAX = 1000.0

#: Base seed for the permuted-bank-label negative control. The effective
#: seed is PERMUTED_LABEL_SEED_BASE + checkpoint_seed, and it is recorded in
#: the frozen state together with a hash of the permuted label vector.
PERMUTED_LABEL_SEED_BASE = 20260920

#: Native DAC clips its temperature at this floor; FV-DAC reuses the value
#: verbatim so S_DAC is bit-identical to the frozen calibrator's.
TEMPERATURE_FLOOR = 1e-12

ARM_PRIMARY = "fv_dac_nll"
ARM_BRIER = "fv_dac_brier"
ARM_LOGNORM = "fv_dac_lognorm"
ARM_SHARED_LAYER = "fv_dac_shared_layer"
ARM_PERMUTED = "fv_dac_permuted"
ARM_LOGIT_SPACE = "fv_dac_logit_space"
ARM_DENSITY_ONLY = "fv_dac_density_only"

#: Every fitted FV-DAC arm, in report order. The diagnostic is excluded
#: because it has no fitted parameter.
FITTED_ARMS = (
    ARM_PRIMARY,
    ARM_BRIER,
    ARM_LOGNORM,
    ARM_SHARED_LAYER,
    ARM_PERMUTED,
    ARM_LOGIT_SPACE,
)


# ==========================================================================
# Preprocessing / distance -- deliberately identical to
# Calibrators/density_aware_calibration.py::LayerKNNScorer
# ==========================================================================


def dac_preprocess(features: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    """DAC's own preprocessing: spatial average (if needed), then L2 normalize.

    Mirrors LayerKNNScorer._preprocess exactly. Kept as a separate function
    so a test can assert the two agree rather than trusting the comment.
    """
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    features = features.float().to(device)
    if features.ndim == 4:
        features = features.mean(dim=(2, 3))
    elif features.ndim == 3:
        features = features.mean(dim=1)
    return F.normalize(features, p=2, dim=-1)


def normalized_euclidean(query: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """||a - b|| for L2-normalized a, b, via ||a-b||^2 = 2 - 2<a,b>.

    This is a TRUE Euclidean distance (the square root IS taken), matching
    native DAC. Note for anyone porting this to FAISS: FAISS's L2 index
    returns SQUARED distances, so a port must take the square root or every
    distance-linear quantity below changes meaning. This implementation uses
    PyTorch and is not affected.
    """
    sq = 2.0 - 2.0 * (query @ bank.t())
    return torch.sqrt(torch.clamp(sq, min=0.0))


def np_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=1, keepdims=True)


# ==========================================================================
# Class-conditioned kNN operator
# ==========================================================================


class ClassConditionalKNN:
    """K_c-th nearest-neighbour distance to each class's slice of one bank.

    The bank is assumed already DAC-preprocessed (spatially averaged and L2
    normalized), so this class performs no preprocessing of its own beyond
    normalizing the queries -- keeping the representation definition in one
    place.
    """

    def __init__(
        self,
        bank: torch.Tensor,
        bank_labels: np.ndarray,
        num_classes: int,
        device: torch.device,
    ) -> None:
        if bank.ndim != 2:
            raise ValueError(f"bank must be 2D [M, d], got {tuple(bank.shape)}")
        if len(bank_labels) != bank.shape[0]:
            raise ValueError(
                f"bank has {bank.shape[0]} rows but {len(bank_labels)} labels"
            )
        self.device = device
        self.num_classes = int(num_classes)
        self.bank = bank.to(device).float()
        self.bank_labels = np.asarray(bank_labels).astype(np.int64)

        counts = np.bincount(self.bank_labels, minlength=self.num_classes)
        if np.any(counts == 0):
            empty = np.flatnonzero(counts == 0).tolist()
            raise ValueError(f"reference bank has no examples for classes {empty}")
        self.class_counts = counts
        self.min_class_count = int(counts.min())

        # (C, max_n) gather table; padded slots are masked to +inf so they can
        # never be selected by topk/kthvalue.
        max_n = int(counts.max())
        idx = np.zeros((self.num_classes, max_n), dtype=np.int64)
        pad = np.ones((self.num_classes, max_n), dtype=bool)
        for k in range(self.num_classes):
            members = np.flatnonzero(self.bank_labels == k)
            idx[k, : len(members)] = members
            pad[k, : len(members)] = False
        self._idx = torch.from_numpy(idx).to(device)
        self._pad = torch.from_numpy(pad).to(device)
        self._max_n = max_n

    def validate_kc(self, kc_values: Sequence[int]) -> None:
        """Every K_c must be a valid neighbour order for EVERY class."""
        for kc in kc_values:
            if kc < 1:
                raise ValueError(f"K_c must be >= 1, got {kc}")
            if kc > self.min_class_count:
                raise ValueError(
                    f"K_c={kc} exceeds the smallest class bank size "
                    f"({self.min_class_count}); no valid K_c-th neighbour exists "
                    "for that class."
                )

    def class_distances(
        self,
        queries: torch.Tensor,
        kc_values: Sequence[int],
        *,
        batch_size: int = 256,
        also_global_k: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """r_{·,k}(x) for each requested K_c.

        Returns
        -------
        r : [N, C, len(kc_values)] float32
        s_global : [N] float32 or None -- the CLASS-AGNOSTIC k-th NN distance
            over the whole bank, i.e. native DAC's own s_l(x). Computed from
            the same distance matrix so the two can be checked against the
            frozen DAC calibrator without a second pass.
        """
        kc_values = list(kc_values)
        self.validate_kc(kc_values)
        max_kc = max(kc_values)
        positions = torch.tensor([kc - 1 for kc in kc_values], device=self.device)

        queries = F.normalize(queries.to(self.device).float(), p=2, dim=-1)
        n = queries.shape[0]
        out = np.empty((n, self.num_classes, len(kc_values)), dtype=np.float32)
        glob = np.empty(n, dtype=np.float32) if also_global_k is not None else None

        flat_idx = self._idx.reshape(-1)
        for start in range(0, n, batch_size):
            chunk = queries[start : start + batch_size]
            dist = normalized_euclidean(chunk, self.bank)  # [b, M]

            if also_global_k is not None:
                if also_global_k <= dist.shape[1]:
                    g, _ = torch.kthvalue(dist, also_global_k, dim=1)
                else:  # native DAC's own fallback
                    g, _ = dist.max(dim=1)
                glob[start : start + chunk.shape[0]] = g.cpu().numpy()

            out[start : start + chunk.shape[0]] = self._gather(dist, max_kc, positions)
            del dist
        return out, glob

    def _gather(
        self, dist: torch.Tensor, max_kc: int, positions: torch.Tensor
    ) -> np.ndarray:
        """Per-class K_c-th smallest distance for one precomputed chunk."""
        per_class = dist[:, self._idx.reshape(-1)].view(
            dist.shape[0], self.num_classes, self._max_n
        )
        per_class = per_class.masked_fill(self._pad.unsqueeze(0), float("inf"))
        vals = torch.topk(per_class, k=max_kc, dim=2, largest=False).values
        result = vals.index_select(2, positions).cpu().numpy()
        del per_class, vals
        return result


def class_distances_multi(
    operators: Sequence[ClassConditionalKNN],
    queries: torch.Tensor,
    kc_values: Sequence[int],
    *,
    batch_size: int = 256,
    also_global_k: Optional[int] = None,
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """Run several label-tables over ONE shared bank with a single distance pass.

    Used so the permuted-bank-label negative control costs a gather rather
    than a second full distance computation -- and, more importantly, so the
    control provably sees the *same* distances as the real-label arm rather
    than a recomputed copy.
    """
    if not operators:
        raise ValueError("at least one operator is required")
    bank = operators[0].bank
    for op in operators[1:]:
        if op.bank.data_ptr() != bank.data_ptr():
            raise ValueError(
                "class_distances_multi requires every operator to share one bank "
                "tensor; otherwise the controls would not see identical distances."
            )
    kc_values = list(kc_values)
    for op in operators:
        op.validate_kc(kc_values)

    device = operators[0].device
    max_kc = max(kc_values)
    positions = torch.tensor([kc - 1 for kc in kc_values], device=device)

    q = F.normalize(queries.to(device).float(), p=2, dim=-1)
    n = q.shape[0]
    outs = [
        np.empty((n, op.num_classes, len(kc_values)), dtype=np.float32)
        for op in operators
    ]
    glob = np.empty(n, dtype=np.float32) if also_global_k is not None else None

    for start in range(0, n, batch_size):
        chunk = q[start : start + batch_size]
        dist = normalized_euclidean(chunk, bank)
        if also_global_k is not None:
            if also_global_k <= dist.shape[1]:
                g, _ = torch.kthvalue(dist, also_global_k, dim=1)
            else:
                g, _ = dist.max(dim=1)
            glob[start : start + chunk.shape[0]] = g.cpu().numpy()
        for op, out in zip(operators, outs):
            out[start : start + chunk.shape[0]] = op._gather(dist, max_kc, positions)
        del dist
    return outs, glob


# ==========================================================================
# Aggregation and the correction itself
# ==========================================================================


def normalized_layer_weights(dac_layer_weights: Sequence[float]) -> Tuple[np.ndarray, bool]:
    """alpha_l = w_l_hat / sum_j w_j_hat, with a documented degenerate fallback.

    Returns (alpha, degenerate). If the fitted DAC layer weights sum to zero
    (or anything non-positive), alpha falls back to uniform 1/L and the flag
    is True so the frozen state records that the native weighting carried no
    information rather than silently pretending it did.
    """
    w = np.asarray(dac_layer_weights, dtype=np.float64)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(w), 1.0 / len(w)), True
    return w / total, False


def dac_temperature(
    global_scores: np.ndarray,
    layer_weights: Sequence[float],
    intercept: float,
) -> np.ndarray:
    """S(x, w) = max(sum_l w_l s_l(x) + w_0, 1e-12) -- native DAC's Eq. 7.

    `global_scores` is [N, L] of class-AGNOSTIC k-th NN distances.
    """
    t = np.asarray(global_scores, dtype=np.float64) @ np.asarray(
        layer_weights, dtype=np.float64
    ) + float(intercept)
    return np.maximum(t, TEMPERATURE_FLOOR)


def aggregate_R(r_layers: np.ndarray, alpha: Sequence[float]) -> np.ndarray:
    """R_k(x) = sum_l alpha_l * r_{l,k}(x).  r_layers is [N, L, C] -> [N, C]."""
    r_layers = np.asarray(r_layers, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if r_layers.shape[1] != alpha.shape[0]:
        raise ValueError(
            f"r_layers has {r_layers.shape[1]} layers but alpha has {alpha.shape[0]}"
        )
    return np.einsum("nlc,l->nc", r_layers, alpha)


def fv_dac_probs(
    logits: np.ndarray,
    correction: np.ndarray,
    temperature: np.ndarray,
    beta: float,
    *,
    sign: float = -1.0,
) -> np.ndarray:
    """q = softmax( (z + sign * beta * correction) / S ).

    sign = -1 for a DISTANCE correction (primary, shared-layer, permuted,
    logit-space): larger distance -> larger penalty.
    sign = +1 for the lognorm SUPPORT correction, whose transform already
    carries the minus sign.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if beta == 0.0:
        shifted = logits
    else:
        correction = np.asarray(correction, dtype=np.float64)
        if correction.shape != logits.shape:
            raise ValueError(
                f"correction shape {correction.shape} != logits shape {logits.shape}"
            )
        shifted = logits + sign * float(beta) * correction
    return np_softmax(shifted / np.asarray(temperature, dtype=np.float64)[:, None])


def shared_layer_probs(
    logits: np.ndarray,
    r_layers: np.ndarray,
    temperature: np.ndarray,
    b: Sequence[float],
) -> np.ndarray:
    """q = softmax( (z - sum_l b_l * r_{l,·}) / S ).

    `b` is SHARED across classes by construction: it is indexed by layer
    only, so there is no way for a per-class parameter to enter. This is
    asserted by tests/test_full_vector_dac.py::test_shared_layer_is_class_symmetric.
    """
    b = np.asarray(b, dtype=np.float64)
    penalty = np.einsum("nlc,l->nc", np.asarray(r_layers, dtype=np.float64), b)
    return np_softmax(
        (np.asarray(logits, dtype=np.float64) - penalty)
        / np.asarray(temperature, dtype=np.float64)[:, None]
    )


def lognorm_transform(
    r_layers: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """v_{l,k}(x) = -( log r_{l,k}(x) - mu_{l,k} ) / sigma_l.

    mu is [L, C] (per layer AND class), sigma is [L] (per layer only), both
    estimated on clean fitting data and then frozen.
    """
    r = np.asarray(r_layers, dtype=np.float64)
    logr = np.log(np.clip(r, 1e-12, None))
    return -(logr - np.asarray(mu, dtype=np.float64)[None, :, :]) / np.asarray(
        sigma, dtype=np.float64
    )[None, :, None]


def lognorm_statistics(r_layers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate (mu[L, C], sigma[L]) from clean fitting data only."""
    logr = np.log(np.clip(np.asarray(r_layers, dtype=np.float64), 1e-12, None))
    mu = logr.mean(axis=0)                       # [L, C] -- per layer AND class
    centered = logr - mu[None, :, :]
    sigma = centered.std(axis=(0, 2))             # [L] -- per layer, pooled over
    return mu, np.maximum(sigma, 1e-8)            #        samples and classes


def centered_logit_representation(logits: np.ndarray) -> np.ndarray:
    """z_c = z - mean(z), then L2 normalize.

    The centering is what makes the output-space control gauge-invariant in
    the same way the hidden-space arm is: a common shift of all logits
    leaves the representation, and therefore every distance, unchanged.
    """
    z = np.asarray(logits, dtype=np.float32)
    zc = z - z.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(zc, axis=1, keepdims=True)
    return zc / np.maximum(norm, 1e-12)


# ==========================================================================
# Objectives and fitting (clean data only -- never reachable from an
# evaluation-only corruption run; see run_fv_dac_experiment.py)
# ==========================================================================


def nll_of(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(labels)), labels])))


def brier_of(probs: np.ndarray, labels: np.ndarray) -> float:
    """Native-DAC-style multiclass Brier: mean over samples of the summed
    squared error across classes."""
    probs = np.asarray(probs, dtype=np.float64)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((onehot - probs) ** 2, axis=1)))


def _objective_fn(name: str):
    if name == "nll":
        return nll_of
    if name == "brier":
        return brier_of
    raise ValueError(f"unknown fitting objective {name!r} (expected 'nll' or 'brier')")


def fit_beta(
    logits: np.ndarray,
    correction: np.ndarray,
    temperature: np.ndarray,
    labels: np.ndarray,
    *,
    objective: str = "nll",
    sign: float = -1.0,
    beta_max: float = BETA_SEARCH_MAX,
) -> Dict[str, Any]:
    """beta_hat = argmin_{beta >= 0} objective( q_beta, y ).

    One-dimensional and bounded below at 0, so a coarse deterministic sweep
    followed by a bounded local refinement is both reproducible and immune
    to the non-convexity a pure local solver could hit. The objective is
    NEVER accuracy and NEVER net flips.
    """
    loss = _objective_fn(objective)

    def f(beta: float) -> float:
        b = max(float(beta), 0.0)
        return loss(fv_dac_probs(logits, correction, temperature, b, sign=sign), labels)

    grid = np.concatenate([[0.0], np.geomspace(1e-3, beta_max, 61)])
    values = np.array([f(b) for b in grid])
    i = int(np.argmin(values))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]

    best_beta, best_val = float(grid[i]), float(values[i])
    if hi > lo:
        res = optimize.minimize_scalar(f, bounds=(lo, hi), method="bounded",
                                       options={"xatol": 1e-6})
        if res.fun < best_val:
            best_beta, best_val = float(max(res.x, 0.0)), float(res.fun)

    return {
        "beta": best_beta,
        "objective": objective,
        "objective_value": best_val,
        "objective_value_at_zero": float(values[0]),
        "at_lower_boundary": bool(best_beta <= 0.0),
        "at_upper_boundary": bool(best_beta >= beta_max * (1 - 1e-9)),
        "beta_max": float(beta_max),
        "n_fit": int(len(labels)),
    }


def fit_shared_layer_weights(
    logits: np.ndarray,
    r_layers: np.ndarray,
    temperature: np.ndarray,
    labels: np.ndarray,
    *,
    objective: str = "nll",
    b_max: float = BETA_SEARCH_MAX,
) -> Dict[str, Any]:
    """b_hat = argmin_{b >= 0} objective, b indexed by LAYER only (O(L) params).

    Analytic gradient for the NLL case; numerical otherwise. L-BFGS-B with a
    non-negativity box, mirroring native DAC's own constrained fit.
    """
    loss = _objective_fn(objective)
    logits = np.asarray(logits, dtype=np.float64)
    r_layers = np.asarray(r_layers, dtype=np.float64)
    temp = np.asarray(temperature, dtype=np.float64)
    n, num_layers = logits.shape[0], r_layers.shape[1]
    onehot = np.zeros_like(logits)
    onehot[np.arange(n), labels] = 1.0

    def value_and_grad(b: np.ndarray):
        probs = shared_layer_probs(logits, r_layers, temp, b)
        if objective != "nll":
            return loss(probs, labels), None
        val = nll_of(probs, labels)
        # d NLL / d u = (p - onehot) / (S * N);  d u / d b_l = -r_l
        du = (probs - onehot) / (temp[:, None] * n)
        grad = -np.einsum("nc,nlc->l", du, r_layers)
        return val, grad

    if objective == "nll":
        res = optimize.minimize(
            fun=value_and_grad,
            x0=np.zeros(num_layers),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, b_max)] * num_layers,
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10},
        )
    else:
        res = optimize.minimize(
            fun=lambda b: loss(shared_layer_probs(logits, r_layers, temp, b), labels),
            x0=np.zeros(num_layers),
            method="L-BFGS-B",
            bounds=[(0.0, b_max)] * num_layers,
            options={"maxiter": 500, "ftol": 1e-12},
        )

    b_hat = np.maximum(np.asarray(res.x, dtype=np.float64), 0.0)
    return {
        "b": b_hat.tolist(),
        "objective": objective,
        "objective_value": float(res.fun),
        "objective_value_at_zero": float(
            loss(shared_layer_probs(logits, r_layers, temp, np.zeros(num_layers)), labels)
        ),
        "all_zero": bool(np.allclose(b_hat, 0.0)),
        "n_params": int(num_layers),
        "n_fit": int(len(labels)),
        "converged": bool(res.success),
    }


# ==========================================================================
# Negative control
# ==========================================================================


def permuted_bank_labels(bank_labels: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic sample-level permutation of the reference-bank labels.

    Everything else is held fixed -- same bank features, same distance code,
    same dimensionality, same K_c, same beta-fitting protocol. Only the
    semantic correspondence between geometry and class is destroyed.

    A permutation of the label VECTOR (rather than a random relabelling)
    preserves the per-class bank counts exactly, so every predeclared K_c
    stays equally valid under the control.
    """
    labels = np.asarray(bank_labels).astype(np.int64)
    rng = np.random.default_rng(int(seed))
    return labels[rng.permutation(len(labels))]


def array_hash(arr: np.ndarray) -> str:
    """Stable content hash of an array's bytes (shape + dtype + data)."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def bank_set_hash(bank: np.ndarray, decimals: int = 5) -> str:
    """Order-INVARIANT hash of a reference bank's row set.

    The benchmark's train loader uses SubsetRandomSampler, so two iterations
    of the same split yield the same rows in a different order. This hashes
    the sorted, rounded rows so set-identity between the frozen DAC bank and
    a freshly extracted labelled bank can be asserted despite that.
    """
    a = np.round(np.asarray(bank, dtype=np.float64), decimals).astype(np.float32)
    keys = np.ascontiguousarray(a).view([("", a.dtype)] * a.shape[1]).ravel()
    order = np.argsort(keys, kind="stable")
    return array_hash(a[order])


# ==========================================================================
# Frozen state
# ==========================================================================


@dataclass
class FVDACFrozenState:
    """Everything needed to evaluate FV-DAC without refitting anything.

    Hashing this object is what makes "corruption evaluation cannot fit"
    checkable after the fact rather than merely asserted.
    """

    # --- identity -----------------------------------------------------
    checkpoint_seed: int
    dataset: str
    model: str
    checkpoint_path: str
    variant_identity: str = "fv_dac_v1"

    # --- frozen native DAC --------------------------------------------
    dac_layers: List[str] = field(default_factory=list)
    dac_k: int = 0
    dac_layer_weights: List[float] = field(default_factory=list)
    dac_intercept: float = 0.0
    dac_state_path: str = ""
    dac_state_sha256: str = ""

    # --- FV-DAC config -------------------------------------------------
    alpha: List[float] = field(default_factory=list)
    alpha_degenerate: bool = False
    selected_kc: int = 0
    predeclared_kc: List[int] = field(default_factory=lambda: list(PREDECLARED_KC))
    kc_selection_rule: str = "clean inner-select NLL of the primary arm"
    kc_selection_table: Dict[str, Any] = field(default_factory=dict)

    # --- fitted parameters (one entry per arm) -------------------------
    fitted: Dict[str, Any] = field(default_factory=dict)
    fitting_objective_primary: str = "nll"

    # --- conditioning statistics (lognorm arm) -------------------------
    lognorm_mu: Optional[List[List[float]]] = None
    lognorm_sigma: Optional[List[float]] = None

    # --- reference bank identity ---------------------------------------
    bank_dir: str = ""
    bank_size: int = 0
    bank_layer_dims: List[int] = field(default_factory=list)
    bank_feature_hashes: List[str] = field(default_factory=list)
    bank_labels_hash: str = ""
    bank_set_hashes: List[str] = field(default_factory=list)
    permuted_label_seed: int = 0
    permuted_labels_hash: str = ""

    # --- representation definitions ------------------------------------
    hidden_representation: str = (
        "DAC layers -> adaptive_avg_pool2d(1,1) -> L2 normalize; "
        "Euclidean distance sqrt(2-2cos)"
    )
    logit_space_representation: str = (
        "z_c = z - mean(z) -> L2 normalize; same Euclidean distance, same "
        "bank membership, same labels, same K_c; S_DAC (hidden-space) unchanged"
    )
    preprocessing: str = ""

    # --- split roles ----------------------------------------------------
    split_roles: Dict[str, Any] = field(default_factory=dict)

    # --- provenance -----------------------------------------------------
    git_commit: str = ""
    created_at: str = ""
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    def state_hash(self) -> str:
        """SHA-256 over the canonical JSON, excluding volatile provenance."""
        payload = asdict(self)
        for volatile in ("created_at", "notes"):
            payload.pop(volatile, None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FVDACFrozenState":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


def required_state_fields() -> Tuple[str, ...]:
    """Fields the frozen state MUST carry (checked by a test).

    Corresponds one-to-one with the preregistered list in
    docs/full_vector_dac_experiment.md.
    """
    return (
        "dac_layer_weights",
        "dac_intercept",
        "fitted",
        "selected_kc",
        "dac_layers",
        "alpha",
        "lognorm_mu",
        "lognorm_sigma",
        "bank_feature_hashes",
        "bank_labels_hash",
        "permuted_label_seed",
        "logit_space_representation",
        "preprocessing",
        "checkpoint_path",
        "fitting_objective_primary",
        "variant_identity",
    )


# ==========================================================================
# Applying a frozen state (the ONLY path a corruption cell may take)
# ==========================================================================


def apply_arm(
    arm: str,
    state: FVDACFrozenState,
    *,
    logits: np.ndarray,
    temperature: np.ndarray,
    r_layers_real: np.ndarray,
    r_layers_permuted: Optional[np.ndarray] = None,
    r_logit_space: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluate one frozen arm. No fitting, no data-dependent choice."""
    alpha = np.asarray(state.alpha, dtype=np.float64)
    params = state.fitted.get(arm)
    if params is None:
        raise RuntimeError(
            f"arm {arm!r} has no frozen parameters in this FV-DAC state -- refusing "
            "to fit at evaluation time."
        )

    if arm in (ARM_PRIMARY, ARM_BRIER):
        return fv_dac_probs(
            logits, aggregate_R(r_layers_real, alpha), temperature,
            float(params["beta"]), sign=-1.0,
        )
    if arm == ARM_LOGNORM:
        if state.lognorm_mu is None or state.lognorm_sigma is None:
            raise RuntimeError("lognorm arm requires frozen mu/sigma statistics")
        v = lognorm_transform(
            r_layers_real,
            np.asarray(state.lognorm_mu, dtype=np.float64),
            np.asarray(state.lognorm_sigma, dtype=np.float64),
        )
        return fv_dac_probs(
            logits, aggregate_R(v, alpha), temperature,
            float(params["beta"]), sign=+1.0,
        )
    if arm == ARM_SHARED_LAYER:
        return shared_layer_probs(logits, r_layers_real, temperature, params["b"])
    if arm == ARM_PERMUTED:
        if r_layers_permuted is None:
            raise RuntimeError("permuted control requires permuted-label distances")
        return fv_dac_probs(
            logits, aggregate_R(r_layers_permuted, alpha), temperature,
            float(params["beta"]), sign=-1.0,
        )
    if arm == ARM_LOGIT_SPACE:
        if r_logit_space is None:
            raise RuntimeError("logit-space control requires output-space distances")
        # Single representation source: alpha is exactly [1.0] by construction.
        return fv_dac_probs(
            logits, aggregate_R(r_logit_space, np.ones(1)), temperature,
            float(params["beta"]), sign=-1.0,
        )
    raise ValueError(f"unknown FV-DAC arm {arm!r}")


def density_only_prediction(r_layers: np.ndarray, alpha: Sequence[float]) -> np.ndarray:
    """Diagnostic: y_hat = argmin_k R_k(x). Never fitted, never a competitor."""
    return np.argmin(aggregate_R(r_layers, alpha), axis=1)
