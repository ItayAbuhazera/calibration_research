"""
Readout, selection and diagnostic math for the layer-selection pilot
(docs/layer_selection_pilot_spec.md). Pure functions over arrays -- no model,
no data loading -- so every property in spec §12 is unit-testable without a GPU.

Nothing in this module reads corruption data. Fitting entry points
(`fit_family_a`, `fit_probe`, `fit_probe_temperature`, the greedy selectors)
take only clean-role arrays; the evaluation helpers (`family_a_probs`,
`family_b_probs`, `flip_decomposition`, ...) take a frozen state and cannot fit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import optimize

from Calibrators.full_vector_dac import (
    TEMPERATURE_FLOOR,
    fit_beta,
    fv_dac_probs,
    np_softmax,
)

# ---------------------------------------------------------------------------
# Candidate layers (spec §4.1). Declared before corrected corruption results.
# ---------------------------------------------------------------------------

#: (module path, pooled dimension). Post-ReLU outputs of Bottleneck blocks.
CANDIDATE_LAYERS: Tuple[Tuple[str, int], ...] = (
    ("layer1.0", 256),
    ("layer1.2", 256),
    ("layer2.1", 512),
    ("layer2.3", 512),
    ("layer3.2", 1024),
    ("layer3.7", 1024),
    ("layer3.12", 1024),
    ("layer3.17", 1024),
    ("layer3.22", 1024),
    ("layer4.0", 2048),
    ("layer4.1", 2048),
    ("layer4.2", 2048),
)
CANDIDATE_NAMES: Tuple[str, ...] = tuple(n for n, _ in CANDIDATE_LAYERS)
N_CANDIDATES = len(CANDIDATE_LAYERS)

#: Native DAC's five layers (`get_dac_target_layers` for a CIFAR ResNet) and the
#: candidate index each non-stem one aliases. `conv1` (pre-BN conv output) is not
#: a residual-block output and is used only to compute S_DAC.
NATIVE_DAC_LAYERS: Tuple[str, ...] = ("conv1", "layer1", "layer2", "layer3", "layer4")
NATIVE_ALIAS_TO_CANDIDATE: Dict[str, int] = {
    "layer1": CANDIDATE_NAMES.index("layer1.2"),
    "layer2": CANDIDATE_NAMES.index("layer2.3"),
    "layer3": CANDIDATE_NAMES.index("layer3.22"),
    "layer4": CANDIDATE_NAMES.index("layer4.2"),
}

#: Inherited from the closed FV-DAC pilot (clean inner-SELECT, legacy protocol).
K_C = 5
#: native DAC's k for CIFAR-100 (`get_dac_k_value`).
K_DAC = 200
LAYER_COUNTS: Tuple[int, ...] = (1, 4, 6, 8)
PROBE_LAMBDAS: Tuple[float, ...] = (1e-4, 1e-3, 1e-2)
PROBE_MAX_ITER = 200
NLL_TIE_TOLERANCE = 1e-9
PERMUTED_LABEL_SEED = 20260921


def depth_spaced_indices(n_layers: int, n_candidates: int = N_CANDIDATES) -> List[int]:
    """Deterministic depth-spaced control set (spec §7): round(linspace(0, n-1, L))
    with numpy's round-half-to-even; L=1 is the final block."""
    if n_layers < 1 or n_layers > n_candidates:
        raise ValueError(f"n_layers must be in [1, {n_candidates}]")
    if n_layers == 1:
        return [n_candidates - 1]
    idx = np.round(np.linspace(0, n_candidates - 1, n_layers)).astype(int).tolist()
    if len(set(idx)) != len(idx):
        raise ValueError(f"depth-spaced indices collide for L={n_layers}: {idx}")
    return idx


def permuted_labels(labels: np.ndarray, seed: int = PERMUTED_LABEL_SEED) -> np.ndarray:
    """Seeded permutation of bank labels: class counts preserved exactly."""
    rng = np.random.default_rng(seed)
    return np.asarray(labels)[rng.permutation(len(labels))]


# ---------------------------------------------------------------------------
# Family A
# ---------------------------------------------------------------------------


def mean_correction(r: np.ndarray, layer_set: Sequence[int]) -> np.ndarray:
    """R_A,k(x) = (1/L) sum_{l in A} r_{l,k}(x), equal weights, no rescaling.

    r is [N, n_candidates, C] (or [N, C] for a single source)."""
    r = np.asarray(r, dtype=np.float64)
    if r.ndim == 2:
        return r
    idx = list(layer_set)
    if len(idx) == 0:
        raise ValueError("empty layer set")
    return r[:, idx, :].mean(axis=1)


def s_dac(global_scores: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    """Native DAC Eq. 7: S(x) = max(sum_l w_l s_l(x) + w_0, 1e-12)."""
    t = np.asarray(global_scores, dtype=np.float64) @ np.asarray(weights, dtype=np.float64) + float(intercept)
    return np.maximum(t, TEMPERATURE_FLOOR)


def family_a_probs(
    logits: np.ndarray,
    r: np.ndarray,
    layer_set: Sequence[int],
    beta: float,
    temperature: np.ndarray,
) -> np.ndarray:
    """q = softmax((z - beta R_A) / S_DAC). beta = 0 is native DAC exactly."""
    return fv_dac_probs(logits, mean_correction(r, layer_set), temperature, beta)


def fit_family_a(
    logits_fit: np.ndarray,
    r_fit: np.ndarray,
    layer_set: Sequence[int],
    temperature_fit: np.ndarray,
    labels_fit: np.ndarray,
) -> Dict[str, Any]:
    """beta_hat = argmin_{beta>=0} NLL on inner-FIT (never accuracy/flips)."""
    return fit_beta(
        logits_fit,
        mean_correction(r_fit, layer_set),
        temperature_fit,
        labels_fit,
        objective="nll",
    )


def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(labels)), labels])))


# ---------------------------------------------------------------------------
# Greedy forward selection (deterministic, nested, full trace)
# ---------------------------------------------------------------------------


def greedy_forward(
    score_fn: Callable[[Tuple[int, ...]], float],
    n_candidates: int,
    n_steps: int,
    tie_tolerance: float = NLL_TIE_TOLERANCE,
) -> Dict[str, Any]:
    """Forward selection of `n_steps` distinct candidates by lowest score.

    `score_fn(set)` is the held-out criterion (inner-SELECT NLL) of the FULLY
    FITTED arm for that set. At every step the best remaining candidate is
    added even if its score is worse than the previous step's, so the trace
    records whether more layers help. Ties within `tie_tolerance` go to the
    smaller candidate index. Every candidate's score at every step is kept,
    including the rejected ones.
    """
    if n_steps > n_candidates:
        raise ValueError("n_steps exceeds the number of candidates")
    chosen: List[int] = []
    trace: List[Dict[str, Any]] = []
    for step in range(1, n_steps + 1):
        scores: Dict[int, float] = {}
        for c in range(n_candidates):
            if c in chosen:
                continue
            scores[c] = float(score_fn(tuple(chosen + [c])))
        best_c, best_s = None, None
        for c in sorted(scores):  # ascending index => smaller index wins ties
            if best_s is None or scores[c] < best_s - tie_tolerance:
                best_c, best_s = c, scores[c]
        chosen.append(int(best_c))
        trace.append(
            {
                "step": step,
                "added": int(best_c),
                "set": list(chosen),
                "score": float(best_s),
                "candidate_scores": {str(k): v for k, v in scores.items()},
                "score_change_vs_previous_step": (
                    None if step == 1 else float(best_s - trace[-1]["score"])
                ),
            }
        )
    return {"order": chosen, "trace": trace}


def nested_sets(order: Sequence[int], counts: Sequence[int] = LAYER_COUNTS) -> Dict[int, List[int]]:
    """A_1 ⊂ A_4 ⊂ A_6 ⊂ A_8: prefixes of the greedy order."""
    return {int(L): [int(i) for i in order[:L]] for L in counts}


# ---------------------------------------------------------------------------
# Family B: linear probes
# ---------------------------------------------------------------------------


@dataclass
class Probe:
    """Multinomial logistic regression on z-scored inputs."""

    weight: torch.Tensor  # [d, C]
    bias: torch.Tensor  # [C]
    mean: torch.Tensor  # [d]
    std: torch.Tensor  # [d]
    lam: float
    temperature: float = 1.0

    @property
    def n_params(self) -> int:
        # weights + biases + the scalar probe temperature
        return int(self.weight.numel() + self.bias.numel() + 1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / self.std) @ self.weight + self.bias


def fit_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    lam: float,
    *,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    max_iter: int = PROBE_MAX_ITER,
) -> Probe:
    """mean-CE + (lam/2)||W||_F^2, full-batch L-BFGS, zero init (deterministic)."""
    x = x_train.float()
    if mean is None:
        mean = x.mean(0)
    if std is None:
        std = x.std(0) + 1e-6
    xs = (x - mean) / std
    d = xs.shape[1]
    w = torch.zeros(d, num_classes, device=x.device, requires_grad=True)
    b = torch.zeros(num_classes, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS(
        [w, b], lr=1.0, max_iter=max_iter, history_size=20,
        line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10,
    )

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(xs @ w + b, y_train) + 0.5 * lam * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return Probe(w.detach(), b.detach(), mean, std, float(lam))


def fit_probe_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Scalar T>0 minimizing NLL of softmax(logits/T), 1-D bounded search on log T."""
    logits = np.asarray(logits, dtype=np.float64)

    def f(logt: float) -> float:
        return nll(np_softmax(logits / np.exp(logt)), labels)

    res = optimize.minimize_scalar(f, bounds=(-3.0, 3.0), method="bounded", options={"xatol": 1e-6})
    return float(np.exp(res.x))


def probe_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    return np_softmax(np.asarray(logits, dtype=np.float64) / float(temperature))


def family_b_probs(
    per_layer_logits: np.ndarray,
    temperatures: Sequence[float],
    layer_set: Sequence[int],
) -> np.ndarray:
    """Equal-weight average of per-layer calibrated probability vectors.

    per_layer_logits: [N, n_probes, C] raw probe logits; temperatures: [n_probes].
    """
    idx = list(layer_set)
    if not idx:
        raise ValueError("empty layer set")
    acc = None
    for l in idx:
        p = probe_probs(per_layer_logits[:, l, :], temperatures[l])
        acc = p if acc is None else acc + p
    return acc / len(idx)


# ---------------------------------------------------------------------------
# Decision diagnostics
# ---------------------------------------------------------------------------


def gt_rank(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """1-based rank of the true class (1 = top-1); ties broken pessimistically."""
    p_true = probs[np.arange(len(labels)), labels][:, None]
    return (probs > p_true).sum(axis=1) + 1


def flip_decomposition(base_probs: np.ndarray, probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """W (wrong->correct), H (correct->wrong), U (wrong->different-wrong).

    DeltaAcc = (W - H)/N. U is neutral for top-1 accuracy. Precisions are None
    when undefined (no flips) rather than 0.
    """
    bp, mp = base_probs.argmax(1), probs.argmax(1)
    n = len(labels)
    changed = bp != mp
    b_ok, m_ok = bp == labels, mp == labels
    W = int(np.sum(changed & ~b_ok & m_ok))
    H = int(np.sum(changed & b_ok & ~m_ok))
    U = int(np.sum(changed & ~b_ok & ~m_ok))
    F_ = W + H + U
    n_err = int((~b_ok).sum())
    r_before, r_after = gt_rank(base_probs, labels), gt_rank(probs, labels)
    out = {
        "n": n, "W": W, "H": H, "U": U, "flips": F_, "net": W - H,
        "delta_acc_from_flips": (W - H) / n,
        "decisive_precision": (W / (W + H)) if (W + H) else None,
        "intervention_precision": (W / F_) if F_ else None,
        "base_errors": n_err,
        "frac_base_errors_repaired": (W / n_err) if n_err else None,
        "argmax_change_rate": F_ / n,
        "gt_rank_mean_base": float(r_before.mean()),
        "gt_rank_mean_method": float(r_after.mean()),
        "gt_rank_improved_frac": float(np.mean(r_after < r_before)),
        "gt_rank_worsened_frac": float(np.mean(r_after > r_before)),
    }
    assert abs(out["delta_acc_from_flips"] - (m_ok.mean() - b_ok.mean())) < 1e-12
    return out


def margin_matched_enrichment(
    base_probs: np.ndarray, probs: np.ndarray, labels: np.ndarray, n_bins: int = 5
) -> Dict[str, Any]:
    """Compare the share of flips landing on base-WRONG examples with the share
    expected if flips fell on examples at the same base top-2 margin uniformly
    at random within margin bins.

    A high share of flips on base errors is NOT evidence of geometric error
    detection by itself: base errors are concentrated at low margin, and the
    flips are too. enrichment ~ 1 means the flips are explained by margin.
    """
    bp = base_probs.argmax(1)
    mp = probs.argmax(1)
    changed = bp != mp
    if changed.sum() == 0:
        return {"flips": 0, "observed_share_on_base_errors": None, "margin_matched_expected_share": None,
                "enrichment_ratio": None, "base_error_rate": float(np.mean(bp != labels))}
    top2 = np.sort(base_probs, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    edges = np.quantile(margin, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-12
    bin_id = np.clip(np.searchsorted(edges, margin, side="right") - 1, 0, n_bins - 1)
    base_wrong = bp != labels
    expected = 0.0
    for b in range(n_bins):
        in_bin = bin_id == b
        n_flip_b = int((changed & in_bin).sum())
        if n_flip_b and in_bin.any():
            expected += n_flip_b * float(base_wrong[in_bin].mean())
    observed = float((changed & base_wrong).sum())
    total = float(changed.sum())
    exp_share = expected / total
    return {
        "flips": int(total),
        "observed_share_on_base_errors": observed / total,
        "margin_matched_expected_share": exp_share,
        "enrichment_ratio": (observed / total) / exp_share if exp_share > 0 else None,
        "base_error_rate": float(base_wrong.mean()),
    }


def state_hash(payload: Dict[str, Any], arrays: Sequence[np.ndarray] = ()) -> str:
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, default=float).encode("utf-8"))
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
