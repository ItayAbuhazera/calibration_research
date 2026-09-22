"""
Shared metric evaluator for unified post-hoc calibration benchmarks.

This module intentionally centralizes all comparable metrics so methods
from different scripts are evaluated with one consistent implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from utils.selective_metrics import selective_metrics


def validate_probability_matrix(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    sum_tolerance: float = 1e-6,
) -> None:
    """
    Validate probability outputs without silently repairing invalid results.

    Rules:
    - shape must be [N, C], labels [N]
    - all values finite and in [0, 1] up to tiny floating-point tolerance
    - each row must sum to 1 within `sum_tolerance`
    - labels must be in [0, C-1]
    """
    if probs.ndim != 2:
        raise ValueError(f"Expected probs with shape [N, C], got {probs.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"Expected labels with shape [N], got {y_true.shape}")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"Probability rows ({probs.shape[0]}) and labels ({y_true.shape[0]}) differ"
        )
    if probs.shape[0] == 0:
        raise ValueError("Empty probability matrix")
    if not np.isfinite(probs).all():
        raise ValueError("Probability matrix contains NaN or Inf values")
    if not np.isfinite(y_true).all():
        raise ValueError("Labels contain NaN or Inf values")

    min_prob = float(np.min(probs))
    max_prob = float(np.max(probs))
    if min_prob < -sum_tolerance or max_prob > 1.0 + sum_tolerance:
        raise ValueError(
            f"Probability bounds violated: min={min_prob:.6g}, max={max_prob:.6g}"
        )

    row_sums = probs.sum(axis=1)
    max_abs_sum_diff = float(np.max(np.abs(row_sums - 1.0)))
    if max_abs_sum_diff > sum_tolerance:
        raise ValueError(
            f"Probabilities do not sum to 1 within tolerance {sum_tolerance}: "
            f"max_abs_row_sum_diff={max_abs_sum_diff:.6g}"
        )

    num_classes = probs.shape[1]
    if np.min(y_true) < 0 or np.max(y_true) >= num_classes:
        raise ValueError(
            f"Label range [{int(np.min(y_true))}, {int(np.max(y_true))}] exceeds "
            f"class range [0, {num_classes - 1}]"
        )


def multiclass_nll(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    idx = np.arange(y_true.shape[0])
    p_true = np.clip(probs[idx, y_true], eps, 1.0)
    return float(-np.mean(np.log(p_true)))


def multiclass_brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def accuracy(probs: np.ndarray, y_true: np.ndarray) -> float:
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == y_true))


def confidence_ece(
    confidence: np.ndarray, correct: np.ndarray, num_bins: int = 15
) -> float:
    """Equal-width binned calibration error of (confidence, correct).

    The single implementation behind top-label ECE for BOTH buckets: a
    full-vector method passes (max(probs), argmax(probs)==y); a scalar
    confidence method passes (calibrated scalar, base_pred==y).
    """
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must align")
    n = confidence.size
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == num_bins - 1:
            in_bin = (confidence >= lo) & (confidence <= hi)
        else:
            in_bin = (confidence >= lo) & (confidence < hi)
        if not np.any(in_bin):
            continue
        prop = np.mean(in_bin)
        ece += prop * abs(float(np.mean(correct[in_bin])) - float(np.mean(confidence[in_bin])))
    return float(ece)


def confidence_adaptive_ece(
    confidence: np.ndarray, correct: np.ndarray, num_bins: int = 15
) -> float:
    """Equal-MASS binned calibration error of (confidence, correct).

    TIE HANDLING. Equal-mass bins cut at fixed rank positions, so when many
    samples share a confidence value a tied block straddles a bin boundary and
    the result depends on the arbitrary order of rows within that block. On
    seed 4 the geometric scalar methods have tie blocks covering 53-66% of the
    test set, and a naive implementation moved adaptive ECE by up to 0.006
    across row permutations -- larger than the effects this benchmark is
    trying to measure.

    We therefore replace each tied block's correctness by the block mean
    before binning (the same device `utils.selective_metrics` uses for AURC).
    Every sample in a tied block then contributes the block's average
    correctness at the block's exact confidence, whichever bin it lands in, so
    the result is deterministic and invariant to input row order.
    """
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must align")
    n = confidence.size
    if n == 0:
        return 0.0
    order = np.argsort(confidence, kind="stable")
    sorted_conf = confidence[order]
    sorted_correct = correct[order]

    # Tie-average correctness within each block of equal confidence.
    new_block = np.empty(n, dtype=bool)
    new_block[0] = True
    new_block[1:] = sorted_conf[1:] != sorted_conf[:-1]
    block_id = np.cumsum(new_block) - 1
    block_mean = np.bincount(block_id, weights=sorted_correct) / np.bincount(block_id)
    sorted_correct = block_mean[block_id]

    edges = np.linspace(0, n, num_bins + 1, dtype=int)
    ece = 0.0
    for i in range(num_bins):
        start, end = edges[i], edges[i + 1]
        if end <= start:
            continue
        prop = (end - start) / n
        ece += prop * abs(
            float(np.mean(sorted_correct[start:end])) - float(np.mean(sorted_conf[start:end]))
        )
    return float(ece)


def top_label_ece(probs: np.ndarray, y_true: np.ndarray, num_bins: int = 15) -> float:
    return confidence_ece(
        np.max(probs, axis=1),
        (np.argmax(probs, axis=1) == y_true).astype(np.float64),
        num_bins=num_bins,
    )


def classwise_ece(probs: np.ndarray, y_true: np.ndarray, num_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    class_eces = []
    for c in range(probs.shape[1]):
        class_conf = probs[:, c]
        class_true = (y_true == c).astype(np.float64)
        ece_c = 0.0
        for i in range(num_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == num_bins - 1:
                in_bin = (class_conf >= lo) & (class_conf <= hi)
            else:
                in_bin = (class_conf >= lo) & (class_conf < hi)
            if not np.any(in_bin):
                continue
            prop = np.mean(in_bin)
            acc_bin = np.mean(class_true[in_bin])
            conf_bin = np.mean(class_conf[in_bin])
            ece_c += prop * abs(acc_bin - conf_bin)
        class_eces.append(ece_c)
    return float(np.mean(class_eces)) if class_eces else 0.0


def adaptive_ece(probs: np.ndarray, y_true: np.ndarray, num_bins: int = 15) -> float:
    return confidence_adaptive_ece(
        np.max(probs, axis=1),
        (np.argmax(probs, axis=1) == y_true).astype(np.float64),
        num_bins=num_bins,
    )


def evaluate_all(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    num_bins: int = 15,
    adaptive_bins: Optional[int] = None,
    eps: float = 1e-12,
    sum_tolerance: float = 1e-6,
    include_selective: bool = True,
) -> Dict[str, Any]:
    """Full-vector metric bucket: the NxC matrix IS the scientific object.

    Only for methods whose registry `output_semantics` is "full_vector".
    Scalar-confidence methods must go through `evaluate_scalar_confidence`,
    which never reads the surrogate matrix's argmax or max.
    """
    validate_probability_matrix(probs, y_true, sum_tolerance=sum_tolerance)
    adapt_bins = num_bins if adaptive_bins is None else adaptive_bins
    out: Dict[str, Any] = {
        "accuracy": accuracy(probs, y_true),
        "nll": multiclass_nll(probs, y_true, eps=eps),
        "brier": multiclass_brier_score(probs, y_true),
        "top_label_ece": top_label_ece(probs, y_true, num_bins=num_bins),
        "classwise_ece": classwise_ece(probs, y_true, num_bins=num_bins),
        "adaptive_ece": adaptive_ece(probs, y_true, num_bins=adapt_bins),
    }
    if include_selective:
        out.update(
            selective_metrics(
                np.max(probs, axis=1),
                (np.argmax(probs, axis=1) == y_true).astype(np.float64),
            )
        )
    return out


def evaluate_scalar_confidence(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    num_bins: int = 15,
    adaptive_bins: Optional[int] = None,
    include_selective: bool = True,
) -> Dict[str, Any]:
    """Scalar metric bucket (frozen BENCHMARK_IMPLEMENTATION_PLAN.md §6).

    `confidence` is the method's native calibrated confidence in the EFFECTIVE
    prediction and `correct` is whether that effective prediction is right.
    No NxC matrix is consulted, so no reconstruction can redefine either one.

    NLL, multiclass Brier and classwise ECE are a real `None`: the plan
    forbids fabricating them from a post-hoc reconstruction, and a null in
    the artifact makes downstream aggregation fail loudly instead of reading
    a missing key as zero.
    """
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct).reshape(-1).astype(np.float64)
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must align")
    if confidence.size == 0:
        raise ValueError("Empty confidence array")
    if not np.isfinite(confidence).all():
        raise ValueError("confidence contains NaN or Inf")
    if not np.all((correct == 0.0) | (correct == 1.0)):
        raise ValueError("correct must be boolean/0-1 valued")
    adapt_bins = num_bins if adaptive_bins is None else adaptive_bins
    out: Dict[str, Any] = {
        "accuracy": float(np.mean(correct)),
        "nll": None,
        "brier": None,
        "top_label_ece": confidence_ece(confidence, correct, num_bins=num_bins),
        "classwise_ece": None,
        "adaptive_ece": confidence_adaptive_ece(confidence, correct, num_bins=adapt_bins),
    }
    if include_selective:
        out.update(selective_metrics(confidence, correct))
    return out


def scalar_confidence_from_surrogate(
    surrogate_probs: np.ndarray, base_probs: np.ndarray
) -> Dict[str, np.ndarray]:
    """Extract the scalar semantics from a stored surrogate matrix.

    Returns the effective prediction (always the BASE argmax), the calibrated
    confidence the surrogate assigns to that class, and the surrogate's own
    argmax for the `surrogate_argmax_change_rate` diagnostic only.
    """
    surrogate_probs = np.asarray(surrogate_probs, dtype=np.float64)
    base_probs = np.asarray(base_probs, dtype=np.float64)
    if surrogate_probs.shape != base_probs.shape:
        raise ValueError(
            f"surrogate {surrogate_probs.shape} and base {base_probs.shape} must match"
        )
    base_pred = np.argmax(base_probs, axis=1)
    rows = np.arange(base_pred.shape[0])
    return {
        "effective_pred": base_pred,
        "confidence": surrogate_probs[rows, base_pred],
        "surrogate_pred": np.argmax(surrogate_probs, axis=1),
    }
