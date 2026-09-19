"""
Shared metric evaluator for unified post-hoc calibration benchmarks.

This module intentionally centralizes all comparable metrics so methods
from different scripts are evaluated with one consistent implementation.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


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


def top_label_ece(probs: np.ndarray, y_true: np.ndarray, num_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == num_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        if not np.any(in_bin):
            continue
        prop = np.mean(in_bin)
        acc_bin = np.mean(correct[in_bin])
        conf_bin = np.mean(confidences[in_bin])
        ece += prop * abs(acc_bin - conf_bin)
    return float(ece if n > 0 else 0.0)


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
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == y_true).astype(np.float64)
    order = np.argsort(confidences)
    sorted_conf = confidences[order]
    sorted_correct = correct[order]
    n = len(y_true)
    if n == 0:
        return 0.0
    edges = np.linspace(0, n, num_bins + 1, dtype=int)
    ece = 0.0
    for i in range(num_bins):
        start = edges[i]
        end = edges[i + 1]
        if end <= start:
            continue
        conf_bin = sorted_conf[start:end]
        corr_bin = sorted_correct[start:end]
        prop = len(conf_bin) / n
        ece += prop * abs(float(np.mean(corr_bin)) - float(np.mean(conf_bin)))
    return float(ece)


def evaluate_all(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    num_bins: int = 15,
    adaptive_bins: Optional[int] = None,
    eps: float = 1e-12,
    sum_tolerance: float = 1e-6,
) -> Dict[str, float]:
    validate_probability_matrix(probs, y_true, sum_tolerance=sum_tolerance)
    adapt_bins = num_bins if adaptive_bins is None else adaptive_bins
    return {
        "accuracy": accuracy(probs, y_true),
        "nll": multiclass_nll(probs, y_true, eps=eps),
        "brier": multiclass_brier_score(probs, y_true),
        "top_label_ece": top_label_ece(probs, y_true, num_bins=num_bins),
        "classwise_ece": classwise_ece(probs, y_true, num_bins=num_bins),
        "adaptive_ece": adaptive_ece(probs, y_true, num_bins=adapt_bins),
    }
