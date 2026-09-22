"""
Decision-audit helper for calibration benchmarks.

Provides a unified `decision_audit` function that computes all decision-level
metrics needed to evaluate whether a calibration method improves model decisions,
and `make_inner_validation_split` for held-out hyperparameter selection.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def decision_audit(
    base_probs: np.ndarray,
    method_probs: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, object]:
    """
    Compute decision-level audit metrics comparing base and calibrated predictions.

    Returns a dict with all fields needed for the paper's decision-improvement table.
    Includes compatibility aliases so existing consumers of _argmax_flip_metadata
    output continue to work without changes.

    Parameters
    ----------
    base_probs : [N, C] float array of base model probabilities.
    method_probs : [N, C] float array of calibrated probabilities.
    labels : [N] int array of ground-truth labels.
    eps : small constant for numerical stability (unused directly but kept for API parity).

    Returns
    -------
    dict with keys:
        base_accuracy, method_accuracy, accuracy_delta,
        argmax_change_rate,
        changed_to_correct_count, changed_to_wrong_count,
        wrong_to_wrong_count, correct_to_correct_count,
        net_flips, net_flip_rate,
        changed_to_correct_rate, changed_to_wrong_rate,
        flip_to_correct_count, flip_to_wrong_count,       # aliases
        flip_to_correct_rate, flip_to_wrong_rate,          # aliases
    """
    if base_probs.ndim != 2:
        raise ValueError(f"base_probs must be 2D [N, C], got shape {base_probs.shape}")
    if method_probs.ndim != 2:
        raise ValueError(f"method_probs must be 2D [N, C], got shape {method_probs.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D [N], got shape {labels.shape}")
    if base_probs.shape != method_probs.shape:
        raise ValueError(
            f"base_probs and method_probs must have identical shape; "
            f"got {base_probs.shape} vs {method_probs.shape}"
        )
    if base_probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Probability matrix rows ({base_probs.shape[0]}) must match label count ({labels.shape[0]})"
        )
    if base_probs.shape[0] == 0:
        raise ValueError("Empty probability matrices")

    base_pred = np.argmax(base_probs, axis=1)
    method_pred = np.argmax(method_probs, axis=1)
    n = len(labels)

    base_correct = base_pred == labels
    method_correct = method_pred == labels

    changed = method_pred != base_pred

    # Decision categories
    changed_to_correct = int(np.sum(changed & ~base_correct & method_correct))
    changed_to_wrong = int(np.sum(changed & base_correct & ~method_correct))
    wrong_to_wrong = int(np.sum(changed & ~base_correct & ~method_correct))
    correct_to_correct = int(np.sum(~changed & base_correct))

    net_flips = changed_to_correct - changed_to_wrong

    result = {
        "base_accuracy": float(np.mean(base_correct)),
        "method_accuracy": float(np.mean(method_correct)),
        "accuracy_delta": float(np.mean(method_correct) - np.mean(base_correct)),
        "argmax_change_rate": float(np.mean(changed)),
        "changed_to_correct_count": changed_to_correct,
        "changed_to_wrong_count": changed_to_wrong,
        "wrong_to_wrong_count": wrong_to_wrong,
        "correct_to_correct_count": correct_to_correct,
        "net_flips": net_flips,
        "net_flip_rate": float(net_flips / n),
        "changed_to_correct_rate": float(changed_to_correct / n),
        "changed_to_wrong_rate": float(changed_to_wrong / n),
        # Compatibility aliases for existing consumers of _argmax_flip_metadata output.
        "flip_to_correct_count": changed_to_correct,
        "flip_to_wrong_count": changed_to_wrong,
        "flip_to_correct_rate": float(changed_to_correct / n),
        "flip_to_wrong_rate": float(changed_to_wrong / n),
    }
    return result


def make_inner_validation_split(
    labels: np.ndarray,
    select_fraction: float = 0.5,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split validation indices into a fit set and a select set.

    The select set is used for held-out hyperparameter selection (e.g. GLAD-PI
    beta grid). The fit set is used to train learned calibrators. Splitting is
    stratified by class to preserve class proportions in both subsets.

    Default select_fraction=0.5 is intentionally generous: the net-flip selection
    metric is discrete and noisy, and 30% selection samples are insufficient for
    CIFAR-100 (100 classes × 5000 val → only 1500 select samples at 30%).

    Parameters
    ----------
    labels : [N] int array of class labels for the validation split.
    select_fraction : fraction of validation data to reserve for selection.
    seed : random seed for reproducibility.

    Returns
    -------
    fit_indices : array of indices for the fit subset.
    select_indices : array of indices for the select subset.
    """
    if not (0.0 < select_fraction < 1.0):
        raise ValueError(f"select_fraction must be in (0, 1), got {select_fraction}")

    n = len(labels)
    indices = np.arange(n)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=select_fraction, random_state=seed)
    fit_idx, select_idx = next(sss.split(indices, labels))
    return fit_idx, select_idx
