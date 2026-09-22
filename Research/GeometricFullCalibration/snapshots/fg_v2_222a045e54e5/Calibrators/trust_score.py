"""
Trust Score calibrator (Jiang et al. 2018) — Phase 1 (alpha=0).

Computes per-sample trust = d_other_class / (d_predicted_class + eps) from
per-class 1-NN distances to the training set.

Two variants:
  - TrustScoreCalibrator.calibrate_diagnostic: returns base_probs unchanged,
    stores trust scores for analysis. can_change_argmax=False.
  - TrustScoreCalibrator.calibrate_switch: thresholds trust < 1.0 and swaps
    predicted/alternative-class probabilities where trust is lowest. can_change_argmax=True.

Phase 2 (alpha > 0 density filtering) is not implemented here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.stability_space import StabilitySpace


class TrustScoreCalibrator:
    """
    Trust Score calibrator.

    fit(train_features, train_labels) — build per-class 1-NN index.
    compute_trust_scores(test_features, base_pred) — returns (trust, alt_pred).
    calibrate_diagnostic(test_features, base_probs) — no-op on probs, attaches metadata.
    calibrate_switch(test_features, base_probs, val_features, val_base_probs, val_labels)
        — selects threshold on val, switches test samples below threshold.
    """

    def __init__(
        self,
        k_filter: int = 10,
        alpha: float = 0.0,
        metric: str = "l2",
    ) -> None:
        if alpha != 0.0:
            raise NotImplementedError("alpha > 0 density filtering is not yet implemented (Phase 2).")
        self.k_filter = k_filter
        self.alpha = alpha
        self.metric = metric
        self._stab_space: Optional[StabilitySpace] = None
        self._num_classes: Optional[int] = None

    def fit(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
        self._num_classes = int(np.max(train_labels)) + 1
        self._stab_space = StabilitySpace(
            train_features,
            train_labels,
            library="fast_separation",
            metric=self.metric,
        )

    def _dist_matrix(self, features: np.ndarray) -> np.ndarray:
        if self._stab_space is None:
            raise RuntimeError("Call fit() before computing trust scores.")
        return self._stab_space.calc_per_class_1nn_distances(features)

    def compute_trust_scores(
        self,
        test_features: np.ndarray,
        base_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (trust_scores [N], alt_pred [N]).

        trust = d_other / (d_pred + eps)
        alt_pred = argmin distance among non-predicted classes.
        """
        dist_matrix = self._dist_matrix(test_features)  # [N, C]
        rows = np.arange(len(base_pred))
        d_pred = dist_matrix[rows, base_pred]

        alt_dm = dist_matrix.copy()
        alt_dm[rows, base_pred] = np.inf
        alt_pred = np.argmin(alt_dm, axis=1)
        d_other = alt_dm[rows, alt_pred]

        trust = d_other / (d_pred + 1e-8)
        return trust, alt_pred

    def calibrate_diagnostic(
        self,
        test_features: np.ndarray,
        base_probs: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        No-op calibration: returns base_probs unchanged.
        Also returns a metadata dict with trust_scores for analysis.
        """
        base_pred = np.argmax(base_probs, axis=1)
        trust, alt_pred = self.compute_trust_scores(test_features, base_pred)
        meta: Dict[str, Any] = {
            "mean_trust_score": float(np.mean(trust)),
            "median_trust_score": float(np.median(trust)),
            "fraction_below_1": float(np.mean(trust < 1.0)),
            "trust_score_mean": float(np.mean(trust)),
        }
        return base_probs.copy(), meta

    @staticmethod
    def _threshold_sweep(
        score: np.ndarray,
        upper_boundary: float,
        base_pred: np.ndarray,
        alt_pred: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """Sweep score values below upper_boundary; select threshold maximising accuracy."""
        eligible_idx = np.flatnonzero(score < upper_boundary)
        order = eligible_idx[np.argsort(score[eligible_idx], kind="stable")]
        ordered_scores = score[order]
        ordered_gain = (
            (alt_pred[order] == labels[order]).astype(np.int64)
            - (base_pred[order] == labels[order]).astype(np.int64)
        )
        cumulative_gain = np.r_[0, np.cumsum(ordered_gain)]
        best_gain = int(np.max(cumulative_gain))
        switch_count = int(np.flatnonzero(cumulative_gain == best_gain)[0])

        if switch_count == 0 or len(ordered_scores) == 0:
            threshold = float(ordered_scores[0]) if len(ordered_scores) else upper_boundary
        elif switch_count >= len(ordered_scores):
            threshold = float(upper_boundary)
        else:
            threshold = float((ordered_scores[switch_count - 1] + ordered_scores[switch_count]) / 2.0)

        switched = score < threshold
        fixes = int(np.sum(switched & (base_pred != labels) & (alt_pred == labels)))
        breaks = int(np.sum(switched & (base_pred == labels) & (alt_pred != labels)))
        return {
            "threshold": threshold,
            "upper_boundary": float(upper_boundary),
            "val_switch_count": int(np.sum(switched)),
            "val_flip_correct": fixes,
            "val_flip_wrong": breaks,
            "val_net_gain": best_gain,
            "val_base_accuracy": float(np.mean(base_pred == labels)),
            "val_thresholded_accuracy": float(
                np.mean(np.where(switched, alt_pred, base_pred) == labels)
            ),
        }

    @staticmethod
    def _swap_probs(
        base_probs: np.ndarray,
        switched: np.ndarray,
        base_pred: np.ndarray,
        alt_pred: np.ndarray,
    ) -> np.ndarray:
        result = base_probs.copy()
        rows = np.flatnonzero(switched)
        for r in rows:
            bp, ap = base_pred[r], alt_pred[r]
            result[r, bp], result[r, ap] = base_probs[r, ap], base_probs[r, bp]
        return result

    def calibrate_switch(
        self,
        test_features: np.ndarray,
        base_probs: np.ndarray,
        val_features: np.ndarray,
        val_base_probs: np.ndarray,
        val_labels: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Threshold trust < 1.0 sweep on val; apply best threshold to test.
        Returns (calibrated_probs [N_test, C], metadata dict).
        """
        val_base_pred = np.argmax(val_base_probs, axis=1)
        test_base_pred = np.argmax(base_probs, axis=1)

        val_trust, val_alt_pred = self.compute_trust_scores(val_features, val_base_pred)
        test_trust, test_alt_pred = self.compute_trust_scores(test_features, test_base_pred)

        val_result = self._threshold_sweep(val_trust, 1.0, val_base_pred, val_alt_pred, val_labels)
        threshold = val_result["threshold"]
        switched = test_trust < threshold
        calibrated = self._swap_probs(base_probs, switched, test_base_pred, test_alt_pred)

        meta: Dict[str, Any] = {
            "selected_threshold": threshold,
            "test_switch_count": int(np.sum(switched)),
            "val_net_gain": val_result["val_net_gain"],
            "val_base_accuracy": val_result["val_base_accuracy"],
            "val_thresholded_accuracy": val_result["val_thresholded_accuracy"],
        }
        meta.update({f"val_{k}": v for k, v in val_result.items()})
        return calibrated, meta

    def get_params(self) -> Dict[str, Any]:
        return {
            "k_filter": self.k_filter,
            "alpha": self.alpha,
            "metric": self.metric,
            "fitted": self._stab_space is not None,
            "num_classes": self._num_classes,
        }
