"""
Mahalanobis confidence calibrator, variant A (Phase 1, §8d of
BENCHMARK_IMPLEMENTATION_PLAN.md).

Per-class mean + Ledoit-Wolf-shrunk shared covariance on penultimate
features; scalar score = -min_c Mahalanobis distance to a class centroid
(higher = more confident). Mapped onto the base model's predicted-class
probability slot via isotonic regression against binary correctness,
scalar-only and argmax-preserving per the plan's frozen §6 bucket table.

Deviation from the plan's literal prose (logged here per §16's "record before
proceeding" rule, since the plan file itself is append-only): §8d describes
this mapping as reusing "the same isotonic-diagnostic convention
Trust-Score-diagnostic already uses (reuse that mapping pattern from
Calibrators/trust_score.py)". Direct inspection of
Calibrators/trust_score.py::TrustScoreCalibrator.calibrate_diagnostic, and its
call site in Experiments/run_unified_benchmark.py
(_ts_diag_name = "trust_score_original_diagnostic"), shows that convention is
a genuine no-op on probabilities: base_probs are returned completely
unchanged, and the scalar trust-score summary is attached only as diagnostic
metadata. There is no isotonic mapping there to reuse. Since §6's frozen
bucket table still requires Mahalanobis confidence to land in the
"scalar only, no NLL/Brier" bucket and be argmax-preserving -- the same
requirement GC-DAC and Top-Label Isotonic satisfy via an "isotonic +
uniform-spread" reconstruction -- this module reuses that actual convention
instead (the same mechanic as
Calibrators/isotonic_regression.py::TopLabelIsotonicCalibrator, generalized to
take an externally supplied scalar confidence score rather than the model's
own softmax top-probability).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.isotonic import IsotonicRegression


class MahalanobisConfidenceCalibrator:
    """
    fit(train_features, train_labels, fit_features, fit_base_probs, fit_labels)
    calibrate(test_features, test_base_probs) -> np.ndarray [N, C]
    get_params() -> dict
    """

    def __init__(self, shrinkage: Optional[float] = None) -> None:
        # shrinkage=None lets sklearn's LedoitWolf pick its own shrinkage
        # coefficient automatically (the standard, non-tuned Ledoit-Wolf
        # estimator); an explicit float in [0, 1] pins it instead.
        self.shrinkage = shrinkage
        self._class_means: Optional[np.ndarray] = None
        self._whitening_matrix: Optional[np.ndarray] = None
        self._whitened_means: Optional[np.ndarray] = None
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._num_classes: Optional[int] = None
        self._feature_dim: Optional[int] = None
        self._shrinkage_used: Optional[float] = None
        self._is_fitted = False

    def _mahalanobis_min_distance(self, features: np.ndarray) -> np.ndarray:
        """Min-over-classes Mahalanobis distance, computed in whitened space
        to avoid materializing an (N, C, D) intermediate."""
        if self._whitening_matrix is None or self._whitened_means is None:
            raise RuntimeError("Call fit() before scoring.")
        w = features @ self._whitening_matrix  # (N, D)
        wm = self._whitened_means  # (C, D)
        a2 = np.sum(w * w, axis=1, keepdims=True)  # (N, 1)
        b2 = np.sum(wm * wm, axis=1)  # (C,)
        cross = w @ wm.T  # (N, C)
        sq_dist = a2 - 2.0 * cross + b2[None, :]
        sq_dist = np.clip(sq_dist, 0.0, None)
        return np.sqrt(sq_dist).min(axis=1)

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        fit_features: np.ndarray,
        fit_base_probs: np.ndarray,
        fit_labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        train_features/train_labels: reference bank (training set, per plan
        §7) used for the per-class means and shared covariance.
        fit_features/fit_base_probs/fit_labels: the calibration/fitting split
        (per plan §7, reused verbatim) used to fit the isotonic mapping from
        Mahalanobis score to correctness.
        """
        train_labels = np.asarray(train_labels)
        self._num_classes = int(np.max(train_labels)) + 1
        d = train_features.shape[1]
        self._feature_dim = int(d)

        class_means = np.zeros((self._num_classes, d), dtype=np.float64)
        for c in range(self._num_classes):
            mask = train_labels == c
            if not np.any(mask):
                raise ValueError(f"No training examples for class {c}")
            class_means[c] = np.asarray(train_features[mask], dtype=np.float64).mean(axis=0)
        self._class_means = class_means

        centered = np.asarray(train_features, dtype=np.float64) - class_means[train_labels]
        if self.shrinkage is None:
            lw = LedoitWolf(assume_centered=True)
            lw.fit(centered)
            covariance = lw.covariance_
            self._shrinkage_used = float(lw.shrinkage_)
        else:
            empirical = np.cov(centered, rowvar=False)
            mu = float(np.trace(empirical) / d)
            covariance = (1.0 - self.shrinkage) * empirical + self.shrinkage * mu * np.eye(d)
            self._shrinkage_used = float(self.shrinkage)

        eigvals, eigvecs = np.linalg.eigh(covariance)
        eigvals = np.clip(eigvals, 1e-12, None)
        # Sigma^{-1/2}, symmetric: whitening features by this matrix makes
        # squared Euclidean distance in the transformed space equal the
        # Mahalanobis distance in the original space.
        self._whitening_matrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        self._whitened_means = class_means @ self._whitening_matrix

        fit_base_probs = np.asarray(fit_base_probs)
        fit_labels = np.asarray(fit_labels)
        fit_pred = np.argmax(fit_base_probs, axis=1)
        fit_score = -self._mahalanobis_min_distance(np.asarray(fit_features, dtype=np.float64))
        fit_correct = (fit_pred == fit_labels).astype(float)
        self._iso.fit(fit_score, fit_correct)
        self._is_fitted = True

        return {
            "shrinkage": self._shrinkage_used,
            "num_classes": self._num_classes,
            "feature_dim": self._feature_dim,
        }

    def calibrate(self, test_features: np.ndarray, test_base_probs: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before calibrate().")
        probs = np.asarray(test_base_probs, dtype=np.float64).copy()
        idx = np.argmax(probs, axis=1)
        rows = np.arange(probs.shape[0])

        score = -self._mahalanobis_min_distance(np.asarray(test_features, dtype=np.float64))
        p_top_new = np.clip(self._iso.predict(score), 1e-6, 1.0 - 1e-6)
        p_top_old = probs[rows, idx]

        scale = (1.0 - p_top_new) / np.clip(1.0 - p_top_old, 1e-6, 1.0)
        probs *= scale[:, None]
        probs[rows, idx] = p_top_new

        probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def get_params(self) -> Dict[str, Any]:
        return {
            "shrinkage": self.shrinkage,
            "shrinkage_used": self._shrinkage_used,
            "num_classes": self._num_classes,
            "feature_dim": self._feature_dim,
            "fitted": self._is_fitted,
        }
