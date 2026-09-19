import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import Any, Dict, List, Optional

from .base_calibrator import BaseCalibrator
from utils.logging_config import get_logger

logger = get_logger(__name__)


class OneVsRestIsotonicCalibration(BaseCalibrator):
    """
    Multiclass one-vs-rest isotonic calibration.

    For each class k, fit a binary isotonic model on:
      feature: class-k score/probability
      target: 1[y == k]

    Supports logits or probabilities as inputs. At inference, each class is
    transformed independently, clipped to [0, 1], and rows are renormalized.
    Degenerate classes (missing positives/negatives) use an explicit constant
    prior fallback for safety.
    """

    def __init__(self, probability_tol: float = 1e-6, **kwargs: Any):
        super().__init__(**kwargs)
        self.n_classes: Optional[int] = None
        self.calibrators: List[Optional[IsotonicRegression]] = []
        self.class_priors: Optional[np.ndarray] = None
        self.class_modes: List[str] = []
        self.class_counts: List[Dict[str, int]] = []
        self.input_mode_last_fit: Optional[str] = None
        self.probability_tol = float(probability_tol)
        self.safe_row_distribution: Optional[np.ndarray] = None

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> None:
        scores = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels)
        if scores.ndim != 2:
            raise ValueError("Expected 2D input array with shape (n_samples, n_classes)")
        if labels.ndim != 1 or labels.shape[0] != scores.shape[0]:
            raise ValueError("labels must be 1D and aligned with input rows")

        probs, input_mode = self._prepare_probabilities(scores)
        self.input_mode_last_fit = input_mode
        self.n_classes = probs.shape[1]

        self.class_priors = np.array(
            [float(np.mean(labels == class_idx)) for class_idx in range(self.n_classes)],
            dtype=np.float64,
        )
        self.safe_row_distribution = self._build_safe_distribution(self.class_priors)

        self.calibrators = []
        self.class_modes = []
        self.class_counts = []

        for class_idx in range(self.n_classes):
            y_binary = (labels == class_idx).astype(np.float64)
            positives = int(np.sum(y_binary))
            negatives = int(len(y_binary) - positives)
            self.class_counts.append(
                {"class": class_idx, "positives": positives, "negatives": negatives}
            )

            if positives == 0 or negatives == 0:
                self.calibrators.append(None)
                self.class_modes.append("constant_prior")
                logger.warning(
                    "OVR Isotonic class %d is degenerate (positives=%d negatives=%d); "
                    "using constant prior fallback.",
                    class_idx,
                    positives,
                    negatives,
                )
                continue

            calibrator = IsotonicRegression(out_of_bounds="clip")
            try:
                calibrator.fit(probs[:, class_idx], y_binary)
                self.calibrators.append(calibrator)
                self.class_modes.append("isotonic")
            except Exception as exc:
                logger.warning(
                    "OVR Isotonic fit failed for class %d (%s); using constant prior fallback.",
                    class_idx,
                    exc,
                )
                self.calibrators.append(None)
                self.class_modes.append("constant_prior")

        self.is_fitted = True

    def calibrate(
        self, logits: np.ndarray, features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calibration")
        if self.n_classes is None or self.class_priors is None:
            raise RuntimeError("Calibrator is in an invalid state; fit again")

        scores = np.asarray(logits, dtype=np.float64)
        probs, _ = self._prepare_probabilities(scores)
        if probs.shape[1] != self.n_classes:
            raise ValueError(
                f"Expected {self.n_classes} classes, received {probs.shape[1]} classes"
            )

        calibrated = np.zeros_like(probs, dtype=np.float64)
        for class_idx in range(self.n_classes):
            class_probs = probs[:, class_idx]
            if self.calibrators[class_idx] is not None:
                transformed = self.calibrators[class_idx].predict(class_probs)
            else:
                transformed = np.full(
                    shape=class_probs.shape,
                    fill_value=float(self.class_priors[class_idx]),
                    dtype=np.float64,
                )
            calibrated[:, class_idx] = np.clip(transformed, 0.0, 1.0)

        calibrated = np.where(np.isfinite(calibrated), calibrated, 0.0)
        row_sums = np.sum(calibrated, axis=1, keepdims=True)
        invalid_rows = (~np.isfinite(row_sums[:, 0])) | (row_sums[:, 0] <= 0.0)
        if np.any(invalid_rows):
            calibrated[invalid_rows] = self.safe_row_distribution
            row_sums = np.sum(calibrated, axis=1, keepdims=True)

        calibrated = calibrated / np.clip(row_sums, 1e-12, None)
        return calibrated

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_classes": self.n_classes,
            "input_mode_last_fit": self.input_mode_last_fit,
            "degenerate_fallback": "constant_prior",
            "class_priors": (
                self.class_priors.tolist() if self.class_priors is not None else None
            ),
            "class_modes": self.class_modes,
            "class_counts": self.class_counts,
            "probability_tol": self.probability_tol,
        }

    def _prepare_probabilities(self, scores: np.ndarray) -> tuple[np.ndarray, str]:
        if self._looks_like_probabilities(scores):
            probs = np.clip(scores, 0.0, 1.0)
            probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), 1e-12, None)
            return probs, "probabilities"
        return self._softmax(scores), "logits"

    def _looks_like_probabilities(self, scores: np.ndarray) -> bool:
        if np.any(~np.isfinite(scores)):
            return False
        if np.any(scores < -self.probability_tol):
            return False
        row_sums = np.sum(scores, axis=1)
        return bool(np.all(np.abs(row_sums - 1.0) <= self.probability_tol))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.clip(np.sum(exp_x, axis=1, keepdims=True), 1e-12, None)

    @staticmethod
    def _build_safe_distribution(class_priors: np.ndarray) -> np.ndarray:
        priors = np.clip(np.asarray(class_priors, dtype=np.float64), 0.0, 1.0)
        total = float(np.sum(priors))
        if not np.isfinite(total) or total <= 0.0:
            return np.full_like(priors, 1.0 / float(len(priors)))
        return priors / total
