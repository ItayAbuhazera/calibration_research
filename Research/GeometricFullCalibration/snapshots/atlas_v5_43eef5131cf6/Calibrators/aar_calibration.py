"""
AAR-Lightweight calibrator — taxonomic fallback (NOT official AAR).

Form:  log q_k(x) ∝ phi(a(x)) * log p_k(x) + S_k

where:
  a(x)     = mean per-class 1-NN distance (geometric atypicality proxy)
  phi(a)   = exp(w0 + w1 * a)   (sample-dependent positive scalar)
  S_k      = global per-class additive log-bias

This is NOT the official AAR implementation. The official AAR uses GMM-based
negative log class-conditional density in penultimate features. This version
approximates atypicality via mean 1-NN distance, which is cheaper but less
principled. Always set official_implementation=False in metadata.

joint_sample_class_dependent=False because phi is sample-only and S_k is
class-only — the correction is not a joint function of (sample, class).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


# Metadata to attach to method entries produced by this calibrator.
METADATA = {
    "official_implementation": False,
    "atypicality_estimator": "nn_distance_proxy",
    "notes": (
        "Lightweight taxonomic fallback. Atypicality estimated via per-sample mean 1-NN distance. "
        "Not official AAR (which uses GMM-based negative log class-conditional density in "
        "penultimate features)."
    ),
}


class AARLightweightCalibrator:
    """
    Lightweight AAR-inspired calibrator.

    fit(logits_val, labels_val, distance_matrix_val) — fits w0, w1, S_k via L-BFGS-B on val NLL.
    calibrate(logits_test, distance_matrix_test) -> np.ndarray [N, C] probabilities.
    get_params() -> dict.
    """

    def __init__(self) -> None:
        self._w: Optional[np.ndarray] = None  # [w0, w1]
        self._S: Optional[np.ndarray] = None  # [C] per-class log-bias
        self._num_classes: Optional[int] = None
        self._fit_nll: Optional[float] = None

    def _atypicality(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Mean per-class 1-NN distance per sample. shape: [N]"""
        return distance_matrix.mean(axis=1)

    def _logits_corrected(
        self,
        logits: np.ndarray,
        distance_matrix: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """Compute corrected logits given flat parameter vector [w0, w1, S_0, ..., S_{C-1}]."""
        C = logits.shape[1]
        w0, w1 = params[0], params[1]
        S = params[2 : 2 + C]
        a = self._atypicality(distance_matrix)  # [N]
        phi = np.exp(w0 + w1 * a)  # [N] — positive scalar per sample
        corrected = phi[:, None] * logits + S[None, :]
        return corrected

    def fit(
        self,
        logits_val: np.ndarray,
        labels_val: np.ndarray,
        distance_matrix_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Fit via L-BFGS-B minimising NLL on the validation set."""
        C = logits_val.shape[1]
        self._num_classes = C

        def _nll(params: np.ndarray) -> float:
            corr = self._logits_corrected(logits_val, distance_matrix_val, params)
            # Numerically stable softmax + cross-entropy
            corr_shifted = corr - corr.max(axis=1, keepdims=True)
            log_sum_exp = np.log(np.exp(corr_shifted).sum(axis=1))
            log_p_true = corr_shifted[np.arange(len(labels_val)), labels_val] - log_sum_exp
            return float(-log_p_true.mean())

        # Initialise: w0=0 (phi≡1), w1=0 (no atypicality correction), S=0
        x0 = np.zeros(2 + C, dtype=np.float64)
        result = minimize(_nll, x0, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-9})

        self._w = result.x[:2].copy()
        self._S = result.x[2 : 2 + C].copy()
        self._fit_nll = float(result.fun)

        return {
            "fit_nll": self._fit_nll,
            "success": bool(result.success),
            "message": str(result.message),
            "w0": float(self._w[0]),
            "w1": float(self._w[1]),
        }

    def calibrate(
        self,
        logits_test: np.ndarray,
        distance_matrix_test: np.ndarray,
    ) -> np.ndarray:
        if self._w is None or self._S is None:
            raise RuntimeError("Call fit() before calibrate()")

        C = self._num_classes
        params = np.concatenate([self._w, self._S])
        corr = self._logits_corrected(logits_test, distance_matrix_test, params)
        corr_shifted = corr - corr.max(axis=1, keepdims=True)
        exp_c = np.exp(corr_shifted)
        probs = exp_c / exp_c.sum(axis=1, keepdims=True)
        return probs.astype(np.float64)

    def get_params(self) -> Dict[str, Any]:
        return {
            "fitted": self._w is not None,
            "num_classes": self._num_classes,
            "w0": float(self._w[0]) if self._w is not None else None,
            "w1": float(self._w[1]) if self._w is not None else None,
            "S_norm": float(np.linalg.norm(self._S)) if self._S is not None else None,
            "fit_nll": self._fit_nll,
            **METADATA,
        }
