"""
KCal-lite: top-k per-class KDE-lite calibration baseline.

This is a lightweight baseline that reuses benchmark embeddings and truncates
KDE to top-k per-class neighbors. It does NOT reproduce the full KCal method,
which requires a learned projection (Pi) and bandwidth selection via cross-validation
on a calibration set. See: Lin et al., "Taking a Step Back with KCal" (2022).

At k=1 the posterior reduces exactly to softmax(-gamma * D_1nn), matching
the SoftmaxKNNBlendCalibrator kernel component.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils.stability_space import StabilitySpace


class KCalLiteCalibrator:
    """
    KCal-lite top-k KDE-lite calibrator.

    For each test point x the per-class affinity is:
        a_c = mean_{r=1..k} exp(-gamma * d_r(x, c))
    where d_r(x, c) is the r-th nearest training distance from x to class c.

    The calibrated posterior is:
        p_kcal[c] = normalize(a[c])   (i.e. softmax over classes)

    The blended output is:
        q = alpha * p_model + (1 - alpha) * p_kcal

    alpha and gamma are jointly tuned on validation NLL using the same grids
    as SoftmaxKNNBlendCalibrator for direct comparability.

    At k=1: a_c = exp(-gamma * d_1nn(x, c)), so p_kcal = softmax(-gamma * D_1nn),
    identical to the SoftmaxKNNBlendCalibrator kernel component.
    """

    DEFAULT_ALPHA_GRID = np.array(
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=np.float64
    )
    DEFAULT_GAMMA_GRID = np.array(
        [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], dtype=np.float64
    )

    def __init__(
        self,
        model,
        y_train: np.ndarray,
        k_per_class: int = 1,
        X_train_embed: Optional[np.ndarray] = None,
        alpha_grid=None,
        gamma_grid=None,
        eps: float = 1e-12,
        stability_space=None,
        metric: str = "l2",
        library: str = "fast_separation",
    ):
        self.model = model
        self.eps = float(eps)
        self.metric = metric.lower()
        self.library = library
        self.y_train = np.asarray(y_train)
        self.num_labels = len(np.unique(self.y_train))
        self.k_per_class = int(k_per_class)

        self.alpha_grid = (
            np.asarray(alpha_grid, dtype=np.float64)
            if alpha_grid is not None
            else self.DEFAULT_ALPHA_GRID.copy()
        )
        self.gamma_grid = (
            np.asarray(gamma_grid, dtype=np.float64)
            if gamma_grid is not None
            else self.DEFAULT_GAMMA_GRID.copy()
        )

        if stability_space is not None:
            self.stab_space = stability_space
        else:
            if X_train_embed is None:
                raise ValueError(
                    "X_train_embed is required when stability_space is not provided."
                )
            self.stab_space = StabilitySpace(
                X_train_embed,
                y_train,
                library=library,
                metric=self.metric,
            )

        self.best_alpha: Optional[float] = None
        self.best_gamma: Optional[float] = None
        self.validation_nll_grid: Optional[np.ndarray] = None
        self.is_fitted = False

    def _kcal_probs(self, knn_distances: np.ndarray, gamma: float) -> np.ndarray:
        """
        Compute p_kcal from [N, C, k] distances using numerically stable log-sum-exp.

        p_kcal[i, c] = softmax_c( logsumexp(-gamma * knn_distances[i, c, :]) )

        The 1/k mean factor cancels under softmax normalization and is omitted.
        At k=1 this equals softmax(-gamma * knn_distances[:, :, 0]).
        """
        gamma = float(gamma)
        knn_distances = np.asarray(knn_distances, dtype=np.float64)
        N, C, k = knn_distances.shape

        if gamma == 0.0:
            return np.full((N, C), 1.0 / C, dtype=np.float64)

        neg_gd = -gamma * knn_distances          # [N, C, k]
        mx = neg_gd.max(axis=2, keepdims=True)   # [N, C, 1]
        log_scores = mx.squeeze(2) + np.log(
            np.sum(np.exp(neg_gd - mx), axis=2)
        )                                         # [N, C]

        # stable softmax over classes
        log_scores -= log_scores.max(axis=1, keepdims=True)
        exp_s = np.exp(log_scores)
        return exp_s / np.clip(exp_s.sum(axis=1, keepdims=True), self.eps, None)

    def _blend(
        self,
        model_probs: np.ndarray,
        knn_distances: np.ndarray,
        alpha: float,
        gamma: float,
    ) -> np.ndarray:
        p_kcal = self._kcal_probs(knn_distances, gamma)
        return float(alpha) * model_probs + (1.0 - float(alpha)) * p_kcal

    def _multiclass_nll(self, probs: np.ndarray, labels: np.ndarray) -> float:
        idx = np.arange(len(labels))
        return float(-np.mean(np.log(np.clip(probs[idx, labels], self.eps, 1.0))))

    def fit(
        self,
        X_val_embed: np.ndarray,
        y_val: np.ndarray,
        model_probs: Optional[np.ndarray] = None,
        X_val_original: Optional[np.ndarray] = None,
        val_knn_distances: Optional[np.ndarray] = None,
    ):
        """
        Tune alpha and gamma on validation NLL.

        Args:
            X_val_embed: Validation embeddings, shape [N_val, D].
            y_val: Validation labels, shape [N_val].
            model_probs: Model softmax probabilities [N_val, C]. If None,
                         X_val_original is used to obtain them via model.predict_proba.
            X_val_original: Raw validation inputs (only needed if model_probs is None).
            val_knn_distances: Pre-computed [N_val, C, k] distances. Computed from
                               X_val_embed when not provided.

        Returns:
            self
        """
        if model_probs is None:
            if X_val_original is None:
                raise ValueError(
                    "Either model_probs or X_val_original must be provided for fitting."
                )
            model_probs = self.model.predict_proba(X_val_original)

        model_probs = np.asarray(model_probs, dtype=np.float64)

        if val_knn_distances is None:
            val_knn_distances = self.stab_space.calc_per_class_knn_distances(
                X_val_embed, self.k_per_class
            )

        n_alpha = len(self.alpha_grid)
        n_gamma = len(self.gamma_grid)
        nll_grid = np.full((n_alpha, n_gamma), np.inf, dtype=np.float64)
        best_nll = float("inf")
        best_alpha: Optional[float] = None
        best_gamma: Optional[float] = None

        for i, alpha in enumerate(self.alpha_grid):
            for j, gamma in enumerate(self.gamma_grid):
                blended = self._blend(model_probs, val_knn_distances, float(alpha), float(gamma))
                nll = self._multiclass_nll(blended, y_val)
                nll_grid[i, j] = nll
                if nll < best_nll:
                    best_nll = nll
                    best_alpha = float(alpha)
                    best_gamma = float(gamma)

        self.best_alpha = best_alpha
        self.best_gamma = best_gamma
        self.validation_nll_grid = nll_grid
        self.is_fitted = True
        return self

    def calibrate(
        self,
        X_test_embed: np.ndarray,
        model_probs: Optional[np.ndarray] = None,
        X_test_original: Optional[np.ndarray] = None,
        test_knn_distances: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply calibration with the tuned alpha and gamma.

        Args:
            X_test_embed: Test embeddings, shape [N_test, D].
            model_probs: Model softmax probabilities [N_test, C].
            X_test_original: Raw test inputs (only needed if model_probs is None).
            test_knn_distances: Pre-computed [N_test, C, k] distances.

        Returns:
            np.ndarray: Calibrated probabilities [N_test, C], rows sum to 1.
        """
        if not self.is_fitted:
            raise ValueError("Call fit() before calibrate().")
        if model_probs is None:
            if X_test_original is None:
                raise ValueError(
                    "Either model_probs or X_test_original must be provided for calibration."
                )
            model_probs = self.model.predict_proba(X_test_original)

        model_probs = np.asarray(model_probs, dtype=np.float64)

        if test_knn_distances is None:
            test_knn_distances = self.stab_space.calc_per_class_knn_distances(
                X_test_embed, self.k_per_class
            )

        return self._blend(model_probs, test_knn_distances, self.best_alpha, self.best_gamma)

    def get_params(self) -> dict:
        return {
            "metric": self.metric,
            "library": self.library,
            "k_per_class": self.k_per_class,
            "num_labels": self.num_labels,
            "is_fitted": self.is_fitted,
            "best_alpha": self.best_alpha,
            "best_gamma": self.best_gamma,
            "alpha_grid": self.alpha_grid.tolist(),
            "gamma_grid": self.gamma_grid.tolist(),
            "validation_nll_grid": (
                self.validation_nll_grid.tolist()
                if self.validation_nll_grid is not None
                else None
            ),
        }
