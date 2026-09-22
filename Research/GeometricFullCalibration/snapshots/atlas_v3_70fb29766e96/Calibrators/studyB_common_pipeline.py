"""
Study-B common factorial pipeline (§5, §8c of BENCHMARK_IMPLEMENTATION_PLAN.md).

Implements the one shared representation-neutral pooling -> per-layer
statistic -> rank-normalize -> unweighted mean -> rank-normalize -> isotonic
mapper used identically by all four Study-B cells:

                    GC statistic       DAC statistic
random internal         A                   B
layers (L=5)
DAC-prescribed          C                   D
layers (L=5)

This module does not reimplement either statistic: the GC-derived per-layer
statistic reuses utils.stability_space.StabilitySpace.calc_stab (the same
"separation" score GeometricCalibrator uses), and the DAC-derived per-layer
statistic reuses Calibrators.density_aware_calibration.LayerKNNScorer (DAC's
own k-th-nearest-neighbor density score) -- both exactly as they exist
elsewhere in this repo. Pooling (spatial-average + L2-normalize) is assumed
already applied upstream by
Experiments/extract_unified_studyAB_representations.py (§8b); this module
consumes already-pooled per-layer feature arrays and does not re-pool.

No learned weights anywhere in the aggregation (equal capacity across all
four cells, per §5): rank-normalization and the unweighted mean use no
fitted parameters beyond the fitting-split empirical CDFs themselves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

from Calibrators.density_aware_calibration import LayerKNNScorer
from utils.stability_space import StabilitySpace

VALID_STATISTICS = ("gc_separation", "dac_density")


def _fit_ecdf_sorted(values: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(values, dtype=np.float64))


def _ecdf_transform(sorted_fit_values: np.ndarray, query_values: np.ndarray) -> np.ndarray:
    """Percentile rank of each query value against a fitting-split ECDF, in [0, 1]."""
    n = len(sorted_fit_values)
    if n == 0:
        raise ValueError("Cannot rank-normalize against an empty fitting split.")
    ranks = np.searchsorted(sorted_fit_values, np.asarray(query_values, dtype=np.float64), side="right")
    return ranks / float(n)


class CommonFactorialCalibrator:
    """
    Shared Study-B mapper: per-layer statistic -> rank-normalize per layer
    (against the fitting split) -> unweighted mean across layers -> a second
    rank-normalization (against the fitting split) -> isotonic regression
    (PAV) against binary correctness -> assign to the predicted class via the
    same isotonic + uniform-spread convention used elsewhere in this repo
    (Calibrators/isotonic_regression.py::TopLabelIsotonicCalibrator).

    `representation_layers`: the fixed, ordered list of layer names this
    instance operates over -- callers pass the same list for A/B (corrected
    random-internal draw layers) or C/D (DAC-prescribed layers), never a
    per-cell-specific list, so the representation axis is controlled outside
    this class.
    """

    def __init__(self, representation_layers: List[str], statistic: str, k: int = 200) -> None:
        if statistic not in VALID_STATISTICS:
            raise ValueError(f"statistic must be one of {VALID_STATISTICS}, got {statistic!r}")
        if not representation_layers:
            raise ValueError("representation_layers must be non-empty")
        self.representation_layers = list(representation_layers)
        self.statistic = statistic
        self.k = k

        self._layer_models: Dict[str, Any] = {}
        self._layer_fit_ecdf: Dict[str, np.ndarray] = {}
        self._g_fit_ecdf: Optional[np.ndarray] = None
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._is_fitted = False

    def _layer_statistic_fit(
        self,
        layer_name: str,
        train_features: np.ndarray,
        train_labels: np.ndarray,
    ) -> Any:
        if self.statistic == "gc_separation":
            model = StabilitySpace(
                np.asarray(train_features, dtype=np.float64),
                np.asarray(train_labels),
                library="fast_separation",
                metric="l2",
            )
        else:
            model = LayerKNNScorer(k=self.k, use_gpu=False)
            model.fit(np.asarray(train_features, dtype=np.float64))
        return model

    def _layer_statistic_score(
        self,
        layer_name: str,
        features: np.ndarray,
        predicted_labels: np.ndarray,
    ) -> np.ndarray:
        model = self._layer_models[layer_name]
        if self.statistic == "gc_separation":
            return np.asarray(model.calc_stab(np.asarray(features, dtype=np.float64), predicted_labels))
        return np.asarray(model.score(np.asarray(features, dtype=np.float64)))

    def _aggregate(
        self,
        layer_features: Dict[str, np.ndarray],
        predicted_labels: np.ndarray,
        fit_mode: bool,
    ) -> np.ndarray:
        rank_normalized_per_layer = []
        for layer_name in self.representation_layers:
            raw_score = self._layer_statistic_score(layer_name, layer_features[layer_name], predicted_labels)
            if fit_mode:
                self._layer_fit_ecdf[layer_name] = _fit_ecdf_sorted(raw_score)
            rank_normalized_per_layer.append(
                _ecdf_transform(self._layer_fit_ecdf[layer_name], raw_score)
            )
        # Unweighted mean across layers -- no learned weights (§5).
        return np.mean(np.stack(rank_normalized_per_layer, axis=0), axis=0)

    def fit(
        self,
        train_layer_features: Dict[str, np.ndarray],
        train_labels: np.ndarray,
        fit_layer_features: Dict[str, np.ndarray],
        fit_base_probs: np.ndarray,
        fit_labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        train_layer_features: {layer_name: (N_train, D_l)} reference bank
            (training set, per plan §7) used to fit each layer's statistic.
        fit_layer_features/fit_base_probs/fit_labels: the calibration/fitting
            split (per plan §7, reused verbatim across all Study-B cells)
            used to fit both ECDFs and the final isotonic mapping.
        """
        missing = [l for l in self.representation_layers if l not in train_layer_features]
        if missing:
            raise ValueError(f"train_layer_features missing layers: {missing}")
        missing = [l for l in self.representation_layers if l not in fit_layer_features]
        if missing:
            raise ValueError(f"fit_layer_features missing layers: {missing}")

        train_labels = np.asarray(train_labels)
        for layer_name in self.representation_layers:
            self._layer_models[layer_name] = self._layer_statistic_fit(
                layer_name, train_layer_features[layer_name], train_labels
            )

        fit_base_probs = np.asarray(fit_base_probs)
        fit_labels = np.asarray(fit_labels)
        fit_pred = np.argmax(fit_base_probs, axis=1)

        g_fit = self._aggregate(fit_layer_features, fit_pred, fit_mode=True)
        self._g_fit_ecdf = _fit_ecdf_sorted(g_fit)
        g_fit_ranked = _ecdf_transform(self._g_fit_ecdf, g_fit)

        fit_correct = (fit_pred == fit_labels).astype(float)
        self._iso.fit(g_fit_ranked, fit_correct)
        self._is_fitted = True

        return {
            "statistic": self.statistic,
            "representation_layers": list(self.representation_layers),
            "k": self.k if self.statistic == "dac_density" else None,
            "num_layers": len(self.representation_layers),
        }

    def calibrate(
        self,
        test_layer_features: Dict[str, np.ndarray],
        test_base_probs: np.ndarray,
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before calibrate().")
        missing = [l for l in self.representation_layers if l not in test_layer_features]
        if missing:
            raise ValueError(f"test_layer_features missing layers: {missing}")

        probs = np.asarray(test_base_probs, dtype=np.float64).copy()
        idx = np.argmax(probs, axis=1)
        rows = np.arange(probs.shape[0])

        g_test = self._aggregate(test_layer_features, idx, fit_mode=False)
        g_test_ranked = _ecdf_transform(self._g_fit_ecdf, g_test)
        p_top_new = np.clip(self._iso.predict(g_test_ranked), 1e-6, 1.0 - 1e-6)
        p_top_old = probs[rows, idx]

        scale = (1.0 - p_top_new) / np.clip(1.0 - p_top_old, 1e-6, 1.0)
        probs *= scale[:, None]
        probs[rows, idx] = p_top_new

        probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def get_params(self) -> Dict[str, Any]:
        return {
            "statistic": self.statistic,
            "representation_layers": list(self.representation_layers),
            "k": self.k if self.statistic == "dac_density" else None,
            "fitted": self._is_fitted,
        }
