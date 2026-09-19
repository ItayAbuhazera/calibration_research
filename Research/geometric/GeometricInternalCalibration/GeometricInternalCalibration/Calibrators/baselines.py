import numpy as np


class UncalibratedBaseline:
    """No fitting, just pass probabilities through safely."""
    def fit(self, *args, **kwargs):
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        probs = np.clip(probs, 1e-8, 1-1e-8)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs


"""
Note: Temperature scaling is provided by Calibrators/temperature_scaling.py in this repo.
This module intentionally does not duplicate it.
"""


from .isotonic_regression import TopLabelIsotonicCalibrator as IsotonicTopLabelCalibrator


