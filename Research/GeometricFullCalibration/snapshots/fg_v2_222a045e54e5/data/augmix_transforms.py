"""Placeholder for optional AugMix-based calibration experiments."""

from __future__ import annotations


class AugMixTransforms:
    """Explicit placeholder for functionality not bundled in this snapshot."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, image):
        raise NotImplementedError(
            "AugMix transforms are referenced by optional calibration paths but "
            "are not bundled in this repository snapshot."
        )
