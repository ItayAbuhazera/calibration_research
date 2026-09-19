"""
DAC Layer Selection Strategy Utilities

Implements the predetermined layer mapping described in
"Beyond In-Domain Scenarios: Robust Density-Aware Calibration"
(Tomani et al., ICML 2023, Appendix C.1).
"""

from __future__ import annotations

import logging
from typing import Dict, List

from utils.logging_config import get_logger
logger = get_logger(__name__)


class DACLayerMapper:
    """
    Map DAC paper layer definitions to discovered layer indices.

    DAC samples representative layers at coarse-grained boundaries. We mimic
    this behaviour by uniformly sampling along the discovered depth while
    always including the first and final layers (logits head).
    """

    DEFAULT_COUNT = 6

    LAYER_CONFIGS: Dict[str, Dict[str, str]] = {
        "resnet18": {
            "description": "PRE-BLOCK, BLOCK-1, BLOCK-2, BLOCK-3, BLOCK-4, LOGITS"
        },
        "resnet50": {
            "description": "PRE-BLOCK, BLOCK-1, BLOCK-2, BLOCK-3, BLOCK-4, LOGITS"
        },
        "densenet121": {
            "description": "PRE-BLOCK, BLOCK-1, BLOCK-2, BLOCK-3, PENULTIMATE, LOGITS"
        },
    }

    @staticmethod
    def _uniform_positions(count: int, target: int) -> List[int]:
        """
        Return `target` evenly spaced positions across `count` items (0-indexed).
        Always includes 0 and count-1 when possible.
        """
        if count <= 0 or target <= 0:
            return []
        if target >= count:
            return list(range(count))
        if target == 1:
            return [0]

        step = (count - 1) / float(target - 1)
        positions = [int(round(i * step)) for i in range(target)]
        # Deduplicate while preserving order
        seen = set()
        deduped: List[int] = []
        for pos in positions:
            pos = max(0, min(count - 1, pos))
            if pos not in seen:
                deduped.append(pos)
                seen.add(pos)
        return deduped

    @staticmethod
    def get_layer_indices(model_name: str, available_layers: List[int]) -> List[int]:
        """
        Map the DAC paper's predetermined layer picks to discovered indices.

        Args:
            model_name: Backbone architecture name (e.g., 'resnet18').
            available_layers: Sorted list of discovered layer indices.

        Returns:
            Subset of available_layers following DAC sampling.
        """

        if not available_layers:
            logger.warning("DAC layer mapper received empty layer list.")
            return []

        model_key = (model_name or "").lower()
        config = DACLayerMapper.LAYER_CONFIGS.get(model_key)
        total_layers = len(available_layers)
        desired = DACLayerMapper.DEFAULT_COUNT

        if config is None:
            logger.warning(
                "Model '%s' not in DAC config table; falling back to uniform sampling.",
                model_name,
            )
        else:
            logger.info("Applying DAC mapping for %s: %s", model_name, config["description"])

        # DenseNet has a dedicated penultimate representation (classifier input).
        if model_key.startswith("densenet") and total_layers >= 2:
            prefix_count = max(0, desired - 2)
            prefix_positions = DACLayerMapper._uniform_positions(total_layers - 2, prefix_count)
            positions = prefix_positions + [total_layers - 2, total_layers - 1]
        else:
            positions = DACLayerMapper._uniform_positions(total_layers, min(desired, total_layers))

        # Map positions back to actual layer indices, preserving order & uniqueness.
        selected: List[int] = []
        seen_layers = set()
        for pos in positions:
            idx = available_layers[min(pos, total_layers - 1)]
            if idx not in seen_layers:
                selected.append(idx)
                seen_layers.add(idx)

        if not selected:
            # Fallback: return first and last to avoid empty selection.
            fallback = sorted({available_layers[0], available_layers[-1]})
            logger.warning(
                "DAC mapping produced no positions (model=%s, layers=%d). "
                "Falling back to endpoints: %s",
                model_name,
                total_layers,
                fallback,
            )
            return fallback

        return selected


def dac_predetermined_selection(
    layer_metrics: Dict[int, Dict[str, float]],
    model_name: str,
    **_: Dict,
) -> List[int]:
    """
    Adapter to match the selection strategy interface.

    Args:
        layer_metrics: Metric dictionary keyed by layer index. Only the keys
                       are used because DAC ignores validation metrics.
        model_name: Backbone identifier used to pick the correct mapping.

    Returns:
        List of layer indices following DAC's predetermined schedule.
    """

    available_layers = sorted(layer_metrics.keys())
    return DACLayerMapper.get_layer_indices(model_name, available_layers)




