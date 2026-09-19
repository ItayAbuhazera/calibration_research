"""
Multi-Objective Layer Selection Strategies

Implements various strategies for selecting optimal neural network layers
using multiple validation metrics simultaneously.
"""

import logging
from collections import Counter
from enum import Enum
from typing import Callable, Dict, List, Set, Tuple, Literal

import numpy as np

from Experiments.metric_config import METRIC_REGISTRY


from utils.logging_config import get_logger
logger = get_logger(__name__)


class SelectionStrategy(str, Enum):
    """Available layer selection strategies."""

    PARETO = "pareto"
    VOTING = "voting"
    EPSILON_BALL = "epsilon_ball"
    SINGLE_METRIC = "single_metric"


def _handle_nan(value: float, direction: str) -> float:
    """
    Replace NaN with worst possible value for the optimization direction.

    Args:
        value: Metric value (possibly NaN)
        direction: 'maximize' or 'minimize'

    Returns:
        Original value or worst-case replacement
    """

    if np.isnan(value) or not np.isfinite(value):
        return -np.inf if direction == "maximize" else np.inf
    return value


def pareto_front_selection(
    layer_metrics: Dict[int, Dict[str, float]],
    metrics: List[str],
    strict: bool = True,
) -> List[int]:
    """
    Select layers on the Pareto front (not dominated by any other layer).

    A layer is on the Pareto front if no other layer is strictly better
    on all metrics. This implements multi-objective optimization without
    needing to weight metrics arbitrarily.

    Args:
        layer_metrics: Mapping from layer_idx to metric values
        metrics: List of metric names to optimize
        strict: If True, require strict domination (better on at least one metric)
               and not worse on any. If False, allow weak domination (better on all).

    Returns:
        Sorted list of layer indices on the Pareto front

    Raises:
        ValueError: If metrics list is empty or contains unknown metrics

    Example:
        >>> layer_metrics = {
        ...     0: {'ece_score': 0.8, 'margin_tail_cvar': 0.6},
        ...     1: {'ece_score': 0.9, 'margin_tail_cvar': 0.5},  # Dominated
        ...     2: {'ece_score': 0.7, 'margin_tail_cvar': 0.9}
        ... }
        >>> pareto_front_selection(layer_metrics, ['ece_score', 'margin_tail_cvar'])
        [0, 2]  # Both layers 0 and 2 are non-dominated
    """

    if not metrics:
        raise ValueError("Must provide at least one metric")

    if not layer_metrics:
        logger.warning("Empty layer_metrics dict")
        return []

    # Validate all metrics are known
    METRIC_REGISTRY.validate_metrics(metrics)

    # Get metric directions
    metric_dirs = {m: METRIC_REGISTRY.get_direction(m) for m in metrics}

    layers = list(layer_metrics.keys())
    pareto_layers: List[int] = []

    for layer_i in layers:
        dominated = False

        for layer_j in layers:
            if layer_i == layer_j:
                continue

            # Check if layer_j dominates layer_i
            better_count = 0
            worse_count = 0

            for metric in metrics:
                val_i = _handle_nan(layer_metrics[layer_i].get(metric, np.nan), metric_dirs[metric])
                val_j = _handle_nan(layer_metrics[layer_j].get(metric, np.nan), metric_dirs[metric])

                if metric_dirs[metric] == "maximize":
                    if val_j > val_i:
                        better_count += 1
                    elif val_j < val_i:
                        worse_count += 1
                else:  # minimize
                    if val_j < val_i:
                        better_count += 1
                    elif val_j > val_i:
                        worse_count += 1

            # Domination criteria
            if strict:
                # Strict: j must be better on at least one AND not worse on any
                if better_count > 0 and worse_count == 0:
                    dominated = True
                    break
            else:
                # Weak: j must be better on ALL metrics
                if better_count == len(metrics):
                    dominated = True
                    break

        if not dominated:
            pareto_layers.append(layer_i)

    result = sorted(pareto_layers)
    logger.info(f"Pareto selection: {len(result)}/{len(layers)} layers on front")
    return result


# ============================================================================
# DAC COMPARISON STRATEGIES
# ============================================================================


def create_dac_comparison_strategies(model_name: str) -> Dict[str, Callable[[Dict[int, Dict[str, float]]], List[int]]]:
    """
    Build a suite of selection strategies used for DAC comparisons.

    Args:
        model_name: Backbone identifier passed to DAC mapping.

    Returns:
        Mapping of strategy name to callable.
    """

    from Experiments.dac_layer_mappings import dac_predetermined_selection

    strategies: Dict[str, Callable[[Dict[int, Dict[str, float]]], List[int]]] = {}

    metric_alias = {
        "kfold_ece": "kfold_ece_utility",
        "decisiveness": "calibration_decisiveness",
    }

    # 1) Single-layer optimal picks for individual metrics
    for metric in ["kfold_ece", "margin_tail_cvar", "decisiveness", "nc1", "nc4"]:
        metric_name = metric_alias.get(metric, metric)
        strategies[f"optimal_{metric}"] = lambda lm, m=metric_name: single_metric_selection(lm, m, k=1)

    # 2) Top-k ensembles for key metrics
    for k in [2, 3]:
        for metric in ["margin_tail_cvar", "decisiveness"]:
            metric_name = metric_alias.get(metric, metric)
            strategies[f"top{k}_{metric}"] = lambda lm, m=metric_name, topk=k: single_metric_selection(lm, m, topk)

    # 3) PSC-style NC filtering (intersection of NC1 / NC4 epsilon-balls)
    strategies["psc_nc_filtered"] = lambda lm: epsilon_ball_selection(
        lm,
        metrics=["nc1", "nc4"],
        epsilon=0.1,
        mode="intersection",
    )

    # 4) Lightweight geometric epsilon union
    strategies["geometric_epsilon"] = lambda lm: epsilon_ball_selection(
        lm,
        metrics=["margin_tail_cvar", "kfold_ece_utility"],
        epsilon=0.005,
        mode="union",
    )

    # 5) DAC predetermined mapping (ignores metrics, depends on model)
    strategies["dac_predetermined"] = lambda lm: dac_predetermined_selection(lm, model_name=model_name)

    # 6) Pareto core (ECE + geometry proxies)
    strategies["pareto_core"] = lambda lm: pareto_front_selection(
        lm,
        metrics=["kfold_ece_utility", "margin_tail_cvar", "nc1"],
        strict=True,
    )

    # 7) Voting across calibration/geometric metrics
    strategies["voting_top3_core"] = lambda lm: metric_voting_selection(
        lm,
        metrics=["kfold_ece_utility", "margin_tail_cvar", "calibration_decisiveness"],
        k_per_metric=3,
    )

    return strategies


def metric_voting_selection(
    layer_metrics: Dict[int, Dict[str, float]],
    metrics: List[str],
    k_per_metric: int = 3,
    return_all: bool = False,
) -> List[int]:
    """
    Select layers using weighted voting from multiple metrics.

    Each metric "votes" for its top-k layers, with rank-based weighting:
    - 1st place: k points
    - 2nd place: k-1 points
    - ...
    - kth place: 1 point

    Args:
        layer_metrics: Mapping from layer_idx to metric values
        metrics: List of metric names to use for voting
        k_per_metric: Number of top layers each metric votes for
        return_all: If True, return all layers sorted by votes
                   If False, return only layers with positive votes

    Returns:
        List of layer indices sorted by vote count (descending)

    Example:
        >>> layer_metrics = {
        ...     0: {'ece_score': 0.9, 'margin_tail_cvar': 0.5},
        ...     1: {'ece_score': 0.7, 'margin_tail_cvar': 0.9},
        ...     2: {'ece_score': 0.8, 'margin_tail_cvar': 0.7}
        ... }
        >>> metric_voting_selection(layer_metrics, ['ece_score', 'margin_tail_cvar'], k_per_metric=2)
        [0, 2, 1]  # Layer 0 gets most votes
    """

    if not metrics:
        raise ValueError("Must provide at least one metric")

    if not layer_metrics:
        return []

    METRIC_REGISTRY.validate_metrics(metrics)

    votes: Counter[int] = Counter()

    for metric in metrics:
        direction = METRIC_REGISTRY.get_direction(metric)

        # Get valid (layer, value) pairs for this metric
        valid_pairs: List[Tuple[int, float]] = []
        for layer, values in layer_metrics.items():
            val = values.get(metric, np.nan)
            val = _handle_nan(val, direction)
            valid_pairs.append((layer, val))

        # Sort by metric value
        reverse = direction == "maximize"
        sorted_layers = sorted(valid_pairs, key=lambda x: x[1], reverse=reverse)

        # Assign votes to top-k
        top_k = min(k_per_metric, len(sorted_layers))
        for rank, (layer, _) in enumerate(sorted_layers[:top_k]):
            points = top_k - rank
            votes[layer] += points
            logger.debug(f"  {metric}: layer {layer} gets {points} points (rank {rank + 1})")

    # Return layers sorted by vote count
    if return_all:
        result = [layer for layer, _ in votes.most_common()]
    else:
        result = [layer for layer, count in votes.most_common() if count > 0]

    logger.info(f"Voting selection: {len(result)} layers received votes")
    return result


def epsilon_ball_selection(
    layer_metrics: Dict[int, Dict[str, float]],
    metrics: List[str],
    epsilon: float = 0.01,
    mode: Literal["union", "intersection"] = "union",
) -> List[int]:
    """
    Select layers within epsilon-ball of best for each metric.

    For each metric, finds the best layer and selects all layers
    within epsilon of that best value. Then combines selections
    using union or intersection.

    Args:
        layer_metrics: Mapping from layer_idx to metric values
        metrics: List of metric names
        epsilon: Tolerance threshold (absolute, not relative)
        mode: How to combine selections across metrics
              'union': Layer selected if within epsilon for ANY metric
              'intersection': Layer selected if within epsilon for ALL metrics

    Returns:
        Sorted list of selected layer indices

    Example:
        >>> layer_metrics = {
        ...     0: {'ece_score': 0.90, 'margin_tail_cvar': 0.60},
        ...     1: {'ece_score': 0.89, 'margin_tail_cvar': 0.95},
        ...     2: {'ece_score': 0.50, 'margin_tail_cvar': 0.92}
        ... }
        >>> # With epsilon=0.05, layers 0 and 1 within epsilon on ece_score
        >>> # layers 1 and 2 within epsilon on margin_tail_cvar
        >>> epsilon_ball_selection(layer_metrics, ['ece_score', 'margin_tail_cvar'], epsilon=0.05, mode='union')
        [0, 1, 2]  # Union: all layers qualify on at least one metric
        >>> epsilon_ball_selection(layer_metrics, ['ece_score', 'margin_tail_cvar'], epsilon=0.05, mode='intersection')
        [1]  # Intersection: only layer 1 qualifies on both metrics
    """

    if not metrics:
        raise ValueError("Must provide at least one metric")

    if not layer_metrics:
        return []

    METRIC_REGISTRY.validate_metrics(metrics)

    metric_selections: Dict[str, Set[int]] = {}

    for metric in metrics:
        direction = METRIC_REGISTRY.get_direction(metric)

        # Find best value
        values: List[Tuple[int, float]] = []
        for layer, vals in layer_metrics.items():
            val = _handle_nan(vals.get(metric, np.nan), direction)
            values.append((layer, val))

        if direction == "maximize":
            best_value = max(val for _, val in values)
            selected = {layer for layer, val in values if val >= best_value - epsilon}
        else:  # minimize
            best_value = min(val for _, val in values)
            selected = {layer for layer, val in values if val <= best_value + epsilon}

        metric_selections[metric] = selected
        logger.debug(
            f"  {metric}: {len(selected)} layers within epsilon of best ({best_value:.6f})"
        )

    # Combine selections
    selections = list(metric_selections.values())
    assert len(selections) > 0, "Expected non-empty metric selections"

    if mode == "union":
        result_set = set().union(*selections)
    elif mode == "intersection":
        result_set = set.intersection(*selections)
    else:
        raise ValueError("mode must be 'union' or 'intersection'")

    result = sorted(result_set)
    logger.info(f"Epsilon-ball ({mode}): {len(result)} layers selected")
    return result


def single_metric_selection(
    layer_metrics: Dict[int, Dict[str, float]],
    metric: str,
    k: int = 5,
) -> List[int]:
    """
    Select top-k layers according to a single metric.

    This is the baseline/comparison method.

    Args:
        layer_metrics: Mapping from layer_idx to metric values
        metric: Single metric name
        k: Number of layers to select

    Returns:
        List of top-k layer indices
    """

    if not layer_metrics:
        return []

    METRIC_REGISTRY.validate_metrics([metric])
    direction = METRIC_REGISTRY.get_direction(metric)

    # Get valid pairs
    valid_pairs: List[Tuple[int, float]] = []
    for layer, values in layer_metrics.items():
        val = _handle_nan(values.get(metric, np.nan), direction)
        valid_pairs.append((layer, val))

    # Sort and take top-k
    reverse = direction == "maximize"
    sorted_layers = sorted(valid_pairs, key=lambda x: x[1], reverse=reverse)

    result = [layer for layer, _ in sorted_layers[:k]]
    logger.info(f"Single-metric ({metric}): selected {len(result)} layers")
    return result


# ============================================================================
# TESTING (pytest-compatible simple tests)
# ============================================================================


def test_pareto_selection_basic():
    """Basic Pareto selection test with simple dominance relations."""

    layer_metrics = {
        0: {"ece_score": 0.8, "margin_tail_cvar": 0.6},  # On front
        1: {"ece_score": 0.9, "margin_tail_cvar": 0.5},  # Dominated by 0 (worse on margin)
        2: {"ece_score": 0.7, "margin_tail_cvar": 0.9},  # On front
    }

    result = pareto_front_selection(layer_metrics, ["ece_score", "margin_tail_cvar"])
    assert result == [0, 2]


def test_pareto_selection_edge_cases():
    """Edge cases: empty input and identical layers."""

    assert pareto_front_selection({}, ["ece_score"]) == []

    layer_metrics = {
        0: {"ece_score": 0.5},
        1: {"ece_score": 0.5},
    }
    result = pareto_front_selection(layer_metrics, ["ece_score"])
    # Both should be non-dominated when identical
    assert result == [0, 1]


def test_voting_selection_ties_and_order():
    """Voting should handle ties and return sorted by votes."""

    layer_metrics = {
        0: {"ece_score": 0.9, "margin_tail_cvar": 0.5},  # Best on ece
        1: {"ece_score": 0.7, "margin_tail_cvar": 0.9},  # Best on margin
        2: {"ece_score": 0.8, "margin_tail_cvar": 0.7},  # Middle on both
    }

    result = metric_voting_selection(layer_metrics, ["ece_score", "margin_tail_cvar"], k_per_metric=2)
    # All layers should receive votes; ordering may vary if tie, but counts should be non-zero
    assert set(result) == {0, 1, 2}


def test_epsilon_ball_union_and_intersection():
    """Epsilon-ball selection 'union' and 'intersection' modes."""

    layer_metrics = {
        0: {"ece_score": 0.90, "margin_tail_cvar": 0.60},
        1: {"ece_score": 0.89, "margin_tail_cvar": 0.95},
        2: {"ece_score": 0.50, "margin_tail_cvar": 0.92},
    }

    res_union = epsilon_ball_selection(layer_metrics, ["ece_score", "margin_tail_cvar"], epsilon=0.05, mode="union")
    res_inter = epsilon_ball_selection(
        layer_metrics, ["ece_score", "margin_tail_cvar"], epsilon=0.05, mode="intersection"
    )

    assert res_union == [0, 1, 2]
    assert res_inter == [1]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simple smoke run of functions
    lm = {
        0: {"ece_score": 0.8, "margin_tail_cvar": 0.6},
        1: {"ece_score": 0.9, "margin_tail_cvar": 0.5},
        2: {"ece_score": 0.7, "margin_tail_cvar": 0.9},
    }

    print("Pareto:", pareto_front_selection(lm, ["ece_score", "margin_tail_cvar"]))
    print("Voting:", metric_voting_selection(lm, ["ece_score", "margin_tail_cvar"], k_per_metric=2))
    print(
        "Epsilon union:",
        epsilon_ball_selection(lm, ["ece_score", "margin_tail_cvar"], epsilon=0.05, mode="union"),
    )
    print(
        "Epsilon intersection:",
        epsilon_ball_selection(lm, ["ece_score", "margin_tail_cvar"], epsilon=0.05, mode="intersection"),
    )



