"""
BT Chooser experiments module.

This module contains data structures and utilities for BT chooser experiments,
including data loading and processing for experiment runs.
"""

from .data_io import RunRecord, load_runs, available_metrics, count_pairs
from .features import (
    DIRECTION_OVERRIDES, 
    canonicalize_metrics, 
    make_layer_features, 
    add_within_run_ranks,
    extract_features_for_modeling
)
from .pareto import pareto_front, filter_pareto
from .pairs import build_pairs
from .bt_model import BradleyTerryChooser
from .infer import prepare_layer_matrix, choose_layer
from .rank_agg import borda_from_metrics
from .bt_integration import extend_two_stage_selector_with_bt, create_bt_wrapper_selector

__all__ = [
    "RunRecord", "load_runs", "available_metrics", "count_pairs",
    "DIRECTION_OVERRIDES", "canonicalize_metrics", "make_layer_features", 
    "add_within_run_ranks", "extract_features_for_modeling",
    "pareto_front", "filter_pareto", "build_pairs", "BradleyTerryChooser",
    "prepare_layer_matrix", "choose_layer", "borda_from_metrics",
    "extend_two_stage_selector_with_bt", "create_bt_wrapper_selector"
]
