#!/usr/bin/env python3
"""
Generate SGC-focused LaTeX tables for the paper.
Updated with Statistical Power Analysis (Required Sample Size estimation).

This script processes calibration results and generates three main tables:
1. Main Comparison Table: Uncal, TS, Isotonic, Pixel Geo, SGC(DAC), SGC(TULIP), SGC
2. Hyperparameter Sensitivity Table: SGC performance across (L, d) configurations
3. ECE Reduction Table: Improvement over uncalibrated baseline

The script also performs statistical significance testing and power analysis:
- Performs one-sided t-tests comparing SGC vs best baseline
- Calculates Cohen's d effect size
- Estimates required sample size for significance (p < 0.05, power = 0.8)
- Generates significance_detailed_report.csv with detailed statistics
"""

import json
import logging
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import sys
# Add project root to path (before importing utils)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_config import get_logger

logger = get_logger(__name__)

# Global list to store detailed significance analysis
SIGNIFICANCE_LOG = []

# TULIP published results (ECE in decimal form)
# From TULIP paper Table 8
# Note: Original paper reports 0.658±0.004 for CIFAR-100, but this appears to be a decimal
# error (should be 0.0658). We've corrected to match their CIFAR-10 scale.
# Note: format_ece_value multiplies by 100, so input 0.030 to display "3.00%"
TULIP_PUBLISHED = {
    ('cifar10', 'baseline_cross_entropy', 'resnet50'): (0.030, 0.005),
    ('cifar10', 'baseline_cross_entropy', 'resnet101'): (0.025, 0.021),
    ('cifar10', 'baseline_cross_entropy', 'resnet152'): (0.026, 0.011),
    # CIFAR-100 values corrected: divided by 10 to fix decimal point error in paper
    # Original paper: 0.658±0.004 (decimal) -> Corrected: 0.0658±0.0004 (decimal)
    # format_ece_value multiplies by 100, so 0.0658 -> displays as 6.58%
    ('cifar100', 'baseline_cross_entropy', 'resnet50'): (0.0658, 0.0004),
    ('cifar100', 'baseline_cross_entropy', 'resnet101'): (0.0671, 0.0021),
    ('cifar100', 'baseline_cross_entropy', 'resnet152'): (0.0662, 0.0055),
}


def load_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON files from results directory."""
    result_files = sorted(results_dir.glob('ablation_*.json'))
    
    if not result_files:
        logger.warning(f"No ablation JSON files found in {results_dir}")
        return []
    
    results = []
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to load {result_file.name}: {e}")
            continue
    
    logger.info(f"Loaded {len(results)} result files")
    return results


def load_coordinate_results(coord_dir: Path, k_value: int = 256) -> pd.DataFrame:
    """
    Load coordinate ablation results and aggregate by (dataset, model, training_method).
    
    Args:
        coord_dir: Directory containing coordinate_ablation_*.json files
        k_value: Number of coordinates to use (default 256 for comparison with L=6 d=256)
    
    Returns:
        DataFrame with columns: dataset, model_name, training_method, coord_ece_mean, coord_ece_std, coord_ece_count, coord_ece_values, coord_acc_mean, coord_acc_std, coord_acc_count
    """
    results = []
    
    for strategy in ["nested", "independent"]:
        pattern = str(coord_dir / f"coordinate_ablation_*_{strategy}.json")
        for fpath in glob.glob(pattern):
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {fpath}: {e}")
                continue
            
            config = data.get("experiment_config", {})
            dataset = config.get("dataset")
            model_name = config.get("model_name")
            training_method = config.get("training_method")
            
            if not all([dataset, model_name, training_method]):
                continue
            
            results_by_k = data.get("results_by_num_coordinates", {})
            k_str = str(k_value)
            
            if k_str in results_by_k:
                for trial in results_by_k[k_str]:
                    ece_geo = trial.get("ece_geometric")
                    acc_geo = trial.get("accuracy_geometric")
                    # DAC on coordinate features (if present)
                    ece_dac = trial.get("ece_dac")
                    acc_dac = trial.get("accuracy_dac")

                    if ece_geo is not None:
                        results.append({
                            "dataset": dataset,
                            "model_name": model_name,
                            "training_method": training_method,
                            "coord_ece": ece_geo,
                            "coord_acc": acc_geo,
                            "coord_dac_ece": ece_dac,
                            "coord_dac_acc": acc_dac,
                        })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Aggregate by (dataset, model_name, training_method)
    agg = df.groupby(["dataset", "model_name", "training_method"]).agg(
        coord_ece_mean=("coord_ece", "mean"),
        coord_ece_std=("coord_ece", "std"),
        coord_ece_count=("coord_ece", "count"),
        coord_ece_values=("coord_ece", list),
        coord_acc_mean=("coord_acc", "mean"),
        coord_acc_std=("coord_acc", "std"),
        coord_acc_count=("coord_acc", "count"),
        coord_dac_ece_mean=("coord_dac_ece", "mean"),
        coord_dac_ece_std=("coord_dac_ece", "std"),
        coord_dac_ece_count=("coord_dac_ece", "count"),
        coord_dac_ece_values=("coord_dac_ece", list),
        coord_dac_acc_mean=("coord_dac_acc", "mean"),
        coord_dac_acc_std=("coord_dac_acc", "std"),
        coord_dac_acc_count=("coord_dac_acc", "count"),
    ).reset_index()
    
    return agg


def extract_experiment_key(result: Dict[str, Any]) -> Tuple[str, str, str]:
    """Extract (training_method, dataset, model_name) key from result."""
    exp_info = result.get('experiment_info', result.get('experiment_config', {}))
    training_method = exp_info.get('training_method', 'unknown')
    dataset = exp_info.get('dataset', 'unknown')
    model_name = exp_info.get('model_name', 'unknown')
    return (training_method, dataset, model_name)


def is_clean_result(result: Dict[str, Any]) -> bool:
    """Check if result is from clean (non-corrupted) dataset."""
    exp_info = result.get('experiment_info', result.get('experiment_config', {}))
    corruption_type = exp_info.get('corruption_type')
    corruption_severity = exp_info.get('corruption_severity')
    return corruption_type is None and corruption_severity is None


def extract_baseline_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract baseline calibration metrics from standard_baselines section."""
    # Try new structure first (standard_baselines), fallback to old (baselines)
    # Be robust to cases where these keys exist but are explicitly set to null/None.
    baselines = result.get('standard_baselines')
    if baselines is None:
        baselines = result.get('baselines') or {}

    metrics = {}
    
    # Uncalibrated
    if 'uncalibrated' in baselines:
        metrics['uncal_ece'] = baselines['uncalibrated'].get('ece', np.nan)
        metrics['uncal_acc'] = baselines['uncalibrated'].get('acc', np.nan)
    
    # Temperature Scaling
    if 'temperature_scaling' in baselines:
        metrics['ts_ece'] = baselines['temperature_scaling'].get('ece', np.nan)
        metrics['ts_acc'] = baselines['temperature_scaling'].get('acc', np.nan)
    
    # Platt Scaling
    if 'platt_scaling' in baselines:
        metrics['platt_ece'] = baselines['platt_scaling'].get('ece', np.nan)
        metrics['platt_acc'] = baselines['platt_scaling'].get('acc', np.nan)
    
    # Isotonic Regression
    if 'isotonic_toplabel' in baselines:
        metrics['isotonic_ece'] = baselines['isotonic_toplabel'].get('ece', np.nan)
        metrics['isotonic_acc'] = baselines['isotonic_toplabel'].get('acc', np.nan)
    
    # Pixel Geo Space
    if 'geometric_physical_space' in baselines:
        metrics['pixel_geo_ece'] = baselines['geometric_physical_space'].get('ece', np.nan)
        metrics['pixel_geo_acc'] = baselines['geometric_physical_space'].get('acc', np.nan)
    
    # Beta Calibration
    if 'beta_calibration' in baselines:
        metrics['beta_ece'] = baselines['beta_calibration'].get('ece', np.nan)
        metrics['beta_acc'] = baselines['beta_calibration'].get('acc', np.nan)
    
    # Density Aware Calibration (actual algorithm, not layer selection)
    if 'density_aware_calibration' in baselines:
        metrics['dac_algorithm_ece'] = baselines['density_aware_calibration'].get('ece', np.nan)
        metrics['dac_algorithm_acc'] = baselines['density_aware_calibration'].get('acc', np.nan)
    
    return metrics


def extract_dac_fixed(result: Dict[str, Any], target_dim: int = 256) -> Optional[float]:
    """Extract DAC Fixed (geo_comb) ECE for specific target_dim."""
    geo_comb = result.get('geo_comb', {})
    results = geo_comb.get('results', [])
    
    # Find best ECE among entries with target_dim
    matching_results = [r for r in results if r.get('target_dim') == target_dim]
    
    if not matching_results:
        return None
    
    # Return best (minimum) ECE
    eces = [r.get('ece') for r in matching_results if 'ece' in r]
    if not eces:
        return None
    
    return min(eces)


def extract_dac_fixed_acc(result: Dict[str, Any], target_dim: int = 256) -> Optional[float]:
    """Extract DAC Fixed (geo_comb) accuracy for specific target_dim."""
    geo_comb = result.get('geo_comb', {})
    results = geo_comb.get('results', [])
    
    # Find best ECE entry, then get its accuracy
    matching_results = [r for r in results if r.get('target_dim') == target_dim]
    
    if not matching_results:
        return None
    
    # Find entry with best (minimum) ECE
    best_result = min(matching_results, key=lambda x: x.get('ece', float('inf')))
    
    # Accuracy is already in percentage (e.g., 93.54)
    acc = best_result.get('accuracy')
    if acc is not None:
        return acc / 100.0  # Convert to [0,1] for consistency
    
    return None


def extract_tulip_fixed(result: Dict[str, Any], target_dim: int = 256) -> Optional[float]:
    """
    Extract TULIP-based ECE for specific target_dim.
    
    Priority:
      1) Original TULIP combined experiment (`tulip_comb.results`).
      2) New SGC-with-TULIP-layers experiment (`sgc_with_tulip_layers_separation`).
    """
    # --- 1) Original tulip_comb structure (if present) ---
    tulip_comb = result.get('tulip_comb', {})
    results = tulip_comb.get('results', [])
    
    # Find best ECE among entries with target_dim
    matching_results = [r for r in results if r.get('target_dim') == target_dim]
    if matching_results:
        eces = [r.get('ece') for r in matching_results if 'ece' in r]
        if eces:
            return min(eces)
    
    # --- 2) Fallback: SGC with TULIP-selected layers (separation variant) ---
    tulip_sgc_sep = result.get('sgc_with_tulip_layers_separation')
    if isinstance(tulip_sgc_sep, dict):
        if tulip_sgc_sep.get('target_dim') == target_dim:
            ece = tulip_sgc_sep.get('ece')
            if ece is not None:
                return ece
    
    return None


def extract_tulip_fixed_acc(result: Dict[str, Any], target_dim: int = 256) -> Optional[float]:
    """
    Extract TULIP-based accuracy for specific target_dim.
    
    Priority:
      1) Original TULIP combined experiment (`tulip_comb.results`).
      2) New SGC-with-TULIP-layers experiment (`sgc_with_tulip_layers_separation`).
    """
    # --- 1) Original tulip_comb structure (if present) ---
    tulip_comb = result.get('tulip_comb', {})
    results = tulip_comb.get('results', [])
    
    matching_results = [r for r in results if r.get('target_dim') == target_dim]
    if matching_results:
        best_result = min(matching_results, key=lambda x: x.get('ece', float('inf')))
        acc = best_result.get('accuracy')
        if acc is not None:
            return acc / 100.0  # Convert to [0,1] for consistency
    
    # --- 2) Fallback: SGC with TULIP-selected layers (separation variant) ---
    tulip_sgc_sep = result.get('sgc_with_tulip_layers_separation')
    if isinstance(tulip_sgc_sep, dict):
        if tulip_sgc_sep.get('target_dim') == target_dim:
            acc = tulip_sgc_sep.get('accuracy')
            if acc is not None:
                # In ablation JSONs this is already a percentage (e.g., 94.01)
                return acc / 100.0
    
    return None


def extract_oracle_ece(result: Dict[str, Any], target_dim: int = 256) -> Optional[float]:
    """Extract Oracle (geo_best) ECE for specific target_dim."""
    geo_best = result.get('geo_best', {})
    
    # Check best_per_target_dim structure
    best_per_target_dim = geo_best.get('best_per_target_dim', {})
    target_dim_str = str(target_dim)
    
    if target_dim_str in best_per_target_dim:
        best_entry = best_per_target_dim[target_dim_str]
        if 'ece' in best_entry:
            return best_entry['ece']
    
    # Fallback: check all_single_layers
    all_single_layers = geo_best.get('all_single_layers', [])
    matching = [r for r in all_single_layers if r.get('target_dim') == target_dim]
    
    if matching:
        eces = [r.get('ece') for r in matching if 'ece' in r]
        if eces:
            return min(eces)
    
    return None


def extract_sgc_result(result: Dict[str, Any], L: int, d: int) -> Optional[float]:
    """Extract SGC ECE for specific (L, d) configuration.
    Tries new structure (global_random_separation, global_random_trust_score) first, falls back to old (random_layer_ablation).
    """
    # Try new structure first: global_random_separation (preferred) or global_random_trust_score
    for key in ['global_random_separation', 'global_random_trust_score', 'global_random']:
        global_random = result.get(key, {})
        if global_random and isinstance(global_random, dict) and not global_random.get('skipped', False):
            # Check if it matches our (L, d) requirements
            num_layers = global_random.get('num_layers')
            target_dim = global_random.get('target_dim')
            if num_layers == L and target_dim == d:
                ece = global_random.get('ece')
                if ece is not None:
                    return ece
    
    # Fallback to old structure: random_layer_ablation
    random_ablation = result.get('random_layer_ablation', {})
    
    if 'results_by_target_dim' not in random_ablation:
        return None
    
    results_by_target_dim = random_ablation['results_by_target_dim']
    target_dim_str = str(d)
    
    if target_dim_str not in results_by_target_dim:
        return None
    
    dim_data = results_by_target_dim[target_dim_str]
    global_random = dim_data.get('global_random', [])
    
    # Find entries with num_layers == L
    matching_entries = [r for r in global_random if r.get('num_layers') == L]
    
    if not matching_entries:
        return None
    
    # Return mean ECE across all matching entries (aggregate over trial_seeds)
    eces = [r.get('ece') for r in matching_entries if 'ece' in r]
    if not eces:
        return None
    
    return np.mean(eces)


def extract_sgc_acc(result: Dict[str, Any], L: int, d: int) -> Optional[float]:
    """Extract SGC accuracy for specific (L, d) configuration.
    Tries new structure (global_random_separation, global_random_trust_score) first, falls back to old (random_layer_ablation).
    """
    # Try new structure first: global_random_separation (preferred) or global_random_trust_score
    for key in ['global_random_separation', 'global_random_trust_score', 'global_random']:
        global_random = result.get(key, {})
        if global_random and isinstance(global_random, dict) and not global_random.get('skipped', False):
            # Check if it matches our (L, d) requirements
            num_layers = global_random.get('num_layers')
            target_dim = global_random.get('target_dim')
            if num_layers == L and target_dim == d:
                acc = global_random.get('accuracy')
                if acc is not None:
                    # Accuracy is in percentage, convert to [0,1]
                    return acc / 100.0
    
    # Fallback to old structure: random_layer_ablation
    random_ablation = result.get('random_layer_ablation', {})
    
    if 'results_by_target_dim' not in random_ablation:
        return None
    
    results_by_target_dim = random_ablation['results_by_target_dim']
    target_dim_str = str(d)
    
    if target_dim_str not in results_by_target_dim:
        return None
    
    dim_data = results_by_target_dim[target_dim_str]
    global_random = dim_data.get('global_random', [])
    
    # Find entries with num_layers == L
    matching_entries = [r for r in global_random if r.get('num_layers') == L]
    
    if not matching_entries:
        return None
    
    # Return mean accuracy across all matching entries (aggregate over trial_seeds)
    # Accuracy is already in percentage, convert to [0,1]
    accs = [r.get('accuracy') for r in matching_entries if 'accuracy' in r]
    if not accs:
        return None
    
    # Convert from percentage to [0,1] and return mean
    return np.mean([acc / 100.0 for acc in accs])


def extract_sgc_faiss_ece(result: Dict[str, Any]) -> Optional[float]:
    """Extract SGC FAISS ECE from sgc_faiss section."""
    sgc_faiss = result.get('sgc_faiss', {})
    if sgc_faiss and isinstance(sgc_faiss, dict) and not sgc_faiss.get('skipped', False):
        ece = sgc_faiss.get('ece')
        if ece is not None:
            return ece
    return None


def extract_sgc_faiss_acc(result: Dict[str, Any]) -> Optional[float]:
    """Extract SGC FAISS accuracy from sgc_faiss section."""
    sgc_faiss = result.get('sgc_faiss', {})
    if sgc_faiss and isinstance(sgc_faiss, dict) and not sgc_faiss.get('skipped', False):
        acc = sgc_faiss.get('accuracy')
        if acc is not None:
            # Accuracy is in percentage, convert to [0,1]
            return acc / 100.0
    return None


def extract_sgc_faiss_brier(result: Dict[str, Any]) -> Optional[float]:
    """Extract SGC FAISS Brier score from sgc_faiss section."""
    sgc_faiss = result.get('sgc_faiss', {})
    if sgc_faiss and isinstance(sgc_faiss, dict) and not sgc_faiss.get('skipped', False):
        brier = sgc_faiss.get('brier')
        if brier is not None:
            return brier
    return None


def extract_sgc_faiss_ood_metrics(result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract SGC FAISS OOD metrics from sgc_faiss section."""
    sgc_faiss = result.get('sgc_faiss', {})
    if sgc_faiss and isinstance(sgc_faiss, dict) and not sgc_faiss.get('skipped', False):
        return {
            'ood_auroc': sgc_faiss.get('ood_auroc'),
            'ood_fpr95': sgc_faiss.get('ood_fpr95'),
            'ood_ece': sgc_faiss.get('ood_ece'),
            'confidence_gap': sgc_faiss.get('confidence_gap')
        }
    return {'ood_auroc': None, 'ood_fpr95': None, 'ood_ece': None, 'confidence_gap': None}


def extract_coordinate_sampling_ece(result: Dict[str, Any]) -> Optional[float]:
    """Extract coordinate sampling ECE from coordinate_sampling section."""
    # Try separation first (preferred), then trust_score, then base key
    for key in ['coordinate_sampling_separation', 'coordinate_sampling_trust_score', 'coordinate_sampling']:
        coord_sampling = result.get(key, {})
        if coord_sampling and isinstance(coord_sampling, dict) and not coord_sampling.get('skipped', False):
            ece = coord_sampling.get('ece')
            if ece is not None:
                return ece
    return None


def extract_coordinate_sampling_acc(result: Dict[str, Any]) -> Optional[float]:
    """Extract coordinate sampling accuracy from coordinate_sampling section."""
    coord_sampling = result.get('coordinate_sampling', {})
    if coord_sampling and isinstance(coord_sampling, dict):
        acc = coord_sampling.get('accuracy')
        if acc is not None:
            # Accuracy is in percentage, convert to [0,1]
            return acc / 100.0
    return None


def extract_geometric_dac_weighted_ece(result: Dict[str, Any]) -> Optional[float]:
    """Extract Geometric + DAC Weighting ECE from geometric_dac_weighted section."""
    geo_dac = result.get('geometric_dac_weighted', {})
    if geo_dac and isinstance(geo_dac, dict):
        ece = geo_dac.get('ece')
        if ece is not None:
            return ece
    return None


def extract_geometric_dac_weighted_acc(result: Dict[str, Any]) -> Optional[float]:
    """Extract Geometric + DAC Weighting accuracy from geometric_dac_weighted section."""
    geo_dac = result.get('geometric_dac_weighted', {})
    if geo_dac and isinstance(geo_dac, dict):
        acc = geo_dac.get('accuracy')
        if acc is not None:
            # Accuracy is in percentage, convert to [0,1]
            return acc / 100.0
    return None


def extract_dac_with_random_layers_ece(result: Dict[str, Any]) -> Optional[float]:
    """Extract DAC + Random Layers ECE from ablation.dac_with_random_layer_selection."""
    ablation = result.get('ablation', {})
    if ablation and isinstance(ablation, dict):
        dac_random = ablation.get('dac_with_random_layer_selection', {})
        if dac_random and isinstance(dac_random, dict):
            ece = dac_random.get('ece')
            if ece is not None:
                return ece
    return None


def extract_dac_with_random_layers_acc(result: Dict[str, Any]) -> Optional[float]:
    """Extract DAC + Random Layers accuracy from ablation.dac_with_random_layer_selection."""
    ablation = result.get('ablation', {})
    if ablation and isinstance(ablation, dict):
        dac_random = ablation.get('dac_with_random_layer_selection', {})
        if dac_random and isinstance(dac_random, dict):
            acc = dac_random.get('accuracy')
            if acc is not None:
                # Accuracy is in percentage, convert to [0,1]
                return acc / 100.0
    return None


def extract_dac_with_random_layers_ood_metrics(result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract DAC with random layer selection OOD metrics."""
    ablation = result.get('ablation', {})
    if ablation and isinstance(ablation, dict):
        dac_random = ablation.get('dac_with_random_layer_selection', {})
        if dac_random and isinstance(dac_random, dict) and not dac_random.get('skipped', False):
            return {
                'ood_auroc': dac_random.get('ood_auroc'),
                'ood_fpr95': dac_random.get('ood_fpr95'),
                'ood_ece': dac_random.get('ood_ece'),
                'confidence_gap': dac_random.get('confidence_gap'),
            }
    return {'ood_auroc': None, 'ood_fpr95': None, 'ood_ece': None, 'confidence_gap': None}


def extract_sgc_with_dac_preprocessing_ece(result: Dict[str, Any]) -> Optional[float]:
    """Extract SGC with DAC Preprocessing ECE from ablation.sgc_with_dac_preprocessing."""
    ablation = result.get('ablation', {})
    if ablation and isinstance(ablation, dict):
        sgc_dac = ablation.get('sgc_with_dac_preprocessing', {})
        if sgc_dac and isinstance(sgc_dac, dict):
            ece = sgc_dac.get('ece')
            if ece is not None:
                return ece
    return None


def extract_sgc_with_dac_preprocessing_acc(result: Dict[str, Any]) -> Optional[float]:
    """Extract SGC with DAC Preprocessing accuracy from ablation.sgc_with_dac_preprocessing."""
    ablation = result.get('ablation', {})
    if ablation and isinstance(ablation, dict):
        sgc_dac = ablation.get('sgc_with_dac_preprocessing', {})
        if sgc_dac and isinstance(sgc_dac, dict):
            acc = sgc_dac.get('accuracy')
            if acc is not None:
                # Accuracy is in percentage, convert to [0,1]
                return acc / 100.0
    return None


def extract_sgc_with_dac_layers_ece(result: Dict[str, Any]) -> Optional[float]:
    """Extract SGC with DAC Layer Selection ECE from sgc_with_dac_layers section."""
    # Try separation first (preferred), then trust_score, then base key
    for key in ['sgc_with_dac_layers_separation', 'sgc_with_dac_layers_trust_score', 'sgc_with_dac_layers']:
        sgc_dac_layers = result.get(key, {})
        if sgc_dac_layers and isinstance(sgc_dac_layers, dict):
            # Skip if it was skipped
            if sgc_dac_layers.get('skipped', False):
                continue
            ece = sgc_dac_layers.get('ece')
            if ece is not None:
                return ece
    return None


def extract_sgc_with_dac_layers_acc(result: Dict[str, Any]) -> Optional[float]:
    """Extract SGC with DAC Layer Selection accuracy from sgc_with_dac_layers section."""
    # Try separation first (preferred), then trust_score, then base key
    for key in ['sgc_with_dac_layers_separation', 'sgc_with_dac_layers_trust_score', 'sgc_with_dac_layers']:
        sgc_dac_layers = result.get(key, {})
        if sgc_dac_layers and isinstance(sgc_dac_layers, dict):
            # Skip if it was skipped
            if sgc_dac_layers.get('skipped', False):
                continue
            acc = sgc_dac_layers.get('accuracy')
            if acc is not None:
                # Accuracy is in percentage, convert to [0,1]
                return acc / 100.0
    return None


def extract_dac_with_coordinate_features_ece(result: Dict[str, Any]) -> Optional[float]:
    """Extract DAC with Coordinate Features ECE from dac_with_coordinate_features section."""
    dac_coord = result.get('dac_with_coordinate_features', {})
    if dac_coord and isinstance(dac_coord, dict):
        # Skip if it was skipped
        if dac_coord.get('skipped', False):
            return None
        ece = dac_coord.get('ece')
        if ece is not None:
            return ece
    return None


def extract_dac_with_coordinate_features_acc(result: Dict[str, Any]) -> Optional[float]:
    """Extract DAC with Coordinate Features accuracy from dac_with_coordinate_features section."""
    dac_coord = result.get('dac_with_coordinate_features', {})
    if dac_coord and isinstance(dac_coord, dict):
        # Skip if it was skipped
        if dac_coord.get('skipped', False):
            return None
        acc = dac_coord.get('accuracy')
        if acc is not None:
            # Accuracy is in percentage, convert to [0,1]
            return acc / 100.0
    return None


def extract_baseline_brier(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract baseline Brier scores from standard_baselines section."""
    # Try new structure first (standard_baselines), fallback to old (baselines)
    # Be robust to explicit null/None values.
    baselines = result.get('standard_baselines')
    if baselines is None:
        baselines = result.get('baselines') or {}
    
    metrics = {}
    
    # Uncalibrated
    if 'uncalibrated' in baselines:
        metrics['uncal_brier'] = baselines['uncalibrated'].get('brier', np.nan)
    
    # Temperature Scaling
    if 'temperature_scaling' in baselines:
        metrics['ts_brier'] = baselines['temperature_scaling'].get('brier', np.nan)
    
    # Platt Scaling
    if 'platt_scaling' in baselines:
        metrics['platt_brier'] = baselines['platt_scaling'].get('brier', np.nan)
    
    # Isotonic Regression
    if 'isotonic_toplabel' in baselines:
        metrics['isotonic_brier'] = baselines['isotonic_toplabel'].get('brier', np.nan)
    
    # Pixel Geo Space
    if 'geometric_physical_space' in baselines:
        metrics['pixel_geo_brier'] = baselines['geometric_physical_space'].get('brier', np.nan)
    
    # Beta Calibration
    if 'beta_calibration' in baselines:
        metrics['beta_brier'] = baselines['beta_calibration'].get('brier', np.nan)
    
    # Density Aware Calibration
    if 'density_aware_calibration' in baselines:
        metrics['dac_algorithm_brier'] = baselines['density_aware_calibration'].get('brier', np.nan)
    
    return metrics


def extract_sgc_brier(result: Dict[str, Any], L: int, d: int) -> Optional[float]:
    """Extract SGC Brier score for specific L and d."""
    # Try new structure first: global_random_separation (preferred) or global_random_trust_score
    for key in ['global_random_separation', 'global_random_trust_score', 'global_random']:
        global_random = result.get(key, {})
        if global_random and isinstance(global_random, dict) and not global_random.get('skipped', False):
            num_layers = global_random.get('num_layers')
            target_dim = global_random.get('target_dim')
            if num_layers == L and target_dim == d:
                brier = global_random.get('brier')
                if brier is not None:
                    return brier
    
    # Fallback to old structure with results list
    global_random = result.get('global_random', {})
    if global_random and isinstance(global_random, dict):
        results = global_random.get('results', [])
        matching = [r for r in results if r.get('num_layers') == L and r.get('target_dim') == d]
        
        if not matching:
            return None
        
        # Return best (minimum ECE) entry's Brier score
        best_result = min(matching, key=lambda x: x.get('ece', float('inf')))
        return best_result.get('brier')
    
    return None


def extract_dac_fixed_brier(result: Dict[str, Any], target_dim: int = 256) -> Optional[float]:
    """Extract DAC Fixed (geo_comb) Brier score for specific target_dim."""
    geo_comb = result.get('geo_comb', {})
    results = geo_comb.get('results', [])
    
    matching_results = [r for r in results if r.get('target_dim') == target_dim]
    if not matching_results:
        return None
    
    best_result = min(matching_results, key=lambda x: x.get('ece', float('inf')))
    return best_result.get('brier')


def extract_tulip_fixed_brier(result: Dict[str, Any], target_dim: int = 256) -> Optional[float]:
    """
    Extract TULIP-based Brier score for specific target_dim.
    
    Priority:
      1) Original TULIP combined experiment (`tulip_comb.results`).
      2) New SGC-with-TULIP-layers experiment (`sgc_with_tulip_layers_separation`).
    """
    # --- 1) Original tulip_comb structure (if present) ---
    tulip_comb = result.get('tulip_comb', {})
    results = tulip_comb.get('results', [])
    
    matching_results = [r for r in results if r.get('target_dim') == target_dim]
    if matching_results:
        best_result = min(matching_results, key=lambda x: x.get('ece', float('inf')))
        brier = best_result.get('brier')
        if brier is not None:
            return brier
    
    # --- 2) Fallback: SGC with TULIP-selected layers (separation variant) ---
    tulip_sgc_sep = result.get('sgc_with_tulip_layers_separation')
    if isinstance(tulip_sgc_sep, dict):
        if tulip_sgc_sep.get('target_dim') == target_dim:
            brier = tulip_sgc_sep.get('brier')
            if brier is not None:
                return brier
    
    return None


def extract_coordinate_sampling_brier(result: Dict[str, Any]) -> Optional[float]:
    """Extract Coordinate Sampling Brier score."""
    coord = result.get('coordinate_sampling', {})
    if coord and isinstance(coord, dict) and not coord.get('skipped', False):
        return coord.get('brier')
    return None


def extract_sgc_with_dac_layers_brier(result: Dict[str, Any]) -> Optional[float]:
    """Extract SGC with DAC layers Brier score."""
    # Try separation first (preferred), then trust_score, then base key
    for key in ['sgc_with_dac_layers_separation', 'sgc_with_dac_layers_trust_score', 'sgc_with_dac_layers']:
        sgc_dac = result.get(key, {})
        if sgc_dac and isinstance(sgc_dac, dict) and not sgc_dac.get('skipped', False):
            brier = sgc_dac.get('brier')
            if brier is not None:
                return brier
    return None


def extract_baseline_ood_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract baseline OOD metrics from standard_baselines section."""
    # Try new structure first (standard_baselines), fallback to old (baselines)
    # Be robust to explicit null/None values.
    baselines = result.get('standard_baselines')
    if baselines is None:
        baselines = result.get('baselines') or {}
    
    metrics = {}
    
    # Helper to extract OOD metrics for a method
    def extract_ood_for_method(method_key: str, prefix: str):
        method_data = baselines.get(method_key, {})
        if method_data:
            metrics[f'{prefix}_ood_auroc'] = method_data.get('ood_auroc', np.nan)
            metrics[f'{prefix}_ood_fpr95'] = method_data.get('ood_fpr95', np.nan)
            metrics[f'{prefix}_ood_ece'] = method_data.get('ood_ece', np.nan)
            metrics[f'{prefix}_confidence_gap'] = method_data.get('confidence_gap', np.nan)
    
    extract_ood_for_method('uncalibrated', 'uncal')
    extract_ood_for_method('temperature_scaling', 'ts')
    extract_ood_for_method('platt_scaling', 'platt')
    extract_ood_for_method('isotonic_toplabel', 'isotonic')
    extract_ood_for_method('geometric_physical_space', 'pixel_geo')
    extract_ood_for_method('beta_calibration', 'beta')
    extract_ood_for_method('density_aware_calibration', 'dac_algorithm')
    
    return metrics


def extract_sgc_ood_metrics(result: Dict[str, Any], L: int, d: int) -> Dict[str, Optional[float]]:
    """Extract SGC OOD metrics for specific L and d."""
    # Try new structure first: global_random_separation (preferred) or global_random_trust_score
    for key in ['global_random_separation', 'global_random_trust_score', 'global_random']:
        global_random = result.get(key, {})
        if global_random and isinstance(global_random, dict) and not global_random.get('skipped', False):
            num_layers = global_random.get('num_layers')
            target_dim = global_random.get('target_dim')
            if num_layers == L and target_dim == d:
                return {
                    'ood_auroc': global_random.get('ood_auroc'),
                    'ood_fpr95': global_random.get('ood_fpr95'),
                    'ood_ece': global_random.get('ood_ece'),
                    'confidence_gap': global_random.get('confidence_gap'),
                }
    
    # Fallback to old structure with results list
    global_random = result.get('global_random', {})
    if global_random and isinstance(global_random, dict):
        results = global_random.get('results', [])
        matching = [r for r in results if r.get('num_layers') == L and r.get('target_dim') == d]
        
        if not matching:
            return {'ood_auroc': None, 'ood_fpr95': None, 'ood_ece': None, 'confidence_gap': None}
        
        best_result = min(matching, key=lambda x: x.get('ece', float('inf')))
        return {
            'ood_auroc': best_result.get('ood_auroc'),
            'ood_fpr95': best_result.get('ood_fpr95'),
            'ood_ece': best_result.get('ood_ece'),
            'confidence_gap': best_result.get('confidence_gap'),
        }
    
    return {'ood_auroc': None, 'ood_fpr95': None, 'ood_ece': None, 'confidence_gap': None}


def extract_dac_fixed_ood_metrics(result: Dict[str, Any], target_dim: int = 256) -> Dict[str, Optional[float]]:
    """Extract DAC Fixed (geo_comb) OOD metrics for specific target_dim."""
    geo_comb = result.get('geo_comb', {})
    results = geo_comb.get('results', [])
    
    matching_results = [r for r in results if r.get('target_dim') == target_dim]
    if not matching_results:
        return {'ood_auroc': None, 'ood_fpr95': None, 'ood_ece': None, 'confidence_gap': None}
    
    best_result = min(matching_results, key=lambda x: x.get('ece', float('inf')))
    return {
        'ood_auroc': best_result.get('ood_auroc'),
        'ood_fpr95': best_result.get('ood_fpr95'),
        'ood_ece': best_result.get('ood_ece'),
        'confidence_gap': best_result.get('confidence_gap'),
    }


def extract_tulip_fixed_ood_metrics(result: Dict[str, Any], target_dim: int = 256) -> Dict[str, Optional[float]]:
    """
    Extract TULIP-based OOD metrics for specific target_dim.
    
    Priority:
      1) Original TULIP combined experiment (`tulip_comb.results`).
      2) New SGC-with-TULIP-layers experiment (`sgc_with_tulip_layers_separation`).
    """
    # Default empty structure
    empty = {'ood_auroc': None, 'ood_fpr95': None, 'ood_ece': None, 'confidence_gap': None}
    
    # --- 1) Original tulip_comb structure (if present) ---
    tulip_comb = result.get('tulip_comb', {})
    results = tulip_comb.get('results', [])
    
    matching_results = [r for r in results if r.get('target_dim') == target_dim]
    if matching_results:
        best_result = min(matching_results, key=lambda x: x.get('ece', float('inf')))
        return {
            'ood_auroc': best_result.get('ood_auroc'),
            'ood_fpr95': best_result.get('ood_fpr95'),
            'ood_ece': best_result.get('ood_ece'),
            'confidence_gap': best_result.get('confidence_gap'),
        }
    
    # --- 2) Fallback: SGC with TULIP-selected layers (separation variant) ---
    tulip_sgc_sep = result.get('sgc_with_tulip_layers_separation')
    if isinstance(tulip_sgc_sep, dict):
        if tulip_sgc_sep.get('target_dim') == target_dim:
            return {
                'ood_auroc': tulip_sgc_sep.get('ood_auroc'),
                'ood_fpr95': tulip_sgc_sep.get('ood_fpr95'),
                'ood_ece': tulip_sgc_sep.get('ood_ece'),
                'confidence_gap': tulip_sgc_sep.get('ood_confidence_gap', tulip_sgc_sep.get('confidence_gap')),
            }
    
    return empty


def extract_coordinate_sampling_ood_metrics(result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract Coordinate Sampling OOD metrics."""
    # Try separation first (preferred), then trust_score, then base key
    for key in ['coordinate_sampling_separation', 'coordinate_sampling_trust_score', 'coordinate_sampling']:
        coord = result.get(key, {})
        if coord and isinstance(coord, dict) and not coord.get('skipped', False):
            return {
                'ood_auroc': coord.get('ood_auroc'),
                'ood_fpr95': coord.get('ood_fpr95'),
                'ood_ece': coord.get('ood_ece'),
                'confidence_gap': coord.get('confidence_gap'),
            }
    return {'ood_auroc': None, 'ood_fpr95': None, 'ood_ece': None, 'confidence_gap': None}


def extract_sgc_with_dac_layers_ood_metrics(result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract SGC with DAC layers OOD metrics."""
    # Try separation first (preferred), then trust_score, then base key
    for key in ['sgc_with_dac_layers_separation', 'sgc_with_dac_layers_trust_score', 'sgc_with_dac_layers']:
        sgc_dac = result.get(key, {})
        if sgc_dac and isinstance(sgc_dac, dict) and not sgc_dac.get('skipped', False):
            return {
                'ood_auroc': sgc_dac.get('ood_auroc'),
                'ood_fpr95': sgc_dac.get('ood_fpr95'),
                'ood_ece': sgc_dac.get('ood_ece'),
                'confidence_gap': sgc_dac.get('confidence_gap'),
            }
    return {'ood_auroc': None, 'ood_fpr95': None, 'ood_ece': None, 'confidence_gap': None}


# ============================================================================
# Trust Score Extraction Functions (explicit trust_score versions)
# ============================================================================

def extract_sgc_trust_score_result(result: Dict[str, Any], L: int, d: int) -> Optional[float]:
    """Extract SGC ECE using trust_score method for specific (L, d) configuration."""
    global_random = result.get('global_random_trust_score', {})
    if global_random and isinstance(global_random, dict) and not global_random.get('skipped', False):
        num_layers = global_random.get('num_layers')
        target_dim = global_random.get('target_dim')
        if num_layers == L and target_dim == d:
            ece = global_random.get('ece')
            if ece is not None:
                return ece
    return None


def extract_sgc_trust_score_acc(result: Dict[str, Any], L: int, d: int) -> Optional[float]:
    """Extract SGC accuracy using trust_score method for specific (L, d) configuration."""
    global_random = result.get('global_random_trust_score', {})
    if global_random and isinstance(global_random, dict) and not global_random.get('skipped', False):
        num_layers = global_random.get('num_layers')
        target_dim = global_random.get('target_dim')
        if num_layers == L and target_dim == d:
            acc = global_random.get('accuracy')
            if acc is not None:
                return acc / 100.0
    return None


def extract_sgc_trust_score_brier(result: Dict[str, Any], L: int, d: int) -> Optional[float]:
    """Extract SGC Brier score using trust_score method for specific L and d."""
    global_random = result.get('global_random_trust_score', {})
    if global_random and isinstance(global_random, dict) and not global_random.get('skipped', False):
        num_layers = global_random.get('num_layers')
        target_dim = global_random.get('target_dim')
        if num_layers == L and target_dim == d:
            brier = global_random.get('brier')
            if brier is not None:
                return brier
    return None


def extract_sgc_trust_score_ood_metrics(result: Dict[str, Any], L: int, d: int) -> Dict[str, Optional[float]]:
    """Extract SGC OOD metrics using trust_score method for specific L and d."""
    global_random = result.get('global_random_trust_score', {})
    if global_random and isinstance(global_random, dict) and not global_random.get('skipped', False):
        num_layers = global_random.get('num_layers')
        target_dim = global_random.get('target_dim')
        if num_layers == L and target_dim == d:
            return {
                'ood_auroc': global_random.get('ood_auroc'),
                'ood_fpr95': global_random.get('ood_fpr95'),
                'ood_ece': global_random.get('ood_ece'),
                'confidence_gap': global_random.get('confidence_gap'),
            }
    return {'ood_auroc': None, 'ood_fpr95': None, 'ood_ece': None, 'confidence_gap': None}


def extract_coordinate_sampling_trust_score_ece(result: Dict[str, Any]) -> Optional[float]:
    """Extract coordinate sampling ECE using trust_score method."""
    coord_sampling = result.get('coordinate_sampling_trust_score', {})
    if coord_sampling and isinstance(coord_sampling, dict) and not coord_sampling.get('skipped', False):
        ece = coord_sampling.get('ece')
        if ece is not None:
            return ece
    return None


def extract_coordinate_sampling_trust_score_acc(result: Dict[str, Any]) -> Optional[float]:
    """Extract coordinate sampling accuracy using trust_score method."""
    coord_sampling = result.get('coordinate_sampling_trust_score', {})
    if coord_sampling and isinstance(coord_sampling, dict) and not coord_sampling.get('skipped', False):
        acc = coord_sampling.get('accuracy')
        if acc is not None:
            return acc / 100.0
    return None


def extract_coordinate_sampling_trust_score_brier(result: Dict[str, Any]) -> Optional[float]:
    """Extract coordinate sampling Brier score using trust_score method."""
    coord_sampling = result.get('coordinate_sampling_trust_score', {})
    if coord_sampling and isinstance(coord_sampling, dict) and not coord_sampling.get('skipped', False):
        brier = coord_sampling.get('brier')
        if brier is not None:
            return brier
    return None


def extract_coordinate_sampling_trust_score_ood_metrics(result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract coordinate sampling OOD metrics using trust_score method."""
    coord_sampling = result.get('coordinate_sampling_trust_score', {})
    if coord_sampling and isinstance(coord_sampling, dict) and not coord_sampling.get('skipped', False):
        return {
            'ood_auroc': coord_sampling.get('ood_auroc'),
            'ood_fpr95': coord_sampling.get('ood_fpr95'),
            'ood_ece': coord_sampling.get('ood_ece'),
            'confidence_gap': coord_sampling.get('confidence_gap'),
        }
    return {'ood_auroc': None, 'ood_fpr95': None, 'ood_ece': None, 'confidence_gap': None}


def extract_sgc_with_dac_layers_trust_score_ece(result: Dict[str, Any]) -> Optional[float]:
    """Extract SGC with DAC Layer Selection ECE using trust_score method."""
    sgc_dac_layers = result.get('sgc_with_dac_layers_trust_score', {})
    if sgc_dac_layers and isinstance(sgc_dac_layers, dict) and not sgc_dac_layers.get('skipped', False):
        ece = sgc_dac_layers.get('ece')
        if ece is not None:
            return ece
    return None


def extract_sgc_with_dac_layers_trust_score_acc(result: Dict[str, Any]) -> Optional[float]:
    """Extract SGC with DAC Layer Selection accuracy using trust_score method."""
    sgc_dac_layers = result.get('sgc_with_dac_layers_trust_score', {})
    if sgc_dac_layers and isinstance(sgc_dac_layers, dict) and not sgc_dac_layers.get('skipped', False):
        acc = sgc_dac_layers.get('accuracy')
        if acc is not None:
            return acc / 100.0
    return None


def extract_sgc_with_dac_layers_trust_score_brier(result: Dict[str, Any]) -> Optional[float]:
    """Extract SGC with DAC Layer Selection Brier score using trust_score method."""
    sgc_dac_layers = result.get('sgc_with_dac_layers_trust_score', {})
    if sgc_dac_layers and isinstance(sgc_dac_layers, dict) and not sgc_dac_layers.get('skipped', False):
        brier = sgc_dac_layers.get('brier')
        if brier is not None:
            return brier
    return None


def extract_sgc_with_dac_layers_trust_score_ood_metrics(result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract SGC with DAC Layer Selection OOD metrics using trust_score method."""
    sgc_dac_layers = result.get('sgc_with_dac_layers_trust_score', {})
    if sgc_dac_layers and isinstance(sgc_dac_layers, dict) and not sgc_dac_layers.get('skipped', False):
        return {
            'ood_auroc': sgc_dac_layers.get('ood_auroc'),
            'ood_fpr95': sgc_dac_layers.get('ood_fpr95'),
            'ood_ece': sgc_dac_layers.get('ood_ece'),
            'confidence_gap': sgc_dac_layers.get('confidence_gap'),
        }
    return {'ood_auroc': None, 'ood_fpr95': None, 'ood_ece': None, 'confidence_gap': None}


def validate_results(results: List[Dict[str, Any]]) -> None:
    """Validate that key configurations exist."""
    clean_results = [r for r in results if is_clean_result(r)]
    
    has_sgc = any(extract_sgc_result(r, L=6, d=256) is not None for r in clean_results)
    if not has_sgc:
        logger.warning("No SGC L=6, d=256 results found in clean data!")
    
    # Check for new structure (standard_baselines) or old structure (baselines)
    has_baselines = any('standard_baselines' in r or 'baselines' in r for r in clean_results)
    if not has_baselines:
        logger.warning("No baseline results found in clean data!")
    
    # Check for new structure sections
    has_global_random = any('global_random' in r for r in clean_results)
    has_coord_sampling = any('coordinate_sampling' in r for r in clean_results)
    has_geo_dac_weighted = any('geometric_dac_weighted' in r for r in clean_results)
    
    logger.info(f"Validated: {len(clean_results)} clean results out of {len(results)} total")
    logger.info(f"Structure check: baselines={has_baselines}, global_random={has_global_random}, "
                f"coordinate_sampling={has_coord_sampling}, geometric_dac_weighted={has_geo_dac_weighted}")


def aggregate_over_seeds(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate results over seeds, computing mean ± std for each metric.
    Also stores raw ECE values as lists for statistical significance testing.
    """
    # Only use clean results for main tables
    clean_results = [r for r in results if is_clean_result(r)]
    
    if not clean_results:
        logger.warning("No clean results found! Using all results.")
        clean_results = results
    
    rows = []
    
    for result in clean_results:
        exp_key = extract_experiment_key(result)
        training_method, dataset, model_name = exp_key
        
        # Extract baseline metrics
        baseline_metrics = extract_baseline_metrics(result)
        
        # Extract baseline Brier scores
        baseline_brier = extract_baseline_brier(result)
        
        # Extract baseline OOD metrics
        baseline_ood = extract_baseline_ood_metrics(result)
        
        # Extract DAC Fixed (best at d=256)
        dac_ece = extract_dac_fixed(result, target_dim=256)
        dac_acc = extract_dac_fixed_acc(result, target_dim=256)
        dac_brier = extract_dac_fixed_brier(result, target_dim=256)
        dac_ood = extract_dac_fixed_ood_metrics(result, target_dim=256)
        
        # Extract TULIP Fixed (best at d=256)
        tulip_ece = extract_tulip_fixed(result, target_dim=256)
        tulip_acc = extract_tulip_fixed_acc(result, target_dim=256)
        tulip_brier = extract_tulip_fixed_brier(result, target_dim=256)
        tulip_ood = extract_tulip_fixed_ood_metrics(result, target_dim=256)
        
        # Extract SGC (L=6, d=256) - separation method
        sgc_ece = extract_sgc_result(result, L=6, d=256)
        sgc_acc = extract_sgc_acc(result, L=6, d=256)
        sgc_brier = extract_sgc_brier(result, L=6, d=256)
        sgc_ood = extract_sgc_ood_metrics(result, L=6, d=256)
        
        # Extract SGC (L=6, d=256) - trust_score method
        sgc_trust_score_ece = extract_sgc_trust_score_result(result, L=6, d=256)
        sgc_trust_score_acc = extract_sgc_trust_score_acc(result, L=6, d=256)
        sgc_trust_score_brier = extract_sgc_trust_score_brier(result, L=6, d=256)
        sgc_trust_score_ood = extract_sgc_trust_score_ood_metrics(result, L=6, d=256)
        
        # Extract SGC FAISS
        sgc_faiss_ece = extract_sgc_faiss_ece(result)
        sgc_faiss_acc = extract_sgc_faiss_acc(result)
        sgc_faiss_brier = extract_sgc_faiss_brier(result)
        sgc_faiss_ood = extract_sgc_faiss_ood_metrics(result)
        
        # Extract Oracle (geo_best at d=256)
        oracle_ece = extract_oracle_ece(result, target_dim=256)

        # Extract new metrics from updated JSON structure
        coord_ece = extract_coordinate_sampling_ece(result)
        coord_acc = extract_coordinate_sampling_acc(result)
        coord_brier = extract_coordinate_sampling_brier(result)
        coord_ood = extract_coordinate_sampling_ood_metrics(result)
        
        # Extract coordinate sampling - trust_score method
        coord_trust_score_ece = extract_coordinate_sampling_trust_score_ece(result)
        coord_trust_score_acc = extract_coordinate_sampling_trust_score_acc(result)
        coord_trust_score_brier = extract_coordinate_sampling_trust_score_brier(result)
        coord_trust_score_ood = extract_coordinate_sampling_trust_score_ood_metrics(result)
        
        geo_dac_weighted_ece = extract_geometric_dac_weighted_ece(result)
        geo_dac_weighted_acc = extract_geometric_dac_weighted_acc(result)
        dac_random_ece = extract_dac_with_random_layers_ece(result)
        dac_random_acc = extract_dac_with_random_layers_acc(result)
        dac_random_ood = extract_dac_with_random_layers_ood_metrics(result)
        sgc_dac_preprocessing_ece = extract_sgc_with_dac_preprocessing_ece(result)
        sgc_dac_preprocessing_acc = extract_sgc_with_dac_preprocessing_acc(result)
        sgc_dac_layers_ece = extract_sgc_with_dac_layers_ece(result)
        sgc_dac_layers_acc = extract_sgc_with_dac_layers_acc(result)
        sgc_dac_layers_brier = extract_sgc_with_dac_layers_brier(result)
        sgc_dac_layers_ood = extract_sgc_with_dac_layers_ood_metrics(result)
        
        # Extract SGC with DAC layers - trust_score method
        sgc_dac_layers_trust_score_ece = extract_sgc_with_dac_layers_trust_score_ece(result)
        sgc_dac_layers_trust_score_acc = extract_sgc_with_dac_layers_trust_score_acc(result)
        sgc_dac_layers_trust_score_brier = extract_sgc_with_dac_layers_trust_score_brier(result)
        sgc_dac_layers_trust_score_ood = extract_sgc_with_dac_layers_trust_score_ood_metrics(result)
        
        dac_with_coord_ece = extract_dac_with_coordinate_features_ece(result)
        dac_with_coord_acc = extract_dac_with_coordinate_features_acc(result)
        
        row = {
            'training_method': training_method,
            'dataset': dataset,
            'model_name': model_name,
            'seed': result.get('experiment_info', {}).get('seed', result.get('experiment_config', {}).get('seed', None)),
            **baseline_metrics,
            **baseline_brier,
            **baseline_ood,
            'dac_ece': dac_ece,
            'dac_acc': dac_acc,
            'dac_brier': dac_brier,
            'dac_ood_auroc': dac_ood.get('ood_auroc'),
            'dac_ood_fpr95': dac_ood.get('ood_fpr95'),
            'dac_ood_ece': dac_ood.get('ood_ece'),
            'dac_confidence_gap': dac_ood.get('confidence_gap'),
            'tulip_ece': tulip_ece,
            'tulip_acc': tulip_acc,
            'tulip_brier': tulip_brier,
            'tulip_ood_auroc': tulip_ood.get('ood_auroc'),
            'tulip_ood_fpr95': tulip_ood.get('ood_fpr95'),
            'tulip_ood_ece': tulip_ood.get('ood_ece'),
            'tulip_confidence_gap': tulip_ood.get('confidence_gap'),
            'sgc_ece': sgc_ece,
            'sgc_acc': sgc_acc,
            'sgc_brier': sgc_brier,
            'sgc_ood_auroc': sgc_ood.get('ood_auroc'),
            'sgc_ood_fpr95': sgc_ood.get('ood_fpr95'),
            'sgc_ood_ece': sgc_ood.get('ood_ece'),
            'sgc_confidence_gap': sgc_ood.get('confidence_gap'),
            # SGC trust_score method
            'sgc_trust_score_ece': sgc_trust_score_ece,
            'sgc_trust_score_acc': sgc_trust_score_acc,
            'sgc_trust_score_brier': sgc_trust_score_brier,
            'sgc_trust_score_ood_auroc': sgc_trust_score_ood.get('ood_auroc'),
            'sgc_trust_score_ood_fpr95': sgc_trust_score_ood.get('ood_fpr95'),
            'sgc_trust_score_ood_ece': sgc_trust_score_ood.get('ood_ece'),
            'sgc_trust_score_confidence_gap': sgc_trust_score_ood.get('confidence_gap'),
            # SGC FAISS
            'sgc_faiss_ece': sgc_faiss_ece,
            'sgc_faiss_acc': sgc_faiss_acc,
            'sgc_faiss_brier': sgc_faiss_brier,
            'sgc_faiss_ood_auroc': sgc_faiss_ood.get('ood_auroc'),
            'sgc_faiss_ood_fpr95': sgc_faiss_ood.get('ood_fpr95'),
            'sgc_faiss_ood_ece': sgc_faiss_ood.get('ood_ece'),
            'sgc_faiss_confidence_gap': sgc_faiss_ood.get('confidence_gap'),
            'oracle_ece': oracle_ece,
            # New metrics
            'coord_ece': coord_ece,
            'coord_acc': coord_acc,
            'coord_brier': coord_brier,
            'coord_ood_auroc': coord_ood.get('ood_auroc'),
            'coord_ood_fpr95': coord_ood.get('ood_fpr95'),
            'coord_ood_ece': coord_ood.get('ood_ece'),
            'coord_confidence_gap': coord_ood.get('confidence_gap'),
            # Coordinate sampling trust_score method
            'coord_trust_score_ece': coord_trust_score_ece,
            'coord_trust_score_acc': coord_trust_score_acc,
            'coord_trust_score_brier': coord_trust_score_brier,
            'coord_trust_score_ood_auroc': coord_trust_score_ood.get('ood_auroc'),
            'coord_trust_score_ood_fpr95': coord_trust_score_ood.get('ood_fpr95'),
            'coord_trust_score_ood_ece': coord_trust_score_ood.get('ood_ece'),
            'coord_trust_score_confidence_gap': coord_trust_score_ood.get('confidence_gap'),
            'geo_dac_weighted_ece': geo_dac_weighted_ece,
            'geo_dac_weighted_acc': geo_dac_weighted_acc,
            'dac_random_ece': dac_random_ece,
            'dac_random_acc': dac_random_acc,
            'dac_random_ood_auroc': dac_random_ood.get('ood_auroc'),
            'dac_random_ood_fpr95': dac_random_ood.get('ood_fpr95'),
            'dac_random_ood_ece': dac_random_ood.get('ood_ece'),
            'dac_random_confidence_gap': dac_random_ood.get('confidence_gap'),
            'sgc_dac_preprocessing_ece': sgc_dac_preprocessing_ece,
            'sgc_dac_preprocessing_acc': sgc_dac_preprocessing_acc,
            'sgc_dac_layers_ece': sgc_dac_layers_ece,
            'sgc_dac_layers_acc': sgc_dac_layers_acc,
            'sgc_dac_layers_brier': sgc_dac_layers_brier,
            'sgc_dac_layers_ood_auroc': sgc_dac_layers_ood.get('ood_auroc'),
            'sgc_dac_layers_ood_fpr95': sgc_dac_layers_ood.get('ood_fpr95'),
            'sgc_dac_layers_ood_ece': sgc_dac_layers_ood.get('ood_ece'),
            'sgc_dac_layers_confidence_gap': sgc_dac_layers_ood.get('confidence_gap'),
            # SGC with DAC layers trust_score method
            'sgc_dac_layers_trust_score_ece': sgc_dac_layers_trust_score_ece,
            'sgc_dac_layers_trust_score_acc': sgc_dac_layers_trust_score_acc,
            'sgc_dac_layers_trust_score_brier': sgc_dac_layers_trust_score_brier,
            'sgc_dac_layers_trust_score_ood_auroc': sgc_dac_layers_trust_score_ood.get('ood_auroc'),
            'sgc_dac_layers_trust_score_ood_fpr95': sgc_dac_layers_trust_score_ood.get('ood_fpr95'),
            'sgc_dac_layers_trust_score_ood_ece': sgc_dac_layers_trust_score_ood.get('ood_ece'),
            'sgc_dac_layers_trust_score_confidence_gap': sgc_dac_layers_trust_score_ood.get('confidence_gap'),
            'dac_with_coordinate_features_ece': dac_with_coord_ece,
            'dac_with_coordinate_features_acc': dac_with_coord_acc,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Aggregate over seeds: group by (training_method, dataset, model_name)
    # For ECE columns, also store lists of raw values for significance testing
    agg_dict = {}
    
    for col in df.columns:
        if col not in ['training_method', 'dataset', 'model_name', 'seed']:
            if col.endswith('_ece') or col.endswith('_acc') or col.endswith('_brier') or \
               col.endswith('_ood_auroc') or col.endswith('_ood_fpr95') or col.endswith('_ood_ece') or \
               col.endswith('_confidence_gap'):
                # For ECE/ACC/Brier/OOD columns, aggregate mean/std/count AND list
                agg_dict[col] = ['mean', 'std', 'count']
            else:
                # For other columns, just mean/std/count
                agg_dict[col] = ['mean', 'std', 'count']
    
    grouped = df.groupby(['training_method', 'dataset', 'model_name']).agg(agg_dict)
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]
    
    # Calculate 95% CI instead of std: CI = 1.96 * std / sqrt(N)
    for col in df.columns:
        if col not in ['training_method', 'dataset', 'model_name', 'seed']:
            mean_col = f'{col}_mean'
            std_col = f'{col}_std'
            count_col = f'{col}_count'
            ci_col = f'{col}_ci'
            
            if mean_col in grouped.columns and std_col in grouped.columns and count_col in grouped.columns:
                # Calculate 95% CI: 1.96 * std / sqrt(N)
                # Handle division by zero and NaN cases
                def calc_ci(row):
                    std_val = row[std_col]
                    count_val = row[count_col]
                    if pd.isna(std_val) or pd.isna(count_val) or count_val <= 1:
                        return np.nan
                    return 1.96 * std_val / np.sqrt(count_val)
                
                grouped[ci_col] = grouped.apply(calc_ci, axis=1)
    
    # Now aggregate lists separately for significance testing
    for col in df.columns:
        if col not in ['training_method', 'dataset', 'model_name', 'seed'] and \
           (col.endswith('_ece') or col.endswith('_acc') or col.endswith('_brier') or
            col.endswith('_ood_auroc') or col.endswith('_ood_fpr95') or col.endswith('_ood_ece') or
            col.endswith('_confidence_gap')):
            list_col = f'{col}_values'
            list_series = df.groupby(['training_method', 'dataset', 'model_name'])[col].apply(
                lambda x: x.dropna().tolist()
            )
            grouped[list_col] = list_series
    
    # Reset index
    grouped = grouped.reset_index()
    
    return grouped


def calculate_significance(scores1: List[float], scores2: List[float], alternative: str = 'less') -> float:
    """Calculate p-value for one-sided t-test comparing two sets of scores.
    
    Args:
        scores1: First set of ECE scores (e.g., SGC)
        scores2: Second set of ECE scores (e.g., best baseline)
        alternative: 'less' for testing if scores1 < scores2, 'greater' for scores1 > scores2
    
    Returns:
        p-value (float). Returns 1.0 if test cannot be performed (insufficient data, NaN values, etc.)
    """
    # Filter out NaN values
    scores1_clean = [s for s in scores1 if pd.notna(s)]
    scores2_clean = [s for s in scores2 if pd.notna(s)]
    
    # Need at least 2 samples for t-test
    if len(scores1_clean) < 2 or len(scores2_clean) < 2:
        return 1.0
    
    # If lengths match, use paired t-test (more powerful)
    if len(scores1_clean) == len(scores2_clean):
        try:
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(scores1_clean, scores2_clean, alternative=alternative)
            return p_value
        except Exception:
            return 1.0
    else:
        # Independent samples t-test
        try:
            t_stat, p_value = stats.ttest_ind(scores1_clean, scores2_clean, alternative=alternative)
            return p_value
        except Exception:
            return 1.0


def perform_tost_test(
    scores1: List[float], 
    scores2: List[float], 
    equivalence_margin: float = 0.0001
) -> Dict[str, Any]:
    """
    Perform Two One-Sided Tests (TOST) for equivalence testing.
    
    Args:
        scores1: First set of paired scores (e.g., SGC ECE values)
        scores2: Second set of paired scores (e.g., Coord ECE values)
        equivalence_margin: Equivalence margin (delta) in same units as scores (default 0.005 = 0.5%)
    
    Returns:
        Dictionary with:
        - tost_pvalue: TOST p-value (max of two one-sided tests)
        - mean_diff: Mean difference (scores2 - scores1)
        - ci_lower: Lower bound of 95% CI for difference
        - ci_upper: Upper bound of 95% CI for difference
        - dz: Paired effect size (Cohen's d_z)
        - equivalence_established: Boolean indicating if equivalence is established (p < 0.05)
        - n_pairs: Number of paired observations
    """
    # Filter out NaN values and ensure paired data
    pairs = [(s1, s2) for s1, s2 in zip(scores1, scores2) if pd.notna(s1) and pd.notna(s2)]
    
    if len(pairs) < 2:
        return {
            'tost_pvalue': 1.0,
            'mean_diff': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'dz': 0.0,
            'equivalence_established': False,
            'n_pairs': len(pairs)
        }
    
    arr1 = np.array([p[0] for p in pairs])
    arr2 = np.array([p[1] for p in pairs])
    diff_arr = arr2 - arr1
    
    n = len(diff_arr)
    mean_diff = diff_arr.mean()
    std_diff = diff_arr.std(ddof=1)
    
    # Standard error of the mean difference
    se_diff = std_diff / np.sqrt(n) if n > 1 else 0.0
    
    # 95% CI for mean difference
    t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else 1.96
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    # TOST: Two One-Sided Tests
    # H01: mean_diff >= margin (reject if p_upper < 0.05)
    # H02: mean_diff <= -margin (reject if p_lower < 0.05)
    # Equivalence established if both are rejected (max(p_upper, p_lower) < 0.05)
    if se_diff > 0 and n > 1:
        t_upper = (mean_diff - equivalence_margin) / se_diff
        t_lower = (mean_diff + equivalence_margin) / se_diff
        p_upper = stats.t.cdf(t_upper, df=n - 1)
        p_lower = 1 - stats.t.cdf(t_lower, df=n - 1)
        tost_pval = max(p_upper, p_lower)
    else:
        tost_pval = 1.0
    
    # Paired effect size (Cohen's d_z)
    dz = mean_diff / std_diff if std_diff > 0 else 0.0
    
    equivalence_established = tost_pval < 0.05
    
    # Minimum margin for equivalence = max of absolute CI bounds
    # This is the tightest margin where equivalence would still be established
    if pd.notna(ci_lower) and pd.notna(ci_upper):
        min_margin_for_equiv = max(abs(ci_lower), abs(ci_upper))
    else:
        min_margin_for_equiv = np.nan
    
    return {
        'tost_pvalue': float(tost_pval),
        'mean_diff': float(mean_diff),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'dz': float(dz),
        'equivalence_established': equivalence_established,
        'n_pairs': n,
        'min_margin_for_equiv': float(min_margin_for_equiv) if pd.notna(min_margin_for_equiv) else np.nan,
        'min_margin_for_equiv_pct': float(min_margin_for_equiv * 100) if pd.notna(min_margin_for_equiv) else np.nan,
    }


def determine_result(
    tost_pvalue: float, 
    mean_diff: float, 
    equivalence_margin: float,
    ci_lower: float,
    ci_upper: float
) -> str:
    """
    Determine the result category based on TOST and CI.
    
    Note: mean_diff = scores2 - scores1. Since lower ECE is better:
    - If mean_diff > 0: Method 2 has higher (worse) ECE, so Method 1 is superior
    - If mean_diff < 0: Method 2 has lower (better) ECE, so Method 2 is superior
    
    Returns:
        "Equivalent" if TOST p < 0.05 (equivalence established)
        "Inconclusive" if TOST p >= 0.05 (equivalence not established, but may be due to insufficient power)
        "Method X superior" if CI doesn't include 0 and doesn't overlap with equivalence region
    """
    if pd.isna(tost_pvalue) or pd.isna(mean_diff) or pd.isna(ci_lower) or pd.isna(ci_upper):
        return "---"
    
    if tost_pvalue < 0.05:
        return "Equivalent"
    
    # Check if CI overlaps with equivalence region [-margin, +margin]
    ci_overlaps_equiv = (ci_lower <= equivalence_margin and ci_upper >= -equivalence_margin)
    
    # Check if CI excludes 0 (statistically significant difference)
    ci_excludes_zero = (ci_lower > 0) or (ci_upper < 0)
    
    # If CI excludes 0 and doesn't overlap with equivalence region, one method is superior
    # Remember: mean_diff = scores2 - scores1, and lower ECE is better
    if ci_excludes_zero and not ci_overlaps_equiv:
        if ci_lower > 0:
            # Entire CI is positive: mean_diff > 0, so Method 2 has higher (worse) ECE
            # Therefore Method 1 is superior
            return "Method 1 superior"
        elif ci_upper < 0:
            # Entire CI is negative: mean_diff < 0, so Method 2 has lower (better) ECE
            # Therefore Method 2 is superior
            return "Method 2 superior"
    
    # Otherwise, inconclusive (equivalence not established, but can't conclude superiority)
    return "Inconclusive"


def calculate_sample_size_requirement(scores1: List[float], scores2: List[float], alpha=0.05, power=0.8) -> Dict[str, Any]:
    """
    Estimate how many samples are needed to see significance.
    Uses Cohen's d effect size.
    
    Args:
        scores1: First set of ECE scores (e.g., SGC)
        scores2: Second set of ECE scores (e.g., best baseline)
        alpha: Significance level (default 0.05)
        power: Statistical power (default 0.8)
    
    Returns:
        Dict with:
        - n_required: Number of samples needed for significance
        - effect_size: Cohen's d effect size
        - current_n: Current number of samples
        - more_needed: How many more samples needed
        - action: Recommended action string
    """
    s1 = np.array([s for s in scores1 if pd.notna(s)])
    s2 = np.array([s for s in scores2 if pd.notna(s)])
    
    if len(s1) < 2 or len(s2) < 2:
        return {
            'n_required': "N/A",
            'effect_size': 0.0,
            'current_n': len(s1),
            'more_needed': "N/A",
            'action': "Collect more data"
        }
    
    mean1, std1 = np.mean(s1), np.std(s1, ddof=1)
    mean2, std2 = np.mean(s2), np.std(s2, ddof=1)
    
    # Pooled standard deviation
    s_pooled = np.sqrt(((len(s1)-1)*std1**2 + (len(s2)-1)*std2**2) / (len(s1)+len(s2)-2))
    
    if s_pooled == 0:
        return {
            'n_required': "N/A",
            'effect_size': 0.0,
            'current_n': len(s1),
            'more_needed': "N/A",
            'action': "Zero variance"
        }
    
    effect_size = abs(mean1 - mean2) / s_pooled
    
    # Estimate N (approximation for one-sided t-test)
    # Z_alpha (0.05) = 1.645, Z_beta (0.8) = 0.84
    # N ~ 2 * ((Z_alpha + Z_beta) / effect_size)^2
    if effect_size < 0.01:
        n_required = ">1000"
        more_needed = "Many"
        action = "Run many more"
    else:
        n_required = int(np.ceil(2 * ((1.645 + 0.84) / effect_size) ** 2))
        more_needed = n_required - len(s1)
        if more_needed <= 0:
            more_needed = 0
            action = "Sufficient"
        else:
            action = f"Run {more_needed} more"
    
    return {
        'n_required': n_required,
        'effect_size': float(effect_size),
        'current_n': len(s1),
        'more_needed': more_needed,
        'action': action
    }


def format_ece_value(mean_val: float, ci_val: float = None, std_val: float = None, count: int = None, inline_math: bool = False) -> str:
    """Format ECE value as mean ± 95% CI or just mean if CI is NaN/zero or count=1.
    
    Args:
        mean_val: Mean ECE value
        ci_val: 95% Confidence Interval (preferred, pre-calculated)
        std_val: Standard deviation (used to calculate CI if ci_val not provided)
        count: Number of samples (required for CI calculation from std)
        inline_math: If True, use inline math mode (1.23$\pm$0.26), else use spaced (1.23 $\\pm$ 0.26)
    
    Note:
        When calculating CI from std, uses t-distribution: t_critical * SEM where
        SEM = std / sqrt(n) and t_critical = stats.t.ppf(0.975, df=n-1)
    """
    if pd.isna(mean_val):
        return "---"
    
    # Use CI if provided, otherwise calculate from std if available
    if ci_val is not None and not pd.isna(ci_val) and ci_val > 0:
        error_val = ci_val
    elif std_val is not None and not pd.isna(std_val) and std_val > 0 and count is not None and count > 1:
        # Calculate 95% CI from std using t-distribution
        # SEM = standard error of the mean
        sem = std_val / np.sqrt(count)
        # t-critical for 95% CI with n-1 degrees of freedom
        t_critical = stats.t.ppf(0.975, df=count - 1)
        error_val = t_critical * sem
    elif std_val is not None and not pd.isna(std_val) and std_val > 0 and (count is None or count == 1):
        # Fallback: if count is missing or 1, use std as approximation
        error_val = std_val
    else:
        # No error bar available
        return f"{mean_val * 100:.2f}"
    
    if inline_math:
        return f"{mean_val * 100:.2f}$\\pm${error_val * 100:.2f}"
    else:
        return f"{mean_val * 100:.2f} $\\pm$ {error_val * 100:.2f}"


def format_accuracy_value(mean_val: float, ci_val: float = None, std_val: float = None, count: int = None, inline_math: bool = False) -> str:
    """Format accuracy value as mean ± 95% CI or just mean if CI is NaN/zero or count=1.
    
    Args:
        mean_val: Mean accuracy value (as percentage, e.g., 95.5 for 95.5%, or in [0,1] range)
        ci_val: 95% Confidence Interval (preferred, calculated as 1.96 * std / sqrt(N))
        std_val: Standard deviation (fallback if ci_val not provided, for backward compatibility)
        count: Number of samples
        inline_math: If True, use inline math mode (95.5$\pm$0.5), else use spaced (95.5 $\\pm$ 0.5)
    """
    if pd.isna(mean_val):
        return "---"
    
    # If accuracy is in [0, 1] range, convert to percentage
    convert_to_pct = mean_val <= 1.0
    if convert_to_pct:
        mean_val = mean_val * 100
        if ci_val is not None and not pd.isna(ci_val):
            ci_val = ci_val * 100
        if std_val is not None and not pd.isna(std_val):
            std_val = std_val * 100
    
    # Use CI if provided, otherwise calculate from std if available
    if ci_val is not None and not pd.isna(ci_val) and ci_val > 0:
        error_val = ci_val
    elif std_val is not None and not pd.isna(std_val) and std_val > 0 and count is not None and count > 1:
        # Calculate CI from std: 1.96 * std / sqrt(N)
        if not convert_to_pct:
            error_val = 1.96 * std_val / np.sqrt(count) * 100
        else:
            error_val = 1.96 * std_val / np.sqrt(count)
    else:
        # No error bar available
        return f"{mean_val:.2f}"
    
    if inline_math:
        return f"{mean_val:.2f}$\\pm${error_val:.2f}"
    else:
        return f"{mean_val:.2f} $\\pm$ {error_val:.2f}"


def format_brier_value(mean_val: float, ci_val: float = None, std_val: float = None, count: int = None, inline_math: bool = False) -> str:
    """Format Brier score value as mean ± 95% CI or just mean if CI is NaN/zero or count=1.
    
    Args:
        mean_val: Mean Brier score value (in [0,1] range)
        ci_val: 95% Confidence Interval (preferred, calculated as 1.96 * std / sqrt(N))
        std_val: Standard deviation (fallback if ci_val not provided)
        count: Number of samples
        inline_math: If True, use inline math mode (0.085$\pm$0.002), else use spaced
    """
    if pd.isna(mean_val):
        return "---"
    
    # Use CI if provided, otherwise calculate from std if available
    if ci_val is not None and not pd.isna(ci_val) and ci_val > 0:
        error_val = ci_val
    elif std_val is not None and not pd.isna(std_val) and std_val > 0 and count is not None and count > 1:
        # Calculate CI from std: 1.96 * std / sqrt(N)
        error_val = 1.96 * std_val / np.sqrt(count)
    else:
        # No error bar available
        return f"{mean_val:.4f}"
    
    if inline_math:
        return f"{mean_val:.4f}$\\pm${error_val:.4f}"
    else:
        return f"{mean_val:.4f} $\\pm$ {error_val:.4f}"


def format_ood_value(mean_val: float, ci_val: float = None, std_val: float = None, count: int = None, inline_math: bool = False, is_percentage: bool = False) -> str:
    """Format OOD metric value (AUROC, FPR95, ECE, confidence gap) as mean ± 95% CI.
    
    Args:
        mean_val: Mean OOD metric value
        ci_val: 95% Confidence Interval
        std_val: Standard deviation (fallback)
        count: Number of samples
        inline_math: If True, use inline math mode
        is_percentage: If True, format as percentage (multiply by 100)
    """
    if pd.isna(mean_val):
        return "---"
    
    # Convert to percentage if needed
    if is_percentage:
        mean_val = mean_val * 100
        if ci_val is not None and not pd.isna(ci_val):
            ci_val = ci_val * 100
        if std_val is not None and not pd.isna(std_val):
            std_val = std_val * 100
    
    # Use CI if provided, otherwise calculate from std if available
    if ci_val is not None and not pd.isna(ci_val) and ci_val > 0:
        error_val = ci_val
    elif std_val is not None and not pd.isna(std_val) and std_val > 0 and count is not None and count > 1:
        # Calculate CI from std: 1.96 * std / sqrt(N)
        if is_percentage:
            error_val = 1.96 * std_val / np.sqrt(count) * 100
        else:
            error_val = 1.96 * std_val / np.sqrt(count)
    else:
        # No error bar available
        if is_percentage:
            return f"{mean_val:.2f}"
        else:
            return f"{mean_val:.4f}"
    
    if inline_math:
        if is_percentage:
            return f"{mean_val:.2f}$\\pm${error_val:.2f}"
        else:
            return f"{mean_val:.4f}$\\pm${error_val:.4f}"
    else:
        if is_percentage:
            return f"{mean_val:.2f} $\\pm$ {error_val:.2f}"
        else:
            return f"{mean_val:.4f} $\\pm$ {error_val:.4f}"


def get_training_method_display_name(training_method: str) -> str:
    """Get display name for training method."""
    mapping = {
        'baseline_cross_entropy': 'Cross Entropy',
        'baseline_focal_adaptive': 'Focal Adaptive',
        'baseline_brier': 'Brier',
        'baseline_mmce_weighted': 'MMCE Weighted',
        'augmix': 'Augmix',
    }
    return mapping.get(training_method, training_method.replace('_', ' ').title())


def generate_single_comparison_table(group_df: pd.DataFrame, dataset: str, training_method: str) -> str:
    """Generate a single comparison table for a specific dataset and training method."""
    # Get display name for training method
    training_display = get_training_method_display_name(training_method)
    dataset_upper = dataset.upper()
    
    # Create label (e.g., tab:cifar10_augmix)
    label = f"tab:{dataset}_{training_method}"
    
    # Track if we have any significant results to add footnote
    has_significant = False
    
    # Sort by model name for consistency
    group_df = group_df.sort_values('model_name')
    
    # Build table structure
    lines = []
    lines.append("\\centering")
    
    # Process rows to check for significance first
    for _, row in group_df.iterrows():
        # Check for SGC significance
        if not pd.isna(row.get('sgc_ece_mean')) and 'sgc_ece_values' in row:
            sgc_scores = row.get('sgc_ece_values', [])
            if isinstance(sgc_scores, list) and len(sgc_scores) >= 2:
                # Find best baseline (excluding SGC)
                baseline_means = {}
                for method in ['ts', 'platt', 'isotonic', 'beta', 'dac_algorithm', 'pixel_geo', 'dac', 'tulip']:
                    if not pd.isna(row.get(f'{method}_ece_mean')):
                        baseline_means[method] = row[f'{method}_ece_mean']
                
                if baseline_means:
                    best_baseline_label = min(baseline_means, key=baseline_means.get)
                    best_baseline_mean = baseline_means[best_baseline_label]
                    
                    if row['sgc_ece_mean'] < best_baseline_mean:
                        baseline_values_col = f'{best_baseline_label}_ece_values'
                        baseline_scores = row.get(baseline_values_col, [])
                        
                        if isinstance(baseline_scores, list) and len(baseline_scores) >= 2:
                            p_value = calculate_significance(sgc_scores, baseline_scores, alternative='less')
                            if p_value < 0.05:
                                has_significant = True
                                break
    
    # Build caption with footnote if needed
    footnote_parts = []
    if dataset.lower() == 'cifar100' and training_method == 'baseline_cross_entropy':
        footnote_parts.append("TULIP published values corrected for apparent decimal error in original paper (Table 8).")
    if has_significant:
        footnote_parts.append("* Indicates statistical significance (p < 0.05) against the best baseline.")
    
    if footnote_parts:
        footnote_text = "\\footnote{" + " ".join(footnote_parts) + "}"
        lines.append(f"\\caption{{ECE (\\%) on {dataset_upper} with {training_display}{footnote_text}}}")
    else:
        lines.append(f"\\caption{{ECE (\\%) on {dataset_upper} with {training_display}}}")
    
    lines.append(f"{{\\footnotesize \\textcolor{{blue}}{{\\textbf{{Best}}}}, \\textcolor{{teal}}{{\\textbf{{2nd}}}}, \\textcolor{{olive}}{{\\textbf{{3rd}}}}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("% Adjust column spacing and font size")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccccccccccc|c|c|c}")
    lines.append("\\toprule")
    lines.append("Model & TS & Platt & Isotonic & Beta & DAC & Pixel Geo & TULIP (pub) & SGC(DAC) & SGC(TULIP) & SGC & Uncal & Acc & Seeds \\\\")
    lines.append("\\midrule")
    
    # Process rows to build table
    for _, row in group_df.iterrows():
        model = row['model_name']
        
        # TULIP published results key (only for Cross Entropy training)
        tulip_pub_key = (dataset.lower(), training_method, model)
        
        # Extract all ECE values with inline math format
        ts = format_ece_value(
            row.get('ts_ece_mean', np.nan),
            row.get('ts_ece_std', np.nan),
            row.get('ts_ece_count', None),
            inline_math=True
        )
        platt = format_ece_value(
            row.get('platt_ece_mean', np.nan),
            row.get('platt_ece_std', np.nan),
            row.get('platt_ece_count', None),
            inline_math=True
        )
        isotonic = format_ece_value(
            row.get('isotonic_ece_mean', np.nan),
            row.get('isotonic_ece_std', np.nan),
            row.get('isotonic_ece_count', None),
            inline_math=True
        )
        beta = format_ece_value(
            row.get('beta_ece_mean', np.nan),
            row.get('beta_ece_std', np.nan),
            row.get('beta_ece_count', None),
            inline_math=True
        )
        dac_algorithm = format_ece_value(
            row.get('dac_algorithm_ece_mean', np.nan),
            row.get('dac_algorithm_ece_std', np.nan),
            row.get('dac_algorithm_ece_count', None),
            inline_math=True
        )
        pixel_geo = format_ece_value(
            row.get('pixel_geo_ece_mean', np.nan),
            row.get('pixel_geo_ece_std', np.nan),
            row.get('pixel_geo_ece_count', None),
            inline_math=True
        )
        
        # TULIP published results (only for Cross Entropy training)
        if tulip_pub_key in TULIP_PUBLISHED:
            tulip_pub_mean, tulip_pub_std = TULIP_PUBLISHED[tulip_pub_key]
            tulip_pub = format_ece_value(tulip_pub_mean, tulip_pub_std, count=None, inline_math=True)
        else:
            tulip_pub = "---"
        
        dac = format_ece_value(
            row.get('dac_ece_mean', np.nan),
            row.get('dac_ece_std', np.nan),
            row.get('dac_ece_count', None),
            inline_math=True
        )
        tulip = format_ece_value(
            row.get('tulip_ece_mean', np.nan),
            row.get('tulip_ece_std', np.nan),
            row.get('tulip_ece_count', None),
            inline_math=True
        )
        sgc = format_ece_value(
            row.get('sgc_ece_mean', np.nan),
            row.get('sgc_ece_std', np.nan),
            row.get('sgc_ece_count', None),
            inline_math=True
        )
        uncal = format_ece_value(
            row.get('uncal_ece_mean', np.nan),
            row.get('uncal_ece_std', np.nan),
            row.get('uncal_ece_count', None),
            inline_math=True
        )
        
        # Extract accuracy
        accuracy = format_accuracy_value(
            row.get('uncal_acc_mean', np.nan),
            row.get('uncal_acc_std', np.nan),
            row.get('uncal_acc_count', None),
            inline_math=True
        )
        
        # Extract seed count (use maximum across all metrics, or SGC if available)
        seed_counts = []
        for metric in ['uncal_ece_count', 'ts_ece_count', 'platt_ece_count', 'isotonic_ece_count', 'beta_ece_count', 
                       'dac_algorithm_ece_count', 'pixel_geo_ece_count', 'dac_ece_count', 
                       'tulip_ece_count', 'sgc_ece_count']:
            count = row.get(metric, 0)
            if pd.notna(count) and count > 0:
                seed_counts.append(int(count))
        
        num_seeds = max(seed_counts) if seed_counts else 0
        
        # Find top 3 calibration methods (exclude uncalibrated)
        ece_values = []
        ece_labels = []
        ece_formatted = {}
        ece_mean_dict = {}  # Store mean values for significance testing
        
        # Don't include uncalibrated in the comparison
        if not pd.isna(row.get('ts_ece_mean')):
            ece_values.append(row['ts_ece_mean'])
            ece_labels.append('ts')
            ece_formatted['ts'] = ts
            ece_mean_dict['ts'] = row['ts_ece_mean']
        if not pd.isna(row.get('platt_ece_mean')):
            ece_values.append(row['platt_ece_mean'])
            ece_labels.append('platt')
            ece_formatted['platt'] = platt
            ece_mean_dict['platt'] = row['platt_ece_mean']
        if not pd.isna(row.get('isotonic_ece_mean')):
            ece_values.append(row['isotonic_ece_mean'])
            ece_labels.append('isotonic')
            ece_formatted['isotonic'] = isotonic
            ece_mean_dict['isotonic'] = row['isotonic_ece_mean']
        if not pd.isna(row.get('beta_ece_mean')):
            ece_values.append(row['beta_ece_mean'])
            ece_labels.append('beta')
            ece_formatted['beta'] = beta
            ece_mean_dict['beta'] = row['beta_ece_mean']
        if not pd.isna(row.get('dac_algorithm_ece_mean')):
            ece_values.append(row['dac_algorithm_ece_mean'])
            ece_labels.append('dac_algorithm')
            ece_formatted['dac_algorithm'] = dac_algorithm
            ece_mean_dict['dac_algorithm'] = row['dac_algorithm_ece_mean']
        if not pd.isna(row.get('pixel_geo_ece_mean')):
            ece_values.append(row['pixel_geo_ece_mean'])
            ece_labels.append('pixel_geo')
            ece_formatted['pixel_geo'] = pixel_geo
            ece_mean_dict['pixel_geo'] = row['pixel_geo_ece_mean']
        # TULIP published (only for Cross Entropy) - skip for significance testing (no raw values)
        if tulip_pub_key is not None and tulip_pub_key in TULIP_PUBLISHED:
            tulip_pub_mean_val, tulip_pub_std_val = TULIP_PUBLISHED[tulip_pub_key]
            ece_values.append(tulip_pub_mean_val)
            ece_labels.append('tulip_pub')
            ece_formatted['tulip_pub'] = tulip_pub
            # Don't add to ece_mean_dict - we can't test significance against published values
        if not pd.isna(row.get('dac_ece_mean')):
            ece_values.append(row['dac_ece_mean'])
            ece_labels.append('dac')
            ece_formatted['dac'] = dac
            ece_mean_dict['dac'] = row['dac_ece_mean']
        if not pd.isna(row.get('tulip_ece_mean')):
            ece_values.append(row['tulip_ece_mean'])
            ece_labels.append('tulip')
            ece_formatted['tulip'] = tulip
            ece_mean_dict['tulip'] = row['tulip_ece_mean']
        if not pd.isna(row.get('sgc_ece_mean')):
            ece_values.append(row['sgc_ece_mean'])
            ece_labels.append('sgc')
            ece_formatted['sgc'] = sgc
            ece_mean_dict['sgc'] = row['sgc_ece_mean']
        
        # Statistical significance testing: SGC vs Best Baseline
        sgc_is_significant = False
        if not pd.isna(row.get('sgc_ece_mean')) and 'sgc_ece_values' in row:
            sgc_scores = row.get('sgc_ece_values', [])
            if isinstance(sgc_scores, list) and len(sgc_scores) >= 2:
                # Find best baseline (excluding SGC and TULIP published)
                baseline_means = {k: v for k, v in ece_mean_dict.items() if k != 'sgc'}
                if baseline_means:
                    best_baseline_label = min(baseline_means, key=baseline_means.get)
                    best_baseline_mean = baseline_means[best_baseline_label]
                    
                    # Only test if SGC is better (lower ECE) than best baseline
                    if row['sgc_ece_mean'] < best_baseline_mean:
                        # Get raw values for best baseline
                        baseline_values_col = f'{best_baseline_label}_ece_values'
                        baseline_scores = row.get(baseline_values_col, [])
                        
                        if isinstance(baseline_scores, list) and len(baseline_scores) >= 2:
                            # Perform one-sided t-test: H0: SGC >= baseline, H1: SGC < baseline
                            p_value = calculate_significance(sgc_scores, baseline_scores, alternative='less')
                            
                            # Power Analysis
                            power_stats = calculate_sample_size_requirement(sgc_scores, baseline_scores)
                            
                            # Log to global CSV list
                            SIGNIFICANCE_LOG.append({
                                'Dataset': dataset,
                                'Training': training_method,
                                'Model': model,
                                'SGC_Mean': row['sgc_ece_mean'] * 100,
                                'Best_Base': best_baseline_label,
                                'Base_Mean': best_baseline_mean * 100,
                                'P_Value': p_value,
                                'Is_Sig': p_value < 0.05,
                                'Effect_Size': power_stats['effect_size'],
                                'Current_N': power_stats['current_n'],
                                'Required_N': power_stats['n_required'],
                                'More_Needed': power_stats['more_needed'],
                                'Action': power_stats['action']
                            })
                            
                            if p_value < 0.05:
                                sgc_is_significant = True
                                # Add asterisk to SGC value (will be applied after color coding)
                                ece_formatted['sgc'] = f"{sgc}*"
        
        # Get top 3 rankings
        if ece_values:
            sorted_indices = np.argsort(ece_values)
            top3_labels = [ece_labels[i] for i in sorted_indices[:3]]
            
            # Apply color coding (preserve asterisk if present)
            if len(top3_labels) > 0:
                best_label = top3_labels[0]
                base_val = ece_formatted[best_label]
                # Remove asterisk temporarily, add color, then restore asterisk
                has_asterisk = base_val.endswith('*')
                if has_asterisk:
                    base_val = base_val[:-1]
                asterisk_str = "*" if has_asterisk else ""
                ece_formatted[best_label] = "\\textcolor{blue}{\\textbf{" + base_val + "}}" + asterisk_str
            if len(top3_labels) > 1:
                second_label = top3_labels[1]
                base_val = ece_formatted[second_label]
                has_asterisk = base_val.endswith('*')
                if has_asterisk:
                    base_val = base_val[:-1]
                asterisk_str = "*" if has_asterisk else ""
                ece_formatted[second_label] = "\\textcolor{teal}{\\textbf{" + base_val + "}}" + asterisk_str
            if len(top3_labels) > 2:
                third_label = top3_labels[2]
                base_val = ece_formatted[third_label]
                has_asterisk = base_val.endswith('*')
                if has_asterisk:
                    base_val = base_val[:-1]
                asterisk_str = "*" if has_asterisk else ""
                ece_formatted[third_label] = "\\textcolor{olive}{\\textbf{" + base_val + "}}" + asterisk_str
        
        # Ensure tulip_pub is in ece_formatted if it exists (for cases where it's not in top 3)
        if tulip_pub_key is not None and tulip_pub_key in TULIP_PUBLISHED and 'tulip_pub' not in ece_formatted:
            ece_formatted['tulip_pub'] = tulip_pub
        
        cells = [
            model,
            ece_formatted.get('ts', ts),
            ece_formatted.get('platt', platt),
            ece_formatted.get('isotonic', isotonic),
            ece_formatted.get('beta', beta),
            ece_formatted.get('dac_algorithm', dac_algorithm),
            ece_formatted.get('pixel_geo', pixel_geo),
            ece_formatted.get('tulip_pub', tulip_pub),
            ece_formatted.get('dac', dac),
            ece_formatted.get('tulip', tulip),
            ece_formatted.get('sgc', sgc),
            uncal,
            accuracy,
            str(num_seeds) if num_seeds > 0 else "---"
        ]
        lines.append(" & ".join(cells) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    
    return "\n".join(lines)


def generate_main_comparison_table(aggregated_df: pd.DataFrame) -> str:
    """Generate multiple comparison tables, one per (dataset, training_method) combination.
    Splits tables across two pages for better fit.
    """
    all_tables = []
    
    # Filter out SVHN dataset (irrelevant)
    filtered_df = aggregated_df[aggregated_df['dataset'] != 'svhn'].copy()
    
    # Group by dataset and training_method, sort for consistent ordering
    grouped = filtered_df.groupby(['dataset', 'training_method'])
    sorted_groups = sorted(grouped, key=lambda x: (x[0][0], x[0][1]))  # Sort by (dataset, training_method)
    
    for (dataset, training_method), group_df in sorted_groups:
        table_content = generate_single_comparison_table(group_df, dataset, training_method)
        all_tables.append(table_content)
    
    # Split tables into two groups (approximately half and half)
    total_tables = len(all_tables)
    mid_point = (total_tables + 1) // 2  # Split roughly in half
    
    first_half = all_tables[:mid_point]
    second_half = all_tables[mid_point:]
    
    lines = []
    lines.append("% Table 1: Main calibration comparison (Part 1)")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{rotating}, \\usepackage{xcolor}")
    
    # First page
    lines.append("\\begin{sidewaystable}[htbp]")
    for i, table_content in enumerate(first_half):
        if i > 0:
            lines.append("")  # Blank line between tables
        lines.append(table_content)
    lines.append("\\end{sidewaystable}")
    
    # Second page
    if second_half:
        lines.append("")
        lines.append("% Table 1: Main calibration comparison (Part 2)")
        lines.append("\\begin{sidewaystable}[htbp]")
        for i, table_content in enumerate(second_half):
            if i > 0:
                lines.append("")  # Blank line between tables
            lines.append(table_content)
        lines.append("\\end{sidewaystable}")
    
    return "\n".join(lines)


def extract_sgc_sensitivity_data(results: List[Dict[str, Any]], aggregate_overall: bool = False) -> pd.DataFrame:
    """Extract SGC sensitivity data across (L, d) configurations.
    
    Args:
        results: List of result dictionaries
        aggregate_overall: If True, aggregate across all configs. If False, keep per-configuration.
    
    Returns:
        DataFrame with sensitivity data. If aggregate_overall=False, includes training_method, dataset, model_name.
    """
    # Only use clean results
    clean_results = [r for r in results if is_clean_result(r)]
    
    rows = []
    
    dimensions = [64, 128, 256, 512, 1024]
    layer_counts = [2, 4, 5, 6, 8, 10, 12]
    
    for result in clean_results:
        exp_key = extract_experiment_key(result)
        training_method, dataset, model_name = exp_key
        
        for d in dimensions:
            for L in layer_counts:
                ece = extract_sgc_result(result, L=L, d=d)
                
                if ece is not None:
                    rows.append({
                        'training_method': training_method,
                        'dataset': dataset,
                        'model_name': model_name,
                        'seed': result.get('experiment_info', {}).get('seed', result.get('experiment_config', {}).get('seed', None)),
                        'target_dim': d,
                        'num_layers': L,
                        'ece': ece,
                    })
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        return df
    
    # Aggregate over seeds only (keep per-configuration)
    agg_dict = {
        'ece': ['mean', 'std', 'count']
    }
    
    if aggregate_overall:
        # Option: Aggregate across all configs (for overall trends)
        overall_grouped = df.groupby(['target_dim', 'num_layers']).agg(agg_dict)
        overall_grouped.columns = ['ece_mean', 'ece_std', 'ece_count']
        overall_grouped = overall_grouped.reset_index()
        return overall_grouped
    else:
        # Option B: Only aggregate over seeds, show one row per (dataset, model, training_method, target_dim, num_layers)
        grouped = df.groupby(['training_method', 'dataset', 'model_name', 'target_dim', 'num_layers']).agg(agg_dict)
        grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]
        grouped = grouped.reset_index()
        return grouped


def generate_sensitivity_table(sensitivity_df: pd.DataFrame, aggregate_overall: bool = False) -> str:
    """Generate Table 2: Hyperparameter sensitivity LaTeX.
    
    Args:
        sensitivity_df: DataFrame with sensitivity data
        aggregate_overall: If True, shows overall averages. If False, shows per-configuration.
    """
    if sensitivity_df.empty:
        return "% No sensitivity data available"
    
    lines = []
    lines.append("% Table 2: SGC hyperparameter sensitivity")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{colortbl}")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    
    if aggregate_overall:
        caption = "SGC hyperparameter sensitivity: ECE (\\%) across layer counts and dimensions (averaged across all datasets/models). Highlighted cell (yellow) is the chosen configuration L=6, d=256."
    else:
        # If per-configuration, we'll need to show multiple tables or aggregate for display
        # For now, aggregate for the table but note in caption
        caption = "SGC hyperparameter sensitivity: ECE (\\%) across layer counts and dimensions (averaged over seeds per configuration). Highlighted cell (yellow) is the chosen configuration L=6, d=256."
        # Aggregate for display if we have per-configuration data
        if 'training_method' in sensitivity_df.columns:
            sensitivity_df = sensitivity_df.groupby(['target_dim', 'num_layers']).agg({
                'ece_mean': 'mean',
                'ece_std': lambda x: np.sqrt(np.mean(x**2)),  # RMS of stds
                'ece_count': 'sum'
            }).reset_index()
    
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrrrrrr}")
    lines.append("\\toprule")
    lines.append("$d$ & $L=2$ & $L=4$ & $L=5$ & $L=6$ & $L=8$ & $L=10$ & $L=12$ \\\\")
    lines.append("\\midrule")
    
    dimensions = sorted(sensitivity_df['target_dim'].unique())
    layer_counts = [2, 4, 5, 6, 8, 10, 12]
    
    for d in dimensions:
        cells = [f"{int(d)}"]
        
        for L in layer_counts:
            mask = (sensitivity_df['target_dim'] == d) & (sensitivity_df['num_layers'] == L)
            matching = sensitivity_df[mask]
            
            if matching.empty:
                cells.append("---")
            else:
                row = matching.iloc[0]
                ece_str = format_ece_value(
                    row['ece_mean'],
                    row.get('ece_std', np.nan),
                    row.get('ece_count', None)
                )
                
                # Highlight L=6, d=256 using \cellcolor (works better with math mode)
                if d == 256 and L == 6:
                    # Use \cellcolor instead of \colorbox for better LaTeX compatibility
                    if " $\\pm$ " in ece_str:
                        # Split the value and highlight both parts
                        parts = ece_str.split(" $\\pm$ ")
                        ece_str = f"\\cellcolor{{yellow!50}}{{{parts[0]}}} $\\pm$ \\cellcolor{{yellow!50}}{{{parts[1]}}}"
                    else:
                        ece_str = f"\\cellcolor{{yellow!50}}{{{ece_str}}}"
                
                cells.append(ece_str)
        
        lines.append(" & ".join(cells) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_improvement_table(aggregated_df: pd.DataFrame) -> str:
    """Generate Table 3: ECE reduction LaTeX."""
    lines = []
    lines.append("% Table 3: ECE reduction from uncalibrated")
    lines.append("% Required packages: \\usepackage{booktabs}")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{ECE reduction from uncalibrated (\\% improvement). Best improvement per row in bold.}")
    lines.append("\\label{tab:improvements}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrrrrrr}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & TS & Platt & Isotonic & Beta & DAC & Pixel Geo & SGC(DAC) & SGC(TULIP) & SGC \\\\")
    lines.append("\\midrule")
    
    for _, row in aggregated_df.iterrows():
        dataset = row['dataset']
        model = row['model_name']
        
        uncal_mean = row.get('uncal_ece_mean', np.nan)
        
        if pd.isna(uncal_mean) or uncal_mean == 0:
            # Skip if no uncalibrated baseline
            continue
        
        # Calculate improvements (including all methods)
        improvements = {}
        
        for method in ['ts', 'platt', 'isotonic', 'beta', 'dac_algorithm', 'pixel_geo', 'dac', 'tulip', 'sgc']:
            method_ece_mean = row.get(f'{method}_ece_mean', np.nan)
            if pd.isna(method_ece_mean):
                improvements[method] = None
            else:
                improvement = ((uncal_mean - method_ece_mean) / uncal_mean) * 100
                improvements[method] = improvement
        
        # Format improvements
        ts_imp = f"{improvements['ts']:.1f}" if improvements['ts'] is not None else "---"
        platt_imp = f"{improvements['platt']:.1f}" if improvements['platt'] is not None else "---"
        isotonic_imp = f"{improvements['isotonic']:.1f}" if improvements['isotonic'] is not None else "---"
        beta_imp = f"{improvements['beta']:.1f}" if improvements['beta'] is not None else "---"
        dac_algorithm_imp = f"{improvements['dac_algorithm']:.1f}" if improvements['dac_algorithm'] is not None else "---"
        pixel_geo_imp = f"{improvements['pixel_geo']:.1f}" if improvements['pixel_geo'] is not None else "---"
        dac_imp = f"{improvements['dac']:.1f}" if improvements['dac'] is not None else "---"
        tulip_imp = f"{improvements['tulip']:.1f}" if improvements['tulip'] is not None else "---"
        sgc_imp = f"{improvements['sgc']:.1f}" if improvements['sgc'] is not None else "---"
        
        # Find best improvement
        valid_improvements = {k: v for k, v in improvements.items() if v is not None}
        if valid_improvements:
            best_method = max(valid_improvements, key=valid_improvements.get)
            
            if best_method == 'ts':
                ts_imp = f"\\textbf{{{ts_imp}}}"
            elif best_method == 'platt':
                platt_imp = f"\\textbf{{{platt_imp}}}"
            elif best_method == 'isotonic':
                isotonic_imp = f"\\textbf{{{isotonic_imp}}}"
            elif best_method == 'beta':
                beta_imp = f"\\textbf{{{beta_imp}}}"
            elif best_method == 'dac_algorithm':
                dac_algorithm_imp = f"\\textbf{{{dac_algorithm_imp}}}"
            elif best_method == 'pixel_geo':
                pixel_geo_imp = f"\\textbf{{{pixel_geo_imp}}}"
            elif best_method == 'dac':
                dac_imp = f"\\textbf{{{dac_imp}}}"
            elif best_method == 'tulip':
                tulip_imp = f"\\textbf{{{tulip_imp}}}"
            elif best_method == 'sgc':
                sgc_imp = f"\\textbf{{{sgc_imp}}}"
        
        cells = [dataset, model, ts_imp, platt_imp, isotonic_imp, beta_imp, dac_algorithm_imp, pixel_geo_imp, dac_imp, tulip_imp, sgc_imp]
        lines.append(" & ".join(cells) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_unified_comparison_table(aggregated_df: pd.DataFrame, training_method_filter: str = None) -> str:
    """Generate a single unified comparison table across all datasets and models.
    
    Args:
        aggregated_df: Aggregated DataFrame with all results
        training_method_filter: If provided, filter to this training method only (e.g., 'baseline_cross_entropy')
    """
    lines = []
    
    # Filter out SVHN
    df = aggregated_df[aggregated_df['dataset'] != 'svhn'].copy()
    
    # Optionally filter to specific training method
    if training_method_filter:
        df = df[df['training_method'] == training_method_filter]
    else:
        # If no filter, aggregate over training methods
        # Group by (dataset, model_name) and aggregate metrics
        # Helper function for RMS aggregation
        def rms_agg(x):
            """Compute root mean square of a Series."""
            if len(x) == 0:
                return np.nan
            return np.sqrt(np.mean(x**2))
        
        agg_dict = {}
        list_cols = []
        for col in df.columns:
            if col not in ['training_method', 'dataset', 'model_name']:
                if col.endswith('_values'):
                    # For list columns, collect them separately
                    list_cols.append(col)
                elif col.endswith('_mean'):
                    # For mean columns, compute mean
                    agg_dict[col] = 'mean'
                elif col.endswith('_std'):
                    # For std columns, compute RMS (root mean square)
                    agg_dict[col] = rms_agg
                elif col.endswith('_count'):
                    # For count columns, sum
                    agg_dict[col] = 'sum'
        
        if agg_dict:
            # First, get the standard mean/std/count dataframe
            df_agg = df.groupby(['dataset', 'model_name']).agg(agg_dict).reset_index()
            
            # Also aggregate the LISTS of values for significance testing
            # Flatten list of lists from all training methods into one big list per dataset/model
            if list_cols:
                def flatten_lists(x):
                    """Flatten a Series of lists into a single list."""
                    result = []
                    for item in x:
                        if isinstance(item, list):
                            result.extend(item)
                        elif pd.notna(item):
                            result.append(item)
                    return result
                
                df_lists = df.groupby(['dataset', 'model_name'])[list_cols].agg(flatten_lists).reset_index()
                
                # Merge lists back into the main df
                df = df_agg.merge(df_lists, on=['dataset', 'model_name'], how='left')
            else:
                df = df_agg
    
    lines.append("% Unified comparison table")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{xcolor}")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    
    # Determine if we should show TULIP published column (only for Cross Entropy)
    show_tulip_pub = (training_method_filter == 'baseline_cross_entropy')
    
    # Check for significance before building caption
    has_significant = False
    for _, row in df.iterrows():
        if not pd.isna(row.get('sgc_ece_mean')) and 'sgc_ece_values' in row:
            sgc_scores = row.get('sgc_ece_values', [])
            if isinstance(sgc_scores, list) and len(sgc_scores) >= 2:
                # Collect means for comparison
                ece_mean_dict = {}
                for method in ['ts', 'platt', 'isotonic', 'beta', 'dac_algorithm', 'pixel_geo', 'dac', 'sgc_dac', 'tulip', 'coord']:
                    if method == 'coord':
                        if not pd.isna(row.get('coord_ece_mean')):
                            ece_mean_dict[method] = row['coord_ece_mean']
                    elif method == 'sgc_dac':
                        if not pd.isna(row.get('sgc_dac_layers_ece_mean')):
                            ece_mean_dict[method] = row['sgc_dac_layers_ece_mean']
                    else:
                        if not pd.isna(row.get(f'{method}_ece_mean')):
                            ece_mean_dict[method] = row[f'{method}_ece_mean']
                if not pd.isna(row.get('sgc_ece_mean')):
                    ece_mean_dict['sgc'] = row['sgc_ece_mean']
                
                # Find best baseline to test against
                baseline_means = {k: v for k, v in ece_mean_dict.items() if k != 'sgc'}
                
                if baseline_means:
                    best_baseline_label = min(baseline_means, key=baseline_means.get)
                    best_baseline_mean = baseline_means[best_baseline_label]
                    
                    # Only test if SGC is winning
                    if row['sgc_ece_mean'] < best_baseline_mean:
                        # Get baseline scores - handle coord and sgc_dac specially
                        if best_baseline_label == 'coord':
                            baseline_scores = row.get('coord_ece_values', [])
                        elif best_baseline_label == 'sgc_dac':
                            baseline_scores = row.get('sgc_dac_layers_ece_values', [])
                        else:
                            baseline_scores = row.get(f'{best_baseline_label}_ece_values', [])
                        if isinstance(baseline_scores, list) and len(baseline_scores) >= 2:
                            p_value = calculate_significance(sgc_scores, baseline_scores, alternative='less')
                            if p_value < 0.05:
                                has_significant = True
                                break
    
    # Build caption without f-string escaping issues
    if training_method_filter:
        training_display = get_training_method_display_name(training_method_filter)
        caption_text = f"ECE (\\%) comparison across datasets and models ({training_display}). Values shown as mean $\\pm$ 95\\% CI."
    else:
        caption_text = "ECE (\\%) comparison across datasets and models (aggregated over training methods). Values shown as mean $\\pm$ 95\\% CI."
    
    if has_significant:
        caption_text += "\\footnote{* Indicates statistical significance (p < 0.05) against the best baseline.}"
    
    lines.append(f"\\caption{{{caption_text} \\textcolor{{blue}}{{\\textbf{{Best}}}}, \\textcolor{{teal}}{{\\textbf{{2nd}}}}, \\textcolor{{olive}}{{\\textbf{{3rd}}}}.}}")
    
    lines.append("\\label{tab:unified_comparison}")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\small")
    
    # Conditionally include TULIP(pub) column
    if show_tulip_pub:
        lines.append("\\begin{tabular}{llccccccccccccc}")
        lines.append("\\toprule")
        lines.append("Dataset & Model & Uncal & TS & Platt & Isotonic & Beta & DAC & Pixel Geo & TULIP(pub) & SGC(DAC) & SGC(TULIP) & SGC & SGC Lite & Acc & Seeds \\\\")
    else:
        lines.append("\\begin{tabular}{llcccccccccccc}")
        lines.append("\\toprule")
        lines.append("Dataset & Model & Uncal & TS & Platt & Isotonic & Beta & DAC & Pixel Geo & SGC(DAC) & SGC(TULIP) & SGC & SGC Lite & Acc & Seeds \\\\")
    lines.append("\\midrule")
    
    # Sort by dataset then model
    df = df.sort_values(['dataset', 'model_name'])
    
    current_dataset = None
    
    for _, row in df.iterrows():
        dataset = row['dataset']
        model = row['model_name']
        
        # Add horizontal line between datasets
        if current_dataset is not None and dataset != current_dataset:
            lines.append("\\midrule")
        current_dataset = dataset
        
        # Extract all ECE values
        methods = {
            'uncal': row.get('uncal_ece_mean', np.nan),
            'ts': row.get('ts_ece_mean', np.nan),
            'platt': row.get('platt_ece_mean', np.nan),
            'isotonic': row.get('isotonic_ece_mean', np.nan),
            'beta': row.get('beta_ece_mean', np.nan),
            'dac_algorithm': row.get('dac_algorithm_ece_mean', np.nan),
            'pixel_geo': row.get('pixel_geo_ece_mean', np.nan),
            'dac': row.get('dac_ece_mean', np.nan),
            'sgc_dac': row.get('sgc_dac_layers_ece_mean', np.nan),  # SGC with DAC layer selection
            'tulip': row.get('tulip_ece_mean', np.nan),
            'sgc': row.get('sgc_ece_mean', np.nan),
            'coord': row.get('coord_ece_mean', np.nan),
        }
        
        # TULIP published results (only for Cross Entropy filter)
        tulip_pub_mean, tulip_pub_std = None, None
        if training_method_filter == 'baseline_cross_entropy':
            tulip_pub_key = (dataset.lower(), 'baseline_cross_entropy', model)
            if tulip_pub_key in TULIP_PUBLISHED:
                tulip_pub_mean, tulip_pub_std = TULIP_PUBLISHED[tulip_pub_key]
        
        # Format TULIP published
        if tulip_pub_mean is not None:
            tulip_pub_formatted = format_ece_value(tulip_pub_mean, std_val=tulip_pub_std, count=None, inline_math=True)
        else:
            tulip_pub_formatted = "---"
        
        # Format each value
        formatted = {}
        for method, mean_val in methods.items():
            if method == 'coord':
                # Coordinate uses different column naming
                std_val = row.get('coord_ece_std', np.nan)
                count = row.get('coord_ece_count', None)
            elif method == 'sgc_dac':
                # SGC(DAC) uses sgc_dac_layers column naming
                std_val = row.get('sgc_dac_layers_ece_std', np.nan)
                count = row.get('sgc_dac_layers_ece_count', None)
            else:
                std_val = row.get(f'{method}_ece_std', np.nan)
                count = row.get(f'{method}_ece_count', None)
            formatted[method] = format_ece_value(mean_val, std_val=std_val, count=count, inline_math=True)
        
        # Add TULIP published to formatted dict (only if we're showing the column)
        if show_tulip_pub:
            formatted['tulip_pub'] = tulip_pub_formatted
        
        # Statistical significance testing: SGC vs Best Baseline
        sgc_is_significant = False
        if not pd.isna(row.get('sgc_ece_mean')) and 'sgc_ece_values' in row:
            sgc_scores = row.get('sgc_ece_values', [])
            if isinstance(sgc_scores, list) and len(sgc_scores) >= 2:
                # Collect means for comparison
                ece_mean_dict = {}
                for method in ['ts', 'platt', 'isotonic', 'beta', 'dac_algorithm', 'pixel_geo', 'dac', 'sgc_dac', 'tulip', 'coord']:
                    if method == 'coord':
                        if not pd.isna(row.get('coord_ece_mean')):
                            ece_mean_dict[method] = row['coord_ece_mean']
                    elif method == 'sgc_dac':
                        if not pd.isna(row.get('sgc_dac_layers_ece_mean')):
                            ece_mean_dict[method] = row['sgc_dac_layers_ece_mean']
                    else:
                        if not pd.isna(row.get(f'{method}_ece_mean')):
                            ece_mean_dict[method] = row[f'{method}_ece_mean']
                if not pd.isna(row.get('sgc_ece_mean')):
                    ece_mean_dict['sgc'] = row['sgc_ece_mean']
                
                # Find best baseline to test against
                baseline_means = {k: v for k, v in ece_mean_dict.items() if k != 'sgc'}
                
                if baseline_means:
                    best_baseline_label = min(baseline_means, key=baseline_means.get)
                    best_baseline_mean = baseline_means[best_baseline_label]
                    
                    # Only test if SGC is winning
                    if row['sgc_ece_mean'] < best_baseline_mean:
                        # Get baseline scores - handle coord and sgc_dac specially
                        if best_baseline_label == 'coord':
                            baseline_scores = row.get('coord_ece_values', [])
                        elif best_baseline_label == 'sgc_dac':
                            baseline_scores = row.get('sgc_dac_layers_ece_values', [])
                        else:
                            baseline_scores = row.get(f'{best_baseline_label}_ece_values', [])
                        if isinstance(baseline_scores, list) and len(baseline_scores) >= 2:
                            p_value = calculate_significance(sgc_scores, baseline_scores, alternative='less')
                            
                            # Power Analysis
                            power_stats = calculate_sample_size_requirement(sgc_scores, baseline_scores)
                            
                            # Determine training method for logging (use filter or 'aggregated')
                            training_method_for_log = training_method_filter if training_method_filter else 'aggregated'
                            
                            # Log to global CSV list
                            SIGNIFICANCE_LOG.append({
                                'Dataset': dataset,
                                'Training': training_method_for_log,
                                'Model': model,
                                'SGC_Mean': row['sgc_ece_mean'] * 100,
                                'Best_Base': best_baseline_label,
                                'Base_Mean': best_baseline_mean * 100,
                                'P_Value': p_value,
                                'Is_Sig': p_value < 0.05,
                                'Effect_Size': power_stats['effect_size'],
                                'Current_N': power_stats['current_n'],
                                'Required_N': power_stats['n_required'],
                                'More_Needed': power_stats['more_needed'],
                                'Action': power_stats['action']
                            })
                            
                            if p_value < 0.05:
                                sgc_is_significant = True
                                # Add asterisk to SGC formatted string
                                formatted['sgc'] = formatted['sgc'] + "*"
        
        # Format accuracy
        accuracy = format_accuracy_value(
            row.get('uncal_acc_mean', np.nan),
            row.get('uncal_acc_std', np.nan),
            row.get('uncal_acc_count', None),
            inline_math=True
        )
        
        # Find top 3 (exclude uncal from ranking)
        ranking_methods = {k: v for k, v in methods.items() if k != 'uncal' and not pd.isna(v)}
        # Include TULIP published in ranking if available
        if tulip_pub_mean is not None:
            ranking_methods['tulip_pub'] = tulip_pub_mean
        # Include coord in ranking
        if 'coord' in methods and not pd.isna(methods['coord']):
            ranking_methods['coord'] = methods['coord']
        if ranking_methods:
            sorted_methods = sorted(ranking_methods.items(), key=lambda x: x[1])
            top3 = [m[0] for m in sorted_methods[:3]]
            
            # Color code top 3, but only if the key exists in formatted
            # Preserve asterisk if present
            if len(top3) > 0 and top3[0] in formatted:
                base_val = formatted[top3[0]]
                has_asterisk = base_val.endswith('*')
                if has_asterisk:
                    base_val = base_val[:-1]
                asterisk_str = "*" if has_asterisk else ""
                formatted[top3[0]] = "\\textcolor{blue}{\\textbf{" + base_val + "}}" + asterisk_str
            if len(top3) > 1 and top3[1] in formatted:
                base_val = formatted[top3[1]]
                has_asterisk = base_val.endswith('*')
                if has_asterisk:
                    base_val = base_val[:-1]
                asterisk_str = "*" if has_asterisk else ""
                formatted[top3[1]] = "\\textcolor{teal}{\\textbf{" + base_val + "}}" + asterisk_str
            if len(top3) > 2 and top3[2] in formatted:
                base_val = formatted[top3[2]]
                has_asterisk = base_val.endswith('*')
                if has_asterisk:
                    base_val = base_val[:-1]
                asterisk_str = "*" if has_asterisk else ""
                formatted[top3[2]] = "\\textcolor{olive}{\\textbf{" + base_val + "}}" + asterisk_str
        
        # Get seed count
        seed_counts = [row.get(f'{m}_ece_count', 0) for m in methods.keys()]
        seed_counts = [int(c) for c in seed_counts if pd.notna(c) and c > 0]
        num_seeds = max(seed_counts) if seed_counts else 0
        
        # Build cells list conditionally based on whether we show TULIP published
        if show_tulip_pub:
            cells = [
                dataset.upper(),
                model,
                formatted['uncal'],
                formatted['ts'],
                formatted.get('platt', '---'),  # Platt Scaling
                formatted['isotonic'],
                formatted['beta'],
                formatted['dac_algorithm'],
                formatted['pixel_geo'],
                formatted.get('tulip_pub', tulip_pub_formatted),  # TULIP published
                formatted.get('sgc_dac', '---'),  # SGC(DAC)
                formatted.get('tulip', '---'),  # SGC(TULIP)
                formatted['sgc'],
                formatted.get('coord', '---'),  # Coord
                accuracy,
                str(num_seeds) if num_seeds > 0 else "---"
            ]
        else:
            cells = [
                dataset.upper(),
                model,
                formatted['uncal'],
                formatted['ts'],
                formatted.get('platt', '---'),  # Platt Scaling
                formatted['isotonic'],
                formatted['beta'],
                formatted['dac_algorithm'],
                formatted['pixel_geo'],
                formatted.get('sgc_dac', '---'),  # SGC(DAC)
                formatted.get('tulip', '---'),  # SGC(TULIP)
                formatted['sgc'],
                formatted.get('coord', '---'),  # Coord
                accuracy,
                str(num_seeds) if num_seeds > 0 else "---"
            ]
        lines.append(" & ".join(cells) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_ece_accuracy_comparison_table(aggregated_df: pd.DataFrame, training_method_filter: str = None) -> str:
    """Generate a table showing ECE and Accuracy side-by-side for each method.
    
    This table demonstrates that calibration methods don't affect accuracy.
    
    Args:
        aggregated_df: Aggregated DataFrame with all results
        training_method_filter: If provided, filter to this training method only
    """
    lines = []
    
    # Filter out SVHN
    df = aggregated_df[aggregated_df['dataset'] != 'svhn'].copy()
    
    # Optionally filter to specific training method
    if training_method_filter:
        df = df[df['training_method'] == training_method_filter]
    else:
        # If no filter, aggregate over training methods
        def rms_agg(x):
            """Compute root mean square of a Series."""
            if len(x) == 0:
                return np.nan
            return np.sqrt(np.mean(x**2))
        
        agg_dict = {}
        for col in df.columns:
            if col not in ['training_method', 'dataset', 'model_name']:
                if col.endswith('_mean'):
                    agg_dict[col] = 'mean'
                elif col.endswith('_std'):
                    agg_dict[col] = rms_agg
                elif col.endswith('_count'):
                    agg_dict[col] = 'sum'
        
        if agg_dict:
            df = df.groupby(['dataset', 'model_name']).agg(agg_dict).reset_index()
    
    lines.append("% ECE and Accuracy comparison table")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{multirow}, \\usepackage{xcolor}")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    
    if training_method_filter:
        training_display = get_training_method_display_name(training_method_filter)
        caption_text = f"ECE (\\%) and Accuracy (\\%) comparison across datasets and models ({training_display}). Each method shows ECE / Accuracy side-by-side to demonstrate that calibration does not affect accuracy. {{\\footnotesize \\textcolor{{blue}}{{\\textbf{{Best}}}}, \\textcolor{{teal}}{{\\textbf{{2nd}}}}, \\textcolor{{olive}}{{\\textbf{{3rd}}}}}}"
    else:
        caption_text = "ECE (\\%) and Accuracy (\\%) comparison across datasets and models (aggregated over training methods). Each method shows ECE / Accuracy side-by-side to demonstrate that calibration does not affect accuracy. {{\\footnotesize \\textcolor{{blue}}{{\\textbf{{Best}}}}, \\textcolor{{teal}}{{\\textbf{{2nd}}}}, \\textcolor{{olive}}{{\\textbf{{3rd}}}}}}"
    
    lines.append(f"\\caption{{{caption_text}}}")
    lines.append("\\label{tab:ece_accuracy_comparison}")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\small")
    
    # Build table with ECE/Accuracy pairs for each method
    # Methods: Uncal, TS, Isotonic, Beta, DAC, Pixel Geo, SGC(DAC), SGC(TULIP), SGC, Coord
    methods = [
        ('Uncal', 'uncal'),
        ('TS', 'ts'),
        ('Isotonic', 'isotonic'),
        ('Beta', 'beta'),
        ('DAC', 'dac_algorithm'),
        ('Pixel Geo', 'pixel_geo'),
        ('SGC(DAC)', 'sgc_dac_layers'),
        ('SGC(TULIP)', 'tulip'),
        ('SGC', 'sgc'),
        ('Coord', 'coord'),
    ]
    
    # Calculate number of columns: Dataset, Model, then for each method: ECE/Acc pair
    num_cols = 2 + len(methods) * 2  # Dataset, Model, then ECE+Acc for each method
    lines.append(f"\\begin{{tabular}}{{ll" + "r" * (num_cols - 2) + "}}")
    lines.append("\\toprule")
    
    # Header row 1: Method names
    header1 = ["Dataset", "Model"]
    for method_name, _ in methods:
        header1.append(f"\\multicolumn{{2}}{{c}}{{{method_name}}}")
    lines.append(" & ".join(header1) + " \\\\")
    
    # Header row 2: ECE / Acc
    header2 = ["", ""]
    for _ in methods:
        header2.append("ECE")
        header2.append("Acc")
    # Update cmidrules to include the new Coord column (now 10 methods = 20 columns after Dataset/Model)
    lines.append("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}\\cmidrule(lr){13-14}\\cmidrule(lr){15-16}\\cmidrule(lr){17-18}\\cmidrule(lr){19-20}\\cmidrule(lr){21-22}")
    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\midrule")
    
    # Sort by dataset then model
    df = df.sort_values(['dataset', 'model_name'])
    
    current_dataset = None
    
    for _, row in df.iterrows():
        dataset = row['dataset']
        model = row['model_name']
        
        # Add horizontal line between datasets
        if current_dataset is not None and dataset != current_dataset:
            lines.append("\\midrule")
        current_dataset = dataset
        
        # First, collect all ECE values and find top 3 (excluding uncalibrated)
        ece_values = []
        ece_methods = []
        ece_formatted = {}
        
        for method_name, method_key in methods:
            # Skip uncalibrated in ranking
            if method_key == 'uncal':
                continue
            
            # Handle coord and sgc_dac_layers specially
            if method_key == 'coord':
                ece_mean = row.get('coord_ece_mean', np.nan)
            elif method_key == 'sgc_dac_layers':
                ece_mean = row.get('sgc_dac_layers_ece_mean', np.nan)
            else:
                ece_mean = row.get(f'{method_key}_ece_mean', np.nan)
            if pd.notna(ece_mean):
                ece_values.append(ece_mean)
                ece_methods.append(method_key)
        
        # Find top 3 ECE values
        if ece_values:
            sorted_indices = np.argsort(ece_values)
            top3_indices = sorted_indices[:3]
            top3_methods = [ece_methods[i] for i in top3_indices]
        else:
            top3_methods = []
        
        cells = [dataset.upper(), model]
        
        # For each method, add ECE and Accuracy
        for method_name, method_key in methods:
            # Format ECE - handle coord and sgc_dac_layers specially
            if method_key == 'coord':
                ece_mean = row.get('coord_ece_mean', np.nan)
                ece_std = row.get('coord_ece_std', np.nan)
                ece_count = row.get('coord_ece_count', None)
            elif method_key == 'sgc_dac_layers':
                ece_mean = row.get('sgc_dac_layers_ece_mean', np.nan)
                ece_std = row.get('sgc_dac_layers_ece_std', np.nan)
                ece_count = row.get('sgc_dac_layers_ece_count', None)
            else:
                ece_mean = row.get(f'{method_key}_ece_mean', np.nan)
                ece_std = row.get(f'{method_key}_ece_std', np.nan)
                ece_count = row.get(f'{method_key}_ece_count', None)
            ece_str = format_ece_value(ece_mean, ece_std, ece_count, inline_math=True)
            
            # Apply color coding for top 3 (excluding uncalibrated)
            if method_key in top3_methods and method_key != 'uncal':
                rank = top3_methods.index(method_key)
                if rank == 0:
                    # Best (blue)
                    ece_str = f"\\textcolor{{blue}}{{\\textbf{{{ece_str}}}}}"
                elif rank == 1:
                    # 2nd (teal)
                    ece_str = f"\\textcolor{{teal}}{{\\textbf{{{ece_str}}}}}"
                elif rank == 2:
                    # 3rd (olive)
                    ece_str = f"\\textcolor{{olive}}{{\\textbf{{{ece_str}}}}}"
            
            # Format Accuracy - handle coord and sgc_dac_layers specially
            if method_key == 'coord':
                acc_mean = row.get('coord_acc_mean', np.nan)
                acc_std = row.get('coord_acc_std', np.nan)
                acc_count = row.get('coord_acc_count', None)
            elif method_key == 'sgc_dac_layers':
                acc_mean = row.get('sgc_dac_layers_acc_mean', np.nan)
                acc_std = row.get('sgc_dac_layers_acc_std', np.nan)
                acc_count = row.get('sgc_dac_layers_acc_count', None)
            else:
                acc_mean = row.get(f'{method_key}_acc_mean', np.nan)
                acc_std = row.get(f'{method_key}_acc_std', np.nan)
                acc_count = row.get(f'{method_key}_acc_count', None)
            acc_str = format_accuracy_value(acc_mean, acc_std, acc_count, inline_math=True)
            
            cells.append(ece_str)
            cells.append(acc_str)
        
        lines.append(" & ".join(cells) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_sgc_vs_coord_comparison_table(aggregated_df: pd.DataFrame, relative_margin_percent: float = 0.10, absolute_margin_floor: float = 0.001) -> str:
    """Compare SGC against Coordinate ECE with TOST equivalence testing.
    
    Args:
        aggregated_df: DataFrame with aggregated results
        relative_margin_percent: Relative margin as fraction of mean ECE (default 0.10 = 10%)
        absolute_margin_floor: Absolute margin floor in same units as ECE (default 0.005 = 0.5pp)
    """
    lines = []

    if aggregated_df.empty:
        return "% No data available for SGC vs Coord comparison"

    df = aggregated_df.copy()

    # Aggregate over training methods to get a single row per dataset/model
    def rms_agg(x):
        if len(x) == 0:
            return np.nan
        return np.sqrt(np.mean(x**2))

    def flatten_lists(series):
        combined = []
        for item in series:
            if isinstance(item, list):
                combined.extend(item)
        return combined

    agg_dict = {
        'sgc_ece_mean': 'mean',
        'sgc_ece_std': rms_agg,
        'sgc_ece_count': 'sum',
        'coord_ece_mean': 'mean',
        'coord_ece_std': rms_agg,
        'coord_ece_count': 'sum',
        'sgc_ece_values': flatten_lists,
        'coord_ece_values': flatten_lists,
    }

    present_cols = set(df.columns)
    agg_dict = {k: v for k, v in agg_dict.items() if k in present_cols}

    if {'dataset', 'model_name', 'training_method'}.issubset(df.columns):
        df = df.groupby(['dataset', 'model_name']).agg(agg_dict).reset_index()

    df = df[
        df.get('sgc_ece_mean').notna()
        & df.get('coord_ece_mean').notna()
    ]

    # Filter out SVHN dataset
    if 'dataset' in df.columns:
        df = df[df['dataset'].astype(str).str.lower() != 'svhn']

    if df.empty:
        return "% No overlapping SGC and Coord results for comparison"

    lines.append("% SGC vs Coordinate calibration comparison table")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{SGC vs Coordinate calibration comparison (ECE \\%; lower is better). We report mean$\\pm$std over $n$ paired trials. We test for equivalence using Paired TOST with a hybrid margin: $\\delta = \\max({relative_margin_percent*100:.0f}\\%\\text{{ of mean}}, {absolute_margin_floor*100:.1f}\\text{{ pp}})$. TOST $p<0.05$ indicates the mean difference falls significantly within $[-\\delta, +\\delta]$. $d_z$ denotes paired effect size.}}")
    lines.append("\\label{tab:sgc_vs_coord}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrrcc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & SGC & Coord & Diff (pp) & Diff 95\\% CI & $d_z$ & TOST $p$ & Result \\\\")
    lines.append("\\midrule")

    equivalence_count = 0

    for _, row in df.sort_values(['dataset', 'model_name']).iterrows():
        dataset = row['dataset']
        model = row['model_name']

        sgc_mean = row.get('sgc_ece_mean', np.nan)
        sgc_std = row.get('sgc_ece_std', np.nan)
        sgc_count = row.get('sgc_ece_count', None)
        coord_mean = row.get('coord_ece_mean', np.nan)
        coord_std = row.get('coord_ece_std', np.nan)
        coord_count = row.get('coord_ece_count', None)

        sgc_str = format_ece_value(sgc_mean, sgc_std, sgc_count, inline_math=True)
        coord_str = format_ece_value(coord_mean, coord_std, coord_count, inline_math=True)

        sgc_vals = [v for v in row.get('sgc_ece_values', []) if pd.notna(v)]
        coord_vals = [v for v in row.get('coord_ece_values', []) if pd.notna(v)]

        # Calculate hybrid margin: max(relative margin, absolute floor)
        # Use the baseline (coord) mean for margin calculation
        baseline_mean = coord_mean if pd.notna(coord_mean) else sgc_mean if pd.notna(sgc_mean) else 0.01
        if pd.notna(baseline_mean) and baseline_mean > 0:
            relative_margin = baseline_mean * relative_margin_percent
            equivalence_margin = max(relative_margin, absolute_margin_floor)
        else:
            equivalence_margin = absolute_margin_floor

        # Perform TOST test
        tost_result = perform_tost_test(sgc_vals, coord_vals, equivalence_margin=equivalence_margin)
        
        mean_diff = tost_result['mean_diff']
        diff_str = f"{mean_diff*100:+.2f}" if pd.notna(mean_diff) else "---"
        
        ci_lower = tost_result['ci_lower']
        ci_upper = tost_result['ci_upper']
        ci_str = f"[{ci_lower*100:+.2f}, {ci_upper*100:+.2f}]" if pd.notna(ci_lower) and pd.notna(ci_upper) else "---"
        
        dz = tost_result['dz']
        dz_str = f"{dz:+.2f}" if pd.notna(dz) else "---"
        
        tost_p = tost_result['tost_pvalue']
        tost_p_str = f"{tost_p:.3f}" if tost_p >= 0.001 else "<0.001"
        
        result_str = determine_result(
            tost_p, mean_diff, equivalence_margin, ci_lower, ci_upper
        )
        # Replace generic method names with actual method names
        result_str = result_str.replace("Method 1", "SGC").replace("Method 2", "Coord")
        
        if tost_result['equivalence_established']:
            equivalence_count += 1

        min_margin_pct = tost_result.get('min_margin_for_equiv_pct', np.nan)
        SIGNIFICANCE_LOG.append({
            'Comparison': 'SGC_vs_Coord',
            'Dataset': dataset,
            'Training': 'aggregated',
            'Model': model,
            'SGC_Mean': sgc_mean * 100 if pd.notna(sgc_mean) else np.nan,
            'Base_Mean': coord_mean * 100 if pd.notna(coord_mean) else np.nan,
            'TOST_P_Value': tost_p,
            'Effect_Size': dz,
            'Min_Margin_Pct': min_margin_pct,
            'Conclusion': 'Equivalent' if tost_result['equivalence_established'] else 'Different',
        })

        cells = [
            dataset.upper(),
            model,
            sgc_str,
            coord_str,
            diff_str,
            ci_str,
            dz_str,
            tost_p_str,
            result_str,
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{2}}{{l}}{{Summary}} & \\multicolumn{{7}}{{r}}{{Equivalence established: {equivalence_count}/{len(df)}}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def generate_sgc_vs_sgc_dac_comparison_table(aggregated_df: pd.DataFrame, relative_margin_percent: float = 0.10, absolute_margin_floor: float = 0.001) -> str:
    """Compare SGC against SGC(DAC) ECE with TOST equivalence testing.
    
    Args:
        aggregated_df: DataFrame with aggregated results
        relative_margin_percent: Relative margin as fraction of mean ECE (default 0.10 = 10%)
        absolute_margin_floor: Absolute margin floor in same units as ECE (default 0.005 = 0.5pp)
    """
    lines = []

    if aggregated_df.empty:
        return "% No data available for SGC vs SGC(DAC) comparison"

    df = aggregated_df.copy()

    # Aggregate over training methods to get a single row per dataset/model
    def rms_agg(x):
        if len(x) == 0:
            return np.nan
        return np.sqrt(np.mean(x**2))

    def flatten_lists(series):
        combined = []
        for item in series:
            if isinstance(item, list):
                combined.extend(item)
        return combined

    agg_dict = {
        'sgc_ece_mean': 'mean',
        'sgc_ece_std': rms_agg,
        'sgc_ece_count': 'sum',
        'sgc_dac_layers_ece_mean': 'mean',
        'sgc_dac_layers_ece_std': rms_agg,
        'sgc_dac_layers_ece_count': 'sum',
        'sgc_ece_values': flatten_lists,
        'sgc_dac_layers_ece_values': flatten_lists,
    }

    present_cols = set(df.columns)
    agg_dict = {k: v for k, v in agg_dict.items() if k in present_cols}

    if {'dataset', 'model_name', 'training_method'}.issubset(df.columns):
        df = df.groupby(['dataset', 'model_name']).agg(agg_dict).reset_index()

    df = df[
        df.get('sgc_ece_mean').notna()
        & df.get('sgc_dac_layers_ece_mean').notna()
    ]

    # Filter out SVHN dataset
    if 'dataset' in df.columns:
        df = df[df['dataset'].astype(str).str.lower() != 'svhn']

    if df.empty:
        return "% No overlapping SGC and SGC(DAC) results for comparison"

    lines.append("% SGC vs SGC(DAC) comparison table")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{SGC vs SGC(DAC) calibration comparison (ECE \\%; lower is better). SGC(DAC) uses DAC-pre-selected layers. We report mean$\\pm$std over $n$ paired trials. We test for equivalence using Paired TOST with a hybrid margin: $\\delta = \\max({relative_margin_percent*100:.0f}\\%\\text{{ of mean}}, {absolute_margin_floor*100:.1f}\\text{{ pp}})$. TOST $p<0.05$ indicates the mean difference falls significantly within $[-\\delta, +\\delta]$. $d_z$ denotes paired effect size.}}")
    lines.append("\\label{tab:sgc_vs_sgc_dac}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrrcc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & SGC & SGC(DAC) & Diff (pp) & Diff 95\\% CI & $d_z$ & TOST $p$ & Result \\\\")
    lines.append("\\midrule")

    equivalence_count = 0

    for _, row in df.sort_values(['dataset', 'model_name']).iterrows():
        dataset = row['dataset']
        model = row['model_name']

        sgc_mean = row.get('sgc_ece_mean', np.nan)
        sgc_std = row.get('sgc_ece_std', np.nan)
        sgc_count = row.get('sgc_ece_count', None)
        sgc_dac_mean = row.get('sgc_dac_layers_ece_mean', np.nan)
        sgc_dac_std = row.get('sgc_dac_layers_ece_std', np.nan)
        sgc_dac_count = row.get('sgc_dac_layers_ece_count', None)

        sgc_str = format_ece_value(sgc_mean, sgc_std, sgc_count, inline_math=True)
        sgc_dac_str = format_ece_value(sgc_dac_mean, sgc_dac_std, sgc_dac_count, inline_math=True)

        sgc_vals = [v for v in row.get('sgc_ece_values', []) if pd.notna(v)]
        sgc_dac_vals = [v for v in row.get('sgc_dac_layers_ece_values', []) if pd.notna(v)]

        # Calculate hybrid margin: max(relative margin, absolute floor)
        # Use the baseline (sgc_dac) mean for margin calculation
        baseline_mean = sgc_dac_mean if pd.notna(sgc_dac_mean) else sgc_mean if pd.notna(sgc_mean) else 0.01
        if pd.notna(baseline_mean) and baseline_mean > 0:
            relative_margin = baseline_mean * relative_margin_percent
            equivalence_margin = max(relative_margin, absolute_margin_floor)
        else:
            equivalence_margin = absolute_margin_floor

        # Perform TOST test
        tost_result = perform_tost_test(sgc_vals, sgc_dac_vals, equivalence_margin=equivalence_margin)
        
        mean_diff = tost_result['mean_diff']
        diff_str = f"{mean_diff*100:+.2f}" if pd.notna(mean_diff) else "---"
        
        ci_lower = tost_result['ci_lower']
        ci_upper = tost_result['ci_upper']
        ci_str = f"[{ci_lower*100:+.2f}, {ci_upper*100:+.2f}]" if pd.notna(ci_lower) and pd.notna(ci_upper) else "---"
        
        dz = tost_result['dz']
        dz_str = f"{dz:+.2f}" if pd.notna(dz) else "---"
        
        tost_p = tost_result['tost_pvalue']
        tost_p_str = f"{tost_p:.3f}" if tost_p >= 0.001 else "<0.001"
        
        result_str = determine_result(
            tost_p, mean_diff, equivalence_margin, ci_lower, ci_upper
        )
        # Replace generic method names with actual method names
        result_str = result_str.replace("Method 1", "SGC").replace("Method 2", "SGC(DAC)")
        
        if tost_result['equivalence_established']:
            equivalence_count += 1

        min_margin_pct = tost_result.get('min_margin_for_equiv_pct', np.nan)
        SIGNIFICANCE_LOG.append({
            'Comparison': 'SGC_vs_SGC_DAC',
            'Dataset': dataset,
            'Training': 'aggregated',
            'Model': model,
            'SGC_Mean': sgc_mean * 100 if pd.notna(sgc_mean) else np.nan,
            'Base_Mean': sgc_dac_mean * 100 if pd.notna(sgc_dac_mean) else np.nan,
            'TOST_P_Value': tost_p,
            'Effect_Size': dz,
            'Min_Margin_Pct': min_margin_pct,
            'Conclusion': 'Equivalent' if tost_result['equivalence_established'] else 'Different',
        })

        cells = [
            dataset.upper(),
            model,
            sgc_str,
            sgc_dac_str,
            diff_str,
            ci_str,
            dz_str,
            tost_p_str,
            result_str,
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{2}}{{l}}{{Summary}} & \\multicolumn{{7}}{{r}}{{Equivalence established: {equivalence_count}/{len(df)}}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def generate_dac_original_vs_random_comparison_table(aggregated_df: pd.DataFrame, relative_margin_percent: float = 0.10, absolute_margin_floor: float = 0.001) -> str:
    """Compare DAC Original against DAC Random Layers ECE with TOST equivalence testing.
    
    Args:
        aggregated_df: DataFrame with aggregated results
        relative_margin_percent: Relative margin as fraction of mean ECE (default 0.10 = 10%)
        absolute_margin_floor: Absolute margin floor in same units as ECE (default 0.005 = 0.5pp)
    """
    lines = []

    if aggregated_df.empty:
        return "% No data available for DAC Original vs DAC Random comparison"

    df = aggregated_df.copy()

    # Aggregate over training methods to get a single row per dataset/model
    def rms_agg(x):
        if len(x) == 0:
            return np.nan
        return np.sqrt(np.mean(x**2))

    def flatten_lists(series):
        combined = []
        for item in series:
            if isinstance(item, list):
                combined.extend(item)
        return combined

    agg_dict = {
        'dac_algorithm_ece_mean': 'mean',
        'dac_algorithm_ece_std': rms_agg,
        'dac_algorithm_ece_count': 'sum',
        'dac_random_ece_mean': 'mean',
        'dac_random_ece_std': rms_agg,
        'dac_random_ece_count': 'sum',
        'dac_algorithm_ece_values': flatten_lists,
        'dac_random_ece_values': flatten_lists,
    }

    present_cols = set(df.columns)
    agg_dict = {k: v for k, v in agg_dict.items() if k in present_cols}

    if {'dataset', 'model_name', 'training_method'}.issubset(df.columns):
        df = df.groupby(['dataset', 'model_name']).agg(agg_dict).reset_index()

    df = df[
        df.get('dac_algorithm_ece_mean').notna()
        & df.get('dac_random_ece_mean').notna()
    ]

    # Filter out SVHN dataset
    if 'dataset' in df.columns:
        df = df[df['dataset'].astype(str).str.lower() != 'svhn']

    if df.empty:
        return "% No overlapping DAC Original and DAC Random results for comparison"

    lines.append("% DAC Original vs DAC Random Layers comparison table")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{DAC Original vs DAC Random Layers calibration comparison (ECE \\%; lower is better). DAC Random uses random layer selection instead of DAC-pre-selected layers. We report mean$\\pm$std over $n$ paired trials. We test for equivalence using Paired TOST with a hybrid margin: $\\delta = \\max({relative_margin_percent*100:.0f}\\%\\text{{ of mean}}, {absolute_margin_floor*100:.1f}\\text{{ pp}})$. TOST $p<0.05$ indicates the mean difference falls significantly within $[-\\delta, +\\delta]$. $d_z$ denotes paired effect size.}}")
    lines.append("\\label{tab:dac_original_vs_random}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrrcc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & DAC Orig & DAC Rand & Diff (pp) & Diff 95\\% CI & $d_z$ & TOST $p$ & Result \\\\")
    lines.append("\\midrule")

    equivalence_count = 0

    for _, row in df.sort_values(['dataset', 'model_name']).iterrows():
        dataset = row['dataset']
        model = row['model_name']

        dac_orig_mean = row.get('dac_algorithm_ece_mean', np.nan)
        dac_orig_std = row.get('dac_algorithm_ece_std', np.nan)
        dac_orig_count = row.get('dac_algorithm_ece_count', None)
        dac_random_mean = row.get('dac_random_ece_mean', np.nan)
        dac_random_std = row.get('dac_random_ece_std', np.nan)
        dac_random_count = row.get('dac_random_ece_count', None)

        dac_orig_str = format_ece_value(dac_orig_mean, dac_orig_std, dac_orig_count, inline_math=True)
        dac_random_str = format_ece_value(dac_random_mean, dac_random_std, dac_random_count, inline_math=True)

        dac_orig_vals = [v for v in row.get('dac_algorithm_ece_values', []) if pd.notna(v)]
        dac_random_vals = [v for v in row.get('dac_random_ece_values', []) if pd.notna(v)]

        # Calculate hybrid margin: max(relative margin, absolute floor)
        # Use the baseline (dac_random) mean for margin calculation
        baseline_mean = dac_random_mean if pd.notna(dac_random_mean) else dac_orig_mean if pd.notna(dac_orig_mean) else 0.01
        if pd.notna(baseline_mean) and baseline_mean > 0:
            relative_margin = baseline_mean * relative_margin_percent
            equivalence_margin = max(relative_margin, absolute_margin_floor)
        else:
            equivalence_margin = absolute_margin_floor

        # Perform TOST test
        tost_result = perform_tost_test(dac_orig_vals, dac_random_vals, equivalence_margin=equivalence_margin)
        
        mean_diff = tost_result['mean_diff']
        diff_str = f"{mean_diff*100:+.2f}" if pd.notna(mean_diff) else "---"
        
        ci_lower = tost_result['ci_lower']
        ci_upper = tost_result['ci_upper']
        ci_str = f"[{ci_lower*100:+.2f}, {ci_upper*100:+.2f}]" if pd.notna(ci_lower) and pd.notna(ci_upper) else "---"
        
        dz = tost_result['dz']
        dz_str = f"{dz:+.2f}" if pd.notna(dz) else "---"
        
        tost_p = tost_result['tost_pvalue']
        tost_p_str = f"{tost_p:.3f}" if tost_p >= 0.001 else "<0.001"
        
        result_str = determine_result(
            tost_p, mean_diff, equivalence_margin, ci_lower, ci_upper
        )
        # Replace generic method names with actual method names
        result_str = result_str.replace("Method 1", "DAC Orig").replace("Method 2", "DAC Rand")
        
        if tost_result['equivalence_established']:
            equivalence_count += 1

        min_margin_pct = tost_result.get('min_margin_for_equiv_pct', np.nan)
        SIGNIFICANCE_LOG.append({
            'Comparison': 'DAC_Original_vs_Random',
            'Dataset': dataset,
            'Training': 'aggregated',
            'Model': model,
            'SGC_Mean': dac_orig_mean * 100 if pd.notna(dac_orig_mean) else np.nan,
            'Base_Mean': dac_random_mean * 100 if pd.notna(dac_random_mean) else np.nan,
            'TOST_P_Value': tost_p,
            'Effect_Size': dz,
            'Min_Margin_Pct': min_margin_pct,
            'Conclusion': 'Equivalent' if tost_result['equivalence_established'] else 'Different',
        })

        cells = [
            dataset.upper(),
            model,
            dac_orig_str,
            dac_random_str,
            diff_str,
            ci_str,
            dz_str,
            tost_p_str,
            result_str,
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{2}}{{l}}{{Summary}} & \\multicolumn{{7}}{{r}}{{Equivalence established: {equivalence_count}/{len(df)}}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def generate_dac_random_vs_coordinates_comparison_table(aggregated_df: pd.DataFrame, relative_margin_percent: float = 0.10, absolute_margin_floor: float = 0.001) -> str:
    """Compare DAC Random Layers against DAC with Coordinate Features ECE with TOST equivalence testing.
    
    Args:
        aggregated_df: DataFrame with aggregated results
        relative_margin_percent: Relative margin as fraction of mean ECE (default 0.10 = 10%)
        absolute_margin_floor: Absolute margin floor in same units as ECE (default 0.005 = 0.5pp)
    """
    lines = []

    if aggregated_df.empty:
        return "% No data available for DAC Random vs DAC Coordinates comparison"

    df = aggregated_df.copy()

    # Aggregate over training methods to get a single row per dataset/model
    def rms_agg(x):
        if len(x) == 0:
            return np.nan
        return np.sqrt(np.mean(x**2))

    def flatten_lists(series):
        combined = []
        for item in series:
            if isinstance(item, list):
                combined.extend(item)
        return combined

    agg_dict = {
        'dac_random_ece_mean': 'mean',
        'dac_random_ece_std': rms_agg,
        'dac_random_ece_count': 'sum',
        'dac_with_coordinate_features_ece_mean': 'mean',
        'dac_with_coordinate_features_ece_std': rms_agg,
        'dac_with_coordinate_features_ece_count': 'sum',
        'dac_random_ece_values': flatten_lists,
        'dac_with_coordinate_features_ece_values': flatten_lists,
    }

    present_cols = set(df.columns)
    agg_dict = {k: v for k, v in agg_dict.items() if k in present_cols}

    if {'dataset', 'model_name', 'training_method'}.issubset(df.columns):
        df = df.groupby(['dataset', 'model_name']).agg(agg_dict).reset_index()

    df = df[
        df.get('dac_random_ece_mean').notna()
        & df.get('dac_with_coordinate_features_ece_mean').notna()
    ]

    # Filter out SVHN dataset
    if 'dataset' in df.columns:
        df = df[df['dataset'].astype(str).str.lower() != 'svhn']

    if df.empty:
        return "% No overlapping DAC Random and DAC Coordinates results for comparison"

    lines.append("% DAC Random Layers vs DAC Coordinates comparison table")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{DAC Random Layers vs DAC with Coordinate Features calibration comparison (ECE \\%; lower is better). DAC Random uses random layer selection; DAC Coordinates uses coordinate features. We report mean$\\pm$std over $n$ paired trials. We test for equivalence using Paired TOST with a hybrid margin: $\\delta = \\max({relative_margin_percent*100:.0f}\\%\\text{{ of mean}}, {absolute_margin_floor*100:.1f}\\text{{ pp}})$. TOST $p<0.05$ indicates the mean difference falls significantly within $[-\\delta, +\\delta]$. $d_z$ denotes paired effect size.}}")
    lines.append("\\label{tab:dac_random_vs_coordinates}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrrcc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & DAC Rand & DAC Coord & Diff (pp) & Diff 95\\% CI & $d_z$ & TOST $p$ & Result \\\\")
    lines.append("\\midrule")

    equivalence_count = 0

    for _, row in df.sort_values(['dataset', 'model_name']).iterrows():
        dataset = row['dataset']
        model = row['model_name']

        dac_random_mean = row.get('dac_random_ece_mean', np.nan)
        dac_random_std = row.get('dac_random_ece_std', np.nan)
        dac_random_count = row.get('dac_random_ece_count', None)
        dac_coord_mean = row.get('dac_with_coordinate_features_ece_mean', np.nan)
        dac_coord_std = row.get('dac_with_coordinate_features_ece_std', np.nan)
        dac_coord_count = row.get('dac_with_coordinate_features_ece_count', None)

        dac_random_str = format_ece_value(dac_random_mean, dac_random_std, dac_random_count, inline_math=True)
        dac_coord_str = format_ece_value(dac_coord_mean, dac_coord_std, dac_coord_count, inline_math=True)

        dac_random_vals = [v for v in row.get('dac_random_ece_values', []) if pd.notna(v)]
        dac_coord_vals = [v for v in row.get('dac_with_coordinate_features_ece_values', []) if pd.notna(v)]

        # Calculate hybrid margin: max(relative margin, absolute floor)
        # Use the baseline (dac_coord) mean for margin calculation
        baseline_mean = dac_coord_mean if pd.notna(dac_coord_mean) else dac_random_mean if pd.notna(dac_random_mean) else 0.01
        if pd.notna(baseline_mean) and baseline_mean > 0:
            relative_margin = baseline_mean * relative_margin_percent
            equivalence_margin = max(relative_margin, absolute_margin_floor)
        else:
            equivalence_margin = absolute_margin_floor

        # Perform TOST test
        tost_result = perform_tost_test(dac_random_vals, dac_coord_vals, equivalence_margin=equivalence_margin)
        
        mean_diff = tost_result['mean_diff']
        diff_str = f"{mean_diff*100:+.2f}" if pd.notna(mean_diff) else "---"
        
        ci_lower = tost_result['ci_lower']
        ci_upper = tost_result['ci_upper']
        ci_str = f"[{ci_lower*100:+.2f}, {ci_upper*100:+.2f}]" if pd.notna(ci_lower) and pd.notna(ci_upper) else "---"
        
        dz = tost_result['dz']
        dz_str = f"{dz:+.2f}" if pd.notna(dz) else "---"
        
        tost_p = tost_result['tost_pvalue']
        tost_p_str = f"{tost_p:.3f}" if tost_p >= 0.001 else "<0.001"
        
        result_str = determine_result(
            tost_p, mean_diff, equivalence_margin, ci_lower, ci_upper
        )
        # Replace generic method names with actual method names
        result_str = result_str.replace("Method 1", "DAC Rand").replace("Method 2", "DAC Coord")
        
        if tost_result['equivalence_established']:
            equivalence_count += 1

        min_margin_pct = tost_result.get('min_margin_for_equiv_pct', np.nan)
        SIGNIFICANCE_LOG.append({
            'Comparison': 'DAC_Random_vs_Coordinates',
            'Dataset': dataset,
            'Training': 'aggregated',
            'Model': model,
            'SGC_Mean': dac_random_mean * 100 if pd.notna(dac_random_mean) else np.nan,
            'Base_Mean': dac_coord_mean * 100 if pd.notna(dac_coord_mean) else np.nan,
            'TOST_P_Value': tost_p,
            'Effect_Size': dz,
            'Min_Margin_Pct': min_margin_pct,
            'Conclusion': 'Equivalent' if tost_result['equivalence_established'] else 'Different',
        })

        cells = [
            dataset.upper(),
            model,
            dac_random_str,
            dac_coord_str,
            diff_str,
            ci_str,
            dz_str,
            tost_p_str,
            result_str,
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{2}}{{l}}{{Summary}} & \\multicolumn{{7}}{{r}}{{Equivalence established: {equivalence_count}/{len(df)}}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def generate_feature_algorithm_ablation_table(aggregated_df: pd.DataFrame) -> str:
    """
    Feature/algorithm ablation table comparing how sensitive performance is
    to the feature extractor (pixels vs DAC vs random layers vs coordinates)
    and to the calibration algorithm (Geometric vs DAC).

    Rows are filtered to baseline_cross_entropy training method (to match main results table)
    and grouped by (dataset, model_name).
    Columns (ECE, mean±std, in %):
      - Geo+Pixel   : geometric calibration on pixel features
      - Geo+DAC     : geometric calibration with DAC-style layer selection (SGC(DAC))
      - Geo+Rand    : geometric calibration with random-layer SGC features (L=6, d=256)
      - Geo+Coord   : geometric calibration with random global coordinates (K=256)
      - DAC+Orig    : DAC algorithm with its original features
      - DAC+Rand    : DAC algorithm with random-layer features (if available)
      - DAC+Coord   : DAC algorithm with coordinate features (if available)
      - Range       : (max - min) ECE across all available methods (in percentage points)

    Methods whose ECE is within 0.5 percentage points of the best method for that
    (dataset, model) are annotated with a dagger (\\textsuperscript{\\dagger}) to
    indicate they are effectively equivalent.
    """
    if aggregated_df.empty:
        return "% No data available for feature/algorithm ablation table"

    # Filter out SVHN and filter to baseline_cross_entropy (to match main results table)
    df = aggregated_df[
        (aggregated_df["dataset"] != "svhn") & 
        (aggregated_df["training_method"] == "baseline_cross_entropy")
    ].copy()

    # Group by (dataset, model_name) - now we only have one training method, so no aggregation needed
    grouped = df.groupby(["dataset", "model_name"])

    rows: List[Dict[str, Any]] = []

    def rms_std(series: pd.Series) -> float:
        vals = series.dropna().to_numpy()
        if vals.size == 0:
            return np.nan
        return float(np.sqrt(np.mean(vals ** 2)))

    for (dataset, model), sub in grouped:
        def agg_method(prefix: str) -> Tuple[float, float, int]:
            """Extract ECE statistics for a given method prefix.
            Since we filter to baseline_cross_entropy, this extracts from a single training method.
            Returns: (mean, std, count)
            """
            mean_col = f"{prefix}_ece_mean"
            std_col = f"{prefix}_ece_std"
            count_col = f"{prefix}_ece_count"
            if mean_col not in sub.columns:
                return (np.nan, np.nan, 0)
            means = sub[mean_col].dropna()
            if means.empty:
                return (np.nan, np.nan, 0)
            mean_val = float(means.mean())
            std_val = rms_std(sub[std_col]) if std_col in sub.columns else np.nan
            # Get count - since we filter to baseline_cross_entropy, this is the seed count for that method
            if count_col in sub.columns:
                counts = sub[count_col].dropna()
                count_val = int(counts.max()) if not counts.empty else 0
            else:
                # If count column doesn't exist, try to infer from mean/std presence
                # If we have mean/std, assume at least 1 seed was used
                count_val = 1 if not pd.isna(mean_val) else 0
            return (mean_val, std_val, count_val)

        geo_pixel_mean, geo_pixel_std, geo_pixel_count = agg_method("pixel_geo")
        geo_dac_mean, geo_dac_std, geo_dac_count = agg_method("sgc_dac_layers")   # Geo + DAC layer selection (SGC(DAC))
        geo_rand_mean, geo_rand_std, geo_rand_count = agg_method("sgc")            # Geo + random layers (L=6, d=256)
        geo_coord_mean, geo_coord_std, geo_coord_count = agg_method("coord")        # Geo + coordinates (K=256)
        dac_orig_mean, dac_orig_std, dac_orig_count = agg_method("dac_algorithm")  # DAC + original features

        # Optional / if available:
        # DAC + random-layer features (not present in current pipeline; placeholder for future use)
        dac_rand_mean, dac_rand_std, dac_rand_count = agg_method("dac_random")
        # DAC + coordinate features: DAC run on coordinate features (from dac_with_coordinate_features)
        dac_coord_mean, dac_coord_std, dac_coord_count = agg_method("dac_with_coordinate_features")

        # Skip rows where we have no meaningful entries
        available_means = [
            geo_pixel_mean,
            geo_dac_mean,
            geo_rand_mean,
            geo_coord_mean,
            dac_orig_mean,
            dac_rand_mean,
            dac_coord_mean,
        ]
        available_means = [v for v in available_means if not pd.isna(v)]
        if not available_means:
            continue

        min_ece = min(available_means)
        max_ece = max(available_means)
        range_pp = (max_ece - min_ece) * 100.0  # percentage points

        rows.append(
            {
                "Dataset": dataset,
                "Model": model,
                "Geo+Pixel_mean": geo_pixel_mean,
                "Geo+Pixel_std": geo_pixel_std,
                "Geo+Pixel_count": geo_pixel_count,
                "Geo+DAC_mean": geo_dac_mean,
                "Geo+DAC_std": geo_dac_std,
                "Geo+DAC_count": geo_dac_count,
                "Geo+Rand_mean": geo_rand_mean,
                "Geo+Rand_std": geo_rand_std,
                "Geo+Rand_count": geo_rand_count,
                "Geo+Coord_mean": geo_coord_mean,
                "Geo+Coord_std": geo_coord_std,
                "Geo+Coord_count": geo_coord_count,
                "DAC+Orig_mean": dac_orig_mean,
                "DAC+Orig_std": dac_orig_std,
                "DAC+Orig_count": dac_orig_count,
                "DAC+Rand_mean": dac_rand_mean,
                "DAC+Rand_std": dac_rand_std,
                "DAC+Rand_count": dac_rand_count,
                "DAC+Coord_mean": dac_coord_mean,
                "DAC+Coord_std": dac_coord_std,
                "DAC+Coord_count": dac_coord_count,
                "Range_pp": range_pp,
                "Best_ece": min_ece,
            }
        )

    if not rows:
        return "% No grouped rows available for feature/algorithm ablation table"

    table_df = pd.DataFrame(rows)

    # Build LaTeX table
    lines: List[str] = []
    lines.append("% Feature/algorithm ablation table")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{xcolor}")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Feature/algorithm ablation: ECE (\\%) for Geometric vs DAC across "
        "pixel, DAC, random-layer (L=6, d=256), and coordinate (K=256) features. "
        "Methods within 0.5 percentage points of the best for a given (dataset, model) "
        "are marked with $\\dagger$ (equivalent).}"
    )
    lines.append("\\label{tab:feature_algorithm_ablation}")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\small")
    lines.append(
        "\\begin{tabular}{llrrrrrrr}"
    )
    lines.append("\\toprule")
    lines.append(
        "Dataset & Model & Geo+Pixel & Geo+DAC & Geo+Rand & Geo+Coord & DAC+Orig & DAC+Rand & DAC+Coord & Range \\\\"
    )
    lines.append("\\midrule")

    for _, row in table_df.sort_values(["Dataset", "Model"]).iterrows():
        best_ece = row["Best_ece"]

        def fmt_cell(mean_val: float, std_val: float, count_val: int) -> str:
            if pd.isna(mean_val):
                return "---"
            # Use existing formatter so display units are consistent
            cell = format_ece_value(mean_val, std_val, count=None, inline_math=True)
            # Add count as subscript: value$_{n}$
            if count_val > 0:
                cell = cell + f"$_{{{count_val}}}$"
            # Check equivalence to best (<= 0.5 percentage points)
            if not pd.isna(best_ece) and abs(mean_val - best_ece) * 100.0 <= 0.5 + 1e-9:
                cell = cell + "$^{\\dagger}$"
            return cell

        geo_pixel_str = fmt_cell(row["Geo+Pixel_mean"], row["Geo+Pixel_std"], row.get("Geo+Pixel_count", 0))
        geo_dac_str = fmt_cell(row["Geo+DAC_mean"], row["Geo+DAC_std"], row.get("Geo+DAC_count", 0))
        geo_rand_str = fmt_cell(row["Geo+Rand_mean"], row["Geo+Rand_std"], row.get("Geo+Rand_count", 0))
        geo_coord_str = fmt_cell(row["Geo+Coord_mean"], row["Geo+Coord_std"], row.get("Geo+Coord_count", 0))
        dac_orig_str = fmt_cell(row["DAC+Orig_mean"], row["DAC+Orig_std"], row.get("DAC+Orig_count", 0))
        dac_rand_str = fmt_cell(row["DAC+Rand_mean"], row["DAC+Rand_std"], row.get("DAC+Rand_count", 0))
        dac_coord_str = fmt_cell(row["DAC+Coord_mean"], row["DAC+Coord_std"], row.get("DAC+Coord_count", 0))
        range_str = f"{row['Range_pp']:.2f}"

        cells = [
            row["Dataset"].upper(),
            row["Model"],
            geo_pixel_str,
            geo_dac_str,
            geo_rand_str,
            geo_coord_str,
            dac_orig_str,
            dac_rand_str,
            dac_coord_str,
            range_str,
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    return "\n".join(lines)

def load_coordinate_results_by_strategy(coord_dir: Path, k_value: int = 256) -> pd.DataFrame:
    """
    Load coordinate ablation results separately for nested and independent strategies.
    Searches in multiple possible directories to find both nested and independent files.
    
    Args:
        coord_dir: Primary directory containing coordinate_ablation_*.json files
        k_value: Number of coordinates to use (default 256)
    
    Returns:
        DataFrame with columns: dataset, model_name, training_method, 
        nested_ece_mean, nested_ece_std, nested_ece_count, nested_ece_values,
        independent_ece_mean, independent_ece_std, independent_ece_count, independent_ece_values,
        nested_acc_mean, nested_acc_std, nested_acc_count,
        independent_acc_mean, independent_acc_std, independent_acc_count
    """
    results = []
    
    # List of possible directories to search (including the primary one and common alternatives)
    search_dirs = [
        coord_dir,
        coord_dir.parent / "calibration_comparison_results_coor",
        coord_dir.parent / "calibration_comparison_results_coori",
        coord_dir.parent / "calibration_comparison_results_coordinates",
    ]
    
    # Remove duplicates and non-existent directories
    search_dirs = list(set([d for d in search_dirs if d.exists()]))
    
    logger.info(f"Searching for coordinate results in {len(search_dirs)} directories: {[str(d) for d in search_dirs]}")
    
    for strategy in ["nested", "independent"]:
        for search_dir in search_dirs:
            pattern = str(search_dir / f"coordinate_ablation_*_{strategy}.json")
            for fpath in glob.glob(pattern):
                try:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load {fpath}: {e}")
                    continue
                
                config = data.get("experiment_config", {})
                dataset = config.get("dataset")
                model_name = config.get("model_name")
                training_method = config.get("training_method")
                
                if not all([dataset, model_name, training_method]):
                    continue
                
                results_by_k = data.get("results_by_num_coordinates", {})
                k_str = str(k_value)
                
                if k_str in results_by_k:
                    for trial in results_by_k[k_str]:
                        ece = trial.get("ece_geometric")
                        acc = trial.get("accuracy_geometric")
                        if ece is not None:
                            results.append({
                                "dataset": dataset,
                                "model_name": model_name,
                                "training_method": training_method,
                                "strategy": strategy,
                                "ece": ece,
                                "acc": acc,
                            })
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Pivot to have nested and independent as separate columns
    # First aggregate by (dataset, model_name, training_method, strategy)
    agg = df.groupby(["dataset", "model_name", "training_method", "strategy"]).agg(
        ece_mean=("ece", "mean"),
        ece_std=("ece", "std"),
        ece_count=("ece", "count"),
        ece_values=("ece", list),
        acc_mean=("acc", "mean"),
        acc_std=("acc", "std"),
        acc_count=("acc", "count"),
    ).reset_index()
    
    # Pivot to separate nested and independent
    pivot_ece = agg.pivot_table(
        index=["dataset", "model_name", "training_method"],
        columns="strategy",
        values=["ece_mean", "ece_std", "ece_count", "ece_values", "acc_mean", "acc_std", "acc_count"],
        aggfunc='first'
    )
    
    # Flatten column names
    pivot_ece.columns = ['_'.join(col).strip() if col[1] else col[0] for col in pivot_ece.columns.values]
    pivot_ece = pivot_ece.reset_index()
    
    return pivot_ece


def generate_nested_vs_independent_table(coord_dir: Path, k_value: int = 256) -> str:
    """Generate a table comparing nested vs independent sampling strategies.
    
    Args:
        coord_dir: Directory containing coordinate ablation results
        k_value: Number of coordinates to use for comparison
    
    Returns:
        LaTeX table string
    """
    lines = []
    
    # Load coordinate results by strategy
    df = load_coordinate_results_by_strategy(coord_dir, k_value)
    
    if df.empty:
        logger.warning("No coordinate results found for nested vs independent comparison")
        return "% No coordinate results available for nested vs independent comparison"
    
    lines.append("% Nested vs Independent sampling strategy comparison")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{xcolor}")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Comparison of Nested vs Independent sampling strategies for coordinate ablation (K=" + str(k_value) + "). \\textcolor{blue}{\\textbf{Best}}, \\textcolor{teal}{\\textbf{2nd}}.}")
    lines.append("\\label{tab:nested_vs_independent}")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\small")
    
    # Build table header
    lines.append("\\begin{tabular}{lllcccccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & Training & \\multicolumn{2}{c}{Nested} & \\multicolumn{2}{c}{Independent} & Winner & Diff \\\\")
    lines.append("\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}")
    lines.append(" & & & ECE & Acc & ECE & Acc & & \\\\")
    lines.append("\\midrule")
    
    # Sort by dataset then model
    df = df.sort_values(['dataset', 'model_name', 'training_method'])
    
    current_dataset = None
    
    for _, row in df.iterrows():
        dataset = row['dataset']
        model = row['model_name']
        training_method = row['training_method']
        
        # Add horizontal line between datasets
        if current_dataset is not None and dataset != current_dataset:
            lines.append("\\midrule")
        current_dataset = dataset
        
        # Get nested values
        nested_ece_mean = row.get('ece_mean_nested', np.nan)
        nested_ece_std = row.get('ece_std_nested', np.nan)
        nested_ece_count = row.get('ece_count_nested', None)
        nested_acc_mean = row.get('acc_mean_nested', np.nan)
        nested_acc_std = row.get('acc_std_nested', np.nan)
        nested_acc_count = row.get('acc_count_nested', None)
        
        # Get independent values
        indep_ece_mean = row.get('ece_mean_independent', np.nan)
        indep_ece_std = row.get('ece_std_independent', np.nan)
        indep_ece_count = row.get('ece_count_independent', None)
        indep_acc_mean = row.get('acc_mean_independent', np.nan)
        indep_acc_std = row.get('acc_std_independent', np.nan)
        indep_acc_count = row.get('acc_count_independent', None)
        
        # Format nested ECE
        nested_ece_str = format_ece_value(nested_ece_mean, nested_ece_std, nested_ece_count, inline_math=True)
        
        # Format nested Accuracy
        nested_acc_str = format_accuracy_value(nested_acc_mean, nested_acc_std, nested_acc_count, inline_math=True)
        
        # Format independent ECE
        indep_ece_str = format_ece_value(indep_ece_mean, indep_ece_std, indep_ece_count, inline_math=True)
        
        # Format independent Accuracy
        indep_acc_str = format_accuracy_value(indep_acc_mean, indep_acc_std, indep_acc_count, inline_math=True)
        
        # Determine winner (lower ECE is better)
        winner = "---"
        diff_str = "---"
        if pd.notna(nested_ece_mean) and pd.notna(indep_ece_mean):
            if nested_ece_mean < indep_ece_mean:
                winner = "\\textcolor{blue}{\\textbf{Nested}}"
                diff = indep_ece_mean - nested_ece_mean
                diff_str = f"+{diff*100:.2f}\\%"
                # Highlight nested ECE
                nested_ece_str = "\\textcolor{blue}{\\textbf{" + nested_ece_str + "}}"
            elif indep_ece_mean < nested_ece_mean:
                winner = "\\textcolor{teal}{\\textbf{Independent}}"
                diff = nested_ece_mean - indep_ece_mean
                diff_str = f"+{diff*100:.2f}\\%"
                # Highlight independent ECE
                indep_ece_str = "\\textcolor{teal}{\\textbf{" + indep_ece_str + "}}"
            else:
                winner = "Tie"
                diff_str = "0.00\\%"
        
        # Get training method display name
        training_display = get_training_method_display_name(training_method)
        
        cells = [
            dataset.upper(),
            model,
            training_display,
            nested_ece_str,
            nested_acc_str,
            indep_ece_str,
            indep_acc_str,
            winner,
            diff_str,
        ]
        
        lines.append(" & ".join(cells) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def compute_equivalence_margin_sensitivity_from_log(
    significance_log: List[Dict],
    test_margins: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]
) -> Dict[str, Any]:
    """
    Analyze how many comparisons remain equivalent at different margins from SIGNIFICANCE_LOG.
    
    Args:
        significance_log: List of dictionaries with TOST results
        test_margins: List of margin percentages to test
    
    Returns:
        Dictionary with:
        - margin_equivalence_counts: {margin: n_equivalent}
        - tightest_all_equiv: smallest margin where ALL comparisons are equivalent
        - per_comparison_min_margin: {comparison: min_margin}
        - tightest_single: smallest min_margin across all comparisons
    """
    per_comparison_min = {}
    for entry in significance_log:
        comparison = entry.get('Comparison', 'Unknown')
        dataset = entry.get('Dataset', 'Unknown')
        model = entry.get('Model', 'Unknown')
        key = f"{comparison}_{dataset}_{model}"
        min_margin = entry.get('Min_Margin_Pct', np.nan)
        if pd.notna(min_margin):
            per_comparison_min[key] = min_margin
    
    if not per_comparison_min:
        return {
            'margin_equivalence_counts': {m: 0 for m in test_margins},
            'tightest_all_equiv': np.nan,
            'per_comparison_min_margin': {},
            'tightest_single': np.nan,
        }
    
    # Tightest margin where all pass
    tightest_all_equiv = max(per_comparison_min.values())
    
    # Count equivalences at each test margin
    margin_counts = {}
    for margin in test_margins:
        n_equiv = sum(1 for m in per_comparison_min.values() if m <= margin)
        margin_counts[margin] = n_equiv
    
    return {
        'margin_equivalence_counts': margin_counts,
        'tightest_all_equiv': float(tightest_all_equiv),
        'per_comparison_min_margin': per_comparison_min,
        'tightest_single': float(min(per_comparison_min.values())),
    }


def save_significance_report(output_dir: Path):
    """Save detailed significance analysis report to CSV."""
    if not SIGNIFICANCE_LOG:
        logger.info("No significance data to report.")
        return
    
    df = pd.DataFrame(SIGNIFICANCE_LOG)
    # Sort by P-Value to see strongest results first
    if 'P_Value' in df.columns:
        df = df.sort_values(['Dataset', 'Training', 'P_Value'])
    elif 'TOST_P_Value' in df.columns:
        df = df.sort_values(['Dataset', 'Training', 'TOST_P_Value'])
    
    path = output_dir / "significance_detailed_report.csv"
    df.to_csv(path, index=False)
    logger.info(f"✓ Saved detailed significance report to: {path}")
    
    # Compute margin sensitivity
    sensitivity = compute_equivalence_margin_sensitivity_from_log(SIGNIFICANCE_LOG)
    n_total = len(df)
    
    # Print summary
    print("\n" + "="*100)
    print("SIGNIFICANCE ANALYSIS REPORT")
    print("="*100)
    if 'P_Value' in df.columns:
        print(df[['Dataset', 'Model', 'SGC_Mean', 'Best_Base', 'Base_Mean', 'P_Value', 'Is_Sig', 
                   'Effect_Size', 'Current_N', 'Required_N', 'More_Needed', 'Action']].to_string(index=False))
    else:
        print(df.to_string(index=False))
    print("="*100)
    
    # Print margin sensitivity summary
    if pd.notna(sensitivity['tightest_single']):
        print("\n" + "="*100)
        print("EQUIVALENCE MARGIN SENSITIVITY ANALYSIS")
        print("="*100)
        print(f"  Tightest single comparison: {sensitivity['tightest_single']:.2f}%")
        print(f"  Margin for ALL equivalent: {sensitivity['tightest_all_equiv']:.2f}%")
        test_margins = [0.1, 0.2, 0.3, 0.4, 0.5]
        for margin in test_margins:
            count = sensitivity['margin_equivalence_counts'].get(margin, 0)
            print(f"  At {margin}% margin: {count}/{n_total} equivalent")
        print("="*100)


def generate_main_results_table(aggregated_df: pd.DataFrame) -> str:
    """Generate Table 1: Main Results Comparison.
    
    Columns: Dataset, Model, Uncal, TS, Platt, Isotonic, Beta, DAC, Pixel Geo, SGC(DAC), SGC, Coord, Acc, Seeds
    
    Args:
        aggregated_df: DataFrame with aggregated results (from aggregate_over_seeds)
    
    Returns:
        LaTeX table string
    """
    # Filter to only include baseline_cross_entropy training method and exclude SVHN dataset
    aggregated_df = aggregated_df[
        (aggregated_df['training_method'] == 'baseline_cross_entropy') & 
        (aggregated_df['dataset'] != 'svhn')
    ].copy()
    
    lines = []
    lines.append("% Table 1: Main Results Comparison")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{xcolor}")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    
    # Build table header (added one more column for Platt Scaling and SGC FAISS)
    lines.append("\\begin{tabular}{lllccccccccccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & Uncal & TS & Platt & Isotonic & Beta & DAC & Pixel Geo & SGC(DAC) & SGC & SGC FAISS & Coord & Acc & Seeds \\\\")
    lines.append("\\midrule")
    
    # Sort by dataset, then model
    aggregated_df = aggregated_df.sort_values(['dataset', 'model_name'])
    
    current_dataset = None
    
    for _, row in aggregated_df.iterrows():
        dataset = row['dataset']
        model_name = row['model_name']
        
        # Group rows by dataset
        is_new_dataset = (current_dataset != dataset)
        if is_new_dataset:
            if current_dataset is not None:
                lines.append("\\midrule")
            current_dataset = dataset
        
        # Extract ECE values for comparison
        methods = {
            'uncal': row.get('uncal_ece_mean', np.nan),
            'ts': row.get('ts_ece_mean', np.nan),
            'platt': row.get('platt_ece_mean', np.nan),
            'isotonic': row.get('isotonic_ece_mean', np.nan),
            'beta': row.get('beta_ece_mean', np.nan),
            'dac': row.get('dac_algorithm_ece_mean', np.nan),
            'pixel_geo': row.get('pixel_geo_ece_mean', np.nan),
            'sgc_dac': row.get('sgc_dac_layers_ece_mean', np.nan),  # SGC with DAC layer selection
            'sgc': row.get('sgc_ece_mean', np.nan),
            'sgc_faiss': row.get('sgc_faiss_ece_mean', np.nan),
            'coord': row.get('coord_ece_mean', np.nan),
        }
        
        # Find best, 2nd, 3rd (excluding NaN)
        valid_methods = {k: v for k, v in methods.items() if not pd.isna(v)}
        if valid_methods:
            sorted_methods = sorted(valid_methods.items(), key=lambda x: x[1])
            best_method = sorted_methods[0][0] if len(sorted_methods) > 0 else None
            second_method = sorted_methods[1][0] if len(sorted_methods) > 1 else None
            third_method = sorted_methods[2][0] if len(sorted_methods) > 2 else None
        else:
            best_method = second_method = third_method = None
        
        # Format dataset and model
        dataset_display = dataset.upper()
        model_display = model_name
        
        # Show dataset name only on first row of each dataset group
        show_dataset = is_new_dataset
        
        # Format each method with coloring
        def format_method(method_key, mean_col, ci_col, count_col, is_best=False, is_second=False, is_third=False):
            mean_val = row.get(mean_col, np.nan)
            ci_val = row.get(ci_col, np.nan)
            count_val = row.get(count_col, None)
            
            if pd.isna(mean_val):
                return "---"
            
            formatted = format_ece_value(mean_val, ci_val=ci_val, count=count_val)
            
            if is_best:
                return f"\\textcolor{{blue}}{{\\textbf{{{formatted}}}}}"
            elif is_second:
                return f"\\textcolor{{teal}}{{\\textbf{{{formatted}}}}}"
            elif is_third:
                return f"\\textcolor{{olive}}{{\\textbf{{{formatted}}}}}"
            return formatted
        
        cells = [
            dataset_display if show_dataset else "",
            model_display,
            format_method('uncal', 'uncal_ece_mean', 'uncal_ece_ci', 'uncal_ece_count', 
                         best_method == 'uncal', second_method == 'uncal', third_method == 'uncal'),
            format_method('ts', 'ts_ece_mean', 'ts_ece_ci', 'ts_ece_count',
                         best_method == 'ts', second_method == 'ts', third_method == 'ts'),
            format_method('platt', 'platt_ece_mean', 'platt_ece_ci', 'platt_ece_count',
                         best_method == 'platt', second_method == 'platt', third_method == 'platt'),
            format_method('isotonic', 'isotonic_ece_mean', 'isotonic_ece_ci', 'isotonic_ece_count',
                         best_method == 'isotonic', second_method == 'isotonic', third_method == 'isotonic'),
            format_method('beta', 'beta_ece_mean', 'beta_ece_ci', 'beta_ece_count',
                         best_method == 'beta', second_method == 'beta', third_method == 'beta'),
            format_method('dac', 'dac_algorithm_ece_mean', 'dac_algorithm_ece_ci', 'dac_algorithm_ece_count',
                         best_method == 'dac', second_method == 'dac', third_method == 'dac'),
            format_method('pixel_geo', 'pixel_geo_ece_mean', 'pixel_geo_ece_ci', 'pixel_geo_ece_count',
                         best_method == 'pixel_geo', second_method == 'pixel_geo', third_method == 'pixel_geo'),
            format_method('sgc_dac', 'sgc_dac_layers_ece_mean', 'sgc_dac_layers_ece_ci', 'sgc_dac_layers_ece_count',
                         best_method == 'sgc_dac', second_method == 'sgc_dac', third_method == 'sgc_dac'),
            format_method('sgc', 'sgc_ece_mean', 'sgc_ece_ci', 'sgc_ece_count',
                         best_method == 'sgc', second_method == 'sgc', third_method == 'sgc'),
            format_method('sgc_faiss', 'sgc_faiss_ece_mean', 'sgc_faiss_ece_ci', 'sgc_faiss_ece_count',
                         best_method == 'sgc_faiss', second_method == 'sgc_faiss', third_method == 'sgc_faiss'),
            format_method('coord', 'coord_ece_mean', 'coord_ece_ci', 'coord_ece_count',
                         best_method == 'coord', second_method == 'coord', third_method == 'coord'),
        ]
        
        # Add accuracy (use SGC accuracy if available, otherwise uncalibrated)
        acc_mean = row.get('sgc_acc_mean', row.get('uncal_acc_mean', np.nan))
        acc_ci = row.get('sgc_acc_ci', row.get('uncal_acc_ci', np.nan))
        acc_count = row.get('sgc_acc_count', row.get('uncal_acc_count', None))
        if pd.isna(acc_mean):
            cells.append("---")
        else:
            cells.append(format_accuracy_value(acc_mean, ci_val=acc_ci, count=acc_count))
        
        # Add seed count
        seed_count = row.get('sgc_ece_count', row.get('uncal_ece_count', 0))
        cells.append(str(int(seed_count)) if not pd.isna(seed_count) else "0")
        
        lines.append(" & ".join(cells) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Main calibration comparison across methods. Values show Mean ECE (\\%) $\\pm$ 95\\% CI. "
                 "\\textcolor{blue}{\\textbf{Best}}, \\textcolor{teal}{\\textbf{2nd}}, \\textcolor{olive}{\\textbf{3rd}}.}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_ablation_matrix_table(aggregated_df: pd.DataFrame) -> str:
    """Generate Table 2: Feature vs. Algorithm Ablation (2x2 comparison).
    
    This table shows that the Algorithm matters more than the Feature source.
    
    Mapping:
    - SGC (Geo Algo + Random Feat): global_random
    - DAC Feat + Geo Alg: sgc_with_dac_preprocessing (fair ablation)
    - DAC (DAC Algo + DAC Feat): standard_baselines.density_aware_calibration
    - Random Feat + DAC Alg: ablation.dac_with_random_layer_selection
    
    Columns: Dataset | Model | N | SGC | DAC Feat + Geo Alg | DAC | Random Feat + DAC Alg
    
    Args:
        aggregated_df: DataFrame with aggregated results (from aggregate_over_seeds)
    
    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("% Table 2: Feature vs. Algorithm Ablation (2x2 comparison)")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{xcolor}, \\usepackage{multirow}")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    
    # Filter out SVHN dataset and keep only baseline_cross_entropy training method
    df = aggregated_df[
        (aggregated_df['dataset'] != 'svhn') & 
        (aggregated_df['training_method'] == 'baseline_cross_entropy')
    ].copy()
    
    # Build table header
    lines.append("\\begin{tabular}{lllcccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & N & SGC & DAC Feat + Geo Alg & DAC & Random Feat + DAC Alg \\\\")
    lines.append("\\midrule")
    
    # Sort by dataset, then model
    df = df.sort_values(['dataset', 'model_name'])
    
    # Count rows per dataset for multirow
    dataset_counts = df.groupby('dataset').size().to_dict()
    
    current_dataset = None
    dataset_row_index = {}  # Track which row we're on for each dataset
    
    for _, row in df.iterrows():
        dataset = row['dataset']
        model_name = row['model_name']
        
        # Initialize row index for new dataset
        if current_dataset != dataset:
            if current_dataset is not None:
                lines.append("\\midrule")
            current_dataset = dataset
            dataset_row_index[dataset] = 0
        
        # Increment row index for current dataset
        dataset_row_index[dataset] += 1
        row_num = dataset_row_index[dataset]
        total_rows = dataset_counts[dataset]
        
        # Extract ECE values for the 2x2 comparison
        # For DAC Feat + Geo Alg: use ONLY the fair ablation "SGC with DAC preprocessing"
        # (Do not substitute geometric_dac_weighted, which is a different algorithmic variant.)
        dac_feat_geo_alg_ece = row.get('sgc_dac_preprocessing_ece_mean', np.nan)
        
        methods = {
            'sgc': row.get('sgc_ece_mean', np.nan),
            'dac_feat_geo_alg': dac_feat_geo_alg_ece,
            'dac': row.get('dac_algorithm_ece_mean', np.nan),
            'sgc_feat_dac_alg': row.get('dac_random_ece_mean', np.nan),
        }
        
        # Find best (lowest ECE)
        valid_methods = {k: v for k, v in methods.items() if not pd.isna(v)}
        if valid_methods:
            best_method = min(valid_methods.items(), key=lambda x: x[1])[0]
        else:
            best_method = None
        
        # Format dataset and model
        dataset_display = dataset.upper()
        model_display = model_name
        
        # Use multirow for dataset on first row, empty for subsequent rows
        if row_num == 1:
            asterisk = "*"
            dataset_cell = f"\\multirow{{{total_rows}}}{{{asterisk}}}{{{dataset_display}}}"
        else:
            dataset_cell = ""
        
        # Format each method with bold for best
        def format_method(method_key, mean_col, ci_col, count_col, is_best=False):
            mean_val = row.get(mean_col, np.nan)
            ci_val = row.get(ci_col, np.nan)
            count_val = row.get(count_col, None)
            
            if pd.isna(mean_val):
                return "---"
            
            formatted = format_ece_value(mean_val, ci_val=ci_val, count=count_val)
            
            if is_best:
                return f"\\textbf{{{formatted}}}"
            return formatted
        
        cells = [
            dataset_cell,
            model_display,
        ]
        
        # Add seed count (N)
        seed_count = row.get('sgc_ece_count', row.get('dac_algorithm_ece_count', 0))
        cells.append(str(int(seed_count)) if not pd.isna(seed_count) else "0")
        
        # Add the 4 methods
        cells.append(format_method('sgc', 'sgc_ece_mean', 'sgc_ece_ci', 'sgc_ece_count',
                                  best_method == 'sgc'))
        
        # For DAC Feat + Geo Alg: use ONLY the fair ablation "SGC with DAC preprocessing"
        # (Do not substitute geometric_dac_weighted, which is a different algorithmic variant.)
        dac_feat_geo_alg_mean = row.get('sgc_dac_preprocessing_ece_mean', np.nan)
        dac_feat_geo_alg_ci = row.get('sgc_dac_preprocessing_ece_ci', np.nan)
        dac_feat_geo_alg_count = row.get('sgc_dac_preprocessing_ece_count', None)
        
        if pd.isna(dac_feat_geo_alg_mean):
            cells.append("---")
        else:
            formatted = format_ece_value(dac_feat_geo_alg_mean, ci_val=dac_feat_geo_alg_ci, count=dac_feat_geo_alg_count)
            if best_method == 'dac_feat_geo_alg':
                formatted = f"\\textbf{{{formatted}}}"
            cells.append(formatted)
        
        cells.append(format_method('dac', 'dac_algorithm_ece_mean', 'dac_algorithm_ece_ci', 'dac_algorithm_ece_count',
                                  best_method == 'dac'))
        cells.append(format_method('sgc_feat_dac_alg', 'dac_random_ece_mean', 'dac_random_ece_ci', 'dac_random_ece_count',
                                  best_method == 'sgc_feat_dac_alg'))
        
        lines.append(" & ".join(cells) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Feature vs. Algorithm Ablation: 2x2 comparison showing that the Geometric algorithm "
                 "is the driver of success, not the feature source. Best result per row is bolded. "
                 "SGC = Geometric algorithm + Random features; DAC Feat + Geo Alg = DAC features + Geometric algorithm; "
                 "DAC = DAC algorithm + DAC features; Random Feat + DAC Alg = Random features + DAC algorithm. "
                 "Values show Mean ECE (\\%) $\\pm$ 95\\% CI.}")
    lines.append("\\label{tab:ablation_matrix}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_brier_score_table(aggregated_df: pd.DataFrame, training_method_filter: str = None) -> str:
    """Generate a unified Brier score comparison table across all datasets and models.
    
    Args:
        aggregated_df: Aggregated DataFrame with all results
        training_method_filter: If provided, filter to this training method only
    """
    lines = []
    
    # Filter out SVHN
    df = aggregated_df[aggregated_df['dataset'] != 'svhn'].copy()
    
    # Optionally filter to specific training method
    if training_method_filter:
        df = df[df['training_method'] == training_method_filter]
    else:
        # Aggregate over training methods
        def rms_agg(x):
            if len(x) == 0:
                return np.nan
            return np.sqrt(np.mean(x**2))
        
        agg_dict = {}
        list_cols = []
        for col in df.columns:
            if col not in ['training_method', 'dataset', 'model_name']:
                if col.endswith('_values'):
                    list_cols.append(col)
                elif col.endswith('_mean'):
                    agg_dict[col] = 'mean'
                elif col.endswith('_std'):
                    agg_dict[col] = rms_agg
                elif col.endswith('_count'):
                    agg_dict[col] = 'sum'
        
        if agg_dict:
            df_agg = df.groupby(['dataset', 'model_name']).agg(agg_dict).reset_index()
            
            if list_cols:
                def flatten_lists(x):
                    result = []
                    for item in x:
                        if isinstance(item, list):
                            result.extend(item)
                        elif pd.notna(item):
                            result.append(item)
                    return result
                
                df_lists = df.groupby(['dataset', 'model_name'])[list_cols].agg(flatten_lists).reset_index()
                df = df_agg.merge(df_lists, on=['dataset', 'model_name'], how='left')
            else:
                df = df_agg
    
    lines.append("% Unified Brier score comparison table")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{xcolor}")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    
    if training_method_filter:
        training_display = get_training_method_display_name(training_method_filter)
        caption_text = f"Brier score comparison across datasets and models ({training_display})."
    else:
        caption_text = "Brier score comparison across datasets and models (aggregated over training methods)."
    
    lines.append(f"\\caption{{{caption_text} \\textcolor{{blue}}{{\\textbf{{Best}}}}, \\textcolor{{teal}}{{\\textbf{{2nd}}}}, \\textcolor{{olive}}{{\\textbf{{3rd}}}}.}}")
    lines.append("\\label{tab:brier_comparison}")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\small")
    
    lines.append("\\begin{tabular}{llcccccccccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & Uncal & TS & Platt & Isotonic & Beta & DAC & Pixel Geo & SGC(DAC) & SGC & SGC Lite \\\\")
    lines.append("\\midrule")
    
    # Sort by dataset then model
    df = df.sort_values(['dataset', 'model_name'])
    
    current_dataset = None
    
    for _, row in df.iterrows():
        dataset = row['dataset']
        model = row['model_name']
        
        # Add horizontal line between datasets
        if current_dataset is not None and dataset != current_dataset:
            lines.append("\\midrule")
        current_dataset = dataset
        
        # Extract all Brier scores
        methods = {
            'uncal': row.get('uncal_brier_mean', np.nan),
            'ts': row.get('ts_brier_mean', np.nan),
            'platt': row.get('platt_brier_mean', np.nan),
            'isotonic': row.get('isotonic_brier_mean', np.nan),
            'beta': row.get('beta_brier_mean', np.nan),
            'dac_algorithm': row.get('dac_algorithm_brier_mean', np.nan),
            'pixel_geo': row.get('pixel_geo_brier_mean', np.nan),
            'dac': row.get('dac_brier_mean', np.nan),
            'sgc_dac': row.get('sgc_dac_layers_brier_mean', np.nan),
            'sgc': row.get('sgc_brier_mean', np.nan),
            'coord': row.get('coord_brier_mean', np.nan),
        }
        
        # Format each value
        formatted = {}
        for method, mean_val in methods.items():
            if method == 'coord':
                std_val = row.get('coord_brier_std', np.nan)
                count = row.get('coord_brier_count', None)
            elif method == 'sgc_dac':
                std_val = row.get('sgc_dac_layers_brier_std', np.nan)
                count = row.get('sgc_dac_layers_brier_count', None)
            else:
                std_val = row.get(f'{method}_brier_std', np.nan)
                count = row.get(f'{method}_brier_count', None)
            formatted[method] = format_brier_value(mean_val, std_val=std_val, count=count, inline_math=True)
        
        # Find top 3 (exclude uncal from ranking)
        ranking_methods = {k: v for k, v in methods.items() if k != 'uncal' and not pd.isna(v)}
        if ranking_methods:
            sorted_methods = sorted(ranking_methods.items(), key=lambda x: x[1])
            top3 = [m[0] for m in sorted_methods[:3]]
            
            # Color code top 3
            if len(top3) > 0 and top3[0] in formatted:
                formatted[top3[0]] = "\\textcolor{blue}{\\textbf{" + formatted[top3[0]] + "}}"
            if len(top3) > 1 and top3[1] in formatted:
                formatted[top3[1]] = "\\textcolor{teal}{\\textbf{" + formatted[top3[1]] + "}}"
            if len(top3) > 2 and top3[2] in formatted:
                formatted[top3[2]] = "\\textcolor{olive}{\\textbf{" + formatted[top3[2]] + "}}"
        
        cells = [
            dataset.upper(),
            model,
            formatted['uncal'],
            formatted['ts'],
            formatted.get('platt', '---'),
            formatted['isotonic'],
            formatted['beta'],
            formatted['dac_algorithm'],
            formatted['pixel_geo'],
            formatted.get('sgc_dac', '---'),
            formatted['sgc'],
            formatted.get('coord', '---'),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_ood_metrics_table(aggregated_df: pd.DataFrame, training_method_filter: str = None) -> str:
    """Generate a unified OOD metrics comparison table across all datasets and models.
    
    Args:
        aggregated_df: Aggregated DataFrame with all results
        training_method_filter: If provided, filter to this training method only
    """
    lines = []
    
    # Filter out SVHN
    df = aggregated_df[aggregated_df['dataset'] != 'svhn'].copy()
    
    # Optionally filter to specific training method
    if training_method_filter:
        df = df[df['training_method'] == training_method_filter]
    else:
        # Aggregate over training methods
        def rms_agg(x):
            if len(x) == 0:
                return np.nan
            return np.sqrt(np.mean(x**2))
        
        agg_dict = {}
        list_cols = []
        for col in df.columns:
            if col not in ['training_method', 'dataset', 'model_name']:
                if col.endswith('_values'):
                    list_cols.append(col)
                elif col.endswith('_mean'):
                    agg_dict[col] = 'mean'
                elif col.endswith('_std'):
                    agg_dict[col] = rms_agg
                elif col.endswith('_count'):
                    agg_dict[col] = 'sum'
        
        if agg_dict:
            df_agg = df.groupby(['dataset', 'model_name']).agg(agg_dict).reset_index()
            
            if list_cols:
                def flatten_lists(x):
                    result = []
                    for item in x:
                        if isinstance(item, list):
                            result.extend(item)
                        elif pd.notna(item):
                            result.append(item)
                    return result
                
                df_lists = df.groupby(['dataset', 'model_name'])[list_cols].agg(flatten_lists).reset_index()
                df = df_agg.merge(df_lists, on=['dataset', 'model_name'], how='left')
            else:
                df = df_agg
    
    lines.append("% Unified OOD metrics comparison table")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{xcolor}")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    
    if training_method_filter:
        training_display = get_training_method_display_name(training_method_filter)
        caption_text = f"OOD metrics comparison across datasets and models ({training_display})."
    else:
        caption_text = "OOD metrics comparison across datasets and models (aggregated over training methods)."
    
    lines.append(f"\\caption{{{caption_text} For AUROC and Confidence Gap: \\textcolor{{blue}}{{\\textbf{{Best}}}}, \\textcolor{{teal}}{{\\textbf{{2nd}}}}. For FPR95 and OOD ECE: \\textcolor{{blue}}{{\\textbf{{Best}}}} (lower is better).}}")
    lines.append("\\label{tab:ood_comparison}")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\small")
    
    lines.append("\\begin{tabular}{llcccccccccccccccccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & \\multicolumn{2}{c}{Uncal} & \\multicolumn{2}{c}{TS} & \\multicolumn{2}{c}{Pixel Geo} & \\multicolumn{2}{c}{DAC} & \\multicolumn{2}{c}{DAC(Rand)} & \\multicolumn{2}{c}{SGC(DAC)} & \\multicolumn{2}{c}{SGC} & \\multicolumn{2}{c}{SGC FAISS} & \\multicolumn{2}{c}{SGC Lite} \\\\")
    lines.append("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}\\cmidrule(lr){13-14}\\cmidrule(lr){15-16}\\cmidrule(lr){17-18}\\cmidrule(lr){19-20}")
    lines.append(" & & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 & AUROC & FPR95 \\\\")
    lines.append("\\midrule")
    
    # Sort by dataset then model
    df = df.sort_values(['dataset', 'model_name'])
    
    current_dataset = None
    
    for _, row in df.iterrows():
        dataset = row['dataset']
        model = row['model_name']
        
        # Add horizontal line between datasets
        if current_dataset is not None and dataset != current_dataset:
            lines.append("\\midrule")
        current_dataset = dataset
        
        # Extract OOD metrics for each method
        def format_ood_metric(prefix, metric_name):
            mean_val = row.get(f'{prefix}_{metric_name}_mean', np.nan)
            std_val = row.get(f'{prefix}_{metric_name}_std', np.nan)
            count = row.get(f'{prefix}_{metric_name}_count', None)
            is_pct = (metric_name == 'ood_fpr95')
            return format_ood_value(mean_val, std_val=std_val, count=count, inline_math=True, is_percentage=is_pct)
        
        # Format metrics for each method
        uncal_auroc = format_ood_metric('uncal', 'ood_auroc')
        uncal_fpr95 = format_ood_metric('uncal', 'ood_fpr95')
        ts_auroc = format_ood_metric('ts', 'ood_auroc')
        ts_fpr95 = format_ood_metric('ts', 'ood_fpr95')
        pixel_geo_auroc = format_ood_metric('pixel_geo', 'ood_auroc')
        pixel_geo_fpr95 = format_ood_metric('pixel_geo', 'ood_fpr95')
        # Regular DAC from standard_baselines (dac_algorithm prefix)
        dac_auroc = format_ood_metric('dac_algorithm', 'ood_auroc')
        dac_fpr95 = format_ood_metric('dac_algorithm', 'ood_fpr95')
        dac_random_auroc = format_ood_metric('dac_random', 'ood_auroc')
        dac_random_fpr95 = format_ood_metric('dac_random', 'ood_fpr95')
        sgc_dac_auroc = format_ood_metric('sgc_dac_layers', 'ood_auroc')
        sgc_dac_fpr95 = format_ood_metric('sgc_dac_layers', 'ood_fpr95')
        sgc_auroc = format_ood_metric('sgc', 'ood_auroc')
        sgc_fpr95 = format_ood_metric('sgc', 'ood_fpr95')
        # SGC FAISS
        sgc_faiss_auroc = format_ood_metric('sgc_faiss', 'ood_auroc')
        sgc_faiss_fpr95 = format_ood_metric('sgc_faiss', 'ood_fpr95')
        # SGC Lite - placeholder for now (can be configured later)
        sgc_lite_auroc = format_ood_metric('sgc_lite', 'ood_auroc')
        sgc_lite_fpr95 = format_ood_metric('sgc_lite', 'ood_fpr95')
        
        # Find best AUROC (higher is better) and best FPR95 (lower is better)
        auroc_means = {
            'uncal': row.get('uncal_ood_auroc_mean', np.nan),
            'ts': row.get('ts_ood_auroc_mean', np.nan),
            'pixel_geo': row.get('pixel_geo_ood_auroc_mean', np.nan),
            'dac': row.get('dac_algorithm_ood_auroc_mean', np.nan),
            'dac_random': row.get('dac_random_ood_auroc_mean', np.nan),
            'sgc_dac': row.get('sgc_dac_layers_ood_auroc_mean', np.nan),
            'sgc': row.get('sgc_ood_auroc_mean', np.nan),
            'sgc_faiss': row.get('sgc_faiss_ood_auroc_mean', np.nan),
            'sgc_lite': row.get('sgc_lite_ood_auroc_mean', np.nan),
        }
        fpr95_means = {
            'uncal': row.get('uncal_ood_fpr95_mean', np.nan),
            'ts': row.get('ts_ood_fpr95_mean', np.nan),
            'pixel_geo': row.get('pixel_geo_ood_fpr95_mean', np.nan),
            'dac': row.get('dac_algorithm_ood_fpr95_mean', np.nan),
            'dac_random': row.get('dac_random_ood_fpr95_mean', np.nan),
            'sgc_dac': row.get('sgc_dac_layers_ood_fpr95_mean', np.nan),
            'sgc': row.get('sgc_ood_fpr95_mean', np.nan),
            'sgc_faiss': row.get('sgc_faiss_ood_fpr95_mean', np.nan),
            'sgc_lite': row.get('sgc_lite_ood_fpr95_mean', np.nan),
        }
        
        # Rank AUROC (higher is better)
        valid_auroc = {k: v for k, v in auroc_means.items() if not pd.isna(v)}
        if valid_auroc:
            sorted_auroc = sorted(valid_auroc.items(), key=lambda x: x[1], reverse=True)
            top2_auroc = [m[0] for m in sorted_auroc[:2]]
            
            if len(top2_auroc) > 0 and top2_auroc[0] == 'uncal':
                uncal_auroc = "\\textcolor{blue}{\\textbf{" + uncal_auroc + "}}"
            elif len(top2_auroc) > 0 and top2_auroc[0] == 'ts':
                ts_auroc = "\\textcolor{blue}{\\textbf{" + ts_auroc + "}}"
            elif len(top2_auroc) > 0 and top2_auroc[0] == 'pixel_geo':
                pixel_geo_auroc = "\\textcolor{blue}{\\textbf{" + pixel_geo_auroc + "}}"
            elif len(top2_auroc) > 0 and top2_auroc[0] == 'dac':
                dac_auroc = "\\textcolor{blue}{\\textbf{" + dac_auroc + "}}"
            elif len(top2_auroc) > 0 and top2_auroc[0] == 'dac_random':
                dac_random_auroc = "\\textcolor{blue}{\\textbf{" + dac_random_auroc + "}}"
            elif len(top2_auroc) > 0 and top2_auroc[0] == 'sgc_dac':
                sgc_dac_auroc = "\\textcolor{blue}{\\textbf{" + sgc_dac_auroc + "}}"
            elif len(top2_auroc) > 0 and top2_auroc[0] == 'sgc':
                sgc_auroc = "\\textcolor{blue}{\\textbf{" + sgc_auroc + "}}"
            elif len(top2_auroc) > 0 and top2_auroc[0] == 'sgc_faiss':
                sgc_faiss_auroc = "\\textcolor{blue}{\\textbf{" + sgc_faiss_auroc + "}}"
            elif len(top2_auroc) > 0 and top2_auroc[0] == 'sgc_lite':
                sgc_lite_auroc = "\\textcolor{blue}{\\textbf{" + sgc_lite_auroc + "}}"
            
            if len(top2_auroc) > 1 and top2_auroc[1] == 'uncal':
                uncal_auroc = "\\textcolor{teal}{\\textbf{" + uncal_auroc.replace("\\textcolor{blue}{\\textbf{", "").replace("}}", "") + "}}"
            elif len(top2_auroc) > 1 and top2_auroc[1] == 'ts':
                ts_auroc = "\\textcolor{teal}{\\textbf{" + ts_auroc.replace("\\textcolor{blue}{\\textbf{", "").replace("}}", "") + "}}"
            elif len(top2_auroc) > 1 and top2_auroc[1] == 'pixel_geo':
                pixel_geo_auroc = "\\textcolor{teal}{\\textbf{" + pixel_geo_auroc.replace("\\textcolor{blue}{\\textbf{", "").replace("}}", "") + "}}"
            elif len(top2_auroc) > 1 and top2_auroc[1] == 'dac':
                dac_auroc = "\\textcolor{teal}{\\textbf{" + dac_auroc.replace("\\textcolor{blue}{\\textbf{", "").replace("}}", "") + "}}"
            elif len(top2_auroc) > 1 and top2_auroc[1] == 'dac_random':
                dac_random_auroc = "\\textcolor{teal}{\\textbf{" + dac_random_auroc.replace("\\textcolor{blue}{\\textbf{", "").replace("}}", "") + "}}"
            elif len(top2_auroc) > 1 and top2_auroc[1] == 'sgc_dac':
                sgc_dac_auroc = "\\textcolor{teal}{\\textbf{" + sgc_dac_auroc.replace("\\textcolor{blue}{\\textbf{", "").replace("}}", "") + "}}"
            elif len(top2_auroc) > 1 and top2_auroc[1] == 'sgc':
                sgc_auroc = "\\textcolor{teal}{\\textbf{" + sgc_auroc.replace("\\textcolor{blue}{\\textbf{", "").replace("}}", "") + "}}"
            elif len(top2_auroc) > 1 and top2_auroc[1] == 'sgc_faiss':
                sgc_faiss_auroc = "\\textcolor{teal}{\\textbf{" + sgc_faiss_auroc.replace("\\textcolor{blue}{\\textbf{", "").replace("}}", "") + "}}"
            elif len(top2_auroc) > 1 and top2_auroc[1] == 'sgc_lite':
                sgc_lite_auroc = "\\textcolor{teal}{\\textbf{" + sgc_lite_auroc.replace("\\textcolor{blue}{\\textbf{", "").replace("}}", "") + "}}"
        
        # Rank FPR95 (lower is better)
        valid_fpr95 = {k: v for k, v in fpr95_means.items() if not pd.isna(v)}
        if valid_fpr95:
            sorted_fpr95 = sorted(valid_fpr95.items(), key=lambda x: x[1])
            best_fpr95 = sorted_fpr95[0][0] if sorted_fpr95 else None
            
            if best_fpr95 == 'uncal':
                uncal_fpr95 = "\\textcolor{blue}{\\textbf{" + uncal_fpr95 + "}}"
            elif best_fpr95 == 'ts':
                ts_fpr95 = "\\textcolor{blue}{\\textbf{" + ts_fpr95 + "}}"
            elif best_fpr95 == 'pixel_geo':
                pixel_geo_fpr95 = "\\textcolor{blue}{\\textbf{" + pixel_geo_fpr95 + "}}"
            elif best_fpr95 == 'dac':
                dac_fpr95 = "\\textcolor{blue}{\\textbf{" + dac_fpr95 + "}}"
            elif best_fpr95 == 'dac_random':
                dac_random_fpr95 = "\\textcolor{blue}{\\textbf{" + dac_random_fpr95 + "}}"
            elif best_fpr95 == 'sgc_dac':
                sgc_dac_fpr95 = "\\textcolor{blue}{\\textbf{" + sgc_dac_fpr95 + "}}"
            elif best_fpr95 == 'sgc':
                sgc_fpr95 = "\\textcolor{blue}{\\textbf{" + sgc_fpr95 + "}}"
            elif best_fpr95 == 'sgc_faiss':
                sgc_faiss_fpr95 = "\\textcolor{blue}{\\textbf{" + sgc_faiss_fpr95 + "}}"
            elif best_fpr95 == 'sgc_lite':
                sgc_lite_fpr95 = "\\textcolor{blue}{\\textbf{" + sgc_lite_fpr95 + "}}"
        
        cells = [
            dataset.upper(),
            model,
            uncal_auroc,
            uncal_fpr95,
            ts_auroc,
            ts_fpr95,
            pixel_geo_auroc if not pd.isna(row.get('pixel_geo_ood_auroc_mean')) else '---',
            pixel_geo_fpr95 if not pd.isna(row.get('pixel_geo_ood_fpr95_mean')) else '---',
            dac_auroc if not pd.isna(row.get('dac_algorithm_ood_auroc_mean')) else '---',
            dac_fpr95 if not pd.isna(row.get('dac_algorithm_ood_fpr95_mean')) else '---',
            dac_random_auroc if not pd.isna(row.get('dac_random_ood_auroc_mean')) else '---',
            dac_random_fpr95 if not pd.isna(row.get('dac_random_ood_fpr95_mean')) else '---',
            sgc_dac_auroc if not pd.isna(row.get('sgc_dac_layers_ood_auroc_mean')) else '---',
            sgc_dac_fpr95 if not pd.isna(row.get('sgc_dac_layers_ood_fpr95_mean')) else '---',
            sgc_auroc if not pd.isna(row.get('sgc_ood_auroc_mean')) else '---',
            sgc_fpr95 if not pd.isna(row.get('sgc_ood_fpr95_mean')) else '---',
            sgc_faiss_auroc if not pd.isna(row.get('sgc_faiss_ood_auroc_mean')) else '---',
            sgc_faiss_fpr95 if not pd.isna(row.get('sgc_faiss_ood_fpr95_mean')) else '---',
            sgc_lite_auroc if not pd.isna(row.get('sgc_lite_ood_auroc_mean')) else '---',
            sgc_lite_fpr95 if not pd.isna(row.get('sgc_lite_ood_fpr95_mean')) else '---',
        ]
        lines.append(" & ".join(cells) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_separation_vs_trust_score_table(aggregated_df: pd.DataFrame, training_method_filter: str = None) -> str:
    """Generate a comparison table showing separation vs trust_score methods side by side.
    
    This table compares the same methods using separation scoring vs trust_score scoring
    to see which performs better.
    
    Args:
        aggregated_df: Aggregated DataFrame with all results
        training_method_filter: If provided, filter to this training method only
    """
    lines = []
    
    # Filter out SVHN
    df = aggregated_df[aggregated_df['dataset'] != 'svhn'].copy()
    
    # Optionally filter to specific training method
    if training_method_filter:
        df = df[df['training_method'] == training_method_filter]
    else:
        # If no filter, aggregate over training methods
        def rms_agg(x):
            """Compute root mean square of a Series."""
            if len(x) == 0:
                return np.nan
            return np.sqrt(np.mean(x**2))
        
        agg_dict = {}
        for col in df.columns:
            if col not in ['training_method', 'dataset', 'model_name']:
                if col.endswith('_mean'):
                    agg_dict[col] = 'mean'
                elif col.endswith('_std'):
                    agg_dict[col] = rms_agg
                elif col.endswith('_count'):
                    agg_dict[col] = 'sum'
        
        if agg_dict:
            df = df.groupby(['dataset', 'model_name']).agg(agg_dict).reset_index()
    
    lines.append("% Separation vs Trust Score comparison table")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{multirow}, \\usepackage{xcolor}")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    
    if training_method_filter:
        training_display = get_training_method_display_name(training_method_filter)
        caption_text = f"ECE (\\%) comparison: Separation vs Trust Score methods ({training_display}). Each method shows Separation / Trust Score side-by-side. {{\\footnotesize \\textcolor{{blue}}{{\\textbf{{Best}}}}, \\textcolor{{teal}}{{\\textbf{{2nd}}}}, \\textcolor{{olive}}{{\\textbf{{3rd}}}}}}"
    else:
        caption_text = "ECE (\\%) comparison: Separation vs Trust Score methods (aggregated over training methods). Each method shows Separation / Trust Score side-by-side. {{\\footnotesize \\textcolor{{blue}}{{\\textbf{{Best}}}}, \\textcolor{{teal}}{{\\textbf{{2nd}}}}, \\textcolor{{olive}}{{\\textbf{{3rd}}}}}}"
    
    lines.append(f"\\caption{{{caption_text}}}")
    lines.append("\\label{tab:separation_vs_trust_score}")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\small")
    
    # Methods to compare: SGC, SGC FAISS, Coordinate Sampling, SGC with DAC layers
    methods = [
        ('SGC', 'sgc', 'sgc_trust_score'),
        ('SGC FAISS', 'sgc_faiss', None),  # FAISS doesn't have trust_score variant
        ('Coord', 'coord', 'coord_trust_score'),
        ('SGC(DAC)', 'sgc_dac_layers', 'sgc_dac_layers_trust_score'),
    ]
    
    # Calculate number of columns: Dataset, Model, then for each method: Sep/Trust pair
    num_cols = 2 + len(methods) * 2  # Dataset, Model, then Sep+Trust for each method
    lines.append(f"\\begin{{tabular}}{{ll" + "r" * (num_cols - 2) + "}}")
    lines.append("\\toprule")
    
    # Header row 1: Method names
    header1 = ["Dataset", "Model"]
    for method_name, _, _ in methods:
        header1.append(f"\\multicolumn{{2}}{{c}}{{{method_name}}}")
    lines.append(" & ".join(header1) + " \\\\")
    
    # Header row 2: Separation / Trust Score
    header2 = ["", ""]
    for method_name, _, trust_key in methods:
        if trust_key is not None:
            header2.append("Sep")
            header2.append("Trust")
        else:
            header2.append("FAISS")
            header2.append("---")
    # Update cmidrules based on number of methods
    if len(methods) == 4:
        lines.append("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}")
    else:
        lines.append("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}")
    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\midrule")
    
    # Sort by dataset then model
    df = df.sort_values(['dataset', 'model_name'])
    
    current_dataset = None
    
    for _, row in df.iterrows():
        dataset = row['dataset']
        model = row['model_name']
        
        # Add horizontal line between datasets
        if current_dataset is not None and dataset != current_dataset:
            lines.append("\\midrule")
        current_dataset = dataset
        
        # Build row: Dataset, Model, then ECE pairs for each method
        row_data = [dataset.upper(), model]
        
        # Collect all ECE values for ranking
        all_ece_values = []
        all_ece_positions = {}  # Track position in row_data for each value
        
        pos = 2  # Start after Dataset and Model
        
        for method_name, sep_key, trust_key in methods:
            if trust_key is not None:
                # Separation ECE
                sep_ece = format_ece_value(
                    row.get(f'{sep_key}_ece_mean', np.nan),
                    row.get(f'{sep_key}_ece_ci', np.nan),
                    row.get(f'{sep_key}_ece_count', None),
                    inline_math=True
                )
                
                # Trust Score ECE
                trust_ece = format_ece_value(
                    row.get(f'{trust_key}_ece_mean', np.nan),
                    row.get(f'{trust_key}_ece_ci', np.nan),
                    row.get(f'{trust_key}_ece_count', None),
                    inline_math=True
                )
                
                # Store for ranking
                sep_mean = row.get(f'{sep_key}_ece_mean', np.nan)
                trust_mean = row.get(f'{trust_key}_ece_mean', np.nan)
                
                if not pd.isna(sep_mean):
                    all_ece_values.append(sep_mean)
                    all_ece_positions[len(all_ece_values) - 1] = pos
                if not pd.isna(trust_mean):
                    all_ece_values.append(trust_mean)
                    all_ece_positions[len(all_ece_values) - 1] = pos + 1
                
                row_data.append(sep_ece)
                row_data.append(trust_ece)
                pos += 2
            else:
                # For methods without trust_score variant (like SGC FAISS), just show the FAISS ECE
                faiss_ece = format_ece_value(
                    row.get(f'{sep_key}_ece_mean', np.nan),
                    row.get(f'{sep_key}_ece_ci', np.nan),
                    row.get(f'{sep_key}_ece_count', None),
                    inline_math=True
                )
                
                # Store for ranking
                faiss_mean = row.get(f'{sep_key}_ece_mean', np.nan)
                if not pd.isna(faiss_mean):
                    all_ece_values.append(faiss_mean)
                    all_ece_positions[len(all_ece_values) - 1] = pos
                
                row_data.append(faiss_ece)
                row_data.append("---")
                pos += 2
        
        # Rank and highlight top 3
        if all_ece_values:
            sorted_indices = sorted(range(len(all_ece_values)), key=lambda i: all_ece_values[i])
            top3_indices = sorted_indices[:3]
            
            # Highlight top 3
            for rank, idx in enumerate(top3_indices):
                pos_in_row = all_ece_positions[idx]
                if rank == 0:
                    row_data[pos_in_row] = f"\\textcolor{{blue}}{{\\textbf{{{row_data[pos_in_row]}}}}}"
                elif rank == 1:
                    row_data[pos_in_row] = f"\\textcolor{{teal}}{{\\textbf{{{row_data[pos_in_row]}}}}}"
                elif rank == 2:
                    row_data[pos_in_row] = f"\\textcolor{{olive}}{{\\textbf{{{row_data[pos_in_row]}}}}}"
        
        lines.append(" & ".join(row_data) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_sgc_vs_faiss_comparison_table(aggregated_df: pd.DataFrame, training_method_filter: str = None) -> str:
    """Generate a simple comparison table showing SGC vs SGC FAISS side by side.
    
    Args:
        aggregated_df: Aggregated DataFrame with all results
        training_method_filter: If provided, filter to this training method only
    """
    lines = []
    
    # Filter out SVHN
    df = aggregated_df[aggregated_df['dataset'] != 'svhn'].copy()
    
    # Optionally filter to specific training method
    if training_method_filter:
        df = df[df['training_method'] == training_method_filter]
    else:
        # If no filter, aggregate over training methods
        def rms_agg(x):
            """Compute root mean square of a Series."""
            if len(x) == 0:
                return np.nan
            return np.sqrt(np.mean(x**2))
        
        agg_dict = {}
        for col in df.columns:
            if col not in ['training_method', 'dataset', 'model_name']:
                if col.endswith('_mean'):
                    agg_dict[col] = 'mean'
                elif col.endswith('_std'):
                    agg_dict[col] = rms_agg
                elif col.endswith('_count'):
                    agg_dict[col] = 'sum'
                elif col.endswith('_ci'):
                    agg_dict[col] = 'mean'
        
        if agg_dict:
            df = df.groupby(['dataset', 'model_name']).agg(agg_dict).reset_index()
    
    lines.append("% SGC vs SGC FAISS comparison table")
    lines.append("% Required packages: \\usepackage{booktabs}, \\usepackage{xcolor}")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    
    if training_method_filter:
        training_display = get_training_method_display_name(training_method_filter)
        caption_text = f"ECE (\\%) comparison: SGC vs SGC FAISS ({training_display}). {{\\footnotesize \\textcolor{{blue}}{{\\textbf{{Best}}}}, \\textcolor{{teal}}{{\\textbf{{2nd}}}}}}"
    else:
        caption_text = "ECE (\\%) comparison: SGC vs SGC FAISS (aggregated over training methods). {{\\footnotesize \\textcolor{{blue}}{{\\textbf{{Best}}}}, \\textcolor{{teal}}{{\\textbf{{2nd}}}}}}"
    
    lines.append(f"\\caption{{{caption_text}}}")
    lines.append("\\label{tab:sgc_vs_faiss}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrr}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & SGC & SGC FAISS \\\\")
    lines.append("\\midrule")
    
    # Sort by dataset then model
    df = df.sort_values(['dataset', 'model_name'])
    
    current_dataset = None
    
    for _, row in df.iterrows():
        dataset = row['dataset']
        model = row['model_name']
        
        # Add horizontal line between datasets
        if current_dataset is not None and dataset != current_dataset:
            lines.append("\\midrule")
        current_dataset = dataset
        
        # Format SGC ECE
        sgc_ece = format_ece_value(
            row.get('sgc_ece_mean', np.nan),
            row.get('sgc_ece_ci', np.nan),
            row.get('sgc_ece_count', None),
            inline_math=True
        )
        
        # Format SGC FAISS ECE
        faiss_ece = format_ece_value(
            row.get('sgc_faiss_ece_mean', np.nan),
            row.get('sgc_faiss_ece_ci', np.nan),
            row.get('sgc_faiss_ece_count', None),
            inline_math=True
        )
        
        # Get mean values for ranking
        sgc_mean = row.get('sgc_ece_mean', np.nan)
        faiss_mean = row.get('sgc_faiss_ece_mean', np.nan)
        
        # Rank and highlight
        values = []
        if not pd.isna(sgc_mean):
            values.append(('sgc', sgc_mean, sgc_ece))
        if not pd.isna(faiss_mean):
            values.append(('faiss', faiss_mean, faiss_ece))
        
        if values:
            sorted_values = sorted(values, key=lambda x: x[1])
            for rank, (method, mean_val, formatted_val) in enumerate(sorted_values):
                if rank == 0:
                    if method == 'sgc':
                        sgc_ece = f"\\textcolor{{blue}}{{\\textbf{{{sgc_ece}}}}}"
                    else:
                        faiss_ece = f"\\textcolor{{blue}}{{\\textbf{{{faiss_ece}}}}}"
                elif rank == 1:
                    if method == 'sgc':
                        sgc_ece = f"\\textcolor{{teal}}{{\\textbf{{{sgc_ece}}}}}"
                    else:
                        faiss_ece = f"\\textcolor{{teal}}{{\\textbf{{{faiss_ece}}}}}"
        
        cells = [
            dataset.upper(),
            model,
            sgc_ece if not pd.isna(sgc_mean) else '---',
            faiss_ece if not pd.isna(faiss_mean) else '---',
        ]
        lines.append(" & ".join(cells) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)


def generate_sgc_vs_faiss_tost_table(aggregated_df: pd.DataFrame, relative_margin_percent: float = 0.10, absolute_margin_floor: float = 0.001) -> str:
    """Compare SGC against SGC FAISS ECE with TOST equivalence testing.
    
    Args:
        aggregated_df: DataFrame with aggregated results
        relative_margin_percent: Relative margin as fraction of mean ECE (default 0.10 = 10%)
        absolute_margin_floor: Absolute margin floor in same units as ECE (default 0.001 = 0.1pp)
    """
    lines = []

    if aggregated_df.empty:
        return "% No data available for SGC vs SGC FAISS comparison"

    df = aggregated_df.copy()

    # Aggregate over training methods to get a single row per dataset/model
    def rms_agg(x):
        if len(x) == 0:
            return np.nan
        return np.sqrt(np.mean(x**2))

    def flatten_lists(series):
        combined = []
        for item in series:
            if isinstance(item, list):
                combined.extend(item)
        return combined

    agg_dict = {
        'sgc_ece_mean': 'mean',
        'sgc_ece_std': rms_agg,
        'sgc_ece_count': 'sum',
        'sgc_faiss_ece_mean': 'mean',
        'sgc_faiss_ece_std': rms_agg,
        'sgc_faiss_ece_count': 'sum',
        'sgc_ece_values': flatten_lists,
        'sgc_faiss_ece_values': flatten_lists,
    }

    present_cols = set(df.columns)
    agg_dict = {k: v for k, v in agg_dict.items() if k in present_cols}

    if {'dataset', 'model_name', 'training_method'}.issubset(df.columns):
        df = df.groupby(['dataset', 'model_name']).agg(agg_dict).reset_index()

    df = df[
        df.get('sgc_ece_mean').notna()
        & df.get('sgc_faiss_ece_mean').notna()
    ]

    # Filter out SVHN dataset
    if 'dataset' in df.columns:
        df = df[df['dataset'].astype(str).str.lower() != 'svhn']

    if df.empty:
        return "% No overlapping SGC and SGC FAISS results for comparison"

    lines.append("% SGC vs SGC FAISS calibration comparison table with TOST")
    lines.append("\\begin{table*}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{SGC vs SGC FAISS calibration comparison (ECE \\%; lower is better). We report mean$\\pm$std over $n$ paired trials. We test for equivalence using Paired TOST with a hybrid margin: $\\delta = \\max({relative_margin_percent*100:.0f}\\%\\text{{ of mean}}, {absolute_margin_floor*100:.1f}\\text{{ pp}})$. TOST $p<0.05$ indicates the mean difference falls significantly within $[-\\delta, +\\delta]$. $d_z$ denotes paired effect size.}}")
    lines.append("\\label{tab:sgc_vs_faiss_tost}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{llrrrrrcc}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & SGC & SGC FAISS & Diff (pp) & Diff 95\\% CI & $d_z$ & TOST $p$ & Result \\\\")
    lines.append("\\midrule")

    equivalence_count = 0

    for _, row in df.sort_values(['dataset', 'model_name']).iterrows():
        dataset = row['dataset']
        model = row['model_name']

        sgc_mean = row.get('sgc_ece_mean', np.nan)
        sgc_std = row.get('sgc_ece_std', np.nan)
        sgc_count = row.get('sgc_ece_count', None)
        faiss_mean = row.get('sgc_faiss_ece_mean', np.nan)
        faiss_std = row.get('sgc_faiss_ece_std', np.nan)
        faiss_count = row.get('sgc_faiss_ece_count', None)

        sgc_str = format_ece_value(sgc_mean, sgc_std, sgc_count, inline_math=True)
        faiss_str = format_ece_value(faiss_mean, faiss_std, faiss_count, inline_math=True)

        sgc_vals = [v for v in row.get('sgc_ece_values', []) if pd.notna(v)]
        faiss_vals = [v for v in row.get('sgc_faiss_ece_values', []) if pd.notna(v)]

        # Calculate hybrid margin: max(relative margin, absolute floor)
        # Use the baseline (faiss) mean for margin calculation
        baseline_mean = faiss_mean if pd.notna(faiss_mean) else sgc_mean if pd.notna(sgc_mean) else 0.01
        if pd.notna(baseline_mean) and baseline_mean > 0:
            relative_margin = baseline_mean * relative_margin_percent
            equivalence_margin = max(relative_margin, absolute_margin_floor)
        else:
            equivalence_margin = absolute_margin_floor

        # Perform TOST test
        tost_result = perform_tost_test(sgc_vals, faiss_vals, equivalence_margin=equivalence_margin)
        
        mean_diff = tost_result['mean_diff']
        diff_str = f"{mean_diff*100:+.2f}" if pd.notna(mean_diff) else "---"
        
        ci_lower = tost_result['ci_lower']
        ci_upper = tost_result['ci_upper']
        ci_str = f"[{ci_lower*100:+.2f}, {ci_upper*100:+.2f}]" if pd.notna(ci_lower) and pd.notna(ci_upper) else "---"
        
        dz = tost_result['dz']
        dz_str = f"{dz:+.2f}" if pd.notna(dz) else "---"
        
        tost_p = tost_result['tost_pvalue']
        tost_p_str = f"{tost_p:.3f}" if tost_p >= 0.001 else "<0.001"
        
        result_str = determine_result(
            tost_p, mean_diff, equivalence_margin, ci_lower, ci_upper
        )
        # Replace generic method names with actual method names
        result_str = result_str.replace("Method 1", "SGC").replace("Method 2", "SGC FAISS")
        
        if tost_result['equivalence_established']:
            equivalence_count += 1

        SIGNIFICANCE_LOG.append({
            'Comparison': 'SGC_vs_SGC_FAISS',
            'Dataset': dataset,
            'Training': 'aggregated',
            'Model': model,
            'SGC_Mean': sgc_mean * 100 if pd.notna(sgc_mean) else np.nan,
            'Base_Mean': faiss_mean * 100 if pd.notna(faiss_mean) else np.nan,
            'TOST_P_Value': tost_p,
            'Effect_Size': dz,
            'Conclusion': 'Equivalent' if tost_result['equivalence_established'] else 'Different',
        })

        cells = [
            dataset.upper(),
            model,
            sgc_str,
            faiss_str,
            diff_str,
            ci_str,
            dz_str,
            tost_p_str,
            result_str,
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{2}}{{l}}{{Summary}} & \\multicolumn{{7}}{{r}}{{Equivalence established: {equivalence_count}/{len(df)}}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def main():
    # Configure logging once at entrypoint.
    from utils.logging_config import setup_logging
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Generate SGC-focused LaTeX tables for the paper"
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        nargs='+',
        default=[Path('calibration_comparison_results/random_ablation'), Path('calibration_comparison')],
        help='Directory(ies) containing ablation JSON files (can specify multiple)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('latex_tables_statistics'),
        help='Directory to save LaTeX tables'
    )
    parser.add_argument(
        '--coord-dir',
        type=Path,
        default=Path('calibration_comparison'),
        help='Directory containing coordinate ablation results'
    )
    
    args = parser.parse_args()
    
    # Load all results from all specified directories
    results = []
    for results_dir in args.results_dir:
        dir_results = load_all_results(results_dir)
        results.extend(dir_results)
        logger.info(f"Loaded {len(dir_results)} results from {results_dir}")
    
    logger.info(f"Total results loaded: {len(results)}")
    
    if not results:
        logger.error("No results loaded. Exiting.")
        return 1
    
    # Validate results
    validate_results(results)
    
    # Aggregate over seeds (only clean results)
    logger.info("Aggregating results over seeds (clean data only)...")
    aggregated_df = aggregate_over_seeds(results)
    
    # Load coordinate results if directory exists
    if args.coord_dir.exists():
        logger.info("Loading coordinate ablation results...")
        coord_df = load_coordinate_results(args.coord_dir, k_value=256)
        if not coord_df.empty:
            # Merge with main aggregated_df
            aggregated_df = aggregated_df.merge(
                coord_df,
                on=["dataset", "model_name", "training_method"],
                how="left"
            )
            logger.info(f"Merged {len(coord_df)} coordinate configurations")
        else:
            logger.info("No coordinate results found in directory")
    else:
        logger.info(f"Coordinate directory not found: {args.coord_dir}")
    
    # Save aggregated CSV for reference
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "sgc_aggregated_results.csv"
    aggregated_df.to_csv(csv_path, index=False)
    logger.info(f"Saved aggregated results to: {csv_path}")
    
    # Generate Table 1: Main Results Comparison (NEW)
    logger.info("Generating Table 1: Main Results Comparison...")
    main_results_latex = generate_main_results_table(aggregated_df)
    main_results_path = args.output_dir / "table_1_main_results.tex"
    with open(main_results_path, 'w') as f:
        f.write(main_results_latex)
    logger.info(f"Saved Table 1 (Main Results) to: {main_results_path}")
    
    # Generate Table 2: Feature vs. Algorithm Ablation (NEW)
    logger.info("Generating Table 2: Feature vs. Algorithm Ablation...")
    ablation_matrix_latex = generate_ablation_matrix_table(aggregated_df)
    ablation_matrix_path = args.output_dir / "table_2_ablation_matrix.tex"
    with open(ablation_matrix_path, 'w') as f:
        f.write(ablation_matrix_latex)
    logger.info(f"Saved Table 2 (Ablation Matrix) to: {ablation_matrix_path}")
    
    # Generate Table 1: Main comparison (OLD - keeping for backward compatibility)
    logger.info("Generating Table 1: Main comparison (legacy)...")
    table1_latex = generate_main_comparison_table(aggregated_df)
    table1_path = args.output_dir / "table_1_main.tex"
    with open(table1_path, 'w') as f:
        f.write(table1_latex)
    logger.info(f"Saved Table 1 (legacy) to: {table1_path}")
    
    # Generate Table 2: Sensitivity
    logger.info("Generating Table 2: Hyperparameter sensitivity...")
    # Use aggregate_overall=True to show overall trends (averaged across all configs)
    sensitivity_df = extract_sgc_sensitivity_data(results, aggregate_overall=True)
    if not sensitivity_df.empty:
        sensitivity_csv_path = args.output_dir / "sgc_sensitivity_data.csv"
        sensitivity_df.to_csv(sensitivity_csv_path, index=False)
        logger.info(f"Saved sensitivity data to: {sensitivity_csv_path}")
    
    table2_latex = generate_sensitivity_table(sensitivity_df, aggregate_overall=True)
    table2_path = args.output_dir / "table_2_sensitivity.tex"
    with open(table2_path, 'w') as f:
        f.write(table2_latex)
    logger.info(f"Saved Table 2 to: {table2_path}")
    
    # Generate Table 3: Improvement
    logger.info("Generating Table 3: ECE reduction...")
    table3_latex = generate_improvement_table(aggregated_df)
    table3_path = args.output_dir / "table_3_improvement.tex"
    with open(table3_path, 'w') as f:
        f.write(table3_latex)
    logger.info(f"Saved Table 3 to: {table3_path}")
    
    # Generate unified table (Cross Entropy only)
    logger.info("Generating unified comparison table (Cross Entropy)...")
    unified_latex = generate_unified_comparison_table(aggregated_df, training_method_filter='baseline_cross_entropy')
    unified_path = args.output_dir / "table_unified_cross_entropy.tex"
    with open(unified_path, 'w') as f:
        f.write(unified_latex)
    logger.info(f"Saved unified table to: {unified_path}")
    
    # Generate unified table (all training methods aggregated)
    logger.info("Generating unified comparison table (all training methods)...")
    unified_all_latex = generate_unified_comparison_table(aggregated_df, training_method_filter=None)
    unified_all_path = args.output_dir / "table_unified_all.tex"
    with open(unified_all_path, 'w') as f:
        f.write(unified_all_latex)
    logger.info(f"Saved unified table (all) to: {unified_all_path}")
    
    # Generate feature/algorithm ablation table (aggregated over training methods)
    logger.info("Generating feature/algorithm ablation table (aggregated over training methods)...")
    feat_algo_latex = generate_feature_algorithm_ablation_table(aggregated_df)
    feat_algo_path = args.output_dir / "table_feature_algorithm_ablation.tex"
    with open(feat_algo_path, 'w') as f:
        f.write(feat_algo_latex)
    logger.info(f"Saved feature/algorithm ablation table to: {feat_algo_path}")
    
    # Generate ECE and Accuracy comparison table (all training methods aggregated)
    logger.info("Generating ECE and Accuracy comparison table (all training methods)...")
    ece_acc_latex = generate_ece_accuracy_comparison_table(aggregated_df, training_method_filter=None)
    ece_acc_path = args.output_dir / "table_ece_accuracy_comparison.tex"
    with open(ece_acc_path, 'w') as f:
        f.write(ece_acc_latex)
    logger.info(f"Saved ECE and Accuracy comparison table to: {ece_acc_path}")
    
    # Generate ECE and Accuracy comparison table (Cross Entropy only)
    logger.info("Generating ECE and Accuracy comparison table (Cross Entropy)...")
    ece_acc_ce_latex = generate_ece_accuracy_comparison_table(aggregated_df, training_method_filter='baseline_cross_entropy')
    ece_acc_ce_path = args.output_dir / "table_ece_accuracy_comparison_cross_entropy.tex"
    with open(ece_acc_ce_path, 'w') as f:
        f.write(ece_acc_ce_latex)
    logger.info(f"Saved ECE and Accuracy comparison table (Cross Entropy) to: {ece_acc_ce_path}")
    
    # Generate Brier score comparison table (Cross Entropy only)
    logger.info("Generating Brier score comparison table (Cross Entropy)...")
    brier_ce_latex = generate_brier_score_table(aggregated_df, training_method_filter='baseline_cross_entropy')
    brier_ce_path = args.output_dir / "table_brier_score_cross_entropy.tex"
    with open(brier_ce_path, 'w') as f:
        f.write(brier_ce_latex)
    logger.info(f"Saved Brier score comparison table (Cross Entropy) to: {brier_ce_path}")
    
    # Generate Brier score comparison table (all training methods aggregated)
    logger.info("Generating Brier score comparison table (all training methods)...")
    brier_all_latex = generate_brier_score_table(aggregated_df, training_method_filter=None)
    brier_all_path = args.output_dir / "table_brier_score_all.tex"
    with open(brier_all_path, 'w') as f:
        f.write(brier_all_latex)
    logger.info(f"Saved Brier score comparison table (all) to: {brier_all_path}")
    
    # Generate OOD metrics comparison table (Cross Entropy only)
    logger.info("Generating OOD metrics comparison table (Cross Entropy)...")
    ood_ce_latex = generate_ood_metrics_table(aggregated_df, training_method_filter='baseline_cross_entropy')
    ood_ce_path = args.output_dir / "table_ood_metrics_cross_entropy.tex"
    with open(ood_ce_path, 'w') as f:
        f.write(ood_ce_latex)
    logger.info(f"Saved OOD metrics comparison table (Cross Entropy) to: {ood_ce_path}")
    
    # Generate OOD metrics comparison table (all training methods aggregated)
    logger.info("Generating OOD metrics comparison table (all training methods)...")
    ood_all_latex = generate_ood_metrics_table(aggregated_df, training_method_filter=None)
    ood_all_path = args.output_dir / "table_ood_metrics_all.tex"
    with open(ood_all_path, 'w') as f:
        f.write(ood_all_latex)
    logger.info(f"Saved OOD metrics comparison table (all) to: {ood_all_path}")
    
    # Generate separation vs trust_score comparison table
    logger.info("Generating Separation vs Trust Score comparison table (Cross Entropy)...")
    sep_vs_trust_ce_latex = generate_separation_vs_trust_score_table(aggregated_df, training_method_filter='baseline_cross_entropy')
    sep_vs_trust_ce_path = args.output_dir / "table_separation_vs_trust_score_cross_entropy.tex"
    with open(sep_vs_trust_ce_path, 'w') as f:
        f.write(sep_vs_trust_ce_latex)
    logger.info(f"Saved Separation vs Trust Score comparison table (Cross Entropy) to: {sep_vs_trust_ce_path}")
    
    logger.info("Generating Separation vs Trust Score comparison table (all training methods)...")
    sep_vs_trust_all_latex = generate_separation_vs_trust_score_table(aggregated_df, training_method_filter=None)
    sep_vs_trust_all_path = args.output_dir / "table_separation_vs_trust_score_all.tex"
    with open(sep_vs_trust_all_path, 'w') as f:
        f.write(sep_vs_trust_all_latex)
    logger.info(f"Saved Separation vs Trust Score comparison table (all) to: {sep_vs_trust_all_path}")
    
    # Generate SGC vs SGC FAISS comparison table (Cross Entropy only)
    logger.info("Generating SGC vs SGC FAISS comparison table (Cross Entropy)...")
    sgc_vs_faiss_ce_latex = generate_sgc_vs_faiss_comparison_table(aggregated_df, training_method_filter='baseline_cross_entropy')
    sgc_vs_faiss_ce_path = args.output_dir / "table_sgc_vs_faiss_cross_entropy.tex"
    with open(sgc_vs_faiss_ce_path, 'w') as f:
        f.write(sgc_vs_faiss_ce_latex)
    logger.info(f"Saved SGC vs SGC FAISS comparison table (Cross Entropy) to: {sgc_vs_faiss_ce_path}")
    
    # Generate SGC vs SGC FAISS comparison table (all training methods aggregated)
    logger.info("Generating SGC vs SGC FAISS comparison table (all training methods)...")
    sgc_vs_faiss_all_latex = generate_sgc_vs_faiss_comparison_table(aggregated_df, training_method_filter=None)
    sgc_vs_faiss_all_path = args.output_dir / "table_sgc_vs_faiss_all.tex"
    with open(sgc_vs_faiss_all_path, 'w') as f:
        f.write(sgc_vs_faiss_all_latex)
    logger.info(f"Saved SGC vs SGC FAISS comparison table (all) to: {sgc_vs_faiss_all_path}")
    
    # Generate comparison tables with different relative margins (10%, 20%, 30%)
    # Using hybrid margin: max(relative_margin_percent * baseline_mean, absolute_margin_floor)
    relative_margins = [0.10, 0.20, 0.30]
    absolute_margin_floor = 0.001  # 0.5 percentage points
    
    for margin_pct in relative_margins:
        margin_label = f"{int(margin_pct*100)}pct"
        
        # Generate SGC vs Coordinate comparison table
        logger.info(f"Generating SGC vs Coordinate comparison table ({margin_label})...")
        sgc_vs_coord_latex = generate_sgc_vs_coord_comparison_table(aggregated_df, relative_margin_percent=margin_pct, absolute_margin_floor=absolute_margin_floor)
        sgc_vs_coord_path = args.output_dir / f"table_sgc_vs_coord_{margin_label}.tex"
        with open(sgc_vs_coord_path, 'w') as f:
            f.write(sgc_vs_coord_latex)
        logger.info(f"Saved SGC vs Coordinate comparison table ({margin_label}) to: {sgc_vs_coord_path}")
        
        # Generate SGC vs SGC(DAC) comparison table
        logger.info(f"Generating SGC vs SGC(DAC) comparison table ({margin_label})...")
        sgc_vs_sgc_dac_latex = generate_sgc_vs_sgc_dac_comparison_table(aggregated_df, relative_margin_percent=margin_pct, absolute_margin_floor=absolute_margin_floor)
        sgc_vs_sgc_dac_path = args.output_dir / f"table_sgc_vs_sgc_dac_{margin_label}.tex"
        with open(sgc_vs_sgc_dac_path, 'w') as f:
            f.write(sgc_vs_sgc_dac_latex)
        logger.info(f"Saved SGC vs SGC(DAC) comparison table ({margin_label}) to: {sgc_vs_sgc_dac_path}")
        
        # Generate DAC Original vs DAC Random comparison table
        logger.info(f"Generating DAC Original vs DAC Random comparison table ({margin_label})...")
        dac_orig_vs_random_latex = generate_dac_original_vs_random_comparison_table(aggregated_df, relative_margin_percent=margin_pct, absolute_margin_floor=absolute_margin_floor)
        dac_orig_vs_random_path = args.output_dir / f"table_dac_original_vs_random_{margin_label}.tex"
        with open(dac_orig_vs_random_path, 'w') as f:
            f.write(dac_orig_vs_random_latex)
        logger.info(f"Saved DAC Original vs DAC Random comparison table ({margin_label}) to: {dac_orig_vs_random_path}")
        
        # Generate DAC Random vs DAC Coordinates comparison table
        logger.info(f"Generating DAC Random vs DAC Coordinates comparison table ({margin_label})...")
        dac_random_vs_coord_latex = generate_dac_random_vs_coordinates_comparison_table(aggregated_df, relative_margin_percent=margin_pct, absolute_margin_floor=absolute_margin_floor)
        dac_random_vs_coord_path = args.output_dir / f"table_dac_random_vs_coordinates_{margin_label}.tex"
        with open(dac_random_vs_coord_path, 'w') as f:
            f.write(dac_random_vs_coord_latex)
        logger.info(f"Saved DAC Random vs DAC Coordinates comparison table ({margin_label}) to: {dac_random_vs_coord_path}")
        
        # Generate SGC vs SGC FAISS TOST comparison table
        logger.info(f"Generating SGC vs SGC FAISS TOST comparison table ({margin_label})...")
        sgc_vs_faiss_tost_latex = generate_sgc_vs_faiss_tost_table(aggregated_df, relative_margin_percent=margin_pct, absolute_margin_floor=absolute_margin_floor)
        sgc_vs_faiss_tost_path = args.output_dir / f"table_sgc_vs_faiss_tost_{margin_label}.tex"
        with open(sgc_vs_faiss_tost_path, 'w') as f:
            f.write(sgc_vs_faiss_tost_latex)
        logger.info(f"Saved SGC vs SGC FAISS TOST comparison table ({margin_label}) to: {sgc_vs_faiss_tost_path}")
    
    # Generate Nested vs Independent comparison table
    if args.coord_dir.exists():
        logger.info("Generating Nested vs Independent comparison table...")
        nested_vs_indep_latex = generate_nested_vs_independent_table(args.coord_dir, k_value=256)
        nested_vs_indep_path = args.output_dir / "table_nested_vs_independent.tex"
        with open(nested_vs_indep_path, 'w') as f:
            f.write(nested_vs_indep_latex)
        logger.info(f"Saved Nested vs Independent comparison table to: {nested_vs_indep_path}")
    else:
        logger.info(f"Skipping Nested vs Independent table (coordinate directory not found: {args.coord_dir})")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal configurations: {len(aggregated_df)}")
    print(f"Datasets: {sorted(aggregated_df['dataset'].unique())}")
    print(f"Models: {sorted(aggregated_df['model_name'].unique())}")
    print(f"Training methods: {sorted(aggregated_df['training_method'].unique())}")
    
    if not sensitivity_df.empty:
        print(f"\nSensitivity table dimensions: {sorted(sensitivity_df['target_dim'].unique())}")
        print(f"Sensitivity table layer counts: {sorted(sensitivity_df['num_layers'].unique())}")
    
    print("\n" + "="*80)
    print("All tables generated successfully!")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Save detailed significance report with power analysis
    logger.info("Generating significance analysis report...")
    save_significance_report(args.output_dir)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

