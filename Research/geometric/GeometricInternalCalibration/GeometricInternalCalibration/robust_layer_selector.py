#!/usr/bin/env python3
"""
Robust Layer Selector implementing advanced selection strategies based on metric analysis.
Addresses brittleness in final layer selection by using ensemble methods, top-K selection,
and learned weights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from scipy import stats
import logging
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from utils.logging_config import get_logger
logger = get_logger(__name__)

@dataclass
class LayerCandidate:
    """Represents a layer candidate with its metrics and scores."""
    layer_idx: int
    metrics: Dict[str, float]
    ece_estimate: float
    ece_ci: Tuple[float, float]  # confidence interval
    selection_confidence: float
    rank_scores: Dict[str, float]
    ensemble_score: float

@dataclass
class SelectionResult:
    """Result of the robust layer selection process."""
    selected_layer: int
    selection_confidence: float
    ece_estimate: float
    ece_ci: Tuple[float, float]
    top_contributors: List[Tuple[str, float]]  # metric name, contribution
    alternatives: List[LayerCandidate]  # other good candidates
    selection_method: str
    stage_a_survivors: List[int]

class RobustLayerSelector:
    """
    Advanced layer selector that implements multiple robustness strategies:
    1. Top-K within epsilon selection
    2. Rank aggregation and ensemble methods
    3. Learned weights from historical data
    4. Bootstrap confidence intervals
    5. Multi-stage selection with cheap/expensive metrics
    """
    
    def __init__(self, 
                 epsilon: float = 0.01,
                 top_k_factor: float = 0.5,  # fraction of layers to keep in stage A
                 bootstrap_samples: int = 200,
                 min_survivors: int = 3,
                 tie_breaker_metrics: List[str] = None):
        """
        Args:
            epsilon: ECE tolerance for considering layers equivalent
            top_k_factor: Fraction of layers to keep after stage A
            bootstrap_samples: Number of bootstrap samples for ECE CI
            min_survivors: Minimum layers to keep in stage A
            tie_breaker_metrics: Ordered list of tie-breaking metrics
        """
        self.epsilon = epsilon
        self.top_k_factor = top_k_factor
        self.bootstrap_samples = bootstrap_samples
        self.min_survivors = min_survivors
        
        # Default tie-breakers based on analysis
        if tie_breaker_metrics is None:
            tie_breaker_metrics = [
                'calibration_decisiveness',
                'boundary_proximity_correlation', 
                'spearman_stability_accuracy'
            ]
        self.tie_breaker_metrics = tie_breaker_metrics
        
        # Learned weights model
        self.learned_weights_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Metric orientation (True = higher is better)
        self.metric_orientations = self._get_metric_orientations()
        
    def _get_metric_orientations(self) -> Dict[str, bool]:
        """Define which metrics should be maximized vs minimized."""
        # Metrics where higher values are better
        maximize_metrics = {
            'boundary_proximity_correlation', 'avg_class_separation_ratio',
            'spearman_stability_accuracy', 'inter_class_distance', 
            'confidence_distance_correlation', 'label_cka_hsic',
            'class_separation_uniformity', 'uncertainty_geometry_alignment',
            'calibration_decisiveness', 'geometry_error_concordance',
            'label_cka', 'logistic_calibratability',
            'good_margin_fraction', 'manifold_trustworthiness', 'manifold_continuity',
            'multiscale_separation', 'silhouette_score', 'spectral_gap',
            'reliability_curve_quality', 'prototype_softmax_ece',
            'feature_stability'
        }
        
        # Metrics where lower values are better
        minimize_metrics = {
            'ece_score', 'within_class_std', 'validation_brier', 'effective_rank',
            'knn_overlap', 'lipschitz_bound', 'local_intrinsic_dimensionality',
            'boundary_smoothness', 'margin_tail_cvar', 'nc1_variance_ratio'
        }
        
        orientations = {}
        orientations.update({m: True for m in maximize_metrics})
        orientations.update({m: False for m in minimize_metrics})
        
        return orientations
    
    def fit_learned_weights(self,
                           historical_metrics: pd.DataFrame,
                           historical_ece_gaps: pd.Series,
                           cv_folds: int = 5,
                           learner: str = 'enet',
                           alphas: Optional[np.ndarray] = None,
                           l1_grid: Optional[List[float]] = None,
                           stability_B: int = 100,
                           stability_subsample: float = 0.5,
                           stability_threshold: float = 0.6) -> pd.DataFrame:
        """
        Train a model to predict ECE gap (lower is better).
        learner ∈ {'ridge','lasso','enet','stability'}.
        Returns a DataFrame with coefficients, |coef|, permutation importance,
        and selection frequency for stability.
        """
        logger.info(f"🧠 Learning metric weights with {learner.upper()}...")

        X = historical_metrics.values
        y = historical_ece_gaps.values
        if alphas is None:
            alphas = np.logspace(-4, 1, 30)
        if l1_grid is None:
            l1_grid = [0.15, 0.3, 0.5, 0.7, 0.85, 1.0]

        def _fit_and_coefs(pipe):
            pipe.fit(X, y)
            steps = pipe.named_steps
            estimator = steps.get('ridgecv') or steps.get('lassocv') or steps.get('elasticnetcv')
            coefs_local = pd.Series(estimator.coef_, index=historical_metrics.columns)
            return pipe, coefs_local

        if learner == 'ridge':
            pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=cv_folds,
                                                          scoring='neg_mean_squared_error'))
            self.learned_weights_model, coefs = _fit_and_coefs(pipe)

        elif learner == 'lasso':
            pipe = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=cv_folds, random_state=0))
            self.learned_weights_model, coefs = _fit_and_coefs(pipe)

        elif learner == 'enet':
            pipe = make_pipeline(StandardScaler(), ElasticNetCV(alphas=alphas, l1_ratio=l1_grid,
                                                               cv=cv_folds, random_state=0))
            self.learned_weights_model, coefs = _fit_and_coefs(pipe)

        elif learner == 'stability':
            rng = np.random.RandomState(0)
            sel_counts = np.zeros(X.shape[1], dtype=float)
            for _ in range(stability_B):
                idx = rng.choice(np.arange(X.shape[0]), size=int(stability_subsample * X.shape[0]), replace=False)
                pipe_b = make_pipeline(StandardScaler(), ElasticNetCV(alphas=alphas, l1_ratio=l1_grid, cv=3, random_state=rng.randint(1e9)))
                pipe_b.fit(X[idx], y[idx])
                est_b = pipe_b.named_steps['elasticnetcv']
                sel_counts += (np.abs(est_b.coef_) > 1e-12).astype(float)
            sel_freq = sel_counts / stability_B
            selected = np.where(sel_freq >= stability_threshold)[0]
            if selected.size == 0:
                selected = np.argsort(sel_freq)[-max(5, X.shape[1] // 10):]

            X_sel = historical_metrics.iloc[:, selected]
            pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=cv_folds, scoring='neg_mean_squared_error'))
            self.learned_weights_model, coefs_refit = _fit_and_coefs(pipe)
            coefs = pd.Series(0.0, index=historical_metrics.columns)
            coefs.iloc[selected] = coefs_refit.values
        else:
            raise ValueError(f"Unknown learner: {learner}")

        self.is_fitted = True

        pi = permutation_importance(self.learned_weights_model, X, y, n_repeats=20,
                                    scoring='neg_mean_squared_error', random_state=0)
        perm = pd.Series(pi.importances_mean, index=historical_metrics.columns)

        out = pd.DataFrame({'coef': coefs, 'abs_coef': coefs.abs(), 'perm_importance': perm})
        if learner == 'stability':
            out['stability_freq'] = sel_freq
            out = out.sort_values(['stability_freq', 'abs_coef'], ascending=[False, False])
        else:
            out = out.sort_values('abs_coef', ascending=False)

        logger.info(f"✅ Learned weights with {learner.upper()}")
        logger.info("   Top drivers: " + ", ".join(out.head(5).index.tolist()))
        return out
    
    def _normalize_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize metrics to [0, 1] with proper orientation.
        
        Args:
            metrics_df: DataFrame with metrics (rows=layers, cols=metrics)
            
        Returns:
            Normalized DataFrame where higher is always better
        """
        # Ensure we have orientations for all metrics
        self._debug_metric_orientations(metrics_df)
        normalized = metrics_df.copy()
        
        for col in normalized.columns:
            values = normalized[col].values
            
            # Skip if all NaN or constant
            if np.all(np.isnan(values)) or np.nanstd(values) == 0:
                normalized[col] = 0.5  # neutral value
                continue
            
            # Winsorize outliers (clip to 5th-95th percentiles)
            p5, p95 = np.nanpercentile(values, [5, 95])
            values = np.clip(values, p5, p95)
            
            # Normalize to [0, 1]
            min_val, max_val = np.nanmin(values), np.nanmax(values)
            if max_val > min_val:
                values = (values - min_val) / (max_val - min_val)
            else:
                values = np.full_like(values, 0.5)
            
            # Flip if metric should be minimized
            if col in self.metric_orientations and not self.metric_orientations[col]:
                values = 1.0 - values
            
            normalized[col] = values
        
        return normalized

    def _debug_metric_orientations(self, metrics_df: pd.DataFrame) -> None:
        """Ensure all metrics have an orientation, defaulting to maximize and log unknowns."""
        unknown = set(metrics_df.columns) - set(self.metric_orientations.keys())
        if unknown:
            logger.debug(f"Metrics without orientation (defaulting to maximize): {sorted(unknown)}")
            for metric in unknown:
                self.metric_orientations[metric] = True
    
    def _rank_aggregate(self, metrics_df: pd.DataFrame, method: str = 'borda') -> pd.Series:
        """
        Aggregate metrics using ranking methods.
        
        Args:
            metrics_df: DataFrame with normalized metrics (higher=better)
            method: 'borda' or 'median_rank'
            
        Returns:
            Series with aggregated scores (higher=better)
        """
        # Convert to ranks (1=best)
        ranks = metrics_df.rank(ascending=False, method='average')
        
        if method == 'borda':
            # Borda count: sum of ranks (lower is better, so we flip)
            scores = ranks.sum(axis=1)
            scores = scores.max() - scores  # flip so higher is better
        elif method == 'median_rank':
            # Median rank (lower is better, so we flip)
            scores = ranks.median(axis=1)
            scores = scores.max() - scores  # flip so higher is better
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return scores
    
    def _learned_weights_score(self, metrics_df: pd.DataFrame) -> pd.Series:
        """
        Score layers using learned weights.
        
        Args:
            metrics_df: DataFrame with metrics
            
        Returns:
            Series with learned weights scores (higher=better)
        """
        if not self.is_fitted or self.learned_weights_model is None:
            raise ValueError("Must call fit_learned_weights() first")
        
        # Predict ECE gaps (lower is better)
        ece_gap_predictions = self.learned_weights_model.predict(metrics_df.values)
        
        # Convert to scores (higher is better)
        scores = -ece_gap_predictions  # flip sign
        
        return pd.Series(scores, index=metrics_df.index)
    
    def _bootstrap_ece_ci(self, 
                         probabilities: np.ndarray, 
                         labels: np.ndarray,
                         n_bins: int = 15) -> Tuple[float, Tuple[float, float]]:
        """
        Compute ECE with bootstrap confidence interval.
        
        Args:
            probabilities: Predicted probabilities
            labels: True labels
            n_bins: Number of bins for ECE calculation
            
        Returns:
            (mean_ece, (lower_ci, upper_ci))
        """
        def calculate_ece(probs, labs):
            confidences = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)
            accuracies = (predictions == labs).astype(float)
            
            ece = 0.0
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            
            for i in range(n_bins):
                in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
                prop_in_bin = np.mean(in_bin)
                
                if prop_in_bin > 0:
                    accuracy_in_bin = np.mean(accuracies[in_bin])
                    avg_confidence_in_bin = np.mean(confidences[in_bin])
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            return ece
        
        # Original ECE
        original_ece = calculate_ece(probabilities, labels)
        
        # Bootstrap
        n_samples = len(labels)
        bootstrap_eces = []
        
        for _ in range(self.bootstrap_samples):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_probs = probabilities[indices]
            boot_labels = labels[indices]
            
            boot_ece = calculate_ece(boot_probs, boot_labels)
            bootstrap_eces.append(boot_ece)
        
        # Confidence interval (2.5th and 97.5th percentiles)
        ci_lower = np.percentile(bootstrap_eces, 2.5)
        ci_upper = np.percentile(bootstrap_eces, 97.5)
        
        return original_ece, (ci_lower, ci_upper)
    
    def select_layer_robust(self,
                           layer_metrics: pd.DataFrame,
                           layer_eces: Optional[pd.Series] = None,
                           layer_probabilities: Optional[Dict[int, np.ndarray]] = None,
                           layer_labels: Optional[Dict[int, np.ndarray]] = None,
                           stage_a_metrics: List[str] = None) -> SelectionResult:
        """
        Robust layer selection using multi-stage approach.
        
        Args:
            layer_metrics: DataFrame with rows=layers, cols=metrics
            layer_eces: Optional pre-computed ECEs for each layer
            layer_probabilities: Optional dict of {layer_idx: probabilities} for bootstrap
            layer_labels: Optional dict of {layer_idx: labels} for bootstrap
            stage_a_metrics: List of cheap metrics for stage A filtering
            
        Returns:
            SelectionResult with chosen layer and metadata
        """
        logger.info("🎯 Starting robust layer selection...")
        
        if layer_metrics.empty:
            raise ValueError("No layer metrics provided")
        
        # Default stage A metrics (fast to compute)
        if stage_a_metrics is None:
            stage_a_metrics = [
                'boundary_proximity_correlation',
                'spearman_stability_accuracy', 
                'confidence_distance_correlation',
                'calibration_decisiveness',
                'class_separation_uniformity'
            ]
        
        # Stage A: Fast filtering with cheap metrics
        logger.info(f"🔍 Stage A: Filtering with {len(stage_a_metrics)} fast metrics...")
        
        # Guard for availability and fall back
        stage_a_metrics = [m for m in (stage_a_metrics or []) if m in layer_metrics.columns]
        if not stage_a_metrics:
            logger.warning("No Stage-A metrics found; falling back to all available metrics.")
            stage_a_metrics = list(layer_metrics.columns)
        stage_a_df = layer_metrics[stage_a_metrics].copy()
        
        # Normalize stage A metrics
        stage_a_normalized = self._normalize_metrics(stage_a_df)
        
        # Rank aggregation for stage A
        stage_a_scores = self._rank_aggregate(stage_a_normalized, method='borda')
        
        # Keep top-K layers
        n_layers = len(stage_a_scores)
        k_survivors = max(self.min_survivors, int(n_layers * self.top_k_factor))
        k_survivors = min(k_survivors, n_layers)  # don't exceed total
        
        stage_a_survivors = stage_a_scores.nlargest(k_survivors).index.tolist()
        
        logger.info(f"   ✅ Stage A kept {len(stage_a_survivors)}/{n_layers} layers: {stage_a_survivors}")
        
        # Stage B: Detailed evaluation of survivors
        logger.info("🔬 Stage B: Detailed evaluation of survivors...")
        
        survivor_metrics = layer_metrics.loc[stage_a_survivors].copy()
        survivor_normalized = self._normalize_metrics(survivor_metrics)
        
        # Multiple scoring methods
        scoring_methods = {}
        
        # 1. Rank aggregation (always available)
        scoring_methods['rank_aggregate'] = self._rank_aggregate(survivor_normalized, method='borda')
        
        # 2. Learned weights (if available)
        if self.is_fitted:
            try:
                scoring_methods['learned_weights'] = self._learned_weights_score(survivor_metrics)
                logger.debug("   ✅ Using learned weights scoring")
            except Exception as e:
                logger.warning(f"   ⚠️ Learned weights failed: {e}")
        
        # 3. Pre-computed ECEs (if available)
        if layer_eces is not None:
            survivor_eces = layer_eces.loc[stage_a_survivors]
            # Convert ECE to score (lower ECE = higher score)
            max_ece = survivor_eces.max()
            scoring_methods['direct_ece'] = max_ece - survivor_eces
        
        # Ensemble scoring: average available methods
        if len(scoring_methods) > 1:
            # Normalize each method to [0, 1]
            normalized_scores = {}
            for method, scores in scoring_methods.items():
                min_score, max_score = scores.min(), scores.max()
                if max_score > min_score:
                    normalized_scores[method] = (scores - min_score) / (max_score - min_score)
                else:
                    normalized_scores[method] = pd.Series(0.5, index=scores.index)
            
            # Equal weighting for now (could be learned)
            ensemble_scores = pd.DataFrame(normalized_scores).mean(axis=1)
            selection_method = f"ensemble_{len(scoring_methods)}_methods"
        else:
            # Use single available method
            method_name, ensemble_scores = list(scoring_methods.items())[0]
            selection_method = method_name
        
        logger.info(f"   📊 Using {selection_method} for final scoring")
        
        # Find top candidates within epsilon
        if layer_eces is not None:
            # Use actual ECE values for epsilon filtering
            survivor_eces = layer_eces.loc[stage_a_survivors]
            best_ece = survivor_eces.min()
            within_epsilon = survivor_eces[survivor_eces <= best_ece + self.epsilon].index
        else:
            # Use top 3 from ensemble scoring
            within_epsilon = ensemble_scores.nlargest(3).index
        
        logger.info(f"   🎯 {len(within_epsilon)} candidates within epsilon: {list(within_epsilon)}")
        
        # Apply tie-breakers if multiple candidates
        if len(within_epsilon) > 1:
            logger.info("   ⚖️ Applying tie-breakers...")
            
            candidates_df = survivor_normalized.loc[within_epsilon]
            
            # Apply tie-breakers in order
            for tie_metric in self.tie_breaker_metrics:
                if tie_metric in candidates_df.columns:
                    best_candidates = candidates_df[candidates_df[tie_metric] == candidates_df[tie_metric].max()]
                    if len(best_candidates) == 1:
                        selected_layer = best_candidates.index[0]
                        logger.info(f"      ✅ Tie broken by {tie_metric}")
                        break
                    candidates_df = best_candidates
            else:
                # Final tie-breaker: shallowest layer (smallest index)
                selected_layer = min(within_epsilon)
                logger.info("      ✅ Tie broken by shallowest layer")
        else:
            selected_layer = within_epsilon[0]
        
        # Compute confidence intervals if probabilities available
        if layer_probabilities and layer_labels and selected_layer in layer_probabilities:
            ece_estimate, ece_ci = self._bootstrap_ece_ci(
                layer_probabilities[selected_layer],
                layer_labels[selected_layer]
            )
        elif layer_eces is not None:
            ece_estimate = layer_eces.loc[selected_layer]
            ece_ci = (ece_estimate * 0.9, ece_estimate * 1.1)  # rough estimate
        else:
            ece_estimate = 0.0
            ece_ci = (0.0, 0.1)
        
        # Selection confidence based on how clearly the winner was chosen
        if len(within_epsilon) == 1:
            selection_confidence = 0.95
        elif len(within_epsilon) <= 3:
            selection_confidence = 0.80
        else:
            selection_confidence = 0.65
        
        # Top contributing metrics
        layer_scores = survivor_normalized.loc[selected_layer]
        top_contributors = [(metric, score) for metric, score in layer_scores.items()]
        top_contributors.sort(key=lambda x: x[1], reverse=True)
        top_contributors = top_contributors[:5]  # top 5
        
        # Alternative candidates
        alternatives = []
        for layer_idx in within_epsilon:
            if layer_idx != selected_layer:
                candidate = LayerCandidate(
                    layer_idx=layer_idx,
                    metrics=survivor_metrics.loc[layer_idx].to_dict(),
                    ece_estimate=layer_eces.loc[layer_idx] if layer_eces is not None else 0.0,
                    ece_ci=(0.0, 0.1),  # placeholder
                    selection_confidence=0.7,
                    rank_scores=survivor_normalized.loc[layer_idx].to_dict(),
                    ensemble_score=ensemble_scores.loc[layer_idx]
                )
                alternatives.append(candidate)
        
        result = SelectionResult(
            selected_layer=selected_layer,
            selection_confidence=selection_confidence,
            ece_estimate=ece_estimate,
            ece_ci=ece_ci,
            top_contributors=top_contributors,
            alternatives=alternatives,
            selection_method=selection_method,
            stage_a_survivors=stage_a_survivors
        )
        
        logger.info(f"🎉 Selected layer {selected_layer} with confidence {selection_confidence:.2f}")
        logger.info(f"   ECE estimate: {ece_estimate:.4f} [{ece_ci[0]:.4f}, {ece_ci[1]:.4f}]")
        logger.info(f"   Top contributors: {[f'{m}({s:.3f})' for m, s in top_contributors[:3]]}")
        
        return result

# Helper functions for integration with existing code

def within_eps_hit_rate(ece_gaps: List[float], eps: float = 0.01) -> float:
    """Calculate hit rate within epsilon tolerance."""
    return float(np.mean(np.array(ece_gaps) <= eps))

def compute_top_k_recall(predicted_layers: List[int], 
                        optimal_layers: List[int], 
                        all_layer_eces: Dict[int, List[float]], 
                        k: int = 3) -> float:
    """
    Compute Top-K recall: how often the predicted layer is in the top-K best layers.
    
    Args:
        predicted_layers: List of predicted layer indices
        optimal_layers: List of empirically optimal layer indices  
        all_layer_eces: Dict mapping layer_idx to list of ECE values across seeds
        k: Top-K to consider
        
    Returns:
        Top-K recall rate
    """
    recalls = []
    
    for pred_layer, opt_layer in zip(predicted_layers, optimal_layers):
        # Get ECE values for this configuration
        layer_eces = {layer: np.mean(eces) for layer, eces in all_layer_eces.items()}
        
        # Find top-K layers by ECE
        sorted_layers = sorted(layer_eces.keys(), key=lambda x: layer_eces[x])
        top_k_layers = sorted_layers[:k]
        
        # Check if prediction is in top-K
        recalls.append(pred_layer in top_k_layers)
    
    return float(np.mean(recalls))

def analyze_regret_distribution(ece_gaps: np.ndarray, 
                               metric_name: str = "Unknown") -> Dict[str, float]:
    """
    Analyze the distribution of ECE gaps (regret) for a metric.
    
    Args:
        ece_gaps: Array of ECE gaps
        metric_name: Name of the metric for logging
        
    Returns:
        Dictionary with regret statistics
    """
    valid_gaps = ece_gaps[np.isfinite(ece_gaps)]
    
    if len(valid_gaps) == 0:
        return {"metric": metric_name, "count": 0}
    
    stats_dict = {
        "metric": metric_name,
        "count": len(valid_gaps),
        "mean_regret": float(np.mean(valid_gaps)),
        "median_regret": float(np.median(valid_gaps)),
        "std_regret": float(np.std(valid_gaps)),
        "q95_regret": float(np.percentile(valid_gaps, 95)),
        "q99_regret": float(np.percentile(valid_gaps, 99)),
        "zero_regret_rate": float(np.mean(valid_gaps == 0)),
        "low_regret_rate": float(np.mean(valid_gaps <= 0.005)),  # within 0.5%
        "high_regret_rate": float(np.mean(valid_gaps > 0.02))    # worse than 2%
    }
    
    return stats_dict

if __name__ == "__main__":
    # Example usage
    print("🚀 Robust Layer Selector")
    print("This module provides advanced layer selection with:")
    print("  ✅ Top-K within epsilon selection")
    print("  ✅ Rank aggregation and ensemble methods") 
    print("  ✅ Learned weights from historical data")
    print("  ✅ Bootstrap confidence intervals")
    print("  ✅ Multi-stage selection pipeline")
