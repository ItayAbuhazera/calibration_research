"""
Geometric Quality Predictors for Calibration Success

This module contains analyzers that predict whether geometric separation (fast-separation)
will work well for calibration on a given layer/dataset/model combination BEFORE running
expensive calibration experiments.

The meta-predictor answers: "Will this geometric signal have predictive power for calibration?"
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from scipy.stats import pearsonr, entropy
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.isotonic import IsotonicRegression
import warnings

from utils.logging_config import get_logger
logger = get_logger(__name__)


@dataclass
class GeometricQualityMetrics:
    """Container for all geometric quality metrics"""
    # Tier 1: Direct Geometric Quality Indicators
    separation_accuracy_correlation: float = 0.0
    separation_entropy: float = 0.0
    safe_dangerous_gap: float = 0.0

    # Tier 2: Geometric Structure Quality
    distance_ratio: float = 0.0
    nn_class_purity: float = 0.0
    impostor_rate: float = 0.0

    # Tier 3: Advanced Indicators
    effective_rank: float = 0.0
    nc1_within_class_scatter: float = 0.0
    nc4_nearest_centroid_accuracy: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'separation_accuracy_correlation': self.separation_accuracy_correlation,
            'separation_entropy': self.separation_entropy,
            'safe_dangerous_gap': self.safe_dangerous_gap,
            'distance_ratio': self.distance_ratio,
            'nn_class_purity': self.nn_class_purity,
            'impostor_rate': self.impostor_rate,
            'effective_rank': self.effective_rank,
            'nc1_within_class_scatter': self.nc1_within_class_scatter,
            'nc4_nearest_centroid_accuracy': self.nc4_nearest_centroid_accuracy,
        }


class GeometricQualityAnalyzer:
    """
    Analyzes whether geometric separation will work for calibration.
    These metrics predict calibration success WITHOUT running full calibration.

    The core idea: compute fast_separation scores and see if they correlate with accuracy.
    fast_separation(x) = [D(x, other_class) - D(x, same_class)] / 2

    Args:
        features: Feature vectors from validation set (N, D)
        labels: True labels for validation set (N,)
        train_features: Feature vectors from training set (M, D)
        train_labels: True labels for training set (M,)
        predictions: Model predictions on validation set (N,) [optional]
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        predictions: Optional[np.ndarray] = None
    ):
        """Initialize the analyzer with validation and training data"""
        # Ensure numpy arrays
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.train_features = np.asarray(train_features, dtype=np.float32)
        self.train_labels = np.asarray(train_labels, dtype=np.int64)

        # Predictions (if not provided, use true labels as proxy)
        if predictions is None:
            logger.warning("No predictions provided, using true labels as proxy")
            self.predictions = self.labels.copy()
        else:
            self.predictions = np.asarray(predictions, dtype=np.int64)

        # Validate shapes
        assert len(self.features) == len(self.labels), "Features and labels must have same length"
        assert len(self.train_features) == len(self.train_labels), "Train features and labels must have same length"
        assert self.features.shape[1] == self.train_features.shape[1], "Feature dimensions must match"

        # Get unique classes
        self.classes = np.unique(self.labels)
        self.n_classes = len(self.classes)

        logger.info(f"Initialized GeometricQualityAnalyzer:")
        logger.info(f"  Validation set: {self.features.shape}")
        logger.info(f"  Training set: {self.train_features.shape}")
        logger.info(f"  Classes: {self.n_classes}")

        # Cache for expensive computations
        self._separation_scores = None
        self._train_nn_index = None

    def _compute_separation_scores(self) -> np.ndarray:
        """
        Compute fast-separation scores for all validation points.

        fast_separation(x) = [D(x, other_class) - D(x, same_class)] / 2

        Returns:
            Array of separation scores (N,)
        """
        if self._separation_scores is not None:
            return self._separation_scores

        logger.info("Computing fast-separation scores...")

        n_val = len(self.features)
        separation_scores = np.zeros(n_val, dtype=np.float32)

        # Build nearest neighbor index for training set
        if self._train_nn_index is None:
            self._train_nn_index = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
            self._train_nn_index.fit(self.train_features)

        # For each validation point, find nearest same-class and other-class training points
        for i in range(n_val):
            query = self.features[i:i+1]
            pred_label = self.predictions[i]

            # Distance to nearest same-class training point
            same_class_mask = self.train_labels == pred_label
            if np.any(same_class_mask):
                same_class_features = self.train_features[same_class_mask]
                dist_same, _ = NearestNeighbors(n_neighbors=1).fit(same_class_features).kneighbors(query)
                d_same = dist_same[0, 0]
            else:
                d_same = np.inf

            # Distance to nearest other-class training point
            other_class_mask = self.train_labels != pred_label
            if np.any(other_class_mask):
                other_class_features = self.train_features[other_class_mask]
                dist_other, _ = NearestNeighbors(n_neighbors=1).fit(other_class_features).kneighbors(query)
                d_other = dist_other[0, 0]
            else:
                d_other = 0.0

            # Compute separation
            separation_scores[i] = (d_other - d_same) / 2.0

        self._separation_scores = separation_scores
        logger.info(f"  Separation range: [{np.min(separation_scores):.4f}, {np.max(separation_scores):.4f}]")

        return separation_scores

    # ============================================================================
    # TIER 1: Direct Geometric Quality Indicators
    # ============================================================================

    def compute_separation_accuracy_correlation(self) -> float:
        """
        Pearson correlation between fast-separation scores and correctness.

        Returns:
            correlation: float in [-1, 1], should be > 0.5 for good calibration

        Implementation:
            1. Compute fast_separation for each val point
            2. Compute correctness (1 if correct, 0 if wrong)
            3. Return pearsonr(separations, correctness)[0]
        """
        try:
            separations = self._compute_separation_scores()
            correctness = (self.predictions == self.labels).astype(float)

            # Filter out any non-finite values
            mask = np.isfinite(separations)
            if mask.sum() < 3:
                logger.warning("Not enough finite separation scores for correlation")
                return 0.0

            corr, p_value = pearsonr(separations[mask], correctness[mask])

            if not np.isfinite(corr):
                logger.warning("Non-finite correlation computed")
                return 0.0

            logger.info(f"Separation-Accuracy Correlation: {corr:.4f} (p={p_value:.4f})")
            return float(corr)

        except Exception as e:
            logger.error(f"Failed to compute separation-accuracy correlation: {e}")
            return 0.0

    def compute_separation_entropy(self, n_bins: int = 50) -> float:
        """
        Shannon entropy of separation score distribution.

        Args:
            n_bins: Number of bins for histogram

        Returns:
            entropy: float, higher = more diverse scores (better)

        Interpretation:
            - Low entropy (<2.0): All points have similar separation → no signal
            - High entropy (>3.5): Good spread of scores → isotonic can fit
        """
        try:
            separations = self._compute_separation_scores()

            # Filter finite values
            mask = np.isfinite(separations)
            if mask.sum() < n_bins:
                logger.warning(f"Not enough samples for entropy calculation (need {n_bins}, have {mask.sum()})")
                return 0.0

            # Compute histogram
            hist, _ = np.histogram(separations[mask], bins=n_bins)

            # Normalize to probabilities
            hist = hist / hist.sum()

            # Compute entropy (use scipy's entropy which handles zero bins)
            ent = entropy(hist + 1e-10)  # Add small value to avoid log(0)

            logger.info(f"Separation Entropy: {ent:.4f}")
            return float(ent)

        except Exception as e:
            logger.error(f"Failed to compute separation entropy: {e}")
            return 0.0

    def compute_safe_dangerous_gap(self) -> float:
        """
        Accuracy difference between safe (sep>0) and dangerous (sep<=0) inputs.

        Returns:
            gap: float in [0, 1], should be > 0.2 for meaningful boundary

        Implementation:
            acc_safe = accuracy[separations > 0]
            acc_dangerous = accuracy[separations <= 0]
            return acc_safe - acc_dangerous
        """
        try:
            separations = self._compute_separation_scores()
            correctness = (self.predictions == self.labels).astype(float)

            # Filter finite values
            mask = np.isfinite(separations)
            sep_finite = separations[mask]
            corr_finite = correctness[mask]

            # Split into safe and dangerous
            safe_mask = sep_finite > 0
            dangerous_mask = sep_finite <= 0

            if safe_mask.sum() == 0 or dangerous_mask.sum() == 0:
                logger.warning("Not enough safe or dangerous samples")
                return 0.0

            acc_safe = corr_finite[safe_mask].mean()
            acc_dangerous = corr_finite[dangerous_mask].mean()

            gap = acc_safe - acc_dangerous

            logger.info(f"Safe/Dangerous Gap: {gap:.4f} (safe_acc={acc_safe:.4f}, dangerous_acc={acc_dangerous:.4f})")
            logger.info(f"  Safe samples: {safe_mask.sum()}, Dangerous samples: {dangerous_mask.sum()}")

            return float(gap)

        except Exception as e:
            logger.error(f"Failed to compute safe/dangerous gap: {e}")
            return 0.0

    # ============================================================================
    # TIER 2: Geometric Structure Quality
    # ============================================================================

    def compute_distance_ratio(self) -> float:
        """
        Ratio: mean(dist_to_other_class) / mean(dist_to_same_class)

        Returns:
            ratio: float, should be >> 1 (ideally > 2.0) for separation

        Implementation:
            For each point:
                d_same = distance to nearest same-class train point
                d_other = distance to nearest other-class train point
            Return mean(d_other) / mean(d_same)
        """
        try:
            n_val = len(self.features)
            d_same_list = []
            d_other_list = []

            for i in range(n_val):
                query = self.features[i:i+1]
                true_label = self.labels[i]

                # Distance to nearest same-class training point
                same_class_mask = self.train_labels == true_label
                if np.any(same_class_mask):
                    same_class_features = self.train_features[same_class_mask]
                    dist_same, _ = NearestNeighbors(n_neighbors=1).fit(same_class_features).kneighbors(query)
                    d_same_list.append(dist_same[0, 0])

                # Distance to nearest other-class training point
                other_class_mask = self.train_labels != true_label
                if np.any(other_class_mask):
                    other_class_features = self.train_features[other_class_mask]
                    dist_other, _ = NearestNeighbors(n_neighbors=1).fit(other_class_features).kneighbors(query)
                    d_other_list.append(dist_other[0, 0])

            if len(d_same_list) == 0 or len(d_other_list) == 0:
                logger.warning("Not enough distance measurements")
                return 0.0

            mean_d_same = np.mean(d_same_list)
            mean_d_other = np.mean(d_other_list)

            if mean_d_same == 0:
                logger.warning("Mean same-class distance is zero")
                return 0.0

            ratio = mean_d_other / mean_d_same

            logger.info(f"Distance Ratio: {ratio:.4f} (d_other={mean_d_other:.4f}, d_same={mean_d_same:.4f})")

            return float(ratio)

        except Exception as e:
            logger.error(f"Failed to compute distance ratio: {e}")
            return 0.0

    def compute_nn_class_purity(self, k: int = 5) -> float:
        """
        Fraction of k-nearest neighbors that share the query's class.

        Args:
            k: Number of nearest neighbors

        Returns:
            purity: float in [0, 1], should be > 0.7 for k=5

        Interpretation:
            - Low (<0.5): Classes mixed in feature space → geometry unreliable
            - High (>0.7): Classes separated → geometry reliable
        """
        try:
            n_val = len(self.features)

            # Adjust k if needed
            k_actual = min(k, len(self.train_features) - 1)
            if k_actual < 1:
                logger.warning("Not enough training samples for NN purity")
                return 0.0

            # Build NN index on training set
            nbrs = NearestNeighbors(n_neighbors=k_actual, algorithm='auto', n_jobs=-1)
            nbrs.fit(self.train_features)

            # Find k nearest training neighbors for each validation point
            _, indices = nbrs.kneighbors(self.features)

            # Compute purity
            purities = []
            for i in range(n_val):
                true_label = self.labels[i]
                neighbor_labels = self.train_labels[indices[i]]
                purity = (neighbor_labels == true_label).sum() / k_actual
                purities.append(purity)

            avg_purity = np.mean(purities)

            logger.info(f"NN Class Purity (k={k_actual}): {avg_purity:.4f}")

            return float(avg_purity)

        except Exception as e:
            logger.error(f"Failed to compute NN class purity: {e}")
            return 0.0

    def compute_impostor_rate(self) -> float:
        """
        Fraction of points where nearest other-class is closer than nearest same-class.

        Returns:
            impostor_rate: float in [0, 1], should be < 0.3

        Note: These are "dangerous" inputs in geometric separation terminology
        """
        try:
            n_val = len(self.features)
            impostor_count = 0
            total_count = 0

            for i in range(n_val):
                query = self.features[i:i+1]
                true_label = self.labels[i]

                # Distance to nearest same-class training point
                same_class_mask = self.train_labels == true_label
                if np.any(same_class_mask):
                    same_class_features = self.train_features[same_class_mask]
                    dist_same, _ = NearestNeighbors(n_neighbors=1).fit(same_class_features).kneighbors(query)
                    d_same = dist_same[0, 0]
                else:
                    continue  # Skip if no same-class neighbors

                # Distance to nearest other-class training point
                other_class_mask = self.train_labels != true_label
                if np.any(other_class_mask):
                    other_class_features = self.train_features[other_class_mask]
                    dist_other, _ = NearestNeighbors(n_neighbors=1).fit(other_class_features).kneighbors(query)
                    d_other = dist_other[0, 0]
                else:
                    continue  # Skip if no other-class neighbors

                # Check if impostor
                if d_other < d_same:
                    impostor_count += 1

                total_count += 1

            if total_count == 0:
                logger.warning("No valid points for impostor rate calculation")
                return 0.0

            rate = impostor_count / total_count

            logger.info(f"Impostor Rate: {rate:.4f} ({impostor_count}/{total_count})")

            return float(rate)

        except Exception as e:
            logger.error(f"Failed to compute impostor rate: {e}")
            return 0.0

    # ============================================================================
    # TIER 3: Advanced Indicators
    # ============================================================================

    def compute_effective_rank(self) -> float:
        """
        Effective rank of feature matrix: exp(H(normalized_singular_values))

        Returns:
            erank: float in [1, min(N, D)]

        Interpretation:
            - Too low (<D*0.3): Features collapsed → no diversity
            - Good range (D*0.5 to D*0.8): Balanced
            - Too high (>D*0.9): May include too much noise
        """
        try:
            # Use validation features for effective rank
            X = self.features

            # Subsample if too large (for efficiency)
            max_samples = 5000
            if len(X) > max_samples:
                logger.info(f"Subsampling {max_samples} points for effective rank calculation")
                indices = np.random.choice(len(X), max_samples, replace=False)
                X = X[indices]

            # Center the data
            X_centered = X - X.mean(axis=0)

            # Compute SVD (use randomized for efficiency if large)
            if X_centered.shape[1] > 500:
                from sklearn.decomposition import TruncatedSVD
                n_components = min(100, X_centered.shape[0] - 1, X_centered.shape[1])
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                svd.fit(X_centered)
                singular_values = svd.singular_values_
            else:
                _, singular_values, _ = np.linalg.svd(X_centered, full_matrices=False)

            # Normalize singular values
            singular_values = singular_values[singular_values > 0]  # Keep only positive
            if len(singular_values) == 0:
                logger.warning("No positive singular values")
                return 0.0

            singular_values_norm = singular_values / singular_values.sum()

            # Compute entropy
            ent = entropy(singular_values_norm + 1e-10)

            # Effective rank
            erank = np.exp(ent)

            # Normalize by feature dimension for interpretability
            n_features = X.shape[1]
            erank_normalized = erank / n_features

            logger.info(f"Effective Rank: {erank:.2f} ({erank_normalized:.2%} of {n_features} features)")

            return float(erank_normalized)

        except Exception as e:
            logger.error(f"Failed to compute effective rank: {e}")
            return 0.0

    def compute_neural_collapse_indicators(self) -> Dict[str, float]:
        """
        NC1 (within-class scatter) and NC4 (nearest-centroid accuracy).
        Only applicable for neural network features.

        Returns:
            {
                'nc1': float in [0, 1], should be 0.1-0.3,
                'nc4': float in [0, 1], nearest-centroid classifier accuracy
            }
        """
        try:
            # Compute class centroids (global mean)
            global_mean = self.features.mean(axis=0)

            # Compute per-class centroids and within-class scatter
            class_centroids = {}
            within_class_vars = []

            for cls in self.classes:
                cls_mask = self.labels == cls
                if cls_mask.sum() == 0:
                    continue

                cls_features = self.features[cls_mask]
                cls_centroid = cls_features.mean(axis=0)
                class_centroids[cls] = cls_centroid

                # Within-class variance
                within_var = ((cls_features - cls_centroid) ** 2).mean()
                within_class_vars.append(within_var)

            # NC1: Average within-class scatter normalized by global variance
            if len(within_class_vars) == 0:
                logger.warning("No class centroids computed for NC1")
                nc1 = 0.0
            else:
                global_var = ((self.features - global_mean) ** 2).mean()
                avg_within_var = np.mean(within_class_vars)

                if global_var == 0:
                    nc1 = 0.0
                else:
                    nc1 = avg_within_var / global_var

            # NC4: Nearest-centroid classifier accuracy
            if len(class_centroids) == 0:
                logger.warning("No class centroids for NC4")
                nc4 = 0.0
            else:
                correct = 0
                total = 0

                for i in range(len(self.features)):
                    query = self.features[i]
                    true_label = self.labels[i]

                    # Find nearest centroid
                    min_dist = np.inf
                    nearest_cls = None

                    for cls, centroid in class_centroids.items():
                        dist = np.linalg.norm(query - centroid)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_cls = cls

                    if nearest_cls == true_label:
                        correct += 1
                    total += 1

                nc4 = correct / total if total > 0 else 0.0

            logger.info(f"Neural Collapse Indicators: NC1={nc1:.4f}, NC4={nc4:.4f}")

            return {'nc1': float(nc1), 'nc4': float(nc4)}

        except Exception as e:
            logger.error(f"Failed to compute neural collapse indicators: {e}")
            return {'nc1': 0.0, 'nc4': 0.0}

    # ============================================================================
    # Summary and Prediction Methods
    # ============================================================================

    def compute_all_predictors(self) -> Dict[str, float]:
        """
        Compute all meta-predictors and return as dictionary.

        Returns:
            Dictionary with all predictor values
        """
        logger.info("\n" + "="*80)
        logger.info("Computing All Geometric Quality Predictors")
        logger.info("="*80)

        # Tier 1
        logger.info("\n[Tier 1: Direct Geometric Quality Indicators]")
        sep_acc_corr = self.compute_separation_accuracy_correlation()
        sep_entropy = self.compute_separation_entropy()
        safe_danger_gap = self.compute_safe_dangerous_gap()

        # Tier 2
        logger.info("\n[Tier 2: Geometric Structure Quality]")
        dist_ratio = self.compute_distance_ratio()
        nn_purity = self.compute_nn_class_purity()
        impostor_rate = self.compute_impostor_rate()

        # Tier 3
        logger.info("\n[Tier 3: Advanced Indicators]")
        eff_rank = self.compute_effective_rank()
        nc_indicators = self.compute_neural_collapse_indicators()

        results = {
            # Tier 1
            'separation_accuracy_correlation': sep_acc_corr,
            'separation_entropy': sep_entropy,
            'safe_dangerous_gap': safe_danger_gap,
            # Tier 2
            'distance_ratio': dist_ratio,
            'nn_class_purity': nn_purity,
            'impostor_rate': impostor_rate,
            # Tier 3
            'effective_rank': eff_rank,
            'nc1_within_class_scatter': nc_indicators['nc1'],
            'nc4_nearest_centroid_accuracy': nc_indicators['nc4'],
        }

        logger.info("\n" + "="*80)
        logger.info("Summary of All Predictors")
        logger.info("="*80)
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("="*80 + "\n")

        return results

    def predict_calibration_success(self) -> Dict[str, any]:
        """
        Make final prediction about whether geometric calibration will work.

        Returns:
            {
                'will_work': bool,
                'confidence': str in ['high', 'medium', 'low'],
                'reason': str,
                'predictors': Dict[str, float],
                'recommendations': List[str]
            }

        Decision Logic:
            - WILL WORK if:
                * separation_accuracy_corr > 0.5 AND
                * safe_dangerous_gap > 0.2 AND
                * distance_ratio > 1.5 AND
                * impostor_rate < 0.3

            - MIGHT WORK if 2-3 conditions met
            - UNLIKELY if < 2 conditions met
        """
        # Compute all predictors
        predictors = self.compute_all_predictors()

        # Decision criteria
        conditions = {
            'separation_accuracy_corr > 0.5': predictors['separation_accuracy_correlation'] > 0.5,
            'safe_dangerous_gap > 0.2': predictors['safe_dangerous_gap'] > 0.2,
            'distance_ratio > 1.5': predictors['distance_ratio'] > 1.5,
            'impostor_rate < 0.3': predictors['impostor_rate'] < 0.3,
        }

        # Count satisfied conditions
        n_satisfied = sum(conditions.values())

        # Make prediction
        if n_satisfied >= 4:
            will_work = True
            confidence = 'high'
            reason = "All key geometric quality indicators are favorable"
        elif n_satisfied == 3:
            will_work = True
            confidence = 'medium'
            reason = "Most geometric quality indicators are favorable"
        elif n_satisfied == 2:
            will_work = False
            confidence = 'medium'
            reason = "Only some geometric quality indicators are favorable"
        else:
            will_work = False
            confidence = 'high'
            reason = "Most geometric quality indicators suggest weak geometric signal"

        # Generate recommendations
        recommendations = []

        if predictors['separation_accuracy_correlation'] < 0.3:
            recommendations.append("⚠️  Low separation-accuracy correlation suggests geometric signal is weak")

        if predictors['safe_dangerous_gap'] < 0.1:
            recommendations.append("⚠️  Small safe/dangerous gap - geometric boundary not meaningful")

        if predictors['distance_ratio'] < 1.2:
            recommendations.append("⚠️  Low distance ratio - classes not well separated in feature space")

        if predictors['impostor_rate'] > 0.4:
            recommendations.append("⚠️  High impostor rate - many points closer to wrong class")

        if predictors['nn_class_purity'] < 0.6:
            recommendations.append("⚠️  Low NN purity - classes are mixed in local neighborhoods")

        if predictors['effective_rank'] < 0.3:
            recommendations.append("⚠️  Low effective rank - features may be collapsed/degenerate")

        if will_work:
            if predictors['separation_entropy'] > 3.5:
                recommendations.append("✅ High separation entropy - good diversity for isotonic regression")
            if predictors['nc4_nearest_centroid_accuracy'] > 0.8:
                recommendations.append("✅ High NC4 - strong class separation in feature space")

        result = {
            'will_work': will_work,
            'confidence': confidence,
            'reason': reason,
            'predictors': predictors,
            'conditions_met': {k: v for k, v in conditions.items()},
            'n_conditions_satisfied': n_satisfied,
            'recommendations': recommendations if recommendations else ["✅ Geometric calibration likely to work well"]
        }

        # Log summary
        logger.info("\n" + "="*80)
        logger.info("CALIBRATION SUCCESS PREDICTION")
        logger.info("="*80)
        logger.info(f"Will Work: {'✅ YES' if will_work else '❌ NO'}")
        logger.info(f"Confidence: {confidence.upper()}")
        logger.info(f"Reason: {reason}")
        logger.info(f"Conditions Satisfied: {n_satisfied}/4")
        logger.info("\nConditions:")
        for condition, satisfied in conditions.items():
            logger.info(f"  {'✅' if satisfied else '❌'} {condition}")
        logger.info("\nRecommendations:")
        for rec in recommendations:
            logger.info(f"  {rec}")
        logger.info("="*80 + "\n")

        return result
