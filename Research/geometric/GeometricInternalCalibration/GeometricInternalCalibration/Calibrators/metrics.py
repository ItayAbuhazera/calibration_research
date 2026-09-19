import numpy as np
import logging
import torch
import time
import torch.nn as nn
import inspect
from typing import Optional, Dict, List, Tuple, Union, Any
from typing import TYPE_CHECKING
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

from scipy.stats import kendalltau, spearmanr

from utils.stability_space import StabilitySpace
from utils.compression_utils import SmartCompression, FixedSizeSPP_JL
from utils.tensor_utils import coerce_to_tensor
if TYPE_CHECKING:
    from Calibrators.geometric_calibrator_new import GeometricCalibrator


from utils.logging_config import get_logger
logger = get_logger(__name__)


class MetricCalculator(ABC):
    """Base class for metric calculators"""
    
    @abstractmethod
    def compute(self, features: torch.Tensor, labels: torch.Tensor, 
                corrupted_features: Optional[torch.Tensor] = None,
                predictions: Optional[np.ndarray] = None,
                confidences: Optional[np.ndarray] = None, **kwargs) -> float:
        """Compute the metric given features and labels"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric"""
        pass


# ============================================================================
# NEW UNCERTAINTY-GEOMETRY CORRELATION METRICS
# ============================================================================

class ConfidenceDistanceCorrelationCalculator(MetricCalculator):
    """Measure correlation between distance to class centroid and prediction confidence"""
    
    @property
    def name(self) -> str:
        return "confidence_distance_correlation"
    
    def compute(self, features: torch.Tensor, labels: torch.Tensor, 
                corrupted_features: Optional[torch.Tensor] = None,
                predictions: Optional[np.ndarray] = None,
                confidences: Optional[np.ndarray] = None, **kwargs) -> float:
        
        if confidences is None:
            logger.warning("No confidences provided for confidence-distance correlation")
            return 0.0
            
        try:
            features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            
            # Compute class centroids
            unique_labels = np.unique(labels_np)
            centroids = {}
            for label in unique_labels:
                mask = labels_np == label
                if np.any(mask):
                    centroids[label] = features_np[mask].mean(axis=0)
            
            # Calculate distance to own class centroid
            distances_to_centroid = []
            valid_indices = []
            
            for i, (feat, label) in enumerate(zip(features_np, labels_np)):
                if label in centroids:
                    dist = np.linalg.norm(feat - centroids[label])
                    distances_to_centroid.append(dist)
                    valid_indices.append(i)
            
            if len(distances_to_centroid) < 2:
                return 0.0
                
            # Convert to proper numpy arrays with finite mask
            distances_to_centroid = np.asarray(distances_to_centroid, dtype=np.float64)
            valid_confidences = np.asarray(confidences[valid_indices], dtype=np.float64)
            
            # Clip confidences to [0,1] to be safe
            valid_confidences = np.clip(valid_confidences, 0.0, 1.0)
            
            # Finite mask
            m = np.isfinite(distances_to_centroid) & np.isfinite(valid_confidences)
            if m.sum() < 3:
                return 0.0
            
            # Uncertainty = 1 - confidence
            uncertainties = 1.0 - valid_confidences[m]
            
            # Spearman correlation; expected sign: distance ↑ ⇒ uncertainty ↑ (positive)
            rho, _ = spearmanr(distances_to_centroid[m], uncertainties)
            if not np.isfinite(rho):
                return 0.0
            
            # Map [-1,1] → [0,1] so 0.5 = no relation; 1 = perfect alignment
            return float(0.5 * (rho + 1.0))
            
        except Exception as e:
            logger.warning(f"Confidence-distance correlation failed: {e}")
            return 0.0


class BoundaryProximityCorrelationCalculator(MetricCalculator):
    """Measure correlation between proximity to class boundaries and uncertainty"""
    
    def __init__(self, k: int = 5):
        self.k = k
    
    @property
    def name(self) -> str:
        return "boundary_proximity_correlation"
    
    def compute(self, features: torch.Tensor, labels: torch.Tensor, 
                corrupted_features: Optional[torch.Tensor] = None,
                predictions: Optional[np.ndarray] = None,
                confidences: Optional[np.ndarray] = None, **kwargs) -> float:
        
        if confidences is None:
            logger.warning("No confidences provided for boundary proximity correlation")
            return 0.0
            
        try:
            features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            
            if len(features_np) < 3:
                return 0.0
            if len(features_np) < self.k + 1:
                return 0.0
            
            # Find k nearest neighbors for each sample
            k_actual = min(self.k, len(features_np) - 1)
            nbrs = NearestNeighbors(n_neighbors=k_actual + 1).fit(features_np)
            distances, indices = nbrs.kneighbors(features_np)
            
            # Calculate boundary proximity scores
            boundary_scores = []
            for i in range(len(features_np)):
                neighbor_indices = indices[i][1:]  # Exclude self
                neighbor_labels = labels_np[neighbor_indices]
                
                # Fraction of neighbors with different class (boundary proximity)
                different_class_fraction = np.mean(neighbor_labels != labels_np[i])
                boundary_scores.append(different_class_fraction)
            
            # Early return if no variance in boundary scores (all same class neighbors)
            boundary_scores = np.array(boundary_scores)
            if np.std(boundary_scores) < 1e-8:
                return 0.0
            
            # Convert to proper numpy arrays with finite mask
            boundary_scores = np.asarray(boundary_scores, dtype=np.float64)
            confidences = np.asarray(confidences, dtype=np.float64)
            confidences = np.clip(confidences, 0.0, 1.0)
            
            m = np.isfinite(boundary_scores) & np.isfinite(confidences)
            if m.sum() < 3:
                return 0.0
            
            uncertainties = 1.0 - confidences[m]
            
            # Expected sign: boundary proximity ↑ (more different-class neighbors) ⇒ uncertainty ↑ (positive)
            rho, _ = spearmanr(boundary_scores[m], uncertainties)
            if not np.isfinite(rho):
                return 0.0
            return float(0.5 * (rho + 1.0))
            
        except Exception as e:
            logger.warning(f"Boundary proximity correlation failed: {e}")
            return 0.0


class UncertaintyGeometryAlignmentCalculator(MetricCalculator):
    """Overall measure of how well geometric structure aligns with uncertainty"""
    
    @property
    def name(self) -> str:
        return "uncertainty_geometry_alignment"
    
    def compute(self, features: torch.Tensor, labels: torch.Tensor, 
                corrupted_features: Optional[torch.Tensor] = None,
                predictions: Optional[np.ndarray] = None,
                confidences: Optional[np.ndarray] = None, **kwargs) -> float:
        
        if confidences is None or predictions is None:
            logger.warning("No confidences/predictions provided for uncertainty-geometry alignment")
            return 0.0
            
        try:
            features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            
            # Early return for tiny datasets
            if len(features_np) < 2:
                return 0.0
            
            # Multiple geometric proxies for uncertainty
            correctness = (predictions == labels_np).astype(float)
            
            # 1. Distance to class centroid
            unique_labels = np.unique(labels_np)
            centroids = {}
            for label in unique_labels:
                mask = labels_np == label
                if np.any(mask):
                    centroids[label] = features_np[mask].mean(axis=0)
            
            centroid_distances = []
            for feat, label in zip(features_np, labels_np):
                if label in centroids:
                    dist = np.linalg.norm(feat - centroids[label])
                    centroid_distances.append(dist)
                else:
                    centroid_distances.append(0)
            
            centroid_distances = np.array(centroid_distances)
            
            # 2. Distance to decision boundary (fast k-NN with labels)
            from sklearn.neighbors import NearestNeighbors
            kn = min(16, len(features_np) - 1) if len(features_np) > 1 else 1
            nbrs = NearestNeighbors(n_neighbors=kn + 1).fit(features_np)
            dists, inds = nbrs.kneighbors(features_np)
            boundary_distances = np.full(len(features_np), np.inf)
            for i in range(len(features_np)):
                mask = labels_np[inds[i, 1:]] != labels_np[i]
                if np.any(mask):
                    boundary_distances[i] = dists[i, 1:][mask][0]
            # Keep non-finite boundary distances as NaN for proper masking in correlations
            boundary_distances = np.array(boundary_distances)
            
            # Convert to proper numpy arrays with finite masks
            centroid_distances = np.asarray(centroid_distances, dtype=np.float64)
            boundary_distances = np.asarray(boundary_distances, dtype=np.float64)
            uncertainties = np.asarray(1.0 - confidences, dtype=np.float64)
            correctness = np.asarray((predictions == labels_np).astype(float), dtype=np.float64)
            
            # Keep any +inf as NaN and mask them out in correlations
            boundary_distances[~np.isfinite(boundary_distances)] = np.nan
            
            def _pref_spearman(a, b, expect_positive=True):
                """Spearman -> [0,1] with expected sign. Returns None if insufficient."""
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 3:
                    return None
                rho, _ = spearmanr(a[m], b[m])
                if not np.isfinite(rho):
                    return None
                # Map to [0,1] with sign preference
                return float(0.5 * ((rho if expect_positive else -rho) + 1.0))
            
            scores = []
            
            # 1) centroid distance vs uncertainty: expect POSITIVE (farther from own centroid ⇒ more uncertain)
            s = _pref_spearman(centroid_distances, uncertainties, expect_positive=True)
            if s is not None:
                scores.append(s)
            
            # 2) centroid distance vs correctness: expect NEGATIVE (farther ⇒ less correct)
            s = _pref_spearman(centroid_distances, correctness, expect_positive=False)
            if s is not None:
                scores.append(s)
            
            # 3) boundary distance vs uncertainty: expect NEGATIVE (farther from boundary ⇒ less uncertain)
            s = _pref_spearman(boundary_distances, uncertainties, expect_positive=False)
            if s is not None:
                scores.append(s)
            
            return float(np.mean(scores)) if scores else 0.0
                
        except Exception as e:
            logger.warning(f"Uncertainty-geometry alignment failed: {e}")
            return 0.0


class ClassSeparationQualityCalculator(MetricCalculator):
    """Enhanced class separation quality with margin analysis"""
    
    @property
    def name(self) -> str:
        return "avg_class_separation_ratio"
    
    def compute(self, features: torch.Tensor, labels: torch.Tensor, 
                corrupted_features: Optional[torch.Tensor] = None,
                predictions: Optional[np.ndarray] = None,
                confidences: Optional[np.ndarray] = None, **kwargs) -> float:
        
        try:
            features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            
            unique_labels = np.unique(labels_np)
            n_classes = len(unique_labels)
            
            if n_classes < 2:
                return 0.0
            
            # Compute class centroids
            centroids = {}
            for label in unique_labels:
                mask = labels_np == label
                if np.any(mask):
                    centroids[label] = features_np[mask].mean(axis=0)
            
            separation_ratios = []
            
            for label in unique_labels:
                if label not in centroids:
                    continue
                    
                class_mask = labels_np == label
                class_features = features_np[class_mask]
                
                if len(class_features) < 2:
                    continue
                
                # Intra-class variance (compactness) - vectorized
                class_centroid = centroids[label]
                diffs = class_features - class_centroid
                avg_intra_distance = float(np.linalg.norm(diffs, axis=1).mean())
                
                # Inter-class distances
                inter_distances = []
                for other_label, other_centroid in centroids.items():
                    if other_label != label:
                        inter_dist = np.linalg.norm(class_centroid - other_centroid)
                        inter_distances.append(inter_dist)
                
                if inter_distances and avg_intra_distance > 0:
                    avg_inter_distance = np.mean(inter_distances)
                    separation_ratio = avg_inter_distance / avg_intra_distance
                    separation_ratios.append(separation_ratio)
            
            return np.mean(separation_ratios) if separation_ratios else 0.0
            
        except Exception as e:
            logger.warning(f"Class separation quality failed: {e}")
            return 0.0


class MarginQualityCalculator(MetricCalculator):
    """Measure quality of class margins
    
    Note: This calculator is defined but not included in the default metric set.
    It can be added if margin quality analysis is specifically needed.
    """
    
    @property
    def name(self) -> str:
        return "good_margin_fraction"
    
    def compute(self, features: torch.Tensor, labels: torch.Tensor, 
                corrupted_features: Optional[torch.Tensor] = None,
                predictions: Optional[np.ndarray] = None,
                confidences: Optional[np.ndarray] = None, **kwargs) -> float:
        
        try:
            # Cap the amount of pairwise work by sampling
            max_pairs = 20000
            features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            
            unique_labels = np.unique(labels_np)
            
            # Compute class centroids
            centroids = {}
            for label in unique_labels:
                mask = labels_np == label
                if np.any(mask):
                    centroids[label] = features_np[mask].mean(axis=0)
            
            good_margin_count = 0
            total_samples = 0
            
            for feat, label in zip(features_np, labels_np):
                if label not in centroids:
                    continue
                    
                # Distance to own class centroid
                own_distance = np.linalg.norm(feat - centroids[label])
                
                # Distance to nearest other class centroid
                min_other_distance = float('inf')
                for other_label, other_centroid in centroids.items():
                    if other_label != label:
                        other_distance = np.linalg.norm(feat - other_centroid)
                        min_other_distance = min(min_other_distance, other_distance)
                
                # Good margin: closer to own class than to any other class
                if min_other_distance != float('inf'):
                    margin_ratio = own_distance / min_other_distance
                    if margin_ratio < 0.8:  # Significantly closer to own class
                        good_margin_count += 1
                    total_samples += 1
            
            return good_margin_count / total_samples if total_samples > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Margin quality calculation failed: {e}")
            return 0.0




class LocalIntrinsicDimensionalityCalculator(MetricCalculator):
    """Estimate local intrinsic dimensionality"""
    
    def __init__(self, k: int = 10):
        self.k = k
    
    @property
    def name(self) -> str:
        return "local_intrinsic_dimensionality"
    
    def compute(self, features: torch.Tensor, labels: torch.Tensor, 
                corrupted_features: Optional[torch.Tensor] = None,
                predictions: Optional[np.ndarray] = None,
                confidences: Optional[np.ndarray] = None, **kwargs) -> float:
        
        try:
            features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
            
            n_samples = len(features_np)
            k = min(self.k, n_samples - 1)
            
            if k <= 1:
                return 0.0
            
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(features_np)
            distances, indices = nbrs.kneighbors(features_np)
            
            local_ids = []
            for i in range(n_samples):
                # Get distances to k nearest neighbors (excluding self)
                neighbor_dists = np.sort(distances[i][1:])
                if len(neighbor_dists) >= 2:
                    rk = neighbor_dists[-1]
                    if rk > 0:
                        logs = np.log(np.maximum(neighbor_dists[:-1], 1e-12) / rk)
                        denom = np.mean(logs)
                        if denom != 0:
                            m_hat = -1.0 / denom
                            if np.isfinite(m_hat) and m_hat > 0:
                                local_ids.append(m_hat)
            
            if local_ids:
                # Filter out any non-finite values before averaging
                finite_lids = [lid for lid in local_ids if np.isfinite(lid)]
                if finite_lids:
                    avg_lid = np.mean(finite_lids)
                    # Clip to reasonable upper bound to prevent extreme values
                    avg_lid = np.clip(avg_lid, 0.0, 100.0)
                    # Normalize: lower intrinsic dimensionality is often better for later layers
                    # Return 1/LID normalized to reasonable range
                    return 1 / (1 + avg_lid)
            return 0.0
                
        except Exception as e:
            logger.warning(f"Local intrinsic dimensionality failed: {e}")
            return 0.0






class ReliabilityCurveQualityCalculator(MetricCalculator):
    """
    Reliability curve quality using a geometric proxy confidence from centroid margins.
    Combines Kendall tau monotonicity and area-to-diagonal quality into a single score.
    """
    def __init__(self, n_bins: int = 15, margin_temp_grid=(0.5, 1.0, 2.0, 5.0, 10.0)):
        self.n_bins = n_bins
        self.temps = margin_temp_grid

    @property
    def name(self) -> str:
        return "reliability_curve_quality"

    def compute(self, features, labels, **kwargs) -> float:
        try:
            X = features.cpu().numpy() if torch.is_tensor(features) else features
            y = labels.cpu().numpy() if torch.is_tensor(labels) else labels
            classes = np.unique(y)
            if len(classes) < 2 or len(X) < 10:
                return 0.0

            mus = {c: X[y == c].mean(axis=0) for c in classes}
            own = np.array([np.linalg.norm(x - mus[yy]) for x, yy in zip(X, y)])
            other = np.array([min(np.linalg.norm(x - mus[c]) for c in classes if c != yy)
                              for x, yy in zip(X, y)])
            margins = other - own
            margins = (margins - np.median(margins)) / (np.std(margins) + 1e-8)

            def proxy_conf(alpha):
                return 1.0 / (1.0 + np.exp(-alpha * margins))

            best_tau, best_aucq = 0.0, 0.0
            for a in self.temps:
                conf = proxy_conf(a)
                bin_edges = np.linspace(0, 1, self.n_bins + 1)
                idx = np.digitize(conf, bin_edges) - 1
                idx = np.clip(idx, 0, self.n_bins - 1)
                centers, accs = [], []
                # approximate correctness via nearest prototype
                nn_d = np.stack([np.linalg.norm(X - mus[c], axis=1) for c in classes], axis=1)
                nn_pred = np.argmin(nn_d, axis=1)
                y_idx = np.array([np.where(classes == yy)[0][0] for yy in y])
                for b in range(self.n_bins):
                    m = (idx == b)
                    if m.sum() >= 10:
                        centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                        accs.append((nn_pred[m] == y_idx[m]).mean())
                if len(centers) < 3:
                    continue
                centers, accs = np.array(centers), np.array(accs)
                order = np.argsort(centers)
                centers = centers[order]
                accs = accs[order]
                tau, _ = kendalltau(centers, accs)
                tau = 0.0 if np.isnan(tau) else max(0.0, float(tau))
                auc_est = float(np.trapz(accs, centers))
                diag_auc = 0.5
                auc_quality = max(0.0, 1.0 - abs(auc_est - diag_auc) / 0.5)
                if 0.5 * tau + 0.5 * auc_quality > 0.5 * best_tau + 0.5 * best_aucq:
                    best_tau, best_aucq = tau, auc_quality
            return 0.6 * best_tau + 0.4 * best_aucq
        except Exception as e:
            logger.warning(f"ReliabilityCurveQuality failed: {e}")
            return 0.0




class PrototypeSoftmaxECECalculator(MetricCalculator):
    """
    ECE of distance-to-prototype softmax with a single temperature.
    Returns NEGATIVE ECE so higher is better.
    """
    def __init__(self, temps=(0.1, 0.3, 1.0, 3.0, 10.0), n_bins: int = 15):
        self.temps = temps
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return "prototype_softmax_ece"

    def compute(self, features, labels, **kwargs) -> float:
        try:
            X = features.cpu().numpy() if torch.is_tensor(features) else features
            y = labels.cpu().numpy() if torch.is_tensor(labels) else labels
            classes = np.unique(y)
            if len(classes) < 2 or len(X) < max(10, self.n_bins):
                return 0.0
            mus = {c: X[y == c].mean(axis=0) for c in classes}
            sig = {c: X[y == c].std(axis=0).mean() + 1e-8 for c in classes}
            D = np.stack([
                np.linalg.norm((X - mus[c]) / sig[c], axis=1) ** 2
                for c in classes
            ], axis=1)

            def ece_of_probs(P, y_idx):
                conf = P.max(axis=1); pred = P.argmax(axis=1)
                acc = (pred == y_idx).astype(float)
                bins = np.linspace(0, 1, self.n_bins + 1)
                ece = 0.0
                for lo, hi in zip(bins[:-1], bins[1:]):
                    m = (conf > lo) & (conf <= hi)
                    if m.any():
                        ece += abs(acc[m].mean() - conf[m].mean()) * m.mean()
                return ece

            y_idx = np.array([np.where(classes == yy)[0][0] for yy in y])
            best = 1.0
            for t in self.temps:
                # Numerical stability: subtract max to prevent overflow
                log_P = -t * D
                log_P_max = np.max(log_P, axis=1, keepdims=True)
                P = np.exp(log_P - log_P_max)
                P_sum = P.sum(axis=1, keepdims=True)
                # Avoid division by zero
                P = np.where(P_sum > 0, P / P_sum, 1.0 / P.shape[1])
                best = min(best, ece_of_probs(P, y_idx))
            return -float(best)
        except Exception as e:
            logger.warning(f"PrototypeSoftmaxECE failed: {e}")
            return np.nan


class LabelCKACalculator(MetricCalculator):
    """Linear CKA between features and one-hot labels (scale-invariant)."""
    @property
    def name(self) -> str:
        return "label_cka"

    def compute(self, features, labels, **kwargs) -> float:
        try:
            X = features.cpu().numpy() if torch.is_tensor(features) else features
            y = labels.cpu().numpy() if torch.is_tensor(labels) else labels
            classes, y_idx = np.unique(y, return_inverse=True)
            C = len(classes)
            if C < 2:
                return 0.0
            Y = np.eye(C)[y_idx]
            Xc = X - X.mean(axis=0, keepdims=True)
            Yc = Y - Y.mean(axis=0, keepdims=True)
            num = np.linalg.norm(Xc.T @ Yc, ord='fro') ** 2
            den = np.linalg.norm(Xc.T @ Xc, ord='fro') * np.linalg.norm(Yc.T @ Yc, ord='fro') + 1e-12
            return float(num / den)
        except Exception as e:
            logger.warning(f"LabelCKA failed: {e}")
            return 0.0


class MarginTailCVaRCalculator(MetricCalculator):
    """
    Conditional Value-at-Risk (CVaR) of the worst margins.
    Higher is better: return 1 - normalized CVaR so larger means safer tails.
    
    Args:
        alpha: Fraction of worst margins to consider for CVaR (default: 0.2 = 20%)
        normalization_method: Method for normalizing CVaR. Options:
            - "linear": Old method: 0.5 + (cvar / denom), clipped to [0, 1]
            - "tanh": New method: 0.5 * (1 + tanh(cvar / (denom * scale))), smooth mapping
            Default: "tanh"
        normalization_scale: Scale factor for tanh normalization (only used with "tanh" method).
            Larger values make the metric less sensitive to CVaR changes. Default: 2.0
    """
    def __init__(self, alpha: float = 0.2, normalization_method: str = "tanh", normalization_scale: float = 2.0):
        self.alpha = alpha
        self.normalization_method = normalization_method
        self.normalization_scale = normalization_scale
        
        if normalization_method not in ["linear", "tanh"]:
            raise ValueError(f"normalization_method must be 'linear' or 'tanh', got '{normalization_method}'")

    @property
    def name(self) -> str:
        return "margin_tail_cvar"

    def compute_raw_cvar(self, features, labels, **kwargs) -> Dict[str, float]:
        """
        Compute raw CVaR values without normalization.
        Returns a dictionary with 'cvar', 'std', and 'margins' for reuse.
        """
        try:
            # Convert to numpy if needed
            X = features.cpu().numpy() if torch.is_tensor(features) else features
            y = labels.cpu().numpy() if torch.is_tensor(labels) else labels
            
            # Validate inputs
            if X is None or y is None:
                logger.warning(f"MarginTailCVaR: features or labels is None")
                return {'cvar': 0.0, 'std': 1.0, 'margins': np.array([])}
            
            if len(X) == 0 or len(y) == 0:
                logger.warning(f"MarginTailCVaR: empty features or labels")
                return {'cvar': 0.0, 'std': 1.0, 'margins': np.array([])}
            
            if len(X) != len(y):
                logger.warning(f"MarginTailCVaR: length mismatch (X: {len(X)}, y: {len(y)})")
                return {'cvar': 0.0, 'std': 1.0, 'margins': np.array([])}
            
            classes = np.unique(y)
            if len(classes) < 2:
                logger.info(f"MarginTailCVaR: only {len(classes)} unique class(es), need at least 2")
                return {'cvar': 0.0, 'std': 1.0, 'margins': np.array([])}
            
            # Compute class means
            mus = {c: X[y == c].mean(axis=0) for c in classes}
            
            # Compute distances to own class and nearest other class
            own = np.array([np.linalg.norm(x - mus[yy]) for x, yy in zip(X, y)])
            other = np.array([min(np.linalg.norm(x - mus[c]) for c in classes if c != yy)
                              for x, yy in zip(X, y)])
            
            # Compute margins (positive = correct side of decision boundary)
            margins = other - own
            
            # Compute CVaR of worst alpha% of margins
            k = max(1, int(self.alpha * len(margins)))
            worst = np.sort(margins)[:k]
            cvar = float(np.mean(worst))
            
            # Normalize by standard deviation
            denom = float(np.std(margins) + 1e-8)
            
            return {'cvar': cvar, 'std': denom, 'margins': margins}
            
        except Exception as e:
            import traceback
            logger.warning(f"MarginTailCVaR compute_raw_cvar failed: {e}")
            logger.debug(f"MarginTailCVaR traceback: {traceback.format_exc()}")
            return {'cvar': 0.0, 'std': 1.0, 'margins': np.array([])}
    
    def normalize_cvar(self, cvar: float, std: float, normalization_method: str = None, normalization_scale: float = None) -> float:
        """
        Apply normalization to raw CVaR value.
        Allows applying different normalization methods/scales to the same raw CVaR.
        """
        if normalization_method is None:
            normalization_method = self.normalization_method
        if normalization_scale is None:
            normalization_scale = self.normalization_scale
        
        denom = std + 1e-8
        
        if normalization_method == "linear":
            # Old method: linear normalization with clipping
            result = float(np.clip(0.5 + (cvar / denom), 0.0, 1.0))
        elif normalization_method == "tanh":
            # New method: tanh-based smooth normalization
            normalized_cvar = cvar / (denom * normalization_scale)
            result = float(0.5 * (1.0 + np.tanh(normalized_cvar)))
        else:
            raise ValueError(f"Unknown normalization_method: {normalization_method}")
        
        return result

    def compute(self, features, labels, **kwargs) -> float:
        """
        Compute normalized CVaR using the instance's normalization settings.
        For efficiency, use compute_raw_cvar() + normalize_cvar() when testing multiple normalizations.
        """
        try:
            logger.info(f"MarginTailCVaR.compute() called with alpha={self.alpha}")
            raw_data = self.compute_raw_cvar(features, labels, **kwargs)
            
            if raw_data['cvar'] == 0.0 and len(raw_data['margins']) == 0:
                return 0.0
            
            result = self.normalize_cvar(
                raw_data['cvar'], 
                raw_data['std'],
                self.normalization_method,
                self.normalization_scale
            )
            
            logger.info(f"MarginTailCVaR(alpha={self.alpha}, method={self.normalization_method}, scale={self.normalization_scale}): computed value={result:.4f} (CVaR={raw_data['cvar']:.4f}, std={raw_data['std']:.4f})")
            
            return result
            
        except Exception as e:
            import traceback
            logger.warning(f"MarginTailCVaR failed: {e}")
            logger.debug(f"MarginTailCVaR traceback: {traceback.format_exc()}")
            return 0.0


# ============================================================================
# MarginTailCVaR Alpha Variants for Ablation Study
# ============================================================================

class MarginTailCVaR_005(MarginTailCVaRCalculator):
    """Margin Tail CVaR with alpha=0.05 (5% worst tail)"""
    def __init__(self):
        super().__init__(alpha=0.05)

    @property
    def name(self) -> str:
        return "margin_tail_cvar_005"


class MarginTailCVaR_010(MarginTailCVaRCalculator):
    """Margin Tail CVaR with alpha=0.10 (10% worst tail)"""
    def __init__(self):
        super().__init__(alpha=0.10)

    @property
    def name(self) -> str:
        return "margin_tail_cvar_010"


class MarginTailCVaR_015(MarginTailCVaRCalculator):
    """Margin Tail CVaR with alpha=0.15 (15% worst tail)"""
    def __init__(self):
        super().__init__(alpha=0.15)

    @property
    def name(self) -> str:
        return "margin_tail_cvar_015"


class MarginTailCVaR_020(MarginTailCVaRCalculator):
    """Margin Tail CVaR with alpha=0.20 (20% worst tail) - BASELINE"""
    def __init__(self):
        super().__init__(alpha=0.20)

    @property
    def name(self) -> str:
        return "margin_tail_cvar_020"


class MarginTailCVaR_025(MarginTailCVaRCalculator):
    """Margin Tail CVaR with alpha=0.25 (25% worst tail)"""
    def __init__(self):
        super().__init__(alpha=0.25)

    @property
    def name(self) -> str:
        return "margin_tail_cvar_025"


class MarginTailCVaR_030(MarginTailCVaRCalculator):
    """Margin Tail CVaR with alpha=0.30 (30% worst tail)"""
    def __init__(self):
        super().__init__(alpha=0.30)

    @property
    def name(self) -> str:
        return "margin_tail_cvar_030"


class GeometryErrorConcordanceCalculator(MetricCalculator):
    """
    Geometry–Error Concordance (GEC): agreement between a geometric proxy score and correctness.
    Uses ROC–AUC (primary) with optional Spearman bonus.
    """
    def __init__(self, proxy: str = "centroid_margin", knn_k: int = 10,
                 use_spearman: bool = True, max_n: int = 20000, rng_seed: int = 0):
        assert proxy in ("centroid_margin", "boundary_knn")
        self.proxy = proxy
        self.knn_k = knn_k
        self.use_spearman = use_spearman
        self.max_n = max_n
        self.rng = np.random.RandomState(rng_seed)

    @property
    def name(self) -> str:
        return "geometry_error_concordance"

    def _maybe_subsample(self, X: np.ndarray, y: np.ndarray, yhat: np.ndarray):
        if len(y) <= self.max_n:
            return X, y, yhat
        idx = self.rng.choice(len(y), self.max_n, replace=False)
        return X[idx], y[idx], yhat[idx]

    def _centroid_margin(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        classes = np.unique(y)
        mus = {c: X[y == c].mean(axis=0) for c in classes}
        own = np.zeros(len(X))
        other_min = np.zeros(len(X))
        for i in range(len(X)):
            yi = y[i]
            xi = X[i]
            own[i] = np.linalg.norm(xi - mus[yi])
            md = np.inf
            for cj, mu in mus.items():
                if cj == yi:
                    continue
                d = np.linalg.norm(xi - mu)
                if d < md:
                    md = d
            other_min[i] = 0.0 if np.isinf(md) else md
        return other_min - own

    def _boundary_knn_purity(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.neighbors import NearestNeighbors
        k = min(self.knn_k, len(X) - 1)
        if k < 1:
            return np.zeros(len(X))
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        _, idx = nbrs.kneighbors(X)
        idx = idx[:, 1:]
        purity = np.zeros(len(X))
        for i in range(len(X)):
            purity[i] = 1.0 - np.mean(y[idx[i]] != y[i])
        return purity

    def compute(self, features: torch.Tensor, labels: torch.Tensor,
                corrupted_features: Optional[torch.Tensor] = None,
                predictions: Optional[np.ndarray] = None,
                confidences: Optional[np.ndarray] = None, **kwargs) -> float:
        from sklearn.metrics import roc_auc_score
        from scipy.stats import spearmanr
        if predictions is None:
            return 0.0
        X = features.cpu().numpy() if torch.is_tensor(features) else features
        y = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        yhat = np.asarray(predictions)
        X, y, yhat = self._maybe_subsample(X, y, yhat)
        s = self._centroid_margin(X, y) if self.proxy == "centroid_margin" else self._boundary_knn_purity(X, y)
        correct = (yhat == y).astype(int)
        if np.std(s) < 1e-9 or len(np.unique(correct)) < 2:
            return 0.0
        try:
            auc = float(roc_auc_score(correct, s))
        except Exception:
            auc = 0.5
        if not self.use_spearman:
            return float(np.clip(auc, 0.0, 1.0))
        rho, _ = spearmanr(s, correct)
        if np.isnan(rho):
            rho = 0.0
        return float(np.clip(0.8 * auc + 0.2 * max(0.0, rho), 0.0, 1.0))


class LabelCKAHSICCalculator(MetricCalculator):
    """Label–CKA / HSIC between features and labels (higher is better)."""
    def __init__(self, mode: str = "linear_cka", rbf_gamma: Optional[float] = None,
                 subsample: Optional[int] = 1500, rng_seed: int = 0):
        assert mode in ("linear_cka", "hsic_rbf")
        self.mode = mode
        self.gamma = rbf_gamma
        self.subsample = subsample
        self.rng = np.random.RandomState(rng_seed)

    @property
    def name(self) -> str:
        return "label_cka_hsic"

    def _maybe_subsample(self, X: np.ndarray, y: np.ndarray):
        if self.subsample is None or len(y) <= self.subsample:
            return X, y
        idx = self.rng.choice(len(y), self.subsample, replace=False)
        return X[idx], y[idx]

    @staticmethod
    def _center_gram(K: np.ndarray) -> np.ndarray:
        n = K.shape[0]
        H = np.eye(n) - np.full((n, n), 1.0 / n)
        return H @ K @ H

    @staticmethod
    def _linear_gram(X: np.ndarray) -> np.ndarray:
        return X @ X.T

    @staticmethod
    def _rbf_gram(X: np.ndarray, gamma: Optional[float]) -> np.ndarray:
        XX = np.sum(X * X, axis=1, keepdims=True)
        dist2 = XX + XX.T - 2.0 * (X @ X.T)
        if gamma is None:
            tri = dist2[np.triu_indices(len(X), 1)]
            med = np.median(tri) if tri.size > 0 else 1.0
            gamma = 1.0 / (med + 1e-12)
        return np.exp(-gamma * dist2)

    @staticmethod
    def _delta_gram(y: np.ndarray) -> np.ndarray:
        return (y[:, None] == y[None, :]).astype(np.float64)

    def _cka_linear(self, X: np.ndarray, y: np.ndarray) -> float:
        Kx = self._center_gram(self._linear_gram(X))
        Ky = self._center_gram(self._delta_gram(y))
        hsic_xy = np.trace(Kx @ Ky)
        hsic_xx = np.trace(Kx @ Kx)
        hsic_yy = np.trace(Ky @ Ky)
        denom = np.sqrt(hsic_xx * hsic_yy) + 1e-12
        if denom <= 0:
            return 0.0
        cka = hsic_xy / denom
        return float(np.clip(cka, 0.0, 1.0))

    def _hsic_rbf(self, X: np.ndarray, y: np.ndarray) -> float:
        Kx = self._center_gram(self._rbf_gram(X, self.gamma))
        Ky = self._center_gram(self._delta_gram(y))
        hsic_xy = np.trace(Kx @ Ky)
        hsic_xx = np.trace(Kx @ Kx)
        hsic_yy = np.trace(Ky @ Ky)
        denom = np.sqrt(hsic_xx * hsic_yy) + 1e-12
        if denom <= 0:
            return 0.0
        nhsic = hsic_xy / denom
        return float(np.clip(nhsic, 0.0, 1.0))

    def compute(self, features: torch.Tensor, labels: torch.Tensor,
                corrupted_features: Optional[torch.Tensor] = None,
                predictions: Optional[np.ndarray] = None,
                confidences: Optional[np.ndarray] = None, **kwargs) -> float:
        X = features.cpu().numpy() if torch.is_tensor(features) else features
        y = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        if X is None or y is None or len(y) < 3:
            return 0.0
        X, y = self._maybe_subsample(X, y)
        try:
            return self._cka_linear(X, y) if self.mode == "linear_cka" else self._hsic_rbf(X, y)
        except Exception:
            return 0.0


class ClassSeparationUniformityCalculator(MetricCalculator):
    """Measure uniformity of separation across classes"""
    
    def __init__(self, max_pairs: int = 20000, rng_seed: int = 0):
        self.max_pairs = int(max_pairs)
        self.rng = np.random.default_rng(int(rng_seed))
    
    @property
    def name(self) -> str:
        return "class_separation_uniformity"
    
    def _sample_uniform_pairs(self, m: int, k: int):
        """Sample k uniform unordered pairs from m items"""
        # Total number of pairs: M = m*(m-1)//2
        M = m * (m - 1) // 2
        if k >= M:
            # Return all pairs
            i_indices, j_indices = [], []
            for i in range(m):
                for j in range(i + 1, m):
                    i_indices.append(i)
                    j_indices.append(j)
            return np.array(i_indices), np.array(j_indices)
        
        # Robust sampling: sample indices directly, avoiding bias
        i = self.rng.integers(0, m, size=k)
        j = self.rng.integers(0, m - 1, size=k)
        j = j + (j >= i)  # shift to avoid i==j
        i, j = np.minimum(i, j), np.maximum(i, j)
        return i, j
    
    def compute(self, features: torch.Tensor, labels: torch.Tensor, 
                corrupted_features: Optional[torch.Tensor] = None,
                predictions: Optional[np.ndarray] = None,
                confidences: Optional[np.ndarray] = None, **kwargs) -> float:
        
        try:
            features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
            labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            
            unique_labels = np.unique(labels_np)
            
            if len(unique_labels) < 2:
                return 0.0
            
            # Compute per-class separation scores
            class_separations = []
            
            for label in unique_labels:
                class_mask = labels_np == label
                class_features = features_np[class_mask]
                
                if len(class_features) < 2:
                    continue
                
                # Intra-class distance (sampled)
                m = len(class_features)
                total_intra_pairs = m * (m - 1) // 2
                intra_distances = []
                if total_intra_pairs <= self.max_pairs:
                    for i in range(m):
                        for j in range(i + 1, m):
                            dist = np.linalg.norm(class_features[i] - class_features[j])
                            intra_distances.append(dist)
                else:
                    # Use uniform sampling for unordered pairs
                    i_indices, j_indices = self._sample_uniform_pairs(m, self.max_pairs)
                    for i, j in zip(i_indices, j_indices):
                        dist = np.linalg.norm(class_features[i] - class_features[j])
                        intra_distances.append(dist)
                avg_intra = float(np.mean(intra_distances)) if intra_distances else 0.0
                
                # Inter-class distance
                other_features = features_np[~class_mask]
                if len(other_features) == 0:
                    continue
                
                # Inter-class distance (sampled)
                n_other = len(other_features)
                total_inter_pairs = m * n_other
                inter_distances = []
                if total_inter_pairs <= self.max_pairs:
                    for i in range(m):
                        class_feat = class_features[i]
                        for j in range(n_other):
                            other_feat = other_features[j]
                            dist = np.linalg.norm(class_feat - other_feat)
                            inter_distances.append(dist)
                else:
                    for _ in range(self.max_pairs):
                        i = self.rng.integers(0, m)
                        j = self.rng.integers(0, n_other)
                        dist = np.linalg.norm(class_features[i] - other_features[j])
                        inter_distances.append(dist)
                avg_inter = float(np.mean(inter_distances)) if inter_distances else 0.0
                
                if avg_intra > 0:
                    separation = avg_inter / avg_intra
                    class_separations.append(separation)
            
            if len(class_separations) < 2:
                return 0.0
            
            # Uniformity = 1 / coefficient_of_variation
            mean_sep = np.mean(class_separations)
            std_sep = np.std(class_separations)
            
            if mean_sep > 0 and std_sep > 0:
                cv = std_sep / mean_sep
                uniformity = 1 / (1 + cv)  # Higher = more uniform
                return uniformity
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Class separation uniformity failed: {e}")
            return 0.0


# ============================================================================
# EXISTING CALCULATORS (Enhanced with normalization)
# ============================================================================









class MultiScaleClassHomogeneityCalculator(MetricCalculator):
    """Multi-scale class homogeneity via k-NN purity at several k (numpy-based)."""
    def __init__(self, k_list: List[int] = (5, 15, 30)):
        self.k_list = list(k_list)

    @property
    def name(self) -> str:
        return "multiscale_separation"

    def compute(self, features: torch.Tensor, labels: torch.Tensor, **kwargs) -> float:
        try:
            from sklearn.neighbors import NearestNeighbors
            X = features.cpu().numpy() if torch.is_tensor(features) else features
            y = labels.cpu().numpy() if torch.is_tensor(labels) else labels
            n = len(X)
            if n < 3:
                return 0.0
            max_k = min(max(self.k_list), n - 1)
            nbrs = NearestNeighbors(n_neighbors=max_k + 1).fit(X)
            _, inds = nbrs.kneighbors(X)
            inds = inds[:, 1:]  # exclude self
            purities = []
            for k in self.k_list:
                k = min(k, max_k)
                same = (y[inds[:, :k]] == y[:, None]).mean()
                purities.append(same)
            return float(np.mean(purities))
        except Exception as e:
            logger.warning(f"MultiScaleClassHomogeneity failed: {e}")
            return 0.0






# =========================================================================
# Layer metrics container and selector utilities
# =========================================================================

@dataclass
class LayerMetrics:
    layer_idx: int
    # Core metrics (kept)
    ece_score: float = 0.0
    spearman_stability_accuracy: float = 0.0
    # uncertainty-geometry
    confidence_distance_correlation: float = 0.0
    boundary_proximity_correlation: float = 0.0
    uncertainty_geometry_alignment: float = 0.0
    # geometry
    avg_class_separation_ratio: float = 0.0
    class_separation_uniformity: float = 0.0
    local_intrinsic_dimensionality: float = 0.0
    # new metrics
    prototype_softmax_ece: float = 0.0
    reliability_curve_quality: float = 0.0
    label_cka: float = 0.0
    label_cka_hsic: float = 0.0
    margin_tail_cvar: float = 0.0
    multiscale_separation: float = 0.0
    logistic_calibratability: float = 0.0

    # ✅ NEW: cross-validated ECE utility in [0,1] (higher is better)
    kfold_ece_utility: float = 0.0

    # NEW DROP-IN METRICS
    decisiveness_tail_mass: float = 0.0
    confidence_entropy: float = 0.0
    confident_error_rate_tau: float = 0.0
    aurc_proxy: float = 0.0
    class_balanced_margin_cvar: float = 0.0
    impostor_gap_cvar: float = 0.0
    margin_skewkurt_safety: float = 0.0
    temperature_estimate_strength: float = 0.0

    composite_score: float = 0.0
    computation_times: Dict[str, float] = None

    def __post_init__(self):
        if self.computation_times is None:
            self.computation_times = {}


class CrossModalLayerMapper:
    """Map layer indices across different model architectures"""

    @staticmethod
    def get_conv_block_indices(model: nn.Module) -> List[int]:
        """Extract indices of convolutional blocks"""
        block_indices: List[int] = []
        modules = list(model.modules())
        for idx, module in enumerate(modules):
            # Check for common block patterns
            if isinstance(module, nn.Sequential):
                # Check if it contains Conv->BN->ReLU pattern
                has_conv = any(isinstance(m, nn.Conv2d) for m in module)
                has_norm = any(isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) for m in module)
                if has_conv and has_norm:
                    block_indices.append(idx)
            elif isinstance(module, nn.Conv2d):
                # Check if followed by normalization or activation
                if idx + 1 < len(modules):
                    next_module = modules[idx + 1]
                    if isinstance(next_module, (nn.BatchNorm2d, nn.ReLU, nn.GroupNorm)):
                        block_indices.append(idx)
        return block_indices

    @staticmethod
    def get_transformer_block_indices(model: nn.Module) -> List[int]:
        """Extract indices of transformer blocks"""
        block_indices: List[int] = []
        for idx, module in enumerate(model.modules()):
            # Common transformer block names
            name = module.__class__.__name__.lower()
            if any(key in name for key in ["transformer", "attention", "encoder", "decoder"]):
                block_indices.append(idx)
        return block_indices

    @staticmethod
    def get_vit_block_indices(model: nn.Module) -> List[int]:
        """Indices of Vision Transformer blocks (e.g., model.blocks[i])."""
        layers: List[nn.Module] = list(model.modules())
        name_to_index: Dict[nn.Module, int] = {m: i for i, m in enumerate(layers)}
        idxs: List[int] = []
        for name, module in model.named_modules():
            # Prefer inner tensor-only submodules:
            if re.search(r"(?:^|\.)(blocks|encoder\.layer|encoder\.layers)\.\d+\.(mlp\.fc2|attn\.proj|norm2)$", name):
                if module in name_to_index:
                    idxs.append(name_to_index[module])
            elif re.search(r"(?:^|\.)(blocks|encoder\.layer|encoder\.layers)\.\d+$", name):
                if module in name_to_index:
                    idxs.append(name_to_index[module])
        return sorted(set(idxs))

    @staticmethod
    def get_candidate_indices(model: nn.Module) -> List[int]:
        """Prefer ViT blocks when present; otherwise use conv blocks; fallback to transformer blocks."""
        vit = CrossModalLayerMapper.get_vit_block_indices(model)
        if vit:
            return vit
        conv = CrossModalLayerMapper.get_conv_block_indices(model)
        if conv:
            return conv
        trans = CrossModalLayerMapper.get_transformer_block_indices(model)
        if trans:
            return trans
        # Fallback: every 4th module index to keep list small
        return list(range(0, len(list(model.modules())), 4)) or [0]

    @staticmethod
    def get_audio_layer_indices(model: nn.Module) -> List[int]:
        """Extract indices suitable for audio models"""
        indices: List[int] = []
        for idx, module in enumerate(model.modules()):
            # Conv1d layers for audio
            if isinstance(module, nn.Conv1d):
                indices.append(idx)
            else:
                # Common audio architecture patterns
                name = module.__class__.__name__.lower()
                if any(key in name for key in ["wav2vec", "conformer", "jasper"]):
                    indices.append(idx)
        return indices


class AdHocLayerSelector:
    """Selector that computes weighted composite from a set of MetricCalculator(s)."""
    def __init__(self,
                 metrics: Optional[List[MetricCalculator]] = None,
                 score_weights: Optional[Dict[str, float]] = None,
                 enable_timing: bool = True,
                 cache_activations: bool = False,
                 fixed_feature_dim: int = 512,
                 compression_ratio: Optional[float] = None,
                 normalize_metrics: bool = True,
                 skip_zero_weight_metrics: bool = True,
                 direction_overrides: Optional[Dict[str, str]] = None):
        self.enable_timing = enable_timing
        self.cache_activations = cache_activations
        self._activation_cache = {}
        self.fixed_feature_dim = fixed_feature_dim
        self.compression_ratio = compression_ratio
        self.spp_projectors: Dict[int, FixedSizeSPP_JL] = {}
        self.normalize_metrics = normalize_metrics
        self.skip_zero_weight_metrics = skip_zero_weight_metrics
        # Direction overrides from empirical sanity check ("max" or "min")
        self.direction_overrides = direction_overrides or {
            # prefer MAX
            "calibration_decisiveness": "max",
            "avg_class_separation_ratio": "max",
            "ece_score": "max",                         # handled as utility inside composite
            "class_separation_uniformity": "max",
            "margin_tail_cvar": "max",
            "kfold_ece_utility": "max",
            "geometry_error_concordance": "max",
            "local_intrinsic_dimensionality": "max",    # your impl returns 1/(1+LID) → higher=better
            "confidence_distance_correlation": "max",
            "boundary_proximity_correlation": "max",
            "uncertainty_geometry_alignment": "max",
            "multiscale_separation": "max",
            "reliability_curve_quality": "max",
            "label_cka_hsic": "max",
            "label_cka": "max",
            "logistic_calibratability": "max",
            "prototype_softmax_ece": "max",             # handled as utility inside composite
            "spearman_stability_accuracy": "max",       # higher correlation is better
            # NEW DROP-IN METRICS - prefer MAX
            "decisiveness_tail_mass": "max",
            "confidence_entropy": "max",
            "class_balanced_margin_cvar": "max",
            "impostor_gap_cvar": "max",
            "margin_skewkurt_safety": "max",
            "temperature_estimate_strength": "max",
            # Alpha ablation variants (all prefer MAX like margin_tail_cvar)
            "margin_tail_cvar_005": "max",
            "margin_tail_cvar_010": "max",
            "margin_tail_cvar_015": "max",
            "margin_tail_cvar_020": "max",
            "margin_tail_cvar_025": "max",
            "margin_tail_cvar_030": "max",

            # prefer MIN (we'll invert them)
            "confident_error_rate_tau": "min",
            "aurc_proxy": "min",
        }

        self.original_metrics = metrics or self._get_default_metrics()
        self.score_weights = score_weights or self._get_default_weights()

        self.metrics = (self._filter_metrics_by_weights()
                        if self.skip_zero_weight_metrics else self.original_metrics)

        # Verify all non-zero weighted metrics have direction overrides
        for name in self.score_weights:
            assert name in self.direction_overrides or self.score_weights[name] == 0.0, \
                f"Missing direction override for metric: {name}"

        # Cached model adapter and last computed probabilities (for auto-forward)
        self.model_adapter = None
        self._cached_probs = None
        self._cached_preds = None
        self._cached_confs = None

        # Log compression strategy
        if self.compression_ratio is not None:
            logger.info(f"AdHocLayerSelector: Using ratio-based compression: {self.compression_ratio}x")
        else:
            logger.info(f"AdHocLayerSelector: Using fixed feature dimension: {self.fixed_feature_dim}")

    def _filter_metrics_by_weights(self) -> List[MetricCalculator]:
        active: List[MetricCalculator] = []
        for m in self.original_metrics:
            if self.score_weights.get(m.name, 0.0) != 0.0:
                active.append(m)
        return active

    def _get_default_metrics(self) -> List[MetricCalculator]:
        return [
            # Core metrics (Stage-A: cheap and effective)
            OptimizedCalibrationDecisivenessCalculator(),
            ConfidenceDistanceCorrelationCalculator(),
            BoundaryProximityCorrelationCalculator(),
            UncertaintyGeometryAlignmentCalculator(),
            ClassSeparationQualityCalculator(),
            ClassSeparationUniformityCalculator(),
            LocalIntrinsicDimensionalityCalculator(),
            ReliabilityCurveQualityCalculator(),
            PrototypeSoftmaxECECalculator(),
            LabelCKACalculator(),
            MarginTailCVaRCalculator(),
            # Alpha ablation variants (weight=0 by default, for experiments only)
            MarginTailCVaR_005(),
            MarginTailCVaR_010(),
            MarginTailCVaR_015(),
            MarginTailCVaR_020(),
            MarginTailCVaR_025(),
            MarginTailCVaR_030(),
            MultiScaleClassHomogeneityCalculator(),
            LogisticCalibratabilityCalculator(),

            # ✅ NEW:
            KFoldECEUtilityCalculator(n_splits=4, n_bins=20),

            # NEW DROP-IN METRICS
            DecisivenessTailMassCalculator(),
            ConfidenceEntropyCalculator(),
            ConfidentErrorRateCalculator(),
            AURCProxyCalculator(),
            ClassBalancedTailCVaRCalculator(),
            ImpostorGapCVaRCalculator(),
            MarginSkewKurtosisCalculator(),
            TemperatureEstimateStrengthCalculator(),

            # Expensive (Stage-B): include but weight-masked in Stage-A
            OptimizedGeometricECECalculator(),
            OptimizedStabilityAccuracyCorr(n_bins=0),
            # New: GEC and Label-CKA/HSIC
            GeometryErrorConcordanceCalculator(),
            LabelCKAHSICCalculator(mode="linear_cka", subsample=20000),
        ]

    def _get_default_weights(self) -> Dict[str, float]:
        # Enhanced weights (higher is better), negatives for "lower is better"
        w = {
            # strong ECE proxies / utilities
            "prototype_softmax_ece": 2.2,     # NOTE: this is -ECE; handled in composite scoring
            "reliability_curve_quality": 1.6,
            "kfold_ece_utility": 2.0,         # ✅ NEW: already in [0,1], higher=better
            # uncertainty↔geometry
            "uncertainty_geometry_alignment": 2.0,
            "confidence_distance_correlation": 2.5,
            "boundary_proximity_correlation": 1.5,
            # calibratability predictors
            "label_cka": 0.8,
            "margin_tail_cvar": 0.8,
            "avg_class_separation_ratio": 1.0,
            "class_separation_uniformity": 0.5,
            # manifold
            "local_intrinsic_dimensionality": 0.3,
            # cheap extras
            "multiscale_separation": 0.7,
            # new
            "geometry_error_concordance": 2.5,
            "label_cka_hsic": 1.2,
            # NEW DROP-IN METRICS (turned off by default - can enable if helpful)
            "decisiveness_tail_mass": 0.0,
            "confidence_entropy": 0.0,
            "confident_error_rate_tau": 0.0,
            "aurc_proxy": 0.0,
            "class_balanced_margin_cvar": 0.0,
            "impostor_gap_cvar": 0.7,
            "margin_skewkurt_safety": 0.0,
            "temperature_estimate_strength": 0.0,
            # Alpha ablation variants (turned off by default - for experiments only)
            "margin_tail_cvar_005": 0.0,
            "margin_tail_cvar_010": 0.0,
            "margin_tail_cvar_015": 0.0,
            "margin_tail_cvar_020": 0.0,
            "margin_tail_cvar_025": 0.0,
            "margin_tail_cvar_030": 0.0,
            # stage-B optional
            "logistic_calibratability": 0.8,
            "calibration_decisiveness": 0.8,
            "spearman_stability_accuracy": 1.2,

            # IMPORTANT: this is NEGATIVE ECE; handled in composite scoring
            "ece_score": 3.0,
        }
        return w

    def compute_layer_metrics(self, features: torch.Tensor, labels: torch.Tensor,
                              corrupted_features: Optional[torch.Tensor] = None,
                              layer_idx: int = 0,
                              predictions: Optional[np.ndarray] = None,
                              confidences: Optional[np.ndarray] = None,
                              **extra) -> LayerMetrics:   # ← add **extra
        # Short-circuit on empty input to avoid neighbor-based crashes
        n = (features.shape[0] if torch.is_tensor(features) else len(features))
        if n == 0:
            logger.warning("compute_layer_metrics: received 0 samples; skipping metrics")
            m = LayerMetrics(layer_idx=layer_idx)
            m.composite_score = 0.0
            return m

        metrics = LayerMetrics(layer_idx=layer_idx)
        raw_values: Dict[str, float] = {}
        for calc in self.metrics:
            start = time.time() if self.enable_timing else None
            try:
                # build kwargs this calc can actually accept
                call_kwargs = dict(
                    corrupted_features=corrupted_features,
                    predictions=predictions,
                    confidences=confidences,
                )
                if extra:
                    sig = inspect.signature(calc.compute)
                    accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
                    for k, v in extra.items():
                        if accepts_var_kw or k in sig.parameters:
                            call_kwargs[k] = v

                # Fallback to cached arrays if not explicitly provided (signature-aware)
                sig = inspect.signature(calc.compute)
                params = set(sig.parameters.keys())
                accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

                def _maybe_put(name, value):
                    if value is None:
                        return
                    if accepts_var_kw or (name in params):
                        call_kwargs[name] = value

                # Fallbacks (only if accepted)
                if "model_probs" not in call_kwargs and self._cached_probs is not None:
                    if len(self._cached_probs) == len(labels):
                        _maybe_put("model_probs", self._cached_probs)
                if "predictions" not in call_kwargs and self._cached_preds is not None:
                    _maybe_put("predictions", self._cached_preds)
                if "confidences" not in call_kwargs and self._cached_confs is not None:
                    _maybe_put("confidences", self._cached_confs)

                # Optional alias for metrics that expect `probs` instead of `model_probs`
                if "probs" in params and "probs" not in call_kwargs and "model_probs" in call_kwargs:
                    call_kwargs["probs"] = call_kwargs["model_probs"]

                value = calc.compute(features, labels, **call_kwargs)
            except Exception as e:
                logger.warning(f"Failed to compute {calc.name}: {e}")
                value = 0.0
            setattr(metrics, calc.name, float(value) if value is not None else 0.0)
            raw_values[calc.name] = float(value) if value is not None else 0.0
            if self.enable_timing:
                metrics.computation_times[calc.name] = time.time() - start
        metrics.composite_score = self._compute_composite_score(metrics)
        return metrics

    def _compute_composite_score(self, layer_metrics: LayerMetrics) -> float:
        """
        Combine metric values with weights into a single [unbounded] composite,
        then divide by the sum of absolute weights for scale invariance.

        IMPORTANT:
          - 'ece_score' and 'prototype_softmax_ece' are NEGATIVE ECE values.
            We convert them to a [0,1] utility: 1 - ECE, so higher is better.
          - 'kfold_ece_utility' is already a [0,1] utility and is used directly.
        """
        score = 0.0
        total = 0.0
        for name, w in self.score_weights.items():
            if w == 0.0:
                continue

            val = getattr(layer_metrics, name, np.nan)
            if not np.isfinite(val):
                continue

            # --- canonicalize into bounded utilities where appropriate ---
            if name in {"ece_score", "prototype_softmax_ece"}:
                # raw val is NEGATIVE ECE; convert to positive ECE in [0,1], then to utility
                if abs(val) < 1e-12:
                    # Treat as missing/neutral if calculator failed and returned ~0
                    continue
                ece = -float(val)                        # make ECE positive
                ece = float(np.clip(ece, 0.0, 1.0))      # safety
                val = 1.0 - ece                          # utility: higher is better
            elif name in {
                "confidence_distance_correlation", "boundary_proximity_correlation",
                "uncertainty_geometry_alignment", "reliability_curve_quality",
                "label_cka", "margin_tail_cvar", "multiscale_separation",
                "logistic_calibratability", "kfold_ece_utility",  # NEW utility stays in [0,1]
                # NEW DROP-IN METRICS
                "decisiveness_tail_mass", "confidence_entropy", "class_balanced_margin_cvar",
                "impostor_gap_cvar", "margin_skewkurt_safety", "temperature_estimate_strength"
            }:
                val = float(np.clip(val, 0.0, 1.0))
            elif name in {"avg_class_separation_ratio"}:
                val = float(np.clip(val, 0.0, 10.0))
            # other metrics use their native scale

            # --- align to empirical direction overrides ---
            want = getattr(self, 'direction_overrides', {}).get(name)
            if want == "min":
                if name in {
                    "confidence_distance_correlation", "boundary_proximity_correlation",
                    "uncertainty_geometry_alignment", "reliability_curve_quality",
                    "label_cka", "margin_tail_cvar", "multiscale_separation",
                    "logistic_calibratability", "kfold_ece_utility",
                    # NEW DROP-IN METRICS
                    "decisiveness_tail_mass", "confidence_entropy", "class_balanced_margin_cvar",
                    "impostor_gap_cvar", "margin_skewkurt_safety", "temperature_estimate_strength"
                }:
                    val = 1.0 - val
                else:
                    val = -val

            score += w * val
            total += abs(w)

        return score / total if total > 0 else 0.0

    def _set_shared_data_for_metrics(self, shared_data: 'SharedCalibrationData', val_labels: np.ndarray):
        """Set shared calibration data for optimized metrics that can use it."""
        for calc in self.original_metrics:
            if isinstance(calc, OptimizedCalibrationDecisivenessCalculator):
                calc.set_shared_data(shared_data)
            elif isinstance(calc, OptimizedGeometricECECalculator):
                calc.set_shared_data(shared_data, val_labels)
            elif isinstance(calc, OptimizedStabilityAccuracyCorr):
                calc.set_shared_data(shared_data, val_labels)
            elif isinstance(calc, KFoldECEUtilityCalculator):
                calc.set_shared_data(shared_data, val_labels)
            elif isinstance(calc, (DecisivenessTailMassCalculator, 
                                   ConfidentErrorRateCalculator,
                                   AURCProxyCalculator,
                                   ConfidenceEntropyCalculator)):
                # these can leverage calibrated_confidences if we pass the shared data
                calc.set_shared_data(shared_data)

    def select_best_layer(self, model: nn.Module, dataloader,
                          candidate_layers: Optional[List[int]] = None,
                          corrupted_dataloader=None,
                          device: torch.device = torch.device('cpu'),
                          model_adapter=None) -> Tuple[int, List[LayerMetrics]]:
        if candidate_layers is None:
            candidate_layers = self._get_candidate_layers(model)
        all_metrics: List[LayerMetrics] = []

        # Build original_data and original_labels once upfront
        original_data = []
        original_labels = []
        for batch in dataloader:
            if len(batch) == 2:
                data, labels = batch
                original_data.append(data.cpu().numpy())
                original_labels.append(labels.cpu().numpy())
        original_data = np.concatenate(original_data, axis=0) if original_data else None
        original_labels = np.concatenate(original_labels, axis=0) if original_labels else None

        predictions = None
        confidences = None
        model_probs = None      # ← keep the full matrix
        if model_adapter is not None and original_data is not None:
            try:
                probs = model_adapter.predict_proba(original_data)
                predictions = np.argmax(probs, axis=1)
                confidences = np.max(probs, axis=1)
                model_probs = probs      # ← keep the full matrix
                # cache for follow-up calls
                self.model_adapter = model_adapter
                self._cached_probs = model_probs
                self._cached_preds = predictions
                self._cached_confs = confidences
            except Exception as e:
                logger.warning(f"Failed to extract predictions/confidences: {e}")

        for layer_idx in candidate_layers:
            feats, labs = self._extract_features(model, dataloader, layer_idx, device)
            corrupted_feats = None
            if corrupted_dataloader is not None:
                corrupted_feats, _ = self._extract_features(model, corrupted_dataloader, layer_idx, device)

            # --- NEW: compute and inject shared calibration data for optimized metrics ---
            try:
                # Build shared data if we have images OR precomputed probs
                if (model_adapter is not None
                    and isinstance(feats, torch.Tensor) and isinstance(labs, torch.Tensor)
                    and (original_data is not None or model_probs is not None)):
                    n = feats.shape[0]
                    if n >= 3:
                        # Deterministic 25%/75% split for better evaluation reliability
                        n_train = max(1, int(0.25 * n))
                        train_idx = np.arange(n_train)
                        eval_idx = np.arange(n_train, n)

                        train_features = feats[train_idx].cpu().numpy()
                        train_labels = labs[train_idx].cpu().numpy()
                        val_features = feats[eval_idx].cpu().numpy()
                        val_labels = labs[eval_idx].cpu().numpy()

                        val_original = original_data[eval_idx] if original_data is not None else None
                        val_model_probs = model_probs[eval_idx] if model_probs is not None else None

                        # Compute shared calibration data and inject
                        shared_calc = SharedCalibrationCalculator(
                            model_adapter=model_adapter,
                            device=str(device)
                        )
                        shared = shared_calc.compute_shared_calibration_data(
                            train_features=train_features,
                            train_labels=train_labels,
                            val_features=val_features,
                            val_labels=val_labels,
                            val_original=val_original,
                            layer_idx=layer_idx,
                            val_model_probs=val_model_probs
                        )
                        self._set_shared_data_for_metrics(shared, val_labels)
            except Exception as e:
                logger.warning(f"Failed to prepare shared calibration data for layer {layer_idx}: {e}")

            m = self.compute_layer_metrics(feats, labs, corrupted_feats, layer_idx,
                                           predictions, confidences,
                                           model_probs=model_probs)  # ← pass it along
            all_metrics.append(m)
        # Optional normalization across layers with percentile clamp
        if self.normalize_metrics and len(all_metrics) > 1:
            all_metrics = self.normalize_metrics_across_layers(all_metrics)
        best_idx = int(np.argmax([m.composite_score for m in all_metrics]))
        return candidate_layers[best_idx], all_metrics

    def _extract_features(self, model: nn.Module, dataloader, layer_idx: int,
                          device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        activation = {}
        def hook(module, input, output):
            activation['output'] = output
        layers = list(model.modules())
        total_modules = len(layers)
        print(f"\n=== Debugging Layer {layer_idx} ===")
        print(f"Model type detected: {type(model)}")
        print(f"Model name: {model.__class__.__name__}")
        print(f"Total modules in model: {total_modules}")
        if 0 <= layer_idx < total_modules:
            target_layer = layers[layer_idx]
            print(f"Layer {layer_idx} type: {type(target_layer)}")
            print(f"Layer {layer_idx} name: {target_layer.__class__.__name__}")
        else:
            print(f"Layer {layer_idx} is out of bounds for total modules ({total_modules}).")
        print("\nFirst 10 layers:")
        for i, layer in enumerate(layers[:10]):
            print(f"  {i}: {layer.__class__.__name__}")
        print("\nLast 10 layers:")
        for i, layer in enumerate(layers[-10:], start=max(0, total_modules - 10)):
            print(f"  {i}: {layer.__class__.__name__}")
        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of bounds")
        layer_module = layers[layer_idx]
        logger.debug(f"Hooked layer {layer_idx}: {layer_module.__class__.__name__}")
        handle = layer_module.register_forward_hook(hook)
        model.eval()
        # one-time sanity toggle (HF backbones)
        try:
            if hasattr(model, "config"):
                model.config.output_attentions = False
                model.config.output_hidden_states = False
                model.config.return_dict = True
        except Exception:
            pass
        features_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        feature_stats_logged = False
        try:
            with torch.no_grad():
                for batch in dataloader:
                    if len(batch) == 2:
                        data, y = batch
                    else:
                        logger.warning("Batch missing labels; many metrics will be unreliable.")
                        data, y = batch[0], torch.zeros(len(batch[0]), dtype=torch.long)
                    data = data.to(device)
                    activation.clear()
                    _ = model(data)
                    raw_out = activation.get('output', None)
                    feat = coerce_to_tensor(raw_out)
                    if feat is None or not torch.is_tensor(feat):
                        logger.warning(f"[L{layer_idx} {layer_module.__class__.__name__}] output type {type(raw_out)}; skipping batch")
                        continue
                    if not feature_stats_logged:
                        try:
                            feat_np = feat.detach().cpu().float().numpy()
                            print(f"Layer {layer_idx} feature stats:")
                            print(f"  Shape: {feat_np.shape}")
                            print(f"  Mean: {feat_np.mean():.6f}")
                            print(f"  Std: {feat_np.std():.6f}")
                            print(f"  Min: {feat_np.min():.6f}")
                            print(f"  Max: {feat_np.max():.6f}")
                            first_sample = feat_np[0].reshape(-1) if feat_np.ndim > 1 else np.array([feat_np[0]])
                            print(f"  First sample L2 norm: {float(np.linalg.norm(first_sample)):.6f}")
                        except Exception as e:
                            logger.warning(f"Failed to log feature stats for layer {layer_idx}: {e}")
                        finally:
                            feature_stats_logged = True
                    if not torch.is_tensor(feat):
                        logger.warning(f"Layer {layer_idx}: coerce failed for type {type(raw_out)}")
                        continue
                    # Handle ViT token sequences [B, T, C]
                    if feat.ndim == 3:
                        B, T, C = feat.shape
                        TT = T - 1
                        has_square_grid = (TT > 0 and int(math.isqrt(TT)) ** 2 == TT)
                        tokens = feat[:, 1:, :] if has_square_grid else feat
                        Tuse = TT if has_square_grid else T
                        s = int(math.isqrt(Tuse))
                        if s * s == Tuse:
                            fmap = tokens.transpose(1, 2).reshape(B, C, s, s)
                            num_channels = fmap.size(1)
                            if num_channels not in self.spp_projectors:
                                if self.compression_ratio is not None:
                                    self.spp_projectors[num_channels] = FixedSizeSPP_JL(
                                        in_channels=num_channels,
                                        final_output_dim=None,
                                        compression_ratio=self.compression_ratio
                                    ).to(device)
                                else:
                                    self.spp_projectors[num_channels] = FixedSizeSPP_JL(
                                        in_channels=num_channels,
                                        final_output_dim=self.fixed_feature_dim
                                    ).to(device)
                            feat = self.spp_projectors[num_channels](fmap)
                        else:
                            feat = tokens.mean(dim=1)
                    elif feat.ndim > 3:
                        num_channels = feat.size(1)
                        if num_channels not in self.spp_projectors:
                            if self.compression_ratio is not None:
                                self.spp_projectors[num_channels] = FixedSizeSPP_JL(
                                    in_channels=num_channels,
                                    final_output_dim=None,
                                    compression_ratio=self.compression_ratio
                                ).to(device)
                            else:
                                self.spp_projectors[num_channels] = FixedSizeSPP_JL(
                                    in_channels=num_channels,
                                    final_output_dim=self.fixed_feature_dim
                                ).to(device)
                        feat = self.spp_projectors[num_channels](feat)
                    features_list.append(feat.cpu())
                    labels_list.append(y.cpu())
        finally:
            handle.remove()
        if not features_list:
            raise ValueError("No features were extracted")
        return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)

    def normalize_metrics_across_layers(self, all_metrics: List[LayerMetrics]) -> List[LayerMetrics]:
        # Percentile clamp (5–95) then min-max per metric
        # Protect raw negative-ECE metrics and bounded utilities from normalization
        protected = {"ece_score", "prototype_softmax_ece"}
        bounded_utilities = {
            "kfold_ece_utility", "reliability_curve_quality", "confidence_distance_correlation",
            "boundary_proximity_correlation", "uncertainty_geometry_alignment", "label_cka",
            "margin_tail_cvar", "multiscale_separation", "logistic_calibratability",
            "decisiveness_tail_mass", "confidence_entropy", "class_balanced_margin_cvar",
            "impostor_gap_cvar", "margin_skewkurt_safety", "temperature_estimate_strength",
            "geometry_error_concordance"  # Already clipped to [0,1] in compute
        }
        metric_names = list(self.score_weights.keys())
        values: Dict[str, List[float]] = {k: [] for k in metric_names}
        for m in all_metrics:
            for k in metric_names:
                if hasattr(m, k):
                    v = getattr(m, k)
                    if not np.isnan(v) and not np.isinf(v) and k not in protected and k not in bounded_utilities:
                        values[k].append(v)
        stats = {}
        for k, arr in values.items():
            if len(arr) >= 2:
                lo, hi = np.percentile(arr, [5, 95])
                stats[k] = (float(lo), float(hi))
        normalized: List[LayerMetrics] = []
        for m in all_metrics:
            nm = LayerMetrics(layer_idx=m.layer_idx, computation_times=m.computation_times.copy())
            for k in metric_names:
                v = getattr(m, k, np.nan)
                if k in stats and k not in protected and k not in bounded_utilities:
                    lo, hi = stats[k]
                    if hi > lo:
                        v = (np.clip(v, lo, hi) - lo) / (hi - lo)
                setattr(nm, k, float(v))
            nm.composite_score = self._compute_composite_score(nm)
            normalized.append(nm)
        return normalized

    def _get_candidate_layers(self, model: nn.Module) -> List[int]:
        model_name = model.__class__.__name__
        print(f"[AdHocLayerSelector] Model type detected: {type(model)}")
        print(f"[AdHocLayerSelector] Model name: {model_name}")
        if self._is_wide_resnet(model):
            candidates = self._get_candidate_layers_wide_resnet(model)
            print(f"[AdHocLayerSelector] WideResNet candidate layers: {candidates}")
            return candidates
        vit = CrossModalLayerMapper.get_vit_block_indices(model)
        if vit:
            print(f"[AdHocLayerSelector] ViT candidate layers: {vit}")
            return vit
        conv = CrossModalLayerMapper.get_conv_block_indices(model)
        if conv:
            print(f"[AdHocLayerSelector] Conv candidate layers: {conv}")
            return conv
        trans = CrossModalLayerMapper.get_transformer_block_indices(model)
        if trans:
            print(f"[AdHocLayerSelector] Transformer candidate layers: {trans}")
            return trans
        fallback = [0]
        print("[AdHocLayerSelector] Fallback candidate layer [0]")
        return fallback

    def _is_wide_resnet(self, model: nn.Module) -> bool:
        name = model.__class__.__name__.lower()
        module_path = model.__class__.__module__.lower()
        return ("wide" in name and "resnet" in name) or \
               ("wrn" in name) or \
               ("wide_resnet" in module_path)

    def _spread_indices(self, idxs: List[int], targets: int = 3) -> List[int]:
        if not idxs:
            return []
        if len(idxs) <= targets:
            return idxs
        positions = {0, len(idxs) - 1, len(idxs) // 2}
        while len(positions) < targets:
            step = max(1, len(idxs) // (targets + 1))
            positions.add(min(len(idxs) - 1, step * len(positions)))
        return sorted({idxs[pos] for pos in positions})

    def _get_candidate_layers_wide_resnet(self, model: nn.Module) -> List[int]:
        layers = list(model.modules())
        module_to_index = {module: idx for idx, module in enumerate(layers)}

        def safe_idx(module: Optional[nn.Module]) -> Optional[int]:
            if module is None:
                return None
            return module_to_index.get(module)

        candidates: List[int] = []

        for attr in ("conv1", "bn1", "relu"):
            idx = safe_idx(getattr(model, attr, None))
            if idx is not None:
                candidates.append(idx)

        def gather_block_indices(seq: Optional[nn.Sequential]) -> List[int]:
            block_indices: List[int] = []
            if seq is None:
                return block_indices
            for name, module in seq.named_modules():
                if name == "":
                    continue
                cls_name = module.__class__.__name__.lower()
                if "basicblock" in cls_name or "bottleneck" in cls_name:
                    idx = module_to_index.get(module)
                    if idx is not None:
                        block_indices.append(idx)
            return sorted(set(block_indices))

        for block_name in ("layer1", "layer2", "layer3"):
            seq = getattr(model, block_name, None)
            block_idxs = gather_block_indices(seq)
            spread = self._spread_indices(block_idxs, targets=3)
            candidates.extend(spread)

        tail_modules = [getattr(model, "avgpool", None), getattr(model, "fc", None)]
        for module in tail_modules:
            idx = safe_idx(module)
            if idx is not None:
                candidates.append(idx)

        seen = set()
        deduped: List[int] = []
        for idx in candidates:
            if idx not in seen:
                deduped.append(idx)
                seen.add(idx)
        return deduped or [0]

    def _coerce_to_tensor(self, out):
        # Deprecated: keep for backward compatibility; use utils.tensor_utils.coerce_to_tensor
        return coerce_to_tensor(out)


class TwoStageLayerSelector:
    """
    Two-stage layer selection:
      Stage-A: cheap metrics on all candidates → shortlist K
      Stage-B: expensive metrics (ECE, logistic_calibratability) on shortlist → pick best
    """
    def __init__(self, base_selector: AdHocLayerSelector,
                 shortlist_k: int = 6,
                 cheap_metric_names: Optional[List[str]] = None):
        self.sel = base_selector
        self.k = shortlist_k
        self.cheap_metric_names = cheap_metric_names or [
            "calibration_decisiveness",
            "prototype_softmax_ece",
            "reliability_curve_quality",
            "confidence_distance_correlation",
            "boundary_proximity_correlation",
            "uncertainty_geometry_alignment",
            "label_cka",
            "label_cka_hsic",
            "margin_tail_cvar",
            "avg_class_separation_ratio",
            "multiscale_separation",
            "local_intrinsic_dimensionality",
            "spearman_stability_accuracy",
            "kfold_ece_utility",
        ]

    def select(self, model: nn.Module, dataloader, candidate_layers: List[int],
               device: torch.device,
               stageB_hook: callable,
               model_adapter=None) -> Tuple[int, List[LayerMetrics]]:
        full_weights = self.sel.score_weights.copy()
        masked = {k: (v if k in self.cheap_metric_names else 0.0)
                  for k, v in full_weights.items()}
        cheap_sel = AdHocLayerSelector(metrics=self.sel.original_metrics,
                                       score_weights=masked,
                                       skip_zero_weight_metrics=True)
        # Ensure model adapter is available for Stage-A so shared data metrics compute
        if model_adapter is not None:
            self.sel.model_adapter = model_adapter
        _, cheap_metrics = cheap_sel.select_best_layer(
            model, dataloader,
            candidate_layers=candidate_layers,
            device=device,
            model_adapter=self.sel.model_adapter
        )
        scores = [(m.layer_idx, m.composite_score) for m in cheap_metrics]
        scores.sort(key=lambda x: x[1], reverse=True)
        shortlist = [idx for idx, _ in scores[:self.k]]

        # restore weights
        self.sel.score_weights = full_weights
        stageB_metrics: List[LayerMetrics] = []

        # Compute original_data and model probabilities once for Stage-B
        predictions = None
        confidences = None
        model_probs = None
        try:
            original_data = []
            for batch in dataloader:
                if len(batch) >= 1:
                    x = batch[0]
                    original_data.append(x.cpu().numpy())
            if original_data:
                original_data = np.concatenate(original_data, axis=0)
                probs = None
                if hasattr(self.sel, 'model_adapter') and self.sel.model_adapter is not None:
                    probs = self.sel.model_adapter.predict_proba(original_data)
                if probs is not None:
                    predictions = np.argmax(probs, axis=1)
                    confidences = np.max(probs, axis=1)
                    model_probs = probs
        except Exception as e:
            logger.warning(f"Stage-B: failed to prepare model probabilities: {e}")
        for layer_idx in shortlist:
            stageB_hook(layer_idx)
            feats, labs = self.sel._extract_features(model, dataloader, layer_idx, device)

            # --- Prepare and inject shared calibration data for Stage-B, mirroring select_best_layer ---
            try:
                if (hasattr(self.sel, 'model_adapter') and self.sel.model_adapter is not None
                        and isinstance(feats, torch.Tensor) and isinstance(labs, torch.Tensor)
                        and (original_data is not None or model_probs is not None)):
                    n = feats.shape[0]
                    if n >= 3:
                        n_train = max(1, int(0.25 * n))
                        train_idx = np.arange(n_train)
                        eval_idx = np.arange(n_train, n)

                        train_features = feats[train_idx].cpu().numpy()
                        train_labels = labs[train_idx].cpu().numpy()
                        val_features = feats[eval_idx].cpu().numpy()
                        val_labels = labs[eval_idx].cpu().numpy()
                        val_original = original_data[eval_idx] if original_data is not None else None
                        val_model_probs = model_probs[eval_idx] if model_probs is not None else None

                        shared_calc = SharedCalibrationCalculator(
                            model_adapter=self.sel.model_adapter,
                            device=str(device)
                        )
                        shared = shared_calc.compute_shared_calibration_data(
                            train_features=train_features,
                            train_labels=train_labels,
                            val_features=val_features,
                            val_labels=val_labels,
                            val_original=val_original,
                            layer_idx=layer_idx,
                            val_model_probs=val_model_probs
                        )
                        self.sel._set_shared_data_for_metrics(shared, val_labels)
            except Exception as e:
                logger.warning(f"Stage-B: failed to prepare shared calibration data for layer {layer_idx}: {e}")

            m = self.sel.compute_layer_metrics(
                feats, labs, layer_idx=layer_idx,
                predictions=predictions, confidences=confidences, model_probs=model_probs
            )
            stageB_metrics.append(m)
        best = max(stageB_metrics, key=lambda m: m.composite_score).layer_idx
        return best, stageB_metrics






# ============================================================================
# OPTIMIZED SHARED CALIBRATION CALCULATOR (for metric reuse)
# ============================================================================

@dataclass
class SharedCalibrationData:
    """Container for shared calibration calculations to avoid redundancy"""
    stability_space: 'StabilitySpace'
    stability_scores: np.ndarray
    calibrated_predictions: np.ndarray
    calibrated_confidences: np.ndarray
    isotonic_regressor: IsotonicRegression
    raw_predictions: np.ndarray
    raw_confidences: np.ndarray
    validation_accuracy: float
    # NEW: bundle required to rerun the exact GeometricCalibrator pipeline
    model_adapter: Any
    train_features: np.ndarray
    train_labels: np.ndarray
    val_features: np.ndarray
    val_labels: np.ndarray
    val_original: Optional[np.ndarray]
    layer_idx: int
    compression_mode: Optional[Any]
    compression_param: Optional[Any]
    library: str
    metric: str
    # Optional: when we don't have raw images, we can still work from precomputed probs
    val_model_probs: Optional[np.ndarray] = None


class SharedCalibrationCalculator:
    """Updated with better logging for the improved split strategy"""
    def __init__(self, model_adapter, device: str = 'cuda', 
                 compression_mode=None, compression_param=None,
                 library='fast_separation', metric='l2'):
        self.model_adapter = model_adapter
        self.device = device
        self.compression_mode = compression_mode
        self.compression_param = compression_param
        self.library = library
        self.metric = metric
    def compute_shared_calibration_data(
        self, 
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,      # This is now the LARGER evaluation set (75%)
        val_labels: np.ndarray,
        val_original: Optional[np.ndarray],
        layer_idx: int
        ,
        val_model_probs: Optional[np.ndarray] = None
    ) -> SharedCalibrationData:
        """
        Compute calibration data with improved logging for 25-75 split.
        """
        logger.debug(f"   🔧 Computing shared calibration data for layer {layer_idx}...")
        logger.debug(f"      Training features: {train_features.shape} (25% for stability space)")
        logger.debug(f"      Validation features: {val_features.shape} (75% for evaluation - BETTER ECE!)")
        logger.debug(
            f"      Original images: "
            f"{None if val_original is None else val_original.shape} (for model predictions)"
        )
        compression = SmartCompression(self.compression_mode, self.compression_param) if self.compression_mode else None
        stability_space = StabilitySpace(
            train_features, 
            train_labels,
            compression=compression,
            library=self.library,
            metric=self.metric
        )
        # Prefer precomputed probabilities when provided
        if val_model_probs is not None:
            val_predictions = val_model_probs
        else:
            if val_original is None:
                raise ValueError("SharedCalibrationCalculator: need val_original or val_model_probs")
            val_predictions = self.model_adapter.predict_proba(val_original)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        raw_confidences = np.max(val_predictions, axis=1)
        stability_scores = stability_space.calc_stab(val_features, val_predictions)
        stability_p5 = np.percentile(stability_scores, 5)
        stability_p95 = np.percentile(stability_scores, 95)
        stability_range = stability_p95 - stability_p5
        if stability_range > 0:
            normalized_stability = np.clip(
                (stability_scores - stability_p5) / stability_range, 0, 1
            )
        else:
            normalized_stability = np.ones_like(stability_scores) * 0.5
        val_accuracies = (val_pred_classes == val_labels).astype(float)
        n_bins = 20
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(normalized_stability, bins=bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Debug: Check bin distribution BEFORE binning loop
        logger.debug(f"Layer {layer_idx} - Binning Debug:")
        logger.debug(f"   Total samples to bin: {len(normalized_stability)}")
        logger.debug(f"   Correct predictions: {np.sum(val_pred_classes == val_labels)} ({np.mean(val_pred_classes == val_labels)*100:.2f}%)")
        logger.debug(f"   Incorrect predictions: {np.sum(val_pred_classes != val_labels)} ({np.mean(val_pred_classes != val_labels)*100:.2f}%)")

        binned_stability = []
        binned_accuracy = []
        for bin_idx in range(n_bins):
            indices_in_bin = np.where(bin_indices == bin_idx)[0]
            if len(indices_in_bin) > 0:
                y_true_bin = val_labels[indices_in_bin]
                y_pred_bin = val_pred_classes[indices_in_bin]
                accuracy = np.mean(y_true_bin == y_pred_bin)

                # Debug first 5 bins
                if bin_idx < 5:
                    correct_in_bin = np.sum(y_true_bin == y_pred_bin)
                    incorrect_in_bin = np.sum(y_true_bin != y_pred_bin)
                    logger.debug(f"   Bin {bin_idx}: {len(indices_in_bin)} samples, {correct_in_bin} correct, {incorrect_in_bin} incorrect, acc={accuracy:.4f}")

                binned_stability.append((bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2)
                binned_accuracy.append(accuracy)
        if len(binned_stability) < 2:
            logger.warning(f"Not enough bins for isotonic regression in layer {layer_idx}")
            return SharedCalibrationData(
                stability_space=stability_space,
                stability_scores=stability_scores,
                calibrated_predictions=val_predictions,  # Use raw predictions
                calibrated_confidences=raw_confidences,
                isotonic_regressor=None,
                raw_predictions=val_predictions,
                raw_confidences=raw_confidences,
                validation_accuracy=np.mean(val_accuracies),
                # exact pipeline bundle
                model_adapter=self.model_adapter,
                train_features=train_features,
                train_labels=train_labels,
                val_features=val_features,
                val_labels=val_labels,
                val_original=val_original,
                layer_idx=layer_idx,
                compression_mode=self.compression_mode,
                compression_param=self.compression_param,
                library=self.library,
                metric=self.metric
            )
        stability_vals = np.array(binned_stability)
        accuracies = np.array(binned_accuracy)

        # ===================================================================
        # 🔍 BINNING VERIFICATION: Check if all bins have 100% accuracy (BUG)
        # ===================================================================
        logger.info("="*60)
        logger.info("🔍 BINNING VERIFICATION:")
        logger.info(f"   Total validation samples: {len(normalized_stability)}")
        logger.info(f"   Overall validation accuracy: {np.mean(val_pred_classes == val_labels):.4f}")
        logger.info(f"   Number of bins created: {len(binned_stability)}")
        logger.info(f"   Bin accuracies range: [{min(binned_accuracy):.4f}, {max(binned_accuracy):.4f}]")
        logger.info(f"   Bin accuracies (all): {[f'{acc:.4f}' for acc in binned_accuracy]}")

        # Detailed check
        if min(binned_accuracy) >= 0.99:
            logger.error("❌ BUG DETECTED: All bins have ≥99% accuracy!")
            logger.error("❌ This indicates binning is using only correct predictions")
        elif max(binned_accuracy) - min(binned_accuracy) < 0.05:
            logger.warning("⚠️ Low variance in bin accuracies - may still have issue")
        else:
            logger.info("✅ Bin accuracies have good diversity - binning appears correct")
        logger.info("="*60)

        isotonic_regressor = IsotonicRegression(out_of_bounds="clip")

        # ===================================================================
        # 🔍 DIAGNOSTIC LOGGING: Check what's being fed to isotonic regressor
        # ===================================================================
        logger.info("="*60)
        logger.info("🔍 ISOTONIC REGRESSOR FITTING DIAGNOSTICS")
        logger.info("="*60)
        logger.info(f"Layer {layer_idx} - Isotonic Regressor Input Analysis:")
        logger.info(f"Input X (stability_vals) shape: {stability_vals.shape}")
        logger.info(f"Input X type: {type(stability_vals)}")
        logger.info(f"First 10 X values: {stability_vals[:10] if len(stability_vals) >= 10 else stability_vals}")
        logger.info(f"X statistics:")
        logger.info(f"   Range: [{stability_vals.min():.6f}, {stability_vals.max():.6f}]")
        logger.info(f"   Mean: {stability_vals.mean():.6f}")
        logger.info(f"   Std: {stability_vals.std():.6f}")
        logger.info(f"   Variance: {stability_vals.var():.6f}")
        logger.info(f"   Unique values: {len(np.unique(stability_vals))}")
        logger.info(f"")
        logger.info(f"Output y (accuracies) distribution:")
        logger.info(f"   Shape: {accuracies.shape}")
        logger.info(f"   Range: [{accuracies.min():.6f}, {accuracies.max():.6f}]")
        logger.info(f"   Mean: {accuracies.mean():.6f}")
        logger.info(f"   First 10 y values: {accuracies[:10] if len(accuracies) >= 10 else accuracies}")
        logger.info(f"")

        # Check if X looks like stability scores or confidence scores
        logger.info(f"📊 Raw stability scores (before binning) analysis:")
        logger.info(f"   Shape: {stability_scores.shape}")
        logger.info(f"   Range: [{stability_scores.min():.6f}, {stability_scores.max():.6f}]")
        logger.info(f"   Mean: {stability_scores.mean():.6f}")
        logger.info(f"   Std: {stability_scores.std():.6f}")
        logger.info(f"   Unique values: {len(np.unique(stability_scores))}")
        logger.info(f"   First 10 raw stability values: {stability_scores[:10]}")
        logger.info(f"")
        logger.info(f"📊 Normalized stability scores analysis:")
        logger.info(f"   Shape: {normalized_stability.shape}")
        logger.info(f"   Range: [{normalized_stability.min():.6f}, {normalized_stability.max():.6f}]")
        logger.info(f"   Mean: {normalized_stability.mean():.6f}")
        logger.info(f"   Std: {normalized_stability.std():.6f}")
        logger.info(f"   Unique values: {len(np.unique(normalized_stability))}")
        logger.info(f"   First 10 normalized values: {normalized_stability[:10]}")
        logger.info(f"")
        logger.info(f"📊 Raw confidence scores (for comparison):")
        logger.info(f"   Shape: {raw_confidences.shape}")
        logger.info(f"   Range: [{raw_confidences.min():.6f}, {raw_confidences.max():.6f}]")
        logger.info(f"   Mean: {raw_confidences.mean():.6f}")
        logger.info(f"   Std: {raw_confidences.std():.6f}")
        logger.info(f"   Unique values: {len(np.unique(raw_confidences))}")
        logger.info(f"   First 10 confidence values: {raw_confidences[:10]}")
        logger.info(f"")

        # Diagnostic checks
        if stability_vals.min() >= 0.0 and stability_vals.max() <= 1.0 and stability_vals.var() < 0.01:
            logger.warning("⚠️ WARNING: X (stability_vals) looks like confidence scores (range [0,1], low variance)")
            logger.warning("⚠️ Expected: Stability-based bin centers (should have some variance)")
        elif len(np.unique(stability_vals)) < 10:
            logger.warning("⚠️ WARNING: X has very few unique values - binning may be too coarse")
        else:
            logger.info("✅ X (stability_vals) appears to be bin centers based on stability scores")

        # Check if normalized_stability correlates with raw_confidences
        from scipy.stats import spearmanr
        corr_stability_confidence, _ = spearmanr(normalized_stability, raw_confidences)
        logger.info(f"")
        logger.info(f"🔍 Correlation Analysis:")
        logger.info(f"   Spearman correlation (normalized_stability vs raw_confidence): {corr_stability_confidence:.6f}")
        if abs(corr_stability_confidence) > 0.95:
            logger.error("❌ CRITICAL: Normalized stability is highly correlated with confidence!")
            logger.error("❌ This suggests stability scores may be derived from confidence instead of geometry!")
        elif abs(corr_stability_confidence) > 0.7:
            logger.warning("⚠️ WARNING: Normalized stability has high correlation with confidence (%.3f)", corr_stability_confidence)
        else:
            logger.info("✅ Normalized stability has low correlation with confidence - this is expected")

        logger.info("="*60)

        isotonic_regressor.fit(stability_vals, accuracies)

        # ===================================================================
        # 🔍 VERIFICATION LOGGING: Test the fitted isotonic regressor
        # ===================================================================
        logger.info("="*60)
        logger.info("🔍 ISOTONIC REGRESSOR VERIFICATION")
        logger.info("="*60)

        # Test the isotonic regressor on sample values
        test_vals = [stability_vals.min(), stability_vals.mean(), stability_vals.max()]
        test_predictions = isotonic_regressor.predict(test_vals)

        logger.info(f"Layer {layer_idx} - Isotonic Regressor Testing:")
        logger.info(f"   Predict(min={stability_vals.min():.6f}) = {test_predictions[0]:.6f}")
        logger.info(f"   Predict(mean={stability_vals.mean():.6f}) = {test_predictions[1]:.6f}")
        logger.info(f"   Predict(max={stability_vals.max():.6f}) = {test_predictions[2]:.6f}")
        logger.info(f"")

        if test_predictions[0] == test_predictions[1] == test_predictions[2]:
            logger.error("❌ ISOTONIC REGRESSOR IS BROKEN: Outputs same value for all inputs!")
            logger.error("❌ This will cause all layers to have identical ECE values!")
        elif abs(test_predictions[2] - test_predictions[0]) < 0.01:
            logger.warning("⚠️ WARNING: Isotonic regressor has very small output range (< 0.01)")
            logger.warning("⚠️ This may cause similar ECE values across layers")
        else:
            logger.info("✅ Isotonic regressor appears to be working correctly")
            logger.info(f"   Output range: {abs(test_predictions[2] - test_predictions[0]):.6f}")

        logger.info("="*60)

        calibrated_values = isotonic_regressor.predict(normalized_stability)

        # Log calibrated values statistics
        logger.info(f"Layer {layer_idx} - Calibrated Values Statistics:")
        logger.info(f"   Shape: {calibrated_values.shape}")
        logger.info(f"   Range: [{calibrated_values.min():.6f}, {calibrated_values.max():.6f}]")
        logger.info(f"   Mean: {calibrated_values.mean():.6f}")
        logger.info(f"   Std: {calibrated_values.std():.6f}")
        logger.info(f"   Unique values: {len(np.unique(calibrated_values))}")
        logger.info(f"   First 10 calibrated values: {calibrated_values[:10]}")

        # Check if all calibrated values are identical
        if len(np.unique(calibrated_values)) == 1:
            logger.error("❌ CRITICAL: All calibrated values are identical!")
            logger.error("❌ This will cause identical ECE across all layers!")
        elif calibrated_values.std() < 0.01:
            logger.warning("⚠️ WARNING: Calibrated values have very low variance (< 0.01)")
            logger.warning("⚠️ This may cause similar ECE values across layers")

        calibrated_values = np.clip(calibrated_values, 0.01, 0.99)
        num_classes = val_predictions.shape[1]
        calibrated_probs = np.zeros_like(val_predictions)
        for i in range(len(val_labels)):
            c = val_pred_classes[i]
            if num_classes > 1:
                rem = (1.0 - calibrated_values[i]) / (num_classes - 1)
                calibrated_probs[i, :] = rem
                calibrated_probs[i, c] = calibrated_values[i]
            else:
                calibrated_probs[i, c] = 1.0
        calibrated_confidences = np.max(calibrated_probs, axis=1)
        validation_accuracy = np.mean(val_accuracies)
        logger.debug(f"   ✅ Shared calibration data computed for layer {layer_idx}")
        logger.debug(f"      📊 Evaluation samples: {len(val_labels)} (larger set = more reliable ECE)")
        logger.debug(f"      🎯 This improves layer selection reliability!")
        return SharedCalibrationData(
            stability_space=stability_space,
            stability_scores=stability_scores,
            calibrated_predictions=calibrated_probs,
            calibrated_confidences=calibrated_confidences,
            isotonic_regressor=isotonic_regressor,
            raw_predictions=val_predictions,
            raw_confidences=raw_confidences,
            validation_accuracy=validation_accuracy,
            # exact pipeline bundle
            model_adapter=self.model_adapter,
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            val_original=val_original,
            layer_idx=layer_idx,
            compression_mode=self.compression_mode,
            compression_param=self.compression_param,
            library=self.library,
            metric=self.metric
            ,
            val_model_probs=val_model_probs
        )


# ============================================================================
# OPTIMIZED METRIC CALCULATORS (using shared data)
# ============================================================================

class OptimizedGeometricECECalculator(MetricCalculator):
    """ECE Calculator that uses pre-computed calibration data"""
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self._shared_data = None
        self._val_labels = None
    @property
    def name(self) -> str:
        return "ece_score"
    def set_shared_data(self, shared_data: SharedCalibrationData, val_labels: np.ndarray):
        self._shared_data = shared_data
        self._val_labels = val_labels
    def compute(self, features: torch.Tensor, labels: torch.Tensor,
                corrupted_features: Optional[torch.Tensor] = None,
                predictions: Optional[np.ndarray] = None,
                confidences: Optional[np.ndarray] = None, **kwargs) -> float:
        if self._shared_data is None or self._val_labels is None:
            logger.warning("No shared calibration data set for ECE calculation")
            return np.nan
        try:
            # Prefer running the exact GeometricCalibrator pipeline if bundle contains required fields
            sd = self._shared_data
            required = all(
                hasattr(sd, k) for k in (
                    'model_adapter', 'train_features', 'train_labels',
                    'val_features', 'val_original', 'compression_mode',
                    'compression_param', 'library', 'metric'
                )
            )
            if required:
                from Calibrators.geometric_calibrator_new import GeometricCalibrator
                gc = GeometricCalibrator(
                    model=sd.model_adapter,
                    X_train_embed=sd.train_features,
                    y_train=sd.train_labels,
                    compression_mode=sd.compression_mode,
                    compression_param=sd.compression_param,
                    metric=sd.metric,
                    library=sd.library,
                    auto_select_layer=False,
                    use_binning=True,
                    n_bins=self.n_bins
                )
                gc.fit(
                    X_val_embed=sd.val_features,
                    y_val=self._val_labels,
                    X_val_original=sd.val_original
                )
                calibrated_probs = gc.calibrate(
                    X_test_embed=sd.val_features,
                    X_test_original=sd.val_original
                )
                ece_value = self._calculate_ece(calibrated_probs, self._val_labels)
                return -float(ece_value)
            # Fallback to using already-computed calibrated predictions (older behavior)
            calibrated_probs = sd.calibrated_predictions
            ece_value = self._calculate_ece(calibrated_probs, self._val_labels)
            logger.debug(f"   📊 ECE from shared data (fallback): {ece_value:.4f}")
            return -float(ece_value)
        except Exception as e:
            logger.warning(f"Optimized ECE calculation failed: {e}")
            return np.nan
    def _calculate_ece(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        confidences = np.max(predictions, axis=1)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracies = (predicted_classes == labels).astype(float)
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        ece = 0.0
        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            # Include exact 0.0 confidence in the first bin
            if i == 0:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class OptimizedCalibrationDecisivenessCalculator(MetricCalculator):
    """Decisiveness Calculator that uses pre-computed calibration data"""
    def __init__(self):
        self._shared_data = None
    @property
    def name(self) -> str:
        return "calibration_decisiveness"
    def set_shared_data(self, shared_data: SharedCalibrationData):
        self._shared_data = shared_data
    def compute(self, features: torch.Tensor, labels: torch.Tensor, 
                corrupted_features: Optional[torch.Tensor] = None,
                predictions: Optional[np.ndarray] = None,
                confidences: Optional[np.ndarray] = None) -> float:
        if self._shared_data is None:
            logger.warning("No shared calibration data set for decisiveness calculation")
            return 0.0
        try:
            calibrated_confidences = self._shared_data.calibrated_confidences
            decisiveness_scores = np.abs(calibrated_confidences - 0.5)
            mean_decisiveness = np.mean(decisiveness_scores)
            return mean_decisiveness
        except Exception as e:
            logger.warning(f"Optimized decisiveness calculation failed: {e}")
            return 0.0


class OptimizedStabilityAccuracyCorr(MetricCalculator):
    """Stability-Accuracy Correlation using pre-computed data"""
    def __init__(self, n_bins: int = 0):
        self.n_bins = n_bins
        self._shared_data = None
        self._val_labels = None
    @property
    def name(self) -> str:
        return "spearman_stability_accuracy"
    def set_shared_data(self, shared_data: SharedCalibrationData, val_labels: np.ndarray):
        self._shared_data = shared_data
        self._val_labels = val_labels
    def compute(self, features: torch.Tensor, labels: torch.Tensor, **kwargs) -> float:
        if self._shared_data is None or self._val_labels is None:
            logger.warning("No shared calibration data set for stability-accuracy correlation")
            return 0.0
        try:
            stability_scores = self._shared_data.stability_scores
            raw_predictions = self._shared_data.raw_predictions
            pred_classes = np.argmax(raw_predictions, axis=1)
            accuracy_scores = (pred_classes == self._val_labels).astype(float)
            if self.n_bins > 1:
                stability_scores, accuracy_scores = self._bin_scores(stability_scores, accuracy_scores)
            if len(np.unique(stability_scores)) < 2:
                logger.warning("Insufficient variance in stability scores for correlation")
                return 0.0
            correlation, p_value = spearmanr(stability_scores, accuracy_scores)
            if np.isnan(correlation):
                logger.warning("Spearman correlation returned NaN")
                return 0.0
            logger.debug(f"   📈 Stability-accuracy correlation from shared data: {correlation:.3f}")
            return float(correlation)
        except Exception as e:
            logger.warning(f"Optimized stability-accuracy correlation failed: {e}")
            return 0.0
    def _bin_scores(self, stability_scores: np.ndarray, accuracy_scores: np.ndarray) -> tuple:
        min_stab, max_stab = stability_scores.min(), stability_scores.max()
        if max_stab - min_stab < 1e-6:
            return stability_scores, accuracy_scores
        bin_edges = np.linspace(min_stab, max_stab, self.n_bins + 1)
        bin_indices = np.digitize(stability_scores, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        binned_stability = []
        binned_accuracy = []
        for bin_idx in range(self.n_bins):
            mask = (bin_indices == bin_idx)
            if np.sum(mask) >= 5:
                binned_stability.append(stability_scores[mask].mean())
                binned_accuracy.append(accuracy_scores[mask].mean())
        if len(binned_stability) < 2:
            return stability_scores, accuracy_scores
        return np.array(binned_stability), np.array(binned_accuracy)


# Additional metric variant used elsewhere
class StabilityAccuracyCorr(MetricCalculator):
    """
    Spearman correlation between stability scores and per-sample accuracy.
    This version efficiently reuses stability scores already computed by the geometric calibrator.
    """
    def __init__(self, model_adapter, n_bins: int = 0):
        self.model = model_adapter          # must expose predict_proba
        self.n_bins = n_bins

    # These will be set from the outside (geometric calibrator)
    _validation_original = None
    _validation_labels = None
    _stability_space = None  # Reuse the existing StabilitySpace

    @property
    def name(self):
        return "spearman_stability_accuracy"

    def set_validation_data(self, X_val_original, y_val, stability_space=None):
        """Set validation data and stability space for metric computation"""
        self._validation_original = X_val_original
        self._validation_labels = y_val
        self._stability_space = stability_space

    def compute(self, features, labels=None, corrupted_features=None, **kwargs):
        """
        Compute Spearman correlation using existing stability space calculations.
        """
        if self._validation_original is None:
            raise RuntimeError("set_validation_data(...) was not called")
        
        if self._stability_space is None:
            logger.warning("No stability space provided - cannot compute stability-accuracy correlation")
            return 0.0

        try:
            # Convert features to numpy if needed
            if torch.is_tensor(features):
                features = features.cpu().numpy()

            # 1) Get model predictions for accuracy calculation
            probs = self.model.predict_proba(self._validation_original)
            pred_classes = np.argmax(probs, axis=1)
            accuracy_scores = (pred_classes == self._validation_labels).astype(float)

            # 2) Compute stability scores using the existing stability space
            stability_scores = self._stability_space.calc_stab(features, probs)
            
            # 3) Apply optional binning for noise reduction
            if self.n_bins > 1:
                stability_scores, accuracy_scores = self._bin_scores(stability_scores, accuracy_scores)

            # 4) Compute Spearman correlation
            if len(np.unique(stability_scores)) < 2:
                logger.warning("Insufficient variance in stability scores for correlation")
                return 0.0
                
            correlation, p_value = spearmanr(stability_scores, accuracy_scores)
            
            if np.isnan(correlation):
                logger.warning("Spearman correlation returned NaN")
                return 0.0
            
            return float(correlation)
            
        except Exception as e:
            logger.warning(f"Failed to compute stability-accuracy correlation: {e}")
            return 0.0

    def _bin_scores(self, stability_scores: np.ndarray, 
                   accuracy_scores: np.ndarray) -> tuple:
        """Bin stability scores and compute average accuracy per bin"""
        min_stab, max_stab = stability_scores.min(), stability_scores.max()
        if max_stab - min_stab < 1e-6:
            return stability_scores, accuracy_scores
            
        bin_edges = np.linspace(min_stab, max_stab, self.n_bins + 1)
        bin_indices = np.digitize(stability_scores, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        binned_stability = []
        binned_accuracy = []
        
        for bin_idx in range(self.n_bins):
            mask = (bin_indices == bin_idx)
            if np.sum(mask) >= 5:  # Minimum samples per bin
                binned_stability.append(stability_scores[mask].mean())
                binned_accuracy.append(accuracy_scores[mask].mean())
        
        if len(binned_stability) < 2:
            return stability_scores, accuracy_scores
            
        return np.array(binned_stability), np.array(binned_accuracy)


class GeometricECECalculator(MetricCalculator):
    """ECE Calculator that uses the geometric calibrator's output for each layer"""
    
    def __init__(self, n_bins: int = 15, model_adapter=None, device: str = 'cuda', 
                 stability_space_class=None, compression=None, library='fast_separation', metric='l2'):
        self.n_bins = n_bins
        self.model_adapter = model_adapter
        self.device = device
        self.stability_space_class = stability_space_class
        self.compression = compression
        self.library = library
        self.metric = metric
        
        # Store validation data for ECE calculation
        self._validation_features = None
        self._validation_labels = None
        self._validation_original = None
        self._training_features = None
        self._training_labels = None
    
    def set_training_data(self, training_features, training_labels):
        """Set training data for building stability space"""
        self._training_features = training_features
        self._training_labels = training_labels
    
    def set_validation_data(self, validation_features, validation_labels, validation_original):
        """Set validation data for ECE calculation"""
        self._validation_features = validation_features
        self._validation_labels = validation_labels
        self._validation_original = validation_original

    
    def compute(self, features: torch.Tensor, labels: torch.Tensor, 
                corrupted_features: Optional[torch.Tensor] = None) -> float:
        """
        Compute ECE using the exact GeometricCalibrator pipeline for this layer.
        """
        try:
            if self._training_features is None or self._training_labels is None:
                logger.warning("No training data set for geometric ECE calculation")
                return 0.0

            # Convert to numpy
            val_features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
            val_labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            train_features_np = self._training_features.cpu().numpy() if isinstance(self._training_features, torch.Tensor) else self._training_features
            train_labels_np = self._training_labels.cpu().numpy() if isinstance(self._training_labels, torch.Tensor) else self._training_labels

            from Calibrators.geometric_calibrator_new import GeometricCalibrator
            gc = GeometricCalibrator(
                model=self.model_adapter,
                X_train_embed=train_features_np,
                y_train=train_labels_np,
                compression_mode=self.compression,
                compression_param=None,
                metric=self.metric,
                library=self.library,
                auto_select_layer=False,
                use_binning=True,
                n_bins=self.n_bins
            )
            gc.fit(
                X_val_embed=val_features_np,
                y_val=val_labels_np,
                X_val_original=self._validation_original
            )
            calibrated_probs = gc.calibrate(
                X_test_embed=val_features_np,
                X_test_original=self._validation_original
            )
            return -self._calculate_ece(calibrated_probs, val_labels_np)
        except Exception as e:
            logger.warning(f"Geometric ECE calculation failed: {e}")
            return 0.0
    
    def _calculate_ece(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Expected Calibration Error"""
        confidences = np.max(predictions, axis=1)
        predicted_classes = np.argmax(predictions, axis=1)
        accuracies = (predicted_classes == labels).astype(float)
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        ece = 0.0
        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = ((confidences >= bin_lower) & (confidences <= bin_upper)) if i == 0 \
                     else ((confidences > bin_lower) & (confidences <= bin_upper))
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece




class LogisticCalibratabilityCalculator(MetricCalculator):
    """
    Fit correctness ~ sigmoid(a * margin + b). Return a combined score:
      + high AUC, + steep but stable slope, + low NLL.
    Uses prototype margin as scalar input.
    """
    def __init__(self, max_iter: int = 200):
        self.max_iter = max_iter

    @property
    def name(self) -> str:
        return "logistic_calibratability"

    def compute(self, features, labels, predictions=None, **kwargs) -> float:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score, log_loss
            X = features.cpu().numpy() if torch.is_tensor(features) else features
            y = labels.cpu().numpy() if torch.is_tensor(labels) else labels
            classes = np.unique(y)
            if len(classes) < 2:
                return 0.0
            if predictions is not None:
                correct = (predictions == y).astype(int)
            else:
                mus = {c: X[y == c].mean(axis=0) for c in classes}
                nn_pred = np.argmin([np.linalg.norm(X - mus[c], axis=1) for c in classes], axis=0)
                y_idx = np.array([np.where(classes == yy)[0][0] for yy in y])
                correct = (nn_pred == y_idx).astype(int)

            mus = {c: X[y == c].mean(axis=0) for c in classes}
            own = np.array([np.linalg.norm(x - mus[yy]) for x, yy in zip(X, y)])
            other = np.array([min(np.linalg.norm(x - mus[c]) for c in classes if c != yy)
                              for x, yy in zip(X, y)])
            margin = (other - own).reshape(-1, 1)

            if len(np.unique(correct)) < 2:
                return 0.0

            clf = LogisticRegression(max_iter=self.max_iter, solver="lbfgs")
            clf.fit(margin, correct)
            prob = clf.predict_proba(margin)[:, 1]
            auc = roc_auc_score(correct, prob)
            nll = log_loss(correct, prob, labels=[0, 1])
            slope = abs(float(clf.coef_[0, 0]))
            score = 0.5 * auc + 0.3 * np.tanh(slope) + 0.2 * (1.0 - np.tanh(nll))
            return float(score)
        except Exception as e:
            logger.warning(f"LogisticCalibratability failed: {e}")
            return 0.0


class KFoldECEUtilityCalculator(MetricCalculator):
    """
    Cross-validated ECE utility (higher is better) using StabilitySpace.
    Uses SharedCalibrationData prepared by the selector; no need to pass model_probs directly.

    For each fold:
      1) Fit StabilitySpace on (X_train, y_train).
      2) Compute stability on X_val with model_probs[val_idx].
      3) Isotonic-regress stability→accuracy to get calibrated confidences.
      4) Build calibrated probability vectors; compute ECE on the fold.
    Returns: mean over folds, as utility = 1 - ECE in [0,1].
    """
    def __init__(self, n_splits: int = 5, n_bins: int = 15,
                 iso_bins: int = 20, rng_seed: int = 0,
                 compression_mode=None, compression_param=None,
                 library: str = 'fast_separation', metric: str = 'l2'):
        self.n_splits = max(2, int(n_splits))
        self.n_bins = int(n_bins)
        self.iso_bins = int(iso_bins)
        self.rng = np.random.RandomState(int(rng_seed))
        self.compression_mode = compression_mode
        self.compression_param = compression_param
        self.library = library
        self.metric = metric
        self._shared_data: Optional[SharedCalibrationData] = None
        self._val_labels: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "kfold_ece_utility"

    def set_shared_data(self, shared_data: SharedCalibrationData, val_labels: np.ndarray):
        self._shared_data = shared_data
        self._val_labels = val_labels

    def compute(self, features: torch.Tensor, labels: torch.Tensor, **kwargs) -> float:
        if self._shared_data is None or self._val_labels is None:
            logger.warning("kfold_ece_utility: no shared data; returning 0.0")
            return 0.0
        try:
            sd = self._shared_data
            X = sd.val_features
            Y = self._val_labels
            Xorig = sd.val_original
            if X is None or Y is None or len(X) != len(Y) or len(np.unique(Y)) < 2:
                return 0.0

            n = len(X)
            if n < max(10, self.n_splits):
                return 0.0

            # Build splitter on evaluation pool
            try:
                from sklearn.model_selection import StratifiedKFold
                splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=int(self.rng.randint(0, 1<<31)))
                splits = splitter.split(X, Y)
            except Exception:
                from sklearn.model_selection import KFold
                splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=int(self.rng.randint(0, 1<<31)))
                splits = splitter.split(X)

            def ece_from_conf(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
                bins = np.linspace(0.0, 1.0, n_bins + 1)
                idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
                ece = 0.0
                for b in range(n_bins):
                    m = (idx == b)
                    if not np.any(m):
                        continue
                    acc = correct[m].mean()
                    cmean = conf[m].mean()
                    ece += float(m.mean()) * abs(acc - cmean)
                return float(ece)

            eces: List[float] = []
            # Two paths:
            # (A) Full pipeline when raw images are available
            if Xorig is not None:
                from Calibrators.geometric_calibrator_new import GeometricCalibrator
                for tr_idx, te_idx in splits:
                    # Fit isotonic on fold's train portion of the eval pool; anchor geometry on global train set
                    fold_X_val = X[tr_idx]
                    fold_y_val = Y[tr_idx]
                    fold_X_val_orig = Xorig[tr_idx]

                    gc = GeometricCalibrator(
                        model=sd.model_adapter,
                        X_train_embed=sd.train_features,
                        y_train=sd.train_labels,
                        compression_mode=sd.compression_mode,
                        compression_param=sd.compression_param,
                        metric=sd.metric,
                        library=sd.library,
                        auto_select_layer=False,
                        use_binning=True,
                        n_bins=self.iso_bins
                    )
                    gc.fit(
                        X_val_embed=fold_X_val,
                        y_val=fold_y_val,
                        X_val_original=fold_X_val_orig
                    )

                    Xte = X[te_idx]
                    Xte_orig = Xorig[te_idx]
                    yte = Y[te_idx]
                    calP = gc.calibrate(X_test_embed=Xte, X_test_original=Xte_orig)
                    conf = calP.max(axis=1)
                    pred = calP.argmax(axis=1)
                    corr = (pred == yte).astype(np.float32)
                    eces.append(ece_from_conf(conf, corr, self.n_bins))
            else:
                # (B) No images: reuse precomputed stability & raw probs to isotonic-calibrate per fold
                stab_all = sd.stability_scores
                rawP_all = sd.raw_predictions if getattr(sd, "raw_predictions", None) is not None else sd.val_model_probs
                if rawP_all is None or len(rawP_all) != len(X):
                    return 0.0
                pred_all = rawP_all.argmax(axis=1)
                corr_all = (pred_all == Y).astype(np.float64)
                # Convert stability to a 0..1 proxy exactly as in SharedCalibrationCalculator
                s5, s95 = np.percentile(stab_all, [5, 95])
                srange = max(1e-12, s95 - s5)
                stab_norm = np.clip((stab_all - s5) / srange, 0.0, 1.0)
                from sklearn.isotonic import IsotonicRegression
                for tr_idx, te_idx in splits:
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(stab_norm[tr_idx], corr_all[tr_idx])
                    conf_te = np.clip(iso.predict(stab_norm[te_idx]), 0.01, 0.99)
                    eces.append(ece_from_conf(conf_te, corr_all[te_idx], self.n_bins))

            mean_ece = float(np.mean(eces)) if eces else 1.0
            return float(np.clip(1.0 - mean_ece, 0.0, 1.0))
        except Exception as e:
            logger.exception(f"KFoldECEUtility failed: {e}")
            return 0.0


# ============================================================================
# NEW DROP-IN METRIC CALCULATORS
# ============================================================================

class DecisivenessTailMassCalculator(MetricCalculator):
    """
    1 - mass of confidences in a band around 0.5 (default [0.4, 0.6]).
    Uses calibrated confidences if present via shared_data hook on OptimizedCalibrationDecisivenessCalculator.
    """
    def __init__(self, band=(0.4, 0.6)):
        self.band = band
        self._shared_data = None  # optional

    @property
    def name(self) -> str:
        return "decisiveness_tail_mass"

    def set_shared_data(self, shared_data):
        self._shared_data = shared_data

    def _get_confs(self, confidences):
        if self._shared_data is not None and getattr(self._shared_data, "calibrated_confidences", None) is not None:
            return np.asarray(self._shared_data.calibrated_confidences, dtype=np.float64)
        if confidences is not None:
            return np.asarray(confidences, dtype=np.float64)
        return None

    def compute(self, features, labels, **kwargs) -> float:
        confs = self._get_confs(kwargs.get("confidences", None))
        if confs is None or len(confs) < 3:
            return 0.0
        lo, hi = self.band
        m = (confs >= lo) & (confs <= hi)
        mass = m.mean()
        return float(np.clip(1.0 - mass, 0.0, 1.0))


class ConfidenceEntropyCalculator(MetricCalculator):
    """1 - normalized entropy of top-class confidences."""
    def __init__(self):
        self._shared_data = None

    @property
    def name(self) -> str:
        return "confidence_entropy"

    def set_shared_data(self, shared_data):
        self._shared_data = shared_data

    def _get_confs(self, confidences):
        if self._shared_data is not None and getattr(self._shared_data, "calibrated_confidences", None) is not None:
            return np.asarray(self._shared_data.calibrated_confidences, dtype=np.float64)
        if confidences is not None:
            return np.asarray(confidences, dtype=np.float64)
        return None

    def compute(self, features, labels, predictions=None, confidences=None, **kwargs) -> float:
        p = self._get_confs(confidences)
        if p is None or len(p) < 3:
            return 0.0
        p = np.clip(p, 1e-6, 1 - 1e-6)
        H = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        # Normalize by ln(2) so 1 bit ⇒ 1.0
        Hn = H / np.log(2.0)
        # map to [0,1]: high entropy -> 0, low entropy -> 1
        return float(np.clip(1.0 - Hn.mean(), 0.0, 1.0))


class ConfidentErrorRateCalculator(MetricCalculator):
    """Error rate among samples with confidence >= tau (uses calibrated confidences if available)."""
    def __init__(self, tau: float = 0.8):
        self.tau = float(tau)
        self._shared_data = None

    @property
    def name(self) -> str:
        return "confident_error_rate_tau"

    def set_shared_data(self, shared_data):
        self._shared_data = shared_data

    def compute(self, features, labels, predictions=None, confidences=None, **kwargs) -> float:
        if predictions is None:
            return 0.0
        if self._shared_data is not None and getattr(self._shared_data, "calibrated_confidences", None) is not None:
            confs = np.asarray(self._shared_data.calibrated_confidences, dtype=np.float64)
        elif confidences is not None:
            confs = np.asarray(confidences, dtype=np.float64)
        else:
            return 0.0
        y = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        yhat = np.asarray(predictions)
        m = confs >= self.tau
        if not m.any():
            return 0.0
        err = (yhat[m] != y[m]).mean()
        return float(err)


class AURCProxyCalculator(MetricCalculator):
    """
    Area under risk-coverage curve using confidence as coverage knob.
    Lower is better. Uses calibrated confidences if available.
    """
    def __init__(self):
        self._shared_data = None

    @property
    def name(self) -> str:
        return "aurc_proxy"

    def set_shared_data(self, shared_data):
        self._shared_data = shared_data

    def _get_confs(self, confidences):
        if self._shared_data is not None and getattr(self._shared_data, "calibrated_confidences", None) is not None:
            return np.asarray(self._shared_data.calibrated_confidences, dtype=np.float64)
        if confidences is not None:
            return np.asarray(confidences, dtype=np.float64)
        return None

    def compute(self, features, labels, predictions=None, confidences=None, **kwargs) -> float:
        if predictions is None:
            return 0.0
        conf = self._get_confs(confidences)
        if conf is None:
            return 0.0
        y = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        yhat = np.asarray(predictions)
        # sort by descending confidence
        order = np.argsort(-conf)
        y_ord = y[order]; yhat_ord = yhat[order]
        correct = (y_ord == yhat_ord).astype(float)
        risks, coverages = [], []
        for k in range(1, len(correct) + 1):
            coverages.append(k / len(correct))
            risks.append(1.0 - correct[:k].mean())
        # simple Riemann sum
        area = float(np.trapz(risks, coverages))
        return area


class ClassBalancedTailCVaRCalculator(MetricCalculator):
    """
    Per-class z-scored margins, then CVaR over the global worst tail.
    Higher is better: return 1 - normalized tail risk (like your margin_tail_cvar).
    """
    def __init__(self, alpha: float = 0.2):
        self.alpha = float(alpha)

    @property
    def name(self) -> str:
        return "class_balanced_margin_cvar"

    def compute(self, features, labels, **kwargs) -> float:
        X = features.cpu().numpy() if torch.is_tensor(features) else features
        y = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        classes = np.unique(y)
        if len(classes) < 2:
            return 0.0
        mus = {c: X[y == c].mean(axis=0) for c in classes}
        own = np.array([np.linalg.norm(x - mus[yy]) for x, yy in zip(X, y)])
        other = np.array([min(np.linalg.norm(x - mus[c]) for c in classes if c != yy)
                          for x, yy in zip(X, y)])
        m = other - own
        # per-class z-score
        mz = np.zeros_like(m, dtype=np.float64)
        for c in classes:
            idx = (y == c)
            mu, sd = m[idx].mean(), m[idx].std() + 1e-8
            mz[idx] = (m[idx] - mu) / sd
        k = max(1, int(self.alpha * len(mz)))
        worst = np.sort(mz)[:k].mean()
        # map to [0,1]: higher means safer tails
        return float(np.clip(0.5 + worst / 4.0, 0.0, 1.0))


class ImpostorGapCVaRCalculator(MetricCalculator):
    """
    For each point: (nearest other-class distance) - (nearest same-class distance).
    CVaR over worst alpha tail; larger is safer.
    """
    def __init__(self, alpha: float = 0.2):
        self.alpha = float(alpha)

    @property
    def name(self) -> str:
        return "impostor_gap_cvar"

    def compute(self, features, labels, **kwargs) -> float:
        from sklearn.neighbors import NearestNeighbors
        X = features.cpu().numpy() if torch.is_tensor(features) else features
        y = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        n = len(X)
        if n < 3:
            return 0.0
        k = min(10, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        D, I = nbrs.kneighbors(X)
        D = D[:, 1:]; I = I[:, 1:]
        gaps = np.zeros(n, dtype=np.float64)
        for i in range(n):
            same = D[i][y[I[i]] == y[i]]
            diff = D[i][y[I[i]] != y[i]]
            ds = same[0] if len(same) else D[i].mean()
            di = diff[0] if len(diff) else D[i].max()
            gaps[i] = di - ds
        ktail = max(1, int(self.alpha * n))
        worst = np.sort(gaps)[:ktail].mean()
        # Robust scaling to avoid brittle zeros with heavy tails
        std = float(np.std(gaps) + 1e-8)
        mad = float(np.median(np.abs(gaps - np.median(gaps))) + 1e-8)
        robust_scale = max(std, 1.4826 * mad)
        score = 0.5 + worst / robust_scale
        return float(np.clip(score, 0.0, 1.0))


class MarginSkewKurtosisCalculator(MetricCalculator):
    """
    Combines negative-skew and excess kurtosis into a 0..1 safety score.
    Higher is better (less heavy/worse left tail).
    """
    @property
    def name(self) -> str:
        return "margin_skewkurt_safety"

    def compute(self, features, labels, **kwargs) -> float:
        X = features.cpu().numpy() if torch.is_tensor(features) else features
        y = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        classes = np.unique(y)
        if len(classes) < 2:
            return 0.0
        mus = {c: X[y == c].mean(axis=0) for c in classes}
        own = np.array([np.linalg.norm(x - mus[yy]) for x, yy in zip(X, y)])
        other = np.array([min(np.linalg.norm(x - mus[c]) for c in classes if c != yy)
                          for x, yy in zip(X, y)])
        m = (other - own)
        z = (m - m.mean()) / (m.std() + 1e-8)
        s = float(np.mean(z ** 3))  # skew
        k = float(np.mean(z ** 4) - 3.0)  # excess kurtosis
        # penalize left skew (negative) and heavy tails (high kurtosis)
        penalty = np.clip(max(0.0, -s) + np.clip(k, 0.0, 10.0) / 10.0, 0.0, 2.0)
        return float(np.clip(1.0 - 0.5 * penalty, 0.0, 1.0))


class TemperatureEstimateStrengthCalculator(MetricCalculator):
    """
    Fit a 1-parameter temperature on prototype-softmax scores (distance-based).
    Return a score combining slope magnitude and NLL improvement.
    """
    def __init__(self, temps=np.logspace(-2, 2, 25), n_bins: int = 15):
        self.temps = np.asarray(list(temps), dtype=np.float64)
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return "temperature_estimate_strength"

    def compute(self, features, labels, **kwargs) -> float:
        X = features.cpu().numpy() if torch.is_tensor(features) else features
        y = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        C = len(np.unique(y))
        if C < 2 or len(X) < 10:
            return 0.0
        mus = {c: X[y == c].mean(axis=0) for c in np.unique(y)}
        # squared Mahalanobis-lite with scalar per-class scale
        sig = {c: X[y == c].std(axis=0).mean() + 1e-8 for c in np.unique(y)}
        D = np.stack([np.linalg.norm((X - mus[c]) / sig[c], axis=1) ** 2 for c in np.unique(y)], axis=1)
        y_idx = np.array([np.where(np.unique(y) == yy)[0][0] for yy in y])

        def nll_for_t(t):
            logP = -t * D
            logP -= logP.max(axis=1, keepdims=True)
            P = np.exp(logP); P /= P.sum(axis=1, keepdims=True)
            # NLL
            return -np.log(np.clip(P[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)).mean()

        nlls = np.array([nll_for_t(t) for t in self.temps])
        best_idx = int(np.argmin(nlls))
        best_t = float(self.temps[best_idx])
        # improvement over t=1.0
        base_nll = nll_for_t(1.0)
        imp = float(np.clip(base_nll - nlls[best_idx], 0.0, 5.0))
        # map: stronger temp (away from 1) + bigger improvement ⇒ higher score
        slope = abs(np.log(best_t))
        return float(np.clip(0.6 * np.tanh(imp) + 0.4 * np.tanh(slope), 0.0, 1.0))


