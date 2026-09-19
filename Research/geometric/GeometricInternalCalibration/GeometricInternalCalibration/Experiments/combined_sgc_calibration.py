"""
Combined SGC + Confidence Calibration Experiments.

This module implements and compares different strategies for combining
geometric separation scores with model confidence for improved calibration.

The key insight:
- Model confidence has high AUROC (~0.86) but poor ECE (overconfident)
- Geometric separation score has excellent ECE (~1-2%) but poor AUROC (~0.5)
- Combining them should yield: Good ECE + Good AUROC + Reasonable Sharpness

Usage:
    python combined_sgc_calibration.py \
        --model resnet50 \
        --dataset cifar100 \
        --seed 11 \
        --results_dir checkpoints \
        --output_dir combined_sgc_results
"""

import numpy as np
import torch
import torch.nn as nn
import faiss
import logging
import argparse
import json
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
from tqdm import tqdm

# Add parent directory for project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)


# ============================================================================
# Metric Computation Utilities
# ============================================================================

def compute_ece(max_probs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Compute ECE (Expected Calibration Error) with equal-width bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(max_probs)
    
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (max_probs >= bin_edges[i]) & (max_probs <= bin_edges[i+1])
        else:
            in_bin = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i+1])
        
        prop_in_bin = in_bin.sum() / n_samples
        if prop_in_bin > 0:
            acc = correct[in_bin].mean()
            conf = max_probs[in_bin].mean()
            ece += np.abs(conf - acc) * prop_in_bin
    
    return ece


def compute_mce(max_probs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Compute MCE (Maximum Calibration Error) - worst-case bin error."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    max_error = 0.0
    
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (max_probs >= bin_edges[i]) & (max_probs <= bin_edges[i+1])
        else:
            in_bin = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i+1])
        
        if in_bin.sum() > 0:
            acc = correct[in_bin].mean()
            conf = max_probs[in_bin].mean()
            max_error = max(max_error, np.abs(conf - acc))
    
    return max_error


def compute_brier_score(max_probs: np.ndarray, correct: np.ndarray) -> float:
    """Compute Brier Score - proper scoring rule (lower is better)."""
    return np.mean((max_probs - correct) ** 2)


def compute_nll(max_probs: np.ndarray, correct: np.ndarray, eps: float = 1e-15) -> float:
    """Compute Negative Log Likelihood (lower is better)."""
    probs_clipped = np.clip(max_probs, eps, 1 - eps)
    nll = -np.mean(correct * np.log(probs_clipped) + (1 - correct) * np.log(1 - probs_clipped))
    return nll


def compute_overconfidence(max_probs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Compute average signed calibration error (positive = overconfident)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    signed_error = 0.0
    n_samples = len(max_probs)
    
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (max_probs >= bin_edges[i]) & (max_probs <= bin_edges[i+1])
        else:
            in_bin = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i+1])
        
        prop_in_bin = in_bin.sum() / n_samples
        if prop_in_bin > 0:
            acc = correct[in_bin].mean()
            conf = max_probs[in_bin].mean()
            signed_error += (conf - acc) * prop_in_bin
    
    return signed_error


def compute_sharpness(max_probs: np.ndarray) -> float:
    """Compute sharpness - standard deviation of confidence scores."""
    return np.std(max_probs)


def compute_auroc_confidence(max_probs: np.ndarray, correct: np.ndarray) -> float:
    """Compute AUROC of using confidence to predict correctness."""
    from sklearn.metrics import roc_auc_score
    try:
        if len(np.unique(correct)) < 2:
            return np.nan
        return roc_auc_score(correct, max_probs)
    except Exception:
        return np.nan


def compute_all_metrics(max_probs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    """Compute all calibration metrics for a single run."""
    return {
        'ece': compute_ece(max_probs, correct, n_bins),
        'mce': compute_mce(max_probs, correct, n_bins),
        'brier': compute_brier_score(max_probs, correct),
        'nll': compute_nll(max_probs, correct),
        'overconfidence': compute_overconfidence(max_probs, correct, n_bins),
        'sharpness': compute_sharpness(max_probs),
        'auroc': compute_auroc_confidence(max_probs, correct),
    }


# ============================================================================
# FAISS-based Stability Calculator (from your codebase)
# ============================================================================

class FastStabilityCalculator:
    """
    Efficient FAISS-based stability calculator.
    Uses single global index for memory efficiency.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.same_indices = {}
        self.global_index = None
        self.y_train = None
        self.X_train = None
        self.gpu_resources = None
        self.num_classes = None
        
        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()
    
    def _create_index(self, features: np.ndarray):
        """Create FAISS index, optionally on GPU."""
        d = features.shape[1]
        index = faiss.IndexFlatL2(d)
        
        if self.use_gpu and self.gpu_resources is not None:
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
        
        index.add(np.ascontiguousarray(features.astype('float32')))
        return index
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Build FAISS indices."""
        self.num_classes = len(np.unique(y_train))
        self.y_train = y_train.copy()
        self.X_train = X_train.copy()
        
        logger.debug(f"Building global FAISS index for {len(X_train)} samples...")
        self.global_index = self._create_index(X_train)
        
        logger.debug(f"Building per-class indices for {self.num_classes} classes...")
        for label in range(self.num_classes):
            idx_same = np.where(y_train == label)[0]
            if len(idx_same) > 0:
                self.same_indices[label] = self._create_index(X_train[idx_same])
    
    def calculate_stability(self, X_test: np.ndarray, predictions: np.ndarray, 
                           k_search: int = 50) -> np.ndarray:
        """Batch stability calculation."""
        X_test = np.ascontiguousarray(X_test.astype('float32'))
        n_samples = len(X_test)
        stability = np.zeros(n_samples)
        
        k_search = min(k_search, len(self.y_train) // self.num_classes)
        global_distances_sq, global_indices = self.global_index.search(X_test, k_search)
        
        for label in range(self.num_classes):
            mask = (predictions == label)
            sample_indices = np.where(mask)[0]
            
            if len(sample_indices) == 0:
                continue
            
            batch_X = X_test[sample_indices]
            
            if label in self.same_indices:
                d_same_sq, _ = self.same_indices[label].search(batch_X, 1)
                d_same = np.sqrt(d_same_sq[:, 0])
            else:
                d_same = np.full(len(sample_indices), np.inf)
            
            d_other = np.full(len(sample_indices), np.inf)
            for i, sample_idx in enumerate(sample_indices):
                for j in range(k_search):
                    neighbor_idx = global_indices[sample_idx, j]
                    if self.y_train[neighbor_idx] != label:
                        d_other[i] = np.sqrt(global_distances_sq[sample_idx, j])
                        break
            
            stability[sample_indices] = (d_other - d_same) / 2
        
        return stability


# ============================================================================
# Temperature Scaling Baseline
# ============================================================================

class TemperatureScaling:
    """Temperature scaling calibration."""
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Fit temperature on validation set using NLL."""
        def nll_loss(T):
            scaled_logits = logits / T
            probs = torch.softmax(torch.from_numpy(scaled_logits.astype(np.float32)), dim=1).numpy()
            log_probs = np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-15, 1))
            return -np.mean(log_probs)
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        logger.info(f"Optimal temperature: {self.temperature:.4f}")
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated confidences."""
        scaled_logits = logits / self.temperature
        probs = torch.softmax(torch.from_numpy(scaled_logits.astype(np.float32)), dim=1).numpy()
        return np.max(probs, axis=1)
    
    def calibrate_probs(self, logits: np.ndarray) -> np.ndarray:
        """Return full calibrated probability distributions."""
        scaled_logits = logits / self.temperature
        return torch.softmax(torch.from_numpy(scaled_logits.astype(np.float32)), dim=1).numpy()


# ============================================================================
# Base Calibrator Class
# ============================================================================

class CombinedCalibrator:
    """Base class for combined calibrators."""
    
    def fit(self, separation_scores: np.ndarray, confidences: np.ndarray, 
            predictions: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit the calibrator on validation data.
        
        Args:
            separation_scores: (N,) raw separation/stability scores
            confidences: (N,) original model max softmax probabilities
            predictions: (N,) predicted class labels
            labels: (N,) true labels
        """
        raise NotImplementedError
    
    def calibrate(self, separation_scores: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """
        Return calibrated confidence scores.
        
        Args:
            separation_scores: (N,) raw separation scores
            confidences: (N,) original model confidences
            
        Returns:
            calibrated_confidences: (N,) calibrated probabilities
        """
        raise NotImplementedError
    
    def _normalize_separation(self, separation_scores: np.ndarray,
                              is_training: bool = False) -> np.ndarray:
        """Normalize separation scores using rank-based normalization."""
        if is_training:
            # Store sorted validation scores for rank-based normalization
            self.val_scores_sorted = np.sort(separation_scores)
            n_val_samples = len(self.val_scores_sorted)
            # Normalize using ranks for fitting
            sep_norm = rankdata(separation_scores, method='average') / n_val_samples
            return sep_norm
        
        # During calibration: map test scores to ranks based on stored validation scores
        if self.val_scores_sorted is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        n_val_samples = len(self.val_scores_sorted)
        # Use searchsorted to find insertion points, then normalize
        sep_norm = np.searchsorted(self.val_scores_sorted, separation_scores, side='right') / n_val_samples
        # Clamp to [0, 1] range
        sep_norm = np.clip(sep_norm, 0.0, 1.0)
        
        return sep_norm


# ============================================================================
# Approach 1: Weighted Combination + Isotonic
# ============================================================================

class WeightedCombinationCalibrator(CombinedCalibrator):
    """
    Learn optimal weights to combine the two signals:
    combined_score = alpha * separation_norm + (1 - alpha) * confidence
    Then apply isotonic regression on combined_score.
    """
    
    def __init__(self, optimize_metric: str = 'brier'):
        """
        Args:
            optimize_metric: Metric to optimize ('brier', 'nll', 'ece')
        """
        self.optimize_metric = optimize_metric
        self.alpha = None
        self.isotonic = IsotonicRegression(out_of_bounds='clip')
        self.val_scores_sorted = None
        self.is_fitted = False
    
    def _compute_combined_score(self, separation_norm: np.ndarray, 
                                confidences: np.ndarray, alpha: float) -> np.ndarray:
        """Compute weighted combination of separation and confidence."""
        return alpha * separation_norm + (1 - alpha) * confidences
    
    def _objective(self, alpha: float, separation_norm: np.ndarray, 
                   confidences: np.ndarray, correctness: np.ndarray) -> float:
        """Objective function for alpha optimization."""
        combined = self._compute_combined_score(separation_norm, confidences, alpha)
        isotonic_temp = IsotonicRegression(out_of_bounds='clip')
        isotonic_temp.fit(combined, correctness)
        calibrated = isotonic_temp.predict(combined)
        calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
        
        if self.optimize_metric == 'brier':
            return compute_brier_score(calibrated, correctness)
        elif self.optimize_metric == 'nll':
            return compute_nll(calibrated, correctness)
        elif self.optimize_metric == 'ece':
            return compute_ece(calibrated, correctness)
        else:
            raise ValueError(f"Unknown optimize_metric: {self.optimize_metric}")
    
    def fit(self, separation_scores: np.ndarray, confidences: np.ndarray,
            predictions: np.ndarray, labels: np.ndarray) -> None:
        """Fit the weighted combination calibrator."""
        correctness = (predictions == labels).astype(float)
        
        # Normalize separation scores
        separation_norm = self._normalize_separation(separation_scores, is_training=True)
        
        # Optimize alpha
        logger.info(f"Optimizing alpha for {self.optimize_metric}...")
        result = minimize_scalar(
            lambda a: self._objective(a, separation_norm, confidences, correctness),
            bounds=(0.0, 1.0),
            method='bounded'
        )
        
        self.alpha = result.x
        logger.info(f"Optimal alpha: {self.alpha:.4f}")
        
        # Fit isotonic regression on combined score
        combined = self._compute_combined_score(separation_norm, confidences, self.alpha)
        self.isotonic.fit(combined, correctness)
        self.is_fitted = True
    
    def calibrate(self, separation_scores: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """Calibrate using weighted combination."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        separation_norm = self._normalize_separation(separation_scores, is_training=False)
        combined = self._compute_combined_score(separation_norm, confidences, self.alpha)
        calibrated = self.isotonic.predict(combined)
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


# ============================================================================
# Approach 2: Logistic Regression on Both Features
# ============================================================================

class LogisticCombinationCalibrator(CombinedCalibrator):
    """
    Use logistic regression with both features:
    X = [separation_norm, confidence]
    y = correctness (binary)
    """
    
    def __init__(self, include_interactions: bool = True):
        """
        Args:
            include_interactions: Whether to include interaction terms
        """
        self.include_interactions = include_interactions
        self.logistic = LogisticRegression(max_iter=1000)
        self.val_scores_sorted = None
        self.is_fitted = False
    
    def _create_features(self, separation_norm: np.ndarray, 
                         confidences: np.ndarray) -> np.ndarray:
        """Create feature matrix."""
        if self.include_interactions:
            return np.column_stack([
                separation_norm,
                confidences,
                confidences ** 2,
                separation_norm * confidences
            ])
        else:
            return np.column_stack([separation_norm, confidences])
    
    def fit(self, separation_scores: np.ndarray, confidences: np.ndarray,
            predictions: np.ndarray, labels: np.ndarray) -> None:
        """Fit logistic regression on both features."""
        correctness = (predictions == labels).astype(int)
        
        # Normalize separation scores
        separation_norm = self._normalize_separation(separation_scores, is_training=True)
        
        # Prepare features
        X = self._create_features(separation_norm, confidences)
        
        # Fit logistic regression
        self.logistic.fit(X, correctness)
        self.is_fitted = True
        
        logger.info(f"Logistic coefficients: {self.logistic.coef_}")
    
    def calibrate(self, separation_scores: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """Calibrate using logistic regression."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        separation_norm = self._normalize_separation(separation_scores, is_training=False)
        X = self._create_features(separation_norm, confidences)
        calibrated = self.logistic.predict_proba(X)[:, 1]
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


# ============================================================================
# Approach 3: Product of Experts
# ============================================================================

class ProductOfExpertsCalibrator(CombinedCalibrator):
    """
    Treat each signal as independent evidence:
    1. Calibrate each signal separately with isotonic
    2. Combine using product of experts (normalized)
    P(correct | both) proportional to P(correct|sep) * P(correct|conf) / P(correct)
    """
    
    def __init__(self):
        self.isotonic_sep = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_conf = IsotonicRegression(out_of_bounds='clip')
        self.prior_accuracy = None
        self.val_scores_sorted = None
        self.is_fitted = False
    
    def fit(self, separation_scores: np.ndarray, confidences: np.ndarray,
            predictions: np.ndarray, labels: np.ndarray) -> None:
        """Fit product of experts calibrator."""
        correctness = (predictions == labels).astype(float)
        self.prior_accuracy = np.mean(correctness)
        
        # Normalize separation scores
        separation_norm = self._normalize_separation(separation_scores, is_training=True)
        
        # Calibrate each signal separately
        self.isotonic_sep.fit(separation_norm, correctness)
        self.isotonic_conf.fit(confidences, correctness)
        self.is_fitted = True
    
    def calibrate(self, separation_scores: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """Calibrate using product of experts."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        separation_norm = self._normalize_separation(separation_scores, is_training=False)
        
        # Get calibrated probabilities from each signal
        p_sep = self.isotonic_sep.predict(separation_norm)
        p_conf = self.isotonic_conf.predict(confidences)
        
        # Clip to avoid numerical issues
        p_sep = np.clip(p_sep, 1e-6, 1 - 1e-6)
        p_conf = np.clip(p_conf, 1e-6, 1 - 1e-6)
        prior = max(self.prior_accuracy, 1e-6)
        
        # Product of experts: P(correct|both) = P(correct|sep) * P(correct|conf) / P(correct)
        combined = (p_sep * p_conf) / prior
        
        return np.clip(combined, 1e-6, 1 - 1e-6)


# ============================================================================
# Approach 4: 2D Binning
# ============================================================================

class Binned2DCalibrator(CombinedCalibrator):
    """
    2D binning approach:
    Bin the 2D space (separation x confidence) and compute accuracy per bin.
    """
    
    def __init__(self, n_sep_bins: int = 10, n_conf_bins: int = 10, 
                 smoothing: bool = True):
        """
        Args:
            n_sep_bins: Number of bins for separation dimension
            n_conf_bins: Number of bins for confidence dimension
            smoothing: Apply Laplace smoothing to avoid empty bins
        """
        self.n_sep_bins = n_sep_bins
        self.n_conf_bins = n_conf_bins
        self.smoothing = smoothing
        self.bin_accuracies = None
        self.bin_counts = None
        self.prior_accuracy = None
        self.val_scores_sorted = None
        self.is_fitted = False
    
    def fit(self, separation_scores: np.ndarray, confidences: np.ndarray,
            predictions: np.ndarray, labels: np.ndarray) -> None:
        """Fit 2D binned calibrator."""
        correctness = (predictions == labels).astype(float)
        self.prior_accuracy = np.mean(correctness)
        
        # Normalize separation scores
        separation_norm = self._normalize_separation(separation_scores, is_training=True)
        
        # Create 2D bins
        sep_bins = np.digitize(separation_norm, np.linspace(0, 1, self.n_sep_bins + 1)[1:-1])
        conf_bins = np.digitize(confidences, np.linspace(0, 1, self.n_conf_bins + 1)[1:-1])
        
        # Clip to valid range
        sep_bins = np.clip(sep_bins, 0, self.n_sep_bins - 1)
        conf_bins = np.clip(conf_bins, 0, self.n_conf_bins - 1)
        
        # Compute accuracy per bin with optional smoothing
        self.bin_accuracies = {}
        self.bin_counts = {}
        
        for sep_bin in range(self.n_sep_bins):
            for conf_bin in range(self.n_conf_bins):
                mask = (sep_bins == sep_bin) & (conf_bins == conf_bin)
                count = mask.sum()
                self.bin_counts[(sep_bin, conf_bin)] = count
                
                if count > 0:
                    if self.smoothing:
                        # Laplace smoothing
                        acc = (correctness[mask].sum() + 1) / (count + 2)
                    else:
                        acc = correctness[mask].mean()
                    self.bin_accuracies[(sep_bin, conf_bin)] = acc
                else:
                    self.bin_accuracies[(sep_bin, conf_bin)] = self.prior_accuracy
        
        self.is_fitted = True
    
    def calibrate(self, separation_scores: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """Calibrate using 2D binning."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        separation_norm = self._normalize_separation(separation_scores, is_training=False)
        
        # Create 2D bins
        sep_bins = np.digitize(separation_norm, np.linspace(0, 1, self.n_sep_bins + 1)[1:-1])
        conf_bins = np.digitize(confidences, np.linspace(0, 1, self.n_conf_bins + 1)[1:-1])
        
        # Clip to valid range
        sep_bins = np.clip(sep_bins, 0, self.n_sep_bins - 1)
        conf_bins = np.clip(conf_bins, 0, self.n_conf_bins - 1)
        
        # Look up accuracy for each bin
        calibrated = np.zeros(len(separation_scores))
        for i in range(len(separation_scores)):
            key = (int(sep_bins[i]), int(conf_bins[i]))
            calibrated[i] = self.bin_accuracies.get(key, self.prior_accuracy)
        
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


# ============================================================================
# Approach 5: Stacked Isotonic
# ============================================================================

class StackedIsotonicCalibrator(CombinedCalibrator):
    """
    Two-stage approach:
    1. Calibrate confidence with isotonic -> p_conf_calibrated
    2. Use weighted combination of (separation, p_conf_calibrated) as input
    """
    
    def __init__(self, stage2_alpha: float = 0.5):
        """
        Args:
            stage2_alpha: Weight for separation in stage 2 combination
        """
        self.stage2_alpha = stage2_alpha
        self.isotonic_conf = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_combined = IsotonicRegression(out_of_bounds='clip')
        self.val_scores_sorted = None
        self.is_fitted = False
    
    def fit(self, separation_scores: np.ndarray, confidences: np.ndarray,
            predictions: np.ndarray, labels: np.ndarray) -> None:
        """Fit stacked isotonic calibrator."""
        correctness = (predictions == labels).astype(float)
        
        # Normalize separation scores
        separation_norm = self._normalize_separation(separation_scores, is_training=True)
        
        # Stage 1: Calibrate confidence
        self.isotonic_conf.fit(confidences, correctness)
        p_conf_calibrated = self.isotonic_conf.predict(confidences)
        p_conf_calibrated = np.clip(p_conf_calibrated, 1e-6, 1 - 1e-6)
        
        # Stage 2: Combine separation and calibrated confidence
        combined_input = self.stage2_alpha * separation_norm + (1 - self.stage2_alpha) * p_conf_calibrated
        self.isotonic_combined.fit(combined_input, correctness)
        self.is_fitted = True
    
    def calibrate(self, separation_scores: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """Calibrate using stacked isotonic."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        separation_norm = self._normalize_separation(separation_scores, is_training=False)
        
        # Stage 1: Calibrate confidence
        p_conf_calibrated = self.isotonic_conf.predict(confidences)
        p_conf_calibrated = np.clip(p_conf_calibrated, 1e-6, 1 - 1e-6)
        
        # Stage 2: Combine and calibrate
        combined_input = self.stage2_alpha * separation_norm + (1 - self.stage2_alpha) * p_conf_calibrated
        calibrated = self.isotonic_combined.predict(combined_input)
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


# ============================================================================
# Approach 6: MLP Combiner
# ============================================================================

class MLPCombinationCalibrator(CombinedCalibrator):
    """
    Small MLP to learn the combination:
    Input: [separation_norm, confidence, confidence^2, sep*conf]
    Output: calibrated probability
    Train with cross-entropy loss on validation set
    """
    
    def __init__(self, hidden_dim: int = 32, n_epochs: int = 200, 
                 lr: float = 0.01, weight_decay: float = 1e-4):
        """
        Args:
            hidden_dim: Hidden dimension of MLP
            n_epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization
        """
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = None
        self.val_scores_sorted = None
        self.is_fitted = False
    
    def _create_features(self, separation_norm: np.ndarray, 
                         confidences: np.ndarray) -> np.ndarray:
        """Create feature vector: [sep, conf, conf^2, sep*conf]"""
        return np.column_stack([
            separation_norm,
            confidences,
            confidences ** 2,
            separation_norm * confidences
        ])
    
    def fit(self, separation_scores: np.ndarray, confidences: np.ndarray,
            predictions: np.ndarray, labels: np.ndarray) -> None:
        """Fit MLP combiner."""
        correctness = (predictions == labels).astype(float)
        
        # Normalize separation scores
        separation_norm = self._normalize_separation(separation_scores, is_training=True)
        
        # Create features
        X = self._create_features(separation_norm, confidences)
        y = correctness
        
        # Create MLP
        self.model = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # Training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, 
                                       weight_decay=self.weight_decay)
        criterion = nn.BCELoss()
        
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > 20:  # Early stopping
                    break
        
        self.is_fitted = True
        logger.info(f"MLP training finished after {epoch+1} epochs, final loss: {loss.item():.4f}")
    
    def calibrate(self, separation_scores: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """Calibrate using MLP."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        separation_norm = self._normalize_separation(separation_scores, is_training=False)
        X = self._create_features(separation_norm, confidences)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            calibrated = self.model(X_tensor).squeeze().numpy()
        
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


# ============================================================================
# Experiment Runner
# ============================================================================

def run_comparison_experiment(
    val_separation: np.ndarray,
    val_confidences: np.ndarray,
    val_predictions: np.ndarray,
    val_labels: np.ndarray,
    val_logits: np.ndarray,
    test_separation: np.ndarray,
    test_confidences: np.ndarray,
    test_predictions: np.ndarray,
    test_labels: np.ndarray,
    test_logits: np.ndarray,
    n_bins: int = 15
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """
    Run comparison of all combination strategies.
    
    Returns:
        Tuple of:
        - Dictionary with metrics for each method
        - Dictionary with calibrated probabilities for each method
    """
    results = {}
    calibrated_probs = {}
    test_correct = (test_predictions == test_labels).astype(float)
    val_correct = (val_predictions == val_labels).astype(float)
    
    # Baseline: Uncalibrated
    logger.info("Computing uncalibrated baseline...")
    results['uncalibrated'] = compute_all_metrics(test_confidences, test_correct, n_bins)
    calibrated_probs['uncalibrated'] = test_confidences
    
    # Baseline: Temperature Scaling
    logger.info("Fitting temperature scaling...")
    temp_scaling = TemperatureScaling()
    temp_scaling.fit(val_logits, val_labels)
    test_temp_calibrated = temp_scaling.calibrate(test_logits)
    results['temperature_scaling'] = compute_all_metrics(test_temp_calibrated, test_correct, n_bins)
    calibrated_probs['temperature_scaling'] = test_temp_calibrated
    
    # Baseline: Top-label Isotonic (confidence only)
    logger.info("Fitting top-label isotonic (confidence only)...")
    isotonic_conf = IsotonicRegression(out_of_bounds='clip')
    isotonic_conf.fit(val_confidences, val_correct)
    test_conf_calibrated = isotonic_conf.predict(test_confidences)
    test_conf_calibrated = np.clip(test_conf_calibrated, 1e-6, 1 - 1e-6)
    results['isotonic_confidence_only'] = compute_all_metrics(test_conf_calibrated, test_correct, n_bins)
    calibrated_probs['isotonic_confidence_only'] = test_conf_calibrated
    
    # Baseline: SGC only (separation only)
    logger.info("Fitting SGC (separation only)...")
    isotonic_sep = IsotonicRegression(out_of_bounds='clip')
    # Rank-based normalization
    val_scores_sorted = np.sort(val_separation)
    n_val_samples = len(val_scores_sorted)
    val_sep_norm = rankdata(val_separation, method='average') / n_val_samples
    # Map test scores to ranks based on validation scores
    test_sep_norm = np.searchsorted(val_scores_sorted, test_separation, side='right') / n_val_samples
    test_sep_norm = np.clip(test_sep_norm, 0.0, 1.0)
    isotonic_sep.fit(val_sep_norm, val_correct)
    test_sep_calibrated = isotonic_sep.predict(test_sep_norm)
    test_sep_calibrated = np.clip(test_sep_calibrated, 1e-6, 1 - 1e-6)
    results['sgc_separation_only'] = compute_all_metrics(test_sep_calibrated, test_correct, n_bins)
    calibrated_probs['sgc_separation_only'] = test_sep_calibrated
    
    # Combined methods
    combined_methods = {
        'combined_weighted_brier': WeightedCombinationCalibrator(optimize_metric='brier'),
        'combined_weighted_ece': WeightedCombinationCalibrator(optimize_metric='ece'),
        'combined_logistic': LogisticCombinationCalibrator(include_interactions=False),
        'combined_logistic_interactions': LogisticCombinationCalibrator(include_interactions=True),
        'combined_product': ProductOfExpertsCalibrator(),
        'combined_2d_binned_10x10': Binned2DCalibrator(n_sep_bins=10, n_conf_bins=10),
        'combined_2d_binned_20x20': Binned2DCalibrator(n_sep_bins=20, n_conf_bins=20),
        'combined_stacked': StackedIsotonicCalibrator(stage2_alpha=0.5),
        'combined_mlp': MLPCombinationCalibrator(hidden_dim=32, n_epochs=200),
    }
    
    for method_name, calibrator in combined_methods.items():
        logger.info(f"Fitting {method_name}...")
        try:
            calibrator.fit(val_separation, val_confidences, val_predictions, val_labels)
            test_calibrated = calibrator.calibrate(test_separation, test_confidences)
            results[method_name] = compute_all_metrics(test_calibrated, test_correct, n_bins)
            calibrated_probs[method_name] = test_calibrated
            logger.info(f"  {method_name} - ECE: {results[method_name]['ece']:.4f}, "
                       f"AUROC: {results[method_name]['auroc']:.4f}, "
                       f"Sharpness: {results[method_name]['sharpness']:.4f}")
        except Exception as e:
            logger.error(f"Error fitting {method_name}: {e}")
            import traceback
            traceback.print_exc()
            results[method_name] = {k: np.nan for k in ['ece', 'mce', 'brier', 'nll', 
                                                         'overconfidence', 'sharpness', 'auroc']}
            calibrated_probs[method_name] = np.full_like(test_confidences, np.nan)
    
    return results, calibrated_probs


def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print results in a nice table format."""
    print("\n" + "="*100)
    print(f"{'Method':<35} {'ECE':>10} {'AUROC':>10} {'Brier':>10} {'Sharpness':>10} {'NLL':>10}")
    print("="*100)
    
    # Sort by ECE
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('ece', float('inf')))
    
    for method, metrics in sorted_results:
        ece = metrics.get('ece', np.nan)
        auroc = metrics.get('auroc', np.nan)
        brier = metrics.get('brier', np.nan)
        sharpness = metrics.get('sharpness', np.nan)
        nll = metrics.get('nll', np.nan)
        
        print(f"{method:<35} {ece:>10.4f} {auroc:>10.4f} {brier:>10.4f} {sharpness:>10.4f} {nll:>10.4f}")
    
    print("="*100)


def compute_reliability_bins(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15
) -> Dict[str, np.ndarray]:
    """
    Compute reliability diagram statistics.
    
    Returns:
        Dictionary with bin_centers, bin_accuracies, bin_confidences, bin_counts, ece
    """
    # Remove NaN values
    valid_mask = ~(np.isnan(confidences) | np.isnan(correct))
    confidences = confidences[valid_mask]
    correct = correct[valid_mask]
    
    if len(confidences) == 0:
        return {
            'bin_centers': np.array([]),
            'bin_accuracies': np.array([]),
            'bin_confidences': np.array([]),
            'bin_counts': np.array([]),
            'ece': np.nan
        }
    
    # Create bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Assign samples to bins
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute statistics per bin
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0
    n_samples = len(confidences)
    
    for i in range(n_bins):
        mask = bin_indices == i
        count = mask.sum()
        bin_counts.append(count)
        
        if count > 0:
            acc = correct[mask].mean()
            conf = confidences[mask].mean()
            bin_accuracies.append(acc)
            bin_confidences.append(conf)
            # ECE contribution
            prop = count / n_samples
            ece += prop * np.abs(conf - acc)
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(bin_centers[i])
    
    return {
        'bin_centers': bin_centers,
        'bin_accuracies': np.array(bin_accuracies),
        'bin_confidences': np.array(bin_confidences),
        'bin_counts': np.array(bin_counts),
        'ece': ece
    }


def plot_reliability_diagrams(
    results: Dict[str, Dict[str, float]],
    calibrated_probs: Dict[str, np.ndarray],
    test_labels: np.ndarray,
    test_predictions: np.ndarray,
    output_path: Path,
    n_bins: int = 15,
    figsize: Tuple[float, float] = (20, 5)
) -> None:
    """
    Plot reliability diagrams for all methods in a single figure.
    
    Args:
        results: Dictionary with metrics for each method
        calibrated_probs: Dictionary with calibrated probabilities for each method
        test_labels: True labels
        test_predictions: Predicted labels (for computing correctness)
        output_path: Path to save the figure
        n_bins: Number of bins for reliability diagram
        figsize: Figure size (width, height)
    """
    # Method display names
    method_names = {
        'uncalibrated': 'Uncalibrated',
        'temperature_scaling': 'Temperature Scaling',
        'isotonic_confidence_only': 'Isotonic (Conf)',
        'sgc_separation_only': 'SGC (Sep)',
        'combined_weighted_brier': 'Weighted (Brier)',
        'combined_weighted_ece': 'Weighted (ECE)',
        'combined_logistic': 'Logistic',
        'combined_logistic_interactions': 'Logistic (Int)',
        'combined_product': 'Product of Experts',
        'combined_2d_binned_10x10': '2D Binned (10×10)',
        'combined_2d_binned_20x20': '2D Binned (20×20)',
        'combined_stacked': 'Stacked Isotonic',
        'combined_mlp': 'MLP Combiner',
    }
    
    # Filter methods that have valid results
    valid_methods = []
    for method in method_names.keys():
        if method in results and method in calibrated_probs:
            if not np.isnan(results[method].get('ece', np.nan)):
                if not np.all(np.isnan(calibrated_probs[method])):
                    valid_methods.append(method)
    
    if len(valid_methods) == 0:
        logger.warning("No valid methods to plot")
        return
    
    # Sort by ECE
    valid_methods = sorted(valid_methods, key=lambda m: results[m].get('ece', float('inf')))
    
    # Create figure with subplots
    n_methods = len(valid_methods)
    n_cols = min(7, n_methods)  # Max 7 columns
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes_array = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * max(1, n_rows / 5)))
    
    # Flatten axes array for easier indexing
    if n_methods == 1:
        axes_list = [axes_array]
    elif n_rows == 1:
        axes_list = axes_array.flatten() if hasattr(axes_array, 'flatten') else [axes_array]
    else:
        axes_list = axes_array.flatten()
    
    # Compute correctness
    test_correct = (test_predictions == test_labels).astype(float)
    
    # Plot each method
    for idx, method in enumerate(valid_methods):
        ax = axes_list[idx]
        
        confidences = calibrated_probs[method]
        correct = test_correct
        
        # Compute reliability bins
        bin_stats = compute_reliability_bins(confidences, correct, n_bins)
        
        if len(bin_stats['bin_centers']) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{method_names.get(method, method)}\nECE: N/A", fontsize=9)
            continue
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect')
        
        # Plot reliability curve
        valid_mask = ~np.isnan(bin_stats['bin_accuracies'])
        if np.any(valid_mask):
            ax.plot(
                bin_stats['bin_confidences'][valid_mask],
                bin_stats['bin_accuracies'][valid_mask],
                'o-',
                linewidth=1.5,
                markersize=4,
                label='Model'
            )
        
        # Set labels and title
        ece = results[method].get('ece', np.nan)
        title = f"{method_names.get(method, method)}\nECE = {ece:.3f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Confidence', fontsize=8)
        if idx % n_cols == 0:
            ax.set_ylabel('Accuracy', fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes_list)):
        axes_list[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Reliability diagrams saved to: {output_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main experiment runner with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Combined SGC + Confidence Calibration Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model/dataset configuration
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'tiny_imagenet'])
    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet18', 'resnet50', 'resnet101', 'resnet152', 'densenet121'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--training_method', type=str, default='baseline_cross_entropy')
    
    # Paths
    parser.add_argument('--results_dir', type=str, default='checkpoints',
                        help='Base directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='combined_sgc_results',
                        help='Output directory for results')
    
    # Experimental parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_bins', type=int, default=15)
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of layers to sample for SGC features')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='Target dimension for feature projection')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import project modules
    try:
        from Experiments.run_post_hoc_calibration import (
            PyTorchModelAdapter,
            get_data_loaders,
            load_trained_model,
            construct_model_path
        )
        from Experiments.compare_dac_geometric import (
            extract_and_aggregate_sgc_features,
            normalize_discovered_layers,
            filter_non_feature_layers
        )
        from Experiments.multi_layer_ensemble import discover_model_layers
    except ImportError as e:
        logger.error(f"Failed to import project modules: {e}")
        logger.error("Make sure you're running from the project root directory.")
        return
    
    # Determine number of classes
    num_classes = {'cifar10': 10, 'cifar100': 100, 'tiny_imagenet': 200}[args.dataset]
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Construct model path
    model_path = construct_model_path(
        base_dir=args.results_dir,
        method=args.training_method,
        dataset=args.dataset,
        model=args.model,
        seed=args.seed
    )
    
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load model
    model = load_trained_model(model_path, args.model, num_classes, device, args.dataset)
    model.eval()
    
    # Get data loaders
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        args.dataset, args.batch_size, args.seed
    )
    
    # Create model adapter
    model_adapter = PyTorchModelAdapter(model, device, args.dataset)
    
    # Extract raw data
    def extract_raw(loader):
        xs, ys = [], []
        for batch in loader:
            data, labels = batch[:2]
            xs.append(data.numpy())
            ys.append(labels.numpy())
        return np.concatenate(xs), np.concatenate(ys)
    
    logger.info("Extracting raw data...")
    train_raw, train_labels = extract_raw(train_loader)
    val_raw, val_labels = extract_raw(val_loader)
    test_raw, test_labels = extract_raw(test_loader)
    
    # Get model predictions and confidences
    logger.info("Computing model predictions...")
    val_probs = model_adapter.predict_proba(val_raw, batch_size=args.batch_size)
    val_logits = model_adapter.predict_logits(val_raw, batch_size=args.batch_size)
    val_predictions = np.argmax(val_probs, axis=1)
    val_confidences = np.max(val_probs, axis=1)
    
    test_probs = model_adapter.predict_proba(test_raw, batch_size=args.batch_size)
    test_logits = model_adapter.predict_logits(test_raw, batch_size=args.batch_size)
    test_predictions = np.argmax(test_probs, axis=1)
    test_confidences = np.max(test_probs, axis=1)
    
    # Discover and select layers for SGC features
    logger.info("Discovering model layers...")
    if args.dataset.lower() in ['tiny_imagenet', 'tinyimagenet']:
        input_shape = (1, 3, 64, 64)
    else:
        input_shape = (1, 3, 32, 32)
    
    discovered_layers = normalize_discovered_layers(
        model, discover_model_layers(model, device=device, input_shape=input_shape)
    )
    filtered_layers, _ = filter_non_feature_layers(discovered_layers, model)
    all_layer_names = [d['name'] for d in filtered_layers]
    
    # Sample random layers
    np.random.seed(args.seed)
    if args.num_layers < len(all_layer_names):
        selected_layers = list(np.random.choice(all_layer_names, size=args.num_layers, replace=False))
    else:
        selected_layers = all_layer_names
    logger.info(f"Selected {len(selected_layers)} layers for SGC features")
    
    # Extract SGC features
    logger.info("Extracting SGC features...")
    train_features, val_features, test_features, _ = extract_and_aggregate_sgc_features(
        model=model,
        layer_names=selected_layers,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        target_dim=args.projection_dim,
        seed=args.seed,
        pooling_mode='max'
    )
    
    # Build stability calculator and compute separation scores
    logger.info("Building FAISS index and computing separation scores...")
    stability_calc = FastStabilityCalculator(use_gpu=args.use_gpu)
    stability_calc.fit(train_features, train_labels)
    
    val_separation = stability_calc.calculate_stability(val_features, val_predictions)
    val_separation = np.nan_to_num(val_separation, nan=0.0, posinf=1e6, neginf=-1e6)
    
    test_separation = stability_calc.calculate_stability(test_features, test_predictions)
    test_separation = np.nan_to_num(test_separation, nan=0.0, posinf=1e6, neginf=-1e6)
    
    logger.info(f"Separation score stats (val): min={val_separation.min():.4f}, "
                f"max={val_separation.max():.4f}, mean={val_separation.mean():.4f}")
    
    # Run comparison experiment
    logger.info("\n" + "="*60)
    logger.info("RUNNING COMPARISON EXPERIMENT")
    logger.info("="*60)
    
    results, calibrated_probs = run_comparison_experiment(
        val_separation=val_separation,
        val_confidences=val_confidences,
        val_predictions=val_predictions,
        val_labels=val_labels,
        val_logits=val_logits,
        test_separation=test_separation,
        test_confidences=test_confidences,
        test_predictions=test_predictions,
        test_labels=test_labels,
        test_logits=test_logits,
        n_bins=args.n_bins
    )
    
    # Print results
    print_results_table(results)
    
    # Plot reliability diagrams
    logger.info("\n" + "="*60)
    logger.info("GENERATING RELIABILITY DIAGRAMS")
    logger.info("="*60)
    
    reliability_fig_path = output_dir / f"{args.dataset}_{args.model}_seed{args.seed}_reliability_diagrams.png"
    plot_reliability_diagrams(
        results=results,
        calibrated_probs=calibrated_probs,
        test_labels=test_labels,
        test_predictions=test_predictions,
        output_path=reliability_fig_path,
        n_bins=args.n_bins
    )
    
    # Save results
    output_file = output_dir / f"{args.dataset}_{args.model}_seed{args.seed}_combined_sgc.json"
    
    # Convert numpy values to Python floats for JSON serialization
    results_serializable = {}
    for method, metrics in results.items():
        results_serializable[method] = {k: float(v) if not np.isnan(v) else None 
                                        for k, v in metrics.items()}
    
    with open(output_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'model': args.model,
            'seed': args.seed,
            'training_method': args.training_method,
            'num_layers': args.num_layers,
            'projection_dim': args.projection_dim,
            'methods': results_serializable
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()