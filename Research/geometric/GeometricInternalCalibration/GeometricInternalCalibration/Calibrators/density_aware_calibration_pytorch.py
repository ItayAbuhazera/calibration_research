"""
Density Aware Calibration (DAC) with PyTorch-based kNN.

This module provides a DAC implementation that uses PyTorch for kNN computation
instead of FAISS. This enables fair timing comparisons with other methods
(like RGCL/RGCC) that also use PyTorch-based kNN.

IMPORTANT: The optimization settings are IDENTICAL to the FAISS version to ensure
fair comparison. Only the kNN backend differs.

Changes from original FAISS version:
- kNN uses PyTorch tensor operations instead of FAISS IndexFlatL2
- Preprocessing (spatial averaging, L2 norm) done on GPU with PyTorch

Identical to FAISS version:
- Optimizer settings (x0, bounds, tol)
- Loss function (squared error)
- Temperature computation
- Final calibration formula
"""

import numpy as np
from scipy import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import time

# Try to import the logger from your project, fall back to standard logging
try:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


def np_softmax(x):
    """Numerically stable softmax"""
    max_val = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_val)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


class LayerKNNScorerPyTorch:
    """
    Computes density scores (s_l) for a specific layer using k-nearest neighbors.
    
    This implementation uses PyTorch for GPU-accelerated kNN, matching the
    backend used by RGCL/RGCC for fair timing comparisons.
    
    Implements the preprocessing described in DAC Section 3.3: 
    "average over spatial dimensions as well as normalize it."
    """
    
    def __init__(self, k: int = 50, device: str = 'cuda'):
        """
        Args:
            k: Number of nearest neighbors
            device: 'cuda' or 'cpu'
        """
        self.k = k
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.train_features = None  # Stored as torch tensor on device
        
    def _preprocess(self, features) -> torch.Tensor:
        """
        1. Spatial Averaging: If features are (N, C, H, W), average to (N, C).
        2. Normalization: L2 normalize.
        
        Returns:
            Preprocessed features as torch.Tensor on self.device
        """
        # Convert to torch tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        elif isinstance(features, torch.Tensor):
            features = features.float()
        
        # Move to device
        features = features.to(self.device)
        
        # Global Average Pooling for 4D tensors (CNN feature maps)
        if features.ndim == 4:
            # shape: (N, C, H, W) -> mean over (2, 3) -> (N, C)
            features = features.mean(dim=(2, 3))
        elif features.ndim == 3:
            # shape: (N, L, C) -> mean over L -> (N, C) (e.g. Transformers)
            features = features.mean(dim=1)
            
        # L2 Normalization
        features = F.normalize(features, p=2, dim=-1)
        
        return features
    
    def fit(self, train_features):
        """
        Stores preprocessed training features for kNN queries.
        
        Args:
            train_features: Training features (N, C, H, W) or (N, C)
        """
        self.train_features = self._preprocess(train_features)
        logger.debug(f"LayerKNNScorerPyTorch fitted with {self.train_features.shape[0]} samples, "
                    f"dim={self.train_features.shape[1]}")
    
    def score(self, test_features) -> np.ndarray:
        """
        Calculates s_l: The distance to the k-th nearest neighbor.
        
        Uses PyTorch for GPU-accelerated distance computation.
        For L2-normalized vectors: ||a - b||^2 = 2 - 2*<a,b>
        
        Args:
            test_features: Test features (N, C, H, W) or (N, C)
            
        Returns:
            knn_distances: Distance to k-th nearest neighbor for each sample (N,)
        """
        if self.train_features is None:
            raise RuntimeError("Scorer not fitted. Call fit() first.")
        
        test_feats = self._preprocess(test_features)
        
        # Batch processing for memory efficiency
        batch_size = 1024
        n_test = test_feats.shape[0]
        knn_distances = []
        
        for i in range(0, n_test, batch_size):
            batch = test_feats[i:i + batch_size]
            
            # Compute cosine similarity (since features are L2 normalized)
            # similarity shape: (batch_size, n_train)
            similarity = torch.mm(batch, self.train_features.t())
            
            # Convert to L2 distance: d = sqrt(2 - 2*sim)
            # Clamp to avoid numerical issues with sqrt of negative numbers
            distances_sq = 2.0 - 2.0 * similarity
            distances_sq = torch.clamp(distances_sq, min=0.0)
            distances = torch.sqrt(distances_sq)
            
            # Get k-th smallest distance (k-th nearest neighbor)
            if self.k <= distances.shape[1]:
                # kthvalue returns (values, indices) for k-th smallest
                kth_distances, _ = torch.kthvalue(distances, self.k, dim=1)
            else:
                # If k > n_train, use the maximum distance
                kth_distances, _ = distances.max(dim=1)
            
            knn_distances.append(kth_distances.cpu().numpy())
        
        return np.concatenate(knn_distances)


class DensityAwareCalibratorPyTorch:
    """
    Density Aware Calibration using PyTorch-based kNN.
    
    This implementation is functionally IDENTICAL to the FAISS version,
    only differing in the kNN backend (PyTorch vs FAISS).
    
    Model:
        S(x, w) = sum_{l=1}^L (w_l * s_l) + w_0  (Eq. 7)
        Q(x, w) = softmax(z_L / S(x, w))         (Eq. 6)
        
    Optimization:
        Minimize Squared Error Loss (Eq. 9)
        Constraints: w_l >= 0 (positive weights for layers)
        
    IMPORTANT: Optimization settings are IDENTICAL to FAISS version:
        - x0 = [1, 1, ..., 1, 0] (bias initialized to 0)
        - bounds = [(0, None)] * L + [(None, None)] (bias unconstrained)
        - tol = 1e-12
    """
    
    def __init__(self, k: int = 50, device: str = 'cuda'):
        """
        Args:
            k: Number of nearest neighbors for density estimation
            device: 'cuda' or 'cpu'
        """
        self.k = k
        self.device = device
        self.layer_scorers: List[LayerKNNScorerPyTorch] = []
        self.weights = None  # [w_1, ..., w_L, w_0]
        self.num_layers = 0
        
        # Timing instrumentation (for fair comparison)
        self.timing = {
            'fit_knn_time': 0.0,
            'score_val_time': 0.0,
            'optimize_time': 0.0,
            'total_fit_time': 0.0,
        }
    
    def fit(self, train_features_list: List, val_features_list: List, 
            val_logits: np.ndarray, val_labels: np.ndarray):
        """
        Fits the DAC calibrator.
        
        Uses IDENTICAL optimization settings to FAISS version for fair comparison.
        
        Args:
            train_features_list: List of training features, one per layer
            val_features_list: List of validation features, one per layer
            val_logits: Validation logits (N, C)
            val_labels: Validation labels (N,)
        """
        total_start = time.perf_counter()
        
        self.num_layers = len(train_features_list)
        
        # Convert inputs to numpy if needed
        if isinstance(val_logits, torch.Tensor):
            val_logits = val_logits.detach().cpu().numpy()
        if isinstance(val_labels, torch.Tensor):
            val_labels = val_labels.detach().cpu().numpy()
        
        # Convert feature lists to numpy if needed
        train_features_list = [
            f.numpy() if isinstance(f, torch.Tensor) else f 
            for f in train_features_list
        ]
        val_features_list = [
            f.numpy() if isinstance(f, torch.Tensor) else f 
            for f in val_features_list
        ]
        
        # 1. Build KNN scorers for each layer (with timing)
        logger.info(f"Building {self.num_layers} layer scorers with PyTorch kNN (k={self.k})")
        knn_start = time.perf_counter()
        
        self.layer_scorers = []
        for i, train_feats in enumerate(train_features_list):
            scorer = LayerKNNScorerPyTorch(k=self.k, device=self.device)
            scorer.fit(train_feats)
            self.layer_scorers.append(scorer)
        
        # Sync GPU before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timing['fit_knn_time'] = time.perf_counter() - knn_start
        
        # 2. Compute density scores for validation set (with timing)
        logger.info("Computing validation density scores...")
        score_start = time.perf_counter()
        
        scores_list = []
        for i, (scorer, val_feats) in enumerate(zip(self.layer_scorers, val_features_list)):
            s_l = scorer.score(val_feats)
            scores_list.append(s_l)
            logger.debug(f"  Layer {i+1} scores: mean={s_l.mean():.4f}, std={s_l.std():.4f}")
        
        # Stack into matrix: (N, num_layers)
        knn_scores = np.column_stack(scores_list)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timing['score_val_time'] = time.perf_counter() - score_start
        
        # 3. Prepare labels as one-hot
        if val_labels.ndim == 1:
            num_classes = val_logits.shape[1]
            labels_onehot = np.eye(num_classes)[val_labels.astype(int)]
        else:
            labels_onehot = val_labels
        
        # 4. Optimize weights using scipy
        # IDENTICAL TO FAISS VERSION:
        # - x0 = [1, 1, ..., 1, 0] (bias=0)
        # - bounds = [(0, None)] * L + [(None, None)] (bias unconstrained)
        # - tol = 1e-12
        logger.info("Optimizing layer weights...")
        opt_start = time.perf_counter()
        
        # Initial guess: weights=1.0, bias=0.0 (SAME AS FAISS)
        x0 = np.ones(self.num_layers + 1)
        x0[-1] = 0.0  # bias = 0
        
        # Constraints: w_l >= 0, bias unconstrained (SAME AS FAISS)
        bounds = [(0.0, None)] * self.num_layers + [(None, None)]
        
        result = optimize.minimize(
            fun=self._squared_error_loss,
            x0=x0,
            args=(val_logits, labels_onehot, knn_scores),
            method='L-BFGS-B',
            bounds=bounds,
            tol=1e-12  # SAME AS FAISS
        )
        
        self.timing['optimize_time'] = time.perf_counter() - opt_start
        
        self.weights = result.x
        self.timing['total_fit_time'] = time.perf_counter() - total_start
        
        logger.info(f"Optimization finished. Loss: {result.fun:.6f}")
        logger.info(f"Learned Weights: {self.weights[:-1]}, Bias: {self.weights[-1]}")
        logger.info(f"Timing breakdown: kNN={self.timing['fit_knn_time']:.2f}s, "
                   f"score={self.timing['score_val_time']:.2f}s, "
                   f"opt={self.timing['optimize_time']:.2f}s, "
                   f"total={self.timing['total_fit_time']:.2f}s")
    
    def _get_temperature(self, knn_scores_matrix: np.ndarray) -> np.ndarray:
        """
        Computes temperature S(x,w) = sum(w_l * s_l) + w_0 (Eq. 7)
        
        IDENTICAL to FAISS version.
        """
        layer_weights = self.weights[:-1]
        bias = self.weights[-1]
        
        T = np.dot(knn_scores_matrix, layer_weights) + bias
        
        # Numerical stability: temperature must be > 0
        return np.maximum(T, 1e-12)
    
    def _squared_error_loss(self, weights, logits, labels_onehot, knn_scores):
        """
        Squared error loss (Eq. 9): L_w = sum_{c=1}^C (I_c - softmax(z/S)^c)^2
        
        IDENTICAL to FAISS version.
        """
        layer_weights = weights[:-1]
        bias = weights[-1]
        
        # Temperature
        T = np.dot(knn_scores, layer_weights) + bias
        T = np.clip(T, 1e-12, None)
        
        # Scaled logits
        scaled_logits = logits / T[:, np.newaxis]
        
        # Probabilities
        probs = np_softmax(scaled_logits)
        
        # Squared error loss
        squared_diff = (labels_onehot - probs) ** 2
        loss = np.sum(squared_diff)
        
        return loss
    
    def calibrate(self, test_features_list: List, test_logits: np.ndarray) -> np.ndarray:
        """
        Applies calibration to test data.
        
        Args:
            test_features_list: List of test features, one per layer
            test_logits: Test logits (N, C)
            
        Returns:
            Calibrated probabilities (N, C)
        """
        if self.weights is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        # Convert to numpy if needed
        if isinstance(test_logits, torch.Tensor):
            test_logits = test_logits.detach().cpu().numpy()
        
        test_features_list = [
            f.numpy() if isinstance(f, torch.Tensor) else f
            for f in test_features_list
        ]
        
        if len(test_features_list) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} layers, got {len(test_features_list)}")
        
        # Compute density scores
        scores_list = []
        for scorer, feats in zip(self.layer_scorers, test_features_list):
            s_l = scorer.score(feats)
            scores_list.append(s_l)
        
        knn_scores = np.column_stack(scores_list)
        
        # Get temperature and scale logits
        T = self._get_temperature(knn_scores)
        scaled_logits = test_logits / T[:, np.newaxis]
        
        # Apply softmax
        calibrated_probs = np_softmax(scaled_logits)
        
        return calibrated_probs
    
    def get_scaled_logits(self, features_list: List, logits: np.ndarray) -> np.ndarray:
        """
        Returns logits scaled by temperature (before softmax).
        Useful for chaining with other calibrators.
        """
        if self.weights is None:
            raise ValueError("Calibrator not fitted.")
        
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        
        features_list = [
            f.numpy() if isinstance(f, torch.Tensor) else f
            for f in features_list
        ]
        
        # Compute density scores
        scores_list = []
        for scorer, feats in zip(self.layer_scorers, features_list):
            s_l = scorer.score(feats)
            scores_list.append(s_l)
        
        knn_scores = np.column_stack(scores_list)
        
        # Scale logits by temperature
        T = self._get_temperature(knn_scores)
        scaled_logits = logits / T[:, np.newaxis]
        
        return scaled_logits
    
    def get_timing(self) -> dict:
        """Returns timing breakdown from last fit() call."""
        return self.timing.copy()


# =============================================================================
# Convenience wrapper for drop-in replacement
# =============================================================================

def create_dac_pytorch(k: int = 50, use_gpu: bool = True) -> DensityAwareCalibratorPyTorch:
    """
    Create a DAC calibrator with PyTorch backend.
    
    This is a drop-in replacement for:
        dac = DensityAwareCalibrator(k=k, use_gpu=use_gpu)
    
    Args:
        k: Number of nearest neighbors
        use_gpu: Whether to use GPU (if available)
        
    Returns:
        DensityAwareCalibratorPyTorch instance
    """
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    return DensityAwareCalibratorPyTorch(k=k, device=device)


# =============================================================================
# Re-export functions from original module for convenience
# =============================================================================

try:
    from Calibrators.density_aware_calibration import (
        get_dac_target_layers,
        extract_dac_features,
        get_dac_k_value,
        find_layer_module
    )
except ImportError:
    logger.warning("Could not import helper functions from density_aware_calibration. "
                  "Make sure to import them separately.")


# =============================================================================
# Test code
# =============================================================================

if __name__ == "__main__":
    print("Testing DensityAwareCalibratorPyTorch...")
    print("=" * 60)
    
    # Create synthetic data
    n_train, n_test = 5000, 1000
    n_classes = 100
    feature_dims = [64, 256, 512, 1024, 2048]  # Typical ResNet dims
    n_layers = len(feature_dims)
    
    print(f"Synthetic data: {n_train} train, {n_test} test, {n_classes} classes, {n_layers} layers")
    
    # Random features and logits
    np.random.seed(42)
    train_features = [np.random.randn(n_train, d).astype(np.float32) 
                     for d in feature_dims]
    val_features = [np.random.randn(n_train, d).astype(np.float32) 
                   for d in feature_dims]
    test_features = [np.random.randn(n_test, d).astype(np.float32) 
                    for d in feature_dims]
    
    val_logits = np.random.randn(n_train, n_classes).astype(np.float32)
    test_logits = np.random.randn(n_test, n_classes).astype(np.float32)
    val_labels = np.random.randint(0, n_classes, n_train)
    
    # Test PyTorch version
    print("\n--- PyTorch kNN Backend ---")
    dac_torch = DensityAwareCalibratorPyTorch(k=200, device='cuda')
    
    fit_start = time.perf_counter()
    dac_torch.fit(train_features, val_features, val_logits, val_labels)
    fit_time = time.perf_counter() - fit_start
    
    cal_start = time.perf_counter()
    probs_torch = dac_torch.calibrate(test_features, test_logits)
    cal_time = time.perf_counter() - cal_start
    
    print(f"Total fit time: {fit_time:.2f}s")
    print(f"Calibration time: {cal_time:.4f}s")
    print(f"Throughput: {n_test / cal_time:.1f} samples/sec")
    print(f"Output shape: {probs_torch.shape}")
    print(f"Sum of probabilities: {probs_torch.sum(axis=1).mean():.4f}")
    print(f"Weights: {dac_torch.weights}")
    
    print("\n" + "=" * 60)
    print("Test passed!")