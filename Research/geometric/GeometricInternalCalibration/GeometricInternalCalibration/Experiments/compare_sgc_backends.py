"""
Compare SGC timing with different KNN backends.
Goal: Apples-to-apples comparison with DAC.
"""

import numpy as np
import time
import json
import faiss
import logging
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
import torch

logger = logging.getLogger(__name__)


class TorchGaussianRandomProjectionJL:
    """
    Deterministic Gaussian random projection implemented in PyTorch.

    Matches sklearn.GaussianRandomProjection scaling:
      components_ ~ N(0, 1 / sqrt(n_components))
    so each entry has variance 1 / n_components.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int, device: torch.device):
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.seed = int(seed)
        self.device = device

        # std = 1 / sqrt(n_components)
        scale = 1.0 / (float(self.out_dim) ** 0.5)

        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)

        # (out_dim, in_dim) so that X @ R^T has shape (N, out_dim)
        self.R = torch.randn(
            (self.out_dim, self.in_dim),
            device=device,
            dtype=torch.float32,
            generator=gen,
        ) * scale

    def project_np(self, x_np: np.ndarray) -> np.ndarray:
        """Project a NumPy array (N, in_dim) -> (N, out_dim), return NumPy float32."""
        x_t = torch.from_numpy(x_np).to(self.device, dtype=torch.float32)
        y_t = torch.nn.functional.linear(x_t, self.R)
        return y_t.detach().cpu().numpy().astype(np.float32, copy=False)


class FastStabilityCalculator:
    """
    Efficient FAISS-based stability calculator.
    Designed to match DAC's computational efficiency.
    
    OPTIMIZED: Uses single global index instead of O(K*N) memory for "other" indices.
    """
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.same_indices = {}  # label -> FAISS index (small, ~N/K samples each)
        self.global_index = None  # Single global index for all training data
        self.y_train = None  # Training labels for filtering
        self.X_train = None  # Training features reference
        self.gpu_resources = None
        
        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()
    
    def _create_index(self, features):
        """Create FAISS index, optionally on GPU."""
        d = features.shape[1]
        index = faiss.IndexFlatL2(d)
        
        if self.use_gpu and self.gpu_resources is not None:
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
        
        index.add(np.ascontiguousarray(features.astype('float32')))
        return index
    
    def fit(self, X_train, y_train):
        """
        Build FAISS indices - optimized to avoid O(K*N) memory.
        
        Instead of creating K "other" indices (each containing ~99% of data),
        we use a single global index and filter by label at query time.
        """
        self.num_classes = len(np.unique(y_train))
        self.y_train = y_train.copy()
        self.X_train = X_train.copy()  # Keep reference for label filtering
        
        # Single global index for "other" class queries (O(N*D) memory, not O(K*N*D))
        logger.info(f"Building global FAISS index for {len(X_train)} samples...")
        self.global_index = self._create_index(X_train)
        
        # Per-class indices only for "same" class (small, ~N/K samples each)
        logger.info(f"Building per-class indices for {self.num_classes} classes...")
        for label in range(self.num_classes):
            idx_same = np.where(y_train == label)[0]
            if len(idx_same) > 0:
                self.same_indices[label] = self._create_index(X_train[idx_same])
    
    def calculate_stability(self, X_test, predictions):
        """
        Batch stability calculation - EXACT equivalent to sklearn NearestNeighbors.
        
        Key fix: Search enough neighbors to GUARANTEE finding different-class neighbor.
        This ensures mathematical equivalence with exact 1-NN search.
        
        Args:
            X_test: Test features (N, D)
            predictions: Predicted class labels (N,)
        
        Returns:
            stability: (N,) array of stability scores
        """
        X_test = np.ascontiguousarray(X_test.astype('float32'))
        n_samples = len(X_test)
        stability = np.zeros(n_samples)
        
        # FIX: Search ALL neighbors to guarantee finding different class
        # This is still faster than sklearn due to FAISS optimizations
        # FAISS GPU has limit of k=2048 for single query
        MAX_K_GPU = 2048
        
        # Determine k_search: need to guarantee finding different-class neighbor
        # Worst case: all neighbors of same class, so need at least (N - same_class_count) + 1
        # For safety, search all training samples
        k_search_other = len(self.y_train)
        
        # Check if we can fit full search in GPU memory
        if self.use_gpu and k_search_other > MAX_K_GPU:
            # Fall back to CPU for large k search
            logger.warning(f"Falling back to CPU search: k={k_search_other} > GPU limit {MAX_K_GPU}")
            # Create CPU index for this search
            cpu_index = faiss.IndexFlatL2(self.X_train.shape[1])
            cpu_index.add(np.ascontiguousarray(self.X_train.astype('float32')))
            global_distances_sq, global_indices = cpu_index.search(X_test, k_search_other)
        else:
            # Use GPU index if k is within limits, otherwise use CPU
            if self.use_gpu and k_search_other <= MAX_K_GPU:
                global_distances_sq, global_indices = self.global_index.search(X_test, k_search_other)
            else:
                # Create CPU index for large k
                cpu_index = faiss.IndexFlatL2(self.X_train.shape[1])
                cpu_index.add(np.ascontiguousarray(self.X_train.astype('float32')))
                global_distances_sq, global_indices = cpu_index.search(X_test, k_search_other)
        
        # Group by predicted class for efficient batch processing
        for label in range(self.num_classes):
            mask = (predictions == label)
            sample_indices = np.where(mask)[0]
            
            if len(sample_indices) == 0:
                continue
            
            batch_X = X_test[sample_indices]
            
            # Same-class distance (from per-class index) - search k=1 (exact)
            if label in self.same_indices:
                d_same_sq, _ = self.same_indices[label].search(batch_X, 1)
                d_same = np.sqrt(d_same_sq[:, 0])
            else:
                d_same = np.full(len(sample_indices), np.inf)
            
            # Other-class distance (from global search, filter by label)
            # Now guaranteed to find different-class neighbor since we searched all
            d_other = np.full(len(sample_indices), np.inf)
            for i, sample_idx in enumerate(sample_indices):
                # Find first neighbor from different class
                for j in range(k_search_other):
                    neighbor_idx = global_indices[sample_idx, j]
                    # FAISS may return -1 for invalid indices (shouldn't happen with k <= n_train)
                    if neighbor_idx < 0 or neighbor_idx >= len(self.y_train):
                        continue
                    if self.y_train[neighbor_idx] != label:
                        d_other[i] = np.sqrt(global_distances_sq[sample_idx, j])
                        break
            
            stability[sample_indices] = (d_other - d_same) / 2
        
        return stability


class SGCCalibratorFast:
    """
    Simplified SGC calibrator using efficient FAISS backend.
    For timing comparison with DAC.
    """
    
    def __init__(self, use_gpu=True):
        self.stability_calc = FastStabilityCalculator(use_gpu=use_gpu)
        self.isotonic = IsotonicRegression(out_of_bounds='clip')
        self.val_scores_sorted = None  # Sorted validation scores for rank-based normalization
        self.is_fitted = False
        self.last_stability = None  # Store last computed stability for verification
        self.last_val_stability = None  # Store validation stability
        # For backward compatibility with older percentile-based normalization code
        # (used in logging / analysis code such as compare_dac_geometric.py)
        self.stability_p5 = None
        self.stability_p95 = None
    
    def fit_index(self, X_train, y_train):
        """Build FAISS indices only (for timing breakdown)."""
        self.stability_calc.fit(X_train, y_train)
    
    def fit_isotonic(self, X_val, y_val, val_predictions):
        """Fit isotonic regression only (assumes index already built)."""
        stability = self.stability_calc.calculate_stability(X_val, val_predictions)
        
        # Store for debugging
        self.last_val_stability = stability.copy()
        # Also store percentile stats for compatibility / analysis
        try:
            self.stability_p5 = float(np.percentile(stability, 5))
            self.stability_p95 = float(np.percentile(stability, 95))
        except Exception:
            self.stability_p5 = None
            self.stability_p95 = None
        
        # Rank-based normalization: store sorted validation scores
        self.val_scores_sorted = np.sort(stability)
        n_val_samples = len(self.val_scores_sorted)
        
        # Normalize using ranks for fitting (preserves ordering)
        from scipy.stats import rankdata
        stability_norm = rankdata(stability, method='average') / n_val_samples
        
        correctness = (val_predictions == y_val).astype(float)
        self.isotonic.fit(stability_norm, correctness)
        self.is_fitted = True
    
    def compute_stability(self, X_test, predictions):
        """Compute stability scores only (for timing breakdown)."""
        stability = self.stability_calc.calculate_stability(X_test, predictions)
        self.last_stability = stability.copy()  # Store for verification
        return stability
    
    def apply_isotonic(self, stability, predictions, logits):
        """Apply isotonic transform only (for timing breakdown)."""
        # Rank-based normalization: use searchsorted to find ranks efficiently
        normalized_stability = np.searchsorted(self.val_scores_sorted, stability) / len(self.val_scores_sorted)
        
        calibrated_conf = self.isotonic.predict(normalized_stability)
        calibrated_conf = np.clip(calibrated_conf, 1e-6, 1 - 1e-6)
        
        # Distribute probabilities
        n_samples, n_classes = logits.shape
        calibrated_probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            c = predictions[i]
            remaining = (1.0 - calibrated_conf[i]) / (n_classes - 1)
            calibrated_probs[i, :] = remaining
            calibrated_probs[i, c] = calibrated_conf[i]
        
        return calibrated_probs
    
    def fit(self, X_train_features, y_train, X_val_features, y_val, val_predictions):
        """
        Fit the SGC calibrator.
        
        Args:
            X_train_features: Training features (N_train, D)
            y_train: Training labels
            X_val_features: Validation features (N_val, D)
            y_val: Validation labels
            val_predictions: Model predictions on validation set (N_val,)
        """
        # Build FAISS indices
        self.stability_calc.fit(X_train_features, y_train)
        
        # Calculate stability on validation set
        stability = self.stability_calc.calculate_stability(X_val_features, val_predictions)
        
        # Store for debugging
        self.last_val_stability = stability.copy()
        # Also store percentile stats for compatibility / analysis
        try:
            self.stability_p5 = float(np.percentile(stability, 5))
            self.stability_p95 = float(np.percentile(stability, 95))
        except Exception:
            self.stability_p5 = None
            self.stability_p95 = None
        
        # Rank-based normalization: store sorted validation scores
        self.val_scores_sorted = np.sort(stability)
        n_val_samples = len(self.val_scores_sorted)
        
        # Normalize using ranks for fitting (preserves ordering)
        from scipy.stats import rankdata
        stability_norm = rankdata(stability, method='average') / n_val_samples
        
        # Fit isotonic regression
        correctness = (val_predictions == y_val).astype(float)
        self.isotonic.fit(stability_norm, correctness)
        self.is_fitted = True
    
    def calibrate(self, X_test_features, predictions, logits):
        """
        Calibrate test predictions.
        
        Args:
            X_test_features: Test features (N_test, D)
            predictions: Predicted class labels (N_test,)
            logits: Model logits (N_test, C)
        
        Returns:
            calibrated_probs: (N_test, C) calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        # Calculate and normalize stability using rank-based normalization
        stability = self.stability_calc.calculate_stability(X_test_features, predictions)
        self.last_stability = stability.copy()  # Store for verification
        
        # Rank-based normalization: use searchsorted to find ranks efficiently
        normalized_stability = np.searchsorted(self.val_scores_sorted, stability) / len(self.val_scores_sorted)
        
        # Get calibrated confidence
        calibrated_conf = self.isotonic.predict(normalized_stability)
        calibrated_conf = np.clip(calibrated_conf, 1e-6, 1 - 1e-6)
        
        # Distribute probabilities
        n_samples, n_classes = logits.shape
        calibrated_probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            c = predictions[i]
            remaining = (1.0 - calibrated_conf[i]) / (n_classes - 1)
            calibrated_probs[i, :] = remaining
            calibrated_probs[i, c] = calibrated_conf[i]
        
        return calibrated_probs


def benchmark_sgc_faiss(
    train_features, train_labels,
    val_features, val_labels, val_predictions,
    test_features, test_labels, test_predictions, test_logits,
    use_gpu=True,
    target_dim=None,
    jl_seed=0,
    jl_device=None,
):
    """
    Benchmark SGC with FAISS backend.
    
    If target_dim is provided, apply the same Gaussian JL projection used in
    `compare_dac_geometric.py` to train/val/test features before SGC, for a
    fully apples-to-apples comparison.
    
    Returns timing breakdown and ECE.
    """
    results = {}

    # Optional JL projection to match DAC/SGC preprocessing
    if target_dim is not None:
        if jl_device is None:
            jl_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            jl_device = torch.device(jl_device)

        in_dim = train_features.shape[1]
        if in_dim != target_dim:
            projector = TorchGaussianRandomProjectionJL(
                in_dim=in_dim,
                out_dim=target_dim,
                seed=jl_seed,
                device=jl_device,
            )
            train_features = projector.project_np(train_features)
            val_features = projector.project_np(val_features)
            test_features = projector.project_np(test_features)
    
    # 1. Fit timing (offline)
    calibrator = SGCCalibratorFast(use_gpu=use_gpu)
    
    if use_gpu and faiss.get_num_gpus() > 0:
        # Synchronize GPU before timing
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    start = time.perf_counter()
    calibrator.fit(train_features, train_labels, val_features, val_labels, val_predictions)
    if use_gpu and faiss.get_num_gpus() > 0:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    results['fit_time_s'] = time.perf_counter() - start
    
    # 2. Calibration timing (online)
    if use_gpu and faiss.get_num_gpus() > 0:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    start = time.perf_counter()
    calibrated_probs = calibrator.calibrate(test_features, test_predictions, test_logits)
    if use_gpu and faiss.get_num_gpus() > 0:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    results['calibrate_time_s'] = time.perf_counter() - start
    
    # 3. Throughput
    n_test = len(test_features)
    results['throughput_samples_per_sec'] = n_test / results['calibrate_time_s']
    
    # 4. ECE calculation
    try:
        from netcal.metrics import ECE
        ece_calc = ECE(bins=15)
        results['ece'] = float(ece_calc.measure(calibrated_probs, test_labels))
    except ImportError:
        # Fallback: use simple ECE calculation if netcal is not available
        # This is a simplified version - for production use, import from Metrics.metrics
        confidences = np.max(calibrated_probs, axis=1)
        predictions = np.argmax(calibrated_probs, axis=1)
        correct = (predictions == test_labels).astype(float)
        n_bins = 15
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        results['ece'] = float(ece)
    
    return results


# Example usage in run_ablation_comparison.py:
def run_sgc_faiss_experiment(
    train_features, train_labels,
    val_features, val_labels, val_logits,
    test_features, test_labels, test_logits,
    target_dim=None,
    jl_seed=0,
    jl_device=None,
):
    """
    Run SGC experiment with FAISS backend for fair timing comparison.
    Optionally applies the same Gaussian JL projection used in
    `compare_dac_geometric.py` if target_dim is provided.
    """
    val_predictions = np.argmax(val_logits, axis=1)
    test_predictions = np.argmax(test_logits, axis=1)
    
    results = benchmark_sgc_faiss(
        train_features, train_labels,
        val_features, val_labels, val_predictions,
        test_features, test_labels, test_predictions, test_logits,
        use_gpu=True,
        target_dim=target_dim,
        jl_seed=jl_seed,
        jl_device=jl_device,
    )
    
    return {
        'sgc_faiss': {
            'ece': results['ece'],
            'fit_time_s': results['fit_time_s'],
            'calibrate_time_s': results['calibrate_time_s'],
            'throughput_samples_per_sec': results['throughput_samples_per_sec'],
            'method': 'SGC with FAISS backend'
        }
    }

