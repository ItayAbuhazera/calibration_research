"""
K-neighbor ablation study for SGC.
Tests the unification hypothesis between geometric separation (k=1) and density-aware calibration (k=50+).

This script runs experiments across:
- k values: [1, 5, 10, 25, 50, 100, 200]
- Layer selections: random_6, dac_layers, last_layer, random_1
- Feature modes: sgc (SPP+JL), coordinate (raw sampling)
"""

import numpy as np
import torch
import faiss
import logging
import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import normalize
from scipy.stats import rankdata

# Import functions from run_post_hoc_calibration.py
from Experiments.run_post_hoc_calibration import (
    get_data_loaders,
    load_trained_model,
    construct_model_path,
    PyTorchModelAdapter,
)

# Import functions from compare_dac_geometric.py
from Experiments.compare_dac_geometric import (
    # Feature extraction
    extract_and_aggregate_sgc_features,
    discover_coordinate_space,
    extract_coordinate_features,
    validate_and_correct_coordinate_space,
    
    # Layer selection
    get_dac_target_layers,
    normalize_discovered_layers,
    discover_model_layers,
    filter_non_feature_layers,
    
    # Metrics
    calculate_ece,
    calculate_adaptive_ece,
    calculate_brier_score,
    calculate_accuracy,
    compute_ood_metrics,
    
    # Utilities  
    save_results_incrementally,
    make_json_serializable,
)

from utils.coordinate_extraction import plan_coordinate_extraction

logger = logging.getLogger(__name__)


class KNeighborSeparationCalculator:
    """
    Computes k-smoothed separation scores:
    s_k(x) = (mean_dist_to_k_other - mean_dist_to_k_same) / 2
    
    k=1 recovers SGC's original separation score
    k=50+ approaches DAC's density-aware behavior
    """
    
    def __init__(self, k=1, use_gpu=True):
        self.k = k
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.same_indices = {}  # label -> FAISS index (per-class indices)
        self.same_index_sizes = {}  # label -> size of per-class index
        self.global_index = None  # Single global index for all training data
        self.y_train = None  # Training labels for filtering
        self.X_train = None  # Training features reference
        self.gpu_resources = None
        self.num_classes = None
        self.MAX_GPU_K = 2048  # FAISS GPU hard limit for k in search
        
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
        Build FAISS indices for k-neighbor search.
        
        Args:
            X_train: Training features (N, D)
            y_train: Training labels (N,)
        """
        self.num_classes = len(np.unique(y_train))
        self.y_train = y_train.copy()
        self.X_train = X_train.copy()
        
        # Single global index for "other" class queries
        logger.info(f"Building global FAISS index for {len(X_train)} samples...")
        self.global_index = self._create_index(X_train)
        
        # Per-class indices for "same" class queries
        logger.info(f"Building per-class indices for {self.num_classes} classes...")
        for label in range(self.num_classes):
            idx_same = np.where(y_train == label)[0]
            if len(idx_same) > 0:
                self.same_indices[label] = self._create_index(X_train[idx_same])
                self.same_index_sizes[label] = len(idx_same)
        
        # Validation: ensure all classes have sufficient samples
        for label in range(self.num_classes):
            if label not in self.same_indices:
                logger.warning(f"  Class {label} has no training samples!")
            elif self.same_index_sizes[label] < self.k:
                logger.warning(f"  Class {label} has only {self.same_index_sizes[label]} samples, less than k={self.k}")
    
    def calculate_scores(self, X_test, predictions):
        """
        Calculate k-smoothed separation scores.
        
        Args:
            X_test: Test features (N, D)
            predictions: Predicted class labels (N,)
        
        Returns:
            separation_scores: (N,) array of k-smoothed separation scores
        """
        X_test = np.ascontiguousarray(X_test.astype('float32'))
        n_samples = len(X_test)
        separation_scores = np.zeros(n_samples)
        
        # For k=200, we need to search enough neighbors to find 200 from other classes
        # Strategy: More aggressive search to guarantee finding other-class neighbors
        k_search_desired = min(self.k * self.num_classes * 2 + 100, len(self.y_train))
        
        # Check if k_search exceeds FAISS GPU limit and fallback to CPU if needed
        if k_search_desired > self.MAX_GPU_K and self.use_gpu:
            logger.warning(
                f"k_search={k_search_desired} exceeds GPU limit of {self.MAX_GPU_K}, "
                f"falling back to CPU for global search"
            )
            # Create temporary CPU index for global search
            global_index_cpu = faiss.IndexFlatL2(X_test.shape[1])
            global_index_cpu.add(np.ascontiguousarray(self.X_train.astype('float32')))
            k_search = k_search_desired
            global_distances_sq, global_indices = global_index_cpu.search(X_test, k_search)
        else:
            k_search = min(k_search_desired, self.MAX_GPU_K, len(self.y_train))
            # Pre-search global index for all test samples
            global_distances_sq, global_indices = self.global_index.search(X_test, k_search)
        
        # Group by predicted class for efficient batch processing
        for label in range(self.num_classes):
            mask = (predictions == label)
            sample_indices = np.where(mask)[0]
            
            if len(sample_indices) == 0:
                continue
            
            batch_X = X_test[sample_indices]
            
            # Same-class distance: find k nearest same-class neighbors
            if label in self.same_indices:
                same_size = self.same_index_sizes[label]
                search_k = min(self.k, same_size)  # Don't search more than available
                d_same_sq, _ = self.same_indices[label].search(batch_X, search_k)
                # Take mean of all k neighbors (no self-exclusion needed - query is not in train set)
                d_same = np.sqrt(d_same_sq[:, :search_k]).mean(axis=1)
            else:
                d_same = np.full(len(sample_indices), np.inf)
            
            # Other-class distance: find k nearest other-class neighbors
            d_other = np.full(len(sample_indices), np.inf)
            for i, sample_idx in enumerate(sample_indices):
                # Find k neighbors from different class
                other_distances = []
                for j in range(k_search):
                    neighbor_idx = global_indices[sample_idx, j]
                    if self.y_train[neighbor_idx] != label:
                        other_distances.append(np.sqrt(global_distances_sq[sample_idx, j]))
                        if len(other_distances) >= self.k:
                            break
                
                if len(other_distances) >= self.k:
                    d_other[i] = np.mean(other_distances[:self.k])
                elif len(other_distances) > 0:
                    # If we couldn't find k neighbors, use what we have
                    d_other[i] = np.mean(other_distances)
            
            # Separation score: (mean_dist_to_k_other - mean_dist_to_k_same) / 2
            raw_separation = (d_other - d_same) / 2
            # Handle edge cases: replace inf with large values, NaN with 0
            raw_separation = np.where(np.isinf(raw_separation), np.sign(raw_separation) * 1e6, raw_separation)
            raw_separation = np.where(np.isnan(raw_separation), 0.0, raw_separation)
            separation_scores[sample_indices] = raw_separation
        
        return separation_scores


class KNeighborSGCCalibrator:
    """
    SGC calibrator using k-smoothed separation scores.
    """
    
    def __init__(self, k=1, use_gpu=True):
        self.k = k
        self.separation_calc = KNeighborSeparationCalculator(k=k, use_gpu=use_gpu)
        self.isotonic = IsotonicRegression(out_of_bounds='clip')
        self.val_scores_sorted = None
        self.is_fitted = False
    
    def fit(self, X_train_features, y_train, X_val_features, y_val, val_predictions):
        """
        Fit the k-neighbor SGC calibrator.
        
        Args:
            X_train_features: Training features (N_train, D)
            y_train: Training labels
            X_val_features: Validation features (N_val, D)
            y_val: Validation labels
            val_predictions: Model predictions on validation set (N_val,)
        """
        # Build FAISS indices
        self.separation_calc.fit(X_train_features, y_train)
        
        # Calculate separation on validation set
        separation = self.separation_calc.calculate_scores(X_val_features, val_predictions)
        
        # DEBUG: Log separation statistics
        logger.info(f"  Separation stats: min={np.min(separation):.4f}, max={np.max(separation):.4f}, "
                    f"mean={np.mean(separation):.4f}, std={np.std(separation):.4f}")
        logger.info(f"  NaN count: {np.sum(np.isnan(separation))}, Inf count: {np.sum(np.isinf(separation))}")
        
        # Handle any remaining NaN/inf (shouldn't happen after fix 1, but defensive)
        separation = np.nan_to_num(separation, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # =====================================================================
        # 📏 STABILITY NORMALIZATION (Rank-based)
        # =====================================================================
        
        logger.info("📏 Storing sorted validation scores for rank-based normalization...")
        
        # Store sorted validation scores for rank-based normalization
        # Isotonic regression only cares about ordering, not absolute values
        self.val_scores_sorted = np.sort(separation)
        n_val_samples = len(self.val_scores_sorted)
        
        logger.info(f"   📊 Stored {n_val_samples} sorted validation scores")
        logger.info(f"   📊 Score range: [{np.min(separation):.6f}, {np.max(separation):.6f}]")
        logger.info(f"   📊 Mean score: {np.mean(separation):.6f}")
        
        # For fitting, we still need normalized values, so normalize using ranks
        # This preserves the ordering information that isotonic regression needs
        separation_norm = rankdata(separation, method='average') / n_val_samples
        
        logger.info(f"   ✅ Applied rank-based normalization")
        logger.info(f"   📊 Normalized range: [{np.min(separation_norm):.6f}, {np.max(separation_norm):.6f}]")
        
        # Fit isotonic regression
        correctness = (val_predictions == y_val).astype(float)
        self.isotonic.fit(separation_norm, correctness)
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
        
        # Calculate separation
        separation = self.separation_calc.calculate_scores(X_test_features, predictions)
        
        # Handle NaN/inf
        separation = np.nan_to_num(separation, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # =====================================================================
        # 📏 RANK-BASED NORMALIZATION (using stored validation scores)
        # =====================================================================
        
        # Map test scores to ranks based on where they fall in sorted validation scores
        # Use searchsorted to find insertion points, then normalize
        n_val_samples = len(self.val_scores_sorted)
        separation_norm = np.searchsorted(self.val_scores_sorted, separation, side='right') / n_val_samples
        # Clamp to [0, 1] range
        separation_norm = np.clip(separation_norm, 0.0, 1.0)
        
        # Get calibrated confidence
        calibrated_conf = self.isotonic.predict(separation_norm)
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


class TrustScoreCalculator:
    """
    Computes Trust Score using FAISS:
        trust_score = d_non / (d_friend + epsilon)
    where d_friend is distance to nearest same-class point and d_non to nearest other-class point.
    """
    
    def __init__(self, use_gpu: bool = True, epsilon: float = 1e-8):
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.epsilon = epsilon
        self.same_indices = {}
        self.other_indices = {}
        self.y_train = None
        self.num_classes = None
        self.gpu_resources = faiss.StandardGpuResources() if self.use_gpu else None
    
    def _create_index(self, features: np.ndarray):
        d = features.shape[1]
        index = faiss.IndexFlatL2(d)
        if self.use_gpu and self.gpu_resources is not None:
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
        index.add(np.ascontiguousarray(features.astype("float32")))
        return index
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Build per-class FAISS indices for nearest-neighbor queries.
        """
        X_train = np.ascontiguousarray(X_train.astype("float32"))
        self.y_train = y_train.copy()
        self.num_classes = len(np.unique(y_train))
        
        logger.info(f"Building Trust Score FAISS indices for {len(X_train)} samples, {self.num_classes} classes...")
        
        for label in range(self.num_classes):
            idx_same = np.where(y_train == label)[0]
            idx_other = np.where(y_train != label)[0]
            
            if len(idx_same) == 0 or len(idx_other) == 0:
                logger.warning(f"  Label {label}: insufficient samples for Trust Score (same={len(idx_same)}, other={len(idx_other)})")
                continue
            
            self.same_indices[label] = self._create_index(X_train[idx_same])
            self.other_indices[label] = self._create_index(X_train[idx_other])
    
    def calculate_scores(self, X_test: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Compute Trust Score for each test sample given predicted labels.
        """
        X_test = np.ascontiguousarray(X_test.astype("float32"))
        n_samples = len(X_test)
        trust_scores = np.zeros(n_samples, dtype=np.float32)
        
        for label in range(self.num_classes):
            mask = (predictions == label)
            sample_indices = np.where(mask)[0]
            if len(sample_indices) == 0:
                continue
            
            if label not in self.same_indices or label not in self.other_indices:
                logger.warning(f"  Skipping Trust Score for label {label} (missing indices)")
                trust_scores[sample_indices] = 0.0
                continue
            
            batch_X = X_test[sample_indices]
            
            # Nearest same-class neighbor
            d_friend_sq, _ = self.same_indices[label].search(batch_X, 1)
            d_friend = np.sqrt(d_friend_sq[:, 0] + self.epsilon)
            
            # Nearest other-class neighbor
            d_non_sq, _ = self.other_indices[label].search(batch_X, 1)
            d_non = np.sqrt(d_non_sq[:, 0] + self.epsilon)
            
            ts = d_non / (d_friend + self.epsilon)
            # Clean up numerical issues
            ts = np.where(np.isinf(ts), 1e6, ts)
            ts = np.where(np.isnan(ts), 0.0, ts)
            
            trust_scores[sample_indices] = ts
        
        return trust_scores


class TrustScoreCalibrator:
    """
    Calibrator using Trust Score computed via FAISS nearest neighbors.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.trust_calc = TrustScoreCalculator(use_gpu=use_gpu)
        self.isotonic = IsotonicRegression(out_of_bounds="clip")
        self.val_scores_sorted = None
        self.is_fitted = False
    
    def fit(self, X_train_features, y_train, X_val_features, y_val, val_predictions):
        """
        Fit trust-score-based calibrator on validation data.
        """
        self.trust_calc.fit(X_train_features, y_train)
        ts = self.trust_calc.calculate_scores(X_val_features, val_predictions)
        
        logger.info(
            f"  Trust Score stats: min={np.min(ts):.4f}, max={np.max(ts):.4f}, "
            f"mean={np.mean(ts):.4f}, std={np.std(ts):.4f}"
        )
        logger.info(f"  NaN count: {np.sum(np.isnan(ts))}, Inf count: {np.sum(np.isinf(ts))}")
        
        ts = np.nan_to_num(ts, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # =====================================================================
        # 📏 STABILITY NORMALIZATION (Rank-based)
        # =====================================================================
        
        logger.info("📏 Storing sorted validation scores for rank-based normalization...")
        
        # Store sorted validation scores for rank-based normalization
        # Isotonic regression only cares about ordering, not absolute values
        self.val_scores_sorted = np.sort(ts)
        n_val_samples = len(self.val_scores_sorted)
        
        logger.info(f"   📊 Stored {n_val_samples} sorted validation scores")
        logger.info(f"   📊 Score range: [{np.min(ts):.6f}, {np.max(ts):.6f}]")
        logger.info(f"   📊 Mean score: {np.mean(ts):.6f}")
        
        # For fitting, we still need normalized values, so normalize using ranks
        # This preserves the ordering information that isotonic regression needs
        ts_norm = rankdata(ts, method='average') / n_val_samples
        
        logger.info(f"   ✅ Applied rank-based normalization")
        logger.info(f"   📊 Normalized range: [{np.min(ts_norm):.6f}, {np.max(ts_norm):.6f}]")
        
        correctness = (val_predictions == y_val).astype(float)
        self.isotonic.fit(ts_norm, correctness)
        self.is_fitted = True
    
    def calibrate(self, X_test_features, predictions, logits):
        """
        Calibrate test predictions using Trust Score.
        """
        if not self.is_fitted:
            raise ValueError("TrustScoreCalibrator not fitted")
        
        ts = self.trust_calc.calculate_scores(X_test_features, predictions)
        ts = np.nan_to_num(ts, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # =====================================================================
        # 📏 RANK-BASED NORMALIZATION (using stored validation scores)
        # =====================================================================
        
        # Map test scores to ranks based on where they fall in sorted validation scores
        # Use searchsorted to find insertion points, then normalize
        n_val_samples = len(self.val_scores_sorted)
        ts_norm = np.searchsorted(self.val_scores_sorted, ts, side='right') / n_val_samples
        # Clamp to [0, 1] range
        ts_norm = np.clip(ts_norm, 0.0, 1.0)
        
        calibrated_conf = self.isotonic.predict(ts_norm)
        calibrated_conf = np.clip(calibrated_conf, 1e-6, 1 - 1e-6)
        
        n_samples, n_classes = logits.shape
        calibrated_probs = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            c = predictions[i]
            remaining = (1.0 - calibrated_conf[i]) / (n_classes - 1)
            calibrated_probs[i, :] = remaining
            calibrated_probs[i, c] = calibrated_conf[i]
        
        return calibrated_probs


def select_layers(
    model: torch.nn.Module,
    model_name: str,
    mode: str,
    seed: int = 42,
    device: torch.device = None,
    dataset_name: str = "cifar10"
) -> List[str]:
    """
    Select layers based on the specified mode.
    
    Args:
        model: PyTorch model
        model_name: Model name (e.g., 'resnet18')
        mode: One of 'random_6', 'random_1', 'dac_layers', 'last_layer'
        seed: Random seed for random selection
        device: torch device
        dataset_name: Dataset name (for input shape determination)
    
    Returns:
        List of layer names
    """
    if mode == 'dac_layers':
        return get_dac_target_layers(model_name, model)
    
    # Determine input shape
    is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
    if is_dinov2:
        input_shape = (1, 3, 224, 224)
    elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
        input_shape = (1, 3, 64, 64)
    else:
        input_shape = (1, 3, 32, 32)
    
    # Discover all layers
    discovered_layers = normalize_discovered_layers(
        model, discover_model_layers(model, device=device, input_shape=input_shape)
    )
    filtered_layers, _ = filter_non_feature_layers(discovered_layers, model)
    all_layer_names = [d["name"] for d in filtered_layers]
    
    if mode == 'last_layer':
        # Return only the last layer
        return [all_layer_names[-1]]
    elif mode == 'random_1':
        np.random.seed(seed)
        return [np.random.choice(all_layer_names)]
    elif mode == 'random_6':
        np.random.seed(seed)
        num_layers = min(6, len(all_layer_names))
        return list(np.random.choice(all_layer_names, size=num_layers, replace=False))
    else:
        raise ValueError(f"Unknown layer selection mode: {mode}")


def extract_features_for_mode(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    layer_names: List[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    feature_mode: str,
    target_dim: int = 256,
    seed: int = 42,
    ood_loader: Optional[DataLoader] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Extract features based on the specified feature mode.
    
    Args:
        model: PyTorch model
        model_name: Model name
        dataset_name: Dataset name
        layer_names: List of layer names to extract from
        train_loader, val_loader, test_loader: DataLoaders
        device: torch device
        feature_mode: 'sgc' or 'coordinate'
        target_dim: Target dimension (for SGC)
        seed: Random seed
        ood_loader: Optional OOD loader
    
    Returns:
        train_features, val_features, test_features, ood_features (or None), info_dict
    """
    if feature_mode == 'sgc':
        # SGC: SPP pooling → random projection → project-then-sum → L2 norm
        train_features, val_features, test_features, info = extract_and_aggregate_sgc_features(
            model=model,
            layer_names=layer_names,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            target_dim=target_dim,
            seed=seed,
            pooling_mode='max'
        )
        
        ood_features = None
        if ood_loader is not None:
            # Extract OOD features using same layers
            _, _, ood_features, _ = extract_and_aggregate_sgc_features(
                model=model,
                layer_names=layer_names,
                train_loader=None,
                val_loader=None,
                test_loader=ood_loader,
                device=device,
                target_dim=target_dim,
                seed=seed,
                pooling_mode='max'
            )
        
        return train_features, val_features, test_features, ood_features, info
    
    elif feature_mode == 'coordinate':
        # NOTE: Coordinate sampling extracts from ALL layers and samples globally
        # The layer_names parameter is ignored in this mode
        # Coordinate: Raw coordinate sampling
        # Determine input shape
        is_dinov2 = model_name is not None and 'dinov2' in model_name.lower()
        if is_dinov2:
            input_shape = (1, 3, 224, 224)
        elif dataset_name.lower() in ["tiny_imagenet", "tinyimagenet"]:
            input_shape = (1, 3, 64, 64)
        else:
            input_shape = (1, 3, 32, 32)
        
        logger.info(f"Coordinate mode: discovering global coordinate space (ignoring layer selection mode '{layer_names}' if provided)")
        
        # Discover coordinate space (initial)
        layer_map, total_size = discover_coordinate_space(
            model, input_shape=input_shape, device=str(device)
        )
        
        # Validate and correct coordinate space with a real batch, mirroring
        # run_coordinate_calibration in compare_dac_geometric.py (3836–4126)
        if train_loader is not None:
            try:
                validation_batch = next(iter(train_loader))
                if isinstance(validation_batch, (list, tuple)):
                    real_batch = validation_batch[0]
                else:
                    real_batch = validation_batch
                
                if not isinstance(real_batch, torch.Tensor):
                    real_batch = torch.from_numpy(real_batch) if isinstance(real_batch, np.ndarray) else real_batch
                real_batch = real_batch.to(device)
                
                layer_map, total_size = validate_and_correct_coordinate_space(
                    model=model,
                    layer_map=layer_map,
                    real_batch=real_batch,
                    device=device,
                )
                logger.info(f"Validated coordinate space: {total_size:,} scalars across {len(layer_map)} layers")
            except Exception as e:
                logger.warning(f"Failed to validate coordinate space, using initial discovery only: {e}")
        
        # Plan extraction
        sampling_plan, global_order = plan_coordinate_extraction(
            total_size=total_size,
            num_coordinates=target_dim,
            layer_map=layer_map,
            seed=seed,
        )
        
        # Extract features
        train_features = extract_coordinate_features(
            model, train_loader, sampling_plan, layer_map, global_order, device=str(device)
        )
        val_features = extract_coordinate_features(
            model, val_loader, sampling_plan, layer_map, global_order, device=str(device)
        )
        test_features = extract_coordinate_features(
            model, test_loader, sampling_plan, layer_map, global_order, device=str(device)
        )
        
        # L2 normalize
        train_features = normalize(train_features.numpy(), norm='l2', axis=1)
        val_features = normalize(val_features.numpy(), norm='l2', axis=1)
        test_features = normalize(test_features.numpy(), norm='l2', axis=1)
        
        ood_features = None
        if ood_loader is not None:
            ood_features = extract_coordinate_features(
                model, ood_loader, sampling_plan, layer_map, global_order, device=str(device)
            )
            ood_features = normalize(ood_features.numpy(), norm='l2', axis=1)
        
        info = {
            'feature_mode': 'coordinate',
            'num_coordinates': target_dim,
            'layer_map_size': total_size
        }
        
        return train_features, val_features, test_features, ood_features, info
    
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")


def run_k_ablation_experiment(
    model, model_name, dataset_name, model_adapter,
    train_raw, train_labels, 
    val_raw, val_labels,
    test_raw, test_labels,
    device,
    k_values=[1, 5, 10, 25, 50, 100, 200],
    layer_selection_modes=['random_6', 'dac_layers', 'last_layer', 'random_1'],
    feature_modes=['sgc', 'coordinate'],
    batch_size=128,
    seed=42,
    ood_raw=None, ood_labels=None,
    output_dir='results/k_ablation',
    include_trust_score: bool = False,
) -> Dict[str, Any]:
    """
    Run k-ablation across all combinations of:
    - k values: [1, 5, 10, 25, 50, 100, 200]
    - Layer selections: random_6, dac_layers, last_layer, random_1
    - Feature modes: sgc (SPP+JL), coordinate (raw sampling)
    
    Returns results dict with ID ECE and OOD AUROC for each configuration.
    """
    logger.info("\n" + "="*80)
    logger.info("K-NEIGHBOR ABLATION STUDY")
    logger.info("="*80)
    logger.info(f"Model: {model_name}, Dataset: {dataset_name}")
    logger.info(f"K values: {k_values}")
    logger.info(f"Layer modes: {layer_selection_modes}")
    logger.info(f"Feature modes: {feature_modes}")
    
    results = {
        "config": {
            "k_values": k_values,
            "layer_modes": layer_selection_modes,
            "feature_modes": feature_modes,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "seed": seed
        },
        "results": {},
        "summary": {}
    }
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_raw), torch.from_numpy(test_labels)),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    ood_loader = None
    if ood_raw is not None:
        ood_labels_tensor = torch.from_numpy(ood_labels) if ood_labels is not None else torch.zeros(len(ood_raw), dtype=torch.long)
        ood_loader = DataLoader(
            TensorDataset(torch.from_numpy(ood_raw), ood_labels_tensor),
            batch_size=batch_size, shuffle=False, num_workers=0
        )
    
    # Get model predictions and logits
    logger.info("Computing model predictions...")
    val_logits = model_adapter.predict_proba(val_raw)
    test_logits = model_adapter.predict_proba(test_raw)
    val_predictions = np.argmax(val_logits, axis=1)
    test_predictions = np.argmax(test_logits, axis=1)
    
    ood_logits = None
    ood_predictions = None
    if ood_raw is not None:
        ood_logits = model_adapter.predict_proba(ood_raw)
        ood_predictions = np.argmax(ood_logits, axis=1)
    
    # Iterate over all combinations
    for layer_mode in layer_selection_modes:
        for feature_mode in feature_modes:
            config_key = f"{feature_mode}_{layer_mode}"
            logger.info("\n" + "-"*80)
            logger.info(f"Configuration: {config_key}")
            logger.info("-"*80)
            
            # Select layers only for SGC feature mode.
            # For coordinate mode, we follow global coordinate sampling which does not
            # use a fixed subset of layers; the layer_mode is kept only as a label.
            if feature_mode == 'sgc':
                try:
                    layer_names = select_layers(
                        model, model_name, layer_mode, seed=seed, device=device, dataset_name=dataset_name
                    )
                    logger.info(f"Selected {len(layer_names)} layers: {layer_names}")
                except Exception as e:
                    logger.error(f"Failed to select layers for {config_key}: {e}")
                    results["results"][config_key] = {"error": str(e)}
                    continue
            else:
                layer_names = []
                logger.info(
                    f"Coordinate feature mode: using global coordinate sampling; "
                    f"layer selection mode '{layer_mode}' is ignored for feature extraction."
                )
            
            # Extract features (once per configuration, reuse across k values)
            try:
                train_features, val_features, test_features, ood_features, extraction_info = extract_features_for_mode(
                    model=model,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    layer_names=layer_names,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    feature_mode=feature_mode,
                    target_dim=256,
                    seed=seed,
                    ood_loader=ood_loader
                )
                logger.info(f"Extracted features: train={train_features.shape}, val={val_features.shape}, test={test_features.shape}")
            except Exception as e:
                logger.error(f"Failed to extract features for {config_key}: {e}")
                results["results"][config_key] = {"error": str(e)}
                continue
            
            # Initialize results for this configuration
            results["results"][config_key] = {}
            
            # Run experiments for each k value
            for k in k_values:
                logger.info(f"\n  Running k={k}...")
                try:
                    # Fit calibrator
                    calibrator = KNeighborSGCCalibrator(k=k, use_gpu=True)
                    calibrator.fit(
                        train_features, train_labels,
                        val_features, val_labels, val_predictions
                    )
                    
                    # Calibrate test set
                    calibrated_probs = calibrator.calibrate(test_features, test_predictions, test_logits)
                    
                    # Compute ID metrics
                    id_ece = calculate_ece(calibrated_probs, test_labels)
                    id_adaptive_ece = calculate_adaptive_ece(calibrated_probs, test_labels)
                    id_brier = calculate_brier_score(calibrated_probs, test_labels)
                    id_accuracy = calculate_accuracy(calibrated_probs, test_labels)
                    
                    # Compute OOD metrics if available
                    ood_metrics = {}
                    if ood_features is not None and ood_logits is not None:
                        ood_calibrated_probs = calibrator.calibrate(ood_features, ood_predictions, ood_logits)
                        ood_metrics = compute_ood_metrics(
                            calibrated_probs_id=calibrated_probs,
                            calibrated_probs_ood=ood_calibrated_probs,
                            id_labels=test_labels,
                            ood_labels=ood_labels
                        )
                    
                    # Store results
                    k_key = f"k_{k}"
                    results["results"][config_key][k_key] = {
                        "id_ece": float(id_ece),
                        "id_adaptive_ece": float(id_adaptive_ece),
                        "id_brier": float(id_brier),
                        "id_accuracy": float(id_accuracy),
                        **{k: float(v) for k, v in ood_metrics.items()}
                    }
                    
                    logger.info(f"    ID ECE: {id_ece:.4f}, OOD AUROC: {ood_metrics.get('ood_auroc', 'N/A')}")
                    
                except Exception as e:
                    logger.error(f"    Failed for k={k}: {e}")
                    traceback.print_exc()
                    results["results"][config_key][f"k_{k}"] = {"error": str(e)}
            
            # Optional Trust Score baseline using the same features (FAISS-based)
            if include_trust_score:
                logger.info("\n  Running Trust Score baseline (FAISS)...")
                try:
                    ts_calibrator = TrustScoreCalibrator(use_gpu=True)
                    ts_calibrator.fit(
                        train_features, train_labels,
                        val_features, val_labels, val_predictions
                    )
                    
                    # Calibrate test set
                    ts_calibrated_probs = ts_calibrator.calibrate(
                        test_features, test_predictions, test_logits
                    )
                    
                    ts_id_ece = calculate_ece(ts_calibrated_probs, test_labels)
                    ts_id_adaptive_ece = calculate_adaptive_ece(ts_calibrated_probs, test_labels)
                    ts_id_brier = calculate_brier_score(ts_calibrated_probs, test_labels)
                    ts_id_accuracy = calculate_accuracy(ts_calibrated_probs, test_labels)
                    
                    ts_ood_metrics = {}
                    if ood_features is not None and ood_logits is not None:
                        ts_ood_calibrated_probs = ts_calibrator.calibrate(
                            ood_features, ood_predictions, ood_logits
                        )
                        ts_ood_metrics = compute_ood_metrics(
                            calibrated_probs_id=ts_calibrated_probs,
                            calibrated_probs_ood=ts_ood_calibrated_probs,
                            id_labels=test_labels,
                            ood_labels=ood_labels
                        )
                    
                    results["results"][config_key]["trust_score"] = {
                        "id_ece": float(ts_id_ece),
                        "id_adaptive_ece": float(ts_id_adaptive_ece),
                        "id_brier": float(ts_id_brier),
                        "id_accuracy": float(ts_id_accuracy),
                        **{k: float(v) for k, v in ts_ood_metrics.items()}
                    }
                    
                    logger.info(
                        f"    Trust Score ID ECE: {ts_id_ece:.4f}, "
                        f"OOD AUROC: {ts_ood_metrics.get('ood_auroc', 'N/A')}"
                    )
                except Exception as e:
                    logger.error(f"    Trust Score baseline failed: {e}")
                    traceback.print_exc()
                    results["results"][config_key]["trust_score"] = {"error": str(e)}
            
            # Save checkpoint after each configuration (including Trust Score if run)
            output_file = os.path.join(output_dir, f"{model_name}_{dataset_name}_seed{seed}_k_ablation.json")
            save_results_incrementally(results, output_file)
            logger.info(f"Checkpoint saved to {output_dir}")
    
    # Compute summary statistics
    logger.info("\n" + "="*80)
    logger.info("COMPUTING SUMMARY STATISTICS")
    logger.info("="*80)
    
    best_id_ece = {"value": float('inf'), "config": None, "k": None}
    best_ood_auroc = {"value": 0.0, "config": None, "k": None}
    
    for config_key, config_results in results["results"].items():
        if "error" in config_results:
            continue
        for k_key, k_results in config_results.items():
            if "error" in k_results:
                continue
            if "id_ece" in k_results and k_results["id_ece"] < best_id_ece["value"]:
                best_id_ece = {
                    "value": k_results["id_ece"],
                    "config": config_key,
                    "k": k_key
                }
            if "ood_auroc" in k_results and k_results["ood_auroc"] > best_ood_auroc["value"]:
                best_ood_auroc = {
                    "value": k_results["ood_auroc"],
                    "config": config_key,
                    "k": k_key
                }
    
    # Unification evidence: compare k=50 SGC with DAC's OOD performance
    unification_evidence = {}
    if "sgc_dac_layers" in results["results"]:
        sgc_k50_key = "k_50"
        if sgc_k50_key in results["results"]["sgc_dac_layers"]:
            sgc_k50_ood_auroc = results["results"]["sgc_dac_layers"][sgc_k50_key].get("ood_auroc")
            if sgc_k50_ood_auroc is not None:
                # Compare with k=1 baseline
                sgc_k1_ood_auroc = results["results"]["sgc_dac_layers"].get("k_1", {}).get("ood_auroc")
                if sgc_k1_ood_auroc is not None:
                    unification_evidence["sgc_k50_vs_k1_ood_auroc_diff"] = float(sgc_k50_ood_auroc - sgc_k1_ood_auroc)
                    if sgc_k50_ood_auroc > sgc_k1_ood_auroc + 0.01:
                        unification_evidence["conclusion"] = "k=50 shows improved OOD detection vs k=1, supporting unification hypothesis"
                    elif abs(sgc_k50_ood_auroc - sgc_k1_ood_auroc) < 0.01:
                        unification_evidence["conclusion"] = "k=50 and k=1 show similar OOD performance"
                    else:
                        unification_evidence["conclusion"] = "k=50 shows worse OOD detection vs k=1"
    
    results["summary"] = {
        "best_id_ece": best_id_ece,
        "best_ood_auroc": best_ood_auroc,
        "unification_evidence": unification_evidence
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="K-neighbor ablation study for SGC")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., cifar10)')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g., resnet18)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--results_base_dir', type=str, required=True,
                        help='Base directory with pre-trained models')
    parser.add_argument('--training_method', type=str, default='baseline_cross_entropy',
                        help='Training method (e.g., baseline_cross_entropy, focal)')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 5, 10, 25, 50, 100, 200],
                        help='K values to test')
    parser.add_argument('--layer_modes', type=str, nargs='+', 
                        default=['random_6', 'dac_layers', 'last_layer', 'random_1'],
                        help='Layer selection modes to test (e.g., random_6 dac_layers last_layer random_1)')
    parser.add_argument('--skip_coordinate', action='store_true',
                        help='Skip coordinate sampling experiments')
    parser.add_argument('--skip_ood', action='store_true',
                        help='Skip OOD evaluation')
    parser.add_argument('--output_dir', type=str, default='results/k_ablation',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--run_trust_score', action='store_true',
                        help='Also run FAISS-based Trust Score baseline per configuration')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data - get_data_loaders returns (train_loader, val_loader, test_loader, num_classes)
    logger.info(f"Loading data for {args.dataset}...")
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset, batch_size=args.batch_size, seed=args.seed
    )
    
    # Extract raw data from loaders
    def extract_from_loader(loader):
        raw_list, label_list = [], []
        for batch in loader:
            data, labels = batch[0], batch[1]
            raw_list.append(data.numpy())
            label_list.append(labels.numpy())
        return np.concatenate(raw_list), np.concatenate(label_list)
    
    train_raw, train_labels = extract_from_loader(train_loader)
    val_raw, val_labels = extract_from_loader(val_loader)
    test_raw, test_labels = extract_from_loader(test_loader)
    
    logger.info(f"Data loaded: train={train_raw.shape}, val={val_raw.shape}, test={test_raw.shape}")
    
    # Load OOD data if not skipping
    ood_raw, ood_labels = None, None
    if not args.skip_ood:
        if args.dataset.lower() in ['cifar10', 'cifar100', 'tiny_imagenet', 'tinyimagenet']:
            try:
                from torchvision.datasets import SVHN
                from torchvision import transforms
                
                # Determine resize size based on dataset
                if args.dataset.lower() in ['tiny_imagenet', 'tinyimagenet']:
                    resize_size = (64, 64)
                else:
                    resize_size = (32, 32)
                
                # Match dataset image size
                svhn_transform = transforms.Compose([
                    transforms.Resize(resize_size),
                    transforms.ToTensor(),
                    # Use same normalization as dataset if your pipeline normalizes
                ])
                svhn_test = SVHN(root='./data', split='test', download=True, transform=svhn_transform)
                svhn_loader = DataLoader(svhn_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
                
                ood_raw, ood_labels = extract_from_loader(svhn_loader)
                logger.info(f"Loaded SVHN OOD data: {ood_raw.shape}")
            except Exception as e:
                logger.warning(f"Could not load OOD data: {e}")
                traceback.print_exc()
    
    # Load model
    logger.info(f"Loading model {args.model_name}...")
    model_path = construct_model_path(
        args.results_base_dir,
        args.training_method,
        args.dataset,
        args.model_name,
        args.seed
    )
    model = load_trained_model(model_path, args.model_name, num_classes, device, dataset=args.dataset)
    model_adapter = PyTorchModelAdapter(model, device, args.dataset)
    
    # Determine feature modes
    feature_modes = ['sgc']
    if not args.skip_coordinate:
        feature_modes.append('coordinate')
    
    # Run experiment
    results = run_k_ablation_experiment(
        model=model,
        model_name=args.model_name,
        dataset_name=args.dataset,
        model_adapter=model_adapter,
        train_raw=train_raw,
        train_labels=train_labels,
        val_raw=val_raw,
        val_labels=val_labels,
        test_raw=test_raw,
        test_labels=test_labels,
        device=device,
        k_values=args.k_values,
        layer_selection_modes=args.layer_modes,
        feature_modes=feature_modes,
        batch_size=args.batch_size,
        seed=args.seed,
        ood_raw=ood_raw,
        ood_labels=ood_labels,
        output_dir=args.output_dir,
        include_trust_score=args.run_trust_score
    )
    
    # Save results
    output_file = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset}_seed{args.seed}_k_ablation.json")
    cleaned_results = make_json_serializable(results)
    with open(output_file, 'w') as f:
        json.dump(cleaned_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Best ID ECE: {results['summary']['best_id_ece']}")
    logger.info(f"Best OOD AUROC: {results['summary']['best_ood_auroc']}")


if __name__ == '__main__':
    main()

