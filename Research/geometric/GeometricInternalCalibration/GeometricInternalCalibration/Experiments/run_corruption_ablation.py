"""
Focused Corruption Evaluation Script for Calibration Methods.
Evaluates multiple calibration methods on CIFAR-C corruptions using FAISS backend.

Methods evaluated:
- Standard baselines: uncalibrated, temperature_scaling, platt_scaling, beta_calibration, isotonic_toplabel
- DAC: density_aware_calibration, dac_with_random_layer_selection, dac_with_coordinate_features
- SGC variants: sgc_faiss, global_random_separation, global_random_trust_score
- Coordinate methods: coordinate_sampling_faiss, coordinate_spp_faiss, coordinate_spp_dac
- Hybrid: sgc_with_dac_preprocessing_separation, sgc_with_dac_layers_separation, sgc_with_dac_layers_trust_score
- Baselines: last_layer_only_baseline, geometric_physical_space
"""

import torch
import numpy as np
import os
import logging
import argparse
import json
import time
import faiss
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset
from Experiments.layer_selection import run_standard_baselines as run_standard_baselines_clean
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# FAISS-BASED STABILITY CALCULATOR
# ==============================================================================

class FastStabilityCalculator:
    """
    Efficient FAISS-based stability calculator.
    Uses single global index for memory efficiency.
    """
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.same_indices = {}
        self.global_index = None
        self.y_train = None
        self.X_train = None
        self.gpu_resources = None
        self.num_classes = None
        
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
    
    def calculate_stability(self, X_test, predictions):
        """Batch stability calculation."""
        X_test = np.ascontiguousarray(X_test.astype('float32'))
        n_samples = len(X_test)
        stability = np.zeros(n_samples)
        
        k_search = min(50, len(self.y_train) // self.num_classes)
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


class TrustScoreCalculator:
    """
    Trust Score calculator using FAISS.
    Trust Score = distance to nearest different-class / distance to nearest same-class
    """
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.same_indices = {}
        self.other_indices = {}
        self.num_classes = None
        self.gpu_resources = None
        
        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()
    
    def _create_index(self, features):
        d = features.shape[1]
        index = faiss.IndexFlatL2(d)
        if self.use_gpu and self.gpu_resources is not None:
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
        index.add(np.ascontiguousarray(features.astype('float32')))
        return index
    
    def fit(self, X_train, y_train):
        self.num_classes = len(np.unique(y_train))
        
        for label in range(self.num_classes):
            idx_same = np.where(y_train == label)[0]
            idx_other = np.where(y_train != label)[0]
            
            if len(idx_same) > 0:
                self.same_indices[label] = self._create_index(X_train[idx_same])
            if len(idx_other) > 0:
                self.other_indices[label] = self._create_index(X_train[idx_other])
    
    def calculate_trust_score(self, X_test, predictions):
        X_test = np.ascontiguousarray(X_test.astype('float32'))
        n_samples = len(X_test)
        trust_scores = np.zeros(n_samples)
        
        for label in range(self.num_classes):
            mask = (predictions == label)
            sample_indices = np.where(mask)[0]
            
            if len(sample_indices) == 0:
                continue
            
            batch_X = X_test[sample_indices]
            
            if label in self.same_indices:
                d_same_sq, _ = self.same_indices[label].search(batch_X, 1)
                d_same = np.sqrt(d_same_sq[:, 0]) + 1e-8
            else:
                d_same = np.ones(len(sample_indices)) * 1e-8
            
            if label in self.other_indices:
                d_other_sq, _ = self.other_indices[label].search(batch_X, 1)
                d_other = np.sqrt(d_other_sq[:, 0])
            else:
                d_other = np.ones(len(sample_indices)) * 1e8
            
            trust_scores[sample_indices] = d_other / d_same
        
        return trust_scores


# ==============================================================================
# CALIBRATORS WITH FAISS BACKEND
# ==============================================================================

class SGCCalibratorFAISS:
    """SGC calibrator using FAISS backend with separation score."""
    
    def __init__(self, use_gpu=True, metric='separation'):
        self.use_gpu = use_gpu
        self.metric = metric  # 'separation' or 'trust_score'
        if metric == 'separation':
            self.score_calc = FastStabilityCalculator(use_gpu=use_gpu)
        else:
            self.score_calc = TrustScoreCalculator(use_gpu=use_gpu)
        self.isotonic = IsotonicRegression(out_of_bounds='clip')
        self.val_scores_sorted = None  # Sorted validation scores for rank-based normalization
        self.is_fitted = False
    
    def fit(self, X_train_features, y_train, X_val_features, y_val, val_predictions):
        """Fit the calibrator."""
        self.score_calc.fit(X_train_features, y_train)
        
        if self.metric == 'separation':
            scores = self.score_calc.calculate_stability(X_val_features, val_predictions)
        else:
            scores = self.score_calc.calculate_trust_score(X_val_features, val_predictions)
        
        # Rank-based normalization: store sorted validation scores
        # Isotonic regression only cares about ordering, not absolute values
        self.val_scores_sorted = np.sort(scores)
        
        # Normalize using ranks: each score's rank / total_samples
        scores_norm = np.searchsorted(self.val_scores_sorted, scores) / len(self.val_scores_sorted)
        
        correctness = (val_predictions == y_val).astype(float)
        self.isotonic.fit(scores_norm, correctness)
        self.is_fitted = True
    
    def calibrate(self, X_test_features, predictions, logits):
        """Calibrate test predictions."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        if self.metric == 'separation':
            scores = self.score_calc.calculate_stability(X_test_features, predictions)
        else:
            scores = self.score_calc.calculate_trust_score(X_test_features, predictions)
        
        # Rank-based normalization: find rank of each test score in sorted validation scores
        # This preserves ordering information even when score ranges are small
        scores_norm = np.searchsorted(self.val_scores_sorted, scores) / len(self.val_scores_sorted)
        
        calibrated_conf = self.isotonic.predict(scores_norm)
        calibrated_conf = np.clip(calibrated_conf, 1e-6, 1 - 1e-6)
        
        n_samples, n_classes = logits.shape
        calibrated_probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            c = predictions[i]
            remaining = (1.0 - calibrated_conf[i]) / (n_classes - 1)
            calibrated_probs[i, :] = remaining
            calibrated_probs[i, c] = calibrated_conf[i]
        
        return calibrated_probs


class SGCCalibratorFAISSPercentile:
    """
    SGC calibrator using FAISS backend with PERCENTILE normalization.
    Used for ablation comparison against rank-based normalization.
    """
    
    def __init__(self, use_gpu=True, metric='separation'):
        self.use_gpu = use_gpu
        self.metric = metric
        if metric == 'separation':
            self.score_calc = FastStabilityCalculator(use_gpu=use_gpu)
        else:
            self.score_calc = TrustScoreCalculator(use_gpu=use_gpu)
        self.isotonic = IsotonicRegression(out_of_bounds='clip')
        self.q05 = None
        self.q95 = None
        self.is_fitted = False
        self.clipping_stats = {}  # Track clipping statistics
    
    def fit(self, X_train_features, y_train, X_val_features, y_val, val_predictions):
        """Fit the calibrator with percentile normalization."""
        self.score_calc.fit(X_train_features, y_train)
        
        if self.metric == 'separation':
            scores = self.score_calc.calculate_stability(X_val_features, val_predictions)
        else:
            scores = self.score_calc.calculate_trust_score(X_val_features, val_predictions)
        
        # Percentile normalization (5th and 95th)
        self.q05 = np.percentile(scores, 5)
        self.q95 = np.percentile(scores, 95)
        
        score_range = self.q95 - self.q05
        if score_range > 0:
            scores_norm = np.clip((scores - self.q05) / score_range, 0, 1)
        else:
            scores_norm = np.ones_like(scores) * 0.5
        
        # Track validation clipping
        n_clipped_low = np.sum(scores < self.q05)
        n_clipped_high = np.sum(scores > self.q95)
        self.clipping_stats['val_clipped_low'] = int(n_clipped_low)
        self.clipping_stats['val_clipped_high'] = int(n_clipped_high)
        self.clipping_stats['val_clipped_pct'] = float((n_clipped_low + n_clipped_high) / len(scores) * 100)
        
        correctness = (val_predictions == y_val).astype(float)
        self.isotonic.fit(scores_norm, correctness)
        self.is_fitted = True
    
    def calibrate(self, X_test_features, predictions, logits, track_clipping=False):
        """Calibrate test predictions with percentile normalization."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        if self.metric == 'separation':
            scores = self.score_calc.calculate_stability(X_test_features, predictions)
        else:
            scores = self.score_calc.calculate_trust_score(X_test_features, predictions)
        
        # Percentile normalization with clipping
        score_range = self.q95 - self.q05
        if score_range > 0:
            scores_norm = np.clip((scores - self.q05) / score_range, 0, 1)
        else:
            scores_norm = np.ones_like(scores) * 0.5
        
        # Track test clipping if requested
        if track_clipping:
            n_clipped_low = np.sum(scores < self.q05)
            n_clipped_high = np.sum(scores > self.q95)
            clipped_pct = (n_clipped_low + n_clipped_high) / len(scores) * 100
        
        calibrated_conf = self.isotonic.predict(scores_norm)
        calibrated_conf = np.clip(calibrated_conf, 1e-6, 1 - 1e-6)
        
        n_samples, n_classes = logits.shape
        calibrated_probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            c = predictions[i]
            remaining = (1.0 - calibrated_conf[i]) / (n_classes - 1)
            calibrated_probs[i, :] = remaining
            calibrated_probs[i, c] = calibrated_conf[i]
        
        if track_clipping:
            return calibrated_probs, clipped_pct
        return calibrated_probs


class AccuracyCalibrator:
    """
    Trivial calibrator that assigns validation accuracy as confidence for all predictions.
    This serves as a baseline showing ECE when confidence = overall accuracy.
    """
    def __init__(self):
        self.val_accuracy = None
        self.is_fitted = False
    
    def fit(self, val_predictions, val_labels):
        """Compute and store validation accuracy."""
        self.val_accuracy = np.mean(val_predictions == val_labels)
        self.is_fitted = True
        return self.val_accuracy
    
    def calibrate(self, predictions, n_classes):
        """Return probabilities where predicted class gets val_accuracy confidence."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        n_samples = len(predictions)
        calibrated_probs = np.zeros((n_samples, n_classes))
        remaining = (1.0 - self.val_accuracy) / (n_classes - 1)
        
        for i in range(n_samples):
            calibrated_probs[i, :] = remaining
            calibrated_probs[i, predictions[i]] = self.val_accuracy
        
        return calibrated_probs


class DACCalibratorFAISS:
    """
    DAC-style calibrator using FAISS backend.
    Uses depth-grouped layer features + logits.
    """
    
    def __init__(self, use_gpu=True, k=10):
        self.use_gpu = use_gpu
        self.k = k
        self.knn_indices = {}
        self.isotonic = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
        self.gpu_resources = None
        self.train_features = None
        self.train_labels = None
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.gpu_resources = faiss.StandardGpuResources()
    
    def _create_index(self, features):
        d = features.shape[1]
        index = faiss.IndexFlatL2(d)
        if self.use_gpu and self.gpu_resources is not None:
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
        index.add(np.ascontiguousarray(features.astype('float32')))
        return index
    
    def fit(self, X_train_features, y_train, X_val_features, y_val, val_logits):
        """Fit DAC calibrator."""
        self.train_features = X_train_features.copy()
        self.train_labels = y_train.copy()
        
        # Build single index (DAC uses k-NN density)
        self.index = self._create_index(X_train_features)
        
        # Compute density scores on validation
        val_features = np.ascontiguousarray(X_val_features.astype('float32'))
        distances, indices = self.index.search(val_features, self.k)
        
        # DAC uses neighbor label agreement as density proxy
        val_predictions = np.argmax(val_logits, axis=1)
        density_scores = np.zeros(len(val_features))
        
        for i in range(len(val_features)):
            neighbor_labels = self.train_labels[indices[i]]
            pred_label = val_predictions[i]
            density_scores[i] = np.mean(neighbor_labels == pred_label)
        
        # Fit isotonic regression
        correctness = (val_predictions == y_val).astype(float)
        self.isotonic.fit(density_scores, correctness)
        self.is_fitted = True
    
    def calibrate(self, X_test_features, test_logits):
        """Calibrate using DAC approach."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        test_features = np.ascontiguousarray(X_test_features.astype('float32'))
        distances, indices = self.index.search(test_features, self.k)
        
        test_predictions = np.argmax(test_logits, axis=1)
        density_scores = np.zeros(len(test_features))
        
        for i in range(len(test_features)):
            neighbor_labels = self.train_labels[indices[i]]
            pred_label = test_predictions[i]
            density_scores[i] = np.mean(neighbor_labels == pred_label)
        
        calibrated_conf = self.isotonic.predict(density_scores)
        calibrated_conf = np.clip(calibrated_conf, 1e-6, 1 - 1e-6)
        
        n_samples, n_classes = test_logits.shape
        calibrated_probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            c = test_predictions[i]
            remaining = (1.0 - calibrated_conf[i]) / (n_classes - 1)
            calibrated_probs[i, :] = remaining
            calibrated_probs[i, c] = calibrated_conf[i]
        
        return calibrated_probs


# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================

def extract_features_from_layer(model, data_loader, layer_name, device, pooling='adaptive_avg'):
    """Extract features from a specific layer."""
    model.eval()
    features = []
    hook_output = []
    
    def hook_fn(module, input, output):
        hook_output.append(output.detach())
    
    # Find and register hook
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
    
    if target_module is None:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    handle = target_module.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                hook_output.clear()
                _ = model(x)
                
                feat = hook_output[0]
                
                # Pool spatial dimensions if needed
                if len(feat.shape) == 4:  # Conv feature map
                    if pooling == 'adaptive_avg':
                        feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
                    elif pooling == 'max':
                        feat = torch.nn.functional.adaptive_max_pool2d(feat, 1).flatten(1)
                    else:
                        feat = feat.flatten(1)
                
                features.append(feat.cpu().numpy())
    finally:
        handle.remove()
    
    return np.concatenate(features, axis=0)


def extract_multi_layer_features(model, data_loader, layer_names, device, pooling='adaptive_avg', 
                                  target_dim=256, seed=42):
    """Extract and concatenate features from multiple layers with random projection."""
    all_features = []
    
    for layer_name in layer_names:
        try:
            feat = extract_features_from_layer(model, data_loader, layer_name, device, pooling)
            all_features.append(feat)
        except Exception as e:
            logger.warning(f"Failed to extract from {layer_name}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No features extracted from any layer")
    
    # Concatenate all features
    combined = np.concatenate(all_features, axis=1)
    
    # Random projection to target dimension
    if combined.shape[1] > target_dim:
        np.random.seed(seed)
        proj_matrix = np.random.randn(combined.shape[1], target_dim) / np.sqrt(target_dim)
        combined = combined @ proj_matrix
    
    return combined.astype(np.float32)


def discover_feature_layers(model, input_shape=(1, 3, 32, 32)):
    """Discover all feature layers in a model."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape).to(device)
    
    layers = []
    layer_shapes = {}
    
    def hook_fn(name):
        def fn(module, input, output):
            if isinstance(output, torch.Tensor):
                layer_shapes[name] = output.shape
        return fn
    
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
            handles.append(module.register_forward_hook(hook_fn(name)))
            layers.append(name)
    
    with torch.no_grad():
        model(dummy_input)
    
    for h in handles:
        h.remove()
    
    # Filter to keep only feature-rich layers
    feature_layers = []
    for name in layers:
        shape = layer_shapes.get(name)
        if shape is not None and len(shape) >= 2:
            feat_dim = np.prod(shape[1:])
            if feat_dim >= 64:  # At least 64-dim features
                feature_layers.append(name)
    
    return feature_layers


def select_random_layers(all_layers, n_layers=6, seed=42):
    """Randomly select n_layers from all available layers."""
    np.random.seed(seed)
    if len(all_layers) <= n_layers:
        return all_layers
    return list(np.random.choice(all_layers, size=n_layers, replace=False))


# ==============================================================================
# CALIBRATION METRICS
# ==============================================================================

def calculate_ece(probs, labels, n_bins=15):
    """Calculate Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels).astype(float)
    
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
    
    return ece


# ==============================================================================
# CIFAR-C EVALUATION
# ==============================================================================

CIFAR_C_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]


def load_cifar_c_data(corruption_name, severity, cifar_c_dir, dataset='cifar10'):
    """Load CIFAR-C corruption data."""
    if dataset == 'cifar10':
        data_path = os.path.join(cifar_c_dir, f"{corruption_name}.npy")
        label_path = os.path.join(cifar_c_dir, "labels.npy")
    else:
        data_path = os.path.join(cifar_c_dir, f"{corruption_name}.npy")
        label_path = os.path.join(cifar_c_dir, "labels.npy")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Corruption file not found: {data_path}")
    
    # Load data - shape is (50000, 32, 32, 3) for 5 severities x 10000 images
    data = np.load(data_path)
    labels = np.load(label_path)
    
    # Extract specific severity (1-indexed: severity 1 = indices 0-9999)
    start_idx = (severity - 1) * 10000
    end_idx = severity * 10000
    
    images = data[start_idx:end_idx]
    y = labels[start_idx:end_idx]
    
    return images, y


def normalize_cifar_images(images, dataset='cifar10'):
    """Normalize CIFAR images to match training preprocessing."""
    # Images are uint8 [0, 255], shape (N, H, W, C)
    images_float = images.astype(np.float32) / 255.0
    
    # Transpose to (N, C, H, W) for PyTorch
    images_float = np.transpose(images_float, (0, 3, 1, 2))
    
    # Apply normalization (CIFAR-10 stats)
    if dataset == 'cifar10':
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
    else:  # cifar100
        mean = np.array([0.5071, 0.4867, 0.4408]).reshape(1, 3, 1, 1)
        std = np.array([0.2675, 0.2565, 0.2761]).reshape(1, 3, 1, 1)
    
    images_normalized = (images_float - mean) / std
    return images_normalized.astype(np.float32)


def precompute_corruption_predictions(model_adapter, cifar_c_dir, batch_size, dataset='cifar10',
                                     corruptions=None, severities=None):
    """
    Pre-compute model predictions (probs, logits) for all corruptions and severities.
    This avoids recomputing the same predictions for each calibration method.
    
    Returns:
        Dict mapping (corruption, severity) -> (images_normalized, probs, logits, labels)
    """
    if corruptions is None:
        corruptions = CIFAR_C_CORRUPTIONS
    if severities is None:
        severities = [1, 2, 3, 4, 5]
    
    cache = {}
    logger.info(f"Pre-computing model predictions for {len(corruptions)} corruptions × {len(severities)} severities...")
    
    for corruption in tqdm(corruptions, desc="Pre-computing predictions"):
        for severity in severities:
            try:
                # Load and normalize images
                images, labels = load_cifar_c_data(corruption, severity, cifar_c_dir, dataset)
                images_normalized = normalize_cifar_images(images, dataset)
                
                # Compute model predictions
                probs = model_adapter.predict_proba(images_normalized, batch_size=batch_size)
                logits = model_adapter.predict_logits(images_normalized, batch_size=batch_size)
                
                cache[(corruption, severity)] = (images_normalized, probs, logits, labels)
            except Exception as e:
                logger.warning(f"Failed to pre-compute predictions for {corruption} severity {severity}: {e}")
                continue
    
    logger.info(f"Pre-computed predictions for {len(cache)} corruption×severity combinations")
    return cache


def evaluate_on_corruption(predict_fn, corruption_name, severity, cifar_c_dir, 
                           test_transform, batch_size, dataset='cifar10',
                           cached_predictions=None):
    """
    Evaluate a calibration method on a single corruption.
    
    Args:
        predict_fn: Function that takes (images_normalized, corruption, severity) and returns calibrated probs
        cached_predictions: Optional dict mapping (corruption, severity) -> (images, probs, logits, labels)
    """
    # Use cached predictions if available
    if cached_predictions and (corruption_name, severity) in cached_predictions:
        images_normalized, probs, logits, labels = cached_predictions[(corruption_name, severity)]
        # Get calibrated predictions (pass corruption/severity for feature caching)
        calibrated_probs = predict_fn(images_normalized, corruption_name, severity)
    else:
        # Load corruption data (fallback if cache not available)
        images, labels = load_cifar_c_data(corruption_name, severity, cifar_c_dir, dataset)
        images_normalized = normalize_cifar_images(images, dataset)
        # Get calibrated predictions
        calibrated_probs = predict_fn(images_normalized, corruption_name, severity)
        # Calculate labels if not cached
        if cached_predictions is None:
            labels = load_cifar_c_data(corruption_name, severity, cifar_c_dir, dataset)[1]
    
    # Calculate ECE
    ece = calculate_ece(calibrated_probs, labels)
    
    return ece


def compute_mce(predict_fn, cifar_c_dir, test_transform, batch_size, dataset='cifar10',
                corruptions=None, severities=None, cached_predictions=None):
    """
    Compute mean Corruption Error (mCE) across all corruptions and severities.
    
    Args:
        predict_fn: Function that takes (images_normalized, corruption, severity) and returns calibrated probs
        cached_predictions: Optional dict mapping (corruption, severity) -> (images, probs, logits, labels)
    """
    if corruptions is None:
        corruptions = CIFAR_C_CORRUPTIONS
    if severities is None:
        severities = [1, 2, 3, 4, 5]
    
    all_eces = []
    corruption_eces = {}
    
    for corruption in corruptions:
        severity_eces = []
        for severity in severities:
            try:
                ece = evaluate_on_corruption(
                    predict_fn, corruption, severity, cifar_c_dir,
                    test_transform, batch_size, dataset,
                    cached_predictions=cached_predictions
                )
                severity_eces.append(ece)
            except Exception as e:
                logger.warning(f"Failed on {corruption} severity {severity}: {e}")
                continue
        
        if severity_eces:
            mean_ece = np.mean(severity_eces)
            corruption_eces[corruption] = mean_ece
            all_eces.append(mean_ece)
    
    mce = np.mean(all_eces) if all_eces else None
    return mce, corruption_eces


# ==============================================================================
# MAIN EVALUATION FUNCTIONS
# ==============================================================================

def evaluate_normalization_comparison(model, model_adapter, train_raw, train_labels,
                                       val_raw, val_labels, cifar_c_dir, test_transform,
                                       batch_size, device, dataset='cifar10',
                                       target_dim=256, n_random_layers=6, seed=42,
                                       test_raw=None, test_labels=None, evaluate_on_clean=False):
    """
    Compare percentile vs rank-based normalization for SGC calibration.
    This ablation demonstrates why rank-based normalization is preferred for OOD robustness.
    
    Args:
        test_raw: Optional clean test data (numpy array). Required if evaluate_on_clean=True.
        test_labels: Optional clean test labels. Required if evaluate_on_clean=True.
        evaluate_on_clean: If True, evaluate on clean test set instead of corruptions.
    """
    results = {}
    
    logger.info("\n" + "="*80)
    if evaluate_on_clean:
        logger.info("NORMALIZATION METHOD COMPARISON (Percentile vs Rank-based) - CLEAN TEST SET")
    else:
        logger.info("NORMALIZATION METHOD COMPARISON (Percentile vs Rank-based)")
    logger.info("="*80)
    
    # Pre-compute model predictions (only if evaluating on corruptions)
    cached_predictions = None
    if not evaluate_on_clean:
        cached_predictions = precompute_corruption_predictions(
            model_adapter, cifar_c_dir, batch_size, dataset
        )
    
    # Discover and select layers
    all_layers = discover_feature_layers(model)
    random_layers = select_random_layers(all_layers, n_layers=n_random_layers, seed=seed)
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_raw).float()),
        batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_raw).float()),
        batch_size=batch_size, shuffle=False
    )
    
    # Extract features (same for both methods)
    train_features = extract_multi_layer_features(
        model, train_loader, random_layers, device, target_dim=target_dim, seed=seed
    )
    val_features = extract_multi_layer_features(
        model, val_loader, random_layers, device, target_dim=target_dim, seed=seed
    )
    
    val_probs = model_adapter.predict_proba(val_raw, batch_size=batch_size)
    val_predictions = np.argmax(val_probs, axis=1)
    
    # Feature cache for test data
    feature_cache = {}
    
    # =========================================================================
    # 1. Rank-based normalization (current method)
    # =========================================================================
    logger.info("\n--- Rank-based Normalization ---")
    sgc_rank = SGCCalibratorFAISS(use_gpu=(device.type == 'cuda'), metric='separation')
    sgc_rank.fit(train_features, train_labels, val_features, val_labels, val_predictions)
    
    # Track clipping stats for rank-based
    rank_clipping_stats = {'per_corruption': {}}
    
    def rank_predict_fn(Xc, corruption=None, severity=None):
        cache_key = (corruption, severity, tuple(random_layers))
        if cache_key and cache_key in feature_cache:
            test_features = feature_cache[cache_key]
        else:
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(Xc).float()),
                batch_size=batch_size, shuffle=False
            )
            test_features = extract_multi_layer_features(
                model, test_loader, random_layers, device, target_dim=target_dim, seed=seed
            )
            if cache_key:
                feature_cache[cache_key] = test_features
        
        if cached_predictions and corruption and severity and (corruption, severity) in cached_predictions:
            _, test_probs, _, _ = cached_predictions[(corruption, severity)]
        else:
            test_probs = model_adapter.predict_proba(Xc, batch_size=batch_size)
        test_predictions = np.argmax(test_probs, axis=1)
        
        return sgc_rank.calibrate(test_features, test_predictions, test_probs)
    
    if evaluate_on_clean:
        # Evaluate on clean test set
        if test_raw is None or test_labels is None:
            raise ValueError("test_raw and test_labels required when evaluate_on_clean=True")
        
        logger.info("Evaluating on clean test set...")
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_raw).float()),
            batch_size=batch_size, shuffle=False
        )
        test_features = extract_multi_layer_features(
            model, test_loader, random_layers, device, target_dim=target_dim, seed=seed
        )
        test_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
        test_predictions = np.argmax(test_probs, axis=1)
        
        calibrated_probs_rank = sgc_rank.calibrate(test_features, test_predictions, test_probs)
        clean_ece_rank = calculate_ece(calibrated_probs_rank, test_labels)
        
        results['rank_based'] = {
            'clean_ece': clean_ece_rank,
            'selected_layers': random_layers,
            'normalization': 'rank_based (empirical CDF)',
            'evaluation_mode': 'clean_test_set'
        }
        logger.info(f"  Clean test ECE = {clean_ece_rank:.4f}")
    else:
        # Evaluate on corruptions (original behavior)
        mce_rank, corruption_eces_rank = compute_mce(
            rank_predict_fn, cifar_c_dir, test_transform, batch_size, dataset,
            cached_predictions=cached_predictions
        )
        
        results['rank_based'] = {
            'mce': mce_rank,
            'corruption_eces': corruption_eces_rank,
            'selected_layers': random_layers,
            'normalization': 'rank_based (empirical CDF)',
            'evaluation_mode': 'corruptions'
        }
        logger.info(f"  mCE = {mce_rank:.4f}")
    
    # =========================================================================
    # 2. Percentile normalization (ablation)
    # =========================================================================
    logger.info("\n--- Percentile Normalization (5th/95th) ---")
    sgc_perc = SGCCalibratorFAISSPercentile(use_gpu=(device.type == 'cuda'), metric='separation')
    sgc_perc.fit(train_features, train_labels, val_features, val_labels, val_predictions)
    
    logger.info(f"  Validation clipping: {sgc_perc.clipping_stats['val_clipped_pct']:.1f}%")
    logger.info(f"  Percentile bounds: q05={sgc_perc.q05:.4f}, q95={sgc_perc.q95:.4f}")
    
    # Track per-corruption clipping
    perc_clipping_stats = {'per_corruption': {}, 'total_clipped': 0, 'total_samples': 0}
    
    def perc_predict_fn(Xc, corruption=None, severity=None):
        cache_key = (corruption, severity, tuple(random_layers))
        if cache_key and cache_key in feature_cache:
            test_features = feature_cache[cache_key]
        else:
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(Xc).float()),
                batch_size=batch_size, shuffle=False
            )
            test_features = extract_multi_layer_features(
                model, test_loader, random_layers, device, target_dim=target_dim, seed=seed
            )
            if cache_key:
                feature_cache[cache_key] = test_features
        
        if cached_predictions and corruption and severity and (corruption, severity) in cached_predictions:
            _, test_probs, _, _ = cached_predictions[(corruption, severity)]
        else:
            test_probs = model_adapter.predict_proba(Xc, batch_size=batch_size)
        test_predictions = np.argmax(test_probs, axis=1)
        test_logits = model_adapter.predict_logits(Xc, batch_size=batch_size)
        
        # Track clipping for this corruption
        if corruption:
            calibrated_probs, clipped_pct = sgc_perc.calibrate(
                test_features, test_predictions, test_logits, track_clipping=True
            )
            if corruption not in perc_clipping_stats['per_corruption']:
                perc_clipping_stats['per_corruption'][corruption] = []
            perc_clipping_stats['per_corruption'][corruption].append(clipped_pct)
            perc_clipping_stats['total_clipped'] += int(clipped_pct * len(Xc) / 100)
            perc_clipping_stats['total_samples'] += len(Xc)
            return calibrated_probs
        else:
            return sgc_perc.calibrate(test_features, test_predictions, test_logits)
    
    if evaluate_on_clean:
        # Evaluate on clean test set
        logger.info("Evaluating on clean test set...")
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_raw).float()),
            batch_size=batch_size, shuffle=False
        )
        test_features = extract_multi_layer_features(
            model, test_loader, random_layers, device, target_dim=target_dim, seed=seed
        )
        test_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
        test_predictions = np.argmax(test_probs, axis=1)
        test_logits = model_adapter.predict_logits(test_raw, batch_size=batch_size)
        
        calibrated_probs_perc = sgc_perc.calibrate(test_features, test_predictions, test_logits)
        clean_ece_perc = calculate_ece(calibrated_probs_perc, test_labels)
        
        results['percentile'] = {
            'clean_ece': clean_ece_perc,
            'selected_layers': random_layers,
            'normalization': 'percentile (5th/95th with clipping)',
            'percentile_bounds': {'q05': float(sgc_perc.q05), 'q95': float(sgc_perc.q95)},
            'clipping_stats': {
                'validation_clipped_pct': sgc_perc.clipping_stats['val_clipped_pct'],
            },
            'evaluation_mode': 'clean_test_set'
        }
        logger.info(f"  Clean test ECE = {clean_ece_perc:.4f}")
    else:
        # Evaluate on corruptions (original behavior)
        mce_perc, corruption_eces_perc = compute_mce(
            perc_predict_fn, cifar_c_dir, test_transform, batch_size, dataset,
            cached_predictions=cached_predictions
        )
        
        # Compute average clipping per corruption
        avg_clipping_per_corruption = {}
        for corr, clips in perc_clipping_stats['per_corruption'].items():
            avg_clipping_per_corruption[corr] = float(np.mean(clips))
        
        total_clipping_pct = perc_clipping_stats['total_clipped'] / max(perc_clipping_stats['total_samples'], 1) * 100
        
        results['percentile'] = {
            'mce': mce_perc,
            'corruption_eces': corruption_eces_perc,
            'selected_layers': random_layers,
            'normalization': 'percentile (5th/95th with clipping)',
            'percentile_bounds': {'q05': float(sgc_perc.q05), 'q95': float(sgc_perc.q95)},
            'clipping_stats': {
                'validation_clipped_pct': sgc_perc.clipping_stats['val_clipped_pct'],
                'test_total_clipped_pct': total_clipping_pct,
                'per_corruption_clipped_pct': avg_clipping_per_corruption
            },
            'evaluation_mode': 'corruptions'
        }
        logger.info(f"  mCE = {mce_perc:.4f}")
        logger.info(f"  Total test clipping: {total_clipping_pct:.1f}%")
    
    # =========================================================================
    # Summary comparison
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("NORMALIZATION COMPARISON SUMMARY")
    logger.info("="*80)
    
    if evaluate_on_clean:
        clean_ece_rank = results['rank_based'].get('clean_ece')
        clean_ece_perc = results['percentile'].get('clean_ece')
        ece_diff = clean_ece_perc - clean_ece_rank if (clean_ece_perc is not None and clean_ece_rank is not None) else None
        relative_diff = (ece_diff / clean_ece_rank * 100) if (ece_diff and clean_ece_rank) else None
        
        results['comparison'] = {
            'rank_clean_ece': clean_ece_rank,
            'percentile_clean_ece': clean_ece_perc,
            'ece_difference': ece_diff,
            'relative_difference_pct': relative_diff,
            'rank_wins': clean_ece_rank < clean_ece_perc if (clean_ece_rank and clean_ece_perc) else None,
            'evaluation_mode': 'clean_test_set',
            'conclusion': 'Rank-based vs percentile normalization on clean test set'
        }
        
        logger.info(f"  Rank-based clean ECE:    {clean_ece_rank:.4f}")
        logger.info(f"  Percentile clean ECE:    {clean_ece_perc:.4f}")
        if ece_diff:
            logger.info(f"  Difference:        {ece_diff:+.4f} ({relative_diff:+.1f}%)")
    else:
        mce_rank = results['rank_based'].get('mce')
        mce_perc = results['percentile'].get('mce')
        total_clipping_pct = results['percentile'].get('clipping_stats', {}).get('test_total_clipped_pct', 0)
        mce_diff = mce_perc - mce_rank if (mce_perc and mce_rank) else None
        relative_diff = (mce_diff / mce_rank * 100) if (mce_diff and mce_rank) else None
        
        results['comparison'] = {
            'rank_mce': mce_rank,
            'percentile_mce': mce_perc,
            'mce_difference': mce_diff,
            'relative_difference_pct': relative_diff,
            'rank_wins': mce_rank < mce_perc if (mce_rank and mce_perc) else None,
            'percentile_total_clipping_pct': total_clipping_pct,
            'evaluation_mode': 'corruptions',
            'conclusion': 'Rank-based normalization avoids information collapse under distribution shift'
        }
        
        logger.info(f"  Rank-based mCE:    {mce_rank:.4f}")
        logger.info(f"  Percentile mCE:    {mce_perc:.4f}")
        if mce_diff:
            logger.info(f"  Difference:        {mce_diff:+.4f} ({relative_diff:+.1f}%)")
        logger.info(f"  Percentile clipping: {total_clipping_pct:.1f}% of OOD samples")
        
        # Check for information collapse (all methods giving same results)
        if mce_rank and mce_perc and abs(mce_rank - mce_perc) < 1e-6:
            logger.warning("  WARNING: Identical mCE suggests possible information collapse!")
    
    return results


def evaluate_standard_baselines_clean(
    model_adapter,
    train_raw,
    train_labels,
    val_raw,
    val_labels,
    test_raw,
    test_labels,
    batch_size,
    output_dir,
    device,
    seed: int = 42,
):
    """
    Thin wrapper that reuses `run_standard_baselines` from `layer_selection.py`
    to compute standard baselines on the clean validation/test sets.
    This avoids re-implementing the baselines in this script.
    """
    logger.info("Delegating standard baseline evaluation to layer_selection.run_standard_baselines")
    return run_standard_baselines_clean(
        model_adapter=model_adapter,
        val_raw=val_raw,
        val_labels=val_labels,
        test_raw=test_raw,
        test_labels=test_labels,
        batch_size=batch_size,
        output_dir=output_dir,
        train_raw=train_raw,
        train_labels=train_labels,
        candidate_layers_all=None,
        seed=seed,
        device=str(device),
    )


def evaluate_geometric_methods(model, model_adapter, train_raw, train_labels,
                                val_raw, val_labels, cifar_c_dir, test_transform,
                                batch_size, device, dataset='cifar10',
                                target_dim=256, n_random_layers=6, seed=42,
                                model_name=None):
    """Evaluate geometric calibration methods using FAISS."""
    results = {}
    
    # Pre-compute model predictions for all corruptions (OPTIMIZATION #1)
    logger.info("Pre-computing model predictions for all corruptions...")
    cached_predictions = precompute_corruption_predictions(
        model_adapter, cifar_c_dir, batch_size, dataset
    )
    
    # Discover available layers
    logger.info("Discovering model layers...")
    all_layers = discover_feature_layers(model)
    logger.info(f"Found {len(all_layers)} feature layers")
    
    if len(all_layers) == 0:
        logger.error("No feature layers found!")
        return results
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_raw).float()),
        batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_raw).float()),
        batch_size=batch_size, shuffle=False
    )
    
    # Get model predictions
    val_probs = model_adapter.predict_proba(val_raw, batch_size=batch_size)
    val_predictions = np.argmax(val_probs, axis=1)
    val_logits = model_adapter.predict_logits(val_raw, batch_size=batch_size)
    
    # Feature cache: (corruption, severity, tuple(layers)) -> features (OPTIMIZATION #2)
    feature_cache = {}
    
    # Helper to create predict function for SGC-style methods with caching
    def create_sgc_predict_fn(calibrator, selected_layers, metric='separation'):
        layers_key = tuple(selected_layers)
        
        def predict_fn(Xc, corruption=None, severity=None):
            # Try to get features from cache
            cache_key = (corruption, severity, layers_key) if (corruption is not None and severity is not None) else None
            
            if cache_key and cache_key in feature_cache:
                test_features = feature_cache[cache_key]
            else:
                # Extract features (create DataLoader only if needed)
                test_loader = DataLoader(
                    TensorDataset(torch.from_numpy(Xc).float()),
                    batch_size=batch_size, shuffle=False
                )
                test_features = extract_multi_layer_features(
                    model, test_loader, selected_layers, device,
                    target_dim=target_dim, seed=seed
                )
                # Cache features if we have corruption/severity info
                if cache_key:
                    feature_cache[cache_key] = test_features
            
            # Use cached predictions if available
            if cached_predictions and corruption and severity and (corruption, severity) in cached_predictions:
                _, test_probs, _, _ = cached_predictions[(corruption, severity)]
                test_predictions = np.argmax(test_probs, axis=1)
            else:
                test_probs = model_adapter.predict_proba(Xc, batch_size=batch_size)
                test_predictions = np.argmax(test_probs, axis=1)
            
            return calibrator.calibrate(test_features, test_predictions, test_probs)
        return predict_fn
    
    def create_dac_predict_fn(calibrator, selected_layers):
        layers_key = tuple(selected_layers)
        
        def predict_fn(Xc, corruption=None, severity=None):
            # Try to get features from cache
            cache_key = (corruption, severity, layers_key) if (corruption is not None and severity is not None) else None
            
            if cache_key and cache_key in feature_cache:
                test_features = feature_cache[cache_key]
            else:
                # Extract features
                test_loader = DataLoader(
                    TensorDataset(torch.from_numpy(Xc).float()),
                    batch_size=batch_size, shuffle=False
                )
                test_features = extract_multi_layer_features(
                    model, test_loader, selected_layers, device,
                    target_dim=target_dim, seed=seed
                )
                # Cache features if we have corruption/severity info
                if cache_key:
                    feature_cache[cache_key] = test_features
            
            # Use cached logits if available
            if cached_predictions and corruption and severity and (corruption, severity) in cached_predictions:
                _, _, test_logits, _ = cached_predictions[(corruption, severity)]
            else:
                test_logits = model_adapter.predict_logits(Xc, batch_size=batch_size)
            
            return calibrator.calibrate(test_features, test_logits)
        return predict_fn
    
    # =========================================================================
    # 1. Last Layer Only Baseline
    # =========================================================================
    logger.info("\nEvaluating: Last Layer Only Baseline")
    try:
        last_layer = [all_layers[-1]]
        train_features = extract_multi_layer_features(
            model, train_loader, last_layer, device, target_dim=target_dim, seed=seed
        )
        val_features = extract_multi_layer_features(
            model, val_loader, last_layer, device, target_dim=target_dim, seed=seed
        )
        
        sgc_cal = SGCCalibratorFAISS(use_gpu=(device.type == 'cuda'), metric='separation')
        sgc_cal.fit(train_features, train_labels, val_features, val_labels, val_predictions)
        
        predict_fn = create_sgc_predict_fn(sgc_cal, last_layer)
        mce, corruption_eces = compute_mce(predict_fn, cifar_c_dir, test_transform, batch_size, dataset, cached_predictions=cached_predictions)
        results['last_layer_only_baseline'] = {
            'mce': mce, 'corruption_eces': corruption_eces, 'selected_layers': last_layer
        }
        logger.info(f"  mCE = {mce:.4f}")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        results['last_layer_only_baseline'] = {'mce': None, 'error': str(e)}
    
    # =========================================================================
    # 2. SGC with Random Layers - Separation Score
    # =========================================================================
    logger.info("\nEvaluating: Global Random (Separation)")
    try:
        random_layers = select_random_layers(all_layers, n_layers=n_random_layers, seed=seed)
        train_features = extract_multi_layer_features(
            model, train_loader, random_layers, device, target_dim=target_dim, seed=seed
        )
        val_features = extract_multi_layer_features(
            model, val_loader, random_layers, device, target_dim=target_dim, seed=seed
        )
        
        sgc_cal = SGCCalibratorFAISS(use_gpu=(device.type == 'cuda'), metric='separation')
        sgc_cal.fit(train_features, train_labels, val_features, val_labels, val_predictions)
        
        predict_fn = create_sgc_predict_fn(sgc_cal, random_layers)
        mce, corruption_eces = compute_mce(predict_fn, cifar_c_dir, test_transform, batch_size, dataset, cached_predictions=cached_predictions)
        results['global_random_separation'] = {
            'mce': mce, 'corruption_eces': corruption_eces, 'selected_layers': random_layers
        }
        logger.info(f"  mCE = {mce:.4f}")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        results['global_random_separation'] = {'mce': None, 'error': str(e)}
    
    # =========================================================================
    # 3. SGC with Random Layers - Trust Score
    # =========================================================================
    logger.info("\nEvaluating: Global Random (Trust Score)")
    try:
        # Reuse random_layers from above
        sgc_cal_ts = SGCCalibratorFAISS(use_gpu=(device.type == 'cuda'), metric='trust_score')
        sgc_cal_ts.fit(train_features, train_labels, val_features, val_labels, val_predictions)
        
        predict_fn = create_sgc_predict_fn(sgc_cal_ts, random_layers, metric='trust_score')
        mce, corruption_eces = compute_mce(predict_fn, cifar_c_dir, test_transform, batch_size, dataset, cached_predictions=cached_predictions)
        results['global_random_trust_score'] = {
            'mce': mce, 'corruption_eces': corruption_eces, 'selected_layers': random_layers
        }
        logger.info(f"  mCE = {mce:.4f}")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        results['global_random_trust_score'] = {'mce': None, 'error': str(e)}
    
    # =========================================================================
    # 4. DAC with Random Layer Selection
    # =========================================================================
    logger.info("\nEvaluating: DAC with Random Layer Selection")
    try:
        if dataset=="cifar10":
            k=50
        elif dataset=="cifar100":
            k=200
        elif dataset=="tiny_imagenet":
            k=200
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        dac_cal = DACCalibratorFAISS(use_gpu=(device.type == 'cuda'), k=k)
        dac_cal.fit(train_features, train_labels, val_features, val_labels, val_logits)
        
        predict_fn = create_dac_predict_fn(dac_cal, random_layers)
        mce, corruption_eces = compute_mce(predict_fn, cifar_c_dir, test_transform, batch_size, dataset, cached_predictions=cached_predictions)
        results['dac_with_random_layer_selection'] = {
            'mce': mce, 'corruption_eces': corruption_eces, 'selected_layers': random_layers
        }
        logger.info(f"  mCE = {mce:.4f}")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        results['dac_with_random_layer_selection'] = {'mce': None, 'error': str(e)}
    
    # =========================================================================
    # 5. Geometric Physical Space (last layer raw features)
    # =========================================================================
    logger.info("\nEvaluating: Geometric Physical Space")
    try:
        # Use raw logits as features (physical/output space)
        train_logits = model_adapter.predict_logits(train_raw, batch_size=batch_size)
        
        sgc_cal = SGCCalibratorFAISS(use_gpu=(device.type == 'cuda'), metric='separation')
        sgc_cal.fit(train_logits.astype(np.float32), train_labels, 
                    val_logits.astype(np.float32), val_labels, val_predictions)
        
        def physical_predict_fn(Xc, corruption=None, severity=None):
            # Use cached logits if available
            if cached_predictions and corruption and severity and (corruption, severity) in cached_predictions:
                _, _, test_logits, _ = cached_predictions[(corruption, severity)]
            else:
                test_logits = model_adapter.predict_logits(Xc, batch_size=batch_size)
            
            # Use cached probs if available
            if cached_predictions and corruption and severity and (corruption, severity) in cached_predictions:
                _, test_probs, _, _ = cached_predictions[(corruption, severity)]
            else:
                test_probs = model_adapter.predict_proba(Xc, batch_size=batch_size)
            
            test_predictions = np.argmax(test_probs, axis=1)
            return sgc_cal.calibrate(test_logits.astype(np.float32), test_predictions, test_probs)
        
        mce, corruption_eces = compute_mce(physical_predict_fn, cifar_c_dir, test_transform, batch_size, dataset, cached_predictions=cached_predictions)
        results['geometric_physical_space'] = {
            'mce': mce, 'corruption_eces': corruption_eces, 'feature_type': 'logits'
        }
        logger.info(f"  mCE = {mce:.4f}")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        results['geometric_physical_space'] = {'mce': None, 'error': str(e)}
    
    # =========================================================================
    # 6-8. SGC with DAC-style Layers (using actual DAC layer selection)
    # =========================================================================
    logger.info("\nEvaluating: SGC with DAC Layers")
    try:
        # Use actual DAC layer selection function to get proper DAC-style layers
        if model_name is None:
            raise ValueError("model_name is required for DAC layer selection. Please pass model_name parameter to evaluate_geometric_methods.")
        
        from Calibrators.density_aware_calibration import get_dac_target_layers
        dac_style_layers = get_dac_target_layers(model_name, model)
        logger.info(f"  Using DAC-style layers: {dac_style_layers}")
        
        train_features_dac = extract_multi_layer_features(
            model, train_loader, dac_style_layers, device, target_dim=target_dim, seed=seed
        )
        val_features_dac = extract_multi_layer_features(
            model, val_loader, dac_style_layers, device, target_dim=target_dim, seed=seed
        )
        
        # Separation score
        sgc_cal_sep = SGCCalibratorFAISS(use_gpu=(device.type == 'cuda'), metric='separation')
        sgc_cal_sep.fit(train_features_dac, train_labels, val_features_dac, val_labels, val_predictions)
        
        predict_fn = create_sgc_predict_fn(sgc_cal_sep, dac_style_layers)
        mce, corruption_eces = compute_mce(predict_fn, cifar_c_dir, test_transform, batch_size, dataset, cached_predictions=cached_predictions)
        results['sgc_with_dac_layers_separation'] = {
            'mce': mce, 'corruption_eces': corruption_eces, 'selected_layers': dac_style_layers
        }
        logger.info(f"  SGC+DAC Layers (Separation) mCE = {mce:.4f}")
        
        # Trust score
        sgc_cal_ts = SGCCalibratorFAISS(use_gpu=(device.type == 'cuda'), metric='trust_score')
        sgc_cal_ts.fit(train_features_dac, train_labels, val_features_dac, val_labels, val_predictions)
        
        predict_fn = create_sgc_predict_fn(sgc_cal_ts, dac_style_layers, metric='trust_score')
        mce, corruption_eces = compute_mce(predict_fn, cifar_c_dir, test_transform, batch_size, dataset, cached_predictions=cached_predictions)
        results['sgc_with_dac_layers_trust_score'] = {
            'mce': mce, 'corruption_eces': corruption_eces, 'selected_layers': dac_style_layers
        }
        logger.info(f"  SGC+DAC Layers (Trust Score) mCE = {mce:.4f}")
        
    except Exception as e:
        logger.error(f"  Failed: {e}")
        results['sgc_with_dac_layers_separation'] = {'mce': None, 'error': str(e)}
        results['sgc_with_dac_layers_trust_score'] = {'mce': None, 'error': str(e)}
    
    # =========================================================================
    # 9. DAC Standard (using original DAC implementation if available)
    # =========================================================================
    logger.info("\nEvaluating: Density Aware Calibration (DAC)")
    try:
        from Calibrators.density_aware_calibration import (
            DensityAwareCalibrator,
            get_dac_target_layers,
            get_dac_k_value,
            extract_dac_features
        )
        
        if model_name is None:
            raise ValueError("model_name is required for DAC layer selection. Please pass model_name parameter to evaluate_geometric_methods.")
        dac_layers = get_dac_target_layers(model_name, model)
        k_value = get_dac_k_value(dataset)
        
        train_loader_dac = DataLoader(
            TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
            batch_size=batch_size, shuffle=False
        )
        val_loader_dac = DataLoader(
            TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
            batch_size=batch_size, shuffle=False
        )
        
        # Extract features from intermediate layers and logits
        logger.info("  Extracting DAC features from train set...")
        train_intermediate_features, train_logits, train_labels_tensor = extract_dac_features(
            model, train_loader_dac, dac_layers, device
        )
        logger.info("  Extracting DAC features from validation set...")
        val_intermediate_features, val_logits, val_labels_tensor = extract_dac_features(
            model, val_loader_dac, dac_layers, device
        )
        
        # According to DAC paper, logits should be included as a feature layer
        # Append logits to the feature lists for KNN density scoring
        train_features_with_logits = train_intermediate_features + [train_logits]
        val_features_with_logits = val_intermediate_features + [val_logits]
        
        # Initialize and fit DAC calibrator
        dac_cal = DensityAwareCalibrator(k=k_value, use_gpu=(device.type == 'cuda'))
        dac_cal.fit(
            train_features_with_logits,
            val_features_with_logits,
            val_logits,  # Used as numerator in temperature scaling
            val_labels_tensor
        )
        
        def dac_predict_fn(Xc, corruption=None, severity=None):
            # Use cached logits if available
            if cached_predictions and corruption and severity and (corruption, severity) in cached_predictions:
                _, _, test_logits_cached, _ = cached_predictions[(corruption, severity)]
                # For DAC, we still need to extract intermediate features, but can use cached logits
                test_loader_c = DataLoader(
                    TensorDataset(torch.from_numpy(Xc), torch.zeros(len(Xc), dtype=torch.long)),
                    batch_size=batch_size, shuffle=False
                )
                test_intermediate_features, _, _ = extract_dac_features(
                    model, test_loader_c, dac_layers, device
                )
                test_logits = test_logits_cached
            else:
                test_loader_c = DataLoader(
                    TensorDataset(torch.from_numpy(Xc), torch.zeros(len(Xc), dtype=torch.long)),
                    batch_size=batch_size, shuffle=False
                )
                # Extract test features
                test_intermediate_features, test_logits, _ = extract_dac_features(
                    model, test_loader_c, dac_layers, device
                )
            # Append logits to feature list
            test_features_with_logits = test_intermediate_features + [test_logits]
            # Calibrate
            return dac_cal.calibrate(test_features_with_logits, test_logits)
        
        mce, corruption_eces = compute_mce(dac_predict_fn, cifar_c_dir, test_transform, batch_size, dataset, cached_predictions=cached_predictions)
        results['density_aware_calibration'] = {
            'mce': mce, 'corruption_eces': corruption_eces, 'selected_layers': dac_layers
        }
        logger.info(f"  mCE = {mce:.4f}")
    except ImportError:
        logger.warning("  DAC implementation not available, using FAISS-based version")
        # Use our FAISS-based DAC
        try:
            dac_cal = DACCalibratorFAISS(use_gpu=(device.type == 'cuda'), k=10)
            dac_cal.fit(train_features, train_labels, val_features, val_labels, val_logits)
            
            predict_fn = create_dac_predict_fn(dac_cal, random_layers)
            mce, corruption_eces = compute_mce(predict_fn, cifar_c_dir, test_transform, batch_size, dataset, cached_predictions=cached_predictions)
            results['density_aware_calibration'] = {
                'mce': mce, 'corruption_eces': corruption_eces, 'selected_layers': random_layers,
                'note': 'FAISS-based implementation'
            }
            logger.info(f"  mCE = {mce:.4f}")
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results['density_aware_calibration'] = {'mce': None, 'error': str(e)}
    except Exception as e:
        logger.error(f"  Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        results['density_aware_calibration'] = {'mce': None, 'error': str(e)}
    
    # =========================================================================
    # 10. SGC with DAC Preprocessing (normalize features like DAC)
    # =========================================================================
    logger.info("\nEvaluating: SGC with DAC Preprocessing")
    try:
        # DAC-style preprocessing: L2 normalize features
        from sklearn.preprocessing import normalize
        train_features_norm = normalize(train_features, norm='l2')
        val_features_norm = normalize(val_features, norm='l2')
        
        sgc_cal = SGCCalibratorFAISS(use_gpu=(device.type == 'cuda'), metric='separation')
        sgc_cal.fit(train_features_norm.astype(np.float32), train_labels, 
                    val_features_norm.astype(np.float32), val_labels, val_predictions)
        
        def sgc_dac_preprocess_predict_fn(Xc, corruption=None, severity=None):
            # Try to get features from cache
            cache_key = (corruption, severity, tuple(random_layers)) if (corruption is not None and severity is not None) else None
            if cache_key and cache_key in feature_cache:
                test_features = feature_cache[cache_key]
            else:
                test_loader = DataLoader(
                    TensorDataset(torch.from_numpy(Xc).float()),
                    batch_size=batch_size, shuffle=False
                )
                test_features = extract_multi_layer_features(
                    model, test_loader, random_layers, device, target_dim=target_dim, seed=seed
                )
                if cache_key:
                    feature_cache[cache_key] = test_features
            test_features_norm = normalize(test_features, norm='l2')
            
            # Use cached predictions if available
            if cached_predictions and corruption and severity and (corruption, severity) in cached_predictions:
                _, test_probs, _, _ = cached_predictions[(corruption, severity)]
            else:
                test_probs = model_adapter.predict_proba(Xc, batch_size=batch_size)
            test_predictions = np.argmax(test_probs, axis=1)
            return sgc_cal.calibrate(test_features_norm.astype(np.float32), test_predictions, test_probs)
        
        mce, corruption_eces = compute_mce(sgc_dac_preprocess_predict_fn, cifar_c_dir, test_transform, batch_size, dataset, cached_predictions=cached_predictions)
        results['sgc_with_dac_preprocessing_separation'] = {
            'mce': mce, 'corruption_eces': corruption_eces, 'selected_layers': random_layers
        }
        logger.info(f"  mCE = {mce:.4f}")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        results['sgc_with_dac_preprocessing_separation'] = {'mce': None, 'error': str(e)}
    
    return results


def evaluate_coordinate_methods(model, model_adapter, train_raw, train_labels,
                                 val_raw, val_labels, cifar_c_dir, test_transform,
                                 batch_size, device, dataset='cifar10',
                                 target_dim=256, seed=42):
    """Evaluate coordinate-based calibration methods."""
    results = {}
    
    # Pre-compute model predictions for all corruptions (OPTIMIZATION #1)
    logger.info("Pre-computing model predictions for all corruptions...")
    cached_predictions = precompute_corruption_predictions(
        model_adapter, cifar_c_dir, batch_size, dataset
    )
    
    # Feature cache: (corruption, severity, tuple(layers), pooling) -> features (OPTIMIZATION #2)
    feature_cache = {}
    
    # Discover layers for coordinate sampling
    all_layers = discover_feature_layers(model)
    
    if len(all_layers) == 0:
        logger.error("No feature layers found for coordinate methods!")
        return results
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_raw).float()),
        batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_raw).float()),
        batch_size=batch_size, shuffle=False
    )
    
    # Get predictions
    val_probs = model_adapter.predict_proba(val_raw, batch_size=batch_size)
    val_predictions = np.argmax(val_probs, axis=1)
    val_logits = model_adapter.predict_logits(val_raw, batch_size=batch_size)
    
    # =========================================================================
    # Coordinate Sampling with FAISS (SGC-style)
    # =========================================================================
    logger.info("\nEvaluating: Coordinate Sampling (FAISS)")
    try:
        # Extract features with coordinate sampling (uniform from all layers)
        train_features = extract_multi_layer_features(
            model, train_loader, all_layers, device, 
            pooling='adaptive_avg', target_dim=target_dim, seed=seed
        )
        val_features = extract_multi_layer_features(
            model, val_loader, all_layers, device,
            pooling='adaptive_avg', target_dim=target_dim, seed=seed
        )
        
        sgc_cal = SGCCalibratorFAISS(use_gpu=(device.type == 'cuda'), metric='separation')
        sgc_cal.fit(train_features, train_labels, val_features, val_labels, val_predictions)
        
        def coord_predict_fn(Xc, corruption=None, severity=None):
            # Try to get features from cache (include pooling mode in key)
            cache_key = (corruption, severity, tuple(all_layers), 'adaptive_avg') if (corruption is not None and severity is not None) else None
            if cache_key and cache_key in feature_cache:
                test_features = feature_cache[cache_key]
            else:
                test_loader = DataLoader(
                    TensorDataset(torch.from_numpy(Xc).float()),
                    batch_size=batch_size, shuffle=False
                )
                test_features = extract_multi_layer_features(
                    model, test_loader, all_layers, device,
                    pooling='adaptive_avg', target_dim=target_dim, seed=seed
                )
                if cache_key:
                    feature_cache[cache_key] = test_features
            
            # Use cached predictions if available
            if cached_predictions and corruption and severity and (corruption, severity) in cached_predictions:
                _, test_probs, _, _ = cached_predictions[(corruption, severity)]
            else:
                test_probs = model_adapter.predict_proba(Xc, batch_size=batch_size)
            test_predictions = np.argmax(test_probs, axis=1)
            return sgc_cal.calibrate(test_features, test_predictions, test_probs)
        
        mce, corruption_eces = compute_mce(coord_predict_fn, cifar_c_dir, test_transform, batch_size, dataset, cached_predictions=cached_predictions)
        results['coordinate_sampling_faiss'] = {
            'mce': mce, 'corruption_eces': corruption_eces, 'n_layers': len(all_layers)
        }
        logger.info(f"  mCE = {mce:.4f}")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        results['coordinate_sampling_faiss'] = {'mce': None, 'error': str(e)}
    
    # =========================================================================
    # Coordinate SPP (Spatial Pyramid Pooling style)
    # =========================================================================
    logger.info("\nEvaluating: Coordinate SPP (FAISS)")
    try:
        # Use max pooling instead of avg (SPP-style)
        train_features_spp = extract_multi_layer_features(
            model, train_loader, all_layers, device,
            pooling='max', target_dim=target_dim, seed=seed
        )
        val_features_spp = extract_multi_layer_features(
            model, val_loader, all_layers, device,
            pooling='max', target_dim=target_dim, seed=seed
        )
        
        sgc_cal_spp = SGCCalibratorFAISS(use_gpu=(device.type == 'cuda'), metric='separation')
        sgc_cal_spp.fit(train_features_spp, train_labels, val_features_spp, val_labels, val_predictions)
        
        def spp_predict_fn(Xc, corruption=None, severity=None):
            # Try to get features from cache (note: pooling='max' creates different cache key)
            cache_key = (corruption, severity, tuple(all_layers), 'max') if (corruption is not None and severity is not None) else None
            if cache_key and cache_key in feature_cache:
                test_features = feature_cache[cache_key]
            else:
                test_loader = DataLoader(
                    TensorDataset(torch.from_numpy(Xc).float()),
                    batch_size=batch_size, shuffle=False
                )
                test_features = extract_multi_layer_features(
                    model, test_loader, all_layers, device,
                    pooling='max', target_dim=target_dim, seed=seed
                )
                if cache_key:
                    feature_cache[cache_key] = test_features
            
            # Use cached predictions if available
            if cached_predictions and corruption and severity and (corruption, severity) in cached_predictions:
                _, test_probs, _, _ = cached_predictions[(corruption, severity)]
            else:
                test_probs = model_adapter.predict_proba(Xc, batch_size=batch_size)
            test_predictions = np.argmax(test_probs, axis=1)
            return sgc_cal_spp.calibrate(test_features, test_predictions, test_probs)
        
        mce, corruption_eces = compute_mce(spp_predict_fn, cifar_c_dir, test_transform, batch_size, dataset, cached_predictions=cached_predictions)
        results['coordinate_spp_faiss'] = {
            'mce': mce, 'corruption_eces': corruption_eces, 'n_layers': len(all_layers)
        }
        logger.info(f"  mCE = {mce:.4f}")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        results['coordinate_spp_faiss'] = {'mce': None, 'error': str(e)}
    
    # =========================================================================
    # Coordinate SPP with DAC algorithm
    # =========================================================================
    logger.info("\nEvaluating: Coordinate SPP + DAC")
    try:
        dac_cal = DACCalibratorFAISS(use_gpu=(device.type == 'cuda'), k=10)
        dac_cal.fit(train_features_spp, train_labels, val_features_spp, val_labels, val_logits)
        
        def spp_dac_predict_fn(Xc, corruption=None, severity=None):
            # Try to get features from cache (reuse max pooling features from spp_predict_fn)
            cache_key = (corruption, severity, tuple(all_layers), 'max') if (corruption is not None and severity is not None) else None
            if cache_key and cache_key in feature_cache:
                test_features = feature_cache[cache_key]
            else:
                test_loader = DataLoader(
                    TensorDataset(torch.from_numpy(Xc).float()),
                    batch_size=batch_size, shuffle=False
                )
                test_features = extract_multi_layer_features(
                    model, test_loader, all_layers, device,
                    pooling='max', target_dim=target_dim, seed=seed
                )
                if cache_key:
                    feature_cache[cache_key] = test_features
            
            # Use cached logits if available
            if cached_predictions and corruption and severity and (corruption, severity) in cached_predictions:
                _, _, test_logits, _ = cached_predictions[(corruption, severity)]
            else:
                test_logits = model_adapter.predict_logits(Xc, batch_size=batch_size)
            return dac_cal.calibrate(test_features, test_logits)
        
        mce, corruption_eces = compute_mce(spp_dac_predict_fn, cifar_c_dir, test_transform, batch_size, dataset, cached_predictions=cached_predictions)
        results['coordinate_spp_dac'] = {
            'mce': mce, 'corruption_eces': corruption_eces, 'n_layers': len(all_layers)
        }
        logger.info(f"  mCE = {mce:.4f}")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        results['coordinate_spp_dac'] = {'mce': None, 'error': str(e)}
    
    # =========================================================================
    # DAC with Coordinate Features
    # =========================================================================
    logger.info("\nEvaluating: DAC with Coordinate Features")
    try:
        # Use avg pooling coordinate features with DAC
        dac_cal_coord = DACCalibratorFAISS(use_gpu=(device.type == 'cuda'), k=10)
        dac_cal_coord.fit(train_features, train_labels, val_features, val_labels, val_logits)
        
        def dac_coord_predict_fn(Xc, corruption=None, severity=None):
            # Try to get features from cache (reuse adaptive_avg pooling features from coord_predict_fn)
            cache_key = (corruption, severity, tuple(all_layers), 'adaptive_avg') if (corruption is not None and severity is not None) else None
            if cache_key and cache_key in feature_cache:
                test_features = feature_cache[cache_key]
            else:
                test_loader = DataLoader(
                    TensorDataset(torch.from_numpy(Xc).float()),
                    batch_size=batch_size, shuffle=False
                )
                test_features = extract_multi_layer_features(
                    model, test_loader, all_layers, device,
                    pooling='adaptive_avg', target_dim=target_dim, seed=seed
                )
                if cache_key:
                    feature_cache[cache_key] = test_features
            
            # Use cached logits if available
            if cached_predictions and corruption and severity and (corruption, severity) in cached_predictions:
                _, _, test_logits, _ = cached_predictions[(corruption, severity)]
            else:
                test_logits = model_adapter.predict_logits(Xc, batch_size=batch_size)
            return dac_cal_coord.calibrate(test_features, test_logits)
        
        mce, corruption_eces = compute_mce(dac_coord_predict_fn, cifar_c_dir, test_transform, batch_size, dataset, cached_predictions=cached_predictions)
        results['dac_with_coordinate_features'] = {
            'mce': mce, 'corruption_eces': corruption_eces, 'n_layers': len(all_layers)
        }
        logger.info(f"  mCE = {mce:.4f}")
    except Exception as e:
        logger.error(f"  Failed: {e}")
        results['dac_with_coordinate_features'] = {'mce': None, 'error': str(e)}
    
    return results


def evaluate_accuracy_calibration(
    model_adapter,
    val_raw,
    val_labels,
    test_raw,
    test_labels,
    cifar_c_dir,
    batch_size,
    dataset='cifar10'
):
    """
    Evaluate accuracy-based calibration baseline.
    Uses validation accuracy as constant confidence for all predictions.
    
    Args:
        model_adapter: Model adapter for predictions
        val_raw: Validation data (numpy array)
        val_labels: Validation labels
        test_raw: Test data (numpy array)
        test_labels: Test labels
        cifar_c_dir: Path to CIFAR-C data directory
        batch_size: Batch size for predictions
        dataset: Dataset name ('cifar10' or 'cifar100')
    
    Returns:
        Dict with val_accuracy, clean_ece, mce, corruption_eces
    """
    results = {}
    
    logger.info("\n" + "="*80)
    logger.info("ACCURACY CALIBRATION BASELINE")
    logger.info("="*80)
    
    # Determine number of classes
    num_classes = 10 if dataset == 'cifar10' else 100
    
    # Get validation predictions
    logger.info("Computing validation accuracy...")
    val_probs = model_adapter.predict_proba(val_raw, batch_size=batch_size)
    val_predictions = np.argmax(val_probs, axis=1)
    
    # Fit calibrator (compute validation accuracy)
    calibrator = AccuracyCalibrator()
    val_accuracy = calibrator.fit(val_predictions, val_labels)
    logger.info(f"  Validation accuracy: {val_accuracy:.4f}")
    
    # Evaluate on clean test set
    logger.info("Evaluating on clean test set...")
    test_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
    test_predictions = np.argmax(test_probs, axis=1)
    
    # Get calibrated probabilities
    calibrated_probs = calibrator.calibrate(test_predictions, num_classes)
    
    # Compute ECE on clean test set
    clean_ece = calculate_ece(calibrated_probs, test_labels)
    logger.info(f"  Clean test ECE: {clean_ece:.4f}")
    
    # Pre-compute corruption predictions for efficiency
    logger.info("Pre-computing corruption predictions...")
    cached_predictions = precompute_corruption_predictions(
        model_adapter, cifar_c_dir, batch_size, dataset
    )
    
    # Create predict function for corruptions
    def acc_cali_predict_fn(Xc, corruption=None, severity=None):
        """Predict function that returns accuracy-based calibrated probabilities."""
        # Use cached predictions if available
        if cached_predictions and corruption and severity and (corruption, severity) in cached_predictions:
            _, test_probs, _, _ = cached_predictions[(corruption, severity)]
            test_predictions = np.argmax(test_probs, axis=1)
        else:
            # Fallback: compute predictions
            test_probs = model_adapter.predict_proba(Xc, batch_size=batch_size)
            test_predictions = np.argmax(test_probs, axis=1)
        
        # Return calibrated probabilities using validation accuracy
        return calibrator.calibrate(test_predictions, num_classes)
    
    # Evaluate on all corruptions
    logger.info("Evaluating on corruptions...")
    test_transform = None  # Not needed for our direct numpy approach
    mce, corruption_eces = compute_mce(
        acc_cali_predict_fn, cifar_c_dir, test_transform, batch_size, dataset,
        cached_predictions=cached_predictions
    )
    
    logger.info(f"  mCE: {mce:.4f}")
    
    results = {
        'val_accuracy': float(val_accuracy),
        'clean_ece': float(clean_ece),
        'mce': float(mce) if mce is not None else None,
        'corruption_eces': {k: float(v) for k, v in corruption_eces.items()} if corruption_eces else {}
    }
    
    return results


# ==============================================================================
# RESULT SAVING
# ==============================================================================

def save_results_incrementally(results: Dict, output_file: str):
    """Save results incrementally to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_file}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Corruption Evaluation for Calibration Methods')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--model_name', type=str, required=True, help='Model architecture (e.g., resnet18)')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--cifar_c_dir', type=str, required=True, help='Path to CIFAR-C data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--target_dim', type=int, default=256, help='Target dimension for features')
    parser.add_argument('--n_random_layers', type=int, default=6, help='Number of random layers for SGC')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='corruption_results')
    parser.add_argument('--output_name', type=str, default=None, help='Custom output filename')
    parser.add_argument('--skip_baselines', action='store_true', help='Skip standard baseline methods')
    parser.add_argument('--skip_geometric', action='store_true', help='Skip geometric methods')
    parser.add_argument('--skip_coordinate', action='store_true', help='Skip coordinate methods')
    parser.add_argument('--run_normalization_ablation', action='store_true', 
                        help='Run normalization method comparison (percentile vs rank-based)')
    parser.add_argument('--run_acc_cali', action='store_true',
                        help='Run accuracy calibration baseline (uses validation accuracy as constant confidence)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    try:
        from Experiments.run_post_hoc_calibration import (
            PyTorchModelAdapter,
            get_data_loaders,
            load_trained_model,
        )
        
        # Load data (and infer number of classes) using the same helper as other experiments
        train_loader, val_loader, test_loader, num_classes = get_data_loaders(
            args.dataset,
            args.batch_size,
            seed=args.seed,
        )

        # Load model using the canonical helper (needs model_name and num_classes)
        model = load_trained_model(
            args.model_path,
            args.model_name,
            num_classes,
            device,
            dataset=args.dataset,
        )
        
        # Extract raw data
        def extract_raw(loader):
            xs, ys = [], []
            for batch in loader:
                data, labels = batch[:2]
                xs.append(data.numpy())
                ys.append(labels.numpy())
            return np.concatenate(xs), np.concatenate(ys)
        
        train_raw, train_labels = extract_raw(train_loader)
        val_raw, val_labels = extract_raw(val_loader)
        test_raw, test_labels = extract_raw(test_loader)
        
        # Create model adapter (pass dataset for consistency with other scripts)
        model_adapter = PyTorchModelAdapter(model, device, args.dataset)
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure your project structure is correct")
        return
    
    # Output file
    if args.output_name:
        output_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    else:
        model_name = args.model_name
        dataset = args.dataset
        seed = args.seed
        output_file = os.path.join(args.output_dir, f"corruption_ablation_{dataset}_{model_name}_s{seed}.json")
    
    # Load existing results if any
    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded existing results from: {output_file}")
    
    results['config'] = {
        'model_path': args.model_path,
        'dataset': args.dataset,
        'cifar_c_dir': args.cifar_c_dir,
        'batch_size': args.batch_size,
        'target_dim': args.target_dim,
        'n_random_layers': args.n_random_layers,
        'seed': args.seed,
        'device': str(device)
    }
    
    # Test transform
    test_transform = None  # Not needed for our direct numpy approach
    
    # Run evaluations
    logger.info("\n" + "="*80)
    if args.run_acc_cali:
        logger.info("STARTING ACCURACY CALIBRATION EVALUATION")
    else:
        logger.info("STARTING CORRUPTION EVALUATION")
    logger.info("="*80)
    
    # When --run-acc-cali is enabled, only run accuracy calibration and normalization ablation
    if args.run_acc_cali:
        # Normalization comparison ablation (on clean test set)
        if args.run_normalization_ablation:
            logger.info("\n--- NORMALIZATION METHOD COMPARISON (CLEAN TEST SET) ---")
            normalization_results = evaluate_normalization_comparison(
                model, model_adapter, train_raw, train_labels,
                val_raw, val_labels, args.cifar_c_dir, test_transform,
                args.batch_size, device, args.dataset,
                args.target_dim, args.n_random_layers, args.seed,
                test_raw=test_raw,
                test_labels=test_labels,
                evaluate_on_clean=True
            )
            results['normalization_comparison'] = normalization_results
            save_results_incrementally(results, output_file)
        
        # Accuracy calibration baseline
        logger.info("\n--- ACCURACY CALIBRATION BASELINE ---")
        acc_cali_results = evaluate_accuracy_calibration(
            model_adapter=model_adapter,
            val_raw=val_raw,
            val_labels=val_labels,
            test_raw=test_raw,
            test_labels=test_labels,
            cifar_c_dir=args.cifar_c_dir,
            batch_size=args.batch_size,
            dataset=args.dataset
        )
        results['accuracy_calibration'] = acc_cali_results
        save_results_incrementally(results, output_file)
    else:
        # Standard evaluation path (when --run-acc-cali is NOT enabled)
        # Standard baselines
        if not args.skip_baselines:
            logger.info("\n--- STANDARD BASELINES (CLEAN) ---")
            baseline_results = evaluate_standard_baselines_clean(
                model_adapter=model_adapter,
                train_raw=train_raw,
                train_labels=train_labels,
                val_raw=val_raw,
                val_labels=val_labels,
                test_raw=test_raw,
                test_labels=test_labels,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                device=device,
                seed=args.seed,
            )
            results['standard_baselines'] = baseline_results
            save_results_incrementally(results, output_file)
        
        # Geometric methods
        if not args.skip_geometric:
            logger.info("\n--- GEOMETRIC METHODS ---")
            geometric_results = evaluate_geometric_methods(
                model, model_adapter, train_raw, train_labels,
                val_raw, val_labels, args.cifar_c_dir, test_transform,
                args.batch_size, device, args.dataset,
                args.target_dim, args.n_random_layers, args.seed,
                model_name=args.model_name
            )
            results['geometric_methods'] = geometric_results
            save_results_incrementally(results, output_file)
        
        # Coordinate methods
        if not args.skip_coordinate:
            logger.info("\n--- COORDINATE METHODS ---")
            coordinate_results = evaluate_coordinate_methods(
                model, model_adapter, train_raw, train_labels,
                val_raw, val_labels, args.cifar_c_dir, test_transform,
                args.batch_size, device, args.dataset,
                args.target_dim, args.seed
            )
            results['coordinate_methods'] = coordinate_results
            save_results_incrementally(results, output_file)
        
        # Normalization comparison ablation (on corruptions)
        if args.run_normalization_ablation:
            logger.info("\n--- NORMALIZATION METHOD COMPARISON ---")
            normalization_results = evaluate_normalization_comparison(
                model, model_adapter, train_raw, train_labels,
                val_raw, val_labels, args.cifar_c_dir, test_transform,
                args.batch_size, device, args.dataset,
                args.target_dim, args.n_random_layers, args.seed,
                test_raw=None,
                test_labels=None,
                evaluate_on_clean=False
            )
            results['normalization_comparison'] = normalization_results
            save_results_incrementally(results, output_file)
    
    # Summary
    logger.info("\n" + "="*80)
    if args.run_acc_cali:
        logger.info("ACCURACY CALIBRATION EVALUATION SUMMARY")
    else:
        logger.info("CORRUPTION EVALUATION SUMMARY")
    logger.info("="*80)
    
    if args.run_acc_cali:
        # Summary for accuracy calibration mode
        if 'accuracy_calibration' in results:
            acc_cali = results['accuracy_calibration']
            logger.info(f"  Validation Accuracy: {acc_cali.get('val_accuracy', 'N/A'):.4f}")
            logger.info(f"  Clean Test ECE: {acc_cali.get('clean_ece', 'N/A'):.4f}")
            if acc_cali.get('mce') is not None:
                logger.info(f"  mCE (corruptions): {acc_cali.get('mce'):.4f}")
        
        # Normalization comparison summary (clean test set)
        if 'normalization_comparison' in results:
            comp = results['normalization_comparison'].get('comparison', {})
            logger.info(f"\nNormalization Comparison (Clean Test Set):")
            if comp.get('evaluation_mode') == 'clean_test_set':
                logger.info(f"  Rank-based clean ECE: {comp.get('rank_clean_ece', 'N/A')}")
                logger.info(f"  Percentile clean ECE: {comp.get('percentile_clean_ece', 'N/A')}")
                if comp.get('ece_difference') is not None:
                    logger.info(f"  Difference: {comp.get('ece_difference'):+.4f} ({comp.get('relative_difference_pct', 0):+.1f}%)")
    else:
        # Summary for standard corruption evaluation mode
        all_mces = []
        for section_name, section_results in results.items():
            if isinstance(section_results, dict) and section_name != 'config':
                for method, data in section_results.items():
                    if isinstance(data, dict) and 'mce' in data and data['mce'] is not None:
                        logger.info(f"  {method}: mCE = {data['mce']:.4f}")
                        all_mces.append((method, data['mce']))
        
        if all_mces:
            best_method, best_mce = min(all_mces, key=lambda x: x[1])
            logger.info(f"\nBest method: {best_method} (mCE = {best_mce:.4f})")
        
        # Normalization comparison summary (corruptions)
        if 'normalization_comparison' in results:
            comp = results['normalization_comparison'].get('comparison', {})
            logger.info(f"\nNormalization Comparison:")
            logger.info(f"  Rank-based mCE: {comp.get('rank_mce', 'N/A')}")
            logger.info(f"  Percentile mCE: {comp.get('percentile_mce', 'N/A')}")
            if comp.get('mce_difference') is not None:
                logger.info(f"  Difference: {comp.get('mce_difference'):+.4f} ({comp.get('relative_difference_pct', 0):+.1f}%)")
            logger.info(f"  Percentile clipping: {comp.get('percentile_total_clipping_pct', 'N/A')}%")
    
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()