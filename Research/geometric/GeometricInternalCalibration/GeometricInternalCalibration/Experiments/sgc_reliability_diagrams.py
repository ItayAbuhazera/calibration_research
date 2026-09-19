#!/usr/bin/env python3
"""
sgc_reliability_diagrams.py

SGC Reliability Diagram Generation Script with CLI support.

This script generates publication-quality reliability diagrams for 
Randomized Geometric Calibration (SGC/RGCL) and its variants.

Usage:
    # Run single experiment
    python sgc_reliability_diagrams.py \
        --model resnet50 \
        --dataset cifar100 \
        --seed 11 \
        --method sgc \
        --results-base-dir /path/to/checkpoints \
        --output-dir reliability_data

    # Run all experiments (no CLI args)
    python sgc_reliability_diagrams.py --run-all

    # Generate plots from existing data
    python sgc_reliability_diagrams.py --plot-only --output-dir reliability_data
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy.stats import rankdata
import faiss
import warnings
warnings.filterwarnings('ignore')

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Experiments.run_post_hoc_calibration import (
    PyTorchModelAdapter,
    get_data_loaders,
    load_trained_model,
    construct_model_path
)
from Experiments.compare_dac_geometric import (
    extract_and_aggregate_sgc_features,
    normalize_discovered_layers,
    filter_non_feature_layers,
    calculate_ece,
    calculate_accuracy,
    calculate_adaptive_ece
)
from Experiments.multi_layer_ensemble import discover_model_layers
from Experiments.compare_dac_geometric import extract_features_from_multiple_layers
from Calibrators.geometric_calibrator_new import GeometricCalibrator
from Calibrators.temperature_scaling import TemperatureScaling
from Calibrators.isotonic_regression import TopLabelIsotonicCalibrator
from Calibrators.density_aware_calibration import (
    DensityAwareCalibrator,
    extract_dac_features_memory_efficient,
    get_dac_target_layers,
    get_dac_k_value
)
from utils.coordinate_extraction import (
    discover_coordinate_space,
    plan_coordinate_extraction,
    extract_coordinate_features
)


# Trust Score classes (from run_k_ablation.py)
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
        
        print(f"Building Trust Score FAISS indices for {len(X_train)} samples, {self.num_classes} classes...")
        
        for label in range(self.num_classes):
            idx_same = np.where(y_train == label)[0]
            idx_other = np.where(y_train != label)[0]
            
            if len(idx_same) == 0 or len(idx_other) == 0:
                print(f"  Label {label}: insufficient samples for Trust Score (same={len(idx_same)}, other={len(idx_other)})")
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
                print(f"  Skipping Trust Score for label {label} (missing indices)")
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
        
        print(f"  Trust Score stats: min={np.min(ts):.4f}, max={np.max(ts):.4f}, "
              f"mean={np.mean(ts):.4f}, std={np.std(ts):.4f}")
        print(f"  NaN count: {np.sum(np.isnan(ts))}, Inf count: {np.sum(np.isinf(ts))}")
        
        ts = np.nan_to_num(ts, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Rank-based normalization
        print("📏 Storing sorted validation scores for rank-based normalization...")
        
        # Store sorted validation scores for rank-based normalization
        self.val_scores_sorted = np.sort(ts)
        n_val_samples = len(self.val_scores_sorted)
        
        print(f"   📊 Stored {n_val_samples} sorted validation scores")
        print(f"   📊 Score range: [{np.min(ts):.6f}, {np.max(ts):.6f}]")
        print(f"   📊 Mean score: {np.mean(ts):.6f}")
        
        # For fitting, we still need normalized values, so normalize using ranks
        ts_norm = rankdata(ts, method='average') / n_val_samples
        
        print(f"   ✅ Applied rank-based normalization")
        print(f"   📊 Normalized range: [{np.min(ts_norm):.6f}, {np.max(ts_norm):.6f}]")
        
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
        
        # Rank-based normalization (using stored validation scores)
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


def get_num_classes(dataset_name: str) -> int:
    """Get number of classes for a dataset."""
    dataset_map = {
        'cifar10': 10,
        'cifar100': 100,
        'tiny_imagenet': 200,
    }
    return dataset_map.get(dataset_name.lower(), 10)


def load_model_and_data(
    model_name: str,
    dataset_name: str,
    seed: int,
    base_dir: str = 'checkpoints',
    training_method: str = 'baseline_cross_entropy',
    batch_size: int = 128,
    device: str = 'cuda'
) -> Tuple[torch.nn.Module, Dict[str, Any], PyTorchModelAdapter]:
    """
    Load pretrained model and data loaders.
    """
    num_classes = get_num_classes(dataset_name)
    
    # Construct model path
    model_path = construct_model_path(
        base_dir=base_dir,
        method=training_method,
        dataset=dataset_name,
        model=model_name,
        seed=seed
    )
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load model
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = load_trained_model(
        model_path=model_path,
        model_name=model_name,
        num_classes=num_classes,
        device=device_obj,
        dataset=dataset_name
    )
    model.eval()
    
    # Load data
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        dataset=dataset_name,
        batch_size=batch_size,
        seed=seed
    )
    
    # Extract raw arrays
    def extract_raw(loader):
        images, labels = [], []
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                img, lbl = batch[0], batch[1]
            else:
                img, lbl = batch, None
            images.append(img.numpy() if isinstance(img, torch.Tensor) else img)
            if lbl is not None:
                labels.append(lbl.numpy() if isinstance(lbl, torch.Tensor) else lbl)
        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0) if labels else None
        return images, labels
    
    train_raw, train_labels = extract_raw(train_loader)
    val_raw, val_labels = extract_raw(val_loader)
    test_raw, test_labels = extract_raw(test_loader)
    
    # Create model adapter
    model_adapter = PyTorchModelAdapter(model, device=device_obj)
    
    data = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_raw': train_raw,
        'train_labels': train_labels,
        'val_raw': val_raw,
        'val_labels': val_labels,
        'test_raw': test_raw,
        'test_labels': test_labels,
    }
    
    return model, data, model_adapter


def get_extractable_layers(
    model: torch.nn.Module,
    dataset_name: str,
    device: torch.device
) -> List[str]:
    """Discover all extractable intermediate representations."""
    if dataset_name.lower() in ['tiny_imagenet', 'tinyimagenet']:
        input_shape = (1, 3, 64, 64)
    else:
        input_shape = (1, 3, 32, 32)
    
    discovered_layers = normalize_discovered_layers(
        model, discover_model_layers(model, device=device, input_shape=input_shape)
    )
    filtered_layers, _ = filter_non_feature_layers(discovered_layers, model)
    
    return [d['name'] for d in filtered_layers]


def sample_random_layers(
    all_layer_names: List[str],
    num_layers: int,
    seed: int
) -> List[str]:
    """Sample L layers uniformly at random."""
    np.random.seed(seed)
    if num_layers > len(all_layer_names):
        return all_layer_names
    return list(np.random.choice(all_layer_names, size=num_layers, replace=False))


def run_sgc_calibration(
    model: torch.nn.Module,
    model_adapter: PyTorchModelAdapter,
    data: Dict[str, Any],
    device: torch.device,
    dataset_name: str,
    L: int = 6,
    d: int = 256,
    seed: int = 42,
    batch_size: int = 128,
    pooling_mode: str = 'max'
) -> Dict[str, np.ndarray]:
    """Run SGC (RGCL) calibration."""
    all_layers = get_extractable_layers(model, dataset_name, device)
    selected_layers = sample_random_layers(all_layers, L, seed)
    print(f"Selected {len(selected_layers)} layers: {selected_layers[:3]}...")
    
    train_features, val_features, test_features, _ = extract_and_aggregate_sgc_features(
        model=model,
        layer_names=selected_layers,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        test_loader=data['test_loader'],
        device=device,
        target_dim=d,
        seed=seed,
        pooling_mode=pooling_mode
    )
    
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=data['train_labels'],
        library='fast_separation',
        auto_select_layer=False,
        device=str(device),
        scoring_method='separation'
    )
    
    geo_cal.fit(
        X_val_embed=val_features,
        y_val=data['val_labels'],
        X_val_original=data['val_raw'],
        fit_batch_size=batch_size
    )
    
    calibrated_probs = geo_cal.calibrate_batched(
        X_test_embed=test_features,
        X_test_original=data['test_raw'],
        batch_size=batch_size
    )
    
    max_probs = np.max(calibrated_probs, axis=1)
    predictions = np.argmax(calibrated_probs, axis=1)
    correct = (predictions == data['test_labels']).astype(float)
    
    return {
        'max_probs': max_probs,
        'correct': correct,
        'predictions': predictions,
        'labels': data['test_labels']
    }


def run_sgc_lite_calibration(
    model: torch.nn.Module,
    model_adapter: PyTorchModelAdapter,
    data: Dict[str, Any],
    device: torch.device,
    K: int = 256,
    seed: int = 42,
    batch_size: int = 128
) -> Dict[str, np.ndarray]:
    """Run SGC-Lite (RGCC) calibration."""
    if data['test_raw'].shape[-1] == 64:
        input_shape = (1, 3, 64, 64)
    else:
        input_shape = (1, 3, 32, 32)
    
    layer_map, total_size = discover_coordinate_space(
        model=model,
        input_shape=input_shape,
        device=str(device)
    )
    
    sampling_plan, global_order = plan_coordinate_extraction(
        total_size=total_size,
        num_coordinates=K,
        layer_map=layer_map,
        seed=seed
    )
    
    # Extract coordinates from each loader
    train_coords = extract_coordinate_features(
        model=model,
        data_loader=data['train_loader'],
        sampling_plan=sampling_plan,
        layer_map=layer_map,
        global_order=global_order,
        device=str(device)
    ).cpu().numpy()
    
    val_coords = extract_coordinate_features(
        model=model,
        data_loader=data['val_loader'],
        sampling_plan=sampling_plan,
        layer_map=layer_map,
        global_order=global_order,
        device=str(device)
    ).cpu().numpy()
    
    test_coords = extract_coordinate_features(
        model=model,
        data_loader=data['test_loader'],
        sampling_plan=sampling_plan,
        layer_map=layer_map,
        global_order=global_order,
        device=str(device)
    ).cpu().numpy()
    
    train_coords = normalize(train_coords, norm='l2', axis=1)
    val_coords = normalize(val_coords, norm='l2', axis=1)
    test_coords = normalize(test_coords, norm='l2', axis=1)
    
    geo_cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_coords,
        y_train=data['train_labels'],
        library='fast_separation',
        auto_select_layer=False,
        device=str(device),
        scoring_method='separation'
    )
    
    geo_cal.fit(
        X_val_embed=val_coords,
        y_val=data['val_labels'],
        X_val_original=data['val_raw'],
        fit_batch_size=batch_size
    )
    
    calibrated_probs = geo_cal.calibrate_batched(
        X_test_embed=test_coords,
        X_test_original=data['test_raw'],
        batch_size=batch_size
    )
    
    max_probs = np.max(calibrated_probs, axis=1)
    predictions = np.argmax(calibrated_probs, axis=1)
    correct = (predictions == data['test_labels']).astype(float)
    
    return {
        'max_probs': max_probs,
        'correct': correct,
        'predictions': predictions,
        'labels': data['test_labels']
    }


def get_uncalibrated_confidences(
    model_adapter: PyTorchModelAdapter,
    data: Dict[str, Any],
    batch_size: int = 128
) -> Dict[str, np.ndarray]:
    """Return max softmax probabilities (uncalibrated baseline)."""
    probs = model_adapter.predict_proba(data['test_raw'], batch_size=batch_size)
    max_probs = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == data['test_labels']).astype(float)
    
    return {
        'max_probs': max_probs,
        'correct': correct,
        'predictions': predictions,
        'labels': data['test_labels']
    }


def run_temperature_scaling(
    model_adapter: PyTorchModelAdapter,
    data: Dict[str, Any],
    batch_size: int = 128
) -> Dict[str, np.ndarray]:
    """Fit temperature scaling on validation, apply to test."""
    val_logits = model_adapter.predict_logits(data['val_raw'], batch_size=batch_size)
    test_logits = model_adapter.predict_logits(data['test_raw'], batch_size=batch_size)
    
    temp_scaler = TemperatureScaling(cross_validate='ece')
    temp_scaler.fit(logits=val_logits, labels=data['val_labels'])
    
    calibrated_probs = temp_scaler.calibrate(logits=test_logits)
    
    max_probs = np.max(calibrated_probs, axis=1)
    predictions = np.argmax(calibrated_probs, axis=1)
    correct = (predictions == data['test_labels']).astype(float)
    
    return {
        'max_probs': max_probs,
        'correct': correct,
        'predictions': predictions,
        'labels': data['test_labels']
    }


def run_top_label_isotonic(
    model_adapter: PyTorchModelAdapter,
    data: Dict[str, Any],
    batch_size: int = 128
) -> Dict[str, np.ndarray]:
    """Fit top-label isotonic regression on validation, apply to test."""
    val_logits = model_adapter.predict_logits(data['val_raw'], batch_size=batch_size)
    test_logits = model_adapter.predict_logits(data['test_raw'], batch_size=batch_size)
    
    top_label_iso = TopLabelIsotonicCalibrator()
    top_label_iso.fit(logits=val_logits, labels=data['val_labels'])
    
    calibrated_probs = top_label_iso.calibrate(logits=test_logits)
    
    max_probs = np.max(calibrated_probs, axis=1)
    predictions = np.argmax(calibrated_probs, axis=1)
    correct = (predictions == data['test_labels']).astype(float)
    
    return {
        'max_probs': max_probs,
        'correct': correct,
        'predictions': predictions,
        'labels': data['test_labels']
    }


def run_sgc_trust_score_calibration(
    model: torch.nn.Module,
    model_adapter: PyTorchModelAdapter,
    data: Dict[str, Any],
    device: torch.device,
    dataset_name: str,
    L: int = 6,
    d: int = 256,
    seed: int = 42,
    batch_size: int = 128,
    pooling_mode: str = 'max'
) -> Dict[str, np.ndarray]:
    """Run SGC with Trust Score calibration (uses SGC features but Trust Score instead of separation)."""
    all_layers = get_extractable_layers(model, dataset_name, device)
    selected_layers = sample_random_layers(all_layers, L, seed)
    print(f"Selected {len(selected_layers)} layers: {selected_layers[:3]}...")
    
    # Extract SGC features (same as regular SGC)
    train_features, val_features, test_features, _ = extract_and_aggregate_sgc_features(
        model=model,
        layer_names=selected_layers,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        test_loader=data['test_loader'],
        device=device,
        target_dim=d,
        seed=seed,
        pooling_mode=pooling_mode
    )
    
    # Get model predictions and logits for validation and test
    val_probs = model_adapter.predict_proba(data['val_raw'], batch_size=batch_size)
    val_predictions = np.argmax(val_probs, axis=1)
    val_logits = model_adapter.predict_logits(data['val_raw'], batch_size=batch_size)
    
    test_probs = model_adapter.predict_proba(data['test_raw'], batch_size=batch_size)
    test_predictions = np.argmax(test_probs, axis=1)
    test_logits = model_adapter.predict_logits(data['test_raw'], batch_size=batch_size)
    
    # Fit Trust Score calibrator
    ts_calibrator = TrustScoreCalibrator(use_gpu=(device.type == 'cuda'))
    ts_calibrator.fit(
        X_train_features=train_features,
        y_train=data['train_labels'],
        X_val_features=val_features,
        y_val=data['val_labels'],
        val_predictions=val_predictions
    )
    
    # Calibrate test set
    calibrated_probs = ts_calibrator.calibrate(
        X_test_features=test_features,
        predictions=test_predictions,
        logits=test_logits
    )
    
    max_probs = np.max(calibrated_probs, axis=1)
    predictions = np.argmax(calibrated_probs, axis=1)
    correct = (predictions == data['test_labels']).astype(float)
    
    return {
        'max_probs': max_probs,
        'correct': correct,
        'predictions': predictions,
        'labels': data['test_labels']
    }


def run_dac_calibration(
    model: torch.nn.Module,
    model_name: str,
    model_adapter: PyTorchModelAdapter,
    data: Dict[str, Any],
    device: torch.device,
    dataset_name: str,
    batch_size: int = 128
) -> Dict[str, np.ndarray]:
    """Run DAC (Density-Aware Calibration) using DAC's standard layer selection."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Get DAC target layers
    layer_names = get_dac_target_layers(model_name, model)
    print(f"Selected {len(layer_names)} DAC layers: {layer_names[:3]}...")
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(data['train_raw']), torch.from_numpy(data['train_labels'])),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(data['val_raw']), torch.from_numpy(data['val_labels'])),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(data['test_raw']), torch.from_numpy(data['test_labels'])),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Extract logits
    def get_logits_and_labels(loader):
        model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 2:
                    data_batch, labels = batch
                else:
                    data_batch = batch[0]
                    labels = torch.zeros(len(data_batch), dtype=torch.long)
                
                data_batch = data_batch.to(device)
                logits = model(data_batch)
                logits_list.append(logits.cpu())
                labels_list.append(labels)
        return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)
    
    train_logits, train_labels_torch = get_logits_and_labels(train_loader)
    val_logits, val_labels_torch = get_logits_and_labels(val_loader)
    test_logits, test_labels_torch = get_logits_and_labels(test_loader)
    
    val_labels = val_labels_torch.numpy() if isinstance(val_labels_torch, torch.Tensor) else val_labels_torch
    test_labels = test_labels_torch.numpy() if isinstance(test_labels_torch, torch.Tensor) else test_labels_torch
    
    # Extract features from each layer using DAC preprocessing
    train_features_list = []
    val_features_list = []
    test_features_list = []
    
    for layer_name in layer_names:
        train_feats, _, _ = extract_dac_features_memory_efficient(
            model, train_loader, [layer_name], device
        )
        val_feats, _, _ = extract_dac_features_memory_efficient(
            model, val_loader, [layer_name], device
        )
        test_feats, _, _ = extract_dac_features_memory_efficient(
            model, test_loader, [layer_name], device
        )
        
        train_features_list.append(train_feats[0])
        val_features_list.append(val_feats[0])
        test_features_list.append(test_feats[0])
    
    # Add logits as final layer
    train_features_for_dac = train_features_list + [train_logits]
    val_features_for_dac = val_features_list + [val_logits]
    test_features_for_dac = test_features_list + [test_logits]
    
    # Convert to numpy for DAC
    train_features_for_dac = [f.numpy() if isinstance(f, torch.Tensor) else f for f in train_features_for_dac]
    val_features_for_dac = [f.numpy() if isinstance(f, torch.Tensor) else f for f in val_features_for_dac]
    test_features_for_dac = [f.numpy() if isinstance(f, torch.Tensor) else f for f in test_features_for_dac]
    
    # Fit DAC
    k_value = get_dac_k_value(dataset_name)
    dac = DensityAwareCalibrator(k=k_value, use_gpu=(device.type == 'cuda'))
    dac.fit(
        train_features_for_dac,
        val_features_for_dac,
        val_logits.numpy(),
        val_labels
    )
    
    # Calibrate test set
    calibrated_probs = dac.calibrate(
        test_features_for_dac,
        test_logits.numpy()
    )
    
    max_probs = np.max(calibrated_probs, axis=1)
    predictions = np.argmax(calibrated_probs, axis=1)
    correct = (predictions == data['test_labels']).astype(float)
    
    return {
        'max_probs': max_probs,
        'correct': correct,
        'predictions': predictions,
        'labels': data['test_labels']
    }


def run_single_experiment(
    model_name: str,
    dataset_name: str,
    seed: int,
    method: str,
    results_base_dir: str,
    training_method: str = 'baseline_cross_entropy',
    batch_size: int = 128,
    sgc_L: int = 6,
    sgc_d: int = 256,
    sgc_lite_K: int = 256,
    device: str = 'cuda'
) -> Optional[Dict[str, np.ndarray]]:
    """Run one experiment configuration."""
    try:
        print(f"\n{'='*60}")
        print(f"Running: {model_name} | {dataset_name} | seed={seed} | {method}")
        print(f"{'='*60}")
        
        model, data, model_adapter = load_model_and_data(
            model_name=model_name,
            dataset_name=dataset_name,
            seed=seed,
            base_dir=results_base_dir,
            training_method=training_method,
            batch_size=batch_size,
            device=device
        )
        
        device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if method == 'uncalibrated':
            results = get_uncalibrated_confidences(model_adapter, data, batch_size)
        elif method == 'temperature_scaling':
            results = run_temperature_scaling(model_adapter, data, batch_size)
        elif method == 'top_label_isotonic':
            results = run_top_label_isotonic(model_adapter, data, batch_size)
        elif method == 'sgc':
            results = run_sgc_calibration(
                model=model,
                model_adapter=model_adapter,
                data=data,
                device=device_obj,
                dataset_name=dataset_name,
                L=sgc_L,
                d=sgc_d,
                seed=seed,
                batch_size=batch_size
            )
        elif method == 'sgc_lite':
            results = run_sgc_lite_calibration(
                model=model,
                model_adapter=model_adapter,
                data=data,
                device=device_obj,
                K=sgc_lite_K,
                seed=seed,
                batch_size=batch_size
            )
        elif method == 'sgc_trust_score':
            results = run_sgc_trust_score_calibration(
                model=model,
                model_adapter=model_adapter,
                data=data,
                device=device_obj,
                dataset_name=dataset_name,
                L=sgc_L,
                d=sgc_d,
                seed=seed,
                batch_size=batch_size
            )
        elif method == 'dac':
            results = run_dac_calibration(
                model=model,
                model_name=model_name,
                model_adapter=model_adapter,
                data=data,
                device=device_obj,
                dataset_name=dataset_name,
                batch_size=batch_size
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute ECE for logging
        num_classes = get_num_classes(dataset_name)
        ece = calculate_ece(
            np.eye(num_classes)[results['predictions']] * results['max_probs'][:, None],
            results['labels']
        )
        acc = calculate_accuracy(
            np.eye(num_classes)[results['predictions']] * results['max_probs'][:, None],
            results['labels']
        )
        print(f"ECE: {ece:.4f}, Accuracy: {acc:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_and_group_results(results_dir: Path) -> Dict[Tuple[str, str, str], List[Dict]]:
    """Group saved .npz files by (model, dataset, method)."""
    grouped = defaultdict(list)
    
    for npz_file in results_dir.glob("*_per_sample.npz"):
        parts = npz_file.stem.replace('_per_sample', '').split('_')
        
        seed_idx = None
        for i, part in enumerate(parts):
            if part.startswith('seed'):
                seed_idx = i
                break
        
        if seed_idx is None:
            continue
        
        model_parts = parts[:seed_idx]
        seed_str = parts[seed_idx]
        method_parts = parts[seed_idx+1:]
        
        # Identify dataset
        # Sort by length (longest first) to avoid substring matching issues
        # e.g., 'cifar100' should match before 'cifar10'
        known_datasets = ['tiny_imagenet', 'tinyimagenet', 'cifar100', 'cifar10']
        model_name = model_parts[0]
        dataset_name = None
        
        for ds in known_datasets:
            if ds in '_'.join(model_parts):
                dataset_name = ds
                break
        
        if dataset_name is None and len(model_parts) > 1:
            dataset_name = '_'.join(model_parts[1:])
        
        method_name = '_'.join(method_parts)
        
        data = np.load(npz_file)
        grouped[(model_name, dataset_name, method_name)].append({
            'seed': seed_str,
            'max_probs': data['max_probs'],
            'correct': data['correct'],
            'predictions': data['predictions'],
            'labels': data['labels']
        })
    
    return dict(grouped)


def compute_binned_statistics(
    grouped_data: List[Dict],
    n_bins: int = 10,
    bin_type: str = 'equal_width'
) -> Dict[str, np.ndarray]:
    """Compute per-bin statistics aggregated across seeds."""
    all_max_probs = [d['max_probs'] for d in grouped_data]
    all_correct = [d['correct'] for d in grouped_data]
    
    # Calculate adaptive ECE for each seed
    per_seed_adaptive_eces = []
    for data_dict in grouped_data:
        max_probs = data_dict['max_probs']
        predictions = data_dict['predictions']
        labels = data_dict['labels']
        
        # Reconstruct full probability distribution from max_probs and predictions
        # Infer num_classes from labels
        num_classes = int(np.max(labels)) + 1
        num_samples = len(max_probs)
        
        # Create probability distribution where predicted class gets max_probs
        # and other classes get equal share of remaining probability
        probs = np.zeros((num_samples, num_classes))
        for i in range(num_samples):
            probs[i, predictions[i]] = max_probs[i]
            remaining_prob = 1.0 - max_probs[i]
            if num_classes > 1:
                other_prob = remaining_prob / (num_classes - 1)
                for j in range(num_classes):
                    if j != predictions[i]:
                        probs[i, j] = other_prob
        
        adaptive_ece = calculate_adaptive_ece(probs, labels, n_bins=n_bins)
        per_seed_adaptive_eces.append(adaptive_ece)
    
    if bin_type == 'equal_width':
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif bin_type == 'adaptive' or bin_type == 'equal_frequency':
        # Adaptive binning: equal number of samples per bin
        all_probs_flat = np.concatenate(all_max_probs)
        sorted_probs = np.sort(all_probs_flat)
        n_samples = len(sorted_probs)
        bin_boundaries = np.interp(
            np.linspace(0, n_samples, n_bins + 1),
            np.arange(n_samples),
            sorted_probs
        )
        bin_edges = bin_boundaries
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:  # percentile-based
        all_probs_flat = np.concatenate(all_max_probs)
        bin_edges = np.percentile(all_probs_flat, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    per_seed_stats = []
    per_seed_eces = []
    
    for max_probs, correct in zip(all_max_probs, all_correct):
        bin_indices = np.digitize(max_probs, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_accs = []
        bin_confs = []
        bin_counts = []
        
        for b in range(n_bins):
            mask = bin_indices == b
            if np.any(mask):
                bin_accs.append(np.mean(correct[mask]))
                bin_confs.append(np.mean(max_probs[mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_accs.append(np.nan)
                bin_confs.append(np.nan)
                bin_counts.append(0)
        
        per_seed_stats.append({
            'acc': np.array(bin_accs),
            'conf': np.array(bin_confs),
            'count': np.array(bin_counts)
        })
        
        ece = np.nansum(np.array(bin_counts) / len(max_probs) * np.abs(np.array(bin_accs) - np.array(bin_confs)))
        per_seed_eces.append(ece)
    
    mean_acc = np.nanmean([s['acc'] for s in per_seed_stats], axis=0)
    std_acc = np.nanstd([s['acc'] for s in per_seed_stats], axis=0)
    mean_conf = np.nanmean([s['conf'] for s in per_seed_stats], axis=0)
    sample_counts = np.mean([s['count'] for s in per_seed_stats], axis=0)
    
    ece_mean = np.mean(per_seed_eces)
    ece_std = np.std(per_seed_eces)
    adaptive_ece_mean = np.mean(per_seed_adaptive_eces)
    adaptive_ece_std = np.std(per_seed_adaptive_eces)
    
    return {
        'bin_centers': bin_centers,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'mean_conf': mean_conf,
        'sample_counts': sample_counts,
        'ece_mean': ece_mean,
        'ece_std': ece_std,
        'adaptive_ece_mean': adaptive_ece_mean,
        'adaptive_ece_std': adaptive_ece_std,
        'n_seeds': len(grouped_data)
    }


def plot_single_method_reliability(
    stats: Dict[str, np.ndarray],
    title: str,
    save_path: Path,
    figsize: Tuple[float, float] = (3.5, 3.5)
) -> None:
    """Publication-quality reliability diagram for one method."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
    
    bin_centers = stats['bin_centers']
    mean_acc = stats['mean_acc']
    std_acc = stats['std_acc']
    mean_conf = stats['mean_conf']
    sample_counts = stats['sample_counts']
    ece_mean = stats['ece_mean']
    ece_std = stats['ece_std']
    adaptive_ece_mean = stats.get('adaptive_ece_mean', None)
    adaptive_ece_std = stats.get('adaptive_ece_std', None)
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration', alpha=0.7)
    ax1.errorbar(
        mean_conf, mean_acc,
        yerr=std_acc,
        fmt='o-',
        capsize=3,
        capthick=1.5,
        linewidth=1.5,
        markersize=5,
        label='Model'
    )
    ax1.set_ylabel('Empirical Accuracy', fontsize=11)
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    if adaptive_ece_mean is not None:
        title_with_ece = f"{title}\nECE = {ece_mean:.2%} ± {ece_std:.2%}, Adaptive ECE = {adaptive_ece_mean:.2%} ± {adaptive_ece_std:.2%}"
    else:
        title_with_ece = f"{title}\nECE = {ece_mean:.2%} ± {ece_std:.2%}"
    ax1.set_title(title_with_ece, fontsize=10)
    
    ax2.bar(bin_centers, sample_counts, width=0.08, alpha=0.6, color='gray')
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_xlabel('Confidence', fontsize=11)
    ax2.set_xlim([0, 1])
    
    plt.tight_layout()
    
    plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path.with_suffix('.png')} and {save_path.with_suffix('.pdf')}")


def plot_method_comparison(
    all_stats: Dict[str, Dict[str, np.ndarray]],
    model: str,
    dataset: str,
    save_path: Path,
    figsize: Tuple[float, float] = (14, 3.5)
) -> None:
    """Side-by-side comparison of calibration methods."""
    method_order = ['uncalibrated', 'temperature_scaling', 'top_label_isotonic', 'sgc', 'sgc_lite', 'sgc_trust_score', 'dac']
    method_labels = {
        'uncalibrated': 'Uncalibrated',
        'temperature_scaling': 'Temperature Scaling',
        'top_label_isotonic': 'Top-Label Isotonic',
        'sgc': 'SGC (Ours)',
        'sgc_lite': 'SGC-Lite (Ours)',
        'sgc_trust_score': 'SGC + Trust Score',
        'dac': 'DAC'
    }
    
    available_methods = [m for m in method_order if m in all_stats]
    if not available_methods:
        print(f"No methods available for {model}/{dataset}")
        return
    
    fig, axes = plt.subplots(1, len(available_methods), figsize=(3.5 * len(available_methods), 3.5), sharey=True)
    
    if len(available_methods) == 1:
        axes = [axes]
    
    for idx, method in enumerate(available_methods):
        stats = all_stats[method]
        ax = axes[idx]
        
        mean_acc = stats['mean_acc']
        std_acc = stats['std_acc']
        mean_conf = stats['mean_conf']
        ece_mean = stats['ece_mean']
        ece_std = stats['ece_std']
        adaptive_ece_mean = stats.get('adaptive_ece_mean', None)
        adaptive_ece_std = stats.get('adaptive_ece_std', None)
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        ax.errorbar(
            mean_conf, mean_acc,
            yerr=std_acc,
            fmt='o-',
            capsize=2,
            capthick=1,
            linewidth=1.2,
            markersize=4
        )
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.set_ylabel('Empirical Accuracy', fontsize=11)
        ax.set_xlabel('Confidence', fontsize=10)
        
        if adaptive_ece_mean is not None:
            title = f"{method_labels.get(method, method)}\nECE = {ece_mean:.2%} ± {ece_std:.2%}\nAdaptive ECE = {adaptive_ece_mean:.2%} ± {adaptive_ece_std:.2%}"
        else:
            title = f"{method_labels.get(method, method)}\nECE = {ece_mean:.2%} ± {ece_std:.2%}"
        ax.set_title(title, fontsize=9)
    
    plt.tight_layout()
    
    plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison: {save_path.with_suffix('.png')} and {save_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(
        description='SGC Reliability Diagram Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode selection
    parser.add_argument('--run-all', action='store_true',
                        help="Run all experiments (default CONFIG)")
    parser.add_argument('--plot-only', action='store_true',
                        help="Only generate plots from existing data")
    
    # Single experiment parameters
    parser.add_argument('--model', type=str, default=None,
                        help="Model name (e.g., resnet50)")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Dataset name (e.g., cifar100)")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed")
    parser.add_argument('--method', type=str, default=None,
                        choices=['sgc', 'sgc_lite', 'sgc_trust_score', 'temperature_scaling', 'top_label_isotonic', 'uncalibrated', 'dac'],
                        help="Calibration method")
    
    # Paths
    parser.add_argument('--results-base-dir', type=str, default='checkpoints',
                        help="Base directory with pre-trained model checkpoints")
    parser.add_argument('--output-dir', type=str, default='reliability_data',
                        help="Output directory for per-sample data")
    parser.add_argument('--figures-dir', type=str, default='figures',
                        help="Output directory for figures")
    parser.add_argument('--training-method', type=str, default='baseline_cross_entropy',
                        help="Training method used for models")
    
    # SGC parameters
    parser.add_argument('--sgc-L', type=int, default=6,
                        dest='sgc_L',
                        help="Number of layers to sample for SGC")
    parser.add_argument('--sgc-d', type=int, default=256,
                        dest='sgc_d',
                        help="Target dimension for SGC projection")
    parser.add_argument('--sgc-lite-K', type=int, default=256,
                        dest='sgc_lite_K',
                        help="Number of coordinates for SGC-Lite")
    
    # Technical settings
    parser.add_argument('--batch-size', type=int, default=128,
                        help="Batch size for data processing")
    parser.add_argument('--n-bins', type=int, default=10,
                        help="Number of bins for reliability diagrams")
    parser.add_argument('--bin-type', type=str, default='equal_width',
                        choices=['equal_width', 'adaptive', 'equal_frequency'],
                        help="Binning strategy: 'equal_width' (default) or 'adaptive'/'equal_frequency'")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Mode 1: Plot only
    if args.plot_only:
        print("Plot-only mode: generating figures from existing data...")
        grouped_results = load_and_group_results(output_dir)
        
        if not grouped_results:
            print(f"No data found in {output_dir}")
            return 1
        
        print(f"Loaded {len(grouped_results)} experiment groups")
        
        # Generate individual plots
        for (model, dataset, method), data_list in grouped_results.items():
            if len(data_list) == 0:
                continue
            
            stats = compute_binned_statistics(data_list, n_bins=args.n_bins, bin_type=args.bin_type)
            title = f"{model.upper()} on {dataset.upper()}\n{method.upper()}"
            bin_suffix = "_adaptive" if args.bin_type in ['adaptive', 'equal_frequency'] else ""
            save_path = figures_dir / f"{model}_{dataset}_{method}_aggregated{bin_suffix}"
            plot_single_method_reliability(stats, title, save_path)
        
        # Generate comparison plots
        models = set(k[0] for k in grouped_results.keys())
        datasets = set(k[1] for k in grouped_results.keys())
        
        for model in models:
            for dataset in datasets:
                all_stats = {}
                for method in ['uncalibrated', 'temperature_scaling', 'top_label_isotonic', 'sgc', 'sgc_lite', 'sgc_trust_score', 'dac']:
                    key = (model, dataset, method)
                    if key in grouped_results and len(grouped_results[key]) > 0:
                        all_stats[method] = compute_binned_statistics(
                            grouped_results[key],
                            n_bins=args.n_bins,
                            bin_type=args.bin_type
                        )
                
                if len(all_stats) > 0:
                    bin_suffix = "_adaptive" if args.bin_type in ['adaptive', 'equal_frequency'] else ""
                    save_path = figures_dir / f"{model}_{dataset}_comparison{bin_suffix}"
                    plot_method_comparison(all_stats, model, dataset, save_path)
        
        print(f"\nAll plots saved to: {figures_dir}")
        return 0
    
    # Mode 2: Single experiment (CLI args provided)
    if args.model and args.dataset and args.seed is not None and args.method:
        print(f"Running single experiment: {args.model}/{args.dataset}/seed{args.seed}/{args.method}")
        
        results = run_single_experiment(
            model_name=args.model,
            dataset_name=args.dataset,
            seed=args.seed,
            method=args.method,
            results_base_dir=args.results_base_dir,
            training_method=args.training_method,
            batch_size=args.batch_size,
            sgc_L=args.sgc_L,
            sgc_d=args.sgc_d,
            sgc_lite_K=args.sgc_lite_K,
            device=args.device
        )
        
        if results is not None:
            filename = output_dir / f"{args.model}_{args.dataset}_seed{args.seed}_{args.method}_per_sample.npz"
            np.savez(
                filename,
                max_probs=results['max_probs'],
                correct=results['correct'],
                predictions=results['predictions'],
                labels=results['labels']
            )
            print(f"Saved: {filename}")
            return 0
        else:
            print("Experiment failed")
            return 1
    
    # Mode 3: Run all experiments
    if args.run_all or (not args.model and not args.plot_only):
        print("Running all experiments with default configuration...")
        
        CONFIG = {
            'models': ['resnet50', 'resnet101', 'densenet121'],
            'datasets': ['cifar10', 'cifar100', 'tiny_imagenet'],
            'seeds': [11, 12, 13, 14, 15, 16],
            'methods': ['uncalibrated', 'temperature_scaling', 'top_label_isotonic', 'sgc', 'sgc_lite', 'sgc_trust_score', 'dac'],
        }
        
        for model_name in CONFIG['models']:
            for dataset_name in CONFIG['datasets']:
                for seed in CONFIG['seeds']:
                    for method in CONFIG['methods']:
                        filename = output_dir / f"{model_name}_{dataset_name}_seed{seed}_{method}_per_sample.npz"
                        
                        if filename.exists():
                            print(f"Skipping {filename.name} (already exists)")
                            continue
                        
                        results = run_single_experiment(
                            model_name=model_name,
                            dataset_name=dataset_name,
                            seed=seed,
                            method=method,
                            results_base_dir=args.results_base_dir,
                            training_method=args.training_method,
                            batch_size=args.batch_size,
                            sgc_L=args.sgc_L,
                            sgc_d=args.sgc_d,
                            sgc_lite_K=args.sgc_lite_K,
                            device=args.device
                        )
                        
                        if results is not None:
                            np.savez(
                                filename,
                                max_probs=results['max_probs'],
                                correct=results['correct'],
                                predictions=results['predictions'],
                                labels=results['labels']
                            )
                            print(f"Saved: {filename}")
                        else:
                            print(f"Failed: {filename}")
        
        print(f"\nAll experiments complete. Data saved to: {output_dir}")
        return 0
    
    # No valid mode selected
    parser.print_help()
    return 1


if __name__ == '__main__':
    exit(main())