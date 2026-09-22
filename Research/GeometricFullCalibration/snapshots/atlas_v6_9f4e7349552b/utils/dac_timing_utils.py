"""
Timing utilities for fair comparison between DAC and RGCL/RGCC.

This module provides consistent timing measurement across all methods,
ensuring apples-to-apples comparison for Figure 2.

End-to-end timing includes:
1. Feature extraction (forward pass through model + layer extraction)
2. Offline fitting (kNN index + calibration mapping)
3. Online calibration (kNN query + probability computation)
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Try to import logger
try:
    from utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


@dataclass
class TimingResult:
    """Structured timing results for a calibration method."""
    method_name: str
    
    # Feature extraction (offline, once per dataset)
    extraction_time_s: float = 0.0
    
    # Fitting/training (offline, once per dataset)
    fit_time_s: float = 0.0
    
    # Calibration (online, per test batch)
    calibrate_time_s: float = 0.0
    
    # Derived metrics
    total_offline_time_s: float = 0.0  # extraction + fit
    total_time_s: float = 0.0  # extraction + fit + calibrate
    
    # Throughput
    n_test_samples: int = 0
    throughput_samples_per_sec: float = 0.0
    
    # Sub-timings (optional, for detailed analysis)
    sub_timings: Dict[str, float] = field(default_factory=dict)
    
    def compute_derived(self):
        """Compute derived timing metrics."""
        self.total_offline_time_s = self.extraction_time_s + self.fit_time_s
        self.total_time_s = self.total_offline_time_s + self.calibrate_time_s
        if self.calibrate_time_s > 0 and self.n_test_samples > 0:
            self.throughput_samples_per_sec = self.n_test_samples / self.calibrate_time_s
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'method': self.method_name,
            'extraction_time_s': self.extraction_time_s,
            'fit_time_s': self.fit_time_s,
            'calibrate_time_s': self.calibrate_time_s,
            'total_offline_time_s': self.total_offline_time_s,
            'total_time_s': self.total_time_s,
            'n_test_samples': self.n_test_samples,
            'throughput_samples_per_sec': self.throughput_samples_per_sec,
            'sub_timings': self.sub_timings,
        }
    
    def __str__(self) -> str:
        return (f"{self.method_name}: "
                f"extract={self.extraction_time_s:.2f}s, "
                f"fit={self.fit_time_s:.2f}s, "
                f"calibrate={self.calibrate_time_s:.2f}s, "
                f"total={self.total_time_s:.2f}s, "
                f"throughput={self.throughput_samples_per_sec:.1f}/s")


class GPUTimer:
    """Context manager for GPU-synchronized timing."""
    
    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.start_time = 0.0
        self.elapsed = 0.0
    
    def __enter__(self):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start_time


def time_dac_full_pipeline(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    test_loader,
    layer_names: List[str],
    k: int,
    device: torch.device,
    backend: str = 'torch',
    extract_dac_features_fn=None,
) -> Tuple[TimingResult, np.ndarray, Dict[str, Any]]:
    """
    Time the full DAC pipeline including feature extraction.
    
    This provides timing that's comparable to RGCL/RGCC end-to-end timing.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data  
        test_loader: DataLoader for test data
        layer_names: List of layer names for DAC feature extraction
        k: Number of nearest neighbors
        device: PyTorch device
        backend: 'torch' or 'faiss'
        extract_dac_features_fn: Function to extract DAC features (imported from your module)
        
    Returns:
        Tuple of (TimingResult, calibrated_probs, metrics_dict)
    """
    timing = TimingResult(method_name=f"DAC ({backend})")
    
    # Import the appropriate DAC calibrator
    if backend == 'torch':
        from Calibrators.density_aware_calibration_pytorch import create_dac_pytorch
        dac = create_dac_pytorch(k=k, use_gpu=(device.type == 'cuda'))
    else:
        from Calibrators.density_aware_calibration import DensityAwareCalibrator
        dac = DensityAwareCalibrator(k=k, use_gpu=(device.type == 'cuda'))
    
    # If no extraction function provided, try to import it
    if extract_dac_features_fn is None:
        from Calibrators.density_aware_calibration import extract_dac_features
        extract_dac_features_fn = extract_dac_features
    
    # =========================================================================
    # 1. FEATURE EXTRACTION (timed)
    # =========================================================================
    logger.info(f"Extracting DAC features ({backend} backend)...")
    
    with GPUTimer() as extract_timer:
        # Extract training features
        train_feats, train_logits, train_labels = extract_dac_features_fn(
            model, train_loader, layer_names, device
        )
        
        # Extract validation features
        val_feats, val_logits, val_labels = extract_dac_features_fn(
            model, val_loader, layer_names, device
        )
        
        # Extract test features
        test_feats, test_logits, test_labels = extract_dac_features_fn(
            model, test_loader, layer_names, device
        )
    
    timing.extraction_time_s = extract_timer.elapsed
    timing.sub_timings['extraction'] = extract_timer.elapsed
    logger.info(f"  Feature extraction: {timing.extraction_time_s:.2f}s")
    
    # Convert to numpy for DAC
    train_feats_np = [f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in train_feats]
    val_feats_np = [f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in val_feats]
    test_feats_np = [f.cpu().numpy() if isinstance(f, torch.Tensor) else f for f in test_feats]
    
    val_logits_np = val_logits.cpu().numpy() if isinstance(val_logits, torch.Tensor) else val_logits
    test_logits_np = test_logits.cpu().numpy() if isinstance(test_logits, torch.Tensor) else test_logits
    val_labels_np = val_labels.cpu().numpy() if isinstance(val_labels, torch.Tensor) else val_labels
    test_labels_np = test_labels.cpu().numpy() if isinstance(test_labels, torch.Tensor) else test_labels
    
    # =========================================================================
    # 2. FITTING (timed)
    # =========================================================================
    logger.info(f"Fitting DAC ({backend} backend)...")
    
    with GPUTimer() as fit_timer:
        dac.fit(
            train_features_list=train_feats_np,
            val_features_list=val_feats_np,
            val_logits=val_logits_np,
            val_labels=val_labels_np,
        )
    
    timing.fit_time_s = fit_timer.elapsed
    timing.sub_timings['fit'] = fit_timer.elapsed
    logger.info(f"  Fit time: {timing.fit_time_s:.2f}s")
    
    # Get sub-timings from PyTorch DAC if available
    if hasattr(dac, 'get_timing'):
        dac_timing = dac.get_timing()
        timing.sub_timings.update({f'fit_{k}': v for k, v in dac_timing.items()})
    
    # =========================================================================
    # 3. CALIBRATION (timed)
    # =========================================================================
    logger.info(f"Calibrating test set ({backend} backend)...")
    timing.n_test_samples = len(test_labels_np)
    
    with GPUTimer() as cal_timer:
        calibrated_probs = dac.calibrate(test_feats_np, test_logits_np)
    
    timing.calibrate_time_s = cal_timer.elapsed
    timing.sub_timings['calibrate'] = cal_timer.elapsed
    
    # Compute derived metrics
    timing.compute_derived()
    
    logger.info(f"  Calibration time: {timing.calibrate_time_s:.4f}s")
    logger.info(f"  Throughput: {timing.throughput_samples_per_sec:.1f} samples/sec")
    logger.info(f"  Total (E2E): {timing.total_time_s:.2f}s")
    
    # =========================================================================
    # 4. COMPUTE METRICS
    # =========================================================================
    # Calculate calibration metrics
    predictions = np.argmax(calibrated_probs, axis=1)
    accuracy = np.mean(predictions == test_labels_np)
    
    # ECE calculation (import from your metrics module or use simple version)
    try:
        from Metrics.metrics import expected_calibration_error
        ece = expected_calibration_error(calibrated_probs, test_labels_np, n_bins=15)
    except ImportError:
        # Simple ECE calculation
        ece = _simple_ece(calibrated_probs, test_labels_np, n_bins=15)
    
    metrics = {
        'ece': float(ece),
        'accuracy': float(accuracy),
        'weights': dac.weights.tolist() if hasattr(dac, 'weights') else None,
    }
    
    return timing, calibrated_probs, metrics


def time_dac_with_precomputed_features(
    train_feats: List[np.ndarray],
    val_feats: List[np.ndarray],
    test_feats: List[np.ndarray],
    val_logits: np.ndarray,
    test_logits: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    k: int,
    device: torch.device,
    backend: str = 'torch',
    extraction_time_s: float = 0.0,  # Pass pre-measured extraction time
) -> Tuple[TimingResult, np.ndarray, Dict[str, Any]]:
    """
    Time DAC with pre-computed features.
    
    Use this when features are already extracted (e.g., shared across methods).
    Pass the extraction_time_s separately to include in total timing.
    
    Args:
        train_feats: Pre-extracted training features (list of arrays)
        val_feats: Pre-extracted validation features
        test_feats: Pre-extracted test features
        val_logits: Validation logits
        test_logits: Test logits
        val_labels: Validation labels
        test_labels: Test labels
        k: Number of nearest neighbors
        device: PyTorch device
        backend: 'torch' or 'faiss'
        extraction_time_s: Pre-measured feature extraction time
        
    Returns:
        Tuple of (TimingResult, calibrated_probs, metrics_dict)
    """
    timing = TimingResult(method_name=f"DAC ({backend})")
    timing.extraction_time_s = extraction_time_s
    
    # Import the appropriate DAC calibrator
    if backend == 'torch':
        from Calibrators.density_aware_calibration_pytorch import create_dac_pytorch
        dac = create_dac_pytorch(k=k, use_gpu=(device.type == 'cuda'))
    else:
        from Calibrators.density_aware_calibration import DensityAwareCalibrator
        dac = DensityAwareCalibrator(k=k, use_gpu=(device.type == 'cuda'))
    
    # Ensure numpy arrays
    train_feats = [f.numpy() if isinstance(f, torch.Tensor) else f for f in train_feats]
    val_feats = [f.numpy() if isinstance(f, torch.Tensor) else f for f in val_feats]
    test_feats = [f.numpy() if isinstance(f, torch.Tensor) else f for f in test_feats]
    
    if isinstance(val_logits, torch.Tensor):
        val_logits = val_logits.cpu().numpy()
    if isinstance(test_logits, torch.Tensor):
        test_logits = test_logits.cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.cpu().numpy()
    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.cpu().numpy()
    
    # FIT
    logger.info(f"Fitting DAC ({backend} backend)...")
    with GPUTimer() as fit_timer:
        dac.fit(train_feats, val_feats, val_logits, val_labels)
    timing.fit_time_s = fit_timer.elapsed
    
    # CALIBRATE
    timing.n_test_samples = len(test_labels)
    logger.info(f"Calibrating test set ({backend} backend)...")
    with GPUTimer() as cal_timer:
        calibrated_probs = dac.calibrate(test_feats, test_logits)
    timing.calibrate_time_s = cal_timer.elapsed
    
    # Compute derived
    timing.compute_derived()
    
    # Metrics
    predictions = np.argmax(calibrated_probs, axis=1)
    accuracy = np.mean(predictions == test_labels)
    
    try:
        from Metrics.metrics import expected_calibration_error
        ece = expected_calibration_error(calibrated_probs, test_labels, n_bins=15)
    except ImportError:
        ece = _simple_ece(calibrated_probs, test_labels, n_bins=15)
    
    metrics = {
        'ece': float(ece),
        'accuracy': float(accuracy),
        'weights': dac.weights.tolist() if hasattr(dac, 'weights') else None,
    }
    
    logger.info(f"  {timing}")
    
    return timing, calibrated_probs, metrics


def _simple_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Simple ECE calculation."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece


# =============================================================================
# Comparison utilities
# =============================================================================

def compare_timing_results(results: List[TimingResult]) -> str:
    """
    Generate a formatted comparison table of timing results.
    
    Args:
        results: List of TimingResult objects
        
    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 80)
    lines.append("TIMING COMPARISON")
    lines.append("=" * 80)
    lines.append(f"{'Method':<25} {'Extract':>10} {'Fit':>10} {'Calibrate':>10} {'Total':>10} {'Throughput':>12}")
    lines.append("-" * 80)
    
    for r in results:
        lines.append(
            f"{r.method_name:<25} "
            f"{r.extraction_time_s:>10.2f} "
            f"{r.fit_time_s:>10.2f} "
            f"{r.calibrate_time_s:>10.2f} "
            f"{r.total_time_s:>10.2f} "
            f"{r.throughput_samples_per_sec:>12.1f}"
        )
    
    lines.append("=" * 80)
    return "\n".join(lines)
