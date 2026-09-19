"""
SGC vs FAISS-SGC: Equivalence Verification and Timing Comparison

Goals:
1. Verify FAISS implementation produces identical ECE to original
2. Measure timing with clear offline/online separation
3. Generate results for paper timing table
"""

import numpy as np
import time
import json
import torch
import faiss
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import logging
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Experiments.compare_sgc_backends import SGCCalibratorFast
from Calibrators.geometric_calibrator import GeometricCalibrator
from utils.stability_space import StabilitySpace

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        # Make bins half-open on the upper side to include confidences exactly equal to 0
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        # Include rightmost edge for the last bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def calculate_adaptive_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Calculate Adaptive Expected Calibration Error (ECE with equal-frequency binning)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    # Adaptive binning: equal number of samples per bin
    n_samples = len(confidences)
    sorted_indices = np.argsort(confidences)
    sorted_confidences = confidences[sorted_indices]
    
    # Create bin boundaries with equal number of samples per bin
    bin_boundaries = np.interp(
        np.linspace(0, n_samples, n_bins + 1),
        np.arange(n_samples),
        sorted_confidences
    )
    
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Include samples in this bin
        if i == n_bins - 1:
            # Last bin includes upper boundary
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calculate_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Calculate Brier Score."""
    num_classes = probs.shape[1]
    one_hot_labels = np.eye(num_classes)[labels]
    return np.mean(np.sum((probs - one_hot_labels)**2, axis=1))


@dataclass
class TimingResult:
    """Structured timing results with offline/online separation."""
    # Offline costs (one-time)
    index_build_time_s: float  # Building FAISS/sklearn index
    isotonic_fit_time_s: float  # Fitting isotonic regression
    total_offline_time_s: float
    
    # Online costs (per-batch)
    feature_extraction_time_s: float  # Per test batch feature extraction
    knn_query_time_s: float  # Per test batch kNN query
    isotonic_transform_time_s: float  # Per test batch isotonic transform
    total_online_time_s: float
    
    # Throughput
    n_test_samples: int
    throughput_samples_per_sec: float
    
    # Quality metrics
    ece: float
    adaptive_ece: float
    brier: float


class SklearnSGCCalibrator:
    """
    Original sklearn-based SGC calibrator for comparison.
    Uses exact NearestNeighbors search.
    """
    
    def __init__(self):
        from sklearn.neighbors import NearestNeighbors
        from sklearn.isotonic import IsotonicRegression
        
        self.same_nbrs = {}  # label -> NearestNeighbors
        self.other_nbrs = {}  # label -> NearestNeighbors
        self.isotonic = IsotonicRegression(out_of_bounds='clip')
        self.val_scores_sorted = None  # Sorted validation scores for rank-based normalization
        self.is_fitted = False
        self.last_stability = None
        self.X_train = None
        self.y_train = None
        self.num_classes = None
    
    def fit_index(self, X_train, y_train):
        """Build sklearn NearestNeighbors indices only (for timing breakdown)."""
        from sklearn.neighbors import NearestNeighbors
        
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.num_classes = len(np.unique(y_train))
        
        logger.info(f"Building sklearn NearestNeighbors indices for {len(X_train)} samples...")
        for label in range(self.num_classes):
            idx_same = np.where(y_train == label)[0]
            idx_other = np.where(y_train != label)[0]
            
            if len(idx_same) > 0:
                self.same_nbrs[label] = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
                self.same_nbrs[label].fit(X_train[idx_same])
            
            if len(idx_other) > 0:
                self.other_nbrs[label] = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
                self.other_nbrs[label].fit(X_train[idx_other])
    
    def fit_isotonic(self, X_val, y_val, val_predictions):
        """Fit isotonic regression only (assumes index already built)."""
        stability = self.compute_stability(X_val, val_predictions)
        
        # Store for debugging
        self.last_val_stability = stability.copy()
        
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
        n_samples = len(X_test)
        stability = np.zeros(n_samples)
        
        for i in range(n_samples):
            label = predictions[i]
            
            # Same-class distance
            if label in self.same_nbrs:
                d_same_sq, _ = self.same_nbrs[label].kneighbors(X_test[i:i+1], n_neighbors=1)
                d_same = d_same_sq[0, 0]
            else:
                d_same = np.inf
            
            # Other-class distance
            if label in self.other_nbrs:
                d_other_sq, _ = self.other_nbrs[label].kneighbors(X_test[i:i+1], n_neighbors=1)
                d_other = d_other_sq[0, 0]
            else:
                d_other = np.inf
            
            stability[i] = (d_other - d_same) / 2
        
        self.last_stability = stability.copy()
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
        """Fit the sklearn-based SGC calibrator."""
        self.fit_index(X_train_features, y_train)
        self.fit_isotonic(X_val_features, y_val, val_predictions)
    
    def calibrate(self, X_test_features, predictions, logits):
        """Calibrate test predictions."""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted")
        
        stability = self.compute_stability(X_test_features, predictions)
        return self.apply_isotonic(stability, predictions, logits)


def run_timing_experiment(
    # Pre-extracted features (to isolate calibration timing)
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_predictions: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_predictions: np.ndarray,
    test_logits: np.ndarray,
    # Configuration
    method: str,  # 'sklearn' or 'faiss'
    use_gpu: bool = True,
    n_warmup: int = 3,
    n_repeat: int = 5,
) -> TimingResult:
    """
    Run timing experiment with proper warmup and repetition.
    
    Timing structure:
    - OFFLINE: index building + isotonic fitting (measured once)
    - ONLINE: kNN query + isotonic transform (measured per-batch, averaged)
    """
    
    if method == 'faiss':
        calibrator = SGCCalibratorFast(use_gpu=use_gpu)
    else:
        calibrator = SklearnSGCCalibrator()
    
    # Synchronize GPU before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # ==================== OFFLINE TIMING ====================
    start_offline = time.perf_counter()
    
    # 1. Index building
    start_index = time.perf_counter()
    calibrator.fit_index(train_features, train_labels)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    index_build_time = time.perf_counter() - start_index
    
    # 2. Isotonic fitting (requires validation stability scores)
    start_isotonic = time.perf_counter()
    calibrator.fit_isotonic(val_features, val_labels, val_predictions)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    isotonic_fit_time = time.perf_counter() - start_isotonic
    
    total_offline_time = time.perf_counter() - start_offline
    
    # ==================== ONLINE TIMING ====================
    # Warmup runs
    for _ in range(n_warmup):
        _ = calibrator.calibrate(test_features, test_predictions, test_logits)
    
    # Timed runs
    online_times = []
    knn_times = []
    isotonic_times = []
    final_probs = None
    
    for _ in range(n_repeat):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_online = time.perf_counter()
        
        # kNN query timing
        start_knn = time.perf_counter()
        stability = calibrator.compute_stability(test_features, test_predictions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        knn_time = time.perf_counter() - start_knn
        
        # Isotonic transform timing
        start_iso = time.perf_counter()
        calibrated_probs = calibrator.apply_isotonic(stability, test_predictions, test_logits)
        isotonic_time = time.perf_counter() - start_iso
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_online = time.perf_counter() - start_online
        
        online_times.append(total_online)
        knn_times.append(knn_time)
        isotonic_times.append(isotonic_time)
        final_probs = calibrated_probs  # Use last run for metrics
    
    # Calculate metrics on final calibrated_probs
    ece = calculate_ece(final_probs, test_labels)
    adaptive_ece = calculate_adaptive_ece(final_probs, test_labels)
    brier = calculate_brier_score(final_probs, test_labels)
    
    n_test = len(test_labels)
    avg_online_time = np.mean(online_times)
    
    return TimingResult(
        index_build_time_s=index_build_time,
        isotonic_fit_time_s=isotonic_fit_time,
        total_offline_time_s=total_offline_time,
        feature_extraction_time_s=0.0,  # Measured separately
        knn_query_time_s=np.mean(knn_times),
        isotonic_transform_time_s=np.mean(isotonic_times),
        total_online_time_s=avg_online_time,
        n_test_samples=n_test,
        throughput_samples_per_sec=n_test / avg_online_time,
        ece=ece,
        adaptive_ece=adaptive_ece,
        brier=brier,
    )


def verify_equivalence(
    train_features, train_labels,
    val_features, val_labels, val_predictions,
    test_features, test_labels, test_predictions, test_logits,
    atol: float = 1e-5,
) -> Dict:
    """
    Verify that FAISS and sklearn implementations produce equivalent results.
    """
    # Run both
    faiss_cal = SGCCalibratorFast(use_gpu=True)
    faiss_cal.fit(train_features, train_labels, val_features, val_labels, val_predictions)
    faiss_probs = faiss_cal.calibrate(test_features, test_predictions, test_logits)
    faiss_stability = faiss_cal.last_stability
    
    sklearn_cal = SklearnSGCCalibrator()
    sklearn_cal.fit(train_features, train_labels, val_features, val_labels, val_predictions)
    sklearn_probs = sklearn_cal.calibrate(test_features, test_predictions, test_logits)
    sklearn_stability = sklearn_cal.last_stability
    
    # Compare stability scores
    stability_match = bool(np.allclose(faiss_stability, sklearn_stability, atol=atol))
    stability_max_diff = float(np.max(np.abs(faiss_stability - sklearn_stability)))
    
    # Compare calibrated probabilities
    probs_match = bool(np.allclose(faiss_probs, sklearn_probs, atol=atol))
    probs_max_diff = float(np.max(np.abs(faiss_probs - sklearn_probs)))
    
    # Compare ECE
    faiss_ece = float(calculate_ece(faiss_probs, test_labels))
    sklearn_ece = float(calculate_ece(sklearn_probs, test_labels))
    ece_diff = float(abs(faiss_ece - sklearn_ece))
    
    return {
        'stability_equivalent': stability_match,
        'stability_max_diff': stability_max_diff,
        'probs_equivalent': probs_match,
        'probs_max_diff': probs_max_diff,
        'faiss_ece': faiss_ece,
        'sklearn_ece': sklearn_ece,
        'ece_diff': ece_diff,
        'is_equivalent': bool(stability_match and probs_match and ece_diff < 0.001),
    }


def run_single_timing_comparison(
    features_dir: Path,
    dataset: str,
    model: str,
    seed: Optional[int],
    output_dir: Path,
    n_warmup: int = 3,
    n_repeat: int = 5,
) -> Dict:
    """Run timing comparison for a single model/seed combination."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {dataset} / {model} / seed {seed}")
    logger.info(f"{'='*60}")
    
    # Load data
    logger.info(f"Loading features from directory: {features_dir}")
    train_features = np.load(features_dir / 'train_features.npy')
    train_labels = np.load(features_dir / 'train_labels.npy')
    val_features = np.load(features_dir / 'val_features.npy')
    val_labels = np.load(features_dir / 'val_labels.npy')
    val_logits = np.load(features_dir / 'val_logits.npy')
    test_features = np.load(features_dir / 'test_features.npy')
    test_labels = np.load(features_dir / 'test_labels.npy')
    test_logits = np.load(features_dir / 'test_logits.npy')
    
    # Load metadata if available to get actual seed
    metadata_path = features_dir / 'extraction_info.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            actual_seed = metadata.get('seed', seed)
            logger.info(f"Loaded features from: {metadata.get('dataset')} / {metadata.get('model_name')} / seed {actual_seed}")
    else:
        actual_seed = seed
    
    val_predictions = np.argmax(val_logits, axis=1)
    test_predictions = np.argmax(test_logits, axis=1)
    
    logger.info(f"Data shapes:")
    logger.info(f"  Train: {train_features.shape}, labels: {train_labels.shape}")
    logger.info(f"  Val: {val_features.shape}, labels: {val_labels.shape}")
    logger.info(f"  Test: {test_features.shape}, labels: {test_labels.shape}")
    
    # 1. Verify equivalence
    logger.info("\n" + "="*60)
    logger.info("VERIFYING EQUIVALENCE")
    logger.info("="*60)
    equiv_result = verify_equivalence(
        train_features, train_labels,
        val_features, val_labels, val_predictions,
        test_features, test_labels, test_predictions, test_logits,
    )
    
    if not equiv_result['is_equivalent']:
        logger.error(f"FAISS not equivalent!")
        logger.error(f"  Max stability diff: {equiv_result['stability_max_diff']:.6e}")
        logger.error(f"  Max probs diff: {equiv_result['probs_max_diff']:.6e}")
        logger.error(f"  ECE diff: {equiv_result['ece_diff']:.6f}")
    else:
        logger.info(" Equivalence verified!")
        logger.info(f"  Stability max diff: {equiv_result['stability_max_diff']:.6e}")
        logger.info(f"  Probs max diff: {equiv_result['probs_max_diff']:.6e}")
        logger.info(f"  ECE diff: {equiv_result['ece_diff']:.6f}")
    
    # 2. Run timing comparison
    logger.info("\n" + "="*60)
    logger.info("TIMING COMPARISON")
    logger.info("="*60)
    
    logger.info("Running sklearn timing...")
    sklearn_timing = run_timing_experiment(
        train_features, train_labels,
        val_features, val_labels, val_predictions,
        test_features, test_labels, test_predictions, test_logits,
        method='sklearn',
        use_gpu=False,
        n_warmup=n_warmup,
        n_repeat=n_repeat,
    )
    
    logger.info("Running FAISS timing...")
    faiss_timing = run_timing_experiment(
        train_features, train_labels,
        val_features, val_labels, val_predictions,
        test_features, test_labels, test_predictions, test_logits,
        method='faiss',
        use_gpu=True,
        n_warmup=n_warmup,
        n_repeat=n_repeat,
    )
    
    return {
        'seed': actual_seed,
        'dataset': dataset,
        'model': model,
        'equivalence': equiv_result,
        'sklearn_timing': asdict(sklearn_timing),
        'faiss_timing': asdict(faiss_timing),
        'speedup': {
            'offline': sklearn_timing.total_offline_time_s / faiss_timing.total_offline_time_s,
            'online': sklearn_timing.total_online_time_s / faiss_timing.total_online_time_s,
            'knn_only': sklearn_timing.knn_query_time_s / faiss_timing.knn_query_time_s,
        }
    }


def main():
    """Run full comparison experiment."""
    import argparse
    parser = argparse.ArgumentParser(description='SGC Timing Comparison: FAISS vs sklearn')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset name')
    parser.add_argument('--model', type=str, nargs='+', default=['resnet50'], help='Model name(s) - can specify multiple')
    parser.add_argument('--seeds', type=int, nargs='+', help='Random seeds. If not provided, will try to infer from features_dir or use all available.')
    parser.add_argument('--output_dir', type=str, default='Results/timing_comparison', help='Output directory')
    parser.add_argument('--features_base_dir', type=str, default='Data/extracted_features', help='Base directory containing extracted features')
    parser.add_argument('--features_dir', type=str, help='Single directory containing extracted features (from extract_sgc_features_for_timing.py). Overrides batch mode.')
    parser.add_argument('--train_features_path', type=str, help='Path to pre-extracted train features (.npy)')
    parser.add_argument('--val_features_path', type=str, help='Path to pre-extracted val features (.npy)')
    parser.add_argument('--test_features_path', type=str, help='Path to pre-extracted test features (.npy)')
    parser.add_argument('--train_labels_path', type=str, help='Path to train labels (.npy)')
    parser.add_argument('--val_labels_path', type=str, help='Path to val labels (.npy)')
    parser.add_argument('--test_labels_path', type=str, help='Path to test labels (.npy)')
    parser.add_argument('--val_logits_path', type=str, help='Path to val logits (.npy)')
    parser.add_argument('--test_logits_path', type=str, help='Path to test logits (.npy)')
    parser.add_argument('--n_warmup', type=int, default=3, help='Number of warmup runs')
    parser.add_argument('--n_repeat', type=int, default=5, help='Number of timed runs')
    args = parser.parse_args()
    
    results = []
    
    # Handle single features_dir mode (original behavior)
    if args.features_dir:
        features_dir = Path(args.features_dir)
        if not features_dir.exists():
            logger.error(f"Features directory does not exist: {features_dir}")
            return
        
        # Try to infer model and seed from directory name or metadata
        metadata_path = features_dir / 'extraction_info.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                inferred_model = metadata.get('model_name', args.model[0] if isinstance(args.model, list) else args.model)
                inferred_seed = metadata.get('seed', None)
                inferred_dataset = metadata.get('dataset', args.dataset)
        else:
            inferred_model = args.model[0] if isinstance(args.model, list) else args.model
            inferred_seed = None
            inferred_dataset = args.dataset
        
        result = run_single_timing_comparison(
            features_dir=features_dir,
            dataset=inferred_dataset,
            model=inferred_model,
            seed=inferred_seed,
            output_dir=Path(args.output_dir),
            n_warmup=args.n_warmup,
            n_repeat=args.n_repeat,
        )
        results.append(result)
    
    # Handle batch mode (multiple models/seeds)
    elif args.features_base_dir:
        base_dir = Path(args.features_base_dir)
        models = args.model if isinstance(args.model, list) else [args.model]
        
        # If seeds not provided, try to discover them from directory names
        if args.seeds is None:
            # Look for directories matching pattern: {dataset}_{model}_seed{seed}
            discovered_seeds = set()
            for model in models:
                pattern = f"{args.dataset.lower()}_{model.lower()}_seed*"
                for dir_path in base_dir.glob(pattern):
                    # Extract seed from directory name
                    parts = dir_path.name.split('_seed')
                    if len(parts) == 2:
                        try:
                            seed = int(parts[1])
                            discovered_seeds.add(seed)
                        except ValueError:
                            pass
            seeds = sorted(discovered_seeds) if discovered_seeds else [None]
            logger.info(f"Discovered seeds: {seeds}")
        else:
            seeds = args.seeds
        
        total_combinations = len(models) * len(seeds)
        current = 0
        
        for model in models:
            for seed in seeds:
                current += 1
                # Construct features directory path
                features_dir = base_dir / f"{args.dataset.lower()}_{model.lower()}_seed{seed}"
                
                if not features_dir.exists():
                    logger.warning(f"Features directory does not exist: {features_dir}, skipping...")
                    continue
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing [{current}/{total_combinations}]: {model} seed {seed}")
                logger.info(f"{'='*60}")
                
                try:
                    result = run_single_timing_comparison(
                        features_dir=features_dir,
                        dataset=args.dataset,
                        model=model,
                        seed=seed,
                        output_dir=Path(args.output_dir),
                        n_warmup=args.n_warmup,
                        n_repeat=args.n_repeat,
                    )
                    results.append(result)
                    logger.info(f" Completed: {model} seed {seed}")
                except Exception as e:
                    logger.error(f" Failed for {model} seed {seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # Handle individual file paths (original behavior)
    elif args.train_features_path:
        # Create a temporary directory structure for this mode
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Copy files to temporary directory
            shutil.copy(args.train_features_path, tmp_path / 'train_features.npy')
            shutil.copy(args.train_labels_path, tmp_path / 'train_labels.npy')
            shutil.copy(args.val_features_path, tmp_path / 'val_features.npy')
            shutil.copy(args.val_labels_path, tmp_path / 'val_labels.npy')
            shutil.copy(args.val_logits_path, tmp_path / 'val_logits.npy')
            shutil.copy(args.test_features_path, tmp_path / 'test_features.npy')
            shutil.copy(args.test_labels_path, tmp_path / 'test_labels.npy')
            shutil.copy(args.test_logits_path, tmp_path / 'test_logits.npy')
            
            result = run_single_timing_comparison(
                features_dir=tmp_path,
                dataset=args.dataset,
                model=args.model[0] if isinstance(args.model, list) else args.model,
                seed=None,
                output_dir=Path(args.output_dir),
                n_warmup=args.n_warmup,
                n_repeat=args.n_repeat,
            )
            results.append(result)
    else:
        logger.error("Please provide one of:")
        logger.error("  1. --features_dir (single directory)")
        logger.error("  2. --features_base_dir with --model and optionally --seeds (batch mode)")
        logger.error("  3. Individual --*_features_path arguments")
        logger.error("\nExample usage (batch mode - recommended):")
        logger.error("  python sgc_timing_comparison.py \\")
        logger.error("    --dataset cifar100 \\")
        logger.error("    --model resnet50 densenet121 \\")
        logger.error("    --seeds 21 22 23 24 25 \\")
        logger.error("    --features_base_dir Data/extracted_features")
        logger.error("\nExample usage (single directory):")
        logger.error("  python sgc_timing_comparison.py \\")
        logger.error("    --features_dir Data/extracted_features/cifar100_resnet50_seed21 \\")
        logger.error("    --dataset cifar100 \\")
        logger.error("    --model resnet50")
        return
    
    if not results:
        logger.error("No results generated. Check that feature directories exist.")
        return
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save combined results
    combined_output = output_path / f'{args.dataset}_timing_comparison.json'
    with open(combined_output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved combined results to: {combined_output}")
    
    # Also save per-model results for backward compatibility
    for model in set(r['model'] for r in results):
        model_results = [r for r in results if r['model'] == model]
        model_output = output_path / f'{args.dataset}_{model}_timing.json'
        with open(model_output, 'w') as f:
            json.dump(model_results, f, indent=2)
        logger.info(f"Saved {model} results to: {model_output}")
    
    # Print summary
    print("\n" + "="*60)
    print("TIMING COMPARISON SUMMARY")
    print("="*60)
    
    for r in results:
        print(f"\nDataset: {r['dataset']}, Model: {r['model']}, Seed: {r['seed']}")
        print(f"  Equivalence: {'PASS' if r['equivalence']['is_equivalent'] else 'FAIL'}")
        print(f"  ECE diff: {r['equivalence']['ece_diff']:.6f}")
        print(f"\n  Timing (sklearn):")
        print(f"    Offline: {r['sklearn_timing']['total_offline_time_s']:.3f}s")
        print(f"    Online: {r['sklearn_timing']['total_online_time_s']:.3f}s")
        print(f"    kNN query: {r['sklearn_timing']['knn_query_time_s']:.3f}s")
        print(f"\n  Timing (FAISS):")
        print(f"    Offline: {r['faiss_timing']['total_offline_time_s']:.3f}s")
        print(f"    Online: {r['faiss_timing']['total_online_time_s']:.3f}s")
        print(f"    kNN query: {r['faiss_timing']['knn_query_time_s']:.3f}s")
        print(f"\n  Speedup:")
        print(f"    Offline: {r['speedup']['offline']:.2f}x")
        print(f"    Online: {r['speedup']['online']:.2f}x")
        print(f"    kNN only: {r['speedup']['knn_only']:.2f}x")
        print(f"\n  Quality metrics:")
        print(f"    sklearn ECE: {r['sklearn_timing']['ece']:.6f}")
        print(f"    FAISS ECE: {r['faiss_timing']['ece']:.6f}")


if __name__ == '__main__':
    main()
