# Calibrators/geometric_calibrator.py
"""
Geometric Calibrator using Fast Separation + Isotonic Regression
"""

import numpy as np
import logging
import torch
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
from .base_calibrator import BaseCalibrator
from utils.stability_space import StabilitySpace
import time
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, TensorDataset
from utils.compression_utils import SmartCompression

from utils.logging_config import get_logger
logger = get_logger(__name__)


def distance_matrix_to_geometry_scores(
    distance_matrix: np.ndarray,
    mode: str,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Convert a per-class 1-NN distance matrix to geometry scores.

    Args:
        distance_matrix: shape [n_samples, n_classes], non-negative distances.
        mode: one of:
            'neg_distance'    — score = -distance (closer = higher)
            'margin'          — score[i,c] = min_{c'!=c} D[i,c'] - D[i,c]
            'log_trust_ratio' — score[i,c] = log((min_{c'!=c} D[i,c'] + eps) / (D[i,c] + eps))
            'rank_log_trust'  — log_trust_ratio scores ranked per sample to [0,1]
        eps: numerical stability constant (used in log_trust_ratio and rank_log_trust).

    Returns:
        scores: same shape as distance_matrix, dtype float64; higher = more trusted.
    """
    D = np.asarray(distance_matrix, dtype=np.float64)
    n, K = D.shape

    if mode == 'neg_distance':
        return -D

    if mode == 'margin':
        scores = np.empty_like(D)
        for c in range(K):
            other = D[:, [k for k in range(K) if k != c]]
            scores[:, c] = other.min(axis=1) - D[:, c]
        return scores

    if mode == 'log_trust_ratio':
        scores = np.empty_like(D)
        for c in range(K):
            other = D[:, [k for k in range(K) if k != c]]
            min_other = other.min(axis=1) if K > 1 else np.full(n, np.inf)
            scores[:, c] = np.log((min_other + eps) / (D[:, c] + eps))
        return scores

    if mode == 'rank_log_trust':
        from scipy.stats import rankdata
        raw = distance_matrix_to_geometry_scores(D, 'log_trust_ratio', eps)
        scores = np.empty_like(raw)
        denom = float(K - 1) if K > 1 else 1.0
        for i in range(n):
            r = rankdata(raw[i], method='average')   # ranks 1..K, ties averaged
            scores[i] = (r - 1.0) / denom
        return scores

    raise ValueError(
        f"Unknown score mode: {mode!r}. "
        "Choose from 'neg_distance', 'margin', 'log_trust_ratio', 'rank_log_trust'."
    )


def extract_toplabel_anchor(
    model_probs: np.ndarray, calibrated_probs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract top-label anchor signals from model and calibrated full vectors.

    Returns:
        top_idx: model argmax class index per sample
        top_conf: calibrated probability at that model argmax class
    """
    if model_probs.ndim != 2 or calibrated_probs.ndim != 2:
        raise ValueError("model_probs and calibrated_probs must both be 2D arrays")
    if model_probs.shape != calibrated_probs.shape:
        raise ValueError("model_probs and calibrated_probs must have the same shape")
    if model_probs.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    top_idx = np.argmax(model_probs, axis=1).astype(np.int64)
    top_conf = calibrated_probs[np.arange(model_probs.shape[0]), top_idx].astype(np.float64)
    return top_idx, top_conf


# ============================================================================
class GeometricCalibrator(BaseCalibrator):
    """
    Geometric calibration using stability/separation scores and isotonic regression.
    """

    def __init__(self, model, X_train_embed=None, y_train=None, X_train_original=None, 
                 compression_mode=None, compression_param=None,
                 metric='l2', stability_space=None, 
                 library='faiss', use_binning=True, n_bins=200,
                 use_augmix=False, augmix_versions=3, augmix_severity=3, 
                 augmix_width=3, augmix_alpha=1.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 fitting_method: str = 'isotonic',  # Options: 'isotonic', 'beta', 'platt', 'logistic'
                 scoring_method: str = 'separation',  # Options: 'separation' (SGC) or 'trust_score' (Trust Score)
                 **kwargs):
        """
        Initializes the GeometricCalibrator with isotonic regression.
        """
        logging.info("\n=== GeometricCalibrator Initialization ===")
        if X_train_embed is not None:
            logging.info(f"Input X_train_embed shape: {X_train_embed.shape}")
        if X_train_original is not None:
            logging.info(f"Input X_train_original shape: {X_train_original.shape}")
        logging.info(f"Compression mode: {compression_mode}")
        logging.info(f"Compression param: {compression_param}")
        logging.info(f"Library: {library}")
        logging.info(f"Metric: {metric}")

        super().__init__(**kwargs)
        self.model = model
        self.isotonic_regressor = None
        self.regressor = None
        self.fitting_method = fitting_method
        self.is_fitted = False
        self.metric = metric.lower()
        self.use_binning = use_binning
        self.n_bins = n_bins
        self.device = device
        self.compression_mode = compression_mode
        self.compression_param = compression_param
        self.library = library
        
        # Validate and store scoring method
        if scoring_method not in ['separation', 'trust_score']:
            raise ValueError(f"scoring_method must be 'separation' or 'trust_score', got '{scoring_method}'")
        self.scoring_method = scoring_method
        self.auto_select_layer = False
        logger.info(f" Scoring method: {scoring_method} ({'SGC separation' if scoring_method == 'separation' else 'Trust Score'})")

        # AugMix parameters
        self.use_augmix = use_augmix
        self.augmix_versions = augmix_versions if use_augmix else 0
        self.augmix_severity = augmix_severity
        self.augmix_width = augmix_width
        self.augmix_alpha = augmix_alpha
        
        if self.use_augmix:
            from data.augmix_transforms import AugMixTransforms
            self.augmix_transform = AugMixTransforms(severity=augmix_severity, width=augmix_width, depth=-1, alpha=augmix_alpha)
            logger.info(f" AugMix enabled for calibration fitting: {augmix_versions} versions, severity={augmix_severity}")
        else:
            self.augmix_transform = None

        if y_train is not None:
            self.num_labels = len(np.unique(y_train))
        else:
            self.num_labels = None

        if X_train_embed is None:
            raise ValueError("X_train_embed is required")
        if y_train is None:
            raise ValueError("y_train is required")
        self.y_train = y_train
        if stability_space:
            self.stab_space = stability_space
        else:
            compression = SmartCompression(method='fixed_spp_jl', target_dims=512, in_channels=X_train_embed.shape[1]) if compression_mode else None
            self.stab_space = StabilitySpace(X_train_embed, y_train, compression=compression, library=library, metric=self.metric)
        
        logger.info(f"Initialized {self.__class__.__name__} with model {self.model.__class__.__name__}")

    def _create_regressor(self):
        """
        Create the appropriate regressor based on fitting_method.
        
        Returns:
            A fitted regressor with .fit(X, y) and .predict(X) methods
        """
        if self.fitting_method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            logger.info("    Using Isotonic Regression (non-parametric)")
            return IsotonicRegression(out_of_bounds="clip")
        
        elif self.fitting_method == 'beta':
            from Calibrators.beta_calibration import _BetaCal
            logger.info("    Using Beta Calibration (3 parameters: a, b, m)")
            return _BetaCal()
        
        elif self.fitting_method == 'platt':
            from sklearn.linear_model import LogisticRegression
            logger.info("    Using Platt Scaling (logistic regression)")
            # Platt scaling: logit(p) = a*s + b
            return LogisticRegression(
                penalty=None,
                max_iter=1000,
                random_state=42
            )
        
        elif self.fitting_method == 'logistic':
            from sklearn.linear_model import LogisticRegression
            logger.info("    Using Logistic Regression with log transforms")
            # Similar to beta but simpler: logit(p) = a + b*log(s)
            return LogisticRegression(
                penalty=None,
                max_iter=1000,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown fitting_method: {self.fitting_method}. "
                            f"Choose from: 'isotonic', 'beta', 'platt', 'logistic'")

    def _compute_score(self, X_val, y_val_pred):
        """
        Compute the appropriate score (separation or trust score) based on scoring_method.
        
        Args:
            X_val (numpy.ndarray): Validation data embeddings
            y_val_pred (numpy.ndarray): Predicted labels or probabilities
            
        Returns:
            numpy.ndarray: Score values (separation or trust score)
        """
        if self.scoring_method == 'separation':
            return self.stab_space.calc_stab(X_val, y_val_pred)
        elif self.scoring_method == 'trust_score':
            return self.stab_space.calc_trust_score(X_val, y_val_pred)
        else:
            raise ValueError(f"Unknown scoring_method: {self.scoring_method}")

    def fit(self, X_val_embed: Optional[np.ndarray] = None, y_val=None, X_val_original: Optional[np.ndarray] = None, features=None, fit_batch_size=1000, model_probs: Optional[np.ndarray] = None):
        """
        Fit the geometric calibrator using validation data.
        
        Args:
            X_val_embed: Validation embeddings/features
            y_val: Validation labels
            X_val_original: Raw validation inputs (required for predict_proba)
            features: Unused, for API compatibility
            fit_batch_size: Batch size for processing
            model_probs: Optional pre-computed model probabilities
        """
        logger.info(f"{self.__class__.__name__}: Fitting calibrator.")
        
        # Require original images for predict_proba
        if X_val_original is None:
            raise ValueError(
                "X_val_original (raw inputs) is required for predict_proba during fitting."
            )
        
        # =====================================================================
        #  AUGMIX HANDLING (if enabled)
        # =====================================================================
        
        # Handle AugMix data augmentation for training
        if self.use_augmix:
            logger.info(f" Creating AugMix augmented validation data ({self.augmix_versions} versions)...")
            X_val_original_expanded, X_val_embed_expanded, y_val_expanded = self._create_augmented_validation_data(
                X_val_embed, y_val, X_val_original
            )
            prediction_data, embedding_data, labels_data = X_val_original_expanded, X_val_embed_expanded, y_val_expanded
            logger.info(f"    Augmented data created: {len(labels_data)} total samples")
        else:
            # X_val_original is guaranteed to be available (checked above)
            prediction_data = X_val_original
            embedding_data = X_val_embed
            labels_data = y_val
            logger.info(f"    Using original validation data: {len(labels_data)} samples")
        
        # =====================================================================
        #  BATCHED STABILITY CALCULATION AND PREDICTION PROCESSING
        # =====================================================================
        
        score_name = "Trust Score" if self.scoring_method == 'trust_score' else "Stability"
        logger.info(f" Computing {score_name.lower()} values and predictions in batches...")
        all_stability_vals, all_predictions, all_labels = [], [], []
        n_samples = len(labels_data)
        n_batches = (n_samples + fit_batch_size - 1) // fit_batch_size
        
        logger.info(f"   Processing {n_samples} samples in {n_batches} batches of {fit_batch_size}")
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * fit_batch_size
            end_idx = min(start_idx + fit_batch_size, n_samples)
            
            batch_pred_data = prediction_data[start_idx:end_idx]
            batch_embed_data = embedding_data[start_idx:end_idx]
            batch_labels = labels_data[start_idx:end_idx]
            
            # Get model predictions for this batch
            batch_predictions = self.model.predict_proba(batch_pred_data)
            
            # Calculate stability/trust score values using the stability space
            batch_stability = self._compute_score(batch_embed_data, batch_predictions)
            
            # Collect results
            all_stability_vals.extend(batch_stability)
            all_predictions.extend(np.argmax(batch_predictions, axis=1))
            all_labels.extend(batch_labels)
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                logger.info(f"   Processed batch {batch_idx + 1}/{n_batches}")
        
        # Convert to numpy arrays
        stability_val = np.array(all_stability_vals)
        y_pred_classes = np.array(all_predictions)
        y_val_processed = np.array(all_labels)
        
        logger.info(f" {score_name} calculation complete:")
        logger.info(f"    {score_name} range: [{np.min(stability_val):.6f}, {np.max(stability_val):.6f}]")
        logger.info(f"    Mean {score_name.lower()}: {np.mean(stability_val):.6f}")
        
        # =====================================================================
        #  STABILITY NORMALIZATION (Rank-based)
        # =====================================================================
        
        logger.info(" Storing sorted validation scores for rank-based normalization...")
        
        # Store sorted validation scores for rank-based normalization
        # Isotonic regression only cares about ordering, not absolute values
        self.val_scores_sorted = np.sort(stability_val)
        n_val_samples = len(self.val_scores_sorted)
        
        logger.info(f"    Stored {n_val_samples} sorted validation scores")
        logger.info(f"    Score range: [{np.min(stability_val):.6f}, {np.max(stability_val):.6f}]")
        logger.info(f"    Mean score: {np.mean(stability_val):.6f}")
        
        # For fitting, we still need normalized values, so normalize using ranks
        # This preserves the ordering information that isotonic regression needs
        from scipy.stats import rankdata
        stability_val = rankdata(stability_val, method='average') / n_val_samples
        
        logger.info(f"    Applied rank-based normalization")
        logger.info(f"    Normalized range: [{np.min(stability_val):.6f}, {np.max(stability_val):.6f}]")
        
        # =====================================================================
        #  BINNING (if enabled)
        # =====================================================================
        
        # Only use binning for isotonic regression; parametric methods need raw data
        use_binning_effective = self.use_binning and (self.fitting_method == 'isotonic')
        
        # Log warning if binning was requested but disabled
        if self.use_binning and not use_binning_effective:
            logger.warning(f" Binning disabled for {self.fitting_method} calibration (requires raw data).")

        if use_binning_effective:
            logger.info(f" Applying binning with {self.n_bins} bins...")
            
            bin_edges = np.linspace(0, 1, self.n_bins + 1, dtype=np.float32)
            bin_indices = np.digitize(stability_val, bin_edges, right=True) - 1
            bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
            
            binned_stability, binned_accuracy = [], []
            
            for bin_idx in range(self.n_bins):
                indices_in_bin = np.where(bin_indices == bin_idx)[0]
                
                if len(indices_in_bin) > 0:
                    # Calculate accuracy for this bin
                    accuracy = np.mean(y_val_processed[indices_in_bin] == y_pred_classes[indices_in_bin])
                    
                    # Use bin center as stability value
                    bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
                    
                    binned_stability.append(bin_center)
                    binned_accuracy.append(accuracy)
            
            stability_vals = np.array(binned_stability)
            accuracies = np.array(binned_accuracy)
            
            logger.info(f"    Created {len(stability_vals)} bins with data")
            logger.info(f"    Accuracy range: [{np.min(accuracies):.4f}, {np.max(accuracies):.4f}]")
            
        else:
            logger.info(" Using individual samples (no binning)")
            stability_vals = stability_val
            accuracies = (y_val_processed == y_pred_classes).astype(float)
            
            logger.info(f"    Accuracy: {np.mean(accuracies):.4f}")
        
        # =====================================================================
        #  REGRESSOR FITTING
        # =====================================================================
        
        logger.info(f" Fitting {self.fitting_method} regressor...")
        
        # Create the appropriate regressor
        self.regressor = self._create_regressor()
        self.isotonic_regressor = self.regressor # Keep for backward compatibility if needed, but regressor is main
        
        # Prepare input data based on method
        if self.fitting_method in ['beta', 'logistic']:
            # These methods need log-transformed features
            eps = 1e-6
            stability_clipped = np.clip(stability_vals, eps, 1 - eps)
            
            if self.fitting_method == 'beta':
                # Beta calibration uses log(s) and log(1-s)
                X_fit = stability_clipped  # _BetaCal handles transform internally
            elif self.fitting_method == 'logistic':
                # Logistic with log transform
                X_fit = np.log(stability_clipped).reshape(-1, 1)
            
            y_fit = accuracies.astype(int)  # Binary labels for classification
            
        elif self.fitting_method in ['isotonic', 'platt']:
            # Direct mapping from stability to accuracy
            X_fit = stability_vals
            
            if self.fitting_method == 'platt':
                # Platt scaling needs binary labels
                y_fit = accuracies.astype(int)
                X_fit = X_fit.reshape(-1, 1)  # sklearn needs 2D input
            else:
                # Isotonic can use continuous targets
                y_fit = accuracies
        
        # Fit the regressor
        logger.info(f"    Fitting on {len(X_fit)} samples...")
        self.regressor.fit(X_fit, y_fit)
        
        # Log fitted parameters if available
        if self.fitting_method == 'beta' and hasattr(self.regressor, 'map_'):
            a, b, m = self.regressor.map_
            logger.info(f"    Beta parameters: a={a:.4f}, b={b:.4f}, m={m:.4f}")
        
        elif self.fitting_method in ['platt', 'logistic']:
            if hasattr(self.regressor, 'coef_') and hasattr(self.regressor, 'intercept_'):
                coef = self.regressor.coef_[0] if len(self.regressor.coef_.shape) > 1 else self.regressor.coef_
                intercept = self.regressor.intercept_[0] if hasattr(self.regressor.intercept_, '__len__') else self.regressor.intercept_
                logger.info(f"    Logistic parameters: coef={coef}, intercept={intercept:.4f}")
        
        elif self.fitting_method == 'isotonic':
            if hasattr(self.regressor, 'X_thresholds_'):
                n_thresholds = len(self.regressor.X_thresholds_)
                logger.info(f"    Isotonic fit: {n_thresholds} thresholds")
        
        # Note: val_scores_sorted is already stored above for rank-based normalization
        # No need to store percentile parameters anymore
        
        # Initialize storage for stability scores (for per-sample saving)
        self.last_stability_scores = None
        
        # Mark as fitted
        self.is_fitted = True
        
        logger.info(f" {self.__class__.__name__} fitting complete!")
        logger.info(f"    Trained on {len(stability_vals)} {'bins' if self.use_binning else 'samples'}")
        logger.info(f"    Method: {self.fitting_method}")
        logger.info(f"    Rank-based normalization: stored {len(self.val_scores_sorted)} sorted validation scores")
        
        # Log optimization summary
        if self.auto_select_layer and hasattr(self, 'layer_selector'):
            total_possible_metrics = len(self.layer_selector.original_metrics) if hasattr(self.layer_selector, 'original_metrics') else len(enhanced_metrics)
            logger.info(f" Optimization Summary:")
            logger.info(f"    Computed {len(enhanced_metrics)}/{total_possible_metrics} metrics")
            logger.info(f"    Estimated speedup: {total_possible_metrics/len(enhanced_metrics):.1f}x")

    def calibrate_batched(self, X_test_embed: Optional[np.ndarray] = None, X_test_original: Optional[np.ndarray] = None, features=None, batch_size=1000, return_probs=False):
        """
        Memory-efficient batched calibration to prevent GPU OOM.
        
        Args:
            X_test_embed: Test embeddings/features
            X_test_original: Raw test inputs (required for predict_proba)
            features: Unused, for API compatibility
            batch_size: Batch size for processing
            return_probs: If True, returns (calibrated_probs, model_probs) tuple.
        """
        if not self.is_fitted:
            raise ValueError("You must fit the calibrator before using it.")

        # Require original images for predict_proba
        if X_test_original is None:
            raise ValueError(
                "X_test_original (raw inputs) is required for predict_proba."
            )
        
        logger.info(f"GeometricCalibrator: Calibrating test data in batches of {batch_size}.")
        n_samples = X_test_original.shape[0]
        
        # Guard against missing num_labels
        if self.num_labels is None:
            # lazily infer from model on a tiny batch
            tmp = self.model.predict_proba(X_test_original[:1])
            self.num_labels = tmp.shape[1]
            logger.info(f" Inferred num_labels = {self.num_labels} from model output")
        
        num_classes = self.num_labels
        all_calibrated_probs = np.zeros((n_samples, num_classes))
        all_model_probs = np.zeros((n_samples, num_classes)) if return_probs else None
        all_stability_scores = np.zeros(n_samples)  # Accumulate stability scores
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        start_time = time.time()
        with tqdm(total=n_batches, desc="Calibrating Test Batches", unit="batch") as pbar:
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_embed = X_test_embed[start_idx:end_idx]
                batch_original = X_test_original[start_idx:end_idx]
                logger.debug(f"   Batch {batch_idx + 1}: embed={batch_embed.shape}, original={batch_original.shape}")
                try:
                    # Removed torch.cuda.empty_cache() from tight loops for performance
                    if return_probs:
                        batch_calibrated, batch_model_probs, batch_stability = self._calibrate_single_batch(batch_embed, batch_original, return_probs=True, return_stability=True)
                        all_calibrated_probs[start_idx:end_idx] = batch_calibrated
                        all_model_probs[start_idx:end_idx] = batch_model_probs
                        all_stability_scores[start_idx:end_idx] = batch_stability
                    else:
                        batch_probs, batch_stability = self._calibrate_single_batch(batch_embed, batch_original, return_stability=True)
                        all_calibrated_probs[start_idx:end_idx] = batch_probs
                        all_stability_scores[start_idx:end_idx] = batch_stability
                    pbar.update(1)
                except Exception as e:
                    logger.error(f" Calibration batch {batch_idx + 1} failed: {e}")
                    all_calibrated_probs[start_idx:end_idx] = 1.0 / num_classes
                    all_stability_scores[start_idx:end_idx] = 0.5  # Default stability
                    pbar.update(1)
        total_time = time.time() - start_time
        logger.info(f" Batched calibration completed in {total_time:.1f}s")
        logger.info(f"GeometricCalibrator: Batched calibration complete.")
        
        # Store stability scores for per-sample saving
        self.last_stability_scores = all_stability_scores
        
        if return_probs:
            return all_calibrated_probs, all_model_probs
        return all_calibrated_probs

    def calibrate_batched_precomputed(self, X_test_embed: np.ndarray, X_test_original: np.ndarray, model_probs: np.ndarray, batch_size=1000):
        """
        Batched calibration using PRE-COMPUTED probabilities to measure calibration throughput 
        without inference overhead (Apples-to-Apples with DAC).
        """
        if not self.is_fitted:
            raise ValueError("You must fit the calibrator before using it.")
            
        if len(model_probs) != len(X_test_original):
            raise ValueError(f"Length of model_probs ({len(model_probs)}) must match data ({len(X_test_original)})")
            
        logger.info(f"GeometricCalibrator: Calibrating test data (PRE-COMPUTED) in batches of {batch_size}.")
        n_samples = X_test_original.shape[0]
        num_classes = model_probs.shape[1]
        
        all_calibrated_probs = np.zeros((n_samples, num_classes))
        all_stability_scores = np.zeros(n_samples)  # Accumulate stability scores
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        with tqdm(total=n_batches, desc="Calibrating (Pre-computed)", unit="batch") as pbar:
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                batch_embed = X_test_embed[start_idx:end_idx]
                # Passed for shape checks/logging but not used for inference
                batch_original = X_test_original[start_idx:end_idx] 
                batch_probs = model_probs[start_idx:end_idx]
                
                try:
                    batch_calibrated, batch_stability = self._calibrate_single_batch_precomputed(batch_embed, batch_probs, return_stability=True)
                    all_calibrated_probs[start_idx:end_idx] = batch_calibrated
                    all_stability_scores[start_idx:end_idx] = batch_stability
                    pbar.update(1)
                except Exception as e:
                    logger.error(f" Calibration batch {batch_idx + 1} failed: {e}")
                    all_calibrated_probs[start_idx:end_idx] = 1.0 / num_classes
                    all_stability_scores[start_idx:end_idx] = 0.5  # Default stability
                    pbar.update(1)
        
        # Store stability scores for per-sample saving
        self.last_stability_scores = all_stability_scores
                    
        return all_calibrated_probs

    def calibrate_batched_toplabel_anchor(
        self,
        X_test_embed: Optional[np.ndarray] = None,
        X_test_original: Optional[np.ndarray] = None,
        batch_size: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convenience wrapper for top-label anchor extraction without changing
        any geometric calibration internals.

        Returns:
            top_idx: model argmax class index per sample
            top_conf: calibrated top-label confidence c_t per sample
        """
        calibrated_probs, model_probs = self.calibrate_batched(
            X_test_embed=X_test_embed,
            X_test_original=X_test_original,
            batch_size=batch_size,
            return_probs=True,
        )
        return extract_toplabel_anchor(model_probs, calibrated_probs)

    def calibrate(self, X_test_embed: Optional[np.ndarray] = None, X_test_original: Optional[np.ndarray] = None, features=None):
        """
        Calibrates the test data using the fitted isotonic regression model.
        
        Args:
            X_test_embed: Test embeddings/features
            X_test_original: Raw test inputs (required for predict_proba)
            features: Unused, for API compatibility
        """
        if not self.is_fitted:
            raise ValueError("You must fit the calibrator before using it.")

        # Require original images for predict_proba
        if X_test_original is None:
            raise ValueError(
                "X_test_original (raw inputs) is required for predict_proba."
            )

        n_samples = len(X_test_embed)
        if n_samples > 1000:
            return self.calibrate_batched(X_test_embed, X_test_original, features, batch_size=1000)
        else:
            logger.info(f"GeometricCalibrator: Calibrating {n_samples} samples (small dataset)")
            # X_test_original is guaranteed to be available (checked above)
            logger.debug(f"Embed data shape: {X_test_embed.shape}")
            logger.debug(f"Original data shape: {X_test_original.shape}")
            try:
                # Removed torch.cuda.empty_cache() from tight loops for performance
                calibrated_probs, stability_scores = self._calibrate_single_batch(X_test_embed, X_test_original, return_stability=True)
                # Store stability scores for per-sample saving
                self.last_stability_scores = stability_scores
                logger.info(f"GeometricCalibrator: Small dataset calibration complete")
                return calibrated_probs
            except Exception as e:
                logger.error(f"GeometricCalibrator: Small dataset calibration failed: {e}")
                num_classes = self.num_labels
                uniform_probs = np.ones((n_samples, num_classes)) / num_classes
                self.last_stability_scores = np.ones(n_samples) * 0.5  # Default stability
                return uniform_probs

    def _calibrate_single_batch_precomputed(self, batch_embed, batch_probs, return_stability=False):
        """
        Process a single batch using PRE-COMPUTED probabilities.
        Skips self.model.predict_proba() to isolate calibration cost.
        
        Args:
            return_stability: If True, returns (calibrated_probs, normalized_stability_batch) tuple
        """
        # Validate that calibrator is properly fitted
        if self.stab_space is None:
            raise ValueError("stab_space is None - calibrator may not be fitted properly")
        if self.regressor is None:
            raise ValueError("regressor is None - calibrator may not be fitted properly")
        
        # Use provided probabilities directly
        y_batch_pred = batch_probs
        y_batch_labels = np.argmax(y_batch_pred, axis=1)

        batch_size = batch_probs.shape[0]
        num_classes = batch_probs.shape[1]
        calibrated_probs = np.zeros((batch_size, num_classes))

        # Compute stability/trust score for batch (using pre-computed logits)
        stability_batch = self._compute_score(batch_embed, y_batch_pred)
        
        # Validate stability_batch
        method_name = 'calc_stab' if self.scoring_method == 'separation' else 'calc_trust_score'
        if stability_batch is None:
            raise ValueError(f"stab_space.{method_name} returned None")
        if not isinstance(stability_batch, np.ndarray):
            raise TypeError(f"stab_space.{method_name} returned {type(stability_batch)}, expected np.ndarray")
        
        # Normalize stability values using rank-based normalization (store before isotonic transform)
        # Use searchsorted to find ranks efficiently
        normalized_stability_batch = np.searchsorted(self.val_scores_sorted, stability_batch) / len(self.val_scores_sorted)
        
        # Apply calibration using the fitted regressor
        if self.fitting_method == 'beta':
            calibrated_values = self.regressor.predict(normalized_stability_batch)
        elif self.fitting_method == 'logistic':
            eps = 1e-6
            stability_clipped = np.clip(normalized_stability_batch, eps, 1 - eps)
            X_pred = np.log(stability_clipped).reshape(-1, 1)
            calibrated_values = self.regressor.predict_proba(X_pred)[:, 1]
        elif self.fitting_method == 'platt':
            X_pred = normalized_stability_batch.reshape(-1, 1)
            calibrated_values = self.regressor.predict_proba(X_pred)[:, 1]
        elif self.fitting_method == 'isotonic':
            calibrated_values = self.regressor.predict(normalized_stability_batch)
        else:
            raise ValueError(f"Unknown fitting_method: {self.fitting_method}")
        
        # Validate calibrated_values
        if calibrated_values is None:
            raise ValueError(f"regressor.predict/predict_proba returned None for fitting_method={self.fitting_method}")
        if not isinstance(calibrated_values, np.ndarray):
            raise TypeError(f"regressor.predict/predict_proba returned {type(calibrated_values)}, expected np.ndarray")
        
        eps = 1e-6
        calibrated_values = np.clip(calibrated_values, eps, 1 - eps)

        # Distribute calibrated values
        for i in range(batch_size):
            c = y_batch_labels[i]
            if self.num_labels > 1:
                rem = (1.0 - calibrated_values[i]) / (self.num_labels - 1)
                calibrated_probs[i, :] = rem
                calibrated_probs[i, c] = calibrated_values[i]
            else:
                calibrated_probs[i, c] = 1.0

        calibrated_probs = np.clip(calibrated_probs, 0, 1)
        
        if return_stability:
            return calibrated_probs, normalized_stability_batch
        return calibrated_probs

    def _calibrate_single_batch(self, batch_embed, batch_original, return_probs=False, return_stability=False):
        """Process a single batch using the original calibration logic
        
        Args:
            return_stability: If True, returns normalized stability scores (after normalization, before isotonic transform)
        """
        
        #  DEBUG: Add detailed logging for data validation
        logger.debug(f" _calibrate_single_batch received:")
        logger.debug(f"   batch_embed shape: {batch_embed.shape}")
        logger.debug(f"   batch_embed dtype: {batch_embed.dtype}")
        logger.debug(f"   batch_original shape: {batch_original.shape}")
        logger.debug(f"   batch_original dtype: {batch_original.dtype}")
        
        #  CRITICAL FIX: Detect if logits are accidentally passed instead of raw images
        if len(batch_original.shape) == 2 and self.num_labels is not None and batch_original.shape[1] == self.num_labels and batch_original.dtype in (np.float32, np.float64):
            logger.error(f" CRITICAL ERROR: Logits detected in batch_original instead of raw images!")
            logger.error(f"   batch_original shape: {batch_original.shape} (looks like logits for {batch_original.shape[1]} classes)")
            logger.error(f"   Expected raw input data, not probabilities/logits")
            logger.error(f"   This indicates a data flow bug in the calibration pipeline")
            raise ValueError(f"Probabilities/logits detected in X_test_original; pass raw inputs instead.")
        
        # Validate data types for physical vs semantic methods
        if len(batch_embed.shape) == 4 and batch_embed.shape[1] in [1, 3]:
            logger.debug(f"    Physical method detected: batch_embed appears to be raw images")
        elif len(batch_embed.shape) == 2:
            logger.debug(f"    Semantic method detected: batch_embed appears to be features")
        else:
            logger.warning(f"    Unexpected batch_embed shape: {batch_embed.shape}")
        
        # Use model adapter to predict on batch
        y_batch_pred = self.model.predict_proba(batch_original)
        y_batch_labels = np.argmax(y_batch_pred, axis=1)

        batch_size = batch_original.shape[0]
        num_classes = y_batch_pred.shape[1]
        calibrated_probs = np.zeros((batch_size, num_classes))

        # Compute stability/trust score for batch
        stability_batch = self._compute_score(batch_embed, y_batch_pred)
        
        # Log stability/trust score statistics before normalization
        score_name = "Trust Score" if self.scoring_method == 'trust_score' else "Stability"
        logger.info(f"Batch {score_name.lower()} stats before normalization:")
        logger.info(f"  Min: {np.min(stability_batch):.6f}")
        logger.info(f"  Max: {np.max(stability_batch):.6f}")
        logger.info(f"  Mean: {np.mean(stability_batch):.6f}")
        logger.info(f"  Std: {np.std(stability_batch):.6f}")
        
        # Normalize stability values using rank-based normalization (store before isotonic transform)
        # Use searchsorted to find ranks efficiently
        normalized_stability_batch = np.searchsorted(self.val_scores_sorted, stability_batch) / len(self.val_scores_sorted)
        
        # Apply calibration using the fitted regressor
        if self.fitting_method == 'beta':
            # Beta calibration expects raw stability values
            calibrated_values = self.regressor.predict(normalized_stability_batch)
        
        elif self.fitting_method == 'logistic':
            # Logistic with log transform
            eps = 1e-6
            stability_clipped = np.clip(normalized_stability_batch, eps, 1 - eps)
            X_pred = np.log(stability_clipped).reshape(-1, 1)
            calibrated_values = self.regressor.predict_proba(X_pred)[:, 1]
        
        elif self.fitting_method == 'platt':
            # Platt scaling on raw stability
            X_pred = normalized_stability_batch.reshape(-1, 1)
            calibrated_values = self.regressor.predict_proba(X_pred)[:, 1]
        
        elif self.fitting_method == 'isotonic':
            # Isotonic regression
            calibrated_values = self.regressor.predict(normalized_stability_batch)
        
        eps = 1e-6
        calibrated_values = np.clip(calibrated_values, eps, 1 - eps)

        # Distribute calibrated values across predicted classes without renormalization
        for i in range(batch_size):
            c = y_batch_labels[i]
            if self.num_labels > 1:
                rem = (1.0 - calibrated_values[i]) / (self.num_labels - 1)
                calibrated_probs[i, :] = rem
                calibrated_probs[i, c] = calibrated_values[i]
            else:
                calibrated_probs[i, c] = 1.0

        # Ensure probabilities are valid
        calibrated_probs = np.clip(calibrated_probs, 0, 1)

        # Return based on flags
        if return_probs and return_stability:
            return calibrated_probs, y_batch_pred, normalized_stability_batch
        elif return_probs:
            return calibrated_probs, y_batch_pred
        elif return_stability:
            return calibrated_probs, normalized_stability_batch
        return calibrated_probs

    def _apply_augmix_to_batch(self, X_val_original):
        """Apply AugMix transformation to a batch of images - SIMPLIFIED VERSION"""
        import torch
        from torchvision import transforms
        
        # Convert numpy to torch if needed
        if isinstance(X_val_original, np.ndarray):
            X_tensor = torch.FloatTensor(X_val_original)
        else:
            X_tensor = X_val_original
        
        augmented_batch = []
        
        # Process in smaller batches to avoid memory issues
        batch_size = min(32, X_tensor.shape[0])  # Process 32 images at a time
        
        for i in range(0, X_tensor.shape[0], batch_size):
            end_idx = min(i + batch_size, X_tensor.shape[0])
            batch_tensor = X_tensor[i:end_idx]
            
            batch_augmented = []
            for j in range(batch_tensor.shape[0]):
                try:
                    # Get single image [C, H, W]
                    image_tensor = batch_tensor[j]
                    
                    # Convert to PIL Image for AugMix (denormalize first)
                    # Denormalize: x = x * std + mean (CIFAR normalization)
                    denorm_image = image_tensor * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
                    denorm_image += torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                    denorm_image = torch.clamp(denorm_image, 0, 1)
                    
                    # Convert to PIL
                    pil_image = transforms.ToPILImage()(denorm_image)
                    
                    # Apply AugMix (returns clean, aug1, aug2 - we'll use aug1)
                    clean, aug1, aug2 = self.augmix_transform(pil_image)
                    
                    # Use one of the augmented versions (aug1)
                    batch_augmented.append(aug1)
                    
                except Exception as e:
                    logger.warning(f"AugMix failed for image {i+j}: {e}, using original")
                    # Fallback to original image
                    batch_augmented.append(image_tensor)
            
            # Add batch to results
            if batch_augmented:
                augmented_batch.extend(batch_augmented)
        
        # Stack into batch tensor
        if augmented_batch:
            augmented_tensor = torch.stack(augmented_batch, dim=0)
            return augmented_tensor.numpy()
        else:
            logger.error("No augmented images created - returning original")
            return X_val_original

    def _extract_features_from_images(self, images):
        """Extract features from augmented images using the model's feature extractor."""
        # This requires access to the feature extraction mechanism
        # Implementation depends on how features were originally extracted
        
        # Option 1: If we have direct access to feature extractor
        if hasattr(self.model, 'extract_features'):
            return self.model.extract_features(images)
        
        # Option 2: If model has a feature extractor attribute
        if hasattr(self.model, 'feature_extractor'):
            return self.model.feature_extractor(images)
        
        # Option 3: If model is wrapped in AugMixFeatureModel
        if hasattr(self.model, 'base_model'):
            # Get features from the base model
            with torch.no_grad():
                features = self.model.base_model(images, return_features=True)[1]
            return features
        
        # Fallback: Use logits as features
        logger.warning("No feature extractor found - using model logits as features")
        with torch.no_grad():
            logits = self.model(images)
        return logits

    def _create_augmented_validation_data(self, X_val_embed, y_val, X_val_original):
        """
        Create augmented validation dataset using the model adapter's feature extraction.
        Handles both Physical (raw images) and Semantic (features) calibration modes.
        """
        if X_val_original is None:
            raise ValueError("X_val_original required for AugMix augmentation")
        
        logger.info(" Creating augmented validation data...")
        
        # Detect calibration type based on data shapes
        is_physical_calibration = (X_val_embed.shape == X_val_original.shape and 
                                 len(X_val_embed.shape) == 4)
        
        logger.info(f" Calibration type: {'Physical' if is_physical_calibration else 'Semantic'}")
        
        # Storage for expanded data
        original_images_list = [X_val_original]
        expanded_labels_list = [y_val]
        
        if is_physical_calibration:
            # For Physical calibration: keep everything as raw images
            augmented_embeddings_list = [X_val_embed]  # Same as original (raw images)
            
            logger.info(" Physical calibration: keeping all data as raw images")
            
            # Create augmented versions
            for aug_idx in range(self.augmix_versions):
                logger.info(f"   Creating augmented version {aug_idx + 1}/{self.augmix_versions}")
                
                try:
                    # Apply AugMix to get augmented images
                    augmented_images = self._apply_augmix_to_batch(X_val_original)
                    
                    # For Physical calibration: use augmented images directly (no feature extraction)
                    original_images_list.append(augmented_images)
                    augmented_embeddings_list.append(augmented_images)  # Same data for geometric calc
                    expanded_labels_list.append(y_val)
                    
                except Exception as e:
                    logger.error(f"Failed to create augmented version {aug_idx + 1}: {e}")
                    continue
        
        else:
            # For Semantic calibration: extract features from augmented images
            augmented_embeddings_list = [X_val_embed]  # Original features
            
            logger.info(" Semantic calibration: extracting features from augmented images")
            
            # Create augmented versions
            for aug_idx in range(self.augmix_versions):
                logger.info(f"   Creating augmented version {aug_idx + 1}/{self.augmix_versions}")
                
                try:
                    # Apply AugMix to get augmented images
                    augmented_images = self._apply_augmix_to_batch(X_val_original)
                    
                    # Extract features from augmented images
                    augmented_embeddings = self._extract_features_from_images(augmented_images)
                    
                    if augmented_embeddings is None:
                        logger.error(f"Feature extraction failed for augmented version {aug_idx + 1}")
                        continue
                    
                    original_images_list.append(augmented_images)
                    augmented_embeddings_list.append(augmented_embeddings)
                    expanded_labels_list.append(y_val)
                    
                except Exception as e:
                    logger.error(f"Failed to create augmented version {aug_idx + 1}: {e}")
                    continue
        
        # Concatenate all versions
        try:
            X_val_original_expanded = np.concatenate(original_images_list, axis=0)
            X_val_embed_expanded = np.concatenate(augmented_embeddings_list, axis=0)
            y_val_expanded = np.concatenate(expanded_labels_list, axis=0)
            
            logger.info(f" Augmented validation data created:")
            logger.info(f"   Original images: {X_val_original_expanded.shape}")
            logger.info(f"   {'Raw images' if is_physical_calibration else 'Features'}: {X_val_embed_expanded.shape}")
            logger.info(f"   Labels: {y_val_expanded.shape}")
            
            return X_val_original_expanded, X_val_embed_expanded, y_val_expanded
            
        except Exception as e:
            logger.error(f"Failed to concatenate augmented data: {e}")
            logger.error(f"Original images shapes: {[x.shape for x in original_images_list]}")
            logger.error(f"Embeddings shapes: {[x.shape for x in augmented_embeddings_list]}")
            raise ValueError(f"Failed to concatenate augmented data: {e}")


    def get_params(self) -> dict:
        params = {
            'metric': self.metric,
            'use_binning': self.use_binning,
            'n_bins': self.n_bins,
            'num_labels': self.num_labels,
            'is_fitted': self.is_fitted,
            'fitting_method': self.fitting_method,
            'fitting_function': self.fitting_method,
            # Rank-based normalization: stored sorted validation scores
            'normalization_method': 'rank_based',
            'val_scores_count': int(len(getattr(self, 'val_scores_sorted', []))),
            # AugMix parameters
            'use_augmix': self.use_augmix,
            'augmix_versions': self.augmix_versions,
            'augmix_severity': self.augmix_severity,
            'augmix_width': self.augmix_width,
            'augmix_alpha': self.augmix_alpha,
        }
        
        # Add method-specific parameters
        if self.is_fitted:
            # === NEW: Export Isotonic Thresholds ===
            if self.fitting_method == 'isotonic' and hasattr(self.regressor, 'X_thresholds_'):
                # FIXED: Calculate y_min/y_max from the thresholds directly
                y_thresholds = self.regressor.y_thresholds_
                
                params['isotonic_params'] = {
                    'X_thresholds': self.regressor.X_thresholds_.tolist(),
                    'y_thresholds': y_thresholds.tolist(),
                    'X_min': float(self.regressor.X_min_),
                    'X_max': float(self.regressor.X_max_),
                    'y_min': float(y_thresholds[0]),  # Use first threshold as min
                    'y_max': float(y_thresholds[-1])  # Use last threshold as max
                }
            # =======================================
            elif self.fitting_method == 'beta' and hasattr(self.regressor, 'map_'):
                params['beta_params'] = {
                    'a': float(self.regressor.map_[0]),
                    'b': float(self.regressor.map_[1]),
                    'm': float(self.regressor.map_[2])
                }
            elif self.fitting_method in ['platt', 'logistic'] and hasattr(self.regressor, 'coef_'):
                params['logistic_params'] = {
                    'coef': self.regressor.coef_.tolist(),
                    'intercept': float(self.regressor.intercept_[0]) if hasattr(self.regressor.intercept_, '__len__') else float(self.regressor.intercept_)
                }

        return params


class FullVectorGeometricFusionCalibrator(BaseCalibrator):
    """
    Full-vector geometric fusion calibrator:
      q = softmax(log(p_model + eps) + lambda * S)
    where S = distance_matrix_to_geometry_scores(D, score_mode, eps) and D is the
    per-class 1-NN distance matrix from training/reference data.

    The lambda hyperparameter is selected by minimising validation NLL over lambda_grid.
    """

    DEFAULT_LAMBDA_GRID = np.array(
        [0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30], dtype=np.float64
    )

    _VALID_SCORE_MODES = ('neg_distance', 'margin', 'log_trust_ratio', 'rank_log_trust')

    def __init__(
        self,
        model,
        X_train_embed,
        y_train,
        metric='l2',
        library='fast_separation',
        compression_mode=None,
        compression_param=None,
        lambda_grid=None,
        score_mode='neg_distance',
        eps=1e-12,
        stability_space=None,
        whitening_components=128,
        whitening_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if score_mode not in self._VALID_SCORE_MODES:
            raise ValueError(
                f"score_mode must be one of {self._VALID_SCORE_MODES}, got {score_mode!r}"
            )
        self.model = model
        self.metric = metric.lower()
        self.library = library
        self.whitening_components = int(whitening_components)
        self.whitening_eps = float(whitening_eps)
        self.eps = float(eps)
        self.score_mode = score_mode
        self.lambda_grid = (
            np.asarray(lambda_grid, dtype=np.float64)
            if lambda_grid is not None
            else self.DEFAULT_LAMBDA_GRID.copy()
        )
        self.y_train = y_train
        self.num_labels = len(np.unique(y_train))

        if stability_space is not None:
            self.stab_space = stability_space
        else:
            compression = (
                SmartCompression(
                    method='fixed_spp_jl',
                    target_dims=512,
                    in_channels=X_train_embed.shape[1],
                )
                if compression_mode
                else None
            )
            self.stab_space = StabilitySpace(
                X_train_embed,
                y_train,
                compression=compression,
                library=library,
                metric=self.metric,
                whitening_components=self.whitening_components,
                whitening_eps=self.whitening_eps,
            )

        self.metric = self.stab_space.metric
        self.library = self.stab_space.library
        self.whitening_components = self.stab_space.whitening_components
        self.whitening_eps = self.stab_space.whitening_eps

        self.best_lambda = None
        self.lambda_selection = {}
        self.lambda0_max_abs_diff = None
        self.is_fitted = False

    def _fuse_probs(self, model_probs, distance_matrix, lam):
        scores = distance_matrix_to_geometry_scores(distance_matrix, self.score_mode, self.eps)
        logits = np.log(np.clip(model_probs, self.eps, 1.0)) + float(lam) * scores
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        denom = np.clip(np.sum(exp_logits, axis=1, keepdims=True), self.eps, None)
        return exp_logits / denom

    def _multiclass_nll(self, probs, labels):
        idx = np.arange(len(labels))
        return float(-np.mean(np.log(np.clip(probs[idx, labels], self.eps, 1.0))))

    def fit(
        self,
        X_val_embed,
        y_val,
        X_val_original=None,
        model_probs=None,
        validate_lambda0=True,
    ):
        if model_probs is None:
            if X_val_original is None:
                raise ValueError("X_val_original or model_probs is required for fitting.")
            model_probs = self.model.predict_proba(X_val_original)

        distance_matrix = self.stab_space.calc_per_class_1nn_distances(X_val_embed)
        lambda_to_nll = {}
        best_lambda = None
        best_nll = float('inf')

        for lam in self.lambda_grid:
            fused = self._fuse_probs(model_probs, distance_matrix, lam)
            nll = self._multiclass_nll(fused, y_val)
            lambda_to_nll[float(lam)] = nll
            if nll < best_nll:
                best_nll = nll
                best_lambda = float(lam)

        self.best_lambda = best_lambda
        self.lambda_selection = {
            'grid': self.lambda_grid.tolist(),
            'nll_by_lambda': lambda_to_nll,
            'best_lambda': self.best_lambda,
            'best_nll': best_nll,
        }

        if validate_lambda0:
            lambda0_fused = self._fuse_probs(model_probs, distance_matrix, lam=0.0)
            self.lambda0_max_abs_diff = float(np.max(np.abs(lambda0_fused - model_probs)))

        self.is_fitted = True
        return self

    def calibrate(self, X_test_embed, X_test_original=None, model_probs=None, return_details=False):
        if not self.is_fitted:
            raise ValueError("You must fit the calibrator before using it.")
        if model_probs is None:
            if X_test_original is None:
                raise ValueError("X_test_original or model_probs is required for calibration.")
            model_probs = self.model.predict_proba(X_test_original)

        distance_matrix = self.stab_space.calc_per_class_1nn_distances(X_test_embed)
        fused = self._fuse_probs(model_probs, distance_matrix, self.best_lambda)

        if return_details:
            return fused, {
                'base_probs': model_probs,
                'distance_matrix': distance_matrix,
                'score_mode': self.score_mode,
                'lambda': self.best_lambda,
            }
        return fused

    def get_params(self):
        return {
            'metric': self.metric,
            'library': self.library,
            'whitening_components': self.whitening_components,
            'whitening_eps': self.whitening_eps,
            'num_labels': self.num_labels,
            'is_fitted': self.is_fitted,
            'score_mode': self.score_mode,
            'best_lambda': self.best_lambda,
            'lambda_grid': self.lambda_grid.tolist(),
            'lambda_selection': self.lambda_selection,
            'lambda0_max_abs_diff': self.lambda0_max_abs_diff,
        }


class FullVectorDistanceFusionCalibrator(BaseCalibrator):
    """
    Legacy distance-fusion calibrator preserved for backward compatibility.
      q = softmax(log(p_model + eps) - beta * d)

    Internally delegates to FullVectorGeometricFusionCalibrator with
    score_mode='neg_distance'.  All public attributes and get_params() keys
    match the original API exactly.
    """

    DEFAULT_BETA_GRID = np.array(
        [0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30], dtype=np.float64
    )

    def __init__(
        self,
        model,
        X_train_embed,
        y_train,
        metric='l2',
        library='fast_separation',
        compression_mode=None,
        compression_param=None,
        beta_grid=None,
        eps=1e-12,
        stability_space=None,
        whitening_components=128,
        whitening_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.metric = metric.lower()
        self.library = library
        self.whitening_components = int(whitening_components)
        self.whitening_eps = float(whitening_eps)
        self.eps = float(eps)
        self.y_train = y_train
        self.num_labels = len(np.unique(y_train))
        self.beta_grid = (
            np.asarray(beta_grid, dtype=np.float64)
            if beta_grid is not None
            else self.DEFAULT_BETA_GRID.copy()
        )

        self._inner = FullVectorGeometricFusionCalibrator(
            model=model,
            X_train_embed=X_train_embed,
            y_train=y_train,
            metric=metric,
            library=library,
            compression_mode=compression_mode,
            compression_param=compression_param,
            lambda_grid=self.beta_grid,
            score_mode='neg_distance',
            eps=eps,
            stability_space=stability_space,
            whitening_components=self.whitening_components,
            whitening_eps=self.whitening_eps,
        )
        self.stab_space = self._inner.stab_space
        self.metric = self.stab_space.metric
        self.library = self.stab_space.library
        self.whitening_components = self.stab_space.whitening_components
        self.whitening_eps = self.stab_space.whitening_eps

        self.best_beta = None
        self.beta_selection = {}
        self.beta0_max_abs_diff = None

    def fit(
        self,
        X_val_embed,
        y_val,
        X_val_original=None,
        model_probs=None,
        validate_beta0=True,
    ):
        self._inner.fit(
            X_val_embed=X_val_embed,
            y_val=y_val,
            X_val_original=X_val_original,
            model_probs=model_probs,
            validate_lambda0=validate_beta0,
        )
        sel = self._inner.lambda_selection
        self.best_beta = self._inner.best_lambda
        self.beta_selection = {
            'grid': sel['grid'],
            'nll_by_beta': sel['nll_by_lambda'],
            'best_beta': sel['best_lambda'],
            'best_nll': sel['best_nll'],
        }
        self.beta0_max_abs_diff = self._inner.lambda0_max_abs_diff
        self.is_fitted = True
        return self

    def calibrate(self, X_test_embed, X_test_original=None, model_probs=None, return_details=False):
        if not self.is_fitted:
            raise ValueError("You must fit the calibrator before using it.")
        result = self._inner.calibrate(
            X_test_embed=X_test_embed,
            X_test_original=X_test_original,
            model_probs=model_probs,
            return_details=return_details,
        )
        if return_details:
            fused, details = result
            return fused, {
                'base_probs': details['base_probs'],
                'distance_matrix': details['distance_matrix'],
                'beta': details['lambda'],
            }
        return result

    def get_params(self):
        return {
            'metric': self.metric,
            'library': self.library,
            'whitening_components': self.whitening_components,
            'whitening_eps': self.whitening_eps,
            'num_labels': self.num_labels,
            'is_fitted': self.is_fitted,
            'best_beta': self.best_beta,
            'beta_grid': self.beta_grid.tolist(),
            'beta_selection': self.beta_selection,
            'beta0_max_abs_diff': self.beta0_max_abs_diff,
        }


class SoftmaxKNNBlendCalibrator(BaseCalibrator):
    """
    Baseline blend: q = alpha*p_model + (1-alpha)*softmax(-gamma*D)
    where D is the per-class 1-NN distance matrix from training data.
    Tunes alpha and gamma jointly on validation NLL.
    """

    DEFAULT_ALPHA_GRID = np.array(
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=np.float64
    )
    DEFAULT_GAMMA_GRID = np.array(
        [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], dtype=np.float64
    )

    def __init__(
        self,
        model,
        y_train: np.ndarray,
        X_train_embed: Optional[np.ndarray] = None,
        alpha_grid=None,
        gamma_grid=None,
        eps: float = 1e-12,
        stability_space=None,
        metric: str = 'l2',
        library: str = 'fast_separation',
        whitening_components: int = 128,
        whitening_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.eps = float(eps)
        self.metric = metric.lower()
        self.library = library
        self.whitening_components = int(whitening_components)
        self.whitening_eps = float(whitening_eps)
        self.y_train = y_train
        self.num_labels = len(np.unique(y_train))
        self.alpha_grid = (
            np.asarray(alpha_grid, dtype=np.float64)
            if alpha_grid is not None
            else self.DEFAULT_ALPHA_GRID.copy()
        )
        self.gamma_grid = (
            np.asarray(gamma_grid, dtype=np.float64)
            if gamma_grid is not None
            else self.DEFAULT_GAMMA_GRID.copy()
        )

        if stability_space is not None:
            self.stab_space = stability_space
        else:
            if X_train_embed is None:
                raise ValueError(
                    "X_train_embed is required when stability_space is not provided."
                )
            self.stab_space = StabilitySpace(
                X_train_embed,
                y_train,
                library=library,
                metric=self.metric,
                whitening_components=self.whitening_components,
                whitening_eps=self.whitening_eps,
            )

        self.metric = self.stab_space.metric
        self.library = self.stab_space.library
        self.whitening_components = self.stab_space.whitening_components
        self.whitening_eps = self.stab_space.whitening_eps

        self.best_alpha: Optional[float] = None
        self.best_gamma: Optional[float] = None
        self.validation_nll_grid: Optional[np.ndarray] = None
        self.is_fitted = False

    def _knn_probs(self, distance_matrix: np.ndarray, gamma: float) -> np.ndarray:
        """Stable softmax(-gamma * D) over per-class distances."""
        logits = -float(gamma) * np.asarray(distance_matrix, dtype=np.float64)
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits)
        return exp_l / np.clip(exp_l.sum(axis=1, keepdims=True), self.eps, None)

    def _blend(
        self,
        model_probs: np.ndarray,
        distance_matrix: np.ndarray,
        alpha: float,
        gamma: float,
    ) -> np.ndarray:
        p_knn = self._knn_probs(distance_matrix, gamma)
        return float(alpha) * model_probs + (1.0 - float(alpha)) * p_knn

    def _multiclass_nll(self, probs: np.ndarray, labels: np.ndarray) -> float:
        idx = np.arange(len(labels))
        return float(-np.mean(np.log(np.clip(probs[idx, labels], self.eps, 1.0))))

    def fit(
        self,
        X_val_embed: np.ndarray,
        y_val: np.ndarray,
        model_probs: Optional[np.ndarray] = None,
        X_val_original: Optional[np.ndarray] = None,
        val_distance_matrix: Optional[np.ndarray] = None,
        features=None,
    ):
        if model_probs is None:
            if X_val_original is None:
                raise ValueError("X_val_original or model_probs is required for fitting.")
            model_probs = self.model.predict_proba(X_val_original)

        model_probs = np.asarray(model_probs, dtype=np.float64)

        if val_distance_matrix is None:
            val_distance_matrix = self.stab_space.calc_per_class_1nn_distances(X_val_embed)

        n_alpha = len(self.alpha_grid)
        n_gamma = len(self.gamma_grid)
        nll_grid = np.full((n_alpha, n_gamma), np.inf, dtype=np.float64)
        best_nll = float('inf')
        best_alpha: Optional[float] = None
        best_gamma: Optional[float] = None

        for i, alpha in enumerate(self.alpha_grid):
            for j, gamma in enumerate(self.gamma_grid):
                blended = self._blend(model_probs, val_distance_matrix, float(alpha), float(gamma))
                nll = self._multiclass_nll(blended, y_val)
                nll_grid[i, j] = nll
                if nll < best_nll:
                    best_nll = nll
                    best_alpha = float(alpha)
                    best_gamma = float(gamma)

        self.best_alpha = best_alpha
        self.best_gamma = best_gamma
        self.validation_nll_grid = nll_grid
        self.is_fitted = True
        return self

    def calibrate(
        self,
        X_test_embed: np.ndarray,
        model_probs: Optional[np.ndarray] = None,
        X_test_original: Optional[np.ndarray] = None,
        test_distance_matrix: Optional[np.ndarray] = None,
        features=None,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("You must fit the calibrator before using it.")
        if model_probs is None:
            if X_test_original is None:
                raise ValueError("X_test_original or model_probs is required for calibration.")
            model_probs = self.model.predict_proba(X_test_original)

        model_probs = np.asarray(model_probs, dtype=np.float64)

        if test_distance_matrix is None:
            test_distance_matrix = self.stab_space.calc_per_class_1nn_distances(X_test_embed)

        return self._blend(model_probs, test_distance_matrix, self.best_alpha, self.best_gamma)

    def get_params(self) -> dict:
        return {
            'metric': self.metric,
            'library': self.library,
            'whitening_components': self.whitening_components,
            'whitening_eps': self.whitening_eps,
            'num_labels': self.num_labels,
            'is_fitted': self.is_fitted,
            'selected_alpha': self.best_alpha,
            'selected_gamma': self.best_gamma,
            'alpha_grid': self.alpha_grid.tolist(),
            'gamma_grid': self.gamma_grid.tolist(),
            'validation_nll_grid': (
                self.validation_nll_grid.tolist()
                if self.validation_nll_grid is not None
                else None
            ),
        }
