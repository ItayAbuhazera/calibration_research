# Calibrators/geometric_calibrator_new.py
"""
Geometric Calibrator using Fast Separation + Isotonic Regression
Enhanced with ECE (Expected Calibration Error) calculation for layer selection
"""

import numpy as np
import logging
import torch
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from .base_calibrator import BaseCalibrator
from utils.stability_space import StabilitySpace
import time
from typing import Dict, List, Tuple, Optional, Callable, Union
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.stats import kendalltau, pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from utils.compression_utils import FixedSizeSPP_JL, SmartCompression
# import nn modules
import torch.nn as nn
from .metrics import (
    MetricCalculator,
    # Containers / utilities
    LayerMetrics,
    AdHocLayerSelector,
    CrossModalLayerMapper,
    # Calibration helpers (new metrics)
    SharedCalibrationCalculator,
    OptimizedGeometricECECalculator,          # provides "ece_score"
    OptimizedCalibrationDecisivenessCalculator,  # "calibration_decisiveness"
    OptimizedStabilityAccuracyCorr,           # "spearman_stability_accuracy"
    
)

from utils.logging_config import get_logger
logger = get_logger(__name__)


def smoke_test_layers(model, dataloader, layer_idxs, device):
    """One-batch smoke test to quickly catch hook/coercion issues."""
    from utils.tensor_utils import coerce_to_tensor
    
    model.eval()
    x, _ = next(iter(dataloader))
    x = x.to(device)
    layers = list(model.modules())

    for idx in layer_idxs[:10]:  # test a subset
        act = {}
        h = layers[idx].register_forward_hook(lambda m, i, o: act.setdefault('out', o))
        try:
            _ = model(x)
            t = coerce_to_tensor(act.get('out'))
            msg = f"ok tensor {tuple(t.shape)}" if torch.is_tensor(t) else f"BAD {type(act.get('out'))}"
            print(f"[{idx:>4}] {layers[idx].__class__.__name__:<28} → {msg}")
        finally:
            h.remove()


# LayerMetrics is now imported from .metrics


"""MetricCalculator moved to Calibrators/metrics.py"""


# ============================================================================
class GeometricCalibrator(BaseCalibrator):
    """
    Geometric calibration using stability/separation scores and isotonic regression.
    Enhanced with ECE-based layer selection.
    """

    def __init__(self, model, X_train_embed=None, y_train=None, X_train_original=None, 
                 compression_mode=None, compression_param=None, compression_ratio=None,
                 metric='l2', stability_space=None, 
                 library='faiss', use_binning=True, n_bins=200,
                 # NEW PARAMETERS
                 use_augmix=False, augmix_versions=3, augmix_severity=3, 
                 augmix_width=3, augmix_alpha=1.0,
                 # Layer selection parameters
                 auto_select_layer: bool = False,
                 candidate_layers: Optional[List[int]] = None,
                 custom_metrics: Optional[List[MetricCalculator]] = None,
                 score_weights: Optional[Dict[str, float]] = None,
                 use_robustness_check: bool = False,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 # NEW: force computing all metrics regardless of weights
                 compute_all_metrics: bool = False,
                 fitting_method: str = 'isotonic',  # Options: 'isotonic', 'beta', 'platt', 'logistic'
                 scoring_method: str = 'separation',  # Options: 'separation' (SGC) or 'trust_score' (Trust Score)
                 normalization_method: str = 'rank',  # 'rank' | 'linear' | 'percentile'
                 **kwargs):
        """
        Initializes the GeometricCalibrator with isotonic regression.
        """
        logging.info("\n=== Enhanced GeometricCalibrator Initialization with ECE ===")
        if X_train_embed is not None:
            logging.info(f"Input X_train_embed shape: {X_train_embed.shape}")
        if X_train_original is not None:
            logging.info(f"Input X_train_original shape: {X_train_original.shape}")
        logging.info(f"Compression mode: {compression_mode}")
        logging.info(f"Compression param: {compression_param}")
        logging.info(f"Library: {library}")
        logging.info(f"Metric: {metric}")
        logging.info(f"Auto select layer: {auto_select_layer}")

        super().__init__(**kwargs)
        self.model = model
        self.isotonic_regressor = None
        self.regressor = None
        self.fitting_method = fitting_method
        if normalization_method not in ('rank', 'linear', 'percentile'):
            raise ValueError(f"Unknown normalization_method: {normalization_method}")
        self.normalization_method = normalization_method
        self.val_scores_sorted = None
        self.val_score_min = None
        self.val_score_max = None
        self.val_score_p5 = None
        self.val_score_p95 = None
        self.is_fitted = False
        self.metric = metric.lower()
        self.use_binning = use_binning
        self.n_bins = n_bins
        self.auto_select_layer = auto_select_layer
        self.candidate_layers = candidate_layers
        self.custom_metrics = custom_metrics
        self.use_robustness_check = use_robustness_check
        self.device = device
        self.layer_selector = None
        self.feature_layer = None
        self.robustness_result = None
        self.layer_metrics = None
        self.compression_mode = compression_mode
        self.compression_param = compression_param
        self.compression_ratio = compression_ratio
        self.library = library
        self.score_weights = score_weights
        self.compute_all_metrics = compute_all_metrics
        
        # Validate and store scoring method
        if scoring_method not in ['separation', 'trust_score']:
            raise ValueError(f"scoring_method must be 'separation' or 'trust_score', got '{scoring_method}'")
        self.scoring_method = scoring_method
        logger.info(f"📊 Scoring method: {scoring_method} ({'SGC separation' if scoring_method == 'separation' else 'Trust Score'})")

        # AugMix parameters
        self.use_augmix = use_augmix
        self.augmix_versions = augmix_versions if use_augmix else 0
        self.augmix_severity = augmix_severity
        self.augmix_width = augmix_width
        self.augmix_alpha = augmix_alpha
        
        if self.use_augmix:
            from Data.augmix_transforms import AugMixTransforms
            self.augmix_transform = AugMixTransforms(severity=augmix_severity, width=augmix_width, depth=-1, alpha=augmix_alpha)
            logger.info(f"🌪️ AugMix enabled for calibration fitting: {augmix_versions} versions, severity={augmix_severity}")
        else:
            self.augmix_transform = None

        if y_train is not None:
            self.num_labels = len(np.unique(y_train))
        else:
            self.num_labels = None

        if self.auto_select_layer:
            if X_train_original is None:
                raise ValueError("X_train_original is required when auto_select_layer is True")
            self.X_train_original = X_train_original
            self.y_train = y_train
            self.stab_space = None
        else:
            if X_train_embed is None:
                raise ValueError("X_train_embed is required when auto_select_layer is False")
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
            logger.info("   📊 Using Isotonic Regression (non-parametric)")
            return IsotonicRegression(out_of_bounds="clip")
        
        elif self.fitting_method == 'beta':
            from Calibrators.beta_calibration import _BetaCal
            logger.info("   📊 Using Beta Calibration (3 parameters: a, b, m)")
            return _BetaCal()
        
        elif self.fitting_method == 'platt':
            from sklearn.linear_model import LogisticRegression
            logger.info("   📊 Using Platt Scaling (logistic regression)")
            # Platt scaling: logit(p) = a*s + b
            return LogisticRegression(
                penalty=None,
                max_iter=1000,
                random_state=42
            )
        
        elif self.fitting_method == 'logistic':
            from sklearn.linear_model import LogisticRegression
            logger.info("   📊 Using Logistic Regression with log transforms")
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

    def fit(self, X_val_embed: Optional[np.ndarray] = None, y_val=None, X_val_original: Optional[np.ndarray] = None, features=None, fit_batch_size=1000, corruption_dataloaders: Optional[Dict[str, DataLoader]] = None, model_probs: Optional[np.ndarray] = None):
        """
        🚀 OPTIMIZED fit method - only computes metrics with non-zero weights
        """
        logger.info(f"{self.__class__.__name__}: OPTIMIZED fitting with selective metric computation.")
        
        if self.auto_select_layer:
            if X_val_original is None or y_val is None:
                raise ValueError("X_val_original and y_val are required for auto_select_layer")
            
            # Split validation data (33-67 split)
            logger.info("🎯 Validation split strategy (33% train, 67% evaluate):")
            total_samples = len(X_val_original)
            from sklearn.model_selection import train_test_split
            cal_train_indices, cal_eval_indices = train_test_split(
                np.arange(total_samples),
                test_size=0.67, train_size=0.33, 
                stratify=y_val, random_state=42
            )
            X_cal_train, y_cal_train = X_val_original[cal_train_indices], y_val[cal_train_indices]
            X_cal_eval, y_cal_eval = X_val_original[cal_eval_indices], y_val[cal_eval_indices]

            # Ensure float32 to reduce memory
            if isinstance(X_cal_train, np.ndarray):
                X_cal_train = X_cal_train.astype(np.float32, copy=False)
            if isinstance(X_cal_eval, np.ndarray):
                X_cal_eval = X_cal_eval.astype(np.float32, copy=False)

            # --- SUBSET THE PRE-CALCULATED PROBABILITIES (if provided) ---
            # Align probabilities with the 67% evaluation split
            initial_probs_cal_eval = None
            if model_probs is not None:
                try:
                    initial_probs_cal_eval = model_probs[cal_eval_indices]
                    logger.info(f"Using precomputed model_probs for eval split: {initial_probs_cal_eval.shape}")
                except Exception as e:
                    logger.warning(f"Failed to subset provided model_probs for eval split: {e}")
            
            logger.info(f"   📊 Calibration training: {len(X_cal_train)} samples")
            logger.info(f"   📈 Calibration evaluation: {len(X_cal_eval)} samples")
            
            # Create dataloaders
            cal_train_dataloader = DataLoader(
                TensorDataset(torch.from_numpy(X_cal_train), torch.from_numpy(y_cal_train)), 
                batch_size=fit_batch_size
            )
            cal_eval_dataloader = DataLoader(
                TensorDataset(torch.from_numpy(X_cal_eval), torch.from_numpy(y_cal_eval)), 
                batch_size=fit_batch_size
            )
            train_dataloader = DataLoader(
                TensorDataset(torch.from_numpy(self.X_train_original), torch.from_numpy(self.y_train)), 
                batch_size=fit_batch_size
            )
            
            # =====================================================================
            # 🚀 OPTIMIZE METRIC SELECTION BASED ON WEIGHTS
            # =====================================================================
            
            logger.info("🚀 Optimizing metric computation based on score weights...")
            
            # Check which special metrics we need (honor compute_all_metrics and None weights)
            w = self.score_weights or {}
            needs_ece = (self.compute_all_metrics or 
                         w.get("ece_score", 0.0) != 0.0)
            needs_decisiveness = (self.compute_all_metrics or 
                                  w.get("calibration_decisiveness", 0.0) != 0.0)
            needs_stability_corr = (self.compute_all_metrics or 
                                    w.get("spearman_stability_accuracy", 0.0) != 0.0)
            
            logger.info(f"   ECE Score needed: {needs_ece}")
            logger.info(f"   Calibration Decisiveness needed: {needs_decisiveness}")
            logger.info(f"   Stability-Accuracy Correlation needed: {needs_stability_corr}")
            
            # Only create the calculators we need
            special_calculators = []
            shared_calibration_calc = None
            
            if needs_ece or needs_decisiveness or needs_stability_corr:
                # We need shared calibration calculator
                shared_calibration_calc = SharedCalibrationCalculator(
                    model_adapter=self.model, 
                    device=self.device,
                    compression_mode=self.compression_mode,
                    compression_param=self.compression_param,
                    library=self.library, 
                    metric=self.metric
                )
                logger.info("   ✅ Created shared calibration calculator")
                
                if needs_ece:
                    ece_calculator = OptimizedGeometricECECalculator(n_bins=15)
                    special_calculators.append(ece_calculator)
                    logger.info("   ✅ Created ECE calculator")
                
                if needs_decisiveness:
                    decisiveness_calculator = OptimizedCalibrationDecisivenessCalculator()
                    special_calculators.append(decisiveness_calculator)
                    logger.info("   ✅ Created decisiveness calculator")
                
                if needs_stability_corr:
                    stab_acc_corr_calc = OptimizedStabilityAccuracyCorr(n_bins=0)
                    special_calculators.append(stab_acc_corr_calc)
                    logger.info("   ✅ Created stability-accuracy correlation calculator")
            else:
                logger.info("   ⏭️ No special calibration metrics needed - skipping expensive calculations")
            
            # Create the layer selector; when compute_all_metrics=True don't skip zero-weight metrics
            self.layer_selector = AdHocLayerSelector(
                metrics=self.custom_metrics,  # Will be filtered based on weights
                score_weights=self.score_weights,
                cache_activations=False,
                compression_ratio=self.compression_ratio,
                skip_zero_weight_metrics=not self.compute_all_metrics
            )
            
            # Add special calculators to the filtered base metrics
            if special_calculators:
                # Get the already-filtered base metrics
                base_metrics = self.layer_selector.metrics
                enhanced_metrics = base_metrics + special_calculators
                logger.info(f"   📊 Total active metrics: {len(enhanced_metrics)} ({len(base_metrics)} base + {len(special_calculators)} special)")
            else:
                enhanced_metrics = self.layer_selector.metrics
                logger.info(f"   📊 Total active metrics: {len(enhanced_metrics)} (base only)")
            
            # Get model and device
            raw_pytorch_model = self.model.model
            if self.candidate_layers is None:
                self.candidate_layers = CrossModalLayerMapper.get_candidate_indices(raw_pytorch_model)
            device_obj = torch.device(self.device)
            
            # Pre-calculate predictions only if needed for uncertainty metrics
            probs_cal_eval = initial_probs_cal_eval
            predictions_cal_eval = None
            confidences_cal_eval = None
            
            # Check if any active metric needs predictions/confidences
            active_metric_names = {getattr(m, "name", "") for m in enhanced_metrics}
            needs_predictions = any(name in {
                "confidence_distance_correlation",
                "boundary_proximity_correlation",
                "uncertainty_geometry_alignment"
            } for name in active_metric_names)
            
            if needs_predictions or special_calculators:
                # Use pre-calculated probs if available, otherwise compute them now
                if probs_cal_eval is not None:
                    logger.info("Using pre-calculated probabilities for uncertainty metrics...")
                else:
                    logger.info("`model_probs` not provided, calculating them now for the eval split (batched)...")
                    bs_pred = min(256, fit_batch_size)
                    probs_cal_eval = self.model.predict_proba(X_cal_eval, batch_size=bs_pred)
                predictions_cal_eval = np.argmax(probs_cal_eval, axis=1)
                confidences_cal_eval = np.max(probs_cal_eval, axis=1)
                logger.info(f"  ✅ Prepared predictions for {len(predictions_cal_eval)} samples")
            else:
                logger.info("   ⏭️ No uncertainty metrics needed - skipping prediction calculation")
            
            logger.info("🚀 Starting OPTIMIZED layer selection...")
            all_metrics = []
            
            # =====================================================================
            # OPTIMIZED LAYER EVALUATION LOOP
            # =====================================================================
            
            for layer_idx in self.candidate_layers:
                logger.info(f"🔍 Evaluating layer {layer_idx} with {len(enhanced_metrics)} active metrics...")
                
                # Extract features for all data splits
                features_cal_train, labels_cal_train = self.layer_selector._extract_features(
                    raw_pytorch_model, cal_train_dataloader, layer_idx, device_obj
                )
                features_cal_eval, labels_cal_eval = self.layer_selector._extract_features(
                    raw_pytorch_model, cal_eval_dataloader, layer_idx, device_obj
                )
                train_features, train_labels = self.layer_selector._extract_features(
                    raw_pytorch_model, train_dataloader, layer_idx, device_obj
                )
                
                # Compute shared calibration data only if needed
                shared_calibration_data = None
                if shared_calibration_calc is not None:
                    logger.debug(f"   🔧 Computing shared calibration data for layer {layer_idx}...")
                    shared_calibration_data = shared_calibration_calc.compute_shared_calibration_data(
                        train_features=train_features.numpy(),
                        train_labels=train_labels.numpy(),
                        val_features=features_cal_eval.numpy(),
                        val_labels=labels_cal_eval.numpy(),
                        val_original=X_cal_eval,
                        layer_idx=layer_idx,
                        # Pass precomputed probabilities when available to avoid recomputation
                        val_model_probs=probs_cal_eval
                    )
                    
                    # Set shared data for special calculators
                    for calculator in special_calculators:
                        if hasattr(calculator, 'set_shared_data'):
                            if isinstance(calculator, OptimizedGeometricECECalculator):
                                calculator.set_shared_data(shared_calibration_data, labels_cal_eval.numpy())
                            elif isinstance(calculator, OptimizedCalibrationDecisivenessCalculator):
                                calculator.set_shared_data(shared_calibration_data)
                            elif isinstance(calculator, OptimizedStabilityAccuracyCorr):
                                calculator.set_shared_data(shared_calibration_data, labels_cal_eval.numpy())
                    # Ensure all metrics that can leverage shared data (including KFoldECEUtilityCalculator)
                    # receive it, not only the special_calculators list.
                    try:
                        self.layer_selector._set_shared_data_for_metrics(shared_calibration_data, labels_cal_eval.numpy())
                    except Exception as e:
                        logger.warning(f"Failed to set shared data for base metrics: {e}")
                
                # Compute metrics using the FILTERED metrics list
                metrics = LayerMetrics(layer_idx=layer_idx)
                
                for calculator in enhanced_metrics:
                    try:
                        value = 0.0
                        
                        # Special calibration-dependent metrics
                        if isinstance(calculator, (OptimizedGeometricECECalculator, 
                                                OptimizedCalibrationDecisivenessCalculator,
                                                OptimizedStabilityAccuracyCorr)):
                            value = calculator.compute(features_cal_eval, labels_cal_eval)
                        
                        # Base geometric/uncertainty metrics
                        else:
                            logger.debug(f"   Computing {calculator.name} for layer {layer_idx}...")
                            logger.debug(f"      features shape: {features_cal_eval.shape if hasattr(features_cal_eval, 'shape') else type(features_cal_eval)}")
                            logger.debug(f"      labels shape: {labels_cal_eval.shape if hasattr(labels_cal_eval, 'shape') else type(labels_cal_eval)}")
                            value = calculator.compute(
                                features=features_cal_eval,
                                labels=labels_cal_eval,
                                corrupted_features=None,
                                predictions=predictions_cal_eval,
                                confidences=confidences_cal_eval,
                                model_probs=probs_cal_eval
                            )
                            logger.debug(f"      {calculator.name} computed value: {value:.6f}")
                        
                        setattr(metrics, calculator.name, value)
                        logger.info(f"   ✓ {calculator.name}: {value:.6f}")
                        
                    except Exception as e:
                        import traceback
                        logger.warning(f"Failed to compute {calculator.name} for layer {layer_idx}: {e}")
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                        setattr(metrics, calculator.name, 0.0)
                
                # Compute composite score
                metrics.composite_score = self.layer_selector._compute_composite_score(metrics)
                all_metrics.append(metrics)
                
                # Enhanced logging
                logger.info(f"✅ Layer {layer_idx} evaluation complete:")
                logger.info(f"   📊 Composite Score: {metrics.composite_score:.4f}")
                
                # Only log metrics that were actually computed
                if needs_ece:
                    logger.info(f"   📉 ECE Score: {getattr(metrics, 'ece_score', 0):.4f}")
                if needs_decisiveness:
                    logger.info(f"   🎯 Decisiveness: {getattr(metrics, 'calibration_decisiveness', 0):.4f}")
                if needs_stability_corr:
                    logger.info(f"   📈 Stability-Accuracy Corr: {getattr(metrics, 'spearman_stability_accuracy', 0):.4f}")
                logger.info(f"   📊 k-fold ECE: {getattr(metrics, 'kfold_ece_utility', 0):.4f}")
                logger.info(f"   📊 Margin Tail CVaR: {getattr(metrics, 'margin_tail_cvar', 0):.4f}")
                
                # --- Memory cleanup for this layer ---
                try:
                    del features_cal_train, labels_cal_train
                    del features_cal_eval, labels_cal_eval
                    del train_features, train_labels
                except Exception:
                    pass
                # Removed torch.cuda.empty_cache() from tight loops for performance
            
            # Select best layer
            best_idx = np.argmax([m.composite_score for m in all_metrics])
            self.feature_layer = self.candidate_layers[best_idx]
            self.layer_metrics = all_metrics
            
            logger.info(f"🏆 Selected layer {self.feature_layer} with score {all_metrics[best_idx].composite_score:.4f}")
            logger.info(f"   🚀 Optimization: Used only {len(enhanced_metrics)} metrics instead of all possible metrics")
            
            # Final feature extraction for the selected layer (use full validation set)
            val_dataloader_full = DataLoader(
                TensorDataset(torch.from_numpy(X_val_original), torch.from_numpy(y_val)), 
                batch_size=fit_batch_size
            )
            val_features_final, _ = self.layer_selector._extract_features(
                raw_pytorch_model, val_dataloader_full, self.feature_layer, device_obj
            )
            train_features_final, _ = self.layer_selector._extract_features(
                raw_pytorch_model, train_dataloader, self.feature_layer, device_obj
            )
            
            X_train_embed = train_features_final.numpy()
            X_val_embed = val_features_final.numpy()
            
            compression = SmartCompression(self.compression_mode, self.compression_param) if self.compression_mode else None
            self.stab_space = StabilitySpace(X_train_embed, self.y_train, compression=compression, library=self.library, metric=self.metric)
        else:
            # Feature mode guard: require original images for predict_proba
            if X_val_original is None:
                raise ValueError(
                    "Feature mode: X_val_original (raw inputs) is required for predict_proba during fitting. "
                    "Alternatively, add a model_probs argument and skip predict_proba."
                )
        
        # =====================================================================
        # 🔄 AUGMIX HANDLING (if enabled)
        # =====================================================================
        
        # Handle AugMix data augmentation for training
        if self.use_augmix:
            logger.info(f"🌪️ Creating AugMix augmented validation data ({self.augmix_versions} versions)...")
            X_val_original_expanded, X_val_embed_expanded, y_val_expanded = self._create_augmented_validation_data(
                X_val_embed, y_val, X_val_original
            )
            prediction_data, embedding_data, labels_data = X_val_original_expanded, X_val_embed_expanded, y_val_expanded
            logger.info(f"   ✅ Augmented data created: {len(labels_data)} total samples")
        else:
            # X_val_original is guaranteed to be available (checked above)
            prediction_data = X_val_original
            embedding_data = X_val_embed
            labels_data = y_val
            logger.info(f"   📊 Using original validation data: {len(labels_data)} samples")
        
        # =====================================================================
        # 📊 BATCHED STABILITY CALCULATION AND PREDICTION PROCESSING
        # =====================================================================
        
        score_name = "Trust Score" if self.scoring_method == 'trust_score' else "Stability"
        logger.info(f"🧮 Computing {score_name.lower()} values and predictions in batches...")
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
        
        logger.info(f"✅ {score_name} calculation complete:")
        logger.info(f"   📊 {score_name} range: [{np.min(stability_val):.6f}, {np.max(stability_val):.6f}]")
        logger.info(f"   📈 Mean {score_name.lower()}: {np.mean(stability_val):.6f}")
        
        # =====================================================================
        # 📏 STABILITY NORMALIZATION
        # =====================================================================
        
        logger.info(f"📏 Applying {self.normalization_method} validation-score normalization...")
        logger.info(f"   📊 Raw score range: [{np.min(stability_val):.6f}, {np.max(stability_val):.6f}]")
        logger.info(f"   📊 Raw mean score: {np.mean(stability_val):.6f}")
        
        if self.normalization_method == 'rank':
            self.val_scores_sorted = np.sort(stability_val)
            from scipy.stats import rankdata
            stability_val = rankdata(stability_val, method='average') / len(self.val_scores_sorted)
            self.val_score_min = None
            self.val_score_max = None
            self.val_score_p5 = None
            self.val_score_p95 = None
            logger.info(f"   📊 Stored {len(self.val_scores_sorted)} sorted validation scores")
        
        elif self.normalization_method == 'linear':
            self.val_score_min = float(np.min(stability_val))
            self.val_score_max = float(np.max(stability_val))
            span = self.val_score_max - self.val_score_min
            if span <= 0:
                raise ValueError("Linear normalization: degenerate val score range")
            stability_val = np.clip((stability_val - self.val_score_min) / span, 0.0, 1.0)
            self.val_scores_sorted = None
            self.val_score_p5 = None
            self.val_score_p95 = None
        
        elif self.normalization_method == 'percentile':
            self.val_score_p5 = float(np.percentile(stability_val, 5))
            self.val_score_p95 = float(np.percentile(stability_val, 95))
            span = self.val_score_p95 - self.val_score_p5
            if span <= 0:
                raise ValueError("Percentile normalization: degenerate p5/p95 range")
            stability_val = np.clip((stability_val - self.val_score_p5) / span, 0.0, 1.0)
            self.val_scores_sorted = None
            self.val_score_min = None
            self.val_score_max = None
        
        else:
            raise ValueError(f"Unknown normalization_method: {self.normalization_method}")
        
        logger.info(f"   ✅ Applied {self.normalization_method} normalization")
        logger.info(f"   📊 Normalized range: [{np.min(stability_val):.6f}, {np.max(stability_val):.6f}]")
        
        # =====================================================================
        # 📦 BINNING (if enabled) 
        # =====================================================================
        
        # Only use binning for isotonic regression; parametric methods need raw data
        use_binning_effective = self.use_binning and (self.fitting_method == 'isotonic')
        
        # Log warning if binning was requested but disabled
        if self.use_binning and not use_binning_effective:
            logger.warning(f"⚠️ Binning disabled for {self.fitting_method} calibration (requires raw data).")

        if use_binning_effective:
            logger.info(f"📦 Applying binning with {self.n_bins} bins...")
            
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
            
            logger.info(f"   ✅ Created {len(stability_vals)} bins with data")
            logger.info(f"   📊 Accuracy range: [{np.min(accuracies):.4f}, {np.max(accuracies):.4f}]")
            
        else:
            logger.info("📊 Using individual samples (no binning)")
            stability_vals = stability_val
            accuracies = (y_val_processed == y_pred_classes).astype(float)
            
            logger.info(f"   📊 Accuracy: {np.mean(accuracies):.4f}")
        
        # =====================================================================
        # 🎯 REGRESSOR FITTING
        # =====================================================================
        
        logger.info(f"🎯 Fitting {self.fitting_method} regressor...")
        
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
        logger.info(f"   📊 Fitting on {len(X_fit)} samples...")
        self.regressor.fit(X_fit, y_fit)
        
        # Log fitted parameters if available
        if self.fitting_method == 'beta' and hasattr(self.regressor, 'map_'):
            a, b, m = self.regressor.map_
            logger.info(f"   ✅ Beta parameters: a={a:.4f}, b={b:.4f}, m={m:.4f}")
        
        elif self.fitting_method in ['platt', 'logistic']:
            if hasattr(self.regressor, 'coef_') and hasattr(self.regressor, 'intercept_'):
                coef = self.regressor.coef_[0] if len(self.regressor.coef_.shape) > 1 else self.regressor.coef_
                intercept = self.regressor.intercept_[0] if hasattr(self.regressor.intercept_, '__len__') else self.regressor.intercept_
                logger.info(f"   ✅ Logistic parameters: coef={coef}, intercept={intercept:.4f}")
        
        elif self.fitting_method == 'isotonic':
            if hasattr(self.regressor, 'X_thresholds_'):
                n_thresholds = len(self.regressor.X_thresholds_)
                logger.info(f"   ✅ Isotonic fit: {n_thresholds} thresholds")
        
        # Initialize storage for stability scores (for per-sample saving)
        self.last_stability_scores = None
        
        # Mark as fitted
        self.is_fitted = True
        
        logger.info(f"✅ {self.__class__.__name__} fitting complete!")
        logger.info(f"   📊 Trained on {len(stability_vals)} {'bins' if self.use_binning else 'samples'}")
        logger.info(f"   🎯 Method: {self.fitting_method}")
        logger.info(f"   💾 Normalization method: {self.normalization_method}")
        
        # Log optimization summary
        if self.auto_select_layer and hasattr(self, 'layer_selector'):
            total_possible_metrics = len(self.layer_selector.original_metrics) if hasattr(self.layer_selector, 'original_metrics') else len(enhanced_metrics)
            logger.info(f"🚀 Optimization Summary:")
            logger.info(f"   📊 Computed {len(enhanced_metrics)}/{total_possible_metrics} metrics")
            logger.info(f"   ⚡ Estimated speedup: {total_possible_metrics/len(enhanced_metrics):.1f}x")

    def calibrate_batched(self, X_test_embed: Optional[np.ndarray] = None, X_test_original: Optional[np.ndarray] = None, features=None, batch_size=1000, return_probs=False):
        """
        Memory-efficient batched calibration to prevent GPU OOM.
        If auto_select_layer, extracts X_test_embed from X_test_original using selected layer.
        
        Args:
            return_probs: If True, returns (calibrated_probs, model_probs) tuple.
                         model_probs are the raw model predictions computed during calibration.
        """
        if not self.is_fitted:
            raise ValueError("You must fit the calibrator before using it.")

        if self.auto_select_layer:
            if X_test_original is None:
                raise ValueError("X_test_original required for auto_select_layer")
            if self.layer_selector is None:
                raise ValueError("Layer selector not initialized - fit first")
            
            dummy_y = np.zeros((len(X_test_original),), dtype=int)
            X_test_original_tensor = torch.tensor(X_test_original) if isinstance(X_test_original, np.ndarray) else X_test_original
            dummy_y_tensor = torch.tensor(dummy_y)
            test_dataset = TensorDataset(X_test_original_tensor, dummy_y_tensor)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            test_features, _ = self.layer_selector._extract_features(self.model.model, test_dataloader, self.feature_layer, torch.device(self.device))
            X_test_embed = test_features.numpy() if isinstance(test_features, torch.Tensor) else test_features
        else:
            # Feature mode guard: require original images for predict_proba
            if X_test_original is None:
                raise ValueError(
                    "Feature mode: X_test_original (raw inputs) is required so we can call predict_proba. "
                    "Alternatively, extend the API to accept precomputed model_probs."
                )
        
        logger.info(f"GeometricCalibrator: Calibrating test data in batches of {batch_size}.")
        n_samples = X_test_original.shape[0]
        
        # Guard against missing num_labels
        if self.num_labels is None:
            # lazily infer from model on a tiny batch
            tmp = self.model.predict_proba(X_test_original[:1])
            self.num_labels = tmp.shape[1]
            logger.info(f"🔧 Inferred num_labels = {self.num_labels} from model output")
        
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
                    logger.error(f"❌ Calibration batch {batch_idx + 1} failed: {e}")
                    all_calibrated_probs[start_idx:end_idx] = 1.0 / num_classes
                    all_stability_scores[start_idx:end_idx] = 0.5  # Default stability
                    pbar.update(1)
        total_time = time.time() - start_time
        logger.info(f"🏁 Batched calibration completed in {total_time:.1f}s")
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
                    logger.error(f"❌ Calibration batch {batch_idx + 1} failed: {e}")
                    all_calibrated_probs[start_idx:end_idx] = 1.0 / num_classes
                    all_stability_scores[start_idx:end_idx] = 0.5  # Default stability
                    pbar.update(1)
        
        # Store stability scores for per-sample saving
        self.last_stability_scores = all_stability_scores
                    
        return all_calibrated_probs

    def calibrate(self, X_test_embed: Optional[np.ndarray] = None, X_test_original: Optional[np.ndarray] = None, features=None):
        """
        Calibrates the test data using the fitted isotonic regression model.
        If auto_select_layer, extracts X_test_embed from X_test_original using selected layer.
        """
        if not self.is_fitted:
            raise ValueError("You must fit the calibrator before using it.")

        if self.auto_select_layer:
            if X_test_original is None:
                raise ValueError("X_test_original required for auto_select_layer")
            if self.layer_selector is None:
                raise ValueError("Layer selector not initialized - fit first")
            
            dummy_y = np.zeros((len(X_test_original),), dtype=int)
            X_test_original_tensor = torch.tensor(X_test_original) if isinstance(X_test_original, np.ndarray) else X_test_original
            dummy_y_tensor = torch.tensor(dummy_y)
            test_dataset = TensorDataset(X_test_original_tensor, dummy_y_tensor)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            test_features, _ = self.layer_selector._extract_features(self.model.model, test_dataloader, self.feature_layer, torch.device(self.device))
            X_test_embed = test_features.numpy() if isinstance(test_features, torch.Tensor) else test_features
        else:
            # Feature mode guard: require original images for predict_proba
            if X_test_original is None:
                raise ValueError(
                    "Feature mode: X_test_original (raw inputs) is required so we can call predict_proba. "
                    "Alternatively, extend the API to accept precomputed model_probs."
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

    def _normalize_scores(self, stability_batch):
        if self.normalization_method == 'rank':
            return np.searchsorted(self.val_scores_sorted, stability_batch) / len(self.val_scores_sorted)
        if self.normalization_method == 'linear':
            span = self.val_score_max - self.val_score_min
            return np.clip((stability_batch - self.val_score_min) / span, 0.0, 1.0)
        if self.normalization_method == 'percentile':
            span = self.val_score_p95 - self.val_score_p5
            return np.clip((stability_batch - self.val_score_p5) / span, 0.0, 1.0)
        raise ValueError(f"Unknown normalization_method: {self.normalization_method}")

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
        
        # Normalize stability values using the validation-set transform.
        normalized_stability_batch = self._normalize_scores(stability_batch)
        
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
        
        # 🔧 DEBUG: Add detailed logging for data validation
        logger.debug(f"🔍 _calibrate_single_batch received:")
        logger.debug(f"   batch_embed shape: {batch_embed.shape}")
        logger.debug(f"   batch_embed dtype: {batch_embed.dtype}")
        logger.debug(f"   batch_original shape: {batch_original.shape}")
        logger.debug(f"   batch_original dtype: {batch_original.dtype}")
        
        # 🔧 CRITICAL FIX: Detect if logits are accidentally passed instead of raw images
        if len(batch_original.shape) == 2 and self.num_labels is not None and batch_original.shape[1] == self.num_labels and batch_original.dtype in (np.float32, np.float64):
            logger.error(f"❌ CRITICAL ERROR: Logits detected in batch_original instead of raw images!")
            logger.error(f"   batch_original shape: {batch_original.shape} (looks like logits for {batch_original.shape[1]} classes)")
            logger.error(f"   Expected raw input data, not probabilities/logits")
            logger.error(f"   This indicates a data flow bug in the calibration pipeline")
            raise ValueError(f"Probabilities/logits detected in X_test_original; pass raw inputs instead.")
        
        # Validate data types for physical vs semantic methods
        if len(batch_embed.shape) == 4 and batch_embed.shape[1] in [1, 3]:
            logger.debug(f"   🔍 Physical method detected: batch_embed appears to be raw images")
        elif len(batch_embed.shape) == 2:
            logger.debug(f"   🔍 Semantic method detected: batch_embed appears to be features")
        else:
            logger.warning(f"   ⚠️ Unexpected batch_embed shape: {batch_embed.shape}")
        
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
        
        # Normalize stability values using the validation-set transform.
        normalized_stability_batch = self._normalize_scores(stability_batch)
        
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
        
        logger.info("🔄 Creating augmented validation data...")
        
        # Detect calibration type based on data shapes
        is_physical_calibration = (X_val_embed.shape == X_val_original.shape and 
                                 len(X_val_embed.shape) == 4)
        
        logger.info(f"📊 Calibration type: {'Physical' if is_physical_calibration else 'Semantic'}")
        
        # Storage for expanded data
        original_images_list = [X_val_original]
        expanded_labels_list = [y_val]
        
        if is_physical_calibration:
            # For Physical calibration: keep everything as raw images
            augmented_embeddings_list = [X_val_embed]  # Same as original (raw images)
            
            logger.info("🔍 Physical calibration: keeping all data as raw images")
            
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
            
            logger.info("🧠 Semantic calibration: extracting features from augmented images")
            
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
            
            logger.info(f"✅ Augmented validation data created:")
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
            'normalization_method': self.normalization_method,
            'val_score_min': self.val_score_min,
            'val_score_max': self.val_score_max,
            'val_score_p5': self.val_score_p5,
            'val_score_p95': self.val_score_p95,
            'val_scores_count': int(0 if self.val_scores_sorted is None else len(self.val_scores_sorted)),
            # AugMix parameters
            'use_augmix': self.use_augmix,
            'augmix_versions': self.augmix_versions,
            'augmix_severity': self.augmix_severity,
            'augmix_width': self.augmix_width,
            'augmix_alpha': self.augmix_alpha,
            # Layer selection params
            'auto_select_layer': self.auto_select_layer,
            'use_robustness_check': self.use_robustness_check,
            'feature_layer': self.feature_layer
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
