"""
In-Training Geometric Calibration Layer - Enhanced Logging Version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.isotonic import IsotonicRegression
from Metrics.metrics import expected_calibration_error

# Import the new simple stability space
from .simple_stability_space import SimpleStabilitySpace

from utils.logging_config import get_logger
logger = get_logger(__name__)


class InTrainingGeometricCalibrator(nn.Module):
    """
    In-training geometric calibration using fast separation method
    """
    
    def __init__(self, 
                 num_classes: int,
                 warmup_epochs: int = 10,
                 accuracy_threshold: float = 0.5,
                 calibration_update_frequency: int = 10,
                 stability_data_ratio: float = 0.75,
                 stability_metric: str = 'l2',
                 min_samples_for_fitting: int = 200,
                 temperature_init: float = 1.0):
        """Same as before..."""
        super().__init__()
        
        self.num_classes = num_classes
        self.warmup_epochs = warmup_epochs
        self.accuracy_threshold = accuracy_threshold
        self.calibration_update_frequency = calibration_update_frequency
        self.stability_data_ratio = stability_data_ratio
        self.stability_metric = stability_metric
        self.min_samples_for_fitting = min_samples_for_fitting
        
        # Temperature scaling for non-max classes (learnable parameter)
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        
        # Training state
        self.current_epoch = 0
        self.calibration_active = False
        self.isotonic_fitted = False
        self.last_calibration_update = -1
        
        # Epoch-based data collection (replaces rolling window)
        self.current_epoch_features = []
        self.current_epoch_labels = []
        self.current_epoch_logits = []
        self.batch_count = 0
        self.data_collection_count = 0  # Track how many times we collect data
        
        # Calibration components
        self.stability_space = SimpleStabilitySpace(
            metric=stability_metric,
            use_cuda=torch.cuda.is_available()
        )
        self.isotonic_regressor = None
        
        # Historical data for weighted isotonic regression
        self.previous_separations = None
        self.previous_accuracies = None
        
        # Statistics tracking
        self.calibration_stats = {
            'stability_scores': [],
            'accuracies': [],
            'ece_history': [],
            'calibration_updates': 0,
            'total_samples_processed': 0,
            'data_collections': 0,
            'calibration_applications': 0
        }
        
        logger.info(f"🔬 InTrainingGeometricCalibrator initialized:")
        logger.info(f"   Warmup epochs: {warmup_epochs}")
        logger.info(f"   Update frequency: {calibration_update_frequency} epochs")
        logger.info(f"   Stability ratio: {stability_data_ratio:.2f}")
        logger.info(f"   Min samples for fitting: {min_samples_for_fitting}")
    
    def should_extract_features(self) -> bool:
        """Check if we should extract features this batch"""
        self.batch_count += 1
        should_extract = self.batch_count % 100 == 0
        if should_extract:
            logger.info(f"🔍 Should extract features: batch {self.batch_count} (every 100th batch)")
        return should_extract
    
    def update_epoch(self, epoch: int, validation_accuracy: float = None):
        """Update the current epoch and trigger calibration updates"""
        self.current_epoch = epoch
        logger.info(f"📅 Epoch {epoch}: Calibration active={self.calibration_active}, "
                   f"Data collected={len(self.current_epoch_features)} batches")
        
        # Check if we should start calibration
        if not self.calibration_active:
            should_start = False
            
            if epoch >= self.warmup_epochs:
                should_start = True
                reason = f"warmup period completed ({epoch} >= {self.warmup_epochs})"
            
            if validation_accuracy is not None and validation_accuracy >= self.accuracy_threshold:
                should_start = True
                reason = f"accuracy threshold reached ({validation_accuracy:.3f} >= {self.accuracy_threshold})"
            
            if should_start:
                self.calibration_active = True
                logger.info(f"🎯 Starting geometric calibration at epoch {epoch}: {reason}")
        
        # Check if we should update calibration (every N epochs)
        should_update = (
            self.calibration_active and 
            epoch > 0 and 
            epoch % self.calibration_update_frequency == 0 and
            epoch != self.last_calibration_update
        )
        
        if should_update:
            logger.info(f"🔄 Triggering calibration update at epoch {epoch}")
            logger.info(f"   📊 Available data: {len(self.current_epoch_features)} feature batches")
            self._process_epoch_data()
            self.last_calibration_update = epoch
        
        # Always process data at the end of each epoch
        self._collect_epoch_data()
    
    def collect_training_data(self, 
                            features: torch.Tensor, 
                            targets: torch.Tensor, 
                            logits: torch.Tensor):
        """Collect training data for current epoch"""
        
        # Check if we have all required data
        if features is None or targets is None or logits is None:
            if self.forward_call_count % 50 == 1:
                missing = []
                if features is None: missing.append("features")
                if targets is None: missing.append("targets") 
                if logits is None: missing.append("logits")
                logger.info(f"   ❌ Data collection skipped: missing {', '.join(missing)}")
            return
        
        if not self.should_extract_features():
            return
        
        self.data_collection_count += 1
        
        # Store data for current epoch processing
        features_np = features.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
        
        self.current_epoch_features.append(features_np)
        self.current_epoch_labels.append(targets_np)
        self.current_epoch_logits.append(logits_np)
        
        self.calibration_stats['data_collections'] += 1
        if self.data_collection_count % 100 == 1:
            logger.info(f"📥 Data collected #{self.data_collection_count}: "
                    f"{len(features_np)} samples, "
                    f"features shape: {features_np.shape}")
            logger.info(f"   📊 Total batches collected this epoch: {len(self.current_epoch_features)}")
    
    def _collect_epoch_data(self):
        """Collect data at end of epoch (called every epoch)"""
        if not self.current_epoch_features:
            logger.info(f"📊 Epoch {self.current_epoch}: No data collected")
            return
        
        total_samples = sum(len(batch) for batch in self.current_epoch_features)
        logger.info(f"📊 Epoch {self.current_epoch}: Collected {total_samples} samples "
                   f"across {len(self.current_epoch_features)} batches")
    
    def _process_epoch_data(self):
        """Process collected epoch data (called every 10 epochs)"""
        logger.info(f"🔄 Starting _process_epoch_data()...")
        
        if not self.current_epoch_features:
            logger.warning(f"❌ No data collected for processing at epoch {self.current_epoch}")
            return
        
        logger.info(f"✅ Processing epoch data for calibration update...")
        
        # Combine all collected data from recent epochs
        try:
            all_features = np.concatenate(self.current_epoch_features, axis=0)
            all_labels = np.concatenate(self.current_epoch_labels, axis=0)
            all_logits = np.concatenate(self.current_epoch_logits, axis=0)
            
            logger.info(f"✅ Combined data successfully: {len(all_features)} samples")
            logger.info(f"   Features shape: {all_features.shape}")
            logger.info(f"   Labels shape: {all_labels.shape}")
            logger.info(f"   Logits shape: {all_logits.shape}")
            
        except Exception as e:
            logger.error(f"❌ Failed to combine epoch data: {e}")
            return
        
        # Get predictions from logits
        predictions = np.argmax(all_logits, axis=1)
        logger.info(f"✅ Computed predictions: unique classes = {np.unique(predictions)}")
        
        # Random split: 75% for stability space, 25% for isotonic fitting
        n_samples = len(all_features)
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * self.stability_data_ratio)
        
        stability_indices = indices[:split_idx]
        isotonic_indices = indices[split_idx:]
        
        logger.info(f"📊 Data split:")
        logger.info(f"   Stability space: {len(stability_indices)} samples")
        logger.info(f"   Isotonic regression: {len(isotonic_indices)} samples")
        
        # Update stability space with 75% of data
        if len(stability_indices) > 0:
            logger.info(f"🔧 Updating stability space...")
            try:
                self.stability_space.update_training_data(
                    all_features[stability_indices], 
                    predictions[stability_indices]
                )
                stability_stats = self.stability_space.get_stats()
                logger.info(f"✅ Stability space updated successfully:")
                logger.info(f"   Total samples: {stability_stats['num_samples']}")
                logger.info(f"   Feature dim: {stability_stats['feature_dim']}")
                logger.info(f"   Unique classes: {stability_stats['num_classes']}")
            except Exception as e:
                logger.error(f"❌ Failed to update stability space: {e}")
        
        # Fit isotonic regression with 25% of data
        if len(isotonic_indices) > 0:
            logger.info(f"🔧 Updating isotonic regression...")
            try:
                self._update_isotonic_regression(
                    all_features[isotonic_indices],
                    all_labels[isotonic_indices], 
                    predictions[isotonic_indices]
                )
            except Exception as e:
                logger.error(f"❌ Failed to update isotonic regression: {e}")
        
        # Update statistics
        self.calibration_stats['calibration_updates'] += 1
        self.calibration_stats['total_samples_processed'] += n_samples
        
        # Clear collected data
        self.current_epoch_features = []
        self.current_epoch_labels = []
        self.current_epoch_logits = []
        
        logger.info(f"✅ Calibration update completed successfully")
        logger.info(f"📊 Total updates so far: {self.calibration_stats['calibration_updates']}")
    
    def _update_isotonic_regression(self, features: np.ndarray, true_labels: np.ndarray, predictions: np.ndarray):
        """Update isotonic regression with weighted recent data"""
        logger.info(f"🔧 Starting isotonic regression update...")
        logger.info(f"   Input: {len(features)} samples")
        
        try:
            # Calculate separation scores for current data
            logger.info(f"🔍 Calculating fast separation scores...")
            separation_scores = self.stability_space.calculate_fast_separation_vectorized(features, predictions, batch_size=1000)
            logger.info(f"✅ Calculated separation scores:")
            logger.info(f"   Count: {len(separation_scores)}")
            logger.info(f"   Range: [{np.min(separation_scores):.4f}, {np.max(separation_scores):.4f}]")
            logger.info(f"   Mean: {np.mean(separation_scores):.4f}")
            
            # Calculate actual accuracies (binary: correct/incorrect)
            actual_accuracies = (predictions == true_labels).astype(float)
            accuracy_rate = np.mean(actual_accuracies)
            logger.info(f"✅ Calculated accuracies: rate = {accuracy_rate:.4f}")
            
            # Handle previous data combination (existing logic...)
            if (hasattr(self, 'previous_separations') and 
                self.previous_separations is not None and 
                self.current_epoch >= 15):
                
                logger.info(f"🔄 Combining with historical data (weighted)...")
                recent_weight = 2.0
                old_weight = 1.0
                
                recent_sep_weighted = np.repeat(separation_scores, int(recent_weight))
                recent_acc_weighted = np.repeat(actual_accuracies, int(recent_weight))
                old_sep_weighted = np.repeat(self.previous_separations, int(old_weight))
                old_acc_weighted = np.repeat(self.previous_accuracies, int(old_weight))
                
                combined_separations = np.concatenate([old_sep_weighted, recent_sep_weighted])
                combined_accuracies = np.concatenate([old_acc_weighted, recent_acc_weighted])
                
                logger.info(f"   Recent weighted: {len(recent_sep_weighted)} samples")
                logger.info(f"   Historical weighted: {len(old_sep_weighted)} samples")
                logger.info(f"   Combined total: {len(combined_separations)} samples")
            else:
                combined_separations = separation_scores
                combined_accuracies = actual_accuracies
                logger.info(f"   Using current data only: {len(separation_scores)} samples")
            
            # Fit isotonic regression if we have enough data
            if len(combined_separations) >= self.min_samples_for_fitting:
                logger.info(f"🔧 Fitting isotonic regression...")
                
                # Filter out invalid values
                valid_mask = np.isfinite(combined_separations) & np.isfinite(combined_accuracies)
                valid_separations = combined_separations[valid_mask]
                valid_accuracies = combined_accuracies[valid_mask]
                
                logger.info(f"   Valid samples: {len(valid_separations)} / {len(combined_separations)}")
                
                if len(valid_separations) >= 10:
                    self.isotonic_regressor = IsotonicRegression(
                        out_of_bounds="clip",
                        increasing=True
                    )
                    self.isotonic_regressor.fit(valid_separations, valid_accuracies)
                    self.isotonic_fitted = True
                    
                    # FIXED: Proper confidence clamping before ECE calculation
                    predicted_confidences = self.isotonic_regressor.predict(valid_separations)
                    
                    # Critical fix: Ensure confidences are in valid range [epsilon, 1-epsilon]
                    epsilon = 1e-7
                    predicted_confidences = np.clip(predicted_confidences, epsilon, 1.0 - epsilon)
                    
                    # Convert accuracies to binary predictions based on confidence threshold
                    predicted_labels = (predicted_confidences > 0.5).astype(int)
                    true_binary_labels = valid_accuracies.astype(int)
                    
                    # Calculate ECE with properly bounded confidences
                    ece = expected_calibration_error(predicted_confidences, predicted_labels, true_binary_labels)
                    
                    # Validate ECE result
                    if np.isnan(ece) or np.isinf(ece) or ece < 0 or ece > 1:
                        logger.warning(f"   ⚠️ Invalid ECE value: {ece}, setting to 1.0")
                        ece = 1.0
                    
                    self.calibration_stats['ece_history'].append(ece)
                    
                    logger.info(f"✅ Isotonic regression fitted successfully:")
                    logger.info(f"   Training samples: {len(valid_separations)}")
                    logger.info(f"   Confidence range: [{np.min(predicted_confidences):.4f}, {np.max(predicted_confidences):.4f}]")
                    logger.info(f"   ECE: {ece:.4f}")
                    logger.info(f"   Fitted: {self.isotonic_fitted}")
                    
                    # Store current data for next update
                    max_history = 1000
                    if len(separation_scores) <= max_history:
                        self.previous_separations = separation_scores
                        self.previous_accuracies = actual_accuracies
                    else:
                        indices = np.random.choice(len(separation_scores), max_history, replace=False)
                        self.previous_separations = separation_scores[indices]
                        self.previous_accuracies = actual_accuracies[indices]
                    
                    logger.info(f"   Stored {len(self.previous_separations)} samples for next update")
                else:
                    logger.warning(f"❌ Not enough valid samples for isotonic regression: {len(valid_separations)}")
            else:
                logger.warning(f"❌ Not enough samples for isotonic regression: {len(combined_separations)} < {self.min_samples_for_fitting}")
                
        except Exception as e:
            logger.error(f"❌ Failed to update isotonic regression: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")    

    def apply_calibration(self, 
                        logits: torch.Tensor, 
                        features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply geometric calibration to logits"""
        if not self.calibration_active or not self.isotonic_fitted or features is None:
            logger.debug(f"🔄 Using temperature scaling")
            return F.softmax(logits / self.temperature, dim=1)
        
        self.calibration_stats['calibration_applications'] += 1
        
        try:
            batch_size = logits.size(0)
            device = logits.device
            
            # Get original probabilities and predictions
            original_probs = F.softmax(logits, dim=1)
            max_probs, predicted_classes = torch.max(original_probs, dim=1)
            
            # Calculate separation scores
            features_np = features.detach().cpu().numpy()
            predictions_np = predicted_classes.detach().cpu().numpy()
            
            separation_scores = self.stability_space.calculate_fast_separation_vectorized(features_np, predictions_np, batch_size=1000)
            
            # Convert separation to calibrated confidence using isotonic regression
            calibrated_confidences = self.isotonic_regressor.predict(separation_scores)
            
            # CRITICAL: Ensure confidences are valid for probability distributions
            calibrated_confidences = np.clip(calibrated_confidences, 0.01, 0.99)
            
            calibrated_confidences_tensor = torch.tensor(
                calibrated_confidences, device=device, dtype=torch.float32
            )
            
            # Create calibrated probabilities
            calibrated_probs = torch.zeros_like(original_probs)
            
            for i in range(batch_size):
                pred_class = predicted_classes[i]
                new_confidence = calibrated_confidences_tensor[i]
                
                # Set the predicted class probability
                calibrated_probs[i, pred_class] = new_confidence
                
                # Distribute remaining probability
                remaining_prob = 1.0 - new_confidence
                other_logits = logits[i].clone()
                other_logits[pred_class] = -float('inf')
                
                if remaining_prob > 0:
                    other_probs = F.softmax(other_logits / self.temperature, dim=0)
                    other_probs = other_probs * remaining_prob / other_probs.sum()
                    
                    mask = torch.ones(self.num_classes, dtype=torch.bool, device=device)
                    mask[pred_class] = False
                    calibrated_probs[i, mask] = other_probs[mask]
            
            return calibrated_probs
            
        except Exception as e:
            logger.error(f"❌ Failed to apply calibration: {e}")
            return F.softmax(logits / self.temperature, dim=1)
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics and status"""
        stability_stats = self.stability_space.get_stats()
        
        return {
            'calibration_active': self.calibration_active,
            'isotonic_fitted': self.isotonic_fitted,
            'current_epoch': self.current_epoch,
            'last_update_epoch': self.last_calibration_update,
            'next_update_epoch': self.last_calibration_update + self.calibration_update_frequency,
            'total_updates': self.calibration_stats['calibration_updates'],
            'total_samples_processed': self.calibration_stats['total_samples_processed'],
            'data_collections': self.calibration_stats['data_collections'],
            'calibration_applications': self.calibration_stats['calibration_applications'],
            'current_ece': self.calibration_stats['ece_history'][-1] if self.calibration_stats['ece_history'] else None,
            'temperature': self.temperature.item(),
            'stability_space_stats': stability_stats,
            'pending_samples': sum(len(batch) for batch in self.current_epoch_features)
        }
    
    def forward(self, 
               logits: torch.Tensor, 
               features: Optional[torch.Tensor] = None,
               targets: Optional[torch.Tensor] = None,
               apply_calibration: bool = True) -> torch.Tensor:
        """Forward pass with optional calibration"""
        # Collect training data if in training mode and targets are provided
        if self.training and targets is not None and features is not None:
            self.collect_training_data(features, targets, logits)
        
        # Apply calibration if requested and ready
        if apply_calibration:
            return self.apply_calibration(logits, features)
        else:
            return F.softmax(logits / self.temperature, dim=1)