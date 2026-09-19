# Calibrators/geometric_calibrator.py
"""
Geometric Calibrator using Fast Separation + Isotonic Regression
Based on the original comprehensive implementation, simplified to use only isotonic regression
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

from utils.logging_config import get_logger
logger = get_logger(__name__)

class GeometricCalibrator(BaseCalibrator):
    """
    Geometric calibration using stability/separation scores and isotonic regression.
    Simplified version using only isotonic regression as the fitting function.
    """

    def __init__(self, model, X_train_embed, y_train, X_train_original=None, 
                 compression_mode=None, compression_param=None, metric='l2', stability_space=None, 
                 library='faiss', use_binning=True, n_bins=200,
                 # NEW PARAMETERS
                 use_augmix=False, augmix_versions=3, augmix_severity=3, 
                 augmix_width=3, augmix_alpha=1.0, **kwargs):
        """
        Initializes the GeometricCalibrator with isotonic regression.

        Args:
            model: The model to be calibrated (with `predict` and `predict_proba` methods).
            X_train_embed: Training data embeddings for geometric calculations.
            y_train: Training labels.
            X_train_original: Original training data for model predictions (if None, uses embeddings).
            compression_mode: Compression mode for data.
            compression_param: Parameter controlling the compression level.
            metric: Distance metric for stability/separation calculations.
            stability_space: Optional custom StabilitySpace instance.
            library: The library used for stability calculation (default is 'faiss').
            use_binning (bool): Whether to bin stability scores and calculate average accuracy.
            n_bins (int): Number of bins for stability scores (default: 200).
            use_augmix (bool): Whether to use AugMix during calibration fitting
            augmix_versions (int): Number of augmented versions per validation sample
            augmix_severity (int): AugMix severity level (1-10)
            augmix_width (int): Number of augmentation chains
            augmix_alpha (float): Beta distribution parameter for mixing
        """
        logging.info("\n=== GeometricCalibrator Initialization ===")
        logging.info(f"Input X_train_embed shape: {X_train_embed.shape}")
        if X_train_original is not None:
            logging.info(f"Input X_train_original shape: {X_train_original.shape}")
        logging.info(f"Compression mode: {compression_mode}")
        logging.info(f"Compression param: {compression_param}")
        logging.info(f"Library: {library}")
        logging.info(f"Metric: {metric}")

        super().__init__(**kwargs)
        self.model = model
        self.X_train_original = X_train_original
        self.isotonic_regressor = None  # Always use isotonic regression
        self.is_fitted = False
        self.metric = metric.lower()
        self.use_binning = use_binning
        self.n_bins = n_bins

        # NEW: AugMix parameters
        self.use_augmix = use_augmix
        self.augmix_versions = augmix_versions if use_augmix else 0
        self.augmix_severity = augmix_severity
        self.augmix_width = augmix_width
        self.augmix_alpha = augmix_alpha
        
        # Initialize AugMix transform if needed
        if self.use_augmix:
            from Data.augmix_transforms import AugMixTransforms
            self.augmix_transform = AugMixTransforms(
                severity=augmix_severity,
                width=augmix_width,
                depth=-1,  # Random depth 1-3
                alpha=augmix_alpha
            )
            logger.info(f"🌪️ AugMix enabled for calibration fitting: "
                       f"{augmix_versions} versions, severity={augmix_severity}")
        else:
            self.augmix_transform = None

        # Determine the number of classes
        self.num_labels = len(np.unique(y_train))
        logging.info(f"Number of unique labels: {self.num_labels}")

        # Use provided stability space or create a new one
        if stability_space:
            self.stab_space = stability_space
            logger.info(f"{self.__class__.__name__}: Using custom StabilitySpace provided by user.")
        else:
            # Initialize compression if needed
            compression = None
            if compression_mode is not None:
                from utils.utils import Compression  # Import here to avoid circular imports
                compression = Compression(compression_mode, compression_param)
            
            # Initialize StabilitySpace
            self.stab_space = StabilitySpace(
                X_train_embed, y_train,
                compression=compression,
                library=library, 
                metric=self.metric
            )
            logger.info(f"{self.__class__.__name__}: Initialized StabilitySpace with default settings"
                        f" (library: {library}, metric: {metric}).")

        logger.info(f"Initialized {self.__class__.__name__} with model {self.model.__class__.__name__}"
                    f" using IsotonicRegression for calibration.")

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

    def fit(self, X_val_embed, y_val, X_val_original=None, features=None, fit_batch_size=1000):
        """
        Enhanced fit method with AugMix augmentation and BATCHED processing for memory efficiency.
        
        Args:
            fit_batch_size: Batch size for fitting process (default: 1000)
        """
        logger.info(f"{self.__class__.__name__}: Fitting with validation data using isotonic regression.")
        
        if self.use_augmix:
            logger.info(f"🌪️ Applying AugMix augmentation: {self.augmix_versions} versions per sample")
            
            # Step 1: Create augmented validation data
            X_val_original_expanded, X_val_embed_expanded, y_val_expanded = self._create_augmented_validation_data(
                X_val_embed, y_val, X_val_original
            )
            
            # Use expanded datasets for fitting
            prediction_data = X_val_original_expanded
            embedding_data = X_val_embed_expanded
            labels_data = y_val_expanded
            
            logger.info(f"📊 Expanded validation set: {len(y_val)} → {len(y_val_expanded)} samples")
            logger.info(f"🔧 Using batched fitting with batch_size={fit_batch_size}")
        else:
            # Use original data
            prediction_data = X_val_original if X_val_original is not None else X_val_embed
            embedding_data = X_val_embed
            labels_data = y_val
            fit_batch_size = min(fit_batch_size, len(labels_data))  # Don't over-batch small datasets

        try:
            # Step 2: BATCHED prediction and stability computation
            all_stability_vals = []
            all_predictions = []
            all_labels = []
            
            n_samples = len(labels_data)
            n_batches = (n_samples + fit_batch_size - 1) // fit_batch_size
            
            logger.info(f"🔄 Processing {n_samples} samples in {n_batches} batches...")
            
            # 🔧 NEW: Add overall batch progress bar with timing
            start_time = time.time()
            
            with tqdm(total=n_batches, desc="Processing Calibration Batches", unit="batch") as pbar:
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * fit_batch_size
                    end_idx = min(start_idx + fit_batch_size, n_samples)
                    
                    # Update progress bar description with current batch info
                    pbar.set_description(f"Processing batch {batch_idx + 1}/{n_batches} ({end_idx - start_idx} samples)")
                    
                    try:
                        # Clear GPU cache before each batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Extract batch data
                        batch_prediction_data = prediction_data[start_idx:end_idx]
                        batch_embedding_data = embedding_data[start_idx:end_idx]
                        batch_labels = labels_data[start_idx:end_idx]
                        
                        # Batch prediction
                        batch_predictions = self.model.predict_proba(batch_prediction_data)
                        batch_pred_classes = np.argmax(batch_predictions, axis=1)
                        
                        # Batch stability computation (this is where the inner progress was showing before)
                        batch_start_time = time.time()
                        batch_stability = self.stab_space.calc_stab(batch_embedding_data, batch_predictions)
                        batch_time = time.time() - batch_start_time
                        
                        # Accumulate results
                        all_stability_vals.extend(batch_stability)
                        all_predictions.extend(batch_pred_classes)
                        all_labels.extend(batch_labels)
                        
                        # Update progress bar with batch timing info
                        avg_time_per_sample = batch_time / (end_idx - start_idx)
                        pbar.set_postfix({
                            'batch_time': f'{batch_time:.1f}s',
                            'samples/s': f'{1/avg_time_per_sample:.1f}'
                        })
                        
                        logger.info(f"      ✅ Batch {batch_idx + 1} completed in {batch_time:.1f}s")
                        
                    except Exception as e:
                        logger.error(f"      ❌ Batch {batch_idx + 1} failed: {e}")
                        # Continue with other batches, but fill with default values
                        batch_size_actual = end_idx - start_idx
                        all_stability_vals.extend([0.0] * batch_size_actual)
                        all_predictions.extend([0] * batch_size_actual)
                        all_labels.extend([0] * batch_size_actual)
                        
                    finally:
                        # Always update the progress bar
                        pbar.update(1)
            
            total_time = time.time() - start_time
            logger.info(f"🏁 Batched processing completed in {total_time:.1f}s")
            logger.info(f"   Average time per batch: {total_time/n_batches:.1f}s")
            logger.info(f"   Average time per sample: {total_time/n_samples:.3f}s")
            
            # Convert to numpy arrays
            stability_val = np.array(all_stability_vals)
            y_pred_classes = np.array(all_predictions)
            y_val_processed = np.array(all_labels)
            
            logger.info(f"✅ Batched processing complete:")
            logger.info(f"   Stability values: {len(stability_val)}")
            logger.info(f"   Predictions: {len(y_pred_classes)}")
            logger.info(f"   Labels: {len(y_val_processed)}")
            
            # Step 3: Continue with existing fitting logic (normalize, bin, fit isotonic)
            # Normalize stability values using percentile-based normalization
            stability_p5 = np.percentile(stability_val, 5)
            stability_p95 = np.percentile(stability_val, 95)
            stability_range = stability_p95 - stability_p5
            
            if stability_range > 0:
                stability_val = np.clip(
                    (stability_val - stability_p5) / stability_range,
                    0, 1
                )
            else:
                # If range is too small, use uniform distribution
                stability_val = np.ones_like(stability_val) * 0.5
                logger.warning("Stability range too small, using uniform distribution")
            
            logger.info(f"Stability normalization stats:")
            logger.info(f"  5th percentile: {stability_p5:.6f}")
            logger.info(f"  95th percentile: {stability_p95:.6f}")
            logger.info(f"  Range: {stability_range:.6f}")
            logger.info(f"  Normalized values (first 5): {stability_val[:5]}")

            if self.use_binning:
                # Step 4: Bin stability values
                logger.info(f"Binning stability values into {self.n_bins} bins.")
                bin_edges = np.linspace(0, 1, self.n_bins + 1)
                bin_indices = np.digitize(stability_val, bins=bin_edges) - 1
                
                # Step 5: Compute average accuracy for each bin
                binned_stability = []
                binned_accuracy = []
                for bin_idx in range(self.n_bins):
                    indices_in_bin = np.where(bin_indices == bin_idx)[0]
                    if len(indices_in_bin) > 0:
                        y_true_bin = y_val_processed[indices_in_bin]
                        y_pred_bin = y_pred_classes[indices_in_bin]
                        accuracy = np.mean(y_true_bin == y_pred_bin)
                        binned_stability.append((bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2)
                        binned_accuracy.append(accuracy)

                stability_vals = np.array(binned_stability)
                accuracies = np.array(binned_accuracy)
                logger.info(f"Created {len(stability_vals)} bins with data")

            else:
                # Use raw stability and accuracies without binning
                unique_stabilities = np.unique(stability_val)
                stability_vals = []
                accuracies = []
                for stab in unique_stabilities:
                    indices = np.where(stability_val == stab)[0]
                    y_true_stab = y_val_processed[indices]
                    y_pred_stab = y_pred_classes[indices]
                    acc = np.mean(y_true_stab == y_pred_stab)
                    stability_vals.append(stab)
                    accuracies.append(acc)
                stability_vals = np.array(stability_vals)
                accuracies = np.array(accuracies)

            # Calculate Pearson correlation
            corr, p_value = np.nan, np.nan
            try:
                if len(stability_vals) > 1:
                    from scipy.stats import pearsonr
                    corr, p_value = pearsonr(stability_vals, accuracies)
                    logger.info(f"Pearson correlation between stability and accuracy: {corr:.4f} (p-value: {p_value:.4f})")
                else:
                    logger.warning("Not enough unique stability values to calculate correlation")
            except ValueError as e:
                logger.warning(f"Could not calculate Pearson correlation: {e}")

            # Fit isotonic regression
            try:
                logger.info(f"Accuracy values - Max: {np.max(accuracies):.4f}, "
                           f"Min: {np.min(accuracies):.4f}, Mean: {np.mean(accuracies):.4f}")

                # Initialize and fit isotonic regression
                self.isotonic_regressor = IsotonicRegression(out_of_bounds="clip")
                self.isotonic_regressor.fit(stability_vals.reshape(-1, 1), accuracies)
                self.is_fitted = True
                
                # Store normalization parameters for later use
                self.stability_p5 = stability_p5
                self.stability_p95 = stability_p95
                
                logger.info(f"{self.__class__.__name__}: Successfully fitted using IsotonicRegression with batched processing.")

            except Exception as e:
                logger.error(f"{self.__class__.__name__}: Failed to fit with error: {e}")
                self.is_fitted = False
                raise

            # Store for later use
            self.stability_vals = stability_vals

            return corr, p_value

        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Fitting failed with error: {e}")
            raise ValueError(f"{self.__class__.__name__}: Fitting failed with error: {e}")

    def calibrate_batched(self, X_test_embed, X_test_original=None, features=None, batch_size=1000):
        """
        Memory-efficient batched calibration to prevent GPU OOM.
        🔧 ENHANCED: Added overall batch progress bar and improved validation
        Args:
            X_test_embed: Test data embeddings.
            X_test_original: Original test data for model predictions.
            features: Optional features parameter for compatibility.
            batch_size: Batch size for processing (default: 1000)
        Returns:
            np.ndarray: Calibrated probability matrix for each sample and class.
        """
        if not self.is_fitted:
            raise ValueError("You must fit the calibrator before using it.")

        logger.info(f"GeometricCalibrator: Calibrating test data in batches of {batch_size}.")
        # 🔧 FIX: Ensure we have proper original data for model predictions
        if X_test_original is None:
            X_test_original = X_test_embed
            logger.info("⚠️ No X_test_original provided - using X_test_embed as original data")
        # 🔧 CRITICAL FIX: Add comprehensive data validation
        logger.info(f"📊 Input validation:")
        logger.info(f"   X_test_embed shape: {X_test_embed.shape}")
        logger.info(f"   X_test_original shape: {X_test_original.shape}")
        # Detect calibration mode based on shapes
        is_physical_calibration = (X_test_embed.shape == X_test_original.shape and 
                                 len(X_test_embed.shape) == 4)
        logger.info(f"   Calibration mode: {'Physical' if is_physical_calibration else 'Semantic'}")
        # 🔧 CRITICAL VALIDATION: Check for data flow bugs
        if len(X_test_original.shape) == 2 and X_test_original.shape[1] <= 20:
            # This looks like logits, not raw images
            logger.error(f"❌ CRITICAL ERROR: Logits detected in batch_original instead of raw images!")
            logger.error(f"   batch_original shape: {X_test_original.shape} (looks like logits for {X_test_original.shape[1]} classes)")
            logger.error(f"   Expected shape: [N, 3, 32, 32] for CIFAR images")
            logger.error(f"   This indicates a data flow bug in the calibration pipeline")
            raise ValueError(f"Logits detected in batch_original instead of raw images: shape {X_test_original.shape}. "
                            f"This indicates a data flow bug in the calibration pipeline.")
        # Validate image data for CIFAR
        if len(X_test_original.shape) == 4:
            if X_test_original.shape[1] == 3 and X_test_original.shape[2:] == (32, 32):
                logger.info("✅ Valid CIFAR raw image data detected")
            elif X_test_original.shape[1] not in [1, 3]:
                logger.warning(f"⚠️ Unusual channel count: {X_test_original.shape[1]} (expected 1 or 3)")
        n_samples = X_test_original.shape[0]
        # Initialize result array
        num_classes = self.num_labels
        all_calibrated_probs = np.zeros((n_samples, num_classes))
        # Process in batches to manage memory
        n_batches = (n_samples + batch_size - 1) // batch_size
        start_time = time.time()
        with tqdm(total=n_batches, desc="Calibrating Test Batches", unit="batch") as pbar:
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                # Extract batch data with validation
                batch_embed = X_test_embed[start_idx:end_idx]
                batch_original = X_test_original[start_idx:end_idx]
                # 🔧 ENHANCED: Add per-batch validation
                logger.debug(f"   Batch {batch_idx + 1}: embed={batch_embed.shape}, original={batch_original.shape}")
                try:
                    # Clear GPU cache before each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Process batch using original calibrate method
                    batch_probs = self._calibrate_single_batch(batch_embed, batch_original)
                    all_calibrated_probs[start_idx:end_idx] = batch_probs
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"❌ Calibration batch {batch_idx + 1} failed: {e}")
                    # Fill with uniform probabilities as fallback
                    all_calibrated_probs[start_idx:end_idx] = 1.0 / num_classes
                    pbar.update(1)
        total_time = time.time() - start_time
        logger.info(f"🏁 Batched calibration completed in {total_time:.1f}s")
        logger.info(f"GeometricCalibrator: Batched calibration complete.")
        return all_calibrated_probs

    def _calibrate_single_batch(self, batch_embed, batch_original):
        """Process a single batch using the original calibration logic"""
        
        # 🔧 DEBUG: Add detailed logging for data validation
        logger.debug(f"🔍 _calibrate_single_batch received:")
        logger.debug(f"   batch_embed shape: {batch_embed.shape}")
        logger.debug(f"   batch_embed dtype: {batch_embed.dtype}")
        logger.debug(f"   batch_original shape: {batch_original.shape}")
        logger.debug(f"   batch_original dtype: {batch_original.dtype}")
        
        # 🔧 CRITICAL FIX: Detect if logits are accidentally passed instead of raw images
        if len(batch_original.shape) == 2 and batch_original.shape[1] in [10, 100]:  # CIFAR-10 or CIFAR-100 logits
            logger.error(f"❌ CRITICAL ERROR: Logits detected in batch_original instead of raw images!")
            logger.error(f"   batch_original shape: {batch_original.shape} (looks like logits for {batch_original.shape[1]} classes)")
            logger.error(f"   Expected shape: [N, 3, 32, 32] for CIFAR images")
            logger.error(f"   This indicates a data flow bug in the calibration pipeline")
            raise ValueError(f"Logits detected in batch_original instead of raw images: shape {batch_original.shape}. "
                           f"This indicates a data flow bug in the calibration pipeline.")
        
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

        # Compute stability for batch
        stability_batch = self.stab_space.calc_stab(batch_embed, y_batch_pred)
        
        # Log stability statistics before normalization
        logger.info(f"Batch stability stats before normalization:")
        logger.info(f"  Min: {np.min(stability_batch):.6f}")
        logger.info(f"  Max: {np.max(stability_batch):.6f}")
        logger.info(f"  Mean: {np.mean(stability_batch):.6f}")
        logger.info(f"  Std: {np.std(stability_batch):.6f}")
        
        # Normalize stability values using stored percentile parameters
        stability_range = self.stability_p95 - self.stability_p5
        if stability_range > 0:
            stability_batch = np.clip(
                (stability_batch - self.stability_p5) / stability_range,
                0, 1
            )
            logger.info(f"Applied percentile-based normalization:")
            logger.info(f"  5th percentile: {self.stability_p5:.6f}")
            logger.info(f"  95th percentile: {self.stability_p95:.6f}")
            logger.info(f"  Range: {stability_range:.6f}")
        else:
            # If range is too small, use uniform distribution
            stability_batch = np.ones_like(stability_batch) * 0.5
            logger.warning("Stability range too small, using uniform distribution")
        
        # Log stability statistics after normalization
        logger.info(f"Batch stability stats after normalization:")
        logger.info(f"  Min: {np.min(stability_batch):.6f}")
        logger.info(f"  Max: {np.max(stability_batch):.6f}")
        logger.info(f"  Mean: {np.mean(stability_batch):.6f}")
        logger.info(f"  Std: {np.std(stability_batch):.6f}")

        # Apply isotonic regression calibration
        calibrated_values = self.isotonic_regressor.predict(stability_batch.reshape(-1, 1))
        calibrated_values = np.clip(calibrated_values, 0.01, 0.99)

        # Distribute calibrated values across predicted classes
        for i in range(batch_size):
            # Assign calibrated probability to predicted class
            calibrated_probs[i, y_batch_labels[i]] = calibrated_values[i]
            
            # Distribute remaining probability equally across other classes
            remaining_prob = (1 - calibrated_values[i]) / (self.num_labels - 1)
            for j in range(self.num_labels):
                if j != y_batch_labels[i]:
                    calibrated_probs[i, j] = remaining_prob

        # Ensure probabilities are valid
        calibrated_probs = np.clip(calibrated_probs, 0, 1)
        calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)

        return calibrated_probs

    def calibrate(self, X_test_embed, X_test_original=None, features=None):
        """
        Calibrates the test data using the fitted isotonic regression model.
        🔧 FIXED: Always uses proper geometric calibration logic, never falls back to base class.

        Args:
            X_test_embed: Test data embeddings.
            X_test_original: Original test data for model predictions (if None, uses embeddings).
            features: Optional features parameter for compatibility with base interface.

        Returns:
            np.ndarray: Calibrated probability matrix for each sample and class.
        """
        if not self.is_fitted:
            raise ValueError("You must fit the calibrator before using it.")

        n_samples = len(X_test_embed)
        
        # 🔧 FIX: Always use geometric calibration logic, don't fall back to base class
        # Use batching for ALL datasets to ensure consistent behavior
        if n_samples > 1000:
            # Large datasets: use existing batched approach
            return self.calibrate_batched(X_test_embed, X_test_original, features, batch_size=1000)
        else:
            # Small datasets: use single batch approach (fixed implementation)
            logger.info(f"GeometricCalibrator: Calibrating {n_samples} samples (small dataset)")
            
            # Ensure we have original data for model predictions
            X_test_original = X_test_original if X_test_original is not None else X_test_embed
            
            # Add debug logging
            logger.debug(f"Embed data shape: {X_test_embed.shape}")
            logger.debug(f"Original data shape: {X_test_original.shape}")
            
            # Process as single batch using the corrected logic
            try:
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Use the single batch method which properly handles data types
                calibrated_probs = self._calibrate_single_batch(X_test_embed, X_test_original)
                
                logger.info(f"GeometricCalibrator: Small dataset calibration complete")
                return calibrated_probs
                
            except Exception as e:
                logger.error(f"GeometricCalibrator: Small dataset calibration failed: {e}")
                # Fallback to uniform probabilities
                num_classes = self.num_labels
                uniform_probs = np.ones((n_samples, num_classes)) / num_classes
                return uniform_probs

    def get_params(self) -> dict:
        """Get calibrator parameters"""
        return {
            'metric': self.metric,
            'use_binning': self.use_binning,
            'n_bins': self.n_bins,
            'num_labels': self.num_labels,
            'is_fitted': self.is_fitted,
            'fitting_function': 'IsotonicRegression',
            # NEW: AugMix parameters
            'use_augmix': self.use_augmix,
            'augmix_versions': self.augmix_versions,
            'augmix_severity': self.augmix_severity,
            'augmix_width': self.augmix_width,
            'augmix_alpha': self.augmix_alpha
        }