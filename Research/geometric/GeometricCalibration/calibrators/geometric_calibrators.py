# geometric_calibrators.py
import numpy as np
import logging
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from calibrators.base_calibrator import BaseCalibrator
from utils.utils import Compression
from utils.logging_config import setup_logging
from sklearn.neighbors import KDTree
import tensorflow as tf
from scipy.special import softmax
from geometric.stability_space import StabilitySpace
# from utils.utils import StabilitySpace


setup_logging()
logger = logging.getLogger(__name__)

import logging
import numpy as np
# Ensure necessary imports are available
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from scipy.optimize import curve_fit
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from scipy.interpolate import interp1d
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False
try:
    from scipy.interpolate import UnivariateSpline
    _has_scipy_interpolate = True
except ImportError:
    _has_scipy_interpolate = False


# --- SigmoidFitter Class ---
class SigmoidFitter:
    """Fits a generalized sigmoid function using scipy.optimize.curve_fit."""
    def __init__(self):
        self.params = None
        self.fitted_func_name = "sigmoid_curve_fit" # For logging/identification

    def _sigmoid_func(self, x, a, b, c, d):
        """Generalized sigmoid: d + (a - d) / (1 + exp(-b * (x - c)))."""
        # Add clipping to prevent overflow in exp
        exponent = np.clip(-b * (x - c), -700, 700)
        denominator = 1 + np.exp(exponent)
        # Avoid division by zero or near-zero
        denominator[denominator < 1e-10] = 1e-10
        result = d + (a - d) / denominator
        # Clip output to ensure it stays within [0, 1] range typically expected for confidence
        return np.clip(result, 0.0, 1.0)

    def fit(self, X, y):
        """Fit sigmoid to X (stability scores) and y (accuracy) data."""
        X_flat = X.ravel()
        if len(X_flat) < 4: # Need at least 4 points to fit 4 parameters
             logger.warning("Not enough data points (<4) to fit sigmoid. Using fallback.")
             self._fallback_fit(X_flat, y)
             return self

        # Robust initial guesses
        a_guess = np.max(y) if len(y)>0 else 1.0
        d_guess = np.min(y) if len(y)>0 else 0.0
        c_guess = np.median(X_flat) if len(X_flat)>0 else 0.0
        # Heuristic for steepness: transition over roughly half the range of X
        x_range = np.ptp(X_flat) if len(X_flat)>1 else 1.0
        b_guess = 4.0 / max(x_range, 1e-6) # Avoid division by zero

        initial_guess = [a_guess, b_guess, c_guess, d_guess]

        # Robust bounds, ensure min/max X are finite
        min_x = np.min(X_flat) if len(X_flat)>0 and np.all(np.isfinite(X_flat)) else -np.inf
        max_x = np.max(X_flat) if len(X_flat)>0 and np.all(np.isfinite(X_flat)) else np.inf
        if not np.isfinite(min_x): min_x = -1e6 # Large negative if infinite
        if not np.isfinite(max_x): max_x = 1e6  # Large positive if infinite

        # Bounds: [a_min, b_min, c_min, d_min], [a_max, b_max, c_max, d_max]
        # Loosen bounds slightly compared to absolute min/max y
        y_min_bound = max(0.0, (np.min(y) if len(y)>0 else 0.0) - 0.1)
        y_max_bound = min(1.0, (np.max(y) if len(y)>0 else 1.0) + 0.1)
        # Ensure min_x <= max_x
        if min_x > max_x: min_x, max_x = max_x, min_x

        bounds = ([y_min_bound, 1e-6, min_x, 0.0],       # Lower bounds (b must be > 0)
                  [y_max_bound, 100.0, max_x, 1.0])      # Upper bounds (limit steepness)

        try:
            self.params, pcov = curve_fit(
                self._sigmoid_func, X_flat, y,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000,
                method='trf' # Trust Region Reflective often more robust with bounds
            )
            if np.any(np.isinf(pcov)): # Check if covariance is infinite (bad fit)
                 logger.warning("Sigmoid curve_fit resulted in infinite covariance. Using fallback.")
                 self._fallback_fit(X_flat, y)

        except (RuntimeError, ValueError) as e:
            logger.warning(f"Sigmoid curve_fit failed: {e}. Using fallback.")
            self._fallback_fit(X_flat, y)

        logger.info(f"Fitted sigmoid params (a,b,c,d): {self.params}")
        return self

    def _fallback_fit(self, X_flat, y):
        """Simple fallback parameter estimation if curve_fit fails."""
        a = np.max(y) if len(y)>0 else 1.0
        d = np.min(y) if len(y)>0 else 0.0
        c = np.median(X_flat) if len(X_flat)>0 else 0.0
        x_range = np.ptp(X_flat) if len(X_flat)>1 else 1.0
        b = 4.0 / max(x_range, 1e-6)
        # Ensure parameters are within reasonable bounds
        a = np.clip(a, 0.0, 1.0)
        d = np.clip(d, 0.0, 1.0)
        b = max(b, 1e-6)
        self.params = [a, b, c, d]
        logger.info(f"Using fallback sigmoid params (a,b,c,d): {self.params}")


    def predict(self, X):
        """Predict using fitted sigmoid function"""
        if self.params is None:
            raise ValueError("SigmoidFitter model not fitted yet")
        # Predict and clip again to be sure
        predictions = self._sigmoid_func(X.ravel(), *self.params)
        return np.clip(predictions, 0.0, 1.0)

# --- LowessWrapper Class ---
class LowessWrapper:
    """Wrapper for statsmodels lowess to provide fit/predict interface."""
    def __init__(self, frac=0.5):
        if not _has_statsmodels:
            raise ImportError("statsmodels is required for LOWESS fitting.")
        self.frac = frac
        self.x_train = None
        self.y_train = None
        self.interp_func = None
        self.fitted_func_name = "lowess"

    def fit(self, X, y):
        self.x_train = X.ravel()
        self.y_train = y
        # Fit lowess and create interpolation function immediately
        smoothed_y = lowess(self.y_train, self.x_train, frac=self.frac, return_sorted=False)
        # Sort points for interpolation
        sort_indices = np.argsort(self.x_train)
        sorted_x = self.x_train[sort_indices]
        sorted_y_smoothed = smoothed_y[sort_indices]
        # Create interpolation function, handle extrapolation carefully
        self.interp_func = interp1d(sorted_x, sorted_y_smoothed,
                                    bounds_error=False,
                                    fill_value=(sorted_y_smoothed[0], sorted_y_smoothed[-1])) # Fill with edge values
        return self

    def predict(self, X):
        if self.interp_func is None:
            raise ValueError("LowessWrapper model not fitted yet")
        x_pred = X.ravel()
        predictions = self.interp_func(x_pred)
        # Clip predictions to [0, 1]
        return np.clip(predictions, 0.0, 1.0)

# --- SplineWrapper Class ---
class SplineWrapper:
    """Wrapper for scipy UnivariateSpline to provide fit/predict interface."""
    def __init__(self, k=3, s=None):
        if not _has_scipy_interpolate:
             raise ImportError("scipy.interpolate is required for spline fitting.")
        self.k = k  # Degree
        self.s = s  # Smoothing factor
        self.spline = None
        self.fitted_func_name = "spline"

    def fit(self, X, y):
        try:
            # UnivariateSpline requires sorted X for fitting if s is None
            sort_indices = np.argsort(X.ravel())
            x_sorted = X.ravel()[sort_indices]
            y_sorted = y[sort_indices]
            # Determine smoothing factor 's' automatically if not provided
            # s = len(y) is a common heuristic for allowing fit close to data
            s_factor = self.s if self.s is not None else len(y)
            self.spline = UnivariateSpline(x_sorted, y_sorted, k=self.k, s=s_factor)
        except Exception as e:
            logger.error(f"Spline fitting failed: {e}. Check input data and parameters.")
            raise # Re-raise error if fitting fails critically
        return self

    def predict(self, X):
        if self.spline is None:
            raise ValueError("SplineWrapper model not fitted yet")
        predictions = self.spline(X.ravel())
        # Clip predictions to [0, 1]
        return np.clip(predictions, 0.0, 1.0)


class GeometricCalibrator(BaseCalibrator):
    """
    Class serving as a wrapper for the geometric calibration method (stability/separation).
    """

    def __init__(self, model, X_train_embed, y_train, X_train_original=None, fitting_func=None, 
                 compression_mode=None, compression_param=None, metric='l2', stability_space=None, 
                 library='faiss', use_binning=True, n_bins=200):
        """
        Initializes the GeometricCalibrator with a model, stability space, and calibration function.

        Args:
            model: The model to be calibrated (with `predict` and `predict_proba` methods).
            X_train_embed: Training data embeddings for geometric calculations.
            y_train: Training labels.
            X_train_original: Original training data for model predictions (if None, uses embeddings).
            fitting_func: Custom fitting function (default: IsotonicRegression).
            compression_mode: Compression mode for data.
            compression_param: Parameter controlling the compression level.
            metric: Distance metric for stability/separation calculations.
            stability_space: Optional custom StabilitySpace instance. If not provided, one is initialized automatically.
            library: The library used for stability calculation (default is 'faiss').
            use_binning (bool): Whether to bin stability scores and calculate average accuracy.
            n_bins (int): Number of bins for stability scores (default: 50).
        """
        logging.info("\n=== GeometricCalibrator Initialization ===")
        logging.info(f"Input X_train_embed shape: {X_train_embed.shape}")
        if X_train_original is not None:
            logging.info(f"Input X_train_original shape: {X_train_original.shape}")
        logging.info(f"Compression mode: {compression_mode}")
        logging.info(f"Compression param: {compression_param}")
        logging.info(f"Library: {library}")
        logging.info(f"Metric: {metric}")

        super().__init__()
        self.model = model
        self.X_train_original = X_train_original  # Store original training data
        self.popt = None
        self._fitted = False
        self.metric = metric.lower()  # Ensure metric is lowercase
        self.use_binning = use_binning
        self.n_bins = n_bins

        # Determine the number of classes (unique labels in y_train)
        self.num_labels = len(np.unique(y_train))
        logging.info(f"Number of unique labels: {self.num_labels}")

        # Default to IsotonicRegression if no custom fitting function is provided
        if isinstance(fitting_func, str):
            logger.info(f"Fitting function name provided: '{fitting_func}'. Creating instance.")
            # Use the helper function to get the actual object instance
            self.fitting_func_instance = self._get_fitting_function(fitting_func)
        elif fitting_func is None:
            # Default to IsotonicRegression if nothing is provided
            logger.info("No fitting function provided. Defaulting to IsotonicRegression.")
            from sklearn.isotonic import IsotonicRegression # Ensure import is here or at top
            self.fitting_func_instance = IsotonicRegression(out_of_bounds="clip")
        else:
            # Assume it's already an instantiated object
            logger.info(f"Custom fitting function object provided: {fitting_func.__class__.__name__}")
            self.fitting_func_instance = fitting_func

        # Use provided stability space or create a new one with the default settings
        if stability_space:
            self.stab_space = stability_space  # User provided custom StabilitySpace
            logger.info(f"{self.__class__.__name__}: Using custom StabilitySpace provided by user.")
        else:
            # Automatically initialize StabilitySpace with embeddings for geometric calculations
            self.stab_space = StabilitySpace(X_train_embed, y_train,
                                             compression=Compression(compression_mode, compression_param),
                                             library=library, metric=self.metric)
            logger.info(f"{self.__class__.__name__}: Initialized StabilitySpace with default settings"
                        f" (library: {library}, metric: {metric}).")

        logger.info(f"Initialized {self.__class__.__name__} with model {self.model.__class__.__name__}"
                    f" and fitting function {self.fitting_func_instance.__class__.__name__}.")
        
    def _get_fitting_function(self, method_name):
        """
        Returns the appropriate fitting function object based on the specified method name.
        """
        method_lower = method_name.lower()

        if method_lower == 'isotonic':
            logger.debug("Creating IsotonicRegression instance.")
            return IsotonicRegression(out_of_bounds="clip")

        elif method_lower == 'sigmoid':
            logger.debug("Creating SigmoidFitter instance.")
            return SigmoidFitter() # Use the new class

        elif method_lower == 'polynomial':
            logger.debug("Creating Polynomial Regression pipeline instance.")
            # Consider adjusting degree based on data complexity if needed
            return Pipeline([
                ('poly', PolynomialFeatures(degree=3)),
                ('linear', LinearRegression(fit_intercept=True)) # Ensure intercept
            ])

        elif method_lower == 'lasso':
            logger.debug("Creating LASSO Regression pipeline instance.")
            alpha = 0.001  # Reduced alpha for potentially better fit with less sparsity
            return Pipeline([
                ('poly', PolynomialFeatures(degree=2)), # Degree 2 might be more stable
                ('lasso', Lasso(alpha=alpha, max_iter=20000, tol=1e-3, fit_intercept=True)) # Increased max_iter, tol
            ])

        elif method_lower == 'elasticnet':
            logger.debug("Creating ElasticNet Regression pipeline instance.")
            alpha = 0.001 # Reduced alpha
            l1_ratio = 0.5 # Balance L1/L2
            return Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('elasticnet', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, tol=1e-3, fit_intercept=True))
            ])

        elif method_lower == 'lowess':
            if _has_statsmodels:
                logger.debug("Creating LowessWrapper instance.")
                # frac parameter might need tuning (0.2 to 0.8 common)
                return LowessWrapper(frac=0.3)
            else:
                logger.warning("statsmodels not installed. Falling back to isotonic regression for lowess request.")
                return IsotonicRegression(out_of_bounds="clip")

        elif method_lower == 'spline':
            if _has_scipy_interpolate:
                logger.debug("Creating SplineWrapper instance.")
                # Smoothing factor 's' is crucial. None uses interpolation, large 's' increases smoothing.
                # s=len(y) is a heuristic to allow fitting close to points.
                return SplineWrapper(k=3, s=None) # Try interpolation first (s=None)
            else:
                logger.warning("scipy.interpolate not installed. Falling back to isotonic regression for spline request.")
                return IsotonicRegression(out_of_bounds="clip")

        else:
            logger.warning(f"Unknown fitting method '{method_name}'. Using default IsotonicRegression.")
            return IsotonicRegression(out_of_bounds="clip")


    def fit(self, X_val_embed, y_val, X_val_original=None):
        """
        Fits the calibrator with the validation data using rounded stability and balanced accuracy.
    
        Args:
            X_val_embed: Validation data embeddings for geometric calculations.
            y_val: Validation labels.
            X_val_original: Original validation data for model predictions (if None, uses embeddings).
        """
        logger.info(f"{self.__class__.__name__}: Fitting with validation data using balanced accuracy and rounded stability.")
    
        try:
            # Step 1: Predict on validation data - use original data if provided
            prediction_data = X_val_original if X_val_original is not None else X_val_embed
            
            if hasattr(self.model, "predict_proba"):
                # For sklearn models
                logger.info(f"Sklearn model detected")
                batch_size = 64  # Adjust based on your GPU memory
                num_samples = len(prediction_data)
                y_pred_val = []
                y_pred_classes = []  # Add this line to initialize the list

                for i in range(0, num_samples, batch_size):
                    batch_data = prediction_data[i:i+batch_size]
                    batch_probs = self.model.predict_proba(batch_data)
                    batch_preds = np.argmax(batch_probs, axis=1)  # Get class predictions directly
                    y_pred_val.append(batch_probs)
                    y_pred_classes.append(batch_preds)

                y_pred_val = np.concatenate(y_pred_val, axis=0)
                y_pred_classes = np.concatenate(y_pred_classes, axis=0)

            elif hasattr(self.model, "predict"):
                # For tf.keras models
                y_pred_val = self.model.predict(prediction_data)  # Probabilities
                y_pred_classes = np.argmax(y_pred_val, axis=1)  # Class labels
            else:
                raise ValueError("Model does not support required prediction methods.")
            # Step 2: Compute stability values using embeddings
            stability_val = self.stab_space.calc_stab(X_val_embed, y_pred_val)
            logger.info(f"Stability values (first 5): {stability_val[:5]}")
    
            # Step 3: Round the stability values for binning
            rounded_stability = np.round(stability_val / 10) * 10
            unique_stabilities = np.unique(stability_val)
    
            if self.use_binning:
                # Step 3: Bin stability values
                self.logger.info(f"Binning stability values into {self.n_bins} bins.")
                min_stability, max_stability = np.min(stability_val), np.max(stability_val)
                bin_edges = np.linspace(min_stability, max_stability, self.n_bins + 1)
                bin_indices = np.digitize(stability_val, bins=bin_edges) - 1  # Bin indices start at 0
                self.logger.info(f"Bin edges: {bin_edges}")


                # Step 4: Compute average accuracy for each bin
                binned_stability = []
                binned_accuracy = []
                for bin_idx in range(self.n_bins):
                    indices_in_bin = np.where(bin_indices == bin_idx)[0]
                    if len(indices_in_bin) > 0:
                        y_true_bin = y_val[indices_in_bin]
                        y_pred_bin = y_pred_classes[indices_in_bin]
                        accuracy = np.mean(y_true_bin == y_pred_bin)
                        binned_stability.append((bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2)
                        binned_accuracy.append(accuracy)

                # Convert to arrays for fitting
                stability_vals = np.array(binned_stability)
                accuracies = np.array(binned_accuracy)
                # show the first five stabilities and their corresponding accuracies
                self.logger.info(f"Stability -> Confidence mapping from fitted function:")
                for stab, conf in zip(stability_vals[:5], accuracies[:5]):
                    self.logger.info(f"  {stab:.6f} -> {conf:.6f}")

            else:
                # Use raw stability and accuracies without binning
                unique_stabilities = np.unique(stability_val)
                stability_vals = []
                accuracies = []
                for stab in unique_stabilities:
                    indices = np.where(stability_val == stab)[0]
                    y_true_stab = y_val[indices]
                    y_pred_stab = y_pred_classes[indices]
                    acc = np.mean(y_true_stab == y_pred_stab)
                    stability_vals.append(stab)
                    accuracies.append(acc)
                stability_vals = np.array(stability_vals)
                accuracies = np.array(accuracies)
            # --- Calculate Pearson Correlation ---
            corr, p_value = np.nan, np.nan # Initialize to NaN
            try:
                # Calculate based on the same data used for fitting
                # (stability_vals and accuracies depend on self.use_binning)
                if len(stability_vals) > 1:
                    from scipy.stats import pearsonr
                    corr, p_value = pearsonr(stability_vals, accuracies)
                    logger.info(f"Pearson correlation between stability and accuracy: {corr:.4f} (p-value: {p_value:.4f})")
                else:
                    logger.warning("Not enough unique stability values to calculate correlation")
            except ValueError as e:
                logger.warning(f"Could not calculate Pearson correlation: {e}") # Handle potential errors

            # --- Fit the calibration model ---
            try:
                # log some info about the accuracy before fitting
                logger.info(f"Accuracy values (first 5): {accuracies[:5]}")
                logger.info(f"Maximum accuracy: {np.max(accuracies):.4f}")
                logger.info(f"Minimum accuracy: {np.min(accuracies):.4f}")
                logger.info(f"Mean accuracy: {np.mean(accuracies):.4f}")

                # Step 6: Fit the instantiated fitting function object
                self.popt = self.fitting_func_instance.fit(stability_vals.reshape(-1, 1), accuracies)
                self._fitted = True
                logger.info(f"{self.__class__.__name__}: Successfully fitted using stability-accuracy pairs and {self.fitting_func_instance.__class__.__name__}.")

            except Exception as e:
                logger.error(f"{self.__class__.__name__}: Failed to fit with error: {e}")
                self._fitted = False # Ensure fitted is false if fit fails
                # Re-raise the exception or handle as appropriate
                raise # Or return None, None, None if you want to handle it in the calling function
            logger.info("\n===== Validation Data Analysis =====")
            logger.info(f"Number of validation samples: {len(stability_val)}")
            logger.info(f"Stability value range: min={np.min(stability_val):.6f}, max={np.max(stability_val):.6f}")
            logger.info(f"Number of unique stability values: {len(np.unique(stability_val))}")

            # Create histogram data for stability values
            from collections import Counter
            stability_bins = np.linspace(np.min(stability_val), np.max(stability_val), 10)
            bin_indices = np.digitize(stability_val, stability_bins)
            bin_counts = Counter(bin_indices)
            for bin_idx, count in sorted(bin_counts.items()):
                # Determine bin boundaries
                if bin_idx > 0:
                    bin_start = f"{stability_bins[bin_idx-1]:.6f}"
                else:
                    bin_start = "-inf"
                
                if bin_idx < len(stability_bins):
                    bin_end = f"{stability_bins[bin_idx]:.6f}"
                else:
                    bin_end = "inf"
                
                logger.info(f"Bin {bin_idx}: [{bin_start}, {bin_end}] -> {count} samples")

            # Check if all stability values are negative
            negative_count = np.sum(stability_val < 0)
            logger.info(f"Negative stability values: {negative_count}/{len(stability_val)} ({negative_count/len(stability_val)*100:.2f}%)")
            # After fitting the function (self.popt = self.fitting_func_instance.fit(...))
            logger.info("\n===== Fitted Function Analysis =====")

            # Generate test points across the full range of stability values
            test_stability = np.linspace(np.min(stability_vals) - 0.1, np.max(stability_vals) + 0.1, 20)
            test_stability = test_stability.reshape(-1, 1)

            # Check output of the fitted function
            fitted_outputs = self.popt.predict(test_stability)

            # Display the function's mapping
            logger.info("Stability -> Confidence mapping from fitted function:")
            for stab, conf in zip(test_stability.flatten(), fitted_outputs.flatten()):
                logger.info(f"  {stab:.6f} -> {conf:.6f}")

            # Calculate output range
            output_range = np.max(fitted_outputs) - np.min(fitted_outputs)
            logger.info(f"Output range of fitted function: {output_range:.6f}")
            if output_range < 0.1:
                logger.info("WARNING: Fitted function has very narrow output range - will produce similar confidence values")
            # Add this line:
            self.stability_vals = stability_vals  # Store as instance variable


            # --- Return correlation values ---
            return corr, p_value
        except Exception as e:
            raise ValueError(f"{self.__class__.__name__}: Fitting failed with error: {e}")
            
    def calibrate(self, X_test_embed, X_test_original=None):
            """
            Calibrates the test data based on the fitted model.
        
            Args:
                X_test: Test data (flattened images).
        
            Returns:
                np.ndarray: Calibrated probability matrix for each image and class.
            """
            if not self._fitted:
                raise ValueError("You must fit the calibrator before using it.")
        
            logger.info(f"{self.__class__.__name__}: Calibrating test data.")
            X_test_original = X_test_original if X_test_original is not None else X_test_embed
            try:
                # Predict on the test data using the trained model (get predicted probabilities for all classes)
                # Step 1: Predict on validation data
                if hasattr(self.model, "predict_proba"):
                    # For sklearn models
                    y_test_pred = self.model.predict_proba(X_test_original)  # Probabilities
                    y_test_labels = self.model.predict(X_test_original)  # Class labels
                elif hasattr(self.model, "predict"):
                    # For tf.keras models
                    y_test_pred = self.model.predict(X_test_original)  # Probabilities
                    y_test_labels = np.argmax(y_test_pred, axis=1)  # Class labels
                else:
                    raise ValueError("Model does not support required prediction methods.")

                # Initialize progress bar using tqdm
                num_samples = X_test_original.shape[0]
                num_classes = y_test_pred.shape[1]
                calibrated_probs = np.zeros((num_samples, num_classes))  # Initialize a matrix to store calibrated probabilities
        
                logger.info(f"Starting calibration for {num_samples} samples and {num_classes} classes.")
        
                # Compute stability for the predicted probabilities
                stability_test = self.stab_space.calc_stab(X_test_embed, y_test_pred)
                logger.info(f"Stability values during calibration (first 10): {stability_test[:10]}")  # Add logging
        
                # Apply the fitted calibration function to the stability values
                calibrated_values = self.popt.predict(stability_test.reshape(-1, 1))
                logger.info(f"Calibrated values (first 10): {calibrated_values[:10]}")  # Add logging
                # Since the y_pred_test is show all the probability of all the classes, we want to use just the maximum of them.
                y_pred_test_max_first_10 = np.max(y_test_pred[:10], axis=1)
                logger.info(f"Uncalibrated probabilities (first 10): {y_pred_test_max_first_10} ")
                # Distribute the calibrated values across the predicted class
                for i in range(X_test_original.shape[0]):
                    # Assign the calibrated probability to the predicted class label
                    calibrated_probs[i, y_test_labels[i]] = calibrated_values[i]
                    
                    # Distribute remaining probability equally across other classes
                    remaining_prob = (1 - calibrated_values[i]) / (self.num_labels - 1)
                    for j in range(self.num_labels):
                        if j != y_test_labels[i]:
                            calibrated_probs[i, j] = remaining_prob
        
                # Ensure probabilities are in [0, 1] and sum to 1
                calibrated_probs = np.clip(calibrated_probs, 0, 1)
                calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
                logger.info("\n===== Test Data Stability Analysis =====")
                logger.info(f"Test stability value range: min={np.min(stability_test):.6f}, max={np.max(stability_test):.6f}")

                # Check if test stability values are within the range seen during fitting
                val_min, val_max = np.min(self.stability_vals), np.max(self.stability_vals)
                test_min, test_max = np.min(stability_test), np.max(stability_test)
                test_in_range = np.logical_and(stability_test >= val_min, stability_test <= val_max)
                in_range_count = np.sum(test_in_range)
                logger.info(f"Validation stability range: [{val_min:.6f}, {val_max:.6f}]")
                logger.info(f"Test stability range: [{test_min:.6f}, {test_max:.6f}]")
                logger.info(f"Test values within validation range: {in_range_count}/{len(stability_test)} ({in_range_count/len(stability_test)*100:.2f}%)")

                # Check distribution of test stability values
                negative_test = np.sum(stability_test < 0)
                logger.info(f"Negative test stability values: {negative_test}/{len(stability_test)} ({negative_test/len(stability_test)*100:.2f}%)")
                # After computing calibrated_values
                logger.info("\n===== Calibrated Values Analysis =====")
                calibrated_unique = np.unique(calibrated_values)
                logger.info(f"Number of unique calibrated values: {len(calibrated_unique)}")
                logger.info(f"Calibrated value range: min={np.min(calibrated_values):.6f}, max={np.max(calibrated_values):.6f}")
                logger.info(f"Standard deviation of calibrated values: {np.std(calibrated_values):.6f}")

                # Count most common values
                from collections import Counter
                rounded_values = np.round(calibrated_values, 3)
                value_counts = Counter(rounded_values)
                logger.info("Most common calibrated values:")
                for value, count in value_counts.most_common(5):
                    logger.info(f"  {value:.3f}: {count} occurrences ({count/len(calibrated_values)*100:.2f}%)")

                logger.info(f"{self.__class__.__name__}: Calibration successful.")
        
                return calibrated_probs
        
            except Exception as e:
                logger.error(f"{self.__class__.__name__}: Calibration failed with error: {e}")
                raise


class SeparationCalibrator(BaseCalibrator):
    """
    A calibrator based on separation calculations.
    """

    def __init__(self, model, X_train, y_train, compression_mode=None, compression_param=None, metric='l2'):
        """
        Initializes the SeparationCalibrator with a model and separation metrics.

        Args:
            model: The model to be calibrated.
            X_train: Training data.
            y_train: Training labels.
            compression_mode: Compression mode for data.
            compression_param: Parameter controlling compression.
            metric: Distance metric for separation.
        """
        super().__init__()
        self.model = model
        self._fitted = False
        self.separation_space = StabilitySpace(X_train, y_train,
                                                compression=Compression(compression_mode, compression_param),
                                                metric=metric)
        logger.info(f"Initialized {self.__class__.__name__} with separation metrics.")

    def fit(self, X_val, y_val):
        """
        Fits the calibrator using separation metrics.

        Args:
            X_val: Validation data.
            y_val: Validation labels.
        """
        logger.info(f"{self.__class__.__name__}: Fitting using separation metrics.")
        try:
            y_pred_val = self.model.predict(X_val)
            separation_val = self.separation_space.calc_sep(X_val, y_pred_val)

            # Fit isotonic regression based on separation values
            correct = y_val == y_pred_val
            self.popt = IsotonicRegression(out_of_bounds="clip").fit(separation_val, correct)
            self._fitted = True
            logger.info(f"{self.__class__.__name__}: Successfully fitted using separation values.")
        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Failed to fit using separation values: {e}")
            raise

    def calibrate(self, X_test):
        """
        Calibrates the test data based on the fitted model.

        Args:
            X_test: Test data.

        Returns:
            Calibrated probabilities.
        """
        if not self._fitted:
            raise ValueError("You must fit the calibrator before using it.")
        logger.info(f"{self.__class__.__name__}: Calibrating based on separation.")

        try:
            y_test_pred = self.model.predict(X_test)
            separation_test = self.separation_space.calc_sep(X_test, y_test_pred)
            calibrated_probs = self.popt.predict(separation_test)
            logger.info(f"{self.__class__.__name__}: Calibration successful.")
            return calibrated_probs
        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Calibration failed with error: {e}")
            raise

class GeometricCalibratorTrust(BaseCalibrator):
    def __init__(self, model, X_train, y_train, fitting_func=None, k=10, min_dist=1e-12, 
                 use_binning=False, n_bins=50, use_filtering=False, alpha=0.0):
        """
        Initializes the GeometricCalibratorTrust.
        """
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.min_dist = min_dist
        self.use_binning = use_binning
        self.n_bins = n_bins
        self.use_filtering = use_filtering
        self.alpha = alpha
        self._fitted = False

        # Flatten the training data if necessary
        if len(self.X_train.shape) > 2:
            logger.info(f"Flattening X_train with original shape: {self.X_train.shape}")
            self.X_train = self.X_train.reshape(self.X_train.shape[0], -1)
            logger.info(f"Flattened X_train shape: {self.X_train.shape}")

        # Initialize KD-trees for each class
        self.num_labels = len(np.unique(y_train))
        self.kdtrees = [None] * self.num_labels

        # Filter training data if needed
        if self.use_filtering:
            self.filter_by_density()
        else:
            # Initialize KD-trees without filtering
            for label in range(self.num_labels):
                X_label = self.X_train[np.where(self.y_train == label)[0]]
                self.kdtrees[label] = KDTree(X_label)

        # Default to IsotonicRegression if no custom fitting function is provided
        self.fitting_func = fitting_func if fitting_func else IsotonicRegression(out_of_bounds="clip")
        
        logger.info(f"Initialized {self.__class__.__name__} with filtering: {use_filtering}")

    def filter_by_density(self):
        """Filter out points with low kNN density for each class."""
        for label in range(self.num_labels):
            X_label = self.X_train[np.where(self.y_train == label)[0]]
            if len(X_label) > 0:
                kdtree = KDTree(X_label)
                knn_radii = kdtree.query(X_label, k=self.k)[0][:, -1]
                eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
                filtered_indices = np.where(knn_radii <= eps)[0]
                filtered_X = X_label[filtered_indices]
                
                if len(filtered_X) > 0:
                    self.kdtrees[label] = KDTree(filtered_X)
                else:
                    logger.warning(f"No points remained after filtering for class {label}")
                    self.kdtrees[label] = KDTree(X_label)  # Use unfiltered data as fallback

    def calculate_trust_scores(self, X, pred_labels):
        """Calculate trust scores for the given data points and predictions."""
        # Flatten X if necessary
        if len(X.shape) > 2:
            logger.info(f"Flattening X with original shape: {X.shape}")
            X = X.reshape(X.shape[0], -1)
            logger.info(f"Flattened X shape: {X.shape}")

        distances = np.zeros((X.shape[0], self.num_labels))
        for label in range(self.num_labels):
            distances[:, label] = self.kdtrees[label].query(X, k=2)[0][:, -1]
        
        sorted_distances = np.sort(distances, axis=1)
        d_to_pred = distances[range(distances.shape[0]), pred_labels]
        d_to_closest_not_pred = np.where(
            sorted_distances[:, 0] != d_to_pred,
            sorted_distances[:, 0],
            sorted_distances[:, 1]
        )
        
        return d_to_closest_not_pred / (d_to_pred + self.min_dist)

    def fit(self, X_val, y_val):
        """
        Fits the calibrator using validation data.
        """
        logger.info(f"{self.__class__.__name__}: Fitting with validation data.")
        try:
            # Flatten X_val if the model is not TensorFlow/Keras
            logger.info(f"Original X_val shape: {X_val.shape}")
            if not isinstance(self.model, tf.keras.Model) and len(X_val.shape) > 2:
                X_val = X_val.reshape(X_val.shape[0], -1)
            logger.info(f"Processed X_val shape: {X_val.shape}")

            # Get model predictions
            if hasattr(self.model, "predict_proba"):
                y_pred_val = self.model.predict_proba(X_val)
                y_pred_classes = self.model.predict(X_val)
            elif hasattr(self.model, "predict"):
                y_pred_val = self.model.predict(X_val)
                y_pred_classes = np.argmax(y_pred_val, axis=1)
            else:
                raise ValueError("Model does not support required prediction methods.")

            # Calculate trust scores
            trust_scores = self.calculate_trust_scores(X_val, y_pred_classes)
            logger.info(f"Trust scores (first 10): {trust_scores[:10]}")

            if self.use_binning:
                # Bin trust scores
                min_score, max_score = np.min(trust_scores), np.max(trust_scores)
                bin_edges = np.linspace(min_score, max_score, self.n_bins + 1)
                bin_indices = np.digitize(trust_scores, bins=bin_edges) - 1
                
                # Compute average accuracy for each bin
                binned_scores = []
                binned_accuracy = []
                for bin_idx in range(self.n_bins):
                    indices_in_bin = np.where(bin_indices == bin_idx)[0]
                    if len(indices_in_bin) > 0:
                        y_true_bin = y_val[indices_in_bin]
                        y_pred_bin = y_pred_classes[indices_in_bin]
                        accuracy = np.mean(y_true_bin == y_pred_bin)
                        binned_scores.append((bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2)
                        binned_accuracy.append(accuracy)
                
                scores = np.array(binned_scores)
                accuracies = np.array(binned_accuracy)
            else:
                # Use raw scores
                unique_scores = np.unique(trust_scores)
                scores = []
                accuracies = []
                for score in unique_scores:
                    indices = np.where(trust_scores == score)[0]
                    accuracy = np.mean(y_val[indices] == y_pred_classes[indices])
                    scores.append(score)
                    accuracies.append(accuracy)
                scores = np.array(scores)
                accuracies = np.array(accuracies)

            # Fit the calibration function
            self.popt = self.fitting_func.fit(scores.reshape(-1, 1), accuracies)
            self._fitted = True
            logger.info(f"{self.__class__.__name__}: Successfully fitted.")

        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Failed to fit with error: {e}")
            raise

    def calibrate(self, X_test):
        """
        Calibrates the test data based on the fitted model.
        
        Args:
            X_test: Test data
            
        Returns:
            Calibrated probability matrix for each sample and class
        """
        if not self._fitted:
            raise ValueError("You must fit the calibrator before using it.")
        
        logger.info(f"{self.__class__.__name__}: Calibrating test data.")
        
        try:
            # Get model predictions
            if hasattr(self.model, "predict_proba"):
                y_test_pred = self.model.predict_proba(X_test)
                y_test_labels = self.model.predict(X_test)
            elif hasattr(self.model, "predict"):
                y_test_pred = self.model.predict(X_test)
                y_test_labels = np.argmax(y_test_pred, axis=1)
            
            # Calculate trust scores
            trust_scores = self.calculate_trust_scores(X_test, y_test_labels)
            
            # Apply calibration function
            calibrated_values = self.popt.predict(trust_scores.reshape(-1, 1))
            
            # Initialize calibrated probabilities matrix
            num_samples = X_test.shape[0]
            calibrated_probs = np.zeros_like(y_test_pred)
            
            # Distribute calibrated values
            for i in tqdm(range(num_samples)):
                pred_class = y_test_labels[i]
                calibrated_probs[i] = y_test_pred[i]  # Copy original probabilities
                calibrated_probs[i] *= calibrated_values[i] / np.sum(calibrated_probs[i])  # Scale by calibrated value
            
            return calibrated_probs

        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Failed to calibrate with error: {e}")
            raise