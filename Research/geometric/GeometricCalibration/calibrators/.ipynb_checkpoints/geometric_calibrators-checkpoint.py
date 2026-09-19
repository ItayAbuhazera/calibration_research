# geometric_calibrators.py
import numpy as np
import logging
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

from calibrators.base_calibrator import BaseCalibrator
from utils.utils import StabilitySpace, Compression, calc_balanced_acc
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class GeometricCalibrator(BaseCalibrator):
    """
    Class serving as a wrapper for the geometric calibration method (stability/separation).
    """

    def __init__(self, model, X_train, y_train, fitting_func=None, compression_mode=None, compression_param=None,
                 metric='l2', stability_space=None, library='faiss'):
        """
        Initializes the GeometricCalibrator with a model, stability space, and calibration function.

        Args:
            model: The model to be calibrated (with `predict` and `predict_proba` methods).
            X_train: Training data (flattened images).
            y_train: Training labels.
            fitting_func: Custom fitting function (default: IsotonicRegression).
            compression_mode: Compression mode for data.
            compression_param: Parameter controlling the compression level.
            metric: Distance metric for stability/separation calculations.
            stability_space: Optional custom StabilitySpace instance. If not provided, one is initialized automatically.
            library: The library used for stability calculation (default is 'faiss').
        """
        super().__init__()
        self.model = model
        self.popt = None
        self._fitted = False

        # Default to IsotonicRegression if no custom fitting function is provided
        self.fitting_func = fitting_func if fitting_func else IsotonicRegression(out_of_bounds="clip")

        # Use provided stability space or create a new one with the default settings
        if stability_space:
            self.stab_space = stability_space  # User provided custom StabilitySpace
            logger.info(f"{self.__class__.__name__}: Using custom StabilitySpace provided by user.")
        else:
            # Automatically initialize StabilitySpace with defaults if not provided
            self.stab_space = StabilitySpace(X_train, y_train,
                                             compression=Compression(compression_mode, compression_param),
                                             library=library, metric=metric)
            logger.info(f"{self.__class__.__name__}: Initialized StabilitySpace with default settings"
                        f" (library: {library}, metric: {metric}).")

        logger.info(f"Initialized {self.__class__.__name__} with model {self.model.__class__.__name__}"
                    f" and fitting function {self.fitting_func.__class__.__name__}.")

    def fit(self, X_val, y_val):
        """
        Fits the calibrator with the validation data using rounded stability and balanced accuracy.
    
        Args:
            X_val: Validation data (flattened images).
            y_val: Validation labels.
        """
        logger.info(f"{self.__class__.__name__}: Fitting with validation data using balanced accuracy and rounded stability.")
    
        try:
            # Step 1: Predict on validation data
            y_pred_val = self.model.predict(X_val)
            y_pred_classes = np.argmax(y_pred_val, axis=1)  # Convert predictions to class labels
    
            # Step 2: Compute stability values based on predictions
            stability_val = self.stab_space.calc_stab(X_val, y_pred_val)
            logger.info(f"Stability values (first 5): {stability_val[:5]}")  # Add logging
    
            # Step 3: Calculate balanced accuracy using the updated calc_balanced_acc function with rounding
            num_classes = len(np.unique(y_val))  # Get the number of unique classes
            s_bal_acc = calc_balanced_acc(stability_val, y_pred_classes, num_classes, round_digits=1)  # Compute balanced accuracy with rounding
    
            # Step 4: Prepare calibration data: (rounded_stability, mean balanced accuracy) pairs
            calibration_data = [(stab, s_bal_acc[stab]) for stab in s_bal_acc]
    
            # Step 5: Fit the provided fitting function (e.g., IsotonicRegression) on the calibration data
            stability_vals, accuracies = zip(*calibration_data)
            self.popt = self.fitting_func.fit(np.array(stability_vals).reshape(-1, 1), np.array(accuracies))
    
            self._fitted = True
            logger.info(f"{self.__class__.__name__}: Successfully fitted using balanced accuracy and {self.fitting_func.__class__.__name__}.")
    
        except Exception as e:
            logger.error(f"{self.__class__.__name__}: Failed to fit with error: {e}")
            raise

    def calibrate(self, X_test):
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
    
        try:
            # Predict on the test data using the trained model (get predicted probabilities for all classes)
            y_test_pred = self.model.predict(X_test)
            y_test_labels = np.argmax(y_test_pred, axis=1)  # Get predicted class labels from probabilities
    
            # Initialize progress bar using tqdm
            num_samples = X_test.shape[0]
            num_classes = y_test_pred.shape[1]
            calibrated_probs = np.zeros((num_samples, num_classes))  # Initialize a matrix to store calibrated probabilities
    
            logger.info(f"Starting calibration for {num_samples} samples and {num_classes} classes.")
    
            # Compute stability for the predicted probabilities
            stability_test = self.stab_space.calc_stab(X_test, y_test_pred)
            logger.info(f"Stability values during calibration (first 5): {stability_test[:5]}")  # Add logging
    
            # Apply the fitted calibration function to the stability values
            calibrated_values = self.popt.predict(stability_test.reshape(-1, 1))
            logger.info(f"Calibrated values (first 5): {calibrated_values[:5]}")  # Add logging
    
            # Distribute the calibrated values across the predicted class
            for i in range(X_test.shape[0]):
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
