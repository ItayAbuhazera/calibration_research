import logging
from sklearn.isotonic import IsotonicRegression
from calibrators.base_calibrator import BaseCalibrator
import numpy as np
from sklearn.preprocessing import label_binarize  # Import label_binarize

from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class IsotonicCalibrator(BaseCalibrator):
    def calibrate(self, probs):
        logger.info("Starting Isotonic Calibration.")
        logger.debug(f"Input probabilities shape: {probs.shape}")

        # Initialize calibrated probabilities array
        calibrated_probs = np.zeros_like(probs)

        # Calibrate each class separately
        for i in range(probs.shape[1]):
            calibrated_probs[:, i] = self.Isotonic[i].predict(probs[:, i])

        # Normalize probabilities to ensure they sum to one
        sum_calibrated = np.sum(calibrated_probs, axis=1, keepdims=True)
        remaining_prob = 1 - sum_calibrated
        remaining_prob = np.clip(remaining_prob, 0, None)  # Avoid negative remaining probabilities

        # Evenly distribute remaining probability across all classes
        n_classes = probs.shape[1]
        for i in range(calibrated_probs.shape[0]):  # Iterate over samples
            if remaining_prob[i] > 0:  # Distribute only if there's leftover probability
                calibrated_probs[i, :] += remaining_prob[i] / n_classes
        return calibrated_probs


    def fit(self, y_prob_val, y_true):
        logger.info("Fitting Isotonic Calibrator.")
        # Initialize list of IsotonicRegression models
        self.Isotonic = []
        n_classes = y_prob_val.shape[1]

        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

        for i in range(n_classes):
            iso_reg = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            iso_reg.fit(y_prob_val[:, i], y_true_bin[:, i])
            self.Isotonic.append(iso_reg)

        logger.info("Isotonic Calibrator fitted successfully.")


class IdentityCalibrator(BaseCalibrator):
    """ A class that implements no recalibration (identity). """

    def fit(self, logits, y):
        """ No fitting needed for identity calibrator. """
        logger.info("IdentityCalibrator: No fitting required.")
        pass

    def calibrate(self, probs):
        """ Return probabilities as-is (no calibration). """
        logger.info("IdentityCalibrator: Returning probabilities without calibration.")
        return probs
