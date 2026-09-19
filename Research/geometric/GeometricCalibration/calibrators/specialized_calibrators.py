import logging
from calibrators.base_calibrator import BaseCalibrator
import numpy as np

from calibrators.calibrators import HB_toplabel
from utils.logging_config import setup_logging
from utils.temperature_utils import temperature_scaling

setup_logging()
logger = logging.getLogger(__name__)

class HBCalibrator(BaseCalibrator):
    """
    Histogram Binning Calibrator (HB).
    Implemented based on https://github.com/aigen/df-posthoc-calibration.
    """

    def __init__(self, bins=50):
        super().__init__()
        self.bins = bins
        logger.info(f"Initialized HBCalibrator with bins={bins}")

    def fit(self, val_proba, y_val):
        logger.info("HBCalibrator: Fitting model using validation data.")
        try:
            self.calibrator = HB_toplabel(points_per_bin=self.bins)
            self.calibrator.fit(val_proba, y_val)
            logger.info("HBCalibrator: Fitting complete.")
        except Exception as e:
            logger.error(f"HBCalibrator: Fitting failed with error: {e}")
            raise

    def calibrate(self, probs):
        logger.info("HBCalibrator: Calibrating probabilities.")
        try:
            return self.calibrator.predict_proba(probs)
        except Exception as e:
            logger.error(f"HBCalibrator: Calibration failed with error: {e}")
            raise



class BBQCalibrator(BaseCalibrator):
    """
    BBQ Calibrator (Binning and Beta Calibration).
    Implemented by https://github.com/EFS-OpenSource/calibration-framework.
    """

    def __init__(self, bins=50):
        super().__init__()
        self.bins = bins
        logger.info(f"Initialized BBQCalibrator with bins={bins}")

    def fit(self, val_proba, y_val):
        logger.info("BBQCalibrator: Fitting model using validation data.")
        try:
            self.calibrator = BBQ()
            self.calibrator.fit(val_proba, y_val)
            logger.info("BBQCalibrator: Fitting complete.")
        except Exception as e:
            logger.error(f"BBQCalibrator: Fitting failed with error: {e}")
            raise

    def calibrate(self, probs):
        logger.info("BBQCalibrator: Calibrating probabilities.")
        try:
            return self.calibrator.transform(probs)
        except Exception as e:
            logger.error(f"BBQCalibrator: Calibration failed with error: {e}")
            raise


class BetaCalibrator(BaseCalibrator):
    """
    Beta Calibration Calibrator.
    Implemented by https://github.com/EFS-OpenSource/calibration-framework.
    """

    def __init__(self, bins=50):
        super().__init__()
        self.bins = bins
        logger.info(f"Initialized BetaCalibrator with bins={bins}")

    def fit(self, val_proba, y_val):
        logger.info("BetaCalibrator: Fitting model using validation data.")
        try:
            self.calibrator = BetaCalibration()
            self.calibrator.fit(val_proba, y_val)
            logger.info("BetaCalibrator: Fitting complete.")
        except Exception as e:
            logger.error(f"BetaCalibrator: Fitting failed with error: {e}")
            raise

    def calibrate(self, probs):
        logger.info("BetaCalibrator: Calibrating probabilities.")
        try:
            return self.calibrator.transform(probs)
        except Exception as e:
            logger.error(f"BetaCalibrator: Calibration failed with error: {e}")
            raise

class SBCCalibrator(BaseCalibrator):
    """ SBC Calibrator based on PlattBinnerMarginalCalibrator """

    def __init__(self, bins=15):
        super().__init__()
        self.bins = bins
        logger.info(f"Initialized SBCCalibrator with bins={bins}")

    def fit(self, val_proba, y_val):
        logger.info("SBCCalibrator: Fitting model.")
        try:
            self.calibrator = SBC.PlattBinnerMarginalCalibrator(len(val_proba), num_bins=self.bins)
            self.calibrator.train_calibration(val_proba, y_val)
            logger.info("SBCCalibrator: Successfully fitted the model.")
        except Exception as e:
            logger.error(f"SBCCalibrator: Fitting failed with error: {e}")
            raise

    def calibrate(self, probs):
        """ Calibrate probabilities using SBC. """
        logger.info("SBCCalibrator: Calibrating probabilities.")
        try:
            return self.calibrator.calibrate(probs)
        except Exception as e:
            logger.error(f"SBCCalibrator: Calibration failed with error: {e}")
            raise

class SBC_TOP_Calibrator(BaseCalibrator):
    """ SBC Top-label calibrator. """

    def __init__(self, bins=15):
        super().__init__()
        self.bins = bins
        logger.info(f"Initialized SBC_TOP_Calibrator with bins={bins}")

    def fit(self, top_probs, correct):
        logger.info("SBC_TOP_Calibrator: Fitting model.")
        try:
            self._platt = SBC.utils.get_platt_scaler(top_probs, correct)
            platt_probs = self._platt(top_probs)
            bins = SBC.utils.get_equal_bins(platt_probs, num_bins=self.bins)
            self._discrete_calibrator = SBC.utils.get_discrete_calibrator(platt_probs, bins)
            logger.info("SBC_TOP_Calibrator: Successfully fitted the model.")
        except Exception as e:
            logger.error(f"SBC_TOP_Calibrator: Fitting failed with error: {e}")
            raise

    def calibrate(self, probs):
        """ Calibrate probabilities. """
        logger.info("SBC_TOP_Calibrator: Calibrating probabilities.")
        try:
            top_probs = self._platt(SBC.utils.get_top_probs(probs))
            return self._discrete_calibrator(top_probs)
        except Exception as e:
            logger.error(f"SBC_TOP_Calibrator: Calibration failed with error: {e}")
            raise