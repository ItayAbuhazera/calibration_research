import logging
from calibrators.base_calibrator import BaseCalibrator
import numpy as np

from calibrators.specialized_calibrators import SBC_TOP_Calibrator
from utils.logging_config import setup_logging
from utils.temperature_utils import temperature_scaling, mse_t, ll_t

setup_logging()
logger = logging.getLogger(__name__)


class EnsembleTSCalibrator(BaseCalibrator):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.weights = None
        logger.info(f"Initialized EnsembleTSCalibrator with temperature={temperature}")

    def fit(self, logits, y):
        logger.info("EnsembleTSCalibrator: Fitting model.")
        try:
            if not isinstance(logits, np.ndarray) or not isinstance(y, np.ndarray):
                raise TypeError("Inputs 'logits' and 'y' must be NumPy arrays.")
            if logits.ndim != 2:
                raise ValueError("Logits must be a 2D array.")
            self.n_classes = logits.shape[1]
            _y = np.eye(self.n_classes)[y]
            self.temperature, self.weights = ets_calibrate(logits, _y, self.n_classes, 'mse')
            logger.info(f"EnsembleTSCalibrator: Successfully fitted the model with temperature: {self.temperature}")
        except Exception as e:
            logger.error(f"EnsembleTSCalibrator: Fitting failed with error: {e}")
            raise

    def calibrate(self, probs):
        logger.info("EnsembleTSCalibrator: Calibrating probabilities.")
        try:
            tempered_probs = probs ** (1. / self.temperature)
            logger.debug(f"Tempered probabilities before normalization: {tempered_probs}")

            denominator = np.sum(tempered_probs, axis=1, keepdims=True)
            logger.debug(f"Denominator for normalization: {denominator}")

            if not np.all(np.isfinite(denominator)):
                raise ValueError("Denominator contains non-finite values (NaN or Inf).")
            if np.any(denominator == 0):
                raise ValueError("Denominator contains zeros, leading to division by zero.")

            tempered_probs /= denominator
            p = np.ones_like(tempered_probs) / self.n_classes
            calibrated_probs = self.weights[0] * tempered_probs + self.weights[1] * probs + self.weights[2] * p
            return calibrated_probs
        except Exception as e:
            logger.error(f"EnsembleTSCalibrator: Calibration failed with error: {e}")
            raise

class stab_SBC_Calibrator(BaseCalibrator):
    """ Stability-based SBC Calibrator that combines Stability and SBC_TOP calibration. """

    def __init__(self, bins=15):
        super().__init__()
        self.bins = bins
        self.stab_cali = StabilityCalibrator()
        self.SBCTOP_cali = SBC_TOP_Calibrator()
        logger.info(f"Initialized stab_SBC_Calibrator with bins={bins}")

    def fit(self, stab_val, val_probs, corrects):
        logger.info("stab_SBC_Calibrator: Fitting StabilityCalibrator.")
        try:
            top_probs = np.max(val_probs, axis=1)
            self.stab_cali.fit(stab_val, corrects)
            logger.info("stab_SBC_Calibrator: StabilityCalibrator fitting complete.")

            calibrated_val_probs = self.stab_cali.calibrate(stab_val)
            self.SBCTOP_cali.fit(calibrated_val_probs, corrects)
            logger.info("stab_SBC_Calibrator: SBCTOPCalibrator fitting complete.")
        except Exception as e:
            logger.error(f"stab_SBC_Calibrator: Fitting failed with error: {e}")
            raise

    def calibrate(self, stab_test):
        logger.info("stab_SBC_Calibrator: Calibrating test data.")
        try:
            first_cali_probs = self.stab_cali.calibrate(stab_test)
            second_cali_probs = self.SBCTOP_cali.calibrate(first_cali_probs)
            logger.info("stab_SBC_Calibrator: Calibration complete.")
            return second_cali_probs
        except Exception as e:
            logger.error(f"stab_SBC_Calibrator: Calibration failed with error: {e}")
            raise

class StabilityHistogramBinningCalibrator(BaseCalibrator):
    """
    Composition of Stability with Histogram Binning.
    """

    def __init__(self, num_bins=50):
        super().__init__()
        self.popt = None
        self.num_bins = num_bins
        logger.info(f"Initialized StabilityHistogramBinningCalibrator with num_bins={num_bins}")

    def fit(self, stab_validation, y_true_validation):
        logger.info("StabilityHistogramBinningCalibrator: Fitting using stability data.")
        try:
            from utils.utils import fitting_function, histogramBinning
            self.popt = fitting_function(stab_validation, y_true_validation)

            # Predictions of the regression on the same problem
            probs = self.popt[0].predict(stab_validation).reshape(-1, 1)

            # Histogram binning
            self.bin_means, self.bins_ranges, self.new_ranges = histogramBinning(probs, y_true_validation, self.num_bins)
            logger.info("StabilityHistogramBinningCalibrator: Fitting complete.")
        except Exception as e:
            logger.error(f"StabilityHistogramBinningCalibrator: Fitting failed with error: {e}")
            raise

    def calibrate(self, stability_test):
        logger.info("StabilityHistogramBinningCalibrator: Calibrating stability test data.")
        try:
            from utils.utils import get_bin
            test_probs = self.popt[0].predict(stability_test).reshape(-1, 1)

            calibrated_probs = []
            for prob in test_probs:
                bin_idx = get_bin(prob, self.bins_ranges)
                if bin_idx > self.num_bins - 1:
                    calibrated_probs.append(self.bin_means[bin_idx - 1])
                else:
                    calibrated_probs.append(self.bin_means[bin_idx])
            logger.info("StabilityHistogramBinningCalibrator: Calibration complete.")
            return calibrated_probs
        except Exception as e:
            logger.error(f"StabilityHistogramBinningCalibrator: Calibration failed with error: {e}")
            raise


import numpy as np
from scipy import optimize
from sklearn.isotonic import IsotonicRegression

"""
auxiliary functions for optimizing the temperature (scaling approaches) and weights of ensembles
*args include logits and labels from the calibration dataset:
"""


def mse_t(t, *args):
    ## find optimal temperature with MSE loss function

    logit, label = args
    logit = logit / t
    n = np.sum(np.exp(logit), 1)
    p = np.exp(logit) / n[:, None]
    mse = np.mean((p - label) ** 2)
    return mse


def ll_t(t, *args):
    ## find optimal temperature with Cross-Entropy loss function

    logit, label = args
    logit = logit / t
    n = np.sum(np.exp(logit), 1)
    p = np.clip(np.exp(logit) / n[:, None], 1e-20, 1 - 1e-20)
    N = p.shape[0]
    ce = -np.sum(label * np.log(p)) / N
    return ce


def mse_w(w, *args):
    ## find optimal weight coefficients with MSE loss function

    p0, p1, p2, label = args
    p = w[0] * p0 + w[1] * p1 + w[2] * p2
    p = p / np.sum(p, 1)[:, None]
    mse = np.mean((p - label) ** 2)
    return mse


def ll_w(w, *args):
    ## find optimal weight coefficients with Cros-Entropy loss function

    p0, p1, p2, label = args
    p = (w[0] * p0 + w[1] * p1 + w[2] * p2)
    N = p.shape[0]
    ce = -np.sum(label * np.log(p)) / N
    return ce


##### Ftting Temperature Scaling
def temperature_scaling(logit, label, loss):
    bnds = ((0.05, 5.0),)
    if loss == 'ce':
        t = optimize.minimize(ll_t, 1.0, args=(logit, label), method='L-BFGS-B', bounds=bnds, tol=1e-12,
                              options={'disp': False})
    if loss == 'mse':
        t = optimize.minimize(mse_t, 1.0, args=(logit, label), method='L-BFGS-B', bounds=bnds, tol=1e-12,
                              options={'disp': False})
    t = t.x
    return t



##### Ftting Enseble Temperature Scaling
def ensemble_scaling(logit, label, loss, t, n_class):
    p1 = np.exp(logit) / np.sum(np.exp(logit), 1)[:, None]
    logit = logit / t
    p0 = np.exp(logit) / np.sum(np.exp(logit), 1)[:, None]
    p2 = np.ones_like(p0) / n_class

    bnds_w = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0),)

    def my_constraint_fun(x):
        return np.sum(x) - 1

    constraints = {"type": "eq", "fun": my_constraint_fun, }
    if loss == 'ce':
        w = optimize.minimize(ll_w, (1.0, 0.0, 0.0), args=(p0, p1, p2, label), method='SLSQP', constraints=constraints,
                              bounds=bnds_w, tol=1e-12, options={'disp': False})
    if loss == 'mse':
        w = optimize.minimize(mse_w, (1.0, 0.0, 0.0), args=(p0, p1, p2, label), method='SLSQP', constraints=constraints,
                              bounds=bnds_w, tol=1e-12, options={'disp': False})
    w = w.x
    return w


"""
Calibration: 
Input: uncalibrated logits, temperature (and weight)
Output: calibrated prediction probabilities
"""


##### Calibration: Temperature Scaling with MSE
def ts_calibrate(logit, label, logit_eval, loss):
    t = temperature_scaling(logit, label, loss)
    print("temperature = " + str(t))
    logit_eval = logit_eval / t
    p = np.exp(logit_eval) / np.sum(np.exp(logit_eval), 1)[:, None]
    return p


##### Calibration: Ensemble Temperature Scaling
def ets_calibrate(logit, label, n_class, loss='mse'):
    t = temperature_scaling(logit, label, loss)  # loss can change to 'ce'
    #print("temperature = " + str(t))
    w = ensemble_scaling(logit, label, loss, t, n_class)
    #print("weight = " + str(w))

    return t, w
    """
    p1 = np.exp(logit_eval) / np.sum(np.exp(logit_eval), 1)[:, None]
    logit_eval = logit_eval / t
    p0 = np.exp(logit_eval) / np.sum(np.exp(logit_eval), 1)[:, None]
    p2 = np.ones_like(p0) / n_class
    p = w[0] * p0 + w[1] * p1 + w[2] * p2
    return p
    """

##### Calibration: Isotonic Regression (Multi-class)
def mir_calibrate(logit, label, logit_eval):
    p = np.exp(logit) / np.sum(np.exp(logit), 1)[:, None]
    p_eval = np.exp(logit_eval) / np.sum(np.exp(logit_eval), 1)[:, None]
    ir = IsotonicRegression(out_of_bounds='clip')
    y_ = ir.fit_transform(p.flatten(), (label.flatten()))
    yt_ = ir.predict(p_eval.flatten())

    p = yt_.reshape(logit_eval.shape) + 1e-9 * p_eval
    return p


def irova_calibrate(logit, label, logit_eval):
    p = np.exp(logit) / np.sum(np.exp(logit), 1)[:, None]
    p_eval = np.exp(logit_eval) / np.sum(np.exp(logit_eval), 1)[:, None]

    for ii in range(p_eval.shape[1]):
        ir = IsotonicRegression(out_of_bounds='clip')
        y_ = ir.fit_transform(p[:, ii], label[:, ii])
        p_eval[:, ii] = ir.predict(p_eval[:, ii]) + 1e-9 * p_eval[:, ii]
    return p_eval
    return p_eval