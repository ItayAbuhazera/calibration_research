import numpy as np
# from utils.metrics import ECE_calc, MCE_calc, ACE_calc, Brier_score_calc, log_loss_calc, binned_likelihood_calc
import logging

class BaseCalibrator:
    """ Abstract base class for all calibrators. """
    def __init__(self, n_classes=None, bins=15, temperature=1.0):
        self.n_classes = n_classes
        self.bins = bins
        self.temperature = temperature
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__} with n_classes={n_classes}, bins={bins}, temperature={temperature}")

    def fit(self, logits, y):
        raise NotImplementedError("Subclasses must implement 'fit' method.")

    def calibrate(self, probs):
        raise NotImplementedError("Subclasses must implement 'calibrate' method.")

    # def compute_metric(self, metric_func, probs_precalibrated, *args):
    #     """ General method for computing a given calibration error metric. """
    #     try:
    #         probs = self.calibrate(probs_precalibrated)
    #         self.logger.info(f"Successfully calibrated probabilities with {self.__class__.__name__}.")
    #         return metric_func(probs, *args)
    #     except Exception as e:
    #         self.logger.error(f"Failed to compute {metric_func.__name__}: {e}")
    #         raise e

    # def ECE(self, probs_precalibrated, y_pred, y_real):
    #     ''' Expected Calibration Error '''
    #     self.logger.info("Calculating Expected Calibration Error (ECE).")
    #     return self.compute_metric(ECE_calc, probs_precalibrated, y_pred, y_real, self.bins)

    # def MCE(self, probs_precalibrated, y_pred, y_real):
    #     ''' Maximum Calibration Error '''
    #     self.logger.info("Calculating Maximum Calibration Error (MCE).")
    #     return self.compute_metric(MCE_calc, probs_precalibrated, y_pred, y_real, self.bins)

    # def ACE(self, probs_precalibrated, y_pred, y_real):
    #     ''' Adaptive Calibration Error '''
    #     self.logger.info("Calculating Adaptive Calibration Error (ACE).")
    #     return self.compute_metric(ACE_calc, probs_precalibrated, y_pred, y_real, self.bins)

    # def Brier_score(self, probs_precalibrated, y_real):
    #     ''' Brier Score '''
    #     self.logger.info("Calculating Brier Score.")
    #     return self.compute_metric(Brier_score_calc, probs_precalibrated, y_real)

    # def Log_Loss(self, probs_precalibrated, y_real):
    #     ''' Log Loss '''
    #     self.logger.info("Calculating Logarithmic Loss (Log Loss).")
    #     return self.compute_metric(log_loss_calc, probs_precalibrated, y_real)

    # def Binned_Likelihood(self, probs_precalibrated, y_real, bins=15):
    #     """ Calculate Binned Likelihood (Binned NLL). """
    #     self.logger.info("Calculating Binned Likelihood (Binned NLL).")
    #     return self.compute_metric(binned_likelihood_calc, probs_precalibrated, y_real, bins)
