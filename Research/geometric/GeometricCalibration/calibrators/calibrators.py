from torch import nn, optim
import calibration as SBC
from netcal.scaling import BetaCalibration
from netcal.binning import BBQ
import torch
from utils.utils import *
from utils.binning_utils import get_uniform_mass_bins, bin_points, nudge, get_bin, histogramBinning
from utils.temperature_utils import temperature_scaling
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG if os.getenv('ENV') == 'dev' else logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class identity:
    def predict_proba(self, x):
        return x

    def predict(self, x):
        return np.argmax(x, axis=1)

class BaseCalibrator:
    """ Abstract base class for all calibrators. """

    def __init__(self, n_classes=None, bins=15, temperature=1.0):
        self.n_classes = n_classes
        self.bins = bins
        self.temperature = temperature
        logger.info(f"Initialized {self.__class__.__name__} with n_classes={n_classes}, bins={bins}, temperature={temperature}")

    def fit(self, logits, y):
        raise NotImplementedError("Subclasses must implement 'fit' method.")

    def calibrate(self, probs):
        raise NotImplementedError("Subclasses must implement 'calibrate' method.")


class HB_binary(object):
    def __init__(self, n_bins=15):
        self.delta = 1e-10
        self.n_bins = n_bins
        self.bin_upper_edges = None
        self.mean_pred_values = None
        self.num_calibration_examples_in_bin = None
        self.fitted = False

    def fit(self, y_score, y):
        y_score = nudge(y_score, self.delta)
        self.bin_upper_edges = get_uniform_mass_bins(y_score, self.n_bins)
        bin_assignment = bin_points(y_score, self.bin_upper_edges)
        self.num_calibration_examples_in_bin = np.zeros([self.n_bins, 1])
        self.mean_pred_values = np.empty(self.n_bins)

        for i in range(self.n_bins):
            bin_idx = (bin_assignment == i)
            self.num_calibration_examples_in_bin[i] = sum(bin_idx)
            if sum(bin_idx) > 0:
                self.mean_pred_values[i] = nudge(y[bin_idx].mean(), self.delta)
            else:
                self.mean_pred_values[i] = nudge(0.5, self.delta)

        self.fitted = True

    def predict_proba(self, y_score):
        assert self.fitted is True, "Call HB_binary.fit() first"
        y_score = nudge(y_score, self.delta)
        y_bins = bin_points(y_score, self.bin_upper_edges)
        y_pred_prob = self.mean_pred_values[y_bins]
        return y_pred_prob

class HB_toplabel(object):
    def __init__(self, points_per_bin=50):
        ### Hyperparameters
        self.points_per_bin = points_per_bin

        ### Parameters to be learnt 
        self.hb_binary_list = []
        
        ### Internal variables
        self.num_classes = None
    
    def fit(self, pred_mat, y):
        y = y + 1  # Convert labels to 1-based indexing

        assert(self.points_per_bin is not None), "Points per bins has to be specified"
        assert(np.size(pred_mat.shape) == 2), "Prediction matrix should be 2 dimensional"
        y = y.squeeze()
        assert(pred_mat.shape[0] == y.size), "Check dimensions of input matrices"
        self.num_classes = pred_mat.shape[1]
        assert(np.min(y) >= 1 and np.max(y) <= self.num_classes), "Labels should be numbered 1 ... L, where L is the number of columns in the prediction matrix"
        
        top_score = np.max(pred_mat, axis=1).squeeze()
        # top_score = np.max(pred_mat, axis=1) if pred_mat.ndim == 2 else pred_mat
        pred_class = (np.argmax(pred_mat, axis=1)+1).squeeze()

        for l in range(1, self.num_classes+1, 1):
            pred_l_indices = np.where(pred_class == l)
            n_l = np.size(pred_l_indices)

            bins_l = np.floor(n_l/self.points_per_bin).astype('int')
            if(bins_l == 0):
               self.hb_binary_list += [identity()]
               # print("Predictions for class {:d} not recalibrated since fewer than {:d} calibration points were predicted as class {:d}.".format(l, self.points_per_bin, l))
            else:
                hb = HB_binary(n_bins = bins_l)
                hb.fit(top_score[pred_l_indices], y[pred_l_indices] == l)
                self.hb_binary_list += [hb]
        
        # top-label histogram binning done
        self.fitted = True

    def predict_proba(self, pred_mat):
        assert(self.fitted is True), "Call HB_binary.fit() first"
        assert(np.size(pred_mat.shape) == 2), "Prediction matrix should be 2 dimensional"
        assert(self.num_classes == pred_mat.shape[1]), "Number of columns of prediction matrix do not match number of labels"
        
        top_score = np.max(pred_mat, axis=1).squeeze()
        pred_class = (np.argmax(pred_mat, axis=1)+1).squeeze()

        n = pred_class.size
        pred_top_score = np.zeros((n))
        for i in range(n):
            pred_top_score[i] = self.hb_binary_list[pred_class[i]-1].predict_proba(top_score[i])

        return pred_top_score

    def fit_top(self, top_score, pred_class, y):
        assert(self.points_per_bin is not None), "Points per bins has to be specified"

        top_score = top_score.squeeze()
        pred_class = pred_class.squeeze()
        y = y.squeeze()

        assert(min(np.min(y), np.min(pred_class)) >= 1), "Labels should be numbered 1 ... L, use HB_binary for a binary problem"
        assert(top_score.size == y.size), "Check dimensions of input matrices"
        assert(pred_class.size == y.size), "Check dimensions of input matrices"
        assert(y.size >= self.n_bins), "Number of bins should be less than the number of calibration points"

        self.num_classes = max(np.max(y), np.max(pred_class))
        
        for l in range(1, self.num_classes+1, 1):
            pred_l_indices = np.where(pred_class == l)
            n_l = np.size(pred_l_indices)

            bins_l = np.floor(n_l/self.points_per_bin).astype('int')
            if(bins_l == 0):
               self.hb_binary_list += [identity()]
               # print("Predictions for class {:d} not recalibrated since fewer than {:d} calibration points were predicted as class {:d}".format(self.points_per_bin, l))
            else:
                hb = HB_binary(n_bins = bins_l)
                hb.fit(top_score[pred_l_indices], y[pred_l_indices] == l)
                self.hb_binary_list += [hb]
        
        # top-label histogram binning done
        self.fitted = True

    def predict_proba_top(self, top_score, pred_class):
        assert(self.fitted is True), "Call HB_binary.fit() first"
        top_score = top_score.squeeze()
        pred_class = pred_class.squeeze()
        assert(top_score.size == pred_class.size), "Check dimensions of input matrices"
        assert(np.min(pred_class) >= 1 and np.min(pred_class) <= self.num_classes), "Some of the predicted labels are not in the range of labels seen while calibrating"
        n = pred_class.size
        pred_top_score = np.zeros((n))
        for i in range(n):
            pred_top_score[i] = self.hb_binary_list[pred_class[i]-1].predict_proba(top_score[i])

        return pred_top_score


class IsotonicCalibrator(BaseCalibrator):
    def fit(self, y_prob_val, y_true):
        logger.info("Fitting Isotonic Calibrator.")
        self.Isotonic = IsotonicRegression(out_of_bounds='clip').fit(y_prob_val, y_true)
        logger.info("Isotonic Calibrator fitted successfully.")

    def calibrate(self, probs):
        logger.info("Calibrating probabilities using Isotonic Regression.")
        return self.Isotonic.predict(np.max(probs, axis=1))


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


# class GeometricCalibrator(BaseCalibrator):
#     """
#     Class serving as a wrapper for the geometric calibration method (stability/separation).
#     """

#     def __init__(self, model, X_train, y_train, fitting_func=None, compression_mode=None, compression_param=None,
#                  metric='l2'):
#         """
#         Initializes the GeometricCalibrator with a model, compression, and metric.

#         Args:
#             model: The model to be calibrated (with `predict` and `predict_proba` methods).
#             X_train: Training data (flattened images).
#             y_train: Training labels.
#             fitting_func: Custom fitting function (default: IsotonicRegression).
#             compression_mode: Compression mode for data.
#             compression_param: Parameter controlling the compression level.
#             metric: Distance metric for stability/separation calculations.
#         """
#         super().__init__()
#         self.model = model
#         self.popt = None
#         self._fitted = False

#         # Default to IsotonicRegression if no custom fitting function is provided
#         self.fitting_func = fitting_func if fitting_func else IsotonicRegression(out_of_bounds="clip")

#         # Compression setup
#         self.stab_space = Stability_space(X_train, y_train,
#                                           compression=Compression(compression_mode, compression_param), metric=metric)

#         logger.info(
#             f"Initialized {self.__class__.__name__} with model {self.model.__class__.__name__} and fitting function {self.fitting_func.__class__.__name__}.")

#     def fit(self, X_val, y_val):
#         """
#         Fits the calibrator with the validation data.

#         Args:
#             X_val: Validation data (flattened images).
#             y_val: Validation labels.
#         """
#         logger.info(f"{self.__class__.__name__}: Fitting with validation data.")

#         try:
#             # Step 1: Predict on validation data
#             y_pred_val = self.model.predict(X_val)

#             # Step 2: Compute stability values based on predictions
#             stability_val = self.stab_space.calc_stab(X_val, y_pred_val)

#             # Step 3: Fit the provided fitting function (e.g., IsotonicRegression) on the stability values
#             correct = y_val == y_pred_val
#             self.popt = self.fitting_func.fit(stability_val, correct)

#             self._fitted = True
#             logger.info(f"{self.__class__.__name__}: Successfully fitted using {self.fitting_func.__class__.__name__}.")

#         except Exception as e:
#             logger.error(f"{self.__class__.__name__}: Failed to fit with error: {e}")
#             raise

#     def calibrate(self, X_test):
#         """
#         Calibrates the test data based on the fitted model.

#         Args:
#             X_test: Test data (flattened images).

#         Returns:
#             Calibrated probability vector for each image.
#         """
#         if not self._fitted:
#             raise ValueError("You must fit the calibrator before using it.")

#         logger.info(f"{self.__class__.__name__}: Calibrating test data.")

#         try:
#             # Predict on the test data using the trained model
#             y_test_pred = self.model.predict(X_test)

#             # Compute stability values for the test data
#             stability_test = self.stab_space.calc_stab(X_test, y_test_pred)

#             # Apply the fitted calibration method
#             calibrated_probs = self.popt.predict(stability_test)
#             logger.info(f"{self.__class__.__name__}: Calibration successful.")

#             return calibrated_probs

#         except Exception as e:
#             logger.error(f"{self.__class__.__name__}: Calibration failed with error: {e}")
#             raise

class PlattCalibrator(BaseCalibrator):
    def fit(self, y_prob_val, y_true):
        from sklearn.linear_model import LogisticRegression
        
        n_classes = y_prob_val.shape[1]
        self.calibrators = []
        
        for i in range(n_classes):
            binary_labels = (y_true == i).astype(int)
            class_scores = y_prob_val[:, i]
            
            if np.sum(binary_labels) > 0 and np.sum(1 - binary_labels) > 0:
                lr = LogisticRegression(random_state=42, max_iter=1000)
                lr.fit(class_scores.reshape(-1, 1), binary_labels)
                self.calibrators.append(lr)
            else:
                self.calibrators.append(None)
    
    def calibrate(self, probs):
        calibrated_probs = np.zeros_like(probs)
        
        for i in range(probs.shape[1]):
            if self.calibrators[i] is not None:
                class_scores = probs[:, i]
                calibrated_probs[:, i] = self.calibrators[i].predict_proba(
                    class_scores.reshape(-1, 1))[:, 1]
            else:
                calibrated_probs[:, i] = probs[:, i]
        
        return calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)

class TSCalibrator(BaseCalibrator):
    """ Maximum likelihood temperature scaling (Guo et al., 2017) """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.loss_trace = None
        logger.info(f"Initialized TSCalibrator with temperature={temperature}")

    def fit(self, logits, y):
        """ Fits temperature scaling using hard labels. """
        logger.info("TSCalibrator: Starting the fitting process.")
        try:
            self.n_classes = logits.shape[1]
            _model_logits = torch.from_numpy(logits)
            _y = torch.from_numpy(y)
            _temperature = torch.tensor(self.temperature, requires_grad=True)

            nll = nn.CrossEntropyLoss()
            optimizer = optim.Adam([_temperature], lr=0.05)
            loss_trace = []

            logger.info("TSCalibrator: Beginning optimization.")
            for step in range(7500):
                optimizer.zero_grad()
                loss = nll(_model_logits / _temperature, _y)
                loss.backward()
                optimizer.step()
                loss_trace.append(loss.item())
                with torch.no_grad():
                    _temperature.clamp_(min=1e-2, max=1e4)

                if np.abs(_temperature.grad) < 1e-3:
                    logger.info("TSCalibrator: Convergence reached.")
                    break

            self.loss_trace = loss_trace
            self.temperature = _temperature.item()
            logger.info(f"TSCalibrator: Fitting complete. Final temperature: {self.temperature}")
        except Exception as e:
            logger.error(f"TSCalibrator: Fitting failed with error: {e}")
            raise

    def calibrate(self, probs):
        """ Apply temperature scaling to probabilities. """
        logger.info("TSCalibrator: Calibrating probabilities.")
        try:
            calibrated_probs = probs ** (1. / self.temperature)
            calibrated_probs /= np.sum(calibrated_probs, axis=1, keepdims=True)
            return calibrated_probs
        except Exception as e:
            logger.error(f"TSCalibrator: Calibration failed with error: {e}")
            raise


class EnsembleTSCalibrator(BaseCalibrator):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.weights = None
        logger.info(f"Initialized EnsembleTSCalibrator with temperature={temperature}")

    def fit(self, logits, y):
        logger.info("EnsembleTSCalibrator: Fitting model.")
        try:
            self.n_classes = logits.shape[1]
            _y = np.eye(self.n_classes)[y]
            self.temperature = temperature_scaling(logits, _y, 'mse')
            logger.info(f"EnsembleTSCalibrator: Successfully fitted the model with temperature: {self.temperature}")
        except Exception as e:
            logger.error(f"EnsembleTSCalibrator: Fitting failed with error: {e}")
            raise

    def calibrate(self, probs):
        logger.info("EnsembleTSCalibrator: Calibrating probabilities.")
        try:
            tempered_probs = probs ** (1. / self.temperature)
            tempered_probs /= np.sum(tempered_probs, axis=1, keepdims=True)
            return tempered_probs
        except Exception as e:
            logger.error(f"EnsembleTSCalibrator: Calibration failed with error: {e}")
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


class SeparationHistogramBinningCalibrator(StabilityHistogramBinningCalibrator):
    pass


