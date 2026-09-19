from .base_calibrator import BaseCalibrator
from sklearn.neighbors import KDTree
import numpy as np
import logging
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class TrustScoreCalibrator:
    """Trust Score calibration method based on Google's implementation."""

    def __init__(self, k=10, alpha=0.0, filtering="density", min_dist=1e-12):
        """
        Args:
            k: Number of nearest neighbors for density estimation.
            alpha: Filtering threshold (0.0 means no filtering).
            filtering: Filtering method ("none", "density", or "uncertainty").
            min_dist: Small constant to avoid division by zero.
        """
        self.k = k
        self.alpha = alpha
        self.filtering = filtering
        self.min_dist = min_dist
        self.kdtrees = None
        self.n_labels = None

    def filter_by_density(self, X):
        """Filter out points with low kNN density."""
        logger.debug(f"Starting density filtering on {X.shape[0]} samples.")
        kdtree = KDTree(X)
        knn_radii = kdtree.query(X, k=self.k)[0][:, -1]
        eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
        filtered_X = X[np.where(knn_radii <= eps)[0], :]
        logger.debug(f"Filtered {X.shape[0] - filtered_X.shape[0]} samples. {filtered_X.shape[0]} samples remain.")
        return filtered_X

    def fit(self, X, y):
        """Initialize trust score computation with training data."""
        logger.info("Fitting Trust Score Calibrator.")
        self.n_labels = np.max(y) + 1
        logger.debug(f"Number of labels detected: {self.n_labels}")
        self.kdtrees = [None] * self.n_labels

        for label in range(self.n_labels):
            X_label = X[np.where(y == label)[0]]
            logger.debug(f"Processing label {label} with {X_label.shape[0]} samples.")
            if self.filtering == "density":
                X_label = self.filter_by_density(X_label)
                logger.debug(f"After filtering, label {label} has {X_label.shape[0]} samples.")
            self.kdtrees[label] = KDTree(X_label)

            if len(X_label) == 0:
                logger.warning(
                    f"No data points for label {label} after filtering. Lower alpha or check data."
                )

        logger.info("Trust Score Calibrator fitted successfully.")

    def calibrate(self, X, predictions):
        """Compute calibrated probabilities using trust scores.
        
        Args:
            X: Test data points.
            predictions: Predicted class labels for the test data.

        Returns:
            Calibrated probabilities as a 2D array.
        """
        logger.info("Calibrating probabilities using Trust Score.")
        logger.debug(f"Test set has {X.shape[0]} samples.")
        distances = np.zeros((X.shape[0], self.n_labels))
        for label in range(self.n_labels):
            logger.debug(f"Querying KDTree for label {label}.")
            distances[:, label] = self.kdtrees[label].query(X, k=2)[0][:, -1]

        logger.debug("Distances matrix computed.")
        logger.debug(f"Distances shape: {distances.shape}")

        # Compute trust scores
        sorted_distances = np.sort(distances, axis=1)
        d_to_pred = distances[np.arange(X.shape[0]), predictions]
        logger.debug(f"d_to_pred calculated: {d_to_pred[:5]}")  # Log first 5 values for debugging
        d_to_closest_not_pred = np.where(
            sorted_distances[:, 0] != d_to_pred,
            sorted_distances[:, 0],
            sorted_distances[:, 1]
        )
        logger.debug(f"d_to_closest_not_pred calculated: {d_to_closest_not_pred[:5]}")  # Log first 5 values for debugging
        trust_scores = d_to_closest_not_pred / (d_to_pred + self.min_dist)
        logger.debug(f"Trust scores calculated: {trust_scores[:5]}")  # Log first 5 values for debugging

        # Convert trust scores to calibrated probabilities
        calibrated_probs = trust_scores / (1 + trust_scores)  # Scale to [0, 1]
        logger.debug(f"Calibrated probabilities calculated: {calibrated_probs[:5]}")  # Log first 5 values for debugging

        # Create probability distribution across classes
        output_probs = np.zeros((X.shape[0], self.n_labels))
        output_probs[np.arange(X.shape[0]), predictions] = calibrated_probs
        logger.info("Calibration completed.")
        return output_probs

