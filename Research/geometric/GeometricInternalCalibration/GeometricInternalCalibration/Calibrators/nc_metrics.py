"""
Neural Collapse Metrics (NC1 and NC4) for Layer Selection
Based on PSC paper (2025) - Probabilistic Skip Connections

NC1: Within-class variability (sensitivity proxy)
NC4: Nearest Centroid Classifier accuracy (smoothness proxy)

Implementation based on standard neural collapse formulation.
"""

import numpy as np
import torch
import logging
from typing import Optional

from utils.logging_config import get_logger
logger = get_logger(__name__)


class MetricCalculator:
    """Base class for metric calculators"""
    @property
    def name(self) -> str:
        raise NotImplementedError


class NC1CollapseMetricCalculator(MetricCalculator):
    """
    NC1: Within-class variability metric (Neural Collapse metric 1)
    
    Computes: NC1 = Tr(Σ_W) / Tr(Σ_T)
    - Σ_W = average of per-class covariance matrices (unweighted)
    - Σ_T = total covariance matrix (centered on mean of class means)
    
    Interpretation:
    - NC1 ≈ 0: High collapse (features from same class map to identical points) - BAD
    - NC1 > 0.2: Features retain variability - GOOD for geometric calibration
    
    PSC paper uses ε = 0.2 as the threshold (NC1 > 0.2 indicates sufficient sensitivity).
    
    Direction: MAXIMIZE (higher is better, filter for NC1 > 0.2)
    """
    
    @property
    def name(self) -> str:
        return "nc1_collapse_metric"
    
    def compute(self, features: torch.Tensor, labels: torch.Tensor,
                corrupted_features: Optional[torch.Tensor] = None,
                predictions: Optional[np.ndarray] = None,
                confidences: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            
            device = features.device
            unique_labels = torch.unique(labels)
            C = len(unique_labels)
            N_total = len(labels)
            
            if C < 2 or N_total < C:
                return 0.0
            
            mu_c_list = []
            Sigma_W_c_list = []
            
            # Compute per-class statistics
            for c in unique_labels:
                class_features = features[labels == c]
                n_c = len(class_features)
                if n_c == 0:
                    continue
                
                # Class mean
                mu_c = class_features.mean(dim=0)
                mu_c_list.append(mu_c)
                
                # Within-class covariance: (1/n_c) * (H - mu_c).T @ (H - mu_c)
                diffs = class_features - mu_c
                Sigma_W_c = (diffs.T @ diffs) / n_c
                Sigma_W_c_list.append(Sigma_W_c)
            
            if not mu_c_list or not Sigma_W_c_list:
                return 0.0
            
            # Σ_W = simple average of all class covariance matrices (unweighted)
            Sigma_W = torch.stack(Sigma_W_c_list).mean(dim=0)
            
            # mu_G = mean of class means (not weighted by class size)
            mu_G = torch.stack(mu_c_list).mean(dim=0)
            
            # Σ_T = total covariance matrix
            diffs_G = features - mu_G
            Sigma_T = (diffs_G.T @ diffs_G) / N_total
            
            # Compute trace ratio
            Tr_W = torch.trace(Sigma_W)
            Tr_T = torch.trace(Sigma_T)
            
            if Tr_T < 1e-12:
                return 0.0
            
            nc1 = Tr_W / Tr_T
            return float(nc1.cpu().item())
            
        except Exception as e:
            logger.warning(f"Failed to compute {self.name}: {e}")
            return 0.0


class NC4SeparabilityCalculator(MetricCalculator):
    """
    NC4: Nearest Centroid Classifier (NCC) accuracy (Neural Collapse metric 4)
    
    Computes accuracy of NCC: y_hat = argmin_c ||h(x) - μ_c||
    
    Interpretation:
    - LOW NC4: Poor class separation - classes are not well-separated in feature space
    - HIGH NC4: Good class separation - good smoothness for geometric calibration
    
    PSC paper: Maximize NC4 subject to NC1 > 0.2 (NO fixed NC4 threshold stated)
    
    Direction: MAXIMIZE (higher is better, indicates better class separability)
    """
    
    @property
    def name(self) -> str:
        return "nc4_ncc_accuracy"
    
    def compute(self, features: torch.Tensor, labels: torch.Tensor,
                corrupted_features: Optional[torch.Tensor] = None,
                predictions: Optional[np.ndarray] = None,
                confidences: Optional[np.ndarray] = None, **kwargs) -> float:
        try:
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            
            device = features.device
            unique_labels = torch.unique(labels)
            C = len(unique_labels)
            
            if C < 2 or len(labels) < C:
                return 0.0
            
            # Calculate all class centroids
            mu_c_list = []
            for c in unique_labels:
                class_features = features[labels == c]
                if len(class_features) == 0:
                    # This class has no samples, add zero-vector placeholder
                    mu_c_list.append(torch.zeros(features.shape[1], device=device))
                else:
                    mu_c_list.append(class_features.mean(dim=0))
            
            mus = torch.stack(mu_c_list)  # Shape [C, D_feat]
            
            # Compute all pairwise distances efficiently using vectorized operations
            # dists[i, j] = distance from features[i] to mus[j]
            dists = torch.cdist(features, mus)  # Shape [N, C]
            
            # Find the index of the closest centroid for each sample
            preds_idx = torch.argmin(dists, dim=1)  # Shape [N]
            
            # Map indices back to class labels
            preds_class = unique_labels[preds_idx]
            
            # Calculate NCC accuracy
            acc = (preds_class == labels).float().mean()
            return float(acc.cpu().item())
            
        except Exception as e:
            logger.warning(f"Failed to compute {self.name}: {e}")
            return 0.0