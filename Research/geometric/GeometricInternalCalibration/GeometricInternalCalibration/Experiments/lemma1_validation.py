import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Dict

class Lemma1Validator:
    """
    Validates Lemma 1: k-NN Structure Preservation Under Corruption
    Measures neighborhood overlap between clean and corrupted versions of the same images.
    """
    def __init__(self, k_values=[1, 3, 5]):
        self.k_values = k_values
        self.max_k = max(k_values)

    def measure_knn_overlap(self, 
                           clean_features: np.ndarray,
                           corrupted_features: np.ndarray,
                           reference_features: np.ndarray,
                           k: int) -> Dict:
        """
        Measure k-NN overlap between clean and corrupted versions.
        Args:
            clean_features: Features of clean images [N, D]
            corrupted_features: Features of corrupted versions of same images [N, D]
            reference_features: Full feature dataset for k-NN search [M, D]
            k: Number of neighbors
        Returns:
            Dict with overlap statistics
        """
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nbrs.fit(reference_features)
        overlaps = []
        preservation_rates = []
        for i in range(len(clean_features)):
            clean_dists, clean_indices = nbrs.kneighbors(
                clean_features[i:i+1], n_neighbors=k+1)
            clean_neighbors = set(clean_indices[0][1:])  # Exclude self
            corrupted_dists, corrupted_indices = nbrs.kneighbors(
                corrupted_features[i:i+1], n_neighbors=k+1)
            corrupted_neighbors = set(corrupted_indices[0][1:])
            overlap = len(clean_neighbors.intersection(corrupted_neighbors))
            preservation_rate = overlap / k
            overlaps.append(overlap)
            preservation_rates.append(preservation_rate)
        return {
            'mean_overlap': float(np.mean(overlaps)),
            'std_overlap': float(np.std(overlaps)),
            'mean_preservation_rate': float(np.mean(preservation_rates)),
            'std_preservation_rate': float(np.std(preservation_rates)),
            'min_overlap': int(np.min(overlaps)),
            'max_overlap': int(np.max(overlaps)),
            'overlaps': overlaps,
            'preservation_rates': preservation_rates
        }

    def compute_inter_class_distances(self, 
                                    features: np.ndarray, 
                                    labels: np.ndarray) -> Dict:
        """
        Compute minimum inter-class distance δ_min^(l).
        Args:
            features: Feature representations [N, D]
            labels: Class labels [N]
        Returns:
            Dict with inter-class distance statistics
        """
        features_np = features
        labels_np = labels
        unique_classes = np.unique(labels_np)
        centroids = {}
        for class_id in unique_classes:
            class_mask = labels_np == class_id
            class_features = features_np[class_mask]
            centroids[class_id] = np.mean(class_features, axis=0)
        inter_class_distances = []
        for i, class1 in enumerate(unique_classes):
            for j, class2 in enumerate(unique_classes):
                if i < j:
                    dist = np.linalg.norm(centroids[class1] - centroids[class2])
                    inter_class_distances.append(dist)
        return {
            'min_inter_class_distance': float(np.min(inter_class_distances)),
            'mean_inter_class_distance': float(np.mean(inter_class_distances)),
            'std_inter_class_distance': float(np.std(inter_class_distances)),
            'all_distances': inter_class_distances,
            'centroids': centroids
        }

    def validate_lemma1_condition(self,
                                 lipschitz_estimate: float,
                                 corruption_magnitude: float,
                                 min_inter_class_distance: float) -> Dict:
        """
        Check if Lemma 1 condition is satisfied:
        L_l · ||x - c(x)||_2 < δ_min^(l) / 2
        """
        left_side = lipschitz_estimate * corruption_magnitude
        right_side = min_inter_class_distance / 2
        condition_satisfied = left_side < right_side
        safety_margin = right_side - left_side
        return {
            'condition_satisfied': bool(condition_satisfied),
            'left_side': float(left_side),
            'right_side': float(right_side),
            'safety_margin': float(safety_margin),
            'margin_ratio': float(safety_margin / right_side) if right_side > 0 else 0
        }

    def analyze_corruption_impact(self,
                                clean_features: np.ndarray,
                                corrupted_features: np.ndarray,
                                reference_features: np.ndarray,
                                clean_labels: np.ndarray,
                                corruption_type: str,
                                severity: int,
                                lipschitz_estimate: float) -> Dict:
        """
        Complete Lemma 1 analysis for a corruption type/severity.
        """
        feature_perturbation = np.linalg.norm(clean_features - corrupted_features, axis=1)
        mean_feature_perturbation = float(feature_perturbation.mean())
        inter_class_stats = self.compute_inter_class_distances(reference_features, clean_labels)
        condition_check = self.validate_lemma1_condition(
            lipschitz_estimate,
            mean_feature_perturbation,
            inter_class_stats['min_inter_class_distance']
        )
        knn_results = {}
        for k in self.k_values:
            knn_results[f'k{k}'] = self.measure_knn_overlap(
                clean_features, corrupted_features, reference_features, k
            )
        return {
            'corruption_type': corruption_type,
            'severity': severity,
            'feature_perturbation_stats': {
                'mean': mean_feature_perturbation,
                'std': float(feature_perturbation.std()),
                'min': float(feature_perturbation.min()),
                'max': float(feature_perturbation.max())
            },
            'inter_class_stats': inter_class_stats,
            'lemma1_condition': condition_check,
            'knn_overlap_results': knn_results,
            'lipschitz_estimate': float(lipschitz_estimate)
        } 