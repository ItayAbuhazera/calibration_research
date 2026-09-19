import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
import json

from utils.logging_config import get_logger
logger = get_logger(__name__)

def convert_numpy_types_comprehensive(obj):
    """
    Comprehensive numpy type converter for JSON serialization.
    Handles all numpy types including nested structures.
    """
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, (np.integer, np.floating, np.bool_)) else k: 
                convert_numpy_types_comprehensive(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types_comprehensive(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    else:
        return obj

class EnhancedLemma1Validator:
    """
    Enhanced Lemma 1 Validator: k-NN Structure Preservation Under Corruption
    
    Validates the theoretical bound:
    |N_k^(l)(x) ∩ N_k^(l)(c(x))| ≥ k - O(L_l · ||x - c(x)||_2 / δ_min^(l))
    
    Where:
    - N_k^(l)(x) = k-nearest neighbors of x in layer l feature space
    - c(x) = corrupted version of x
    - L_l = Lipschitz constant at layer l
    - δ_min^(l) = minimum inter-class distance at layer l
    """
    
    def __init__(self, k_values=[1, 3, 5]):
        self.k_values = k_values
        self.max_k = max(k_values)

    def measure_knn_overlap_detailed(self, 
                                   clean_features: np.ndarray,
                                   corrupted_features: np.ndarray,
                                   reference_features: np.ndarray,
                                   k: int) -> Dict:
        """
        Detailed k-NN overlap measurement with per-sample analysis.
        
        Returns both aggregate statistics and per-sample detailed results
        for theoretical validation.
        """
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nbrs.fit(reference_features)
        
        per_sample_results = []
        overlaps = []
        preservation_rates = []
        
        for i in range(len(clean_features)):
            # Find k-NN for clean image
            clean_dists, clean_indices = nbrs.kneighbors(
                clean_features[i:i+1], n_neighbors=k+1)
            clean_neighbors = set(clean_indices[0][1:])  # Exclude self
            
            # Find k-NN for corrupted image  
            corrupted_dists, corrupted_indices = nbrs.kneighbors(
                corrupted_features[i:i+1], n_neighbors=k+1)
            corrupted_neighbors = set(corrupted_indices[0][1:])
            
            # Compute overlap
            overlap = len(clean_neighbors.intersection(corrupted_neighbors))
            preservation_rate = overlap / k
            
            overlaps.append(overlap)
            preservation_rates.append(preservation_rate)
            
            # Store detailed per-sample info for theoretical validation
            per_sample_results.append({
                'sample_idx': int(i),  # Convert to native int
                'clean_neighbors': [int(x) for x in clean_neighbors],  # Convert to native int
                'corrupted_neighbors': [int(x) for x in corrupted_neighbors],  # Convert to native int
                'overlap': int(overlap),  # Convert to native int
                'preservation_rate': float(preservation_rate),  # Convert to native float
                'clean_distances': [float(x) for x in clean_dists[0][1:]],  # Convert to native float
                'corrupted_distances': [float(x) for x in corrupted_dists[0][1:]]  # Convert to native float
            })
        
        return {
            'mean_overlap': float(np.mean(overlaps)),
            'std_overlap': float(np.std(overlaps)),
            'mean_preservation_rate': float(np.mean(preservation_rates)),
            'std_preservation_rate': float(np.std(preservation_rates)),
            'min_overlap': int(np.min(overlaps)),
            'max_overlap': int(np.max(overlaps)),
            'overlaps': [int(x) for x in overlaps],  # Convert to native int
            'preservation_rates': [float(x) for x in preservation_rates],  # Convert to native float
            'per_sample_results': per_sample_results
        }

    def compute_theoretical_bound(self,
                                 lipschitz_estimate: float,
                                 feature_perturbations: np.ndarray,
                                 min_inter_class_distance: float,
                                 k: int) -> Dict:
        """
        Compute theoretical bound for k-NN overlap:
        overlap ≥ k - O(L_l · ||x - c(x)||_2 / δ_min^(l))
        
        Args:
            lipschitz_estimate: L_l estimate
            feature_perturbations: ||φ_l(x) - φ_l(c(x))||_2 for each sample
            min_inter_class_distance: δ_min^(l)
            k: Number of neighbors
        
        Returns:
            Dictionary with theoretical predictions and bounds
        """
        # Compute the perturbation ratio term: L_l · ||x - c(x)||_2 / δ_min^(l)
        perturbation_ratios = (lipschitz_estimate * feature_perturbations) / (min_inter_class_distance + 1e-8)
        
        # The O(·) term represents the expected number of neighbors lost
        # We use a conservative estimate: lost_neighbors ≈ C * perturbation_ratio * k
        # where C is an empirical constant (typically 1-3 for well-separated classes)
        C_conservative = 2.0  # Conservative bound
        C_tight = 1.0        # Tighter bound
        
        # Theoretical lower bounds
        expected_lost_neighbors_conservative = np.minimum(k, C_conservative * perturbation_ratios * k)
        expected_lost_neighbors_tight = np.minimum(k, C_tight * perturbation_ratios * k)
        
        theoretical_overlap_conservative = k - expected_lost_neighbors_conservative
        theoretical_overlap_tight = k - expected_lost_neighbors_tight
        
        # Ensure bounds are non-negative
        theoretical_overlap_conservative = np.maximum(0, theoretical_overlap_conservative)
        theoretical_overlap_tight = np.maximum(0, theoretical_overlap_tight)
        
        return {
            'perturbation_ratios': [float(x) for x in perturbation_ratios],
            'expected_lost_neighbors_conservative': [float(x) for x in expected_lost_neighbors_conservative],
            'expected_lost_neighbors_tight': [float(x) for x in expected_lost_neighbors_tight],
            'theoretical_overlap_conservative': [float(x) for x in theoretical_overlap_conservative],
            'theoretical_overlap_tight': [float(x) for x in theoretical_overlap_tight],
            'mean_perturbation_ratio': float(np.mean(perturbation_ratios)),
            'max_perturbation_ratio': float(np.max(perturbation_ratios)),
            'bound_constants': {'C_conservative': float(C_conservative), 'C_tight': float(C_tight)}
        }

    def validate_theoretical_bound(self,
                                  actual_overlaps: List[int],
                                  theoretical_bounds: Dict,
                                  k: int) -> Dict:
        """
        Validate that actual k-NN overlaps satisfy the theoretical bound.
        
        Returns statistical validation of the theoretical lemma.
        """
        actual_overlaps = np.array(actual_overlaps)
        conservative_bounds = np.array(theoretical_bounds['theoretical_overlap_conservative'])
        tight_bounds = np.array(theoretical_bounds['theoretical_overlap_tight'])
        
        # Check how many samples satisfy the bounds
        conservative_violations = int(np.sum(actual_overlaps < conservative_bounds))
        tight_violations = int(np.sum(actual_overlaps < tight_bounds))
        
        total_samples = len(actual_overlaps)
        conservative_success_rate = (total_samples - conservative_violations) / total_samples
        tight_success_rate = (total_samples - tight_violations) / total_samples
        
        # Compute margins (how much better we do than the bound)
        conservative_margins = actual_overlaps - conservative_bounds
        tight_margins = actual_overlaps - tight_bounds
        
        # Statistical analysis
        mean_actual = float(np.mean(actual_overlaps))
        mean_conservative_bound = float(np.mean(conservative_bounds))
        mean_tight_bound = float(np.mean(tight_bounds))
        
        return {
            'validation_summary': {
                'total_samples': int(total_samples),
                'conservative_success_rate': float(conservative_success_rate),
                'tight_success_rate': float(tight_success_rate),
                'conservative_violations': int(conservative_violations),
                'tight_violations': int(tight_violations)
            },
            'margin_analysis': {
                'mean_conservative_margin': float(np.mean(conservative_margins)),
                'std_conservative_margin': float(np.std(conservative_margins)),
                'min_conservative_margin': float(np.min(conservative_margins)),
                'mean_tight_margin': float(np.mean(tight_margins)),
                'std_tight_margin': float(np.std(tight_margins)),
                'min_tight_margin': float(np.min(tight_margins))
            },
            'bound_quality': {
                'mean_actual_overlap': mean_actual,
                'mean_conservative_bound': mean_conservative_bound,
                'mean_tight_bound': mean_tight_bound,
                'conservative_tightness': mean_actual - mean_conservative_bound,
                'tight_tightness': mean_actual - mean_tight_bound
            },
            'detailed_results': {
                'actual_overlaps': [int(x) for x in actual_overlaps],
                'conservative_bounds': [float(x) for x in conservative_bounds],
                'tight_bounds': [float(x) for x in tight_bounds],
                'conservative_margins': [float(x) for x in conservative_margins],
                'tight_margins': [float(x) for x in tight_margins]
            }
        }

    def compute_inter_class_distances(self, 
                                    features: np.ndarray, 
                                    labels: np.ndarray) -> Dict:
        """
        Compute minimum inter-class distance δ_min^(l) and related statistics.
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
        class_pairs = []
        for i, class1 in enumerate(unique_classes):
            for j, class2 in enumerate(unique_classes):
                if i < j:
                    dist = np.linalg.norm(centroids[class1] - centroids[class2])
                    inter_class_distances.append(dist)
                    class_pairs.append((int(class1), int(class2)))  # Convert to native int
        
        # Convert centroids to native types
        centroids_serializable = {}
        for class_id, centroid in centroids.items():
            centroids_serializable[int(class_id)] = [float(x) for x in centroid]
        
        return {
            'min_inter_class_distance': float(np.min(inter_class_distances)),
            'mean_inter_class_distance': float(np.mean(inter_class_distances)),
            'std_inter_class_distance': float(np.std(inter_class_distances)),
            'max_inter_class_distance': float(np.max(inter_class_distances)),
            'all_distances': [float(x) for x in inter_class_distances],
            'class_pairs': class_pairs,
            'centroids': centroids_serializable,
            'num_classes': int(len(unique_classes))
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
            'margin_ratio': float(safety_margin / right_side) if right_side > 0 else 0.0,
            'interpretation': {
                'lipschitz_constant': float(lipschitz_estimate),
                'corruption_magnitude': float(corruption_magnitude),
                'min_inter_class_distance': float(min_inter_class_distance),
                'required_threshold': float(right_side)
            }
        }

    def analyze_corruption_impact_enhanced(self,
                                         clean_features: np.ndarray,
                                         corrupted_features: np.ndarray,
                                         reference_features: np.ndarray,
                                         clean_labels: np.ndarray,
                                         corruption_type: str,
                                         severity: int,
                                         lipschitz_estimate: float) -> Dict:
        """
        Complete enhanced Lemma 1 analysis with theoretical validation.
        """
        # Compute feature perturbations
        feature_perturbation = np.linalg.norm(clean_features - corrupted_features, axis=1)
        mean_feature_perturbation = float(feature_perturbation.mean())
        
        # Compute inter-class distances
        inter_class_stats = self.compute_inter_class_distances(reference_features, clean_labels)
        
        # Check Lemma 1 condition
        condition_check = self.validate_lemma1_condition(
            lipschitz_estimate,
            mean_feature_perturbation,
            inter_class_stats['min_inter_class_distance']
        )
        
        # Detailed k-NN analysis for each k
        knn_results = {}
        theoretical_validation = {}
        
        for k in self.k_values:
            # Measure actual k-NN overlap with detailed results
            knn_detailed = self.measure_knn_overlap_detailed(
                clean_features, corrupted_features, reference_features, k
            )
            
            # Compute theoretical bounds
            theoretical_bounds = self.compute_theoretical_bound(
                lipschitz_estimate,
                feature_perturbation,
                inter_class_stats['min_inter_class_distance'],
                k
            )
            
            # Validate theoretical bound
            bound_validation = self.validate_theoretical_bound(
                knn_detailed['overlaps'],
                theoretical_bounds,
                k
            )
            
            knn_results[f'k{k}'] = knn_detailed
            theoretical_validation[f'k{k}'] = {
                'theoretical_bounds': theoretical_bounds,
                'bound_validation': bound_validation
            }
        
        result = {
            'corruption_type': str(corruption_type),
            'severity': int(severity),
            'feature_perturbation_stats': {
                'mean': mean_feature_perturbation,
                'std': float(feature_perturbation.std()),
                'min': float(feature_perturbation.min()),
                'max': float(feature_perturbation.max()),
                'per_sample': [float(x) for x in feature_perturbation]
            },
            'inter_class_stats': inter_class_stats,
            'lemma1_condition': condition_check,
            'knn_overlap_results': knn_results,
            'theoretical_validation': theoretical_validation,
            'lipschitz_estimate': float(lipschitz_estimate),
            'lemma_proof_summary': self._generate_proof_summary(
                knn_results, theoretical_validation, condition_check
            )
        }
        
        # Final conversion to ensure all numpy types are converted
        return convert_numpy_types_comprehensive(result)

    def _generate_proof_summary(self, knn_results: Dict, theoretical_validation: Dict, condition_check: Dict) -> Dict:
        """
        Generate a summary of the theoretical proof validation.
        """
        summary = {
            'lemma1_condition_satisfied': bool(condition_check['condition_satisfied']),
            'k_specific_validation': {}
        }
        
        for k_str in knn_results.keys():
            k = int(k_str[1:])  # Extract k from 'k3' -> 3
            validation = theoretical_validation[k_str]['bound_validation']
            
            summary['k_specific_validation'][k_str] = {
                'conservative_bound_success_rate': float(validation['validation_summary']['conservative_success_rate']),
                'tight_bound_success_rate': float(validation['validation_summary']['tight_success_rate']),
                'mean_actual_overlap': float(validation['bound_quality']['mean_actual_overlap']),
                'mean_theoretical_lower_bound': float(validation['bound_quality']['mean_conservative_bound']),
                'bound_exceeded': bool(validation['bound_quality']['mean_actual_overlap'] >= validation['bound_quality']['mean_conservative_bound']),
                'average_margin_above_bound': float(validation['margin_analysis']['mean_conservative_margin'])
            }
        
        # Overall assessment
        all_k_success = all(
            info['conservative_bound_success_rate'] >= 0.9  # 90% success rate threshold
            for info in summary['k_specific_validation'].values()
        )
        
        summary['overall_lemma_validation'] = {
            'theoretical_bound_validated': bool(all_k_success),
            'condition_and_bound_both_satisfied': bool(condition_check['condition_satisfied'] and all_k_success),
            'proof_confidence': 'HIGH' if all_k_success else 'MEDIUM' if any(
                info['conservative_bound_success_rate'] >= 0.8 
                for info in summary['k_specific_validation'].values()
            ) else 'LOW'
        }
        
        return summary

    def create_validation_plots(self, results: Dict, output_dir: str = None) -> Dict[str, str]:
        """
        Create comprehensive visualization plots for Lemma 1 validation.
        
        Returns paths to generated plots.
        """
        plt.style.use('seaborn-v0_8')
        plot_paths = {}
        
        try:
            # Plot 1: Theoretical vs Actual Overlap
            fig, axes = plt.subplots(1, len(self.k_values), figsize=(5*len(self.k_values), 5))
            if len(self.k_values) == 1:
                axes = [axes]
            
            for idx, k in enumerate(self.k_values):
                k_str = f'k{k}'
                validation = results['theoretical_validation'][k_str]
                
                actual = validation['bound_validation']['detailed_results']['actual_overlaps']
                conservative = validation['bound_validation']['detailed_results']['conservative_bounds']
                tight = validation['bound_validation']['detailed_results']['tight_bounds']
                
                x = range(len(actual))
                axes[idx].scatter(x, actual, alpha=0.7, label='Actual Overlap', color='blue')
                axes[idx].plot(x, conservative, '--', color='red', alpha=0.8, label='Conservative Bound')
                axes[idx].plot(x, tight, '--', color='orange', alpha=0.8, label='Tight Bound')
                
                axes[idx].set_title(f'k={k}: Theoretical vs Actual Overlap')
                axes[idx].set_xlabel('Sample Index')
                axes[idx].set_ylabel('k-NN Overlap')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            if output_dir:
                path = f"{output_dir}/lemma1_theoretical_validation.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plot_paths['theoretical_validation'] = path
            plt.close()
            
            # Plot 2: Perturbation Ratio vs Overlap Loss
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            for k in self.k_values:
                k_str = f'k{k}'
                validation = results['theoretical_validation'][k_str]
                
                perturbation_ratios = validation['theoretical_bounds']['perturbation_ratios']
                actual_overlaps = validation['bound_validation']['detailed_results']['actual_overlaps']
                overlap_loss = [k - overlap for overlap in actual_overlaps]
                
                ax.scatter(perturbation_ratios, overlap_loss, alpha=0.6, label=f'k={k}')
            
            ax.set_xlabel('Perturbation Ratio (L_l · ||x - c(x)||_2 / δ_min)')
            ax.set_ylabel('k-NN Overlap Loss (k - actual_overlap)')
            ax.set_title('Perturbation Ratio vs Neighborhood Disruption')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if output_dir:
                path = f"{output_dir}/lemma1_perturbation_analysis.png"
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plot_paths['perturbation_analysis'] = path
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error creating plots: {e}")
        
        return plot_paths

# Backward compatibility: alias to original class name
Lemma1Validator = EnhancedLemma1Validator