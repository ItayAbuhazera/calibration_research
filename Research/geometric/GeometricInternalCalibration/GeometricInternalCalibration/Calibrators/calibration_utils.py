import numpy as np
import torch
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import label_binarize

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute the Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels of shape (n_samples,)
        n_bins: Number of bins for probability discretization
        
    Returns:
        Expected Calibration Error
    """
    # Get the predicted class and its probability
    pred_probs = np.max(probs, axis=1)
    pred_labels = np.argmax(probs, axis=1)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get samples in this bin
        in_bin = np.logical_and(pred_probs >= bin_lower, pred_probs < bin_upper)
        if np.sum(in_bin) > 0:
            # Calculate accuracy and confidence in this bin
            accuracy = np.mean(pred_labels[in_bin] == labels[in_bin])
            confidence = np.mean(pred_probs[in_bin])
            # Add to ECE
            ece += np.abs(accuracy - confidence) * np.sum(in_bin) / len(labels)
    
    return ece

def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Brier score for multiclass classification.

    Args:
        probs: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels of shape (n_samples,)

    Returns:
        Brier score
    """
    n_classes = probs.shape[1]
    labels_bin = label_binarize(labels, classes=range(n_classes))
    return brier_score_loss(labels_bin, probs)

def compute_error_detection_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute AUROC for error detection using model confidence as the score.
    Positive class = error (incorrect prediction).
    Score = 1 - max_prob so higher scores mean more likely to be an error.
    Returns AUROC in [0,1], or NaN if undefined.
    """
    pred_labels = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)
    errors = (pred_labels != labels).astype(int)
    scores = 1.0 - max_probs

    if errors.sum() == 0 or errors.sum() == len(errors):
        return float("nan")

    return float(roc_auc_score(errors, scores))

def compute_risk_coverage_curve(probs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute risk-coverage curve for selective prediction.
    Coverage c: fraction of points kept (highest confidence first).
    Risk(c): error rate on kept subset.
    """
    pred_labels = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    correct = (pred_labels == labels).astype(int)
    n = len(labels)

    if n == 0:
        return np.array([]), np.array([])

    order = np.argsort(-confidences)
    correct_sorted = correct[order]
    cumulative_errors = np.cumsum(1 - correct_sorted)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = cumulative_errors / k

    return coverage, risk

def compute_aurc(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Area Under Risk-Coverage curve (AURC).
    Lower is better (less risk for given coverage).
    """
    coverage, risk = compute_risk_coverage_curve(probs, labels)
    if coverage.size == 0:
        return float("nan")
    return float(np.trapz(risk, coverage))

def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10,
                           title: str = "Reliability Diagram") -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a reliability diagram.
    
    Args:
        probs: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels of shape (n_samples,)
        n_bins: Number of bins for probability discretization
        title: Title for the plot
        
    Returns:
        Figure and axes objects
    """
    # Get the predicted class and its probability
    pred_probs = np.max(probs, axis=1)
    pred_labels = np.argmax(probs, axis=1)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Calculate accuracy and confidence for each bin
    accuracies = []
    confidences = []
    counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(pred_probs >= bin_lower, pred_probs < bin_upper)
        if np.sum(in_bin) > 0:
            accuracy = np.mean(pred_labels[in_bin] == labels[in_bin])
            confidence = np.mean(pred_probs[in_bin])
            count = np.sum(in_bin)
            
            accuracies.append(accuracy)
            confidences.append(confidence)
            counts.append(count)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the reliability curve
    ax.plot(confidences, accuracies, 'bo-', label='Model')
    
    # Plot the perfect calibration line
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    
    # Add labels and title
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    
    # Set the limits and grid
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    
    return fig, ax

def evaluate_calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """
    Evaluate calibration using multiple metrics.
    
    Args:
        probs: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels of shape (n_samples,)
        n_bins: Number of bins for probability discretization
        
    Returns:
        Dictionary containing calibration metrics
    """
    ece = compute_ece(probs, labels, n_bins)
    brier = compute_brier_score(probs, labels)
    
    return {
        'ece': ece,
        'brier_score': brier
    } 


def compute_ood_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """
    Compute AUROC for OOD detection.
    Higher score = more likely OOD.
    
    Args:
        id_scores: Uncertainty scores for in-distribution samples (N_id,)
        ood_scores: Uncertainty scores for out-of-distribution samples (N_ood,)
    
    Returns:
        AUROC where 1.0 = perfect separation, 0.5 = random
    """
    from sklearn.metrics import roc_auc_score
    
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([
        np.zeros(len(id_scores)),  # 0 = ID
        np.ones(len(ood_scores))   # 1 = OOD
    ])
    
    if len(np.unique(labels)) < 2:
        return float('nan')
    
    return float(roc_auc_score(labels, scores))


def compute_fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, 
                       tpr_threshold: float = 0.95) -> float:
    """
    Compute FPR@95%TPR for OOD detection.
    
    At 95% true positive rate (accepting 95% of ID samples),
    what fraction of OOD samples are incorrectly accepted?
    
    Lower is better (0 = no OOD accepted).
    """
    from sklearn.metrics import roc_curve
    
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores))
    ])
    
    if len(np.unique(labels)) < 2:
        return float('nan')
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Find threshold where TPR >= tpr_threshold
    idx = np.where(tpr >= tpr_threshold)[0]
    if len(idx) == 0:
        return 1.0
    
    return float(fpr[idx[0]])


def compute_geometric_uncertainty_score(
    model_adapter,
    test_raw: np.ndarray,
    selected_layer_indices: List[int],
    discovered_layers: List[Dict],
    train_features: torch.Tensor,
    train_labels: np.ndarray,
    target_dim: int,
    batch_size: int,
    device: str = "cuda"
) -> np.ndarray:
    """
    Compute uncertainty scores for OOD detection using geometric method.
    Higher score = more uncertain = more likely OOD.
    
    Returns max softmax probability (1 - confidence can be uncertainty score)
    or you can return geometric distance-based scores.
    """
    from Experiments.layer_selection import extract_features_directly
    from Calibrators.geometric_calibrator_new import GeometricCalibrator
    
    # Extract features from test set
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(test_raw),
            torch.zeros(len(test_raw), dtype=torch.long)
        ),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Extract and compress features
    test_features_list = []
    for layer_idx in selected_layer_indices:
        layer_name = discovered_layers[layer_idx]['name']
        test_feats, _ = extract_features_directly(
            model_adapter.model,
            test_loader,
            layer_name,
            torch.device(device),
            compression_ratio=calculate_compression_ratio(test_raw.shape, target_dim),
            output_dim=target_dim
        )
        test_features_list.append(test_feats)
    
    # Combine features
    if len(test_features_list) > 1:
        test_features = torch.cat(test_features_list, dim=1)
    else:
        test_features = test_features_list[0]
    
    # Fit calibrator and get probabilities
    calibrator = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False
    )
    calibrator.fit(X_val_embed=train_features, y_val=train_labels, X_val_original=train_raw)
    test_probs = calibrator.calibrate_batched(X_test_embed=test_features, X_test_original=test_raw, batch_size=batch_size)
    
    # Uncertainty score: 1 - max_probability (higher = more uncertain)
    max_probs = np.max(test_probs, axis=1)
    uncertainty_scores = 1.0 - max_probs
    
    return uncertainty_scores