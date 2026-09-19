"""
Metric Configuration Registry for Multi-Metric Layer Selection

This module provides a centralized, type-safe registry of all validation metrics
used for layer selection in geometric calibration experiments.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal
from enum import Enum


class MetricCategory(str, Enum):
    """Categories of validation metrics."""

    CALIBRATION = "calibration"
    GEOMETRIC = "geometric"
    STABILITY = "stability"
    MARGIN = "margin"
    FEATURE_QUALITY = "feature_quality"


@dataclass(frozen=True)
class MetricInfo:
    """
    Immutable information about a validation metric.

    Attributes:
        name: Unique metric identifier
        direction: Whether to maximize or minimize for optimization
        category: Semantic category of the metric
        description: Brief explanation of what the metric measures
    """

    name: str
    direction: Literal["maximize", "minimize"]
    category: MetricCategory
    description: str

    def should_maximize(self) -> bool:
        """Returns True if higher values are better."""

        return self.direction == "maximize"


class MetricRegistry:
    """
    Type-safe registry for validation metrics.

    Provides validation, lookup, and filtering capabilities for metrics
    used in multi-objective layer selection.
    """

    def __init__(self):
        self._metrics: Dict[str, MetricInfo] = {}
        self._register_default_metrics()

    def _register_default_metrics(self) -> None:
        """Register all known validation metrics."""

        metrics = [
            # Calibration Quality Proxies
            MetricInfo(
                "ece_score",
                "maximize",
                MetricCategory.CALIBRATION,
                "Negative ECE from geometric calibrator (higher = better calibration)",
            ),
            MetricInfo(
                "kfold_ece_utility",
                "maximize",
                MetricCategory.CALIBRATION,
                "Cross-validated ECE utility score (1 - ECE)",
            ),
            MetricInfo(
                "calibration_decisiveness",
                "maximize",
                MetricCategory.CALIBRATION,
                "How decisive the calibration predictions are",
            ),
            MetricInfo(
                "reliability_curve_quality",
                "maximize",
                MetricCategory.CALIBRATION,
                "Monotonicity and area quality of reliability curve",
            ),
            MetricInfo(
                "temperature_estimate_strength",
                "maximize",
                MetricCategory.CALIBRATION,
                "Strength of temperature scaling fit (slope + NLL gain)",
            ),

            # Geometric/Separation Metrics
            MetricInfo(
                "confidence_distance_correlation",
                "maximize",
                MetricCategory.GEOMETRIC,
                "Correlation between confidence and geometric distance",
            ),
            MetricInfo(
                "boundary_proximity_correlation",
                "maximize",
                MetricCategory.GEOMETRIC,
                "k-NN boundary proximity ↔ uncertainty alignment",
            ),
            MetricInfo(
                "multiscale_separation",
                "maximize",
                MetricCategory.GEOMETRIC,
                "k-NN class purity averaged over multiple k values",
            ),
            MetricInfo(
                "avg_class_separation_ratio",
                "maximize",
                MetricCategory.GEOMETRIC,
                "Inter-class / intra-class distance ratio",
            ),
            MetricInfo(
                "local_intrinsic_dimensionality",
                "minimize",
                MetricCategory.GEOMETRIC,
                "Estimated local intrinsic dimensionality (lower may indicate tighter manifolds)",
            ),
            MetricInfo(
                "uncertainty_geometry_alignment",
                "maximize",
                MetricCategory.GEOMETRIC,
                "Aggregate alignment of geometry with uncertainty/correctness",
            ),

            # Stability/Robustness
            MetricInfo(
                "spearman_stability_accuracy",
                "maximize",
                MetricCategory.STABILITY,
                "Correlation between stability score and accuracy",
            ),

            # Margin Safety
            MetricInfo(
                "margin_tail_cvar",
                "maximize",
                MetricCategory.MARGIN,
                "Conditional value at risk for decision margin tails (higher = safer)",
            ),
            MetricInfo(
                "class_balanced_margin_cvar",
                "maximize",
                MetricCategory.MARGIN,
                "Per-class z-scored margin tail safety",
            ),
            MetricInfo(
                "impostor_gap_cvar",
                "maximize",
                MetricCategory.MARGIN,
                "Tail of (nearest impostor − nearest same-class) gaps",
            ),
            MetricInfo(
                "margin_skewkurt_safety",
                "maximize",
                MetricCategory.MARGIN,
                "Penalty for left-skew/heavy tails in margins",
            ),

            # Feature Quality
            MetricInfo(
                "label_cka",
                "maximize",
                MetricCategory.FEATURE_QUALITY,
                "Linear CKA of features vs labels (feature-label association)",
            ),
            MetricInfo(
                "label_cka_hsic",
                "maximize",
                MetricCategory.FEATURE_QUALITY,
                "HSIC/CKA hybrid for label-feature association",
            ),
            MetricInfo(
                "nc1",
                "maximize",
                MetricCategory.FEATURE_QUALITY,
                "Neural Collapse 1 criterion (between-class variance alignment).",
            ),
            MetricInfo(
                "nc4",
                "maximize",
                MetricCategory.FEATURE_QUALITY,
                "Neural Collapse 4 criterion (classifier-feature alignment).",
            ),
            MetricInfo(
                "logistic_calibratability",
                "maximize",
                MetricCategory.CALIBRATION,
                "How well correctness fits a 1D sigmoid of margin",
            ),

            # Risk/Error Metrics (minimize)
            MetricInfo(
                "confident_error_rate_tau",
                "minimize",
                MetricCategory.CALIBRATION,
                "Error rate among high-confidence samples",
            ),
            MetricInfo(
                "aurc_proxy",
                "minimize",
                MetricCategory.CALIBRATION,
                "Area under risk-coverage curve (lower is better)",
            ),
        ]

        for metric in metrics:
            self._metrics[metric.name] = metric

    def get(self, metric_name: str) -> MetricInfo:
        """
        Get metric info by name.

        Args:
            metric_name: Name of the metric

        Returns:
            MetricInfo object

        Raises:
            KeyError: If metric name not found
        """

        if metric_name not in self._metrics:
            raise KeyError(
                f"Unknown metric: '{metric_name}'. "
                f"Available metrics: {sorted(self._metrics.keys())}"
            )
        return self._metrics[metric_name]

    def get_direction(self, metric_name: str) -> Literal["maximize", "minimize"]:
        """Get optimization direction for a metric."""

        return self.get(metric_name).direction

    def validate_metrics(self, metric_names: List[str]) -> None:
        """
        Validate that all metric names are known.

        Args:
            metric_names: List of metric names to validate

        Raises:
            ValueError: If any metric name is unknown
        """

        unknown = [m for m in metric_names if m not in self._metrics]
        if unknown:
            raise ValueError(
                f"Unknown metrics: {unknown}. "
                f"Available: {sorted(self._metrics.keys())}"
            )

    def get_by_category(self, category: MetricCategory) -> List[MetricInfo]:
        """Get all metrics in a category."""

        return [m for m in self._metrics.values() if m.category == category]

    def get_all_names(self) -> List[str]:
        """Get all registered metric names."""

        return sorted(self._metrics.keys())

    def __contains__(self, metric_name: str) -> bool:
        """Check if metric is registered."""

        return metric_name in self._metrics

    def __len__(self) -> int:
        """Number of registered metrics."""

        return len(self._metrics)


# Global registry instance
METRIC_REGISTRY = MetricRegistry()


class MetricPresets:
    """Predefined metric combinations for different research goals."""

    PARETO_CORE = [
        "ece_score",
        "margin_tail_cvar",
        "multiscale_separation",
    ]

    PARETO_EXTENDED = [
        "ece_score",
        "kfold_ece_utility",
        "margin_tail_cvar",
        "calibration_decisiveness",
        "multiscale_separation",
    ]

    CALIBRATION_FOCUSED = [
        "ece_score",
        "kfold_ece_utility",
        "reliability_curve_quality",
        "temperature_estimate_strength",
    ]

    GEOMETRIC_FOCUSED = [
        "confidence_distance_correlation",
        "boundary_proximity_correlation",
        "multiscale_separation",
        "margin_tail_cvar",
    ]

    STABILITY_FOCUSED = [
        "spearman_stability_accuracy",
        "margin_tail_cvar",
        "impostor_gap_cvar",
    ]


if __name__ == "__main__":
    # Demo usage
    print("=== Metric Registry Demo ===\n")

    # 1. Get metric info
    metric = METRIC_REGISTRY.get("ece_score")
    print(f"Metric: {metric.name}")
    print(f"Direction: {metric.direction}")
    print(f"Category: {metric.category.value}")
    print(f"Description: {metric.description}")
    print(f"Should maximize: {metric.should_maximize()}\n")

    # 2. Validate metrics
    try:
        METRIC_REGISTRY.validate_metrics(["ece_score", "margin_tail_cvar"])
        print("✓ Metrics validated successfully\n")
    except ValueError as e:
        print(f"✗ Validation failed: {e}\n")

    # 3. Get metrics by category
    cal_metrics = METRIC_REGISTRY.get_by_category(MetricCategory.CALIBRATION)
    print(f"Calibration metrics ({len(cal_metrics)}):")
    for m in cal_metrics:
        print(f"  - {m.name}: {m.direction}")
    print()

    # 4. Use presets
    print("PARETO_CORE preset:")
    for metric_name in MetricPresets.PARETO_CORE:
        direction = METRIC_REGISTRY.get_direction(metric_name)
        print(f"  - {metric_name}: {direction}")


