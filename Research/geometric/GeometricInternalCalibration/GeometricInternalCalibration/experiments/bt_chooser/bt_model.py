"""
Bradley-Terry chooser model for layer selection.

This module implements a trainable Bradley-Terry chooser that learns to rank layers
based on pairwise comparisons using logistic regression with grouped cross-validation.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

from .data_io import RunRecord
from .pairs import build_pairs

# Configure logging
from utils.logging_config import get_logger
logger = get_logger(__name__)


class BradleyTerryChooser:
    """
    Bradley-Terry chooser for layer selection based on pairwise comparisons.
    
    This model learns to rank layers by training on pairwise comparisons between
    layers within runs, using logistic regression with grouped cross-validation.
    """
    
    def __init__(
        self,
        metric_names: List[str],
        use_ranks: bool = True,
        standardize: bool = True,
        penalty: str = "l2",
        C_grid: List[float] = [0.1, 0.3, 1, 3, 10],
        class_weight: str = "balanced",
        eps_tie: float = 0.003,
        max_pairs_per_run: Optional[int] = 400,
        random_state: int = 0
    ):
        """
        Initialize Bradley-Terry chooser.
        
        Args:
            metric_names: List of metric names to use for features
            use_ranks: Whether to include rank features
            standardize: Whether to standardize pair differences
            penalty: Regularization penalty type
            C_grid: Grid of regularization parameters to try
            class_weight: Class weight strategy for imbalanced data
            eps_tie: Minimum ECE difference threshold for keeping pairs
            max_pairs_per_run: Maximum pairs per run (None for no limit)
            random_state: Random seed for reproducibility
        """
        self.metric_names = metric_names
        self.use_ranks = use_ranks
        self.standardize = standardize
        self.penalty = penalty
        self.C_grid = C_grid
        self.class_weight = class_weight
        self.eps_tie = eps_tie
        self.max_pairs_per_run = max_pairs_per_run
        self.random_state = random_state
        
        # Model components (set during fit)
        self.model_ = None
        self.scaler_ = None
        self.column_order_ = None
        self.feature_names_ = None
        
        # Training metadata
        self.n_runs_ = 0
        self.n_pairs_ = 0
        self.cv_results_ = None
        
    def _build_column_order(self) -> List[str]:
        """Build column order identical to pairs.build_pairs."""
        columns = []
        for metric in self.metric_names:
            columns.append(metric)
        if self.use_ranks:
            for metric in self.metric_names:
                columns.append(f"{metric}_rank")
        return columns
    
    def _extract_group_label(self, run: RunRecord, group_by: str) -> str:
        """Extract group label from run based on group_by parameter."""
        if group_by == "dataset":
            return run.dataset
        elif group_by == "dataset_backbone":
            return f"{run.dataset}_{run.backbone}"
        else:
            raise ValueError(f"Unknown group_by: {group_by}")
    
    def _build_layer_feature_matrix(
        self, 
        layer_features: Dict[int, Dict[str, float]], 
        layer_order: List[int]
    ) -> np.ndarray:
        """Build feature matrix per layer with consistent column order."""
        if not layer_features:
            return np.array([]).reshape(0, len(self.column_order_))
        
        n_layers = len(layer_order)
        n_features = len(self.column_order_)
        X = np.full((n_layers, n_features), np.nan)
        
        # Fill in available values
        for i, layer_idx in enumerate(layer_order):
            if layer_idx in layer_features:
                layer_data = layer_features[layer_idx]
                for j, col_name in enumerate(self.column_order_):
                    if col_name in layer_data:
                        X[i, j] = layer_data[col_name]
        
        # Impute missing values with neutral values
        for j in range(n_features):
            col_data = X[:, j]
            valid_mask = ~np.isnan(col_data)
            
            if np.any(valid_mask):
                median_val = np.median(col_data[valid_mask])
                X[np.isnan(col_data), j] = median_val
            else:
                # Use neutral imputation: 0.5 for rank features, 0.0 for raw features
                if self.column_order_[j].endswith("_rank"):
                    X[:, j] = 0.5  # Neutral rank
                else:
                    X[:, j] = 0.0  # Neutral raw value
        
        return X
    
    def _evaluate_model(
        self, 
        model: LogisticRegression, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        test_runs: List[RunRecord],
        test_layer_orders: List[List[int]]
    ) -> Tuple[float, float]:
        """
        Evaluate model on test data using hit rate and regret.
        
        Args:
            model: Trained logistic regression model
            X_test: Test feature matrix
            y_test: Test labels
            test_runs: Test runs
            test_layer_orders: Layer orders for test runs
            
        Returns:
            Tuple of (hit_rate, mean_regret)
        """
        if len(test_runs) == 0:
            return 0.0, float('inf')
        
        hit_count = 0
        total_runs = 0
        regrets = []
        
        # Process each test run
        run_start_idx = 0
        for run_idx, (run, layer_order) in enumerate(zip(test_runs, test_layer_orders)):
            # Find pairs for this run
            run_pairs = []
            for i in range(len(layer_order)):
                for j in range(len(layer_order)):
                    if i != j:
                        pair_idx = run_start_idx + i * len(layer_order) + j - (i if i < j else i + 1)
                        if pair_idx < len(y_test):
                            run_pairs.append((i, j, pair_idx))
            
            if not run_pairs:
                continue
                
            # Build layer feature matrix for this run
            from .features import make_layer_features, add_within_run_ranks
            layer_features = make_layer_features(run, self.metric_names)
            if self.use_ranks:
                layer_features = add_within_run_ranks(layer_features)
            
            X_layers = self._build_layer_feature_matrix(layer_features, layer_order)
            
            if X_layers.size == 0:
                continue
            
            # Score layers using learned weights
            layer_scores = self.score_layers(X_layers)
            
            # Find predicted best layer
            predicted_best = np.argmax(layer_scores)
            
            # Find true best layer by ECE
            layer_to_ece = {layer["layer_idx"]: layer["target_ece"] for layer in run.layers}
            true_eces = [layer_to_ece.get(layer_idx, float('inf')) for layer_idx in layer_order]
            true_best = np.argmin(true_eces)
            
            # Check hit rate
            if predicted_best == true_best:
                hit_count += 1
            total_runs += 1
            
            # Compute regret
            chosen_ece = true_eces[predicted_best]
            min_ece = min(true_eces)
            regret = chosen_ece - min_ece
            regrets.append(regret)
            
            run_start_idx += len(run_pairs)
        
        hit_rate = hit_count / total_runs if total_runs > 0 else 0.0
        mean_regret = np.mean(regrets) if regrets else float('inf')
        
        return hit_rate, mean_regret
    
    def fit(self, runs: List[RunRecord], group_by: str = "dataset") -> Dict:
        """
        Fit the Bradley-Terry chooser using grouped cross-validation.
        
        Args:
            runs: List of experiment runs
            group_by: How to group runs for CV ("dataset" or "dataset_backbone")
            
        Returns:
            Dictionary with training report including CV metrics and feature weights
        """
        logger.info(f"Fitting Bradley-Terry chooser on {len(runs)} runs")
        
        if not runs:
            raise ValueError("No runs provided")
        
        # Build column order
        self.column_order_ = self._build_column_order()
        self.feature_names_ = self.column_order_.copy()
        
        # Build pairs for all runs
        all_X = []
        all_y = []
        all_groups = []
        all_runs = []
        all_layer_orders = []
        
        for run in runs:
            X, y, w, layer_order = build_pairs(
                run, 
                self.metric_names, 
                eps=self.eps_tie,
                max_pairs=self.max_pairs_per_run,
                use_ranks=self.use_ranks
            )
            
            if X.size > 0:
                all_X.append(X)
                all_y.append(y)
                group_label = self._extract_group_label(run, group_by)
                all_groups.extend([group_label] * len(y))
                all_runs.extend([run] * len(y))
                all_layer_orders.extend([layer_order] * len(y))
        
        if not all_X:
            raise ValueError("No valid pairs found across all runs")
        
        # Combine all pairs
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        groups_combined = np.array(all_groups)
        
        self.n_runs_ = len(runs)
        self.n_pairs_ = len(y_combined)
        
        logger.info(f"Built {self.n_pairs_} pairs from {self.n_runs_} runs")
        logger.info(f"Unique groups: {np.unique(groups_combined)}")
        
        # Group runs for CV
        unique_groups = np.unique(groups_combined)
        group_to_runs = {}
        for i, group in enumerate(unique_groups):
            group_mask = groups_combined == group
            group_runs = [all_runs[j] for j in np.where(group_mask)[0]]
            group_to_runs[group] = group_runs  # Keep all runs for this group
        
        # Leave-one-group-out cross-validation
        cv_results = []
        
        for C in self.C_grid:
            logger.info(f"Testing C={C}")
            
            fold_hit_rates = []
            fold_regrets = []
            
            # leave-one-group-out: train on all other groups; evaluate on every run in the held-out group
            for test_group in unique_groups:
                # Split data
                train_mask = groups_combined != test_group
                test_mask = groups_combined == test_group
                X_train = X_combined[train_mask]
                y_train = y_combined[train_mask]
                if len(X_train) == 0:
                    continue
                
                # Standardize if requested
                if self.standardize:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                else:
                    scaler = None
                    X_train_scaled = X_train
                
                # Train model on train pairs
                model = LogisticRegression(
                    penalty=self.penalty,
                    C=C,
                    class_weight=self.class_weight,
                    random_state=self.random_state,
                    max_iter=1000
                )
                model.fit(X_train_scaled, y_train)

                # Evaluate on held-out group: go over **unique runs** whose group == test_group
                test_runs_unique = [r for r in runs if self._extract_group_label(r, group_by) == test_group]
                fold_hits, fold_regrets = [], []
                from .features import make_layer_features, add_within_run_ranks
                for run in test_runs_unique:
                    layer_features = make_layer_features(run, self.metric_names)
                    if self.use_ranks:
                        layer_features = add_within_run_ranks(layer_features)
                    X_layers = self._build_layer_feature_matrix(layer_features, sorted(layer_features.keys()))
                    if X_layers.size == 0:
                        continue
                    # standardize with the **train** scaler
                    X_layers_s = scaler.transform(X_layers) if scaler is not None else X_layers
                    scores = X_layers_s @ model.coef_[0] + model.intercept_[0]
                    pred_idx = np.argmax(scores)
                    layer_order = sorted(layer_features.keys())
                    pred_layer = layer_order[pred_idx]
                    # compute hit & regret against true best ECE in this run
                    eces = {l["layer_idx"]: l["target_ece"] for l in run.layers}
                    true_best = min(eces, key=eces.get)
                    hit = 1 if pred_layer == true_best else 0
                    regret = eces[pred_layer] - eces[true_best]
                    fold_hits.append(hit)
                    fold_regrets.append(regret)
                if not fold_hits:
                    continue
                hit_rate = float(np.mean(fold_hits))
                mean_regret = float(np.mean(fold_regrets))
                
                fold_hit_rates.append(hit_rate)
                fold_regrets.append(mean_regret)
            
            if fold_hit_rates:
                cv_results.append({
                    'C': C,
                    'hit_rate_mean': np.mean(fold_hit_rates),
                    'hit_rate_std': np.std(fold_hit_rates),
                    'regret_mean': np.mean(fold_regrets),
                    'regret_std': np.std(fold_regrets),
                    'n_folds': len(fold_hit_rates)
                })
                
                logger.info(f"C={C}: hit_rate={np.mean(fold_hit_rates):.3f}±{np.std(fold_hit_rates):.3f}, "
                          f"regret={np.mean(fold_regrets):.3f}±{np.std(fold_regrets):.3f}")
        
        # Select best C
        if not cv_results:
            raise ValueError("No valid CV results")
        
        # Sort by hit rate (descending), then by regret (ascending)
        cv_results.sort(key=lambda x: (-x['hit_rate_mean'], x['regret_mean']))
        best_result = cv_results[0]
        best_C = best_result['C']
        
        logger.info(f"Selected C={best_C} with hit_rate={best_result['hit_rate_mean']:.3f}")
        
        # Refit on all data with best C
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_final = self.scaler_.fit_transform(X_combined)
        else:
            self.scaler_ = None
            X_final = X_combined
        
        self.model_ = LogisticRegression(
            penalty=self.penalty,
            C=best_C,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=1000
        )
        self.model_.fit(X_final, y_combined)
        
        # Store CV results
        self.cv_results_ = cv_results
        
        # Build report
        report = {
            'n_runs': self.n_runs_,
            'n_pairs': self.n_pairs_,
            'best_C': best_C,
            'cv_results': cv_results,
            'feature_names': self.feature_names_,
            'feature_weights': self.model_.coef_[0].tolist(),
            'intercept': float(self.model_.intercept_[0]),
            'metric_names': self.metric_names,
            'use_ranks': self.use_ranks,
            'standardize': self.standardize
        }
        
        logger.info("Training completed successfully")
        return report
    
    def score_layers(self, layer_feature_matrix: np.ndarray) -> np.ndarray:
        """
        Score layers using learned weights.
        
        Args:
            layer_feature_matrix: Feature matrix with shape (n_layers, n_features)
            
        Returns:
            Array of layer scores
        """
        if self.model_ is None:
            raise ValueError("Model not fitted yet")
        
        if layer_feature_matrix.shape[1] != len(self.column_order_):
            raise ValueError(f"Expected {len(self.column_order_)} features, got {layer_feature_matrix.shape[1]}")
        
        # Standardize if needed
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(layer_feature_matrix)
        else:
            X_scaled = layer_feature_matrix
        
        # Compute scores: s = w^T x
        scores = X_scaled @ self.model_.coef_[0] + self.model_.intercept_[0]
        
        return scores
    
    def save(self, path: str) -> None:
        """Save model to JSON file."""
        if self.model_ is None:
            raise ValueError("Model not fitted yet")
        
        model_data = {
            'metric_names': self.metric_names,
            'use_ranks': self.use_ranks,
            'standardize': self.standardize,
            'penalty': self.penalty,
            'class_weight': self.class_weight,
            'eps_tie': self.eps_tie,
            'max_pairs_per_run': self.max_pairs_per_run,
            'random_state': self.random_state,
            'column_order': self.column_order_,
            'feature_names': self.feature_names_,
            'coef': self.model_.coef_[0].tolist(),
            'intercept': float(self.model_.intercept_[0]),
            'scaler_mean': self.scaler_.mean_.tolist() if self.scaler_ else None,
            'scaler_scale': self.scaler_.scale_.tolist() if self.scaler_ else None,
            'cv_results': self.cv_results_
        }
        
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from JSON file."""
        with open(path, 'r') as f:
            model_data = json.load(f)
        
        # Restore parameters
        self.metric_names = model_data['metric_names']
        self.use_ranks = model_data['use_ranks']
        self.standardize = model_data['standardize']
        self.penalty = model_data['penalty']
        self.class_weight = model_data['class_weight']
        self.eps_tie = model_data['eps_tie']
        self.max_pairs_per_run = model_data['max_pairs_per_run']
        self.random_state = model_data['random_state']
        self.column_order_ = model_data['column_order']
        self.feature_names_ = model_data['feature_names']
        self.cv_results_ = model_data.get('cv_results')
        
        # Restore model
        coef = np.array(model_data['coef'])
        intercept = np.array([model_data['intercept']])
        
        self.model_ = LogisticRegression(
            penalty=self.penalty,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        self.model_.coef_ = coef.reshape(1, -1)
        self.model_.intercept_ = intercept
        
        # Restore scaler
        if model_data['scaler_mean'] is not None:
            self.scaler_ = StandardScaler()
            self.scaler_.mean_ = np.array(model_data['scaler_mean'])
            self.scaler_.scale_ = np.array(model_data['scaler_scale'])
        else:
            self.scaler_ = None
        
        logger.info(f"Model loaded from {path}")


def test_bt_chooser():
    """Test the BradleyTerryChooser with synthetic data."""
    from .data_io import RunRecord
    
    # Create test runs
    runs = [
        RunRecord(
            run_id="test_1",
            dataset="cifar10",
            backbone="resnet50",
            seed=42,
            layers=[
                {"layer_idx": 0, "metrics": {"metric1": 0.8, "metric2": 0.6}, "target_ece": 0.15},
                {"layer_idx": 1, "metrics": {"metric1": 0.7, "metric2": 0.8}, "target_ece": 0.12},
            ]
        ),
        RunRecord(
            run_id="test_2", 
            dataset="cifar10",
            backbone="resnet50",
            seed=43,
            layers=[
                {"layer_idx": 0, "metrics": {"metric1": 0.9, "metric2": 0.5}, "target_ece": 0.18},
                {"layer_idx": 1, "metrics": {"metric1": 0.6, "metric2": 0.9}, "target_ece": 0.14},
            ]
        )
    ]
    
    # Test chooser
    chooser = BradleyTerryChooser(
        metric_names=["metric1", "metric2"],
        use_ranks=False,
        C_grid=[1.0, 10.0]
    )
    
    print("Testing BradleyTerryChooser...")
    
    # Fit model
    report = chooser.fit(runs, group_by="dataset")
    
    print(f"Training report: {report}")
    print("✓ BradleyTerryChooser test passed!")


if __name__ == "__main__":
    test_bt_chooser()
