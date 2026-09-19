"""
Integration module for Bradley-Terry chooser with TwoStageLayerSelector.

This module provides a thin wrapper to integrate the BT chooser into the existing
TwoStageLayerSelector without breaking current behavior.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np

from .data_io import RunRecord
from .infer import choose_layer

# Configure logging
from utils.logging_config import get_logger
logger = get_logger(__name__)


def extend_two_stage_selector_with_bt():
    """
    Extend TwoStageLayerSelector with BT chooser integration.
    
    This function monkey-patches the TwoStageLayerSelector class to add
    the select_with_bt method without breaking existing functionality.
    """
    # Import here to avoid circular imports
    from Calibrators.metrics import TwoStageLayerSelector, AdHocLayerSelector, LayerMetrics
    
    def select_with_bt(
        self,
        model: torch.nn.Module,
        dataloader,
        candidate_layers: List[int],
        device: torch.device,
        model_adapter,
        bt_path: str,
        metric_names: List[str],
        tie_delta: float = 1e-4,
        apply_pareto: bool = True
    ) -> Tuple[int, List[LayerMetrics]]:
        """
        Select layer using Bradley-Terry chooser with optional Stage-B fallback.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader for computing metrics
            candidate_layers: List of layer indices to consider
            device: Device to run on
            model_adapter: Model adapter for feature extraction
            bt_path: Path to trained Bradley-Terry model
            metric_names: List of metric names to use for BT chooser
            tie_delta: Threshold for tie detection
            apply_pareto: Whether to apply Pareto filtering
            
        Returns:
            Tuple of (selected_layer_idx, all_layer_metrics)
        """
        logger.info(f"🎯 Starting BT chooser selection on {len(candidate_layers)} layers")
        
        # Step 1: Compute cheap metrics for all candidate layers (Stage-A style)
        logger.info("📊 Computing cheap metrics for all candidate layers...")
        
        # Create a cheap selector with only the specified metrics
        cheap_metrics = []
        for calc in self.sel.original_metrics:
            if calc.name in metric_names:
                cheap_metrics.append(calc)
        
        if not cheap_metrics:
            logger.warning(f"No metrics found for {metric_names}, falling back to original select method")
            return self.select(model, dataloader, candidate_layers, device, None, model_adapter)
        
        # Create cheap selector with only the specified metrics
        cheap_sel = AdHocLayerSelector(
            metrics=cheap_metrics,
            score_weights={name: 1.0 for name in metric_names},  # Equal weights
            skip_zero_weight_metrics=True
        )
        
        # Ensure model adapter is available
        if model_adapter is not None:
            self.sel.model_adapter = model_adapter
        
        # Compute metrics for all candidate layers
        _, cheap_metrics_list = cheap_sel.select_best_layer(
            model, dataloader,
            candidate_layers=candidate_layers,
            device=device,
            model_adapter=self.sel.model_adapter
        )
        
        logger.info(f"✅ Computed metrics for {len(cheap_metrics_list)} layers")
        
        # Step 2: Build temporary RunRecord
        logger.info("🔧 Building RunRecord for BT chooser...")
        
        # Extract metrics from LayerMetrics objects
        layers_data = []
        for metrics in cheap_metrics_list:
            layer_data = {
                "layer_idx": metrics.layer_idx,
                "metrics": {},
                "target_ece": 0.0  # Not needed for BT chooser
            }
            
            # Extract metric values
            for metric_name in metric_names:
                if hasattr(metrics, metric_name):
                    value = getattr(metrics, metric_name)
                    if not (np.isnan(value) or np.isinf(value)):
                        layer_data["metrics"][metric_name] = float(value)
            
            layers_data.append(layer_data)
        
        # Create RunRecord
        run_record = RunRecord(
            run_id="bt_inference",
            dataset="inference",
            backbone="unknown",
            seed=0,
            layers=layers_data
        )
        
        logger.info(f"📋 Created RunRecord with {len(layers_data)} layers")
        
        # Step 3: Call BT chooser
        logger.info("🤖 Running BT chooser...")
        
        try:
            bt_result = choose_layer(
                bt_path, 
                run_record, 
                apply_pareto=apply_pareto, 
                tie_delta=tie_delta
            )
            
            logger.info(f"🎯 BT chooser result: {bt_result['strategy']}")
            
        except Exception as e:
            logger.error(f"BT chooser failed: {e}, falling back to original select method")
            return self.select(model, dataloader, candidate_layers, device, None, model_adapter)
        
        # Step 4: Handle BT chooser result
        if bt_result["strategy"] == "single":
            # Single clear winner
            selected_layer = bt_result["layer_idx"]
            logger.info(f"✅ BT chooser selected layer {selected_layer}")
            
            # Return the selected layer and all metrics
            return selected_layer, cheap_metrics_list
            
        elif bt_result["strategy"] == "tie":
            # Tie detected - run Stage-B on top-2
            logger.info("🤝 Tie detected, running Stage-B on top-2 layers...")
            
            tied_layers = [item[0] for item in bt_result["top2"]]
            logger.info(f"🔍 Running Stage-B on tied layers: {tied_layers}")
            
            # Run Stage-B on tied layers using original method
            stageB_metrics = []
            
            # Compute original_data and model probabilities once for Stage-B
            predictions = None
            confidences = None
            model_probs = None
            
            try:
                original_data = []
                original_labels = []
                
                for batch_data, batch_labels in dataloader:
                    if isinstance(batch_data, (list, tuple)):
                        batch_data = batch_data[0]
                    original_data.append(batch_data)
                    original_labels.append(batch_labels)
                
                original_data = torch.cat(original_data, dim=0)
                original_labels = torch.cat(original_labels, dim=0)
                
                # Get model predictions
                with torch.no_grad():
                    model.eval()
                    model_outputs = model(original_data.to(device))
                    
                    if isinstance(model_outputs, (list, tuple)):
                        model_probs = torch.softmax(model_outputs[0], dim=1).cpu().numpy()
                    else:
                        model_probs = torch.softmax(model_outputs, dim=1).cpu().numpy()
                    
                    predictions = np.argmax(model_probs, axis=1)
                    confidences = np.max(model_probs, axis=1)
                
            except Exception as e:
                logger.warning(f"Failed to prepare shared data for Stage-B: {e}")
            
            # Compute Stage-B metrics for tied layers
            for layer_idx in tied_layers:
                try:
                    # Get features for this layer
                    features, labels = self.sel._extract_features_for_layer(
                        model, dataloader, layer_idx, device, model_adapter
                    )
                    
                    # Set shared data for metrics that need it
                    if hasattr(self.sel, '_set_shared_data_for_metrics'):
                        try:
                            shared = type('SharedData', (), {
                                'original_data': original_data,
                                'original_labels': original_labels,
                                'predictions': predictions,
                                'confidences': confidences,
                                'model_probs': model_probs
                            })()
                            self.sel._set_shared_data_for_metrics(shared, labels)
                        except Exception as e:
                            logger.warning(f"Failed to set shared data for layer {layer_idx}: {e}")
                    
                    # Compute Stage-B metrics
                    stageB_metric = self.sel.compute_layer_metrics(
                        features, labels, layer_idx=layer_idx,
                        predictions=predictions, confidences=confidences, model_probs=model_probs
                    )
                    stageB_metrics.append(stageB_metric)
                    
                except Exception as e:
                    logger.warning(f"Failed to compute Stage-B metrics for layer {layer_idx}: {e}")
                    # Create dummy metrics
                    dummy_metrics = LayerMetrics(layer_idx=layer_idx)
                    dummy_metrics.composite_score = 0.0
                    stageB_metrics.append(dummy_metrics)
            
            if not stageB_metrics:
                logger.error("No Stage-B metrics computed, falling back to first tied layer")
                selected_layer = tied_layers[0]
            else:
                # Choose best by composite score
                best_metric = max(stageB_metrics, key=lambda m: m.composite_score)
                selected_layer = best_metric.layer_idx
                logger.info(f"✅ Stage-B selected layer {selected_layer} (composite score: {best_metric.composite_score:.4f})")
            
            # Combine cheap metrics with Stage-B metrics
            all_metrics = cheap_metrics_list.copy()
            
            # Update Stage-B metrics in the combined list
            for stageB_metric in stageB_metrics:
                # Find and replace the corresponding cheap metric
                for i, cheap_metric in enumerate(all_metrics):
                    if cheap_metric.layer_idx == stageB_metric.layer_idx:
                        all_metrics[i] = stageB_metric
                        break
                else:
                    # Add new Stage-B metric if not found
                    all_metrics.append(stageB_metric)
            
            return selected_layer, all_metrics
            
        else:
            # Error case - fall back to original method
            logger.warning(f"BT chooser returned error: {bt_result.get('message', 'Unknown error')}")
            return self.select(model, dataloader, candidate_layers, device, None, model_adapter)
    
    # Add the method to TwoStageLayerSelector
    TwoStageLayerSelector.select_with_bt = select_with_bt
    
    logger.info("✅ Successfully extended TwoStageLayerSelector with BT chooser integration")


def create_bt_wrapper_selector(base_selector, bt_path: str, metric_names: List[str], **bt_kwargs):
    """
    Create a wrapper selector that uses BT chooser as the primary method.
    
    Args:
        base_selector: Base TwoStageLayerSelector instance
        bt_path: Path to trained Bradley-Terry model
        metric_names: List of metric names to use for BT chooser
        **bt_kwargs: Additional arguments for BT chooser
        
    Returns:
        Wrapper selector with BT chooser integration
    """
    # Ensure the extension is applied
    extend_two_stage_selector_with_bt()
    
    class BTWrapperSelector:
        def __init__(self, base_selector, bt_path, metric_names, **bt_kwargs):
            self.base_selector = base_selector
            self.bt_path = bt_path
            self.metric_names = metric_names
            self.bt_kwargs = bt_kwargs
        
        def select_with_bt(self, model, dataloader, candidate_layers, device, model_adapter=None):
            """Use BT chooser for selection."""
            return self.base_selector.select_with_bt(
                model, dataloader, candidate_layers, device, model_adapter,
                self.bt_path, self.metric_names, **self.bt_kwargs
            )
        
        def select(self, model, dataloader, candidate_layers, device, stageB_hook, model_adapter=None):
            """Fallback to original selection method."""
            return self.base_selector.select(
                model, dataloader, candidate_layers, device, stageB_hook, model_adapter
            )
    
    return BTWrapperSelector(base_selector, bt_path, metric_names, **bt_kwargs)


# Auto-apply the extension when this module is imported
extend_two_stage_selector_with_bt()
