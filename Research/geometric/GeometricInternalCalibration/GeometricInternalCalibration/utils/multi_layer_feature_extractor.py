"""
Multi-layer feature extraction system for in-training calibration
Uses focused collection strategy for reliable layer selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
from Metrics.metrics import expected_calibration_error

from utils.logging_config import get_logger
logger = get_logger(__name__)


class LayerFeatureExtractor:
    """
    Extracts features from multiple layers using focused collection strategy.
    
    Instead of continuously collecting features throughout training, this approach
    collects features during one specific epoch when the model representations
    have stabilized, then makes the layer selection decision.
    """
    
    def __init__(self, model, collection_epoch: int = 3, min_batches_for_decision: int = 50):
        """
        Initialize the feature extractor with focused collection strategy.
        
        Think of this like planning a single, comprehensive survey rather than
        trying to gather data continuously. We wait until the model has learned
        stable representations, then do one focused collection session.
        
        Args:
            model: The neural network model to extract features from
            collection_epoch: Which epoch to collect features for layer selection
            min_batches_for_decision: Minimum batches needed to make a good decision
        """
        self.model = model
        self.collection_epoch = collection_epoch
        self.min_batches_for_decision = min_batches_for_decision
        
        # Collection state tracking
        self.focused_collection_active = False
        self.collected_batches = 0
        self.target_batches = min_batches_for_decision
        self.current_epoch = 0
        
        # Storage for the focused collection session
        # This is like having organized filing cabinets for our one collection event
        self.collection_data = {
            'features': {},  # layer_name -> list of feature arrays
            'labels': [],    # all labels from collection session
            'predictions': []  # all predictions from collection session
        }
        
        # Infrastructure for hook management
        self.layer_features = {}  # Temporary storage during forward pass
        self.layer_names = []     # Names of all layers we're monitoring
        self.hooks = []          # Forward hooks for feature extraction
        
        # Results after layer selection
        self.best_layer = None
        self.layer_ece_scores = {}
        self.layer_selection_complete = False
        
        # Initialize the hook system
        self._register_hooks()
        
        logger.info(f"🔍 LayerFeatureExtractor initialized for focused collection")
        logger.info(f"   Collection epoch: {collection_epoch}")
        logger.info(f"   Target batches: {min_batches_for_decision}")
        logger.info(f"   Registered hooks for {len(self.layer_names)} layers")
    
    def update_epoch(self, epoch: int):
        """
        Update the current epoch and manage collection timing.
        
        This is like a project manager checking if it's time to start
        the big data collection effort.
        
        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        
        # Log progress toward collection epoch
        if epoch < self.collection_epoch:
            epochs_remaining = self.collection_epoch - epoch
            if epoch % 5 == 0:  # Log every 5 epochs to avoid spam
                logger.info(f"⏳ Epoch {epoch}: {epochs_remaining} epochs until layer selection begins")
    
    def begin_focused_collection(self):
        """
        Start the focused collection process for layer selection.
        
        This is like announcing "attention everyone, we're now conducting
        our comprehensive survey - please participate in data collection."
        """
        if self.layer_selection_complete:
            logger.info(f"✅ Layer selection already complete, skipping collection")
            return
            
        logger.info(f"🎯 Beginning focused feature collection at epoch {self.current_epoch}")
        logger.info(f"   Target: {self.target_batches} batches")
        logger.info(f"   Will collect from {len(self.layer_names)} layers")
        logger.info(f"   Model representations should now be stable")
        
        self.focused_collection_active = True
        self.collected_batches = 0
        
        # Initialize clean collection storage
        # Like setting up fresh data collection forms
        self.collection_data = {
            'features': {layer: [] for layer in self.layer_names},
            'labels': [],
            'predictions': []
        }
    
    def collect_if_active(self, targets: torch.Tensor, logits: torch.Tensor):
        """
        Collect data only if focused collection is currently active.
        
        This replaces the complex continuous collection with simple,
        controlled data gathering during our planned collection window.
        
        Args:
            targets: True labels for this batch
            logits: Model outputs for this batch
        """
        # Quick exit if collection isn't active
        if not self.focused_collection_active:
            return
        
        # Check if we've collected enough data
        if self.collected_batches >= self.target_batches:
            logger.info(f"✅ Focused collection complete: {self.collected_batches} batches")
            self.focused_collection_active = False
            return
        
        # Extract ground truth information from this batch
        batch_labels = targets.cpu().numpy()
        probs = F.softmax(logits, dim=1)
        _, predictions = torch.max(probs, dim=1)
        batch_predictions = predictions.cpu().numpy()
        
        # Collect features from all layers that fired their hooks
        features_collected = 0
        batch_size = len(targets)
        
        for layer_name in self.layer_names:
            if layer_name in self.layer_features:
                features = self.layer_features[layer_name]
                
                # Validate that this layer gave us the right amount of data
                # This is like checking that each survey respondent filled out
                # the complete form before accepting their response
                if features.size(0) == batch_size:
                    features_cpu = features.detach().cpu().numpy()
                    self.collection_data['features'][layer_name].append(features_cpu)
                    features_collected += 1
                else:
                    logger.warning(f"⚠️ Size mismatch for {layer_name}: "
                                 f"expected {batch_size}, got {features.size(0)}")
        
        # Store the reference data (labels and predictions)
        self.collection_data['labels'].extend(batch_labels)
        self.collection_data['predictions'].extend(batch_predictions)
        
        self.collected_batches += 1
        
        # Progress reporting - like giving updates on survey completion
        if self.collected_batches % 10 == 0 or self.collected_batches <= 5:
            progress_pct = (self.collected_batches / self.target_batches) * 100
            logger.info(f"📊 Collection progress: {self.collected_batches}/{self.target_batches} "
                       f"batches ({progress_pct:.1f}%), {features_collected} layers per batch")
        
        # Clear temporary storage to prevent memory buildup
        self.layer_features.clear()
    
    def finalize_layer_selection(self) -> Optional[str]:
        """
        Analyze all collected data and select the best layer.
        
        This is like analyzing all the survey responses to make an informed
        decision. Since all data came from the same controlled collection
        session, we don't have synchronization issues.
        
        Returns:
            Name of the best layer, or None if selection failed
        """
        logger.info(f"🔍 Finalizing layer selection from focused collection")
        
        # Basic validation of collected data
        if not self.collection_data['labels']:
            logger.error(f"❌ No data collected during focused collection")
            return None
        
        # Convert collected data to analysis-ready format
        all_labels = np.array(self.collection_data['labels'])
        all_predictions = np.array(self.collection_data['predictions'])
        
        logger.info(f"📊 Collected data summary:")
        logger.info(f"   Total samples: {len(all_labels)}")
        logger.info(f"   From {self.collected_batches} batches")
        logger.info(f"   Unique labels: {len(np.unique(all_labels))}")
        logger.info(f"   Labels range: [{all_labels.min()}, {all_labels.max()}]")
        
        # Evaluate each layer that has collected data
        best_ece = float('inf')
        best_layer = None
        successful_evaluations = 0
        failed_evaluations = 0
        
        for layer_name in self.layer_names:
            # Skip layers that didn't collect any data
            if (layer_name not in self.collection_data['features'] or 
                not self.collection_data['features'][layer_name]):
                logger.info(f"⏭️ Skipping {layer_name}: no data collected")
                continue
            
            logger.info(f"🔍 Evaluating {layer_name}...")
            
            try:
                # Combine all feature batches for this layer
                layer_batches = self.collection_data['features'][layer_name]
                layer_features = np.concatenate(layer_batches, axis=0)
                
                # Validation check - since everything came from the same collection
                # session, the sizes should match perfectly
                if len(layer_features) != len(all_labels):
                    logger.warning(f"   ❌ Size mismatch: {len(layer_features)} vs {len(all_labels)}")
                    failed_evaluations += 1
                    continue
                
                # Compute confidence scores based on feature magnitudes
                # This is a simple but effective approach: layers that produce
                # more discriminative features tend to have different magnitudes
                # for different classes
                feature_norms = np.linalg.norm(layer_features, axis=1)
                norm_range = feature_norms.max() - feature_norms.min()
                
                # Handle edge case where all features have the same magnitude
                if norm_range < 1e-10:
                    logger.warning(f"   ⚠️ Features have uniform magnitude - using default confidence")
                    confidences = np.ones_like(feature_norms) * 0.5
                else:
                    # Normalize feature magnitudes to [0,1] range as confidence scores
                    confidences = (feature_norms - feature_norms.min()) / norm_range
                
                # Apply robust clamping to prevent numerical issues in ECE calculation
                # This is like adding safety margins to prevent edge cases
                epsilon = 1e-7
                confidences_clamped = np.clip(confidences, epsilon, 1.0 - epsilon)
                
                # Compute Expected Calibration Error (ECE)
                # ECE measures how well the confidence scores match actual accuracy
                ece = expected_calibration_error(confidences_clamped, all_predictions, all_labels)
                
                # Validate the ECE result
                if np.isnan(ece) or np.isinf(ece) or ece < 0 or ece > 1:
                    logger.warning(f"   ⚠️ Invalid ECE value: {ece}")
                    failed_evaluations += 1
                    continue
                
                # Store successful evaluation
                self.layer_ece_scores[layer_name] = ece
                successful_evaluations += 1
                
                logger.info(f"   ✅ ECE: {ece:.6f}")
                logger.info(f"      Feature shape: {layer_features.shape}")
                logger.info(f"      Norm range: [{feature_norms.min():.3f}, {feature_norms.max():.3f}]")
                
                # Track the best layer (lowest ECE is better)
                if ece < best_ece:
                    best_ece = ece
                    best_layer = layer_name
                    logger.info(f"   🏆 New best layer!")
                    
            except Exception as e:
                logger.error(f"   ❌ Evaluation failed: {e}")
                failed_evaluations += 1
                continue
        
        # Report final results
        total_attempts = successful_evaluations + failed_evaluations
        logger.info(f"🏁 Layer selection results:")
        logger.info(f"   ✅ Successful evaluations: {successful_evaluations}")
        logger.info(f"   ❌ Failed evaluations: {failed_evaluations}")
        if total_attempts > 0:
            success_rate = (successful_evaluations / total_attempts) * 100
            logger.info(f"   📊 Success rate: {success_rate:.1f}%")
        
        if best_layer:
            self.best_layer = best_layer
            self.layer_selection_complete = True
            logger.info(f"   🎯 Selected best layer: {best_layer} (ECE: {best_ece:.6f})")
            
            # Show comparison with other layers
            if len(self.layer_ece_scores) > 1:
                sorted_layers = sorted(self.layer_ece_scores.items(), key=lambda x: x[1])
                num_to_show = min(5, len(sorted_layers))
                logger.info(f"   📊 Top {num_to_show} layers by ECE:")
                for rank, (name, ece) in enumerate(sorted_layers[:num_to_show], 1):
                    status = "🏆" if name == best_layer else f"#{rank}"
                    logger.info(f"      {status} {name}: {ece:.6f}")
            
            # Clean up collection data to free memory
            self._cleanup_collection_data()
            
        else:
            logger.warning(f"   ⚠️ No valid layers found - all evaluations failed")
            logger.warning(f"   This suggests an issue with feature extraction or model architecture")
        
        return best_layer
    
    def _cleanup_collection_data(self):
        """
        Clean up collection data to free memory after layer selection.
        
        This is like disposing of survey forms after the analysis is complete.
        """
        logger.info(f"🧹 Cleaning up collection data to free memory")
        
        # Calculate approximate memory being freed
        total_samples = len(self.collection_data['labels'])
        num_layers = len([k for k, v in self.collection_data['features'].items() if v])
        
        # Clear the collection data
        self.collection_data = {
            'features': {},
            'labels': [],
            'predictions': []
        }
        
        logger.info(f"   Freed data from {total_samples} samples across {num_layers} layers")
        
        # Force garbage collection to actually free the memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_current_features(self) -> Optional[torch.Tensor]:
        """
        Get features from the currently selected best layer.
        
        This is called during normal training after layer selection is complete.
        
        Returns:
            Features from the best layer, or None if no layer selected
        """
        if not self.best_layer or self.best_layer not in self.layer_features:
            return None
        return self.layer_features[self.best_layer]
    
    def get_layer_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the layer selection process.
        
        This is like generating a report about our survey and analysis results.
        
        Returns:
            Dictionary with layer selection information
        """
        return {
            'selected_layer': self.best_layer,
            'layer_ece_scores': self.layer_ece_scores.copy(),
            'selection_complete': self.layer_selection_complete,
            'available_layers': self.layer_names.copy(),
            'collection_epoch': self.collection_epoch,
            'current_epoch': self.current_epoch,
            'collection_active': self.focused_collection_active,
            'collected_batches': self.collected_batches,
            'target_batches': self.target_batches
        }
    
    def _register_hooks(self):
        """
        Register forward hooks on relevant layers for feature extraction.
        
        This is like installing sensors throughout a building to monitor activity.
        We want to capture features from layers that are likely to be good
        representation points.
        """
        logger.info(f"🔎 Scanning model architecture for feature extraction points...")
        
        # Detect what type of architecture we're working with
        architecture_type = self._detect_architecture_type()
        logger.info(f"   🏗️ Detected architecture: {architecture_type}")
        
        # Get all modules in the model
        all_modules = list(self.model.named_modules())
        logger.info(f"   📊 Total modules: {len(all_modules)}")
        
        # Find potential layers based on architecture
        if architecture_type == 'resnet':
            potential_layers = self._find_resnet_layers(all_modules)
        elif architecture_type == 'densenet':
            potential_layers = self._find_densenet_layers(all_modules)
        else:
            potential_layers = self._find_generic_layers(all_modules)
        
        logger.info(f"🎯 Found {len(potential_layers)} potential feature extraction layers")
        
        # Register hooks on selected layers
        for name, module in potential_layers:
            hook = module.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)
            self.layer_names.append(name)
            logger.info(f"   ✅ Registered hook for {name}")
        
        logger.info(f"🔗 Successfully registered {len(self.hooks)} feature extraction hooks")
        
        if len(self.hooks) == 0:
            logger.error(f"❌ CRITICAL: No hooks registered!")
            logger.error(f"   Architecture: {architecture_type}")
            logger.error(f"   This will prevent feature extraction from working")
    
    def _detect_architecture_type(self):
        """
        Detect the neural network architecture type.
        
        This is like identifying what kind of building you're in so you know
        where to look for the important structural elements.
        """
        model_class_name = self.model.__class__.__name__.lower()
        
        # Check class name first (most reliable)
        if 'resnet' in model_class_name:
            return 'resnet'
        elif 'densenet' in model_class_name:
            return 'densenet'
        elif 'vgg' in model_class_name:
            return 'vgg'
        elif 'inception' in model_class_name:
            return 'inception'
        
        # If class name doesn't help, analyze module structure
        module_names = [name for name, _ in self.model.named_modules()]
        
        # Look for characteristic naming patterns
        if any('layer1' in name or 'layer2' in name for name in module_names):
            return 'resnet'
        elif any('denseblock' in name or 'transition' in name for name in module_names):
            return 'densenet'
        elif any('features.' in name and any('classifier' in n for n in module_names) for name in module_names):
            return 'vgg'
        
        return 'generic'
    
    def _find_resnet_layers(self, all_modules):
        """Find good feature extraction points in ResNet architectures."""
        potential_layers = []
        
        for name, module in all_modules:
            # Skip the final classifier
            if isinstance(module, nn.Linear) and 'fc' in name.lower():
                continue
            
            # Look for ResNet-specific layer patterns
            resnet_patterns = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'shortcut']
            if any(pattern in name.lower() for pattern in resnet_patterns):
                if isinstance(module, (nn.Sequential, nn.AdaptiveAvgPool2d, nn.AvgPool2d, 
                                     nn.MaxPool2d, nn.BatchNorm2d)):
                    potential_layers.append((name, module))
        
        return potential_layers
    
    def _find_densenet_layers(self, all_modules):
        """Find good feature extraction points in DenseNet architectures."""
        potential_layers = []
        
        for name, module in all_modules:
            # Skip classifier layers
            if isinstance(module, nn.Linear) and ('classifier' in name.lower() or 'fc' in name.lower()):
                continue
            
            # Look for DenseNet-specific patterns
            densenet_patterns = ['features', 'denseblock', 'transition', 'norm']
            if any(pattern in name.lower() for pattern in densenet_patterns):
                if isinstance(module, (nn.Sequential, nn.BatchNorm2d, nn.ReLU, 
                                     nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                    potential_layers.append((name, module))
            
            # Include pooling layers
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
                potential_layers.append((name, module))
        
        return potential_layers
    
    def _find_generic_layers(self, all_modules):
        """Find feature extraction points for unknown architectures."""
        potential_layers = []
        
        for name, module in all_modules:
            # Skip obvious classifier layers
            if isinstance(module, nn.Linear):
                continue
            
            # Include various layer types that might have useful features
            if isinstance(module, (nn.Sequential, nn.BatchNorm2d, nn.ReLU, 
                                 nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d, nn.Conv2d)):
                potential_layers.append((name, module))
        
        return potential_layers
    
    def _make_hook(self, layer_name: str):
        """
        Create a forward hook for a specific layer.
        
        This is like creating a specific sensor that activates when
        data flows through a particular layer.
        """
        def hook_fn(module, input, output):
            # Only capture features if we're actively collecting
            if self.focused_collection_active:
                if isinstance(output, torch.Tensor):
                    # Flatten multi-dimensional outputs to 2D
                    if len(output.shape) > 2:
                        features = torch.flatten(output, 1)
                    else:
                        features = output
                    
                    # Store features temporarily (will be processed in collect_if_active)
                    self.layer_features[layer_name] = features
        
        return hook_fn
    
    def cleanup(self):
        """
        Remove all hooks and clean up resources.
        
        This is like removing all the sensors when the monitoring is complete.
        """
        logger.info(f"🧹 Cleaning up LayerFeatureExtractor")
        
        # Remove all forward hooks
        for hook in self.hooks:
            hook.remove()
        
        # Clear all data structures
        self.hooks.clear()
        self.layer_features.clear()
        self.collection_data = {'features': {}, 'labels': [], 'predictions': []}
        
        logger.info(f"   Removed {len(self.hooks)} hooks and cleared all data")

    def evaluate_layers_and_select_best(self, min_samples: int = 200) -> Optional[str]:
        """
        Evaluate all layers and select the best one based on geometric separation.
        
        Args:
            min_samples: Minimum number of samples needed for evaluation
            
        Returns:
            str: Name of the best layer, or None if selection failed
        """
        if not self.collection_data['labels']:
            logger.warning("No data collected for layer evaluation")
            return None
            
        if len(self.collection_data['labels']) < min_samples:
            logger.warning(f"Not enough samples for layer evaluation: {len(self.collection_data['labels'])} < {min_samples}")
            return None
        
        logger.info(f"Evaluating {len(self.layer_names)} layers for selection")
        
        # Convert collected data to numpy arrays
        labels = np.array(self.collection_data['labels'])
        predictions = np.array(self.collection_data['predictions'])
        
        # Score each layer
        layer_scores = {}
        for layer_name in self.layer_names:
            if layer_name not in self.collection_data['features']:
                continue
                
            # Get features for this layer
            features = np.concatenate(self.collection_data['features'][layer_name], axis=0)
            
            # Compute pairwise distances
            distances = np.zeros((len(features), len(features)))
            for i in range(len(features)):
                distances[i] = np.linalg.norm(features[i] - features, axis=1)
            
            # Compute separation scores
            separation_scores = []
            for i in range(len(features)):
                # Find distances to same class and different class samples
                same_class_mask = (labels == labels[i])
                diff_class_mask = ~same_class_mask
                
                if np.sum(same_class_mask) > 1 and np.sum(diff_class_mask) > 0:
                    # Get minimum distances
                    d_same = np.min(distances[i, same_class_mask & (np.arange(len(features)) != i)])
                    d_other = np.min(distances[i, diff_class_mask])
                    
                    # Compute separation score (Fast-Separation metric)
                    separation_score = (d_other - d_same) / 2.0
                    separation_scores.append(separation_score)
            
            if separation_scores:
                # Compute final layer score
                mean_score = np.mean(separation_scores)
                score_variance = np.var(separation_scores)
                
                # Penalize high variance
                layer_score = mean_score / (1.0 + score_variance)
                layer_scores[layer_name] = layer_score
                
                logger.info(f"Layer {layer_name}: score = {layer_score:.4f}")
        
        if not layer_scores:
            logger.warning("No valid layer scores computed")
            return None
        
        # Select best layer
        best_layer = max(layer_scores.items(), key=lambda x: x[1])[0]
        best_score = layer_scores[best_layer]
        
        # Store scores for later reference
        self.layer_ece_scores = layer_scores
        
        logger.info(f"Layer selection results:")
        for layer_name, score in sorted(layer_scores.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {layer_name}: {score:.4f}")
        logger.info(f"Selected layer: {best_layer} (score: {best_score:.4f})")
        
        return best_layer


# Test the focused collection implementation
if __name__ == "__main__":
    # Simple test with a dummy model
    import torch.nn as nn
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128)
            )
            self.avgpool = nn.AdaptiveAvgPool2d(4)
            self.classifier = nn.Linear(128 * 16, 10)
            
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    # Test the focused collection approach
    model = TestModel()
    extractor = LayerFeatureExtractor(model, collection_epoch=2, min_batches_for_decision=5)
    
    print(f"Registered layers: {extractor.layer_names}")
    
    # Simulate training epochs
    for epoch in range(5):
        extractor.update_epoch(epoch)
        
        if epoch == 2:  # Collection epoch
            extractor.begin_focused_collection()
        
        # Simulate batches in this epoch
        for batch in range(10):
            x = torch.randn(8, 3, 32, 32)
            targets = torch.randint(0, 10, (8,))
            logits = model(x)
            
            # Try to collect data
            extractor.collect_if_active(targets, logits)
        
        if epoch == 3:  # Finalize selection
            best_layer = extractor.finalize_layer_selection()
            print(f"Best layer selected: {best_layer}")
            print(f"Layer info: {extractor.get_layer_info()}")
            break
    
    extractor.cleanup()