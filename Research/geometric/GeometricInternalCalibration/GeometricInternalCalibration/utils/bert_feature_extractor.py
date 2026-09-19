# utils/bert_feature_extractor.py
"""
Feature extractor for BERT models compatible with geometric calibration
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict
import logging

from utils.logging_config import get_logger
logger = get_logger(__name__)

class BERTFeatureExtractor:
    """Feature extractor for BERT that handles text data"""
    
    def __init__(self, model, forced_layer: Optional[int] = None):
        self.model = model
        self.forced_layer = forced_layer
        self.device = next(model.parameters()).device
        
        # BERT typically has 12 layers
        self.num_layers = 12
        self.features = {}
        self.hooks = []
        
    def extract_features(self, data_loader, layer_indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Extract features from specified layers
        
        Args:
            data_loader: DataLoader for text data
            layer_indices: List of layer indices to extract (0-11 for BERT-base)
            
        Returns:
            Dictionary mapping layer names to features
        """
        if layer_indices is None:
            layer_indices = [self.forced_layer] if self.forced_layer else [8]  # Default to layer 8
        
        self.model.eval()
        all_features = {f'layer_{idx}': [] for idx in layer_indices}
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                # Get BERT outputs with all hidden states
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Extract features from specified layers
                hidden_states = outputs['hidden_states']
                
                for idx in layer_indices:
                    # Use [CLS] token representation
                    layer_features = hidden_states[idx][:, 0, :].cpu().numpy()
                    all_features[f'layer_{idx}'].append(layer_features)
                
                all_labels.append(labels.numpy())
        
        # Concatenate all batches
        for key in all_features:
            all_features[key] = np.concatenate(all_features[key], axis=0)
        
        all_labels = np.concatenate(all_labels, axis=0)
        
        return all_features, all_labels
    
    def extract_single_layer(self, data_loader, layer_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract features from a single layer (compatible with existing interface)"""
        features_dict, labels = self.extract_features(data_loader, [layer_idx])
        
        # Get logits as well
        all_logits = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits'].cpu().numpy()
                all_logits.append(logits)
        
        all_logits = np.concatenate(all_logits, axis=0)
        features = features_dict[f'layer_{layer_idx}']
        
        return features, all_logits, labels