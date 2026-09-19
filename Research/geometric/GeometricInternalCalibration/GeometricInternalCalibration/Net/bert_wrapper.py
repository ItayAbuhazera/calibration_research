# models/bert_wrapper.py
"""
BERT model wrapper for geometric calibration experiments
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification
from typing import Dict, Optional

class BERTWrapper(nn.Module):
    """Wrapper for BERT that exposes intermediate layers for geometric calibration"""
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased',
                 num_classes: int = 2,
                 fine_tune: bool = True):
        super().__init__()
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(model_name)
        self.num_layers = self.bert.config.num_hidden_layers
        self.hidden_size = self.bert.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Fine-tuning settings
        if not fine_tune:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Storage for intermediate representations
        self.intermediate_outputs = {}
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                output_hidden_states: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with intermediate layer outputs
        
        Returns:
            Dict containing:
                - logits: Classification logits
                - hidden_states: All hidden states if requested
                - pooled_output: [CLS] token representation
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
        )
        
        # Get pooled output for classification
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        result = {'logits': logits}
        
        if output_hidden_states:
            # Store all hidden states
            result['hidden_states'] = outputs.hidden_states
            result['pooled_output'] = pooled_output
        
        return result
    
    def extract_features(self, input_ids: torch.Tensor, 
                        attention_mask: torch.Tensor,
                        layer_idx: int = -1) -> torch.Tensor:
        """Extract features from a specific layer"""
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Get features from specified layer
            hidden_states = outputs.hidden_states[layer_idx]
            
            # Use [CLS] token representation
            features = hidden_states[:, 0, :]
            
        return features