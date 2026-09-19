# Experiments/bert_trainer.py
"""
BERT Fine-tuning Trainer for Semantic-Space Geometric Calibration
================================================================

Integrates BERT fine-tuning with the existing training infrastructure,
following the same patterns as vision experiments but adapted for NLP.

This trainer:
1. Fine-tunes BERT on SST-2 using standard cross-entropy
2. Extracts features from all BERT layers for calibration
3. Integrates with existing calibration pipeline
4. Supports the same training methods as vision experiments
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import BERT components
from Data.sst2 import get_sst2_loaders, SST2Dataset
from Net.bert_wrapper import BERTWrapper

# Import base trainer for consistency
from Experiments.run_single_experiment import SingleExperimentRunner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from utils.logging_config import get_logger
logger = get_logger(__name__)


class BERTTrainer(SingleExperimentRunner):
    """
    BERT trainer that extends the existing SingleExperimentRunner
    to maintain consistency with vision experiments
    """
    
    def __init__(self, 
                 method: str = 'baseline_cross_entropy',
                 dataset: str = 'sst2',
                 model_name: str = 'bert-base-uncased',
                 seed: int = 42,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 **config):
        """
        Initialize BERT trainer
        
        Args:
            method: Training method (baseline_cross_entropy, etc.)
            dataset: Dataset name (sst2)
            model_name: BERT model name
            seed: Random seed
            device: Device to use
            **config: Additional configuration
        """
        # Initialize parent class
        super().__init__(
            method=method,
            dataset=dataset,
            model=model_name,
            seed=seed,
            device=device,
            **config
        )
        
        # BERT-specific configuration
        self.model_name = model_name
        self.num_classes = 2  # SST-2 binary classification
        self.max_length = config.get('max_length', 128)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.weight_decay = config.get('weight_decay', 0.01)
        
        # Override parent class attributes for BERT
        self.use_geometric_calibration = method in [
            'constellation', 'geometric_focal_calibration', 
            'ce_fast_separation', 'augmix_constellation'
        ]
        
        logger.info(f"Initialized BERT trainer: {method} on {dataset}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Use geometric calibration: {self.use_geometric_calibration}")
    
    def get_data_loaders(self, batch_size: int = 16) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get SST-2 data loaders
        
        Args:
            batch_size: Batch size for data loaders
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        return get_sst2_loaders(
            batch_size=batch_size,
            seed=self.seed,
            tokenizer_name=self.model_name,
            max_length=self.max_length
        )
    
    def create_model(self) -> BERTWrapper:
        """
        Create BERT model
        
        Returns:
            BERTWrapper model
        """
        model = BERTWrapper(
            model_name=self.model_name,
            num_classes=self.num_classes,
            fine_tune=True
        )
        
        return model
    
    def create_optimizer(self, model: nn.Module, train_loader: DataLoader, epochs: int) -> Tuple[optim.Optimizer, Any]:
        """
        Create optimizer and scheduler for BERT fine-tuning
        
        Args:
            model: BERT model
            train_loader: Training data loader
            epochs: Number of training epochs
            
        Returns:
            Tuple of (optimizer, scheduler)
        """
        # Use AdamW optimizer (recommended for transformers)
        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Linear warmup scheduler
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    def get_loss_function(self):
        """
        Get loss function based on training method
        
        Returns:
            Loss function
        """
        if self.method == 'baseline_cross_entropy':
            return lambda logits, targets, **kwargs: F.cross_entropy(logits, targets)
        elif self.method == 'baseline_focal':
            gamma = self.config.get('gamma', 2.0)
            def focal_loss(logits, targets, **kwargs):
                ce_loss = F.cross_entropy(logits, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = (1 - pt) ** gamma * ce_loss
                return focal_loss.mean()
            return focal_loss
        elif self.method == 'baseline_brier':
            def brier_loss(logits, targets, **kwargs):
                probs = F.softmax(logits, dim=1)
                targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
                return torch.mean(torch.sum((probs - targets_one_hot) ** 2, dim=1))
            return brier_loss
        else:
            # Default to cross-entropy for other methods
            logger.warning(f"Using cross-entropy loss for method: {self.method}")
            return lambda logits, targets, **kwargs: F.cross_entropy(logits, targets)
    
    def train_epoch_bert(self, 
                        model: nn.Module, 
                        train_loader: DataLoader, 
                        optimizer: optim.Optimizer, 
                        scheduler: Any,
                        loss_fn) -> Tuple[float, float]:
        """
        Train BERT for one epoch
        
        Args:
            model: BERT model
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            loss_fn: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask, output_hidden_states=self.use_geometric_calibration)
            logits = outputs['logits']
            
            # Calculate loss
            if self.use_geometric_calibration and 'pooled_output' in outputs:
                # For geometric methods, pass features to loss function
                loss = loss_fn(logits, labels, features=outputs['pooled_output'])
            else:
                loss = loss_fn(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def evaluate_epoch_bert(self, 
                           model: nn.Module, 
                           eval_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate BERT for one epoch
        
        Args:
            model: BERT model
            eval_loader: Evaluation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
                logits = outputs['logits']
                
                # Calculate loss
                loss = F.cross_entropy(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(eval_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train_model(self, 
                   epochs: int = 3, 
                   batch_size: int = 16,
                   save_model: bool = True) -> Tuple[nn.Module, Dict, DataLoader]:
        """
        Train BERT model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            save_model: Whether to save the model
            
        Returns:
            Tuple of (trained_model, training_history, test_loader)
        """
        logger.info(f"🚀 Starting BERT training: {self.method}")
        
        # Get data loaders
        train_loader, val_loader, test_loader = self.get_data_loaders(batch_size)
        
        # Create model
        model = self.create_model()
        model.to(self.device)
        
        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer(model, train_loader, epochs)
        
        # Get loss function
        loss_fn = self.get_loss_function()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0,
            'best_epoch': 0
        }
        
        best_val_acc = 0.0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch_bert(
                model, train_loader, optimizer, scheduler, loss_fn
            )
            
            # Validation phase
            val_loss, val_acc = self.evaluate_epoch_bert(model, val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                history['best_val_acc'] = best_val_acc
                history['best_epoch'] = epoch
                best_model_state = model.state_dict().copy()
                self.best_val_acc = best_val_acc
                self.best_epoch = epoch
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Save model
        if save_model:
            model_path = self.exp_dir / f'model_best.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'model_name': self.model_name,
                    'num_classes': self.num_classes,
                    'fine_tune': True
                },
                'training_config': {
                    'method': self.method,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': self.learning_rate,
                    'seed': self.seed
                },
                'history': history
            }, model_path)
            logger.info(f"Model saved to {model_path}")
        
        logger.info(f"✅ Training complete! Best val accuracy: {best_val_acc:.2f}%")
        return model, history, test_loader
    
    def evaluate_model(self, 
                      model: nn.Module, 
                      test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate trained BERT model
        
        Args:
            model: Trained BERT model
            test_loader: Test data loader
            
        Returns:
            Evaluation results
        """
        logger.info("🔍 Evaluating BERT model")
        
        model.eval()
        all_logits = []
        all_labels = []
        all_features = {}  # Store features from different layers
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                # Get model outputs with hidden states
                outputs = model(input_ids, attention_mask, output_hidden_states=True)
                logits = outputs['logits']
                hidden_states = outputs['hidden_states']
                pooled_output = outputs['pooled_output']
                
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.numpy())
                
                # Extract features from different layers for calibration
                for i, hidden_state in enumerate(hidden_states[1:]):  # Skip embedding layer
                    layer_name = f'layer_{i}'
                    if layer_name not in all_features:
                        all_features[layer_name] = []
                    # Use [CLS] token representation
                    all_features[layer_name].append(hidden_state[:, 0, :].cpu().numpy())
                
                # Add pooler features
                if 'pooler' not in all_features:
                    all_features['pooler'] = []
                all_features['pooler'].append(pooled_output.cpu().numpy())
        
        # Concatenate all batches
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        for layer_name in all_features:
            all_features[layer_name] = np.concatenate(all_features[layer_name], axis=0)
        
        # Calculate metrics
        predictions = np.argmax(logits, axis=1)
        accuracy = accuracy_score(labels, predictions)
        
        # Calculate ECE using existing metric
        from Metrics.metrics import expected_calibration_error
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        ece = expected_calibration_error(labels, probs)
        
        eval_results = {
            'accuracy': accuracy,
            'ece': ece,
            'logits': logits,
            'labels': labels,
            'features': all_features,
            'num_samples': len(labels)
        }
        
        logger.info(f"Test Accuracy: {accuracy:.2f}%")
        logger.info(f"Test ECE: {ece:.4f}")
        logger.info(f"Features extracted from {len(all_features)} layers")
        
        return eval_results


def create_bert_experiment(method: str, 
                          model_name: str = 'bert-base-uncased',
                          seed: int = 42,
                          epochs: int = 3,
                          batch_size: int = 16,
                          **config) -> Dict[str, Any]:
    """
    Create and run a complete BERT experiment
    
    Args:
        method: Training method
        model_name: BERT model name
        seed: Random seed
        epochs: Number of training epochs
        batch_size: Batch size
        **config: Additional configuration
        
    Returns:
        Complete experiment results
    """
    # Initialize trainer
    trainer = BERTTrainer(
        method=method,
        dataset='sst2',
        model_name=model_name,
        seed=seed,
        **config
    )
    
    # Run experiment
    results = trainer.run_experiment(
        epochs=epochs,
        batch_size=batch_size,
        evaluate_only=False
    )
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Test BERT trainer
    logging.basicConfig(level=logging.INFO)
    
    # Quick test experiment
    results = create_bert_experiment(
        method='baseline_cross_entropy',
        model_name='bert-base-uncased',
        seed=42,
        epochs=1,  # Quick test
        batch_size=8
    )
    
    print("✅ BERT trainer test complete!")
    print(f"Test accuracy: {results['evaluation_results']['accuracy']:.2f}%")
    print(f"Test ECE: {results['evaluation_results']['ece']:.4f}")
    print(f"Extracted features from {len(results['evaluation_results']['features'])} layers")