"""
Integration module for geometric calibration experiments
Works with existing training infrastructure
"""

import torch
import torch.nn as nn
from Losses.geometric_calibration_layer import create_geometric_model
from Losses.loss import get_loss_function, LOSS_CONFIGS
import logging

from utils.logging_config import get_logger
logger = get_logger(__name__)


class GeometricExperimentConfig:
    """Configuration for geometric calibration experiments"""
    
    GEOMETRIC_METHODS = {
        'none': 'No geometric calibration',
        'calibration_layer': 'Geometric calibration layer only',
        'constellation_loss': 'Constellation loss only', 
        'geometric_focal': 'Geometric focal loss only',
        'full_geometric': 'Calibration layer + constellation loss',
        'adaptive_geometric': 'Calibration layer + geometric focal loss'
    }
    
    @classmethod
    def get_available_methods(cls):
        """Get list of available geometric methods"""
        return list(cls.GEOMETRIC_METHODS.keys())
    
    @classmethod
    def get_method_description(cls, method):
        """Get description of geometric method"""
        return cls.GEOMETRIC_METHODS.get(method, "Unknown method")


def create_geometric_experiment_model(base_model, method='none', num_classes=10, **kwargs):
    """
    Create model with geometric calibration based on experiment method
    
    Args:
        base_model: Base neural network model
        method: Geometric calibration method
        num_classes: Number of output classes
        **kwargs: Additional parameters
    
    Returns:
        Configured model for experiment
    """
    architecture = kwargs.get('architecture', 'resnet')
    temp = kwargs.get('temp', 1.0)
    
    if method == 'none':
        # No modification - return base model
        return base_model
    
    elif method in ['calibration_layer', 'full_geometric', 'adaptive_geometric']:
        # Add geometric calibration layer
        geometric_model = create_geometric_model(
            base_model=base_model,
            num_classes=num_classes,
            architecture=architecture,
            temp=temp
        )
        return geometric_model
    
    else:
        # For loss-only methods, return base model (loss will be handled separately)
        return base_model


def create_geometric_experiment_loss(method='none', num_classes=10, **kwargs):
    """
    Create loss function based on geometric method
    
    Args:
        method: Geometric calibration method  
        num_classes: Number of classes
        **kwargs: Loss-specific parameters
    
    Returns:
        Loss function
    """
    device = kwargs.get('device', 'cuda')
    
    if method == 'none' or method == 'calibration_layer':
        # Standard cross-entropy loss
        return get_loss_function('cross_entropy')
    
    elif method == 'constellation_loss' or method == 'full_geometric':
        # Constellation loss
        loss_config = LOSS_CONFIGS.get('constellation', {})
        loss_config.update(kwargs)
        return get_loss_function('constellation', **loss_config)
    
    elif method == 'geometric_focal' or method == 'adaptive_geometric':
        # Geometric focal loss
        loss_config = LOSS_CONFIGS.get('geometric_focal', {})
        loss_config.update(kwargs)
        return get_loss_function('geometric_focal', **loss_config)
    
    else:
        raise ValueError(f"Unknown geometric method: {method}")


def run_geometric_experiment(base_model, 
                           train_loader, 
                           val_loader,
                           method='none',
                           num_classes=10,
                           epochs=10,
                           device='cuda',
                           **kwargs):
    """
    Run a complete geometric calibration experiment
    
    Args:
        base_model: Base neural network
        train_loader: Training data loader
        val_loader: Validation data loader  
        method: Geometric calibration method
        num_classes: Number of classes
        epochs: Number of training epochs
        device: Device for training
        **kwargs: Additional experiment parameters
    
    Returns:
        Dictionary with experiment results
    """
    # Create experimental model and loss
    model = create_geometric_experiment_model(
        base_model=base_model,
        method=method,
        num_classes=num_classes,
        architecture=kwargs.get('architecture', 'resnet'),
        **kwargs
    )
    
    loss_fn = create_geometric_experiment_loss(
        method=method,
        num_classes=num_classes,
        device=device,
        **kwargs
    )
    
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=kwargs.get('lr', 0.1),
        momentum=kwargs.get('momentum', 0.9),
        weight_decay=kwargs.get('weight_decay', 5e-4)
    )
    
    # Training loop
    results = {
        'method': method,
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': [],
        'calibration_stats': []
    }
    
    for epoch in range(epochs):
        # Training phase
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, device, method
        )
        
        # Validation phase  
        val_loss, val_acc, cal_stats = validate_epoch(
            model, val_loader, loss_fn, device, method
        )
        
        # Record results
        results['train_losses'].append(train_loss)
        results['train_accuracies'].append(train_acc)
        results['val_losses'].append(val_loss)
        results['val_accuracies'].append(val_acc)
        results['calibration_stats'].append(cal_stats)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs} - "
                       f"Train: {train_loss:.4f}/{train_acc:.4f} - "
                       f"Val: {val_loss:.4f}/{val_acc:.4f}")
    
    return results


def train_epoch(model, data_loader, loss_fn, optimizer, device, method):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Handle different model types
        if method in ['calibration_layer', 'full_geometric', 'adaptive_geometric']:
            # Geometric model - may return components
            outputs = model(data, targets=targets)
            if isinstance(outputs, dict):
                logits = outputs['logits']
                features = outputs.get('features', None)
            else:
                logits = outputs
                features = None
        else:
            # Standard model
            logits = model(data)
            features = None
        
        # Compute loss
        if features is not None and method in ['constellation_loss', 'full_geometric', 'geometric_focal', 'adaptive_geometric']:
            loss = loss_fn(logits, targets, features=features)
        else:
            loss = loss_fn(logits, targets)
        
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(data_loader), 100.0 * correct / total


def validate_epoch(model, data_loader, loss_fn, device, method):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    calibration_stats = {}
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            
            # Handle different model types
            if method in ['calibration_layer', 'full_geometric', 'adaptive_geometric']: 
                outputs = model(data, return_components=True)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                    features = outputs.get('features', None)
                    calibration_stats = outputs.get('calibration_stats', {})
                else:
                    logits = outputs
                    features = None
            else:
                logits = model(data)
                features = None
            
            # Compute loss
            if features is not None and method in ['constellation_loss', 'full_geometric', 'geometric_focal', 'adaptive_geometric']:
                loss = loss_fn(logits, targets, features=features)
            else:
                loss = loss_fn(logits, targets)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(data_loader), 100.0 * correct / total, calibration_stats


# Example usage and testing
def test_geometric_integration():
    """Test geometric integration with dummy data"""
    print("Testing Geometric Integration...")
    
    # Dummy model (replace with actual model)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    base_model = DummyModel()
    
    # Test different methods
    methods = ['none', 'calibration_layer', 'constellation_loss', 'full_geometric']
    
    for method in methods:
        try:
            print(f"\nTesting method: {method}")
            model = create_geometric_experiment_model(base_model, method=method)
            loss_fn = create_geometric_experiment_loss(method=method)
            print(f"✅ Method {method} created successfully")
        except Exception as e:
            print(f"❌ Method {method} failed: {e}")
    
    print("\nGeometric integration test completed!")


if __name__ == "__main__":
    test_geometric_integration()