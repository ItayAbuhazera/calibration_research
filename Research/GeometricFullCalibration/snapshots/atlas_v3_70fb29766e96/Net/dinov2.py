import torch
import torch.nn as nn
import torch.optim as optim
from transformers import Dinov2Model, Dinov2Config
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def get_dinov2_optimizer_and_scheduler(model, config):
    """
    Create optimized optimizer and scheduler for DINOv2 from-scratch training.
    
    Args:
        model: DINOv2 model
        config: Training configuration dictionary
    
    Returns:
        optimizer: AdamW optimizer with proper learning rates
        scheduler: Combined warmup + cosine annealing scheduler
    """
    # Separate parameters for backbone and classifier
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
    
    # Learning rates for from-scratch training
    backbone_lr = config.get('backbone_lr', 5e-4)
    classifier_lr = config.get('classifier_lr', 1e-3)
    weight_decay = config.get('weight_decay', 0.05)
    
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': classifier_params, 'lr': classifier_lr, 'weight_decay': weight_decay}
    ]
    
    # Use AdamW optimizer (better for transformers)
    optimizer = optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Create scheduler with warmup
    epochs = config.get('epochs', 300)
    warmup_epochs = config.get('warmup_epochs', 20)
    
    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    
    # Main scheduler (cosine annealing)
    main_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=epochs - warmup_epochs,
        eta_min=1e-6
    )
    
    # Combined scheduler: warmup then cosine annealing
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    print(f"DINOv2 Optimizer Setup:")
    print(f"   Backbone LR: {backbone_lr}")
    print(f"   Classifier LR: {classifier_lr}")
    print(f"   Weight decay: {weight_decay}")
    print(f"   Warmup epochs: {warmup_epochs}")
    print(f"   Total epochs: {epochs}")
    print(f"   Backbone params: {sum(p.numel() for p in backbone_params):,}")
    print(f"   Classifier params: {sum(p.numel() for p in classifier_params):,}")
    
    return optimizer, scheduler

class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes=10, model_size='small', freeze_backbone=True, use_pretrained=True):
        super().__init__()
        self.model_size = model_size
        self.freeze_backbone = freeze_backbone
        self.use_pretrained = use_pretrained
        
        # Feature dimensions by model size
        feature_dims = {
            'small': 384,
            'base': 768, 
            'large': 1024,
            'giant': 1536
        }
        
        self.feature_dim = feature_dims[model_size]
        
        if use_pretrained:
            # Load pretrained DINOv2
            model_name = f"facebook/dinov2-{model_size}"
            self.backbone = Dinov2Model.from_pretrained(model_name)
            
            # Freeze backbone if specified
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        else:
            # Create from-scratch DINOv2 with same architecture
            print(f"Creating DINOv2-{model_size} from scratch for CIFAR-10 training")
            
            # Load config but not weights
            model_name = f"facebook/dinov2-{model_size}"
            config = Dinov2Config.from_pretrained(model_name)
            
            # Initialize model with random weights
            self.backbone = Dinov2Model(config)
            
            # All parameters trainable for from-scratch training
            for param in self.backbone.parameters():
                param.requires_grad = True
                
            print(f"Initialized DINOv2-{model_size} with random weights")
            print(f"   Trainable parameters: {sum(p.numel() for p in self.backbone.parameters() if p.requires_grad):,}")
        
        # Classification head (always trainable)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        # For feature extraction hooks
        self.feature_hook_layer = None
        self.extracted_features = None
        
    def forward(self, x, return_features=False):
        # DINOv2 expects [B, 3, 224, 224] - handle resizing
        if x.size(-1) != 224:
            x = nn.functional.interpolate(x, size=224, mode='bilinear', align_corners=False)
        
        # Extract features from backbone
        outputs = self.backbone(x)
        features = outputs.last_hidden_state[:, 0]  # CLS token
        
        # Classification
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def get_feature_extraction_layer(self):
        """Return the layer for feature extraction (before classifier)"""
        return self.backbone.layernorm  # Final layer norm before CLS token
    
    def get_training_info(self):
        """Get information about model training setup"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        
        return {
            'model_size': self.model_size,
            'use_pretrained': self.use_pretrained,
            'freeze_backbone': self.freeze_backbone,
            'backbone_trainable_params': backbone_params,
            'classifier_trainable_params': classifier_params,
            'total_trainable_params': backbone_params + classifier_params
        }
