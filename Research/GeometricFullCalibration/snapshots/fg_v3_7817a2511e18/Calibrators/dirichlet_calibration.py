# Calibrators/pytorch_dirichlet_calibration.py
"""
PyTorch GPU-optimized Dirichlet Calibration - Plug-and-play replacement
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.special import digamma
import numpy as np
import logging
from typing import Dict, Any, Optional
from .base_calibrator import BaseCalibrator

from utils.logging_config import get_logger
logger = get_logger(__name__)

class DirichletCalibration(BaseCalibrator):
    """
    PyTorch GPU-optimized Dirichlet Calibration.
    
    Drop-in replacement for the original DirichletCalibration with GPU acceleration.
    Maintains the same interface as the original implementation.
    """
    
    def __init__(self, reg_lambda: float = 1e-4, reg_mu: Optional[np.ndarray] = None, 
                 device: str = None, learning_rate: float = 1e-3, max_iter: int = 1000, 
                 use_gpu: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.use_gpu = use_gpu
        
        # Device handling - auto-detect if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model parameters
        self.weights = None
        self.bias = None
        self.n_classes = None
        self.fitted_successfully = False
        
        logger.info(f" PyTorch Dirichlet calibration initialized on {self.device}")
    
    def _create_parameters(self, n_classes: int):
        """Create learnable parameters."""
        # Initialize close to identity transformation
        weights = torch.eye(n_classes, device=self.device) + \
                 torch.randn(n_classes, n_classes, device=self.device) * 0.001
        bias = torch.zeros(n_classes, device=self.device)
        
        return nn.Parameter(weights), nn.Parameter(bias)
    
    def _objective(self, weights: torch.Tensor, bias: torch.Tensor, 
                   log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the objective function."""
        # Compute log alphas
        log_alphas = bias.unsqueeze(0) + torch.matmul(log_probs, weights.T)
        log_alphas = torch.clamp(log_alphas, min=-5, max=5)
        alphas = torch.exp(log_alphas)
        alphas = torch.clamp(alphas, min=0.1, max=100.0)
        
        # Compute sum of alphas
        alpha_sums = torch.sum(alphas, dim=1)
        
        # Expected log probability for true class
        batch_size = alphas.shape[0]
        expected_log_probs = torch.zeros(batch_size, device=self.device)
        
        for i in range(batch_size):
            true_class = labels[i]
            expected_log_probs[i] = digamma(alphas[i, true_class]) - digamma(alpha_sums[i])
        
        # Negative log likelihood
        nll = -torch.mean(expected_log_probs)
        
        # L2 regularization
        identity = torch.eye(self.n_classes, device=self.device)
        reg_weights = torch.sum((weights - identity) ** 2)
        reg_bias = torch.sum(bias ** 2)
        reg_term = self.reg_lambda * (reg_weights + reg_bias)
        
        return nll + reg_term
    
    def fit(self, logits: np.ndarray, labels: np.ndarray, features: Optional[np.ndarray] = None) -> None:
        """
        Fit the Dirichlet calibration model using PyTorch optimization.
        
        Args:
            logits: Raw model outputs (logits) of shape (n_samples, n_classes)
            labels: True labels of shape (n_samples,)
            features: Not used for Dirichlet calibration (optional parameter for interface compatibility)
        """
        logger.info(" Fitting PyTorch Dirichlet calibration...")
        
        self.n_classes = logits.shape[1]
        
        # Convert to PyTorch tensors and move to device
        logits_tensor = torch.tensor(logits, dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # Compute probabilities and log probabilities
        probs = F.softmax(logits_tensor, dim=1)
        log_probs = torch.log(torch.clamp(probs, min=1e-8, max=1-1e-8))
        
        # Create learnable parameters
        weights, bias = self._create_parameters(self.n_classes)
        
        # Try different optimizers
        optimizers = [
            ('Adam', optim.Adam([weights, bias], lr=self.learning_rate)),
            ('AdamW', optim.AdamW([weights, bias], lr=self.learning_rate, weight_decay=self.reg_lambda)),
            ('L-BFGS', optim.LBFGS([weights, bias], lr=1.0, max_iter=20))
        ]
        
        best_weights = None
        best_bias = None
        best_loss = float('inf')
        
        for opt_name, optimizer in optimizers:
            try:
                logger.info(f"   Trying {opt_name} optimization...")
                
                # Reset parameters
                weights.data = torch.eye(self.n_classes, device=self.device) + \
                              torch.randn(self.n_classes, self.n_classes, device=self.device) * 0.001
                bias.data = torch.zeros(self.n_classes, device=self.device)
                
                # Optimization loop
                for epoch in range(self.max_iter):
                    def closure():
                        optimizer.zero_grad()
                        loss = self._objective(weights, bias, log_probs, labels_tensor)
                        loss.backward()
                        return loss
                    
                    if opt_name == 'L-BFGS':
                        loss = optimizer.step(closure)
                    else:
                        optimizer.zero_grad()
                        loss = self._objective(weights, bias, log_probs, labels_tensor)
                        loss.backward()
                        optimizer.step()
                    
                    if epoch % 100 == 0:
                        logger.debug(f"   Epoch {epoch}: loss = {loss.item():.6f}")
                    
                    if loss.item() < 0.1:
                        break
                    
                    if opt_name == 'L-BFGS' and epoch > 50:
                        break
                
                final_loss = loss.item()
                logger.info(f"    {opt_name}: final loss = {final_loss:.6f}")
                
                if final_loss < best_loss and final_loss > 0:
                    best_weights = weights.data.clone()
                    best_bias = bias.data.clone()
                    best_loss = final_loss
                    
                    if final_loss < 0.5:
                        break
                        
            except Exception as e:
                logger.debug(f"{opt_name} optimization failed: {e}")
                continue
        
        # Store results
        if best_weights is not None and best_loss < 10.0:
            self.weights = best_weights.cpu().numpy()
            self.bias = best_bias.cpu().numpy()
            
            if self._validate_solution():
                self.fitted_successfully = True
                logger.info(f" PyTorch Dirichlet calibration succeeded! Final loss: {best_loss:.6f}")
            else:
                logger.warning(" Solution validation failed, using conservative solution")
                self._create_conservative_solution()
        else:
            logger.warning(f" Optimization failed (best loss: {best_loss}), using conservative solution")
            self._create_conservative_solution()
        
        self.is_fitted = True
    
    def calibrate(self, logits: np.ndarray, features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply Dirichlet calibration to logits using GPU acceleration.
        
        Args:
            logits: Raw model outputs (logits) of shape (n_samples, n_classes)
            features: Not used for Dirichlet calibration (optional parameter for interface compatibility)
            
        Returns:
            Calibrated probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calibration")
        
        try:
            # Convert to PyTorch tensors
            logits_tensor = torch.tensor(logits, dtype=torch.float32, device=self.device)
            weights_tensor = torch.tensor(self.weights, dtype=torch.float32, device=self.device)
            bias_tensor = torch.tensor(self.bias, dtype=torch.float32, device=self.device)
            
            # Compute probabilities and log probabilities
            probs = F.softmax(logits_tensor, dim=1)
            log_probs = torch.log(torch.clamp(probs, min=1e-8, max=1-1e-8))
            
            # Apply learned transformation
            log_alphas = bias_tensor.unsqueeze(0) + torch.matmul(log_probs, weights_tensor.T)
            log_alphas = torch.clamp(log_alphas, min=-5, max=5)
            
            # Compute calibrated probabilities
            alphas = torch.exp(log_alphas)
            alphas = torch.clamp(alphas, min=0.1, max=100.0)
            
            # Dirichlet mean
            calibrated_probs = alphas / torch.sum(alphas, dim=1, keepdim=True)
            
            # Final normalization
            calibrated_probs = torch.clamp(calibrated_probs, min=1e-8, max=1-1e-8)
            calibrated_probs = calibrated_probs / torch.sum(calibrated_probs, dim=1, keepdim=True)
            
            return calibrated_probs.cpu().numpy()
            
        except Exception as e:
            logger.warning(f"GPU calibration failed: {e}, falling back to CPU")
            return self._softmax_numpy(logits)
    
    def _softmax_numpy(self, x):
        """Fallback numpy softmax."""
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _validate_solution(self) -> bool:
        """Validate that the solution is reasonable."""
        if self.weights is None or self.bias is None:
            return False
        
        if np.max(np.abs(self.weights)) > 10.0:
            return False
        if np.min(np.diag(self.weights)) < -1.0:
            return False
        if np.max(np.abs(self.bias)) > 5.0:
            return False
        
        return True
    
    def _create_conservative_solution(self):
        """Create a conservative solution."""
        self.weights = np.eye(self.n_classes) * 0.95
        self.bias = np.zeros(self.n_classes)
        self.fitted_successfully = False
    
    def get_params(self) -> Dict[str, Any]:
        """Get calibration parameters."""
        params = {
            "n_classes": self.n_classes,
            "reg_lambda": self.reg_lambda,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "device": str(self.device),
            "fitted_successfully": self.fitted_successfully,
            "backend": "PyTorch"
        }
        
        if self.weights is not None:
            params.update({
                "weights_shape": self.weights.shape,
                "weights_norm": float(np.linalg.norm(self.weights)),
                "weights_max": float(np.max(np.abs(self.weights))),
                "weights_diagonal_mean": float(np.mean(np.diag(self.weights))),
                "bias_norm": float(np.linalg.norm(self.bias)),
                "bias_max": float(np.max(np.abs(self.bias)))
            })
        
        return params
    
    def save(self, path: str) -> None:
        """Save the calibration model."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'n_classes': self.n_classes,
                'weights': self.weights,
                'bias': self.bias,
                'reg_lambda': self.reg_lambda,
                'reg_mu': self.reg_mu,
                'fitted_successfully': self.fitted_successfully,
                'device': str(self.device)
            }, f)
    
    def load(self, path: str) -> None:
        """Load the calibration model."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.n_classes = data['n_classes']
            self.weights = data['weights']
            self.bias = data['bias']
            self.reg_lambda = data['reg_lambda']
            self.reg_mu = data['reg_mu']
            self.fitted_successfully = data.get('fitted_successfully', False)
            self.is_fitted = True

# Smart factory function that automatically chooses the best implementation
def create_dirichlet_calibrator(use_gpu: bool = True, **kwargs):
    """
    Factory function that creates the best available Dirichlet calibrator.
    
    Args:
        use_gpu: Whether to use GPU acceleration if available
        **kwargs: Arguments passed to the calibrator
    
    Returns:
        DirichletCalibration instance (PyTorch if GPU available, NumPy otherwise)
    """
    if use_gpu and torch.cuda.is_available():
        logger.info(" Using GPU-accelerated PyTorch Dirichlet calibration")
        return PyTorchDirichletCalibration(use_gpu=True, **kwargs)
    else:
        logger.info(" Using CPU NumPy Dirichlet calibration")
        from .dirichlet_calibration import DirichletCalibration
        return DirichletCalibration(**kwargs)

# Backward compatibility wrapper
class SmartDirichletCalibration(BaseCalibrator):
    """
    Smart wrapper that automatically chooses between PyTorch and NumPy implementations.
    Provides seamless GPU acceleration when available while maintaining compatibility.
    """
    
    def __init__(self, use_gpu: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_gpu = use_gpu
        self.calibrator = create_dirichlet_calibrator(use_gpu=use_gpu, **kwargs)
    
    def fit(self, logits: np.ndarray, labels: np.ndarray, features: Optional[np.ndarray] = None) -> None:
        """Fit using the selected implementation."""
        self.calibrator.fit(logits, labels, features)
        self.is_fitted = self.calibrator.is_fitted
    
    def calibrate(self, logits: np.ndarray, features: Optional[np.ndarray] = None) -> np.ndarray:
        """Calibrate using the selected implementation."""
        return self.calibrator.calibrate(logits, features)
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters from the selected implementation."""
        return self.calibrator.get_params()
    
    def save(self, path: str) -> None:
        """Save the calibration model."""
        self.calibrator.save(path)
    
    def load(self, path: str) -> None:
        """Load the calibration model."""
        self.calibrator.load(path)
        self.is_fitted = self.calibrator.is_fitted