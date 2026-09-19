import numpy as np
import torch
import logging
from typing import Optional, Dict, Any
from tqdm import tqdm

from utils.logging_config import get_logger
logger = get_logger(__name__)


class SimpleStabilitySpace:
    def __init__(self,
                 X_train: Optional[np.ndarray] = None,
                 y_train: Optional[np.ndarray] = None,
                 metric: str = 'l2',
                 use_cuda: bool = False):
        """
        Initialize stability space with GPU support
        """
        self.metric = metric
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        # Initialize storage
        if X_train is not None and y_train is not None:
            self.X_train = X_train
            self.y_train = y_train
        else:
            self.X_train = np.array([])
            self.y_train = np.array([])
        
        # Pre-compute class indices for GPU efficiency
        self._precompute_class_data()
        
        logger.info(f"🔬 SimpleStabilitySpace initialized:")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Metric: {metric}")
        logger.info(f"   CUDA available: {torch.cuda.is_available()}")
        if self.use_cuda:
            logger.info(f"   GPU: {torch.cuda.get_device_name()}")
            logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def _precompute_class_data(self):
        """Precompute class-specific data structures for efficient GPU operations"""
        if len(self.y_train) == 0:
            self.class_indices = {}
            return
            
        self.class_indices = {}
        unique_classes = np.unique(self.y_train)
        
        for class_id in unique_classes:
            self.class_indices[int(class_id)] = np.where(self.y_train == class_id)[0]
        
        logger.info(f"📊 Precomputed indices for {len(unique_classes)} classes")
    
    def update_training_data(self, new_features: np.ndarray, new_predictions: np.ndarray):
        """Update training data and recompute class indices"""
        if len(self.X_train) == 0:
            self.X_train = new_features
            self.y_train = new_predictions
        else:
            self.X_train = np.vstack([self.X_train, new_features])
            self.y_train = np.concatenate([self.y_train, new_predictions])
        
        # Keep only recent samples to prevent memory explosion
        max_samples = 10000 if self.use_cuda else 5000  # More samples on GPU
        if len(self.X_train) > max_samples:
            self.X_train = self.X_train[-max_samples:]
            self.y_train = self.y_train[-max_samples:]
        
        # Recompute class indices
        self._precompute_class_data()
        
        logger.info(f"📊 Updated training data: {len(self.X_train):,} samples")
    
    def calculate_fast_separation_vectorized(self, 
                                           features: np.ndarray, 
                                           predictions: np.ndarray, 
                                           batch_size: int = None) -> np.ndarray:
        """
        GPU-accelerated vectorized fast-separation calculation with smart batching
        """
        if len(self.X_train) == 0:
            logger.warning("🚨 No training data available for separation calculation")
            return np.zeros(len(features))
        
        # Adaptive batch sizing based on available memory
        if batch_size is None:
            if self.use_cuda:
                # Estimate memory usage and set appropriate batch size
                feature_dim = features.shape[1]
                n_train = len(self.X_train)
                
                # Rough memory estimate: batch_size * n_train * feature_dim * 4 bytes (float32)
                available_memory = torch.cuda.get_device_properties(0).total_memory * 0.5  # Use 50% of GPU memory
                estimated_memory_per_sample = n_train * feature_dim * 4 * 2  # 2x for intermediate computations
                batch_size = max(50, min(1000, int(available_memory / estimated_memory_per_sample)))
            else:
                batch_size = 200  # Conservative CPU batch size
        
        n_query = len(features)
        n_train = len(self.X_train)
        
        logger.info(f"🚀 Computing GPU-accelerated separation for {n_query:,} samples using {n_train:,} training points")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Metric: {self.metric}")
        
        # Convert training data to GPU tensors once
        try:
            X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32, device=self.device)
            y_train_tensor = torch.tensor(self.y_train, dtype=torch.long, device=self.device)
            
            if self.use_cuda:
                memory_used = X_train_tensor.element_size() * X_train_tensor.nelement()
                logger.info(f"   GPU memory for training data: {memory_used / 1e6:.1f} MB")
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"⚠️ GPU out of memory, falling back to CPU")
                self.use_cuda = False
                self.device = torch.device('cpu')
                X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32, device=self.device)
                y_train_tensor = torch.tensor(self.y_train, dtype=torch.long, device=self.device)
            else:
                raise e
        
        separation_scores = []
        n_batches = (n_query + batch_size - 1) // batch_size
        
        with tqdm(total=n_batches, desc=f"🚀 Fast Separation ({'GPU' if self.use_cuda else 'CPU'})", unit="batch") as pbar:
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_query)
                
                batch_features = features[start_idx:end_idx]
                batch_predictions = predictions[start_idx:end_idx]
                
                try:
                    # Process entire batch on GPU
                    batch_scores = self._compute_batch_separation_gpu(
                        batch_features, batch_predictions, X_train_tensor, y_train_tensor
                    )
                    separation_scores.extend(batch_scores)
                    
                    # Update progress
                    completed_samples = end_idx
                    pbar.set_postfix({
                        'samples': f"{completed_samples:,}/{n_query:,}",
                        'batch_mean': f"{np.mean(batch_scores):.3f}",
                        'overall_mean': f"{np.mean(separation_scores):.3f}",
                        'device': 'GPU' if self.use_cuda else 'CPU'
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.use_cuda:
                        logger.warning(f"⚠️ GPU OOM at batch {batch_idx}, reducing batch size")
                        torch.cuda.empty_cache()
                        
                        # Fall back to smaller batches for this batch
                        mini_batch_size = batch_size // 4
                        batch_scores = []
                        
                        for mini_start in range(start_idx, end_idx, mini_batch_size):
                            mini_end = min(mini_start + mini_batch_size, end_idx)
                            mini_features = features[mini_start:mini_end]
                            mini_predictions = predictions[mini_start:mini_end]
                            
                            mini_scores = self._compute_batch_separation_gpu(
                                mini_features, mini_predictions, X_train_tensor, y_train_tensor
                            )
                            batch_scores.extend(mini_scores)
                        
                        separation_scores.extend(batch_scores)
                    else:
                        raise e
                
                pbar.update(1)
                
                # Clear GPU cache periodically
                if self.use_cuda and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        scores_array = np.array(separation_scores)
        
        logger.info(f"✅ GPU-accelerated separation computation complete:")
        logger.info(f"   📊 Processed: {len(scores_array):,} samples in {n_batches} batches")
        logger.info(f"   📈 Stats: mean={np.mean(scores_array):.4f}, std={np.std(scores_array):.4f}")
        logger.info(f"   📏 Range: [{np.min(scores_array):.4f}, {np.max(scores_array):.4f}]")
        
        # Final GPU cleanup
        if self.use_cuda:
            torch.cuda.empty_cache()
        
        return scores_array
    
    def _compute_batch_separation_gpu(self, 
                                     batch_features: np.ndarray, 
                                     batch_predictions: np.ndarray,
                                     X_train_tensor: torch.Tensor,
                                     y_train_tensor: torch.Tensor) -> list:
        """
        Compute separation scores for a batch using GPU-accelerated operations
        """
        batch_size = len(batch_features)
        batch_features_tensor = torch.tensor(batch_features, dtype=torch.float32, device=self.device)
        batch_predictions_tensor = torch.tensor(batch_predictions, dtype=torch.long, device=self.device)
        
        batch_scores = []
        
        if self.metric == 'l2':
            # Vectorized L2 distance computation using torch.cdist
            # Shape: [batch_size, n_train]
            distances = torch.cdist(batch_features_tensor, X_train_tensor, p=2)
            
            for i in range(batch_size):
                pred_class = batch_predictions_tensor[i].item()
                sample_distances = distances[i]  # [n_train]
                
                # Create masks for same/different classes
                same_class_mask = (y_train_tensor == pred_class)
                diff_class_mask = (y_train_tensor != pred_class)
                
                # Compute minimum distances
                if torch.any(same_class_mask):
                    d_same = torch.min(sample_distances[same_class_mask]).item()
                else:
                    d_same = float('inf')
                
                if torch.any(diff_class_mask):
                    d_diff = torch.min(sample_distances[diff_class_mask]).item()
                else:
                    d_diff = float('inf')
                
                # Fast-separation calculation
                fast_sep = self._compute_separation_score(d_same, d_diff)
                batch_scores.append(fast_sep)
        
        else:  # cosine metric
            # Normalize vectors for cosine similarity
            batch_features_norm = torch.nn.functional.normalize(batch_features_tensor, p=2, dim=1)
            X_train_norm = torch.nn.functional.normalize(X_train_tensor, p=2, dim=1)
            
            # Compute cosine similarities and convert to distances
            similarities = torch.mm(batch_features_norm, X_train_norm.t())  # [batch_size, n_train]
            distances = 1 - similarities  # Convert to cosine distance
            
            for i in range(batch_size):
                pred_class = batch_predictions_tensor[i].item()
                sample_distances = distances[i]  # [n_train]
                
                # Create masks for same/different classes
                same_class_mask = (y_train_tensor == pred_class)
                diff_class_mask = (y_train_tensor != pred_class)
                
                # Compute minimum distances
                if torch.any(same_class_mask):
                    d_same = torch.min(sample_distances[same_class_mask]).item()
                else:
                    d_same = float('inf')
                
                if torch.any(diff_class_mask):
                    d_diff = torch.min(sample_distances[diff_class_mask]).item()
                else:
                    d_diff = float('inf')
                
                # Fast-separation calculation
                fast_sep = self._compute_separation_score(d_same, d_diff)
                batch_scores.append(fast_sep)
        
        return batch_scores
    
    def _compute_separation_score(self, d_same: float, d_diff: float) -> float:
        """Compute fast-separation score from same and different class distances"""
        if d_same == float('inf') and d_diff == float('inf'):
            return 0.0
        elif d_same == float('inf'):
            return d_diff / 2.0
        elif d_diff == float('inf'):
            return -d_same / 2.0
        else:
            return (d_diff - d_same) / 2.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the stability space"""
        return {
            'num_samples': len(self.X_train),
            'feature_dim': self.X_train.shape[1] if len(self.X_train) > 0 else 0,
            'num_classes': len(np.unique(self.y_train)) if len(self.y_train) > 0 else 0,
            'metric': self.metric,
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'using_gpu': self.use_cuda
        }