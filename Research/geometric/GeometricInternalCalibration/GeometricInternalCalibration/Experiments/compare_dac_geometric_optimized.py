"""
Optimized compare_dac_geometric.py

Key optimizations:
1. Centralized caching for model predictions (logits/probs)
2. Centralized DataLoader factory to avoid recreation
3. Cached OOD feature extraction
4. Pre-computed logits passed to optimization functions
5. Lazy evaluation patterns where possible
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache
import hashlib


# =============================================================================
# CACHING INFRASTRUCTURE
# =============================================================================

class ExperimentCache:
    """
    Centralized cache for expensive computations that should only happen once.
    
    This class eliminates redundant:
    - Model forward passes (predict_proba/predict_logits)
    - DataLoader creation
    - Feature extraction for OOD data
    - Coordinate space discovery
    """
    
    def __init__(self, model_adapter, device: torch.device, batch_size: int = 128):
        self.model_adapter = model_adapter
        self.device = device
        self.batch_size = batch_size
        
        # Caches
        self._logits_cache: Dict[str, np.ndarray] = {}
        self._probs_cache: Dict[str, np.ndarray] = {}
        self._predictions_cache: Dict[str, np.ndarray] = {}
        self._dataloader_cache: Dict[str, DataLoader] = {}
        self._features_cache: Dict[str, Any] = {}
        self._coordinate_space_cache: Dict[str, Any] = {}
        
    def _hash_array(self, arr: np.ndarray) -> str:
        """Create a hash key for a numpy array based on shape and sample."""
        # Use shape + first/last few elements for fast hashing
        key_parts = [str(arr.shape), str(arr.dtype)]
        if arr.size > 0:
            flat = arr.flatten()
            key_parts.append(str(flat[:min(10, len(flat))].tobytes()))
            key_parts.append(str(flat[-min(10, len(flat)):].tobytes()))
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()[:16]
    
    # -------------------------------------------------------------------------
    # LOGITS / PROBS / PREDICTIONS CACHING
    # -------------------------------------------------------------------------
    
    def get_logits(self, data_raw: np.ndarray, data_name: str = None) -> np.ndarray:
        """Get logits for data, using cache if available."""
        key = data_name or self._hash_array(data_raw)
        
        if key not in self._logits_cache:
            self._logits_cache[key] = self.model_adapter.predict_logits(
                data_raw, batch_size=self.batch_size
            )
        return self._logits_cache[key]
    
    def get_probs(self, data_raw: np.ndarray, data_name: str = None) -> np.ndarray:
        """Get softmax probabilities for data, using cache if available."""
        key = data_name or self._hash_array(data_raw)
        
        if key not in self._probs_cache:
            self._probs_cache[key] = self.model_adapter.predict_proba(
                data_raw, batch_size=self.batch_size
            )
        return self._probs_cache[key]
    
    def get_predictions(self, data_raw: np.ndarray, data_name: str = None) -> np.ndarray:
        """Get argmax predictions for data, using cache if available."""
        key = data_name or self._hash_array(data_raw)
        
        if key not in self._predictions_cache:
            probs = self.get_probs(data_raw, data_name)
            self._predictions_cache[key] = np.argmax(probs, axis=1)
        return self._predictions_cache[key]
    
    def precompute_all_predictions(
        self,
        train_raw: np.ndarray,
        val_raw: np.ndarray, 
        test_raw: np.ndarray,
        ood_raw: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Precompute all logits/probs/predictions upfront.
        Call this once at the start of main() to populate caches.
        """
        results = {}
        
        # Train
        results['train_logits'] = self.get_logits(train_raw, 'train')
        results['train_probs'] = self.get_probs(train_raw, 'train')
        results['train_preds'] = self.get_predictions(train_raw, 'train')
        
        # Val
        results['val_logits'] = self.get_logits(val_raw, 'val')
        results['val_probs'] = self.get_probs(val_raw, 'val')
        results['val_preds'] = self.get_predictions(val_raw, 'val')
        
        # Test
        results['test_logits'] = self.get_logits(test_raw, 'test')
        results['test_probs'] = self.get_probs(test_raw, 'test')
        results['test_preds'] = self.get_predictions(test_raw, 'test')
        
        # OOD (if available)
        if ood_raw is not None:
            results['ood_logits'] = self.get_logits(ood_raw, 'ood')
            results['ood_probs'] = self.get_probs(ood_raw, 'ood')
            results['ood_preds'] = self.get_predictions(ood_raw, 'ood')
        
        return results
    
    # -------------------------------------------------------------------------
    # DATALOADER CACHING
    # -------------------------------------------------------------------------
    
    def get_dataloader(
        self,
        data_raw: np.ndarray,
        labels: np.ndarray,
        name: str,
        shuffle: bool = False
    ) -> DataLoader:
        """Get or create a DataLoader, caching for reuse."""
        key = f"{name}_shuffle{shuffle}"
        
        if key not in self._dataloader_cache:
            self._dataloader_cache[key] = DataLoader(
                TensorDataset(
                    torch.from_numpy(data_raw),
                    torch.from_numpy(labels)
                ),
                batch_size=self.batch_size,
                shuffle=shuffle
            )
        return self._dataloader_cache[key]
    
    def get_train_loader(self, train_raw: np.ndarray, train_labels: np.ndarray) -> DataLoader:
        return self.get_dataloader(train_raw, train_labels, 'train', shuffle=False)
    
    def get_val_loader(self, val_raw: np.ndarray, val_labels: np.ndarray) -> DataLoader:
        return self.get_dataloader(val_raw, val_labels, 'val', shuffle=False)
    
    def get_test_loader(self, test_raw: np.ndarray, test_labels: np.ndarray = None) -> DataLoader:
        if test_labels is None:
            test_labels = np.zeros(len(test_raw), dtype=np.int64)
        return self.get_dataloader(test_raw, test_labels, 'test', shuffle=False)
    
    def get_ood_loader(self, ood_raw: np.ndarray, ood_labels: np.ndarray = None) -> DataLoader:
        if ood_labels is None:
            ood_labels = np.zeros(len(ood_raw), dtype=np.int64)
        return self.get_dataloader(ood_raw, ood_labels, 'ood', shuffle=False)
    
    # -------------------------------------------------------------------------
    # FEATURE EXTRACTION CACHING
    # -------------------------------------------------------------------------
    
    def cache_features(self, key: str, features: Any):
        """Store extracted features in cache."""
        self._features_cache[key] = features
    
    def get_cached_features(self, key: str) -> Optional[Any]:
        """Retrieve cached features if available."""
        return self._features_cache.get(key)
    
    def cache_ood_features(
        self,
        layer_config: str,  # e.g., "sgc_random_L6" or "dac_layers" or "coordinate_K256"
        features: np.ndarray
    ):
        """Cache OOD features for a specific feature extraction configuration."""
        self._features_cache[f'ood_{layer_config}'] = features
    
    def get_cached_ood_features(self, layer_config: str) -> Optional[np.ndarray]:
        """Get cached OOD features for a specific configuration."""
        return self._features_cache.get(f'ood_{layer_config}')
    
    # -------------------------------------------------------------------------
    # COORDINATE SPACE CACHING
    # -------------------------------------------------------------------------
    
    def cache_coordinate_space(
        self,
        model_name: str,
        input_shape: tuple,
        layer_map: List[Dict],
        total_size: int,
        sampling_plan: Dict = None,
        global_order: np.ndarray = None
    ):
        """Cache coordinate space discovery results."""
        key = f"{model_name}_{input_shape}"
        self._coordinate_space_cache[key] = {
            'layer_map': layer_map,
            'total_size': total_size,
            'sampling_plan': sampling_plan,
            'global_order': global_order
        }
    
    def get_cached_coordinate_space(
        self,
        model_name: str,
        input_shape: tuple
    ) -> Optional[Dict]:
        """Get cached coordinate space if available."""
        key = f"{model_name}_{input_shape}"
        return self._coordinate_space_cache.get(key)
    
    def clear(self):
        """Clear all caches."""
        self._logits_cache.clear()
        self._probs_cache.clear()
        self._predictions_cache.clear()
        self._dataloader_cache.clear()
        self._features_cache.clear()
        self._coordinate_space_cache.clear()


# =============================================================================
# OPTIMIZED HELPER FUNCTIONS
# =============================================================================

def extract_ood_features_once(
    model: torch.nn.Module,
    ood_raw: np.ndarray,
    layer_names: List[str],
    device: torch.device,
    batch_size: int,
    cache: ExperimentCache,
    feature_config: str,  # Unique identifier for this feature extraction config
    extract_fn,  # Function to use for extraction
    **extract_kwargs
) -> np.ndarray:
    """
    Extract OOD features with caching to avoid redundant extraction.
    
    Args:
        feature_config: Unique string identifying this extraction configuration
                       e.g., "sgc_L6_d256_seed42" or "coordinate_K256"
        extract_fn: The extraction function to call if not cached
        **extract_kwargs: Arguments to pass to extract_fn
    """
    # Check cache first
    cached = cache.get_cached_ood_features(feature_config)
    if cached is not None:
        return cached
    
    # Extract
    ood_loader = cache.get_ood_loader(ood_raw)
    features = extract_fn(model, ood_loader, layer_names, device, **extract_kwargs)
    
    # Cache for future use
    cache.cache_ood_features(feature_config, features)
    
    return features


def run_geometric_with_dac_weighting_optimized(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    cache: ExperimentCache,  # NEW: Pass the cache
    batch_size: int = 128,
    precomputed_features: Dict[str, Any] = None,
    precomputed_separations: Dict[str, Any] = None,
    pooling_mode: str = 'max',
    target_dimension: int = None,
    ood_raw: np.ndarray = None,
    ood_labels: np.ndarray = None
) -> Dict[str, Any]:
    """
    OPTIMIZED version of run_geometric_with_dac_weighting.
    
    Key optimization: val_logits is computed ONCE and passed to objective function
    instead of being recomputed on every optimization iteration.
    """
    from scipy.optimize import minimize
    from sklearn.preprocessing import normalize
    import time
    import logging
    
    logger = logging.getLogger(__name__)
    
    # ... (setup code similar to original) ...
    
    # CRITICAL OPTIMIZATION: Get val_logits ONCE from cache
    val_logits = cache.get_probs(val_raw, 'val')
    val_pred = cache.get_predictions(val_raw, 'val')
    test_pred = cache.get_predictions(test_raw, 'test')
    
    # ... (separation score computation - same as original) ...
    
    # Step 3: Learn optimal weights
    # OPTIMIZED: val_logits is captured in closure, not recomputed
    n_classes = len(np.unique(train_labels))
    val_labels_onehot = np.zeros((len(val_labels), n_classes))
    val_labels_onehot[np.arange(len(val_labels)), val_labels] = 1
    
    def objective_function(params, val_sep_matrix, val_logits_precomputed, val_labels_oh):
        """
        OPTIMIZED: val_logits_precomputed is passed as argument, not recomputed.
        This saves ~1000x forward passes during optimization!
        """
        layer_weights = params[:-1]
        bias = params[-1]
        
        # Compute weighted separation scores
        weighted_scores = val_sep_matrix @ layer_weights + bias
        
        # Use PRE-COMPUTED logits (the key optimization!)
        temperatures = np.maximum(weighted_scores, 0.1)
        calibrated_logits = val_logits_precomputed / temperatures[:, np.newaxis]
        
        # Softmax
        max_logits = np.max(calibrated_logits, axis=1, keepdims=True)
        exp_logits = np.exp(calibrated_logits - max_logits)
        calibrated_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Squared error loss
        return np.sum((val_labels_oh - calibrated_probs) ** 2)
    
    # ... (rest of optimization and calibration) ...


# =============================================================================
# EXAMPLE: OPTIMIZED MAIN FUNCTION STRUCTURE
# =============================================================================

def main_optimized():
    """
    Example of how to structure main() with the caching infrastructure.
    """
    import argparse
    import logging
    
    logger = logging.getLogger(__name__)
    
    # ... (argument parsing) ...
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ... (load model, data) ...
    
    # CREATE CACHE EARLY
    cache = ExperimentCache(model_adapter, device, batch_size=args.batch_size)
    
    # PRECOMPUTE ALL PREDICTIONS UPFRONT
    # This single call replaces dozens of scattered predict_proba() calls
    logger.info("Pre-computing model predictions for all datasets...")
    predictions = cache.precompute_all_predictions(
        train_raw, val_raw, test_raw, ood_raw
    )
    logger.info("Pre-computation complete. Predictions cached for reuse.")
    
    # Now all experiment functions can use cache.get_probs(), cache.get_predictions()
    # instead of calling model_adapter.predict_proba() repeatedly
    
    # Example: Running experiments with cache
    if should_run_global_random:
        result = run_global_random_calibration_optimized(
            model, model_name, dataset_name, model_adapter,
            train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
            device,
            cache=cache,  # Pass cache
            # ... other args ...
        )
    
    if should_run_dac_weighted:
        result = run_geometric_with_dac_weighting_optimized(
            model, model_name, dataset_name, model_adapter,
            train_raw, train_labels, val_raw, val_labels, test_raw, test_labels,
            device,
            cache=cache,  # Pass cache - this avoids the optimization loop bug!
            # ... other args ...
        )


# =============================================================================
# QUICK FIXES YOU CAN APPLY TO YOUR EXISTING CODE
# =============================================================================

"""
QUICK FIX #1: Fix the optimization loop bug (HIGHEST PRIORITY)
--------------------------------------------------------------------------------
In run_geometric_with_dac_weighting(), around line 5152:

BEFORE:
    def objective_function(params):
        ...
        val_logits_local = model_adapter.predict_proba(val_raw)  # BUG: Called every iteration!
        ...

AFTER:
    # Compute ONCE before the optimization loop
    val_logits_precomputed = model_adapter.predict_proba(val_raw)
    
    def objective_function(params):
        ...
        # Use the precomputed value (captured in closure)
        val_logits_local = val_logits_precomputed
        ...


QUICK FIX #2: Cache test predictions at top of each experiment function
--------------------------------------------------------------------------------
Instead of calling model_adapter.predict_proba(test_raw) multiple times,
compute once and reuse:

    # At start of function
    test_probs = model_adapter.predict_proba(test_raw, batch_size=batch_size)
    test_preds = np.argmax(test_probs, axis=1)
    
    # Later, use test_probs instead of calling predict_proba again


QUICK FIX #3: Create DataLoaders once in main() and pass them around
--------------------------------------------------------------------------------
In main():
    train_loader = DataLoader(TensorDataset(...), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(...), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(...), batch_size=batch_size, shuffle=False)
    if ood_raw is not None:
        ood_loader = DataLoader(TensorDataset(...), batch_size=batch_size, shuffle=False)

Then pass these loaders to functions instead of raw arrays.


QUICK FIX #4: Cache OOD feature extraction results
--------------------------------------------------------------------------------
After extracting OOD features in one experiment, store them:

    ood_features_cache = {}
    
    def get_ood_features(config_key, extract_fn, *args, **kwargs):
        if config_key not in ood_features_cache:
            ood_features_cache[config_key] = extract_fn(*args, **kwargs)
        return ood_features_cache[config_key]
"""


# =============================================================================
# ESTIMATED SPEEDUP
# =============================================================================

"""
Based on the analysis:

1. Optimization loop fix (objective_function):
   - Before: ~500-1000 forward passes through model during optimization
   - After: 1 forward pass
   - Speedup: 500-1000x for that section alone
   - Time saved: Could be 5-30 minutes depending on model/data size

2. Centralized prediction caching:
   - Before: ~20-30 predict_proba() calls across all experiments
   - After: 4 predict_proba() calls (train, val, test, ood)
   - Speedup: ~5-7x for prediction-heavy code paths
   - Time saved: 2-10 minutes depending on data size

3. DataLoader reuse:
   - Minor savings (~seconds) but cleaner code

4. OOD feature caching:
   - Before: OOD features extracted ~5-8 times
   - After: OOD features extracted once per configuration
   - Speedup: 5-8x for OOD evaluation
   - Time saved: 1-5 minutes depending on number of experiments

TOTAL ESTIMATED SPEEDUP: 2-4x for the full experiment suite
(More if you run many experiments with DAC weighting optimization)
"""
