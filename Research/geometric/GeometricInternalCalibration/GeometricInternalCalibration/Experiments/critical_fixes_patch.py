"""
CRITICAL FIXES FOR compare_dac_geometric.py
============================================

This file contains specific patches you can apply to your code.
Priority is ranked from P0 (critical/easy fix) to P2 (nice to have).

"""

# =============================================================================
# P0 FIX #1: OPTIMIZATION LOOP BUG (CRITICAL - Saves 500-1000x forward passes)
# =============================================================================
"""
Location: run_geometric_with_dac_weighting() around line 5148-5188

PROBLEM:
The objective_function() calls model_adapter.predict_proba() on every iteration
of the optimization. With L-BFGS-B potentially running 100-1000 iterations,
this means 100-1000 unnecessary forward passes!

FIND THIS CODE (around line 5152-5188):
"""
# --- BEFORE ---
BEFORE_CODE_BLOCK_1 = '''
    # Step 3: Learn optimal weights using DAC's optimization approach
    logger.info("\\n3. Learning optimal layer weights (DAC-style optimization)...")
    start_fit = time.perf_counter()
    
    # Get model predictions for validation set (to determine correctness)
    val_logits = model_adapter.predict_proba(val_raw)
    val_preds = np.argmax(val_logits, axis=1)
    
    # Prepare data in one-hot format (like DAC does)
    n_classes = len(np.unique(train_labels))
    val_labels_onehot = np.zeros((len(val_labels), n_classes))
    val_labels_onehot[np.arange(len(val_labels)), val_labels] = 1
    
    # Define objective function (same as DAC's squared error loss)
    def objective_function(params):
        """
        Minimize squared error between predicted confidences and true labels.
        params = [w1, w2, w3, w4, bias]
        """
        layer_weights = params[:-1]
        bias = params[-1]
        
        # Compute weighted separation scores: S(x,w) = w1*s1 + w2*s2 + ... + bias
        weighted_scores = val_sep_matrix @ layer_weights + bias
        
        # Get logits from model
        val_logits_local = model_adapter.predict_proba(val_raw)  # <-- BUG: Called every iteration!
        
        # Apply temperature scaling: Q(x,w) = softmax(z_L / S(x,w))
        # Avoid division by zero
        temperatures = np.maximum(weighted_scores, 0.1)
        calibrated_logits = val_logits_local / temperatures[:, np.newaxis]
'''

# --- AFTER ---
AFTER_CODE_BLOCK_1 = '''
    # Step 3: Learn optimal weights using DAC's optimization approach
    logger.info("\\n3. Learning optimal layer weights (DAC-style optimization)...")
    start_fit = time.perf_counter()
    
    # Get model predictions for validation set ONCE (to determine correctness)
    # OPTIMIZATION: This is now computed once and reused in objective function
    val_logits_precomputed = model_adapter.predict_proba(val_raw)
    val_preds = np.argmax(val_logits_precomputed, axis=1)
    
    # Prepare data in one-hot format (like DAC does)
    n_classes = len(np.unique(train_labels))
    val_labels_onehot = np.zeros((len(val_labels), n_classes))
    val_labels_onehot[np.arange(len(val_labels)), val_labels] = 1
    
    # Define objective function (same as DAC's squared error loss)
    # OPTIMIZATION: val_logits_precomputed is captured in closure - no recomputation!
    def objective_function(params):
        """
        Minimize squared error between predicted confidences and true labels.
        params = [w1, w2, w3, w4, bias]
        """
        layer_weights = params[:-1]
        bias = params[-1]
        
        # Compute weighted separation scores: S(x,w) = w1*s1 + w2*s2 + ... + bias
        weighted_scores = val_sep_matrix @ layer_weights + bias
        
        # FIXED: Use pre-computed logits (captured in closure)
        # This saves ~500-1000 forward passes during optimization!
        
        # Apply temperature scaling: Q(x,w) = softmax(z_L / S(x,w))
        # Avoid division by zero
        temperatures = np.maximum(weighted_scores, 0.1)
        calibrated_logits = val_logits_precomputed / temperatures[:, np.newaxis]
'''


# =============================================================================
# P0 FIX #2: ADD CENTRALIZED PREDICTION CACHING IN MAIN()
# =============================================================================
"""
Location: main() around line 6850, right after extracting raw data

Add this code block to precompute all predictions once:
"""
MAIN_PREDICTION_CACHING = '''
    # ===========================================================================
    # OPTIMIZATION: Precompute model predictions for all datasets
    # This eliminates redundant forward passes across all experiments
    # ===========================================================================
    logger.info("\\nPre-computing model predictions for all datasets...")
    prediction_cache = {}
    
    # Training set
    prediction_cache['train_probs'] = model_adapter.predict_proba(train_raw, batch_size=args.batch_size)
    prediction_cache['train_preds'] = np.argmax(prediction_cache['train_probs'], axis=1)
    
    # Validation set
    prediction_cache['val_probs'] = model_adapter.predict_proba(val_raw, batch_size=args.batch_size)
    prediction_cache['val_preds'] = np.argmax(prediction_cache['val_probs'], axis=1)
    
    # Test set
    prediction_cache['test_probs'] = model_adapter.predict_proba(test_raw, batch_size=args.batch_size)
    prediction_cache['test_preds'] = np.argmax(prediction_cache['test_probs'], axis=1)
    
    # OOD set (if available)
    if ood_raw is not None:
        prediction_cache['ood_probs'] = model_adapter.predict_proba(ood_raw, batch_size=args.batch_size)
        prediction_cache['ood_preds'] = np.argmax(prediction_cache['ood_probs'], axis=1)
    
    logger.info("Pre-computation complete.")
    # ===========================================================================
'''


# =============================================================================
# P1 FIX #3: ADD DATALOADER CACHING IN MAIN()
# =============================================================================
"""
Location: main() around line 6850, after prediction caching

Add this code to create all DataLoaders once:
"""
MAIN_DATALOADER_CACHING = '''
    # ===========================================================================
    # OPTIMIZATION: Create DataLoaders once and reuse
    # ===========================================================================
    logger.info("Creating data loaders...")
    dataloader_cache = {}
    
    dataloader_cache['train'] = DataLoader(
        TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
        batch_size=args.batch_size, shuffle=False
    )
    dataloader_cache['val'] = DataLoader(
        TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
        batch_size=args.batch_size, shuffle=False
    )
    dataloader_cache['test'] = DataLoader(
        TensorDataset(torch.from_numpy(test_raw), torch.from_numpy(test_labels.astype(np.int64))),
        batch_size=args.batch_size, shuffle=False
    )
    if ood_raw is not None:
        dataloader_cache['ood'] = DataLoader(
            TensorDataset(torch.from_numpy(ood_raw), torch.from_numpy(np.zeros(len(ood_raw), dtype=np.int64))),
            batch_size=args.batch_size, shuffle=False
        )
    # ===========================================================================
'''


# =============================================================================
# P1 FIX #4: ADD OOD FEATURE CACHING
# =============================================================================
"""
Add this at the top of main() or as a global, then use it across all functions
that extract OOD features:
"""
OOD_FEATURE_CACHING = '''
    # ===========================================================================
    # OPTIMIZATION: OOD feature caching
    # Each unique feature extraction configuration stores its OOD features
    # ===========================================================================
    ood_feature_cache = {}
    
    def get_or_extract_ood_features(
        config_key: str,  # e.g., "sgc_L6_d256" or "coordinate_K256"
        extract_fn,       # Function to extract features
        *args, **kwargs   # Arguments for extract_fn
    ):
        """Extract OOD features with caching."""
        if config_key not in ood_feature_cache:
            logger.info(f"Extracting OOD features for config: {config_key}")
            ood_feature_cache[config_key] = extract_fn(*args, **kwargs)
        else:
            logger.info(f"Using cached OOD features for config: {config_key}")
        return ood_feature_cache[config_key]
    # ===========================================================================
'''


# =============================================================================
# P1 FIX #5: UPDATE FUNCTION SIGNATURES TO ACCEPT CACHED DATA
# =============================================================================
"""
Update experiment functions to accept precomputed data instead of computing it themselves.

Example for run_global_random_calibration (around line 2838):
"""
UPDATED_FUNCTION_SIGNATURE = '''
def run_global_random_calibration(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    model_adapter: PyTorchModelAdapter,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    val_raw: np.ndarray,
    val_labels: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    num_layers: int = 6,
    target_dim: int = 256,
    batch_size: int = 128,
    seed: int = 42,
    pooling_mode: str = 'max',
    output_file: str = None,
    results_dict: Dict[str, Any] = None,
    ood_raw: np.ndarray = None,
    ood_labels: np.ndarray = None,
    scoring_method: str = 'separation',
    precomputed_train_features: np.ndarray = None,
    precomputed_val_features: np.ndarray = None,
    precomputed_test_features: np.ndarray = None,
    precomputed_extraction_info: Dict[str, Any] = None,
    precomputed_selected_layers: List[str] = None,
    # NEW PARAMETERS FOR CACHING:
    precomputed_test_probs: np.ndarray = None,  # <-- ADD THIS
    precomputed_ood_probs: np.ndarray = None,   # <-- ADD THIS
    train_loader: DataLoader = None,            # <-- ADD THIS
    val_loader: DataLoader = None,              # <-- ADD THIS
    test_loader: DataLoader = None,             # <-- ADD THIS
    ood_loader: DataLoader = None,              # <-- ADD THIS
) -> Dict[str, Any]:
'''


# =============================================================================
# P2 FIX #6: BATCH OOD EVALUATION INTO A SEPARATE FUNCTION
# =============================================================================
"""
Instead of duplicating OOD evaluation code in every experiment function,
create a reusable function:
"""
REUSABLE_OOD_EVALUATION = '''
def evaluate_ood_metrics(
    geo_cal,  # Fitted calibrator
    test_features: np.ndarray,
    test_raw: np.ndarray,
    test_labels: np.ndarray,
    test_probs_cached: np.ndarray,  # Use cached instead of recomputing
    ood_features: np.ndarray,
    ood_raw: np.ndarray,
    ood_labels: np.ndarray,
    ood_probs_cached: np.ndarray,  # Use cached instead of recomputing
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Reusable OOD evaluation that uses cached model predictions.
    """
    # Calibrate test using fitted calibrator
    calibrated_probs = geo_cal.calibrate_batched_precomputed(
        X_test_embed=test_features,
        X_test_original=test_raw,
        model_probs=test_probs_cached,  # Cached!
        batch_size=batch_size
    )
    
    # Calibrate OOD using same calibrator
    ood_calibrated_probs = geo_cal.calibrate_batched_precomputed(
        X_test_embed=ood_features,
        X_test_original=ood_raw,
        model_probs=ood_probs_cached,  # Cached!
        batch_size=batch_size
    )
    
    # Compute metrics
    return compute_ood_metrics(
        calibrated_probs_id=calibrated_probs,
        calibrated_probs_ood=ood_calibrated_probs,
        id_labels=test_labels,
        ood_labels=ood_labels,
    )
'''


# =============================================================================
# SUMMARY OF CHANGES AND ESTIMATED TIME SAVINGS
# =============================================================================
"""
PRIORITY | FIX                          | TIME SAVED    | DIFFICULTY
---------|------------------------------|---------------|------------
P0       | Optimization loop bug        | 5-30 minutes  | Easy (1 line)
P0       | Prediction caching in main   | 2-10 minutes  | Easy (add block)
P1       | DataLoader caching           | ~30 seconds   | Easy (add block)
P1       | OOD feature caching          | 1-5 minutes   | Medium
P1       | Updated function signatures  | Enables above | Medium
P2       | Reusable OOD evaluation      | Code quality  | Medium

TOTAL ESTIMATED SPEEDUP: 2-4x for full experiment suite
MOST CRITICAL FIX: P0 #1 (optimization loop) - this alone can save 5-30 minutes
"""


# =============================================================================
# STEP-BY-STEP APPLICATION GUIDE
# =============================================================================
"""
1. FIRST (takes 1 minute):
   Apply FIX #1 to run_geometric_with_dac_weighting()
   - Change line 5173 from:
       val_logits_local = model_adapter.predict_proba(val_raw)
     to:
       (delete the line, use val_logits_precomputed from earlier)
   - Make sure val_logits_precomputed is defined before objective_function

2. SECOND (takes 5 minutes):
   Add prediction caching in main() (FIX #2)
   - Add the prediction_cache block after extracting raw data
   - Pass prediction_cache to experiment functions that need it

3. THIRD (takes 5 minutes):
   Add DataLoader caching in main() (FIX #3)
   - Add the dataloader_cache block
   - Replace DataLoader(...) calls in experiment functions with cached loaders

4. FOURTH (takes 10-15 minutes):
   Add OOD feature caching (FIX #4)
   - Add the ood_feature_cache
   - Modify experiment functions to check cache before extracting

5. FIFTH (takes 15-20 minutes):
   Update function signatures (FIX #5)
   - Add new parameters for cached data
   - Update all call sites in main()

6. SIXTH (optional, takes 30 minutes):
   Create reusable OOD evaluation function (FIX #6)
   - Extract common OOD evaluation code to a function
   - Replace duplicate code in all experiment functions
"""
