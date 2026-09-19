import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import logging
from typing import List, Tuple, Optional

# Import existing NC metrics
from Calibrators.nc_metrics import NC1CollapseMetricCalculator, NC4SeparabilityCalculator

from utils.logging_config import get_logger
logger = get_logger(__name__)


class TuckerProjector:
    """
    Project high-dimensional convolutional features using Tucker decomposition.
    
    For conv layer output of shape (N, C, H, W), we:
    1. Reshape to (N, C, H*W)
    2. Compute channel-wise covariance
    3. Tucker decomposition to get projection matrices
    4. Project and flatten to (N, c_proj * d_proj)
    
    NOTE: This is OPTIONAL since extract_features_directly() already applies SPP+JL.
    For simplicity, we skip Tucker and use features as-is.
    """
    
    def __init__(self, c_proj=10, d_proj=10):
        """
        Args:
            c_proj: target channel dimension
            d_proj: target spatial dimension per channel
        """
        self.c_proj = c_proj
        self.d_proj = d_proj
        
        # Learned parameters (fit on training data)
        self.A = None  # Channel projection
        self.B = None  # Spatial projection
        self.channel_mean = None
        self.channel_std = None
    
    def fit(self, features_4d):
        """
        Fit Tucker decomposition on training features.
        
        Args:
            features_4d: (N, C, H, W) array of conv features
        """
        N, C, H, W = features_4d.shape
        
        # Reshape to (N, C, H*W)
        features_reshaped = features_4d.reshape(N, C, H * W)
        
        # Compute channel-wise statistics
        self.channel_mean = features_reshaped.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
        self.channel_std = features_reshaped.std(axis=(0, 2), keepdims=True) + 1e-8
        
        # Normalize
        features_norm = (features_reshaped - self.channel_mean) / self.channel_std
        
        # Compute channel-wise covariance (C, H*W, H*W)
        channel_covs = []
        for c in range(C):
            cov_c = np.cov(features_norm[:, c, :].T)  # (H*W, H*W)
            channel_covs.append(cov_c)
        channel_covs = np.array(channel_covs)  # (C, H*W, H*W)
        
        # Simplified Tucker: SVD on average covariance
        avg_cov = channel_covs.mean(axis=0)  # (H*W, H*W)
        
        # Get spatial projection (top d_proj singular vectors)
        U, S, Vh = np.linalg.svd(avg_cov, full_matrices=False)
        self.B = U[:, :self.d_proj]  # (H*W, d_proj)
        
        # Get channel projection (simple PCA on channel means)
        channel_features = features_norm.mean(axis=2).T  # (C, N)
        U_c, S_c, Vh_c = np.linalg.svd(channel_features @ channel_features.T, full_matrices=False)
        self.A = U_c[:, :self.c_proj]  # (C, c_proj)
        
        logger.info(f"TuckerProjector fitted: {C}×{H*W} → {self.c_proj}×{self.d_proj}")
    
    def transform(self, features_4d):
        """
        Project features using learned Tucker decomposition.
        
        Args:
            features_4d: (N, C, H, W) array
            
        Returns:
            projected: (N, c_proj * d_proj) array
        """
        N, C, H, W = features_4d.shape
        
        # Reshape and normalize
        features_reshaped = features_4d.reshape(N, C, H * W)
        features_norm = (features_reshaped - self.channel_mean) / self.channel_std
        
        # Project spatially: (N, C, H*W) @ (H*W, d_proj) → (N, C, d_proj)
        features_spatial = features_norm @ self.B  # (N, C, d_proj)
        
        # Project channels: (N, C, d_proj) → (N, c_proj, d_proj)
        # Reshape to (N, d_proj, C) for matmul
        features_t = features_spatial.transpose(0, 2, 1)  # (N, d_proj, C)
        features_both = features_t @ self.A  # (N, d_proj, c_proj)
        features_both = features_both.transpose(0, 2, 1)  # (N, c_proj, d_proj)
        
        # Flatten
        projected = features_both.reshape(N, self.c_proj * self.d_proj)
        
        return projected


class ProbabilisticSkipConnection:
    """
    PSC: Probabilistic Skip Connections for deterministic UQ.
    
    Steps:
    1. Identify candidate layer using NC metrics (NC1 and NC4)
    2. Extract features from that layer
    3. Project using Tucker decomposition (or SPP+JL from extract_features_directly)
    4. Fit probabilistic model (QDA for OOD, isotonic for iD)
    """
    
    def __init__(self, model_adapter, candidate_layers, device='cuda', epsilon=0.2,
                 use_tucker=True, c_proj=64, d_proj=16):
        """
        Args:
            model_adapter: PyTorchModelAdapter instance
            candidate_layers: list of layer indices to consider
            device: 'cuda' or 'cpu'
            epsilon: NC1 threshold (default 0.2 from PSC paper)
            use_tucker: If True, use TuckerCompression; if False, use SPP+JL from extract_features_directly
            c_proj: Target channel dimension for Tucker compression
            d_proj: Target spatial dimension for Tucker compression
        """
        self.model_adapter = model_adapter
        self.candidate_layers = candidate_layers
        self.device = device
        self.epsilon = epsilon
        self.use_tucker = use_tucker
        self.c_proj = c_proj
        self.d_proj = d_proj
        
        # Use existing NC metric calculators
        self.nc1_calculator = NC1CollapseMetricCalculator()
        self.nc4_calculator = NC4SeparabilityCalculator()
        
        self.selected_layer = None
        self.tucker_compressor = None  # TuckerCompression module
        self.qda = None  # For OOD detection
        
        # For iD calibration (isotonic on projected features)
        self.iso_calibrator = None
    
    def _extract_raw_layer_output(self, model, dataloader, layer_idx):
        """
        Extract raw 4D features (N, C, H, W) from a specific layer.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader
            layer_idx: Layer index to extract from
            
        Returns:
            features: (N, C, H, W) tensor of raw features
        """
        activation = {}
        def hook(module, input, output):
            activation['output'] = output
        
        layers = list(model.modules())
        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of bounds")
        
        target_layer = layers[layer_idx]
        handle = target_layer.register_forward_hook(hook)
        
        features_list = []
        model.eval()
        
        try:
            with torch.no_grad():
                for batch in dataloader:
                    if len(batch) == 2:
                        data, _ = batch
                    else:
                        data = batch[0]
                    
                    data = data.to(self.device).float()
                    activation.clear()
                    
                    _ = model(data)
                    
                    feat = activation.get('output', None)
                    if feat is None or not torch.is_tensor(feat):
                        continue
                    
                    # Ensure 4D (N, C, H, W)
                    if feat.ndim == 2:
                        # If 2D, assume it's (N, C) and add dummy spatial dims
                        feat = feat.unsqueeze(-1).unsqueeze(-1)
                    elif feat.ndim == 3:
                        # If 3D, might be (N, L, C) - need to handle based on context
                        # For now, assume it needs reshaping
                        logger.warning(f"Layer {layer_idx}: 3D features detected, shape {feat.shape}")
                        continue
                    
                    features_list.append(feat.cpu())
        finally:
            handle.remove()
        
        if not features_list:
            raise ValueError(f"No features extracted from layer {layer_idx}")
        
        features = torch.cat(features_list, dim=0)
        return features
    
    def select_layer(self, train_raw, train_labels, val_raw, val_labels, batch_size):
        """
        Select best layer using Neural Collapse metrics on validation set.
        
        Strategy from PSC paper:
        1. Compute NC1 and NC4 for each candidate layer
        2. Filter to layers with NC1 > ε (not collapsed)
        3. Among non-collapsed layers, pick layer with highest NC4
        
        Returns:
            selected_layer: int, the chosen layer index
        """
        logger.info("="*80)
        logger.info("PSC LAYER SELECTION: Analyzing Neural Collapse metrics")
        logger.info("="*80)
        logger.info(f"Analyzing {len(self.candidate_layers)} candidate layers...")
        logger.info(f"NC1 threshold (ε): {self.epsilon}")
        
        raw_model = self.model_adapter.model
        dev = torch.device(self.device)
        
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
            batch_size=batch_size, shuffle=False
        )
        
        layer_scores = []
        
        for layer_idx in self.candidate_layers:
            # Extract features using existing infrastructure
            from Experiments.layer_selection import extract_features_directly
            
            val_feats, _ = extract_features_directly(
                raw_model, val_loader, layer_idx, dev
            )
            
            # Convert to tensor for NC metrics
            val_feats_tensor = val_feats.to(dev)
            val_labels_tensor = torch.from_numpy(val_labels).to(dev)
            
            # Compute NC1 using existing implementation
            nc1 = self.nc1_calculator.compute(val_feats_tensor, val_labels_tensor)
            
            # Compute NC4 using existing implementation
            nc4 = self.nc4_calculator.compute(val_feats_tensor, val_labels_tensor)
            
            is_collapsed = nc1 < self.epsilon
            is_good = (not is_collapsed) and (nc4 > 0.85)
            
            logger.info(
                f"  Layer {layer_idx:3d}: NC1={nc1:.3f} {'✗ COLLAPSED' if is_collapsed else '✓'}, "
                f"NC4={nc4:.3f} {'✓ GOOD' if is_good else ''}"
            )
            
            layer_scores.append({
                'layer': layer_idx,
                'nc1': nc1,
                'nc4': nc4,
                'is_collapsed': is_collapsed,
                'is_good': is_good
            })
        
        # Filter to non-collapsed layers (NC1 > ε)
        non_collapsed = [s for s in layer_scores if s['nc1'] >= self.epsilon]
        
        if not non_collapsed:
            logger.warning(f"⚠️  No non-collapsed layers found (NC1 > {self.epsilon})!")
            logger.warning("    Falling back to layer with highest NC1")
            self.selected_layer = max(layer_scores, key=lambda x: x['nc1'])['layer']
            selected_nc1 = max(layer_scores, key=lambda x: x['nc1'])['nc1']
            selected_nc4 = max(layer_scores, key=lambda x: x['nc1'])['nc4']
        else:
            # Among non-collapsed, pick highest NC4
            best = max(non_collapsed, key=lambda x: x['nc4'])
            self.selected_layer = best['layer']
            selected_nc1 = best['nc1']
            selected_nc4 = best['nc4']
            
            logger.info(f"\n✓ Found {len(non_collapsed)} non-collapsed layers")
        
        logger.info("="*80)
        logger.info(f"🎯 SELECTED LAYER: {self.selected_layer}")
        logger.info(f"   NC1 = {selected_nc1:.4f} (threshold: {self.epsilon})")
        logger.info(f"   NC4 = {selected_nc4:.4f} (NCC accuracy)")
        logger.info("="*80 + "\n")
        
        return self.selected_layer
    
    def fit(self, train_raw, train_labels, val_raw, val_labels, batch_size):
        """
        Fit PSC calibrator.
        
        Steps:
        1. Select layer if not already done
        2. Extract features from selected layer
        3. Fit Tucker projector (if use_tucker=True) or use SPP+JL from extract_features_directly
        4. Fit QDA for OOD detection
        5. Fit isotonic calibrator for iD
        """
        from Calibrators.isotonic_regression import TopLabelIsotonicCalibrator
        from utils.compression_utils import TuckerCompression
        
        # Step 1: Select layer
        if self.selected_layer is None:
            self.select_layer(train_raw, train_labels, val_raw, val_labels, batch_size)
        
        logger.info(f"Fitting PSC on layer {self.selected_layer}...")
        logger.info(f"  Compression method: {'Tucker' if self.use_tucker else 'SPP+JL'}")
        
        raw_model = self.model_adapter.model
        dev = torch.device(self.device)
        
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
            batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
            batch_size=batch_size, shuffle=False
        )
        
        if self.use_tucker:
            # Step 2: Extract raw 4D features for Tucker compression
            logger.info("  Extracting raw train features (4D)...")
            train_feats_4d = self._extract_raw_layer_output(raw_model, train_loader, self.selected_layer)
            
            logger.info("  Extracting raw val features (4D)...")
            val_feats_4d = self._extract_raw_layer_output(raw_model, val_loader, self.selected_layer)
            
            # Get channel count from first batch
            _, in_channels, _, _ = train_feats_4d.shape
            
            # Step 3: Fit Tucker compression
            logger.info(f"  Fitting TuckerCompression: {in_channels} channels -> {self.c_proj}x{self.d_proj}")
            self.tucker_compressor = TuckerCompression(
                in_channels=in_channels,
                c_proj=self.c_proj,
                d_proj=self.d_proj,
                device=self.device
            ).to(dev)
            
            # Fit on training data
            self.tucker_compressor.fit(
                train_loader, 
                self.model_adapter, 
                self.selected_layer,
                num_batches=50
            )
            
            # Apply compression to get final features
            logger.info("  Applying Tucker compression to train features...")
            train_feats_compressed = []
            with torch.no_grad():
                for i in range(0, len(train_feats_4d), batch_size):
                    batch_feats = train_feats_4d[i:i+batch_size].to(dev)
                    compressed = self.tucker_compressor(batch_feats)
                    train_feats_compressed.append(compressed.cpu())
            train_feats = torch.cat(train_feats_compressed, dim=0)
            
            logger.info("  Applying Tucker compression to val features...")
            val_feats_compressed = []
            with torch.no_grad():
                for i in range(0, len(val_feats_4d), batch_size):
                    batch_feats = val_feats_4d[i:i+batch_size].to(dev)
                    compressed = self.tucker_compressor(batch_feats)
                    val_feats_compressed.append(compressed.cpu())
            val_feats = torch.cat(val_feats_compressed, dim=0)
            
            train_feats_np = train_feats.numpy()
            val_feats_np = val_feats.numpy()
        else:
            # Step 2: Extract features using SPP+JL (existing method)
            from Experiments.layer_selection import extract_features_directly
            
            logger.info("  Extracting train features (SPP+JL)...")
            train_feats, _ = extract_features_directly(raw_model, train_loader, self.selected_layer, dev)
            
            logger.info("  Extracting val features (SPP+JL)...")
            val_feats, _ = extract_features_directly(raw_model, val_loader, self.selected_layer, dev)
            
            train_feats_np = train_feats.cpu().numpy()
            val_feats_np = val_feats.cpu().numpy()
        
        self.final_train_feats = train_feats_np
        self.final_val_feats = val_feats_np
        
        # Step 4: Fit QDA for OOD detection (epistemic uncertainty)
        logger.info("  Fitting QDA for epistemic uncertainty (OOD detection)...")
        self.qda = QuadraticDiscriminantAnalysis()
        self.qda.fit(self.final_train_feats, train_labels)
        
        # Compute density statistics on validation set for normalization
        val_log_densities = self.qda.predict_log_proba(self.final_val_feats).max(axis=1)
        self.density_mean = val_log_densities.mean()
        self.density_std = val_log_densities.std()
        logger.info(f"    Density stats: mean={self.density_mean:.2f}, std={self.density_std:.2f}")
        
        # Step 5: Fit isotonic calibrator for iD (aleatoric uncertainty)
        logger.info("  Fitting isotonic calibrator for aleatoric uncertainty (iD)...")
        val_logits = self.model_adapter.predict_logits(val_raw, batch_size=batch_size)
        self.iso_calibrator = TopLabelIsotonicCalibrator()
        self.iso_calibrator.fit(val_logits, val_labels)
        
        logger.info("✅ PSC fitting complete\n")
        self.is_fitted = True
    
    def get_feature_density(self, test_raw, batch_size):
        """
        Compute feature density for OOD detection (epistemic uncertainty).
        
        Returns:
            densities: (N,) array of log-densities (higher = more in-distribution)
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before get_feature_density()")
        
        raw_model = self.model_adapter.model
        dev = torch.device(self.device)
        
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
            batch_size=batch_size, shuffle=False
        )
        
        if self.use_tucker and self.tucker_compressor is not None:
            # Extract raw 4D features and apply Tucker compression
            test_feats_4d = self._extract_raw_layer_output(raw_model, test_loader, self.selected_layer)
            
            test_feats_compressed = []
            with torch.no_grad():
                for i in range(0, len(test_feats_4d), batch_size):
                    batch_feats = test_feats_4d[i:i+batch_size].to(dev)
                    compressed = self.tucker_compressor(batch_feats)
                    test_feats_compressed.append(compressed.cpu())
            test_feats = torch.cat(test_feats_compressed, dim=0)
            test_feats_np = test_feats.numpy()
        else:
            from Experiments.layer_selection import extract_features_directly
            test_feats, _ = extract_features_directly(raw_model, test_loader, self.selected_layer, dev)
            test_feats_np = test_feats.cpu().numpy()
        
        # Get log probability from QDA (epistemic uncertainty)
        log_probs = self.qda.predict_log_proba(test_feats_np)
        densities = log_probs.max(axis=1)  # Max log prob across classes
        
        return densities
    
    def calibrate(self, test_raw, batch_size):
        """
        Calibrate test predictions using PSC.
        
        PSC combines two types of uncertainty:
        - Epistemic: feature density from selected layer (OOD detection via QDA)
        - Aleatoric: isotonic-calibrated probabilities (in-distribution calibration)
        
        For low-density (OOD-like) samples, reduce confidence by interpolating
        toward uniform distribution.
        
        Returns:
            calibrated_probs: (N, C) array of calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before calibrate()")
        
        # Extract features from selected layer for epistemic uncertainty
        raw_model = self.model_adapter.model
        dev = torch.device(self.device)
        
        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(test_raw),
                torch.zeros(len(test_raw), dtype=torch.long)
            ),
            batch_size=batch_size, shuffle=False
        )
        
        if self.use_tucker and self.tucker_compressor is not None:
            # Extract raw 4D features and apply Tucker compression
            test_feats_4d = self._extract_raw_layer_output(raw_model, test_loader, self.selected_layer)
            
            test_feats_compressed = []
            with torch.no_grad():
                for i in range(0, len(test_feats_4d), batch_size):
                    batch_feats = test_feats_4d[i:i+batch_size].to(dev)
                    compressed = self.tucker_compressor(batch_feats)
                    test_feats_compressed.append(compressed.cpu())
            test_feats = torch.cat(test_feats_compressed, dim=0)
            test_feats_np = test_feats.numpy()
        else:
            from Experiments.layer_selection import extract_features_directly
            test_feats, _ = extract_features_directly(raw_model, test_loader, self.selected_layer, dev)
            test_feats_np = test_feats.cpu().numpy()
        
        # Epistemic uncertainty: QDA log-density (higher = more in-distribution)
        log_densities = self.qda.predict_log_proba(test_feats_np).max(axis=1)
        
        # Convert to confidence weights [0, 1] using sigmoid on normalized densities
        # Normalize by validation set statistics
        normalized_densities = (log_densities - self.density_mean) / (self.density_std + 1e-8)
        density_weights = 1.0 / (1.0 + np.exp(-normalized_densities))  # sigmoid
        
        # Aleatoric uncertainty: isotonic calibration on original logits
        test_logits = self.model_adapter.predict_logits(test_raw, batch_size=batch_size)
        iso_probs = self.iso_calibrator.calibrate(test_logits)
        
        # Combine: interpolate between isotonic probs and uniform for low-density samples
        # High density_weight (in-distribution) → use iso_probs
        # Low density_weight (OOD) → move toward uniform distribution
        num_classes = iso_probs.shape[1]
        uniform = np.ones_like(iso_probs) / num_classes
        
        # Weighted combination: density_weights[:, None] broadcasts to (N, C)
        calibrated_probs = (density_weights[:, None] * iso_probs + 
                           (1.0 - density_weights[:, None]) * uniform)
        
        return calibrated_probs

