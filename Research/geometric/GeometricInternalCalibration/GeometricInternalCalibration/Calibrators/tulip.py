import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegressionCV
import logging
from typing import List, Tuple, Optional, Dict, Union

from utils.logging_config import get_logger
logger = get_logger(__name__)

# ==============================================================================
# Layer selection helpers for TULIP IC placement
# ==============================================================================
def select_tulip_layers(model, model_name: str, num_fallback: int = 7) -> List[str]:
    """
    Select strategic layers for TULIP internal classifiers.
    Strategy: Target the last BatchNorm of residual blocks.
    - ResNet-18/34 (BasicBlock): Target .bn2
    - ResNet-50+ (Bottleneck): Target .bn3
    """
    all_layers = []
    for name, module in model.named_modules():
        # Keep only leaf modules (no children) with a non-empty name
        if name and not list(module.children()):
            all_layers.append(name)

    model_name = model_name.lower()

    # Architecture-specific target patterns
    if "resnet18" in model_name:
        # 9 Layers: Stem + End of every block
        target_patterns = [
            "bn1",             # Stem
            "layer1.0.bn2", "layer1.1.bn2",
            "layer2.0.bn2", "layer2.1.bn2",
            "layer3.0.bn2", "layer3.1.bn2",
            "layer4.0.bn2", "layer4.1.bn2"
        ]
    elif "resnet34" in model_name:
        # ~9 Layers spaced out (ResNet34 has 3,4,6,3 blocks)
        target_patterns = [
            "bn1",             # Stem
            "layer1.2.bn2",    # L1 End
            "layer2.1.bn2", "layer2.3.bn2",  # L2 Mid/End
            "layer3.1.bn2", "layer3.3.bn2", "layer3.5.bn2", # L3 Early/Mid/End
            "layer4.0.bn2", "layer4.2.bn2"   # L4 Start/End
        ]
    elif "resnet50" in model_name:
        # ~9 Layers spaced out (ResNet50 has 3,4,6,3 Bottleneck blocks)
        # Note: Bottleneck blocks end with .bn3, not .bn2
        target_patterns = [
            "bn1",             # Stem
            "layer1.2.bn3",    # L1 End
            "layer2.1.bn3", "layer2.3.bn3",  # L2 Mid/End
            "layer3.1.bn3", "layer3.3.bn3", "layer3.5.bn3", # L3 Early/Mid/End
            "layer4.1.bn3", "layer4.2.bn3"   # L4 Mid/End
        ]
    elif "resnet101" in model_name:
        # Spaced out for depth (3,4,23,3 blocks)
        target_patterns = [
            "bn1", "layer1.2.bn3", "layer2.3.bn3",
            "layer3.3.bn3", "layer3.7.bn3", "layer3.11.bn3", "layer3.15.bn3", "layer3.19.bn3", "layer3.22.bn3",
            "layer4.2.bn3"
        ]
    elif "resnet152" in model_name:
        target_patterns = [
            "bn1",                  # Stem
            "layer1.2.bn3",         # L1 End
            "layer2.3.bn3", "layer2.7.bn3",  # L2 Mid/End
            "layer3.7.bn3", "layer3.15.bn3", "layer3.23.bn3", "layer3.31.bn3", "layer3.35.bn3",  # L3 Distributed
            "layer4.0.bn3", "layer4.2.bn3"   # L4 Start/End
        ]

    else:
        # Generic evenly spaced fallback
        n_layers = len(all_layers)
        step = max(1, n_layers // num_fallback)
        return [all_layers[i] for i in range(0, n_layers, step)][:num_fallback]

    # Match target patterns to actual layer names
    selected: List[str] = []
    # Create a set for O(1) lookups to avoid partial matching issues
    all_layers_set = set(all_layers)
    
    for pattern in target_patterns:
        # Exact match check first (safest)
        if pattern in all_layers_set:
            selected.append(pattern)
            continue
            
        # Fallback to substring matching if exact name not found
        # (Useful if model wrappers add prefixes)
        for layer_name in all_layers:
            if pattern in layer_name:
                # Ensure we take the most specific match
                if not any(pattern in other and len(other) > len(layer_name) for other in all_layers):
                    selected.append(layer_name)
                    break

    # Fallback if pattern matching fails
    if len(selected) < 5:
        logger.warning(f"Pattern matching found only {len(selected)} layers for {model_name}; using evenly spaced fallback")
        n_layers = len(all_layers)
        step = max(1, n_layers // num_fallback)
        selected = [all_layers[i] for i in range(0, n_layers, step)][:num_fallback]

    return selected


def select_tulip_layers_simple(model, num_ics: int = 7) -> List[str]:
    """
    Simpler rule-based selection: choose ~num_ics leaf layers at block boundaries,
    evenly spaced through the network.
    """
    block_layers: List[str] = []
    for name, module in model.named_modules():
        if any(f"layer{i}" in name for i in [1, 2, 3, 4]) and name and not list(module.children()):
            block_layers.append(name)

    if len(block_layers) <= num_ics:
        return block_layers

    step = len(block_layers) / num_ics
    indices = [int(i * step) for i in range(num_ics)]
    return [block_layers[i] for i in indices]


# ==============================================================================
# 1. Random Fourier Features (RFF) Layer
#    Implements the kernel approximation for "Distance Awareness"
#    NOTE: matches original TULIP implementation (orthogonal blocks + norm)
# ==============================================================================
def random_ortho(n: int, m: int) -> torch.Tensor:
    """QR-based orthogonal initializer."""
    q, _ = torch.linalg.qr(torch.randn(n, m))
    return q


class RandomFourierFeatures(nn.Module):
    """
    Orthogonal RFF layer to approximate the RBF kernel.
    Uses block-orthogonal weights with feature normalization and divides
    by feature_scale (as in the original codebase).
    """
    def __init__(self, in_dim: int, num_random_features: int, feature_scale: Optional[float] = None):
        super().__init__()
        if feature_scale is None:
            feature_scale = math.sqrt(num_random_features / 2)

        self.register_buffer("feature_scale", torch.tensor(feature_scale))

        # Build orthogonal blocks
        if num_random_features <= in_dim:
            W = random_ortho(in_dim, num_random_features)
        else:
            dim_left = num_random_features
            ws = []
            while dim_left > in_dim:
                ws.append(random_ortho(in_dim, in_dim))
                dim_left -= in_dim
            ws.append(random_ortho(in_dim, dim_left))
            W = torch.cat(ws, 1)

        # Feature normalization (edward2-style)
        feature_norm = torch.randn(W.shape) ** 2
        W = W * feature_norm.sum(0).sqrt()

        self.register_buffer("W", W)
        b = torch.empty(num_random_features).uniform_(0, 2 * math.pi)
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.cos(x @ self.W + self.b)
        k = k / self.feature_scale
        return k

# ==============================================================================
# 2. Laplace GP Head (JL -> LayerNorm -> RFF -> beta)
#    Precision updated inside training forward pass (matches reference)
# ==============================================================================
class Laplace(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        num_deep_features: int,
        num_gp_features: int,
        normalize_gp_features: bool,
        num_random_features: int,
        num_outputs: int,
        num_data: int,
        train_batch_size: int,
        ridge_penalty: float = 1.0,
        feature_scale: Optional[float] = None,
        mean_field_factor: Optional[float] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.mean_field_factor = mean_field_factor
        self.ridge_penalty = ridge_penalty

        if num_gp_features > 0:
            self.num_gp_features = num_gp_features
            self.register_buffer(
                "random_matrix",
                torch.normal(0, 0.05, (num_gp_features, num_deep_features)),
            )
            self.jl = lambda x: nn.functional.linear(x, self.random_matrix)
        else:
            self.num_gp_features = num_deep_features
            self.jl = nn.Identity()

        self.normalize_gp_features = normalize_gp_features
        if normalize_gp_features:
            self.normalize = nn.LayerNorm(num_gp_features)

        self.rff = RandomFourierFeatures(self.num_gp_features, num_random_features, feature_scale)
        self.beta = nn.Linear(num_random_features, num_outputs)

        precision = torch.eye(num_random_features) * self.ridge_penalty
        self.register_buffer("precision", precision)
        self.register_buffer("seen_data", torch.tensor(0))
        self.register_buffer("covariance", torch.eye(num_random_features))
        self.recompute_covariance = True

    def mean_field_logits(self, logits: torch.Tensor, pred_cov: torch.Tensor) -> torch.Tensor:
        logits_scale = torch.sqrt(1.0 + torch.diag(pred_cov) * self.mean_field_factor)
        if self.mean_field_factor is not None and self.mean_field_factor > 0:
            logits = logits / logits_scale.unsqueeze(-1)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature_extractor(x)
        f_reduc = self.jl(f)
        if self.normalize_gp_features:
            f_reduc = self.normalize(f_reduc)

        k = self.rff(f_reduc)
        pred = self.beta(k)

        if self.training:
            # Detach features before precision update to avoid graph growth / memory leak
            k_detached = k.detach()
            precision_minibatch = k_detached.t() @ k_detached
            self.precision += precision_minibatch
            self.seen_data += x.shape[0]
            self.recompute_covariance = True
            return pred

        if self.recompute_covariance:
            with torch.no_grad():
                eps = 1e-7
                jitter = eps * torch.eye(self.precision.shape[1], device=self.precision.device)
                u, _ = torch.linalg.cholesky_ex(self.precision + jitter)
                torch.cholesky_inverse(u, out=self.covariance)
            self.recompute_covariance = False

        with torch.no_grad():
            pred_cov = k @ ((self.covariance @ k.t()) * self.ridge_penalty)

        if self.mean_field_factor is not None:
            pred = self.mean_field_logits(pred, pred_cov)

        return pred


def feature_reduction_formula(input_size: int) -> int:
    """
    Simplified reduction kernel size: reduce to 1x1 if possible.
    """
    if input_size <= 0:
        return -1
    return input_size


class InternalFeatureExtractor(nn.Module):
    def __init__(self, input_size: int, output_channels: int, num_classes: int, num_features: int = 128):
        super().__init__()
        red_kernel_size = feature_reduction_formula(input_size)

        if red_kernel_size == -1 or input_size <= 1:
            self.linear1 = nn.Linear(output_channels * max(1, input_size) * max(1, input_size), num_features)
            self.use_pooling = False
        else:
            red_input_size = int(input_size / red_kernel_size)
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            self.linear1 = nn.Linear(output_channels * red_input_size * red_input_size, num_features)
            self.use_pooling = True

        self.bn1 = nn.BatchNorm1d(num_features)

    def features_w_pooling(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_pooling:
            out = self.linear1(x.view(x.size(0), -1))
            return self.bn1(out)
        avgp = self.alpha * self.max_pool(x)
        maxp = (1 - self.alpha) * self.avg_pool(x)
        mixed = avgp + maxp
        out = self.bn1(self.linear1(mixed.view(mixed.size(0), -1)))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            out = self.linear1(x)
            return self.bn1(out)
        return self.features_w_pooling(x)


class InternalClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_channels: int,
        num_classes: int,
        data_cfg: Optional[object] = None,
        batch_size: int = 256,
        num_features: int = 128,
    ):
        super().__init__()
        num_data = getattr(data_cfg, "train_len", 1) if data_cfg is not None else 1
        feature_extractor = InternalFeatureExtractor(input_size, output_channels, num_classes, num_features)

        self.head = Laplace(
            feature_extractor,
            num_deep_features=num_features,
            num_gp_features=64,
            normalize_gp_features=True,
            num_random_features=512,
            num_outputs=num_classes,
            num_data=num_data,
            train_batch_size=batch_size,
            ridge_penalty=1.0,
            feature_scale=2,
            mean_field_factor=25,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

# ==============================================================================
# 3. Efficient TULIP Calibrator (Optimized for Cluster/Thesis)
# ==============================================================================
class TULIPCalibrator:
    def __init__(
        self,
        model,  # The raw PyTorch model
        candidate_layers: List[str],  # Layer names, e.g. "layer1", "layer2.0.conv1"
        num_classes: int = 10,
        device: str = 'cuda'
    ):
        self.model = model
        self.candidate_layers = candidate_layers
        self.num_classes = num_classes
        self.device = device
        
        # Storage
        self.internal_classifiers: Dict[str, InternalClassifier] = {}
        self.combination_weights: Optional[np.ndarray] = None
        self.is_fitted = False

    def _compute_ds_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        # Eq 19 in paper: u(x) = K / (K + sum(exp(g(x))))
        # Note: logits here are already variance-scaled by the GP Head
        K = logits.shape[1]
        log_sum_exp = torch.logsumexp(logits, dim=1)
        uncertainty = K / (K + torch.exp(log_sum_exp))
        return uncertainty

    def _extract_all_features(self, loader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Extract features incrementally to avoid memory spikes.
        """
        self.model.eval()
        self.model.to(self.device)
        
        num_samples = len(loader.dataset)
        features_dict: Dict[str, np.ndarray] = {}
        
        # First pass: determine feature dimensions
        with torch.no_grad():
            sample_batch = next(iter(loader))[0][:1].to(self.device)
            
            feature_dims: Dict[str, int] = {}
            def get_dim_hook(name):
                def hook(model, input, output):
                    val = output.detach()
                    if len(val.shape) == 4:
                        feature_dims[name] = val.shape[1]
                    else:
                        feature_dims[name] = val.shape[1]
                return hook
            
            hooks = []
            for name, module in self.model.named_modules():
                if name in self.candidate_layers:
                    hooks.append(module.register_forward_hook(get_dim_hook(name)))
            
            _ = self.model(sample_batch)
            for h in hooks:
                h.remove()
        
        # Pre-allocate numpy arrays
        for name in self.candidate_layers:
            features_dict[name] = np.zeros((num_samples, feature_dims[name]), dtype=np.float32)
        
        # Second pass: fill arrays directly
        current_idx = [0]  # mutable for nested scope
        
        def get_activation(name):
            def hook(model, input, output):
                val = output.detach()
                
                if len(val.shape) == 4:
                    val = F.adaptive_avg_pool2d(val, (1, 1)).flatten(1)
                
                batch_size = val.shape[0]
                start = current_idx[0]
                end = start + batch_size
                features_dict[name][start:end] = val.cpu().numpy()
            return hook
        
        hooks = []
        for name, module in self.model.named_modules():
            if name in self.candidate_layers:
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        logger.info("Extracting features (single forward pass)...")
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                batch_size = batch_x.shape[0]
                
                _ = self.model(batch_x)
                current_idx[0] += batch_size
                
                del batch_x
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        for h in hooks:
            h.remove()
        
        return features_dict

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100):
        """
        Fit TULIP in two stages:
        1. Train Internal Classifiers (ICs) on Training Set
        2. Fit Combination Head on Validation Set (Unsupervised Switches)
        """
        dev = torch.device(self.device)
        
        # --- Stage 1: Extract Training Features & Train ICs ---
        logger.info("Extracting Training Features...")
        train_feats_by_layer = self._extract_all_features(train_loader)
        
        # Get labels from loader (assumed accessible via loop)
        train_labels = []
        for _, y in train_loader:
            train_labels.append(y)
        train_labels = torch.cat(train_labels).numpy()

        for layer_name, feats in train_feats_by_layer.items():
            logger.info(f"Training IC for layer: {layer_name}")
            
            # Setup IC
            if feats.ndim == 4:
                input_size = feats.shape[2]
                output_channels = feats.shape[1]
            else:
                input_size = 1
                output_channels = feats.shape[1]

            ic = InternalClassifier(
                input_size=input_size,
                output_channels=output_channels,
                num_classes=self.num_classes,
                data_cfg=None,
                batch_size=train_loader.batch_size or 256,
            ).to(dev)
            
            # Train IC (Standard optimization) – higher LR now that Laplace update is detached
            optimizer = torch.optim.Adam(ic.parameters(), lr=1e-1)
            criterion = nn.CrossEntropyLoss()
            
            ic_feats = torch.from_numpy(feats)
            ic_labels = torch.from_numpy(train_labels).long()
            
            # Mini-batch training for IC
            ds = TensorDataset(ic_feats, ic_labels)
            dl = DataLoader(ds, batch_size=256, shuffle=True)
            
            ic.train()
            
            # 1. Optimize Weights (Beta)
            for ep in range(epochs):
                correct = 0
                total = 0
                epoch_loss = 0.0
                for b_x, b_y in dl:
                    b_x, b_y = b_x.to(dev), b_y.to(dev)
                    optimizer.zero_grad()
                    # Standard forward (returns logits)
                    logits = ic(b_x)
                    loss = criterion(logits, b_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == b_y).sum().item()
                    total += b_y.size(0)
                if (ep + 1) % 10 == 0 or ep == 0:
                    acc = 100 * correct / total if total > 0 else 0.0
                    logger.info(f"  [Epoch {ep+1}/{epochs}] Loss: {epoch_loss/len(dl):.4f} | Acc: {acc:.2f}%")
            
            self.internal_classifiers[layer_name] = ic

        # --- Stage 2: Fit Combination Head on Validation (Switches) ---
        logger.info("Extracting Validation Features...")
        val_feats_by_layer = self._extract_all_features(val_loader)
        
        val_uncertainties: List[torch.Tensor] = []
        val_preds: List[torch.Tensor] = []
        
        # IMPORTANT: Ensure layers are sorted by depth for correct switch logic
        sorted_layers = self.candidate_layers 
        
        for name in sorted_layers:
            ic = self.internal_classifiers[name]
            ic.eval()
            feats = torch.from_numpy(val_feats_by_layer[name]).to(dev)
            
            # Process in batches to avoid OOM on validation inference
            batch_u = []
            batch_p = []
            ds_val_layer = TensorDataset(feats)
            dl_val_layer = DataLoader(ds_val_layer, batch_size=256, shuffle=False)

            with torch.no_grad():
                for (bx,) in dl_val_layer:
                    logits = ic(bx) # Returns variance-scaled logits now
                    u = self._compute_ds_uncertainty(logits).cpu()
                    p = logits.argmax(dim=1).cpu()
                    batch_u.append(u)
                    batch_p.append(p)

            val_uncertainties.append(torch.cat(batch_u))
            val_preds.append(torch.cat(batch_p))
                
        # Shape: [N_val, N_layers]
        val_uncertainties = torch.stack(val_uncertainties, dim=1).numpy()
        val_preds = torch.stack(val_preds, dim=1).numpy()
        
        # Calculate Switches (Disagreement between consecutive layers)
        # switch[i] = 1 if pred[layer k] != pred[layer k-1]
        switches = np.zeros(val_preds.shape[0])
        for i in range(1, len(sorted_layers)):
            switches += (val_preds[:, i] != val_preds[:, i-1]).astype(int)

        # --- Diagnostics to catch single-class y_proxy issues ---
        logger.info(f"Number of layers: {len(sorted_layers)}")
        logger.info(f"Layer names: {sorted_layers}")
        logger.info("Switches distribution:")
        logger.info(f"  Min: {switches.min()}, Max: {switches.max()}")
        logger.info(f"  Mean: {switches.mean():.2f}, Median: {np.median(switches):.2f}")
        hist_counts, _ = np.histogram(switches, bins=10)
        logger.info(f"  Histogram: {hist_counts}")

        # Original threshold (reference)
        y_proxy_original = (switches > 1).astype(int)
        logger.info(f"Original y_proxy (>1): {np.bincount(y_proxy_original)}")

        # Adaptive thresholding to avoid degenerate labels
        threshold = np.median(switches)
        y_proxy = (switches > threshold).astype(int)
        logger.info(f"Median threshold ({threshold}): {np.bincount(y_proxy)}")

        if len(np.unique(y_proxy)) < 2:
            logger.warning("Only one class with median threshold; trying mean threshold")
            threshold = np.mean(switches)
            y_proxy = (switches > threshold).astype(int)
            logger.info(f"Mean threshold ({threshold}): {np.bincount(y_proxy)}")

        if len(np.unique(y_proxy)) < 2:
            logger.warning("Still one class; forcing 50/50 split at median")
            threshold = np.median(switches)
            y_proxy = (switches >= threshold).astype(int)
            logger.info(f"Forced 50/50 split: {np.bincount(y_proxy)}")

        # Fit Logistic Regression (cross-validated) to map [u1, u2, ...] -> y_proxy
        lr = LogisticRegressionCV(n_jobs=-1, max_iter=500)
        lr.fit(val_uncertainties, y_proxy)
        
        # Softmax normalize coefficients to get positive weights summing to 1
        coefs = torch.from_numpy(lr.coef_[0])
        self.combination_weights = F.softmax(coefs, dim=0).numpy()
        
        logger.info(f"Learned Combination Weights: {self.combination_weights}")
        self.is_fitted = True

    def predict_uncertainty(self, test_loader: DataLoader) -> np.ndarray:
        if not self.is_fitted: raise RuntimeError("Call fit() first")
        
        feats_map = self._extract_all_features(test_loader)
        layer_uncertainties = []
        
        dev = torch.device(self.device)
        
        for name in self.candidate_layers:
            ic = self.internal_classifiers[name]
            ic.eval()
            feats = torch.from_numpy(feats_map[name]).to(dev)
            
            # Batch processing for inference
            batch_u = []
            ds_layer = TensorDataset(feats)
            dl_layer = DataLoader(ds_layer, batch_size=256, shuffle=False)

            with torch.no_grad():
                for (bx,) in dl_layer:
                    logits = ic(bx)
                    u = self._compute_ds_uncertainty(logits).cpu()
                    batch_u.append(u)
            
            layer_uncertainties.append(torch.cat(batch_u))
                
        # [N_test, N_layers]
        layer_uncertainties = torch.stack(layer_uncertainties, dim=1).numpy()
        
        # Weighted Sum (Eq 6)
        weights = self.combination_weights[None, :] # [1, N_layers]
        u_final = (layer_uncertainties * weights).sum(axis=1)
        
        return u_final

    def fit_from_features(self, train_features_dict: Dict[str, np.ndarray], train_labels: np.ndarray,
                          val_features_dict: Dict[str, np.ndarray], val_labels: np.ndarray, epochs: int = 50):
        """
        Fit TULIP using pre-extracted features.
        Useful if features are computed/compressed externally.
        """
        dev = torch.device(self.device)
        
        # --- Stage 1: Train Internal Classifiers (ICs) on Training Features ---
        logger.info("Training ICs from pre-extracted features...")
        
        for layer_name, feats in train_features_dict.items():
            if layer_name not in self.candidate_layers:
                continue
                
            logger.info(f"Training IC for layer: {layer_name} (features shape: {feats.shape})")
            
            if feats.ndim == 4:
                input_size = feats.shape[2]
                output_channels = feats.shape[1]
            else:
                input_size = 1
                output_channels = feats.shape[1]

            ic = InternalClassifier(
                input_size=input_size,
                output_channels=output_channels,
                num_classes=self.num_classes,
                data_cfg=None,
                batch_size=256,
            ).to(dev)
            
            optimizer = torch.optim.Adam(ic.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            ic_feats = torch.from_numpy(feats).float()
            ic_labels = torch.from_numpy(train_labels).long()
            
            ds = TensorDataset(ic_feats, ic_labels)
            dl = DataLoader(ds, batch_size=256, shuffle=True)
            
            ic.train()
            
            # 1. Optimize Weights
            for ep in range(epochs):
                correct = 0
                total = 0
                epoch_loss = 0.0
                for b_x, b_y in dl:
                    b_x, b_y = b_x.to(dev), b_y.to(dev)
                    optimizer.zero_grad()
                    logits = ic(b_x)
                    loss = criterion(logits, b_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == b_y).sum().item()
                    total += b_y.size(0)
                if (ep + 1) % 10 == 0 or ep == 0:
                    acc = 100 * correct / total if total > 0 else 0.0
                    logger.info(f"  [Epoch {ep+1}/{epochs}] (features) Loss: {epoch_loss/len(dl):.4f} | Acc: {acc:.2f}%")
            
            self.internal_classifiers[layer_name] = ic

        # --- Stage 2: Fit Combination Head ---
        logger.info("Fitting combination weights from validation features...")
        
        val_uncertainties = []
        val_preds = []
        
        sorted_layers = self.candidate_layers
        
        for name in sorted_layers:
            if name not in self.internal_classifiers or name not in val_features_dict:
                continue
                
            ic = self.internal_classifiers[name]
            ic.eval()
            feats = torch.from_numpy(val_features_dict[name]).float().to(dev)
            
            # Batched inference
            batch_u = []
            batch_p = []
            ds = TensorDataset(feats)
            dl = DataLoader(ds, batch_size=256, shuffle=False)

            with torch.no_grad():
                for (bx,) in dl:
                    logits = ic(bx)
                    u = self._compute_ds_uncertainty(logits).cpu()
                    p = logits.argmax(dim=1).cpu()
                    batch_u.append(u)
                    batch_p.append(p)

            val_uncertainties.append(torch.cat(batch_u))
            val_preds.append(torch.cat(batch_p))
        
        if not val_uncertainties:
            raise RuntimeError("No valid layers found for combination weight learning")
                
        val_uncertainties = torch.stack(val_uncertainties, dim=1).numpy()
        val_preds = torch.stack(val_preds, dim=1).numpy()
        
        switches = np.zeros(val_preds.shape[0])
        for i in range(1, len(sorted_layers)):
            if i < val_preds.shape[1]:
                switches += (val_preds[:, i] != val_preds[:, i-1]).astype(int)

        # --- Diagnostics to catch single-class y_proxy issues ---
        logger.info(f"Number of layers: {len(sorted_layers)}")
        logger.info(f"Layer names: {sorted_layers}")
        logger.info("Switches distribution:")
        logger.info(f"  Min: {switches.min()}, Max: {switches.max()}")
        logger.info(f"  Mean: {switches.mean():.2f}, Median: {np.median(switches):.2f}")
        hist_counts, _ = np.histogram(switches, bins=10)
        logger.info(f"  Histogram: {hist_counts}")

        # Original threshold (reference)
        y_proxy_original = (switches > 1).astype(int)
        logger.info(f"Original y_proxy (>1): {np.bincount(y_proxy_original)}")

        # Adaptive thresholding to avoid degenerate labels
        threshold = np.median(switches)
        y_proxy = (switches > threshold).astype(int)
        logger.info(f"Median threshold ({threshold}): {np.bincount(y_proxy)}")

        if len(np.unique(y_proxy)) < 2:
            logger.warning("Only one class with median threshold; trying mean threshold")
            threshold = np.mean(switches)
            y_proxy = (switches > threshold).astype(int)
            logger.info(f"Mean threshold ({threshold}): {np.bincount(y_proxy)}")

        if len(np.unique(y_proxy)) < 2:
            logger.warning("Still one class; forcing 50/50 split at median")
            threshold = np.median(switches)
            y_proxy = (switches >= threshold).astype(int)
            logger.info(f"Forced 50/50 split: {np.bincount(y_proxy)}")

        lr = LogisticRegressionCV(n_jobs=-1, max_iter=500)
        lr.fit(val_uncertainties, y_proxy)
        
        coefs = torch.from_numpy(lr.coef_[0])
        self.combination_weights = F.softmax(coefs, dim=0).numpy()
        
        logger.info(f"Learned Combination Weights: {self.combination_weights}")
        self.is_fitted = True

    def predict_uncertainty_from_features(self, test_features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit_from_features() first")
        
        layer_uncertainties = []
        dev = torch.device(self.device)
        
        for name in self.candidate_layers:
            if name not in self.internal_classifiers or name not in test_features_dict:
                continue
                
            ic = self.internal_classifiers[name]
            ic.eval()
            feats = torch.from_numpy(test_features_dict[name]).float().to(dev)
            
            batch_u = []
            ds = TensorDataset(feats)
            dl = DataLoader(ds, batch_size=256, shuffle=False)

            with torch.no_grad():
                for (bx,) in dl:
                    logits = ic(bx)
                    u = self._compute_ds_uncertainty(logits).cpu()
                    batch_u.append(u)
            
            layer_uncertainties.append(torch.cat(batch_u))
        
        if not layer_uncertainties:
            raise RuntimeError("No valid layers found for uncertainty prediction")
        
        layer_uncertainties = torch.stack(layer_uncertainties, dim=1).numpy()
        
        weights = self.combination_weights[None, :]  # [1, N_layers]
        u_final = (layer_uncertainties * weights).sum(axis=1)
        
        return u_final