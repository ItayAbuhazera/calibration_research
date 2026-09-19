import numpy as np
from scipy import optimize
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional
import logging

from utils.logging_config import get_logger
logger = get_logger(__name__)

def np_softmax(x):
    """Numerically stable softmax"""
    max_val = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_val)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def get_dac_target_layers(model_name: str, model: nn.Module) -> List[str]:
    """
    Get target layers for DAC feature extraction according to DAC paper (Table 6, Appendix C.1).
    
    IMPORTANT: This function returns INTERMEDIATE layers only.
    The logits layer is extracted separately and added later in the pipeline.
    
    For DenseNet: PRE-BLOCK + BLOCK-1-3 + PENULTIMATE = 5 intermediate layers
                  Logits added later → Total 6 layers
    
    For ResNet: PRE-BLOCK + BLOCK-1-4 = 5 intermediate layers  
                Logits added later → Total 6 layers
    
    Args:
        model_name: Architecture name (e.g., 'resnet18', 'densenet121')
        model: PyTorch model instance
        
    Returns:
        List of INTERMEDIATE layer names (logits added separately)
    """
    model_name_lower = model_name.lower()
    
    if 'resnet' in model_name_lower:
        # According to DAC paper Table 6: PRE-BLOCK + BLOCK-1-4 (logits added later)
        target_layers = []
        
        # PRE-BLOCK: Try maxpool first (ImageNet), then conv1 (CIFAR)
        if find_layer_module(model, 'maxpool') is not None:
            target_layers.append('maxpool')
        elif find_layer_module(model, 'conv1') is not None:
            target_layers.append('conv1')
        
        # BLOCK-1 to BLOCK-4
        target_layers.extend(['layer1', 'layer2', 'layer3', 'layer4'])
        
        logger.info(f"DAC intermediate layers for {model_name}: {target_layers}")
        logger.info(f"  Total with logits: {len(target_layers) + 1} layers")
        return target_layers
        
    elif 'densenet' in model_name_lower:
        layers = []
        
        # Try standard torchvision structure first
        if hasattr(model, 'features'):
            features_module = model.features
            features_modules = features_module._modules
            
            # PRE-BLOCK: Programmatically find the module immediately before 'denseblock1'
            # This is typically 'pool0' for ImageNet models, but we determine it dynamically
            pre_block_name = None
            if 'denseblock1' in features_modules:
                # Get ordered list of module names
                module_names = list(features_modules.keys())
                denseblock1_idx = module_names.index('denseblock1')
                if denseblock1_idx > 0:
                    pre_block_name = module_names[denseblock1_idx - 1]
                    layers.append(f'features.{pre_block_name}')
                    logger.info(f"  Found PRE-BLOCK: features.{pre_block_name} (immediately before denseblock1)")
                else:
                    raise ValueError(f"DenseNet {model_name}: denseblock1 is the first module in features, "
                                   f"cannot determine PRE-BLOCK. Expected a module before denseblock1.")
            else:
                raise ValueError(f"DenseNet {model_name}: 'denseblock1' not found in model.features")
            
            # BLOCK-1 to BLOCK-3: Add denseblocks only (per paper Table 6)
            # Explicitly exclude transitions (transition1/2/3)
            for block_idx in [1, 2, 3]:
                block_name = f'denseblock{block_idx}'
                if block_name in features_modules:
                    layers.append(f'features.{block_name}')
                    logger.info(f"  Found BLOCK-{block_idx}: features.{block_name}")
                else:
                    raise ValueError(f"DenseNet {model_name}: Required block 'features.{block_name}' not found")
            
            # PENULTIMATE: Last module in model.features (typically 'norm5')
            module_names = list(features_modules.keys())
            if module_names:
                penultimate_name = module_names[-1]
                layers.append(f'features.{penultimate_name}')
                logger.info(f"  Found PENULTIMATE: features.{penultimate_name} (last module in features)")
            else:
                raise ValueError(f"DenseNet {model_name}: model.features has no modules")
            
            # DO NOT add transitions - paper excludes these
            # Verify we didn't accidentally include any
            for layer_name in layers:
                if 'transition' in layer_name.lower() or 'trans' in layer_name.lower():
                    raise ValueError(f"DenseNet {model_name}: Transition layer '{layer_name}' should not be included. "
                                   f"Paper specifies: PRE-BLOCK + 3 dense blocks + PENULTIMATE only.")
        
        # Custom DenseNet implementation (no .features attribute)
        else:
            logger.info(f"Using custom DenseNet structure for {model_name}")
            
            # Get ordered list of top-level modules
            module_names = list(model._modules.keys())
            
            # Find first dense block (accept variations: denseblock1, dense1, etc.)
            first_dense_idx = None
            first_dense_name = None
            for idx, name in enumerate(module_names):
                name_lower = name.lower()
                if ('dense' in name_lower and 'block' in name_lower) or \
                   (name_lower.startswith('dense') and name_lower[5:6].isdigit()):
                    # Check it's not a transition
                    if 'trans' not in name_lower:
                        first_dense_idx = idx
                        first_dense_name = name
                        break
            
            if first_dense_idx is None:
                raise ValueError(f"DenseNet {model_name}: Could not find first dense block in model structure")
            
            # PRE-BLOCK: Module immediately before first dense block
            if first_dense_idx > 0:
                pre_block_name = module_names[first_dense_idx - 1]
                layers.append(pre_block_name)
                logger.info(f"  Found PRE-BLOCK: {pre_block_name} (immediately before {first_dense_name})")
            else:
                raise ValueError(f"DenseNet {model_name}: First dense block '{first_dense_name}' is the first module, "
                               f"cannot determine PRE-BLOCK. Expected a module before it.")
            
            # BLOCK-1 to BLOCK-3: Find first 3 dense blocks (exclude transitions)
            dense_blocks_found = []
            for name in module_names:
                name_lower = name.lower()
                # Match dense blocks but exclude transitions
                if (('dense' in name_lower and 'block' in name_lower) or \
                    (name_lower.startswith('dense') and name_lower[5:6].isdigit())) and \
                   'trans' not in name_lower:
                    dense_blocks_found.append(name)
                    if len(dense_blocks_found) >= 3:
                        break
            
            if len(dense_blocks_found) < 3:
                raise ValueError(f"DenseNet {model_name}: Found only {len(dense_blocks_found)} dense blocks, "
                               f"need at least 3 for BLOCK-1..BLOCK-3")
            
            for i, block_name in enumerate(dense_blocks_found[:3], 1):
                layers.append(block_name)
                logger.info(f"  Found BLOCK-{i}: {block_name}")
            
            # PENULTIMATE: Last normalization layer before classifier, or last module before classifier
            # Look for norm5, bn2, or similar normalization layers
            penultimate_name = None
            for name in reversed(module_names):
                name_lower = name.lower()
                # Skip classifier/classifier-related modules
                if 'classifier' in name_lower or 'fc' in name_lower or 'linear' in name_lower:
                    continue
                # Prefer normalization layers
                if 'norm' in name_lower or name_lower.startswith('bn'):
                    penultimate_name = name
                    break
            
            # If no normalization found, use last non-classifier module
            if penultimate_name is None:
                for name in reversed(module_names):
                    name_lower = name.lower()
                    if 'classifier' not in name_lower and 'fc' not in name_lower and 'linear' not in name_lower:
                        penultimate_name = name
                        break
            
            if penultimate_name is None:
                raise ValueError(f"DenseNet {model_name}: Could not determine PENULTIMATE layer "
                               f"(last pre-classifier module)")
            
            layers.append(penultimate_name)
            logger.info(f"  Found PENULTIMATE: {penultimate_name}")
            
            # DO NOT add transitions (trans1, trans2, trans3)
            # Verify we didn't accidentally include any
            for layer_name in layers:
                if 'transition' in layer_name.lower() or 'trans' in layer_name.lower():
                    raise ValueError(f"DenseNet {model_name}: Transition layer '{layer_name}' should not be included. "
                                   f"Paper specifies: PRE-BLOCK + 3 dense blocks + PENULTIMATE only.")
        
        # Enforce correctness: exactly 5 intermediate layers
        if len(layers) != 5:
            raise ValueError(f"DenseNet {model_name}: Expected exactly 5 intermediate layers "
                           f"(PRE-BLOCK + BLOCK-1..BLOCK-3 + PENULTIMATE), got {len(layers)}. "
                           f"Layers found: {layers}")
        
        logger.info(f"DAC intermediate layers for {model_name}: {layers}")
        logger.info(f"  Expected: 5 intermediate + 1 logits = 6 total")
        logger.info(f"  Actual intermediate: {len(layers)} ✓")
        
        return layers
        
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}. "
                       f"Supported: ResNet, DenseNet")


def find_layer_module(model: nn.Module, layer_name: str) -> Optional[nn.Module]:
    """
    Find a module in the model by its name (supports dot notation for nested modules).
    Also handles DINOv2 block names like 'block_0', 'block_4', etc.
    
    Args:
        model: PyTorch model
        layer_name: Layer name, can use dot notation (e.g., 'features.denseblock1')
                    or DINOv2 block format (e.g., 'block_0')
        
    Returns:
        The module if found, None otherwise
    """
    # Handle DINOv2 block names (e.g., 'block_0', 'block_4')
    if layer_name.startswith('block_'):
        try:
            block_index = int(layer_name.split('_')[1])
            
            # Try different access patterns for DINOv2 models
            # Pattern 1: timm-style (model.backbone.blocks[index])
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'blocks'):
                blocks = model.backbone.blocks
                if isinstance(blocks, (nn.ModuleList, nn.Sequential)) and block_index < len(blocks):
                    return blocks[block_index]
            
            # Pattern 2: Direct blocks access (model.blocks[index])
            if hasattr(model, 'blocks'):
                blocks = model.blocks
                if isinstance(blocks, (nn.ModuleList, nn.Sequential)) and block_index < len(blocks):
                    return blocks[block_index]
            
            # Pattern 3: HuggingFace-style (model.backbone.encoder.layer[index])
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'encoder'):
                encoder = model.backbone.encoder
                if hasattr(encoder, 'layer') and block_index < len(encoder.layer):
                    return encoder.layer[block_index]
                elif hasattr(encoder, 'layers') and block_index < len(encoder.layers):
                    return encoder.layers[block_index]
            
            # Pattern 4: Direct encoder access (model.encoder.layer[index])
            if hasattr(model, 'encoder'):
                encoder = model.encoder
                if hasattr(encoder, 'layer') and block_index < len(encoder.layer):
                    return encoder.layer[block_index]
                elif hasattr(encoder, 'layers') and block_index < len(encoder.layers):
                    return encoder.layers[block_index]
            
        except (ValueError, IndexError):
            pass  # Fall through to standard handling
    
    # Standard handling for dot notation or direct attributes
    if '.' in layer_name:
        parts = layer_name.split('.')
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current
    else:
        return getattr(model, layer_name) if hasattr(model, layer_name) else None


def extract_dac_features(
    model: nn.Module,
    dataloader: DataLoader,
    layer_names: List[str],
    device: torch.device
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Extract intermediate features and logits separately for DAC.
    
    NOTE: 
    - Extracts features for INTERMEDIATE layers (not logits)
    - Extracts logits separately by running full forward pass
    - Returns: (intermediate_features_list, logits, labels)
    
    To match the paper's experimental setup, append logits to the feature list:
        features_with_logits = intermediate_features_list + [logits]
    
    This allows logits to be used both:
    1. As a feature layer for KNN density scoring (denominator)
    2. As the numerator in temperature scaling
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for the dataset
        layer_names: List of intermediate layer names (from get_dac_target_layers)
        device: Device to run inference on
        
    Returns:
        Tuple of:
        - intermediate_features_list: List of tensors, one per intermediate layer
        - logits: Final layer logits (N, C)
        - labels: Ground truth labels (N,)
    """
    model.eval()
    
    # Storage for features from each layer
    layer_features = {name: [] for name in layer_names}
    logits_list = []
    labels_list = []
    
    # Register hooks for intermediate layers
    hooks = []
    
    def make_hook(layer_name):
        def hook_fn(module, input, output):
            # Handle different output types
            if isinstance(output, torch.Tensor):
                feat = output
            elif isinstance(output, (tuple, list)):
                feat = output[0] if len(output) > 0 else None
            else:
                logger.warning(f"Unexpected output type for {layer_name}: {type(output)}")
                return
            
            if feat is not None:
                # CRITICAL: Apply global average pooling if 4D to reduce memory
                # Features should be 2D [batch, channels], not 4D [batch, channels, height, width]
                if feat.ndim == 4:
                    # Global average pooling: [B, C, H, W] -> [B, C, 1, 1]
                    pooled = F.adaptive_avg_pool2d(feat, (1, 1))
                    # Squeeze to 2D: [B, C, 1, 1] -> [B, C]
                    feat = pooled.squeeze(-1).squeeze(-1)
                elif feat.ndim == 3:
                    # For 3D tensors (e.g., [B, L, C] from transformers), average over sequence length
                    feat = feat.mean(dim=1)  # [B, L, C] -> [B, C]
                
                # Move to CPU and store
                layer_features[layer_name].append(feat.detach().cpu())
        return hook_fn
    
    # Register hooks for all intermediate layers
    for layer_name in layer_names:
        module = find_layer_module(model, layer_name)
        if module is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")
        hook = module.register_forward_hook(make_hook(layer_name))
        hooks.append(hook)
    
    try:
        from tqdm import tqdm
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting intermediate layer features"):
                if len(batch) == 2:
                    data, labels = batch
                else:
                    data = batch[0]
                    labels = torch.zeros(len(data), dtype=torch.long)
                
                data = data.to(device)
                labels_list.append(labels)
                
                # Forward pass - this will trigger hooks and also produce logits
                logits = model(data)
                logits_list.append(logits.detach().cpu())
    
    finally:
        # Remove all hooks
        for hook in hooks:
            hook.remove()
    
    # Concatenate features from each layer
    intermediate_features_list = []
    for layer_name in layer_names:
        if not layer_features[layer_name]:
            raise ValueError(f"No features extracted for layer '{layer_name}'")
        features = torch.cat(layer_features[layer_name], dim=0)
        
        # VERIFY: Features should be 2D [N, C] after pooling
        if features.ndim != 2:
            raise ValueError(f"Features from layer '{layer_name}' must be 2D [N, C], got shape {features.shape}")
        
        intermediate_features_list.append(features)
    
    # Concatenate logits and labels
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    logger.info(f"Extracted features from {len(layer_names)} layers:")
    for i, (layer_name, feat) in enumerate(zip(layer_names, intermediate_features_list)):
        logger.info(f"  ✓ Layer {i+1} ({layer_name}): {feat.shape} (verified 2D [N, C])")
    logger.info(f"Logits shape: {logits.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    
    return intermediate_features_list, logits, labels


def extract_dac_features_memory_efficient(
    model: nn.Module,
    dataloader: DataLoader,
    layer_names: List[str],
    device: torch.device,
    max_layers_per_batch: int = 4  # Process 4 layers at a time
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Extract DAC features with memory-efficient batching.
    Processes layers in groups to avoid OOM.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for the dataset
        layer_names: List of intermediate layer names (from get_dac_target_layers)
        device: Device to run inference on
        max_layers_per_batch: Maximum number of layers to process simultaneously
        
    Returns:
        Tuple of:
        - intermediate_features_list: List of tensors, one per layer
        - logits: Final layer logits (N, C)
        - labels: Ground truth labels (N,)
    """
    model.eval()
    
    # Split layers into chunks
    layer_chunks = [layer_names[i:i + max_layers_per_batch] 
                    for i in range(0, len(layer_names), max_layers_per_batch)]
    
    all_features = {name: [] for name in layer_names}
    logits_list = []
    labels_list = []
    
    logger.info(f"Processing {len(layer_names)} layers in {len(layer_chunks)} chunks of {max_layers_per_batch}")
    
    for chunk_idx, layer_chunk in enumerate(layer_chunks, 1):
        logger.info(f"Processing layer chunk {chunk_idx}/{len(layer_chunks)}: {layer_chunk}")
        
        # Extract features for this chunk
        chunk_features, chunk_logits, chunk_labels = extract_dac_features(
            model, dataloader, layer_chunk, device
        )
        
        # Store features
        for layer_name, feats in zip(layer_chunk, chunk_features):
            all_features[layer_name] = feats
        
        # Store logits and labels (only once)
        if chunk_idx == 1:
            logits_list = chunk_logits
            labels_list = chunk_labels
        
        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Reconstruct in original order
    features_list = [all_features[name] for name in layer_names]
    
    return features_list, logits_list, labels_list


def get_dac_k_value(dataset_name: str) -> int:
    """
    Get the k value for KNN based on dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'cifar10', 'cifar100', 'tiny_imagenet')
        
    Returns:
        k value: 50 for CIFAR-10, 200 for CIFAR-100 and Tiny ImageNet
    """
    dataset_lower = dataset_name.lower()
    # Important: check for CIFAR-100 *before* CIFAR-10, since the string "cifar10"
    # is a substring of "cifar100". Otherwise CIFAR-100 would incorrectly match
    # the CIFAR-10 condition and use k=50 instead of k=200.
    if 'cifar100' in dataset_lower or 'cifar-100' in dataset_lower:
        return 200
    elif 'cifar10' in dataset_lower or 'cifar-10' in dataset_lower:
        return 50
    elif 'tiny_imagenet' in dataset_lower or 'tinyimagenet' in dataset_lower:
        return 200  # Tiny ImageNet has 200 classes, similar to CIFAR-100
    else:
        logger.warning(f"Unknown dataset {dataset_name}, defaulting to k=200")
        return 200


class LayerKNNScorer:
    """
    Computes density scores (s_l) for a specific layer using k-nearest neighbors.
    Implements the preprocessing described in Section 3.3: 
    "average over spatial dimensions as well as normalize it."
    """
    def __init__(self, k=50, use_gpu=True):
        self.k = k
        self.use_gpu = False
        self.index = None
        self.faiss_res = None
        
        # Initialize GPU resources if available and requested
        if use_gpu and torch.cuda.is_available():
            try:
                if hasattr(faiss, 'StandardGpuResources'):
                    self.faiss_res = faiss.StandardGpuResources()
                    self.use_gpu = True
                else:
                    logger.info("FAISS GPU support not detected. Using CPU.")
            except Exception as e:
                logger.warning(f"FAISS GPU init failed: {e}. Using CPU.")



    def _preprocess(self, features):
        """
        1. Spatial Averaging: If features are (N, C, H, W), average to (N, C).
        2. Normalization: L2 normalize.
        """
        # Handle torch tensors
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
            
        # Global Average Pooling for 4D tensors (CNN feature maps)
        if features.ndim == 4:
            # shape: (N, C, H, W) -> mean over (2, 3) -> (N, C)
            features = np.mean(features, axis=(2, 3))
        elif features.ndim == 3:
             # shape: (N, L, C) -> mean over L -> (N, C) (e.g. Transformers)
            features = np.mean(features, axis=1)
            
        # L2 Normalization
        norms = np.linalg.norm(features, ord=2, axis=-1, keepdims=True) + 1e-12
        return np.ascontiguousarray(features / norms, dtype=np.float32)



    def fit(self, train_features):
        """Builds the KNN index for this layer."""
        features = self._preprocess(train_features)
        d = features.shape[1]
        
        # Initialize Index
        if self.use_gpu:
            index_flat = faiss.IndexFlatL2(d)
            self.index = faiss.index_cpu_to_gpu(self.faiss_res, 0, index_flat)
        else:
            self.index = faiss.IndexFlatL2(d)
            
        self.index.add(features)



    def score(self, test_features):
        """
        Calculates s_l: The distance to the k-th nearest neighbor.
        Eq (8): d_{i,l} = ||z_{i,l} - z_l||
        """
        if self.index is None:
            raise RuntimeError("Scorer not fitted.")
            
        features = self._preprocess(test_features)
        
        # Search for k nearest neighbors
        # distances shape: (N, k)
        distances, _ = self.index.search(features, self.k)
        
        # s_l is the k-th smallest element (last column of sorted distances)
        # Note: FAISS returns squared L2 distances by default, square root to get Euclidean
        knn_distances = np.sqrt(distances[:, -1])
        
        return knn_distances



class DensityAwareCalibrator:
    """
    Implements the DAC Algorithm exactly as per the paper.
    
    Model:
        S(x, w) = sum_{l=1}^L (w_l * s_l) + w_0  (Eq. 7)
        Q(x, w) = softmax(z_L / S(x, w))         (Eq. 6)
        
    Optimization:
        Minimize Squared Error Loss (Eq. 9)
        Constraints: w_l >= 0 (positive weights for layers)
    """
    def __init__(self, k=50, use_gpu=True):
        self.k = k
        self.use_gpu = use_gpu
        self.layer_scorers = []
        self.weights = None # [w_1, ..., w_L, w_0]
        self.num_layers = 0



    def fit(self, train_features_list, val_features_list, val_logits, val_labels):
        """
        Fit DAC calibrator using layer features and logits.
        
        NOTE: To match the paper's experimental setup, logits should be included as a feature layer.
        The paper uses logits both:
        1. As a feature layer for KNN density scoring (contributing w_logits * s_logits to S(x,w))
        2. As the numerator (z_L) in temperature scaling: Q(x,w) = softmax(z_L / S(x,w))
        
        Args:
            train_features_list: List of arrays/tensors, one per layer [Layer1, Layer2, ..., Logits]
                                These are used for KNN density scoring. Can include logits as the last element.
            val_features_list: List of arrays/tensors, one per layer [Layer1, Layer2, ..., Logits]
                               These are used for KNN density scoring. Can include logits as the last element.
            val_logits: Final layer logits (N, C) - these will be temperature-scaled (used as numerator)
            val_labels: Ground truth labels (N,) integer labels OR (N, C) one-hot
        """
        # Type conversions: Convert torch tensors to numpy arrays
        if isinstance(val_logits, torch.Tensor):
            val_logits = val_logits.detach().cpu().numpy()
        if isinstance(val_labels, torch.Tensor):
            val_labels = val_labels.detach().cpu().numpy()
        
        # Convert feature lists from torch tensors to numpy arrays if needed
        train_features_list = [
            f.numpy() if isinstance(f, torch.Tensor) else f 
            for f in train_features_list
        ]
        val_features_list = [
            f.numpy() if isinstance(f, torch.Tensor) else f 
            for f in val_features_list
        ]
        
        self.num_layers = len(train_features_list)
        if len(val_features_list) != self.num_layers:
            raise ValueError("Mismatch in number of layers between train and val.")



        # 1. Initialize and Fit KNN for each layer
        logger.info(f"Fitting KNN for {self.num_layers} layers...")
        self.layer_scorers = []
        val_scores_list = []
        
        for i in range(self.num_layers):
            scorer = LayerKNNScorer(k=self.k, use_gpu=self.use_gpu)
            scorer.fit(train_features_list[i])
            self.layer_scorers.append(scorer)
            
            # Pre-calculate validation scores s_l for efficiency
            s_l = scorer.score(val_features_list[i])
            val_scores_list.append(s_l)
            
        # Stack scores: Shape (N_val, L)
        val_knn_scores = np.column_stack(val_scores_list)
        
        # Prepare labels (convert to one-hot for Eq. 9)
        if val_labels.ndim == 1:
            num_classes = val_logits.shape[1]
            val_labels_onehot = np.eye(num_classes)[val_labels]
        else:
            val_labels_onehot = val_labels



        # 2. Optimization
        # Initial guess: weights=1.0, bias=0.0
        # x = [w_1, ..., w_L, w_0]
        x0 = np.ones(self.num_layers + 1)
        x0[-1] = 0.0 
        
        # Constraints: w_l >= 0 (Eq 7 commentary: "constrain weights to be positive")
        # w_0 (bias) is unconstrained
        bounds = [(0.0, None)] * self.num_layers + [(None, None)]
        
        logger.info("Optimizing DAC weights using Squared Error Loss...")
        result = optimize.minimize(
            fun=self._squared_error_loss,
            x0=x0,
            args=(val_logits, val_labels_onehot, val_knn_scores),
            method='L-BFGS-B',
            bounds=bounds,
            tol=1e-12
        )
        
        self.weights = result.x
        logger.info(f"Optimization finished. Loss: {result.fun:.6f}")
        logger.info(f"Learned Weights: {self.weights[:-1]}, Bias: {self.weights[-1]}")



    def _get_temperature(self, knn_scores_matrix):
        """
        Computes Eq (7): S(x,w) = sum(w_l * s_l) + w_0
        
        Following the original DAC paper, we do not clip the temperature to an arbitrary range.
        However, we add minimal numerical stability to prevent division by zero when computing
        scaled logits (z_L / S(x,w)).
        """
        layer_weights = self.weights[:-1]
        bias = self.weights[-1]
        
        # Linear combination
        T = np.dot(knn_scores_matrix, layer_weights) + bias
        
        # Minimal numerical stability: prevent division by zero (temperature must be > 0)
        # This is necessary for numerical stability but doesn't alter the paper's formula
        return np.maximum(T, 1e-12)



    def _squared_error_loss(self, weights, logits, labels_onehot, knn_scores):
        """
        Eq (9): L_w = sum_{c=1}^C (I_c - sigma(z/S)^(c))^2
        """
        # Current weights for this step
        current_layer_weights = weights[:-1]
        current_bias = weights[-1]
        
        # Calculate T
        T = np.dot(knn_scores, current_layer_weights) + current_bias
        T = np.clip(T, 1e-12, None) # Numerical stability
        
        # Scaled Logits: z_L / S(x,w)
        scaled_logits = logits / T[:, np.newaxis]
        
        # Probabilities
        probs = np_softmax(scaled_logits)
        
        # Squared Error: Sum over classes, Mean over samples
        # The paper implies summing over samples "accumulate L_w over all samples"
        squared_diff = (labels_onehot - probs) ** 2
        loss = np.sum(squared_diff) # Sum over all classes and samples
        
        return loss



    def get_scaled_logits(self, features_list, logits):
        """
        Returns the logits scaled by the density temperature S(x,w).
        
        Used for chaining with other calibrators (e.g., TS + DAC).
        
        Args:
            features_list: List of feature arrays/tensors, one per layer
            logits: Logits to scale (N, C)
            
        Returns:
            Scaled logits (N, C) - NOT softmaxed, just divided by temperature
        """
        if self.weights is None:
            raise ValueError("Calibrator not fitted.")
            
        # Type conversions
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        
        # Convert features
        features_list = [
            f.numpy() if isinstance(f, torch.Tensor) else f 
            for f in features_list
        ]
            
        if len(features_list) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} feature layers, got {len(features_list)}")
            
        # 1. Compute density scores
        scores_list = []
        for i, scorer in enumerate(self.layer_scorers):
            s_l = scorer.score(features_list[i])
            scores_list.append(s_l)
            
        knn_scores = np.column_stack(scores_list)
        
        # 2. Compute Temperature S(x, w)
        T = self._get_temperature(knn_scores)
        
        # 3. Scale Logits (No Softmax!)
        scaled_logits = logits / T[:, np.newaxis]
        
        return scaled_logits

    def calibrate(self, test_features_list, test_logits):
        """
        Applies the learned calibration to test data.
        Returns: Calibrated probabilities
        """
        if self.weights is None:
            raise ValueError("Calibrator not fitted.")
        
        # Type conversions: Convert torch tensors to numpy arrays
        if isinstance(test_logits, torch.Tensor):
            test_logits = test_logits.detach().cpu().numpy()
        
        # Convert feature list from torch tensors to numpy arrays if needed
        test_features_list = [
            f.numpy() if isinstance(f, torch.Tensor) else f 
            for f in test_features_list
        ]
            
        if len(test_features_list) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} feature layers, got {len(test_features_list)}")
        
        # Get scaled logits and apply softmax
        scaled_logits = self.get_scaled_logits(test_features_list, test_logits)
        calibrated_probs = np_softmax(scaled_logits)
        
        return calibrated_probs


def test_densenet121_layer_selection():
    """
    Sanity test for DenseNet121 layer selection.
    Verifies that get_dac_target_layers returns the expected layers for torchvision DenseNet121.
    """
    try:
        import torchvision.models as models
    except ImportError:
        logger.warning("torchvision not available, skipping DenseNet121 sanity test")
        return
    
    # Instantiate torchvision DenseNet121
    model = models.densenet121(pretrained=False)
    
    # Get target layers
    layers = get_dac_target_layers('densenet121', model)
    
    # Expected layers for torchvision DenseNet121
    expected_layers = ['features.pool0', 'features.denseblock1', 'features.denseblock2', 
                       'features.denseblock3', 'features.norm5']
    
    # Verify we got exactly 5 layers
    assert len(layers) == 5, f"Expected 5 layers, got {len(layers)}: {layers}"
    
    # Verify PRE-BLOCK is the module immediately before denseblock1
    features_modules = list(model.features._modules.keys())
    denseblock1_idx = features_modules.index('denseblock1')
    expected_pre_block = features_modules[denseblock1_idx - 1]
    actual_pre_block = layers[0].replace('features.', '')
    
    assert actual_pre_block == expected_pre_block, \
        f"PRE-BLOCK mismatch: expected '{expected_pre_block}' (immediately before denseblock1), " \
        f"got '{actual_pre_block}'. Full layers: {layers}"
    
    # Verify BLOCK-1..BLOCK-3
    assert layers[1] == 'features.denseblock1', f"Expected BLOCK-1 to be 'features.denseblock1', got {layers[1]}"
    assert layers[2] == 'features.denseblock2', f"Expected BLOCK-2 to be 'features.denseblock2', got {layers[2]}"
    assert layers[3] == 'features.denseblock3', f"Expected BLOCK-3 to be 'features.denseblock3', got {layers[3]}"
    
    # Verify PENULTIMATE is the last module in features
    expected_penultimate = features_modules[-1]
    actual_penultimate = layers[4].replace('features.', '')
    assert actual_penultimate == expected_penultimate, \
        f"PENULTIMATE mismatch: expected '{expected_penultimate}' (last module in features), " \
        f"got '{actual_penultimate}'. Full layers: {layers}"
    
    # Verify all layers can be found by find_layer_module
    for layer_name in layers:
        module = find_layer_module(model, layer_name)
        assert module is not None, f"Layer '{layer_name}' not found by find_layer_module()"
    
    # If expected_layers match exactly, assert that
    if layers == expected_layers:
        logger.info("✓ DenseNet121 test passed: layers match expected exactly")
    else:
        logger.info(f"✓ DenseNet121 test passed: layers are correct (PRE-BLOCK={actual_pre_block}, "
                   f"PENULTIMATE={actual_penultimate})")
    
    return True


if __name__ == "__main__":
    # Run sanity test if script is executed directly
    test_densenet121_layer_selection()
