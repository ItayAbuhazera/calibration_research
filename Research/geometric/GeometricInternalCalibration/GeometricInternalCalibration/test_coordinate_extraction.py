#!/usr/bin/env python3
"""
Quick sanity test for coordinate extraction functions.
Run this before submitting SLURM jobs to verify everything works.
"""

import torch
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coordinate_extraction import (
    discover_coordinate_space,
    plan_coordinate_extraction,
    extract_coordinate_features,
)
from torchvision.models import resnet18
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("COORDINATE EXTRACTION SANITY TEST")
print("=" * 60)

# Quick sanity test
print("\n1. Creating model...")
model = resnet18(pretrained=False)
model.eval()

# 1. Discover
print("\n2. Discovering coordinate space...")
layer_map, total_size = discover_coordinate_space(
    model, input_shape=(1, 3, 32, 32), device='cpu'
)
print(f"   ✓ Total coordinates: {total_size:,}")
print(f"   ✓ Layers discovered: {len(layer_map)}")
print(f"   ✓ Sample layers:")
for i, layer in enumerate(layer_map[:3]):
    print(f"      - {layer['name']}: size={layer['size']:,}, shape={layer['shape']}")

# 2. Plan
print("\n3. Planning coordinate extraction (K=256)...")
sampling_plan, global_order = plan_coordinate_extraction(
    total_size=total_size,
    num_coordinates=256,
    layer_map=layer_map,
    seed=42
)
print(f"   ✓ Layers with coordinates: {len(sampling_plan)}")
print(f"   ✓ Global order shape: {global_order.shape}")
print(f"   ✓ Sample layers in plan:")
for i, (layer_name, indices) in enumerate(list(sampling_plan.items())[:3]):
    print(f"      - {layer_name}: {len(indices)} coordinates")

# 3. Extract (small batch)
print("\n4. Extracting features from dummy data...")
dummy_data = torch.randn(10, 3, 32, 32)
dummy_labels = torch.zeros(10, dtype=torch.long)
loader = DataLoader(TensorDataset(dummy_data, dummy_labels), batch_size=5, num_workers=0)

features = extract_coordinate_features(
    model, loader, sampling_plan, layer_map, global_order, device='cpu'
)
print(f"   ✓ Features shape: {features.shape}")  # Should be (10, 256)

# Verify no NaNs
if torch.isnan(features).any():
    print("   ✗ ERROR: NaNs found in features!")
    sys.exit(1)
else:
    print("   ✓ No NaNs in features")

# Verify shape is correct
if features.shape != (10, 256):
    print(f"   ✗ ERROR: Expected shape (10, 256), got {features.shape}")
    sys.exit(1)
else:
    print("   ✓ Feature shape is correct")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED! ✓")
print("=" * 60)
print("\nCoordinate extraction is working correctly.")
print("You can now run the full experiment.")



