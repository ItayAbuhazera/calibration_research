# Save as: Experiments/patch_compression_baselines.py

"""
Patch existing compression experiment results with baseline ECE values.
Adds relative_ece_change to results that are missing it.
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import defaultdict
import argparse


def find_baseline_file(
    base_dir: Path,
    layer_type: str,
    baseline_dim: int
) -> Optional[Path]:
    """
    Find baseline file with multiple fallback strategies.
    """
    # Strategy 1: ratio1 with exact dimension
    ratio1_exact = base_dir / f"{layer_type}_ratio1_dim{baseline_dim}_results.json"
    if ratio1_exact.exists():
        return ratio1_exact
    
    # Strategy 2: legacy naming with exact dimension
    legacy_exact = base_dir / f"{layer_type}_dim{baseline_dim}_results.json"
    if legacy_exact.exists():
        return legacy_exact
    
    # Strategy 3: any ratio1 file (for semantic layers where dim varies)
    ratio1_pattern = f"{layer_type}_ratio1_dim*_results.json"
    candidates = list(base_dir.glob(ratio1_pattern))
    if candidates:
        # Pick the one with largest dimension (most uncompressed)
        return max(candidates, key=lambda p: int(p.stem.split('_dim')[1].split('_')[0]))
    
    # Strategy 4: legacy pattern wildcard
    legacy_pattern = f"{layer_type}_dim*_results.json"
    candidates = list(base_dir.glob(legacy_pattern))
    for candidate in candidates:
        try:
            with open(candidate) as f:
                data = json.load(f)
            ratio = data.get('compression_ratio') or data.get('requested_compression_ratio')
            if ratio and abs(float(ratio) - 1.0) <= 0.05:
                return candidate
        except:
            continue
    
    # Strategy 5: fall back to lowest available compression ratio
    lowest_ratio_baseline = find_lowest_ratio_baseline(base_dir, layer_type)
    if lowest_ratio_baseline:
        return lowest_ratio_baseline

    return None


def load_baseline_ece(baseline_path: Path) -> Optional[float]:
    """Load ECE from baseline file."""
    try:
        with open(baseline_path) as f:
            data = json.load(f)
        ece = data.get('ece')
        if ece is not None:
            return float(ece)
    except Exception as e:
        print(f"  Warning: Failed to load {baseline_path}: {e}")
    return None


def find_lowest_ratio_baseline(base_dir: Path, layer_type: str) -> Optional[Path]:
    """Find result file with lowest compression ratio for this layer type."""
    pattern = f"{layer_type}_ratio*_dim*_results.json"
    candidates = list(base_dir.glob(pattern))

    best_ratio = float('inf')
    best_path = None

    for candidate in candidates:
        try:
            with open(candidate) as f:
                data = json.load(f)
            ratio = data.get('compression_ratio') or data.get('requested_compression_ratio')
            if ratio and float(ratio) < best_ratio:
                best_ratio = float(ratio)
                best_path = candidate
        except:
            continue

    return best_path


def compute_baseline_dim(layer_type: str, original_dim: Optional[int]) -> Optional[int]:
    """
    Determine the correct baseline dimension for a layer type.
    
    Physical layers: Use original_dim (uncompressed)
    Semantic layers: Use fixed reference (8192)
    """
    if layer_type == 'data_layer':
        return original_dim
    else:
        # Semantic layers use reference baseline
        return 8192


def patch_result_file(
    result_path: Path,
    baseline_ece: float,
    dry_run: bool = False
) -> Tuple[bool, str]:
    """
    Patch a single result file with baseline ECE and relative change.
    
    Returns:
        (success, message)
    """
    try:
        with open(result_path) as f:
            data = json.load(f)
        
        current_ece = data.get('ece')
        if current_ece is None:
            return False, "No ECE value in result"
        
        # Check if already has baseline
        if data.get('baseline_ece') is not None and data.get('relative_ece_change') is not None:
            return False, "Already has baseline (skipped)"
        
        # Compute relative change
        relative_change = (float(current_ece) - baseline_ece) / baseline_ece * 100
        
        # Update data
        data['baseline_ece'] = float(baseline_ece)
        data['relative_ece_change'] = float(relative_change)
        
        if not dry_run:
            # Write back
            with open(result_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        return True, f"Patched: ΔECE = {relative_change:+.2f}%"
    
    except Exception as e:
        return False, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Patch compression results with missing baselines"
    )
    parser.add_argument(
        '--results_dir',
        type=Path,
        default=Path('aaai_full_experiments/results/compression_experiments_ratio'),
        help='Directory containing compression results'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be patched without actually modifying files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed information for each file'
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1
    
    print("="*80)
    print("COMPRESSION RESULTS BASELINE PATCHER")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE PATCHING'}")
    print()
    
    # Find all result files
    all_files = list(args.results_dir.rglob("*_results.json"))
    print(f"Found {len(all_files)} result files")
    
    # Group by configuration
    stats = {
        'total': len(all_files),
        'patched': 0,
        'skipped': 0,
        'failed': 0,
        'baseline_missing': 0
    }
    
    # Process each file
    for result_path in sorted(all_files):
        # Parse path components
        parts = result_path.relative_to(args.results_dir).parts
        if len(parts) < 5:
            continue
        
        training_loss = parts[0]
        dataset = parts[1]
        model = parts[2]
        seed_dir = parts[3]  # e.g., "seed15"
        
        # Load current result
        try:
            with open(result_path) as f:
                data = json.load(f)
        except:
            stats['failed'] += 1
            continue
        
        layer_type = data.get('layer_type')
        original_dim = data.get('original_dim')
        current_ratio = data.get('compression_ratio') or data.get('requested_compression_ratio')
        
        # Skip if this IS the baseline
        if current_ratio and abs(float(current_ratio) - 1.0) <= 0.05:
            if args.verbose:
                print(f"  [BASELINE] {result_path.name}")
            stats['skipped'] += 1
            continue
        
        # Skip if already has baseline
        if data.get('baseline_ece') is not None:
            if args.verbose:
                print(f"  [HAS BASELINE] {result_path.name}")
            stats['skipped'] += 1
            continue
        
        # Determine baseline dimension
        baseline_dim = compute_baseline_dim(layer_type, original_dim)
        if baseline_dim is None:
            if args.verbose:
                print(f"  [ERROR] Cannot determine baseline dim for {result_path.name}")
            stats['failed'] += 1
            continue
        
        # Find baseline file
        base_dir = result_path.parent
        baseline_path = find_baseline_file(base_dir, layer_type, baseline_dim)
        
        if baseline_path is None:
            if args.verbose or True:  # Always show missing baselines
                print(f"  [MISSING BASELINE] {result_path.name}")
                print(f"    Looked for: {layer_type}_ratio1_dim{baseline_dim}_results.json")
            stats['baseline_missing'] += 1
            continue
        
        # Load baseline ECE
        baseline_ece = load_baseline_ece(baseline_path)
        if baseline_ece is None:
            if args.verbose:
                print(f"  [BASELINE READ ERROR] {result_path.name}")
            stats['failed'] += 1
            continue
        
        # Patch the file
        success, message = patch_result_file(result_path, baseline_ece, args.dry_run)
        
        if success:
            stats['patched'] += 1
            if args.verbose or not args.dry_run:
                print(f"  [PATCHED] {result_path.name}")
                print(f"    {message}")
        else:
            if "Already has baseline" in message:
                stats['skipped'] += 1
            else:
                stats['failed'] += 1
            if args.verbose:
                print(f"  [FAILED] {result_path.name}: {message}")
    
    # Print summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files:          {stats['total']}")
    print(f"✓ Patched:            {stats['patched']}")
    print(f"− Skipped (OK):       {stats['skipped']}")
    print(f"✗ Failed:             {stats['failed']}")
    print(f"⚠ Missing baseline:   {stats['baseline_missing']}")
    
    if args.dry_run:
        print()
        print("This was a DRY RUN. Run without --dry_run to apply changes.")
    
    return 0


if __name__ == "__main__":
    exit(main())