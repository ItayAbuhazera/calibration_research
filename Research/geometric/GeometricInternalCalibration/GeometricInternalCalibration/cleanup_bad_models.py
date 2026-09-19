#!/usr/bin/env python3
"""
Find and delete model checkpoints with accuracy below threshold.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple


def find_results_files(base_dir: str) -> List[Path]:
    """Find all results.json files in the directory tree."""
    results_files = []
    base_path = Path(base_dir)
    
    for results_file in base_path.rglob("results.json"):
        results_files.append(results_file)
    
    return results_files


def check_accuracy(results_file: Path, threshold: float = 0.4) -> Tuple[bool, float]:
    """
    Check if model accuracy is below threshold.
    
    Returns:
        (should_delete, accuracy)
    """
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        eval_results = data.get("evaluation_results", {})
        
        # PRIORITY FIX: Calculate from raw counts if available
        correct = eval_results.get("correct_samples")
        total = eval_results.get("total_samples")
        
        if correct is not None and total is not None and total > 0:
            accuracy = float(correct) / float(total)
        else:
            # Fallback to the explicit accuracy field
            accuracy = eval_results.get("accuracy", None)
            
            if accuracy is None:
                print(f"⚠️  No accuracy found in {results_file}")
                return False, -1.0
            
            # Normalize accuracy (handle both 0-1 and 0-100 scales)
            # Warning: This still fails for percentages < 1% if they are stored as e.g. 0.6
            if accuracy > 1.0:
                accuracy = accuracy / 100.0
        
        should_delete = accuracy < threshold
        return should_delete, accuracy
        
    except Exception as e:
        print(f"❌ Error reading {results_file}: {e}")
        return False, -1.0


def delete_seed_directory(results_file: Path, dry_run: bool = True) -> bool:
    """
    Delete the seed directory (2 levels up from results.json).
    
    Path structure:
    .../seed{X}/{method}_{dataset}_{model}_seed{X}/results.json
    We want to delete: .../seed{X}/
    """
    # Go up 2 levels: results.json -> {method}_{dataset}_{model}_seed{X} -> seed{X}
    seed_dir = results_file.parent.parent
    
    if not seed_dir.name.startswith("seed"):
        print(f"⚠️  Unexpected directory structure: {seed_dir}")
        return False
    
    if dry_run:
        print(f"[DRY RUN] Would delete: {seed_dir}")
        return True
    else:
        try:
            shutil.rmtree(seed_dir)
            print(f"✅ Deleted: {seed_dir}")
            return True
        except Exception as e:
            print(f"❌ Failed to delete {seed_dir}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Find and delete model checkpoints with low accuracy"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/ptamar/geometric-internal-calibration/aaai_full_experiments/results/baseline",
        help="Base directory to search for results.json files"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Accuracy threshold (models below this will be deleted). Default: 0.4 (40%%)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Don't actually delete, just show what would be deleted (default: True)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform deletion (overrides --dry-run)"
    )
    
    args = parser.parse_args()
    
    # If --execute is specified, turn off dry_run
    if args.execute:
        args.dry_run = False
        print("\n⚠️  EXECUTE MODE: Will actually delete directories!\n")
    else:
        print("\n🔍 DRY RUN MODE: No files will be deleted\n")
    
    print(f"Searching for results.json files in: {args.base_dir}")
    print(f"Accuracy threshold: {args.threshold * 100:.1f}%\n")
    
    # Find all results files
    results_files = find_results_files(args.base_dir)
    print(f"Found {len(results_files)} results.json files\n")
    
    # Check each file
    to_delete = []
    for results_file in results_files:
        should_delete, accuracy = check_accuracy(results_file, args.threshold)
        
        if should_delete:
            acc_pct = accuracy * 100 if accuracy >= 0 else -1
            print(f"❌ Low accuracy ({acc_pct:.1f}%): {results_file}")
            to_delete.append(results_file)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total models checked: {len(results_files)}")
    print(f"Models below {args.threshold * 100:.1f}% accuracy: {len(to_delete)}")
    
    if to_delete:
        print(f"\n{'='*80}")
        print(f"DELETION PLAN")
        print(f"{'='*80}")
        
        deleted_count = 0
        for results_file in to_delete:
            success = delete_seed_directory(results_file, dry_run=args.dry_run)
            if success:
                deleted_count += 1
        
        print(f"\n{'='*80}")
        if args.dry_run:
            print(f"DRY RUN COMPLETE")
            print(f"Would delete {deleted_count} directories")
            print(f"\nTo actually delete, run with --execute flag")
        else:
            print(f"DELETION COMPLETE")
            print(f"Successfully deleted {deleted_count} directories")
    else:
        print(f"\n✅ No models below threshold found!")


if __name__ == "__main__":
    main()