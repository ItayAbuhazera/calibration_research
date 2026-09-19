#!/usr/bin/env python3
"""
compute_mce.py

Compute Mean Corruption Error (mCE) and aggregate all corruption results.

Auto-detects corruption types and severities from the directory.
Computes mCE for ECE, mean accuracy, and mean brier score.
Includes clean (non-corrupted) baseline for comparison.

Usage:
    python Experiments/compute_mce.py \
        --results-dir calibration_comparison_results/corruption_ablation \
        --model resnet50 \
        --dataset cifar10 \
        --training-method baseline_cross_entropy \
        --seed 11
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np


def discover_corruption_files(
    results_dir: str,
    model: str,
    dataset: str,
    training_method: str,
    seed: int,
) -> Tuple[Optional[str], Dict[str, Set[int]]]:
    """
    Discover clean file and all corruption files in the directory.
    
    Returns:
        - clean_file: path to clean results file (or None if not found)
        - corruption_map: dict mapping corruption_type -> set of severities
    """
    # Pattern for clean file
    clean_filename = f"ablation_{training_method}_{dataset}_{model}_seed{seed}.json"
    clean_path = os.path.join(results_dir, clean_filename)
    clean_file = clean_path if os.path.exists(clean_path) else None
    
    if clean_file is None:
        print(f"Warning: Clean results file not found: {clean_filename}")
    
    # Pattern for corruption files
    # Format: ablation_{training_method}_{dataset}_{model}_seed{seed}_{corruption}_sev{severity}.json
    pattern = re.compile(
        rf"ablation_{re.escape(training_method)}_{re.escape(dataset)}_{re.escape(model)}_seed{seed}_(.+)_sev(\d+)\.json"
    )
    
    corruption_map: Dict[str, Set[int]] = defaultdict(set)
    
    for filename in os.listdir(results_dir):
        match = pattern.match(filename)
        if match:
            corruption_type = match.group(1)
            severity = int(match.group(2))
            corruption_map[corruption_type].add(severity)
    
    return clean_file, dict(corruption_map)


def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None on failure."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {filepath}: {e}")
        return None


def extract_method_metrics(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Extract ECE, accuracy, and brier score for all methods from a result file.
    
    Returns:
        dict mapping method_name -> {"ece": float, "accuracy": float, "brier": float}
    """
    results = {}
    
    # Baselines
    baselines = data.get("baselines", {})
    for method_name, method_data in baselines.items():
        if isinstance(method_data, dict):
            metrics = {}
            if "ece" in method_data:
                metrics["ece"] = method_data["ece"]
            if "acc" in method_data:
                metrics["accuracy"] = method_data["acc"]
            elif "accuracy" in method_data:
                metrics["accuracy"] = method_data["accuracy"]
            if "brier" in method_data:
                metrics["brier"] = method_data["brier"]
            # Normalize accuracy: convert decimal (<1) to percentage
            if "accuracy" in metrics:
                acc = metrics["accuracy"]
                if acc < 1.0:
                    metrics["accuracy"] = acc * 100
            if metrics:
                results[method_name] = metrics
    
    # geo_comb (SGC with DAC fixed layers)
    geo_comb = data.get("geo_comb", {}).get("results", [])
    if geo_comb:
        metrics = {}
        if "ece" in geo_comb[0]:
            metrics["ece"] = geo_comb[0]["ece"]
        if "accuracy" in geo_comb[0]:
            metrics["accuracy"] = geo_comb[0]["accuracy"]
        if "brier" in geo_comb[0]:
            metrics["brier"] = geo_comb[0]["brier"]
        # Normalize accuracy: convert decimal (<1) to percentage
        if "accuracy" in metrics:
            acc = metrics["accuracy"]
            if acc < 1.0:
                metrics["accuracy"] = acc * 100
        if metrics:
            results["sgc_dac_fixed"] = metrics
    
    # tulip_comb (SGC with TULIP fixed layers)
    tulip_comb = data.get("tulip_comb", {}).get("results", [])
    if tulip_comb:
        metrics = {}
        if "ece" in tulip_comb[0]:
            metrics["ece"] = tulip_comb[0]["ece"]
        if "accuracy" in tulip_comb[0]:
            metrics["accuracy"] = tulip_comb[0]["accuracy"]
        if "brier" in tulip_comb[0]:
            metrics["brier"] = tulip_comb[0]["brier"]
        # Normalize accuracy: convert decimal (<1) to percentage
        if "accuracy" in metrics:
            acc = metrics["accuracy"]
            if acc < 1.0:
                metrics["accuracy"] = acc * 100
        if metrics:
            results["sgc_tulip_fixed"] = metrics
    
    # random_layer_ablation - get best ECE result across all target dims
    rla = data.get("random_layer_ablation", {}).get("results_by_target_dim", {})
    best_random_ece = None
    best_random_metrics = None
    for td_key, td_data in rla.items():
        for trial in td_data.get("global_random", []):
            ece = trial.get("ece")
            if ece is not None:
                if best_random_ece is None or ece < best_random_ece:
                    best_random_ece = ece
                    best_random_metrics = {
                        "ece": ece,
                        "accuracy": trial.get("accuracy"),
                        "brier": trial.get("brier"),
                    }
    if best_random_metrics:
        # Remove None values
        best_random_metrics = {k: v for k, v in best_random_metrics.items() if v is not None}
        # Normalize accuracy: convert decimal (<1) to percentage
        if "accuracy" in best_random_metrics:
            acc = best_random_metrics["accuracy"]
            if acc < 1.0:
                best_random_metrics["accuracy"] = acc * 100
        if best_random_metrics:
            results["sgc_random"] = best_random_metrics
    
    return results


def compute_mce(
    results_dir: str,
    model: str,
    dataset: str,
    training_method: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Compute Mean Corruption Error and aggregate all results.
    """
    # Discover files
    clean_file, corruption_map = discover_corruption_files(
        results_dir, model, dataset, training_method, seed
    )
    
    # Report discovered corruptions
    all_severities = set()
    for sevs in corruption_map.values():
        all_severities.update(sevs)
    all_severities = sorted(all_severities)
    corruption_types = sorted(corruption_map.keys())
    
    print(f"\nDiscovered {len(corruption_types)} corruption types: {corruption_types}")
    print(f"Discovered {len(all_severities)} severities: {all_severities}")
    
    expected_file_count = sum(len(sevs) for sevs in corruption_map.values())
    print(f"Total corruption files found: {expected_file_count}")
    
    # Validate completeness
    print("\nCompleteness check:")
    missing_count = 0
    for corruption in corruption_types:
        found_sevs = corruption_map[corruption]
        missing_sevs = set(all_severities) - found_sevs
        if missing_sevs:
            print(f"  Warning: {corruption} missing severities: {sorted(missing_sevs)}")
            missing_count += len(missing_sevs)
    
    if missing_count == 0:
        print(f"  All {len(corruption_types)} x {len(all_severities)} = {len(corruption_types) * len(all_severities)} files present")
    else:
        print(f"  Missing {missing_count} files")
    
    # Load clean results
    clean_metrics = {}
    if clean_file:
        clean_data = load_json(clean_file)
        if clean_data:
            clean_metrics = extract_method_metrics(clean_data)
            print(f"\nClean results loaded: {len(clean_metrics)} methods")
    
    # Structure: method -> corruption -> severity -> {ece, accuracy, brier}
    detailed: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    
    # Load all corruption files
    for corruption in corruption_types:
        for severity in corruption_map[corruption]:
            filename = f"ablation_{training_method}_{dataset}_{model}_seed{seed}_{corruption}_sev{severity}.json"
            filepath = os.path.join(results_dir, filename)
            
            data = load_json(filepath)
            if data is None:
                continue
            
            method_metrics = extract_method_metrics(data)
            for method, metrics in method_metrics.items():
                detailed[method][corruption][severity] = metrics
    
    # Compute aggregations
    # 1. Per-corruption (mean across severities)
    per_corruption: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    
    for method, corr_data in detailed.items():
        for corruption, sev_data in corr_data.items():
            if sev_data:
                eces = [m["ece"] for m in sev_data.values() if "ece" in m]
                accs = [m["accuracy"] for m in sev_data.values() if "accuracy" in m]
                briers = [m["brier"] for m in sev_data.values() if "brier" in m]
                
                if eces:
                    per_corruption[method][corruption]["ece"] = float(np.mean(eces))
                if accs:
                    per_corruption[method][corruption]["accuracy"] = float(np.mean(accs))
                if briers:
                    per_corruption[method][corruption]["brier"] = float(np.mean(briers))
    
    # 2. Per-severity (mean across corruption types)
    per_severity: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    
    for method, corr_data in detailed.items():
        for severity in all_severities:
            eces = []
            accs = []
            briers = []
            for corruption in corruption_types:
                if corruption in corr_data and severity in corr_data[corruption]:
                    m = corr_data[corruption][severity]
                    if "ece" in m:
                        eces.append(m["ece"])
                    if "accuracy" in m:
                        accs.append(m["accuracy"])
                    if "brier" in m:
                        briers.append(m["brier"])
            
            if eces:
                per_severity[method][severity]["ece"] = float(np.mean(eces))
            if accs:
                per_severity[method][severity]["accuracy"] = float(np.mean(accs))
            if briers:
                per_severity[method][severity]["brier"] = float(np.mean(briers))
    
    # 3. mCE (mean across all corruptions and severities)
    mce: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    for method, corr_means in per_corruption.items():
        if corr_means:
            eces = [m["ece"] for m in corr_means.values() if "ece" in m]
            accs = [m["accuracy"] for m in corr_means.values() if "accuracy" in m]
            briers = [m["brier"] for m in corr_means.values() if "brier" in m]
            
            if eces:
                mce[method]["ece"] = float(np.mean(eces))
            if accs:
                mce[method]["accuracy"] = float(np.mean(accs))
            if briers:
                mce[method]["brier"] = float(np.mean(briers))
    
    # Convert detailed to serializable format
    detailed_serializable = {}
    for method, corr_data in detailed.items():
        detailed_serializable[method] = {}
        for corruption, sev_data in corr_data.items():
            detailed_serializable[method][corruption] = {
                str(sev): metrics for sev, metrics in sev_data.items()
            }
    
    per_severity_serializable = {}
    for method, sev_data in per_severity.items():
        per_severity_serializable[method] = {
            str(sev): metrics for sev, metrics in sev_data.items()
        }
    
    # DAC-style mCE: includes clean (severity=0) in the calculation
    # Methodology: For EACH corruption type, compute the mean of (Clean + Sev 1-5).
    # Then take the mean across all corruption types.
    mce_dac_style: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    all_methods = sorted(set(list(clean_metrics.keys()) + list(mce.keys())))
    
    for method in all_methods:
        corruption_means = []
        
        for corruption in corruption_types:
            # Sequence: [Clean, Sev1, Sev2, Sev3, Sev4, Sev5]
            current_corruption_sequence = []
            
            # 1. Add Severity 0 (Clean)
            if method in clean_metrics and "ece" in clean_metrics[method]:
                current_corruption_sequence.append(clean_metrics[method]["ece"])
            
            # 2. Add Severities 1-5 from detailed results
            if method in detailed and corruption in detailed[method]:
                for sev_val in sorted(detailed[method][corruption].keys()):
                    if "ece" in detailed[method][corruption][sev_val]:
                        current_corruption_sequence.append(detailed[method][corruption][sev_val]["ece"])
            
            # Compute mean for this specific corruption type
            if current_corruption_sequence:
                corruption_means.append(np.mean(current_corruption_sequence))
        
        # Final metric is the mean of corruption means
        if corruption_means:
            mce_dac_style[method]["ece"] = float(np.mean(corruption_means))
    
    return {
        "config": {
            "model": model,
            "dataset": dataset,
            "training_method": training_method,
            "seed": seed,
            "results_dir": results_dir,
            "corruption_types": corruption_types,
            "severities": all_severities,
            "num_corruption_types": len(corruption_types),
            "num_severities": len(all_severities),
            "total_files": expected_file_count,
        },
        "clean": clean_metrics,
        "mce": dict(mce),
        "mce_dac_style": dict(mce_dac_style),
        "per_corruption": dict(per_corruption),
        "per_severity": per_severity_serializable,
        "detailed": detailed_serializable,
    }


def print_report(results: Dict[str, Any]):
    """Print a formatted report."""
    config = results["config"]
    clean = results["clean"]
    mce = results["mce"]
    mce_dac_style = results.get("mce_dac_style", {})
    per_corruption = results["per_corruption"]
    per_severity = results["per_severity"]
    
    print("\n" + "=" * 100)
    print("MEAN CORRUPTION ERROR (mCE) REPORT")
    print("=" * 100)
    print(f"Model: {config['model']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Training: {config['training_method']}")
    print(f"Seed: {config['seed']}")
    print(f"Corruptions: {config['num_corruption_types']}")
    print(f"Severities: {config['severities']}")
    print(f"Total files: {config['total_files']}")
    
    # Get all methods
    all_methods = sorted(set(list(clean.keys()) + list(mce.keys())))
    
    # Clean results
    print("\n" + "-" * 100)
    print("CLEAN (NON-CORRUPTED) RESULTS")
    print("-" * 100)
    print(f"{'Method':<35} {'ECE':<15} {'Accuracy':<15} {'Brier':<15}")
    print("-" * 100)
    
    for method in all_methods:
        if method in clean:
            m = clean[method]
            ece_str = f"{m.get('ece', 0)*100:.3f}%" if 'ece' in m else "N/A"
            acc_str = f"{m.get('accuracy', 0):.2f}%" if 'accuracy' in m else "N/A"
            brier_str = f"{m.get('brier', 0):.4f}" if 'brier' in m else "N/A"
            print(f"{method:<35} {ece_str:<15} {acc_str:<15} {brier_str:<15}")
    
    # mCE results (sorted by ECE)
    print("\n" + "-" * 100)
    print("MEAN CORRUPTION ERROR (mCE) - averaged across all corruptions and severities")
    print("-" * 100)
    print(f"{'Method':<35} {'mCE (ECE)':<15} {'Mean Acc':<15} {'Mean Brier':<15}")
    print("-" * 100)
    
    sorted_by_ece = sorted(
        [(m, mce[m]) for m in mce if "ece" in mce[m]],
        key=lambda x: x[1]["ece"]
    )
    
    for method, m in sorted_by_ece:
        ece_str = f"{m.get('ece', 0)*100:.3f}%" if 'ece' in m else "N/A"
        acc_str = f"{m.get('accuracy', 0):.2f}%" if 'accuracy' in m else "N/A"
        brier_str = f"{m.get('brier', 0):.4f}" if 'brier' in m else "N/A"
        print(f"{method:<35} {ece_str:<15} {acc_str:<15} {brier_str:<15}")
    
    # DAC-style mCE (includes clean as severity=0)
    print("\n" + "-" * 100)
    print("DAC-STYLE mCE - includes clean (severity=0) in average")
    print("-" * 100)
    print(f"{'Method':<35} {'mCE (ECE)':<15}")
    print("-" * 100)
    
    sorted_dac_style = sorted(
        [(m, mce_dac_style[m]) for m in mce_dac_style if "ece" in mce_dac_style[m]],
        key=lambda x: x[1]["ece"]
    )
    
    for method, m in sorted_dac_style:
        ece_str = f"{m.get('ece', 0)*100:.3f}%"
        print(f"{method:<35} {ece_str:<15}")
    
    # Per-severity breakdown
    print("\n" + "-" * 100)
    print("PER-SEVERITY BREAKDOWN (mean across corruption types)")
    print("-" * 100)
    
    severities = config["severities"]
    
    # ECE by severity
    print("\nECE by Severity:")
    header = f"{'Method':<35}"
    for sev in severities:
        header += f" Sev {sev:<10}"
    print(header)
    print("-" * 100)
    
    for method, m in sorted_by_ece[:10]:  # Top 10 methods
        row = f"{method:<35}"
        for sev in severities:
            sev_str = str(sev)
            if method in per_severity and sev_str in per_severity[method]:
                ece = per_severity[method][sev_str].get("ece", 0) * 100
                row += f" {ece:>10.3f}%"
            else:
                row += f" {'N/A':>11}"
        print(row)
    
    # Accuracy by severity
    print("\nAccuracy by Severity:")
    header = f"{'Method':<35}"
    for sev in severities:
        header += f" Sev {sev:<10}"
    print(header)
    print("-" * 100)
    
    for method, m in sorted_by_ece[:10]:
        row = f"{method:<35}"
        for sev in severities:
            sev_str = str(sev)
            if method in per_severity and sev_str in per_severity[method]:
                acc = per_severity[method][sev_str].get("accuracy", 0)
                row += f" {acc:>10.2f}%"
            else:
                row += f" {'N/A':>11}"
        print(row)
    
    # Degradation analysis (clean vs corrupted)
    print("\n" + "-" * 100)
    print("DEGRADATION ANALYSIS (Clean -> Corrupted)")
    print("-" * 100)
    print(f"{'Method':<35} {'Clean ECE':<12} {'mCE':<12} {'ECE Increase':<15} {'Clean Acc':<12} {'mAcc':<12} {'Acc Drop':<12}")
    print("-" * 100)
    
    for method, m in sorted_by_ece[:10]:
        if method in clean and "ece" in clean[method]:
            clean_ece = clean[method]["ece"] * 100
            corrupt_ece = m.get("ece", 0) * 100
            ece_increase = corrupt_ece - clean_ece
            
            clean_acc = clean[method].get("accuracy", 0)
            corrupt_acc = m.get("accuracy", 0)
            acc_drop = clean_acc - corrupt_acc
            
            print(f"{method:<35} {clean_ece:>10.3f}% {corrupt_ece:>10.3f}% {ece_increase:>+12.3f}% {clean_acc:>10.2f}% {corrupt_acc:>10.2f}% {acc_drop:>+10.2f}%")
    
    # Best method summary
    if sorted_by_ece:
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        best_method, best_metrics = sorted_by_ece[0]
        print(f"Best method by mCE: {best_method}")
        print(f"  mCE (ECE): {best_metrics.get('ece', 0)*100:.3f}%")
        print(f"  Mean Accuracy: {best_metrics.get('accuracy', 0):.2f}%")
        
        # Compare SGC methods to baselines
        sgc_methods = ["sgc_random", "sgc_dac_fixed", "sgc_tulip_fixed"]
        baseline_methods = ["uncalibrated", "temperature_scaling", "density_aware_calibration"]
        
        print("\nSGC vs Baselines (mCE ECE):")
        for sgc in sgc_methods:
            if sgc in mce and "ece" in mce[sgc]:
                sgc_ece = mce[sgc]["ece"]
                for baseline in baseline_methods:
                    if baseline in mce and "ece" in mce[baseline]:
                        baseline_ece = mce[baseline]["ece"]
                        improvement = (baseline_ece - sgc_ece) / baseline_ece * 100
                        print(f"  {sgc} vs {baseline}: {improvement:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Compute Mean Corruption Error")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing corruption ablation results")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--training-method", type=str, required=True)
    parser.add_argument("--seed", type=int, default=11)
    
    args = parser.parse_args()
    
    results = compute_mce(
        results_dir=args.results_dir,
        model=args.model,
        dataset=args.dataset,
        training_method=args.training_method,
        seed=args.seed,
    )
    
    print_report(results)
    
    # Save results
    output_filename = f"mce_{args.model}_{args.dataset}_seed{args.seed}.json"
    output_path = os.path.join(args.results_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()