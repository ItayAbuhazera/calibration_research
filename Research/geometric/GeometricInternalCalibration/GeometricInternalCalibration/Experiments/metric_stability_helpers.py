#!/usr/bin/env python3
"""
Metric Stability Helpers

Analyzes stability of metric-based layer selection across multiple runs/seeds.
"""

import json
import os
from typing import Dict, List, Any
import numpy as np


def summarize_metric_stability_across_runs(json_paths: List[str], topk: int = 5) -> Dict[str, Any]:
    """
    Given a list of paths to metric_docs_and_topk.json (e.g., different seeds),
    returns:
      - top1_repeat_fraction: fraction of runs that pick the same top-1 layer
      - mean_jaccard_topK: mean Jaccard of Top-K sets across all run pairs
      - depth_bias_var: variance of normalized depth percentile across runs for each metric
    
    Args:
        json_paths: List of paths to metric_docs_and_topk.json files
        topk: Number of top layers to analyze (default: 5)
    
    Returns:
        Dictionary with stability metrics for each selector metric
    """
    docs = []
    for p in json_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r") as f:
                    docs.append(json.load(f))
            except Exception:
                pass

    if not docs:
        return {"error": "No valid docs"}

    # collect common metric names
    common_metrics = set(docs[0].get("metrics_topk", {}).keys())
    for d in docs[1:]:
        common_metrics &= set(d.get("metrics_topk", {}).keys())
    common_metrics = sorted(list(common_metrics))

    out = {}
    for m in common_metrics:
        top1_layers = []
        topK_sets   = []
        depth_ps    = []
        for doc in docs:
            entry = doc["metrics_topk"][m]
            top1_layers.append(int(entry["top1_layer"]))
            Kset = set(map(int, entry.get("topk_layers", [])[:topk]))
            topK_sets.append(Kset)

            # depth percentile of Top-K
            candidates = list(map(int, doc.get("candidate_layers", [])))
            if candidates:
                idx_map = {L: i for i, L in enumerate(sorted(candidates))}
                ranks = [idx_map.get(int(L), 0) for L in Kset] or [0]
                mean_rank = float(np.mean(ranks))
                depth_ps.append(mean_rank / max(1, len(candidates) - 1))

        # top-1 repeat fraction
        vals, counts = np.unique(top1_layers, return_counts=True)
        repeat_frac = float(np.max(counts) / max(1, len(top1_layers)))

        # mean pairwise Jaccard among runs
        jac = []
        for i in range(len(topK_sets)):
            for j in range(i + 1, len(topK_sets)):
                A, B = topK_sets[i], topK_sets[j]
                u = len(A | B) or 1
                jac.append(len(A & B) / u)
        mean_j = float(np.mean(jac)) if jac else None

        out[m] = {
            "top1_repeat_fraction": repeat_frac,
            "mean_jaccard_topK": mean_j,
            "depth_bias_variance": float(np.var(depth_ps)) if depth_ps else None,
        }

    return out


def compare_two_runs(path1: str, path2: str, topk: int = 5) -> Dict[str, Any]:
    """
    Compare metric selections between two runs (e.g., different seeds).
    
    Returns per-metric:
      - top1_match: bool (whether Top-1 layer is the same)
      - topK_jaccard: Jaccard similarity of Top-K sets
      - rank_correlation: Spearman correlation of full metric value vectors
    """
    try:
        with open(path1, "r") as f:
            doc1 = json.load(f)
        with open(path2, "r") as f:
            doc2 = json.load(f)
    except Exception as e:
        return {"error": str(e)}
    
    metrics1 = set(doc1.get("metrics_topk", {}).keys())
    metrics2 = set(doc2.get("metrics_topk", {}).keys())
    common = sorted(list(metrics1 & metrics2))
    
    comparisons = {}
    for m in common:
        e1 = doc1["metrics_topk"][m]
        e2 = doc2["metrics_topk"][m]
        
        # Top-1 match
        top1_match = (int(e1["top1_layer"]) == int(e2["top1_layer"]))
        
        # Top-K Jaccard
        set1 = set(map(int, e1.get("topk_layers", [])[:topk]))
        set2 = set(map(int, e2.get("topk_layers", [])[:topk]))
        union = len(set1 | set2) or 1
        jaccard = len(set1 & set2) / union
        
        # Rank correlation of full metric values
        all_vals1 = doc1.get("all_metric_values", {}).get(m, {})
        all_vals2 = doc2.get("all_metric_values", {}).get(m, {})
        candidates = sorted(set(all_vals1.keys()) & set(all_vals2.keys()))
        
        if len(candidates) >= 3:
            v1 = np.array([float(all_vals1[k]) for k in candidates])
            v2 = np.array([float(all_vals2[k]) for k in candidates])
            r1 = np.argsort(np.argsort(v1)).astype(float)
            r2 = np.argsort(np.argsort(v2)).astype(float)
            if r1.std() > 0 and r2.std() > 0:
                rank_corr = float(np.corrcoef(r1, r2)[0, 1])
            else:
                rank_corr = None
        else:
            rank_corr = None
        
        comparisons[m] = {
            "top1_match": bool(top1_match),
            "topK_jaccard": float(jaccard),
            "rank_correlation": rank_corr,
        }
    
    return comparisons


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 3:
        print("Usage: python metric_stability_helpers.py <path1> <path2> ... [--topk K]")
        print("  Analyzes stability across multiple metric_docs_and_topk.json files")
        sys.exit(1)
    
    paths = []
    topk_val = 5
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--topk":
            topk_val = int(sys.argv[i+1])
            i += 2
        else:
            paths.append(sys.argv[i])
            i += 1
    
    if len(paths) >= 2:
        print("="*80)
        print("METRIC STABILITY ANALYSIS")
        print("="*80)
        print(f"Analyzing {len(paths)} runs with Top-{topk_val}")
        print()
        
        stability = summarize_metric_stability_across_runs(paths, topk=topk_val)
        
        print("Stability Summary:")
        print(f"{'Metric':<45} {'Top1 Repeat':<12} {'Mean Jaccard':<13} {'Depth Var':<10}")
        print("-"*80)
        for m, stats in sorted(stability.items()):
            if not isinstance(stats, dict):
                continue
            repeat = stats.get('top1_repeat_fraction', 0.0)
            jaccard = stats.get('mean_jaccard_topK')
            var = stats.get('depth_bias_variance')
            
            repeat_str = f"{repeat:.3f}" if repeat is not None else "N/A"
            jaccard_str = f"{jaccard:.3f}" if jaccard is not None else "N/A"
            var_str = f"{var:.4f}" if var is not None else "N/A"
            
            print(f"{m:<45} {repeat_str:<12} {jaccard_str:<13} {var_str:<10}")
        
        # Save to JSON
        out_json = "metric_stability_summary.json"
        with open(out_json, "w") as f:
            json.dump(stability, f, indent=2)
        print()
        print(f"✅ Saved to {out_json}")
