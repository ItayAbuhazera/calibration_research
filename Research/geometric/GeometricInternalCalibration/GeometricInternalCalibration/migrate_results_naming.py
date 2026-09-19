#!/usr/bin/env python3
"""
Comprehensive recovery script for DAC comparison experiments.
Extracts complete results from log files including timing, memory, and all metrics.
"""

import json
import re
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# Directories
JSON_DIR = Path("calibration_comparison_results_full")
LOG_DIR = Path("slurm_jobs/_dac_comparison_logs")
BACKUP_DIR = Path("calibration_comparison_results_full_backups")

# Required top-level keys for a complete JSON
REQUIRED_KEYS = [
    'experiment_config',
    'uncalibrated',
    'baseline_ts',
    'dac_standalone',
    'ts_plus_dac',
    'layer_quality_metrics',
    'metric_based_selections',
    'metric_guided_calibration',
    'geometric_original',
    'geometric_concatenated',
    'geometric_dac_weighted',
    'geometric_precomputed',
    'ablation',
    'analysis'
]


def is_json_corrupted(json_path: Path) -> Tuple[bool, str, Optional[Dict]]:
    """
    Check if JSON file is corrupted or incomplete.
    
    Returns:
        (is_corrupted, reason, partial_data)
    """
    try:
        with open(json_path, 'r') as f:
            content = f.read()
        
        # Try to load what we can
        try:
            data = json.loads(content)
            partial_data = data
        except json.JSONDecodeError:
            # Try to salvage what we can by finding the last valid closing brace
            # Find last complete section
            last_complete = content.rfind('},')
            if last_complete > 0:
                try:
                    truncated = content[:last_complete + 1] + '\n}'
                    partial_data = json.loads(truncated)
                except:
                    partial_data = None
            else:
                partial_data = None
        
        # Check for missing keys
        if partial_data:
            missing_keys = [key for key in REQUIRED_KEYS if key not in partial_data]
            if missing_keys:
                return True, f"Missing keys: {', '.join(missing_keys)}", partial_data
        
        # Check for incomplete val_calibrated_probs
        if '"val_calibrated_probs":' in content and '"val_calibrated_probs": [' not in content:
            return True, "Incomplete metric_guided_calibration (val_calibrated_probs cutoff)", partial_data
        
        return False, "Complete", partial_data
        
    except Exception as e:
        return True, f"Error reading file: {e}", None


def find_matching_log(json_path: Path, log_dir: Path) -> Optional[Path]:
    """Find matching log file for a JSON file."""
    json_name = json_path.stem
    pattern = r'ablation_(.+)_(.+)_(.+)_seed(\d+)'
    match = re.match(pattern, json_name)
    
    if not match:
        return None
    
    training_method, dataset, model, seed = match.groups()
    log_pattern = f"dac_cmp_{model}_{dataset}_{training_method}_s{seed}_*.out"
    
    matching_logs = list(log_dir.glob(log_pattern))
    
    if not matching_logs:
        return None
    
    if len(matching_logs) > 1:
        matching_logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return matching_logs[0]


def extract_calibration_result(section: str) -> Dict[str, Any]:
    """
    Extract calibration results from a log section.
    
    Expects section to contain:
    - Calibration time: X.XXs
    - Throughput: X.XX samples/sec
    - Peak GPU memory: X.XX MB
    - ECE: X.XXXXXX
    - Accuracy: XX.XXXX
    """
    result = {}
    
    # Extract fit time (appears before calibration time)
    fit_time = re.search(r'Geometric fitting time:\s*(\d+\.\d+)s', section)
    if fit_time:
        result['fit_time_s'] = float(fit_time.group(1))
    
    # Extract calibration metrics
    calibrate_time = re.search(r'Calibration time:\s*(\d+\.\d+)s', section)
    if calibrate_time:
        result['calibrate_time_s'] = float(calibrate_time.group(1))
    
    throughput = re.search(r'Throughput:\s*(\d+\.\d+)\s*samples/sec', section)
    if throughput:
        result['throughput_samples_per_sec'] = float(throughput.group(1))
    
    memory = re.search(r'Peak GPU memory:\s*(\d+\.\d+)\s*MB', section)
    if memory:
        result['peak_memory_mb'] = float(memory.group(1))
    
    ece = re.search(r'ECE:\s*(\d+\.\d+)', section)
    if ece:
        result['ece'] = float(ece.group(1))
    
    accuracy = re.search(r'Accuracy:\s*(\d+\.\d+)', section)
    if accuracy:
        result['accuracy'] = float(accuracy.group(1))
    
    return result


def parse_single_layer_calibrations(log_content: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse individual layer calibration results.
    
    Format:
    Calibrating layer X/Y: {layer_name} ({semantic_name})
    ...
    ECE: X.XXXXXX
    Accuracy: XX.XXXX
    """
    results = {}
    
    # Find all single layer calibration sections
    pattern = r'Calibrating layer \d+/\d+:\s*(\S+)\s*\((.+?)\)(.*?)(?=Calibrating layer \d+/\d+:|✓ All single layer calibrations completed)'
    
    matches = re.findall(pattern, log_content, re.DOTALL)
    
    for layer_name, semantic_name, section in matches:
        result = extract_calibration_result(section)
        if result:
            result['method'] = f'Geometric ({semantic_name})'
            result['layer_names'] = layer_name
            result['extract_time_s'] = 0.0  # Precomputed
            results[layer_name] = result
    
    return results


def parse_combined_layers(log_content: str) -> Optional[Dict[str, Any]]:
    """Parse combined layers experiment results."""
    # Find section
    section_match = re.search(
        r'MEMORY-INTENSIVE EXPERIMENT: COMBINED LAYERS.*?'
        r'RUNNING GEOMETRIC CALIBRATION: Geometric \(Combined (\d+) Main Blocks\)(.*?)'
        r'(?=✓ Combined layers experiment completed|RUNNING DAC-STYLE WEIGHTED ENSEMBLE)',
        log_content,
        re.DOTALL
    )
    
    if not section_match:
        return None
    
    num_blocks = int(section_match.group(1))
    section = section_match.group(2)
    
    # Extract layer names
    layers_match = re.search(r"Using main blocks:\s*\[(.+?)\]", log_content)
    layer_names = []
    if layers_match:
        # Parse list: ['dense1', 'dense2', 'dense3', 'dense4']
        layers_str = layers_match.group(1)
        layer_names = [l.strip().strip("'\"") for l in layers_str.split(',')]
    
    result = extract_calibration_result(section)
    if result:
        result['method'] = f'Geometric (Combined {num_blocks} Main Blocks)'
        result['layer_names'] = layer_names
        result['extract_time_s'] = 0.0
    
    return result


def parse_dac_weighted(log_content: str) -> Optional[Dict[str, Any]]:
    """Parse DAC-style weighted ensemble results."""
    section_match = re.search(
        r'GEOMETRIC WITH DAC-STYLE LAYER WEIGHTING(.*?)'
        r'(?=✓ DAC-style weighted experiment completed|EXTRACTING SINGLE LAYER)',
        log_content,
        re.DOTALL
    )
    
    if not section_match:
        return None
    
    section = section_match.group(1)
    
    result = extract_calibration_result(section)
    if result:
        result['method'] = 'Geometric + DAC Weighting'
        result['extract_time_s'] = 0.0
        result['compute_separation_time_s'] = 0.0
        
        # Extract learned weights
        weights_match = re.search(r'Learned Weights:\s*\[(.+?)\]', section)
        if weights_match:
            weights_str = weights_match.group(1)
            result['layer_weights'] = [float(w.strip()) for w in weights_str.split()]
        
        bias_match = re.search(r'Learned Bias:\s*(-?\d+\.\d+)', section)
        if bias_match:
            result['bias'] = float(bias_match.group(1))
        
        # Extract layer names
        layers_match = re.search(r"Using \d+ layers:\s*\[(.+?)\]", section)
        if layers_match:
            layers_str = layers_match.group(1)
            result['layer_names'] = [l.strip().strip("'\"") for l in layers_str.split(',')]
            result['num_layers'] = len(result['layer_names'])
        
        result['note'] = 'Uses DAC-style weighted ensemble of per-layer geometric separation scores'
    
    return result


def parse_geometric_with_dac_features(log_content: str) -> List[Dict[str, Any]]:
    """Parse Geometric + DAC Features ablation results."""
    results = []
    
    # Find all "GEOMETRIC WITH DAC FEATURES (Layer X)" sections
    pattern = r'GEOMETRIC WITH DAC FEATURES \(Layer (\d+)\)(.*?)(?=GEOMETRIC WITH DAC FEATURES \(Layer \d+\)|DAC WITH GEOMETRIC FEATURES|RUNNING METRIC-GUIDED)'
    
    matches = re.findall(pattern, log_content, re.DOTALL)
    
    for layer_num, section in matches:
        # Extract DAC layer name
        layer_match = re.search(r'Using DAC layer:\s*(\S+)', section)
        layer_name = layer_match.group(1) if layer_match else f'layer{layer_num}'
        
        result = extract_calibration_result(section)
        if result:
            result['method'] = f'Geometric + DAC Features (L{layer_num})'
            result['layer_name'] = layer_name
            result['note'] = 'Uses DAC feature extraction with Geometric calibration algorithm'
            
            # Extract feature extraction time
            extract_time = re.search(r'Feature extraction time:\s*(\d+\.\d+)s', section)
            if extract_time:
                result['extract_time_s'] = float(extract_time.group(1))
            
            results.append(result)
    
    return results


def parse_dac_with_geometric_features(log_content: str) -> Optional[Dict[str, Any]]:
    """Parse DAC + Geometric Features ablation results."""
    section_match = re.search(
        r'DAC WITH GEOMETRIC FEATURES \(All Layers\)(.*?)'
        r'(?=RUNNING METRIC-GUIDED LAYER SELECTION|RUNNING PRE-COMPUTED)',
        log_content,
        re.DOTALL
    )
    
    if not section_match:
        return None
    
    section = section_match.group(1)
    
    result = extract_calibration_result(section)
    if result:
        result['method'] = 'DAC + Geometric Features (All Layers)'
        result['note'] = 'Uses Geometric feature extraction (all layers) with DAC multi-layer weighted calibration'
        
        # Extract feature extraction time
        extract_time = re.search(r'Feature extraction time:\s*(\d+\.\d+)s', section)
        if extract_time:
            result['extract_time_s'] = float(extract_time.group(1))
        
        # Extract k-value
        k_match = re.search(r'DAC k-value:\s*(\d+)', section)
        if k_match:
            result['k_value'] = int(k_match.group(1))
        
        # Extract layer names
        layers_match = re.search(r'Using Geometric layers:\s*\[(.+?)\]', section)
        if layers_match:
            layers_str = layers_match.group(1)
            result['layer_names'] = [l.strip().strip("'\"") for l in layers_str.split(',')]
            result['num_layers'] = len(result['layer_names'])
    
    return result


def parse_metric_guided_calibration(log_content: str, single_layer_results: Dict) -> List[Dict[str, Any]]:
    """
    Parse metric-guided layer selection results.
    
    These experiments reuse precomputed single-layer results, so we match them up.
    """
    results = []
    
    # Find the metric-guided section
    section_match = re.search(
        r'RUNNING METRIC-GUIDED LAYER SELECTION EXPERIMENTS(.*?)'
        r'(?=RUNNING PRE-COMPUTED THROUGHPUT|PRE-COMPUTED THROUGHPUT EXPERIMENT)',
        log_content,
        re.DOTALL
    )
    
    if not section_match:
        return results
    
    section = section_match.group(1)
    
    # Find each metric-guided test
    pattern = r'Testing layer selected by (.+?):\s*(\S+).*?Reusing precomputed calibration result for (\S+)'
    
    matches = re.findall(pattern, section, re.DOTALL)
    
    for metric_name, selected_layer, layer_name in matches:
        # Get the precomputed result for this layer
        if layer_name in single_layer_results:
            result = single_layer_results[layer_name].copy()
            result['method'] = f'Geometric ({metric_name} → {layer_name})'
            result['selection_metric'] = metric_name
            
            # Extract metric value from layer_quality_metrics (we'll need to pass this)
            # For now, set to NaN - we'll fill this in from the partial JSON data
            result['metric_value'] = float('nan')
            
            results.append(result)
    
    return results


def parse_precomputed_throughput(log_content: str, single_layer_results: Dict) -> Optional[Dict[str, Any]]:
    """Parse pre-computed throughput experiment."""
    section_match = re.search(
        r'PRE-COMPUTED THROUGHPUT EXPERIMENT.*?'
        r'Testing throughput on best layer:\s*(\S+)(.*?)'
        r'(?=ABLATION ANALYSIS|$)',
        log_content,
        re.DOTALL
    )
    
    if not section_match:
        return None
    
    layer_name = section_match.group(1)
    section = section_match.group(2)
    
    # Get the base result from single layer results
    if layer_name not in single_layer_results:
        return None
    
    result = single_layer_results[layer_name].copy()
    
    # Update with precomputed throughput
    precomputed_throughput = re.search(r'Pre-computed Throughput:\s*(\d+\.\d+)\s*samples/sec', section)
    if precomputed_throughput:
        result['throughput_samples_per_sec'] = float(precomputed_throughput.group(1))
    
    # Update calibration time (different in precomputed mode)
    precomputed_time = re.search(r'Calibration time:\s*(\d+\.\d+)s', section)
    if precomputed_time:
        result['calibrate_time_s'] = float(precomputed_time.group(1))
    
    # Get semantic name for method
    semantic_match = re.search(r'Geometric Pre-computed \((.+?)\)', section)
    if semantic_match:
        result['method'] = f'Geometric Pre-computed ({semantic_match.group(1)})'
    else:
        result['method'] = f'Geometric Pre-computed ({layer_name})'
    
    return result


def parse_analysis_section(log_content: str) -> Dict[str, Any]:
    """Parse the analysis section."""
    analysis = {}
    
    # Find overall comparisons section
    section_match = re.search(
        r'OVERALL KEY COMPARISONS(.*?)(?=METRIC-GUIDED SELECTION ANALYSIS|$)',
        log_content,
        re.DOTALL
    )
    
    if not section_match:
        return analysis
    
    section = section_match.group(1)
    
    # Extract best overall
    best_overall = re.search(r'Best Overall:.*?(\S+.*?)\s+(\d+\.\d+)', section)
    if best_overall:
        analysis['best_overall_method'] = best_overall.group(1).strip()
        analysis['best_overall_ece'] = float(best_overall.group(2))
    
    # Extract improvements
    dac_improvement = re.search(r'DAC ECE improvement:\s*([-+]?\d+\.\d+)%', section)
    if dac_improvement:
        analysis['dac_ece_improvement_over_uncalibrated_percent'] = float(dac_improvement.group(1))
    
    best_improvement = re.search(r'Best Overall ECE improvement:\s*([-+]?\d+\.\d+)%', section)
    if best_improvement:
        analysis['best_overall_ece_improvement_over_uncalibrated_percent'] = float(best_improvement.group(1))
    
    # Extract improvement over DAC (from multi-layer analysis)
    dac_comparison = re.search(r'improvement over DAC by\s*(\d+\.\d+)%', log_content)
    if dac_comparison:
        # This is usually negative (best overall worse than DAC) or positive
        # We need to calculate it from ECE values
        pass
    
    # Extract best metric-guided from the metric-guided section
    metric_section = re.search(
        r'METRIC-GUIDED SELECTION ANALYSIS(.*?)(?=Traceback|$)',
        log_content,
        re.DOTALL
    )
    
    if metric_section:
        metric_text = metric_section.group(1)
        
        best_metric = re.search(r'Method:\s*(.+)', metric_text)
        if best_metric:
            analysis['best_metric_guided_method'] = best_metric.group(1).strip()
        
        best_metric_ece = re.search(r'ECE:\s*(\d+\.\d+)', metric_text)
        if best_metric_ece:
            analysis['best_metric_guided_ece'] = float(best_metric_ece.group(1))
        
        best_metric_selection = re.search(r'Selection metric:\s*(.+)', metric_text)
        if best_metric_selection:
            analysis['best_metric_guided_selection_metric'] = best_metric_selection.group(1).strip()
    
    analysis['note'] = 'Full ablation: features (DAC vs Geo) × algorithms (single-layer, concat, weighted-ensemble) + metric-guided selection'
    
    return analysis


def reconstruct_complete_json(
    partial_json: Dict[str, Any],
    log_path: Path
) -> Dict[str, Any]:
    """
    Reconstruct complete JSON from partial data and log file.
    """
    print(f"  📖 Parsing log file...")
    
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    # Start with partial JSON data (has experiment_config, uncalibrated, etc.)
    complete_json = partial_json.copy()
    
    # Parse all sections
    print(f"  🔍 Extracting single layer calibrations...")
    single_layer_results = parse_single_layer_calibrations(log_content)
    print(f"     Found {len(single_layer_results)} layers")
    
    print(f"  🔍 Extracting combined layers experiment...")
    combined_result = parse_combined_layers(log_content)
    
    print(f"  🔍 Extracting DAC-weighted experiment...")
    dac_weighted_result = parse_dac_weighted(log_content)
    
    print(f"  🔍 Extracting ablation studies...")
    geo_with_dac_features = parse_geometric_with_dac_features(log_content)
    dac_with_geo_features = parse_dac_with_geometric_features(log_content)
    print(f"     Geo+DAC features: {len(geo_with_dac_features)} layers")
    print(f"     DAC+Geo features: {'found' if dac_with_geo_features else 'not found'}")
    
    print(f"  🔍 Extracting metric-guided calibrations...")
    metric_guided_results = parse_metric_guided_calibration(log_content, single_layer_results)
    print(f"     Found {len(metric_guided_results)} metric selections")
    
    print(f"  🔍 Extracting precomputed throughput...")
    precomputed_result = parse_precomputed_throughput(log_content, single_layer_results)
    
    print(f"  🔍 Extracting analysis section...")
    analysis = parse_analysis_section(log_content)
    
    # Fill in metric values for metric-guided results from partial JSON
    if 'layer_quality_metrics' in partial_json:
        for result in metric_guided_results:
            layer_name = result['layer_names']
            metric_key = result['selection_metric']
            
            # Extract base metric name
            if "PSC" in metric_key:
                base_metric = "nc4"
            else:
                base_metric = metric_key.replace(" (Max)", "").replace(" (Min)", "").replace(" (Med)", "")
            
            if layer_name in partial_json['layer_quality_metrics']:
                if base_metric in partial_json['layer_quality_metrics'][layer_name]:
                    result['metric_value'] = partial_json['layer_quality_metrics'][layer_name][base_metric]
    
    # Calculate improvement over DAC if we have the data
    if 'dac_standalone' in partial_json and 'best_overall_ece' in analysis:
        dac_ece = partial_json['dac_standalone']['ece']
        best_ece = analysis['best_overall_ece']
        improvement = ((dac_ece - best_ece) / dac_ece) * 100
        analysis['improvement_over_dac_percent'] = improvement
    
    # Reconstruct geometric_original with proper layer indexing
    geometric_original = {}
    for i, (layer_name, result) in enumerate(sorted(single_layer_results.items()), 1):
        geometric_original[f'layer{i}'] = result
    
    complete_json['geometric_original'] = geometric_original
    
    if combined_result:
        complete_json['geometric_concatenated'] = combined_result
    
    if dac_weighted_result:
        complete_json['geometric_dac_weighted'] = dac_weighted_result
    
    if precomputed_result:
        complete_json['geometric_precomputed'] = precomputed_result
    
    # Metric-guided calibration - need to handle properly
    # Check if partial JSON has incomplete metric_guided_calibration
    if 'metric_guided_calibration' in partial_json:
        # Use partial data for entries that were saved, fill rest from log
        existing_count = len([x for x in partial_json['metric_guided_calibration'] if 'ece' in x])
        print(f"     Partial JSON has {existing_count} complete entries")
        
        # Merge: use partial data where available (more detailed), log data for missing
        merged_metric_guided = []
        
        for entry in partial_json['metric_guided_calibration']:
            if 'ece' in entry and 'selection_metric' in entry:
                merged_metric_guided.append(entry)
        
        # Add any from log that aren't in partial
        partial_metrics = {x['selection_metric'] for x in merged_metric_guided if 'selection_metric' in x}
        for entry in metric_guided_results:
            if entry['selection_metric'] not in partial_metrics:
                merged_metric_guided.append(entry)
        
        complete_json['metric_guided_calibration'] = merged_metric_guided
    else:
        complete_json['metric_guided_calibration'] = metric_guided_results
    
    # Ablation section
    ablation = {
        'geometric_with_dac_features': {f'layer{i+1}': result for i, result in enumerate(geo_with_dac_features)}
    }
    
    if dac_with_geo_features:
        ablation['dac_with_geometric_features'] = dac_with_geo_features
    
    complete_json['ablation'] = ablation
    
    # Analysis section
    complete_json['analysis'] = analysis
    
    return complete_json


def main():
    """Main recovery process."""
    print("="*80)
    print("COMPREHENSIVE DAC COMPARISON JSON RECOVERY SCRIPT")
    print("="*80)
    print(f"\nJSON Directory: {JSON_DIR}")
    print(f"Log Directory:  {LOG_DIR}")
    print(f"Backup Directory: {BACKUP_DIR}")
    print()
    
    # Create backup directory
    BACKUP_DIR.mkdir(exist_ok=True)
    
    # Find all JSON files
    json_files = list(JSON_DIR.glob("ablation_*.json"))
    print(f"Found {len(json_files)} JSON files to check\n")
    
    corrupted_files = []
    fixed_files = []
    complete_files = []
    failed_files = []
    
    for json_path in sorted(json_files):
        print(f"\n{'='*80}")
        print(f"Processing: {json_path.name}")
        print('='*80)
        
        # Check if corrupted
        is_corrupted, reason, partial_data = is_json_corrupted(json_path)
        
        if not is_corrupted:
            print(f"✅ Complete - skipping")
            complete_files.append(json_path)
            continue
        
        print(f"⚠️  CORRUPTED: {reason}")
        
        if not partial_data:
            print(f"❌ Cannot load any data from JSON - too corrupted")
            failed_files.append(json_path)
            continue
        
        corrupted_files.append(json_path)
        
        # Find matching log
        log_path = find_matching_log(json_path, LOG_DIR)
        
        if not log_path:
            print(f"❌ Cannot fix without log file")
            failed_files.append(json_path)
            continue
        
        print(f"📄 Found log: {log_path.name}")
        
        try:
            # Reconstruct complete JSON
            complete_json = reconstruct_complete_json(partial_data, log_path)
            
            # Backup original
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = BACKUP_DIR / f"{json_path.stem}_{timestamp}.json.bak"
            shutil.copy2(json_path, backup_path)
            print(f"💾 Backed up to: {backup_path.name}")
            
            # Write reconstructed JSON
            with open(json_path, 'w') as f:
                json.dump(complete_json, f, indent=2)
            
            print(f"✅ Successfully reconstructed complete JSON!")
            fixed_files.append(json_path)
            
        except Exception as e:
            print(f"❌ Error reconstructing JSON: {e}")
            import traceback
            traceback.print_exc()
            failed_files.append(json_path)
    
    # Summary
    print("\n" + "="*80)
    print("RECOVERY SUMMARY")
    print("="*80)
    print(f"Total files:        {len(json_files)}")
    print(f"Complete:           {len(complete_files)} ✅")
    print(f"Corrupted found:    {len(corrupted_files)} ⚠️")
    print(f"Successfully fixed: {len(fixed_files)} ✅")
    print(f"Failed to fix:      {len(failed_files)} ❌")
    print()
    
    if fixed_files:
        print("Successfully recovered:")
        for f in fixed_files:
            print(f"  ✅ {f.name}")
        print()
    
    if failed_files:
        print("Failed files (may need manual intervention):")
        for f in failed_files:
            print(f"  ❌ {f.name}")
        print()
    
    print(f"✅ Recovery complete!")
    print(f"   Backups saved to: {BACKUP_DIR}")


if __name__ == "__main__":
    main()