#!/usr/bin/env python3
"""
Example workflow demonstrating the complete BT chooser pipeline.

This script shows how to train, evaluate, and use the Bradley-Terry chooser
for layer selection in calibration tasks.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.bt_chooser import (
    load_runs, BradleyTerryChooser, choose_layer,
    extend_two_stage_selector_with_bt
)


def create_sample_data():
    """Create sample experiment data for demonstration."""
    print("📊 Creating sample experiment data...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    runs_dir = temp_dir / "runs"
    runs_dir.mkdir()
    
    # Create sample runs
    sample_runs = [
        {
            "run_id": "cifar10_resnet50_001",
            "dataset": "cifar10",
            "backbone": "resnet50",
            "seed": 42,
            "layers": [
                {
                    "layer_idx": 0,
                    "metrics": {
                        "prototype_softmax_ece": 0.12,
                        "reliability_curve_quality": 0.85,
                        "confidence_distance_correlation": 0.72,
                        "boundary_proximity_correlation": 0.68
                    },
                    "target_ece": 0.15
                },
                {
                    "layer_idx": 1,
                    "metrics": {
                        "prototype_softmax_ece": 0.14,
                        "reliability_curve_quality": 0.82,
                        "confidence_distance_correlation": 0.68,
                        "boundary_proximity_correlation": 0.65
                    },
                    "target_ece": 0.18
                },
                {
                    "layer_idx": 2,
                    "metrics": {
                        "prototype_softmax_ece": 0.10,
                        "reliability_curve_quality": 0.90,
                        "confidence_distance_correlation": 0.75,
                        "boundary_proximity_correlation": 0.72
                    },
                    "target_ece": 0.12
                }
            ]
        },
        {
            "run_id": "cifar10_resnet50_002",
            "dataset": "cifar10",
            "backbone": "resnet50",
            "seed": 43,
            "layers": [
                {
                    "layer_idx": 0,
                    "metrics": {
                        "prototype_softmax_ece": 0.13,
                        "reliability_curve_quality": 0.83,
                        "confidence_distance_correlation": 0.70,
                        "boundary_proximity_correlation": 0.66
                    },
                    "target_ece": 0.16
                },
                {
                    "layer_idx": 1,
                    "metrics": {
                        "prototype_softmax_ece": 0.11,
                        "reliability_curve_quality": 0.88,
                        "confidence_distance_correlation": 0.73,
                        "boundary_proximity_correlation": 0.70
                    },
                    "target_ece": 0.13
                }
            ]
        },
        {
            "run_id": "imagenet_resnet18_001",
            "dataset": "imagenet",
            "backbone": "resnet18",
            "seed": 44,
            "layers": [
                {
                    "layer_idx": 0,
                    "metrics": {
                        "prototype_softmax_ece": 0.15,
                        "reliability_curve_quality": 0.80,
                        "confidence_distance_correlation": 0.65,
                        "boundary_proximity_correlation": 0.62
                    },
                    "target_ece": 0.20
                },
                {
                    "layer_idx": 1,
                    "metrics": {
                        "prototype_softmax_ece": 0.12,
                        "reliability_curve_quality": 0.85,
                        "confidence_distance_correlation": 0.68,
                        "boundary_proximity_correlation": 0.65
                    },
                    "target_ece": 0.17
                }
            ]
        }
    ]
    
    # Save sample runs
    for i, run in enumerate(sample_runs):
        run_file = runs_dir / f"run_{i:03d}_per_layer_ground_truth.json"
        with open(run_file, 'w') as f:
            json.dump(run, f, indent=2)
    
    print(f"✅ Created {len(sample_runs)} sample runs in {runs_dir}")
    return temp_dir, runs_dir


def train_bt_chooser(runs_dir, output_path):
    """Train a Bradley-Terry chooser on the sample data."""
    print("\n🤖 Training Bradley-Terry chooser...")
    
    # Load runs
    runs = load_runs(str(runs_dir))
    print(f"📊 Loaded {len(runs)} runs")
    
    # Create chooser
    metric_names = ["prototype_softmax_ece", "reliability_curve_quality", 
                   "confidence_distance_correlation", "boundary_proximity_correlation"]
    
    chooser = BradleyTerryChooser(
        metric_names=metric_names,
        use_ranks=True,
        standardize=True,
        penalty="l2",
        C_grid=[0.1, 1.0, 10.0],
        class_weight="balanced",
        eps_tie=0.003,
        max_pairs_per_run=100,
        random_state=42
    )
    
    # Train model
    report = chooser.fit(runs, group_by="dataset")
    
    print(f"✅ Training completed!")
    print(f"   Best C: {report['best_C']}")
    print(f"   CV Hit Rate: {report['cv_results'][0]['hit_rate_mean']:.3f}")
    print(f"   CV Regret: {report['cv_results'][0]['regret_mean']:.4f}")
    
    # Save model
    chooser.save(output_path)
    print(f"💾 Model saved to {output_path}")
    
    return chooser


def evaluate_bt_chooser(bt_path, runs_dir):
    """Evaluate the trained chooser."""
    print("\n📈 Evaluating Bradley-Terry chooser...")
    
    # Load runs
    runs = load_runs(str(runs_dir))
    
    # Load chooser
    chooser = BradleyTerryChooser(
        metric_names=[],  # Will be loaded from file
        use_ranks=True,
        random_state=42
    )
    chooser.load(bt_path)
    
    # Evaluate each run
    hit_count = 0
    total_regret = 0.0
    tie_count = 0
    borda_resolved = 0
    
    for run in runs:
        try:
            result = choose_layer(bt_path, run, apply_pareto=True, tie_delta=1e-4)
            
            # Find true best layer
            layer_to_ece = {layer["layer_idx"]: layer["target_ece"] for layer in run.layers}
            true_best = min(layer_to_ece.keys(), key=lambda x: layer_to_ece[x])
            
            if result["strategy"] == "single":
                chosen = result["layer_idx"]
                hit_count += 1 if chosen == true_best else 0
                regret = layer_to_ece[chosen] - layer_to_ece[true_best]
                total_regret += regret
                
            elif result["strategy"] == "tie":
                tied_layers = [item[0] for item in result["top2"]]
                hit_count += 1 if true_best in tied_layers else 0
                tied_eces = [layer_to_ece[layer] for layer in tied_layers]
                regret = min(tied_eces) - layer_to_ece[true_best]
                total_regret += regret
                tie_count += 1
                borda_resolved += 1 if result.get("borda_used", False) else 0
                
        except Exception as e:
            print(f"⚠️ Error evaluating run {run.run_id}: {e}")
    
    # Print results
    n_runs = len(runs)
    hit_rate = hit_count / n_runs
    avg_regret = total_regret / n_runs
    tie_rate = tie_count / n_runs
    borda_rate = borda_resolved / tie_count if tie_count > 0 else 0.0
    
    print(f"✅ Evaluation completed!")
    print(f"   Hit Rate: {hit_rate:.3f} ({hit_count}/{n_runs})")
    print(f"   Avg Regret: {avg_regret:.4f}")
    print(f"   Tie Rate: {tie_rate:.3f} ({tie_count}/{n_runs})")
    print(f"   Borda Resolution: {borda_rate:.3f} ({borda_resolved}/{tie_count})")


def demonstrate_integration():
    """Demonstrate integration with TwoStageLayerSelector."""
    print("\n🔗 Demonstrating TwoStageLayerSelector integration...")
    
    try:
        # Import required modules
        from Calibrators.metrics import TwoStageLayerSelector, AdHocLayerSelector
        
        # Extend the class
        extend_two_stage_selector_with_bt()
        
        # Create selector
        base_selector = AdHocLayerSelector()
        two_stage_selector = TwoStageLayerSelector(base_selector)
        
        print("✅ TwoStageLayerSelector extended with BT chooser")
        print("   - select_with_bt() method available")
        print("   - Original select() method preserved")
        print("   - Ready for integration in production code")
        
    except ImportError as e:
        print(f"⚠️ Could not demonstrate integration: {e}")
        print("   This is expected if Calibrators module is not available")


def main():
    """Run the complete example workflow."""
    print("🚀 Bradley-Terry Chooser Example Workflow")
    print("=" * 50)
    
    # Create sample data
    temp_dir, runs_dir = create_sample_data()
    
    try:
        # Train chooser
        model_path = temp_dir / "bt_chooser.json"
        chooser = train_bt_chooser(runs_dir, str(model_path))
        
        # Evaluate chooser
        evaluate_bt_chooser(str(model_path), runs_dir)
        
        # Demonstrate integration
        demonstrate_integration()
        
        print("\n🎉 Example workflow completed successfully!")
        print(f"📁 Sample data available in: {temp_dir}")
        print(f"🤖 Trained model saved to: {model_path}")
        
    except Exception as e:
        print(f"❌ Error in workflow: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temporary files")


if __name__ == "__main__":
    main()
