
import os
import json
import shutil
import tempfile
import sys
# Add project root to path
sys.path.append("/home/ptamar/geometric-internal-calibration")
from Experiments.run_random_ablation_jobs import check_ood_results_exist

def test_ood_check():
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing in {temp_dir}")
        training_method = "baseline"
        dataset = "cifar10"
        model = "resnet18"
        seed = 42
        
        filename = f"ablation_{training_method}_{dataset}_{model}_seed{seed}.json"
        
        # Case 1: File doesn't exist
        print("Test 1: File missing ->", end=" ")
        result = check_ood_results_exist(temp_dir, training_method, dataset, model, seed)
        print("PASS" if result is False else f"FAIL (got {result})")
        
        # Case 2: File exists but missing OOD metrics (partial)
        print("Test 2: File exists, no OOD ->", end=" ")
        data = {
            "baselines": {
                "temp_scaling": {"accuracy": 0.9, "ece": 0.05} 
            }
        }
        with open(os.path.join(temp_dir, filename), 'w') as f:
            json.dump(data, f)
        result = check_ood_results_exist(temp_dir, training_method, dataset, model, seed)
        print("PASS" if result is False else f"FAIL (got {result})")
        
        # Case 3: File exists and has OOD (complete)
        print("Test 3: File exists, with OOD ->", end=" ")
        data["baselines"]["temp_scaling"]["ood_auroc"] = 0.8
        data["baselines"]["temp_scaling"]["ood_fpr95"] = 0.1
        with open(os.path.join(temp_dir, filename), 'w') as f:
            json.dump(data, f)
        result = check_ood_results_exist(temp_dir, training_method, dataset, model, seed)
        print("PASS" if result is True else f"FAIL (got {result})")

        # Case 4: mixed (one complete, one incomplete)
        print("Test 4: Mixed results ->", end=" ")
        data["baselines"]["uncalibrated"] = {"accuracy": 0.85, "ece": 0.1} # Missing OOD
        with open(os.path.join(temp_dir, filename), 'w') as f:
            json.dump(data, f)
        result = check_ood_results_exist(temp_dir, training_method, dataset, model, seed)
        print("PASS" if result is False else f"FAIL (got {result})")
    
if __name__ == "__main__":
    test_ood_check()
