"""
NC-Guided Multi-Layer Ensemble Calibration Runner

Uses NC metrics (NC1, NC4) to select layers, then runs multi-layer ensemble
calibration with different weighting strategies:
1. Uniform: All layers weighted equally
2. Learned: Optimize weights via Brier score on validation
3. NC4-rank: Weight layers by their NC4 scores
4. NC1-rank: Weight layers by their NC1 scores
5. Pareto-rank: Weight layers by their position on Pareto front

Usage:
    python run_nc_ensemble.py --mode slurm --submit
"""

import argparse
import json
import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import itertools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
from utils.logging_config import get_logger
logger = get_logger(__name__)

# Import model loading utilities for accuracy check
try:
    import torch
    from Experiments.run_post_hoc_calibration import get_data_loaders, load_trained_model
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - accuracy checks will be skipped")


@dataclass
class NCEnsembleConfig:
    """Configuration for a single NC-based ensemble experiment"""
    dataset: str
    model: str
    training_loss: str
    seed: int
    layer_selection_strategy: str  # 'pareto_top5', 'nc4_top5', 'nc1_top5', 'strict', 'psc'
    weighting_method: str  # 'uniform', 'learned', 'nc4_rank', 'nc1_rank', 'pareto_rank'
    selected_layers: List[int]
    nc_metrics_file: Path
    output_dir: Path


class NCEnsembleJobManager:
    """Manages NC-based ensemble experiments"""
    
    def __init__(
        self,
        datasets: List[str],
        models: List[str],
        training_losses: List[str],
        seeds: List[int],
        nc_metrics_dir: str = "nc_metrics_results",
        checkpoint_base_dir: str = "aaai_full_experiments/results/baseline",
        output_dir: str = "nc_ensemble_results",
        ensemble_script: str = "Experiments/multi_layer_ensemble.py",
        conda_env: str = "tamar_n_env"
    ):
        self.datasets = datasets
        self.models = models
        self.training_losses = training_losses
        self.seeds = seeds
        self.nc_metrics_dir = Path(nc_metrics_dir)
        self.checkpoint_base_dir = Path(checkpoint_base_dir)
        self.output_dir = Path(output_dir)
        self.ensemble_script = Path(ensemble_script)
        self.conda_env = conda_env
        
        # SLURM settings
        self.slurm_partition = "gpu_partition"
        self.slurm_time = "2-00:00:00"
        self.slurm_mem = "32G"
        self.slurm_cpus = 4
        self.slurm_gpus = "rtx_4090:1"
        
        # Cache for model accuracy checks (to avoid re-checking same model)
        self._accuracy_cache: Dict[Tuple[str, str, str, int], Optional[float]] = {}
    
    @staticmethod
    def get_clean_dataset_name(dataset: str) -> str:
        """Map corruption-C datasets to their clean counterparts."""
        if dataset == "cifar10c":
            return "cifar10"
        if dataset == "cifar100c":
            return "cifar100"
        return dataset
    
    def check_model_accuracy(
        self,
        training_loss: str,
        dataset: str,
        model: str,
        seed: int,
        min_accuracy: float = 0.6
    ) -> Optional[float]:
        """
        Check if model accuracy meets the minimum threshold.
        Returns accuracy if >= min_accuracy, None otherwise.
        Uses caching to avoid re-checking the same model.
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - skipping accuracy check")
            return 1.0  # Assume valid if we can't check
        
        clean_dataset = self.get_clean_dataset_name(dataset)
        
        cache_key = (training_loss, clean_dataset, model, seed)
        if cache_key in self._accuracy_cache:
            return self._accuracy_cache[cache_key]
        
        try:
            # Build model path
            dynamic_folder_name = f"{training_loss}_{clean_dataset}_{model}_seed{seed}"
            model_path = (
                self.checkpoint_base_dir /
                training_loss /
                clean_dataset /
                model /
                f"seed{seed}" /
                dynamic_folder_name /
                "best_model.pth"
            )
            
            if not model_path.exists():
                logger.warning(f"Model checkpoint not found: {model_path}")
                self._accuracy_cache[cache_key] = None
                return None
            
            # Setup device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Get data loaders to determine num_classes
            _, _, test_loader, num_classes = get_data_loaders(
                dataset=clean_dataset,
                batch_size=256,
                seed=seed
            )
            
            # Load model
            model_obj = load_trained_model(
                str(model_path),
                model,
                num_classes,
                device,
                dataset=clean_dataset
            )
            model_obj = model_obj.to(device)
            model_obj.eval()
            
            # Evaluate accuracy
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    outputs = model_obj(batch_x)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            accuracy = correct / total if total > 0 else 0.0
            
            # Cache result
            if accuracy >= min_accuracy:
                self._accuracy_cache[cache_key] = accuracy
                logger.debug(f"Model {training_loss}/{clean_dataset}/{model}/seed{seed} accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) - PASSED")
                return accuracy
            else:
                self._accuracy_cache[cache_key] = None
                logger.warning(
                    f"Model {training_loss}/{clean_dataset}/{model}/seed{seed} accuracy: {accuracy:.4f} ({accuracy*100:.2f}%) "
                    f"is below {min_accuracy*100:.0f}% threshold - SKIPPING"
                )
                return None
                
        except Exception as e:
            logger.error(f"Failed to check accuracy for {training_loss}/{clean_dataset}/{model}/seed{seed}: {e}")
            self._accuracy_cache[cache_key] = None
            return None
    
    def load_nc_metrics(self, training_loss: str, dataset: str, model: str, seed: int) -> Optional[Dict]:
        """Load NC metrics results for a specific model"""
        clean_dataset = self.get_clean_dataset_name(dataset)
        metrics_file = (
            self.nc_metrics_dir / 
            training_loss / 
            clean_dataset / 
            model / 
            f"seed{seed}" / 
            "nc_metrics_validation.json"
        )
        
        if not metrics_file.exists():
            logger.debug(f"NC metrics not found: {metrics_file}")
            return None
        
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {metrics_file}: {e}")
            return None
    
    def compute_analysis_on_the_fly(self, nc_data: Dict) -> Dict:
        """
        Compute analysis fields on-the-fly from raw NC metrics.
        This is used as a fallback when precomputed analysis is missing or incomplete.
        """
        nc_metrics = nc_data.get('nc_metrics', {})
        if not nc_metrics:
            return {}

        # Filter valid layers (no errors)
        valid_layers = [(int(k), v['nc1'], v['nc4']) for k, v in nc_metrics.items() if 'error' not in v]

        if not valid_layers:
            return {}

        # Compute Pareto front (maximize both NC1 and NC4)
        pareto_front = []
        for idx1, nc1_1, nc4_1 in valid_layers:
            is_dominated = False
            for idx2, nc1_2, nc4_2 in valid_layers:
                if idx1 != idx2:
                    # idx2 dominates idx1 if it's better or equal in both, and strictly better in at least one
                    if (nc1_2 >= nc1_1 and nc4_2 >= nc4_1) and (nc1_2 > nc1_1 or nc4_2 > nc4_1):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_front.append({'layer_idx': int(idx1), 'nc1': float(nc1_1), 'nc4': float(nc4_1)})

        # Sort by NC4 descending
        pareto_front.sort(key=lambda x: x['nc4'], reverse=True)

        # Find best layers by different criteria
        best_nc4_layer = max(valid_layers, key=lambda x: x[2]) if valid_layers else (None, 0, 0)
        best_nc1_layer = max(valid_layers, key=lambda x: x[1]) if valid_layers else (None, 0, 0)

        # PSC recommended layer (best NC4 among NC1 > 0.2)
        psc_viable = [(idx, nc1, nc4) for idx, nc1, nc4 in valid_layers if nc1 > 0.2]
        psc_recommended = max(psc_viable, key=lambda x: x[2]) if psc_viable else (None, 0, 0)

        # Strict criteria layers (NC1 > 0.2 AND NC4 > 0.8)
        strict_layers = [int(idx) for idx, nc1, nc4 in valid_layers if nc1 > 0.2 and nc4 > 0.8]

        analysis = {
            'pareto_front': pareto_front,
            'best_nc4_layer': {
                'layer_idx': int(best_nc4_layer[0]) if best_nc4_layer[0] is not None else None,
                'nc1': float(best_nc4_layer[1]),
                'nc4': float(best_nc4_layer[2]),
            },
            'best_nc1_layer': {
                'layer_idx': int(best_nc1_layer[0]) if best_nc1_layer[0] is not None else None,
                'nc1': float(best_nc1_layer[1]),
                'nc4': float(best_nc1_layer[2]),
            },
            'psc_recommended_layer': {
                'layer_idx': int(psc_recommended[0]) if psc_recommended[0] is not None else None,
                'nc1': float(psc_recommended[1]),
                'nc4': float(psc_recommended[2]),
            },
            'strict_criteria_layers': strict_layers,
            'pareto_recommended': pareto_front[0] if pareto_front else None,
        }

        logger.info(f"Computed analysis on-the-fly: {len(pareto_front)} Pareto-optimal layers, {len(strict_layers)} strict layers")

        return analysis

    def select_layers_by_strategy(self, nc_data: Dict, strategy: str) -> List[int]:
        """Select layers based on NC metrics using different strategies"""

        # Check if analysis exists, if not compute it on-the-fly
        if 'analysis' not in nc_data or not nc_data.get('analysis'):
            logger.info(f"Analysis field missing in NC data, computing on-the-fly for strategy '{strategy}'")
            nc_data['analysis'] = self.compute_analysis_on_the_fly(nc_data)

        # Also check if specific fields are empty and compute if needed
        analysis = nc_data.get('analysis', {})
        needs_recompute = False

        # Check if pareto_front is needed but empty/missing
        if strategy in ['pareto_top5', 'pareto_top3'] and not analysis.get('pareto_front'):
            needs_recompute = True

        # Check if strict_criteria_layers is needed but empty/missing
        if strategy == 'strict' and not analysis.get('strict_criteria_layers'):
            needs_recompute = True

        # Check if psc_recommended_layer is needed but missing/None
        if strategy == 'psc' and (not analysis.get('psc_recommended_layer') or analysis.get('psc_recommended_layer', {}).get('layer_idx') is None):
            needs_recompute = True

        # Check if best_nc4_layer is needed but missing/None
        if strategy == 'best_nc4' and (not analysis.get('best_nc4_layer') or analysis.get('best_nc4_layer', {}).get('layer_idx') is None):
            needs_recompute = True

        if needs_recompute:
            logger.info(f"Required analysis field missing/empty for strategy '{strategy}', computing on-the-fly")
            nc_data['analysis'] = self.compute_analysis_on_the_fly(nc_data)
            analysis = nc_data['analysis']

        if strategy == 'pareto_top5':
            # Top 5 from Pareto front (sorted by NC4)
            pareto = analysis.get('pareto_front', [])
            return [p['layer_idx'] for p in pareto[:5]]

        elif strategy == 'pareto_top3':
            pareto = analysis.get('pareto_front', [])
            return [p['layer_idx'] for p in pareto[:3]]

        elif strategy == 'nc4_top5':
            # Top 5 layers by NC4 score
            nc_metrics = nc_data.get('nc_metrics', {})
            valid = [(int(k), v['nc4']) for k, v in nc_metrics.items() if 'error' not in v]
            valid.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, _ in valid[:5]]

        elif strategy == 'nc4_top3':
            nc_metrics = nc_data.get('nc_metrics', {})
            valid = [(int(k), v['nc4']) for k, v in nc_metrics.items() if 'error' not in v]
            valid.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, _ in valid[:3]]

        elif strategy == 'nc1_top5':
            # Top 5 layers by NC1 score
            nc_metrics = nc_data.get('nc_metrics', {})
            valid = [(int(k), v['nc1']) for k, v in nc_metrics.items() if 'error' not in v]
            valid.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, _ in valid[:5]]

        elif strategy == 'nc1_top3':
            # Top 3 layers by NC1 score
            nc_metrics = nc_data.get('nc_metrics', {})
            valid = [(int(k), v['nc1']) for k, v in nc_metrics.items() if 'error' not in v]
            valid.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, _ in valid[:3]]

        elif strategy == 'strict':
            # Layers meeting strict criteria (NC1 > 0.2 AND NC4 > 0.8)
            return analysis.get('strict_criteria_layers', [])

        elif strategy == 'psc':
            # PSC recommended layer (best NC4 with NC1 > 0.2)
            psc = analysis.get('psc_recommended_layer', {})
            layer_idx = psc.get('layer_idx')
            return [layer_idx] if layer_idx is not None else []

        elif strategy == 'best_nc4':
            # Single best NC4 layer
            best = analysis.get('best_nc4_layer', {})
            layer_idx = best.get('layer_idx')
            return [layer_idx] if layer_idx is not None else []
        
        elif strategy == 'best_nc1':
            # Single best NC1 layer
            best = analysis.get('best_nc1_layer', {})
            layer_idx = best.get('layer_idx')
            return [layer_idx] if layer_idx is not None else []
        
        elif strategy == "single_layer":
            # layer that nc1>0.2 and argmax on nc4
            nc_metrics = nc_data.get('nc_metrics', {})
            valid = [(int(k), v['nc1'], v['nc4']) for k, v in nc_metrics.items() if 'error' not in v and v['nc1'] > 0.2]
            valid.sort(key=lambda x: x[2], reverse=True)
            return [idx for idx, nc1, nc4 in valid if nc1 > 0.2]


        else:
            raise ValueError(f"Unknown layer selection strategy: {strategy}")
    
    def get_layer_weights(self, nc_data: Dict, layers: List[int], method: str) -> Optional[List[float]]:
        """
        Get initial weights for layers based on NC metrics.
        Returns None for methods that learn weights from scratch (uniform, learned).
        """
        if method in ['uniform', 'learned']:
            return None  # Will be handled by ensemble script
        
        nc_metrics = nc_data.get('nc_metrics', {})
        
        if method == 'nc4_rank':
            # Weight by NC4 scores (normalized)
            scores = [nc_metrics[str(idx)]['nc4'] for idx in layers]
            total = sum(scores)
            return [s / total for s in scores] if total > 0 else None
        
        elif method == 'nc1_rank':
            # Weight by NC1 scores (normalized)
            scores = [nc_metrics[str(idx)]['nc1'] for idx in layers]
            total = sum(scores)
            return [s / total for s in scores] if total > 0 else None
        
        elif method == 'pareto_rank':
            # Weight by inverse rank in Pareto front
            pareto = nc_data.get('analysis', {}).get('pareto_front', [])
            layer_to_rank = {p['layer_idx']: i+1 for i, p in enumerate(pareto)}
            
            # Inverse rank weighting
            inv_ranks = [1.0 / layer_to_rank.get(idx, len(pareto)+1) for idx in layers]
            total = sum(inv_ranks)
            return [r / total for r in inv_ranks] if total > 0 else None
        
        return None
    
    def generate_experiments(
        self,
        layer_strategies: List[str],
        weighting_methods: List[str]
    ) -> List[NCEnsembleConfig]:
        """Generate all experiment configurations"""
        
        experiments = []
        
        for dataset, model, loss, seed in itertools.product(
            self.datasets, self.models, self.training_losses, self.seeds
        ):
            # Check model accuracy first - skip if below 60%
            accuracy = self.check_model_accuracy(loss, dataset, model, seed, min_accuracy=0.6)
            if accuracy is None:
                logger.debug(f"Skipping {loss}/{dataset}/{model}/seed{seed} - accuracy below 60% threshold")
                continue
            
            # Load NC metrics
            nc_data = self.load_nc_metrics(loss, dataset, model, seed)
            if nc_data is None:
                logger.warning(f"No NC metrics for {loss}/{dataset}/{model}/seed{seed}")
                continue
            
            # Try each layer selection strategy
            for layer_strategy in layer_strategies:
                try:
                    selected_layers = self.select_layers_by_strategy(nc_data, layer_strategy)
                    
                    if not selected_layers:
                        logger.warning(f"No layers selected with {layer_strategy} for {loss}/{dataset}/{model}/seed{seed}")
                        continue
                    
                    # Create experiment for each weighting method
                    for weighting_method in weighting_methods:
                        output_dir = (
                            self.output_dir /
                            loss /
                            dataset /
                            model /
                            f"seed{seed}" /
                            layer_strategy /
                            weighting_method
                        )
                        
                        experiments.append(NCEnsembleConfig(
                            dataset=dataset,
                            model=model,
                            training_loss=loss,
                            seed=seed,
                            layer_selection_strategy=layer_strategy,
                            weighting_method=weighting_method,
                            selected_layers=selected_layers,
                            nc_metrics_file=self.nc_metrics_dir / loss / dataset / model / f"seed{seed}" / "nc_metrics_test.json",
                            output_dir=output_dir
                        ))
                
                except Exception as e:
                    logger.error(f"Failed to create experiments for {layer_strategy}: {e}")
                    continue
        
        return experiments
    
    @staticmethod
    def make_config_signature(
        layers: List[int],
        weights: Optional[List[float]],
        dataset: str,
        model: str,
        seed: int
    ) -> str:
        """
        Create a canonical signature for an ensemble configuration.
        """
        if weights is None:
            sig_data = {
                'layers': sorted(int(idx) for idx in layers),
                'weights': None,
                'dataset': dataset,
                'model': model,
                'seed': seed
            }
        else:
            paired = sorted(zip(layers, weights))
            sorted_layers = [int(layer) for layer, _ in paired]
            sorted_weights = [round(float(weight), 3) for _, weight in paired]
            sig_data = {
                'layers': sorted_layers,
                'weights': sorted_weights,
                'dataset': dataset,
                'model': model,
                'seed': seed
            }
        
        config_str = json.dumps(sig_data, sort_keys=True)
        import hashlib
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    @staticmethod
    def weights_are_nearly_uniform(weights: List[float], tolerance: float = 0.01) -> bool:
        """Check if weights are effectively uniform (all within tolerance of 1/K)."""
        if not weights:
            return False
        
        count = len(weights)
        if count == 0:
            return False
        
        uniform_weight = 1.0 / count
        return all(abs(float(w) - uniform_weight) < tolerance for w in weights)
    
    def compute_nc_weights(
        self,
        nc_data: Dict,
        selected_layers: List[int],
        method: str
    ) -> Optional[List[float]]:
        """Compute NC-derived weights for a given configuration."""
        if method not in ['nc4_rank', 'nc1_rank', 'pareto_rank']:
            return None
        
        if 'analysis' not in nc_data or not nc_data.get('analysis'):
            nc_data['analysis'] = self.compute_analysis_on_the_fly(nc_data)
        
        return self.get_layer_weights(nc_data, selected_layers, method)
    
    def check_experiment_completed(self, config: NCEnsembleConfig) -> bool:
        """Check if experiment already completed"""
        # Check for results file
        results_file = config.output_dir / "ensemble_results.json"
        return results_file.exists()
    
    def generate_sbatch_command(self, config: NCEnsembleConfig) -> str:
        """Generate sbatch command for a single experiment"""
        job_name = f"NCEns_{config.layer_selection_strategy[:4]}_{config.weighting_method[:4]}_{config.model}_{config.dataset}_s{config.seed}"

        project_root = Path.cwd()

        # Create logs directory
        log_dir = Path("slurm_logs") / "nc_ensemble"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{job_name}_%j.log"

        # Build Python command
        layers_str = ",".join(map(str, config.selected_layers))

        # Shared cache directory at the seed level
        shared_cache_dir = (
            self.output_dir /
            config.training_loss /
            config.dataset /
            config.model /
            f"seed{config.seed}" /
            "_shared_feat_cache"
        )

        python_cmd = [
            sys.executable,
            str(self.ensemble_script),
            # IMPORTANT: pass the ORIGINAL dataset name (e.g., cifar10c/cifar100c)
            # The ensemble script is responsible for mapping to the clean dataset
            "--dataset", config.dataset,
            "--model-name", config.model,
            "--seed", str(config.seed),
            "--training-method", config.training_loss,
            "--results-base-dir", str(self.checkpoint_base_dir),
            "--output-dir", str(config.output_dir),
            "--feat-cache", str(shared_cache_dir),
            "--selection-method", config.layer_selection_strategy,
            "--layer_indices", layers_str,
            "--weighting-methods", config.weighting_method,
            "--device", "cuda",
            "--bins", "15",
            "--batch-size", "512",
            "--learned-max-iter", "200",
            "--learned-use-lbfgs",
            # Note: --compression-ratio is auto-selected in multi_layer_ensemble.py
            # (32x for CIFAR-100, 16x for others). Can be overridden via __init__ if needed.
        ]
        
        # Add corruption benchmark path if dataset is a -c variant
        corruption_path = None
        if config.dataset == "cifar10c":
            corruption_path = "data/cifar10-c"
        elif config.dataset == "cifar100c":
            corruption_path = "data/cifar100-c"
        
        if corruption_path:
            python_cmd.extend(["--corruption-dataset-path", corruption_path])

        # Compute and add initial weights for NC-based weighting methods
        if config.weighting_method in ['nc4_rank', 'nc1_rank', 'pareto_rank']:
            # Load NC data to compute weights
            nc_data = self.load_nc_metrics(
                config.training_loss,
                config.dataset,
                config.model,
                config.seed
            )
            if nc_data is not None:
                weights = self.get_layer_weights(nc_data, config.selected_layers, config.weighting_method)
                if weights is not None:
                    weights_str = ",".join(f"{w:.10f}" for w in weights)
                    python_cmd.extend(["--initial-weights", weights_str])
                    logger.info(f"Computed {config.weighting_method} weights for {config.model}/{config.dataset}/seed{config.seed}: {weights}")
                else:
                    logger.warning(f"Failed to compute weights for {config.weighting_method}, will fall back to uniform")
            else:
                logger.warning(f"Failed to load NC data for computing weights, will fall back to uniform")
        
        safe_python_cmd = " ".join(shlex.quote(str(arg)) for arg in python_cmd)
        
        # Build wrap script
        wrap_script = f"""
echo '========================================'
echo 'SLURM JOB: NC-Based Multi-Layer Ensemble'
echo '========================================'
echo 'Job ID       : $SLURM_JOB_ID'
echo 'Host         : $(hostname)'
echo 'Start Time   : $(date)'
echo '----------------------------------------'
echo 'Config:'
echo '  Strategy   : {config.layer_selection_strategy}'
echo '  Weighting  : {config.weighting_method}'
echo '  Model      : {config.model}'
echo '  Dataset    : {config.dataset}'
echo '  Loss       : {config.training_loss}'
echo '  Seed       : {config.seed}'
echo '  Layers     : {layers_str}'
echo '----------------------------------------'
module load anaconda || echo "Anaconda module not found, assuming env is active."
source activate {shlex.quote(self.conda_env)} || echo "Conda env activation failed."
export PYTHONPATH='{shlex.quote(str(project_root))}:$PYTHONPATH'
export CUDA_LAUNCH_BLOCKING=1
cd {shlex.quote(str(project_root))}
echo 'CMD: {safe_python_cmd}'
{safe_python_cmd}
echo '----------------------------------------'
echo 'End Time     : $(date)'
echo 'Exit Code    : $?'
echo '========================================'
"""
        clean_wrap_script = "\n".join(line.lstrip() for line in wrap_script.strip().split('\n'))
        
        # Build sbatch command
        sbatch_cmd = [
            "sbatch",
            f"--partition={shlex.quote(self.slurm_partition)}",
            f"--job-name={shlex.quote(job_name)}",
            f"--output={shlex.quote(str(log_file))}",
            f"--time={shlex.quote(self.slurm_time)}",
            "--ntasks=1",
            f"--gpus={shlex.quote(self.slurm_gpus)}",
            f"--cpus-per-task={shlex.quote(str(self.slurm_cpus))}",
            f"--mem={shlex.quote(self.slurm_mem)}",
            f"--wrap={shlex.quote(clean_wrap_script)}",
        ]
        
        return " ".join(sbatch_cmd)
    
    def run_experiments(
        self,
        layer_strategies: List[str],
        weighting_methods: List[str],
        dry_run: bool = False,
        submit: bool = False
    ):
        """Generate and optionally submit experiments with deduplication."""
        
        logger.info("="*60)
        logger.info("NC-Based Multi-Layer Ensemble Experiments")
        logger.info("="*60)
        logger.info(f"Layer strategies: {layer_strategies}")
        logger.info(f"Weighting methods: {weighting_methods}")
        logger.info("="*60)
        
        # Generate experiments
        experiments = self.generate_experiments(layer_strategies, weighting_methods)
        if not experiments:
            logger.warning("No experiments generated. Exiting.")
            return
        
        logger.info(f"Generated {len(experiments)} experiment configurations")
        
        # Sort experiments for better cache locality (seed -> dataset -> model -> loss -> strategy -> method)
        logger.info("Sorting experiments for cache-friendly submission order...")
        experiments.sort(key=lambda e: (
            e.seed,
            e.dataset,
            e.model,
            e.training_loss,
            e.layer_selection_strategy,
            e.weighting_method
        ))
        logger.info("Experiments sorted by seed and dataset grouping.")
        
        # Deduplicate configurations
        logger.info("Detecting duplicate experiment configurations...")
        metrics_cache: Dict[Tuple[str, str, str, int], Optional[Dict]] = {}
        seen_configs: Dict[str, Tuple[NCEnsembleConfig, List[NCEnsembleConfig]]] = {}
        unique_experiments: List[NCEnsembleConfig] = []
        
        for config in experiments:
            weights = None
            
            if config.weighting_method in ['nc4_rank', 'nc1_rank', 'pareto_rank']:
                cache_key = (config.training_loss, config.dataset, config.model, config.seed)
                if cache_key not in metrics_cache:
                    nc_data = self.load_nc_metrics(*cache_key)
                    if nc_data is not None and (not nc_data.get('analysis')):
                        nc_data['analysis'] = self.compute_analysis_on_the_fly(nc_data)
                    metrics_cache[cache_key] = nc_data
                nc_data = metrics_cache[cache_key]
                if nc_data is not None:
                    weights = self.compute_nc_weights(
                        nc_data,
                        config.selected_layers,
                        config.weighting_method
                    )
            
            if weights is not None and self.weights_are_nearly_uniform(weights):
                weights = None
            
            signature = self.make_config_signature(
                config.selected_layers,
                weights,
                config.dataset,
                config.model,
                config.seed
            )
            
            if signature in seen_configs:
                canonical_config, duplicates = seen_configs[signature]
                duplicates.append(config)
                logger.debug(
                    "Duplicate detected: %s/%s matches %s/%s",
                    config.layer_selection_strategy,
                    config.weighting_method,
                    canonical_config.layer_selection_strategy,
                    canonical_config.weighting_method
                )
            else:
                seen_configs[signature] = (config, [])
                unique_experiments.append(config)
        
        n_original = len(experiments)
        n_unique = len(unique_experiments)
        n_duplicates = n_original - n_unique
        
        logger.info("="*60)
        logger.info("Deduplication report")
        logger.info("="*60)
        logger.info(f"Original configurations : {n_original}")
        logger.info(f"Unique configurations   : {n_unique}")
        logger.info(f"Duplicates skipped      : {n_duplicates} ({(100*n_duplicates/n_original):.1f}%)" if n_original else "Duplicates skipped      : 0 (0.0%)")
        logger.info("="*60)
        
        if n_duplicates > 0:
            logger.info("Sample duplicate groups:")
            sample_count = 0
            for canonical, duplicates in seen_configs.values():
                if duplicates and sample_count < 3:
                    logger.info("  Group %d:", sample_count + 1)
                    logger.info("    Canonical : %s/%s",
                                canonical.layer_selection_strategy,
                                canonical.weighting_method)
                    logger.info("    Layers    : %s", canonical.selected_layers)
                    for dup in duplicates[:3]:
                        logger.info("    Duplicate : %s/%s",
                                    dup.layer_selection_strategy,
                                    dup.weighting_method)
                    remaining = len(duplicates) - 3
                    if remaining > 0:
                        logger.info("    ... %d more duplicates", remaining)
                    sample_count += 1
            logger.info("")
        
        # Persist duplicate mapping for post-processing
        duplicate_mapping_file = self.output_dir / "duplicate_mappings.json"
        mapping: Dict[str, str] = {}
        for canonical, duplicates in seen_configs.values():
            if not duplicates:
                continue
            try:
                canonical_path = str(canonical.output_dir.relative_to(self.output_dir))
            except ValueError:
                canonical_path = str(canonical.output_dir)
            for dup in duplicates:
                try:
                    dup_path = str(dup.output_dir.relative_to(self.output_dir))
                except ValueError:
                    dup_path = str(dup.output_dir)
                mapping[dup_path] = canonical_path
        
        if mapping:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(duplicate_mapping_file, 'w') as f:
                json.dump(mapping, f, indent=2)
            logger.info("Duplicate mapping saved to %s", duplicate_mapping_file)
            logger.info("Use copy_duplicate_results.py after runs complete to populate duplicate paths.")
        
        experiments = unique_experiments
        
        pending = [e for e in experiments if not self.check_experiment_completed(e)]
        skipped = len(experiments) - len(pending)
        
        if skipped > 0:
            logger.info(f"Skipping {skipped} completed experiments")
        
        if not pending:
            logger.info("No pending experiments. Exiting.")
            return
        
        logger.info(f"Pending experiments: {len(pending)}")
        
        if dry_run:
            logger.info("\nDRY RUN - First 5 experiments:")
            for i, exp in enumerate(pending[:5], 1):
                logger.info(f"{i}. {exp.layer_selection_strategy}/{exp.weighting_method}: "
                          f"{exp.dataset}/{exp.model}/seed{exp.seed} "
                          f"(layers: {exp.selected_layers})")
            return
        
        if not submit:
            logger.info("\nUse --submit flag to submit jobs to SLURM")
            logger.info(f"Would submit {len(pending)} jobs")
            return
        
        # Submit jobs
        logger.info("\nSubmitting jobs to SLURM...")
        submitted = 0
        failed = 0
        job_ids = []
        
        for i, config in enumerate(pending, 1):
            logger.info(f"[{i}/{len(pending)}] Submitting {config.layer_selection_strategy}/{config.weighting_method}")
            
            try:
                cmd = self.generate_sbatch_command(config)
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=120
                )
                job_id = result.stdout.strip().split()[-1]
                logger.info(f"  Submitted: Job ID {job_id}")
                submitted += 1
                job_ids.append(job_id)
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"  Failed: {e}")
                failed += 1
        
        logger.info("\n" + "="*60)
        logger.info("SUBMISSION SUMMARY")
        logger.info("="*60)
        logger.info(f"Submitted: {submitted}")
        logger.info(f"Failed: {failed}")
        if submitted > 0:
            logger.info(f"\nJob IDs: {', '.join(job_ids[:10])}{'...' if len(job_ids) > 10 else ''}")
            logger.info(f"\nMonitor jobs: squeue -u $USER")
            logger.info(f"View logs: slurm_logs/nc_ensemble/")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Run NC-based multi-layer ensemble calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['local', 'slurm'],
        default='slurm',
        help='Execution mode'
    )
    
    # Experiment configuration
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['cifar10', 'cifar100', 'cifar100c', 'cifar10c'],
        help='Datasets to process'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['resnet18', 'resnet50', 'densenet121'],
        help='Models to process'
    )
    parser.add_argument(
        '--training_losses',
        nargs='+',
        default=['baseline_cross_entropy'],
        help='Training losses to process'
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        help='Seeds to process'
    )
    
    # Layer selection strategies
    parser.add_argument(
        '--layer_strategies',
        nargs='+',
        default=['pareto_top5', 'nc4_top5', 'psc','nc1_top3', 'nc4_top3'],
        choices=['pareto_top5', 'pareto_top3', 'nc4_top5', 'nc4_top3', 'nc1_top5', 'nc1_top3', 'strict', 'psc', 'best_nc4', 'single_layer'],
        help='Layer selection strategies to try'
    )
    
    # Weighting methods
    parser.add_argument(
        '--weighting_methods',
        nargs='+',
        default=['uniform', 'learned', 'nc4_rank', 'nc1_rank'],
        choices=['uniform', 'learned', 'nc4_rank', 'nc1_rank', 'pareto_rank'],
        help='Weighting methods to try'
    )
    
    # Paths
    parser.add_argument(
        '--nc_metrics_dir',
        type=str,
        default='nc_metrics_results',
        help='Directory containing NC metrics results'
    )
    parser.add_argument(
        '--checkpoint_base_dir',
        type=str,
        default='aaai_full_experiments/results/baseline',
        help='Base directory with trained models'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='nc_ensemble_results',
        help='Output directory for ensemble results'
    )
    parser.add_argument(
        '--ensemble_script',
        type=str,
        default='Experiments/multi_layer_ensemble.py',
        help='Path to multi-layer ensemble script'
    )
    
    # Execution options
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be done without executing'
    )
    parser.add_argument(
        '--submit',
        action='store_true',
        help='Submit jobs to SLURM (only for --mode slurm)'
    )
    parser.add_argument(
        '--conda_env',
        type=str,
        default='tamar_n_env',
        help='Conda environment name'
    )
    
    args = parser.parse_args()
    
    # Create job manager
    manager = NCEnsembleJobManager(
        datasets=args.datasets,
        models=args.models,
        training_losses=args.training_losses,
        seeds=args.seeds,
        nc_metrics_dir=args.nc_metrics_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        output_dir=args.output_dir,
        ensemble_script=args.ensemble_script,
        conda_env=args.conda_env
    )
    
    # Run experiments
    manager.run_experiments(
        layer_strategies=args.layer_strategies,
        weighting_methods=args.weighting_methods,
        dry_run=args.dry_run,
        submit=args.submit
    )


if __name__ == '__main__':
    main()