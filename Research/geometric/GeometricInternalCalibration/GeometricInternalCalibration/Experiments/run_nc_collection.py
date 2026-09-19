"""
Python Job Manager for NC Metrics Collection - Final Version

Uses exact conda activation from working scripts with explicit PATH management.
"""

import argparse
import json
import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import itertools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
from utils.logging_config import get_logger
logger = get_logger(__name__)


class NCMetricsJobManager:
    """Manages NC metrics collection jobs across multiple configurations"""
    
    def __init__(
        self,
        datasets: List[str],
        models: List[str],
        training_losses: List[str],
        seeds: List[int],
        models_dir: str = "aaai_full_experiments/results",
        output_dir: str = "nc_metrics_results",
        batch_size: int = 256,
        use_validation: bool = False,
        conda_env: str = "tamar_n_env"
    ):
        self.datasets = datasets
        self.models = models
        self.training_losses = training_losses
        self.seeds = seeds
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.use_validation = use_validation
        self.conda_env = conda_env
        
        # SLURM settings
        self.slurm_partition = "gpu_partition"
        self.slurm_time = "02:00:00"
        self.slurm_mem = "32G"
        self.slurm_cpus = 4
        self.slurm_gpus = "rtx_4090:1"
    
    def get_all_jobs(self) -> List[Dict[str, str]]:
        """Generate all job configurations"""
        jobs = []
        for dataset, model, loss, seed in itertools.product(
            self.datasets, self.models, self.training_losses, self.seeds
        ):
            jobs.append({
                'dataset': dataset,
                'model': model,
                'training_loss': loss,
                'seed': seed
            })
        return jobs
    
    def check_model_exists(self, job: Dict[str, str]) -> bool:
        """Check if the trained model exists"""
        model_path = (
            self.models_dir / 
            "baseline" /
            job['training_loss'] / 
            job['dataset'] / 
            job['model'] /
            f"seed{job['seed']}" /
            f"{job['training_loss']}_{job['dataset']}_{job['model']}_seed{job['seed']}" /
            "best_model.pth"
        )
        exists = model_path.exists()
        if not exists:
            logger.debug(f"Model not found: {model_path}")
        return exists
    
    def check_results_exist(self, job: Dict[str, str]) -> bool:
        """Check if results already exist"""
        split = 'validation' if self.use_validation else 'test'
        results_path = (
            self.output_dir /
            job['training_loss'] /
            job['dataset'] /
            job['model'] /
            f"seed{job['seed']}" /
            f"nc_metrics_{split}.json"
        )
        return results_path.exists()
    
    def get_job_status(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Categorize all jobs into:
        - to_run: Jobs that need to be executed
        - skipped_no_model: Jobs skipped because model doesn't exist
        - skipped_exists: Jobs skipped because results already exist
        """
        all_jobs = self.get_all_jobs()
        to_run = []
        skipped_no_model = []
        skipped_exists = []
        
        for job in all_jobs:
            if not self.check_model_exists(job):
                skipped_no_model.append(job)
            elif self.check_results_exist(job):
                skipped_exists.append(job)
            else:
                to_run.append(job)
        
        return to_run, skipped_no_model, skipped_exists
    
    def format_job_name(self, job: Dict[str, str]) -> str:
        """Create a readable job name"""
        return f"{job['dataset']}-{job['model']}-{job['training_loss']}-seed{job['seed']}"
    
    def run_job_local(self, job: Dict[str, str]) -> bool:
        """Run a single job locally"""
        cmd = [
            sys.executable,
            "Experiments/collect_nc_metrics.py",
            "--dataset", job['dataset'],
            "--model", job['model'],
            "--training_loss", job['training_loss'],
            "--seed", str(job['seed']),
            "--models_dir", str(self.models_dir),
            "--output_dir", str(self.output_dir),
            "--batch_size", str(self.batch_size),
        ]
        
        if self.use_validation:
            cmd.append("--use_validation")
        
        logger.info(f"Running: {self.format_job_name(job)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Success: {self.format_job_name(job)}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed: {self.format_job_name(job)}")
            logger.error(f"Error: {e.stderr}")
            return False
    
    def generate_sbatch_command(self, job: Dict[str, str]) -> str:
        """Generate sbatch command string - exact copy from working scripts"""
        job_name = self.format_job_name(job)
        
        # Get current working directory
        project_root = Path.cwd()
        
        # Create logs directory
        log_dir = Path("slurm_logs") / "nc_collection"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{job_name}_%j.log"
        
        # Build Python command WITH FULL PATH (like compression_sweep does)
        python_cmd = [
            sys.executable,  # FULL PATH TO PYTHON - this is the key!
            "Experiments/collect_nc_metrics.py",
            "--dataset", job['dataset'],
            "--model", job['model'],
            "--training_loss", job['training_loss'],
            "--seed", str(job['seed']),
            "--models_dir", str(self.models_dir),
            "--output_dir", str(self.output_dir),
            "--batch_size", str(self.batch_size),
        ]
        
        if self.use_validation:
            python_cmd.append("--use_validation")
        
        safe_python_cmd = " ".join(shlex.quote(str(arg)) for arg in python_cmd)
        
        # Build wrap script - execute the full command including Python path
        wrap_script = f"""
echo '========================================'
echo 'SLURM JOB: NC Metrics Collection'
echo '========================================'
echo 'Job ID       : $SLURM_JOB_ID'
echo 'Host         : $(hostname)'
echo 'Start Time   : $(date)'
echo '----------------------------------------'
echo 'Config:'
echo '  Dataset    : {job["dataset"]}'
echo '  Model      : {job["model"]}'
echo '  Loss       : {job["training_loss"]}'
echo '  Seed       : {job["seed"]}'
echo '----------------------------------------'
module load anaconda || echo "Anaconda module not found, assuming env is active."
source activate {shlex.quote(self.conda_env)} || echo "Conda env '{shlex.quote(self.conda_env)}' activation failed."
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
        # Clean up indentation
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
    
    def submit_slurm_job(self, job: Dict[str, str]) -> Tuple[bool, str]:
        """Submit a SLURM job using sbatch --wrap"""
        job_name = self.format_job_name(job)
        
        try:
            cmd = self.generate_sbatch_command(job)
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
                timeout=120
            )
            job_id = result.stdout.strip().split()[-1]
            logger.info(f"Submitted: {job_name} (Job ID: {job_id})")
            return True, job_id
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout submitting: {job_name}")
            logger.error("SLURM scheduler may be overloaded")
            return False, ""
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit: {job_name}")
            logger.error(f"Error: {e.stderr}")
            return False, ""
        except Exception as e:
            logger.error(f"Unexpected error: {job_name}")
            logger.error(f"Error: {str(e)}")
            return False, ""
    
    def run_local(self, dry_run: bool = False, parallel: bool = False):
        """Run all jobs locally"""
        to_run, skipped_no_model, skipped_exists = self.get_job_status()
        
        logger.info("="*60)
        logger.info("NC Metrics Collection - Local Execution")
        logger.info("="*60)
        logger.info(f"Total jobs: {len(self.get_all_jobs())}")
        logger.info(f"To run: {len(to_run)}")
        logger.info(f"Skipped (no model): {len(skipped_no_model)}")
        logger.info(f"Skipped (exists): {len(skipped_exists)}")
        logger.info("="*60)
        
        if dry_run:
            logger.info("\nDRY RUN - Jobs that would be executed:")
            for i, job in enumerate(to_run, 1):
                logger.info(f"{i}. {self.format_job_name(job)}")
            return
        
        if parallel:
            logger.warning("Parallel execution not yet implemented. Running sequentially.")
        
        # Run jobs sequentially
        success_count = 0
        fail_count = 0
        
        for i, job in enumerate(to_run, 1):
            logger.info(f"\n[{i}/{len(to_run)}] {self.format_job_name(job)}")
            if self.run_job_local(job):
                success_count += 1
            else:
                fail_count += 1
        
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {fail_count}")
        logger.info(f"Skipped: {len(skipped_no_model) + len(skipped_exists)}")
        logger.info("="*60)
    
    def show_missing_models(self):
        """Show which models are missing (for debugging)"""
        _, skipped_no_model, _ = self.get_job_status()
        
        if not skipped_no_model:
            logger.info("All models found!")
            return
        
        logger.info(f"\nMissing {len(skipped_no_model)} models:")
        logger.info("="*60)
        
        for job in skipped_no_model[:20]:
            model_path = (
                self.models_dir / 
                "baseline" /
                job['training_loss'] / 
                job['dataset'] / 
                job['model'] /
                f"seed{job['seed']}" /
                f"{job['training_loss']}_{job['dataset']}_{job['model']}_seed{job['seed']}" /
                "best_model.pth"
            )
            logger.info(f"  {self.format_job_name(job)}")
            logger.info(f"    Expected: {model_path}")
        
        if len(skipped_no_model) > 20:
            logger.info(f"  ... and {len(skipped_no_model) - 20} more")
        
        logger.info("="*60)
        logger.info("\nTip: Check your --models_dir path and model naming convention")
    
    def run_slurm(self, submit: bool = False, dry_run: bool = False):
        """Submit SLURM jobs using sbatch --wrap"""
        to_run, skipped_no_model, skipped_exists = self.get_job_status()
        
        logger.info("="*60)
        logger.info("NC Metrics Collection - SLURM Job Submission")
        logger.info("="*60)
        logger.info(f"Total jobs: {len(self.get_all_jobs())}")
        logger.info(f"To run: {len(to_run)}")
        logger.info(f"Skipped (no model): {len(skipped_no_model)}")
        logger.info(f"Skipped (exists): {len(skipped_exists)}")
        logger.info("="*60)
        
        if dry_run:
            logger.info("\nDRY RUN - Jobs that would be submitted:")
            for i, job in enumerate(to_run, 1):
                logger.info(f"{i}. {self.format_job_name(job)}")
            logger.info("\nExample sbatch command:")
            if to_run:
                print(self.generate_sbatch_command(to_run[0]))
            return
        
        if not submit:
            logger.info("\nUse --submit flag to actually submit jobs to SLURM")
            logger.info(f"Would submit {len(to_run)} jobs")
            return
        
        # Submit jobs
        logger.info("\nSubmitting jobs to SLURM...")
        submitted = 0
        failed = 0
        job_ids = []
        
        for i, job in enumerate(to_run, 1):
            logger.info(f"[{i}/{len(to_run)}] {self.format_job_name(job)}")
            
            try:
                success, job_id = self.submit_slurm_job(job)
                if success:
                    submitted += 1
                    job_ids.append(job_id)
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Unexpected error submitting {self.format_job_name(job)}: {e}")
                failed += 1
            
            # Be nice to the scheduler
            time.sleep(0.1)
        
        logger.info("\n" + "="*60)
        logger.info("SUBMISSION SUMMARY")
        logger.info("="*60)
        logger.info(f"Submitted: {submitted}")
        logger.info(f"Failed: {failed}")
        if submitted > 0:
            logger.info(f"\nJob IDs: {', '.join(job_ids[:10])}{'...' if len(job_ids) > 10 else ''}")
            logger.info(f"\nMonitor jobs with: squeue -u $USER")
            logger.info(f"View logs in: slurm_logs/nc_collection/")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Manage NC metrics collection jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check which models exist
  python run_nc_collection.py --mode check-models
  
  # Dry run to see what would be submitted
  python run_nc_collection.py --mode local --dry_run
  
  # Submit to SLURM
  python run_nc_collection.py --mode slurm --submit
        """
    )
    
    # Execution mode
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['local', 'slurm', 'check-models'],
        default='local',
        help='Execution mode: local, slurm, or check-models'
    )
    
    # Job configuration
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['cifar10', 'cifar100'],
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
    
    # Paths
    parser.add_argument(
        '--models_dir',
        type=str,
        default='aaai_full_experiments/results',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='nc_metrics_results',
        help='Output directory for results'
    )
    
    # Options
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for feature extraction'
    )
    parser.add_argument(
        '--use_validation',
        action='store_true',
        help='Use validation split instead of test'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be done without executing'
    )
    parser.add_argument(
        '--conda_env',
        type=str,
        default='tamar_n_env',
        help='Conda environment name'
    )
    
    # SLURM specific
    parser.add_argument(
        '--submit',
        action='store_true',
        help='Submit SLURM jobs immediately (only for --mode slurm)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run jobs in parallel (only for --mode local, not yet implemented)'
    )
    
    args = parser.parse_args()
    
    # Create job manager
    manager = NCMetricsJobManager(
        datasets=args.datasets,
        models=args.models,
        training_losses=args.training_losses,
        seeds=args.seeds,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_validation=args.use_validation,
        conda_env=args.conda_env
    )
    
    # Execute based on mode
    if args.mode == 'check-models':
        manager.show_missing_models()
    elif args.mode == 'local':
        manager.run_local(dry_run=args.dry_run, parallel=args.parallel)
    elif args.mode == 'slurm':
        manager.run_slurm(submit=args.submit, dry_run=args.dry_run)


if __name__ == '__main__':
    main()