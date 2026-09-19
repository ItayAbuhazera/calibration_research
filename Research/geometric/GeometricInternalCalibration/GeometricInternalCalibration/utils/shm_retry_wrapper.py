"""
Robust Shared Memory Retry System for PyTorch

This module provides comprehensive shared memory management and retry mechanisms
specifically designed for PyTorch DataLoaders and memory-intensive operations.

Features:
1. SharedMemoryMonitor - Real-time monitoring and cleanup of /dev/shm
2. @retry_on_memory_error - Decorator with exponential backoff for memory errors
3. MemorySafeDataLoader - DataLoader wrapper with automatic worker reduction
4. Utility functions for memory management and PyTorch configuration

Designed for robust operation in multi-worker PyTorch environments with
shared memory constraints.
"""

import logging
import os
import time
import random
import functools
import shutil
import glob
import psutil
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Union, Tuple, Iterator
import torch
from torch.utils.data import DataLoader

# Import additional modules for enhanced functionality
import gc
import tracemalloc
import hashlib

from utils.logging_config import get_logger
logger = get_logger(__name__)


def _is_memory_error(error: Exception) -> bool:
    """
    Enhanced check if error is related to shared memory issues.
    
    Includes multiprocessing-specific patterns and error chain checking.
    
    Args:
        error: Exception to check
        
    Returns:
        True if error is memory-related
    """
    error_str = str(error).lower()
    
    # Enhanced error patterns including multiprocessing-specific issues
    error_patterns = [
        "shared memory", 
        "no space left on device", 
        "bus error",
        "dataloader worker", 
        "cannot allocate memory", 
        "shm_open failed",
        "unable to write to file </torch_",  # Specific error pattern from user
        "killed by signal",  # Worker killed
        "out of shared memory",
        "raise your shared memory limit",
        "worker process died",
        "semaphore failure",
        "too many open files",
        "resource temporarily unavailable",
        "broken pipe in multiprocessing",
        "fork: cannot allocate memory"
    ]
    
    # Check the exception chain for nested errors
    if hasattr(error, '__cause__') and error.__cause__:
        error_str += " " + str(error.__cause__).lower()
    
    if hasattr(error, '__context__') and error.__context__:
        error_str += " " + str(error.__context__).lower()
    
    # Also check exception type names
    error_type = type(error).__name__.lower()
    type_patterns = [
        "runtimeerror",
        "oserror", 
        "memoryerror",
        "brokenpipeerror"
    ]
    
    pattern_match = any(pattern in error_str for pattern in error_patterns)
    type_match = any(pattern in error_type for pattern in type_patterns) and ("memory" in error_str or "worker" in error_str)
    
    return pattern_match or type_match


class SharedMemoryMonitor:
    """
    Enhanced real-time shared memory monitoring with pre-emptive checks.
    """
    
    def __init__(self, log_level: str = "INFO", enable_profiling: bool = False):
        """
        Initialize SharedMemoryMonitor with enhanced capabilities.
        
        Args:
            log_level: Logging level for monitoring messages
            enable_profiling: Enable memory profiling with tracemalloc
        """
        self.log_level = getattr(logging, log_level.upper())
        self.shm_path = Path("/dev/shm")
        self.enable_profiling = enable_profiling
        self.memory_history = []
        
        if self.enable_profiling:
            tracemalloc.start()
            logger.info("🔬 Memory profiling enabled")
        
        if not self.shm_path.exists():
            logger.warning("⚠️ /dev/shm not available - shared memory monitoring disabled")
    
    def get_memory_profile(self) -> Dict[str, float]:
        """
        Get detailed memory profile if profiling is enabled.
        
        Returns:
            Dictionary with memory statistics in GB
        """
        profile = {}
        
        if self.enable_profiling and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            profile.update({
                'tracemalloc_current_gb': current / 1e9,
                'tracemalloc_peak_gb': peak / 1e9
            })
        
        # System memory info
        try:
            mem_info = psutil.virtual_memory()
            profile.update({
                'system_total_gb': mem_info.total / 1e9,
                'system_available_gb': mem_info.available / 1e9,
                'system_used_gb': mem_info.used / 1e9,
                'system_percent': mem_info.percent
            })
        except Exception as e:
            logger.debug(f"Could not get system memory info: {e}")
        
        # GPU memory if available
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    profile[f'gpu_{i}_allocated_gb'] = allocated
                    profile[f'gpu_{i}_reserved_gb'] = reserved
            except Exception as e:
                logger.debug(f"Could not get GPU memory info: {e}")
        
        return profile
    
    def log_memory_statistics(self, context: str = ""):
        """Enhanced memory logging with profiling support."""
        super().log_memory_statistics(context)
        
        if self.enable_profiling:
            profile = self.get_memory_profile()
            self.memory_history.append({
                'context': context,
                'timestamp': time.time(),
                'profile': profile
            })
            
            if profile:
                logger.info(f"🔬 Memory profile ({context}):")
                for key, value in profile.items():
                    if 'gb' in key:
                        logger.info(f"   {key}: {value:.2f}GB")
                    else:
                        logger.info(f"   {key}: {value:.1f}%")
    
    def get_usage_percentage(self) -> float:
        """
        Get current shared memory usage percentage.
        
        Returns:
            Usage percentage (0.0 to 100.0)
        """
        try:
            if not self.shm_path.exists():
                return 0.0
                
            shm_stats = shutil.disk_usage(self.shm_path)
            total_bytes = shm_stats.total
            used_bytes = total_bytes - shm_stats.free
            usage_percent = (used_bytes / total_bytes) * 100.0
            
            logger.log(self.log_level, 
                      f"📊 Shared memory usage: {usage_percent:.1f}% "
                      f"({used_bytes / (1024**2):.1f}MB / {total_bytes / (1024**2):.1f}MB)")
            
            return usage_percent
            
        except Exception as e:
            logger.error(f"❌ Failed to get shared memory usage: {e}")
            return 0.0
    
    def clean_torch_files(self) -> Dict[str, int]:
        """
        Clean PyTorch-related files from /dev/shm.
        
        Returns:
            Dictionary with cleanup statistics:
            - files_removed: Number of files removed
            - space_freed_mb: Space freed in MB
            - files_failed: Number of files that failed to remove
        """
        files_removed = 0
        space_freed_bytes = 0
        files_failed = 0
        
        logger.info("🧹 Cleaning PyTorch shared memory files...")
        
        try:
            if not self.shm_path.exists():
                logger.warning("⚠️ /dev/shm not available")
                return {'files_removed': 0, 'space_freed_mb': 0.0, 'files_failed': 0}
            
            # Find PyTorch-specific files
            torch_patterns = [
                "torch_*",
                "*_torch_*", 
                "pytorch_*",
                "*_pytorch_*"
            ]
            
            torch_files = []
            for pattern in torch_patterns:
                torch_files.extend(self.shm_path.glob(pattern))
            
            # Also find files with numeric names (likely PyTorch worker files)
            for item in self.shm_path.iterdir():
                if item.is_file():
                    # Check for numeric patterns or large files that might be PyTorch related
                    if (item.name.isdigit() or 
                        any(char.isdigit() for char in item.name) and 
                        item.stat().st_size > 1024):
                        torch_files.append(item)
            
            # Remove duplicates
            torch_files = list(set(torch_files))
            
            logger.info(f"   Found {len(torch_files)} potential PyTorch files")
            
            for file_path in torch_files:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    files_removed += 1
                    space_freed_bytes += file_size
                    logger.debug(f"   🗑️ Removed: {file_path.name} ({file_size} bytes)")
                    
                except (OSError, PermissionError) as e:
                    files_failed += 1
                    logger.debug(f"   ❌ Failed to remove {file_path.name}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Torch file cleanup failed: {e}")
            
        space_freed_mb = space_freed_bytes / (1024 * 1024)
        
        logger.info(f"✅ Cleanup complete: {files_removed} files removed, "
                   f"{space_freed_mb:.2f}MB freed, {files_failed} failed")
        
        return {
            'files_removed': files_removed,
            'space_freed_mb': round(space_freed_mb, 2),
            'files_failed': files_failed
        }
    
    def wait_for_memory(self, threshold: float = 80.0, timeout: float = 300.0) -> bool:
        """
        Wait for shared memory usage to drop below threshold.
        
        Args:
            threshold: Maximum acceptable usage percentage (default: 80%)
            timeout: Maximum time to wait in seconds (default: 300s)
            
        Returns:
            True if memory is available, False if timeout reached
        """
        start_time = time.time()
        check_interval = 5.0  # Check every 5 seconds
        
        logger.info(f"⏳ Waiting for shared memory usage < {threshold}% (timeout: {timeout}s)")
        
        while time.time() - start_time < timeout:
            current_usage = self.get_usage_percentage()
            
            if current_usage < threshold:
                elapsed = time.time() - start_time
                logger.info(f"✅ Memory available: {current_usage:.1f}% < {threshold}% "
                           f"(waited {elapsed:.1f}s)")
                return True
            
            # Log progress every 30 seconds
            elapsed = time.time() - start_time
            if elapsed % 30 < check_interval:
                remaining = timeout - elapsed
                logger.info(f"   Still waiting: {current_usage:.1f}% usage "
                           f"(remaining: {remaining:.0f}s)")
            
            time.sleep(check_interval)
        
        # Timeout reached
        final_usage = self.get_usage_percentage()
        logger.warning(f"⏰ Timeout reached: {final_usage:.1f}% usage still > {threshold}%")
        return False

    def check_memory_threshold(self, threshold: float = 85.0) -> bool:
        """
        Pre-emptive memory check with threshold.
        
        Args:
            threshold: Memory usage threshold (0-100)
            
        Returns:
            True if memory usage is below threshold
        """
        current_usage = self.get_usage_percentage()
        
        if current_usage > threshold:
            logger.warning(f"⚠️ Memory usage ({current_usage:.1f}%) exceeds threshold ({threshold}%)")
            return False
        
        return True
    
    def emergency_cleanup(self) -> Dict[str, Any]:
        """
        Emergency memory cleanup for critical situations.
        
        Returns:
            Cleanup statistics
        """
        logger.warning("🚨 Performing emergency memory cleanup")
        
        # Clean shared memory files
        shm_stats = self.clean_torch_files()
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # System memory cleanup attempts
        try:
            # Force Python garbage collection multiple times
            for _ in range(3):
                collected = gc.collect()
                if collected == 0:
                    break
        except Exception as e:
            logger.warning(f"Garbage collection issue: {e}")
        
        # Updated usage after cleanup
        final_usage = self.get_usage_percentage()
        
        cleanup_stats = {
            **shm_stats,
            'gc_collections': 3,
            'final_memory_usage': final_usage
        }
        
        logger.warning(f"🧹 Emergency cleanup complete: {cleanup_stats}")
        return cleanup_stats


def retry_on_memory_error(
    max_retries: int = 5,
    initial_delay: float = 30.0,
    max_delay: float = 300.0,
    backoff_factor: float = 2.0
) -> Callable:
    """Enhanced decorator with improved error detection."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            monitor = SharedMemoryMonitor()
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    # Log attempt
                    if attempt == 0:
                        logger.debug(f"🎯 Executing {func.__name__}")
                    else:
                        logger.info(f"🔄 Retry attempt {attempt}/{max_retries} for {func.__name__}")
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(f"✅ {func.__name__} succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except Exception as e:
                    # Use enhanced error detection
                    if not _is_memory_error(e):
                        # Not a memory error, re-raise immediately
                        logger.error(f"❌ {func.__name__} failed with non-memory error: {e}")
                        raise
                    
                    last_exception = e
                    
                    if attempt >= max_retries:
                        logger.error(f"❌ {func.__name__} failed after {max_retries} retries")
                        logger.error(f"   Final error: {e}")
                        break
                    
                    # Log the error and plan for retry
                    logger.warning(f"⚠️ Memory error in {func.__name__}: {e}")
                    
                    # Enhanced cleanup based on attempt number
                    if attempt >= 2:  # Third attempt or later - emergency mode
                        logger.warning("🚨 Entering emergency recovery mode")
                        cleanup_stats = monitor.emergency_cleanup()
                    else:
                        cleanup_stats = monitor.clean_torch_files()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Calculate delay with exponential backoff
                    delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
                    
                    logger.info(f"⏳ Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}")
                    logger.info(f"   Cleanup: {cleanup_stats.get('files_removed', 0)} files, "
                               f"{cleanup_stats.get('space_freed_mb', 0):.1f}MB freed")
                    
                    # Log current memory status
                    usage = monitor.get_usage_percentage()
                    logger.info(f"   Current shared memory usage: {usage:.1f}%")
                    
                    time.sleep(delay)
            
            # All retries exhausted
            if last_exception:
                logger.error(f"❌ All {max_retries} retries exhausted for {func.__name__}")
                raise last_exception
            else:
                logger.error(f"❌ Unexpected error in retry logic for {func.__name__}")
                raise RuntimeError("Retry logic failed unexpectedly")
                
        return wrapper
    return decorator


class MemorySafeDataLoader:
    """
    Enhanced memory-safe wrapper with worker process error recovery.
    """
    
    def __init__(self, *args, **kwargs):
        # Call parent constructor
        super().__init__(*args, **kwargs)
        
        # Enhanced tracking
        self.worker_restart_count = 0
        self.max_worker_restarts = 3
        self.last_error = None
        self.error_history = []
    
    def _pre_flight_memory_check(self) -> bool:
        """
        Pre-emptive memory check before DataLoader creation.
        
        Returns:
            True if memory is safe for DataLoader creation
        """
        logger.debug("🔍 Pre-flight memory check for DataLoader creation")
        
        mem_usage = self.monitor.get_usage_percentage()
        
        if mem_usage > 85.0:
            logger.warning(f"⚠️ High memory usage ({mem_usage:.1f}%) before DataLoader creation")
            
            # Aggressive cleanup
            cleanup_stats = self.monitor.emergency_cleanup()
            
            # Wait for memory to stabilize
            if not self.monitor.wait_for_memory(threshold=70.0, timeout=60.0):
                logger.error(f"❌ Memory still too high after cleanup")
                return False
        
        logger.debug(f"✅ Pre-flight check passed: {mem_usage:.1f}% usage")
        return True
    
    @retry_on_memory_error(max_retries=2, initial_delay=10.0, max_delay=60.0)
    def _create_dataloader(self, num_workers: int) -> DataLoader:
        """Enhanced DataLoader creation with pre-flight checks."""
        
        # Pre-flight memory check
        if not self._pre_flight_memory_check():
            raise RuntimeError("Pre-flight memory check failed - insufficient memory for DataLoader creation")
        
        # Call original implementation
        return super()._create_dataloader(num_workers)
    
    def __iter__(self) -> Iterator:
        """
        Enhanced iteration with worker process error recovery.
        
        Returns:
            Iterator over batches with automatic worker restart
        """
        if self.dataloader is None:
            self.create_dataloader()
        
        while self.worker_restart_count < self.max_worker_restarts:
            try:
                # Reset for fresh iteration
                dataloader_iter = iter(self.dataloader)
                
                # Iterate through all batches
                for batch in dataloader_iter:
                    yield batch
                
                # Successful completion
                break
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for worker-specific errors
                if any(pattern in error_str for pattern in [
                    "dataloader worker", "killed by signal", "worker process died",
                    "broken pipe", "connection reset"
                ]):
                    self.worker_restart_count += 1
                    self.last_error = e
                    self.error_history.append({
                        'error': str(e),
                        'timestamp': time.time(),
                        'restart_attempt': self.worker_restart_count
                    })
                    
                    logger.warning(f"🔄 Worker process error detected, restart attempt {self.worker_restart_count}/{self.max_worker_restarts}")
                    logger.warning(f"   Error: {e}")
                    
                    if self.worker_restart_count < self.max_worker_restarts:
                        # More aggressive cleanup
                        self.monitor.emergency_cleanup()
                        
                        # Reduce workers for next attempt
                        self.current_workers = max(0, self.current_workers // 2)
                        logger.info(f"   Reducing workers to {self.current_workers}")
                        
                        # Wait before retry
                        time.sleep(5.0 * self.worker_restart_count)
                        
                        # Recreate DataLoader
                        try:
                            self.dataloader = self._create_dataloader(self.current_workers)
                        except Exception as create_error:
                            logger.error(f"Failed to recreate DataLoader: {create_error}")
                            raise
                        
                        continue  # Try again with new DataLoader
                    else:
                        logger.error(f"❌ Max worker restarts ({self.max_worker_restarts}) exceeded")
                        raise RuntimeError(f"DataLoader failed after {self.max_worker_restarts} worker restarts. Last error: {e}")
                
                # For non-worker errors, check if it's a memory error
                elif _is_memory_error(e):
                    logger.warning(f"⚠️ Memory error in DataLoader iteration: {e}")
                    
                    # Clean memory and retry once
                    self.monitor.emergency_cleanup()
                    time.sleep(2.0)
                    
                    # Try to recreate with minimal resources
                    self.current_workers = 0
                    self.dataloader = self._create_dataloader(0)
                    continue
                
                else:
                    # Non-recoverable error
                    logger.error(f"❌ Non-recoverable DataLoader error: {e}")
                    raise
        
        # If we exit the loop without breaking, all restart attempts failed
        if self.worker_restart_count >= self.max_worker_restarts:
            raise RuntimeError(f"DataLoader iteration failed after {self.max_worker_restarts} restart attempts")


# Utility functions

def setup_memory_safe_pytorch():
    """
    Configure PyTorch for memory-safe operation.
    
    Sets various PyTorch environment variables and configurations
    to reduce memory usage and improve stability.
    """
    logger.info("🔧 Setting up memory-safe PyTorch configuration...")
    
    # Set environment variables for memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:False'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Set multiprocessing sharing strategy
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Configure CUDA settings if available
    if torch.cuda.is_available():
        # Enable memory management optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False  # Consistent memory usage
        
        # Clear cache
        torch.cuda.empty_cache()
        
        logger.info("   ✅ CUDA memory management configured")
    
    # Set number of threads for CPU operations
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(min(4, os.cpu_count()))
        logger.info(f"   ✅ Set PyTorch threads to {torch.get_num_threads()}")
    
    logger.info("✅ Memory-safe PyTorch configuration complete")


def get_adaptive_batch_size(base_size: int, memory_usage: float) -> int:
    """
    Calculate adaptive batch size based on memory usage.
    
    Args:
        base_size: Base batch size
        memory_usage: Current memory usage percentage (0-100)
        
    Returns:
        Adjusted batch size
    """
    if memory_usage < 50:
        # Low memory usage - can use full batch size
        return base_size
    elif memory_usage < 70:
        # Moderate usage - reduce by 25%
        return max(1, int(base_size * 0.75))
    elif memory_usage < 85:
        # High usage - reduce by 50%
        return max(1, int(base_size * 0.5))
    else:
        # Very high usage - reduce to minimum
        return max(1, int(base_size * 0.25))


def cleanup_all_torch_memory():
    """
    Force cleanup of all PyTorch shared memory.
    
    Returns:
        Cleanup statistics
    """
    logger.info("🧹 Forcing cleanup of all PyTorch memory...")
    
    monitor = SharedMemoryMonitor()
    
    # Clean shared memory files
    shm_stats = monitor.clean_torch_files()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info("   ✅ Cleared CUDA cache and IPC")
    
    # Force garbage collection
    import gc
    gc.collect()
    
    logger.info(f"✅ Memory cleanup complete: {shm_stats['files_removed']} files removed, "
               f"{shm_stats['space_freed_mb']:.1f}MB freed")
    
    return shm_stats


def enable_emergency_recovery_mode():
    """
    Enable ultra-safe mode for critical experiments.
    
    This function sets the most conservative PyTorch settings
    to minimize memory usage and multiprocessing issues.
    """
    logger.warning("🚨 Enabling emergency recovery mode - ultra-conservative settings")
    
    # Disable CuDNN for deterministic behavior and lower memory
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Limit CPU threads
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    # Force file system sharing strategy
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Set minimal CUDA memory allocation
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,expandable_segments:False'
    
    logger.warning("   ⚙️ CuDNN disabled")
    logger.warning("   ⚙️ CPU threads limited to 1")
    logger.warning("   ⚙️ File system sharing strategy enabled")
    logger.warning("   ⚙️ Minimal CUDA memory allocation")


def create_memory_safe_hash(data) -> str:
    """
    Create a memory-safe hash for caching purposes.
    
    Args:
        data: Data to hash (should be serializable)
        
    Returns:
        Hexadecimal hash string
    """
    try:
        # Convert data to string representation
        data_str = str(data)
        # Create hash
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    except Exception as e:
        logger.warning(f"Failed to create hash: {e}")
        return f"fallback_{int(time.time())}"


# Example usage and testing functions

def example_memory_safe_dataloader():
    """Example demonstrating MemorySafeDataLoader usage."""
    
    class MockDataset:
        def __len__(self):
            return 1000
        def __getitem__(self, idx):
            return torch.randn(3, 32, 32), idx % 10
    
    logger.info("🧪 Testing MemorySafeDataLoader...")
    
    try:
        # Setup memory-safe configuration
        setup_memory_safe_pytorch()
        
        dataset = MockDataset()
        
        # Create memory-safe DataLoader
        dataloader = MemorySafeDataLoader(
            dataset=dataset,
            batch_size=64,
            num_workers=0,
            shuffle=True,
            pin_memory=True
        )
        
        # Test creation
        dl = dataloader.create_dataloader()
        
        # Test iteration
        for i, (data, target) in enumerate(dataloader):
            if i >= 3:  # Test a few batches
                break
            logger.info(f"   Batch {i}: data shape {data.shape}, targets shape {target.shape}")
        
        # Show creation stats
        stats = dataloader.get_creation_stats()
        logger.info(f"✅ DataLoader stats: {stats}")
        
        # Test adaptive batch size
        monitor = SharedMemoryMonitor()
        usage = monitor.get_usage_percentage()
        adaptive_size = get_adaptive_batch_size(64, usage)
        logger.info(f"   Adaptive batch size: {64} → {adaptive_size} (usage: {usage:.1f}%)")
        
    except Exception as e:
        logger.error(f"❌ MemorySafeDataLoader test failed: {e}")


if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("🧪 Testing shm_retry_wrapper module...")
    
    # Test shared memory monitoring
    monitor = SharedMemoryMonitor()
    usage = monitor.get_usage_percentage()
    logger.info(f"Initial usage: {usage:.1f}%")
    
    # Test cleanup
    cleanup_stats = cleanup_all_torch_memory()
    
    # Test memory-safe DataLoader
    example_memory_safe_dataloader()
    
    logger.info("✅ All tests completed!") 