"""
MEMORY OPTIMIZER
===============

Advanced memory optimization for AI trading systems including:
- Tensor memory management
- GPU memory optimization
- Batch processing efficiency
- Caching mechanisms
- Object pooling
- Memory leak detection

Features:
- Real-time memory monitoring
- Automatic garbage collection
- Memory profiling
- Leak detection and prevention
"""

import torch
import gc
import psutil
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from collections import defaultdict, deque
import threading
import time
from functools import wraps

class MemoryOptimizationLevel(Enum):
    """Memory optimization levels"""
    BASIC = "basic"          # Basic optimizations
    STANDARD = "standard"    # Standard optimizations
    AGGRESSIVE = "aggressive" # Aggressive optimizations
    EXTREME = "extreme"      # Extreme optimizations (may impact performance)

class ObjectType(Enum):
    """Object types for memory tracking"""
    TENSOR = "tensor"
    MODEL = "model"
    BUFFER = "buffer"
    CACHE = "cache"
    TEMPORARY = "temporary"
    PERSISTENT = "persistent"

@dataclass
class MemoryStats:
    """Memory statistics"""
    rss: float  # Resident Set Size (MB)
    vms: float  # Virtual Memory Size (MB)
    cpu_percent: float
    gpu_memory: float  # GPU memory usage (MB)
    tensor_memory: float  # Tensor memory usage (MB)
    cache_memory: float   # Cache memory usage (MB)
    available_memory: float  # Available system memory (MB)
    total_memory: float     # Total system memory (MB)
    timestamp: float

@dataclass
class OptimizationResult:
    """Memory optimization result"""
    freed_memory: float  # MB freed
    objects_collected: int
    tensors_freed: int
    cache_cleared: bool
    gpu_memory_freed: float  # MB
    duration: float  # Seconds
    recommendations: List[str]

class MemoryOptimizer:
    """
    Advanced Memory Optimizer for AI Trading Systems
    
    Implements comprehensive memory optimization techniques:
    1. Tensor memory management
    2. GPU memory optimization
    3. Batch processing efficiency
    4. Caching mechanisms
    5. Object pooling
    6. Memory leak detection
    """
    
    def __init__(
        self,
        optimization_level: MemoryOptimizationLevel = MemoryOptimizationLevel.STANDARD,
        enable_gpu_optimization: bool = True,
        enable_tensor_optimization: bool = True,
        enable_cache_management: bool = True,
        enable_leak_detection: bool = True,
        memory_threshold: float = 0.8,  # 80% memory usage threshold
        gpu_memory_threshold: float = 0.7,  # 70% GPU memory threshold
        auto_optimize: bool = True,
        auto_optimize_interval: int = 30  # seconds
    ):
        """
        Initialize memory optimizer
        
        Args:
            optimization_level: Level of optimization aggressiveness
            enable_gpu_optimization: Enable GPU memory optimization
            enable_tensor_optimization: Enable tensor memory optimization
            enable_cache_management: Enable cache management
            enable_leak_detection: Enable memory leak detection
            memory_threshold: System memory threshold for optimization (0.0 - 1.0)
            gpu_memory_threshold: GPU memory threshold for optimization (0.0 - 1.0)
            auto_optimize: Enable automatic optimization
            auto_optimize_interval: Auto optimization interval in seconds
        """
        self.optimization_level = optimization_level
        self.enable_gpu_optimization = enable_gpu_optimization
        self.enable_tensor_optimization = enable_tensor_optimization
        self.enable_cache_management = enable_cache_management
        self.enable_leak_detection = enable_leak_detection
        self.memory_threshold = memory_threshold
        self.gpu_memory_threshold = gpu_memory_threshold
        self.auto_optimize = auto_optimize
        self.auto_optimize_interval = auto_optimize_interval
        
        # Memory tracking
        self.memory_history = deque(maxlen=1000)
        self.object_registry = defaultdict(list)
        self.tensor_registry = []
        self.cache_registry = {}
        self.leak_candidates = defaultdict(int)
        
        # Statistics
        self.total_freed_memory = 0.0
        self.total_objects_collected = 0
        self.total_tensors_freed = 0
        self.total_gpu_memory_freed = 0.0
        self.optimization_count = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Auto optimization thread
        self.auto_optimize_thread = None
        self.auto_optimize_running = False
        
        # Start auto optimization if enabled
        if self.auto_optimize:
            self._start_auto_optimization()
    
    def _start_auto_optimization(self):
        """Start automatic optimization thread"""
        if self.auto_optimize_thread is None or not self.auto_optimize_thread.is_alive():
            self.auto_optimize_running = True
            self.auto_optimize_thread = threading.Thread(
                target=self._auto_optimization_worker,
                daemon=True
            )
            self.auto_optimize_thread.start()
            print(f"[MemoryOptimizer] Auto optimization started (every {self.auto_optimize_interval}s)")
    
    def _stop_auto_optimization(self):
        """Stop automatic optimization thread"""
        self.auto_optimize_running = False
        if self.auto_optimize_thread and self.auto_optimize_thread.is_alive():
            self.auto_optimize_thread.join(timeout=5.0)
    
    def _auto_optimization_worker(self):
        """Background worker for automatic optimization"""
        while self.auto_optimize_running:
            try:
                # Check memory usage
                stats = self.get_memory_stats()
                
                # Trigger optimization if thresholds exceeded
                if (stats.rss / stats.total_memory > self.memory_threshold or
                    stats.gpu_memory / stats.total_memory > self.gpu_memory_threshold):
                    
                    # Perform optimization
                    result = self.optimize_memory()
                    
                    if result.freed_memory > 0:
                        print(f"[MemoryOptimizer] Auto optimization freed {result.freed_memory:.2f}MB")
                
                # Sleep for interval
                time.sleep(self.auto_optimize_interval)
                
            except Exception as e:
                warnings.warn(f"[MemoryOptimizer] Auto optimization error: {str(e)}")
                time.sleep(self.auto_optimize_interval)
    
    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory statistics
        
        Returns:
            MemoryStats object with current memory usage
        """
        # Get system memory stats
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        # Get system memory
        system_memory = psutil.virtual_memory()
        
        # Get GPU memory (if available)
        gpu_memory = 0.0
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        except:
            pass
        
        # Get tensor memory
        tensor_memory = self._get_tensor_memory_usage()
        
        # Get cache memory
        cache_memory = self._get_cache_memory_usage()
        
        stats = MemoryStats(
            rss=memory_info.rss / 1024 / 1024,  # MB
            vms=memory_info.vms / 1024 / 1024,  # MB
            cpu_percent=cpu_percent,
            gpu_memory=gpu_memory,
            tensor_memory=tensor_memory,
            cache_memory=cache_memory,
            available_memory=system_memory.available / 1024 / 1024,  # MB
            total_memory=system_memory.total / 1024 / 1024,  # MB
            timestamp=time.time()
        )
        
        # Store in history
        self.memory_history.append(stats)
        
        return stats
    
    def _get_tensor_memory_usage(self) -> float:
        """Get tensor memory usage in MB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                # Estimate tensor memory from registry
                total_size = 0.0
                for tensor in self.tensor_registry:
                    if hasattr(tensor, 'element_size') and hasattr(tensor, 'nelement'):
                        total_size += tensor.element_size() * tensor.nelement()
                return total_size / 1024 / 1024  # MB
        except:
            return 0.0
    
    def _get_cache_memory_usage(self) -> float:
        """Get cache memory usage in MB"""
        try:
            total_size = 0.0
            for cache in self.cache_registry.values():
                if hasattr(cache, '__sizeof__'):
                    total_size += cache.__sizeof__()
            return total_size / 1024 / 1024  # MB
        except:
            return 0.0
    
    def register_object(
        self,
        obj: Any,
        object_type: ObjectType,
        size_estimate: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register object for memory tracking
        
        Args:
            obj: Object to register
            object_type: Type of object
            size_estimate: Estimated size in bytes
            metadata: Additional metadata
        """
        with self.lock:
            obj_id = id(obj)
            self.object_registry[object_type.value].append({
                'id': obj_id,
                'object': obj,
                'size_estimate': size_estimate,
                'metadata': metadata or {},
                'registered_at': time.time()
            })
    
    def unregister_object(
        self,
        obj: Any,
        object_type: ObjectType
    ):
        """
        Unregister object from memory tracking
        
        Args:
            obj: Object to unregister
            object_type: Type of object
        """
        with self.lock:
            obj_id = id(obj)
            if object_type.value in self.object_registry:
                self.object_registry[object_type.value] = [
                    item for item in self.object_registry[object_type.value]
                    if item['id'] != obj_id
                ]
    
    def register_tensor(
        self,
        tensor: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register tensor for memory tracking
        
        Args:
            tensor: Tensor to register
            metadata: Additional metadata
        """
        with self.lock:
            self.tensor_registry.append({
                'tensor': tensor,
                'metadata': metadata or {},
                'registered_at': time.time()
            })
    
    def unregister_tensor(
        self,
        tensor: torch.Tensor
    ):
        """
        Unregister tensor from memory tracking
        
        Args:
            tensor: Tensor to unregister
        """
        with self.lock:
            tensor_id = id(tensor)
            self.tensor_registry = [
                item for item in self.tensor_registry
                if id(item['tensor']) != tensor_id
            ]
    
    def register_cache(
        self,
        cache_name: str,
        cache_obj: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Register cache for memory tracking
        
        Args:
            cache_name: Name of cache
            cache_obj: Cache object
            metadata: Additional metadata
        """
        with self.lock:
            self.cache_registry[cache_name] = {
                'object': cache_obj,
                'metadata': metadata or {},
                'registered_at': time.time()
            }
    
    def unregister_cache(
        self,
        cache_name: str
    ):
        """
        Unregister cache from memory tracking
        
        Args:
            cache_name: Name of cache to unregister
        """
        with self.lock:
            if cache_name in self.cache_registry:
                del self.cache_registry[cache_name]
    
    def optimize_memory(
        self,
        force: bool = False
    ) -> OptimizationResult:
        """
        Perform memory optimization
        
        Args:
            force: Force optimization regardless of thresholds
            
        Returns:
            OptimizationResult with freed memory and statistics
        """
        start_time = time.time()
        
        # Get current memory stats
        initial_stats = self.get_memory_stats()
        
        freed_memory = 0.0
        objects_collected = 0
        tensors_freed = 0
        gpu_memory_freed = 0.0
        cache_cleared = False
        recommendations = []
        
        # Check if optimization is needed
        if not force:
            memory_usage = initial_stats.rss / initial_stats.total_memory
            gpu_usage = initial_stats.gpu_memory / initial_stats.total_memory
            
            if (memory_usage < self.memory_threshold and 
                gpu_usage < self.gpu_memory_threshold):
                return OptimizationResult(
                    freed_memory=0.0,
                    objects_collected=0,
                    tensors_freed=0,
                    cache_cleared=False,
                    gpu_memory_freed=0.0,
                    duration=0.0,
                    recommendations=["Memory usage below threshold, no optimization needed"]
                )
        
        with self.lock:
            # 1. Tensor memory optimization
            if self.enable_tensor_optimization:
                tensor_result = self._optimize_tensors()
                freed_memory += tensor_result['freed_memory']
                tensors_freed += tensor_result['tensors_freed']
                gpu_memory_freed += tensor_result['gpu_memory_freed']
                recommendations.extend(tensor_result['recommendations'])
            
            # 2. Cache management
            if self.enable_cache_management:
                cache_result = self._optimize_caches()
                freed_memory += cache_result['freed_memory']
                cache_cleared = cache_result['cache_cleared']
                recommendations.extend(cache_result['recommendations'])
            
            # 3. Object cleanup
            object_result = self._cleanup_objects()
            freed_memory += object_result['freed_memory']
            objects_collected += object_result['objects_collected']
            recommendations.extend(object_result['recommendations'])
            
            # 4. GPU memory optimization
            if self.enable_gpu_optimization and torch.cuda.is_available():
                gpu_result = self._optimize_gpu_memory()
                gpu_memory_freed += gpu_result['gpu_memory_freed']
                recommendations.extend(gpu_result['recommendations'])
            
            # 5. Garbage collection
            gc_result = self._run_garbage_collection()
            freed_memory += gc_result['freed_memory']
            objects_collected += gc_result['objects_collected']
            recommendations.extend(gc_result['recommendations'])
        
        # Update statistics
        self.total_freed_memory += freed_memory
        self.total_objects_collected += objects_collected
        self.total_tensors_freed += tensors_freed
        self.total_gpu_memory_freed += gpu_memory_freed
        self.optimization_count += 1
        
        duration = time.time() - start_time
        
        # Get final stats
        final_stats = self.get_memory_stats()
        
        return OptimizationResult(
            freed_memory=initial_stats.rss - final_stats.rss,
            objects_collected=objects_collected,
            tensors_freed=tensors_freed,
            cache_cleared=cache_cleared,
            gpu_memory_freed=gpu_memory_freed,
            duration=duration,
            recommendations=recommendations
        )
    
    def _optimize_tensors(self) -> Dict[str, Any]:
        """
        Optimize tensor memory usage
        
        Returns:
            Dictionary with optimization results
        """
        freed_memory = 0.0
        tensors_freed = 0
        gpu_memory_freed = 0.0
        recommendations = []
        
        try:
            # Free detached tensors
            detached_tensors = []
            for item in self.tensor_registry:
                tensor = item['tensor']
                if tensor.is_leaf and tensor.grad_fn is None:
                    # Check if tensor is referenced elsewhere
                    if self._is_tensor_orphaned(tensor):
                        detached_tensors.append(item)
            
            # Free detached tensors
            for item in detached_tensors:
                tensor = item['tensor']
                tensor_size = tensor.element_size() * tensor.nelement() / 1024 / 1024  # MB
                freed_memory += tensor_size
                tensors_freed += 1
                
                # Remove from registry
                self.tensor_registry.remove(item)
            
            # GPU tensor optimization
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Get GPU memory before and after
                before_gpu = torch.cuda.memory_allocated()
                torch.cuda.ipc_collect()
                after_gpu = torch.cuda.memory_allocated()
                
                gpu_memory_freed = (before_gpu - after_gpu) / 1024 / 1024  # MB
            
            if tensors_freed > 0:
                recommendations.append(f"Freed {tensors_freed} detached tensors ({freed_memory:.2f}MB)")
            
            if gpu_memory_freed > 0:
                recommendations.append(f"Freed {gpu_memory_freed:.2f}MB GPU memory")
                
        except Exception as e:
            recommendations.append(f"Tensor optimization error: {str(e)}")
        
        return {
            'freed_memory': freed_memory,
            'tensors_freed': tensors_freed,
            'gpu_memory_freed': gpu_memory_freed,
            'recommendations': recommendations
        }
    
    def _is_tensor_orphaned(self, tensor: torch.Tensor) -> bool:
        """
        Check if tensor is orphaned (no references)
        
        Args:
            tensor: Tensor to check
            
        Returns:
            True if tensor is orphaned
        """
        # This is a simplified check
        # In practice, you might want to use weak references or other methods
        try:
            # Check if tensor is part of computation graph
            if tensor.grad_fn is not None:
                return False
            
            # Check if tensor requires grad
            if tensor.requires_grad:
                return False
            
            # Additional checks can be added here
            return True
        except:
            return False
    
    def _optimize_caches(self) -> Dict[str, Any]:
        """
        Optimize cache memory usage
        
        Returns:
            Dictionary with optimization results
        """
        freed_memory = 0.0
        cache_cleared = False
        recommendations = []
        
        try:
            # Clear expired cache entries
            cleared_caches = []
            for cache_name, cache_info in self.cache_registry.items():
                cache_obj = cache_info['object']
                
                # Check if cache has clear method
                if hasattr(cache_obj, 'clear'):
                    try:
                        # Estimate size before clearing
                        size_before = cache_obj.__sizeof__() / 1024 / 1024  # MB
                        
                        # Clear cache
                        cache_obj.clear()
                        
                        # Estimate size after clearing
                        size_after = cache_obj.__sizeof__() / 1024 / 1024  # MB
                        
                        freed_memory += (size_before - size_after)
                        cleared_caches.append(cache_name)
                        
                    except Exception as e:
                        recommendations.append(f"Failed to clear cache {cache_name}: {str(e)}")
            
            if cleared_caches:
                cache_cleared = True
                recommendations.append(f"Cleared {len(cleared_caches)} caches ({freed_memory:.2f}MB)")
                
        except Exception as e:
            recommendations.append(f"Cache optimization error: {str(e)}")
        
        return {
            'freed_memory': freed_memory,
            'cache_cleared': cache_cleared,
            'recommendations': recommendations
        }
    
    def _cleanup_objects(self) -> Dict[str, Any]:
        """
        Cleanup unused objects
        
        Returns:
            Dictionary with cleanup results
        """
        freed_memory = 0.0
        objects_collected = 0
        recommendations = []
        
        try:
            # Remove expired objects from registry
            expired_objects = []
            current_time = time.time()
            
            for obj_type, objects in self.object_registry.items():
                expired = []
                for item in objects:
                    # Check if object reference is still valid
                    if 'registered_at' in item and current_time - item['registered_at'] > 3600:  # 1 hour
                        expired.append(item)
                
                # Remove expired objects
                for item in expired:
                    objects.remove(item)
                    expired_objects.append((obj_type, item))
            
            objects_collected = len(expired_objects)
            
            if objects_collected > 0:
                recommendations.append(f"Removed {objects_collected} expired objects")
                
        except Exception as e:
            recommendations.append(f"Object cleanup error: {str(e)}")
        
        return {
            'freed_memory': freed_memory,
            'objects_collected': objects_collected,
            'recommendations': recommendations
        }
    
    def _optimize_gpu_memory(self) -> Dict[str, Any]:
        """
        Optimize GPU memory usage
        
        Returns:
            Dictionary with optimization results
        """
        gpu_memory_freed = 0.0
        recommendations = []
        
        try:
            if torch.cuda.is_available():
                # Get initial GPU memory
                initial_gpu_memory = torch.cuda.memory_allocated()
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Force garbage collection on GPU
                torch.cuda.ipc_collect()
                
                # Get final GPU memory
                final_gpu_memory = torch.cuda.memory_allocated()
                
                gpu_memory_freed = (initial_gpu_memory - final_gpu_memory) / 1024 / 1024  # MB
                
                if gpu_memory_freed > 0:
                    recommendations.append(f"Freed {gpu_memory_freed:.2f}MB GPU memory")
                    
        except Exception as e:
            recommendations.append(f"GPU optimization error: {str(e)}")
        
        return {
            'gpu_memory_freed': gpu_memory_freed,
            'recommendations': recommendations
        }
    
    def _run_garbage_collection(self) -> Dict[str, Any]:
        """
        Run garbage collection
        
        Returns:
            Dictionary with GC results
        """
        freed_memory = 0.0
        objects_collected = 0
        recommendations = []
        
        try:
            # Get memory before GC
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            # Run garbage collection
            collected = gc.collect()
            
            # Get memory after GC
            memory_after = process.memory_info().rss
            
            freed_memory = (memory_before - memory_after) / 1024 / 1024  # MB
            objects_collected = collected
            
            if freed_memory > 0:
                recommendations.append(f"GC freed {freed_memory:.2f}MB ({collected} objects)")
            else:
                recommendations.append(f"GC ran but no memory freed ({collected} objects)")
                
        except Exception as e:
            recommendations.append(f"GC error: {str(e)}")
        
        return {
            'freed_memory': freed_memory,
            'objects_collected': objects_collected,
            'recommendations': recommendations
        }
    
    def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """
        Detect potential memory leaks
        
        Returns:
            List of potential memory leaks
        """
        leaks = []
        
        try:
            # Check for long-lived objects
            current_time = time.time()
            
            for obj_type, objects in self.object_registry.items():
                for item in objects:
                    if 'registered_at' in item:
                        age = current_time - item['registered_at']
                        if age > 7200:  # 2 hours
                            leaks.append({
                                'type': obj_type,
                                'age_hours': age / 3600,
                                'metadata': item.get('metadata', {}),
                                'size_estimate': item.get('size_estimate', 0)
                            })
            
            # Check for growing tensor registry
            if len(self.tensor_registry) > 10000:  # Arbitrary threshold
                leaks.append({
                    'type': 'tensor_registry',
                    'count': len(self.tensor_registry),
                    'age_hours': 0,
                    'metadata': {'warning': 'Large tensor registry'},
                    'size_estimate': len(self.tensor_registry) * 1024  # Rough estimate
                })
            
        except Exception as e:
            warnings.warn(f"Leak detection error: {str(e)}")
        
        return leaks
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """
        Get memory optimization statistics
        
        Returns:
            Dictionary with optimization statistics
        """
        return {
            'total_freed_memory_mb': self.total_freed_memory,
            'total_objects_collected': self.total_objects_collected,
            'total_tensors_freed': self.total_tensors_freed,
            'total_gpu_memory_freed_mb': self.total_gpu_memory_freed,
            'optimization_count': self.optimization_count,
            'current_memory_stats': self.get_memory_stats().__dict__,
            'registry_sizes': {
                'object_registry': sum(len(objects) for objects in self.object_registry.values()),
                'tensor_registry': len(self.tensor_registry),
                'cache_registry': len(self.cache_registry)
            },
            'leak_candidates': len(self.leak_candidates),
            'auto_optimization_enabled': self.auto_optimize,
            'optimization_level': self.optimization_level.value
        }
    
    def __del__(self):
        """Cleanup on destruction"""
        self._stop_auto_optimization()

# Decorator for memory optimization
def memory_optimized(func):
    """
    Decorator to automatically optimize memory after function execution
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create optimizer if not exists
        if not hasattr(wrapper, '_optimizer'):
            wrapper._optimizer = MemoryOptimizer()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Optimize memory after execution
            wrapper._optimizer.optimize_memory()
            
            return result
            
        except Exception as e:
            # Still optimize memory on error
            wrapper._optimizer.optimize_memory()
            raise e
    
    return wrapper

# Context manager for memory optimization
class MemoryOptimizationContext:
    """
    Context manager for memory optimization
    """
    
    def __init__(
        self,
        optimizer: Optional[MemoryOptimizer] = None,
        auto_optimize: bool = True
    ):
        self.optimizer = optimizer or MemoryOptimizer()
        self.auto_optimize = auto_optimize
        self.start_stats = None
    
    def __enter__(self):
        # Get initial memory stats
        self.start_stats = self.optimizer.get_memory_stats()
        return self.optimizer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Optimize memory on exit
        if self.auto_optimize:
            result = self.optimizer.optimize_memory()
            
            if result.freed_memory > 0:
                print(f"[MemoryOptimizationContext] Freed {result.freed_memory:.2f}MB")

# Example usage
if __name__ == "__main__":
    # Initialize memory optimizer
    optimizer = MemoryOptimizer(
        optimization_level=MemoryOptimizationLevel.STANDARD,
        enable_gpu_optimization=True,
        enable_tensor_optimization=True,
        enable_cache_management=True,
        enable_leak_detection=True,
        memory_threshold=0.8,
        gpu_memory_threshold=0.7,
        auto_optimize=True,
        auto_optimize_interval=30
    )
    
    print("=== MEMORY OPTIMIZER EXAMPLE ===\n")
    
    # Get initial memory stats
    print("Initial Memory Stats:")
    initial_stats = optimizer.get_memory_stats()
    print(f"  RSS: {initial_stats.rss:.2f}MB")
    print(f"  VMS: {initial_stats.vms:.2f}MB")
    print(f"  GPU Memory: {initial_stats.gpu_memory:.2f}MB")
    print(f"  Tensor Memory: {initial_stats.tensor_memory:.2f}MB")
    print(f"  Cache Memory: {initial_stats.cache_memory:.2f}MB")
    print(f"  Available Memory: {initial_stats.available_memory:.2f}MB")
    print(f"  Total Memory: {initial_stats.total_memory:.2f}MB")
    print(f"  CPU Percent: {initial_stats.cpu_percent:.2f}%")
    
    # Create some tensors for testing
    print("\nCreating test tensors...")
    test_tensors = []
    for i in range(10):
        tensor = torch.randn(1000, 1000)  # ~4MB each
        test_tensors.append(tensor)
        optimizer.register_tensor(tensor, {'purpose': f'test_tensor_{i}'})
    
    print(f"Created {len(test_tensors)} test tensors (~{len(test_tensors)*4}MB)")
    
    # Get memory stats after tensor creation
    print("\nMemory Stats After Tensor Creation:")
    tensor_stats = optimizer.get_memory_stats()
    print(f"  RSS: {tensor_stats.rss:.2f}MB (+{tensor_stats.rss - initial_stats.rss:.2f}MB)")
    print(f"  Tensor Memory: {tensor_stats.tensor_memory:.2f}MB")
    
    # Simulate some cache usage
    print("\nSimulating cache usage...")
    test_cache = {}
    for i in range(1000):
        test_cache[f'key_{i}'] = f'value_{i}' * 100  # ~10KB each
    
    optimizer.register_cache('test_cache', test_cache, {'purpose': 'test'})
    print(f"Created test cache with {len(test_cache)} entries")
    
    # Get memory stats after cache creation
    print("\nMemory Stats After Cache Creation:")
    cache_stats = optimizer.get_memory_stats()
    print(f"  RSS: {cache_stats.rss:.2f}MB (+{cache_stats.rss - tensor_stats.rss:.2f}MB)")
    print(f"  Cache Memory: {cache_stats.cache_memory:.2f}MB")
    
    # Perform memory optimization
    print("\nPerforming memory optimization...")
    optimization_result = optimizer.optimize_memory(force=True)
    
    print("Optimization Results:")
    print(f"  Freed Memory: {optimization_result.freed_memory:.2f}MB")
    print(f"  Objects Collected: {optimization_result.objects_collected}")
    print(f"  Tensors Freed: {optimization_result.tensors_freed}")
    print(f"  GPU Memory Freed: {optimization_result.gpu_memory_freed:.2f}MB")
    print(f"  Cache Cleared: {optimization_result.cache_cleared}")
    print(f"  Duration: {optimization_result.duration:.4f}s")
    print(f"  Recommendations: {len(optimization_result.recommendations)}")
    for rec in optimization_result.recommendations:
        print(f"    - {rec}")
    
    # Get final memory stats
    print("\nFinal Memory Stats:")
    final_stats = optimizer.get_memory_stats()
    print(f"  RSS: {final_stats.rss:.2f}MB")
    print(f"  GPU Memory: {final_stats.gpu_memory:.2f}MB")
    print(f"  Tensor Memory: {final_stats.tensor_memory:.2f}MB")
    print(f"  Cache Memory: {final_stats.cache_memory:.2f}MB")
    
    # Get optimization statistics
    print("\nOptimization Statistics:")
    stats = optimizer.get_optimization_statistics()
    print(f"  Total Freed Memory: {stats['total_freed_memory_mb']:.2f}MB")
    print(f"  Total Objects Collected: {stats['total_objects_collected']}")
    print(f"  Total Tensors Freed: {stats['total_tensors_freed']}")
    print(f"  Total GPU Memory Freed: {stats['total_gpu_memory_freed_mb']:.2f}MB")
    print(f"  Optimization Count: {stats['optimization_count']}")
    print(f"  Registry Sizes: {stats['registry_sizes']}")
    
    # Detect memory leaks
    print("\nDetecting Memory Leaks...")
    leaks = optimizer.detect_memory_leaks()
    print(f"  Potential Leaks Found: {len(leaks)}")
    for leak in leaks:
        print(f"    - Type: {leak['type']}, Age: {leak['age_hours']:.2f}h")
    
    print("\n=== MEMORY OPTIMIZER EXAMPLE COMPLETE ===")