#!/usr/bin/env python3
"""
Performance Optimization Manager
Centralized caching and performance monitoring for QBRAINS
"""

import hashlib
import json
import time
import threading
from typing import Dict, Any, Optional, Callable
from functools import wraps
import numpy as np
from pathlib import Path
import pickle


class PerformanceCache:
    """Centralized cache for performance optimization"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                self._remove(key)
                self.misses += 1
                return None
            
            self.access_times[key] = time.time()
            self.hits += 1
            return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        with self.lock:
            # Remove oldest if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times, key=self.access_times.get)
                self._remove(oldest_key)
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.access_times[key] = time.time()
    
    def _remove(self, key: str) -> None:
        """Remove key from cache"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
        self.access_times.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.access_times.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache),
                'max_size': self.max_size
            }


class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()
    
    def record_timing(self, operation: str, duration: float) -> None:
        """Record operation timing"""
        with self.lock:
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics"""
        with self.lock:
            stats = {}
            for operation, timings in self.metrics.items():
                if timings:
                    stats[operation] = {
                        'count': len(timings),
                        'avg': np.mean(timings),
                        'min': np.min(timings),
                        'max': np.max(timings),
                        'std': np.std(timings)
                    }
            return stats


# Global instances
performance_cache = PerformanceCache()
performance_monitor = PerformanceMonitor()


def cached(ttl_seconds: int = 3600, key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                for arg in args:
                    if isinstance(arg, (str, int, float, bool)):
                        key_parts.append(str(arg))
                    elif isinstance(arg, np.ndarray):
                        key_parts.append(hashlib.md5(arg.tobytes()).hexdigest())
                    else:
                        key_parts.append(str(hash(arg)))
                
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}:{v}")
                
                cache_key = "_".join(key_parts)
            
            # Check cache
            result = performance_cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Cache result
            performance_cache.set(cache_key, result)
            
            # Record performance
            performance_monitor.record_timing(func.__name__, duration)
            
            return result
        
        return wrapper
    return decorator


def timed(operation_name: Optional[str] = None):
    """Decorator for timing function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            performance_monitor.record_timing(op_name, duration)
            
            return result
        
        return wrapper
    return decorator


def create_audio_cache_key(audio_data: np.ndarray, sample_rate: int) -> str:
    """Create cache key for audio data"""
    audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()
    return f"audio_{audio_hash}_{sample_rate}"


def create_video_cache_key(video_path: str, frame_interval: int) -> str:
    """Create cache key for video processing"""
    file_hash = hashlib.md5(video_path.encode()).hexdigest()
    return f"video_{file_hash}_{frame_interval}"


def create_emotion_cache_key(text: str, context: Dict[str, Any]) -> str:
    """Create cache key for emotion analysis"""
    context_str = json.dumps(context, sort_keys=True)
    combined = f"{text}_{context_str}"
    return f"emotion_{hashlib.md5(combined.encode()).hexdigest()}"


def get_performance_report() -> Dict[str, Any]:
    """Get comprehensive performance report"""
    return {
        'cache_stats': performance_cache.get_stats(),
        'timing_stats': performance_monitor.get_stats(),
        'timestamp': time.time()
    }


def clear_all_caches():
    """Clear all performance caches"""
    performance_cache.clear()


# Batch processing utilities
def batch_process(items: list, batch_size: int = 10, process_func: Callable = None):
    """Process items in batches for better performance"""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        if process_func:
            batch_results = process_func(batch)
            results.extend(batch_results)
        else:
            results.extend(batch)
    return results


# Memory optimization
def optimize_memory_usage():
    """Optimize memory usage by clearing caches and garbage collection"""
    import gc
    clear_all_caches()
    gc.collect()


# Preloading utilities
class ModelPreloader:
    """Preload models for faster inference"""
    
    def __init__(self):
        self.loaded_models = {}
        self.lock = threading.Lock()
    
    def preload_model(self, model_path: str, model_type: str = "general"):
        """Preload a model"""
        with self.lock:
            if model_path not in self.loaded_models:
                try:
                    # This would be implemented based on specific model types
                    # For now, just mark as preloaded
                    self.loaded_models[model_path] = {
                        'type': model_type,
                        'loaded_at': time.time(),
                        'path': model_path
                    }
                except Exception as e:
                    print(f"Failed to preload model {model_path}: {e}")
    
    def is_loaded(self, model_path: str) -> bool:
        """Check if model is loaded"""
        return model_path in self.loaded_models


# Global model preloader
model_preloader = ModelPreloader()
