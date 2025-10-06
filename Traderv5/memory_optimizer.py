"""Memory optimization utilities for TraderV5."""

from __future__ import annotations

import gc
import logging
import sys
import tracemalloc
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

_LOG = logging.getLogger("traderv5.memory_optimizer")


class MemoryOptimizer:
    """Memory optimization and monitoring utilities."""
    
    def __init__(self, enable_tracing: bool = False):
        self.enable_tracing = enable_tracing
        self.memory_stats = {
            "peak_memory_mb": 0.0,
            "current_memory_mb": 0.0,
            "gc_collections": 0,
            "objects_created": 0,
            "objects_destroyed": 0,
        }
        
        if enable_tracing:
            tracemalloc.start()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        current_mb = memory_info.rss / 1024 / 1024
        self.memory_stats["current_memory_mb"] = current_mb
        self.memory_stats["peak_memory_mb"] = max(
            self.memory_stats["peak_memory_mb"], 
            current_mb
        )
        
        return {
            "current_mb": current_mb,
            "peak_mb": self.memory_stats["peak_memory_mb"],
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Convert object columns to category if beneficial
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        # Downcast numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert boolean columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype('bool')
                except (ValueError, TypeError):
                    pass
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        _LOG.debug(f"DataFrame memory optimized: {original_memory:.2f}MB -> {optimized_memory:.2f}MB ({reduction:.1f}% reduction)")
        
        return df
    
    def optimize_numpy_array(self, arr: np.ndarray) -> np.ndarray:
        """Optimize numpy array memory usage."""
        if arr.dtype == np.float64:
            # Try to downcast to float32
            if np.all(np.isfinite(arr)) and np.all(arr >= np.finfo(np.float32).min) and np.all(arr <= np.finfo(np.float32).max):
                arr = arr.astype(np.float32)
        elif arr.dtype == np.int64:
            # Try to downcast to smaller integer types
            if np.all(arr >= np.iinfo(np.int32).min) and np.all(arr <= np.iinfo(np.int32).max):
                arr = arr.astype(np.int32)
            elif np.all(arr >= np.iinfo(np.int16).min) and np.all(arr <= np.iinfo(np.int16).max):
                arr = arr.astype(np.int16)
            elif np.all(arr >= np.iinfo(np.int8).min) and np.all(arr <= np.iinfo(np.int8).max):
                arr = arr.astype(np.int8)
        
        return arr
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return stats."""
        before = len(gc.get_objects())
        collected = gc.collect()
        after = len(gc.get_objects())
        
        self.memory_stats["gc_collections"] += 1
        self.memory_stats["objects_destroyed"] += (before - after)
        
        return {
            "objects_before": before,
            "objects_after": after,
            "objects_collected": collected,
            "objects_destroyed": before - after,
        }
    
    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Get detailed memory snapshot."""
        memory_usage = self.get_memory_usage()
        
        snapshot = {
            "memory_usage": memory_usage,
            "gc_stats": {
                "collections": self.memory_stats["gc_collections"],
                "objects_created": self.memory_stats["objects_created"],
                "objects_destroyed": self.memory_stats["objects_destroyed"],
            },
            "system_memory": self._get_system_memory_info(),
        }
        
        if self.enable_tracing:
            snapshot["tracemalloc"] = self._get_tracemalloc_info()
        
        return snapshot
    
    def _get_system_memory_info(self) -> Dict[str, float]:
        """Get system memory information."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total_mb": memory.total / 1024 / 1024,
                "available_mb": memory.available / 1024 / 1024,
                "used_mb": memory.used / 1024 / 1024,
                "percent_used": memory.percent,
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    def _get_tracemalloc_info(self) -> Dict[str, Any]:
        """Get tracemalloc information."""
        if not self.enable_tracing:
            return {"error": "tracing not enabled"}
        
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            return {
                "current_size_mb": tracemalloc.get_traced_memory()[0] / 1024 / 1024,
                "peak_size_mb": tracemalloc.get_traced_memory()[1] / 1024 / 1024,
                "top_allocations": [
                    {
                        "filename": stat.traceback.format()[0],
                        "size_mb": stat.size / 1024 / 1024,
                        "count": stat.count,
                    }
                    for stat in top_stats[:10]
                ],
            }
        except Exception as e:
            return {"error": str(e)}
    
    def monitor_memory_usage(self, operation_name: str, func, *args, **kwargs):
        """Monitor memory usage during an operation."""
        memory_before = self.get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            memory_after = self.get_memory_usage()
            
            memory_delta = memory_after["current_mb"] - memory_before["current_mb"]
            
            _LOG.info(f"Memory usage for {operation_name}: {memory_delta:+.2f}MB (peak: {memory_after['peak_mb']:.2f}MB)")
            
            return result
            
        except Exception as e:
            memory_after = self.get_memory_usage()
            memory_delta = memory_after["current_mb"] - memory_before["current_mb"]
            
            _LOG.error(f"Memory usage for {operation_name} (failed): {memory_delta:+.2f}MB")
            raise
    
    def cleanup_memory(self) -> Dict[str, Any]:
        """Comprehensive memory cleanup."""
        cleanup_stats = {}
        
        # Force garbage collection
        gc_stats = self.force_garbage_collection()
        cleanup_stats["garbage_collection"] = gc_stats
        
        # Get memory usage after cleanup
        memory_after = self.get_memory_usage()
        cleanup_stats["memory_after_cleanup"] = memory_after
        
        _LOG.info(f"Memory cleanup completed. Current usage: {memory_after['current_mb']:.2f}MB")
        
        return cleanup_stats
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        
        memory_usage = self.get_memory_usage()
        
        if memory_usage["current_mb"] > 1000:  # More than 1GB
            recommendations.append("Consider using chunked processing for large datasets")
        
        if memory_usage["current_mb"] > 500:  # More than 500MB
            recommendations.append("Enable DataFrame memory optimization")
            recommendations.append("Consider using categorical data types for string columns")
        
        if self.memory_stats["gc_collections"] < 10:
            recommendations.append("Consider more frequent garbage collection")
        
        if memory_usage["current_mb"] > memory_usage["available_mb"] * 0.8:
            recommendations.append("Memory usage is high - consider reducing batch sizes")
        
        return recommendations
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.enable_tracing:
            tracemalloc.stop()


class MemoryEfficientDataFrame:
    """Memory-efficient DataFrame wrapper with lazy loading and chunked processing."""
    
    def __init__(self, data_source, chunk_size: int = 10000):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self._current_chunk = None
        self._chunk_index = 0
        self._total_rows = None
    
    def __len__(self):
        if self._total_rows is None:
            # Estimate total rows (this might be expensive)
            self._total_rows = len(self.data_source)
        return self._total_rows
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._get_slice(key)
        else:
            return self._get_row(key)
    
    def _get_slice(self, slice_obj):
        """Get a slice of the DataFrame."""
        start = slice_obj.start or 0
        stop = slice_obj.stop or len(self)
        step = slice_obj.step or 1
        
        # Load chunks as needed
        chunks = []
        for i in range(start, stop, step):
            chunk_idx = i // self.chunk_size
            if chunk_idx != self._chunk_index:
                self._load_chunk(chunk_idx)
            
            row_idx = i % self.chunk_size
            if row_idx < len(self._current_chunk):
                chunks.append(self._current_chunk.iloc[row_idx])
        
        if chunks:
            return pd.DataFrame(chunks)
        else:
            return pd.DataFrame()
    
    def _get_row(self, index):
        """Get a single row."""
        chunk_idx = index // self.chunk_size
        if chunk_idx != self._chunk_index:
            self._load_chunk(chunk_idx)
        
        row_idx = index % self.chunk_size
        if row_idx < len(self._current_chunk):
            return self._current_chunk.iloc[row_idx]
        else:
            raise IndexError("Index out of range")
    
    def _load_chunk(self, chunk_index):
        """Load a specific chunk."""
        start_idx = chunk_index * self.chunk_size
        end_idx = start_idx + self.chunk_size
        
        # This would be implemented based on the data source
        # For now, return a mock implementation
        self._current_chunk = pd.DataFrame()  # Mock implementation
        self._chunk_index = chunk_index
    
    def iterrows(self):
        """Iterate over rows efficiently."""
        for i in range(len(self)):
            yield i, self[i]
    
    def apply(self, func, axis=0):
        """Apply function to DataFrame efficiently."""
        results = []
        for i in range(0, len(self), self.chunk_size):
            chunk = self._get_slice(slice(i, i + self.chunk_size))
            if not chunk.empty:
                results.append(chunk.apply(func, axis=axis))
        
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()


def optimize_trading_system_memory() -> Dict[str, Any]:
    """Optimize memory usage for the entire trading system."""
    optimizer = MemoryOptimizer(enable_tracing=True)
    
    # Get initial memory state
    initial_memory = optimizer.get_memory_usage()
    
    # Force garbage collection
    gc_stats = optimizer.force_garbage_collection()
    
    # Get final memory state
    final_memory = optimizer.get_memory_usage()
    
    # Get recommendations
    recommendations = optimizer.get_optimization_recommendations()
    
    return {
        "initial_memory_mb": initial_memory["current_mb"],
        "final_memory_mb": final_memory["current_mb"],
        "memory_saved_mb": initial_memory["current_mb"] - final_memory["current_mb"],
        "gc_stats": gc_stats,
        "recommendations": recommendations,
        "memory_snapshot": optimizer.get_memory_snapshot(),
    }


if __name__ == "__main__":
    # Demo memory optimization
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    
    print("Memory Optimization Demo")
    print("=" * 40)
    
    # Create optimizer
    optimizer = MemoryOptimizer(enable_tracing=True)
    
    # Get initial memory
    initial = optimizer.get_memory_usage()
    print(f"Initial memory: {initial['current_mb']:.2f}MB")
    
    # Create some test data
    test_data = pd.DataFrame({
        'col1': np.random.randn(10000),
        'col2': np.random.randint(0, 100, 10000),
        'col3': ['string_' + str(i) for i in range(10000)],
    })
    
    # Optimize DataFrame
    optimized_data = optimizer.optimize_dataframe(test_data)
    
    # Get memory after optimization
    after_optimization = optimizer.get_memory_usage()
    print(f"Memory after optimization: {after_optimization['current_mb']:.2f}MB")
    
    # Get recommendations
    recommendations = optimizer.get_optimization_recommendations()
    print(f"Recommendations: {recommendations}")
    
    # Get final snapshot
    snapshot = optimizer.get_memory_snapshot()
    print(f"Final memory snapshot: {snapshot['memory_usage']}")
    
    print("Demo complete!")
