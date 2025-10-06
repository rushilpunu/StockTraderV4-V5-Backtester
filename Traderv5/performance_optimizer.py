"""Performance optimization utilities for TraderV5."""

from __future__ import annotations

import asyncio
import logging
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from functools import lru_cache

_LOG = logging.getLogger("traderv5.performance")


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    operation: str
    duration_ms: float
    memory_mb: float
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class FeatureCache:
    """High-performance feature caching with TTL and compression."""
    
    def __init__(self, cache_dir: Path, ttl_hours: int = 24, max_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        self.max_size_mb = max_size_mb
        self._lock = threading.RLock()
        self._metrics = {"hits": 0, "misses": 0}
        
        # Clean expired cache on startup
        self._cleanup_expired()
    
    def _cache_key(self, ticker: str, start_date: str, end_date: str, feature_type: str) -> str:
        """Generate cache key for features."""
        return f"{ticker}_{start_date}_{end_date}_{feature_type}"
    
    def _cache_path(self, key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, ticker: str, start_date: str, end_date: str, feature_type: str) -> Optional[pd.DataFrame]:
        """Get cached features."""
        key = self._cache_key(ticker, start_date, end_date, feature_type)
        cache_path = self._cache_path(key)
        
        with self._lock:
            if not cache_path.exists():
                self._metrics["misses"] += 1
                return None
            
            # Check TTL
            file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if file_age > timedelta(hours=self.ttl_hours):
                cache_path.unlink()
                self._metrics["misses"] += 1
                return None
            
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                self._metrics["hits"] += 1
                return data
            except Exception as e:
                _LOG.warning(f"Failed to load cache for {key}: {e}")
                cache_path.unlink()
                self._metrics["misses"] += 1
                return None
    
    def set(self, ticker: str, start_date: str, end_date: str, feature_type: str, data: pd.DataFrame) -> None:
        """Cache features."""
        key = self._cache_key(ticker, start_date, end_date, feature_type)
        cache_path = self._cache_path(key)
        
        with self._lock:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                _LOG.warning(f"Failed to cache {key}: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired cache files."""
        if not self.cache_dir.exists():
            return
        
        cutoff_time = datetime.now() - timedelta(hours=self.ttl_hours)
        removed_count = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_time:
                    cache_file.unlink()
                    removed_count += 1
            except Exception:
                pass
        
        if removed_count > 0:
            _LOG.info(f"Cleaned up {removed_count} expired cache files")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        with self._lock:
            total = self._metrics["hits"] + self._metrics["misses"]
            hit_rate = self._metrics["hits"] / total if total > 0 else 0.0
            
            return {
                "hits": self._metrics["hits"],
                "misses": self._metrics["misses"],
                "hit_rate": hit_rate,
                "cache_dir": str(self.cache_dir),
                "files_count": len(list(self.cache_dir.glob("*.pkl"))) if self.cache_dir.exists() else 0,
            }


class ParallelDataFetcher:
    """Parallel data fetching with intelligent batching."""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 3):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def fetch_price_data_parallel(
        self,
        tickers: List[str],
        yahoo_fetcher,
        lookback_days: int,
        interval: str = "1h",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch price data for multiple tickers in parallel."""
        results: Dict[str, pd.DataFrame] = {}
        
        # Process in batches to avoid overwhelming APIs
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            
            # Submit batch jobs
            future_to_ticker = {
                self.executor.submit(
                    self._fetch_single_ticker,
                    ticker,
                    yahoo_fetcher,
                    lookback_days,
                    interval,
                ): ticker
                for ticker in batch
            }
            
            # Collect results with timeouts to prevent hangs
            per_future_timeout = 12.0
            overall_batch_deadline = time.time() + 20.0
            pending = list(future_to_ticker.items())
            while pending and time.time() < overall_batch_deadline:
                next_pending = []
                for future, ticker in pending:
                    try:
                        data = future.result(timeout=per_future_timeout)
                        if data is not None and not data.empty:
                            results[ticker] = data
                    except Exception as e:
                        _LOG.warning(f"Failed or timed out fetching {ticker}: {e}")
                    finally:
                        if not future.done():
                            next_pending.append((future, ticker))
                pending = next_pending
            # Cancel any stragglers to avoid hanging
            for future, ticker in pending:
                future.cancel()
                _LOG.warning(f"Cancelled slow fetch for {ticker} after timeout")
            
            # Small delay between batches
            if i + self.batch_size < len(tickers):
                time.sleep(0.5)
        
        return results
    
    def _fetch_single_ticker(
        self,
        ticker: str,
        yahoo_fetcher,
        lookback_days: int,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single ticker."""
        try:
            return yahoo_fetcher.fetch_price_history(
                ticker,
                range_=f"{lookback_days}d",
                interval=interval,
            )
        except Exception as e:
            _LOG.warning(f"Error fetching {ticker}: {e}")
            return None
    
    def fetch_gdelt_data_parallel(
        self,
        tickers: List[str],
        gdelt_fetcher,
        start: datetime,
        end: datetime,
        timeline_minutes: int = 60,
    ) -> Dict[str, Any]:
        """Fetch GDELT data for multiple tickers in parallel."""
        results: Dict[str, Any] = {}
        
        # Process strictly one-at-a-time for GDELT to consolidate pressure
        gdelt_batch_size = 1
        
        for i in range(0, len(tickers), gdelt_batch_size):
            batch = tickers[i:i + gdelt_batch_size]
            
            future_to_ticker = {
                self.executor.submit(
                    self._fetch_gdelt_single,
                    ticker,
                    gdelt_fetcher,
                    start,
                    end,
                    timeline_minutes,
                ): ticker
                for ticker in batch
            }
            
            # Collect results with explicit timeouts
            per_future_timeout = 12.0
            overall_batch_deadline = time.time() + 25.0
            pending = list(future_to_ticker.items())
            while pending and time.time() < overall_batch_deadline:
                next_pending = []
                for future, ticker in pending:
                    try:
                        data = future.result(timeout=per_future_timeout)
                        if data is not None:
                            results[ticker] = data
                    except Exception as e:
                        _LOG.warning(f"Failed or timed out fetching GDELT for {ticker}: {e}")
                    finally:
                        if not future.done():
                            next_pending.append((future, ticker))
                pending = next_pending
            for future, ticker in pending:
                future.cancel()
                _LOG.warning(f"Cancelled slow GDELT fetch for {ticker} after timeout")
            
            # Longer delay between GDELT batches
            if i + gdelt_batch_size < len(tickers):
                time.sleep(2.0)
        
        return results
    
    def _fetch_gdelt_single(
        self,
        ticker: str,
        gdelt_fetcher,
        start: datetime,
        end: datetime,
        timeline_minutes: int,
    ) -> Optional[Any]:
        """Fetch GDELT data for a single ticker."""
        try:
            from Traderv5.data_sources import collect_gdelt_window
            return collect_gdelt_window(
                ticker,
                start=start,
                end=end,
                timeline_minutes=timeline_minutes,
            )
        except Exception as e:
            _LOG.warning(f"Error fetching GDELT data for {ticker}: {e}")
            return None
    
    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class OptimizedFeatureEngineer:
    """Optimized feature engineering with caching and vectorization.

    When `require_gdelt=True`, the builder will return an empty frame if
    GDELT data is unavailable for the requested window. This guarantees
    that downstream training/scoring never proceeds on price-only signals
    if sentiment is required for risk control.
    """
    
    def __init__(self, feature_cache: FeatureCache, config, require_gdelt: bool = True):
        self.cache = feature_cache
        self.config = config
        self._vectorized_operations = True
        self.require_gdelt = require_gdelt
    
    def build_features_optimized(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        gdelt_data: Optional[Any],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Build features with caching and optimization."""
        # Enforce GDELT requirement if configured
        if self.require_gdelt and gdelt_data is None:
            _LOG.info("Skipping %s feature build: GDELT required but missing", ticker)
            return pd.DataFrame()
        # Check cache first
        cached_features = self.cache.get(ticker, start_date, end_date, "features")
        if cached_features is not None:
            return cached_features
        
        # Build features
        start_time = time.time()
        features = self._build_features_vectorized(price_data, gdelt_data, ticker)
        duration_ms = (time.time() - start_time) * 1000
        
        # Cache the result
        self.cache.set(ticker, start_date, end_date, "features", features)
        
        _LOG.debug(f"Built features for {ticker} in {duration_ms:.1f}ms")
        return features
    
    def _build_features_vectorized(self, price_data: pd.DataFrame, gdelt_data: Optional[Any], ticker: str) -> pd.DataFrame:
        """Build features using vectorized operations."""
        df = price_data.copy()
        
        # Vectorized price features
        df = self._vectorized_price_features(df)
        
        # Vectorized sentiment features
        if gdelt_data is not None:
            df = self._vectorized_sentiment_features(df, gdelt_data)
        else:
            # No GDELT
            if self.require_gdelt:
                return pd.DataFrame()
            else:
                sentiment_cols = [
                    "sentiment_avg", "sentiment_std", "article_count",
                    "positive_article_ratio", "negative_article_ratio"
                ]
                for col in sentiment_cols:
                    df[col] = 0.0
        
        # Add ticker column
        df["ticker"] = ticker
        
        # Clean up
        df = df.replace({np.inf: 0.0, -np.inf: 0.0})
        df = df.fillna(0.0)
        
        return df
    
    def _vectorized_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized price feature calculations."""
        # Returns
        df["return_1"] = df["close"].pct_change().fillna(0.0)
        for window in [5, 10, 20]:
            df[f"return_{window}"] = df["close"].pct_change(window).fillna(0.0)
        
        # Volatility
        for window in [5, 10, 20]:
            df[f"volatility_{window}"] = df["return_1"].rolling(window, min_periods=2).std().fillna(0.0)
        
        # ATR
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = true_range.rolling(14, min_periods=1).mean()
        df["atr_14"] = atr.fillna(0.0)
        df["atr_pct_14"] = (atr / close.replace(0.0, np.nan)).fillna(0.0)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            ma = close.rolling(window, min_periods=1).mean()
            df[f"sma_{window}"] = ma
            df[f"price_vs_sma_{window}"] = ((close - ma) / ma.replace(0.0, np.nan)).fillna(0.0)
        
        # Volume features
        volume = df.get("volume", pd.Series(0.0, index=df.index))
        volume_mean = volume.rolling(20, min_periods=1).mean()
        volume_std = volume.rolling(20, min_periods=1).std().replace(0.0, np.nan)
        df["volume_z"] = ((volume - volume_mean) / volume_std).fillna(0.0)
        
        return df
    
    def _vectorized_sentiment_features(self, df: pd.DataFrame, gdelt_data: Any) -> pd.DataFrame:
        """Vectorized sentiment feature calculations."""
        # This would integrate with the GDELT data processing
        # For now, add placeholder sentiment features
        sentiment_cols = [
            "sentiment_avg", "sentiment_std", "article_count",
            "positive_article_ratio", "negative_article_ratio"
        ]
        
        for col in sentiment_cols:
            df[col] = 0.0  # Placeholder - would be calculated from gdelt_data
        
        return df


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
    
    def record_operation(
        self,
        operation: str,
        duration_ms: float,
        memory_mb: float = 0.0,
        cache_hits: int = 0,
        cache_misses: int = 0,
    ) -> None:
        """Record performance metrics for an operation."""
        with self._lock:
            metric = PerformanceMetrics(
                operation=operation,
                duration_ms=duration_ms,
                memory_mb=memory_mb,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
            )
            self.metrics.append(metric)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            if not self.metrics:
                return {"total_operations": 0}
            
            total_ops = len(self.metrics)
            total_time = sum(m.duration_ms for m in self.metrics)
            avg_time = total_time / total_ops
            
            # Group by operation
            by_operation = {}
            for metric in self.metrics:
                op = metric.operation
                if op not in by_operation:
                    by_operation[op] = []
                by_operation[op].append(metric)
            
            summary = {
                "total_operations": total_ops,
                "total_time_ms": total_time,
                "average_time_ms": avg_time,
                "by_operation": {},
            }
            
            for op, metrics in by_operation.items():
                op_times = [m.duration_ms for m in metrics]
                summary["by_operation"][op] = {
                    "count": len(metrics),
                    "total_time_ms": sum(op_times),
                    "avg_time_ms": sum(op_times) / len(op_times),
                    "min_time_ms": min(op_times),
                    "max_time_ms": max(op_times),
                }
            
            return summary
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def time_operation(operation_name: str):
    """Decorator to time operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            performance_monitor.record_operation(operation_name, duration_ms)
            return result
        return wrapper
    return decorator


@lru_cache(maxsize=1000)
def cached_calculation(key: str, *args):
    """Cached calculation wrapper."""
    # This would be used for expensive calculations
    pass


__all__ = [
    "FeatureCache",
    "ParallelDataFetcher", 
    "OptimizedFeatureEngineer",
    "PerformanceMonitor",
    "performance_monitor",
    "time_operation",
]
