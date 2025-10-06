"""Optimized trading system with performance improvements."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from Traderv5.cash_ledger import CashLedger
from Traderv5.bucket_scheduler import BucketScheduler
from Traderv5.decision_router import DecisionRouter
from Traderv5.performance_optimizer import (
    FeatureCache, 
    ParallelDataFetcher, 
    OptimizedFeatureEngineer,
    performance_monitor,
    time_operation
)
from Traderv5.configuration import load_trading_parameters
from Traderv5.data_sources import YahooFinanceDataFetcher, collect_gdelt_window
from Traderv5.model.predictor import ModelPredictor

_LOG = logging.getLogger("traderv5.optimized_trader")


class OptimizedTrader:
    """High-performance trading system with parallel processing and caching."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = load_trading_parameters(config_path)
        
        # Initialize core components
        self.cash_ledger = CashLedger(
            starting_equity=self.config.account.starting_equity,
            calendar=self.config.calendar
        )
        
        self.bucket_scheduler = BucketScheduler(
            total_equity=self.config.account.starting_equity,
            bucket_a_days=self.config.trading.bucket_a_days,
            bucket_b_days=self.config.trading.bucket_b_days
        )
        
        self.router = DecisionRouter(
            risk_per_trade_pct=self.config.risk.risk_per_trade_pct,
            atr_stop_multiplier=self.config.risk.atr_stop_multiplier,
            take_profit_multiplier=self.config.risk.take_profit_multiplier
        )
        
        # Initialize performance optimizations
        self.parallel_fetcher = ParallelDataFetcher(
            max_workers=min(4, len(self.config.trading.tickers)),
            batch_size=3
        )
        
        # Initialize feature cache
        cache_dir = self.config.data.cache_dir / "features"
        self.feature_cache = FeatureCache(cache_dir, ttl_hours=24)
        
        # Initialize optimized feature engineer
        self.optimized_feature_engineer = OptimizedFeatureEngineer(
            self.feature_cache,
            None  # Will be set when we have feature config
        )
        
        # Initialize data fetchers
        self.yahoo_fetcher = YahooFinanceDataFetcher()
        
        # Initialize model predictor
        try:
            self.predictor = ModelPredictor.load_default()
        except FileNotFoundError:
            _LOG.warning("No trained model found. Run training first.")
            self.predictor = None
        
        # Performance tracking
        self.performance_stats = {
            "cycles_completed": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "trades_executed": 0,
        }
        
        self._lock = threading.RLock()
    
    @time_operation("trading_cycle")
    def run_trading_cycle(self, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Run a single trading cycle with optimizations."""
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        with self._lock:
            cycle_start = time.time()
            
            try:
                # Get active bucket
                active_bucket = self.bucket_scheduler.get_active_bucket(current_time.date())
                bucket_allocation = self.bucket_scheduler.get_bucket_allocation(active_bucket)
                
                # Fetch market data in parallel
                market_data = self._fetch_market_data_parallel(current_time)
                
                # Process trading decisions
                decisions = self._process_trading_decisions(market_data, active_bucket, bucket_allocation)
                
                # Execute trades
                executed_trades = self._execute_trades(decisions)
                
                # Update performance stats
                cycle_time = time.time() - cycle_start
                self.performance_stats["cycles_completed"] += 1
                self.performance_stats["total_processing_time"] += cycle_time
                self.performance_stats["trades_executed"] += len(executed_trades)
                
                return {
                    "timestamp": current_time.isoformat(),
                    "active_bucket": active_bucket.value,
                    "bucket_allocation": bucket_allocation,
                    "market_data_count": len(market_data),
                    "decisions_count": len(decisions),
                    "executed_trades": executed_trades,
                    "cycle_time_ms": cycle_time * 1000,
                    "cash_ledger_state": {
                        "settled_cash": self.cash_ledger.settled_cash,
                        "total_equity": self.cash_ledger.get_total_equity(),
                        "available_cash": self.cash_ledger.get_available_cash(),
                    }
                }
                
            except Exception as e:
                _LOG.error(f"Error in trading cycle: {e}")
                return {
                    "timestamp": current_time.isoformat(),
                    "error": str(e),
                    "cycle_time_ms": (time.time() - cycle_start) * 1000,
                }
    
    def _fetch_market_data_parallel(self, current_time: datetime) -> Dict[str, Dict[str, Any]]:
        """Fetch market data for all tickers in parallel."""
        end_time = current_time
        start_time = end_time - timedelta(days=self.config.data.lookback_days)
        
        # Fetch price data in parallel
        price_data = self.parallel_fetcher.fetch_price_data_parallel(
            list(self.config.trading.tickers),
            self.yahoo_fetcher,
            self.config.data.lookback_days,
            self.config.data.price_interval,
        )
        
        self.performance_stats["api_calls"] += len(price_data)
        
        # Fetch GDELT data in parallel
        gdelt_data = self.parallel_fetcher.fetch_gdelt_data_parallel(
            list(price_data.keys()),
            None,  # Uses default GDELT client
            start_time.replace(tzinfo=None),
            end_time.replace(tzinfo=None),
            self.config.data.timeline_minutes,
        )
        
        self.performance_stats["api_calls"] += len(gdelt_data)
        
        # Combine data
        market_data = {}
        for ticker in price_data.keys():
            market_data[ticker] = {
                "price_data": price_data[ticker],
                "gdelt_data": gdelt_data.get(ticker),
                "timestamp": current_time,
            }
        
        return market_data
    
    def _process_trading_decisions(
        self, 
        market_data: Dict[str, Dict[str, Any]], 
        active_bucket: Any,
        bucket_allocation: float
    ) -> List[Dict[str, Any]]:
        """Process trading decisions for all tickers."""
        decisions = []
        
        for ticker, data in market_data.items():
            try:
                # Build features with caching
                start_date = (data["timestamp"] - timedelta(days=self.config.data.lookback_days)).strftime("%Y-%m-%d")
                end_date = data["timestamp"].strftime("%Y-%m-%d")
                
                features_df = self.optimized_feature_engineer.build_features_optimized(
                    ticker,
                    data["price_data"],
                    data["gdelt_data"],
                    start_date,
                    end_date,
                )
                
                if features_df.empty:
                    continue
                
                # Get latest features
                latest_features = features_df.iloc[-1]
                feature_dict = latest_features.to_dict()
                
                # Get model prediction
                if self.predictor is not None:
                    prediction = self.predictor.predict_proba(feature_dict)
                else:
                    prediction = 0.5  # Neutral prediction
                
                # Make trading decision
                decision = self.router.make_decision(
                    prediction=prediction,
                    horizon="swing",  # Could be determined by model or config
                    cash_ledger_state=self.cash_ledger.get_state(),
                    active_bucket=active_bucket,
                    bucket_allocation=bucket_allocation,
                )
                
                if decision.action != "HOLD":
                    decisions.append({
                        "ticker": ticker,
                        "decision": decision,
                        "prediction": prediction,
                        "features": feature_dict,
                        "timestamp": data["timestamp"],
                    })
                
            except Exception as e:
                _LOG.warning(f"Failed to process decision for {ticker}: {e}")
                continue
        
        return decisions
    
    def _execute_trades(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute trading decisions."""
        executed_trades = []
        
        for decision_data in decisions:
            try:
                decision = decision_data["decision"]
                ticker = decision_data["ticker"]
                
                if decision.action == "BUY":
                    # Check if we can buy
                    if self.cash_ledger.can_buy(decision.size):
                        self.cash_ledger.buy(decision.size)
                        executed_trades.append({
                            "ticker": ticker,
                            "action": "BUY",
                            "size": decision.size,
                            "timestamp": decision_data["timestamp"],
                            "prediction": decision_data["prediction"],
                        })
                        _LOG.info(f"Executed BUY: {ticker} size={decision.size}")
                
                elif decision.action == "SELL":
                    # Check if we can sell
                    if self.cash_ledger.can_sell(decision.size):
                        self.cash_ledger.sell(decision.size)
                        executed_trades.append({
                            "ticker": ticker,
                            "action": "SELL",
                            "size": decision.size,
                            "timestamp": decision_data["timestamp"],
                            "prediction": decision_data["prediction"],
                        })
                        _LOG.info(f"Executed SELL: {ticker} size={decision.size}")
                
            except Exception as e:
                _LOG.warning(f"Failed to execute trade for {decision_data['ticker']}: {e}")
                continue
        
        return executed_trades
    
    def end_of_day_settlement(self, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Process end-of-day settlement."""
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        with self._lock:
            # Process cash settlement
            self.cash_ledger.end_of_day()
            
            # Get cache metrics
            cache_metrics = self.feature_cache.get_metrics()
            self.performance_stats["cache_hits"] = cache_metrics["hits"]
            self.performance_stats["cache_misses"] = cache_metrics["misses"]
            
            return {
                "timestamp": current_time.isoformat(),
                "settlement_completed": True,
                "cash_ledger_state": {
                    "settled_cash": self.cash_ledger.settled_cash,
                    "total_equity": self.cash_ledger.get_total_equity(),
                    "available_cash": self.cash_ledger.get_available_cash(),
                },
                "cache_metrics": cache_metrics,
                "performance_stats": self.performance_stats.copy(),
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            cache_metrics = self.feature_cache.get_metrics()
            perf_summary = performance_monitor.get_summary()
            
            return {
                "performance_stats": self.performance_stats.copy(),
                "cache_metrics": cache_metrics,
                "operation_metrics": perf_summary,
                "cash_ledger_state": {
                    "settled_cash": self.cash_ledger.settled_cash,
                    "total_equity": self.cash_ledger.get_total_equity(),
                    "available_cash": self.cash_ledger.get_available_cash(),
                },
                "bucket_scheduler_state": {
                    "total_allocated_equity": self.bucket_scheduler.get_total_allocated_equity(),
                },
            }
    
    def shutdown(self) -> None:
        """Shutdown the trader and cleanup resources."""
        with self._lock:
            self.parallel_fetcher.shutdown()
            _LOG.info("OptimizedTrader shutdown complete")


def run_optimized_trader_demo() -> None:
    """Demo function to show optimized trader performance."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    
    _LOG.info("Starting OptimizedTrader demo...")
    
    # Initialize trader
    trader = OptimizedTrader()
    
    try:
        # Run a few trading cycles
        for i in range(3):
            _LOG.info(f"Running trading cycle {i+1}/3...")
            result = trader.run_trading_cycle()
            
            print(f"\nCycle {i+1} Results:")
            print(f"  Active bucket: {result.get('active_bucket', 'N/A')}")
            print(f"  Market data: {result.get('market_data_count', 0)} tickers")
            print(f"  Decisions: {result.get('decisions_count', 0)}")
            print(f"  Executed trades: {len(result.get('executed_trades', []))}")
            print(f"  Cycle time: {result.get('cycle_time_ms', 0):.1f}ms")
            print(f"  Available cash: ${result.get('cash_ledger_state', {}).get('available_cash', 0):.2f}")
            
            # Small delay between cycles
            time.sleep(1)
        
        # End of day settlement
        _LOG.info("Processing end-of-day settlement...")
        settlement_result = trader.end_of_day_settlement()
        
        # Print performance summary
        perf_summary = trader.get_performance_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Cycles completed: {perf_summary['performance_stats']['cycles_completed']}")
        print(f"Total processing time: {perf_summary['performance_stats']['total_processing_time']:.2f}s")
        print(f"Average cycle time: {perf_summary['performance_stats']['total_processing_time'] / max(1, perf_summary['performance_stats']['cycles_completed']):.2f}s")
        print(f"API calls: {perf_summary['performance_stats']['api_calls']}")
        print(f"Trades executed: {perf_summary['performance_stats']['trades_executed']}")
        
        cache_metrics = perf_summary['cache_metrics']
        if cache_metrics['hits'] + cache_metrics['misses'] > 0:
            hit_rate = cache_metrics['hits'] / (cache_metrics['hits'] + cache_metrics['misses'])
            print(f"Cache hit rate: {hit_rate:.1%} ({cache_metrics['hits']} hits, {cache_metrics['misses']} misses)")
        
        print(f"Final equity: ${perf_summary['cash_ledger_state']['total_equity']:.2f}")
        print(f"Available cash: ${perf_summary['cash_ledger_state']['available_cash']:.2f}")
        
        # Print operation metrics
        if perf_summary['operation_metrics']['total_operations'] > 0:
            print(f"\nOperation metrics:")
            for op, metrics in perf_summary['operation_metrics']['by_operation'].items():
                print(f"  {op}: {metrics['avg_time_ms']:.1f}ms avg ({metrics['count']} ops)")
        
        print("="*60)
        
    finally:
        trader.shutdown()
        _LOG.info("OptimizedTrader demo complete")


if __name__ == "__main__":
    run_optimized_trader_demo()
