"""Integration tests for the complete trading system."""

import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List

import numpy as np
import pandas as pd

from Traderv5.backtester import CashAccountBacktester, BacktestResult
from Traderv5.cash_ledger import CashLedger
from Traderv5.bucket_scheduler import BucketScheduler, BucketType
from Traderv5.decision_router import DecisionRouter, ModelPrediction, PositionInfo, TradeAction
from Traderv5.execution_manager import ExecutionManager
from Traderv5.risk_manager import RiskManager
from Traderv5.configuration import TradingParameters, AccountSettings, RiskSettings, CalendarSettings, SchedulerSettings, RoutingSettings, DataSettings, TrainingSettings


class TestIntegration:
    """Integration tests for the complete trading system."""
    
    @pytest.fixture
    def sample_config(self):
        """Create sample trading configuration."""
        return TradingParameters(
            account=AccountSettings(
                type="cash",
                starting_equity=25000.0,
                pattern_day_trade_limit=3,
                min_equity_for_margin=25000.0,
            ),
            risk=RiskSettings(
                risk_per_trade_pct=0.02,
                atr_stop_multiplier=1.8,
                take_profit_multiple=2.5,
                max_drawdown_pct=0.06,
                drawdown_size_reduction=0.5,
            ),
            calendar=CalendarSettings(
                exchange="XNYS",
                holiday_calendar="XNYS",
                custom_holidays=[],
                settlement_lag_days=1,
            ),
            data=DataSettings(
                queries_per_second=0.5,
                max_retries=4,
                backoff_factor=1.8,
                cache_ttl_hours=24,
                cache_dir="cache/http",
                gdelt_chunk_minutes=240,
                gdelt_pause_seconds=1.0,
                yahoo_pause_seconds=0.8,
            ),
            scheduler=SchedulerSettings(
                bucket_a_days=("MONDAY", "WEDNESDAY", "FRIDAY"),
                bucket_b_days=("TUESDAY", "THURSDAY"),
                allocation_pct=0.5,
            ),
            routing=RoutingSettings(
                swing_threshold=0.6,
                intraday_threshold=0.75,
                intraday_min_cash_pct=0.2,
                intraday_top_decile=0.9,
                hold_if_unsettled_needed=True,
            ),
            training=TrainingSettings(
                tickers=("AAPL", "MSFT", "AMZN"),
                lookback_days=120,
                price_interval="1h",
                timeline_minutes=60,
                label_horizon_minutes=90,
                positive_threshold=0.0035,
                negative_threshold=-0.0035,
                min_samples=200,
                max_tickers_per_batch=2,
            ),
        )
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 150.0
        
        data = {}
        for symbol in ["AAPL", "MSFT", "AMZN"]:
            prices = []
            current_price = base_price
            
            for _ in dates:
                # Random walk with slight upward bias
                change = np.random.normal(0.001, 0.02)  # 0.1% mean return, 2% volatility
                current_price *= (1 + change)
                prices.append(current_price)
            
            # Create OHLC data
            df = pd.DataFrame({
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, len(dates)),
            }, index=dates)
            
            data[symbol] = df
        
        return data
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample model predictions."""
        predictions = {}
        
        for symbol in ["AAPL", "MSFT", "AMZN"]:
            symbol_predictions = []
            
            # Generate predictions for each trading day in January 2024
            for day in range(1, 32):
                try:
                    pred_date = date(2024, 1, day)
                    if pred_date.weekday() < 5:  # Only weekdays
                        # Generate realistic prediction scores
                        swing_score = np.random.uniform(-0.5, 0.9)
                        intraday_score = np.random.uniform(-0.5, 0.9)
                        
                        prediction = ModelPrediction(
                            symbol=symbol,
                            swing_score=swing_score,
                            intraday_score=intraday_score,
                            confidence=np.random.uniform(0.6, 0.95),
                            timestamp=datetime.combine(pred_date, datetime.min.time()),
                            features={
                                "rsi": np.random.uniform(20, 80),
                                "macd": np.random.uniform(-0.1, 0.1),
                                "volume_ratio": np.random.uniform(0.5, 2.0),
                            },
                        )
                        symbol_predictions.append(prediction)
                except ValueError:
                    # Invalid date (e.g., Feb 30)
                    continue
            
            predictions[symbol] = symbol_predictions
        
        return predictions
    
    def test_complete_system_integration(self, sample_config, sample_price_data, sample_predictions):
        """Test the complete system working together."""
        # Create backtester
        backtester = CashAccountBacktester(sample_config)
        
        # Run backtest
        start_date = date(2024, 1, 2)  # Tuesday
        end_date = date(2024, 1, 31)   # Wednesday
        
        result = backtester.run_backtest(sample_price_data, sample_predictions, start_date, end_date)
        
        # Verify results
        assert isinstance(result, BacktestResult)
        assert result.start_date == start_date
        assert result.end_date == end_date
        assert result.initial_equity == Decimal('25000.00')
        assert result.final_equity > 0
        assert result.total_trades >= 0
        assert len(result.violations) >= 0
        
        # Check that no critical violations occurred
        critical_violations = [v for v in result.violations if v.severity == "CRITICAL"]
        assert len(critical_violations) == 0, f"Critical violations found: {[v.description for v in critical_violations]}"
        
        # Verify equity curve
        assert len(result.equity_curve) > 0
        assert result.equity_curve.index[0] <= start_date
        assert result.equity_curve.index[-1] <= end_date
    
    def test_cash_ledger_integration(self, sample_config):
        """Test cash ledger integration with other components."""
        cash_ledger = CashLedger(sample_config.account.starting_equity, sample_config.calendar)
        bucket_scheduler = BucketScheduler(sample_config.scheduler, sample_config.calendar)
        risk_manager = RiskManager(sample_config.risk)
        execution_manager = ExecutionManager(cash_ledger, risk_manager)
        
        # Update bucket scheduler with equity
        bucket_scheduler.update_equity(Decimal(str(sample_config.account.starting_equity)))
        
        # Test buy execution
        monday = date(2024, 1, 15)  # Monday (Bucket A)
        
        # Check if we can trade
        can_trade, reason = bucket_scheduler.can_trade(Decimal('5000.00'), monday)
        assert can_trade is True
        
        can_buy, reason = cash_ledger.can_buy(Decimal('5000.00'), monday)
        assert can_buy is True
        
        # Execute buy
        transaction = cash_ledger.execute_buy("AAPL", 10, Decimal('150.00'), monday)
        assert transaction is not None
        assert transaction.symbol == "AAPL"
        assert transaction.quantity == 10
        
        # Check cash ledger state
        summary = cash_ledger.get_cash_summary()
        assert summary["settled_cash"] < 25000.0  # Cash should be reduced
        assert summary["reserved_cash"] > 0  # Some cash should be reserved
    
    def test_bucket_scheduler_integration(self, sample_config):
        """Test bucket scheduler integration."""
        bucket_scheduler = BucketScheduler(sample_config.scheduler, sample_config.calendar)
        bucket_scheduler.update_equity(Decimal('25000.00'))
        
        # Test alternating days
        monday = date(2024, 1, 15)  # Monday
        tuesday = date(2024, 1, 16)  # Tuesday
        wednesday = date(2024, 1, 17)  # Wednesday
        
        assert bucket_scheduler.get_active_bucket(monday) == BucketType.A
        assert bucket_scheduler.get_active_bucket(tuesday) == BucketType.B
        assert bucket_scheduler.get_active_bucket(wednesday) == BucketType.A
        
        # Test fund allocation
        bucket_a = bucket_scheduler.buckets[BucketType.A]
        bucket_b = bucket_scheduler.buckets[BucketType.B]
        
        assert bucket_a.allocated_amount == Decimal('12500.00')  # 50% of 25000
        assert bucket_b.allocated_amount == Decimal('12500.00')  # 50% of 25000
    
    def test_decision_router_integration(self, sample_config):
        """Test decision router integration."""
        cash_ledger = CashLedger(sample_config.account.starting_equity, sample_config.calendar)
        bucket_scheduler = BucketScheduler(sample_config.scheduler, sample_config.calendar)
        bucket_scheduler.update_equity(Decimal('25000.00'))
        
        router = DecisionRouter(cash_ledger, bucket_scheduler, sample_config.routing)
        
        # Create test prediction
        prediction = ModelPrediction(
            symbol="AAPL",
            swing_score=0.7,  # Above threshold
            intraday_score=0.8,  # Above threshold
            confidence=0.85,
            timestamp=datetime.utcnow(),
            features={},
        )
        
        # Test buy decision
        monday = date(2024, 1, 15)  # Monday (Bucket A)
        decision = router._route_buy_decision(prediction, Decimal('150.00'), monday)
        
        assert decision.action == TradeAction.BUY
        assert decision.symbol == "AAPL"
        assert decision.bucket_type == BucketType.A
        assert decision.quantity > 0
    
    def test_risk_manager_integration(self, sample_config):
        """Test risk manager integration."""
        risk_manager = RiskManager(sample_config.risk)
        
        # Create sample price data for ATR calculation
        dates = pd.date_range('2024-01-01', '2024-01-20', freq='D')
        prices = 150.0 * (1 + np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        
        price_data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
        }, index=dates)
        
        # Calculate ATR
        atr = risk_manager.calculate_atr(price_data)
        assert atr > 0
        
        # Test position sizing
        entry_price = Decimal('150.00')
        stop_loss = risk_manager.calculate_stop_loss(entry_price, atr)
        quantity, risk_amount = risk_manager.calculate_position_size(
            entry_price, stop_loss, Decimal('10000.00')
        )
        
        assert quantity > 0
        assert risk_amount > 0
        assert stop_loss < entry_price
    
    def test_execution_manager_integration(self, sample_config):
        """Test execution manager integration."""
        cash_ledger = CashLedger(sample_config.account.starting_equity, sample_config.calendar)
        risk_manager = RiskManager(sample_config.risk)
        execution_manager = ExecutionManager(cash_ledger, risk_manager)
        
        # Create test decision
        from Traderv5.decision_router import TradeDecision, TradeHorizon
        decision = TradeDecision(
            action=TradeAction.BUY,
            symbol="AAPL",
            quantity=10,
            price=Decimal('150.00'),
            horizon=TradeHorizon.SWING,
            stop_loss=Decimal('142.50'),
            take_profit=Decimal('168.75'),
            risk_amount=Decimal('75.00'),
            reason="Test buy signal",
            timestamp=datetime.utcnow(),
            model_score=0.7,
            bucket_type=BucketType.A,
        )
        
        # Execute trade
        atr = Decimal('5.00')
        current_price = Decimal('150.00')
        result = execution_manager.execute_trade_decision(decision, atr, current_price)
        
        assert result.success is True
        assert result.symbol == "AAPL"
        assert result.quantity == 10
        assert result.price == current_price
    
    def test_settlement_flow(self, sample_config):
        """Test the complete settlement flow."""
        cash_ledger = CashLedger(sample_config.account.starting_equity, sample_config.calendar)
        
        # Execute buy on Monday
        monday = date(2024, 1, 15)  # Monday
        buy_transaction = cash_ledger.execute_buy("AAPL", 10, Decimal('150.00'), monday)
        
        # Check that cash is reserved
        assert cash_ledger.state.reserved_cash > 0
        assert cash_ledger.state.settled_cash < 25000.0
        
        # Execute sell on Tuesday
        tuesday = date(2024, 1, 16)  # Tuesday
        sell_transaction = cash_ledger.execute_sell("AAPL", 10, Decimal('160.00'), tuesday)
        
        # Check that proceeds go to unsettled
        settlement_date = sell_transaction.settlement_date
        assert settlement_date in cash_ledger.state.unsettled_cash_by_date
        assert cash_ledger.state.unsettled_cash_by_date[settlement_date] > 0
        
        # Process settlement
        settled_amount = cash_ledger.end_of_day_settlement(settlement_date)
        assert settled_amount > 0
        assert settlement_date not in cash_ledger.state.unsettled_cash_by_date
    
    def test_violation_prevention(self, sample_config):
        """Test that the system prevents violations."""
        cash_ledger = CashLedger(sample_config.account.starting_equity, sample_config.calendar)
        bucket_scheduler = BucketScheduler(sample_config.scheduler, sample_config.calendar)
        bucket_scheduler.update_equity(Decimal('25000.00'))
        
        router = DecisionRouter(cash_ledger, bucket_scheduler, sample_config.routing)
        
        # Try to buy with insufficient cash
        cash_ledger.execute_buy("MSFT", 100, Decimal('200.00'))  # Use up most cash
        
        prediction = ModelPrediction(
            symbol="AAPL",
            swing_score=0.8,
            intraday_score=0.9,
            confidence=0.9,
            timestamp=datetime.utcnow(),
            features={},
        )
        
        monday = date(2024, 1, 15)  # Monday (Bucket A)
        decision = router._route_buy_decision(prediction, Decimal('150.00'), monday)
        
        # Should be skipped due to insufficient cash
        assert decision.action == TradeAction.SKIP
        assert "Insufficient cash" in decision.reason
    
    def test_performance_metrics(self, sample_config, sample_price_data, sample_predictions):
        """Test that performance metrics are calculated correctly."""
        backtester = CashAccountBacktester(sample_config)
        
        start_date = date(2024, 1, 2)
        end_date = date(2024, 1, 31)
        
        result = backtester.run_backtest(sample_price_data, sample_predictions, start_date, end_date)
        
        # Check basic metrics
        assert result.total_return_pct is not None
        assert result.max_drawdown_pct is not None
        assert result.win_rate >= 0.0
        assert result.win_rate <= 1.0
        
        # Check trade statistics
        if result.total_trades > 0:
            assert result.winning_trades + result.losing_trades == result.total_trades
            assert result.avg_win is not None
            assert result.avg_loss is not None
        
        # Check equity curve
        assert len(result.equity_curve) > 0
        assert result.equity_curve.iloc[0]['equity'] == 25000.0  # Starting equity
        assert result.equity_curve.iloc[-1]['equity'] == result.final_equity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
