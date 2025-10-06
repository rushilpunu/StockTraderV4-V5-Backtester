"""Demo script showing how to use the TraderV5 system."""

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from Traderv5.configuration import load_trading_parameters
from Traderv5.cash_ledger import CashLedger
from Traderv5.bucket_scheduler import BucketScheduler, BucketType
from Traderv5.decision_router import DecisionRouter, ModelPrediction, TradeAction
from Traderv5.risk_manager import RiskManager
from Traderv5.execution_manager import ExecutionManager
from Traderv5.backtester import CashAccountBacktester

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def demo_cash_ledger():
    """Demonstrate cash ledger functionality."""
    print("\n=== Cash Ledger Demo ===")
    
    # Load configuration
    config = load_trading_parameters()
    
    # Create cash ledger
    cash_ledger = CashLedger(config.account.starting_equity, config.calendar)
    
    print(f"Initial equity: ${cash_ledger.state.equity:,.2f}")
    print(f"Available cash: ${cash_ledger.state.available_cash:,.2f}")
    
    # Execute a buy
    monday = date(2024, 1, 15)  # Monday
    transaction = cash_ledger.execute_buy("AAPL", 10, Decimal('150.00'), monday)
    
    print(f"\nExecuted BUY: {transaction.symbol} {transaction.quantity} shares @ ${transaction.price}")
    print(f"Settlement date: {transaction.settlement_date}")
    print(f"Available cash after buy: ${cash_ledger.state.available_cash:,.2f}")
    
    # Execute a sell
    tuesday = date(2024, 1, 16)  # Tuesday
    sell_transaction = cash_ledger.execute_sell("AAPL", 10, Decimal('160.00'), tuesday)
    
    print(f"\nExecuted SELL: {sell_transaction.symbol} {sell_transaction.quantity} shares @ ${sell_transaction.price}")
    print(f"Settlement date: {sell_transaction.settlement_date}")
    
    # Show unsettled schedule
    schedule = cash_ledger.get_unsettled_schedule()
    print(f"\nUnsettled cash schedule: {schedule}")
    
    # Process settlement
    settled_amount = cash_ledger.end_of_day_settlement(sell_transaction.settlement_date)
    print(f"Settled amount: ${settled_amount:,.2f}")
    print(f"Final available cash: ${cash_ledger.state.available_cash:,.2f}")


def demo_bucket_scheduler():
    """Demonstrate bucket scheduler functionality."""
    print("\n=== Bucket Scheduler Demo ===")
    
    # Load configuration
    config = load_trading_parameters()
    
    # Create bucket scheduler
    bucket_scheduler = BucketScheduler(config.scheduler, config.calendar)
    bucket_scheduler.update_equity(Decimal('20000.00'))
    
    # Show bucket allocations
    status = bucket_scheduler.get_bucket_status()
    print("Bucket allocations:")
    for bucket, info in status.items():
        print(f"  Bucket {bucket}: ${info['allocated']:,.2f} allocated, ${info['free']:,.2f} free")
    
    # Test different days
    test_dates = [
        (date(2024, 1, 15), "Monday"),
        (date(2024, 1, 16), "Tuesday"),
        (date(2024, 1, 17), "Wednesday"),
        (date(2024, 1, 18), "Thursday"),
        (date(2024, 1, 19), "Friday"),
    ]
    
    print("\nActive buckets by day:")
    for test_date, day_name in test_dates:
        active_bucket = bucket_scheduler.get_active_bucket(test_date)
        info = bucket_scheduler.get_active_bucket_info(test_date)
        print(f"  {day_name}: Bucket {active_bucket.value} - {'Can trade' if info['can_trade'] else 'No trading'}")
    
    # Test trading schedule
    start_date = date(2024, 1, 15)
    end_date = date(2024, 1, 19)
    schedule = bucket_scheduler.get_trading_schedule(start_date, end_date)
    
    print(f"\nTrading schedule ({start_date} to {end_date}):")
    for date_str, bucket in schedule.items():
        print(f"  {date_str}: Bucket {bucket}")


def demo_decision_router():
    """Demonstrate decision router functionality."""
    print("\n=== Decision Router Demo ===")
    
    # Load configuration
    config = load_trading_parameters()
    
    # Create components
    cash_ledger = CashLedger(config.account.starting_equity, config.calendar)
    bucket_scheduler = BucketScheduler(config.scheduler, config.calendar)
    bucket_scheduler.update_equity(Decimal('20000.00'))
    router = DecisionRouter(cash_ledger, bucket_scheduler, config.routing)
    
    # Create test prediction
    prediction = ModelPrediction(
        symbol="AAPL",
        swing_score=0.7,  # Above threshold
        intraday_score=0.8,  # Above threshold
        confidence=0.85,
        timestamp=datetime.utcnow(),
        features={"rsi": 65.0, "macd": 0.05, "volume_ratio": 1.2},
    )
    
    print(f"Model prediction for {prediction.symbol}:")
    print(f"  Swing score: {prediction.swing_score:.3f}")
    print(f"  Intraday score: {prediction.intraday_score:.3f}")
    print(f"  Confidence: {prediction.confidence:.3f}")
    
    # Test buy decision
    monday = date(2024, 1, 15)  # Monday (Bucket A)
    decision = router._route_buy_decision(prediction, Decimal('150.00'), monday)
    
    print(f"\nBuy decision:")
    print(f"  Action: {decision.action.value}")
    print(f"  Symbol: {decision.symbol}")
    print(f"  Quantity: {decision.quantity}")
    print(f"  Price: ${decision.price}")
    print(f"  Horizon: {decision.horizon.value if decision.horizon else 'None'}")
    print(f"  Bucket: {decision.bucket_type.value if decision.bucket_type else 'None'}")
    print(f"  Reason: {decision.reason}")
    
    # Test with insufficient cash
    cash_ledger.execute_buy("MSFT", 100, Decimal('200.00'))  # Use up most cash
    
    decision_no_cash = router._route_buy_decision(prediction, Decimal('150.00'), monday)
    print(f"\nBuy decision (insufficient cash):")
    print(f"  Action: {decision_no_cash.action.value}")
    print(f"  Reason: {decision_no_cash.reason}")


def demo_risk_manager():
    """Demonstrate risk manager functionality."""
    print("\n=== Risk Manager Demo ===")
    
    # Load configuration
    config = load_trading_parameters()
    
    # Create risk manager
    risk_manager = RiskManager(config.risk)
    
    # Create sample price data
    dates = pd.date_range('2024-01-01', '2024-01-20', freq='D')
    np.random.seed(42)
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
    print(f"ATR (14-period): ${atr:.2f}")
    
    # Calculate stop loss and take profit
    entry_price = Decimal('150.00')
    stop_loss = risk_manager.calculate_stop_loss(entry_price, atr)
    take_profit = risk_manager.calculate_take_profit(entry_price, stop_loss)
    
    print(f"\nRisk levels for entry at ${entry_price}:")
    print(f"  Stop loss: ${stop_loss:.2f}")
    print(f"  Take profit: ${take_profit:.2f}")
    print(f"  Risk per share: ${entry_price - stop_loss:.2f}")
    print(f"  Reward per share: ${take_profit - entry_price:.2f}")
    
    # Calculate position size
    available_cash = Decimal('10000.00')
    quantity, risk_amount = risk_manager.calculate_position_size(
        entry_price, stop_loss, available_cash
    )
    
    print(f"\nPosition sizing:")
    print(f"  Available cash: ${available_cash:,.2f}")
    print(f"  Recommended quantity: {quantity} shares")
    print(f"  Total risk: ${risk_amount:.2f}")
    print(f"  Risk percentage: {(risk_amount / Decimal('20000.00')) * 100:.2f}%")


def demo_simple_backtest():
    """Demonstrate a simple backtest."""
    print("\n=== Simple Backtest Demo ===")
    
    # Load configuration
    config = load_trading_parameters()
    
    # Create sample price data
    dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
    np.random.seed(42)
    
    price_data = {}
    predictions = {}
    
    for symbol in ["AAPL", "MSFT"]:
        # Generate price data
        prices = 150.0 * (1 + np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        price_data[symbol] = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates)),
        }, index=dates)
        
        # Generate predictions
        symbol_predictions = []
        for i, date_val in enumerate(dates):
            if date_val.weekday() < 5:  # Only weekdays
                prediction = ModelPrediction(
                    symbol=symbol,
                    swing_score=np.random.uniform(0.3, 0.9),
                    intraday_score=np.random.uniform(0.3, 0.9),
                    confidence=np.random.uniform(0.6, 0.95),
                    timestamp=datetime.combine(date_val.date(), datetime.min.time()),
                    features={"rsi": np.random.uniform(20, 80)},
                )
                symbol_predictions.append(prediction)
        predictions[symbol] = symbol_predictions
    
    # Create and run backtester
    backtester = CashAccountBacktester(config)
    
    start_date = date(2024, 1, 2)
    end_date = date(2024, 1, 31)
    
    print(f"Running backtest from {start_date} to {end_date}...")
    result = backtester.run_backtest(price_data, predictions, start_date, end_date)
    
    # Display results
    print(f"\nBacktest Results:")
    print(f"  Initial equity: ${result.initial_equity:,.2f}")
    print(f"  Final equity: ${result.final_equity:,.2f}")
    print(f"  Total return: {result.total_return_pct:.2f}%")
    print(f"  Max drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"  Total trades: {result.total_trades}")
    print(f"  Win rate: {result.win_rate:.1%}")
    print(f"  Violations: {len(result.violations)}")
    
    if result.violations:
        print(f"\nViolations:")
        for violation in result.violations[:5]:  # Show first 5
            print(f"  {violation.date}: {violation.violation_type} - {violation.description}")


def main():
    """Run all demos."""
    print("TraderV5 System Demo")
    print("=" * 50)
    
    try:
        demo_cash_ledger()
        demo_bucket_scheduler()
        demo_decision_router()
        demo_risk_manager()
        demo_simple_backtest()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
