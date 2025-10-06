# TraderV5 - Cash Account Trading System

A comprehensive trading system designed specifically for sub-$25k cash accounts, with intelligent pattern day trading limit management and settlement tracking.

## Features

### Core Components

1. **Configuration Loader** - Validates and loads all trading parameters
2. **Trading Calendar** - Handles trading days, holidays, and T+1 settlement
3. **Cash Ledger** - Tracks settled vs unsettled cash with settlement dates
4. **Bucket Scheduler** - Alternates trading days between two buckets (A/B)
5. **Decision Router** - Combines model predictions with cash account constraints
6. **Risk Manager** - ATR-based stops and position sizing
7. **Execution Manager** - Bracket orders and trade execution
8. **Backtester** - Cash account mode with violation prevention

### Key Innovations

- **Intelligent PDT Management**: Uses bucket scheduling to trade daily while respecting cash account rules
- **Settlement Tracking**: Prevents good faith violations by tracking T+1 settlement
- **Cash-Aware Routing**: Blocks trades that would require unsettled funds
- **Rate-Limited Data**: Built-in caching and rate limiting for Yahoo/GDELT APIs

## Quick Start

### 1. Configuration

Edit `config/trader_v5.json`:

```json
{
  "account": {
    "type": "cash",
    "starting_equity": 20000.0,
    "pattern_day_trade_limit": 3,
    "min_equity_for_margin": 25000.0
  },
  "risk": {
    "risk_per_trade_pct": 0.02,
    "atr_stop_multiplier": 1.8,
    "take_profit_multiple": 2.5,
    "max_drawdown_pct": 0.06,
    "drawdown_size_reduction": 0.5
  },
  "scheduler": {
    "bucket_a_days": ["MONDAY", "WEDNESDAY", "FRIDAY"],
    "bucket_b_days": ["TUESDAY", "THURSDAY"],
    "allocation_pct": 0.5
  },
  "training": {
    "tickers": ["AAPL", "MSFT", "AMZN"],
    "lookback_days": 120,
    "max_tickers_per_batch": 2
  }
}
```

### 2. Train Models

```bash
cd Traderv5
python model/training.py
```

This will:
- Fetch price data from Yahoo Finance (with rate limiting)
- Collect GDELT sentiment data (with chunking and pauses)
- Train ensemble models (Gradient Boosting, Random Forest, Logistic Regression)
- Save models to `Traderv5/models/`

### 3. Run Backtest

```python
from Traderv5.backtester import CashAccountBacktester
from Traderv5.configuration import load_trading_config

# Load configuration
config_dict = load_trading_config()
config = TradingParameters.from_dict(config_dict)

# Create backtester
backtester = CashAccountBacktester(config)

# Run backtest
result = backtester.run_backtest(price_data, predictions, start_date, end_date)

print(f"Total Return: {result.total_return_pct:.2f}%")
print(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
print(f"Win Rate: {result.win_rate:.1%}")
print(f"Violations: {len(result.violations)}")
```

### 4. Run Tests

```bash
cd Traderv5
python run_tests.py
```

## System Architecture

### Cash Account Constraints

The system handles cash account limitations through:

1. **Settlement Tracking**: All sell proceeds go to `unsettled_cash_by_date` with T+1 settlement
2. **Bucket Scheduling**: Split equity into two buckets, alternate trading days
3. **Cash-Aware Routing**: Block trades requiring unsettled funds
4. **Violation Prevention**: Prevent good faith violations automatically

### Trading Flow

```
Model Prediction → Decision Router → Risk Manager → Execution Manager
       ↓                ↓              ↓              ↓
   Score/Confidence → Buy/Sell/Hold → Position Size → Bracket Order
       ↓                ↓              ↓              ↓
   Features         Cash Check    ATR Stops    Cash Ledger Update
```

### Bucket System

- **Bucket A**: Monday, Wednesday, Friday
- **Bucket B**: Tuesday, Thursday
- Each bucket gets 50% of equity
- While Bucket A trades, Bucket B settles (and vice versa)
- Enables daily trading without PDT violations

## Configuration Reference

### Account Settings
- `type`: "cash" or "margin"
- `starting_equity`: Initial account value
- `pattern_day_trade_limit`: PDT limit (usually 3)
- `min_equity_for_margin`: Minimum for margin account

### Risk Settings
- `risk_per_trade_pct`: Maximum risk per trade (0.01-0.05)
- `atr_stop_multiplier`: ATR multiplier for stop loss
- `take_profit_multiple`: Risk-reward ratio
- `max_drawdown_pct`: Maximum drawdown before size reduction
- `drawdown_size_reduction`: Position size reduction factor

### Scheduler Settings
- `bucket_a_days`: Trading days for bucket A
- `bucket_b_days`: Trading days for bucket B
- `allocation_pct`: Percentage of equity per bucket

### Data Settings
- `queries_per_second`: API rate limit
- `cache_ttl_hours`: Cache expiration time
- `gdelt_pause_seconds`: Pause between GDELT requests
- `yahoo_pause_seconds`: Pause between Yahoo requests

## API Rate Limiting

The system includes built-in rate limiting to avoid 429 errors:

- **Yahoo Finance**: 0.5 queries/second with 0.8s pause
- **GDELT**: 1.0s pause between requests, 240-minute chunks
- **Caching**: 24-hour TTL with filesystem storage
- **Batch Processing**: Process tickers in small batches

## Risk Management

### Position Sizing
- Based on ATR and risk percentage
- Maximum 2% of equity per trade
- Reduced size during drawdowns
- Respects available cash constraints

### Stop Loss/Take Profit
- ATR-based stop loss (1.8x ATR default)
- Take profit at 2.5x risk (configurable)
- Bracket orders for automatic execution

### Drawdown Protection
- Reduce position sizes by 50% if drawdown > 6%
- Track peak equity and current drawdown
- Automatic risk reduction

## Testing

### Unit Tests
- `test_cash_ledger.py`: Settlement and cash tracking
- `test_bucket_scheduler.py`: Bucket alternation logic
- `test_decision_router.py`: Trade decision routing
- `test_integration.py`: End-to-end system tests

### Test Coverage
- Settlement date calculation (Friday → Monday)
- Good faith violation prevention
- Bucket scheduler alternation
- Cash constraint enforcement
- Risk limit compliance

## Performance Optimization

### For Sub-$25k Accounts
- Optimized for smaller position sizes
- Efficient cash utilization
- Minimal slippage assumptions
- Realistic execution costs

### Rate Limiting
- Configurable API limits
- Exponential backoff on 429 errors
- Intelligent caching strategy
- Batch processing for efficiency

## Monitoring and Logging

The system provides comprehensive logging:
- Trade executions and settlements
- Violation detection and prevention
- Performance metrics and drawdowns
- API rate limiting and caching

## Limitations

- Designed for US equity markets only
- Requires reliable internet for data feeds
- Model performance depends on market conditions
- Cash account constraints limit position sizes

## Contributing

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update configuration validation
4. Document new parameters

## License

This project is for educational and research purposes. Use at your own risk.