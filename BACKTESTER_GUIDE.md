# Backtester System Guide

This guide explains how to use the comprehensive backtesting system for TraderV4 and TraderV5.

## Overview

The backtesting system allows you to test trading strategies on historical data to evaluate performance before deploying with real capital.

### Architecture

```
┌─────────────────┐
│   Dashboard     │  ← Main control interface (React Native/Web)
│   (App.tsx)     │
└────────┬────────┘
         │
         ├──────────────────────────────────┐
         │                                  │
┌────────▼────────┐              ┌─────────▼──────────┐
│  TraderV4 API   │              │  TraderV5 API      │
│  (Port 8001)    │              │  (Port 8002)       │
└────────┬────────┘              └─────────┬──────────┘
         │                                  │
         └──────────┬───────────────────────┘
                    │
         ┌──────────▼────────────┐
         │  Backtester System    │
         │  (/backtester/)       │
         └──────────┬────────────┘
                    │
         ┌──────────┴────────────┐
         │                       │
    ┌────▼─────┐        ┌───────▼────────┐
    │ TraderV4 │        │   TraderV5     │
    │   Bot    │        │     Bot        │
    └──────────┘        └────────────────┘
```

## Components

### 1. Backtester Runner (`/backtester/`)
- **Purpose**: Core backtesting engine
- **Features**:
  - Parallel execution for multiple tickers
  - Sentiment-based trading simulation
  - Portfolio tracking and metrics calculation
  - Support for multiple bot variants

### 2. TraderV5Bot (`/Traderv5/backtest_adapter.py`)
- **Purpose**: Adapter that connects TraderV5 ML model to the backtester
- **Features**:
  - Feature engineering from price and GDELT data
  - ML-based trade decisions
  - Risk management integration
  - Position tracking

### 3. CashAccountBacktester (`/Traderv5/backtester.py`)
- **Purpose**: Advanced backtester with cash settlement simulation
- **Features**:
  - T+1 settlement tracking
  - Good faith violation prevention
  - Bucket-based trading (alternating days)
  - Comprehensive violation tracking

### 4. Optimized Backtester (`/Traderv5/optimized_backtester.py`)
- **Purpose**: High-performance backtesting with caching
- **Features**:
  - Parallel data fetching
  - Feature caching
  - Performance metrics tracking
  - Up to 10x faster than standard backtester

## Quick Start

### 1. Simple Demo

Run a quick 7-day backtest on AAPL:

```bash
python run_backtest_demo.py
```

This will:
- Auto-detect available bots (TraderV4 or TraderV5)
- Run a simple backtest
- Show basic results
- Provide next steps

### 2. Comprehensive Backtest

Run a detailed backtest with multiple tickers:

```bash
# Last 30 days with TraderV5
python backtester/run_comprehensive_backtest.py \
    --tickers AAPL MSFT GOOGL \
    --days 30 \
    --bot traderv5

# Specific date range
python backtester/run_comprehensive_backtest.py \
    --tickers TSLA \
    --dates 2024-01-01 2024-12-31 \
    --bot traderv5 \
    --output results.json

# Full year with comparison
python backtester/run_comprehensive_backtest.py \
    --tickers AAPL MSFT \
    --year 2024 \
    --bot traderv4 traderv5 \
    --compare
```

### 3. API-Based Backtesting

Start the TraderV5 API server:

```bash
python Traderv5/api.py
```

Then make API calls:

```bash
curl -X POST http://localhost:8002/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT"],
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "starting_cash": 10000.0,
    "use_optimized": true
  }'
```

### 4. Dashboard Integration

1. Start the API server:
   ```bash
   python Traderv5/api.py
   ```

2. Start the dashboard:
   ```bash
   cd dashboard
   npm start
   ```

3. Navigate to the "Backtest" tab in the dashboard

## Configuration Options

### Command Line Arguments

```bash
# Ticker selection
--tickers AAPL MSFT GOOGL    # Stocks to test

# Date range (choose one)
--days 30                     # Last N days
--year 2024                   # Entire year
--dates 2024-01-01 2024-12-31 # Specific range

# Trading parameters
--cash 10000                  # Starting capital
--bot traderv5                # Bot variant(s)

# Data parameters
--window 60                   # Sentiment window (minutes)
--timeframe 1Hour             # Price bar timeframe

# Features
--finbert                     # Enable FinBERT sentiment (slower)
--keybert                     # Enable KeyBERT keywords
--no-vader                    # Disable VADER sentiment
--trace                       # Record detailed trace

# Performance
--workers 4                   # Parallel workers (0=auto)

# Output
--output results.json         # Save results to file
--compare                     # Show comparison table
--quiet                       # Suppress output
```

## Understanding Results

### Metrics Explained

- **Total Return**: Dollar and percentage gain/loss
- **Win Rate**: Percentage of profitable trades
- **Alert Precision**: Percentage of alerts that led to profitable trades
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return metric
- **Profit Factor**: Gross profit / Gross loss

### Sample Output

```
BACKTEST RESULTS
================================================================================

Configuration:
  Tickers: AAPL, MSFT
  Period: 2024-01-01 to 2024-12-31
  Duration: 365 days
  Starting Cash: $10,000.00
  Bots: traderv5

────────────────────────────────────────────────────────────────────────────────
BOT: TRADERV5
────────────────────────────────────────────────────────────────────────────────

  AAPL:
    Total Return: $1,234.56 (12.35%)
    Trades: 45
    Alerts: 120
    Profitable Trades: 28
    Win Rate: 62.2%
    Alert Precision: 23.3%
    Max Drawdown: -5.42%
    Average Score: 0.156
    Top Keywords: earnings, revenue, growth

  MSFT:
    Total Return: $876.54 (8.77%)
    Trades: 38
    Alerts: 98
    Profitable Trades: 24
    Win Rate: 63.2%
    Alert Precision: 24.5%
    Max Drawdown: -4.21%

  Summary:
    Combined Return: $2,111.10
    Total Trades: 83
    Total Alerts: 218
    Overall Win Rate: 62.7%
```

## Best Practices

### 1. Model Training

Before using TraderV5, ensure the model is trained:

```bash
python Traderv5/model/training.py
```

Check model status:

```bash
python check_model_status.py
```

### 2. Date Range Selection

- **Short-term (7-30 days)**: Quick testing, recent market conditions
- **Medium-term (1-6 months)**: Good balance of speed and coverage
- **Long-term (1+ year)**: Comprehensive evaluation, slower execution

### 3. Performance Optimization

Use the optimized backtester for large-scale testing:

```bash
# Standard backtester
python backtester/run_comprehensive_backtest.py --tickers AAPL --year 2024

# Optimized backtester (via API)
curl -X POST http://localhost:8002/backtest/run -d '{"use_optimized": true, ...}'
```

### 4. Parallel Execution

For multiple tickers, use parallel workers:

```bash
python backtester/run_comprehensive_backtest.py \
    --tickers AAPL MSFT GOOGL AMZN TSLA \
    --days 30 \
    --workers 4
```

## Troubleshooting

### "Model not found" error

**Solution**: Train the TraderV5 model first
```bash
python Traderv5/model/training.py
```

### "No results generated"

**Possible causes**:
- Market was closed during the period
- No sentiment data available
- Insufficient historical data

**Solutions**:
- Use a longer date range
- Check API credentials (GDELT, Alpaca)
- Verify data availability with test scripts

### Slow performance

**Solutions**:
- Use optimized backtester
- Enable parallel workers
- Reduce date range or number of tickers
- Disable FinBERT and KeyBERT if not needed

### API connection errors

**Solutions**:
- Check if API server is running
- Verify port numbers (8001 for V4, 8002 for V5)
- Check firewall settings
- Review CORS configuration

## Advanced Usage

### Custom Bot Development

Create your own bot by extending `BaseBot`:

```python
from backtester.bots.base import BaseBot, register_bot

@register_bot
class MyCustomBot(BaseBot):
    name = "custom"
    
    def setup(self, config, baseline):
        # Initialize your bot
        pass
    
    def on_snapshot(self, timestamp, ticker, price, sentiment_snapshot, 
                    portfolio_cash, portfolio_equity, baseline, position, open_positions):
        # Return trading decision
        return TradeDecision(...)
```

### Integration with External Systems

The backtester can be integrated with:
- Jupyter notebooks for analysis
- Custom dashboards
- Automated testing pipelines
- Performance monitoring systems

### Data Export

Export results for external analysis:

```bash
python backtester/run_comprehensive_backtest.py \
    --tickers AAPL \
    --year 2024 \
    --output results.json

# Results include:
# - Trade-by-trade breakdown
# - Equity curve data
# - Feature importance
# - Sentiment analysis
```

## API Reference

### TraderV5 API Endpoints

```
GET  /                    # API index
GET  /health              # Health check
GET  /config              # Get configuration
GET  /model/status        # Model status

POST /backtest/run        # Start backtest
GET  /backtest/status/:id # Check status
GET  /backtest/list       # List all backtests
GET  /backtest/result/:id # Get results
```

### Backtester API Endpoints

```
GET  /backtester/available-bots  # List available bots
POST /backtester/run             # Run backtest
```

## Performance Metrics

### Optimized Backtester Performance

- **Data Fetch**: Parallel fetching (4x faster)
- **Feature Building**: Cached features (10x faster on cache hits)
- **Overall**: 3-5x faster than standard backtester

### Recommended Configurations

**Development/Testing:**
```bash
--days 7 --workers 1 --timeframe 1Hour
```

**Production Evaluation:**
```bash
--days 90 --workers 4 --timeframe 15Min --trace
```

**Comprehensive Analysis:**
```bash
--year 2024 --workers 4 --finbert --keybert --output full_results.json
```

## Next Steps

1. **Run the demo**: `python run_backtest_demo.py`
2. **Train the model**: `python Traderv5/model/training.py`
3. **Start the API**: `python Traderv5/api.py`
4. **Run comprehensive tests**: See examples above
5. **Integrate with dashboard**: Start React Native app

## Support

For issues or questions:
1. Check the logs in `/logs/`
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify API credentials are configured

## License

See project LICENSE file for details.



