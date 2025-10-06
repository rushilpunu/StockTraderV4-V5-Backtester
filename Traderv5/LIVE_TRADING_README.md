# TraderV5 Live Trading with Alpaca

This guide will help you set up and run TraderV5 with Alpaca paper trading for real-time automated trading.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
cd Traderv5
python3 setup_alpaca.py
python3 start_live_trading.py
```

### Option 2: Manual Setup

1. **Get Alpaca credentials**:
   - Go to [Alpaca Paper Trading](https://app.alpaca.markets/paper/dashboard/overview)
   - Sign up and generate API keys

2. **Set environment variables**:
   ```bash
   export ALPACA_API_KEY="your_api_key_here"
   export ALPACA_API_SECRET="your_secret_key_here"
   export ALPACA_API_BASE="https://paper-api.alpaca.markets"
   ```

3. **Run the system**:
   ```bash
   cd Traderv5
   python3 start_live_trading.py
   ```

## 📁 Files Created

- `setup_alpaca.py` - Interactive credential setup
- `start_live_trading.py` - Complete startup script
- `run_live_trading.py` - Main trading engine
- `ensure_models_trained.py` - Model training checker
- `SETUP_INSTRUCTIONS.md` - Detailed setup guide

## 🔧 What the System Does

### Automated Trading Features
- **Real-time market data** from Yahoo Finance and GDELT
- **ML predictions** using trained ensemble models
- **Risk management** with ATR-based stops and position sizing
- **Cash account compliance** with settlement tracking
- **Bucket scheduling** to avoid PDT violations
- **Paper trading** with real market prices (no real money)

### Trading Flow
1. **Data Collection**: Fetches price and sentiment data
2. **Feature Engineering**: Creates ML features from raw data
3. **Model Prediction**: Generates buy/sell/hold signals
4. **Risk Assessment**: Calculates position sizes and stops
5. **Order Execution**: Places trades via Alpaca API
6. **Monitoring**: Tracks performance and violations

## 📊 Configuration

The system uses `config/trader_v5.json` for settings:

```json
{
  "account": {
    "type": "cash",
    "starting_equity": 25000.0,
    "pattern_day_trade_limit": 3
  },
  "risk": {
    "risk_per_trade_pct": 0.02,
    "atr_stop_multiplier": 1.8,
    "take_profit_multiple": 2.5
  },
  "training": {
    "tickers": ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA"]
  }
}
```

## 📈 Monitoring

### Console Output
Real-time logging shows:
- Trading cycles and decisions
- Order executions and fills
- Account status and positions
- Error handling and recovery

### Log Files
- `logs/trading_YYYY-MM-DD.log` - Daily trading logs
- `logs/main/app_YYYY-MM-DD.log` - Application logs
- `logs/trades/trading_YYYY-MM-DD.log` - Trade-specific logs

### Alpaca Dashboard
Monitor your paper trading account at:
https://app.alpaca.markets/paper/dashboard/overview

## 🛡️ Safety Features

### Paper Trading
- **Virtual money only** - No real financial risk
- **Real market prices** - Authentic trading simulation
- **Full order execution** - Includes fees and slippage

### Risk Controls
- **Position limits** - Maximum positions and risk per trade
- **Stop losses** - ATR-based automatic stops
- **Cash management** - Prevents overdraft and violations
- **Market hours** - Only trades during market hours

### Error Handling
- **Graceful shutdown** - Clean exit on interruption
- **Retry logic** - Handles temporary failures
- **Circuit breakers** - Stops trading on repeated errors

## 🎛️ Controls

### Starting Trading
```bash
python3 start_live_trading.py
```

### Stopping Trading
- Press `Ctrl+C` for graceful shutdown
- System completes current cycle and exits cleanly

### Pausing Trading
- Stop the process and restart when ready
- System maintains state between sessions

## 🔍 Troubleshooting

### Common Issues

**"No trained models found"**
```bash
python3 ensure_models_trained.py
```

**"Alpaca credentials not found"**
```bash
python3 setup_alpaca.py
```

**"Market is closed"**
- System waits for market hours automatically
- Check logs for next market open time

**"Connection failed"**
- Verify internet connection
- Check Alpaca API status
- Validate credentials

### Debug Mode
For detailed logging, modify the logging level in the scripts:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📋 System Requirements

### Dependencies
- Python 3.8+
- alpaca-trade-api
- pandas, numpy, scikit-learn
- requests, yfinance

### Install Dependencies
```bash
pip install alpaca-trade-api pandas numpy scikit-learn requests yfinance
```

## 🎯 Performance Expectations

### Paper Trading Results
- **Realistic simulation** of live trading
- **Market impact** included in fills
- **Commission costs** applied
- **Settlement delays** enforced

### Typical Performance
- **Cycle time**: 5-10 minutes per cycle
- **Data latency**: 1-2 minutes for fresh data
- **Order execution**: 1-5 seconds for fills
- **Memory usage**: ~200MB typical

## 🔄 Next Steps

1. **Monitor Performance**: Watch paper trading results for 1-2 weeks
2. **Tune Configuration**: Adjust risk parameters and tickers
3. **Scale Up**: Increase position sizes gradually
4. **Live Trading**: Consider real money trading after validation

## ⚠️ Important Notes

- **Paper trading only** - Default configuration uses virtual money
- **Educational purpose** - Use for learning and testing strategies
- **No guarantees** - Past performance doesn't predict future results
- **Risk management** - Always use appropriate position sizing
- **Market hours** - System only trades during market hours

## 📞 Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review configuration in `config/trader_v5.json`
3. Verify Alpaca account status
4. Test with paper trading first

---

**Happy Trading! 🚀**
