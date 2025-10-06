# TraderV5 Live Trading Setup Instructions

## Quick Start

1. **Set up Alpaca credentials** (create a `.env` file in the project root):
   ```bash
   # Copy this template to .env and fill in your credentials
   ALPACA_API_KEY=your_paper_api_key_here
   ALPACA_API_SECRET=your_paper_secret_key_here
   ALPACA_API_BASE=https://paper-api.alpaca.markets
   ```

2. **Get Alpaca credentials**:
   - Go to [Alpaca Paper Trading Dashboard](https://app.alpaca.markets/paper/dashboard/overview)
   - Create an account if you don't have one
   - Generate API keys for paper trading
   - Copy the keys to your `.env` file

3. **Run the startup script**:
   ```bash
   cd Traderv5
   python start_live_trading.py
   ```

## What the startup script does:

1. ✅ **Environment Check**: Verifies all required files are present
2. 🔑 **Credentials Check**: Validates Alpaca API credentials
3. 🤖 **Model Training**: Ensures models are trained (trains if needed)
4. 🚀 **Live Trading**: Starts the trading system

## Manual Steps (if needed)

If the automated startup doesn't work, you can run steps manually:

### 1. Train Models
```bash
cd Traderv5
python model/training.py
```

### 2. Check Models
```bash
cd Traderv5
python ensure_models_trained.py
```

### 3. Start Trading
```bash
cd Traderv5
python run_live_trading.py
```

## Configuration

The system uses `config/trader_v5.json` for configuration. Key settings:

- **Tickers**: List of stocks to trade
- **Risk Settings**: Position sizing and stop losses
- **Trading Schedule**: Which days to trade (bucket system)
- **Data Sources**: Yahoo Finance and GDELT settings

## Paper Trading Features

- ✅ **Real market data**: Uses actual market prices
- ✅ **Real-time execution**: Orders execute at market prices
- ✅ **Virtual money**: No real money at risk
- ✅ **Full simulation**: Includes fees, slippage, and settlement
- ✅ **Account tracking**: Monitors equity, cash, and positions

## Monitoring

The system logs all activities to:
- Console output (real-time)
- `logs/trading_YYYY-MM-DD.log` (daily log files)

## Stopping the System

Press `Ctrl+C` to stop trading gracefully. The system will:
- Complete current trading cycle
- Close any open positions (optional)
- Save state and shutdown cleanly

## Troubleshooting

### "No trained models found"
Run: `python ensure_models_trained.py`

### "Alpaca credentials not found"
Check your `.env` file and ensure variables are set correctly.

### "Market is closed"
The system will wait for market hours. Check `logs/` for status updates.

### Connection issues
Verify your internet connection and Alpaca API status.

## Safety Features

- **Paper trading only**: Default configuration uses paper trading
- **Position limits**: Maximum positions and risk per trade
- **Market hours**: Only trades during market hours
- **Graceful shutdown**: Clean exit on interruption
- **Error handling**: Continues running despite individual trade failures

## Next Steps

Once paper trading is working well:
1. Monitor performance in the Alpaca dashboard
2. Adjust configuration in `config/trader_v5.json`
3. Consider live trading (with real money) if desired
4. Scale up position sizes gradually
