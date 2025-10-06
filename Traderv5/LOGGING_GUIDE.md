# TraderV5 Enhanced Logging Guide

## What You'll See

When you run `start_live_trading.py`, you'll get detailed logging for every trading decision:

### 1. Cycle Start
```
================================================================================
🔄 TRADING CYCLE STARTED: 2025-10-03 14:30:00 UTC
================================================================================
💰 Account: Cash=$10,000.00 | Equity=$10,000.00
📊 Open Positions: 0
🏦 Market Status: OPEN ✅
```

### 2. For Each Stock (e.g., AAPL)
```
--------------------------------------------------------------------------------
🎯 Analyzing: AAPL
--------------------------------------------------------------------------------
📈 Fetching Yahoo Finance data...
   💵 Price: $175.43 (+2.34% over 120 periods)
   📊 Volume: 45,234,567

📰 Fetching GDELT sentiment data...
   📰 Articles: 234
   😊 Avg Sentiment: 0.156

🔧 Building ML features...

🤖 Running ML model prediction...
   🎲 Probability (Long): 68.5%
   📊 Signal Strength: +0.185

⚖️  DECISION BREAKDOWN:
   📰 GDELT Sentiment: 0.156 → BULLISH ✅
   📈 Yahoo Price Trend: +2.34% → UP ✅
   🤖 Model Prediction: 68.5% → BUY ✅

   🔍 Top Features:
      • close_momentum_5: 0.023
      • volume_ratio: 1.234
      • sentiment_avg_60: 0.156
      • rsi_14: 62.5
      • macd_signal: 0.012

📋 FINAL DECISION:
   Action: BUY
   Reason: strong bullish signal
   Quantity: 15 shares
   Notional: $2,631.45
   Stop Loss: $168.21
   Take Profit: $186.34
   ✅ Order placed: a1b2c3d4-5678-90ef-ghij-klmnopqrstuv
```

### 3. Cycle Complete
```
================================================================================
✅ Cycle completed in 45.3s
================================================================================
```

## Key Information Displayed

### Data Sources
- **📈 Yahoo Finance**: Current price, volume, price trend
- **📰 GDELT**: News article count, average sentiment
- **🤖 ML Model**: Probability prediction, signal strength

### Decision Components
Each stock shows:
1. **Data Collection** - What data was fetched
2. **Feature Engineering** - Top contributing features
3. **Individual Signals**:
   - GDELT: Bullish/Bearish sentiment
   - Yahoo: Price trend (up/down)
   - Model: Buy/Sell/Hold recommendation
4. **Final Decision** - Action taken and why

### Trade Execution
- Order details (quantity, price)
- Risk management (stop loss, take profit)
- Order confirmation or failure

## Understanding the Signals

### GDELT Sentiment
- **> 0**: Bullish news (positive sentiment)
- **< 0**: Bearish news (negative sentiment)
- Higher article count = more market attention

### Yahoo Price Trend
- **Positive %**: Stock is trending up
- **Negative %**: Stock is trending down
- Based on lookback period (default 7 days)

### Model Prediction
- **> 60%**: Strong buy signal
- **40-60%**: Hold/Neutral
- **< 40%**: Sell signal

## Logging Levels

All output goes to:
- **Console**: Real-time display
- **File**: `logs/trading_YYYY-MM-DD.log`

## Example Full Cycle

Market Open → 
  Fetch Data (AAPL, MSFT, AMZN, NVDA, TSLA) →
  Analyze Each Stock →
  Make Decisions →
  Execute Trades →
  Wait 5 minutes →
  Repeat

## Tips

1. **Watch the Decision Breakdown**: Shows if all signals agree
2. **Monitor Top Features**: See what's driving the model
3. **Check Order IDs**: Verify trades in Alpaca dashboard
4. **Track Performance**: Review logs to understand win/loss patterns

