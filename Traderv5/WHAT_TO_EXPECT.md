# What to Expect When Running TraderV5

## Startup (First 10 seconds)

```
[2025-10-03 14:30:00] INFO - 🎯 TraderV5 Live Trading Startup
[2025-10-03 14:30:00] INFO - ==================================================
[2025-10-03 14:30:00] INFO - 🔍 Checking environment setup...
[2025-10-03 14:30:00] INFO - ✅ Environment setup looks good
[2025-10-03 14:30:00] INFO - 🔑 Checking Alpaca credentials...
[2025-10-03 14:30:00] INFO - ✅ Alpaca credentials found
[2025-10-03 14:30:00] INFO -    Base URL: https://paper-api.alpaca.markets
[2025-10-03 14:30:00] INFO -    API Key: PKQ0AZWB...

[2025-10-03 14:30:01] INFO - 🤖 Checking model training status...
[2025-10-03 14:30:02] INFO - ✅ Models loaded successfully
[2025-10-03 14:30:02] INFO -    📁 Model Path: /path/to/trade_classifier.joblib
[2025-10-03 14:30:02] INFO -    🏷️  Model Type: GradientBoostingClassifier
[2025-10-03 14:30:02] INFO -    📊 Features: 47 features
[2025-10-03 14:30:02] INFO -    🎯 Training Samples: 1,234
[2025-10-03 14:30:02] INFO -    📈 Model Performance:
[2025-10-03 14:30:02] INFO -       • Accuracy: 67.50%
[2025-10-03 14:30:02] INFO -       • Precision: 71.20%
[2025-10-03 14:30:02] INFO -       • Recall: 64.80%
[2025-10-03 14:30:02] INFO -       • F1 Score: 67.87%
[2025-10-03 14:30:02] INFO -    📅 Last Updated: 2025-10-02 18:45:23

[2025-10-03 14:30:03] INFO - Loading configuration...
[2025-10-03 14:30:03] INFO - Trading 5 tickers: AAPL, MSFT, AMZN, NVDA, TSLA
[2025-10-03 14:30:03] INFO - Connecting to Alpaca...
[2025-10-03 14:30:04] INFO - Connected to Alpaca account: PA3F13CGD3SG
[2025-10-03 14:30:04] INFO - Equity: $10,000.00
[2025-10-03 14:30:04] INFO - Cash: $10,000.00
[2025-10-03 14:30:04] INFO - Market is OPEN

[2025-10-03 14:30:05] INFO - ✅ TraderV5 initialized successfully!

================================================================================
🎯 LIVE TRADING MODE ACTIVE
================================================================================
📊 Trading Schedule:
   • Cycle Interval: 5 minutes
   • Trading Tickers: AAPL, MSFT, AMZN, NVDA, TSLA
   • Lookback Period: 120 days

💡 The system is now running. You will see:
   1. Market status checks every 30 seconds
   2. Full trading cycles every 5 minutes (when market is open)
   3. Detailed analysis for each ticker

Press Ctrl+C to stop gracefully
================================================================================
```

## While Running (Every 30 seconds)

```
⏰ [14:30:35] System active - Next cycle in 265s | Cycles completed: 0
⏰ [14:31:05] System active - Next cycle in 235s | Cycles completed: 0
⏰ [14:31:35] System active - Next cycle in 205s | Cycles completed: 0
```

This tells you:
- **Current time**: [14:30:35]
- **System is alive**: "System active"
- **When next cycle runs**: "Next cycle in 265s" (4 minutes 25 seconds)
- **How many cycles done**: "Cycles completed: 0"

## Trading Cycle (Every 5 minutes when market is open)

```
================================================================================
🔄 TRADING CYCLE STARTED: 2025-10-03 14:35:00 UTC
================================================================================
💰 Account: Cash=$10,000.00 | Equity=$10,000.00
📊 Open Positions: 0
🏦 Market Status: OPEN ✅

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

[... repeats for MSFT, AMZN, NVDA, TSLA ...]

================================================================================
✅ CYCLE SUMMARY
================================================================================
⏱️  Duration: 67.3s
💰 Account Change: Cash $10,000.00 → $7,368.55 (-$2,631.45)
📊 Equity Change: $10,000.00 → $10,000.00 (+$0.00)
📈 Actions Taken: 1
   • BUY AAPL: 15 shares ($2,631.45)
🏦 Current Positions: 1
   • AAPL: 15 shares ($2,631.45)
================================================================================
```

## How to Know It's Working

### ✅ Good Signs:
1. **Status updates every 30s**: Shows system is alive
2. **Countdown decreasing**: "Next cycle in Xs" gets smaller
3. **Cycles complete**: "Cycles completed" number increases
4. **Market status shown**: OPEN ✅ or CLOSED ❌
5. **Data being fetched**: Yahoo and GDELT messages appear
6. **Decisions made**: BUY/SELL/HOLD for each ticker
7. **Summary at end**: Shows what happened in cycle

### ❌ Warning Signs:
1. **No status updates**: System may be frozen
2. **Market CLOSED**: Won't trade until market opens
3. **Errors in red**: Check the error messages
4. **No cycles completing**: May be stuck on data fetch

## Model Performance Indicators

At startup, you'll see:
- **Accuracy**: Overall correctness (aim for >60%)
- **Precision**: How often BUY signals are correct (aim for >65%)
- **Recall**: How many opportunities caught (aim for >60%)
- **F1 Score**: Balance of precision/recall (aim for >65%)

Higher numbers = better model performance

## What Each Signal Means

### GDELT Sentiment:
- **> 0.1**: Strong positive news
- **0.0 to 0.1**: Mild positive
- **-0.1 to 0.0**: Mild negative  
- **< -0.1**: Strong negative news

### Yahoo Price Trend:
- **> +5%**: Strong uptrend
- **+1% to +5%**: Mild uptrend
- **-1% to +1%**: Sideways
- **< -1%**: Downtrend

### Model Prediction:
- **> 70%**: Very confident BUY
- **60-70%**: Confident BUY
- **40-60%**: HOLD (neutral)
- **30-40%**: Confident SELL
- **< 30%**: Very confident SELL

## Troubleshooting

**"System active - Next cycle in 300s" keeps repeating**
- This is normal! It's waiting for the 5-minute interval

**"Market is CLOSED"**
- Normal outside market hours (9:30 AM - 4:00 PM ET)
- System will wait until market opens

**No trades being made**
- Check if all signals say HOLD
- Model might not be confident enough
- Risk limits might be preventing trades

**Errors during data fetch**
- Yahoo Finance may be slow - system will retry
- GDELT may have rate limits - system handles this

## Files to Check

**Live logs**: `logs/trading_YYYY-MM-DD.log`
**Model info**: `Traderv5/model/artifacts/trade_classifier_meta.json`
**Configuration**: `config/trader_v5.json`
**Alpaca trades**: https://app.alpaca.markets/paper/dashboard/overview

