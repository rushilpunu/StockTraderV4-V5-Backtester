# Small Capital Stock Trader ($1,000 Variant)

## Overview

This is a specialized variant of the main StockTrader (V4/V5) monorepo optimized for smaller capital ($1,000) that focuses on strategies that don't require pattern day trading. The goal is to achieve the same 1-2% daily returns while working within SEC regulations for accounts under $25,000.

## Key Differences from Original Trader

### 1. **No Pattern Day Trading**
- Avoids the 4+ day trades per 5-day rolling period restriction
- Uses swing trading and position holding strategies
- Focuses on longer-term positions (hours to days)

### 2. **Smaller Position Sizes**
- Maximum position size: $100-200 per trade
- Uses fractional shares when possible
- Diversifies across multiple small positions

### 3. **Enhanced Risk Management**
- Tighter stop losses (1-2% vs 2.5%)
- Smaller take profit targets (1-2% vs 2.5%)
- More conservative entry criteria

### 4. **Alternative Strategies**
- **Swing Trading**: Hold positions for 1-3 days
- **Options Trading**: Use covered calls and cash-secured puts
- **ETF Focus**: Trade ETFs for better diversification
- **Sector Rotation**: Focus on sector-specific ETFs

### 5. **Capital Efficiency**
- Uses margin for covered calls (if available)
- Implements cash-secured put strategies
- Focuses on high-probability setups

## Strategy Components

### Primary Strategies

1. **Swing Trading with GDELT Sentiment**
   - Use GDELT data to identify multi-day trends
   - Enter positions based on sentiment momentum
   - Hold for 1-3 days to avoid PDT restrictions

2. **Options Income Generation**
   - Sell covered calls on existing positions
   - Sell cash-secured puts on desired stocks
   - Generate income while waiting for opportunities

3. **ETF Momentum Trading**
   - Trade sector ETFs (XLK, XLF, XLE, etc.)
   - Use momentum indicators for entry/exit
   - Lower volatility than individual stocks

4. **News-Based Swing Trades**
   - Identify news events with multi-day impact
   - Enter positions before news breaks
   - Hold through the news cycle

### Risk Management

- **Position Sizing**: Max 10-20% of capital per position
- **Stop Losses**: 1-2% per trade
- **Take Profits**: 1-2% per trade
- **Daily Loss Limit**: 3% of capital
- **Weekly Loss Limit**: 8% of capital

## Configuration

The system uses the same core components as the original trader but with modified parameters:

- `MAX_POSITION_SIZE=150` (vs 1000 in original)
- `STOP_LOSS_PERCENTAGE=0.015` (vs 0.025 in original)
- `TAKE_PROFIT_PERCENTAGE=0.015` (vs 0.025 in original)
- `SWING_TRADING_ENABLED=true`
- `OPTIONS_TRADING_ENABLED=true`

## Expected Performance

- **Daily Returns**: 1-2% (same as original)
- **Risk**: Lower due to smaller position sizes
- **Volatility**: Reduced due to swing trading approach
- **Drawdown**: Expected to be lower than original

## Usage

```bash
# Run the small capital trader
python main.py

# Run in dry-run mode
python main.py --dry-run

# Run system check
python main.py --check
```

## Requirements

- Alpaca account with $1,000+ balance
- Options trading enabled (recommended)
- Same dependencies as original trader

## Legal Compliance

This variant is designed to comply with SEC regulations for accounts under $25,000:
- No pattern day trading
- Uses swing trading strategies
- Implements proper position sizing
- Follows all applicable trading rules
