# Automated Trading System

A comprehensive automated trading system that uses GDELT's event and sentiment data to detect high-volatility triggers for major stock tickers, generate internal alerts, and execute trades via Alpaca's paper trading API.

## 🚀 Features

- **Real-time Sentiment Analysis**: Leverages GDELT's global news and event data
- **Advanced Volatility Detection**: ML-based anomaly detection and statistical analysis  
- **Intelligent Alert System**: Multi-level alert generation with confidence scoring
- **Risk Management**: Comprehensive position sizing, stop-loss, and portfolio risk controls
- **Paper Trading**: Safe testing environment using Alpaca's paper trading API
- **Automated Execution**: Fully autonomous operation with human oversight capabilities
- **Comprehensive Logging**: Detailed audit trails for all decisions and trades
- **Health Monitoring**: Real-time system health checks and performance metrics

## 📋 Requirements

- Python 3.8+
- Alpaca paper trading account
- Internet connection for GDELT API access

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd StockTraderV2
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create environment configuration**:
   ```bash
   python main.py --create-env
   ```

4. **Edit the `.env` file** with your Alpaca API credentials:
   ```bash
   # Get your API keys from: https://alpaca.markets/
   ALPACA_API_KEY=your_alpaca_api_key_here
   ALPACA_SECRET_KEY=your_alpaca_secret_key_here
   ```

## 🚦 Quick Start

1. **Run system check**:
   ```bash
   python main.py --check
   ```

2. **Start in dry-run mode** (recommended first):
   ```bash
   python main.py --dry-run
   ```

3. **Start live trading**:
   ```bash
   python main.py
   ```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ALPACA_API_KEY` | Your Alpaca API key | Required |
| `ALPACA_SECRET_KEY` | Your Alpaca secret key | Required |
| `MAX_POSITION_SIZE` | Maximum position size in USD | 1000 |
| `STOP_LOSS_PERCENTAGE` | Stop loss percentage | 0.05 (5%) |
| `TAKE_PROFIT_PERCENTAGE` | Take profit percentage | 0.10 (10%) |
| `LOG_LEVEL` | Logging level | INFO |
| `TRADING_ENABLED` | Enable actual trading | true |

### Monitored Stocks

By default, the system monitors these major stocks:
- AAPL (Apple)
- MSFT (Microsoft) 
- GOOGL (Alphabet)
- AMZN (Amazon)
- TSLA (Tesla)
- META (Meta)
- NVDA (NVIDIA)
- NFLX (Netflix)
- BABA (Alibaba)
- V (Visa)

## 🔄 How It Works

### 1. Data Collection
- Queries GDELT API for recent news and events
- Filters for company-specific content
- Respects API rate limits with intelligent throttling

### 2. Sentiment Analysis  
- Processes article sentiment scores
- Aggregates sentiment over rolling time windows
- Calculates sentiment momentum and acceleration

### 3. Volatility Detection
- Uses statistical and ML-based anomaly detection
- Combines multiple volatility indicators
- Generates confidence-weighted signals

### 4. Alert Generation
- Creates multi-level alerts (LOW, MEDIUM, HIGH, CRITICAL)
- Includes trading recommendations and risk assessments
- Implements cooldown periods to prevent spam

### 5. Trading Decisions
- Evaluates alerts against risk management rules
- Calculates optimal position sizes
- Sets stop-loss and take-profit levels

### 6. Trade Execution
- Executes trades via Alpaca paper trading API
- Monitors order status and portfolio positions
- Maintains comprehensive audit logs

## 📊 Monitoring & Logging

### Log Files
- `logs/main/` - Main application logs
- `logs/trades/` - All trading activity  
- `logs/alerts/` - Alert generation history
- `logs/errors/` - Error tracking and debugging
- `logs/performance/` - System performance metrics

### System Health
The system continuously monitors:
- API connectivity (GDELT, Alpaca)
- Data processing pipeline health
- Portfolio risk metrics
- System resource usage
- Error rates and recovery

## 🛡️ Risk Management

### Position Limits
- Maximum single position: 10% of portfolio
- Maximum daily loss: 5% of portfolio
- Stop after 3 consecutive losses
- Cooldown periods between trades

### Risk Scoring
- Comprehensive risk assessment for each trade
- Portfolio-level risk monitoring
- Automatic trading halt on risk threshold breach

### Safety Features
- Paper trading only (no real money at risk)
- Dry-run mode for testing
- Manual override capabilities
- Comprehensive error handling

## 🔧 Advanced Usage

### Custom Configuration
You can customize the system behavior by modifying `src/config.py`:

```python
# Example: Adjust sentiment thresholds
sentiment_threshold_high = 0.8  # More conservative
sentiment_threshold_low = -0.8

# Example: Change analysis frequency  
analysis_interval_minutes = 10  # More frequent analysis
```

### Adding New Stocks
To monitor additional stocks, update the `stock_tickers` list in `src/config.py`:

```python
stock_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
    "NVDA", "META", "NFLX", "BABA", "V",
    "JPM", "JNJ", "WMT"  # Add new tickers
]
```

## 📈 Performance Metrics

The system tracks comprehensive metrics:
- Total trades executed
- Win rate percentage
- Average return per trade
- Maximum drawdown
- Sharpe ratio
- Portfolio value over time

## 🐛 Troubleshooting

### Common Issues

1. **"GDELT API connection failed"**
   - Check internet connectivity
   - Verify GDELT API is accessible
   - Check rate limiting

2. **"Alpaca API connection failed"**  
   - Verify API keys in `.env` file
   - Ensure paper trading account is active
   - Check Alpaca API status

3. **"No data for ticker"**
   - Some tickers may have limited news coverage
   - Check if ticker symbol is correct
   - Verify company name mapping

### Debug Mode
Run with debug logging for detailed troubleshooting:
```bash
LOG_LEVEL=DEBUG python main.py --dry-run
```

## 📚 Architecture

### Core Components
- **GDELT Client**: News data retrieval with rate limiting
- **Data Processor**: Sentiment aggregation and analysis
- **Volatility Analyzer**: Statistical anomaly detection
- **Alert System**: Multi-level alert generation
- **Trading Engine**: Decision making and risk management
- **Alpaca Client**: Trade execution and portfolio management
- **Automation System**: Orchestration and monitoring

### Data Flow
```
GDELT API → Data Processing → Volatility Analysis → Alert Generation → Trading Decision → Trade Execution → Monitoring
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## ⚠️ Disclaimers

- **This system is for educational and research purposes only**
- **Past performance does not guarantee future results**
- **Always test thoroughly in paper trading before considering real money**
- **The authors are not responsible for any financial losses**
- **Sentiment analysis and automated trading carry inherent risks**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs in the `logs/` directory
3. Open an issue on GitHub with detailed error information

---

**Happy Trading! 📈🤖**