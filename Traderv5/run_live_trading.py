#!/usr/bin/env python3
"""Live trading runner for TraderV5 with Alpaca paper trading."""

import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Change to parent directory so imports work
os.chdir(parent_dir)

from Traderv5.configuration import load_trading_parameters
from Traderv5.trader import ModelDrivenTrader, TraderV5Config
from Traderv5.model.predictor import ModelPredictor
from Traderv4.funcs import RiskConfig
from config.credentials import load_alpaca_credentials

# Set up logging
# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)

# Use current date for log file
from datetime import date
log_file = logs_dir / f"trading_{date.today().strftime('%Y-%m-%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(log_file))
    ]
)
logger = logging.getLogger(__name__)

# Global variables for graceful shutdown
trader_instance: Optional[ModelDrivenTrader] = None
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


def load_alpaca_client():
    """Load and configure Alpaca client for paper trading."""
    try:
        from alpaca_trade_api import REST
    except ImportError as exc:
        raise RuntimeError(
            "alpaca-trade-api library is required. Install with: pip install alpaca-trade-api"
        ) from exc
    
    # Load credentials - this will use paper trading by default
    creds = load_alpaca_credentials()
    
    # Ensure we're using paper trading URL
    paper_url = "https://paper-api.alpaca.markets"
    if "paper-api.alpaca.markets" not in creds.base_url:
        logger.warning(f"Using non-paper URL: {creds.base_url}. Paper trading recommended.")
    
    client = REST(
        key_id=creds.api_key,
        secret_key=creds.api_secret,
        base_url=creds.base_url
    )
    
    # Verify connection
    try:
        account = client.get_account()
        logger.info(f"Connected to Alpaca account: {account.account_number}")
        logger.info(f"Account status: {account.status}")
        logger.info(f"Buying power: ${float(account.buying_power):,.2f}")
        logger.info(f"Equity: ${float(account.equity):,.2f}")
        logger.info(f"Cash: ${float(account.cash):,.2f}")
        
        # Check if market is open
        clock = client.get_clock()
        logger.info(f"Market is {'OPEN' if clock.is_open else 'CLOSED'}")
        if not clock.is_open:
            logger.info(f"Next open: {clock.next_open}")
            logger.info(f"Next close: {clock.next_close}")
            
    except Exception as exc:
        logger.error(f"Failed to connect to Alpaca: {exc}")
        raise
    
    return client


def load_account_balance(alpaca_client):
    """Load current account balance."""
    try:
        account = alpaca_client.get_account()
        return float(account.cash)
    except Exception as exc:
        logger.error(f"Failed to get account balance: {exc}")
        return 0.0


def create_trader_config() -> TraderV5Config:
    """Create TraderV5 configuration from config file."""
    try:
        # Load configuration
        config_params = load_trading_parameters()
        
        # Convert to TraderV5Config
        risk_config = RiskConfig(
            max_capital_fraction=config_params.risk.risk_per_trade_pct,
            max_positions=5,  # Conservative for paper trading
            stop_loss_pct=config_params.risk.atr_stop_multiplier * 0.01,  # Approximate
            take_profit_pct=config_params.risk.take_profit_multiple * 0.01,  # Approximate
            cooldown_minutes=30,
            entry_sentiment_threshold=0.6,
            exit_sentiment_threshold=0.4,
        )
        
        trader_config = TraderV5Config(
            tickers=list(config_params.training.tickers),
            lookback_days=config_params.training.lookback_days,
            price_interval=config_params.training.price_interval,
            gdelt_timeline_minutes=config_params.training.timeline_minutes,
            gdelt_delay=config_params.data.gdelt_pause_seconds,
            sentiment_window_minutes=config_params.training.timeline_minutes,
            cycle_pause_seconds=300,  # 5 minutes between cycles
            risk=risk_config,
            label_horizon_minutes=config_params.training.label_horizon_minutes,
            positive_threshold=config_params.training.positive_threshold,
            negative_threshold=config_params.training.negative_threshold,
        )
        
        logger.info(f"Loaded configuration for {len(trader_config.tickers)} tickers")
        logger.info(f"Tickers: {', '.join(trader_config.tickers)}")
        
        return trader_config
        
    except Exception as exc:
        logger.error(f"Failed to load configuration: {exc}")
        raise


def check_model_availability():
    """Check if trained models are available."""
    try:
        predictor = ModelPredictor.load_default()
        logger.info("Trained models loaded successfully")
        return predictor
    except FileNotFoundError as exc:
        logger.error("Trained models not found!")
        logger.error("Please run the following to train models first:")
        logger.error("  cd Traderv5 && python model/training.py")
        raise
    except Exception as exc:
        logger.error(f"Failed to load models: {exc}")
        raise


def run_trading_cycle():
    """Run a single trading cycle."""
    global trader_instance, shutdown_requested
    
    if trader_instance is None:
        logger.error("Trader instance not initialized")
        return
    
    try:
        logger.info("Starting trading cycle...")
        cycle_start = datetime.utcnow()
        
        trader_instance.run_cycle()
        
        cycle_duration = datetime.utcnow() - cycle_start
        logger.info(f"Trading cycle completed in {cycle_duration.total_seconds():.1f} seconds")
        
    except Exception as exc:
        logger.error(f"Error in trading cycle: {exc}")
        logger.exception("Trading cycle failed")


def main():
    """Main trading loop."""
    global trader_instance, shutdown_requested
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting TraderV5 Live Trading System")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = create_trader_config()
        
        # Check models
        logger.info("Checking trained models...")
        predictor = check_model_availability()
        
        # Initialize Alpaca client
        logger.info("Connecting to Alpaca...")
        alpaca_client = load_alpaca_client()
        
        # Create trader instance
        logger.info("Initializing trader...")
        trader_instance = ModelDrivenTrader(
            config=config,
            alpaca_client=alpaca_client,
            balance_fetcher=lambda: load_account_balance(alpaca_client),
            predictor=predictor
        )
        
        logger.info("TraderV5 initialized successfully!")
        logger.info("Starting trading loop...")
        logger.info("Press Ctrl+C to stop trading gracefully")
        logger.info("-" * 60)
        
        # Main trading loop
        cycle_count = 0
        last_cycle_time = datetime.utcnow()
        
        while not shutdown_requested:
            try:
                current_time = datetime.utcnow()
                
                # Check if enough time has passed since last cycle
                time_since_last_cycle = current_time - last_cycle_time
                if time_since_last_cycle.total_seconds() >= config.cycle_pause_seconds:
                    cycle_count += 1
                    logger.info(f"=== Trading Cycle #{cycle_count} ===")
                    
                    run_trading_cycle()
                    last_cycle_time = current_time
                else:
                    # Wait a bit before checking again
                    time.sleep(30)  # Check every 30 seconds
                    
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
                shutdown_requested = True
                break
            except Exception as exc:
                logger.error(f"Error in main loop: {exc}")
                logger.exception("Main loop error")
                time.sleep(60)  # Wait a minute before retrying
        
        logger.info("Trading loop stopped")
        
    except Exception as exc:
        logger.error(f"Failed to start trading system: {exc}")
        logger.exception("Startup error")
        sys.exit(1)
    
    finally:
        logger.info("TraderV5 Live Trading System shutdown complete")


if __name__ == "__main__":
    main()
