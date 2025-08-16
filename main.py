#!/usr/bin/env python3
"""
Automated Trading System - Main Entry Point

A comprehensive automated trading system that uses GDELT's event and sentiment data
to detect high-volatility triggers for major stock tickers, generate internal alerts,
and execute trades via Alpaca's paper trading API.

Usage:
    python main.py [--config CONFIG_FILE] [--dry-run] [--help]

Environment Variables:
    ALPACA_API_KEY: Your Alpaca API key
    ALPACA_SECRET_KEY: Your Alpaca secret key
    LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
    TRADING_ENABLED: Enable/disable actual trading (true/false)
"""

import asyncio
import argparse
import signal
import sys
import os
from pathlib import Path
from typing import Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.automation_system import AutomationSystem
from src.logging_config import (
    logging_setup, error_handler, log_startup_info, log_shutdown_info, LoggedOperation
)
from src.config import config
from loguru import logger


class TradingSystemManager:
    """Main system manager for the automated trading system."""
    
    def __init__(self):
        self.automation_system: Optional[AutomationSystem] = None
        self.shutdown_requested = False
    
    async def start(self, dry_run: bool = False):
        """Start the trading system."""
        try:
            with LoggedOperation("system_startup", "main"):
                logger.info("🚀 Starting Automated Trading System")
                
                # Log startup information
                log_startup_info()
                
                # Display configuration
                self._display_configuration(dry_run)
                
                # Initialize automation system
                self.automation_system = AutomationSystem()
                
                # Setup signal handlers for graceful shutdown
                self._setup_signal_handlers()
                
                # Start the automation system
                if dry_run:
                    logger.info("DRY RUN MODE - No actual trades will be executed")
                    # Temporarily disable trading for dry run
                    original_trading_enabled = config.trading_enabled
                    config.trading_enabled = False
                    
                    try:
                        await self.automation_system.start()
                    finally:
                        config.trading_enabled = original_trading_enabled
                else:
                    await self.automation_system.start()
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
            await self.shutdown()
        except Exception as e:
            error_id = error_handler.handle_exception(
                e, "system_startup", critical=True
            )
            logger.critical(f"System startup failed with error ID: {error_id}")
            sys.exit(1)
    
    async def shutdown(self):
        """Graceful shutdown of the trading system."""
        if self.shutdown_requested:
            return
        
        self.shutdown_requested = True
        
        try:
            with LoggedOperation("system_shutdown", "main"):
                logger.info("🛑 Shutting down Automated Trading System")
                
                if self.automation_system:
                    await self.automation_system.stop()
                
                # Log shutdown information
                log_shutdown_info()
                
                logger.info("✅ System shutdown completed")
                
        except Exception as e:
            error_id = error_handler.handle_exception(
                e, "system_shutdown", critical=True
            )
            logger.error(f"Error during shutdown with error ID: {error_id}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _display_configuration(self, dry_run: bool):
        """Display current configuration."""
        logger.info("📋 System Configuration:")
        logger.info(f"  • Trading Enabled: {config.trading_enabled and not dry_run}")
        logger.info(f"  • Analysis Interval: {config.analysis_interval_minutes} minutes")
        logger.info(f"  • Stock Tickers: {', '.join(config.stock_tickers)}")
        logger.info(f"  • Max Position Size: ${config.max_position_size}")
        logger.info(f"  • Stop Loss: {config.stop_loss_percentage:.1%}")
        logger.info(f"  • Take Profit: {config.take_profit_percentage:.1%}")
        logger.info(f"  • Log Level: {config.log_level}")
        
        if dry_run:
            logger.info("  • Mode: DRY RUN (no actual trades)")
        else:
            logger.info("  • Mode: LIVE TRADING")


def validate_environment():
    """Validate required environment variables and configuration."""
    errors = []
    
    # Check required environment variables
    required_env_vars = [
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY"
    ]
    
    for var in required_env_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")
    
    # Check if configuration is valid
    try:
        # Test configuration access
        _ = config.stock_tickers
        _ = config.max_position_size
    except Exception as e:
        errors.append(f"Configuration error: {e}")
    
    # Check log directory permissions
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        test_file = log_dir / "test_permissions.tmp"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        errors.append(f"Cannot write to logs directory: {e}")
    
    if errors:
        logger.error("Environment validation failed:")
        for error in errors:
            logger.error(f"  • {error}")
        return False
    
    logger.info("✅ Environment validation passed")
    return True


def create_sample_env_file():
    """Create a sample environment file."""
    sample_env_content = """# Alpaca API Configuration
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Trading Configuration
MAX_POSITION_SIZE=1000
STOP_LOSS_PERCENTAGE=0.05
TAKE_PROFIT_PERCENTAGE=0.10
COOLDOWN_MINUTES=30

# GDELT Configuration
GDELT_RATE_LIMIT_REQUESTS_PER_MINUTE=60

# Monitoring Configuration
LOG_LEVEL=INFO
TRADING_ENABLED=true
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        env_file.write_text(sample_env_content)
        logger.info(f"Created sample environment file: {env_file}")
        logger.info("Please edit .env file with your actual API keys")
        return True
    
    return False


async def run_system_check():
    """Run comprehensive system check."""
    logger.info("🔍 Running system check...")
    
    try:
        # Import and test all major components
        from src.gdelt_client import GDELTClient
        from src.alpaca_client import AlpacaClient
        from src.data_processor import DataProcessor
        from src.volatility_analyzer import VolatilityAnalyzer
        from src.alert_system import AlertSystem
        from src.trading_engine import TradingEngine
        
        # Test GDELT client
        logger.info("  • Testing GDELT API connection...")
        async with GDELTClient() as gdelt_client:
            from datetime import datetime, timedelta
            test_data = await gdelt_client.get_doc_search(
                query="AAPL",
                start_date=datetime.utcnow() - timedelta(hours=1),
                end_date=datetime.utcnow(),
                max_records=1
            )
            if test_data:
                logger.info("    ✅ GDELT API connection successful")
            else:
                logger.warning("    ⚠️  GDELT API returned no data")
        
        # Test Alpaca client
        logger.info("  • Testing Alpaca API connection...")
        alpaca_client = AlpacaClient()
        account_info = await alpaca_client.get_account_info()
        if account_info:
            logger.info("    ✅ Alpaca API connection successful")
            logger.info(f"    Portfolio Value: ${account_info.get('portfolio_value', 0):.2f}")
        else:
            logger.error("    ❌ Alpaca API connection failed")
            return False
        
        # Test other components
        logger.info("  • Testing data processing components...")
        data_processor = DataProcessor()
        volatility_analyzer = VolatilityAnalyzer()
        alert_system = AlertSystem()
        trading_engine = TradingEngine()
        
        logger.info("    ✅ All components initialized successfully")
        
        logger.info("✅ System check completed successfully")
        return True
        
    except Exception as e:
        error_id = error_handler.handle_exception(e, "system_check", critical=True)
        logger.error(f"❌ System check failed with error ID: {error_id}")
        return False


def display_banner():
    """Display system banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        Automated Trading System                              ║
║                                                                              ║
║  📈 GDELT Sentiment Analysis + Alpaca Paper Trading                         ║
║  🤖 Automated volatility detection and trade execution                      ║
║  🛡️  Comprehensive risk management and monitoring                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Automated Trading System using GDELT and Alpaca",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (optional)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no actual trades)"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run system check and exit"
    )
    
    parser.add_argument(
        "--create-env",
        action="store_true",
        help="Create sample .env file and exit"
    )
    
    args = parser.parse_args()
    
    # Display banner
    display_banner()
    
    # Handle special commands
    if args.create_env:
        if create_sample_env_file():
            sys.exit(0)
        else:
            logger.info(".env file already exists")
            sys.exit(1)
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Use --create-env to create sample configuration.")
        sys.exit(1)
    
    # Run system check if requested
    if args.check:
        success = await run_system_check()
        sys.exit(0 if success else 1)
    
    # Start the main system
    system_manager = TradingSystemManager()
    
    try:
        await system_manager.start(dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        error_id = error_handler.handle_exception(e, "main", critical=True)
        logger.critical(f"System failed with error ID: {error_id}")
        sys.exit(1)
    finally:
        await system_manager.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program interrupted")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
