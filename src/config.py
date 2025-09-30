"""Configuration management for the automated trading system."""

import os
from typing import List
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from config.credentials import load_alpaca_credentials

# Load environment variables only if not already loaded
if not os.getenv("ALPACA_API_KEY"):
    load_dotenv()

_DEFAULT_CREDENTIALS = load_alpaca_credentials(required=False)


class TradingConfig(BaseSettings):
    """Trading configuration settings."""
    
    # Alpaca API Configuration
    alpaca_api_key: str = Field(
        default=_DEFAULT_CREDENTIALS.api_key if _DEFAULT_CREDENTIALS else "",
        env="ALPACA_API_KEY",
    )
    alpaca_secret_key: str = Field(
        default=_DEFAULT_CREDENTIALS.api_secret if _DEFAULT_CREDENTIALS else "",
        env="ALPACA_SECRET_KEY",
    )
    alpaca_base_url: str = Field(
        default=_DEFAULT_CREDENTIALS.base_url if _DEFAULT_CREDENTIALS else "https://paper-api.alpaca.markets",
        env="ALPACA_BASE_URL"
    )
    
    # Trading Parameters
    max_position_size: float = Field(default=1000.0, env="MAX_POSITION_SIZE")
    stop_loss_percentage: float = Field(default=0.0025, env="STOP_LOSS_PERCENTAGE")
    take_profit_percentage: float = Field(default=0.0025, env="TAKE_PROFIT_PERCENTAGE")
    cooldown_minutes: int = Field(default=8, env="COOLDOWN_MINUTES")
    
    # GDELT Configuration
    gdelt_rate_limit_requests_per_minute: int = Field(
        default=60, env="GDELT_RATE_LIMIT_REQUESTS_PER_MINUTE"
    )
    
    # Monitoring Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    trading_enabled: bool = Field(default=True, env="TRADING_ENABLED")
    
    # Aggressive Mode Configuration
    aggressive_mode_enabled: bool = Field(default=True, env="AGGRESSIVE_MODE_ENABLED")
    min_trades_per_cycle: int = Field(default=1, env="MIN_TRADES_PER_CYCLE")
    aggressive_confidence: float = Field(default=0.7, env="AGGRESSIVE_CONFIDENCE")
    aggressive_min_trade_value: float = Field(default=50.0, env="AGGRESSIVE_MIN_TRADE_VALUE")
    aggressive_prefer_alerts: bool = Field(default=True, env="AGGRESSIVE_PREFER_ALERTS")
    
    # Stock Tickers to Monitor (Top 10 large caps)
    stock_tickers: List[str] = [
        # Tech Giants
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "BABA", "V"
    ]
    
    # Sentiment Analysis Thresholds
    sentiment_threshold_high: float = 0.5
    sentiment_threshold_low: float = -0.5
    sentiment_change_threshold: float = 0.15
    event_count_threshold: int = 5
    
    # Time Windows (in minutes)
    rolling_window_minutes: int = 60
    analysis_interval_minutes: int = 3

    # Exit Enforcement
    enforce_config_exits: bool = Field(default=True, env="ENFORCE_CONFIG_EXITS")
    
    @model_validator(mode="after")
    def _validate_credentials(self):
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            raise ValueError("Alpaca credentials are missing. Set environment variables or update config.")
        return self

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global configuration instance - only create if not already loaded
try:
    config = TradingConfig()
except Exception:
    # If this fails, it means we're in the small capital trader context
    # and should use the SmallCapitalTradingConfig instead
    config = None
