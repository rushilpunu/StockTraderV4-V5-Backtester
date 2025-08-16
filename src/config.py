"""Configuration management for the automated trading system."""

import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TradingConfig(BaseSettings):
    """Trading configuration settings."""
    
    # Alpaca API Configuration
    alpaca_api_key: str = Field(..., env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(..., env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        env="ALPACA_BASE_URL"
    )
    
    # Trading Parameters
    max_position_size: float = Field(default=1000.0, env="MAX_POSITION_SIZE")
    stop_loss_percentage: float = Field(default=0.05, env="STOP_LOSS_PERCENTAGE")
    take_profit_percentage: float = Field(default=0.10, env="TAKE_PROFIT_PERCENTAGE")
    cooldown_minutes: int = Field(default=30, env="COOLDOWN_MINUTES")
    
    # GDELT Configuration
    gdelt_rate_limit_requests_per_minute: int = Field(
        default=60, env="GDELT_RATE_LIMIT_REQUESTS_PER_MINUTE"
    )
    
    # Monitoring Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    trading_enabled: bool = Field(default=True, env="TRADING_ENABLED")
    
    # Stock Tickers to Monitor
    stock_tickers: List[str] = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "BABA", "V"
    ]
    
    # Sentiment Analysis Thresholds
    sentiment_threshold_high: float = 0.7
    sentiment_threshold_low: float = -0.7
    sentiment_change_threshold: float = 0.3
    event_count_threshold: int = 10
    
    # Time Windows (in minutes)
    rolling_window_minutes: int = 60
    analysis_interval_minutes: int = 3
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global configuration instance
config = TradingConfig()
