"""Configuration helpers for the backtesting suite."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class BacktestConfig:
    tickers: List[str]
    start: datetime
    end: datetime
    starting_cash: float = 100_000.0
    sentiment_window_minutes: int = 60
    bar_timeframe: str = "15Min"
    bot_variants: List[str] = None
    use_vader: bool = True
    use_finbert: bool = False
    use_keybert: bool = False
    record_trace: bool = False
    max_workers: Optional[int] = None
    risk_profile: str = "balanced"
    aggressiveness: Optional[float] = None
    entry_signal_bias: Optional[float] = None
    max_trade_leverage: Optional[float] = None
    entry_threshold: Optional[float] = None
    exit_threshold: Optional[float] = None
    cooldown_minutes: Optional[int] = None
    max_positions_override: Optional[int] = None
    max_capital_fraction_override: Optional[float] = None

    def __post_init__(self) -> None:
        if self.bot_variants is None:
            self.bot_variants = ["traderv4"]
        if self.max_workers is not None and self.max_workers <= 0:
            self.max_workers = None
        if not self.risk_profile:
            self.risk_profile = "balanced"
        self.risk_profile = self.risk_profile.lower()


__all__ = ["BacktestConfig"]
