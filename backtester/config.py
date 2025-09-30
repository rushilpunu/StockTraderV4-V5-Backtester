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

    def __post_init__(self) -> None:
        if self.bot_variants is None:
            self.bot_variants = ["traderv4"]
        if self.max_workers is not None and self.max_workers <= 0:
            self.max_workers = None


__all__ = ["BacktestConfig"]
