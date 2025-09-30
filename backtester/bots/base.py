"""Bot adapter interface for backtesting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from Traderv4.funcs import TradeDecision

from backtester.config import BacktestConfig


class BaseBot(ABC):
    name: str = "base"

    def setup(self, config: BacktestConfig, baseline: Dict[str, float]) -> None:
        """Optional hook invoked before processing a ticker."""

    @abstractmethod
    def on_snapshot(
        self,
        timestamp: str,
        ticker: str,
        price: float,
        sentiment_snapshot,
        portfolio_cash: float,
        portfolio_equity: float,
        baseline: Dict[str, float],
        position: Optional[Dict[str, Any]],
        open_positions: int,
    ) -> TradeDecision:
        """Return trading decision for a given ticker/snapshot."""

    def on_cycle_end(self) -> None:
        """Hook invoked when backtest completes."""

    def on_trade_filled(
        self,
        *,
        decision: TradeDecision,
        timestamp: str,
        price: float,
        quantity: float,
    ) -> None:
        """Hook invoked when the simulator executes a trade."""


BOT_REGISTRY: Dict[str, BaseBot] = {}


def register_bot(bot_cls):
    instance = bot_cls()
    BOT_REGISTRY[instance.name] = instance
    return bot_cls


__all__ = ["BaseBot", "BOT_REGISTRY", "register_bot"]
