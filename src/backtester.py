"""Lightweight backtesting utilities for the trading engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from loguru import logger

from .alert_system import TradingAlert
from .trading_engine import TradeAction, TradingDecision, TradingEngine


@dataclass
class PriceBar:
    """Simple OHLC price container used for simulations."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class BacktestTrade:
    """Details about an individual simulated trade."""

    ticker: str
    action: TradeAction
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    return_pct: float
    outcome: str  # win, loss, or breakeven


@dataclass
class BacktestResult:
    """Aggregated metrics for a backtest run."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_return_pct: float
    total_pnl: float
    accuracy_rating: str
    trades: List[BacktestTrade] = field(default_factory=list)


class Backtester:
    """Run basic historical simulations on top of the trading engine."""

    def __init__(self, engine: Optional[TradingEngine] = None):
        self.engine = engine or TradingEngine()

    async def run_backtest_async(
        self,
        alerts: Iterable[TradingAlert],
        price_history: Dict[str, List[PriceBar]],
    ) -> BacktestResult:
        """Asynchronously run the backtest for the provided alerts and prices."""

        trades: List[BacktestTrade] = []
        winning_trades = 0
        losing_trades = 0
        returns: List[float] = []

        alerts_sorted = sorted(alerts, key=lambda alert: alert.timestamp)

        for alert in alerts_sorted:
            price_bars = price_history.get(alert.ticker)
            if not price_bars:
                logger.debug(f"Missing price history for {alert.ticker}; skipping alert")
                continue

            entry_index = self._find_entry_index(alert.timestamp, price_bars)
            if entry_index is None:
                logger.debug(
                    f"No price data after alert timestamp for {alert.ticker}; skipping"
                )
                continue

            current_price = price_bars[entry_index].close
            decision = await self.engine.evaluate_trading_decision(alert, current_price)

            if not decision or decision.action not in {TradeAction.BUY, TradeAction.SELL}:
                continue

            trade = self._simulate_trade(decision, price_bars, entry_index)
            if trade is None:
                continue

            trades.append(trade)
            returns.append(trade.return_pct)

            if trade.outcome == "win":
                winning_trades += 1
            elif trade.outcome == "loss":
                losing_trades += 1

        win_rate = (winning_trades / len(trades) * 100) if trades else 0.0
        average_return_pct = sum(returns) / len(returns) if returns else 0.0
        total_pnl = sum(trade.pnl for trade in trades)
        accuracy_rating = TradingEngine._calculate_accuracy_rating(win_rate)

        return BacktestResult(
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            average_return_pct=average_return_pct,
            total_pnl=total_pnl,
            accuracy_rating=accuracy_rating,
            trades=trades,
        )

    def run_backtest(
        self,
        alerts: Iterable[TradingAlert],
        price_history: Dict[str, List[PriceBar]],
    ) -> BacktestResult:
        """Convenience synchronous wrapper for the asynchronous backtest."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError(
                "run_backtest cannot be called from an active event loop. "
                "Use run_backtest_async instead."
            )

        return asyncio.run(self.run_backtest_async(alerts, price_history))

    @staticmethod
    def _find_entry_index(timestamp: datetime, price_bars: List[PriceBar]) -> Optional[int]:
        for idx, bar in enumerate(price_bars):
            if bar.timestamp >= timestamp:
                return idx
        return None

    def _simulate_trade(
        self,
        decision: TradingDecision,
        price_bars: List[PriceBar],
        entry_index: int,
    ) -> Optional[BacktestTrade]:
        quantity = max(decision.quantity, 0)
        if quantity == 0:
            return None

        entry_bar = price_bars[entry_index]
        entry_price = entry_bar.close
        direction = 1 if decision.action == TradeAction.BUY else -1

        stop_loss = decision.stop_loss
        take_profit = decision.take_profit

        exit_price = entry_price
        exit_time = entry_bar.timestamp
        outcome = "breakeven"

        for bar in price_bars[entry_index + 1 :]:
            if direction == 1:
                if stop_loss and bar.low <= stop_loss:
                    exit_price = stop_loss
                    exit_time = bar.timestamp
                    outcome = "loss"
                    break
                if take_profit and bar.high >= take_profit:
                    exit_price = take_profit
                    exit_time = bar.timestamp
                    outcome = "win"
                    break
            else:
                if stop_loss and bar.high >= stop_loss:
                    exit_price = stop_loss
                    exit_time = bar.timestamp
                    outcome = "loss"
                    break
                if take_profit and bar.low <= take_profit:
                    exit_price = take_profit
                    exit_time = bar.timestamp
                    outcome = "win"
                    break
        else:
            final_bar = price_bars[-1]
            exit_price = final_bar.close
            exit_time = final_bar.timestamp
            gain = (exit_price - entry_price) * direction
            if gain > 0:
                outcome = "win"
            elif gain < 0:
                outcome = "loss"

        pnl = (exit_price - entry_price) * quantity * direction
        return_pct = ((exit_price - entry_price) / entry_price) * 100 * direction

        return BacktestTrade(
            ticker=decision.ticker,
            action=decision.action,
            entry_time=entry_bar.timestamp,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            return_pct=return_pct,
            outcome=outcome,
        )
