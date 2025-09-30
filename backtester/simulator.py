"""Portfolio and trade simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from Traderv4.funcs import TradeDecision


@dataclass
class ExecutedTrade:
    ticker: str
    action: str
    price: float
    quantity: float
    notional: float
    timestamp: str
    reason: str


class PortfolioSimulator:
    def __init__(self, starting_cash: float) -> None:
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: Dict[str, float] = {}
        self.trade_log: List[ExecutedTrade] = []
        self.equity_curve: List[float] = []
        self.unrealized: Dict[str, float] = {}

    def step(self, timestamp: str, prices: Dict[str, float]) -> None:
        equity = self.cash
        for ticker, shares in self.positions.items():
            price = prices.get(ticker)
            if price is None or price <= 0:
                continue
            equity += shares * price
            self.unrealized[ticker] = shares * price
        self.equity_curve.append(equity)

    def execute(self, decision: TradeDecision, price: float, timestamp: str) -> float:
        if decision.action == "HOLD" or price <= 0 or decision.notional <= 0:
            return 0.0
        ticker = decision.ticker
        notional = decision.notional
        shares = decision.quantity if decision.quantity is not None else None
        if decision.action == "BUY":
            if shares is None:
                notional = min(notional, self.cash)
                if notional <= 0:
                    return 0.0
                shares = notional / price
            else:
                cost = shares * price
                if cost > self.cash:
                    shares = min(self.cash / price, shares)
                notional = shares * price
            self.cash -= notional
            self.positions[ticker] = self.positions.get(ticker, 0.0) + shares
        else:  # SELL
            if shares is None:
                shares = notional / price
            proceeds = shares * price
            self.cash += proceeds
            self.positions[ticker] = self.positions.get(ticker, 0.0) - shares
        self.trade_log.append(
            ExecutedTrade(
                ticker=ticker,
                action=decision.action,
                price=price,
                quantity=shares,
                notional=shares * price,
                timestamp=timestamp,
                reason=decision.reason,
            )
        )
        return shares

    def finalize(self, prices: Dict[str, float], timestamp: Optional[str] = None) -> List[ExecutedTrade]:
        forced_trades: List[ExecutedTrade] = []
        liquidation_ts = timestamp or "forced_liquidation"
        for ticker, shares in list(self.positions.items()):
            price = prices.get(ticker)
            if price is None:
                continue
            quantity = abs(shares)
            if quantity <= 0:
                self.positions[ticker] = 0.0
                self.unrealized.pop(ticker, None)
                continue
            notional = quantity * price
            if shares > 0:
                action = "SELL"
                self.cash += notional
            else:
                action = "BUY"
                self.cash -= notional
            trade = ExecutedTrade(
                ticker=ticker,
                action=action,
                price=price,
                quantity=quantity,
                notional=notional,
                timestamp=liquidation_ts,
                reason="forced liquidation",
            )
            self.trade_log.append(trade)
            forced_trades.append(trade)
            self.positions[ticker] = 0.0
            self.unrealized.pop(ticker, None)
        self.step(liquidation_ts, prices)
        return forced_trades

    def realized_pnl(self) -> float:
        return self.equity_curve[-1] - self.starting_cash if self.equity_curve else 0.0


__all__ = ["PortfolioSimulator", "ExecutedTrade"]
