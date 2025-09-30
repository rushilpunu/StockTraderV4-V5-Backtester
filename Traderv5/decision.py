"""Model-driven decision engine for Trader V5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import math

from Traderv4.funcs import (
    AccountSnapshot,
    RiskConfig,
    TradeDecision,
    TradeFrequencyTracker,
)

from Traderv5.model.predictor import ModelPredictor


@dataclass
class SignalContext:
    probability_long: float
    features: Dict[str, float]

    @property
    def signal_strength(self) -> float:
        return float(self.probability_long - 0.5)


class ModelDecisionEngine:
    def __init__(
        self,
        predictor: ModelPredictor,
        risk: RiskConfig,
        trade_tracker: TradeFrequencyTracker,
    ) -> None:
        self.predictor = predictor
        self.risk = risk
        self.trade_tracker = trade_tracker
        self.last_trade_at: Dict[str, datetime] = {}

    def decide(
        self,
        ticker: str,
        features: Dict[str, float],
        price_snapshot: Dict[str, Optional[float]],
        account: AccountSnapshot,
        *,
        position: Optional[Dict[str, Any]] = None,
        open_positions: int = 0,
    ) -> TradeDecision:
        price_raw = price_snapshot.get("close") if price_snapshot else None
        try:
            price = float(price_raw) if price_raw is not None else 0.0
        except (TypeError, ValueError):
            price = 0.0
        if price <= 0:
            return self._hold(ticker, "missing price data")

        probability = float(self.predictor.predict_proba(features))
        signal = SignalContext(probability_long=probability, features=features)
        strength = signal.signal_strength

        pos_side, quantity, market_value = self._parse_position(position, price)

        available_funds, per_position_cap = self._position_budget(account, market_value)

        # Exit logic for open positions
        if pos_side == "LONG" and quantity > 0:
            if strength <= self.risk.exit_sentiment_threshold:
                notional = market_value if market_value > 0 else quantity * price
                return self._exit_trade(
                    ticker,
                    action="SELL",
                    confidence=abs(strength),
                    notional=notional,
                    quantity=quantity,
                    reason="model confidence faded",
                    metadata={"probability_long": f"{probability:.3f}"},
                )
            return self._hold(ticker, "maintain long position")

        if pos_side == "SHORT" and quantity > 0:
            if strength >= -self.risk.exit_sentiment_threshold:
                notional = market_value if market_value > 0 else quantity * price
                return self._exit_trade(
                    ticker,
                    action="BUY",
                    confidence=abs(strength),
                    notional=notional,
                    quantity=quantity,
                    reason="model confidence faded",
                    metadata={"probability_long": f"{probability:.3f}"},
                )
            return self._hold(ticker, "maintain short position")

        # Entry gates
        strength_threshold = max(self.risk.entry_sentiment_threshold, 0.05)
        cooldown_active = self._cooldown_active(ticker)
        if cooldown_active:
            return self._hold(ticker, "cooldown active")
        if open_positions >= self.risk.max_positions:
            return self._hold(ticker, "max positions reached")
        if not self.trade_tracker.can_enter():
            return self._hold(ticker, "day trade limit reached")

        if strength >= strength_threshold:
            notional = self._size_trade(available_funds, per_position_cap, strength)
            if notional <= 0:
                return self._hold(ticker, "insufficient capital")
            return self._enter_trade(
                ticker,
                action="BUY",
                confidence=abs(strength),
                notional=notional,
                price=price,
                reason="model long conviction",
                metadata={"probability_long": f"{probability:.3f}"},
            )

        if strength <= -strength_threshold and self.risk.allow_shorting:
            notional = self._size_trade(available_funds, per_position_cap, abs(strength))
            if notional <= 0:
                return self._hold(ticker, "insufficient capital")
            return self._enter_trade(
                ticker,
                action="SELL",
                confidence=abs(strength),
                notional=notional,
                price=price,
                reason="model short conviction",
                metadata={"probability_long": f"{probability:.3f}"},
            )

        return self._hold(ticker, "signal below threshold", metadata={"probability_long": f"{probability:.3f}"})

    def _parse_position(self, position: Optional[Dict[str, Any]], price: float) -> tuple[str, float, float]:
        if not position:
            return "FLAT", 0.0, 0.0
        try:
            quantity = abs(float(position.get("quantity", 0.0)))
        except (TypeError, ValueError):
            quantity = 0.0
        try:
            market_value = abs(float(position.get("market_value", quantity * price)))
        except (TypeError, ValueError):
            market_value = quantity * price
        raw_side = str(position.get("side", "")).upper()
        if quantity <= 0:
            return "FLAT", 0.0, 0.0
        if raw_side in {"LONG", "BUY"}:
            side = "LONG"
        elif raw_side in {"SHORT", "SELL"}:
            side = "SHORT"
        else:
            side = "LONG"
        return side, quantity, market_value

    def _position_budget(self, account: AccountSnapshot, market_value: float) -> tuple[float, float]:
        equity = account.equity or account.portfolio_value or account.cash
        cash = max(account.cash, 0.0)
        buying_power = max(account.buying_power, cash)
        available_funds = max(0.0, min(cash if cash > 0 else buying_power, buying_power))
        per_position_cap = (equity or buying_power or cash) * self.risk.max_capital_fraction if equity else available_funds
        if self.risk.max_position_value is not None:
            per_position_cap = min(per_position_cap, self.risk.max_position_value)
        if market_value > per_position_cap:
            per_position_cap = max(per_position_cap, market_value)
        return available_funds, per_position_cap

    def _size_trade(self, available_funds: float, per_position_cap: float, strength: float) -> float:
        base = min(available_funds, per_position_cap)
        scaled = base * min(max(strength, 0.05), 1.0)
        return float(scaled)

    def _cooldown_active(self, ticker: str) -> bool:
        last_trade = self.last_trade_at.get(ticker)
        if not last_trade:
            return False
        return datetime.utcnow() - last_trade < timedelta(minutes=self.risk.cooldown_minutes)

    def _hold(self, ticker: str, reason: str, metadata: Optional[Dict[str, str]] = None) -> TradeDecision:
        return TradeDecision(
            ticker=ticker,
            action="HOLD",
            confidence=0.0,
            notional=0.0,
            time_in_force="gtc",
            stop_loss=None,
            take_profit=None,
            reason=reason,
            intent="hold",
            metadata=metadata or {},
        )

    def _enter_trade(
        self,
        ticker: str,
        *,
        action: str,
        confidence: float,
        notional: float,
        price: float,
        reason: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> TradeDecision:
        if notional <= 0:
            return self._hold(ticker, "invalid notional")
        quantity = notional / price if price > 0 else None
        decision = TradeDecision(
            ticker=ticker,
            action=action,
            confidence=float(min(max(confidence, 0.0), 1.0)),
            notional=float(notional),
            time_in_force="gtc",
            stop_loss=None,
            take_profit=None,
            reason=reason,
            intent="entry",
            quantity=quantity,
            metadata=metadata or {},
        )
        self.last_trade_at[ticker] = datetime.utcnow()
        return decision

    def _exit_trade(
        self,
        ticker: str,
        *,
        action: str,
        confidence: float,
        notional: float,
        quantity: float,
        reason: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> TradeDecision:
        decision = TradeDecision(
            ticker=ticker,
            action=action,
            confidence=float(min(max(confidence, 0.0), 1.0)),
            notional=float(notional),
            time_in_force="gtc",
            stop_loss=None,
            take_profit=None,
            reason=reason,
            intent="exit",
            quantity=quantity,
            metadata=metadata or {},
        )
        self.last_trade_at[ticker] = datetime.utcnow()
        return decision


__all__ = ["ModelDecisionEngine", "SignalContext"]
