"""Adapter that reuses TraderV4 decision engine for backtests."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime, timezone
import statistics
from typing import Any, Deque, Dict, Optional

from Traderv4.funcs import (
    AccountSnapshot,
    Alert,
    DecisionEngine,
    RiskConfig,
    TradeDecision,
    TradeFrequencyTracker,
    VolatilityThresholds,
)

from backtester.config import BacktestConfig

from .base import BaseBot, register_bot


@register_bot
class TraderV4Bot(BaseBot):
    name = "traderv4"

    def __init__(self) -> None:
        self.thresholds = VolatilityThresholds(min_articles=1, sentiment_spike=0.1, sentiment_extreme=0.25)
        self.tracker = TradeFrequencyTracker(max_day_trades=10_000)
        self.risk = RiskConfig()
        self.engine = DecisionEngine(self.risk, self.tracker)
        self._baseline: Dict[str, float] = {}
        self._open_entries: Dict[str, Deque[datetime]] = defaultdict(deque)
        self._starting_cash: float = 0.0
        self._tiny_notional_threshold: float = 0.0
        self._score_memory: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=24))

    def setup(self, config: BacktestConfig, baseline: Dict[str, float]) -> None:
        self._baseline = baseline
        self._starting_cash = float(config.starting_cash)
        base_fraction = min(0.5, max(0.05, self._starting_cash / 1_000_000))
        if self._starting_cash <= 25_000:
            base_fraction = max(base_fraction, 0.15)
        if self._starting_cash <= 5_000:
            base_fraction = max(base_fraction, 0.25)
        self.risk.max_capital_fraction = base_fraction
        self.risk.allow_shorting = True
        self._tiny_notional_threshold = max(25.0, self._starting_cash * 0.01)
        # Reset stateful components so consecutive runs don't inherit cooldown/PDT history
        self.tracker = TradeFrequencyTracker(max_day_trades=10_000)
        self.engine = DecisionEngine(self.risk, self.tracker)
        self._open_entries = defaultdict(deque)
        self._score_memory = defaultdict(lambda: deque(maxlen=24))
        # Tighten sentiment thresholds slightly to prefer higher conviction entries
        self.risk.entry_sentiment_threshold = max(0.12, self.risk.entry_sentiment_threshold)
        self.risk.exit_sentiment_threshold = max(0.08, self.risk.exit_sentiment_threshold)

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
        article_boost = math.log1p(sentiment_snapshot.article_count)
        sentiment_score = (
            0.35 * sentiment_snapshot.delta_15
            + 0.2 * (sentiment_snapshot.tone_15 - sentiment_snapshot.tone_60)
            + 0.1 * sentiment_snapshot.delta_60
            + 0.05 * sentiment_snapshot.delta_1440
            + 0.1 * sentiment_snapshot.vader
            + 0.1 * (sentiment_snapshot.finbert or 0.0)
            + 0.05 * sentiment_snapshot.source_score
            + 0.05 * article_boost
        )
        threshold = max(0.01, self.thresholds.sentiment_spike / 3)
        keywords = ",".join(sentiment_snapshot.keywords[:3])
        recent_scores = self._score_memory[ticker]
        recent_scores.append(sentiment_score)
        median_score = statistics.median(recent_scores) if recent_scores else 0.0
        mad = statistics.median([abs(score - median_score) for score in recent_scores]) if len(recent_scores) >= 3 else 0.0
        conviction_gate = abs(median_score) + max(0.05, mad * 0.8)
        if sentiment_snapshot.article_count < self.thresholds.min_articles:
            conviction_gate = max(conviction_gate, threshold * 1.5)
        volatility_penalty = 0.0
        baseline_range = self._baseline.get("avg_range_pct", 0.0)
        if baseline_range and baseline_range < 0.01:
            volatility_penalty = 0.02
        gated_threshold = conviction_gate + volatility_penalty
        if abs(sentiment_score) < max(threshold, gated_threshold):
            return TradeDecision(
                ticker=ticker,
                action="HOLD",
                confidence=0.0,
                notional=0.0,
                time_in_force="gtc",
                stop_loss=None,
                take_profit=None,
                reason="signal below conviction gate",
                intent="hold",
                metadata={
                    "score": f"{sentiment_score:.3f}",
                    "medianScore": f"{median_score:.3f}",
                    "keywords": keywords,
                    "articles": sentiment_snapshot.article_count,
                },
            )
        if abs(sentiment_score) < threshold:
            return TradeDecision(
                ticker=ticker,
                action="HOLD",
                confidence=0.0,
                notional=0.0,
                time_in_force="gtc",
                stop_loss=None,
                take_profit=None,
                reason="no alert",
                intent="hold",
                metadata={
                    "score": f"{sentiment_score:.3f}",
                    "keywords": keywords,
                    "articles": sentiment_snapshot.article_count,
                },
            )
        alert = Alert(
            ticker=ticker,
            average_sentiment=sentiment_snapshot.tone_15,
            sentiment_delta=sentiment_snapshot.delta_15,
            article_count=sentiment_snapshot.article_count,
            trigger_reason="score",
            created_at=sentiment_snapshot.timestamp,
        )
        price_snapshot = {"close": price}
        equity = max(portfolio_equity, 0.0) + 1e-6
        cash = max(portfolio_cash, 0.0)
        account = AccountSnapshot(
            equity=equity,
            cash=cash,
            buying_power=max(cash, equity),
            portfolio_value=equity,
        )
        decision = self.engine.decide(
            alert,
            account=account,
            price_snapshot=price_snapshot,
            position=position,
            open_positions=open_positions,
        )
        max_notional = min(equity * self.risk.max_capital_fraction, portfolio_cash)
        if decision.notional > max_notional:
            decision = TradeDecision(
                ticker=decision.ticker,
                action=decision.action,
                confidence=decision.confidence,
                notional=max_notional,
                time_in_force=decision.time_in_force,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                reason=decision.reason,
                intent=decision.intent,
                quantity=decision.quantity,
                metadata=decision.metadata,
            )
        if decision.notional <= 0 or decision.notional < price * 0.01:
            return TradeDecision(
                ticker=ticker,
                action="HOLD",
                confidence=0.0,
                notional=0.0,
                time_in_force="gtc",
                stop_loss=None,
                take_profit=None,
                reason="insufficient capital",
                intent="hold",
                metadata={
                    "score": f"{sentiment_score:.3f}",
                    "keywords": keywords,
                    "articles": sentiment_snapshot.article_count,
                },
            )
        strength = min(max(abs(sentiment_score), 0.2), 2.0)
        adjusted_notional = min(decision.notional * strength, max_notional)
        if adjusted_notional <= price * 0.01:
            return TradeDecision(
                ticker=ticker,
                action="HOLD",
                confidence=0.0,
                notional=0.0,
                time_in_force="gtc",
                stop_loss=None,
                take_profit=None,
                reason="signal below sizing floor",
                intent="hold",
                metadata={
                    "score": f"{sentiment_score:.3f}",
                    "keywords": keywords,
                    "articles": sentiment_snapshot.article_count,
                    "medianScore": f"{median_score:.3f}",
                },
            )
        decision = TradeDecision(
            ticker=decision.ticker,
            action=decision.action,
            confidence=min(1.0, decision.confidence * strength),
            notional=adjusted_notional,
            time_in_force=decision.time_in_force,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            reason=decision.reason,
            intent=decision.intent,
            quantity=decision.quantity,
            metadata=decision.metadata,
        )
        decision.metadata.update({
            "score": f"{sentiment_score:.3f}",
            "medianScore": f"{median_score:.3f}",
            "mad": f"{mad:.3f}",
            "keywords": keywords,
            "articles": sentiment_snapshot.article_count,
            "strength": f"{strength:.2f}",
            "volatility": f"{baseline_range:.4f}",
        })
        if decision.notional > 0 and decision.notional < self._tiny_notional_threshold:
            decision.metadata["notional_warning"] = f"tiny notional ${decision.notional:.2f}"
        return decision

    def on_trade_filled(
        self,
        *,
        decision: TradeDecision,
        timestamp: str,
        price: float,
        quantity: float,
    ) -> None:
        if quantity <= 0:
            return
        fill_time = self._parse_timestamp(timestamp)
        if fill_time is None:
            return
        ticker = decision.ticker.upper()
        if decision.intent == "entry":
            self.tracker.register_trade(ticker, opened=fill_time)
            self._open_entries[ticker].append(fill_time)
            return
        if decision.intent == "exit":
            opened_time: Optional[datetime] = None
            queue = self._open_entries.get(ticker)
            if queue:
                opened_time = queue.popleft()
                if not queue:
                    self._open_entries.pop(ticker, None)
            self.tracker.register_trade(ticker, opened=opened_time, closed=fill_time)
            return
        if decision.action == "BUY":
            self.tracker.register_trade(ticker, opened=fill_time)
            self._open_entries[ticker].append(fill_time)
        elif decision.action == "SELL":
            opened_time = None
            queue = self._open_entries.get(ticker)
            if queue:
                opened_time = queue.popleft()
                if not queue:
                    self._open_entries.pop(ticker, None)
            self.tracker.register_trade(ticker, opened=opened_time, closed=fill_time)

    def _parse_timestamp(self, raw: str) -> Optional[datetime]:
        if not raw:
            return None
        text = raw.strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is not None:
            try:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                parsed = parsed.replace(tzinfo=None)
        return parsed


__all__ = ["TraderV4Bot"]
