"""Shared in-memory state for the TraderV4 automation stack."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, List, Optional

from Traderv4.funcs import Alert, TradeDecision


@dataclass
class AlertRecord:
    ticker: str
    average_sentiment: float
    sentiment_delta: float
    article_count: int
    reason: str
    seen_at: datetime

    @classmethod
    def from_alert(cls, alert: Alert) -> "AlertRecord":
        return cls(
            ticker=alert.ticker,
            average_sentiment=alert.average_sentiment,
            sentiment_delta=alert.sentiment_delta,
            article_count=alert.article_count,
            reason=alert.trigger_reason,
            seen_at=alert.created_at,
        )

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "averageSentiment": self.average_sentiment,
            "sentimentDelta": self.sentiment_delta,
            "articleCount": self.article_count,
            "reason": self.reason,
            "seenAt": self.seen_at.isoformat(),
        }


@dataclass
class DecisionRecord:
    ticker: str
    action: str
    notional: float
    confidence: float
    reason: str
    intent: str
    created_at: datetime

    @classmethod
    def from_decision(cls, decision: TradeDecision) -> "DecisionRecord":
        return cls(
            ticker=decision.ticker,
            action=decision.action,
            notional=decision.notional,
            confidence=decision.confidence,
            reason=decision.reason,
            intent=decision.intent,
            created_at=datetime.utcnow(),
        )

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "action": self.action,
            "notional": self.notional,
            "confidence": self.confidence,
            "reason": self.reason,
            "intent": self.intent,
            "createdAt": self.created_at.isoformat(),
        }


@dataclass
class PositionRecord:
    ticker: str
    side: str
    quantity: float
    market_value: float
    cost_basis: Optional[float]
    last_updated: datetime

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "side": self.side,
            "quantity": self.quantity,
            "marketValue": self.market_value,
            "costBasis": self.cost_basis,
            "lastUpdated": self.last_updated.isoformat(),
        }


@dataclass
class DayTradeStatus:
    used: int
    limit: int
    next_reset: Optional[datetime]

    def as_dict(self) -> dict:
        return {
            "used": self.used,
            "limit": self.limit,
            "nextReset": self.next_reset.isoformat() if self.next_reset else None,
        }


@dataclass
class SentimentPoint:
    seen_at: datetime
    average_sentiment: float
    sentiment_delta: float
    article_count: int
    ticker: str

    def as_dict(self) -> dict:
        return {
            "seenAt": self.seen_at.isoformat(),
            "averageSentiment": self.average_sentiment,
            "sentimentDelta": self.sentiment_delta,
            "articleCount": self.article_count,
            "ticker": self.ticker,
        }

@dataclass
class MarketStatus:
    is_open: Optional[bool]
    next_open: Optional[datetime]
    next_close: Optional[datetime]
    timestamp: datetime

    def as_dict(self) -> dict:
        return {
            "isOpen": self.is_open,
            "nextOpen": self.next_open.isoformat() if self.next_open else None,
            "nextClose": self.next_close.isoformat() if self.next_close else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CycleDiagnostics:
    started_at: datetime
    completed_at: datetime
    tickers_processed: int
    alerts_generated: int
    decisions_made: int
    holds: int
    entries: int
    exits: int
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "startedAt": self.started_at.isoformat(),
            "completedAt": self.completed_at.isoformat(),
            "tickersProcessed": self.tickers_processed,
            "alertsGenerated": self.alerts_generated,
            "decisionsMade": self.decisions_made,
            "holds": self.holds,
            "entries": self.entries,
            "exits": self.exits,
            "errors": self.errors,
        }

@dataclass
class TraderSnapshot:
    alerts: List[AlertRecord] = field(default_factory=list)
    decisions: List[DecisionRecord] = field(default_factory=list)
    positions: List[PositionRecord] = field(default_factory=list)
    day_trade: Optional[DayTradeStatus] = None
    last_updated: Optional[datetime] = None
    sentiment_history: List[SentimentPoint] = field(default_factory=list)
    market_status: Optional[MarketStatus] = None
    diagnostics: Optional[CycleDiagnostics] = None

    def as_dict(self) -> dict:
        return {
            "alerts": [alert.as_dict() for alert in self.alerts],
            "decisions": [decision.as_dict() for decision in self.decisions],
            "positions": [position.as_dict() for position in self.positions],
            "dayTrade": self.day_trade.as_dict() if self.day_trade else None,
            "sentimentHistory": [point.as_dict() for point in self.sentiment_history],
            "lastUpdated": self.last_updated.isoformat() if self.last_updated else None,
            "marketStatus": self.market_status.as_dict() if self.market_status else None,
            "diagnostics": self.diagnostics.as_dict() if self.diagnostics else None,
        }


class StateStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = TraderSnapshot()

    def update(
        self,
        *,
        alerts: List[AlertRecord],
        decisions: List[DecisionRecord],
        positions: List[PositionRecord],
        day_trade: Optional[DayTradeStatus],
        sentiment_history: List[SentimentPoint],
        market_status: Optional[MarketStatus],
        diagnostics: Optional[CycleDiagnostics],
    ) -> None:
        with self._lock:
            self._snapshot.alerts = alerts
            self._snapshot.decisions = decisions
            self._snapshot.positions = positions
            self._snapshot.day_trade = day_trade
            self._snapshot.last_updated = datetime.utcnow()
            self._snapshot.sentiment_history = sentiment_history
            self._snapshot.market_status = market_status
            self._snapshot.diagnostics = diagnostics

    def snapshot(self) -> TraderSnapshot:
        with self._lock:
            return TraderSnapshot(
                alerts=list(self._snapshot.alerts),
                decisions=list(self._snapshot.decisions),
                positions=list(self._snapshot.positions),
                day_trade=self._snapshot.day_trade,
                last_updated=self._snapshot.last_updated,
                sentiment_history=list(self._snapshot.sentiment_history),
                market_status=self._snapshot.market_status,
                diagnostics=self._snapshot.diagnostics,
            )


__all__ = [
    "AlertRecord",
    "DecisionRecord",
    "PositionRecord",
    "DayTradeStatus",
    "TraderSnapshot",
    "SentimentPoint",
    "StateStore",
    "MarketStatus",
    "CycleDiagnostics",
]
