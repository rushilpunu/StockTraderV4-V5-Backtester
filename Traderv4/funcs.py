from __future__ import annotations

import inspect
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from Traderv4.GDELT import GDELTArticle

import requests


@dataclass
class ArticleSentiment:
    ticker: str
    sentiment: float
    timestamp: datetime
    source: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class TickerSentimentSnapshot:
    ticker: str
    average_sentiment: float
    total_articles: int
    weighted_sentiment: float
    sentiment_delta: float
    time_window: timedelta
    raw_articles: List[ArticleSentiment] = field(default_factory=list)


@dataclass
class Alert:
    ticker: str
    average_sentiment: float
    sentiment_delta: float
    article_count: int
    trigger_reason: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RiskConfig:
    max_capital_fraction: float = 0.2
    max_positions: int = 5
    stop_loss_pct: float = 0.04
    take_profit_pct: float = 0.08
    take_profit_tolerance: float = 0.10
    expected_hold_minutes: int = 720
    cooldown_minutes: int = 30
    entry_sentiment_threshold: float = 0.1
    exit_sentiment_threshold: float = 0.05
    max_position_value: Optional[float] = None
    allow_shorting: bool = False
    aggressiveness: float = 1.0
    max_trade_leverage: float = 1.0
    entry_signal_bias: float = 0.0


@dataclass
class VolatilityThresholds:
    min_articles: int = 1
    sentiment_spike: float = 0.2
    sentiment_extreme: float = 0.45
    average_sentiment_floor: float = 0.25
    single_article_spike: float = 0.35


@dataclass
class TradeDecision:
    ticker: str
    action: str
    confidence: float
    notional: float
    time_in_force: str
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reason: str
    intent: str = "entry"
    quantity: Optional[float] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float

    @classmethod
    def from_account(cls, account: Any) -> "AccountSnapshot":
        def _to_float(value: Any, fallback: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        cash = _to_float(getattr(account, "cash", None))
        buying_power = _to_float(getattr(account, "buying_power", None), cash)
        equity = _to_float(getattr(account, "equity", None), buying_power or cash)
        portfolio_value = _to_float(getattr(account, "portfolio_value", None), equity)
        return cls(equity=equity, cash=cash, buying_power=buying_power, portfolio_value=portfolio_value)


class YahooFinanceClient:
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    def _fetch_chart(self, ticker: str, range_: str, interval: str) -> Optional[Dict[str, Any]]:
        url = self.BASE_URL.format(ticker=ticker)
        params = {
            "range": range_,
            "interval": interval,
            "includePrePost": "false",
            "events": "div,splits",
        }
        try:
            response = requests.get(url, params=params, timeout=6)
            response.raise_for_status()
        except requests.RequestException:
            return None
        payload = response.json()
        result = payload.get("chart", {}).get("result")
        if not result:
            return None
        return result[0]

    def fetch_intraday_snapshot(self, ticker: str, period: str = "5d", interval: str = "15m") -> Dict[str, Optional[float]]:
        chart = self._fetch_chart(ticker, range_=period, interval=interval)
        if not chart:
            return {"close": None, "volume": None}
        indicators = chart.get("indicators", {})
        quote = (indicators.get("quote") or [{}])[0]
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
        if not closes:
            return {"close": None, "volume": None}
        close = closes[-1]
        volume = volumes[-1] if volumes else None
        return {
            "close": float(close) if close is not None else None,
            "volume": float(volume) if volume is not None else None,
        }

    def fetch_volatility_baseline(self, ticker: str, lookback_days: int = 20) -> Dict[str, Optional[float]]:
        chart = self._fetch_chart(ticker, range_=f"{lookback_days}d", interval="1d")
        if not chart:
            return {"avg_volume": None, "avg_range_pct": None}
        high = chart.get("indicators", {}).get("quote", [{}])[0].get("high") or []
        low = chart.get("indicators", {}).get("quote", [{}])[0].get("low") or []
        close = chart.get("indicators", {}).get("quote", [{}])[0].get("close") or []
        volume = chart.get("indicators", {}).get("quote", [{}])[0].get("volume") or []
        if not close:
            return {"avg_volume": None, "avg_range_pct": None}
        ranges: List[float] = []
        for h, l, c in zip(high, low, close):
            if h is None or l is None or c in (None, 0):
                continue
            ranges.append(abs((h - l) / c))
        avg_range = sum(ranges) / len(ranges) if ranges else None
        vols = [v for v in volume if v is not None]
        avg_volume = (sum(vols) / len(vols)) if vols else None
        return {
            "avg_volume": float(avg_volume) if avg_volume is not None else None,
            "avg_range_pct": float(avg_range) if avg_range is not None else None,
        }


class SentimentProcessor:
    def normalize_gdelt_payload(self, payload: Dict[str, Any]) -> List[ArticleSentiment]:
        articles: List[ArticleSentiment] = []
        for item in payload.get("articles", []):
            ticker = item.get("semtag", "").upper() or item.get("title", "").split(" ")[0]
            timestamp_raw = item.get("seendate") or item.get("publishtime")
            timestamp = datetime.strptime(timestamp_raw, "%Y%m%dT%H%M%S") if timestamp_raw else datetime.utcnow()
            sentiment = float(item.get("tone", 0.0))
            source = item.get("sourceurl", "unknown")
            metadata = {"title": item.get("title", ""), "url": source}
            articles.append(ArticleSentiment(ticker=ticker, sentiment=sentiment, timestamp=timestamp, source=source, metadata=metadata))
        return articles

    def from_gdelt_articles(self, records: Iterable["GDELTArticle"]) -> List[ArticleSentiment]:
        normalized: List[ArticleSentiment] = []
        for record in records:
            ticker = (record.ticker or "").upper()
            if not ticker:
                continue
            normalized.append(
                ArticleSentiment(
                    ticker=ticker,
                    sentiment=record.tone,
                    timestamp=record.published_at,
                    source=record.url or "gdelt",
                    metadata={"title": record.title},
                )
            )
        return normalized

    def group_by_ticker(self, articles: Iterable[ArticleSentiment]) -> Dict[str, List[ArticleSentiment]]:
        grouped: Dict[str, List[ArticleSentiment]] = defaultdict(list)
        for article in articles:
            if article.ticker:
                grouped[article.ticker].append(article)
        return grouped

    def aggregate_sentiment(self, grouped_articles: Dict[str, List[ArticleSentiment]], window: timedelta) -> List[TickerSentimentSnapshot]:
        snapshots: List[TickerSentimentSnapshot] = []
        cutoff = datetime.utcnow() - window
        for ticker, articles in grouped_articles.items():
            recent = [a for a in articles if a.timestamp >= cutoff]
            if not recent:
                continue
            sentiments = [a.sentiment for a in recent]
            weights = [1.0 for _ in recent]
            weighted = sum(s * w for s, w in zip(sentiments, weights)) / max(sum(weights), 1.0)
            avg_sentiment = sum(sentiments) / len(sentiments)
            delta = sentiments[-1] - sentiments[0] if len(sentiments) > 1 else sentiments[0]
            snapshots.append(TickerSentimentSnapshot(
                ticker=ticker,
                average_sentiment=avg_sentiment,
                total_articles=len(recent),
                weighted_sentiment=weighted,
                sentiment_delta=delta,
                time_window=window,
                raw_articles=recent,
            ))
        return snapshots


class VolatilityAnalyzer:
    def __init__(self, thresholds: VolatilityThresholds):
        self.thresholds = thresholds

    def evaluate(self, snapshot: TickerSentimentSnapshot, baseline: Optional[Dict[str, float]] = None) -> Optional[Alert]:
        article_count = snapshot.total_articles
        if article_count < self.thresholds.min_articles:
            return None

        reason: Optional[str] = None
        abs_delta = abs(snapshot.sentiment_delta)
        abs_avg = abs(snapshot.average_sentiment)

        if abs_delta >= self.thresholds.sentiment_spike:
            reason = "sentiment spike"
        elif abs_avg >= self.thresholds.sentiment_extreme:
            reason = "sentiment extreme"
        else:
            if abs_avg >= self.thresholds.average_sentiment_floor and article_count >= max(1, self.thresholds.min_articles // 2 or 1):
                reason = "average sentiment floor"

        if not reason and snapshot.raw_articles:
            strongest = max(snapshot.raw_articles, key=lambda a: abs(a.sentiment))
            if abs(strongest.sentiment) >= self.thresholds.single_article_spike:
                reason = f"single article spike:{strongest.source}"

        if not reason:
            return None

        if baseline and baseline.get("avg_range_pct"):
            reason = f"{reason} | baseline_range={baseline['avg_range_pct']:.3f}"

        return Alert(
            ticker=snapshot.ticker,
            average_sentiment=snapshot.average_sentiment,
            sentiment_delta=snapshot.sentiment_delta,
            article_count=snapshot.total_articles,
            trigger_reason=reason,
        )


@dataclass
class DayTradeRecord:
    opened: datetime
    closed: Optional[datetime] = None
    ticker: str = ""

    @property
    def intraday(self) -> bool:
        if not self.closed:
            return False
        return self.opened.date() == self.closed.date()


class TradeFrequencyTracker:
    def __init__(self, max_day_trades: int = 3):
        self.max_day_trades = max_day_trades
        self.records: Deque[DayTradeRecord] = deque()
        self._open_records: Dict[str, Deque[DayTradeRecord]] = defaultdict(deque)

    def register_trade(
        self,
        ticker: str,
        opened: Optional[datetime] = None,
        closed: Optional[datetime] = None,
    ) -> None:
        if opened is None and closed is None:
            raise ValueError("trade registration requires opened or closed timestamp")

        if opened is not None and closed is None:
            record = DayTradeRecord(opened=opened, ticker=ticker)
            self.records.append(record)
            self._open_records[ticker].append(record)
        elif closed is not None:
            queue = self._open_records.get(ticker)
            if queue:
                record = queue.popleft()
                if opened and opened != record.opened:
                    record.opened = opened
                record.closed = closed
                if not queue:
                    self._open_records.pop(ticker, None)
            else:
                opened_ts = opened or closed
                self.records.append(DayTradeRecord(opened=opened_ts, closed=closed, ticker=ticker))

        self._prune()

    def _prune(self) -> None:
        cutoff = datetime.utcnow() - timedelta(days=5)
        while self.records and self.records[0].opened < cutoff:
            record = self.records.popleft()
            if record.closed is None:
                queue = self._open_records.get(record.ticker)
                if queue:
                    try:
                        queue.remove(record)
                    except ValueError:
                        pass
                    if not queue:
                        self._open_records.pop(record.ticker, None)

    def intraday_trades(self) -> int:
        self._prune()
        return sum(1 for record in self.records if record.intraday)

    def can_enter(self) -> bool:
        return self.intraday_trades() < self.max_day_trades

    def next_reset_date(self) -> Optional[datetime]:
        self._prune()
        intraday_records = [record for record in self.records if record.intraday]
        if not intraday_records:
            return None
        oldest_close = min(
            (record.closed or record.opened) for record in intraday_records
        )
        return oldest_close + timedelta(days=5)


class DecisionEngine:
    def __init__(self, risk: RiskConfig, trade_tracker: TradeFrequencyTracker):
        self.risk = risk
        self.trade_tracker = trade_tracker
        self.last_trade_at: Dict[str, datetime] = {}
        self.signal_memory: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=6))

    def decide(
        self,
        alert: Alert,
        account: AccountSnapshot,
        price_snapshot: Dict[str, Optional[float]],
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
            return TradeDecision(
                ticker=alert.ticker,
                action="HOLD",
                confidence=0.0,
                notional=0.0,
                time_in_force="gtc",
                stop_loss=None,
                take_profit=None,
                reason="missing price data",
                intent="hold",
            )

        quantity = 0.0
        market_value = 0.0
        side = None
        if position:
            try:
                quantity = abs(float(position.get("quantity", 0.0)))
            except (TypeError, ValueError):
                quantity = 0.0
            try:
                market_value = abs(float(position.get("market_value", 0.0)))
            except (TypeError, ValueError):
                market_value = quantity * price
            raw_side = str(position.get("side", "")).upper()
            if quantity > 0:
                if raw_side in {"LONG", "BUY"}:
                    side = "LONG"
                elif raw_side in {"SHORT", "SELL"}:
                    side = "SHORT"
                else:
                    side = "LONG"

        sentiment = alert.average_sentiment
        delta = alert.sentiment_delta
        article_boost = 1.0 + min(alert.article_count, 12) / 24.0
        signal_score = (sentiment * 0.6 + delta * 0.4) * article_boost
        history = self.signal_memory[alert.ticker]
        history.append(signal_score)
        trend_score = sum(history) / len(history)
        confidence = min(abs(trend_score), 1.0)

        equity = account.equity or account.portfolio_value or account.cash
        cash = max(account.cash, 0.0)
        buying_power = max(account.buying_power, 0.0)
        if cash > 0 and buying_power > 0:
            available_funds = min(cash, buying_power)
        else:
            available_funds = buying_power or cash
        per_position_cap = equity * self.risk.max_capital_fraction if equity else available_funds
        if self.risk.max_position_value is not None:
            per_position_cap = min(per_position_cap, self.risk.max_position_value)
        if market_value > per_position_cap:
            per_position_cap = max(per_position_cap, market_value)
        ticker_capacity = max(0.0, per_position_cap - market_value)
        budget = max(0.0, min(available_funds, ticker_capacity))

        exit_threshold = max(self.risk.exit_sentiment_threshold, 0.0)
        entry_threshold = max(self.risk.entry_sentiment_threshold, 0.0)
        day_trade_pressure = max(0, self.trade_tracker.intraday_trades() - (self.trade_tracker.max_day_trades - 2))
        if day_trade_pressure > 0:
            entry_threshold += 0.05 * day_trade_pressure

        # Exit logic for existing positions
        if side == "LONG":
            should_exit = (
                trend_score <= -exit_threshold
                or sentiment <= -exit_threshold
                or delta <= -exit_threshold
            )
            if should_exit and quantity > 0:
                notional = market_value if market_value > 0 else quantity * price
                decision = TradeDecision(
                    ticker=alert.ticker,
                    action="SELL",
                    confidence=confidence,
                    notional=notional,
                    time_in_force="gtc",
                    stop_loss=None,
                    take_profit=None,
                    reason="exit: sentiment reversal",
                    intent="exit",
                    quantity=quantity,
                    metadata={
                        "avg_sentiment": f"{sentiment:.3f}",
                        "sentiment_delta": f"{delta:.3f}",
                        "articles": str(alert.article_count),
                        "score": f"{signal_score:.3f}",
                        "trend": f"{trend_score:.3f}",
                        "position_side": side,
                    },
                )
                self.last_trade_at[alert.ticker] = datetime.utcnow()
                return decision
            return TradeDecision(
                ticker=alert.ticker,
                action="HOLD",
                confidence=0.0,
                notional=0.0,
                time_in_force="gtc",
                stop_loss=None,
                take_profit=None,
                reason="holding long position",
                intent="hold",
            )

        if side == "SHORT":
            should_cover = (
                trend_score >= exit_threshold
                or sentiment >= exit_threshold
                or delta >= exit_threshold
            )
            if should_cover and quantity > 0:
                notional = market_value if market_value > 0 else quantity * price
                decision = TradeDecision(
                    ticker=alert.ticker,
                    action="BUY",
                    confidence=confidence,
                    notional=notional,
                    time_in_force="gtc",
                    stop_loss=None,
                    take_profit=None,
                    reason="exit: sentiment reversal",
                    intent="exit",
                    quantity=quantity,
                    metadata={
                        "avg_sentiment": f"{sentiment:.3f}",
                        "sentiment_delta": f"{delta:.3f}",
                        "articles": str(alert.article_count),
                        "score": f"{signal_score:.3f}",
                        "trend": f"{trend_score:.3f}",
                        "position_side": side,
                    },
                )
                self.last_trade_at[alert.ticker] = datetime.utcnow()
                return decision
            return TradeDecision(
                ticker=alert.ticker,
                action="HOLD",
                confidence=0.0,
                notional=0.0,
                time_in_force="gtc",
                stop_loss=None,
                take_profit=None,
                reason="holding short position",
                intent="hold",
            )

        # No existing position: consider new entries
        if open_positions >= self.risk.max_positions:
            return TradeDecision(
                ticker=alert.ticker,
                action="HOLD",
                confidence=0.0,
                notional=0.0,
                time_in_force="gtc",
                stop_loss=None,
                take_profit=None,
                reason="max positions reached",
                intent="hold",
            )

        if not self.trade_tracker.can_enter():
            return TradeDecision(
                ticker=alert.ticker,
                action="HOLD",
                confidence=0.0,
                notional=0.0,
                time_in_force="gtc",
                stop_loss=None,
                take_profit=None,
                reason="day trade limit reached",
                intent="hold",
            )

        cooldown = self.risk.cooldown_minutes
        last_trade = self.last_trade_at.get(alert.ticker)
        if last_trade and datetime.utcnow() - last_trade < timedelta(minutes=cooldown):
            return TradeDecision(
                ticker=alert.ticker,
                action="HOLD",
                confidence=0.0,
                notional=0.0,
                time_in_force="gtc",
                stop_loss=None,
                take_profit=None,
                reason="cooldown active",
                intent="hold",
            )

        if signal_score >= entry_threshold:
            if budget <= 0 or budget < price * 0.01:
                return TradeDecision(
                    ticker=alert.ticker,
                    action="HOLD",
                    confidence=0.0,
                    notional=0.0,
                    time_in_force="gtc",
                    stop_loss=None,
                    take_profit=None,
                    reason="insufficient buying power",
                    intent="hold",
                )
            stop_loss = price * (1 - self.risk.stop_loss_pct)
            take_profit = price * (1 + self.risk.take_profit_pct)
            normalized_score = min(max(signal_score, 0.0), 1.25)
            notional = budget * min(normalized_score, 1.0)
            if notional < price * 0.01:
                return TradeDecision(
                    ticker=alert.ticker,
                    action="HOLD",
                    confidence=0.0,
                    notional=0.0,
                    time_in_force="gtc",
                    stop_loss=None,
                    take_profit=None,
                    reason="signal below sizing floor",
                    intent="hold",
                )
            decision = TradeDecision(
                ticker=alert.ticker,
                action="BUY",
                confidence=confidence,
                notional=notional,
                time_in_force="gtc",
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=alert.trigger_reason,
                intent="entry",
                metadata={
                    "avg_sentiment": f"{sentiment:.3f}",
                    "sentiment_delta": f"{delta:.3f}",
                    "articles": str(alert.article_count),
                    "score": f"{signal_score:.3f}",
                    "trend": f"{trend_score:.3f}",
                    "article_boost": f"{article_boost:.3f}",
                },
            )
            self.last_trade_at[alert.ticker] = datetime.utcnow()
            return decision

        if self.risk.allow_shorting and signal_score <= -entry_threshold:
            if budget <= 0 or budget < price * 0.01:
                return TradeDecision(
                    ticker=alert.ticker,
                    action="HOLD",
                    confidence=0.0,
                    notional=0.0,
                    time_in_force="gtc",
                    stop_loss=None,
                    take_profit=None,
                    reason="insufficient buying power",
                    intent="hold",
                )
            stop_loss = price * (1 + self.risk.stop_loss_pct)
            take_profit = price * (1 - self.risk.take_profit_pct)
            normalized_score = min(max(abs(signal_score), 0.0), 1.25)
            notional = budget * min(normalized_score, 1.0)
            if notional < price * 0.01:
                return TradeDecision(
                    ticker=alert.ticker,
                    action="HOLD",
                    confidence=0.0,
                    notional=0.0,
                    time_in_force="gtc",
                    stop_loss=None,
                    take_profit=None,
                    reason="signal below sizing floor",
                    intent="hold",
                )
            decision = TradeDecision(
                ticker=alert.ticker,
                action="SELL",
                confidence=confidence,
                notional=notional,
                time_in_force="gtc",
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=alert.trigger_reason,
                intent="entry",
                metadata={
                    "avg_sentiment": f"{sentiment:.3f}",
                    "sentiment_delta": f"{delta:.3f}",
                    "articles": str(alert.article_count),
                    "score": f"{signal_score:.3f}",
                    "trend": f"{trend_score:.3f}",
                    "article_boost": f"{article_boost:.3f}",
                },
            )
            self.last_trade_at[alert.ticker] = datetime.utcnow()
            return decision

        return TradeDecision(
            ticker=alert.ticker,
            action="HOLD",
            confidence=0.0,
            notional=0.0,
            time_in_force="gtc",
            stop_loss=None,
            take_profit=None,
            reason="no actionable sentiment",
            intent="hold",
        )


class TradeExecutor:
    def __init__(self, alpaca_client, trade_tracker: TradeFrequencyTracker):
        self.alpaca = alpaca_client
        self.trade_tracker = trade_tracker
        self.log = logging.getLogger("trade_executor")
        self._open_entries: Dict[str, Deque[datetime]] = defaultdict(deque)
        self._seen_entry_orders: Set[str] = set()
        self._seen_exit_orders: Set[str] = set()

    def place_trade(self, decision: TradeDecision) -> Optional[str]:
        if decision.action == "HOLD":
            self.log.info("Skipping trade for %s: %s", decision.ticker, decision.reason)
            return None

        if decision.intent != "exit":
            guard_reason = self._pdt_guard_reason()
            if guard_reason:
                self.log.info("Skipping trade for %s: %s", decision.ticker, guard_reason)
                return None

        order_kwargs: Dict[str, Any] = {
            "symbol": decision.ticker,
            "side": "buy" if decision.action == "BUY" else "sell",
            "type": "market",
            "time_in_force": decision.time_in_force,
        }

        if decision.quantity is not None:
            if decision.quantity <= 0:
                self.log.info("Skipping trade for %s: zero quantity", decision.ticker)
                return None
            quantity = round(decision.quantity, 6)
            order_kwargs["qty"] = quantity
            if not math.isclose(quantity, round(quantity), rel_tol=0.0, abs_tol=1e-6):
                # Fractional share orders must be DAY orders when routed to Alpaca.
                if order_kwargs.get("time_in_force", "").lower() != "day":
                    self.log.debug(
                        "Adjusting time_in_force to DAY for fractional qty %.6f on %s",
                        quantity,
                        decision.ticker,
                    )
                    order_kwargs["time_in_force"] = "day"
        elif decision.notional > 0:
            notional = round(decision.notional, 2)
            order_kwargs["notional"] = notional
            if order_kwargs.get("time_in_force", "").lower() != "day":
                self.log.debug(
                    "Adjusting time_in_force to DAY for fractional notional %.2f on %s",
                    notional,
                    decision.ticker,
                )
                order_kwargs["time_in_force"] = "day"
        else:
            self.log.info("Skipping trade for %s: zero notional", decision.ticker)
            return None

        if (
            decision.intent == "entry"
            and decision.action == "BUY"
            and decision.take_profit is not None
            and decision.stop_loss is not None
        ):
            order_kwargs["order_class"] = "bracket"
            order_kwargs["take_profit"] = {"limit_price": decision.take_profit}
            order_kwargs["stop_loss"] = {"stop_price": decision.stop_loss}

        try:
            self._log_trade_intent(decision, order_kwargs)
            order = self.alpaca.submit_order(**order_kwargs)
            order_id = getattr(order, "id", None)
            if order_id is None and isinstance(order, dict):
                order_id = order.get("id")
            self.log.info("Submitted order %s for %s", order_id or "?", decision.ticker)
            return order_id
        except Exception as exc:
            self.log.exception("Failed to submit order for %s: %s", decision.ticker, exc)
            return None

    def _pdt_guard_reason(self) -> Optional[str]:
        try:
            account = self.alpaca.get_account()
        except Exception as exc:  # pragma: no cover - guard should not stop trading on fetch issues
            self.log.debug("Unable to fetch account for PDT guard: %s", exc)
            return None

        def _to_float(value: Any, fallback: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        def _to_int(value: Any, fallback: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        equity = _to_float(getattr(account, "equity", None), _to_float(getattr(account, "cash", 0.0)))
        last_equity = _to_float(getattr(account, "last_equity", None))
        if equity == 0.0 and last_equity:
            equity = last_equity
        daytrade_count = _to_int(getattr(account, "daytrade_count", 0))
        used = max(daytrade_count, self.trade_tracker.intraday_trades())
        limit = self.trade_tracker.max_day_trades

        if equity < 25_000 and used >= limit:
            return (
                f"PDT guard active: equity ${equity:,.2f}, intraday trades {used}/{limit}"
            )

        return None

    def _log_trade_intent(self, decision: TradeDecision, order_kwargs: Dict[str, Any]) -> None:
        """Emit a structured log line that explains why a trade is being submitted."""

        side = order_kwargs.get("side", decision.action).upper()
        qty = order_kwargs.get("qty")
        notional = order_kwargs.get("notional")
        pieces = [
            f"intent={decision.intent}",
            f"confidence={decision.confidence:.3f}",
            f"reason={decision.reason}",
        ]
        if qty is not None:
            pieces.append(f"qty={qty}")
        if notional is not None:
            pieces.append(f"notional={notional}")
        tif = order_kwargs.get("time_in_force")
        if tif:
            pieces.append(f"tif={tif.upper()}")
        metadata = decision.metadata or {}
        if metadata:
            preview_items = list(metadata.items())[:8]
            meta_repr = ", ".join(f"{key}={value}" for key, value in preview_items)
        else:
            meta_repr = "none"
        self.log.info(
            "Routing %s order for %s | %s | metadata: %s",
            side,
            decision.ticker,
            ", ".join(pieces),
            meta_repr,
        )

    def sync_trade_activity(self) -> None:
        fetcher = getattr(self.alpaca, "list_activities", None)
        args: Tuple[Any, ...] = tuple()
        kwargs: Dict[str, Any]
        if fetcher is not None:
            kwargs = {"activity_types": "FILL", "page_size": 100}
        else:
            fetcher = getattr(self.alpaca, "get_activities", None)
            if fetcher is None:
                self.log.warning("Alpaca client does not support activity listing")
                return
            params = inspect.signature(fetcher).parameters
            if "activity_type" in params:
                kwargs = {"activity_type": "FILL"}
            elif "activity_types" in params:
                kwargs = {"activity_types": "FILL"}
            else:
                args = ("FILL",)
                kwargs = {}

        try:
            activities = fetcher(*args, **kwargs)  # type: ignore[misc]
        except Exception as exc:
            self.log.warning("Unable to sync trade activities: %s", exc)
            return

        for activity in reversed(list(activities)):
            order_id = getattr(activity, "order_id", None)
            symbol = getattr(activity, "symbol", "").upper()
            side = getattr(activity, "side", "").lower()
            timestamp_raw = getattr(activity, "transaction_time", None) or getattr(activity, "date", None)

            if not symbol or side not in {"buy", "sell"}:
                continue

            key = (order_id or f"{symbol}:{timestamp_raw}")
            if side == "buy" and key in self._seen_entry_orders:
                continue
            if side == "sell" and key in self._seen_exit_orders:
                continue

            fill_time = self._parse_activity_time(timestamp_raw)
            if not fill_time:
                continue

            if side == "buy":
                self.trade_tracker.register_trade(symbol, opened=fill_time)
                self._open_entries[symbol].append(fill_time)
                self._seen_entry_orders.add(key)
            else:
                opened_time: Optional[datetime] = None
                queue = self._open_entries.get(symbol)
                if queue:
                    opened_time = queue.popleft()
                    if not queue:
                        self._open_entries.pop(symbol, None)
                self.trade_tracker.register_trade(symbol, opened=opened_time, closed=fill_time)
                self._seen_exit_orders.add(key)

    def _parse_activity_time(self, raw: Optional[str]) -> Optional[datetime]:
        if not raw:
            return None
        if isinstance(raw, datetime):
            parsed = raw
        elif hasattr(raw, "to_pydatetime"):
            try:
                parsed = raw.to_pydatetime()  # type: ignore[attr-defined]
            except Exception:
                parsed = None
        else:
            try:
                text = str(raw).strip()
            except Exception:
                return None
            if not text:
                return None
            normalized = text.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                self.log.debug("Unable to parse activity timestamp %s", raw)
                return None
        if parsed is None:
            return None
        try:
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        except Exception:
            pass
        return parsed

    def get_open_positions(self) -> List[Dict[str, float]]:
        try:
            positions = self.alpaca.list_positions()
        except Exception as exc:
            self.log.warning("Unable to fetch positions: %s", exc)
            return []

        normalized: List[Dict[str, float]] = []
        for position in positions:
            symbol = getattr(position, "symbol", getattr(position, "ticker", ""))
            if not symbol:
                continue
            try:
                quantity = float(getattr(position, "qty", getattr(position, "quantity", 0.0)))
            except Exception:
                quantity = 0.0
            try:
                market_value = float(getattr(position, "market_value", 0.0))
            except Exception:
                market_value = 0.0
            try:
                cost_basis = float(getattr(position, "cost_basis", 0.0))
            except Exception:
                cost_basis = None
            try:
                avg_entry_price = float(
                    getattr(position, "avg_entry_price", getattr(position, "avg_price", 0.0))
                )
            except Exception:
                avg_entry_price = None
            normalized.append(
                {
                    "ticker": symbol.upper(),
                    "side": getattr(position, "side", ""),
                    "quantity": quantity,
                    "market_value": market_value,
                    "cost_basis": cost_basis,
                    "avg_entry_price": avg_entry_price,
                }
            )
        return normalized

    def get_account_snapshot(self) -> AccountSnapshot:
        try:
            account = self.alpaca.get_account()
        except Exception as exc:
            self.log.warning("Unable to fetch account snapshot: %s", exc)
            return AccountSnapshot(equity=0.0, cash=0.0, buying_power=0.0, portfolio_value=0.0)
        return AccountSnapshot.from_account(account)

    def get_market_clock(self) -> Optional[Dict[str, Any]]:
        try:
            clock = self.alpaca.get_clock()
        except Exception as exc:
            self.log.warning("Unable to fetch market clock: %s", exc)
            return None

        def _extract(obj: Any, key: str) -> Any:
            value = getattr(obj, key, None)
            if value is None and isinstance(obj, dict):
                value = obj.get(key)
            return value

        is_open = bool(_extract(clock, "is_open"))
        next_open_raw = _extract(clock, "next_open")
        next_close_raw = _extract(clock, "next_close")
        timestamp_raw = _extract(clock, "timestamp")

        next_open = self._parse_activity_time(next_open_raw) if next_open_raw else None
        next_close = self._parse_activity_time(next_close_raw) if next_close_raw else None
        timestamp = self._parse_activity_time(timestamp_raw) if timestamp_raw else datetime.utcnow()

        return {
            "is_open": is_open,
            "next_open": next_open,
            "next_close": next_close,
            "timestamp": timestamp,
        }


class EventLogger:
    def __init__(self):
        self.log = logging.getLogger("trader")

    def log_alert(self, alert: Alert) -> None:
        self.log.info(
            "Alert for %s | avg=%.3f delta=%.3f count=%s reason=%s",
            alert.ticker,
            alert.average_sentiment,
            alert.sentiment_delta,
            alert.article_count,
            alert.trigger_reason,
        )

    def log_decision(self, decision: TradeDecision) -> None:
        quantity_note = ""
        if decision.quantity is not None:
            quantity_note = f" qty={decision.quantity:.4f}"
        self.log.info(
            "Decision[%s] %s %s notional=%.2f%s reason=%s",
            decision.intent,
            decision.action,
            decision.ticker,
            decision.notional,
            quantity_note,
            decision.reason,
        )

    def log_trade_frequency(self, tracker: TradeFrequencyTracker) -> None:
        self.log.info("Intraday trades in window: %s", tracker.intraday_trades())

    def log_market_closed(self, next_open: Optional[datetime]) -> None:
        if next_open:
            self.log.info("Market closed; next open at %s", next_open.isoformat())
        else:
            self.log.info("Market closed; awaiting next session schedule")

    def log_cycle_summary(
        self,
        *,
        tickers: int,
        alerts: int,
        decisions: int,
        entries: int,
        exits: int,
        holds: int,
        errors: int,
    ) -> None:
        self.log.info(
            "Cycle summary | tickers=%s alerts=%s decisions=%s entries=%s exits=%s holds=%s errors=%s",
            tickers,
            alerts,
            decisions,
            entries,
            exits,
            holds,
            errors,
        )
